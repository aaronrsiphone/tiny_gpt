# coding: utf-8
'''
Accelerate / BLAS bindings, and the model's working dtype.

Every dense matmul in tiny_gpt goes through cblas_sgemm from Apple's
Accelerate framework when it's available (macOS, iOS/Pythonista), and
through plain NumPy everywhere else. This module is the only place that
touches ctypes or knows Accelerate exists -- every other module calls
gemm()/bmm() and gets the same numbers back regardless of platform.

It also owns DT, the model's working dtype. DT is a mutable module
attribute rather than a constant so that diagnostics.gradcheck() can flip
the whole model into float64 for finite differences and flip it back
afterwards. Other modules must read backend.DT at the point of use (e.g.
`backend.DT`), not `from tinygpt.backend import DT`, or they will keep
seeing whatever dtype was active the moment they were first imported
instead of the current one.
'''

import ctypes
import os

import numpy as np

CblasRowMajor = 101
CblasNoTrans = 111
CblasTrans = 112

try:
    _lib = ctypes.CDLL(
        '/System/Library/Frameworks/Accelerate.framework/Accelerate')
    _sgemm = _lib.cblas_sgemm
    _sgemm.restype = None
    _sgemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
                       ctypes.c_void_p, ctypes.c_int,
                       ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
    HAVE_ACCELERATE = True
except OSError:
    HAVE_ACCELERATE = False


DT = np.float32


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _vforce(name):
    """Bind an Accelerate vForce function, or None if unavailable.

    Softmax needs an exp per logit and the logit softcap a tanh per logit.
    On an install whose numpy has no SIMD dispatch those run one element at
    a time in scalar C. vForce is the vectorised version and ships with
    Accelerate, so it costs one more ctypes binding and no new dependency.

    With --act relu2 the activation no longer needs a transcendental at
    all, which removes the largest consumer of this binding.
    """
    if not HAVE_ACCELERATE:
        return None
    try:
        f = getattr(_lib, name)
    except AttributeError:
        return None
    f.restype = None
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.POINTER(ctypes.c_int)]
    return f


# vForce adds a ctypes call and a possible copy per elementwise op. Whether
# that pays for itself at these array sizes is unconfirmed on-device, so it
# is a switch rather than an assumption:  TINY_GPT_VFORCE=0 to A/B it.
USE_VFORCE = os.environ.get('TINY_GPT_VFORCE', '1') != '0'

_vvtanhf = _vforce('vvtanhf') if USE_VFORCE else None
_vvexpf = _vforce('vvexpf') if USE_VFORCE else None
_vvsqrtf = _vforce('vvsqrtf') if USE_VFORCE else None

# Whether the vForce transcendentals actually bound. Exposed so train.py /
# diagnostics.py can report it without reaching into the private _vv* names.
HAVE_VFORCE = _vvtanhf is not None


def _vf_apply(fn, np_fn, x, out=None):
    if fn is None or not USE_VFORCE or x.dtype != np.float32:
        return np_fn(x, out=out) if out is not None else np_fn(x)
    x = np.ascontiguousarray(x, dtype=np.float32)
    if (out is None or out.shape != x.shape or out.dtype != np.float32
            or not out.flags.c_contiguous):
        out = np.empty_like(x)
    n = ctypes.c_int(x.size)
    fn(_p(out), _p(x), ctypes.byref(n))
    return out


def vtanh(x, out=None):
    return _vf_apply(_vvtanhf, np.tanh, x, out)


def vexp(x, out=None):
    return _vf_apply(_vvexpf, np.exp, x, out)


def vsqrt(x, out=None):
    return _vf_apply(_vvsqrtf, np.sqrt, x, out)


def set_dtype(d):
    '''Switch model precision. cblas_sgemm is float32-only, so float64 mode
    silently routes every GEMM through numpy. Used by gradcheck, where
    float32 loss resolution (~1e-7 on a loss of ~2.4) divided by 2*eps
    swamps any gradient smaller than ~1e-2 and makes the check measure its
    own noise floor rather than the derivative.'''
    global DT
    DT = d


def gemm(A, B, C, transA=False, transB=False, beta=0.0):
    '''C = op(A) @ op(B) + beta*C, row-major float32, written in place.

    For row-major CBLAS the leading dimension is always the stored column
    count regardless of the transpose flag; only logical M/N/K change.
    '''
    if not HAVE_ACCELERATE or A.dtype != np.float32:
        a = A.T if transA else A
        b = B.T if transB else B
        if beta == 0.0:
            np.matmul(a, b, out=C)
        else:
            C *= beta
            C += a @ b
        return
    M, N = C.shape
    K = A.shape[0] if transA else A.shape[1]
    _sgemm(CblasRowMajor,
           CblasTrans if transA else CblasNoTrans,
           CblasTrans if transB else CblasNoTrans,
           M, N, K, 1.0,
           _p(A), A.shape[1], _p(B), B.shape[1],
           beta, _p(C), C.shape[1])


# Attention is ~7.6% of arithmetic but ~98% of GEMM call count: thousands
# of tiny per-head matmuls per step. 'accelerate' pays ctypes overhead per
# call but each runs on AMX; 'numpy' issues one batched call but gets no
# BLAS. Measured on-device: accelerate wins by 2.3x once Accelerate exists
# at all (356ms -> 154ms/step). Defaulting to a static string was the bug --
# every time this file gets regenerated, the setting reverts and silently
# undoes whatever was measured on the actual device. Auto-detect instead.
# TINY_GPT_BMM=numpy to A/B the attention backend without editing this.
BMM_BACKEND = os.environ.get(
    'TINY_GPT_BMM', 'accelerate' if HAVE_ACCELERATE else 'numpy')


def bmm(A, B, C, transA=False, transB=False):
    '''Batched GEMM over the leading axis of three 3-D arrays.'''
    if (BMM_BACKEND == 'numpy' or not HAVE_ACCELERATE
            or A.dtype != np.float32):
        a = A.transpose(0, 2, 1) if transA else A
        b = B.transpose(0, 2, 1) if transB else B
        np.matmul(a, b, out=C)
        return
    for i in range(C.shape[0]):
        gemm(A[i], B[i], C[i], transA, transB)
