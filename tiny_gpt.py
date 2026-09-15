# coding: utf-8
'''
train big.txt --steps 3700 --block-size 256 --n-embd 128 --n-layer 4 --seed 42 --ckpt base192.npz --rope 0 --qk-norm 0 --act gelu --zero-init 0 --value-residual 0 --softcap 0


A character-level GPT trained from scratch, on an iPhone, in Pythonista.

Real backprop. Real multi-head causal self-attention. No autograd, no
PyTorch, no pretrained weights. Every matmul goes through cblas_sgemm on
Apple's AMX coprocessor, measured at 1763 GFLOP/s on an A18 Pro.

    python tiny_gpt.py gradcheck       finite-difference every gradient
    python tiny_gpt.py train [file]    train, then sample
    python tiny_gpt.py bench           throughput by phase
    python tiny_gpt.py profile         per-op timing

Six architecture levers, each independently switchable, all ON by default:

    --rope 1            rotary position embeddings instead of learned wpe
    --qk-norm 1         non-parametric RMSNorm on q and k, after RoPE
    --act relu2         squared ReLU instead of tanh-GELU
    --zero-init 1       output projections and head start at zero
    --value-residual 1  per-layer learnable shortcut to layer-0 values
    --softcap 30        c*tanh(logits/c) before cross-entropy

Set any of them to 0 (or --act gelu) to A/B a single change against the
baseline. All six off reproduces the previous architecture exactly, and
version-1 checkpoints load that way automatically.

Falls back to numpy when Accelerate is absent, so the same file runs and
verifies off-device. Only the GEMM binding differs.
'''

import ctypes
import json
import math
import os
import sys
import time

import numpy as np

#'', '', args hack

# ----------------------------------------------------------------- Accelerate

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


# --------------------------------------------------------------------- pieces

_GELU_C = 0.7978845608028654  # sqrt(2/pi)


def gelu(x):
    return gelu_cached(x)[0]


def gelu_cached(x):
    """GELU, also returning the tanh it computed.

    Note x*x*x rather than x**3. numpy fast-paths the squaring operator but
    not the cubing one, so x**3 falls through to a general per-element pow:
    measured 64.34 ms against 0.76 ms for the explicit product on a
    2048x512 float32 array, an 85x difference. np.tanh on the same array is
    0.41 ms. Profiling attributed 67% of a training step to GELU and the
    exponent operator was essentially all of it -- the transcendental was
    never the problem.

    The tanh is returned so the backward pass can reuse it instead of
    recomputing the identical value from the same input.
    """
    x3 = x * x * x
    t = vtanh(_GELU_C * (x + 0.044715 * x3))
    return 0.5 * x * (1.0 + t), t


def dgelu_from(x, t):
    """GELU derivative, given the tanh cached by the forward pass."""
    dinner = _GELU_C * (1.0 + 3.0 * 0.044715 * (x * x))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner


def dgelu(x):
    x3 = x * x * x
    return dgelu_from(x, vtanh(_GELU_C * (x + 0.044715 * x3)))


def act_fwd(x, kind):
    """MLP activation. Returns (output, cache-for-backward).

    relu2 is max(x,0)**2, written as r*r for the same reason x**3 was
    avoided above. It costs one compare and one multiply per element
    against GELU's cube plus tanh, and its cache is the ReLU output rather
    than a separately computed tanh.
    """
    if kind == 'relu2':
        r = np.maximum(x, 0.0).astype(DT)
        return (r * r).astype(DT), r
    return gelu_cached(x)


def act_bwd(dout, x, cache, kind):
    """d/dx of act_fwd. For relu2: d(relu(x)^2)/dx = 2*relu(x)."""
    if kind == 'relu2':
        return (dout * (2.0 * cache)).astype(DT)
    return (dout * dgelu_from(x, cache)).astype(DT)


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = xc / std
    return (xhat * g + b).astype(DT), xhat, std


def dlayernorm(dy, xhat, std, g):
    C = xhat.shape[-1]
    dg = (dy * xhat).reshape(-1, C).sum(axis=0)
    db = dy.reshape(-1, C).sum(axis=0)
    dxhat = dy * g
    dx = (dxhat
          - dxhat.mean(axis=-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)) / std
    return dx.astype(DT), dg.astype(DT), db.astype(DT)


def rms_norm(x, eps=1e-6):
    """Non-parametric RMSNorm over the last axis. Returns (out, scale).

    No learnable gain: QK-norm exists to bound the attention logits, and a
    per-head gain would just reintroduce the degree of freedom that lets
    them grow. Keeping it parameter-free also keeps the parameter count and
    the gradcheck table unchanged.
    """
    ms = (x * x).mean(axis=-1, keepdims=True)
    s = np.sqrt(ms + eps)
    return (x / s).astype(DT), s.astype(DT)


def drms_norm(dr, r, s):
    """Backward of rms_norm, in terms of its own output.

    r = x/s with s = sqrt(mean(x^2)+eps), so
        dx = (dr - r*mean(dr*r)) / s
    which needs the output and the scale but not the input.
    """
    return ((dr - r * (dr * r).mean(axis=-1, keepdims=True)) / s).astype(DT)


# ------------------------------------------------------------------- rotary

_ROPE = {}


def rope_tables(hs, n, base=10000.0):
    """cos/sin for positions [0,n) and half-dimension hs/2, memoised.

    Kept in float64 and cast at use, so gradcheck's float64 mode and the
    float32 training path share one table. Grown, never shrunk: cached
    generation walks past block_size and keeps asking for larger n.
    """
    c = _ROPE.get(hs)
    if c is not None and c[2] >= n:
        return c[0], c[1]
    n2 = max(int(n * 1.5) + 64, 512)
    half = hs // 2
    inv = 1.0 / (base ** (np.arange(half, dtype=np.float64) * 2.0 / hs))
    ang = np.arange(n2, dtype=np.float64)[:, None] * inv[None, :]
    cos = np.ascontiguousarray(np.cos(ang))
    sin = np.ascontiguousarray(np.sin(ang))
    _ROPE[hs] = (cos, sin, n2)
    return cos, sin


def _rope_cs(hs, pos0, T):
    cos, sin = rope_tables(hs, pos0 + T)
    c = np.ascontiguousarray(cos[pos0:pos0 + T], dtype=DT)[None]
    s = np.ascontiguousarray(sin[pos0:pos0 + T], dtype=DT)[None]
    return c, s


def rope_apply(x, pos0=0):
    """Rotate the first and second halves of each head vector as a pair.

    x is (N, T, hs). The attention logit between positions i and j then
    depends only on i-j, which is what makes the KV cache sliceable: past
    block_size the cache drops its oldest entries and the surviving
    relative offsets are still exactly the ones training saw. A learned
    absolute wpe cannot do that -- it forces the cache to be rebuilt.
    """
    hs = x.shape[-1]
    half = hs // 2
    c, s = _rope_cs(hs, pos0, x.shape[1])
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * c - x2 * s
    out[..., half:] = x1 * s + x2 * c
    return out


def rope_bwd(d, pos0=0):
    """Transpose of rope_apply: the same rotation by the negated angle."""
    hs = d.shape[-1]
    half = hs // 2
    c, s = _rope_cs(hs, pos0, d.shape[1])
    d1 = d[..., :half]
    d2 = d[..., half:]
    out = np.empty_like(d)
    out[..., :half] = d1 * c + d2 * s
    out[..., half:] = -d1 * s + d2 * c
    return out


def softmax_rows(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = vexp(np.ascontiguousarray(z, dtype=DT))
    return (e / e.sum(axis=-1, keepdims=True)).astype(DT)


# ---------------------------------------------------------------------- model

class Config(object):

    def __init__(self, vocab, n_layer=4, n_head=4, n_embd=128,
                 block_size=64, rope=True, qk_norm=True, act='relu2',
                 zero_init=True, value_residual=True, softcap=30.0):
        self.vocab = vocab
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head

        self.rope = bool(rope)
        self.qk_norm = bool(qk_norm)
        self.act = str(act)
        self.zero_init = bool(zero_init)
        self.value_residual = bool(value_residual) and n_layer > 1
        self.softcap = float(softcap or 0.0)

        assert self.act in ('gelu', 'relu2')
        if self.rope:
            assert self.head_size % 2 == 0, 'rope needs an even head size'

    def flags(self):
        return {'rope': self.rope, 'qk_norm': self.qk_norm, 'act': self.act,
                'zero_init': self.zero_init,
                'value_residual': self.value_residual,
                'softcap': self.softcap}

    def describe(self):
        return ('rope=%s qk_norm=%s act=%s zero_init=%s value_residual=%s '
                'softcap=%g' % (self.rope, self.qk_norm, self.act,
                                self.zero_init, self.value_residual,
                                self.softcap))


LEGACY_FLAGS = {'rope': False, 'qk_norm': False, 'act': 'gelu',
                'zero_init': False, 'value_residual': False, 'softcap': 0.0}


class GPT(object):

    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        rng = np.random.RandomState(seed)
        C, L = cfg.n_embd, cfg.n_layer
        f = lambda *s: (rng.randn(*s) * 0.02).astype(DT)
        z = lambda *s: np.zeros(s, dtype=DT)
        o = lambda *s: np.ones(s, dtype=DT)

        # Residual projections and the classifier either start at zero
        # (muP-like: every block is an exact identity at init and the
        # residual stream carries only the embedding) or at the GPT-2
        # 1/sqrt(2L) scale, which keeps the residual variance from growing
        # with depth.
        s = 0.02 / math.sqrt(2 * L)
        out = (lambda *sh: z(*sh)) if cfg.zero_init else \
            (lambda *sh: (rng.randn(*sh) * s).astype(DT))

        self.p = {
            'wte': f(cfg.vocab, C),
            'lnf_g': o(C), 'lnf_b': z(C),
            'head': z(C, cfg.vocab) if cfg.zero_init else f(C, cfg.vocab),
        }
        if not cfg.rope:
            self.p['wpe'] = f(cfg.block_size, C)
        if cfg.value_residual:
            # One scalar per layer above the first. Zero-initialised, so at
            # step 0 this lever is a bit-exact no-op against the same model
            # without it: it can only earn its keep as training moves it.
            self.p['vres'] = z(L - 1)

        for i in range(L):
            self.p['ln1_g%d' % i] = o(C)
            self.p['ln1_b%d' % i] = z(C)
            self.p['qkv_w%d' % i] = f(C, 3 * C)
            self.p['qkv_b%d' % i] = z(3 * C)
            self.p['proj_w%d' % i] = out(C, C)
            self.p['proj_b%d' % i] = z(C)
            self.p['ln2_g%d' % i] = o(C)
            self.p['ln2_b%d' % i] = z(C)
            self.p['fc_w%d' % i] = f(C, 4 * C)
            self.p['fc_b%d' % i] = z(4 * C)
            self.p['fp_w%d' % i] = out(4 * C, C)
            self.p['fp_b%d' % i] = z(C)

        self.n_params = sum(v.size for v in self.p.values())

        # causal mask, built once
        m = np.tril(np.ones((cfg.block_size, cfg.block_size), DT))
        self.neg = np.where(m == 0, -1e9, 0.0).astype(DT)

    # -- forward ------------------------------------------------------------

    def forward(self, idx, targets=None):
        cfg = self.cfg
        p = self.p
        B, T = idx.shape
        C, nh, hs = cfg.n_embd, cfg.n_head, cfg.head_size
        BT = B * T
        ca = {'idx': idx, 'B': B, 'T': T}

        x = p['wte'][idx].astype(DT)
        if not cfg.rope:
            x = (x + p['wpe'][:T]).astype(DT)
        ca['blocks'] = []
        v0 = None

        for i in range(cfg.n_layer):
            bc = {}
            h, xhat1, std1 = layernorm(x, p['ln1_g%d' % i], p['ln1_b%d' % i])
            bc['xhat1'], bc['std1'] = xhat1, std1

            h2 = np.ascontiguousarray(h.reshape(BT, C))
            qkv = np.empty((BT, 3 * C), DT)
            gemm(h2, p['qkv_w%d' % i], qkv)
            qkv += p['qkv_b%d' % i]
            bc['h2'] = h2

            q = np.ascontiguousarray(
                qkv[:, :C].reshape(B, T, nh, hs).transpose(0, 2, 1, 3)
                ).reshape(B * nh, T, hs)
            k = np.ascontiguousarray(
                qkv[:, C:2 * C].reshape(B, T, nh, hs).transpose(0, 2, 1, 3)
                ).reshape(B * nh, T, hs)
            v = np.ascontiguousarray(
                qkv[:, 2 * C:].reshape(B, T, nh, hs).transpose(0, 2, 1, 3)
                ).reshape(B * nh, T, hs)

            if cfg.rope:
                q = rope_apply(q)
                k = rope_apply(k)
            if cfg.qk_norm:
                q, bc['qs'] = rms_norm(q)
                k, bc['ks'] = rms_norm(k)

            if i == 0:
                v0 = v
            if cfg.value_residual and i > 0:
                vu = (v + p['vres'][i - 1] * v0).astype(DT)
            else:
                vu = v

            att = np.empty((B * nh, T, T), DT)
            bmm(q, k, att, transB=True)
            att *= (1.0 / math.sqrt(hs))
            att += self.neg[:T, :T]
            pr = softmax_rows(att)

            y = np.empty((B * nh, T, hs), DT)
            bmm(pr, vu, y)
            y = np.ascontiguousarray(
                y.reshape(B, nh, T, hs).transpose(0, 2, 1, 3)).reshape(BT, C)
            bc['q'], bc['k'], bc['v'], bc['vu'] = q, k, v, vu
            bc['pr'], bc['y'] = pr, y

            a = np.empty((BT, C), DT)
            gemm(y, p['proj_w%d' % i], a)
            a += p['proj_b%d' % i]
            x = (x + a.reshape(B, T, C)).astype(DT)

            h, xhat2, std2 = layernorm(x, p['ln2_g%d' % i], p['ln2_b%d' % i])
            bc['xhat2'], bc['std2'] = xhat2, std2
            h3 = np.ascontiguousarray(h.reshape(BT, C))
            bc['h3'] = h3

            fc = np.empty((BT, 4 * C), DT)
            gemm(h3, p['fc_w%d' % i], fc)
            fc += p['fc_b%d' % i]
            act, acache = act_fwd(fc, cfg.act)
            bc['fc'], bc['act'], bc['ac'] = fc, act, acache

            m = np.empty((BT, C), DT)
            gemm(act, p['fp_w%d' % i], m)
            m += p['fp_b%d' % i]
            x = (x + m.reshape(B, T, C)).astype(DT)

            bc['x_out'] = x
            ca['blocks'].append(bc)

        xf, xhatf, stdf = layernorm(x, p['lnf_g'], p['lnf_b'])
        ca['xhatf'], ca['stdf'] = xhatf, stdf
        xf2 = np.ascontiguousarray(xf.reshape(BT, C))
        ca['xf2'] = xf2

        logits = np.empty((BT, cfg.vocab), DT)
        gemm(xf2, p['head'], logits)

        if cfg.softcap:
            # c*tanh(z/c): bounded logits, so a single overconfident
            # character cannot dominate the gradient. Monotonic, so top-k
            # ordering at sampling time is unchanged.
            c = cfg.softcap
            t = vtanh(np.ascontiguousarray(logits / c, dtype=DT))
            ca['capt'] = t
            logits = (c * t).astype(DT)
        ca['logits'] = logits

        if targets is None:
            return logits.reshape(B, T, cfg.vocab), ca

        t = targets.reshape(-1)

        mx = logits.max(axis=1, keepdims=True)

        # Compute exp only once. vexp uses Accelerate/vForce in float32.
        z = np.ascontiguousarray(logits - mx, dtype=DT)
        e = vexp(z)

        den = e.sum(axis=1, keepdims=True)
        lse = mx + np.log(den)

        loss = float(np.mean(
          lse[:, 0] - logits[np.arange(BT), t]
        ))

        # Cache the softmax probabilities for cross-entropy backward.
        # This trades ~BT*vocab*4 bytes of cache for eliminating another exp().
        ca['xent_pr'] = (e / den).astype(DT)
        ca['lse'] = lse
        ca['t'] = t

        return loss, ca

    # -- backward -----------------------------------------------------------

    def backward(self, ca):
        cfg = self.cfg
        p = self.p
        B, T = ca['B'], ca['T']
        C, nh, hs = cfg.n_embd, cfg.n_head, cfg.head_size
        BT = B * T
        g = {k: np.zeros_like(v) for k, v in p.items()}

        dlogits = ca['xent_pr'].copy()
        dlogits[np.arange(BT), ca['t']] -= 1.0
        dlogits /= BT

        if cfg.softcap:
            # d/dz of c*tanh(z/c) is 1 - tanh(z/c)^2, and the forward pass
            # already cached that tanh.
            t = ca['capt']
            dlogits = (dlogits * (1.0 - t * t)).astype(DT)

        gemm(ca['xf2'], dlogits, g['head'], transA=True)
        dxf = np.empty((BT, C), DT)
        gemm(dlogits, p['head'], dxf, transB=True)

        dx, g['lnf_g'], g['lnf_b'] = dlayernorm(
            dxf.reshape(B, T, C), ca['xhatf'], ca['stdf'], p['lnf_g'])

        v0 = ca['blocks'][0]['v']
        dv0_acc = np.zeros_like(v0) if cfg.value_residual else None

        for i in reversed(range(cfg.n_layer)):
            bc = ca['blocks'][i]

            dm = np.ascontiguousarray(dx.reshape(BT, C))
            g['fp_b%d' % i] = dm.sum(axis=0)
            gemm(bc['act'], dm, g['fp_w%d' % i], transA=True)
            dact = np.empty((BT, 4 * C), DT)
            gemm(dm, p['fp_w%d' % i], dact, transB=True)

            dfc = act_bwd(dact, bc['fc'], bc['ac'], cfg.act)
            g['fc_b%d' % i] = dfc.sum(axis=0)
            gemm(bc['h3'], dfc, g['fc_w%d' % i], transA=True)
            dh3 = np.empty((BT, C), DT)
            gemm(dfc, p['fc_w%d' % i], dh3, transB=True)

            dres, g['ln2_g%d' % i], g['ln2_b%d' % i] = dlayernorm(
                dh3.reshape(B, T, C), bc['xhat2'], bc['std2'],
                p['ln2_g%d' % i])
            dx = dx + dres

            da = np.ascontiguousarray(dx.reshape(BT, C))
            g['proj_b%d' % i] = da.sum(axis=0)
            gemm(bc['y'], da, g['proj_w%d' % i], transA=True)
            dy = np.empty((BT, C), DT)
            gemm(da, p['proj_w%d' % i], dy, transB=True)

            dy = np.ascontiguousarray(
                dy.reshape(B, T, nh, hs).transpose(0, 2, 1, 3)
                ).reshape(B * nh, T, hs)

            pr, q, k, vu = bc['pr'], bc['q'], bc['k'], bc['vu']
            dvu = np.empty_like(vu)
            bmm(pr, dy, dvu, transA=True)
            dpr = np.empty_like(pr)
            bmm(dy, vu, dpr, transB=True)

            datt = pr * (dpr - (dpr * pr).sum(axis=-1, keepdims=True))
            datt = (datt * (1.0 / math.sqrt(hs))).astype(DT)

            dq = np.empty_like(q)
            bmm(datt, k, dq)
            dk = np.empty_like(k)
            bmm(datt, q, dk, transA=True)

            # vu = v + lam*v0, so dv is dvu unchanged and the scalar picks
            # up the full inner product with layer 0's values. The dv0 term
            # is held until the i==0 iteration, where v0 is actually born.
            if cfg.value_residual and i > 0:
                g['vres'][i - 1] = float(np.sum(dvu * v0))
                dv0_acc += (p['vres'][i - 1] * dvu).astype(dv0_acc.dtype)
                dv = dvu
            elif cfg.value_residual and i == 0:
                dv = dvu + dv0_acc
            else:
                dv = dvu

            # Unwind q and k through QK-norm, then through the rotation.
            if cfg.qk_norm:
                dq = drms_norm(dq, q, bc['qs'])
                dk = drms_norm(dk, k, bc['ks'])
            if cfg.rope:
                dq = rope_bwd(dq)
                dk = rope_bwd(dk)

            def unhead(a):
                return np.ascontiguousarray(
                    a.reshape(B, nh, T, hs).transpose(0, 2, 1, 3)
                    ).reshape(BT, C)

            dqkv = np.concatenate([unhead(dq), unhead(dk), unhead(dv)],
                                  axis=1).astype(DT)
            g['qkv_b%d' % i] = dqkv.sum(axis=0)
            gemm(bc['h2'], dqkv, g['qkv_w%d' % i], transA=True)
            dh2 = np.empty((BT, C), DT)
            gemm(dqkv, p['qkv_w%d' % i], dh2, transB=True)

            dres, g['ln1_g%d' % i], g['ln1_b%d' % i] = dlayernorm(
                dh2.reshape(B, T, C), bc['xhat1'], bc['std1'],
                p['ln1_g%d' % i])
            dx = dx + dres

        np.add.at(g['wte'], ca['idx'], dx)
        if not cfg.rope:
            g['wpe'][:T] = dx.sum(axis=0)
        return g

    # -- cached generation --------------------------------------------------

    def _step_cached(self, tok, pos, kv, window=None):
        """One token through the network, reusing cached keys/values.

        Naive sampling recomputes the whole context for every new token, so
        generating n tokens costs O(n*T) forwards. Here each layer keeps its
        k and v for all previous positions, the new token contributes one
        row to each, and attention is a single (1 x t+1) row per head. No
        causal mask is needed: everything in the cache is already past.

        With RoPE the key is rotated once, on the way in, and `window`
        trims the oldest entries in place. Relative offsets in the surviving
        window are unchanged, so this is exact sliding-window attention
        rather than the approximation the absolute-wpe path has to make.
        """
        cfg = self.cfg
        p = self.p
        C, nh, hs = cfg.n_embd, cfg.n_head, cfg.head_size

        x = p['wte'][tok].reshape(1, C).astype(DT)
        if not cfg.rope:
            x = (x + p['wpe'][pos]).astype(DT)
        v0 = None

        for i in range(cfg.n_layer):
            h, _, _ = layernorm(x, p['ln1_g%d' % i], p['ln1_b%d' % i])
            qkv = np.empty((1, 3 * C), DT)
            gemm(np.ascontiguousarray(h), p['qkv_w%d' % i], qkv)
            qkv += p['qkv_b%d' % i]

            q = qkv[0, :C].reshape(nh, 1, hs)
            k_new = qkv[0, C:2 * C].reshape(nh, 1, hs)
            v_new = qkv[0, 2 * C:].reshape(nh, 1, hs)

            if cfg.rope:
                q = rope_apply(q, pos)
                k_new = rope_apply(k_new, pos)
            if cfg.qk_norm:
                q, _ = rms_norm(q)
                k_new, _ = rms_norm(k_new)

            if i == 0:
                v0 = v_new
            if cfg.value_residual and i > 0:
                v_new = (v_new + p['vres'][i - 1] * v0).astype(DT)

            if kv[i][0] is None:
                kc, vc = k_new.copy(), v_new.copy()
            else:
                kc = np.concatenate([kv[i][0], k_new], axis=1)
                vc = np.concatenate([kv[i][1], v_new], axis=1)
            if window and kc.shape[1] > window:
                kc = np.ascontiguousarray(kc[:, -window:])
                vc = np.ascontiguousarray(vc[:, -window:])
            kv[i] = (kc, vc)

            att = np.matmul(q, kc.transpose(0, 2, 1)) * (1.0 / math.sqrt(hs))
            pr = softmax_rows(att)
            y = np.matmul(pr, vc).reshape(1, C)

            a = np.empty((1, C), DT)
            gemm(np.ascontiguousarray(y), p['proj_w%d' % i], a)
            a += p['proj_b%d' % i]
            x = x + a

            h, _, _ = layernorm(x, p['ln2_g%d' % i], p['ln2_b%d' % i])
            fc = np.empty((1, 4 * C), DT)
            gemm(np.ascontiguousarray(h), p['fc_w%d' % i], fc)
            fc += p['fc_b%d' % i]
            act, _ = act_fwd(fc, cfg.act)
            m = np.empty((1, C), DT)
            gemm(np.ascontiguousarray(act), p['fp_w%d' % i], m)
            m += p['fp_b%d' % i]
            x = x + m

        xf, _, _ = layernorm(x, p['lnf_g'], p['lnf_b'])
        logits = np.empty((1, cfg.vocab), DT)
        gemm(np.ascontiguousarray(xf), p['head'], logits)
        if cfg.softcap:
            c = cfg.softcap
            logits = (c * vtanh(np.ascontiguousarray(logits / c, dtype=DT))
                      ).astype(DT)
        return logits[0]

    def generate(self, stoi, itos, prompt='\n', n=400, temp=0.8, top_k=40,
                 top_p=None, seed=0, stop=None, on_token=None):
        """Sample with a KV cache. Returns the generated text.

        Verified against the uncached sample(): logits agree to ~1e-07 at
        every position, and sampled sequences are byte-identical for the
        first block_size tokens. Measured 12x faster at n_embd=128,
        block_size=256.

        Past block_size the behaviour depends on the position encoding.
        With RoPE the cache is simply trimmed and generation stays exact:
        attention depends on relative offset, and every surviving offset is
        one training saw. With a learned absolute wpe the cache has to be
        rebuilt in chunks, which is cheaper than sample()'s per-token
        recompute but leaves the far end of the context up to block_size
        tokens stale. Use sample() when exact per-token sliding matters
        more than speed.
        """
        cfg = self.cfg
        rng = np.random.RandomState(seed)
        ids = [stoi[c] for c in prompt if c in stoi] or [0]
        win = cfg.block_size if cfg.rope else None

        kv = [(None, None) for _ in range(cfg.n_layer)]
        logits = None
        for j, t in enumerate(ids):
            logits = self._step_cached(t, j, kv, window=win)
        pos = len(ids)
        if stop:
            stops = (stop,) if isinstance(stop, str) else tuple(stop)
        out = []
        for _ in range(n):
            z = logits.astype(np.float64) / max(temp, 1e-6)
            if top_k and top_k < z.size:
                cut = np.partition(z, -top_k)[-top_k]
                z = np.where(z < cut, -np.inf, z)
            z = z - z.max()
            pr = np.exp(z)
            pr /= pr.sum()
            if top_p is not None and 0 < top_p < 1:
                order = np.argsort(-pr)
                keep = np.cumsum(pr[order]) <= top_p
                keep[0] = True
                mask = np.zeros_like(pr, dtype=bool)
                mask[order[keep]] = True
                pr = np.where(mask, pr, 0.0)
                pr /= pr.sum()

            nxt = int(rng.choice(len(pr), p=pr))
            ch = itos[nxt]
            out.append(ch)
            if on_token:
                on_token(ch)
            if stop and ''.join(out).endswith(tuple(stop)):
                break

            ids.append(nxt)
            if cfg.rope:
                logits = self._step_cached(nxt, pos, kv, window=win)
                pos += 1
            elif pos >= cfg.block_size:
                ids = ids[-(cfg.block_size - 1):]
                kv = [(None, None) for _ in range(cfg.n_layer)]
                for j, t in enumerate(ids):
                    logits = self._step_cached(t, j, kv)
                pos = len(ids)
            else:
                logits = self._step_cached(nxt, pos, kv)
                pos += 1

        return ''.join(out)

    # -- sampling -----------------------------------------------------------

    def sample(self, stoi, itos, prompt='\n', n=400, temp=0.8, top_k=40,
               seed=0):
        rng = np.random.RandomState(seed)
        ids = [stoi.get(c, 0) for c in prompt]
        out = list(prompt)
        for _ in range(n):
            ctx = ids[-self.cfg.block_size:]
            idx = np.array([ctx], dtype=np.int64)
            logits, _ = self.forward(idx)
            z = logits[0, len(ctx) - 1].astype(np.float64) / max(temp, 1e-6)
            if top_k and top_k < z.size:
                cut = np.partition(z, -top_k)[-top_k]
                z = np.where(z < cut, -np.inf, z)
            z -= z.max()
            pr = np.exp(z)
            pr /= pr.sum()
            nxt = int(rng.choice(len(pr), p=pr))
            ids.append(nxt)
            out.append(itos[nxt])
        return ''.join(out)


# ------------------------------------------------------------- checkpoints

CKPT_VERSION = 2


def save_checkpoint(path, model, stoi, step=0, opt=None, val_loss=None,
                    sched=None):
    """Write params, vocab, config, arch flags and optimizer state to .npz.

    The vocabulary must travel with the weights: it is built from
    sorted(set(text)) of whatever corpus was used, so a different corpus
    yields different integer ids and the embedding rows stop meaning
    anything. Storing it removes the chance of silently loading a model
    against the wrong mapping.

    The architecture flags travel for the same reason -- a RoPE checkpoint
    has no wpe and a value-residual checkpoint has an extra parameter, so
    the flags have to be known before GPT() is constructed.

    `sched` is (k, total): position within the LR schedule. Without it a
    resume re-runs warmup on a converged model and validation jumps before
    recovering.
    """
    blob = {}
    for k, v in model.p.items():
        blob['p__' + k] = v
    if opt is not None:
        for k, v in opt.m.items():
            blob['m__' + k] = v
        for k, v in opt.v.items():
            blob['v__' + k] = v
        blob['opt_t'] = np.array(opt.t)
    cfg = model.cfg
    meta = {
        'version': CKPT_VERSION,
        'vocab': cfg.vocab, 'n_layer': cfg.n_layer, 'n_head': cfg.n_head,
        'n_embd': cfg.n_embd, 'block_size': cfg.block_size,
        'step': int(step), 'val_loss': val_loss,
        'n_params': int(model.n_params),
        'stoi': {c: int(i) for c, i in stoi.items()},
        'arch': cfg.flags(),
        'sched': list(sched) if sched else None,
    }
    blob['meta'] = np.frombuffer(
        json.dumps(meta).encode('utf-8'), dtype=np.uint8)
    np.savez_compressed(path, **blob)
    return path


def load_checkpoint(path, with_optimizer=False):
    """Return (model, stoi, itos, meta[, opt]).

    Version-1 checkpoints predate the arch flags and are loaded with every
    lever off, which is exactly the architecture they were trained with.
    That keeps the 1.2521-val model.npz loadable by this file.
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z['meta']).decode('utf-8'))
    ver = meta.get('version')
    if ver not in (1, 2):
        raise SystemExit('checkpoint version %r, expected 1 or %d'
                         % (ver, CKPT_VERSION))
    arch = dict(LEGACY_FLAGS)
    arch.update(meta.get('arch') or {})
    cfg = Config(meta['vocab'], meta['n_layer'], meta['n_head'],
                 meta['n_embd'], meta['block_size'], **arch)
    model = GPT(cfg)
    for k in list(model.p):
        key = 'p__' + k
        if key not in z:
            raise SystemExit('checkpoint missing parameter %r' % k)
        model.p[k] = np.ascontiguousarray(z[key], dtype=DT)
    stoi = {c: int(i) for c, i in meta['stoi'].items()}
    itos = {i: c for c, i in stoi.items()}
    if not with_optimizer:
        return model, stoi, itos, meta
    opt = Adam(model.p)
    if 'opt_t' in z:
        for k in model.p:
            opt.m[k] = np.ascontiguousarray(z['m__' + k], dtype=DT)
            opt.v[k] = np.ascontiguousarray(z['v__' + k], dtype=DT)
        opt.t = int(z['opt_t'])
    return model, stoi, itos, meta, opt


# ------------------------------------------------------------------- Adam

class Adam(object):
    '''Adam with bias correction. SGD does not train transformers.'''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
    
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
    
        sqrt_bc2 = math.sqrt(bc2)
    
        # Equivalent to:
        #
        #   mh = m / bc1
        #   vh = v / bc2
        #   prm -= lr * mh / (sqrt(vh) + eps)
        #
        # but avoids materialising mh and vh.
        step_scale = lr * sqrt_bc2 / bc1
        eps_scaled = self.eps * sqrt_bc2
    
        for k, prm in params.items():
            gr = grads[k]
    
            if self.wd and prm.ndim >= 2:
                gr = gr + self.wd * prm
    
            m = self.m[k]
            v = self.v[k]
    
            m *= self.b1
            m += (1.0 - self.b1) * gr
    
            v *= self.b2
            v += (1.0 - self.b2) * (gr * gr)
    
            # tmp = sqrt(v)
            #
            # Then recycle tmp for the rest of the update instead of creating
            # sqrt(v), denominator, normalized moment, and update arrays.
            tmp = vsqrt(v)
            tmp += eps_scaled
    
            np.divide(m, tmp, out=tmp)
            tmp *= step_scale
    
            prm -= tmp

# -------------------------------------------------------------------- data

def load_corpus(path=None):
    if path and os.path.isfile(path):
        with open(path, 'r', errors='replace') as f:
            return f.read(), path
    chunks, names = [], []
    for fn in sorted(os.listdir('.')):
        if fn.endswith('.py') or fn.endswith('.txt'):
            try:
                with open(fn, 'r', errors='replace') as f:
                    chunks.append(f.read())
                names.append(fn)
            except Exception:
                pass
    if not chunks:
        raise SystemExit('no corpus: pass a text file as an argument')
    return '\n\n'.join(chunks), '%d files (%s)' % (len(names),
                                                   ', '.join(names[:4]))


def make_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def batcher(data, B, T, rng):
    while True:
        ix = rng.randint(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
        yield x, y


# ------------------------------------------------------------------- modes

def gradcheck(**flags):
    '''Finite differences against every analytic gradient, tiny config.

    Runs in float64. In float32 the loss carries ~1e-7 of resolution, so a
    central difference over eps=1e-3 has an absolute noise floor near 6e-5 --
    larger than the LayerNorm gain/bias gradients themselves (~2e-3), which
    made those entries appear to fail at 3-6e-2 relative when the analytic
    values were in fact correct.

    eps is 1e-5, not the 1e-6 used before. The same noise floor bites again
    one precision down: roundoff in a central difference is about
    |loss|*2^-52/(2*eps), which at eps=1e-6 is ~3e-10 absolute. The
    smallest gradients in the table (ln2_g under relu2, ~2e-4) are then
    only ~1e-6 away in relative terms, and the check reports its own noise
    as a failure. Central-difference error is roundoff/eps plus truncation
    times eps^2, minimised near (3*2^-52)^(1/3) ~ 1e-5, which drops the
    floor to ~1e-7 relative while truncation is still negligible.

    Zero-init is honoured for construction and then undone: with a zero
    head the gradient of every upstream parameter is exactly zero at step
    0, so the check would compare 0 against 0 and pass without testing
    anything. The output projections are re-randomised before checking.

    python tiny_gpt.py gradcheck --rope 0 --qk-norm 0 ...  to isolate a
    single lever's derivation.
    '''
    set_dtype(np.float64)
    rng = np.random.RandomState(0)
    cfg = Config(vocab=11, n_layer=3, n_head=2, n_embd=16, block_size=8,
                 **flags)
    model = GPT(cfg, seed=1)

    if cfg.zero_init:
        for name in model.p:
            if name.startswith(('proj_w', 'fp_w')) or name == 'head':
                model.p[name] = (rng.randn(*model.p[name].shape)
                                 * 0.02).astype(DT)
    if cfg.value_residual:
        # A zero lambda has a perfectly good gradient, but a nonzero one
        # also exercises the dv0 path back into layer 0.
        model.p['vres'] = (rng.randn(*model.p['vres'].shape)
                           * 0.3).astype(DT)

    B, T = 2, 6
    idx = rng.randint(0, cfg.vocab, size=(B, T)).astype(np.int64)
    tgt = rng.randint(0, cfg.vocab, size=(B, T)).astype(np.int64)

    loss, ca = model.forward(idx, tgt)
    grads = model.backward(ca)
    print('params %d   loss %.6f' % (model.n_params, loss))
    print('arch   %s' % cfg.describe())
    print('accelerate: %s\n' % HAVE_ACCELERATE)

    eps = 1e-5
    worst = 0.0
    worst_name = '-'
    print('%-12s %14s %14s %10s' % ('param', 'numeric', 'analytic', 'rel'))
    for name in sorted(model.p):
        a = model.p[name]
        flat = a.reshape(-1)
        gflat = grads[name].reshape(-1)
        j = int(np.argmax(np.abs(gflat)))
        old = float(flat[j])

        flat[j] = old + eps
        hi, _ = model.forward(idx, tgt)
        flat[j] = old - eps
        lo, _ = model.forward(idx, tgt)
        flat[j] = old

        num = (hi - lo) / (2 * eps)
        ana = float(gflat[j])
        rel = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
        if rel > worst:
            worst, worst_name = rel, name
        print('%-12s %14.6e %14.6e %10.2e' % (name, num, ana, rel))

    set_dtype(np.float32)
    print('\nworst relative %.3e (%s) -> %s'
          % (worst, worst_name, 'PASS' if worst < 1e-6 else 'FAIL'))
    print('(float64 forward + central differences)')
    return worst < 1e-6


def gradcheck_all():
    """gradcheck the baseline, each lever alone, and everything together.

    Nine runs, a few seconds total. A lever that is correct in isolation
    and wrong in combination is the failure mode this catches -- QK-norm
    and RoPE compose on the same tensor, and value residual reaches across
    layers.
    """
    base = dict(LEGACY_FLAGS)
    trials = [('baseline', {})]
    for k, v in (('rope', True), ('qk_norm', True), ('act', 'relu2'),
                 ('zero_init', True), ('value_residual', True),
                 ('softcap', 30.0)):
        trials.append((k, {k: v}))
    trials.append(('all', {'rope': True, 'qk_norm': True, 'act': 'relu2',
                           'zero_init': True, 'value_residual': True,
                           'softcap': 30.0}))
    results = []
    for name, over in trials:
        f = dict(base)
        f.update(over)
        print('\n' + '=' * 60)
        print('gradcheck: %s' % name)
        print('=' * 60)
        results.append((name, gradcheck(**f)))
    print('\n' + '-' * 60)
    for name, ok in results:
        print('%-16s %s' % (name, 'PASS' if ok else 'FAIL'))
    return all(ok for _, ok in results)


def estimate_loss(model, data, B, batches=20, seed=1234):
    """Mean loss over random batches. Used for the held-out split."""
    rng = np.random.RandomState(seed)
    T = model.cfg.block_size
    tot = 0.0
    for _ in range(batches):
        ix = rng.randint(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
        loss, _ = model.forward(x, y)
        tot += loss
    return tot / batches


def train(path=None, steps=2000, B=32, lr=2e-3, seed=0, block_size=None,
          n_layer=4, n_head=4, n_embd=128, ckpt='model.npz',
          ckpt_every=250, val_frac=0.05, resume=None, rope=1, qk_norm=1,
          act='relu2', zero_init=1, value_residual=1, softcap=30.0):
    text, src = load_corpus(path)
    stoi, itos = make_vocab(text)
    data = np.array([stoi[c] for c in text], dtype=np.int64)

    # Held-out tail. A char model on a few MB overfits readily, and train
    # loss alone will keep dropping while the model memorises the corpus.
    n_val = max(model_min_val(data, val_frac), 0)
    train_data, val_data = (data[:-n_val], data[-n_val:]) if n_val else \
        (data, None)

    if resume:
        model, stoi_r, itos_r, meta, opt = load_checkpoint(
            resume, with_optimizer=True)
        if stoi_r != stoi:
            raise SystemExit(
                'resume vocabulary differs from this corpus: the checkpoint '
                'was trained on a different text, so its embedding rows do '
                'not match these character ids. Train on the same corpus or '
                'start fresh.')
        stoi, itos = stoi_r, itos_r
        start = int(meta.get('step', 0))
        cfg = model.cfg
        # Continue the cosine where it stopped rather than re-warming a
        # converged model. The new total extends the old schedule by
        # however many steps were asked for.
        sch = meta.get('sched')
        k0 = int(sch[0]) if sch else 0
        total = k0 + steps
        print('resumed  %s at step %d (schedule %d/%d)'
              % (resume, start, k0, total))
    else:
        cfg = Config(vocab=len(stoi), n_layer=n_layer, n_head=n_head,
                     n_embd=n_embd, block_size=block_size or 64,
                     rope=rope, qk_norm=qk_norm, act=act,
                     zero_init=zero_init, value_residual=value_residual,
                     softcap=softcap)
        model = GPT(cfg, seed=seed)
        opt = Adam(model.p, lr=lr, weight_decay=0.01)
        start = 0
        k0, total = 0, steps

    print('corpus   %s' % src)
    print('chars    %d train / %d val   vocab %d'
          % (len(train_data), n_val, cfg.vocab))
    print('model    %d params  (%d layers, %d heads, %d dim, block %d)'
          % (model.n_params, cfg.n_layer, cfg.n_head, cfg.n_embd,
             cfg.block_size))
    print('arch     %s' % cfg.describe())
    print('tokens/step %d   accelerate %s   bmm %s   vForce %s'
          % (B * cfg.block_size, HAVE_ACCELERATE, BMM_BACKEND,
             _vvtanhf is not None))
    print('checkpoint %s every %d steps\n' % (ckpt, ckpt_every))

    rng = np.random.RandomState(seed)
    gen = batcher(train_data, B, cfg.block_size, rng)
    warmup = max(1, min(100, total // 10))
    best = float('inf')
    t0 = time.time()
    print('%-7s %10s %9s %9s %9s %9s' % ('step', 'lr',  'loss', 'val', 'ms/step',
                                    'elapsed'))
    for s in range(start, start + steps):
        j = s - start
        k = k0 + j
        cur = lr * (k + 1) / warmup if k < warmup else \
            lr * (0.1 + 0.9 * 0.5 *
                  (1 + math.cos(math.pi * (k - warmup) /
                                max(1, total - warmup))))
        x, y = next(gen)
        loss, ca = model.forward(x, y)
        grads = model.backward(ca)
        opt.step(model.p, grads, lr=cur)

        last = (j == steps - 1)
        if j % ckpt_every == 0 or last:
            vl = estimate_loss(model, val_data, B, batches=32) \
                if val_data is not None and len(val_data) > cfg.block_size \
                else float('nan')
            el = time.time() - t0
            print('%-7d %.4e %9.4f %9.4f %9.1f %8.1fs'
                  % (s, cur, loss, vl, el / (j + 1) * 1000.0, el))
            if not (vl == vl) or vl < best:       # nan-safe
                best = vl
                save_checkpoint(ckpt, model, stoi, step=s + 1, opt=opt,
                                val_loss=None if vl != vl else float(vl),
                                sched=(k + 1, total))
        elif j % 100 == 0:
            el = time.time() - t0
            print('%-7d %.4e %9.4f %9s %9.1f %8.1fs'
                  % (s, cur, loss, '-', el / (j + 1) * 1000.0, el))

    dt = time.time() - t0
    tok = steps * B * cfg.block_size
    print('\n%d steps, %d tokens in %.1f s  (%.0f tokens/s)'
          % (steps, tok, dt, tok / dt))
    print('%.1f GFLOP/s sustained end-to-end'
          % (6.0 * model.n_params * tok / dt / 1e9))
    print('saved    %s  (best val %.4f)' % (ckpt, best))

    print('\n' + '-' * 60)
    print(model.generate(stoi, itos, prompt='\n', n=600))
    print('-' * 60)


def model_min_val(data, frac):
    """Validation split size, but never more than a quarter of the data."""
    return int(min(len(data) * min(frac, 0.25), len(data) // 4))


def profile(steps=12, B=32, block_size=64, **flags):
    """Time each op category separately. Measurement, not estimation.

    The dense GEMMs run on AMX and the attention matmuls run on whichever
    bmm backend is selected, but everything else -- the activation,
    LayerNorm, softmax, RoPE, QK-norm, the head-splitting copies, Adam --
    is plain numpy. On an install with no SIMD dispatch that residue can
    dominate a step even though it is a rounding error in the FLOP count.
    """
    import collections
    text, _ = load_corpus(None)
    stoi, _ = make_vocab(text)
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    cfg = Config(vocab=len(stoi), block_size=block_size, **flags)
    model = GPT(cfg)
    opt = Adam(model.p)
    rng = np.random.RandomState(0)
    gen = batcher(data, B, cfg.block_size, rng)

    T = collections.OrderedDict()
    BT, C = B * cfg.block_size, cfg.n_embd
    nh, hs = cfg.n_head, cfg.head_size

    for _ in range(3):
        x, y = next(gen)
        opt.step(model.p, model.backward(model.forward(x, y)[1]))

    T.clear()
    for _ in range(steps):
        x, y = next(gen)
        t0 = time.time(); loss, ca = model.forward(x, y)
        T['forward'] = T.get('forward', 0) + time.time() - t0
        t0 = time.time(); g = model.backward(ca)
        T['backward'] = T.get('backward', 0) + time.time() - t0
        t0 = time.time(); opt.step(model.p, g)
        T['adam'] = T.get('adam', 0) + time.time() - t0

    # isolate the pure-numpy pieces at representative sizes
    a4 = np.ascontiguousarray(np.random.randn(BT, 4 * C).astype(DT))
    a1 = np.ascontiguousarray(np.random.randn(B, cfg.block_size, C).astype(DT))
    at = np.ascontiguousarray(
        np.random.randn(B * nh, cfg.block_size, cfg.block_size).astype(DT))
    qh = np.ascontiguousarray(
        np.random.randn(B * nh, cfg.block_size, hs).astype(DT))
    g1 = np.ascontiguousarray(np.random.randn(C).astype(DT))
    b1 = np.zeros(C, DT)
    reps = steps * cfg.n_layer

    def bulk(key, fn, n):
        t0 = time.time()
        for _ in range(n):
            fn()
        T[key] = time.time() - t0

    _, ac_ref = act_fwd(a4, cfg.act)
    bulk('  act', lambda: act_fwd(a4, cfg.act), reps)
    bulk('  dact', lambda: act_bwd(a4, a4, ac_ref, cfg.act), reps)
    bulk('  softmax', lambda: softmax_rows(at), reps)
    bulk('  layernorm', lambda: layernorm(a1, g1, b1), reps * 2)
    if cfg.rope:
        bulk('  rope', lambda: rope_apply(qh), reps * 2)
    if cfg.qk_norm:
        bulk('  qknorm', lambda: rms_norm(qh), reps * 2)

    tot = (T['forward'] + T['backward'] + T['adam']) / steps * 1000
    print('B=%d block_size=%d  bmm=%s  accelerate=%s  vForce=%s (enabled=%s)'
          % (B, cfg.block_size, BMM_BACKEND, HAVE_ACCELERATE,
             _vvtanhf is not None, USE_VFORCE))
    print('arch %s' % cfg.describe())
    print('%d params, %d tokens/step\n' % (model.n_params, BT))
    print('%-12s %10s %8s' % ('phase', 'ms/step', 'share'))
    for k in ('forward', 'backward', 'adam'):
        ms = T[k] / steps * 1000
        print('%-12s %10.2f %7.1f%%' % (k, ms, 100 * ms / tot))
    print('%-12s %10.2f' % ('TOTAL', tot))
    print('\ncomponent costs, summed over all layers per step:')
    for k in ('  act', '  dact', '  softmax', '  layernorm', '  rope',
              '  qknorm'):
        if k not in T:
            continue
        ms = T[k] / steps * 1000
        print('%-12s %10.2f %7.1f%% of step' % (k, ms, 100 * ms / tot))
    flop = 6.0 * model.n_params * BT
    print('\n%.1f GFLOP/s end-to-end' % (flop / (tot * 1e-3) / 1e9))


def bench(steps=20, B=32, **flags):
    global BMM_BACKEND
    saved = BMM_BACKEND
    for backend in ('numpy', 'accelerate'):
        if backend == 'accelerate' and not HAVE_ACCELERATE:
            continue
        BMM_BACKEND = backend
        print('=== bmm backend: %s ===' % backend)
        _bench_one(steps, B, **flags)
        print()
    BMM_BACKEND = saved


def _bench_one(steps=20, B=32, **flags):
    text, _ = load_corpus(None)
    stoi, _ = make_vocab(text)
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    cfg = Config(vocab=len(stoi), **flags)
    model = GPT(cfg)
    opt = Adam(model.p)
    rng = np.random.RandomState(0)
    gen = batcher(data, B, cfg.block_size, rng)

    for _ in range(3):
        x, y = next(gen)
        _, ca = model.forward(x, y)
        opt.step(model.p, model.backward(ca))

    tf = tb = to = 0.0
    for _ in range(steps):
        x, y = next(gen)
        t = time.time(); loss, ca = model.forward(x, y); tf += time.time() - t
        t = time.time(); g = model.backward(ca); tb += time.time() - t
        t = time.time(); opt.step(model.p, g); to += time.time() - t

    tot = (tf + tb + to) / steps * 1000.0
    print('%d params, %d tokens/step, accelerate %s'
          % (model.n_params, B * cfg.block_size, HAVE_ACCELERATE))
    print('arch %s\n' % cfg.describe())
    print('%-10s %10s %8s' % ('phase', 'ms/step', 'share'))
    for n, v in (('forward', tf), ('backward', tb), ('adam', to)):
        ms = v / steps * 1000.0
        print('%-10s %10.2f %7.1f%%' % (n, ms, ms / tot * 100))
    print('%-10s %10.2f' % ('TOTAL', tot))
    flop = 6.0 * model.n_params * B * cfg.block_size
    print('\n%.1f GFLOP/s   %.0f tokens/s'
          % (flop / (tot * 1e-3) / 1e9, B * cfg.block_size / (tot * 1e-3)))


# --------------------------------------------------------------------- cli

_STR_KEYS = ('ckpt', 'resume', 'act')


def parse_kv(args, start=0):
    """--key value pairs. 'act' holds a string that would parse as a float
    if it were not excluded: both 'gelu' and 'relu2' contain an 'e'."""
    kw = {}
    i = start
    while i < len(args) - 1:
        key = args[i].lstrip('-').replace('-', '_')
        val = args[i + 1]
        if key in _STR_KEYS:
            kw[key] = val
        else:
            kw[key] = (float(val) if '.' in val or 'e' in val.lower()
                       else int(val))
        i += 2
    return kw


if __name__ == '__main__':
    sys.argv = [
    'tiny_gpt.py',
    'train',
    'big.txt',
    '--steps', '3600',
    '--B', '32',
    '--block-size', '256',
    '--n-layer', '4',
    '--n-head', '4',
    '--n-embd', '128',
    '--rope', '1',
    '--qk-norm', '1',
    '--act', 'relu2',
    '--zero-init', '0',
    '--value-residual', '0',
    '--softcap', '0',
    '--ckpt', 'modelv3.npz',
    '--lr', '0.00053'
    ]
    mode = sys.argv[1] if len(sys.argv) > 1 else 'gradcheck'
    if mode == 'gradcheck':
        ok = gradcheck(**parse_kv(sys.argv, 2))
        sys.exit(0 if ok else 1)
    elif mode == 'gradcheck-all':
        sys.exit(0 if gradcheck_all() else 1)
    elif mode == 'train':
        args = sys.argv[2:]
        path = args[0] if args and not args[0].startswith('-') else None
        train(path, **parse_kv(args, 1 if path else 0))
    elif mode == 'bench':
        bench(**parse_kv(sys.argv, 2))
    elif mode == 'profile':
        profile(**parse_kv(sys.argv, 2))
    else:
        raise SystemExit('modes: gradcheck | gradcheck-all | train [file] '
                         '| bench | profile')