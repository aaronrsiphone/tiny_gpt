# coding: utf-8
'''
Elementwise math: activations, LayerNorm, RMSNorm, rotary position
embeddings, softmax -- every op with a hand-written forward *and* backward,
since there is no autograd anywhere in this codebase.

Each op is written as a mirrored fwd/bwd pair (gelu_cached()/dgelu_from(),
layernorm()/dlayernorm(), rms_norm()/drms_norm(), rope_apply()/rope_bwd())
so a reader can hold one screen with both directions of a single
computation instead of hunting for the backward pass across the file.
'''

import numpy as np

from tinygpt import backend
from tinygpt.backend import vtanh, vexp

# --------------------------------------------------------------- activation

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
        r = np.maximum(x, 0.0).astype(backend.DT)
        return (r * r).astype(backend.DT), r
    return gelu_cached(x)


def act_bwd(dout, x, cache, kind):
    """d/dx of act_fwd. For relu2: d(relu(x)^2)/dx = 2*relu(x)."""
    if kind == 'relu2':
        return (dout * (2.0 * cache)).astype(backend.DT)
    return (dout * dgelu_from(x, cache)).astype(backend.DT)


# --------------------------------------------------------------- LayerNorm

def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = xc / std
    return (xhat * g + b).astype(backend.DT), xhat, std


def dlayernorm(dy, xhat, std, g):
    C = xhat.shape[-1]
    dg = (dy * xhat).reshape(-1, C).sum(axis=0)
    db = dy.reshape(-1, C).sum(axis=0)
    dxhat = dy * g
    dx = (dxhat
          - dxhat.mean(axis=-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)) / std
    return dx.astype(backend.DT), dg.astype(backend.DT), db.astype(backend.DT)


# ----------------------------------------------------------------- RMSNorm

def rms_norm(x, eps=1e-6):
    """Non-parametric RMSNorm over the last axis. Returns (out, scale).

    No learnable gain: QK-norm exists to bound the attention logits, and a
    per-head gain would just reintroduce the degree of freedom that lets
    them grow. Keeping it parameter-free also keeps the parameter count and
    the gradcheck table unchanged.
    """
    ms = (x * x).mean(axis=-1, keepdims=True)
    s = np.sqrt(ms + eps)
    return (x / s).astype(backend.DT), s.astype(backend.DT)


def drms_norm(dr, r, s):
    """Backward of rms_norm, in terms of its own output.

    r = x/s with s = sqrt(mean(x^2)+eps), so
        dx = (dr - r*mean(dr*r)) / s
    which needs the output and the scale but not the input.
    """
    return ((dr - r * (dr * r).mean(axis=-1, keepdims=True))
             / s).astype(backend.DT)


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
    c = np.ascontiguousarray(cos[pos0:pos0 + T], dtype=backend.DT)[None]
    s = np.ascontiguousarray(sin[pos0:pos0 + T], dtype=backend.DT)[None]
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


# ------------------------------------------------------------------ softmax

def softmax_rows(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = vexp(np.ascontiguousarray(z, dtype=backend.DT))
    return (e / e.sum(axis=-1, keepdims=True)).astype(backend.DT)
