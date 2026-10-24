# coding: utf-8
'''
Llama-architecture inference over NumPy and Accelerate.

Covers what Hugging Face's LlamaForCausalLM computes, for inference only:
RMSNorm, rotary position embeddings (rotate-half layout), grouped-query
attention with a preallocated KV cache, the SiLU-gated (SwiGLU) MLP, and
tied or untied output embeddings. Every dimension comes from config.json.

Weights keep Hugging Face's nn.Linear orientation, W of shape (out, in),
so no transposed copy is ever made: decoding a token is y = W @ x, which
is cblas_sgemv(NoTrans) on the row-major matrix, and prefilling a prompt
is Y = X @ W.T, which is cblas_sgemm(transB). The q/k/v projections are
fused into one matrix per layer, as are gate/up, so a layer is four
matrix-vector products rather than seven.

Two weight modes:

  'float32'  every matrix widened to float32 once, at load. Fastest; for
             SmolLM2-135M about 540 MB.
  'lowmem'   matrices stay in the file's own dtype (BF16 for SmolLM2),
             memory-mapped, and are widened a slab at a time into one
             small scratch buffer as each product needs them. Resident
             weight memory is the OS's clean, droppable file cache; the
             cost is a conversion pass over every weight per token.

Which routine computes a matrix-vector product -- cblas_sgemv, a
one-row cblas_sgemm, or NumPy -- is chosen at load by timing each on the
model's actual matrices (see MatVec), and each candidate is checked
against NumPy before it is allowed to win.
'''

import math
import os
import time

import numpy as np

from tinygpt import backend
from tinygpt.backend import gemm, gemv, vexp
from tinygpt.hf.safetensors import Checkpoint, to_f32


class UnsupportedModel(ValueError):
    pass


class ContextFull(RuntimeError):
    pass


# ---------------------------------------------------------------- config

class LlamaConfig(object):

    SUPPORTED_TYPES = ('llama',)

    def __init__(self, d):
        mt = d.get('model_type')
        if mt not in self.SUPPORTED_TYPES:
            raise UnsupportedModel(
                'model_type %r is not supported (supported: %s)'
                % (mt, ', '.join(self.SUPPORTED_TYPES)))
        archs = d.get('architectures') or ['LlamaForCausalLM']
        if 'LlamaForCausalLM' not in archs:
            raise UnsupportedModel('architectures %r: only LlamaForCausalLM'
                                   % (archs,))
        act = d.get('hidden_act', 'silu')
        if act != 'silu':
            raise UnsupportedModel('hidden_act %r: only silu' % act)
        if d.get('pretraining_tp', 1) not in (None, 1):
            raise UnsupportedModel('pretraining_tp != 1 is not supported')

        self.hidden = int(d['hidden_size'])
        self.inter = int(d['intermediate_size'])
        self.n_layer = int(d['num_hidden_layers'])
        self.n_head = int(d['num_attention_heads'])
        self.n_kv = int(d.get('num_key_value_heads') or self.n_head)
        self.head_dim = int(d.get('head_dim') or self.hidden // self.n_head)
        self.vocab = int(d['vocab_size'])
        self.eps = float(d.get('rms_norm_eps', 1e-6))
        # transformers 4 writes rope_theta and rope_scaling at the top
        # level; transformers 5 folds both into one rope_parameters dict.
        rp = d.get('rope_parameters') or {}
        self.theta = float(rp.get('rope_theta',
                                  d.get('rope_theta', 10000.0)))
        self.rope_scaling = d.get('rope_scaling') or rp or None
        self.max_pos = int(d.get('max_position_embeddings', 2048))
        self.tied = bool(d.get('tie_word_embeddings', False))
        self.attn_bias = bool(d.get('attention_bias', False))
        self.mlp_bias = bool(d.get('mlp_bias', False))
        self.bos_id = d.get('bos_token_id')
        eos = d.get('eos_token_id')
        self.eos_ids = set(eos if isinstance(eos, list) else
                           ([eos] if eos is not None else []))

        if self.n_head % self.n_kv:
            raise UnsupportedModel('%d heads not divisible by %d kv heads'
                                   % (self.n_head, self.n_kv))
        if self.head_dim % 2:
            raise UnsupportedModel('rotary embedding needs an even head_dim')
        self.group = self.n_head // self.n_kv
        self.q_dim = self.n_head * self.head_dim
        self.kv_dim = self.n_kv * self.head_dim

    def n_params(self):
        H, I = self.hidden, self.inter
        per = (H * (self.q_dim + 2 * self.kv_dim) + self.q_dim * H
               + 3 * H * I + 2 * H)
        emb = self.vocab * H * (1 if self.tied else 2)
        return self.n_layer * per + emb + H

    def kv_bytes(self, ctx):
        return 2 * self.n_layer * self.kv_dim * ctx * 4

    def describe(self):
        return ('%d layers, hidden %d, %d heads / %d kv heads x %d, mlp %d, '
                'vocab %d, rope theta %g%s'
                % (self.n_layer, self.hidden, self.n_head, self.n_kv,
                   self.head_dim, self.inter, self.vocab, self.theta,
                   ', tied embeddings' if self.tied else ''))


def rope_inv_freq(cfg):
    '''Inverse frequencies, in float32 as transformers computes them.'''
    hd = cfg.head_dim
    exps = np.arange(0, hd, 2).astype(np.float32) / np.float32(hd)
    inv = (np.float32(1.0) / (np.float32(cfg.theta) ** exps)).astype(
        np.float32)
    rs = cfg.rope_scaling
    kind = None
    if rs:
        kind = rs.get('rope_type', rs.get('type'))
    if kind in (None, 'default'):
        return inv
    if kind == 'linear':
        return (inv / np.float32(rs['factor'])).astype(np.float32)
    if kind == 'llama3':
        factor = float(rs['factor'])
        lo = float(rs.get('low_freq_factor', 1.0))
        hi = float(rs.get('high_freq_factor', 4.0))
        old = float(rs['original_max_position_embeddings'])
        lo_wl, hi_wl = old / lo, old / hi
        wl = 2 * math.pi / inv
        out = np.where(wl > lo_wl, inv / factor, inv)
        smooth = (old / wl - lo) / (hi - lo)
        mid = (1 - smooth) * out / factor + smooth * out
        is_mid = ~(wl < hi_wl) & ~(wl > lo_wl)
        return np.where(is_mid, mid, out).astype(np.float32)
    raise UnsupportedModel('rope_scaling type %r is not supported' % kind)


def rope_tables(cfg, n):
    '''cos, sin of shape (n, head_dim/2) for positions [0, n), float32.'''
    inv = rope_inv_freq(cfg)
    pos = np.arange(n, dtype=np.float32)
    ang = (pos[:, None] * inv[None, :]).astype(np.float32)
    return (np.ascontiguousarray(np.cos(ang), dtype=np.float32),
            np.ascontiguousarray(np.sin(ang), dtype=np.float32))


# ------------------------------------------------------------ elementwise

def rms_norm(x, w, eps):
    '''x * rsqrt(mean(x^2) + eps) * w over the last axis, in float32.'''
    if x.ndim == 1:
        # One token: a dot product is several times cheaper than mean().
        r = np.float32(1.0 / math.sqrt(float(np.dot(x, x)) / x.size + eps))
        return x * r * w
    ms = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x * (np.float32(1.0) / np.sqrt(ms + np.float32(eps)))) * w


def silu_mul(g, u):
    '''silu(g) * u = g * sigmoid(g) * u. exp(-g) overflowing to inf for
    very negative g gives the correct limit of 0.'''
    with np.errstate(over='ignore'):
        e = vexp(np.ascontiguousarray(-g))
    return g / (np.float32(1.0) + e) * u


def softmax_last(z):
    z = z - z.max(axis=-1, keepdims=True)
    with np.errstate(under='ignore'):
        e = vexp(np.ascontiguousarray(z, dtype=np.float32))
    return e / e.sum(axis=-1, keepdims=True)


def rope_rotate(x, cos, sin):
    '''Rotate-half RoPE on x (..., T, n_heads, head_dim), in place.

    cos/sin are (T, head_dim/2). Pairs (i, i + head_dim/2) rotate
    together, the layout of Hugging Face Llama checkpoints (whose q/k
    weights were permuted at conversion time to make it so).
    '''
    half = x.shape[-1] // 2
    c = cos[:, None, :]
    s = sin[:, None, :]
    x1 = x[..., :half].copy()
    x2 = x[..., half:]
    x[..., :half] = x1 * c - x2 * s
    x[..., half:] = x1 * s + x2 * c
    return x


# --------------------------------------------------------- matrix-vector

def _mv_numpy(W, x, y):
    np.dot(W, x, out=y)


def _mv_gemv(W, x, y):
    gemv(W, x, y)


def _mv_gemm(W, x, y):
    gemm(x.reshape(1, -1), W, y.reshape(1, -1), transB=True)


MATVEC_IMPLS = {'numpy': _mv_numpy, 'gemv': _mv_gemv, 'gemm': _mv_gemm}


def _mm_numpy(W, X, Y):
    np.matmul(X, W.T, out=Y)


def _mm_gemm(W, X, Y):
    gemm(X, W, Y, transB=True)


# Prefill: Y = X @ W.T for a block of rows. Tuned like the matvec, at a
# representative block height, because a slow BLAS can lose to NumPy here
# just as it can for matrix-vector products.
MATMUL_IMPLS = {'numpy': _mm_numpy, 'gemm': _mm_gemm}
MATMUL_TUNE_ROWS = 32


def matvec_candidates():
    c = ['numpy']
    if backend.HAVE_ACCELERATE:
        c.append('gemm')
    if backend.HAVE_SGEMV:
        c.append('gemv')
    return c


def _median_time(fn, args, reps):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        t.append(time.perf_counter() - t0)
    return sorted(t)[len(t) // 2]


class MatVec(object):
    '''Pick the fastest correct routine per matrix shape, for both
    y = W @ x (decoding) and Y = X @ W.T (prefill).

    TINY_GPT_MATVEC=gemv|gemm|numpy and TINY_GPT_MATMUL=gemm|numpy force
    one, for A/B runs.
    '''

    def __init__(self, force=None, force_mm=None):
        self.force = force or os.environ.get('TINY_GPT_MATVEC') or None
        if self.force and self.force not in MATVEC_IMPLS:
            raise ValueError('TINY_GPT_MATVEC=%r: choose from %s'
                             % (self.force, ', '.join(MATVEC_IMPLS)))
        self.force_mm = force_mm or os.environ.get('TINY_GPT_MATMUL') or None
        if self.force_mm and self.force_mm not in MATMUL_IMPLS:
            raise ValueError('TINY_GPT_MATMUL=%r: choose from %s'
                             % (self.force_mm, ', '.join(MATMUL_IMPLS)))
        self.choice = {}
        self.timings = {}
        self.mm_choice = {}
        self.mm_timings = {}

    def tune_mm(self, W, reps=3):
        shape = W.shape
        if shape in self.mm_choice:
            return self.mm_choice[shape]
        cands = ['numpy'] + (['gemm'] if backend.HAVE_ACCELERATE else [])
        if self.force_mm or len(cands) == 1:
            name = self.force_mm or 'numpy'
            self.mm_choice[shape] = MATMUL_IMPLS[name]
            self.mm_timings[shape] = {name: None}
            return self.mm_choice[shape]
        rng = np.random.RandomState(0)
        X = rng.standard_normal((MATMUL_TUNE_ROWS, shape[1])).astype(
            np.float32)
        ref = np.empty((MATMUL_TUNE_ROWS, shape[0]), np.float32)
        _mm_numpy(W, X, ref)
        tol = 1e-4 * (np.abs(W).max() * np.abs(X).sum(axis=1).max() + 1e-30)
        best, best_t, times = 'numpy', None, {}
        for name in cands:
            fn = MATMUL_IMPLS[name]
            Y = np.empty_like(ref)
            first = _median_time(fn, (W, X, Y), 1)
            err = float(np.abs(Y - ref).max())
            if not err <= tol:
                times[name] = 'rejected (err %.3g)' % err
                continue
            dt = first if best_t is not None and first > 10 * best_t \
                else _median_time(fn, (W, X, Y), reps)
            times[name] = dt
            if best_t is None or dt < best_t:
                best, best_t = name, dt
        self.mm_choice[shape] = MATMUL_IMPLS[best]
        self.mm_timings[shape] = times
        return self.mm_choice[shape]

    def tune(self, W, reps=5):
        shape = W.shape
        if shape in self.choice:
            return self.choice[shape]
        if self.force:
            self.choice[shape] = MATVEC_IMPLS[self.force]
            self.timings[shape] = {self.force: None}
            return self.choice[shape]
        rng = np.random.RandomState(0)
        x = rng.standard_normal(shape[1]).astype(np.float32)
        ref = np.empty(shape[0], np.float32)
        _mv_numpy(W, x, ref)
        tol = 1e-4 * (np.abs(W).max() * np.abs(x).sum() + 1e-30)
        best, best_t, times = None, None, {}
        for name in matvec_candidates():
            fn = MATVEC_IMPLS[name]
            y = np.empty(shape[0], np.float32)
            first = _median_time(fn, (W, x, y), 1)
            err = float(np.abs(y - ref).max())
            if not err <= tol:
                times[name] = 'rejected (err %.3g)' % err
                continue
            # A candidate 10x behind on its first call is not re-timed.
            dt = first if best_t is not None and first > 10 * best_t \
                else _median_time(fn, (W, x, y), reps)
            times[name] = dt
            if best_t is None or dt < best_t:
                best, best_t = name, dt
        self.choice[shape] = MATVEC_IMPLS[best]
        self.timings[shape] = times
        return self.choice[shape]

    def report(self):
        lines = []
        for title, impls, choice, timings in (
                ('matvec (decode)', MATVEC_IMPLS, self.choice, self.timings),
                ('matmul (prefill, %d rows)' % MATMUL_TUNE_ROWS,
                 MATMUL_IMPLS, self.mm_choice, self.mm_timings)):
            if not timings:
                continue
            lines.append('  ' + title)
            names = {v: k for k, v in impls.items()}
            for shape, times in sorted(timings.items()):
                parts = []
                for k, v in times.items():
                    parts.append('%s %s' % (k, 'untimed' if v is None else
                                            (v if isinstance(v, str)
                                             else '%.3f ms' % (v * 1e3))))
                lines.append('    %5d x %-5d -> %-5s  (%s)'
                             % (shape[0], shape[1], names[choice[shape]],
                                ', '.join(parts)))
        return '\n'.join(lines)


class Linear(object):
    '''y = W @ x (+ b) for one weight matrix, possibly several fused.

    `parts` are (raw array, stored dtype) pairs stacked along the output
    axis. In float32 mode they are widened once into one contiguous
    matrix; in lowmem mode they are kept as the raw mapped arrays and
    widened a slab of rows at a time into the shared scratch buffer.
    '''

    def __init__(self, parts, mode, mv, scratch=None, bias=None):
        self.n_in = parts[0][0].shape[1]
        for raw, _ in parts:
            if raw.ndim != 2 or raw.shape[1] != self.n_in:
                raise UnsupportedModel('fused weights disagree on input '
                                       'width')
        self.n_out = sum(raw.shape[0] for raw, _ in parts)
        self.mode = mode
        self.bias = bias
        self.scratch = scratch
        if mode == 'float32':
            self.W = np.empty((self.n_out, self.n_in), np.float32)
            r = 0
            for raw, dt in parts:
                to_f32(raw, dt, out=self.W[r:r + raw.shape[0]])
                r += raw.shape[0]
            self.parts = None
            self._mv = mv.tune(self.W)
            self._mm = mv.tune_mm(self.W)
        else:
            self.W = None
            self.parts = parts
            self._mv = None
            self.mvtuner = mv

    def matvec(self, x, y):
        if self.W is not None:
            self._mv(self.W, x, y)
        else:
            rows = max(1, self.scratch.size // self.n_in)
            r = 0
            for raw, dt in self.parts:
                n = raw.shape[0]
                for a in range(0, n, rows):
                    b = min(n, a + rows)
                    W = self.scratch[:(b - a) * self.n_in].reshape(
                        b - a, self.n_in)
                    to_f32(raw[a:b], dt, out=W)
                    self.mvtuner.tune(W)(W, x, y[r + a:r + b])
                r += n
        if self.bias is not None:
            y += self.bias
        return y

    def matmul(self, X, Y):
        '''Y = X @ W.T for a block of rows X (T, n_in).'''
        if self.W is not None:
            self._mm(self.W, X, Y)
        else:
            rows = max(1, self.scratch.size // self.n_in)
            r = 0
            for raw, dt in self.parts:
                n = raw.shape[0]
                for a in range(0, n, rows):
                    b = min(n, a + rows)
                    W = self.scratch[:(b - a) * self.n_in].reshape(
                        b - a, self.n_in)
                    to_f32(raw[a:b], dt, out=W)
                    tmp = np.empty((X.shape[0], b - a), np.float32)
                    self.mvtuner.tune_mm(W)(W, X, tmp)
                    Y[:, r + a:r + b] = tmp
                r += n
        if self.bias is not None:
            Y += self.bias
        return Y


# ------------------------------------------------------------------ model

class KVCache(object):
    '''Preallocated keys/values for every layer: (n_kv, capacity, head_dim).

    Keys are stored already rotated, so a cached key never needs touching
    again. `length` is how many positions are filled.
    '''

    def __init__(self, cfg, capacity):
        shape = (cfg.n_kv, capacity, cfg.head_dim)
        self.k = [np.zeros(shape, np.float32) for _ in range(cfg.n_layer)]
        self.v = [np.zeros(shape, np.float32) for _ in range(cfg.n_layer)]
        self.capacity = capacity
        self.length = 0
        self.tokens = []

    def truncate(self, n):
        '''Forget everything after the first n positions.'''
        self.length = min(self.length, n)
        del self.tokens[self.length:]


class _Layer(object):
    __slots__ = ('ln1', 'ln2', 'qkv', 'o', 'gate_up', 'down')


class LlamaModel(object):

    def __init__(self, cfg, ckpt, ctx_limit=1024, weights='float32',
                 matvec=None, scratch_mb=8):
        if weights not in ('float32', 'lowmem'):
            raise ValueError("weights must be 'float32' or 'lowmem'")
        self.cfg = cfg
        self.weights = weights
        self.ctx_limit = int(min(ctx_limit, cfg.max_pos))
        self.mv = MatVec(matvec)
        self.ckpt = ckpt
        scratch = None
        if weights == 'lowmem':
            scratch = np.empty(max(scratch_mb << 20, 4 * cfg.hidden * 4) // 4,
                               np.float32)
        self.scratch = scratch
        H = cfg.hidden

        def raw(name, shape):
            if name not in ckpt:
                raise UnsupportedModel('checkpoint is missing %r' % name)
            t = ckpt.info(name)
            if t.shape != tuple(shape):
                raise UnsupportedModel('%s has shape %r, config implies %r'
                                       % (name, t.shape, tuple(shape)))
            return ckpt.get_raw(name), t.dtype

        def vec(name, n):
            r, dt = raw(name, (n,))
            return to_f32(r, dt)

        def lin(names_shapes, bias_names=None):
            parts = [raw(n, s) for n, s in names_shapes]
            b = None
            if bias_names:
                b = np.concatenate([vec(n, s[0]) for n, s in
                                    zip(bias_names, [s for _, s in
                                                     names_shapes])])
            out = Linear(parts, weights, self.mv, scratch, b)
            if weights == 'float32':
                for n, _ in names_shapes:
                    ckpt.drop_pages(n)
            return out

        qd, kvd, I = cfg.q_dim, cfg.kv_dim, cfg.inter
        self.layers = []
        for i in range(cfg.n_layer):
            p = 'model.layers.%d.' % i
            a = p + 'self_attn.'
            m = p + 'mlp.'
            L = _Layer()
            L.ln1 = vec(p + 'input_layernorm.weight', H)
            L.ln2 = vec(p + 'post_attention_layernorm.weight', H)
            L.qkv = lin([(a + 'q_proj.weight', (qd, H)),
                         (a + 'k_proj.weight', (kvd, H)),
                         (a + 'v_proj.weight', (kvd, H))],
                        [a + 'q_proj.bias', a + 'k_proj.bias',
                         a + 'v_proj.bias'] if cfg.attn_bias else None)
            L.o = lin([(a + 'o_proj.weight', (H, qd))],
                      [a + 'o_proj.bias'] if cfg.attn_bias else None)
            L.gate_up = lin([(m + 'gate_proj.weight', (I, H)),
                             (m + 'up_proj.weight', (I, H))],
                            [m + 'gate_proj.bias', m + 'up_proj.bias']
                            if cfg.mlp_bias else None)
            L.down = lin([(m + 'down_proj.weight', (H, I))],
                         [m + 'down_proj.bias'] if cfg.mlp_bias else None)
            self.layers.append(L)
        self.norm = vec('model.norm.weight', H)

        emb_raw = raw('model.embed_tokens.weight', (cfg.vocab, H))
        if cfg.tied or 'lm_head.weight' not in ckpt:
            if not cfg.tied:
                raise UnsupportedModel('untied model has no lm_head.weight')
            self.head = lin([('model.embed_tokens.weight', (cfg.vocab, H))])
        else:
            self.head = lin([('lm_head.weight', (cfg.vocab, H))])
        if weights == 'float32' and cfg.tied:
            self._emb = self.head.W        # one copy serves both
        elif weights == 'float32':
            self._emb = to_f32(*emb_raw)
        else:
            self._emb = None
        self._emb_raw = emb_raw

        self.cos, self.sin = rope_tables(cfg, self.ctx_limit)
        self.scale = np.float32(1.0 / math.sqrt(cfg.head_dim))
        if weights == 'float32':
            # Everything now lives in float32 copies. Dropping the last
            # views onto the file lets the mapping go, so the BF16 pages
            # are not held alongside the widened weights.
            self._emb_raw = None
            self.ckpt = None
            ckpt.close()

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_dir(cls, path, **kw):
        import json
        with open(os.path.join(path, 'config.json'), 'rb') as f:
            cfg = LlamaConfig(json.loads(f.read().decode('utf-8')))
        ckpt = Checkpoint(path)
        try:
            return cls(cfg, ckpt, **kw)
        except BaseException:
            ckpt.close()
            raise

    def close(self):
        '''Release the weight file (lowmem mode keeps it open).'''
        if self.ckpt is not None:
            self.ckpt.close()
            self.ckpt = None

    def new_cache(self, capacity=None):
        return KVCache(self.cfg, capacity or self.ctx_limit)

    def embed(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.cfg.vocab):
            raise ValueError('token id out of range for vocab %d'
                             % self.cfg.vocab)
        if self._emb is not None:
            return self._emb[ids]
        r, dt = self._emb_raw
        return to_f32(r[ids], dt)

    # -- forward -----------------------------------------------------------

    def step(self, tok, cache):
        '''Feed one token at position cache.length; return its logits.'''
        cfg = self.cfg
        pos = cache.length
        if pos >= cache.capacity or pos >= self.ctx_limit:
            raise ContextFull('context limit of %d tokens reached'
                              % min(cache.capacity, self.ctx_limit))
        nkv, g, hd = cfg.n_kv, cfg.group, cfg.head_dim
        qd, kvd = cfg.q_dim, cfg.kv_dim
        cos, sin = self.cos[pos:pos + 1], self.sin[pos:pos + 1]
        x = self.embed([tok])[0].astype(np.float32, copy=True)
        qkv = np.empty(qd + 2 * kvd, np.float32)
        gu = np.empty(2 * cfg.inter, np.float32)
        hbuf = np.empty(cfg.hidden, np.float32)
        T = pos + 1
        for li, L in enumerate(self.layers):
            h = rms_norm(x, L.ln1, cfg.eps)
            L.qkv.matvec(h, qkv)
            q = qkv[:qd].reshape(1, cfg.n_head, hd)
            k = qkv[qd:qd + kvd].reshape(1, nkv, hd)
            rope_rotate(q, cos, sin)
            rope_rotate(k, cos, sin)
            cache.k[li][:, pos] = k[0]
            cache.v[li][:, pos] = qkv[qd + kvd:].reshape(nkv, hd)
            K = cache.k[li][:, :T]
            V = cache.v[li][:, :T]
            s = np.matmul(q.reshape(nkv, g, hd), K.transpose(0, 2, 1))
            s *= self.scale
            y = np.matmul(softmax_last(s), V).reshape(qd)
            L.o.matvec(y, hbuf)
            x += hbuf
            h = rms_norm(x, L.ln2, cfg.eps)
            L.gate_up.matvec(h, gu)
            a = silu_mul(gu[:cfg.inter], gu[cfg.inter:])
            L.down.matvec(np.ascontiguousarray(a), hbuf)
            x += hbuf
        cache.length = T
        cache.tokens.append(int(tok))
        logits = np.empty(cfg.vocab, np.float32)
        self.head.matvec(rms_norm(x, self.norm, cfg.eps), logits)
        return logits

    def prefill(self, ids, cache, all_logits=False, chunk=64):
        '''Feed a block of tokens with matrix-matrix products.

        Returns the logits of the last token, or of every token with
        all_logits. Processed in chunks so the attention scores and
        activations stay bounded however long the prompt is.
        '''
        ids = list(ids)
        if not ids:
            raise ValueError('prefill needs at least one token')
        if cache.length + len(ids) > min(cache.capacity, self.ctx_limit):
            raise ContextFull('%d tokens do not fit: %d cached, limit %d'
                              % (len(ids), cache.length,
                                 min(cache.capacity, self.ctx_limit)))
        outs = []
        last = None
        for a in range(0, len(ids), chunk):
            last = self._prefill_chunk(ids[a:a + chunk], cache,
                                       all_logits or a + chunk >= len(ids))
            if all_logits:
                outs.append(last)
        if all_logits:
            return np.concatenate(outs, axis=0)
        return last[-1]

    def _prefill_chunk(self, ids, cache, want_logits):
        cfg = self.cfg
        nkv, g, hd = cfg.n_kv, cfg.group, cfg.head_dim
        qd, kvd = cfg.q_dim, cfg.kv_dim
        T = len(ids)
        p0 = cache.length
        S = p0 + T
        cos, sin = self.cos[p0:S], self.sin[p0:S]
        x = np.array(self.embed(ids), dtype=np.float32)
        qkv = np.empty((T, qd + 2 * kvd), np.float32)
        gu = np.empty((T, 2 * cfg.inter), np.float32)
        hbuf = np.empty((T, cfg.hidden), np.float32)
        # Query t (absolute p0+t) may attend to keys 0..p0+t.
        mask = np.arange(S)[None, :] > (p0 + np.arange(T))[:, None]
        for li, L in enumerate(self.layers):
            h = np.ascontiguousarray(rms_norm(x, L.ln1, cfg.eps))
            L.qkv.matmul(h, qkv)
            q = qkv[:, :qd].reshape(T, cfg.n_head, hd).copy()
            k = qkv[:, qd:qd + kvd].reshape(T, nkv, hd).copy()
            rope_rotate(q, cos, sin)
            rope_rotate(k, cos, sin)
            cache.k[li][:, p0:S] = k.transpose(1, 0, 2)
            cache.v[li][:, p0:S] = \
                qkv[:, qd + kvd:].reshape(T, nkv, hd).transpose(1, 0, 2)
            K = cache.k[li][:, :S]
            V = cache.v[li][:, :S]
            # (nkv, g*T, hd): each kv head serves its g query heads.
            qg = q.reshape(T, nkv, g, hd).transpose(1, 2, 0, 3).reshape(
                nkv, g * T, hd)
            s = np.matmul(qg, K.transpose(0, 2, 1)).reshape(nkv, g, T, S)
            s *= self.scale
            s[:, :, mask] = -np.inf
            pr = softmax_last(s).reshape(nkv, g * T, S)
            y = np.matmul(pr, V).reshape(nkv, g, T, hd).transpose(
                2, 0, 1, 3).reshape(T, qd)
            L.o.matmul(np.ascontiguousarray(y), hbuf)
            x += hbuf
            h = np.ascontiguousarray(rms_norm(x, L.ln2, cfg.eps))
            L.gate_up.matmul(h, gu)
            a = silu_mul(gu[:, :cfg.inter], gu[:, cfg.inter:])
            L.down.matmul(np.ascontiguousarray(a), hbuf)
            x += hbuf
        cache.length = S
        cache.tokens.extend(int(t) for t in ids)
        if not want_logits:
            return None
        h = np.ascontiguousarray(rms_norm(x, self.norm, cfg.eps))
        logits = np.empty((T, cfg.vocab), np.float32)
        self.head.matmul(h, logits)
        return logits

    def feed(self, ids, cache):
        '''Bring the cache up to date with `ids`, reusing any shared prefix.

        Returns the logits after the last id. A multi-turn chat re-sends
        the whole conversation each turn; only the new tail is computed.
        '''
        ids = list(ids)
        n = 0
        old = cache.tokens
        while n < min(len(old), len(ids) - 1) and old[n] == ids[n]:
            n += 1
        cache.truncate(n)
        rest = ids[n:]
        if len(rest) == 1:
            return self.step(rest[0], cache)
        return self.prefill(rest, cache)


# --------------------------------------------------------------- sampling

class Sampler(object):
    '''Greedy at temperature 0; otherwise temperature, top-k, top-p.

    Mirrors GPT.generate()'s sampling in tinygpt/model.py, in float64.
    '''

    def __init__(self, temperature=0.0, top_k=0, top_p=1.0,
                 repetition_penalty=1.0, seed=0):
        self.temperature = float(temperature)
        self.top_k = int(top_k or 0)
        self.top_p = float(top_p if top_p is not None else 1.0)
        self.rep = float(repetition_penalty or 1.0)
        self.rng = np.random.RandomState(seed)

    def __call__(self, logits, history=()):
        z = logits.astype(np.float64)
        if self.rep != 1.0 and len(history):
            idx = np.unique(np.asarray(history, dtype=np.int64))
            v = z[idx]
            z[idx] = np.where(v > 0, v / self.rep, v * self.rep)
        if self.temperature <= 0:
            return int(np.argmax(z))
        z = z / self.temperature
        if self.top_k and self.top_k < z.size:
            cut = np.partition(z, -self.top_k)[-self.top_k]
            z = np.where(z < cut, -np.inf, z)
        z -= z.max()
        pr = np.exp(z)
        pr /= pr.sum()
        if 0 < self.top_p < 1:
            order = np.argsort(-pr)
            keep = np.cumsum(pr[order]) - pr[order] < self.top_p
            keep[0] = True
            mask = np.zeros(pr.size, bool)
            mask[order[keep]] = True
            pr = np.where(mask, pr, 0.0)
            pr /= pr.sum()
        return int(self.rng.choice(pr.size, p=pr))


def generate(model, cache, prompt_ids, max_new_tokens, sampler=None,
             stop_ids=(), on_token=None):
    '''Generate up to max_new_tokens after prompt_ids.

    Returns (new_ids, reason, stats) where reason is 'stop', 'length' or
    'context', and stats has the prefill time and per-token decode times.
    The stop token itself is not included in new_ids.
    '''
    sampler = sampler or Sampler()
    stop_ids = set(stop_ids)
    t0 = time.perf_counter()
    logits = model.feed(prompt_ids, cache)
    t_first = time.perf_counter() - t0
    out = []
    reason = 'length'
    t1 = time.perf_counter()
    for _ in range(max_new_tokens):
        nxt = sampler(logits, cache.tokens)
        if nxt in stop_ids:
            reason = 'stop'
            break
        out.append(nxt)
        if on_token:
            on_token(nxt)
        if len(out) == max_new_tokens:
            break
        try:
            logits = model.step(nxt, cache)
        except ContextFull:
            reason = 'context'
            break
    t_dec = time.perf_counter() - t1
    return out, reason, {'prefill_s': t_first, 'decode_s': t_dec,
                         'prompt_tokens': len(prompt_ids),
                         'new_tokens': len(out)}
