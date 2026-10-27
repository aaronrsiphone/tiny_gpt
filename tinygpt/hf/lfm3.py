# coding: utf-8
'''
LFM3 (Liquid AI, model_type 'lfm3') inference over NumPy and Accelerate.

A hybrid of short convolutions and attention. Each layer is one of

  conv               B, C, x = in_proj(h) split in three; Bx = B * x;
                     a causal depthwise convolution over time (kernel 7)
                     of Bx; then out_proj(C * conv)
  full_attention     grouped-query attention with RMSNorm on q and k
                     (before RoPE), partial "zero-tail" RoPE (theta 1e6),
                     and a sigmoid gate per head on the output
  sliding_attention  the same with full RoPE (theta 1e4), attending only
                     to the last `sliding_window` positions

wrapped in "sandwich" norms, followed by a SwiGLU MLP:

  h = x + post_operator_norm(op(operator_norm(x)))
  y = h + post_ffn_norm(w2(silu(w1(ffn_norm(h))) * w3(ffn_norm(h))))

and a final RMSNorm (named embedding_norm in the checkpoint) before the
tied output head. The reference is the model repository's own
modeling_lfm3.py; transformers has no built-in lfm3.

Zero-tail partial RoPE: the inverse frequencies follow the progression
of the full head_dim (theta^(-2i/head_dim)), but only the first
partial_rotary_factor * head_dim / 2 are kept; the rest are zero, so
those dimension pairs see cos 1, sin 0 and pass through unrotated. With
that table the ordinary rotate-half rope_rotate() does the right thing.

The cache keeps, per attention layer, keys and values for every position
(sliding layers read only their window of it), and per conv layer the
Bx of every position. Keeping all of Bx rather than the last six costs
memory (hidden floats per conv layer per position) but lets feed() reuse
any prefix of a conversation exactly as LlamaModel does.

Weights are handled as in llama.py: 'float32' or 'lowmem', with each
matrix product routed through the autotuned MatVec.
'''

import json
import math
import os

import numpy as np

from tinygpt.hf.llama import (ContextFull, Linear, MatVec, UnsupportedModel,
                              rms_norm, rope_rotate, silu_mul, softmax_last)
from tinygpt.hf.safetensors import Checkpoint, to_f32

CONV = 'conv'
FULL = 'full_attention'
SLIDING = 'sliding_attention'


class Lfm3Config(object):
    '''Architecture of an LFM3 model, read from config.json.'''

    model_type = 'lfm3'
    arch = 'Lfm3ForCausalLM'

    def __init__(self, d):
        if d.get('model_type') != self.model_type:
            raise UnsupportedModel('model_type %r is not lfm3'
                                   % d.get('model_type'))
        archs = d.get('architectures') or [self.arch]
        if self.arch not in archs:
            raise UnsupportedModel('architectures %r: only %s'
                                   % (archs, self.arch))
        self.hidden = int(d['hidden_size'])
        self.inter = int(d['intermediate_size'])
        self.n_layer = int(d['num_hidden_layers'])
        self.n_head = int(d['num_attention_heads'])
        self.n_kv = int(d.get('num_key_value_heads') or self.n_head)
        self.head_dim = int(d.get('head_dim') or self.hidden // self.n_head)
        self.vocab = int(d['vocab_size'])
        self.eps = float(d.get('norm_eps', 1e-5))
        self.max_pos = int(d.get('max_position_embeddings', 128000))
        self.tied = bool(d.get('tie_word_embeddings',
                               d.get('tie_embedding', True)))
        self.sandwich = bool(d.get('use_sandwich_norm', True))
        self.conv_k = int(d.get('conv_L_cache', 7))
        if d.get('conv_bias'):
            raise UnsupportedModel('conv_bias is not supported')
        self.sliding_window = d.get('sliding_window')
        self.layer_types = list(d.get('layer_types') or [
            FULL if i % 4 == 2 else SLIDING if i % 4 == 0 and i else CONV
            for i in range(self.n_layer)])
        if len(self.layer_types) != self.n_layer:
            raise UnsupportedModel('layer_types has %d entries for %d layers'
                                   % (len(self.layer_types), self.n_layer))
        bad = set(self.layer_types) - {CONV, FULL, SLIDING}
        if bad:
            raise UnsupportedModel('unknown layer types %s' % sorted(bad))
        if SLIDING in self.layer_types and not self.sliding_window:
            raise UnsupportedModel('sliding_attention needs sliding_window')
        if self.sliding_window:
            self.sliding_window = int(self.sliding_window)

        # RoPE per attention layer type: (theta, partial_rotary_factor).
        rp = d.get('rope_parameters') or {
            FULL: {'rope_type': 'default', 'rope_theta': 1e6,
                   'partial_rotary_factor': 0.5},
            SLIDING: {'rope_type': 'default', 'rope_theta': 1e4}}
        self.rope = {}
        for t in (FULL, SLIDING):
            if t not in self.layer_types:
                continue
            p = rp.get(t)
            if p is None:
                raise UnsupportedModel('no rope_parameters for %s' % t)
            if p.get('rope_type', 'default') != 'default':
                raise UnsupportedModel('rope_type %r is not supported'
                                       % p.get('rope_type'))
            self.rope[t] = (float(p['rope_theta']),
                            float(p.get('partial_rotary_factor', 1.0)))

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
        self.n_conv = self.layer_types.count(CONV)
        self.n_attn = self.n_layer - self.n_conv
        # The sliding window is a property of some layers, not a limit on
        # the context.
        self.max_context = self.max_pos

    def n_params(self):
        H, I = self.hidden, self.inter
        attn = (H * (self.q_dim + 2 * self.kv_dim + self.n_head)
                + self.q_dim * H + 2 * self.head_dim)
        conv = H * 3 * H + H * self.conv_k + H * H
        per = 3 * H * I + (4 if self.sandwich else 2) * H
        emb = self.vocab * H * (1 if self.tied else 2)
        return (self.n_layer * per + self.n_attn * attn + self.n_conv * conv
                + emb + H)

    def kv_bytes(self, ctx):
        return (2 * self.n_attn * self.kv_dim + self.n_conv * self.hidden) \
            * ctx * 4

    def describe(self):
        return ('lfm3: %d layers (%d conv, %d full attention, %d sliding '
                'window %s), hidden %d, %d heads / %d kv heads x %d, mlp %d, '
                'vocab %d%s'
                % (self.n_layer, self.n_conv, self.layer_types.count(FULL),
                   self.layer_types.count(SLIDING), self.sliding_window,
                   self.hidden, self.n_head, self.n_kv, self.head_dim,
                   self.inter, self.vocab,
                   ', tied embeddings' if self.tied else ''))


def rope_inv_freq(head_dim, theta, partial):
    '''Zero-tail inverse frequencies, in float32 as the reference does.'''
    n = int(partial * head_dim) // 2
    exps = np.arange(0, 2 * n, 2).astype(np.float32) / np.float32(head_dim)
    inv = (np.float32(1.0) / (np.float32(theta) ** exps)).astype(np.float32)
    return np.concatenate([inv, np.zeros(head_dim // 2 - n, np.float32)])


def rope_tables(head_dim, theta, partial, n):
    inv = rope_inv_freq(head_dim, theta, partial)
    ang = (np.arange(n, dtype=np.float32)[:, None] * inv[None, :]).astype(
        np.float32)
    return (np.ascontiguousarray(np.cos(ang), dtype=np.float32),
            np.ascontiguousarray(np.sin(ang), dtype=np.float32))


def head_norm(x, w, eps):
    '''RMSNorm over the last axis of (..., n_heads, head_dim).'''
    ms = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x * (np.float32(1.0) / np.sqrt(ms + np.float32(eps)))) * w


def sigmoid(z):
    with np.errstate(over='ignore'):
        return np.float32(1.0) / (np.float32(1.0) + np.exp(-z))


class Lfm3Cache(object):
    '''Per attention layer: rotated keys and values (n_kv, capacity,
    head_dim). Per conv layer: Bx (capacity, hidden). `length` positions
    are filled.'''

    def __init__(self, cfg, capacity):
        self.k, self.v, self.bx = [], [], []
        for t in cfg.layer_types:
            if t == CONV:
                self.k.append(None)
                self.v.append(None)
                self.bx.append(np.zeros((capacity, cfg.hidden), np.float32))
            else:
                shape = (cfg.n_kv, capacity, cfg.head_dim)
                self.k.append(np.zeros(shape, np.float32))
                self.v.append(np.zeros(shape, np.float32))
                self.bx.append(None)
        self.capacity = capacity
        self.length = 0
        self.tokens = []

    def truncate(self, n):
        '''Forget everything after the first n positions.'''
        self.length = min(self.length, n)
        del self.tokens[self.length:]


class _Layer(object):
    __slots__ = ('kind', 'op_norm', 'post_op_norm', 'ffn_norm',
                 'post_ffn_norm', 'qkvg', 'q_norm', 'k_norm', 'o',
                 'in_proj', 'conv_w', 'out_proj', 'w13', 'w2')


class Lfm3Model(object):

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
        H, I, hd = cfg.hidden, cfg.inter, cfg.head_dim

        def raw(name, shape):
            if name not in ckpt:
                raise UnsupportedModel('checkpoint is missing %r' % name)
            t = ckpt.info(name)
            if t.shape != tuple(shape):
                raise UnsupportedModel('%s has shape %r, config implies %r'
                                       % (name, t.shape, tuple(shape)))
            return ckpt.get_raw(name), t.dtype

        def vec(name, shape):
            return to_f32(*raw(name, shape)).reshape(-1)

        def lin(names_shapes):
            return Linear([raw(n, s) for n, s in names_shapes], weights,
                          self.mv, scratch)

        self.layers = []
        for i, kind in enumerate(cfg.layer_types):
            p = 'model.layers.%d.' % i
            L = _Layer()
            L.kind = kind
            L.op_norm = vec(p + 'operator_norm.weight', (H,))
            L.ffn_norm = vec(p + 'ffn_norm.weight', (H,))
            L.post_op_norm = L.post_ffn_norm = None
            if cfg.sandwich:
                L.post_op_norm = vec(p + 'post_operator_norm.weight', (H,))
                L.post_ffn_norm = vec(p + 'post_ffn_norm.weight', (H,))
            if kind == CONV:
                c = p + 'conv.'
                L.in_proj = lin([(c + 'in_proj.weight', (3 * H, H))])
                # (hidden, 1, k) -> (k, hidden): row j weights Bx[t-k+1+j].
                L.conv_w = np.ascontiguousarray(
                    vec(c + 'conv.weight', (H, 1, cfg.conv_k)).reshape(
                        H, cfg.conv_k).T)
                L.out_proj = lin([(c + 'out_proj.weight', (H, H))])
            else:
                a = p + 'self_attn.'
                # The per-head gate reads the same normed input as q/k/v,
                # so it rides along as a few extra output rows.
                L.qkvg = lin([(a + 'q_proj.weight', (cfg.q_dim, H)),
                              (a + 'k_proj.weight', (cfg.kv_dim, H)),
                              (a + 'v_proj.weight', (cfg.kv_dim, H)),
                              (a + 'gate_proj.weight', (cfg.n_head, H))])
                L.q_norm = vec(a + 'q_norm.weight', (hd,))
                L.k_norm = vec(a + 'k_norm.weight', (hd,))
                L.o = lin([(a + 'o_proj.weight', (H, cfg.q_dim))])
            f = p + 'feed_forward.'
            L.w13 = lin([(f + 'w1.weight', (I, H)), (f + 'w3.weight', (I, H))])
            L.w2 = lin([(f + 'w2.weight', (H, I))])
            self.layers.append(L)
        self.norm = vec('model.embedding_norm.weight', (H,))

        emb_shape = (cfg.vocab, H)
        if cfg.tied or 'lm_head.weight' not in ckpt:
            if not cfg.tied:
                raise UnsupportedModel('untied model has no lm_head.weight')
            self.head = lin([('model.embed_tokens.weight', emb_shape)])
        else:
            self.head = lin([('lm_head.weight', emb_shape)])
        if weights == 'float32':
            self._emb_raw = None
            self._emb = self.head.W if cfg.tied else \
                to_f32(*raw('model.embed_tokens.weight', emb_shape))
        else:
            self._emb = None
            self._emb_raw = raw('model.embed_tokens.weight', emb_shape)

        self.rope = {t: rope_tables(hd, theta, partial, self.ctx_limit)
                     for t, (theta, partial) in cfg.rope.items()}
        self.scale = np.float32(1.0 / math.sqrt(hd))
        if weights == 'float32':
            self.ckpt = None
            ckpt.close()

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_dir(cls, path, **kw):
        with open(os.path.join(path, 'config.json'), 'rb') as f:
            cfg = Lfm3Config(json.loads(f.read().decode('utf-8')))
        # As LlamaModel.from_dir: only lowmem reads through a memory map.
        ckpt = Checkpoint(path, use_mmap='require'
                          if kw.get('weights') == 'lowmem' else False)
        try:
            return cls(cfg, ckpt, **kw)
        except BaseException:
            ckpt.close()
            raise

    def close(self):
        if self.ckpt is not None:
            self.ckpt.close()
            self.ckpt = None

    def new_cache(self, capacity=None):
        return Lfm3Cache(self.cfg, capacity or self.ctx_limit)

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

    def _check_room(self, n, cache):
        limit = min(cache.capacity, self.ctx_limit)
        if cache.length + n > limit:
            raise ContextFull('context limit of %d tokens reached' % limit)

    def step(self, tok, cache):
        '''Feed one token at position cache.length; return its logits.'''
        self._check_room(1, cache)
        return self._forward([tok], cache, False)[0]

    def prefill(self, ids, cache, all_logits=False, chunk=64):
        '''Feed a block of tokens; the logits of the last (or every) one.'''
        ids = list(ids)
        if not ids:
            raise ValueError('prefill needs at least one token')
        self._check_room(len(ids), cache)
        outs = []
        last = None
        for a in range(0, len(ids), chunk):
            want = all_logits or a + chunk >= len(ids)
            last = self._forward(ids[a:a + chunk], cache, not want)
            if all_logits:
                outs.append(last)
        if all_logits:
            return np.concatenate(outs, axis=0)
        return last[-1]

    def _forward(self, ids, cache, skip_logits):
        '''Run T tokens at positions cache.length.. through every layer.
        One code path for decode (T = 1) and prefill: with T = 1 each
        matmul is a single row, which Linear routes to its matvec.'''
        cfg = self.cfg
        H, T = cfg.hidden, len(ids)
        p0 = cache.length
        S = p0 + T
        x = np.array(self.embed(ids), dtype=np.float32).reshape(T, H)
        buf = {}

        def scratch(name, n):
            b = buf.get(name)
            if b is None:
                b = buf[name] = np.empty((T, n), np.float32)
            return b

        for li, L in enumerate(self.layers):
            h = np.ascontiguousarray(rms_norm(x, L.op_norm, cfg.eps))
            if L.kind == CONV:
                y = self._conv(L, li, h, cache, p0, scratch)
            else:
                y = self._attention(L, li, h, cache, p0, scratch)
            if L.post_op_norm is not None:
                y = rms_norm(y, L.post_op_norm, cfg.eps)
            x += y
            h = np.ascontiguousarray(rms_norm(x, L.ffn_norm, cfg.eps))
            gu = self._lin(L.w13, h, scratch('w13', 2 * cfg.inter))
            a = np.ascontiguousarray(silu_mul(gu[:, :cfg.inter],
                                              gu[:, cfg.inter:]))
            y = self._lin(L.w2, a, scratch('out', H))
            if L.post_ffn_norm is not None:
                y = rms_norm(y, L.post_ffn_norm, cfg.eps)
            x += y
        cache.length = S
        cache.tokens.extend(int(t) for t in ids)
        if skip_logits:
            return None
        h = np.ascontiguousarray(rms_norm(x, self.norm, cfg.eps))
        return self._lin(self.head, h, np.empty((T, cfg.vocab), np.float32))

    @staticmethod
    def _lin(lin, X, Y):
        if X.shape[0] == 1:
            lin.matvec(X[0], Y[0])
            return Y
        return lin.matmul(X, Y)

    def _conv(self, L, li, h, cache, p0, scratch):
        H, K = self.cfg.hidden, self.cfg.conv_k
        T = h.shape[0]
        S = p0 + T
        bcx = self._lin(L.in_proj, h, scratch('in_proj', 3 * H))
        hist = cache.bx[li]
        np.multiply(bcx[:, :H], bcx[:, 2 * H:], out=hist[p0:S])
        # out[t] = sum_j w[j] * Bx[t - (K-1) + j], zeros before position 0.
        a = max(0, p0 - (K - 1))
        win = hist[a:S]
        pad = (K - 1) - (p0 - a)
        if pad:
            win = np.concatenate([np.zeros((pad, H), np.float32), win])
        w = L.conv_w
        conv = win[0:T] * w[0]
        for j in range(1, K):
            conv += win[j:j + T] * w[j]
        conv *= bcx[:, H:2 * H]
        return self._lin(L.out_proj, np.ascontiguousarray(conv),
                         scratch('out', H))

    def _attention(self, L, li, h, cache, p0, scratch):
        cfg = self.cfg
        nkv, g, hd = cfg.n_kv, cfg.group, cfg.head_dim
        qd, kvd, nh = cfg.q_dim, cfg.kv_dim, cfg.n_head
        T = h.shape[0]
        S = p0 + T
        qkvg = self._lin(L.qkvg, h, scratch('qkvg', qd + 2 * kvd + nh))
        q = head_norm(qkvg[:, :qd].reshape(T, nh, hd), L.q_norm, cfg.eps)
        k = head_norm(qkvg[:, qd:qd + kvd].reshape(T, nkv, hd), L.k_norm,
                      cfg.eps)
        cos, sin = self.rope[L.kind]
        rope_rotate(q, cos[p0:S], sin[p0:S])
        rope_rotate(k, cos[p0:S], sin[p0:S])
        cache.k[li][:, p0:S] = k.transpose(1, 0, 2)
        cache.v[li][:, p0:S] = \
            qkvg[:, qd + kvd:qd + 2 * kvd].reshape(T, nkv, hd).transpose(
                1, 0, 2)
        # Keys [k0, S). A sliding layer's query at p sees keys
        # (p - window, p]; the earliest query is p0.
        k0 = 0
        W = cfg.sliding_window if L.kind == SLIDING else None
        if W:
            k0 = max(0, p0 - W + 1)
        K = cache.k[li][:, k0:S]
        V = cache.v[li][:, k0:S]
        n = S - k0
        qg = q.reshape(T, nkv, g, hd).transpose(1, 2, 0, 3).reshape(
            nkv, g * T, hd)
        s = np.matmul(qg, K.transpose(0, 2, 1)).reshape(nkv, g, T, n)
        s *= self.scale
        if T > 1 or W:
            kp = k0 + np.arange(n)[None, :]
            qp = p0 + np.arange(T)[:, None]
            mask = kp > qp
            if W:
                mask |= kp <= qp - W
            if mask.any():
                s[:, :, mask] = -np.inf
        pr = softmax_last(s).reshape(nkv, g * T, n)
        y = np.matmul(pr, V).reshape(nkv, g, T, hd).transpose(2, 0, 1, 3)
        gate = sigmoid(qkvg[:, qd + 2 * kvd:]).reshape(T, nkv, g, 1)
        y = np.ascontiguousarray((y * gate).reshape(T, qd))
        return self._lin(L.o, y, scratch('out', cfg.hidden))

    def feed(self, ids, cache):
        '''Bring the cache up to date with `ids`, reusing any shared prefix
        (as LlamaModel.feed). Returns the logits after the last id.'''
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
