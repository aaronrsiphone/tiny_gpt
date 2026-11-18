# coding: utf-8
'''
Aurora2 (AuroraAI-Research, model_type 'aurora2') inference over NumPy and
Accelerate: a small top-1 mixture-of-experts language model.

Per token, on top of the usual decoder:

  embedding   tok_emb[id] + ng_gate * ng_proj(concat of three hashed n-gram
              embeddings). Each n-gram table (n = 2, 3, 4) is indexed by a
              causal hash of the n most recent token ids, the current one
              included; ids before the start of the text count as 0.
  every block x = lam0 * x + lam1 * x0        (x0 = the embedding above)
              a = rmsnorm(x, ln1)
              q, k, v = a @ wq, wk, wv; layers after the first mix
              v = vl0 * v + vl1 * v_of_layer_0
              q, k: RMSNorm per head with its own weight (before RoPE),
              rotate-half RoPE over the whole head, grouped-query attention
              scaled by head_dim ** -0.5
              each head's output is multiplied by 2 * sigmoid(ag_w . a[:12])
              x += attention_out @ wo
              m = rmsnorm(x, ln2)
              x += shared SwiGLU(m) + gate * expert(m)
              where the router picks ONE expert per token (softmax over
              m @ router_w + router_b, argmax) and gate is its probability
  head        rmsnorm(x, final_norm) @ tok_emb.T, then 30 * tanh(z / 30)

The reference is the model repository's modeling_aurora2.py, read as text
(transformers has no built-in aurora2). The config says
tie_word_embeddings false, yet the checkpoint has no lm_head: the head is
tok_emb, as in the reference's forward.

Every expert is its own Linear, so a token only reads (in lowmem mode: only
widens) the one expert it was routed to: about 19M of the 400M parameters.
The cache keeps per layer the rotated keys and the (mixed) values.
'''

import json
import math
import os

import numpy as np

from tinygpt.hf.lfm3 import head_norm, rope_tables, sigmoid
from tinygpt.hf.llama import (ContextFull, Linear, MatVec, UnsupportedModel,
                              rms_norm, rope_rotate, silu_mul, softmax_last)
from tinygpt.hf.safetensors import Checkpoint, to_f32

HASH_MASK = 0x7FFFFFFF


class Aurora2Config(object):
    '''Architecture of an Aurora2 model, read from config.json.'''

    model_type = 'aurora2'
    arch = 'Aurora2ForCausalLM'

    def __init__(self, d):
        if d.get('model_type') != self.model_type:
            raise UnsupportedModel('model_type %r is not aurora2'
                                   % d.get('model_type'))
        archs = d.get('architectures') or [self.arch]
        if self.arch not in archs:
            raise UnsupportedModel('architectures %r: only %s'
                                   % (archs, self.arch))
        # Defaults are those of configuration_aurora2.py.
        self.vocab = int(d.get('vocab_size', 16384))
        self.hidden = int(d.get('d_model', 192))
        self.n_layer = int(d.get('n_layers', 12))
        self.n_head = int(d.get('n_heads', 6))
        self.n_kv = int(d.get('n_kv_heads', 2))
        self.head_dim = int(d.get('head_dim', 32))
        self.n_expert = int(d.get('num_experts', 32))
        self.ffn = int(d.get('ffn_dim', 1776))
        self.shared = int(d.get('ffn_shared_dim', 256))
        self.ng_buckets = int(d.get('ngram_buckets', 16384))
        self.ng_dim = int(d.get('ngram_dim', 12))
        self.ng_sizes = [int(n) for n in d.get('ngram_sizes', (2, 3, 4))]
        self.ng_seed = int(d.get('ngram_seed', 1337))
        self.rope_theta = float(d.get('rope_theta', 10000.0))
        self.eps = float(d.get('rms_norm_eps', 1e-6))
        self.max_pos = int(d.get('max_position_embeddings', 2048))
        self.gate_dim = int(d.get('gate_dim', 12))
        self.softcap = float(d.get('logit_softcap') or 0.0)
        self.bos_id = d.get('bos_token_id', 0)
        eos = d.get('eos_token_id', 1)
        self.eos_ids = set(eos if isinstance(eos, list) else
                           ([eos] if eos is not None else []))
        if self.n_head % self.n_kv:
            raise UnsupportedModel('%d heads not divisible by %d kv heads'
                                   % (self.n_head, self.n_kv))
        if self.head_dim % 2:
            raise UnsupportedModel('rotary embedding needs an even head_dim')
        if not 0 < self.gate_dim <= self.hidden:
            raise UnsupportedModel('gate_dim %d outside 1..%d'
                                   % (self.gate_dim, self.hidden))
        if not self.ng_sizes or min(self.ng_sizes) < 1:
            raise UnsupportedModel('ngram_sizes must be positive')
        if max(self.ng_sizes) - 1 > self.max_pos:
            raise UnsupportedModel('ngram_sizes larger than the context')
        self.group = self.n_head // self.n_kv
        self.q_dim = self.n_head * self.head_dim
        self.kv_dim = self.n_kv * self.head_dim
        self.max_context = self.max_pos

    @property
    def tied(self):
        return True

    def n_params(self):
        H, E = self.hidden, self.n_expert
        per = (2 * H + self.q_dim * H + 2 * self.kv_dim * H + H * self.q_dim
               + self.q_dim + self.kv_dim + E * H + E + 2 * self.shared * H
               + H * self.shared + E * 3 * self.ffn * H + 2
               + self.n_head * self.gate_dim)
        return (self.n_layer * per + 2 * (self.n_layer - 1) + self.vocab * H
                + len(self.ng_sizes) * self.ng_buckets * self.ng_dim
                + H * len(self.ng_sizes) * self.ng_dim + 2 * H)

    def n_active_params(self):
        H = self.hidden
        per = (2 * H + self.q_dim * H + 2 * self.kv_dim * H + H * self.q_dim
               + self.q_dim + self.kv_dim + self.n_expert * H
               + self.n_expert + 2 * self.shared * H + H * self.shared
               + 3 * self.ffn * H + 2 + self.n_head * self.gate_dim)
        return (self.n_layer * per + 2 * (self.n_layer - 1) + self.vocab * H
                + len(self.ng_sizes) * self.ng_buckets * self.ng_dim
                + H * len(self.ng_sizes) * self.ng_dim + 2 * H)

    def kv_bytes(self, ctx):
        return 2 * self.n_layer * self.kv_dim * ctx * 4

    def describe(self):
        return ('aurora2: %d layers, hidden %d, %d heads / %d kv heads x %d, '
                '%d experts (top-1) of %d + shared %d, %d-gram hash '
                'embeddings, vocab %d, tied embeddings; ~%.0fM of %.0fM '
                'parameters used per token'
                % (self.n_layer, self.hidden, self.n_head, self.n_kv,
                   self.head_dim, self.n_expert, self.ffn, self.shared,
                   max(self.ng_sizes), self.vocab,
                   self.n_active_params() / 1e6, self.n_params() / 1e6))


def ngram_hashes(ids, n, seed, buckets):
    '''Bucket of the causal hash of the n ids ending at each position.

    ids: int64 (T,). Positions before the first id read as 0. The
    arithmetic is the reference's, in int64 where torch used it too: every
    intermediate stays below 2**52.'''
    ids = np.asarray(ids, dtype=np.int64)
    T = ids.shape[0]
    h = np.full(T, seed + 7919 * n, dtype=np.int64)
    for k in range(n):
        xk = np.zeros(T, np.int64)
        if k < T:
            xk[k:] = ids[:T - k]
        h = (h * 1000003 + xk + 1) & HASH_MASK
        h = h ^ (h >> 13)
    h = (h * 40503) & HASH_MASK
    return (h >> 4) % buckets


class Aurora2Cache(object):
    '''Per layer: rotated keys and values (n_kv, capacity, head_dim).
    `length` positions are filled; `tokens` are the ids behind them (the
    n-gram hash of a new token reads the last three of them).'''

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
    __slots__ = ('ln1', 'ln2', 'qkv', 'q_norm', 'k_norm', 'o', 'ag_w', 'lam',
                 'vl', 'router', 's13', 's2', 'w13', 'w2')


class Aurora2Model(object):

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
            # One slab must hold an expert's fused gate|up matrix.
            need = max(2 * cfg.ffn, cfg.vocab // 8, 1) * cfg.hidden
            scratch = np.empty(max(scratch_mb << 20, need * 4) // 4,
                               np.float32)
        self.scratch = scratch
        H, hd, E = cfg.hidden, cfg.head_dim, cfg.n_expert

        def raw(name, shape):
            if name not in ckpt:
                raise UnsupportedModel('checkpoint is missing %r' % name)
            t = ckpt.info(name)
            if t.shape != tuple(shape):
                raise UnsupportedModel('%s has shape %r, config implies %r'
                                       % (name, t.shape, tuple(shape)))
            return ckpt.get_raw(name), t.dtype

        def vec(name, shape):
            return to_f32(*raw(name, shape))

        def lin(names_shapes, bias=None):
            return Linear([raw(n, s) for n, s in names_shapes], weights,
                          self.mv, scratch, bias)

        self.layers = []
        for i in range(cfg.n_layer):
            p = 'model.blocks.%d.' % i
            L = _Layer()
            L.ln1 = vec(p + 'ln1', (H,))
            L.ln2 = vec(p + 'ln2', (H,))
            L.qkv = lin([(p + 'wq', (cfg.q_dim, H)),
                         (p + 'wk', (cfg.kv_dim, H)),
                         (p + 'wv', (cfg.kv_dim, H))])
            L.q_norm = vec(p + 'q_norm', (cfg.n_head, hd))
            L.k_norm = vec(p + 'k_norm', (cfg.n_kv, hd))
            L.o = lin([(p + 'wo', (H, cfg.q_dim))])
            L.ag_w = vec(p + 'ag_w', (cfg.n_head, cfg.gate_dim))
            lam = vec(p + 'lam', (2,))
            L.lam = (np.float32(lam[0]), np.float32(lam[1]))
            if i > 0:
                vl = vec(p + 'vl', (2,))
                L.vl = (np.float32(vl[0]), np.float32(vl[1]))
            else:
                L.vl = None
            L.router = lin([(p + 'router_w', (E, H))],
                           bias=vec(p + 'router_b', (E,)))
            L.s13 = lin([(p + 's13', (2 * cfg.shared, H))])
            L.s2 = lin([(p + 's2', (H, cfg.shared))])
            w13, dt13 = raw(p + 'w13', (E, 2 * cfg.ffn, H))
            w2, dt2 = raw(p + 'w2', (E, H, cfg.ffn))
            L.w13 = [Linear([(w13[e], dt13)], weights, self.mv, scratch)
                     for e in range(E)]
            L.w2 = [Linear([(w2[e], dt2)], weights, self.mv, scratch)
                    for e in range(E)]
            self.layers.append(L)
        self.norm = vec('model.final_norm', (H,))
        self.ng_emb = vec('model.ng_emb', (len(cfg.ng_sizes), cfg.ng_buckets,
                                           cfg.ng_dim))
        self.ng_proj = vec('model.ng_proj', (H, len(cfg.ng_sizes)
                                             * cfg.ng_dim))
        self.ng_gate = vec('model.ng_gate', (H,))

        emb_shape = (cfg.vocab, H)
        self.head = lin([('model.tok_emb.weight', emb_shape)])
        if weights == 'float32':
            self._emb_raw = None
            self._emb = self.head.W
        else:
            self._emb = None
            self._emb_raw = raw('model.tok_emb.weight', emb_shape)

        self.rope = rope_tables(hd, cfg.rope_theta, 1.0, self.ctx_limit)
        self.scale = np.float32(1.0 / math.sqrt(hd))
        if weights == 'float32':
            self.ckpt = None
            ckpt.close()

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_dir(cls, path, **kw):
        with open(os.path.join(path, 'config.json'), 'rb') as f:
            cfg = Aurora2Config(json.loads(f.read().decode('utf-8')))
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
        return Aurora2Cache(self.cfg, capacity or self.ctx_limit)

    def embed(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.cfg.vocab):
            raise ValueError('token id out of range for vocab %d'
                             % self.cfg.vocab)
        if self._emb is not None:
            return self._emb[ids]
        r, dt = self._emb_raw
        return to_f32(r[ids], dt)

    def _embed_tokens(self, ids, cache):
        '''Token embedding plus the gated n-gram features, (T, hidden).'''
        cfg = self.cfg
        T = len(ids)
        e = np.array(self.embed(ids), dtype=np.float32).reshape(T, cfg.hidden)
        # The hashes read up to max(n) - 1 ids before the new ones.
        keep = max(cfg.ng_sizes) - 1
        hist = cache.tokens[-keep:] if keep else []
        full = np.array(list(hist) + [int(t) for t in ids], dtype=np.int64)
        feats = []
        for i, n in enumerate(cfg.ng_sizes):
            b = ngram_hashes(full, n, cfg.ng_seed, cfg.ng_buckets)[-T:]
            feats.append(self.ng_emb[i][b])
        f = np.concatenate(feats, axis=1)
        e += self.ng_gate * np.dot(f, self.ng_proj.T)
        return e

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

    @staticmethod
    def _lin(lin, X, Y):
        if X.shape[0] == 1:
            lin.matvec(X[0], Y[0])
            return Y
        return lin.matmul(X, Y)

    def _forward(self, ids, cache, skip_logits):
        '''Run T tokens at positions cache.length.. through every layer.
        One code path for decode (T = 1) and prefill.'''
        cfg = self.cfg
        H, T = cfg.hidden, len(ids)
        p0 = cache.length
        S = p0 + T
        x0 = self._embed_tokens(ids, cache)
        x = x0.copy()
        v1 = None
        buf = {}

        def scratch(name, n):
            b = buf.get(name)
            if b is None:
                b = buf[name] = np.empty((T, n), np.float32)
            return b

        for li, L in enumerate(self.layers):
            x = L.lam[0] * x + L.lam[1] * x0
            a = np.ascontiguousarray(rms_norm(x, L.ln1, cfg.eps))
            y, v1 = self._attention(L, li, a, v1, cache, p0, scratch)
            x += y
            m = np.ascontiguousarray(rms_norm(x, L.ln2, cfg.eps))
            gu = self._lin(L.s13, m, scratch('s13', 2 * cfg.shared))
            s = self._lin(L.s2, np.ascontiguousarray(
                silu_mul(gu[:, :cfg.shared], gu[:, cfg.shared:])),
                scratch('s2', H))
            x = x + s + self._moe(L, m)
        cache.length = S
        cache.tokens.extend(int(t) for t in ids)
        if skip_logits:
            return None
        h = np.ascontiguousarray(rms_norm(x, self.norm, cfg.eps))
        z = self._lin(self.head, h, np.empty((T, cfg.vocab), np.float32))
        if cfg.softcap > 0:
            cap = np.float32(cfg.softcap)
            z = cap * np.tanh(z / cap)
        return z

    def _moe(self, L, m):
        '''Dropless top-1 routed SwiGLU, scaled by the router probability.'''
        cfg = self.cfg
        T = m.shape[0]
        rl = self._lin(L.router, m, np.empty((T, cfg.n_expert), np.float32))
        pr = softmax_last(rl)
        idx = pr.argmax(axis=-1)
        gate = pr[np.arange(T), idx]
        out = np.empty((T, cfg.hidden), np.float32)
        f = cfg.ffn
        for e in np.unique(idx):
            sel = np.nonzero(idx == e)[0]
            X = np.ascontiguousarray(m[sel])
            gu = self._lin(L.w13[e], X, np.empty((len(sel), 2 * f),
                                                 np.float32))
            a = np.ascontiguousarray(silu_mul(gu[:, :f], gu[:, f:]))
            out[sel] = self._lin(L.w2[e], a, np.empty((len(sel), cfg.hidden),
                                                      np.float32))
        out *= gate[:, None]
        return out

    def _attention(self, L, li, a, v1, cache, p0, scratch):
        '''Returns the attention block's contribution to the residual
        stream and the first layer's raw values (v1, passed along).'''
        cfg = self.cfg
        nkv, g, hd = cfg.n_kv, cfg.group, cfg.head_dim
        qd, kvd, nh = cfg.q_dim, cfg.kv_dim, cfg.n_head
        T = a.shape[0]
        S = p0 + T
        qkv = self._lin(L.qkv, a, scratch('qkv', qd + 2 * kvd))
        q = head_norm(qkv[:, :qd].reshape(T, nh, hd), L.q_norm, cfg.eps)
        k = head_norm(qkv[:, qd:qd + kvd].reshape(T, nkv, hd), L.k_norm,
                      cfg.eps)
        v = qkv[:, qd + kvd:].reshape(T, nkv, hd)
        if li == 0:
            v1 = v.copy()
        else:
            v = L.vl[0] * v + L.vl[1] * v1
        cos, sin = self.rope
        rope_rotate(q, cos[p0:S], sin[p0:S])
        rope_rotate(k, cos[p0:S], sin[p0:S])
        cache.k[li][:, p0:S] = k.transpose(1, 0, 2)
        cache.v[li][:, p0:S] = v.transpose(1, 0, 2)
        K = cache.k[li][:, :S]
        V = cache.v[li][:, :S]
        qg = q.reshape(T, nkv, g, hd).transpose(1, 2, 0, 3).reshape(
            nkv, g * T, hd)
        s = np.matmul(qg, K.transpose(0, 2, 1)).reshape(nkv, g, T, S)
        s *= self.scale
        if T > 1:
            mask = np.arange(S)[None, :] > (p0 + np.arange(T))[:, None]
            s[:, :, mask] = -np.inf
        pr = softmax_last(s).reshape(nkv, g * T, S)
        y = np.matmul(pr, V).reshape(nkv, g, T, hd).transpose(2, 0, 1, 3)
        # One multiplier per head: 2 * sigmoid(ag_w . first gate_dim
        # channels of the normed input).
        gate = np.float32(2.0) * sigmoid(
            np.dot(a[:, :cfg.gate_dim], L.ag_w.T))
        y = np.ascontiguousarray((y.reshape(T, nh, hd)
                                  * gate[:, :, None]).reshape(T, qd))
        return self._lin(L.o, y, scratch('out', cfg.hidden)), v1

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
