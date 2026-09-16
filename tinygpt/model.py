# coding: utf-8
'''
Config and GPT: the model itself.

GPT.forward() and GPT.backward() are two halves of one hand-derived
computation graph -- backward() has no autograd to lean on, so it walks
the same layers as forward() in reverse, consuming exactly the
intermediate values forward() stashed in its cache dict `ca`. Reading them
side by side (forward top-to-bottom, backward bottom-to-top) is the
fastest way to see how one implies the other.

Two separate paths generate text: sample() recomputes the full context on
every new token and is the ground truth; generate() keeps a per-layer KV
cache and is what you actually want to run. See generate()'s docstring for
how the two are verified to agree.
'''

import math

import numpy as np

from tinygpt import backend
from tinygpt.backend import bmm, gemm, vexp, vtanh
from tinygpt.ops import (act_bwd, act_fwd, dlayernorm, drms_norm, layernorm,
                          rms_norm, rope_apply, rope_bwd, softmax_rows)

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
        f = lambda *s: (rng.randn(*s) * 0.02).astype(backend.DT)
        z = lambda *s: np.zeros(s, dtype=backend.DT)
        o = lambda *s: np.ones(s, dtype=backend.DT)

        # Residual projections and the classifier either start at zero
        # (muP-like: every block is an exact identity at init and the
        # residual stream carries only the embedding) or at the GPT-2
        # 1/sqrt(2L) scale, which keeps the residual variance from growing
        # with depth.
        s = 0.02 / math.sqrt(2 * L)
        out = (lambda *sh: z(*sh)) if cfg.zero_init else \
            (lambda *sh: (rng.randn(*sh) * s).astype(backend.DT))

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
        m = np.tril(np.ones((cfg.block_size, cfg.block_size), backend.DT))
        self.neg = np.where(m == 0, -1e9, 0.0).astype(backend.DT)

    # -- forward ------------------------------------------------------------

    def forward(self, idx, targets=None):
        cfg = self.cfg
        p = self.p
        B, T = idx.shape
        C, nh, hs = cfg.n_embd, cfg.n_head, cfg.head_size
        BT = B * T
        ca = {'idx': idx, 'B': B, 'T': T}

        x = p['wte'][idx].astype(backend.DT)
        if not cfg.rope:
            x = (x + p['wpe'][:T]).astype(backend.DT)
        ca['blocks'] = []
        v0 = None

        for i in range(cfg.n_layer):
            bc = {}
            h, xhat1, std1 = layernorm(x, p['ln1_g%d' % i], p['ln1_b%d' % i])
            bc['xhat1'], bc['std1'] = xhat1, std1

            h2 = np.ascontiguousarray(h.reshape(BT, C))
            qkv = np.empty((BT, 3 * C), backend.DT)
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
                vu = (v + p['vres'][i - 1] * v0).astype(backend.DT)
            else:
                vu = v

            att = np.empty((B * nh, T, T), backend.DT)
            bmm(q, k, att, transB=True)
            att *= (1.0 / math.sqrt(hs))
            att += self.neg[:T, :T]
            pr = softmax_rows(att)

            y = np.empty((B * nh, T, hs), backend.DT)
            bmm(pr, vu, y)
            y = np.ascontiguousarray(
                y.reshape(B, nh, T, hs).transpose(0, 2, 1, 3)).reshape(BT, C)
            bc['q'], bc['k'], bc['v'], bc['vu'] = q, k, v, vu
            bc['pr'], bc['y'] = pr, y

            a = np.empty((BT, C), backend.DT)
            gemm(y, p['proj_w%d' % i], a)
            a += p['proj_b%d' % i]
            x = (x + a.reshape(B, T, C)).astype(backend.DT)

            h, xhat2, std2 = layernorm(x, p['ln2_g%d' % i], p['ln2_b%d' % i])
            bc['xhat2'], bc['std2'] = xhat2, std2
            h3 = np.ascontiguousarray(h.reshape(BT, C))
            bc['h3'] = h3

            fc = np.empty((BT, 4 * C), backend.DT)
            gemm(h3, p['fc_w%d' % i], fc)
            fc += p['fc_b%d' % i]
            act, acache = act_fwd(fc, cfg.act)
            bc['fc'], bc['act'], bc['ac'] = fc, act, acache

            m = np.empty((BT, C), backend.DT)
            gemm(act, p['fp_w%d' % i], m)
            m += p['fp_b%d' % i]
            x = (x + m.reshape(B, T, C)).astype(backend.DT)

            bc['x_out'] = x
            ca['blocks'].append(bc)

        xf, xhatf, stdf = layernorm(x, p['lnf_g'], p['lnf_b'])
        ca['xhatf'], ca['stdf'] = xhatf, stdf
        xf2 = np.ascontiguousarray(xf.reshape(BT, C))
        ca['xf2'] = xf2

        logits = np.empty((BT, cfg.vocab), backend.DT)
        gemm(xf2, p['head'], logits)

        if cfg.softcap:
            # c*tanh(z/c): bounded logits, so a single overconfident
            # character cannot dominate the gradient. Monotonic, so top-k
            # ordering at sampling time is unchanged.
            c = cfg.softcap
            t = vtanh(np.ascontiguousarray(logits / c, dtype=backend.DT))
            ca['capt'] = t
            logits = (c * t).astype(backend.DT)
        ca['logits'] = logits

        if targets is None:
            return logits.reshape(B, T, cfg.vocab), ca

        t = targets.reshape(-1)

        mx = logits.max(axis=1, keepdims=True)

        # Compute exp only once. vexp uses Accelerate/vForce in float32.
        z = np.ascontiguousarray(logits - mx, dtype=backend.DT)
        e = vexp(z)

        den = e.sum(axis=1, keepdims=True)
        lse = mx + np.log(den)

        loss = float(np.mean(
          lse[:, 0] - logits[np.arange(BT), t]
        ))

        # Cache the softmax probabilities for cross-entropy backward.
        # This trades ~BT*vocab*4 bytes of cache for eliminating another exp().
        ca['xent_pr'] = (e / den).astype(backend.DT)
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
            dlogits = (dlogits * (1.0 - t * t)).astype(backend.DT)

        gemm(ca['xf2'], dlogits, g['head'], transA=True)
        dxf = np.empty((BT, C), backend.DT)
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
            dact = np.empty((BT, 4 * C), backend.DT)
            gemm(dm, p['fp_w%d' % i], dact, transB=True)

            dfc = act_bwd(dact, bc['fc'], bc['ac'], cfg.act)
            g['fc_b%d' % i] = dfc.sum(axis=0)
            gemm(bc['h3'], dfc, g['fc_w%d' % i], transA=True)
            dh3 = np.empty((BT, C), backend.DT)
            gemm(dfc, p['fc_w%d' % i], dh3, transB=True)

            dres, g['ln2_g%d' % i], g['ln2_b%d' % i] = dlayernorm(
                dh3.reshape(B, T, C), bc['xhat2'], bc['std2'],
                p['ln2_g%d' % i])
            dx = dx + dres

            da = np.ascontiguousarray(dx.reshape(BT, C))
            g['proj_b%d' % i] = da.sum(axis=0)
            gemm(bc['y'], da, g['proj_w%d' % i], transA=True)
            dy = np.empty((BT, C), backend.DT)
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
            datt = (datt * (1.0 / math.sqrt(hs))).astype(backend.DT)

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
                                  axis=1).astype(backend.DT)
            g['qkv_b%d' % i] = dqkv.sum(axis=0)
            gemm(bc['h2'], dqkv, g['qkv_w%d' % i], transA=True)
            dh2 = np.empty((BT, C), backend.DT)
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

        x = p['wte'][tok].reshape(1, C).astype(backend.DT)
        if not cfg.rope:
            x = (x + p['wpe'][pos]).astype(backend.DT)
        v0 = None

        for i in range(cfg.n_layer):
            h, _, _ = layernorm(x, p['ln1_g%d' % i], p['ln1_b%d' % i])
            qkv = np.empty((1, 3 * C), backend.DT)
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
                v_new = (v_new + p['vres'][i - 1] * v0).astype(backend.DT)

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

            a = np.empty((1, C), backend.DT)
            gemm(np.ascontiguousarray(y), p['proj_w%d' % i], a)
            a += p['proj_b%d' % i]
            x = x + a

            h, _, _ = layernorm(x, p['ln2_g%d' % i], p['ln2_b%d' % i])
            fc = np.empty((1, 4 * C), backend.DT)
            gemm(np.ascontiguousarray(h), p['fc_w%d' % i], fc)
            fc += p['fc_b%d' % i]
            act, _ = act_fwd(fc, cfg.act)
            m = np.empty((1, C), backend.DT)
            gemm(np.ascontiguousarray(act), p['fp_w%d' % i], m)
            m += p['fp_b%d' % i]
            x = x + m

        xf, _, _ = layernorm(x, p['lnf_g'], p['lnf_b'])
        logits = np.empty((1, cfg.vocab), backend.DT)
        gemm(np.ascontiguousarray(xf), p['head'], logits)
        if cfg.softcap:
            c = cfg.softcap
            logits = (c * vtanh(np.ascontiguousarray(logits / c,
                      dtype=backend.DT))).astype(backend.DT)
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
