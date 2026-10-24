# coding: utf-8
'''Llama inference against transformers' float32 reference logits.

Tiny random models (GQA, MHA, biases, untied head, llama3 RoPE scaling,
BF16/F16/F32 storage) always run. The SmolLM2 tests run when the model
is cached.
'''

import socket
import unittest

import numpy as np

from tests.hf_common import b64_array, fixture, smollm2_dir, tiny_model_dir
from tinygpt.hf.llama import (ContextFull, LlamaConfig, LlamaModel, Sampler,
                              UnsupportedModel, generate)

TINY = ['tiny_llama_tied_bf16.json', 'tiny_llama_untied_f16.json',
        'tiny_llama_mha_f32.json']


def load_tiny(name, **kw):
    td, d = tiny_model_dir(name)
    ref = b64_array(d['logits'], d['logits_shape'])
    return LlamaModel.from_dir(td, **kw), d, ref


class TinyLlamaTest(unittest.TestCase):

    def assertClose(self, got, ref, what):
        err = float(np.abs(got - ref).max())
        self.assertLess(err, 2e-4 * max(1.0, float(np.abs(ref).max())),
                        '%s: max error %.3g' % (what, err))

    def test_against_transformers(self):
        for name in TINY:
            for w in ('float32', 'lowmem'):
                m, d, ref = load_tiny(name, ctx_limit=128, weights=w,
                                      scratch_mb=0)
                self.addCleanup(m.close)
                for chunk in (1, 5, 64):
                    c = m.new_cache()
                    got = m.prefill(d['ids'], c, all_logits=True,
                                    chunk=chunk)
                    self.assertClose(got, ref, '%s %s prefill chunk %d'
                                     % (name, w, chunk))
                c = m.new_cache()
                got = np.stack([m.step(t, c) for t in d['ids']])
                self.assertClose(got, ref, '%s %s cached steps' % (name, w))

    def test_cached_equals_uncached(self):
        # Step-by-step decoding with the KV cache vs a fresh full-sequence
        # forward for every prefix: the uncached ground truth.
        m, d, _ = load_tiny(TINY[0], ctx_limit=64)
        ids = d['ids'][:16]
        c = m.new_cache()
        steps = [m.step(t, c) for t in ids]
        for n in range(1, len(ids) + 1):
            full = m.prefill(ids[:n], m.new_cache())
            np.testing.assert_allclose(steps[n - 1], full, atol=1e-4,
                                       rtol=1e-4)

    def test_feed_reuses_prefix(self):
        m, d, ref = load_tiny(TINY[1], ctx_limit=64)
        ids = d['ids']
        c = m.new_cache()
        m.feed(ids[:10], c)
        m.feed(ids[:6] + [7, 8, 9], c)              # diverge, then return
        got = m.feed(ids, c)
        self.assertEqual(c.tokens, list(ids))
        np.testing.assert_allclose(got, ref[-1], atol=2e-4, rtol=1e-4)

    def test_context_limit(self):
        m, d, _ = load_tiny(TINY[0], ctx_limit=8)
        c = m.new_cache()
        m.prefill(d['ids'][:8], c)
        with self.assertRaises(ContextFull):
            m.step(3, c)
        with self.assertRaises(ContextFull):
            m.prefill(d['ids'][:9], m.new_cache())
        out, reason, _ = generate(m, m.new_cache(), d['ids'][:5], 50)
        self.assertEqual(reason, 'context')
        self.assertEqual(len(out), 4)

    def test_generate_stops(self):
        m, d, _ = load_tiny(TINY[0], ctx_limit=64)
        out, _, _ = generate(m, m.new_cache(), d['ids'][:4], 10)
        out2, reason, _ = generate(m, m.new_cache(), d['ids'][:4], 10,
                                   stop_ids={out[2]})
        self.assertEqual(reason, 'stop')
        self.assertEqual(out2, out[:out.index(out[2])])

    def test_bad_token_id(self):
        m, _, _ = load_tiny(TINY[0], ctx_limit=16)
        with self.assertRaises(ValueError):
            m.step(10 ** 6, m.new_cache())

    def test_unsupported(self):
        base = fixture(TINY[0])['config']
        for k, v in (('model_type', 'mistral'), ('hidden_act', 'gelu'),
                     ('architectures', ['GPT2LMHeadModel']),
                     ('num_key_value_heads', 4)):
            with self.assertRaises(UnsupportedModel):
                LlamaConfig(dict(base, **{k: v}))
        with self.assertRaises(UnsupportedModel):
            cfg = dict(base, rope_parameters={'rope_type': 'yarn',
                                              'rope_theta': 1e4})
            LlamaModel.__new__(LlamaModel)
            from tinygpt.hf.llama import rope_inv_freq
            rope_inv_freq(LlamaConfig(cfg))

    def test_missing_tensor(self):
        import os
        from tests.hf_common import write_safetensors
        from tinygpt.hf.safetensors import Checkpoint
        td, d = tiny_model_dir(TINY[0])
        ck = Checkpoint(td)
        t = {k: (ck.info(k).dtype, np.array(ck.get_raw(k)))
             for k in ck.keys() if 'layers.1.mlp.up' not in k}
        ck.close()
        write_safetensors(os.path.join(td, 'model.safetensors'), t)
        with self.assertRaises(UnsupportedModel):
            LlamaModel.from_dir(td)


class SamplerTest(unittest.TestCase):

    def test_greedy_and_top_k1(self):
        z = np.array([0.1, 3.0, -1.0, 2.9], np.float32)
        self.assertEqual(Sampler(0.0)(z), 1)
        self.assertEqual(Sampler(1.0, top_k=1)(z), 1)
        self.assertEqual(Sampler(1.0, top_p=1e-6)(z), 1)

    def test_seeded(self):
        z = np.random.RandomState(0).standard_normal(100).astype(np.float32)
        a = [Sampler(0.8, seed=5)(z) for _ in range(3)]
        s1, s2 = Sampler(0.8, seed=5), Sampler(0.8, seed=5)
        self.assertEqual([s1(z) for _ in range(20)],
                         [s2(z) for _ in range(20)])
        self.assertEqual(len(set(a)), 1)

    def test_top_k_support(self):
        z = np.arange(10, dtype=np.float32)
        s = Sampler(5.0, top_k=3, seed=0)
        self.assertTrue(set(s(z) for _ in range(200)) <= {7, 8, 9})

    def test_repetition_penalty(self):
        z = np.array([2.0, 1.9], np.float32)
        self.assertEqual(Sampler(0.0, repetition_penalty=1.2)(z, [0]), 1)


class _NoNetwork(object):
    def __enter__(self):
        self.saved = socket.socket.connect

        def refuse(*a, **k):
            raise AssertionError('network access during offline inference')
        socket.socket.connect = refuse

    def __exit__(self, *a):
        socket.socket.connect = self.saved


@unittest.skipUnless(smollm2_dir(), 'SmolLM2 not cached')
class SmolLM2Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with _NoNetwork():
            cls.model = LlamaModel.from_dir(smollm2_dir(), ctx_limit=256)
        cls.d = fixture('smollm2_logits.json')

    @classmethod
    def tearDownClass(cls):
        del cls.model

    def test_logits(self):
        d = self.d
        ids, cols = d['ids'], d['logits_cols']
        ref = b64_array(d['logits_sel'], (len(ids), len(cols)))
        last = b64_array(d['last_logits'])
        got = self.model.prefill(ids, self.model.new_cache(),
                                 all_logits=True)
        self.assertLess(np.abs(got[:, cols] - ref).max(), 1e-3)
        self.assertLess(np.abs(got[-1] - last).max(), 1e-3)
        c = self.model.new_cache()
        steps = np.stack([self.model.step(t, c) for t in ids])
        self.assertLess(np.abs(steps[:, cols] - ref).max(), 1e-3)

    def test_greedy_offline(self):
        want = list(self.d['greedy'])
        if want and want[-1] == 2:
            want = want[:-1]           # transformers keeps the eos token
        with _NoNetwork():
            out, reason, _ = generate(self.model, self.model.new_cache(),
                                      self.d['ids'], 48, Sampler(0.0),
                                      stop_ids={2})
        self.assertEqual(out, want)


if __name__ == '__main__':
    unittest.main()
