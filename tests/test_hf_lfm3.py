# coding: utf-8
'''LFM3 inference (tinygpt/hf/lfm3.py).

tiny_lfm3_via_lfm2.json checks the short-conv block, QK-norm attention
and final norm against transformers' built-in LFM2. tiny_lfm3.json adds
what LFM2 lacks (sandwich norms, the output gate, zero-tail partial RoPE,
sliding-window layers) against the PyTorch transcription in
tests/make_hf_fixtures.py. The LFM3-1.7B test runs when it is cached.
'''

import unittest

import numpy as np

from tests.hf_common import b64_array, fixture, lfm3_dir, tiny_model_dir
from tinygpt.hf.lfm3 import Lfm3Config, Lfm3Model, rope_inv_freq
from tinygpt.hf.llama import ContextFull, Sampler, UnsupportedModel, generate
from tinygpt.hf.models import config_from_dict, model_from_dir

TINY = ['tiny_lfm3_via_lfm2.json', 'tiny_lfm3.json']


def load_tiny(name, **kw):
    td, d = tiny_model_dir(name)
    ref = b64_array(d['logits'], d['logits_shape'])
    m = model_from_dir(td, **kw)
    return m, d, ref


class TinyLfm3Test(unittest.TestCase):

    def assertClose(self, got, ref, what):
        err = float(np.abs(got - ref).max())
        self.assertLess(err, 2e-4 * max(1.0, float(np.abs(ref).max())),
                        '%s: max error %.3g' % (what, err))

    def test_against_reference(self):
        for name in TINY:
            for w in ('float32', 'lowmem'):
                m, d, ref = load_tiny(name, ctx_limit=64, weights=w,
                                      scratch_mb=0)
                self.addCleanup(m.close)
                self.assertIsInstance(m, Lfm3Model)
                for chunk in (1, 4, 64):
                    c = m.new_cache()
                    got = m.prefill(d['ids'], c, all_logits=True,
                                    chunk=chunk)
                    self.assertClose(got, ref, '%s %s prefill chunk %d'
                                     % (name, w, chunk))
                c = m.new_cache()
                got = np.stack([m.step(t, c) for t in d['ids']])
                self.assertClose(got, ref, '%s %s cached steps' % (name, w))

    def test_feed_reuses_prefix(self):
        # The conv layers' history and the sliding window must both be
        # right after rewinding into the middle of the cache.
        m, d, ref = load_tiny(TINY[1], ctx_limit=64)
        ids = d['ids']
        c = m.new_cache()
        m.feed(ids[:20], c)
        m.feed(ids[:9] + [7, 8, 9, 10, 11, 12, 13], c)
        got = m.feed(ids, c)
        self.assertEqual(c.tokens, list(ids))
        self.assertClose(got, ref[-1], 'feed')

    def test_context_not_capped_by_window(self):
        m, d, _ = load_tiny(TINY[1], ctx_limit=40)
        self.assertEqual(m.cfg.sliding_window, 5)
        self.assertEqual(m.ctx_limit, 40)
        c = m.new_cache()
        m.prefill(d['ids'], c)
        out, reason, _ = generate(m, c, d['ids'], 50, Sampler(0.0))
        self.assertEqual(reason, 'context')
        with self.assertRaises(ContextFull):
            m.step(3, c)

    def test_zero_tail_rope(self):
        inv = rope_inv_freq(128, 1e6, 0.5)
        self.assertEqual(inv.shape, (64,))
        self.assertTrue(np.all(inv[32:] == 0))
        np.testing.assert_allclose(inv[:32], 1e6 ** (-np.arange(32) / 64.0),
                                   rtol=1e-6)
        self.assertTrue(np.all(rope_inv_freq(128, 1e4, 1.0) > 0))

    def test_config(self):
        base = fixture(TINY[1])['config']
        cfg = config_from_dict(base)
        self.assertIsInstance(cfg, Lfm3Config)
        self.assertEqual(cfg.max_context, base['max_position_embeddings'])
        self.assertIn('2 sliding window 5', cfg.describe())
        for k, v in (('conv_bias', True), ('layer_types', ['conv'] * 5),
                     ('layer_types', ['conv'] * 5 + ['mamba']),
                     ('architectures', ['Lfm2ForCausalLM']),
                     ('rope_parameters', {
                         'full_attention': {'rope_type': 'yarn',
                                            'rope_theta': 1e6},
                         'sliding_attention': {'rope_theta': 1e4}})):
            with self.assertRaises(UnsupportedModel):
                Lfm3Config(dict(base, **{k: v}))
        with self.assertRaises(UnsupportedModel):
            config_from_dict(dict(base, model_type='lfm9'))

    def test_n_params(self):
        from tinygpt.hf.safetensors import Checkpoint
        for name in TINY:
            td, d = tiny_model_dir(name)
            ck = Checkpoint(td)
            n = sum(int(np.prod(ck.info(k).shape)) for k in ck.keys())
            ck.close()
            self.assertEqual(Lfm3Config(d['config']).n_params(), n)


@unittest.skipUnless(lfm3_dir(), 'LFM3-1.7B not cached')
class Lfm3RealTest(unittest.TestCase):
    '''The real checkpoint, lowmem. There is no float32 reference fixture
    (that would mean running the repository's own modeling code), so this
    checks the parameter count and that greedy decoding says something
    sensible.'''

    def test_greedy(self):
        from tinygpt.hf.tokenizer import BPETokenizer
        path = lfm3_dir()
        m = model_from_dir(path, ctx_limit=64, weights='lowmem')
        self.addCleanup(m.close)
        self.assertEqual(m.cfg.n_params(), 1728801536)
        tok = BPETokenizer.from_dir(path)
        ids = tok.encode('<|startoftext|>The capital of France is')
        self.assertEqual(ids[0], 124894)
        out, _, _ = generate(m, m.new_cache(), ids, 3, Sampler(0.0))
        self.assertEqual(tok.decode(out), ' Paris. Therefore')


if __name__ == '__main__':
    unittest.main()
