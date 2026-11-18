# coding: utf-8
'''Aurora2 inference (tinygpt/hf/aurora2.py).

tiny_aurora2.json: a random tiny model with every Aurora2 feature (hashed
n-gram embeddings, embedding skip, value residual, per-head output gate,
several routed experts, a logit softcap that bites) against the PyTorch
transcription in tests/make_hf_fixtures.py. transformers has no built-in
aurora2 and the repository's own code is never run, so that transcription
is the reference; aurora2_logits.json is the same transcription run on the
real checkpoint. Real-model tests run when it is cached.
'''

import unittest

import numpy as np

from tests.hf_common import (aurora2_dir, b64_array, fixture,
                             tiny_model_dir)
from tinygpt.hf.aurora2 import (Aurora2Config, Aurora2Model, ngram_hashes)
from tinygpt.hf.llama import ContextFull, Sampler, UnsupportedModel, generate
from tinygpt.hf.models import config_from_dict, model_from_dir
from tinygpt.hf.tokenizer import BPETokenizer, UnsupportedTokenizer

TINY = 'tiny_aurora2.json'


def load_tiny(**kw):
    td, d = tiny_model_dir(TINY)
    ref = b64_array(d['logits'], d['logits_shape'])
    m = model_from_dir(td, **kw)
    return m, d, ref


class TinyAurora2Test(unittest.TestCase):

    def assertClose(self, got, ref, what):
        err = float(np.abs(got - ref).max())
        self.assertLess(err, 2e-4 * max(1.0, float(np.abs(ref).max())),
                        '%s: max error %.3g' % (what, err))

    def test_against_reference(self):
        for w in ('float32', 'lowmem'):
            m, d, ref = load_tiny(ctx_limit=64, weights=w, scratch_mb=0)
            self.addCleanup(m.close)
            self.assertIsInstance(m, Aurora2Model)
            for chunk in (1, 4, 7, 64):
                c = m.new_cache()
                got = m.prefill(d['ids'], c, all_logits=True, chunk=chunk)
                self.assertClose(got, ref, '%s prefill chunk %d' % (w, chunk))
            c = m.new_cache()
            got = np.stack([m.step(t, c) for t in d['ids']])
            self.assertClose(got, ref, '%s cached steps' % w)

    def test_softcap_bites(self):
        # The fixture would not notice a missing softcap otherwise.
        _, d, ref = load_tiny(ctx_limit=64)
        cap = d['config']['logit_softcap']
        self.assertGreater(np.abs(ref).max(), 0.5 * cap)
        self.assertLess(np.abs(ref).max(), cap)

    def test_feed_reuses_prefix(self):
        # The n-gram hash reads the cached ids, so rewinding into the
        # middle of the cache must leave them right.
        m, d, ref = load_tiny(ctx_limit=64)
        self.addCleanup(m.close)
        ids = d['ids']
        c = m.new_cache()
        m.feed(ids[:20], c)
        m.feed(ids[:9] + [7, 8, 9, 10, 11, 12, 13], c)
        got = m.feed(ids, c)
        self.assertEqual(c.tokens, list(ids))
        self.assertClose(got, ref[-1], 'feed')

    def test_context_full(self):
        m, d, _ = load_tiny(ctx_limit=40)
        self.addCleanup(m.close)
        self.assertEqual(m.ctx_limit, 40)
        c = m.new_cache()
        m.prefill(d['ids'], c)
        out, reason, _ = generate(m, c, d['ids'], 50, Sampler(0.0))
        self.assertEqual(reason, 'context')
        with self.assertRaises(ContextFull):
            m.step(3, c)

    def test_ngram_hash(self):
        # Hand-computed from the reference's arithmetic for n = 2 at the
        # second position: ids (5, 9), the one before the start counts as 0.
        seed, n, nb = 1337, 2, 16384
        h = seed + 7919 * n
        for x in (9, 5):                 # k = 0: current id, k = 1: before
            h = (h * 1000003 + x + 1) & 0x7FFFFFFF
            h ^= h >> 13
        want = (((h * 40503) & 0x7FFFFFFF) >> 4) % nb
        got = ngram_hashes([5, 9], n, seed, nb)
        self.assertEqual(int(got[1]), want)
        # Causal: later ids never change an earlier hash.
        longer = ngram_hashes([5, 9, 77, 4], n, seed, nb)
        self.assertEqual(list(longer[:2]), list(got))
        # Position 0 sees zeros before it, whatever the table size.
        self.assertTrue(all(0 <= v < 64 for v in ngram_hashes(
            list(range(40)), 4, seed, 64)))

    def test_config(self):
        base = fixture(TINY)['config']
        cfg = config_from_dict(base)
        self.assertIsInstance(cfg, Aurora2Config)
        self.assertEqual(cfg.max_context, base['max_position_embeddings'])
        self.assertEqual(cfg.eos_ids, {1})
        self.assertIn('5 experts', cfg.describe())
        for k, v in (('n_kv_heads', 3), ('head_dim', 7), ('gate_dim', 0),
                     ('gate_dim', 25), ('ngram_sizes', []),
                     ('architectures', ['Aurora1ForCausalLM'])):
            with self.assertRaises(UnsupportedModel):
                Aurora2Config(dict(base, **{k: v}))
        with self.assertRaises(UnsupportedModel):
            config_from_dict(dict(base, model_type='aurora9'))

    def test_checkpoint_mismatch_is_refused(self):
        td, d = tiny_model_dir(TINY)
        import json
        import os
        cfg = dict(d['config'], num_experts=6)
        with open(os.path.join(td, 'config.json'), 'w') as f:
            json.dump(cfg, f)
        with self.assertRaises(UnsupportedModel):
            model_from_dir(td, ctx_limit=16)

    def test_n_params(self):
        from tinygpt.hf.safetensors import Checkpoint
        td, d = tiny_model_dir(TINY)
        ck = Checkpoint(td)
        n = sum(int(np.prod(ck.info(k).shape)) for k in ck.keys())
        ck.close()
        self.assertEqual(Aurora2Config(d['config']).n_params(), n)
        # The shipped config: the totals the model card states.
        real = Aurora2Config({'model_type': 'aurora2'})
        self.assertEqual(real.n_params(), 399597454)
        self.assertEqual(real.n_active_params(), 19050382)


class Aurora2TokenizerTest(unittest.TestCase):
    '''Needs the real tokenizer.json; the fixture holds `tokenizers`' ids.'''

    @unittest.skipUnless(aurora2_dir(), 'Aurora2-400M-A20M not cached')
    def test_matches_reference(self):
        tok = BPETokenizer.from_dir(aurora2_dir())
        d = fixture('aurora2_tokens.json')
        self.assertEqual(tok.prefix_ids, [0])
        for t, ids, plain, dec in zip(d['texts'], d['ids'], d['ids_plain'],
                                      d['decoded']):
            self.assertEqual(tok.encode(t, add_special_tokens=True), ids, t)
            self.assertEqual(tok.encode(t), plain, t)
            self.assertEqual(tok.decode(ids), dec, t)

    def test_post_processors(self):
        spec = fixture('bpe_tiny.json')['tokenizer']
        self.assertEqual(BPETokenizer(spec).prefix_ids, [])
        bos = {'type': 'TemplateProcessing',
               'single': [{'SpecialToken': {'id': '<|endoftext|>',
                                            'type_id': 0}},
                          {'Sequence': {'id': 'A', 'type_id': 0}}],
               'special_tokens': {'<|endoftext|>': {
                   'id': '<|endoftext|>', 'ids': [0],
                   'tokens': ['<|endoftext|>']}}}
        tok = BPETokenizer(dict(spec, post_processor=bos))
        text = 'hello'
        plain = tok.encode(text)
        self.assertEqual(tok.encode(text, add_special_tokens=True),
                         [0] + plain)
        eos = dict(bos, single=bos['single'] + [bos['single'][0]])
        tok = BPETokenizer(dict(spec, post_processor=eos))
        self.assertEqual(tok.encode(text, add_special_tokens=True),
                         [0] + plain + [0])
        for bad in (dict(bos, single=[bos['single'][0]]),
                    dict(bos, special_tokens={}),
                    {'type': 'BertProcessing'}):
            with self.assertRaises(UnsupportedTokenizer):
                BPETokenizer(dict(spec, post_processor=bad))


@unittest.skipUnless(aurora2_dir(), 'Aurora2-400M-A20M not cached')
class Aurora2RealTest(unittest.TestCase):
    '''The real checkpoint against aurora2_logits.json, in both weight
    modes, through prefill and cached decoding.'''

    def test_logits_and_greedy(self):
        d = fixture('aurora2_logits.json')
        cols = d['cols']
        ref = b64_array(d['logits_cols'], (len(d['ids']), len(cols)))
        last = b64_array(d['last_logits'])
        tok = BPETokenizer.from_dir(aurora2_dir())
        self.assertEqual(tok.encode(d['prompt'], add_special_tokens=True),
                         d['ids'])
        for w in ('float32', 'lowmem'):
            m = model_from_dir(aurora2_dir(), ctx_limit=128, weights=w)
            self.addCleanup(m.close)
            self.assertEqual(m.cfg.n_params(), 399597454)
            c = m.new_cache()
            got = m.prefill(d['ids'], c, all_logits=True)
            err = float(np.abs(got[:, cols] - ref).max())
            self.assertLess(err, 1e-3, '%s prefill %.3g' % (w, err))
            self.assertLess(float(np.abs(got[-1] - last).max()), 1e-3)
            c = m.new_cache()
            steps = np.stack([m.step(t, c) for t in d['ids']])
            self.assertLess(float(np.abs(steps[:, cols] - ref).max()), 1e-3)
            out, _, _ = generate(m, m.new_cache(), d['ids'], 12,
                                 Sampler(0.0))
            self.assertEqual(out, d['greedy'][:12], w)


if __name__ == '__main__':
    unittest.main()
