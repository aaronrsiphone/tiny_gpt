# coding: utf-8
'''Byte-level BPE: exact ids against the reference `tokenizers` library.

bpe_tiny.json is a small tokenizer trained with SmolLM2's pipeline and is
always available. The SmolLM2 tests run when the model is cached
(python hf_run.py download, or import of the zip).
'''

import os
import unittest

from tests.hf_common import fixture, smollm2_dir
from tinygpt.hf.tokenizer import (BPETokenizer, ChatFormat,
                                  UnsupportedTokenizer, bytes_to_unicode)


class ByteMapTest(unittest.TestCase):

    def test_bijective(self):
        m = bytes_to_unicode()
        self.assertEqual(len(m), 256)
        self.assertEqual(len(set(m.values())), 256)
        self.assertEqual(m[ord(' ')], 'Ġ')      # the familiar 'Ġ'
        self.assertEqual(m[ord('A')], 'A')


def check_against(test, tok, texts, ids, decoded=None):
    bad = []
    for i, t in enumerate(texts):
        got = tok.encode(t)
        if got != ids[i]:
            bad.append((t, ids[i][:12], got[:12]))
        if decoded is not None and tok.decode(got) != decoded[i]:
            bad.append(('decode', t, decoded[i]))
    test.assertEqual(bad[:3], [], '%d of %d differ' % (len(bad), len(texts)))


class TinyBPETest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.d = fixture('bpe_tiny.json')
        cls.tok = BPETokenizer(cls.d['tokenizer'])

    def test_encode_decode_exact(self):
        check_against(self, self.tok, self.d['texts'], self.d['ids'],
                      self.d['decoded'])

    def test_stream_matches_decode(self):
        for t in self.d['texts'][:80]:
            ids = self.tok.encode(t)
            st = self.tok.stream(skip_special=False)
            got = ''.join(st.feed(i) for i in ids) + st.flush()
            self.assertEqual(got, self.tok.decode(ids))

    def test_special_tokens_optional(self):
        t = '<|im_end|>'
        self.assertEqual(len(self.tok.encode(t)), 1)
        self.assertGreater(len(self.tok.encode(t, allow_special=False)), 1)

    def test_unsupported_configs_rejected(self):
        import copy
        for mutate in (
                lambda s: s.__setitem__('normalizer', {'type': 'NFC'}),
                lambda s: s['model'].__setitem__('byte_fallback', True),
                lambda s: s['model'].__setitem__('dropout', 0.1),
                lambda s: s.__setitem__('pre_tokenizer',
                                        {'type': 'Whitespace'}),
                lambda s: s['model'].__setitem__('type', 'WordPiece')):
            spec = copy.deepcopy(self.d['tokenizer'])
            mutate(spec)
            with self.assertRaises(UnsupportedTokenizer):
                BPETokenizer(spec)


@unittest.skipUnless(smollm2_dir(), 'SmolLM2 not cached')
class SmolLM2TokenizerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.d = fixture('smollm2_tokens.json')
        cls.tok = BPETokenizer.from_dir(smollm2_dir())

    def test_texts(self):
        check_against(self, self.tok, self.d['texts'], self.d['ids'])

    def test_readme(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8')
        ids = self.tok.encode(text)
        if ids != self.d['readme_ids']:
            self.skipTest('README.md changed since the fixture was made')
        self.assertEqual(self.tok.decode(ids), text)

    def test_special_ids(self):
        self.assertEqual(self.tok.encode('<|im_start|>'), [1])
        self.assertEqual(self.tok.encode('<|im_end|>'), [2])
        self.assertEqual(self.tok.eos_id, 2)

    def test_chat_template(self):
        cf = ChatFormat(self.tok.chat_template)
        for msgs, text, ids in zip(self.d['chats'], self.d['chat_text'],
                                   self.d['chat_ids']):
            self.assertEqual(cf.render(msgs), text)
            self.assertEqual(self.tok.encode(cf.render(msgs)), ids)


if __name__ == '__main__':
    unittest.main()
