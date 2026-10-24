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
        # A long real document (a snapshot of README.md stored in the
        # fixture, so later README edits cannot make this test skip).
        text = self.d['readme_text']
        ids = self.tok.encode(text)
        self.assertEqual(ids, self.d['readme_ids'])
        self.assertEqual(self.tok.decode(ids), text)

    def test_special_ids(self):
        self.assertEqual(self.tok.encode('<|im_start|>'), [1])
        self.assertEqual(self.tok.encode('<|im_end|>'), [2])
        self.assertEqual(self.tok.eos_id, 2)

    def test_chat_template(self):
        cf = ChatFormat(self.tok.chat_template, self.tok.special_tokens)
        for msgs, text, ids in zip(self.d['chats'], self.d['chat_text'],
                                   self.d['chat_ids']):
            self.assertEqual(cf.render(msgs), text)
            self.assertEqual(self.tok.encode(cf.render(msgs)), ids)


try:
    import jinja2  # noqa: F401
    HAVE_JINJA = True
except ImportError:
    HAVE_JINJA = False


class ChatTemplateTest(unittest.TestCase):
    '''Real templates from several repos against transformers' renderer.'''

    @classmethod
    def setUpClass(cls):
        import datetime
        cls.d = fixture('chat_templates.json')
        when = datetime.datetime(*cls.d['now'])
        cls.now = staticmethod(lambda: when)

    @unittest.skipUnless(HAVE_JINJA, 'jinja2 not installed')
    def test_jinja_matches_transformers(self):
        n = 0
        for repo, r in sorted(self.d['repos'].items()):
            cf = ChatFormat(r['template'], r['special_tokens'], now=self.now)
            self.assertEqual(cf.engine, 'jinja2')
            for c in r['cases']:
                got = cf.render(c['messages'], c['add_generation_prompt'],
                                **c['kwargs'])
                self.assertEqual(got, c['text'], '%s %r' % (repo, c))
                n += 1
        self.assertGreaterEqual(n, 40)

    def test_fallback_only_for_known_templates(self):
        for repo, r in sorted(self.d['repos'].items()):
            if repo == 'HuggingFaceTB/SmolLM2-135M-Instruct':
                cf = ChatFormat(r['template'], r['special_tokens'],
                                use_jinja=False)
                self.assertEqual(cf.engine, 'chatml')
                for c in r['cases']:
                    if c['kwargs']:
                        with self.assertRaises(ValueError):
                            cf.render(c['messages'], **c['kwargs'])
                        continue
                    self.assertEqual(
                        cf.render(c['messages'], c['add_generation_prompt']),
                        c['text'])
            else:
                with self.assertRaises(UnsupportedTokenizer):
                    ChatFormat(r['template'], use_jinja=False)

    @unittest.skipUnless(HAVE_JINJA, 'jinja2 not installed')
    def test_sandbox_and_errors(self):
        msgs = [{'role': 'user', 'content': 'x'}]
        evil = ChatFormat("{{ ''.__class__.__mro__[1].__subclasses__() }}")
        with self.assertRaises(ValueError):
            evil.render(msgs)
        strict = ChatFormat("{% if messages[0]['role'] != 'system' %}"
                            "{{ raise_exception('system first') }}{% endif %}")
        with self.assertRaises(ValueError) as cm:
            strict.render(msgs)
        self.assertIn('system first', str(cm.exception))

    def test_no_template(self):
        with self.assertRaises(UnsupportedTokenizer):
            ChatFormat(None)

    def test_template_file_and_named_list(self):
        import json
        import shutil
        import tempfile
        spec = fixture('bpe_tiny.json')['tokenizer']
        td = tempfile.mkdtemp()
        try:
            with open(os.path.join(td, 'tokenizer.json'), 'w') as f:
                json.dump(spec, f)
            with open(os.path.join(td, 'tokenizer_config.json'), 'w') as f:
                json.dump({'chat_template': [
                    {'name': 'tool_use', 'template': 'T'},
                    {'name': 'default', 'template': 'D'}],
                    'eos_token': {'content': '<|im_end|>'}}, f)
            tok = BPETokenizer.from_dir(td)
            self.assertEqual(tok.chat_template, 'D')
            self.assertEqual(tok.special_tokens, {'eos_token': '<|im_end|>'})
            with open(os.path.join(td, 'chat_template.jinja'), 'w') as f:
                f.write('J')
            self.assertEqual(BPETokenizer.from_dir(td).chat_template, 'J')
        finally:
            shutil.rmtree(td)


if __name__ == '__main__':
    unittest.main()
