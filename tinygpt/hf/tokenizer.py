# coding: utf-8
'''
Byte-level BPE, read from a Hugging Face tokenizer.json, in pure Python.

This reproduces what the Rust `tokenizers` library does for GPT-2-style
byte-level BPE models (SmolLM2, GPT-2, ...), step for step, so the ids are
the ones the model was trained on rather than an approximation of them:

  1. Added tokens (<|im_start|> and friends) are cut out of the text
     first, leftmost-longest, and map straight to their ids.
  2. The remaining text is pre-tokenized: optionally every numeric char is
     isolated (the Digits step), then the GPT-2 regex splits it into
     words, and each byte of a word is mapped to a printable stand-in char.
  3. Each word is merged by the BPE merge ranks with the same priority
     queue the Rust code uses, so even oddly ordered merge tables agree.

Decoding maps the stand-in chars back to bytes and decodes UTF-8 with
replacement, as the ByteLevel decoder does.

The regex classes \\p{L}, \\p{N}, \\s are not available in Python's re
module, so they come from unicode_tables.py, which was measured from the
reference library itself (see tools/gen_unicode_tables.py).

Anything in tokenizer.json this module does not implement exactly -- a
normalizer, a different pre-tokenizer, byte fallback, BPE dropout --
raises at load time instead of silently tokenizing differently.
'''

import codecs
import heapq
import json
import os
import re

from tinygpt.hf import unicode_tables as _ut


class UnsupportedTokenizer(ValueError):
    pass


# ------------------------------------------------------------ byte mapping

def bytes_to_unicode():
    '''GPT-2's reversible byte -> printable char table.'''
    bs = (list(range(ord('!'), ord('~') + 1))
          + list(range(ord('\xa1'), ord('\xac') + 1))
          + list(range(ord('\xae'), ord('\xff') + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {c: b for b, c in BYTE_ENCODER.items()}


def _cls(rs):
    parts = []
    for a, b in rs:
        if a == b:
            parts.append('\\U%08X' % a)
        else:
            parts.append('\\U%08X-\\U%08X' % (a, b))
    return ''.join(parts)


_L = _cls(_ut.LETTER)
_N = _cls(_ut.NUMBER)
_W = _cls(_ut.WHITE_SPACE)

# The GPT-2 pattern, with \p{L}, \p{N}, \s spelled out as explicit classes.
GPT2_PATTERN = (
    "'s|'t|'re|'ve|'m|'ll|'d"
    "| ?[%(L)s]+| ?[%(N)s]+| ?[^%(W)s%(L)s%(N)s]+"
    "|[%(W)s]+(?![^%(W)s])|[%(W)s]+" % {'L': _L, 'N': _N, 'W': _W})

_numeric_re = None
_gpt2_re = None


def _regexes():
    global _numeric_re, _gpt2_re
    if _gpt2_re is None:
        _gpt2_re = re.compile(GPT2_PATTERN)
        _numeric_re = re.compile('([%s])' % _cls(_ut.NUMERIC))
    return _gpt2_re, _numeric_re


# ----------------------------------------------------------------- config

def _check_pre_tokenizer(pt):
    '''Return (isolate_digits, add_prefix_space) or raise.'''
    if pt is None:
        raise UnsupportedTokenizer('tokenizer has no pre_tokenizer; only '
                                   'byte-level BPE is supported')
    steps = pt['pretokenizers'] if pt.get('type') == 'Sequence' else [pt]
    digits = False
    bytelevel = None
    for i, st in enumerate(steps):
        t = st.get('type')
        if t == 'Digits' and i == 0 and len(steps) > 1:
            if not st.get('individual_digits', False):
                raise UnsupportedTokenizer(
                    'Digits pre-tokenizer without individual_digits')
            digits = True
        elif t == 'ByteLevel' and i == len(steps) - 1:
            if not st.get('use_regex', True):
                raise UnsupportedTokenizer('ByteLevel with use_regex=false')
            bytelevel = st
        else:
            raise UnsupportedTokenizer(
                'unsupported pre-tokenizer step %r (supported: optional '
                'Digits, then ByteLevel)' % t)
    if bytelevel is None:
        raise UnsupportedTokenizer('pre-tokenizer does not end in ByteLevel')
    return digits, bool(bytelevel.get('add_prefix_space', False))


def _check_model(m):
    if m.get('type') != 'BPE':
        raise UnsupportedTokenizer('tokenizer model %r, only BPE is '
                                   'supported' % m.get('type'))
    if m.get('dropout') not in (None, 0, 0.0):
        raise UnsupportedTokenizer('BPE dropout is not supported')
    if m.get('byte_fallback'):
        raise UnsupportedTokenizer('BPE byte_fallback is not supported')
    if m.get('continuing_subword_prefix') or m.get('end_of_word_suffix'):
        raise UnsupportedTokenizer('BPE subword prefixes/suffixes are not '
                                   'supported')
    if m.get('ignore_merges'):
        raise UnsupportedTokenizer('BPE ignore_merges is not supported')


# -------------------------------------------------------------- tokenizer

class BPETokenizer(object):
    '''Byte-level BPE tokenizer loaded from a tokenizer.json dict.

    Follows the same small protocol as tinygpt/tokenizer.py where it makes
    sense (token_to_id, id_to_token, vocab_size, encode, decode).
    '''

    name = 'hf-bpe'
    strict = True

    def __init__(self, spec, config=None):
        config = config or {}
        if spec.get('normalizer') is not None:
            raise UnsupportedTokenizer('tokenizer normalizers are not '
                                       'supported')
        pp = spec.get('post_processor')
        if pp is not None and pp.get('type') != 'ByteLevel':
            raise UnsupportedTokenizer(
                'post-processor %r is not supported' % pp.get('type'))
        dec = spec.get('decoder')
        if dec is not None and dec.get('type') != 'ByteLevel':
            raise UnsupportedTokenizer('decoder %r is not supported'
                                       % dec.get('type'))
        m = spec['model']
        _check_model(m)
        self.isolate_digits, self.add_prefix_space = \
            _check_pre_tokenizer(spec.get('pre_tokenizer'))

        self.token_to_id = {t: int(i) for t, i in m['vocab'].items()}
        self.added = {}
        self.special_ids = set()
        for a in spec.get('added_tokens', ()):
            if a.get('lstrip') or a.get('rstrip') or a.get('single_word'):
                raise UnsupportedTokenizer(
                    'added token %r uses lstrip/rstrip/single_word'
                    % a['content'])
            self.added[a['content']] = int(a['id'])
            self.token_to_id[a['content']] = int(a['id'])
            if a.get('special'):
                self.special_ids.add(int(a['id']))
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = max(self.id_to_token) + 1

        self.merges = {}
        for rank, mg in enumerate(m['merges']):
            a, b = mg.split(' ') if isinstance(mg, str) else mg
            try:
                ia, ib = self.token_to_id[a], self.token_to_id[b]
                new = self.token_to_id[a + b]
            except KeyError:
                raise UnsupportedTokenizer('merge %r names a token outside '
                                           'the vocabulary' % (mg,))
            self.merges.setdefault((ia, ib), (rank, new))

        if self.added:
            alts = sorted(self.added, key=len, reverse=True)
            self._added_re = re.compile('|'.join(re.escape(a)
                                                 for a in alts))
        else:
            self._added_re = None
        self._cache = {}

        def text(v):
            return v.get('content') if isinstance(v, dict) else v

        # The special tokens by role, as strings: chat templates refer to
        # them as variables ({{ bos_token }}), just as transformers passes
        # its special_tokens_map into the template.
        self.special_tokens = {}
        for key in ('bos_token', 'eos_token', 'unk_token', 'pad_token'):
            if text(config.get(key)):
                self.special_tokens[key] = text(config[key])
        extra = config.get('additional_special_tokens')
        if extra:
            self.special_tokens['additional_special_tokens'] = [
                text(t) for t in extra]
        self.bos_id = self.token_to_id.get(self.special_tokens.get(
            'bos_token'))
        self.eos_id = self.token_to_id.get(self.special_tokens.get(
            'eos_token'))
        tpl = config.get('chat_template')
        if isinstance(tpl, list):
            # Several named templates (e.g. 'default', 'tool_use').
            named = dict((t.get('name'), t.get('template')) for t in tpl)
            tpl = named.get('default')
        self.chat_template = tpl

    @classmethod
    def from_dir(cls, path):
        with open(os.path.join(path, 'tokenizer.json'), 'rb') as f:
            spec = json.loads(f.read().decode('utf-8'))
        cfg = {}
        p = os.path.join(path, 'tokenizer_config.json')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                cfg = json.loads(f.read().decode('utf-8'))
        # Newer repos keep the template in its own file, which then wins
        # over any copy in tokenizer_config.json (as in transformers).
        p = os.path.join(path, 'chat_template.jinja')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                cfg['chat_template'] = f.read().decode('utf-8')
        return cls(spec, cfg)

    # -- encoding ---------------------------------------------------------

    def _words(self, text):
        gpt2, numeric = _regexes()
        if self.add_prefix_space and text and not text[0].isspace():
            text = ' ' + text
        pieces = numeric.split(text) if self.isolate_digits else [text]
        for piece in pieces:
            if piece:
                for w in gpt2.findall(piece):
                    yield w

    def _bpe(self, word):
        ids = self._cache.get(word)
        if ids is not None:
            return ids
        sym = [self.token_to_id.get(BYTE_ENCODER[b])
               for b in word.encode('utf-8')]
        sym = [s for s in sym if s is not None]
        n = len(sym)
        if n == 0:
            return []
        nxt = list(range(1, n + 1))
        prv = list(range(-1, n - 1))
        nxt[-1] = -1
        alive = [True] * n
        merges = self.merges
        heap = []
        for i in range(n - 1):
            m = merges.get((sym[i], sym[i + 1]))
            if m is not None:
                heap.append((m[0], i, m[1]))
        heapq.heapify(heap)
        # Same loop as tokenizers' Word::merge_all: lowest rank first, then
        # leftmost; entries made stale by an earlier merge are skipped.
        while heap:
            rank, i, new = heapq.heappop(heap)
            if not alive[i] or nxt[i] == -1:
                continue
            j = nxt[i]
            m = merges.get((sym[i], sym[j]))
            if m is None or m[0] != rank or m[1] != new:
                continue
            sym[i] = new
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[j] != -1:
                prv[nxt[j]] = i
            p = prv[i]
            if p != -1:
                m = merges.get((sym[p], new))
                if m is not None:
                    heapq.heappush(heap, (m[0], p, m[1]))
            q = nxt[i]
            if q != -1:
                m = merges.get((new, sym[q]))
                if m is not None:
                    heapq.heappush(heap, (m[0], i, m[1]))
        ids = [sym[i] for i in range(n) if alive[i]]
        if len(self._cache) > 50000:
            self._cache.clear()
        self._cache[word] = ids
        return ids

    def _encode_plain(self, text, out):
        for w in self._words(text):
            out.extend(self._bpe(w))

    def encode(self, text, allow_special=True):
        '''Text -> ids. With allow_special, added-token strings in the text
        become their ids, exactly as `tokenizers` does by default.'''
        out = []
        if not allow_special or self._added_re is None:
            self._encode_plain(text, out)
            return out
        pos = 0
        for m in self._added_re.finditer(text):
            if m.start() > pos:
                self._encode_plain(text[pos:m.start()], out)
            out.append(self.added[m.group(0)])
            pos = m.end()
        if pos < len(text):
            self._encode_plain(text[pos:], out)
        return out

    # -- decoding ---------------------------------------------------------

    def token_bytes(self, i):
        t = self.id_to_token[i]
        try:
            return bytes(BYTE_DECODER[c] for c in t)
        except KeyError:
            # The ByteLevel decoder's rule: a token that is not entirely
            # stand-in chars (an added token with a raw space, say) is
            # passed through as its own UTF-8.
            return t.encode('utf-8')

    def decode_bytes(self, ids, skip_special=False):
        return b''.join(self.token_bytes(i) for i in ids
                        if not (skip_special and i in self.special_ids))

    def decode(self, ids, skip_special=False):
        return self.decode_bytes(ids, skip_special).decode('utf-8',
                                                           'replace')

    def stream(self, skip_special=True):
        '''An incremental decoder: feed(id) -> the text it completes.

        A multi-byte character can be split across tokens; the incremental
        UTF-8 decoder holds the partial bytes back until they complete.
        '''
        return _Stream(self, skip_special)


class _Stream(object):

    def __init__(self, tok, skip_special):
        self.tok = tok
        self.skip = skip_special
        self.dec = codecs.getincrementaldecoder('utf-8')('replace')

    def feed(self, i):
        if self.skip and i in self.tok.special_ids:
            return ''
        return self.dec.decode(self.tok.token_bytes(i))

    def flush(self):
        return self.dec.decode(b'', final=True)


# ----------------------------------------------------------- chat format

def _compile_template(template, now):
    '''Compile a chat template exactly as transformers does.

    Same sandbox (ImmutableSandboxedEnvironment: templates cannot reach
    files, modules or object internals), same whitespace options, same
    extras: loopcontrols, the {% generation %} tag, a tojson that does
    not HTML-escape, raise_exception() and strftime_now().
    '''
    import jinja2
    from jinja2 import nodes
    from jinja2.ext import Extension, loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class GenerationTag(Extension):
        # transformers uses {% generation %} to find assistant text for
        # training masks. For rendering it is transparent.
        tags = set(['generation'])

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(['name:endgeneration'],
                                           drop_needle=True)
            return nodes.CallBlock(self.call_method('_body'), [], [],
                                   body).set_lineno(lineno)

        def _body(self, caller):
            return caller()

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=[GenerationTag, loopcontrols])
    env.filters['tojson'] = tojson
    env.globals['raise_exception'] = raise_exception
    env.globals['strftime_now'] = lambda fmt: now().strftime(fmt)
    return env.from_string(template), jinja2.exceptions.TemplateError


# Without jinja2, only these exact templates are rendered, by hand. {SYSTEM}
# stands for a default system prompt. Anything else is refused rather
# than approximated.
_KNOWN_CHATML = (
    # SmolLM2
    "{% for message in messages %}{% if loop.first and messages[0]['role']"
    " != 'system' %}{{ '<|im_start|>system\n{SYSTEM}<|im_end|>\n' }}"
    "{% endif %}{{'<|im_start|>' + message['role'] + '\n' + "
    "message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{% endif %}",
    # plain ChatML, no default system prompt
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + "
    "'\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{% endif %}",
)


def _match_known(template):
    '''(True, default system prompt or None) if template is one of
    _KNOWN_CHATML, else (False, None).'''
    for known in _KNOWN_CHATML:
        if '{SYSTEM}' not in known:
            if template == known:
                return True, None
            continue
        head, tail = known.split('{SYSTEM}')
        if (template.startswith(head) and template.endswith(tail)
                and len(template) >= len(head) + len(tail)):
            system = template[len(head):len(template) - len(tail)]
            if "'" not in system and '{' not in system:
                return True, system
    return False, None


class ChatFormat(object):
    '''Turns a list of {'role', 'content'} messages into prompt text
    using the model's own chat template.

    With jinja2 (Pythonista ships it) the template itself is rendered,
    in the same sandbox and with the same variables as transformers'
    apply_chat_template: messages, add_generation_prompt, tools=None,
    documents=None, the special tokens by name, plus any extra keyword
    arguments the template reads (e.g. enable_thinking=False).

    Without jinja2, a known plain ChatML template (SmolLM2's) is rendered
    by hand, and any other template is refused.

    The rendered string is tokenized like any other text, which is what
    apply_chat_template(tokenize=True) does too.
    '''

    def __init__(self, template, special_tokens=None, use_jinja=None,
                 now=None):
        if not template:
            raise UnsupportedTokenizer('model has no chat template; use '
                                       'completion mode')
        import datetime
        self.template = template
        self.special_tokens = dict(special_tokens or {})
        self.now = now or datetime.datetime.now
        self._jinja = None
        if use_jinja is not False:
            try:
                self._jinja, self._TemplateError = _compile_template(
                    template, self.now)
            except ImportError:
                if use_jinja:
                    raise
        if self._jinja is not None:
            self.engine = 'jinja2'
            return
        ok, self.default_system = _match_known(template)
        if not ok:
            raise UnsupportedTokenizer(
                'rendering this chat template needs jinja2 (pip install '
                'jinja2); use completion mode')
        self.engine = 'chatml'

    def render(self, messages, add_generation_prompt=True, **kwargs):
        if self._jinja is not None:
            ctx = dict(self.special_tokens)
            ctx.update(kwargs)
            try:
                return self._jinja.render(
                    messages=messages, tools=None, documents=None,
                    add_generation_prompt=add_generation_prompt, **ctx)
            except self._TemplateError as e:
                raise ValueError('chat template: %s' % e)
        if kwargs:
            raise ValueError('template options %s need jinja2'
                             % ', '.join(sorted(kwargs)))
        out = []
        if (self.default_system is not None and messages
                and messages[0]['role'] != 'system'):
            out.append('<|im_start|>system\n%s<|im_end|>\n'
                       % self.default_system)
        for msg in messages:
            out.append('<|im_start|>%s\n%s<|im_end|>\n'
                       % (msg['role'], msg['content']))
        if add_generation_prompt:
            out.append('<|im_start|>assistant\n')
        return ''.join(out)
