# coding: utf-8
'''Chat completions with tool calling (tinygpt/hf/chat_api.py).

The parsing and message conversion are tested directly. The engine is
tested on a scripted stand-in model with a one-character-per-token
tokenizer, so what it sends to the model and what it does with the answer
can be read as text. The round trip through LFM3's real chat template
runs when LFM3-1.7B is cached.
'''

import json
import unittest

import numpy as np

from tests.hf_common import lfm3_dir
from tinygpt.hf.chat_api import (STUB, ChatEngine, ChatError, parse_reply,
                                 template_messages)
from tinygpt.hf.llama import ContextFull

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'read_file', 'description': 'Read a file.',
        'parameters': {'type': 'object', 'properties': {
            'path': {'type': 'string'},
            'max_lines': {'type': 'integer'},
            'numbered': {'type': 'boolean'}}}}},
    {'type': 'function', 'function': {
        'name': 'multi_edit', 'description': 'Edit.',
        'parameters': {'type': 'object', 'properties': {
            'path': {'type': 'string'},
            'edits': {'type': 'array'}}}}},
]


def call(name, **params):
    body = ''.join('<parameter=%s>\n%s\n</parameter>\n' % kv
                   for kv in params.items())
    return '<tool_call>\n<function=%s>\n%s</function>\n</tool_call>' % (
        name, body)


class ParseReplyTest(unittest.TestCase):

    def test_plain_answer(self):
        self.assertEqual(parse_reply('Paris.'), ('', 'Paris.', []))

    def test_thinking_then_call(self):
        text = 'I should look.</think>' + call('read_file', path='a.py')
        self.assertEqual(
            parse_reply(text, TOOLS, thinking_open=True),
            ('I should look.', '', [('read_file', {'path': 'a.py'})]))

    def test_text_around_calls_and_several_calls(self):
        text = ('<think>hm</think>Reading both.\n' + call('read_file', path='a')
                + '\n\n' + call('read_file', path='b'))
        reasoning, content, calls = parse_reply(text, TOOLS)
        self.assertEqual(reasoning, 'hm')
        self.assertEqual(content, 'Reading both.')
        self.assertEqual([c[1]['path'] for c in calls], ['a', 'b'])

    def test_values_follow_the_schema(self):
        text = call('read_file', path='123', max_lines='40',
                    numbered='true')
        self.assertEqual(parse_reply(text, TOOLS)[2], [
            ('read_file', {'path': '123', 'max_lines': 40,
                           'numbered': True})])
        text = call('multi_edit', path='x',
                    edits='[{"old": "a", "new": "b"}]')
        self.assertEqual(parse_reply(text, TOOLS)[2][0][1]['edits'],
                         [{'old': 'a', 'new': 'b'}])
        # A value that is not what the schema says stays text, for the
        # tool to refuse with a message the model can act on.
        text = call('read_file', max_lines='forty')
        self.assertEqual(parse_reply(text, TOOLS)[2][0][1],
                         {'max_lines': 'forty'})

    def test_unknown_tool_guesses_types(self):
        args = parse_reply(call('other', n='3', s='three', b='false',
                                o='{"a": 1}'))[2][0][1]
        self.assertEqual(args, {'n': 3, 's': 'three', 'b': False,
                                'o': {'a': 1}})

    def test_multiline_value_keeps_its_shape(self):
        code = 'def f():\n    return 1\n\n</parameter> is just text\n'
        text = call('read_file', path=code)
        self.assertEqual(parse_reply(text, TOOLS)[2][0][1]['path'], code)

    def test_json_notation(self):
        text = ('<tool_call>\n{"name": "read_file", "arguments": '
                '{"path": "a.py"}}\n</tool_call>')
        self.assertEqual(parse_reply(text, TOOLS)[2],
                         [('read_file', {'path': 'a.py'})])

    def test_cut_off(self):
        # Out of tokens while thinking: nothing was said.
        self.assertEqual(parse_reply('still thin', thinking_open=True),
                         ('still thin', '', []))
        # Out of tokens inside a call: the complete parameters survive.
        text = ('</think><tool_call>\n<function=read_file>\n<parameter=path>'
                '\na.py\n</parameter>\n<parameter=max_l')
        self.assertEqual(parse_reply(text, TOOLS, True)[2],
                         [('read_file', {'path': 'a.py'})])

    def test_stops_at_a_turn_marker(self):
        self.assertEqual(parse_reply('Done.<|im_end|>\n<|im_start|>user\nhi'),
                         ('', 'Done.', []))


class TemplateMessagesTest(unittest.TestCase):

    def test_conversion(self):
        msgs = template_messages([
            {'role': 'system', 'content': 'S'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'look'},
                                         {'type': 'image_url',
                                          'image_url': {'url': 'x'}}]},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'c1', 'type': 'function', 'function': {
                    'name': 'read_file',
                    'arguments': '{"path": "a.py"}'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'name': 'read_file',
             'content': 'A' * 100},
            {'role': 'tool', 'tool_call_id': 'c2', 'content': 'kept'},
        ], reasoning={'c1': 'why'}, stubbed={'c1'})
        self.assertEqual(msgs[1]['content'].split('\n')[0], 'look')
        self.assertIn('cannot see images', msgs[1]['content'])
        a = msgs[2]
        self.assertEqual(a['content'], '')
        self.assertEqual(a['reasoning_content'], 'why')
        # Templates want the arguments as a mapping, not a JSON string.
        self.assertEqual(a['tool_calls'][0]['function']['arguments'],
                         {'path': 'a.py'})
        self.assertEqual(msgs[3]['content'], STUB)
        self.assertEqual(msgs[4]['content'], 'kept')

    def test_bad_arguments_do_not_raise(self):
        m = template_messages([{'role': 'assistant', 'content': '',
                                'tool_calls': [{'id': 'c', 'function': {
                                    'name': 'f', 'arguments': '{oops'}}]}])
        self.assertEqual(m[0]['tool_calls'][0]['function']['arguments'],
                         {'arguments': '{oops'})


# ------------------------------------------------- a scripted stand-in

END = 0x10FFFF          # the "end of turn" token


class CharTok(object):
    '''One token per character.'''

    def encode(self, text, allow_special=True):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special=False):
        return ''.join(chr(i) for i in ids)

    def stream(self, skip_special=True):
        class S(object):
            def feed(self, i):
                return chr(i)

            def flush(self):
                return ''
        return S()


class Format(object):
    '''A chat "template" that is easy to read back.'''

    def render(self, messages, add_generation_prompt=True, tools=None,
               **kw):
        out = ['[tools:%s]' % ','.join(t['function']['name']
                                       for t in tools or [])]
        for m in messages:
            out.append('<%s>%s' % (m['role'], m['content']))
            if m.get('reasoning_content'):
                out.append('{thought:%s}' % m['reasoning_content'])
            for c in m.get('tool_calls') or []:
                out.append('{call:%s %s}' % (
                    c['function']['name'],
                    json.dumps(c['function']['arguments'], sort_keys=True)))
        out.append('<assistant>')
        if kw.get('enable_thinking', True):
            out.append('<think>')
        return ''.join(out)


class Cache(object):
    def __init__(self):
        self.tokens = []

    def truncate(self, n):
        del self.tokens[n:]


class Scripted(object):
    '''A "model" that answers with the next scripted reply.'''

    def __init__(self, replies, ctx_limit=2000):
        self.replies = list(replies)
        self.ctx_limit = ctx_limit
        self.prompts = []       # the text of each prompt it was given
        self.computed = []      # how many prompt tokens were not cached
        self._left = []

    def new_cache(self, capacity=None):
        return Cache()

    def _logits(self):
        z = np.zeros(END + 1, dtype=np.float32)
        z[self._left.pop(0) if self._left else END] = 1.0
        return z

    def feed(self, ids, cache):
        n = 0
        while n < min(len(cache.tokens), len(ids) - 1) \
                and cache.tokens[n] == ids[n]:
            n += 1
        self.computed.append(len(ids) - n)
        cache.tokens[:] = ids
        self.prompts.append(''.join(chr(i) for i in ids))
        self._left = [ord(c) for c in self.replies.pop(0)]
        return self._logits()

    def step(self, tok, cache):
        if len(cache.tokens) >= self.ctx_limit:
            raise ContextFull('full')
        cache.tokens.append(tok)
        return self._logits()


def engine(replies, ctx_limit=2000, **kw):
    model = Scripted(replies, ctx_limit)
    return ChatEngine(model, CharTok(), Format(), {END}, name='stand-in',
                      **kw), model


class ChatEngineTest(unittest.TestCase):

    def test_tool_loop(self):
        eng, model = engine([
            'look first</think>' + call('read_file', path='a.py'),
            'got it</think>The answer is 4.'])
        msgs = [{'role': 'system', 'content': 'S'},
                {'role': 'user', 'content': 'What is in a.py?'}]
        seen = []
        r = eng.chat(msgs, TOOLS, on_text=seen.append,
                     on_prompt=lambda new, cached: seen.append((new, cached)))
        m = r['choices'][0]['message']
        self.assertEqual(r['object'], 'chat.completion')
        self.assertEqual(r['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(m['content'], '')
        self.assertEqual(m['reasoning_content'], 'look first')
        f = m['tool_calls'][0]['function']
        self.assertEqual((f['name'], json.loads(f['arguments'])),
                         ('read_file', {'path': 'a.py'}))
        self.assertEqual(r['usage']['prompt_tokens'], len(model.prompts[0]))
        self.assertEqual(r['usage']['completion_tokens'],
                         len(eng.last['raw']))
        self.assertEqual(seen[0], (len(model.prompts[0]), 0))
        self.assertEqual(''.join(seen[1:]), eng.last['raw'])
        self.assertTrue(model.prompts[0].startswith(
            '[tools:read_file,multi_edit]<system>S<user>'))

        # The agent sends back what it was given, minus the reasoning,
        # plus the tool's result. The thinking behind the call is put
        # back for the template.
        msgs += [{'role': 'assistant', 'content': m['content'],
                  'tool_calls': m['tool_calls']},
                 {'role': 'tool', 'tool_call_id': m['tool_calls'][0]['id'],
                  'name': 'read_file', 'content': 'x = 4'}]
        r = eng.chat(msgs, TOOLS)
        self.assertIn('{thought:look first}{call:read_file {"path": "a.py"}}'
                      '<tool>x = 4<assistant><think>', model.prompts[1])
        self.assertEqual(r['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(r['choices'][0]['message']['content'],
                         'The answer is 4.')
        # Only what came after the shared prefix was computed.
        shared = len('[tools:read_file,multi_edit]<system>S<user>'
                     'What is in a.py?<assistant>')
        self.assertEqual(model.computed[1], len(model.prompts[1]) - shared)
        self.assertEqual(eng.last['cached_tokens'],
                         len(model.prompts[1]) - model.computed[1])

    def test_thinking_off_and_length(self):
        eng, model = engine(['x' * 50], enable_thinking=False)
        r = eng.chat([{'role': 'user', 'content': 'hi'}], max_tokens=10)
        self.assertTrue(model.prompts[0].endswith('<assistant>'))
        self.assertEqual(r['choices'][0]['message']['content'], 'x' * 10)
        self.assertEqual(r['choices'][0]['finish_reason'], 'length')
        self.assertNotIn('reasoning_content', r['choices'][0]['message'])

    def test_cut_off_while_thinking_says_so(self):
        eng, model = engine(['t' * 50])
        r = eng.chat([{'role': 'user', 'content': 'hi'}], max_tokens=10)
        m = r['choices'][0]['message']
        self.assertEqual(m['reasoning_content'], 't' * 10)
        self.assertIn('ran out of room after 10 tokens', m['content'])
        self.assertEqual(r['choices'][0]['finish_reason'], 'length')

    def conversation(self, sizes):
        msgs = [{'role': 'user', 'content': 'go'}]
        for i, n in enumerate(sizes):
            msgs.append({'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'c%d' % i, 'function': {'name': 'read_file',
                                               'arguments': '{}'}}]})
            msgs.append({'role': 'tool', 'tool_call_id': 'c%d' % i,
                         'content': chr(ord('A') + i) * n})
        return msgs

    def test_old_tool_results_are_dropped_to_fit(self):
        eng, model = engine(['</think>ok', '</think>ok'], ctx_limit=1000)
        msgs = self.conversation([300, 300, 300])
        eng.chat(msgs, TOOLS, max_tokens=100)
        p = model.prompts[0]
        self.assertLessEqual(len(p), 900)
        self.assertNotIn('A' * 300, p)          # oldest: gone
        self.assertIn(STUB, p)
        self.assertIn('C' * 300, p)             # newest: whole
        # Dropped results stay dropped, so the prompt keeps its prefix
        # and the cache keeps working on the next step.
        eng.chat(msgs + [{'role': 'user', 'content': 'and?'}], TOOLS,
                 max_tokens=100)
        self.assertTrue(model.prompts[1].startswith(p[:-len('<assistant><think>')]))
        self.assertLess(model.computed[1], 60)

    def test_old_turns_are_dropped_to_fit(self):
        eng, model = engine(['</think>ok', '</think>ok'], ctx_limit=1000)
        msgs = [{'role': 'system', 'content': 'S'}]
        for i in range(6):
            msgs += [{'role': 'user', 'content': 'q%d ' % i + 'x' * 200},
                     {'role': 'assistant', 'content': 'a%d' % i}]
        msgs.append({'role': 'user', 'content': 'last'})
        eng.chat(msgs, TOOLS, max_tokens=100)
        p = model.prompts[0]
        self.assertLessEqual(len(p), 900)
        self.assertTrue(p.startswith('[tools:read_file,multi_edit]<system>S'
                                     '<user>q'))
        self.assertNotIn('q0 ', p)
        self.assertIn('<user>q5 ', p)
        self.assertTrue(p.endswith('<user>last<assistant><think>'))
        # One more exchange is cut at the same place: the prefix holds.
        msgs += [{'role': 'assistant', 'content': 'ok'},
                 {'role': 'user', 'content': 'more'}]
        eng.chat(msgs, TOOLS, max_tokens=100)
        self.assertTrue(model.prompts[1].startswith(
            p[:-len('<assistant><think>')]))

    def test_newest_result_is_cut_to_fit(self):
        eng, model = engine(['</think>ok'], ctx_limit=1000)
        eng.chat(self.conversation([5000]), TOOLS, max_tokens=100)
        p = model.prompts[0]
        self.assertLessEqual(len(p), 900)
        self.assertIn('A' * 500, p)
        self.assertIn('cut to fit', p)

    def test_too_long_is_an_error(self):
        eng, model = engine(['</think>ok'], ctx_limit=1000)
        with self.assertRaises(ChatError) as cm:
            eng.chat([{'role': 'user', 'content': 'x' * 3000}], TOOLS)
        self.assertIn('context', str(cm.exception))

    def test_reset(self):
        eng, model = engine(['</think>a', '</think>b'])
        msgs = [{'role': 'user', 'content': 'hi'}]
        eng.chat(msgs)
        eng.reset()
        eng.chat(msgs)
        self.assertEqual(model.computed[1], len(model.prompts[1]))


@unittest.skipUnless(lfm3_dir(), 'LFM3-1.7B is not cached')
class Lfm3TemplateTest(unittest.TestCase):
    '''What the template writes for a tool call is what parse_reply
    reads, so calls survive the trip through the history.'''

    def test_round_trip(self):
        from tinygpt.hf.tokenizer import BPETokenizer, ChatFormat
        tok = BPETokenizer.from_dir(lfm3_dir())
        chat = ChatFormat(tok.chat_template, tok.special_tokens)
        args = {'path': 'a b.py', 'max_lines': 40, 'numbered': True}
        msgs = template_messages([
            {'role': 'system', 'content': 'Be brief.'},
            {'role': 'user', 'content': 'Read it.'},
            {'role': 'assistant', 'content': 'Sure.', 'tool_calls': [
                {'id': 'c1', 'type': 'function', 'function': {
                    'name': 'read_file',
                    'arguments': json.dumps(args)}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'x = 4'},
        ], reasoning={'c1': 'I need the file.'})
        text = chat.render(msgs, tools=TOOLS)
        self.assertIn('<tools>\n{"name": "read_file"', text)
        self.assertIn('<tool_response>\nx = 4\n</tool_response>', text)
        self.assertTrue(text.endswith('<|im_start|>assistant\n<think>'))
        said = text.split('<|im_start|>assistant\n')[1].split('<|im_end|>')[0]
        self.assertEqual(parse_reply(said, TOOLS), (
            'I need the file.', 'Sure.', [('read_file', args)]))
        off = chat.render(msgs, tools=TOOLS, enable_thinking=False)
        self.assertTrue(off.endswith('<think></think>'))
        # Every marker the parser looks for is a single token.
        for marker in ('<think>', '</think>', '<tool_call>', '</tool_call>'):
            self.assertEqual(len(tok.encode(marker)), 1, marker)


if __name__ == '__main__':
    unittest.main()
