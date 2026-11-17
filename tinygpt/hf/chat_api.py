# coding: utf-8
'''
Chat completions with tool calling, in the OpenAI wire shape, on a local
model. This is what lets an agent harness written for a hosted API (Jeb's
miniagent) be driven by a model running in this process:

    engine = hf_run.open_chat_engine('LiquidAI/LFM3-1.7B', ctx=4096)
    reply = engine.chat(messages, tools=schemas)
    reply['choices'][0]['message']   # content, reasoning_content, tool_calls

`messages` and `tools` are what an OpenAI-compatible client would send;
the reply is what such a server would answer. In between:

- the messages are rendered with the model's own chat template, which is
  also what describes the tools to the model;
- the model's tool calls are parsed back out of its text. Two notations
  are understood: LFM3's
      <tool_call>
      <function=NAME>
      <parameter=ARG>
      value
      </parameter>
      </function>
      </tool_call>
  and the JSON one Qwen uses, <tool_call>{"name": ..., "arguments": ...}
  </tool_call>;
- <think>...</think> comes back as `reasoning_content`, and is given back
  to the template for the calls of the current tool loop, as LFM3 expects;
- the KV cache is kept between calls, so each step of an agent loop only
  computes the tokens added since the previous one (the system prompt and
  tool list are read once);
- a conversation that outgrows the context loses its oldest tool results
  (replaced by a one-line stub), then its oldest turns, rather than
  failing.
'''

import json
import os
import re
import time

from .llama import Sampler, generate

__all__ = ['ChatEngine', 'ChatError', 'parse_reply', 'template_messages']


class ChatError(Exception):
    '''A request that cannot be answered; the message says why.'''


STUB = '[earlier tool output removed to fit the local model\'s context]'
CUT = '\n[... cut to fit the local model\'s context]'
# Room kept free for the reply when deciding whether a prompt fits.
RESERVE = 512
NO_IMAGE = '[image omitted: this model cannot see images]'

_TOOL_CALL = re.compile(r'<tool_call>(.*?)(?:</tool_call>|\Z)', re.S)
_FUNCTION = re.compile(r'<function=([^>\n]+)>(.*?)(?:</function>|\Z)', re.S)
# A value runs to the </parameter> that is followed by the next parameter
# or the end of the call, so a value may itself contain "</parameter>".
_PARAM = re.compile(
    r'<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>'
    r'(?=\s*(?:<parameter=|</function>|</tool_call>|\Z))', re.S)
_SPECIAL = re.compile(r'<\|im_(?:start|end)\|>|<\|endoftext\|>')


# ------------------------------------------------------------ requests

def _text(content):
    '''Message content as plain text (a string, or OpenAI content parts).'''
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get('text'), str):
                    if part.get('type') in (None, 'text'):
                        out.append(part['text'])
                elif part.get('type') in ('image_url', 'image'):
                    out.append(NO_IMAGE)
        return '\n'.join(p for p in out if p)
    return json.dumps(content, ensure_ascii=False)


def _arguments(raw):
    '''Tool-call arguments as a mapping. Clients send a JSON string;
    templates want the mapping.'''
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except ValueError:
            return {'arguments': raw}
        if isinstance(value, dict):
            return value
    return {}


def template_messages(messages, reasoning=None, stubbed=()):
    '''OpenAI-shaped messages as the chat template wants them.

    `reasoning` maps a tool-call id to the thinking that preceded that
    call; `stubbed` holds the tool-call ids whose results were dropped.
    '''
    reasoning = reasoning or {}
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role')
        if role == 'tool':
            cid = m.get('tool_call_id') or ''
            text = STUB if cid and cid in stubbed else _text(m.get('content'))
            out.append({'role': 'tool', 'content': text,
                        'tool_call_id': cid})
        elif role == 'assistant':
            msg = {'role': 'assistant', 'content': _text(m.get('content'))}
            calls = []
            for c in m.get('tool_calls') or []:
                f = c.get('function') or {}
                calls.append({'type': 'function', 'id': c.get('id', ''),
                              'function': {
                                  'name': str(f.get('name') or ''),
                                  'arguments': _arguments(
                                      f.get('arguments'))}})
            if calls:
                msg['tool_calls'] = calls
            thought = m.get('reasoning_content') or m.get('reasoning')
            if not isinstance(thought, str) and calls:
                thought = reasoning.get(calls[0]['id'])
            if isinstance(thought, str) and thought:
                msg['reasoning_content'] = thought
            out.append(msg)
        elif role in ('system', 'user', 'developer'):
            out.append({'role': 'system' if role == 'developer' else role,
                        'content': _text(m.get('content'))})
    return out


# ------------------------------------------------------------- replies

def _schema_types(tools):
    '''{tool name: {parameter name: JSON-schema type}}.'''
    out = {}
    for t in tools or []:
        f = t.get('function', t) if isinstance(t, dict) else {}
        props = (f.get('parameters') or {}).get('properties') or {}
        out[f.get('name')] = dict(
            (k, v.get('type') if isinstance(v, dict) else None)
            for k, v in props.items())
    return out


def _value(text, kind):
    '''A parameter's text as a JSON value. Strings stay as written; other
    declared types are parsed, and a value that does not parse is passed
    on as text for the tool to complain about.'''
    if kind == 'string':
        return text
    if isinstance(kind, list) and 'string' in kind:
        return text
    stripped = text.strip()
    if kind is None and not re.match(
            r'^(-?\d+(\.\d+)?|true|false|null|[\[{].*[\]}])$', stripped,
            re.S):
        return text
    try:
        return json.loads(stripped)
    except ValueError:
        low = stripped.lower()
        if low in ('true', 'false'):
            return low == 'true'
        return text


def parse_reply(text, tools=None, thinking_open=False):
    '''Split generated text into (reasoning, content, calls).

    calls is a list of (name, arguments dict). `thinking_open` says the
    prompt ended inside <think>, so the text starts mid-thought.
    '''
    m = _SPECIAL.search(text)
    if m:
        text = text[:m.start()]
    reasoning = ''
    if '</think>' in text:
        reasoning, text = text.split('</think>', 1)
        reasoning = reasoning.replace('<think>', '', 1)
    elif thinking_open or text.lstrip().startswith('<think>'):
        # Ran out of tokens while still thinking.
        reasoning, text = text.replace('<think>', '', 1), ''
    types = _schema_types(tools)
    calls = []
    for block in _TOOL_CALL.finditer(text):
        body = block.group(1).strip()
        if body.startswith('{'):
            try:
                obj = json.loads(body)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get('name'):
                calls.append((str(obj['name']), _arguments(
                    obj.get('arguments', obj.get('parameters')))))
            continue
        for fn in _FUNCTION.finditer(body):
            name = fn.group(1).strip()
            kinds = types.get(name, {})
            args = {}
            for p in _PARAM.finditer(fn.group(2)):
                key = p.group(1).strip()
                args[key] = _value(p.group(2), kinds.get(key))
            calls.append((name, args))
    content = _TOOL_CALL.sub('', text).strip()
    return reasoning.strip(), content, calls


# -------------------------------------------------------------- engine

class ChatEngine(object):
    '''One loaded model answering chat-completion requests.

    sampling: {'temperature', 'top_k', 'top_p', 'repetition_penalty'},
    the defaults for requests that do not say otherwise.
    '''

    def __init__(self, model, tok, chat, stops, sampling=None, name='',
                 max_tokens=1024, enable_thinking=None, seed=0):
        self.model = model
        self.tok = tok
        self.chat_format = chat
        self.stops = set(stops)
        self.sampling = dict(sampling or {})
        self.name = name
        self.max_tokens = int(max_tokens)
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.cache = model.new_cache()
        self.last = {}              # timings of the latest request
        self._calls = 0
        self._reasoning = {}        # tool-call id -> thinking before it
        self._stubbed = set()       # tool-call ids whose output was dropped

    @property
    def context(self):
        return self.model.ctx_limit

    def reset(self):
        '''Forget the cached conversation (the model stays loaded).'''
        self.cache = self.model.new_cache()
        self._reasoning.clear()
        self._stubbed.clear()

    # -- prompt ------------------------------------

    def _encode(self, msgs, tools, options):
        try:
            text = self.chat_format.render(msgs, tools=tools, **options)
        except ValueError as e:
            raise ChatError(str(e))
        return text, self.tok.encode(text)

    def _render(self, messages, tools, options):
        msgs = template_messages(messages, self._reasoning, self._stubbed)
        return (msgs,) + self._encode(msgs, tools, options)

    def _prompt(self, messages, tools, options, max_new):
        '''(prompt text, ids), shrinking the conversation if it has to.'''
        room = self.context - min(max_new, RESERVE)
        msgs, text, ids = self._render(messages, tools, options)
        if len(ids) <= room:
            return text, ids
        # 1. Drop old tool results, oldest first, down to 3/4 of the room
        #    so this does not happen again on the very next step. Results
        #    of the current step (after the last assistant message) stay.
        last_call = max([i for i, m in enumerate(msgs)
                         if m['role'] == 'assistant'] or [0])
        size = len(ids)
        stub_cost = len(self.tok.encode(STUB))
        for m in msgs[:last_call]:
            if size <= room * 3 // 4:
                break
            cid = m.get('tool_call_id')
            if m['role'] != 'tool' or not cid or cid in self._stubbed:
                continue
            saved = len(self.tok.encode(m['content'])) - stub_cost
            if saved > 0:
                self._stubbed.add(cid)
                size -= saved
        msgs, text, ids = self._render(messages, tools, options)
        if len(ids) <= room:
            return text, ids
        # 2. Forget the oldest turns, again down to 3/4 of the room. The
        #    turn being answered always stays. Nothing is remembered: the
        #    same request is cut at the same place every time.
        head = 1 if msgs and msgs[0]['role'] == 'system' else 0
        users = [i for i, m in enumerate(msgs)
                 if m['role'] == 'user' and i > head]
        for start in users:
            trial = msgs[:head] + msgs[start:]
            text, ids = self._encode(trial, tools, options)
            if len(ids) <= room * 3 // 4 or start == users[-1]:
                msgs = trial
                break
        if len(ids) <= room:
            return text, ids
        # 3. Cut the newest tool results (a file too big to read whole).
        last_call = max([i for i, m in enumerate(msgs)
                         if m['role'] == 'assistant'] or [0])
        over = len(ids) - room
        cut_cost = len(self.tok.encode(CUT)) + 8
        new = [m for m in msgs[last_call:] if m['role'] == 'tool']
        new.sort(key=lambda m: -len(m['content']))
        for m in new:
            if over <= 0:
                break
            part = self.tok.encode(m['content'], allow_special=False)
            keep = max(64, len(part) - over - cut_cost)
            if keep < len(part):
                over -= len(part) - keep
                m['content'] = self.tok.decode(part[:keep]) + CUT
        text, ids = self._encode(msgs, tools, options)
        if len(ids) <= room:
            return text, ids
        raise ChatError(
            'the conversation is %d tokens and the local model\'s context '
            'is %d. Start a new conversation, or load the model with a '
            'larger context (and fewer tools: their descriptions are part '
            'of every prompt).' % (len(ids), self.context))

    def cached_prefix(self, ids):
        '''How many leading tokens of ids the cache already holds.'''
        old = self.cache.tokens
        n = 0
        while n < min(len(old), len(ids) - 1) and old[n] == ids[n]:
            n += 1
        return n

    # -- request -----------------------------------

    def chat(self, messages, tools=None, max_tokens=None, temperature=None,
             top_k=None, top_p=None, repetition_penalty=None,
             enable_thinking=None, on_text=None, on_prompt=None):
        '''Answer one chat-completion request; returns the response dict.

        on_text(piece) is called with the raw text as it is generated
        (thinking and tool-call markup included); on_prompt(new, cached)
        before the prompt is read, with how many of its tokens have to be
        computed and how many the cache already holds.
        '''
        max_new = int(max_tokens or self.max_tokens)
        options = {}
        think = self.enable_thinking if enable_thinking is None \
            else enable_thinking
        if think is not None:
            options['enable_thinking'] = bool(think)
        text, ids = self._prompt(messages, tools or None, options, max_new)
        max_new = min(max_new, self.context - len(ids))

        p = dict(self.sampling)
        for key, v in (('temperature', temperature), ('top_k', top_k),
                       ('top_p', top_p),
                       ('repetition_penalty', repetition_penalty)):
            if v is not None:
                p[key] = v
        sampler = Sampler(p.get('temperature', 0.0), p.get('top_k', 0),
                          p.get('top_p', 1.0),
                          p.get('repetition_penalty', 1.0),
                          seed=self.seed + self._calls)
        self._calls += 1

        cached = self.cached_prefix(ids)
        if on_prompt:
            on_prompt(len(ids) - cached, cached)
        stream = self.tok.stream(skip_special=False)
        pieces = []

        def on_token(i):
            piece = stream.feed(i)
            if piece:
                pieces.append(piece)
                if on_text:
                    on_text(piece)
        out, reason, stats = generate(self.model, self.cache, ids, max_new,
                                      sampler, self.stops, on_token)
        tail = stream.flush()
        if tail:
            pieces.append(tail)
            if on_text:
                on_text(tail)
        raw = ''.join(pieces)

        reasoning, content, calls = parse_reply(
            raw, tools, thinking_open=text.rstrip().endswith('<think>'))
        if not content and not calls and reason != 'stop':
            # Out of tokens before saying anything: say that, rather
            # than hand the agent an empty answer.
            content = ('[the local model ran out of room after %d tokens '
                       'without answering; raise max_tokens or the context, '
                       'or start a new conversation]' % len(out))
        message = {'role': 'assistant', 'content': content}
        if reasoning:
            message['reasoning_content'] = reasoning
        if calls:
            stamp = '%x%s' % (int(time.time()), os.urandom(3).hex())
            message['tool_calls'] = [
                {'id': 'call_%s_%d' % (stamp, i), 'type': 'function',
                 'function': {'name': name,
                              'arguments': json.dumps(args,
                                                      ensure_ascii=False)}}
                for i, (name, args) in enumerate(calls)]
            if reasoning:
                self._reasoning[message['tool_calls'][0]['id']] = reasoning
        finish = 'tool_calls' if calls else \
            ('stop' if reason == 'stop' else 'length')
        rate = stats['new_tokens'] / stats['decode_s'] \
            if stats['decode_s'] else 0.0
        self.last = {'prompt_tokens': len(ids), 'cached_tokens': cached,
                     'new_tokens': len(out), 'prefill_s': stats['prefill_s'],
                     'decode_s': stats['decode_s'], 'tokens_per_s': rate,
                     'stop': reason, 'raw': raw}
        return {
            'id': 'chatcmpl-local-%d' % self._calls,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': self.name,
            'choices': [{'index': 0, 'message': message,
                         'finish_reason': finish}],
            'usage': {'prompt_tokens': len(ids),
                      'completion_tokens': len(out),
                      'total_tokens': len(ids) + len(out)},
        }
