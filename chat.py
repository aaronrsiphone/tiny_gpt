# coding: utf-8
'''
Inference and evaluation for a trained tiny_gpt checkpoint.

    python chat.py chat model.npz               interactive REPL
    python chat.py ask model.npz "why is the sky blue?"
    python chat.py sample model.npz --n 800     free-running generation
    python chat.py eval model.npz chat.txt      loss, perplexity, bits/char
    python chat.py info model.npz               what is in the file

Generation uses tiny_gpt's KV cache: 12x faster than recomputing the full
context per token, and verified to produce byte-identical output to the
uncached path within block_size.

Sampling knobs:
    --temp T    below 1 sharpens, above 1 flattens. 0 is greedy.
    --top-k K   keep only the K most likely tokens (default 40)
    --top-p P   nucleus: keep the smallest set summing to P
    --seed S    reproducible output

The tokenizer comes out of the checkpoint, so nothing here needs to be told
whether a model is character-level or token-level. A token-level model's
output is shown in the tokenizer's own compact form -- for the math
tokenizer that means <step>1: rather than "Step 1:", because the spacing and
letter case were never in the ids to begin with.

Expectation, stated plainly: this is a model of under a million parameters.
A character-level one learns spelling, the U:/A: turn structure, sentence
rhythm and answer shape; it does not learn to answer questions. Judge it on
whether the output looks like its training data, not on whether it is right.
'''

import math
import sys
import time

import numpy as np

from tinygpt.checkpoint import load_checkpoint


# ------------------------------------------------------------------ helpers

def parse(args, defaults):
    '''Pull --key value pairs off the tail of argv.'''
    out = dict(defaults)
    rest = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith('--'):
            key = a[2:].replace('-', '_')
            if key not in out:
                raise SystemExit('unknown option --%s' % a[2:])
            i += 1
            if i >= len(args):
                raise SystemExit('--%s needs a value' % key)
            cur = out[key]
            out[key] = type(cur)(args[i]) if cur is not None \
                else float(args[i])
        else:
            rest.append(a)
        i += 1
    return out, rest


def load(path):
    model, tok, meta = load_checkpoint(path)
    return model, tok, meta


def describe(meta, tok):
    print('%s params, %d layers, %d heads, %d dim, block_size %d'
          % ('{:,}'.format(meta['n_params']), meta['n_layer'],
             meta['n_head'], meta['n_embd'], meta['block_size']))
    print('vocab %d (%s tokenizer), trained %d steps'
          % (meta['vocab'], tok.name, meta['step']))
    if meta.get('val_loss') is not None:
        v = meta['val_loss']
        print('val loss %.4f  (perplexity %.1f, %.2f bits/%s)'
              % (v, math.exp(v), v / math.log(2), unit(tok)))


def unit(tok):
    """What one id is worth, for labelling bits-per-thing."""
    return 'char' if tok.name == 'char' else 'token'


def usable(tok, text):
    """Can this tokenizer turn `text` into ids? Says so when it cannot.

    Two tokenizers, two policies. A character tokenizer drops what it has
    never seen, because its vocabulary is only ever whatever the corpus
    happened to contain, so an unseen character is ordinary. A
    fixed-vocabulary tokenizer refuses, because unknown text means the input
    is outside the language the model was trained on, and an answer
    assembled from <unk> would read like a reply while being noise.
    """
    miss = tok.unknown(text)
    if not miss:
        return True
    if tok.strict:
        print('the %s tokenizer has no rule for: %s' % (tok.name, ''.join(miss)))
        print('it only knows the language its corpus is generated in.')
        return False
    print('(dropping text absent from the training vocabulary: %s)'
          % ''.join(miss))
    return True


def trim_turn(text, tok):
    """Cut a generated reply at the start of the next turn."""
    for s in tok.stops:
        text = text.split(s)[0]
    return text.strip()


# -------------------------------------------------------------------- modes

def info(argv):
    if not argv:
        raise SystemExit('usage: chat.py info model.npz')
    model, tok, meta = load(argv[0])
    describe(meta, tok)
    print('\nvocabulary:\n%s' % render_vocab(tok))


def render_vocab(tok):
    """The whole vocabulary, one line.

    A character vocabulary reads best as a run of characters with the
    unprintable ones dotted out; multi-character tokens need separating or
    the line is unreadable.
    """
    tokens = sorted(tok.token_to_id)
    if all(len(t) == 1 for t in tokens):
        return ''.join(t if 32 <= ord(t) < 127 else '.' for t in tokens)
    return ' '.join(t.replace('\n', '\\n') for t in tokens)


def ask(argv):
    opt, rest = parse(argv, {'temp': 0.8, 'top_k': 40, 'top_p': None,
                             'seed': 0, 'n': 300})
    if len(rest) < 2:
        raise SystemExit('usage: chat.py ask model.npz "your question"')
    model, tok, meta = load(rest[0])
    question = ' '.join(rest[1:])
    reply(model, tok, question, opt)


def reply(model, tok, question, opt):
    if not usable(tok, question):
        return

    prompt = "U: %s\nA:" % question
    t0 = time.time()
    out = model.generate(
        tok, prompt=prompt, n=int(opt['n']),
        temp=float(opt['temp']), top_k=int(opt['top_k']),
        top_p=opt['top_p'], seed=int(opt['seed']),
        stop=tok.stops)
    dt = time.time() - t0
    print('A: %s' % trim_turn(out, tok))
    print('\n[%d chars in %.2fs, %.0f chars/s]'
          % (len(out), dt, len(out) / max(dt, 1e-9)))


def chat(argv):
    opt, rest = parse(argv, {'temp': 0.8, 'top_k': 40, 'top_p': None,
                             'seed': 0, 'n': 300})
    if not rest:
        raise SystemExit('usage: chat.py chat model.npz')
    model, tok, meta = load(rest[0])
    describe(meta, tok)
    print('\nType a message. Blank line or Ctrl-D to quit.')
    print(f'Commands: /temp 0.2  /topk 20 /seed 42  /n 20\n')

    turn = 0
    while True:
        try:
            line = input('U: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            return
        if line.startswith('/'):
            parts = line[1:].split()
            if len(parts) == 2 and parts[0] in ('temp', 'topk', 'top_k',
                                                'seed', 'n'):
                key = 'top_k' if parts[0] in ('topk', 'top_k') else parts[0]
                opt[key] = float(parts[1]) if key == 'temp' \
                    else int(parts[1])
                print('  %s = %s' % (key, opt[key]))
            else:
                print('  commands: /temp /topk /seed /n')
            continue
        opt['seed'] = int(opt['seed']) + turn
        turn += 1
        reply(model, tok, line, opt)
        print()


def sample(argv):
    opt, rest = parse(argv, {'temp': 0.2, 'top_k': 40, 'top_p': None,
                             'seed': 0, 'n': 600})
    if not rest:
        raise SystemExit('usage: chat.py sample model.npz')
    model, tok, meta = load(rest[0])
    t0 = time.time()
    out = model.generate(tok, prompt='\n', n=int(opt['n']),
                         temp=float(opt['temp']), top_k=int(opt['top_k']),
                         top_p=opt['top_p'], seed=int(opt['seed']))
    dt = time.time() - t0
    print(out)
    print('\n[%d chars in %.2fs, %.0f chars/s]'
          % (len(out), dt, len(out) / max(dt, 1e-9)))


def evaluate(argv):
    '''Loss / perplexity / bits-per-token over a text file.

    Deterministic: strides through the file at fixed offsets rather than
    sampling random windows, so two runs on the same file give the same
    number and two checkpoints are directly comparable.

    Careful when comparing two tokenizers on the same file: a token-level
    model predicts fewer, larger units, so its bits-per-token is not
    comparable to a character model's bits-per-character. Bits per character
    is the common denominator, and is reported alongside.
    '''
    opt, rest = parse(argv, {'batches': 100, 'B': 8})
    if len(rest) < 2:
        raise SystemExit('usage: chat.py eval model.npz corpus.txt')
    model, tok, meta = load(rest[0])
    with open(rest[1], 'r', errors='replace') as f:
        text = f.read()

    miss = tok.unknown(text)
    if miss:
        if tok.strict:
            raise SystemExit(
                'the %s tokenizer has no rule for %d characters in this '
                'file: %s\nIt can only score text in the language it was '
                'built for.' % (tok.name, len(miss), ''.join(miss[:40])))
        print('WARNING: %d characters in this file are not in the model '
              'vocabulary.' % len(miss))
        print('They are dropped, which makes the score optimistic. '
              'Characters: %s' % ''.join(miss[:40]))

    n_chars = len(text)
    data = np.array(tok.encode(text), dtype=np.int64)
    T = model.cfg.block_size
    B = int(opt['B'])
    nb = int(opt['batches'])
    span = len(data) - T - 1
    if span <= 0:
        raise SystemExit('file shorter than block_size')

    stride = max(1, span // (nb * B))
    tot, seen = 0.0, 0
    t0 = time.time()
    for b in range(nb):
        ix = [(b * B + j) * stride % span for j in range(B)]
        x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
        loss, _ = model.forward(x, y)
        tot += loss
        seen += 1
        if b % 20 == 0:
            sys.stderr.write('\r  %d/%d batches' % (b, nb))
            sys.stderr.flush()
    sys.stderr.write('\r' + ' ' * 30 + '\r')

    mean = tot / seen
    bits = mean / math.log(2)
    print('file     %s  (%d chars, %d tokens, %d evaluated)'
          % (rest[1], n_chars, len(data), seen * B * T))
    describe(meta, tok)
    print('\nloss           %.4f  (natural log)' % mean)
    print('perplexity     %.2f' % math.exp(mean))
    print('%-15s%.3f' % ('bits/' + unit(tok), bits))
    if tok.name != 'char':
        print('%-15s%.3f  (%.2f chars/token)'
              % ('bits/char', bits * len(data) / max(n_chars, 1),
                 n_chars / max(len(data), 1)))
    print('%.1f s' % (time.time() - t0))
    print('\nreference: uniform over %d %ss is %.3f bits. English text at '
          'the character' % (meta['vocab'], unit(tok),
                             math.log(meta['vocab'], 2)))
    print('level bottoms out near 1.2-1.5 bits/char for small models.')


MODES = {'chat': chat, 'ask': ask, 'sample': sample, 'eval': evaluate,
         'info': info}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(__doc__)
        print('modes: %s' % ', '.join(sorted(MODES)))
    MODES[sys.argv[1]](sys.argv[2:])


if __name__ == '__main__':
    main()