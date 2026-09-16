# coding: utf-8
'''
Corpus loading, character vocabulary, and batching.
'''

import os

import numpy as np


def load_corpus(path=None):
    if path and os.path.isfile(path):
        with open(path, 'r', errors='replace') as f:
            return f.read(), path
    chunks, names = [], []
    for fn in sorted(os.listdir('.')):
        if fn.endswith('.py') or fn.endswith('.txt'):
            try:
                with open(fn, 'r', errors='replace') as f:
                    chunks.append(f.read())
                names.append(fn)
            except Exception:
                pass
    if not chunks:
        raise SystemExit('no corpus: pass a text file as an argument')
    return '\n\n'.join(chunks), '%d files (%s)' % (len(names),
                                                   ', '.join(names[:4]))


def make_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def batcher(data, B, T, rng):
    while True:
        ix = rng.randint(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
        yield x, y
