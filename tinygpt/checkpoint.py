# coding: utf-8
'''
Saving and loading .npz checkpoints.

A checkpoint has to be self-contained: the vocabulary, the architecture
flags, and the optimizer state all travel with the weights, because none
of them can be safely guessed back from the array shapes alone.
'''

import json

import numpy as np

from tinygpt import backend, tokenizer
from tinygpt.model import Config, GPT, LEGACY_FLAGS
from tinygpt.optim import Adam

# Version 3 adds the 'tokenizer' field. Older checkpoints are all
# character-level, which is exactly what their absence of the field means.
CKPT_VERSION = 3


def save_checkpoint(path, model, tok, step=0, opt=None, val_loss=None,
                    sched=None):
    """Write params, vocab, config, arch flags and optimizer state to .npz.

    The vocabulary must travel with the weights: a character vocabulary is
    built from sorted(set(text)) of whatever corpus was used, so a different
    corpus yields different integer ids and the embedding rows stop meaning
    anything. Storing it removes the chance of silently loading a model
    against the wrong mapping.

    The tokenizer's name travels with it, because the mapping alone does not
    say how to split text: {'<u>': 5} in a table gives no hint that "U: "
    should become one id rather than three. Loading reconstructs the named
    tokenizer and checks it still agrees with the stored mapping.

    The architecture flags travel for the same reason -- a RoPE checkpoint
    has no wpe and a value-residual checkpoint has an extra parameter, so
    the flags have to be known before GPT() is constructed.

    `sched` is (k, total): position within the LR schedule. Without it a
    resume re-runs warmup on a converged model and validation jumps before
    recovering.
    """
    blob = {}
    for k, v in model.p.items():
        blob['p__' + k] = v
    if opt is not None:
        for k, v in opt.m.items():
            blob['m__' + k] = v
        for k, v in opt.v.items():
            blob['v__' + k] = v
        blob['opt_t'] = np.array(opt.t)
    cfg = model.cfg
    meta = {
        'version': CKPT_VERSION,
        'vocab': cfg.vocab, 'n_layer': cfg.n_layer, 'n_head': cfg.n_head,
        'n_embd': cfg.n_embd, 'block_size': cfg.block_size,
        'step': int(step), 'val_loss': val_loss,
        'n_params': int(model.n_params),
        'tokenizer': tok.name,
        'stoi': {c: int(i) for c, i in tok.token_to_id.items()},
        'arch': cfg.flags(),
        'sched': list(sched) if sched else None,
    }
    blob['meta'] = np.frombuffer(
        json.dumps(meta).encode('utf-8'), dtype=np.uint8)
    np.savez_compressed(path, **blob)
    return path


def load_checkpoint(path, with_optimizer=False):
    """Return (model, tok, meta[, opt]).

    Version-1 checkpoints predate the arch flags and are loaded with every
    lever off, which is exactly the architecture they were trained with.
    That keeps the 1.2521-val model.npz loadable by this file. Versions 1
    and 2 also predate the tokenizer field and are character-level.
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z['meta']).decode('utf-8'))
    ver = meta.get('version')
    if ver not in (1, 2, 3):
        raise SystemExit('checkpoint version %r, expected 1, 2 or %d'
                         % (ver, CKPT_VERSION))
    arch = dict(LEGACY_FLAGS)
    arch.update(meta.get('arch') or {})
    cfg = Config(meta['vocab'], meta['n_layer'], meta['n_head'],
                 meta['n_embd'], meta['block_size'], **arch)
    model = GPT(cfg)
    for k in list(model.p):
        key = 'p__' + k
        if key not in z:
            raise SystemExit('checkpoint missing parameter %r' % k)
        model.p[k] = np.ascontiguousarray(z[key], dtype=backend.DT)
    stoi = {c: int(i) for c, i in meta['stoi'].items()}
    tok = tokenizer.from_vocab(meta.get('tokenizer', 'char'), stoi)
    if not with_optimizer:
        return model, tok, meta
    opt = Adam(model.p)
    if 'opt_t' in z:
        for k in model.p:
            opt.m[k] = np.ascontiguousarray(z['m__' + k], dtype=backend.DT)
            opt.v[k] = np.ascontiguousarray(z['v__' + k], dtype=backend.DT)
        opt.t = int(z['opt_t'])
    return model, tok, meta, opt
