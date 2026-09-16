# coding: utf-8
'''
The training loop.
'''

import math
import time

import numpy as np

from tinygpt import backend
from tinygpt.checkpoint import load_checkpoint, save_checkpoint
from tinygpt.data import batcher, load_corpus, make_vocab
from tinygpt.model import Config, GPT
from tinygpt.optim import Adam


def estimate_loss(model, data, B, batches=20, seed=1234):
    """Mean loss over random batches. Used for the held-out split."""
    rng = np.random.RandomState(seed)
    T = model.cfg.block_size
    tot = 0.0
    for _ in range(batches):
        ix = rng.randint(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
        loss, _ = model.forward(x, y)
        tot += loss
    return tot / batches


def model_min_val(data, frac):
    """Validation split size, but never more than a quarter of the data."""
    return int(min(len(data) * min(frac, 0.25), len(data) // 4))


def train(path=None, steps=2000, B=32, lr=2e-3, seed=0, block_size=None,
          n_layer=4, n_head=4, n_embd=128, ckpt='model.npz',
          ckpt_every=250, val_frac=0.05, resume=None, rope=1, qk_norm=1,
          act='relu2', zero_init=1, value_residual=1, softcap=30.0):
    text, src = load_corpus(path)
    stoi, itos = make_vocab(text)
    data = np.array([stoi[c] for c in text], dtype=np.int64)

    # Held-out tail. A char model on a few MB overfits readily, and train
    # loss alone will keep dropping while the model memorises the corpus.
    n_val = max(model_min_val(data, val_frac), 0)
    train_data, val_data = (data[:-n_val], data[-n_val:]) if n_val else \
        (data, None)

    if resume:
        model, stoi_r, itos_r, meta, opt = load_checkpoint(
            resume, with_optimizer=True)
        if stoi_r != stoi:
            raise SystemExit(
                'resume vocabulary differs from this corpus: the checkpoint '
                'was trained on a different text, so its embedding rows do '
                'not match these character ids. Train on the same corpus or '
                'start fresh.')
        stoi, itos = stoi_r, itos_r
        start = int(meta.get('step', 0))
        cfg = model.cfg
        # Continue the cosine where it stopped rather than re-warming a
        # converged model. The new total extends the old schedule by
        # however many steps were asked for.
        sch = meta.get('sched')
        k0 = int(sch[0]) if sch else 0
        total = k0 + steps
        print('resumed  %s at step %d (schedule %d/%d)'
              % (resume, start, k0, total))
    else:
        cfg = Config(vocab=len(stoi), n_layer=n_layer, n_head=n_head,
                     n_embd=n_embd, block_size=block_size or 64,
                     rope=rope, qk_norm=qk_norm, act=act,
                     zero_init=zero_init, value_residual=value_residual,
                     softcap=softcap)
        model = GPT(cfg, seed=seed)
        opt = Adam(model.p, lr=lr, weight_decay=0.01)
        start = 0
        k0, total = 0, steps

    print('corpus   %s' % src)
    print('chars    %d train / %d val   vocab %d'
          % (len(train_data), n_val, cfg.vocab))
    print('model    %d params  (%d layers, %d heads, %d dim, block %d)'
          % (model.n_params, cfg.n_layer, cfg.n_head, cfg.n_embd,
             cfg.block_size))
    print('arch     %s' % cfg.describe())
    print('tokens/step %d   accelerate %s   bmm %s   vForce %s'
          % (B * cfg.block_size, backend.HAVE_ACCELERATE,
             backend.BMM_BACKEND, backend.HAVE_VFORCE))
    print('checkpoint %s every %d steps\n' % (ckpt, ckpt_every))

    rng = np.random.RandomState(seed)
    gen = batcher(train_data, B, cfg.block_size, rng)
    warmup = max(1, min(100, total // 10))
    best = float('inf')
    t0 = time.time()
    print('%-7s %10s %9s %9s %9s %9s' % ('step', 'lr',  'loss', 'val', 'ms/step',
                                    'elapsed'))
    for s in range(start, start + steps):
        j = s - start
        k = k0 + j
        cur = lr * (k + 1) / warmup if k < warmup else \
            lr * (0.1 + 0.9 * 0.5 *
                  (1 + math.cos(math.pi * (k - warmup) /
                                max(1, total - warmup))))
        x, y = next(gen)
        loss, ca = model.forward(x, y)
        grads = model.backward(ca)
        opt.step(model.p, grads, lr=cur)

        last = (j == steps - 1)
        if j % ckpt_every == 0 or last:
            vl = estimate_loss(model, val_data, B, batches=32) \
                if val_data is not None and len(val_data) > cfg.block_size \
                else float('nan')
            el = time.time() - t0
            print('%-7d %.4e %9.4f %9.4f %9.1f %8.1fs'
                  % (s, cur, loss, vl, el / (j + 1) * 1000.0, el))
            if not (vl == vl) or vl < best:       # nan-safe
                best = vl
                save_checkpoint(ckpt, model, stoi, step=s + 1, opt=opt,
                                val_loss=None if vl != vl else float(vl),
                                sched=(k + 1, total))
        elif j % 100 == 0:
            el = time.time() - t0
            print('%-7d %.4e %9.4f %9s %9.1f %8.1fs'
                  % (s, cur, loss, '-', el / (j + 1) * 1000.0, el))

    dt = time.time() - t0
    tok = steps * B * cfg.block_size
    print('\n%d steps, %d tokens in %.1f s  (%.0f tokens/s)'
          % (steps, tok, dt, tok / dt))
    print('%.1f GFLOP/s sustained end-to-end'
          % (6.0 * model.n_params * tok / dt / 1e9))
    print('saved    %s  (best val %.4f)' % (ckpt, best))

    print('\n' + '-' * 60)
    print(model.generate(stoi, itos, prompt='\n', n=600))
    print('-' * 60)
