# coding: utf-8
'''
Verification and performance tools: gradcheck, gradcheck_all, bench,
profile.

gradcheck is what makes the hand-derived backward() in model.py
trustworthy: it compares every analytic gradient against a finite
difference computed independently in float64, with no shared code path
that could hide the same mistake in both.
'''

import collections
import time

import numpy as np

from tinygpt import backend
from tinygpt.backend import set_dtype
from tinygpt.data import batcher, load_corpus
from tinygpt.model import Config, GPT, LEGACY_FLAGS
from tinygpt.ops import act_bwd, act_fwd, layernorm, rms_norm, rope_apply, softmax_rows
from tinygpt.optim import Adam
from tinygpt.tokenizer import build as build_tokenizer


def gradcheck(**flags):
    '''Finite differences against every analytic gradient, tiny config.

    Runs in float64. In float32 the loss carries ~1e-7 of resolution, so a
    central difference over eps=1e-3 has an absolute noise floor near 6e-5 --
    larger than the LayerNorm gain/bias gradients themselves (~2e-3), which
    made those entries appear to fail at 3-6e-2 relative when the analytic
    values were in fact correct.

    eps is 1e-5, not the 1e-6 used before. The same noise floor bites again
    one precision down: roundoff in a central difference is about
    |loss|*2^-52/(2*eps), which at eps=1e-6 is ~3e-10 absolute. The
    smallest gradients in the table (ln2_g under relu2, ~2e-4) are then
    only ~1e-6 away in relative terms, and the check reports its own noise
    as a failure. Central-difference error is roundoff/eps plus truncation
    times eps^2, minimised near (3*2^-52)^(1/3) ~ 1e-5, which drops the
    floor to ~1e-7 relative while truncation is still negligible.

    Zero-init is honoured for construction and then undone: with a zero
    head the gradient of every upstream parameter is exactly zero at step
    0, so the check would compare 0 against 0 and pass without testing
    anything. The output projections are re-randomised before checking.

    python tiny_gpt.py gradcheck --rope 0 --qk-norm 0 ...  to isolate a
    single lever's derivation.
    '''
    set_dtype(np.float64)
    rng = np.random.RandomState(0)
    cfg = Config(vocab=11, n_layer=3, n_head=2, n_embd=16, block_size=8,
                 **flags)
    model = GPT(cfg, seed=1)

    if cfg.zero_init:
        for name in model.p:
            if name.startswith(('proj_w', 'fp_w')) or name == 'head':
                model.p[name] = (rng.randn(*model.p[name].shape)
                                 * 0.02).astype(backend.DT)
    if cfg.value_residual:
        # A zero lambda has a perfectly good gradient, but a nonzero one
        # also exercises the dv0 path back into layer 0.
        model.p['vres'] = (rng.randn(*model.p['vres'].shape)
                           * 0.3).astype(backend.DT)

    B, T = 2, 6
    idx = rng.randint(0, cfg.vocab, size=(B, T)).astype(np.int64)
    tgt = rng.randint(0, cfg.vocab, size=(B, T)).astype(np.int64)

    loss, ca = model.forward(idx, tgt)
    grads = model.backward(ca)
    print('params %d   loss %.6f' % (model.n_params, loss))
    print('arch   %s' % cfg.describe())
    print('accelerate: %s\n' % backend.HAVE_ACCELERATE)

    eps = 1e-5
    worst = 0.0
    worst_name = '-'
    print('%-12s %14s %14s %10s' % ('param', 'numeric', 'analytic', 'rel'))
    for name in sorted(model.p):
        a = model.p[name]
        flat = a.reshape(-1)
        gflat = grads[name].reshape(-1)
        j = int(np.argmax(np.abs(gflat)))
        old = float(flat[j])

        flat[j] = old + eps
        hi, _ = model.forward(idx, tgt)
        flat[j] = old - eps
        lo, _ = model.forward(idx, tgt)
        flat[j] = old

        num = (hi - lo) / (2 * eps)
        ana = float(gflat[j])
        rel = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
        if rel > worst:
            worst, worst_name = rel, name
        print('%-12s %14.6e %14.6e %10.2e' % (name, num, ana, rel))

    set_dtype(np.float32)
    print('\nworst relative %.3e (%s) -> %s'
          % (worst, worst_name, 'PASS' if worst < 1e-6 else 'FAIL'))
    print('(float64 forward + central differences)')
    return worst < 1e-6


def gradcheck_all():
    """gradcheck the baseline, each lever alone, and everything together.

    Nine runs, a few seconds total. A lever that is correct in isolation
    and wrong in combination is the failure mode this catches -- QK-norm
    and RoPE compose on the same tensor, and value residual reaches across
    layers.
    """
    base = dict(LEGACY_FLAGS)
    trials = [('baseline', {})]
    for k, v in (('rope', True), ('qk_norm', True), ('act', 'relu2'),
                 ('zero_init', True), ('value_residual', True),
                 ('softcap', 30.0)):
        trials.append((k, {k: v}))
    trials.append(('all', {'rope': True, 'qk_norm': True, 'act': 'relu2',
                           'zero_init': True, 'value_residual': True,
                           'softcap': 30.0}))
    results = []
    for name, over in trials:
        f = dict(base)
        f.update(over)
        print('\n' + '=' * 60)
        print('gradcheck: %s' % name)
        print('=' * 60)
        results.append((name, gradcheck(**f)))
    print('\n' + '-' * 60)
    for name, ok in results:
        print('%-16s %s' % (name, 'PASS' if ok else 'FAIL'))
    return all(ok for _, ok in results)


def profile(steps=12, B=32, block_size=64, **flags):
    """Time each op category separately. Measurement, not estimation.

    The dense GEMMs run on AMX and the attention matmuls run on whichever
    bmm backend is selected, but everything else -- the activation,
    LayerNorm, softmax, RoPE, QK-norm, the head-splitting copies, Adam --
    is plain numpy. On an install with no SIMD dispatch that residue can
    dominate a step even though it is a rounding error in the FLOP count.
    """
    # Throughput is measured on whatever is lying around, character-level:
    # the point is ms/step at a given shape, not what the ids mean.
    text, _ = load_corpus(None)
    tok = build_tokenizer('char', text)
    data = np.array(tok.encode(text), dtype=np.int64)
    cfg = Config(vocab=tok.vocab_size, block_size=block_size, **flags)
    model = GPT(cfg)
    opt = Adam(model.p)
    rng = np.random.RandomState(0)
    gen = batcher(data, B, cfg.block_size, rng)

    T = collections.OrderedDict()
    BT, C = B * cfg.block_size, cfg.n_embd
    nh, hs = cfg.n_head, cfg.head_size

    for _ in range(3):
        x, y = next(gen)
        opt.step(model.p, model.backward(model.forward(x, y)[1]))

    T.clear()
    for _ in range(steps):
        x, y = next(gen)
        t0 = time.time(); loss, ca = model.forward(x, y)
        T['forward'] = T.get('forward', 0) + time.time() - t0
        t0 = time.time(); g = model.backward(ca)
        T['backward'] = T.get('backward', 0) + time.time() - t0
        t0 = time.time(); opt.step(model.p, g)
        T['adam'] = T.get('adam', 0) + time.time() - t0

    # isolate the pure-numpy pieces at representative sizes
    a4 = np.ascontiguousarray(np.random.randn(BT, 4 * C).astype(backend.DT))
    a1 = np.ascontiguousarray(
        np.random.randn(B, cfg.block_size, C).astype(backend.DT))
    at = np.ascontiguousarray(
        np.random.randn(B * nh, cfg.block_size,
                        cfg.block_size).astype(backend.DT))
    qh = np.ascontiguousarray(
        np.random.randn(B * nh, cfg.block_size, hs).astype(backend.DT))
    g1 = np.ascontiguousarray(np.random.randn(C).astype(backend.DT))
    b1 = np.zeros(C, backend.DT)
    reps = steps * cfg.n_layer

    def bulk(key, fn, n):
        t0 = time.time()
        for _ in range(n):
            fn()
        T[key] = time.time() - t0

    _, ac_ref = act_fwd(a4, cfg.act)
    bulk('  act', lambda: act_fwd(a4, cfg.act), reps)
    bulk('  dact', lambda: act_bwd(a4, a4, ac_ref, cfg.act), reps)
    bulk('  softmax', lambda: softmax_rows(at), reps)
    bulk('  layernorm', lambda: layernorm(a1, g1, b1), reps * 2)
    if cfg.rope:
        bulk('  rope', lambda: rope_apply(qh), reps * 2)
    if cfg.qk_norm:
        bulk('  qknorm', lambda: rms_norm(qh), reps * 2)

    tot = (T['forward'] + T['backward'] + T['adam']) / steps * 1000
    print('B=%d block_size=%d  bmm=%s  accelerate=%s  vForce=%s (enabled=%s)'
          % (B, cfg.block_size, backend.BMM_BACKEND, backend.HAVE_ACCELERATE,
             backend.HAVE_VFORCE, backend.USE_VFORCE))
    print('arch %s' % cfg.describe())
    print('%d params, %d tokens/step\n' % (model.n_params, BT))
    print('%-12s %10s %8s' % ('phase', 'ms/step', 'share'))
    for k in ('forward', 'backward', 'adam'):
        ms = T[k] / steps * 1000
        print('%-12s %10.2f %7.1f%%' % (k, ms, 100 * ms / tot))
    print('%-12s %10.2f' % ('TOTAL', tot))
    print('\ncomponent costs, summed over all layers per step:')
    for k in ('  act', '  dact', '  softmax', '  layernorm', '  rope',
              '  qknorm'):
        if k not in T:
            continue
        ms = T[k] / steps * 1000
        print('%-12s %10.2f %7.1f%% of step' % (k, ms, 100 * ms / tot))
    flop = 6.0 * model.n_params * BT
    print('\n%.1f GFLOP/s end-to-end' % (flop / (tot * 1e-3) / 1e9))


def bench(steps=20, B=32, **flags):
    saved = backend.BMM_BACKEND
    for name in ('numpy', 'accelerate'):
        if name == 'accelerate' and not backend.HAVE_ACCELERATE:
            continue
        backend.BMM_BACKEND = name
        print('=== bmm backend: %s ===' % name)
        _bench_one(steps, B, **flags)
        print()
    backend.BMM_BACKEND = saved


def _bench_one(steps=20, B=32, **flags):
    text, _ = load_corpus(None)
    tok = build_tokenizer('char', text)
    data = np.array(tok.encode(text), dtype=np.int64)
    cfg = Config(vocab=tok.vocab_size, **flags)
    model = GPT(cfg)
    opt = Adam(model.p)
    rng = np.random.RandomState(0)
    gen = batcher(data, B, cfg.block_size, rng)

    for _ in range(3):
        x, y = next(gen)
        _, ca = model.forward(x, y)
        opt.step(model.p, model.backward(ca))

    tf = tb = to = 0.0
    for _ in range(steps):
        x, y = next(gen)
        t = time.time(); loss, ca = model.forward(x, y); tf += time.time() - t
        t = time.time(); g = model.backward(ca); tb += time.time() - t
        t = time.time(); opt.step(model.p, g); to += time.time() - t

    tot = (tf + tb + to) / steps * 1000.0
    print('%d params, %d tokens/step, accelerate %s'
          % (model.n_params, B * cfg.block_size, backend.HAVE_ACCELERATE))
    print('arch %s\n' % cfg.describe())
    print('%-10s %10s %8s' % ('phase', 'ms/step', 'share'))
    for n, v in (('forward', tf), ('backward', tb), ('adam', to)):
        ms = v / steps * 1000.0
        print('%-10s %10.2f %7.1f%%' % (n, ms, ms / tot * 100))
    print('%-10s %10.2f' % ('TOTAL', tot))
    flop = 6.0 * model.n_params * B * cfg.block_size
    print('\n%.1f GFLOP/s   %.0f tokens/s'
          % (flop / (tot * 1e-3) / 1e9, B * cfg.block_size / (tot * 1e-3)))
