# coding: utf-8
'''
Time single-token matrix-vector products on real BF16 weights, every way
we know how, on this device.

One "pass" = y = W @ x for every projection in the model (q, k, v, o,
gate, up, down for each layer, plus the vocabulary head) -- the matvec
part of decoding one token. Methods:

  slab8m     the lowmem path: tinygpt.hf.llama.Linear (8 MB scratch,
             BF16 -> F32 slab, then the autotuned sgemv/gemm). Widens with
             BNNSCopy where available; run with TINY_GPT_BF16_CONVERT=numpy
             for the old NumPy widening
  slab1m     same, 1 MB scratch (does a cache-sized slab help?)
  copyslab   slabs widened with BNNSCopy instead of NumPy, then sgemv
  mm_bf16    BNNSMatMul, BF16 weights straight from the mmap, F32 x -> F32 y
  mm_bf16x   BNNSMatMul, BF16 weights, x rounded to BF16 first -> F32 y
  fc_bf16    BNNS fully connected filter, BNNSFlagsUseClientPtr, BF16 W
  f32        float32 mode: weights widened once, plain sgemv (speed
             ceiling; costs 2x the weight size in memory)

Methods are timed round-robin (one pass of each per round) so thermal
drift hits them all equally; the median per method is reported. Each is
also checked against a float64 reference on a few tensors.

Pythonista: press Run (finds the model under tiny_gpt/hf_models/). Shell:

    python tools/native16_bench.py [--model DIR] [--rounds 30] [--only a,b]

Writes native16_results/bench_<time>.txt and .json.
'''

import ctypes
import datetime
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import native16_diagnostics as nd                      # noqa: E402
from tinygpt import backend                            # noqa: E402
from tinygpt.hf import llama                           # noqa: E402
from tinygpt.hf.safetensors import Checkpoint, bf16_to_f32  # noqa: E402

ALL = ['slab8m', 'slab1m', 'copyslab', 'mm_bf16', 'mm_bf16x', 'fc_bf16',
       'f32']


# ------------------------------------------------------------------ weights

def load_tensors(model_dir):
    '''[(name, uint16 (rows, cols) mmap view)] for every projection.'''
    ck = Checkpoint(model_dir, use_mmap=True)
    names = sorted((n for n in ck.keys() if n.endswith('proj.weight')),
                   key=_layer_order)
    head = 'lm_head.weight' if 'lm_head.weight' in ck else \
        'model.embed_tokens.weight'
    names.append(head)
    out = []
    for n in names:
        if ck.info(n).dtype != 'BF16':
            raise SystemExit('%s is %s; this benchmark wants BF16'
                             % (n, ck.info(n).dtype))
        out.append((n, ck.get_raw(n)))
    return ck, out


def _layer_order(name):
    parts = name.split('.')
    try:
        layer = int(parts[2])
    except (IndexError, ValueError):
        layer = 1 << 30
    order = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj',
             'down_proj']
    kind = next((i for i, k in enumerate(order) if k in name), 99)
    return (layer, kind)


def synthetic_tensors():
    '''SmolLM2-135M shapes, random BF16, deliberately 8-byte (not 16-byte)
    aligned like the real mapped file. Used only when no model is found.'''
    rng = np.random.RandomState(0)
    H, I, KV, L, V = 576, 1536, 192, 30, 49152
    shapes = []
    for i in range(L):
        p = 'model.layers.%d.' % i
        shapes += [(p + 'q_proj', H, H), (p + 'k_proj', KV, H),
                   (p + 'v_proj', KV, H), (p + 'o_proj', H, H),
                   (p + 'gate_proj', I, H), (p + 'up_proj', I, H),
                   (p + 'down_proj', H, I)]
    shapes.append(('lm_head', V, H))
    out = []
    for name, r, c in shapes:
        buf = np.empty(r * c + 32, np.uint16)
        off = (8 - buf.ctypes.data % 64) % 64 // 2   # address = 8 mod 64
        w = buf[off:off + r * c].reshape(r, c)
        w[...] = nd.f32_to_bf16_bits(
            (rng.standard_normal((r, c)) * 0.02).astype(np.float32))
        out.append((name, w))
    return None, out


# ------------------------------------------------------------------ methods
# Each method: setup(tensors) -> run(xs, ys) doing one full pass, where
# xs[n_in] is the shared input for that width and ys[i] the i-th output.

def m_slab(tensors, mb):
    mv = llama.MatVec()
    scratch = np.empty((mb << 20) // 4, np.float32)
    lins = [llama.Linear([(w, 'BF16')], 'lowmem', mv, scratch)
            for _, w in tensors]

    def run(xs, ys):
        for i, lin in enumerate(lins):
            lin.matvec(xs[lin.n_in], ys[i])
    return run


def _bnns():
    return ctypes.CDLL(nd.ACCELERATE)


def m_copyslab(tensors, mb=8):
    lib = _bnns()
    copy = nd._bind(lib, 'BNNSCopy', ctypes.c_int,
                    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p])
    scratch = np.empty((mb << 20) // 4, np.float32)
    plans = []
    for _, w in tensors:
        n_out, n_in = w.shape
        rows = max(1, scratch.size // n_in)
        slabs = []
        for a in range(0, n_out, rows):
            b = min(n_out, a + rows)
            dst = scratch[:(b - a) * n_in]
            sd = nd.NDArrayDesc()
            sd.layout = nd.LAYOUT_VECTOR
            sd.size[0] = (b - a) * n_in
            sd.data = w[a:b].ctypes.data
            sd.data_type = nd.DT_BF16
            dd = nd._vec_desc(dst, nd.DT_F32)
            slabs.append((a, b, ctypes.byref(dd), ctypes.byref(sd), dd, sd,
                          dst.reshape(b - a, n_in)))
        plans.append(slabs)
    gemv = backend.gemv

    def run(xs, ys):
        for i, slabs in enumerate(plans):
            y = ys[i]
            for a, b, pd, ps, _, _, W in slabs:
                if copy(pd, ps, None):
                    raise RuntimeError('BNNSCopy failed')
                gemv(W, xs[W.shape[1]], y[a:b])
    return run


def _mm_descs(w, x, y, x_dt):
    n_out, n_in = w.shape

    def mat(ptr, rows, cols, dt):
        d = nd.NDArrayDesc()
        d.layout = nd.LAYOUT_2D_FIRST_MAJOR
        d.size[0], d.size[1] = rows, cols
        d.stride[0], d.stride[1] = 1, cols
        d.data = ptr
        d.data_type = dt
        return d
    return (mat(w.ctypes.data, n_out, n_in, nd.DT_BF16),
            mat(x.ctypes.data, n_in, 1, x_dt),
            mat(y.ctypes.data, n_out, 1, nd.DT_F32))


def m_matmul(tensors, ys, bf16_input):
    '''Descriptors point at fixed x/y buffers, so run() copies the caller's
    x in (and rounds it to BF16 for mm_bf16x) and reads y out by view.'''
    lib = _bnns()
    argt = [ctypes.c_bool, ctypes.c_bool, ctypes.c_float, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p]
    mm = nd._bind(lib, 'BNNSMatMul', ctypes.c_int, argt)
    wsz = nd._bind(lib, 'BNNSMatMulWorkspaceSize', ctypes.c_ssize_t,
                   argt[:6] + [ctypes.c_void_p])
    x_dt = nd.DT_BF16 if bf16_input else nd.DT_F32
    xbuf = {}
    for _, w in tensors:
        n = w.shape[1]
        if n not in xbuf:
            xbuf[n] = np.zeros(n, np.uint16 if bf16_input else np.float32)
    calls, keep = [], []
    for i, (_, w) in enumerate(tensors):
        A, B, C = _mm_descs(w, xbuf[w.shape[1]], ys[i], x_dt)
        n = int(wsz(False, False, 1.0, ctypes.byref(A), ctypes.byref(B),
                    ctypes.byref(C), None))
        if n < 0:
            raise RuntimeError('BNNSMatMul refused %r' % (w.shape,))
        ws = np.empty(max(n, 16), np.uint8)
        keep += [A, B, C, ws]
        calls.append((ctypes.byref(A), ctypes.byref(B), ctypes.byref(C),
                      ws.ctypes.data))

    def run(xs, ys_):
        for n, buf in xbuf.items():
            if bf16_input:
                buf[...] = nd.f32_to_bf16_bits(xs[n])
            else:
                buf[...] = xs[n]
        for A, B, C, ws in calls:
            if mm(False, False, 1.0, A, B, C, ws, None):
                raise RuntimeError('BNNSMatMul failed')
    run.keep = keep
    return run


def m_fc(tensors):
    lib = _bnns()
    create = nd._bind(lib, 'BNNSFilterCreateLayerFullyConnected',
                      ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p])
    apply_ = nd._bind(lib, 'BNNSFilterApply', ctypes.c_int,
                      [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p])
    destroy = nd._bind(lib, 'BNNSFilterDestroy', None, [ctypes.c_void_p])
    filters = []
    for _, w in tensors:
        n_out, n_in = w.shape
        p = nd.FCParams()
        p.i_desc.layout = nd.LAYOUT_VECTOR
        p.i_desc.size[0] = n_in
        p.i_desc.data_type = nd.DT_F32
        p.o_desc.layout = nd.LAYOUT_VECTOR
        p.o_desc.size[0] = n_out
        p.o_desc.data_type = nd.DT_F32
        p.w_desc.layout = nd.LAYOUT_ROW_MAJOR
        p.w_desc.size[0], p.w_desc.size[1] = n_in, n_out
        p.w_desc.data = w.ctypes.data
        p.w_desc.data_type = nd.DT_BF16
        p.bias.layout = nd.LAYOUT_VECTOR
        p.bias.size[0] = n_out
        p.bias.data_type = nd.DT_F32
        fp = nd.FilterParams()
        fp.flags = nd.FLAG_USE_CLIENT_PTR
        f = create(ctypes.byref(p), ctypes.byref(fp))
        if not f:
            raise RuntimeError('FC refused %r' % (w.shape,))
        filters.append((f, n_in))

    def run(xs, ys):
        for i, (f, n_in) in enumerate(filters):
            if apply_(f, xs[n_in].ctypes.data, ys[i].ctypes.data):
                raise RuntimeError('BNNSFilterApply failed')
    run.close = lambda: [destroy(f) for f, _ in filters]
    return run


def m_f32(tensors):
    Ws = [bf16_to_f32(w) for _, w in tensors]
    mv = llama.MatVec()
    fns = [mv.tune(W) for W in Ws]

    def run(xs, ys):
        for i, W in enumerate(Ws):
            fns[i](W, xs[W.shape[1]], ys[i])
    return run


# ---------------------------------------------------------------- harness

def footprint_mb():
    b = nd._task_footprint()
    return None if b is None else b / 1e6


def bench(model=None, rounds=30, warmup=3, only=None, log=print):
    if model is None:
        found = nd.find_models()
        model = found[0] if found else None
    if model:
        ck, tensors = load_tensors(model)
        src = model
    else:
        ck, tensors = synthetic_tensors()
        src = 'synthetic SmolLM2-135M shapes (no model found)'
    n_params = sum(w.size for _, w in tensors)
    log('weights   %s' % src)
    log('          %d matrices, %.1fM params, %.0f MB as BF16'
        % (len(tensors), n_params / 1e6, n_params * 2 / 1e6))

    rng = np.random.RandomState(1)
    widths = sorted(set(w.shape[1] for _, w in tensors))
    xs = {n: rng.standard_normal(n).astype(np.float32) for n in widths}

    methods = [m for m in (only or ALL) if m in ALL]
    outs, runs, setup = {}, {}, {}
    mem = {'baseline': footprint_mb()}
    for m in methods:
        ys = [np.zeros(w.shape[0], np.float32) for _, w in tensors]
        t0 = time.perf_counter()
        try:
            if m == 'slab8m':
                run = m_slab(tensors, 8)
            elif m == 'slab1m':
                run = m_slab(tensors, 1)
            elif m == 'copyslab':
                run = m_copyslab(tensors)
            elif m == 'mm_bf16':
                run = m_matmul(tensors, ys, False)
            elif m == 'mm_bf16x':
                run = m_matmul(tensors, ys, True)
            elif m == 'fc_bf16':
                run = m_fc(tensors)
            else:
                run = m_f32(tensors)
            run(xs, ys)
        except Exception as e:
            log('  %-9s setup failed: %r' % (m, e))
            setup[m] = {'error': repr(e)}
            continue
        setup[m] = {'seconds': time.perf_counter() - t0}
        mem['after_' + m] = footprint_mb()
        runs[m], outs[m] = run, ys
        log('  %-9s ready (%.2fs setup)' % (m, setup[m]['seconds']))

    # Correctness on the first q, the first down and the head.
    pick = [0, min(6, len(tensors) - 1), len(tensors) - 1]
    errs = {}
    for m, ys in outs.items():
        runs[m](xs, ys)
        e = []
        for i in pick:
            w = tensors[i][1]
            ref = bf16_to_f32(w).astype(np.float64) @ xs[w.shape[1]]
            d = np.abs(ys[i] - ref).max() / (np.abs(ref).max() + 1e-30)
            e.append(float(d))
        errs[m] = max(e)

    log('timing: %d warmup + %d rounds, round-robin' % (warmup, rounds))
    times = {m: [] for m in runs}
    for r in range(warmup + rounds):
        for m, run in runs.items():
            t0 = time.perf_counter()
            run(xs, outs[m])
            dt = time.perf_counter() - t0
            if r >= warmup:
                times[m].append(dt)
    for run in runs.values():
        if hasattr(run, 'close'):
            run.close()

    res = {'source': src, 'matrices': len(tensors), 'params': n_params,
           'rounds': rounds, 'methods': {}, 'memory_mb': mem,
           'setup': setup, 'environment': {
               'machine': nd.sysctl_info().get('hw.machine'),
               'ios': nd.ios_info().get('systemVersion'),
               'python': sys.version.split()[0],
               'numpy': np.__version__,
               'time': datetime.datetime.now().isoformat()}}
    base = None
    if 'slab8m' in times:
        base = float(np.median(times['slab8m']))
    for m, t in times.items():
        med = float(np.median(t))
        res['methods'][m] = {
            'median_ms': med * 1e3, 'min_ms': min(t) * 1e3,
            'p90_ms': float(np.percentile(t, 90)) * 1e3,
            'speedup_vs_slab8m': (base / med) if base else None,
            'GBps_bf16': n_params * 2 / med / 1e9,
            'max_rel_err': errs[m]}
    return res


def report(res):
    L = ['native16 bench   %s  iOS %s  python %s  numpy %s' % (
        res['environment']['machine'], res['environment']['ios'],
        res['environment']['python'], res['environment']['numpy']),
        'weights          %s' % res['source'],
        '                 %d matrices, %.1fM params; %d rounds'
        % (res['matrices'], res['params'] / 1e6, res['rounds']),
        '',
        '  method     median ms/pass   min    p90   vs slab8m  GB/s(bf16)  '
        'max rel err']
    for m in ALL:
        r = res['methods'].get(m)
        if r is None:
            if m in res['setup'] and 'error' in res['setup'][m]:
                L.append('  %-9s  failed: %s' % (m, res['setup'][m]['error']))
            continue
        L.append('  %-9s  %10.2f   %6.2f %6.2f   %6s     %6.1f      %.1e' % (
            m, r['median_ms'], r['min_ms'], r['p90_ms'],
            '%.2fx' % r['speedup_vs_slab8m'] if r['speedup_vs_slab8m']
            else '-', r['GBps_bf16'], r['max_rel_err']))
    L.append('')
    mem = res['memory_mb']
    prev = mem.get('baseline')
    parts = []
    for m in ALL:
        k = 'after_' + m
        if k in mem and mem[k] is not None and prev is not None:
            parts.append('%s %+.0f' % (m, mem[k] - prev))
            prev = mem[k]
    if parts:
        L.append('footprint MB    baseline %.0f; added by setup: %s'
                 % (mem['baseline'], ', '.join(parts)))
    L.append('A pass is the matvec work of one decoded token (no attention, '
             'norms or sampling).')
    return '\n'.join(L) + '\n'


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--rounds', type=int, default=30)
    ap.add_argument('--only', help='comma list of: ' + ','.join(ALL))
    a = ap.parse_args(argv)
    res = bench(a.model, a.rounds,
                only=a.only.split(',') if a.only else None)
    text = report(res)
    print('')
    print(text)
    out = os.path.join(REPO, 'native16_results')
    if not os.path.isdir(out):
        os.makedirs(out)
    base = os.path.join(out, 'bench_%s' % datetime.datetime.now().strftime(
        '%Y%m%d_%H%M%S'))
    with open(base + '.json', 'w') as f:
        json.dump(res, f, indent=1, sort_keys=True)
    with open(base + '.txt', 'w') as f:
        f.write(text)
    print('wrote %s.txt and .json' % base)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
