# coding: utf-8
'''
Run a pretrained Hugging Face language model on-device: NumPy plus, on
macOS/iOS, Apple's Accelerate framework. No inference API, no PyTorch.

In Pythonista: edit the settings below and press Run. Elsewhere, the
same settings are command-line options:

    python hf_run.py download                       fetch the model once
    python hf_run.py chat                           interactive, offline
    python hf_run.py ask "Why is the sky blue?"     one answer, offline
    python hf_run.py complete "Once upon a time"    raw continuation
    python hf_run.py import SmolLM2-135M-Instruct.zip   from a local zip/dir
    python hf_run.py bench                          speed and memory
    python hf_run.py selftest                       tests + bench -> report
    python hf_run.py info                           what is cached

    options: --model ID  --max-new N  --temp T  --top-k K  --top-p P
             --rep-penalty R
             --ctx N  --weights float32|lowmem  --seed S  --system TEXT
             --cache DIR  --report FILE

Download once with a connection; every other mode reads only the local
cache (hf_models/ next to this script) and never touches the network.
If the model is not cached but SmolLM2-135M-Instruct.zip (or a zip named
after the model) sits next to this script, it is imported automatically.

See tinygpt/hf/ for the implementation.
'''

import os
import sys
import time

# ============================================================== settings
# Edit these when running from Pythonista (no command-line arguments).

MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'   # or a huggingface.co URL
# Also: 'Qwen/Qwen2.5-Coder-1.5B-Instruct', 'LiquidAI/LFM3-1.7B' (3.5 GB)
MODE = 'chat'           # download | chat | ask | complete | import |
                        # bench | selftest | info
PROMPT = 'What is the capital of France?'   # for ask / complete
MAX_NEW_TOKENS = 256
# Sampling. None = what the model recommends in its generation_config.json
# (as transformers would use), else temperature 0.2 / top-p 0.9.
TEMPERATURE = None      # 0 = greedy (deterministic)
TOP_K = None            # 0 = off
TOP_P = None            # 1.0 = off
REPETITION_PENALTY = None   # 1.0 = off
CONTEXT_LIMIT = 1024    # tokens of KV cache; ~7 MB per 100 for SmolLM2-135M,
                        # ~15 MB per 100 for LFM3-1.7B
WEIGHTS = 'auto'        # 'auto': lowmem with Accelerate (faster there);
                        #   elsewhere float32 for small models, else lowmem
                        # 'float32': widened once at load, 4 bytes/param
                        # 'lowmem': weights stay in the file (~0 MB counted
                        #           against the app), widened as used
SYSTEM_PROMPT = None    # None = the model's own default
SEED = 0
CACHE_DIR = None        # None = hf_models/ next to this script
IMPORT_SOURCE = None    # zip or directory for MODE = 'import'
REPORT_FILE = 'hf_report.txt'   # written by bench / selftest

# =======================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def settings_from_argv(argv):
    import argparse
    p = argparse.ArgumentParser(
        description='Offline inference for Hugging Face models '
                    '(Llama, Qwen2, LFM3).')
    p.add_argument('mode', choices=['download', 'chat', 'ask', 'complete',
                                    'import', 'bench', 'selftest', 'info'])
    p.add_argument('text', nargs='?', default=None,
                   help='prompt (ask/complete) or source (import)')
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--max-new', type=int, default=MAX_NEW_TOKENS)
    p.add_argument('--temp', type=float, default=TEMPERATURE)
    p.add_argument('--top-k', type=int, default=TOP_K)
    p.add_argument('--top-p', type=float, default=TOP_P)
    p.add_argument('--rep-penalty', type=float, default=REPETITION_PENALTY)
    p.add_argument('--ctx', type=int, default=CONTEXT_LIMIT)
    p.add_argument('--weights', choices=['auto', 'float32', 'lowmem'],
                   default=WEIGHTS)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--system', default=SYSTEM_PROMPT)
    p.add_argument('--cache', default=CACHE_DIR)
    p.add_argument('--revision', default='main')
    p.add_argument('--report', default=REPORT_FILE)
    a = p.parse_args(argv)
    return {
        'mode': a.mode, 'model': a.model, 'max_new': a.max_new,
        'temp': a.temp, 'top_k': a.top_k, 'top_p': a.top_p, 'ctx': a.ctx,
        'rep': a.rep_penalty,
        'weights': a.weights, 'seed': a.seed, 'system': a.system,
        'cache': a.cache, 'revision': a.revision, 'report': a.report,
        'prompt': a.text if a.mode in ('ask', 'complete') else PROMPT,
        'source': a.text if a.mode == 'import' else IMPORT_SOURCE,
    }


def settings_from_globals():
    return {
        'mode': MODE, 'model': MODEL_ID, 'max_new': MAX_NEW_TOKENS,
        'temp': TEMPERATURE, 'top_k': TOP_K, 'top_p': TOP_P,
        'rep': REPETITION_PENALTY,
        'ctx': CONTEXT_LIMIT, 'weights': WEIGHTS, 'seed': SEED,
        'system': SYSTEM_PROMPT, 'cache': CACHE_DIR, 'revision': 'main',
        'report': REPORT_FILE, 'prompt': PROMPT, 'source': IMPORT_SOURCE,
    }


# ----------------------------------------------------------------- memory

def peak_rss_mb():
    '''Peak resident memory of this process in MB, or None.'''
    try:
        import resource
    except ImportError:
        return None
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB; macOS and iOS report bytes.
    return r / 1e6 if sys.platform in ('darwin', 'ios') else r / 1e3


def _task_vm_info():
    '''Mach task_vm_info for this process (macOS / iOS), or None.

    phys_footprint is the number iOS compares against an app's memory
    limit: dirty and compressed memory, but not clean file pages the OS
    can drop and re-read. Unlike ru_maxrss it reports *now*, so it means
    something even inside Pythonista, whose interpreter outlives a run.
    '''
    import ctypes
    u64 = ctypes.c_uint64

    class TaskVMInfo(ctypes.Structure):
        _fields_ = ([('virtual_size', u64), ('region_count', ctypes.c_int32),
                     ('page_size', ctypes.c_int32)]
                    + [(n, u64) for n in (
                        'resident_size', 'resident_size_peak', 'device',
                        'device_peak', 'internal', 'internal_peak',
                        'external', 'external_peak', 'reusable',
                        'reusable_peak', 'purgeable_volatile_pmap',
                        'purgeable_volatile_resident',
                        'purgeable_volatile_virtual', 'compressed',
                        'compressed_peak', 'compressed_lifetime',
                        'phys_footprint', 'min_address', 'max_address')]
                    + [('ledger_phys_footprint_peak', ctypes.c_int64)])

    try:
        lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        task = ctypes.c_uint32.in_dll(lib, 'mach_task_self_')
        task_info = lib.task_info
    except (OSError, ValueError, AttributeError):
        return None
    task_info.restype = ctypes.c_int
    task_info.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p,
                          ctypes.POINTER(ctypes.c_uint32)]
    info = TaskVMInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
    TASK_VM_INFO = 22
    if task_info(task.value, TASK_VM_INFO, ctypes.byref(info),
                 ctypes.byref(count)) != 0:
        return None
    # The kernel fills as many fields as it supports and says how many in
    # `count` (in 4-byte units); fields past that were not written.
    have = count.value * 4
    out = {'resident': info.resident_size}
    if have >= TaskVMInfo.min_address.offset:
        out['footprint'] = info.phys_footprint
    if have >= TaskVMInfo.ledger_phys_footprint_peak.offset + 8:
        out['footprint_peak'] = info.ledger_phys_footprint_peak
    return out


def memory_now():
    '''(MB, what it measures) for this process right now, or (None, '').

    iOS/macOS: phys_footprint, the figure the OS memory limit uses.
    Linux: resident set size from /proc.
    '''
    if sys.platform in ('darwin', 'ios'):
        info = _task_vm_info()
        if info and 'footprint' in info:
            return info['footprint'] / 1e6, 'footprint'
        if info:
            return info['resident'] / 1e6, 'resident'
        return None, ''
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1e3, 'resident'
    except (OSError, ValueError):
        pass
    return None, ''


def memory_line():
    mb, what = memory_now()
    return '%.0f MB %s' % (mb, what) if mb is not None else 'n/a'


# ------------------------------------------------------------------ model

def locate(s):
    '''The local snapshot for the model, importing a zip if one is next to
    this script. Never touches the network.'''
    from tinygpt.hf import hub
    path = hub.find_local(s['model'], s['cache'], s['revision'])
    if path:
        return path
    base = s['model'].split('/')[-1]
    for cand in (os.path.join(HERE, base + '.zip'), os.path.join(HERE, base)):
        if os.path.exists(cand):
            print('model not cached; importing %s' % os.path.basename(cand))
            return hub.import_local(cand, s['model'], s['cache'])
    raise SystemExit(
        '%s is not downloaded yet. Run once with a connection:\n'
        '    MODE = "download"   (or: python hf_run.py download)'
        % s['model'])


# 'auto' picks lowmem whenever BF16 is widened by BNNSCopy: on an iPhone 16
# Pro, SmolLM2-135M then decoded at ~88 tok/s in lowmem against ~79 in
# float32, at 180 MB footprint instead of 683 MB. Without BNNSCopy (NumPy
# widening, e.g. Linux or an old iOS) it picks float32 up to this many MB
# of float32 weights: there float32 decoded 1.5x faster than lowmem at 135M
# params (540 MB), and no faster at 360M (1.45 GB) while using ~1.8 GB more
# memory.
AUTO_FLOAT32_MAX_MB = 1000
# Explicit float32 above this is refused on iOS: the system would kill
# Pythonista without any Python-level MemoryError.
IOS_FLOAT32_MAX_MB = 2500


def resolve_weights(mode, cfg, have_bnns_copy=None):
    if mode != 'auto':
        return mode
    if have_bnns_copy is None:
        from tinygpt import backend
        have_bnns_copy = backend.HAVE_BNNS_COPY
    if have_bnns_copy:
        return 'lowmem'
    return 'float32' if cfg.n_params() * 4 <= AUTO_FLOAT32_MAX_MB << 20 \
        else 'lowmem'


def load(s, quiet=False):
    from tinygpt import backend
    from tinygpt.hf.llama import UnsupportedModel
    from tinygpt.hf.models import config_from_dict, model_from_dir
    from tinygpt.hf.safetensors import SafetensorsError
    from tinygpt.hf.tokenizer import BPETokenizer, ChatFormat
    import json
    path = locate(s)
    with open(os.path.join(path, 'config.json'), 'rb') as f:
        raw_cfg = json.loads(f.read().decode('utf-8'))
    try:
        cfg = config_from_dict(raw_cfg)
    except UnsupportedModel as e:
        raise SystemExit('unsupported model: %s' % e)
    ctx = min(s['ctx'], cfg.max_context)
    weights = resolve_weights(s['weights'], cfg)
    est = (cfg.n_params() * 4 if weights == 'float32' else 8 << 20) \
        + cfg.kv_bytes(ctx)
    if (weights == 'float32' and sys.platform == 'ios'
            and est > IOS_FLOAT32_MAX_MB << 20):
        raise SystemExit(
            'float32 weights for this model need about %d MB, more than '
            'iOS lets Pythonista use. Use WEIGHTS = "lowmem" (or "auto").'
            % (est >> 20))
    if not quiet:
        print('%s  (%s params)' % (s['model'], '{:,}'.format(cfg.n_params())))
        print(cfg.describe())
        print('weights %s%s, context %d, needs about %d MB; Accelerate %s'
              % (weights, ' (auto)' if s['weights'] == 'auto' else '', ctx,
                 est >> 20,
                 'yes' if backend.HAVE_ACCELERATE else 'no (NumPy only)'))
    t0 = time.time()
    try:
        model = model_from_dir(path, ctx_limit=ctx, weights=weights)
    except MemoryError:
        raise SystemExit(
            'out of memory loading the weights (needs about %d MB).\n'
            'Try WEIGHTS = "lowmem" and/or a smaller CONTEXT_LIMIT.'
            % (est >> 20))
    except UnsupportedModel as e:
        raise SystemExit('unsupported model: %s' % e)
    except SafetensorsError as e:
        raise SystemExit('cannot load the weights: %s' % e)
    t_load = time.time() - t0
    tok = BPETokenizer.from_dir(path)
    try:
        chat = ChatFormat(tok.chat_template, tok.special_tokens)
    except ValueError as e:
        chat = None
        chat_note = str(e)
    else:
        chat_note = {'jinja2': "the model's template (jinja2)",
                     'chatml': 'built-in ChatML (no jinja2)'}[chat.engine]
    stops = set(cfg.eos_ids)
    model.generation_config = {}
    gen_cfg = os.path.join(path, 'generation_config.json')
    if os.path.exists(gen_cfg):
        with open(gen_cfg, 'rb') as f:
            model.generation_config = json.loads(f.read().decode('utf-8'))
        e = model.generation_config.get('eos_token_id')
        stops.update(e if isinstance(e, list) else ([e] if e is not None
                                                    else []))
    if tok.eos_id is not None:
        stops.add(tok.eos_id)
    # A ChatML turn ends at <|im_end|>. Instruct models declare it as an
    # end token; base models (Qwen2.5-Coder-1.5B) do not, which is how a
    # base model in chat mode is recognised. It may still emit it.
    model.base_model_chat = False
    if chat is not None and '<|im_end|>' in (tok.chat_template or ''):
        if '<|im_end|>' in tok.token_to_id:
            end = tok.token_to_id['<|im_end|>']
            model.base_model_chat = end not in stops
            stops.add(end)
    if not quiet:
        pk = peak_rss_mb()
        print('loaded in %.1fs, memory now %s (peak so far %s MB)'
              % (t_load, memory_line(), '%.0f' % pk if pk else '?'))
        print('chat format: %s' % chat_note)
    return model, tok, chat, stops, t_load


# Used when neither the settings nor generation_config.json say otherwise.
DEFAULT_SAMPLING = {'temperature': 0.2, 'top_k': 0, 'top_p': 0.9,
                    'repetition_penalty': 1.0}


def sampling_settings(s, gen):
    '''Resolve sampling parameters: explicit settings, then the model's
    generation_config.json (`gen`), then DEFAULT_SAMPLING.

    Follows transformers: a config with do_sample true samples with its
    temperature/top_k/top_p (transformers' own defaults 1.0/50/1.0 for any
    it leaves out), do_sample false means greedy, and repetition_penalty
    applies either way. Returns (dict, where each value came from).
    '''
    out = dict(DEFAULT_SAMPLING)
    src = 'defaults'
    if gen.get('do_sample') is True:
        out.update(temperature=float(gen.get('temperature', 1.0)),
                   top_k=int(gen.get('top_k', 50)),
                   top_p=float(gen.get('top_p', 1.0)))
        src = 'generation_config.json'
    elif gen.get('do_sample') is False:
        out.update(temperature=0.0)
        src = 'generation_config.json'
    if 'repetition_penalty' in gen:
        out['repetition_penalty'] = float(gen['repetition_penalty'])
        src = 'generation_config.json'
    explicit = False
    for key, name in (('temp', 'temperature'), ('top_k', 'top_k'),
                      ('top_p', 'top_p'), ('rep', 'repetition_penalty')):
        if s.get(key) is not None:
            out[name] = s[key]
            explicit = True
    if explicit:
        src = 'settings' if src == 'defaults' else src + ' + settings'
    return out, src


def make_sampler(s, model=None):
    from tinygpt.hf.llama import Sampler
    gen = getattr(model, 'generation_config', None) or {}
    p, src = sampling_settings(s, gen)
    print('sampling: %s (%s)' % (
        'greedy' if p['temperature'] <= 0 else
        'temperature %g, top_k %d, top_p %g' % (p['temperature'], p['top_k'],
                                               p['top_p'])
        + ('' if p['repetition_penalty'] == 1.0 else
           ', repetition penalty %g' % p['repetition_penalty']), src))
    return Sampler(p['temperature'], p['top_k'], p['top_p'],
                   p['repetition_penalty'], seed=s['seed'])


def warn_base_model(model, s):
    if getattr(model, 'base_model_chat', False):
        print('note: %s looks like a base model -- its config does not end '
              'turns at <|im_end|> -- so it will tend to continue the text '
              'rather than answer. Use MODE = "complete", or the chat '
              'version of the model (often named ...-Instruct).'
              % s['model'])


def run_generation(model, tok, cache, ids, s, stops, sampler):
    from tinygpt.hf.llama import generate
    st = tok.stream()

    def show(i):
        sys.stdout.write(st.feed(i))
        sys.stdout.flush()
    out, reason, stats = generate(model, cache, ids, s['max_new'], sampler,
                                  stops, show)
    sys.stdout.write(st.flush() + '\n')
    rate = stats['new_tokens'] / stats['decode_s'] if stats['decode_s'] \
        else 0.0
    note = {'context': '  [context limit reached]',
            'length': '  [max tokens]'}.get(reason, '')
    print('[%d prompt tokens in %.2fs, %d new at %.1f tok/s]%s'
          % (stats['prompt_tokens'], stats['prefill_s'],
             stats['new_tokens'], rate, note))
    return out, reason


def messages_for(s, history):
    msgs = []
    if s['system']:
        msgs.append({'role': 'system', 'content': s['system']})
    return msgs + history


def fit_history(model, tok, chat, s, history):
    '''Drop the oldest exchanges until prompt + answer fit the context.'''
    while True:
        ids = tok.encode(chat.render(messages_for(s, history)))
        if len(ids) + min(s['max_new'], 64) <= model.ctx_limit \
                or len(history) <= 1:
            return ids, history
        history = history[2:]


# ------------------------------------------------------------------ modes

def mode_download(s):
    from tinygpt.hf import hub
    print('downloading %s into %s' % (s['model'],
                                      s['cache'] or hub.default_cache_dir()))
    path = hub.snapshot_download(s['model'], s['revision'], s['cache'])
    print('ready: %s' % path)
    print('You can now go offline and use MODE = "chat".')


def mode_import(s):
    from tinygpt.hf import hub
    src = s['source']
    if not src:
        raise SystemExit('import needs a zip file or directory')
    if not os.path.isabs(src):
        src = os.path.join(os.getcwd(), src) if os.path.exists(src) \
            else os.path.join(HERE, src)
    path = hub.import_local(src, s['model'], s['cache'])
    print('imported into %s' % path)


def mode_info(s):
    from tinygpt.hf import hub
    path = hub.find_local(s['model'], s['cache'], s['revision'])
    if not path:
        print('%s: not cached' % s['model'])
        return
    print('%s: %s' % (s['model'], path))
    print('verifying hashes...')
    for name in hub.verify_snapshot(path):
        print('  ok  %-28s %10d bytes'
              % (name, os.path.getsize(os.path.join(path, name))))


def mode_ask(s, raw=False):
    model, tok, chat, stops, _ = load(s)
    if raw or chat is None:
        ids = tok.encode(s['prompt'])
        sys.stdout.write(s['prompt'])
    else:
        warn_base_model(model, s)
        ids = tok.encode(chat.render(messages_for(
            s, [{'role': 'user', 'content': s['prompt']}])))
        print('\n> %s\n' % s['prompt'])
    run_generation(model, tok, model.new_cache(), ids, s, stops,
                   make_sampler(s, model))


def mode_chat(s):
    model, tok, chat, stops, _ = load(s)
    if chat is None:
        raise SystemExit('this model has no chat template; use complete')
    print('\nchat with %s -- empty line or /quit to stop, /reset to clear'
          % s['model'].split('/')[-1])
    warn_base_model(model, s)
    cache = model.new_cache()
    sampler = make_sampler(s, model)
    history = []
    while True:
        try:
            line = input('\nyou> ')
        except (EOFError, KeyboardInterrupt):
            break
        line = line.strip()
        if not line or line == '/quit':
            break
        if line == '/reset':
            history = []
            cache = model.new_cache()
            print('(cleared)')
            continue
        history.append({'role': 'user', 'content': line})
        try:
            ids, history = fit_history(model, tok, chat, s, history)
        except ValueError as e:      # the template rejected the messages
            print(e)
            history.pop()
            continue
        sys.stdout.write('\nbot> ')
        out, _ = run_generation(model, tok, cache, ids, s, stops, sampler)
        history.append({'role': 'assistant',
                        'content': tok.decode(out, skip_special=True)})


def bench(s, out=print):
    '''Load time, first-token latency, decode speed, memory.'''
    import gc
    from tinygpt import backend
    from tinygpt.hf.llama import Sampler, generate
    gc.collect()                    # let an earlier bench's model go first
    mem0 = memory_line()
    model, tok, chat, stops, t_load = load(s, quiet=True)
    mem1 = memory_line()
    out('model        %s (%s)' % (s['model'], model.weights))
    out('platform     %s, python %s, numpy %s'
        % (sys.platform, sys.version.split()[0], __import__('numpy')
           .__version__))
    out('accelerate   %s (sgemv %s)' % (backend.HAVE_ACCELERATE,
                                       backend.HAVE_SGEMV))
    out('load         %.2f s' % t_load)
    if model.weights == 'float32':
        out('backend choice per weight shape (timed at load):')
        out(model.mv.report())
    prompt = chat.render([{'role': 'user', 'content':
                           'Explain in a few sentences how rainbows form, '
                           'and why the colours always appear in the same '
                           'order.'}]) if chat else 'Rainbows form when'
    ids = tok.encode(prompt)
    for run in range(2):
        cache = model.new_cache()
        res, reason, st = generate(model, cache, ids, 64, Sampler(0.0),
                                   stop_ids=(), on_token=None)
        out('run %d        prompt %d tok -> first token %.3f s; '
            '%d tokens at %.2f tok/s (%.1f ms/token)'
            % (run + 1, len(ids), st['prefill_s'], st['new_tokens'],
               st['new_tokens'] / st['decode_s'],
               1e3 * st['decode_s'] / st['new_tokens']))
    cache = model.new_cache()
    t0 = time.time()
    for t in ids:
        model.step(t, cache)
    out('token-by-token prompt (no prefill): %.3f s' % (time.time() - t0))
    pk = peak_rss_mb()
    out('memory       before load %s; after load %s; after generating %s'
        % (mem0, mem1, memory_line()))
    out('peak memory  %s MB (this process so far)'
        % ('%.0f' % pk if pk else 'n/a'))
    out('greedy text  %r' % tok.decode(res, skip_special=True)[:200])


class _Tee(object):
    def __init__(self, path):
        self.f = open(path, 'w')

    def __call__(self, line=''):
        print(line)
        self.f.write(line + '\n')
        self.f.flush()


def report_path(s):
    p = s['report']
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def mode_bench(s):
    tee = _Tee(report_path(s))
    tee('== bench %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    bench(s, tee)
    print('\nreport written to %s' % report_path(s))


def mode_selftest(s):
    import io
    import unittest
    tee = _Tee(report_path(s))
    tee('== selftest %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    tee('(peak memory only ever grows within one process, so the lowmem '
        'bench runs first and the tests last)')
    tee('')
    for w in ('lowmem', 'float32'):
        try:
            bench(dict(s, weights=w), tee)
        except SystemExit as e:
            tee('bench %s skipped: %s' % (w, e))
        tee('')
    buf = io.StringIO()
    suite = unittest.defaultTestLoader.discover(
        os.path.join(HERE, 'tests'), pattern='test_hf*.py', top_level_dir=HERE)
    res = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)
    for line in buf.getvalue().splitlines():
        tee(line)
    tee('')
    tee('tests: %s' % ('PASS' if res.wasSuccessful() else 'FAIL'))
    print('\nreport written to %s' % report_path(s))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    s = settings_from_argv(argv) if argv else settings_from_globals()
    from tinygpt.hf import hub
    s['model'] = hub.normalize_repo_id(s['model'])
    fn = {'download': mode_download, 'import': mode_import,
          'info': mode_info, 'chat': mode_chat,
          'ask': mode_ask, 'complete': lambda s: mode_ask(s, raw=True),
          'bench': mode_bench, 'selftest': mode_selftest}.get(s['mode'])
    if fn is None:
        raise SystemExit('unknown MODE %r' % s['mode'])
    try:
        fn(s)
    except hub.HubError as e:
        raise SystemExit('error: %s' % e)


if __name__ == '__main__':
    main()
