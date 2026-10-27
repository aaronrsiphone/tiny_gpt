# coding: utf-8
'''
Describe a Hugging Face model repository the way tinygpt/hf/ needs to see
it, and say what in it is not supported yet. Step 1 of
tinygpt/hf/ADDING_A_MODEL.md.

    python tools/hf_probe.py Qwen/Qwen2.5-Coder-1.5B          # cached
    python tools/hf_probe.py path/to/snapshot_dir
    python tools/hf_probe.py --remote LiquidAI/LFM3-1.7B       # no download

--remote reads only metadata: config, tokenizer and template files, and the
safetensors header through an HTTP Range request (a few hundred KB, not the
weights). Python files in the repository are listed but never imported or
run. Runs in Pythonista too (stdlib + requests; numpy for the support
check).
'''

import collections
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

TEXT_FILES = ('config.json', 'generation_config.json', 'tokenizer.json',
              'tokenizer_config.json', 'special_tokens_map.json',
              'chat_template.jinja', 'model.safetensors.index.json')

# config.json keys tinygpt/hf/llama.py reads or knowingly ignores. Anything
# else is printed, since an unfamiliar key often means an unfamiliar feature.
KNOWN_KEYS = set('''
architectures model_type hidden_size intermediate_size num_hidden_layers
num_attention_heads num_key_value_heads head_dim vocab_size rms_norm_eps
rope_theta rope_scaling rope_parameters max_position_embeddings
tie_word_embeddings attention_bias mlp_bias bos_token_id eos_token_id
pad_token_id hidden_act pretraining_tp sliding_window use_sliding_window
max_window_layers layer_types torch_dtype dtype transformers_version
use_cache initializer_range attention_dropout rope_interleaved
is_llama_config transformers.js_config _name_or_path
'''.split())


# ------------------------------------------------------------------ sources

class Local(object):

    def __init__(self, path):
        self.path = path
        self.files = {n: os.path.getsize(os.path.join(path, n))
                      for n in os.listdir(path)
                      if os.path.isfile(os.path.join(path, n))}

    def text(self, name):
        p = os.path.join(self.path, name)
        if not os.path.exists(p):
            return None
        with open(p, 'rb') as f:
            return f.read().decode('utf-8')

    def st_header(self, name):
        with open(os.path.join(self.path, name), 'rb') as f:
            (n,) = struct.unpack('<Q', f.read(8))
            return json.loads(f.read(n).decode('utf-8'))


class Remote(object):

    def __init__(self, repo, revision='main'):
        from tinygpt.hf import hub
        self.repo = hub.check_repo_id(hub.normalize_repo_id(repo))
        self.s = hub._session()
        info = hub.repo_info(self.repo, revision, self.s)
        self.sha = info['sha']
        self.files = {n: s.get('size') for n, s in info['files'].items()}
        self.base = '%s/%s/resolve/%s/' % (hub.ENDPOINT, self.repo, self.sha)

    def text(self, name):
        if name not in self.files:
            return None
        r = self.s.get(self.base + name, timeout=(15, 60))
        r.raise_for_status()
        return r.content.decode('utf-8')

    def _range(self, name, a, b):
        r = self.s.get(self.base + name, headers={'Range': 'bytes=%d-%d'
                                                  % (a, b)}, timeout=(15, 60))
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError('server ignored the Range request')
        return r.content

    def st_header(self, name):
        (n,) = struct.unpack('<Q', self._range(name, 0, 7))
        if n > 64 << 20:
            raise RuntimeError('implausible header length %d' % n)
        return json.loads(self._range(name, 8, 7 + n).decode('utf-8'))


# ------------------------------------------------------------------ report

def section(title):
    print('\n== %s' % title)


def show_files(src):
    section('files')
    for n in sorted(src.files):
        size = src.files[n]
        note = ''
        if n.endswith('.py'):
            note = '  <- repository code: read as a spec, never run'
        elif n.endswith(('.bin', '.pt', '.pth', '.ckpt', '.pkl')):
            note = '  <- pickle: never loaded'
        print('  %-40s %14s%s' % (n, '{:,}'.format(size) if size else '?',
                                  note))


def show_config(cfg):
    section('config.json')
    for k in ('model_type', 'architectures', 'hidden_size',
              'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
              'num_key_value_heads', 'head_dim', 'vocab_size',
              'tie_word_embeddings', 'hidden_act', 'max_position_embeddings',
              'sliding_window', 'rope_theta', 'rope_scaling',
              'rope_parameters'):
        if k in cfg:
            print('  %-24s %s' % (k, json.dumps(cfg[k])))
    if cfg.get('auto_map'):
        print('  auto_map                 %s  <- custom modeling code; check '
              'whether transformers has it natively' % json.dumps(
                  cfg['auto_map']))
    lt = cfg.get('layer_types')
    if lt:
        print('  layer_types              %s' % dict(collections.Counter(lt)))
        print('                           %s' % ' '.join(
            '%d:%s' % (i, t[:4]) for i, t in enumerate(lt)))
    extra = sorted(set(cfg) - KNOWN_KEYS - {'auto_map'})
    if extra:
        print('  not read by llama.py:    %s' % ', '.join(
            '%s=%s' % (k, json.dumps(cfg[k])) for k in extra))


def show_tensors(src):
    section('tensors (layer index replaced by N; which layers have each)')
    names = [n for n in src.files if n.endswith('.safetensors')]
    if not names:
        print('  no safetensors files')
        return 0, 0
    header = {}
    for n in names:
        h = src.st_header(n)
        h.pop('__metadata__', None)
        header.update(h)
    groups = collections.OrderedDict()
    dtypes = collections.Counter()
    params = 0
    for k in sorted(header):
        t = header[k]
        dtypes[t['dtype']] += 1
        n = 1
        for d in t['shape']:
            n *= d
        params += n
        m = re.search(r'\.(\d+)\.', k)
        key = (re.sub(r'\.\d+\.', '.N.', k, count=1), t['dtype'],
               tuple(t['shape']))
        groups.setdefault(key, []).append(int(m.group(1)) if m else None)
    for (pat, dt, shape), layers in groups.items():
        where = ''
        if layers[0] is not None:
            where = ' layers ' + _ranges(sorted(layers))
        print('  %-52s %-5s %-16s%s' % (pat, dt, list(shape), where))
    size = sum(src.files[n] or 0 for n in names)
    print('  %d tensors, dtypes %s, %s params; files %s bytes'
          % (len(header), dict(dtypes), '{:,}'.format(params),
             '{:,}'.format(size)))
    return params, size


def _ranges(xs):
    out, a = [], None
    for i, x in enumerate(xs):
        if a is None:
            a = x
        if i + 1 == len(xs) or xs[i + 1] != x + 1:
            out.append(str(a) if a == x else '%d-%d' % (a, x))
            a = None
    s = ','.join(out)
    return s if len(s) < 60 else s[:57] + '...'


def show_tokenizer(spec, tcfg):
    section('tokenizer.json')
    for k in ('normalizer', 'pre_tokenizer', 'decoder', 'post_processor'):
        print('  %-15s %s' % (k, json.dumps(spec.get(k))[:300]))
    m = spec.get('model', {})
    print('  model           %s, vocab %d, merges %d'
          % (json.dumps({k: v for k, v in m.items()
                         if k not in ('vocab', 'merges')}),
             len(m.get('vocab') or ()), len(m.get('merges') or ())))
    flags = collections.Counter(
        tuple(k for k in ('lstrip', 'rstrip', 'single_word', 'normalized',
                          'special') if a.get(k))
        for a in spec.get('added_tokens', ()))
    print('  added tokens    %d, flag combinations %s'
          % (sum(flags.values()), dict(flags)))
    for k in ('bos_token', 'eos_token', 'pad_token', 'unk_token'):
        if tcfg.get(k):
            print('  %-15s %s' % (k, json.dumps(tcfg[k])))
    try:
        from tinygpt.hf.tokenizer import BPETokenizer
        BPETokenizer(spec, tcfg)
        print('  support: OK (tinygpt/hf/tokenizer.py loads it)')
    except Exception as e:
        print('  support: NOT YET -- %s: %s' % (e.__class__.__name__, e))


def show_template(tpl, tcfg, src_name):
    section('chat template (%s)' % src_name)
    if not tpl:
        print('  none: completion only')
        return
    checks = (('bos_token', r'\bbos_token\b'),
              ('strftime_now', r'\bstrftime_now\b'),
              ('{% generation %}', r'\{%-?\s*generation\b'),
              ('raise_exception', r'\braise_exception\b'),
              ('tools', r'\btools\b'),
              ('enable_thinking', r'\benable_thinking\b'),
              ('<think>', r'<think>'), ('tojson', r'\btojson\b'),
              ('namespace', r'\bnamespace\('))
    feats = [name for name, rx in checks if re.search(rx, tpl)]
    print('  %d chars; uses: %s' % (len(tpl), ', '.join(feats) or '-'))
    try:
        from tinygpt.hf.tokenizer import BPETokenizer, ChatFormat  # noqa
        special = {k: (v.get('content') if isinstance(v, dict) else v)
                   for k, v in tcfg.items()
                   if k.endswith('_token') and v}
        cf = ChatFormat(tpl, special)
        out = cf.render([{'role': 'user', 'content': 'Hi'}])
        print('  render (%s): %r' % (cf.engine, out[-160:]))
    except Exception as e:
        print('  render: FAILED -- %s: %s' % (e.__class__.__name__, e))


def show_support(cfg, params, size):
    section('support check')
    try:
        from tinygpt.hf.llama import LlamaConfig
        c = LlamaConfig(cfg)
        print('  model: OK -- %s' % c.describe())
    except Exception as e:
        print('  model: NOT YET -- %s: %s' % (e.__class__.__name__, e))
    if params:
        print('  float32 weights %.2f GB, lowmem file %.2f GB'
              % (params * 4 / 1e9, size / 1e9))


def main(argv):
    remote = '--remote' in argv
    args = [a for a in argv if a != '--remote']
    if len(args) != 1:
        raise SystemExit(__doc__)
    target = args[0]
    if remote:
        src = Remote(target)
        print('%s @ %s' % (src.repo, src.sha))
    elif os.path.isdir(target):
        src = Local(target)
        print(target)
    else:
        from tinygpt.hf import hub
        path = hub.find_local(hub.normalize_repo_id(target))
        if not path:
            raise SystemExit('%s is not cached; use --remote' % target)
        src = Local(path)
        print(path)

    def load_json(name):
        t = src.text(name)
        return json.loads(t) if t else {}

    cfg = load_json('config.json')
    tcfg = load_json('tokenizer_config.json')
    show_files(src)
    show_config(cfg)
    gen = load_json('generation_config.json')
    if gen:
        section('generation_config.json')
        print('  %s' % json.dumps(gen))
    params, size = show_tensors(src)
    spec = load_json('tokenizer.json')
    if spec:
        show_tokenizer(spec, tcfg)
    tpl = src.text('chat_template.jinja')
    show_template(tpl or tcfg.get('chat_template'), tcfg,
                  'chat_template.jinja' if tpl else 'tokenizer_config.json')
    show_support(cfg, params, size)


if __name__ == '__main__':
    main(sys.argv[1:])
