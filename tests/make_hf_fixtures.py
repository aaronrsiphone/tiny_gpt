# coding: utf-8
'''
Regenerate the reference fixtures in tests/fixtures/ from the official
Hugging Face libraries. The tests themselves need only NumPy; this script
is the one place the reference implementations are used:

    pip install torch transformers tokenizers safetensors
    python tests/make_hf_fixtures.py [--lfm3=LFM3-1.7B dir]
                                     [SmolLM2-135M-Instruct dir
                                      [Qwen2.5-Coder-1.5B dir
                                       [Qwen2.5-Coder-1.5B-Instruct dir]]]
    python tests/make_hf_fixtures.py --only-lfm3 [--lfm3=dir]

Without model directories only the synthetic fixtures are rebuilt.

Fixtures, all JSON (weights base64-encoded) so they need neither git-lfs
nor the *.npz/*.safetensors patterns this repo ignores or LFS-tracks:

  tiny_llama_*.json   random tiny Llama models written by `safetensors`,
                      with transformers' float32 logits for a sequence
  safetensors_dtypes.json
                      F32/F16/BF16 tensors written by `safetensors`
  bpe_tiny.json       a small byte-level BPE trained by `tokenizers` with
                      SmolLM2's pipeline (Digits + ByteLevel), plus
                      reference encodings
  bpe_tiny_qwen.json  the same with Qwen2's pipeline (NFC, Split regex,
                      ByteLevel without its own regex)
  tiny_qwen2_bf16.json a random tiny Qwen2 (q/k/v bias) with logits
  smollm2_tokens.json SmolLM2 encodings and chat-template renderings
  smollm2_logits.json SmolLM2 float32 reference logits / greedy decode
  qwen_tokens.json,   the same for Qwen2.5-Coder-1.5B
  qwen_logits.json
  qwen_instruct_chat.json  Qwen2.5-Coder-1.5B-Instruct answering a chat
                      message, greedy with its repetition penalty
  tiny_lfm3_via_lfm2.json
                      a random tiny LFM3 reduced to what transformers'
                      built-in LFM2 computes, with LFM2's logits
  tiny_lfm3.json      a random tiny LFM3 with every LFM3 feature, with
                      logits from lfm3_reference_forward() below
  lfm3_tokens.json    LFM3-1.7B encodings (ignore_merges, digit groups)
  chat_templates.json real chat templates from several model repos,
                      rendered by transformers' own renderer (fetches the
                      templates from the Hub; skipped when offline)
'''

import base64
import io
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, 'fixtures')


def b64(b):
    return base64.b64encode(b).decode('ascii')


def dump(name, obj):
    path = os.path.join(FIX, name)
    with open(path, 'w') as f:
        json.dump(obj, f, separators=(',', ':'), sort_keys=True)
    print('wrote %s (%d KB)' % (os.path.relpath(path), os.path.getsize(path)
                                >> 10))


# ------------------------------------------------------------------ text

TEXTS = [
    'Hello, world!', ' leading space', 'trailing  ',
    'multiple   spaces\n\n\nnewlines\t\ttabs',
    'Numbers 12345 and 3.14159, -42, 1e10, ٣٤ ½ Ⅻ',
    "I'm you're they've we'll he'd it's IT'S",
    '<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n',
    '<|im_start|><|im_end|><|endoftext|>x<|im_start',
    '日本語のテキスト、中文。'
    '한국어',
    'emoji \U0001F600\U0001F44D\U0001F3FD \U0001F468‍\U0001F469‍'
    '\U0001F467',
    'Ünïcödé café naïve',
    ' nbsp ls　ideo\x85nel\x1cfs​zw',
    'def f(x):\n    return x**2  # comment\n',
    'https://huggingface.co/HuggingFaceTB?x=1&y=2', '   ', '', '\n',
    'a' * 200, 'x\r\ny', 'Ελληνικά '
    'русский עברית',
    '$$\\int_0^1 x^2 dx = \\frac{1}{3}$$',
    'The quick brown fox jumps over the lazy dog. ' * 3,
]


def random_texts(n, seed=0):
    rng = random.Random(seed)
    pool = ([chr(c) for c in list(range(32, 127)) * 20
             + list(range(0xa0, 0x3000, 7))
             + [0x1F600, 0x1F44D, 0x10D40, 0x1C89, 0x20000, 0x85, 0x1c,
                0xa0, 0x2028, 0x3000, 0xFFFD]]
            + [' ', ' ', '\n', '  ', '<|im_end|>', "'s", '123'])
    return [''.join(rng.choice(pool) for _ in range(rng.randint(1, 60)))
            for _ in range(n)]


def readme_text():
    with open(os.path.join(HERE, '..', 'README.md'), 'rb') as f:
        return f.read().decode('utf-8')


# ------------------------------------------------------------ tiny llama

def tiny_llama(name, cfg_kw, dtype, seed, seq_len=24, kind='llama'):
    import torch
    from safetensors.torch import save
    import transformers
    Config, Model = {
        'llama': (transformers.LlamaConfig, transformers.LlamaForCausalLM),
        'qwen2': (transformers.Qwen2Config, transformers.Qwen2ForCausalLM),
    }[kind]

    torch.manual_seed(seed)
    cfg = Config(**cfg_kw)
    model = Model(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
        for n, p in model.named_parameters():
            if 'norm' in n:
                p.add_(1.0)
    tdt = {'BF16': torch.bfloat16, 'F16': torch.float16,
           'F32': torch.float32}[dtype]
    sd = {k: v.detach().to(tdt).contiguous()
          for k, v in model.state_dict().items()}
    if cfg.tie_word_embeddings:
        sd.pop('lm_head.weight', None)
    blob = save(sd, metadata={'format': 'pt'})
    # Reference: the stored (rounded) weights, computed in float32.
    model.load_state_dict({k: v.float() for k, v in sd.items()},
                          strict=False)
    if cfg.tie_word_embeddings:
        model.tie_weights()
    model = model.float().eval()
    g = torch.Generator().manual_seed(seed + 1)
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len), generator=g)
    with torch.no_grad():
        logits = model(ids).logits[0].numpy().astype(np.float32)
    cfg_json = json.loads(cfg.to_json_string())
    dump(name, {'config': cfg_json, 'safetensors': b64(blob),
                'ids': ids[0].tolist(), 'logits': b64(logits.tobytes()),
                'logits_shape': list(logits.shape)})


def safetensors_dtypes():
    import torch
    from safetensors.torch import save
    torch.manual_seed(0)
    x = torch.randn(7, 5) * 10
    special = torch.tensor([0.0, -0.0, 1.0, -2.5, 65504.0, 1e-8, 3.0e38,
                            float('inf'), float('-inf'), 1.0 / 3])
    t = {'f32': x.float(), 'f16': x.half(), 'bf16': x.bfloat16(),
         'bf16_special': special.bfloat16(), 'scalar': torch.tensor(3.5),
         'empty': torch.zeros(0, 4)}
    ref = {k: v.float().numpy().tolist() for k, v in t.items()}
    ref['bf16_special'] = [repr(float(v)) for v in t['bf16_special'].float()]
    dump('safetensors_dtypes.json',
         {'safetensors': b64(save(t)), 'values': ref,
          'shapes': {k: list(v.shape) for k, v in t.items()}})


# -------------------------------------------------------------- tokenizer

def bpe_tiny():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers import trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(
        vocab_size=600, special_tokens=['<|endoftext|>', '<|im_start|>',
                                        '<|im_end|>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([readme_text()] * 2 + TEXTS, trainer=tr)
    spec = json.loads(tok.to_str())
    texts = TEXTS + random_texts(300, seed=1)
    dump('bpe_tiny.json', {
        'tokenizer': spec, 'texts': texts,
        'ids': [tok.encode(t).ids for t in texts],
        'decoded': [tok.decode(tok.encode(t).ids, skip_special_tokens=False)
                    for t in texts]})


def chat_ids(at, msgs):
    ids = at.apply_chat_template(msgs, tokenize=True,
                                 add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, 'keys'):
        ids = ids['input_ids']
    return [int(i) for i in ids]


QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
              r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

NFC_TEXTS = ['é café Å Å 각',
             "I'M HE'LL She'S it'Ll ſ 'ſ 'K",
             'x̣́ x̣́ क़ ﬁ ①',
             'a࣍́ á࣍ ᫁é',
             'def f(x):\r\n    return x  # été\n\n\n']


def bpe_tiny_qwen():
    from tokenizers import (Regex, Tokenizer, decoders, models, normalizers,
                            pre_tokenizers, trainers)
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(
        vocab_size=700, special_tokens=['<|endoftext|>', '<|im_start|>',
                                        '<|im_end|>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([readme_text()] * 2 + TEXTS + NFC_TEXTS,
                            trainer=tr)
    spec = json.loads(tok.to_str())
    texts = TEXTS + NFC_TEXTS + random_texts(300, seed=3)
    dump('bpe_tiny_qwen.json', {
        'tokenizer': spec, 'texts': texts,
        'ids': [tok.encode(t).ids for t in texts],
        'decoded': [tok.decode(tok.encode(t).ids, skip_special_tokens=False)
                    for t in texts]})


CODE_PROMPT = ('def quicksort(arr):\n'
               '    """Sort a list of numbers in ascending order."""\n')


def qwen_tokens(model_dir):
    from tokenizers import Tokenizer
    ref = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer.json'))
    texts = TEXTS + NFC_TEXTS + random_texts(500, seed=4)
    for f in ('make_hf_fixtures.py', 'hf_common.py'):
        with open(os.path.join(HERE, f), 'rb') as fh:
            texts.append(fh.read().decode('utf-8'))
    dump('qwen_tokens.json', {'texts': texts,
                              'ids': [ref.encode(t).ids for t in texts]})


def qwen_logits(model_dir):
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    ref = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer.json'))
    ids = ref.encode(CODE_PROMPT).ids
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32).eval()
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x).logits[0].numpy()
        gen = model.generate(x, max_new_tokens=40, do_sample=False,
                             eos_token_id=151643,
                             pad_token_id=151643)[0].tolist()
    sel = list(range(0, logits.shape[1], 211))
    dump('qwen_logits.json', {
        'prompt': CODE_PROMPT, 'ids': ids, 'logits_cols': sel,
        'logits_sel': b64(np.ascontiguousarray(
            logits[:, sel], dtype=np.float32).tobytes()),
        'last_logits': b64(logits[-1].astype(np.float32).tobytes()),
        'greedy': gen[len(ids):], 'greedy_text': ref.decode(gen[len(ids):]),
    })


COUNT_WORDS = (
    'Write a Python function called `count_words` that takes a string and '
    'returns a dictionary mapping each word to how many times it appears.'
    '\n\nRules:\n- Treat words case-insensitively ("The" and "the" are the '
    'same word).\n- Ignore punctuation: . , ! ? ; :\n- Split on whitespace.'
    '\n- Use only the Python standard library.\n\nExample:\n'
    'count_words("The cat sat. The cat ran!")\nshould return:\n'
    '{"the": 2, "cat": 2, "sat": 1, "ran": 1}\n\n'
    'Return only the code, with no explanation.')


def qwen_instruct_chat(model_dir):
    # Greedy decoding with the model's own repetition_penalty (1.1, from
    # its generation_config.json), which transformers applies even when
    # not sampling.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    at = AutoTokenizer.from_pretrained(model_dir)
    msgs = [{'role': 'user', 'content': COUNT_WORDS}]
    ids = chat_ids(at, msgs)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        gen = model.generate(torch.tensor([ids]), max_new_tokens=48,
                             do_sample=False, repetition_penalty=1.1,
                             pad_token_id=151643)[0].tolist()
    dump('qwen_instruct_chat.json', {
        'messages': msgs, 'ids': ids, 'repetition_penalty': 1.1,
        'greedy': gen[len(ids):], 'greedy_text': at.decode(gen[len(ids):])})


def smollm2_tokens(model_dir):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    ref = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer.json'))
    texts = TEXTS + random_texts(500, seed=2)
    readme = readme_text()
    chats = [
        [{'role': 'user', 'content': 'What is the capital of France?'}],
        [{'role': 'system', 'content': 'You are terse.'},
         {'role': 'user', 'content': 'Hi'}],
        [{'role': 'user', 'content': 'Write a haiku about rain.'},
         {'role': 'assistant', 'content': 'Soft drops on the roof\n...'},
         {'role': 'user', 'content': 'Another, about 2 cats \U0001F431'}],
    ]
    at = AutoTokenizer.from_pretrained(model_dir)
    dump('smollm2_tokens.json', {
        'texts': texts, 'ids': [ref.encode(t).ids for t in texts],
        'readme_text': readme,
        'readme_ids': ref.encode(readme).ids,
        'chats': chats,
        'chat_text': [at.apply_chat_template(c, tokenize=False,
                                             add_generation_prompt=True)
                      for c in chats],
        'chat_ids': [chat_ids(at, c) for c in chats],
    })


def smollm2_logits(model_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    at = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32).eval()
    msgs = [{'role': 'user', 'content': 'What is the capital of France?'}]
    ids = chat_ids(at, msgs)
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x).logits[0].numpy()
        gen = model.generate(x, max_new_tokens=48, do_sample=False,
                             eos_token_id=2, pad_token_id=2)[0].tolist()
    sel = list(range(0, 49152, 97))     # a fixed spread of vocab columns
    dump('smollm2_logits.json', {
        'ids': ids,
        'logits_cols': sel,
        'logits_sel': b64(np.ascontiguousarray(
            logits[:, sel], dtype=np.float32).tobytes()),
        'last_logits': b64(logits[-1].astype(np.float32).tobytes()),
        'greedy': gen[len(ids):],
        'greedy_text': at.decode(gen[len(ids):], skip_special_tokens=True),
        'raw_prompt_ids': at('The capital of France is')['input_ids'],
    })


TEMPLATE_REPOS = ['HuggingFaceTB/SmolLM2-135M-Instruct',
                  'Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen3-0.6B',
                  'HuggingFaceTB/SmolLM3-3B',
                  'microsoft/Phi-3.5-mini-instruct',
                  'Qwen/Qwen2.5-Coder-1.5B']
FIXED_NOW = (2026, 1, 15, 9, 30)       # templates that print today's date


def chat_templates():
    import datetime
    from huggingface_hub import hf_hub_download
    import transformers.utils.chat_template_utils as ctu

    class FixedClock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(*FIXED_NOW)
    ctu.datetime = FixedClock

    def text(v):
        return v.get('content') if isinstance(v, dict) else v

    convs = [
        ([{'role': 'user', 'content': 'What is 2+2?'}], {}),
        ([{'role': 'system', 'content': 'Answer in French.'},
          {'role': 'user', 'content': 'Hello!'}], {}),
        ([{'role': 'user', 'content': 'Name a colour.'},
          {'role': 'assistant', 'content': 'Blue.'},
          {'role': 'user', 'content': 'Another, with an emoji \U0001F308'}],
         {}),
        ([{'role': 'user', 'content': 'Think briefly: 3*7?'}],
         {'enable_thinking': False}),
    ]
    out = {}
    for repo in TEMPLATE_REPOS:
        try:
            cfg_path = hf_hub_download(repo, 'tokenizer_config.json')
        except Exception as e:
            print('skipping chat templates (%s)' % e.__class__.__name__)
            return
        with open(cfg_path, 'rb') as f:
            cfg = json.loads(f.read().decode('utf-8'))
        tpl = cfg.get('chat_template')
        try:
            with open(hf_hub_download(repo, 'chat_template.jinja'),
                      'rb') as f:
                tpl = f.read().decode('utf-8')
        except Exception:
            pass
        if isinstance(tpl, list):
            tpl = dict((t['name'], t['template']) for t in tpl)['default']
        special = {k: text(cfg[k]) for k in ('bos_token', 'eos_token',
                                              'unk_token', 'pad_token')
                   if text(cfg.get(k))}
        if cfg.get('additional_special_tokens'):
            special['additional_special_tokens'] = [
                text(t) for t in cfg['additional_special_tokens']]
        cases = []
        for msgs, kw in convs:
            for agp in (True, False):
                r, _ = ctu.render_jinja_template(
                    [msgs], chat_template=tpl, add_generation_prompt=agp,
                    **dict(special, **kw))
                cases.append({'messages': msgs, 'kwargs': kw,
                              'add_generation_prompt': agp, 'text': r[0]})
        out[repo] = {'template': tpl, 'special_tokens': special,
                     'cases': cases}
    dump('chat_templates.json', {'now': list(FIXED_NOW), 'repos': out})


# ----------------------------------------------------------------- lfm3

LFM3_BASE = dict(vocab_size=160, hidden_size=48, intermediate_size=96,
                 num_attention_heads=6, num_key_value_heads=2, head_dim=8,
                 max_position_embeddings=256, norm_eps=1e-5, conv_L_cache=7,
                 conv_bias=False, bos_token_id=1, eos_token_id=2,
                 pad_token_id=0, tie_word_embeddings=True)


def _randomize(model, seed):
    import torch
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
        for n, p in model.named_parameters():
            if 'norm' in n:
                p.add_(1.0)


def _lfm3_dump(name, cfg_json, sd, ids, logits, note):
    from safetensors.torch import save
    import torch
    sd = {k: v.detach().to(torch.bfloat16).contiguous()
          for k, v in sd.items()}
    dump(name, {'config': cfg_json, 'safetensors': b64(save(sd)),
                'ids': ids, 'logits': b64(logits.tobytes()),
                'logits_shape': list(logits.shape), 'reference': note})


def _bf16_round(sd):
    import torch
    return {k: v.detach().to(torch.bfloat16).float() for k, v in sd.items()}


def tiny_lfm3_via_lfm2():
    '''A tiny LFM3 whose forward pass equals transformers' built-in LFM2
    (no sandwich norms, full RoPE, gate weights zero so the gate is a
    constant 1/2, made up for by doubling o_proj). LFM2 shares LFM3's
    short-conv block, QK-norm attention and final embedding_norm, so this
    checks those against library code without running repository code.'''
    import torch
    import transformers
    layer_types = ['conv', 'full_attention', 'conv', 'conv',
                   'full_attention']
    kw = dict(LFM3_BASE, num_hidden_layers=len(layer_types),
              layer_types=layer_types, block_auto_adjust_ff_dim=False,
              rope_parameters={'rope_type': 'default',
                               'rope_theta': 10000.0})
    cfg = transformers.Lfm2Config(**kw)
    model = transformers.Lfm2ForCausalLM(cfg).eval()
    _randomize(model, 11)
    sd = _bf16_round(model.state_dict())
    sd.pop('lm_head.weight', None)
    model.load_state_dict(sd, strict=False)
    model.tie_weights()
    g = torch.Generator().manual_seed(12)
    ids = torch.randint(0, cfg.vocab_size, (1, 30), generator=g)
    with torch.no_grad():
        logits = model(ids).logits[0].numpy().astype(np.float32)
    out = {}
    for k, v in sd.items():
        k = (k.replace('self_attn.out_proj', 'self_attn.o_proj')
              .replace('q_layernorm', 'q_norm')
              .replace('k_layernorm', 'k_norm'))
        if k.endswith('self_attn.o_proj.weight'):
            v = v * 2   # exact in bf16
        out[k] = v
    for i, t in enumerate(layer_types):
        if t != 'conv':
            out['model.layers.%d.self_attn.gate_proj.weight' % i] = \
                torch.zeros(cfg.num_attention_heads, cfg.hidden_size)
    cfg_json = dict(kw, model_type='lfm3',
                    architectures=['Lfm3ForCausalLM'],
                    use_sandwich_norm=False, sliding_window=None,
                    rope_parameters={'full_attention': {
                        'rope_type': 'default', 'rope_theta': 10000.0}})
    del cfg_json['block_auto_adjust_ff_dim']
    _lfm3_dump('tiny_lfm3_via_lfm2.json', cfg_json, out, ids[0].tolist(),
               logits, 'transformers %s Lfm2ForCausalLM'
               % transformers.__version__)


def lfm3_reference_forward(cfg, sd, ids):
    '''LFM3's forward pass written out in PyTorch, transcribed from the
    model repository's modeling_lfm3.py (revision 40ebe0a): whole-sequence
    and unoptimised, sharing no code with tinygpt/hf/lfm3.py. Covers what
    LFM2 lacks: sandwich norms, the per-head output gate, zero-tail
    partial RoPE and sliding-window layers.'''
    import torch
    import torch.nn.functional as F
    H, nh, nkv, hd = (cfg['hidden_size'], cfg['num_attention_heads'],
                      cfg['num_key_value_heads'], cfg['head_dim'])
    eps = cfg['norm_eps']
    T = len(ids)

    def norm(x, w):
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    def rope(layer_type):
        p = cfg['rope_parameters'][layer_type]
        angles = int(p.get('partial_rotary_factor', 1.0) * hd) // 2
        inv = 1.0 / (p['rope_theta'] ** (
            torch.arange(0, 2 * angles, 2, dtype=torch.float) / hd))
        inv = torch.cat((inv, torch.zeros(hd // 2 - angles)))
        f = torch.arange(T, dtype=torch.float)[:, None] * inv[None]
        emb = torch.cat((f, f), -1)
        return emb.cos(), emb.sin()

    def rot(x):
        return torch.cat((-x[..., hd // 2:], x[..., :hd // 2]), -1)

    x = sd['model.embed_tokens.weight'][torch.tensor(ids)]
    for i, t in enumerate(cfg['layer_types']):
        p = 'model.layers.%d.' % i
        h = norm(x, sd[p + 'operator_norm.weight'])
        if t == 'conv':
            B, C, xx = (h @ sd[p + 'conv.in_proj.weight'].T).T.chunk(3, 0)
            conv = F.conv1d((B * xx)[None], sd[p + 'conv.conv.weight'],
                            padding=cfg['conv_L_cache'] - 1,
                            groups=H)[0, :, :T]
            y = (C * conv).T @ sd[p + 'conv.out_proj.weight'].T
        else:
            a = p + 'self_attn.'
            q = norm((h @ sd[a + 'q_proj.weight'].T).view(T, nh, hd),
                     sd[a + 'q_norm.weight']).transpose(0, 1)
            k = norm((h @ sd[a + 'k_proj.weight'].T).view(T, nkv, hd),
                     sd[a + 'k_norm.weight']).transpose(0, 1)
            v = (h @ sd[a + 'v_proj.weight'].T).view(T, nkv, hd).transpose(
                0, 1)
            cos, sin = rope(t)
            q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
            k = k.repeat_interleave(nh // nkv, 0)
            v = v.repeat_interleave(nh // nkv, 0)
            qi = torch.arange(T)[:, None]
            ki = torch.arange(T)[None, :]
            allowed = ki <= qi
            if t == 'sliding_attention':
                allowed &= ki > qi - cfg['sliding_window']
            s = (q @ k.transpose(1, 2)) * hd ** -0.5
            s = s.masked_fill(~allowed, float('-inf')).softmax(-1)
            o = (s @ v).transpose(0, 1)
            gate = torch.sigmoid(h @ sd[a + 'gate_proj.weight'].T)
            o = (o * gate[..., None]).reshape(T, nh * hd)
            y = o @ sd[a + 'o_proj.weight'].T
        if cfg['use_sandwich_norm']:
            y = norm(y, sd[p + 'post_operator_norm.weight'])
        x = x + y
        f = p + 'feed_forward.'
        h = norm(x, sd[p + 'ffn_norm.weight'])
        y = (F.silu(h @ sd[f + 'w1.weight'].T) * (h @ sd[f + 'w3.weight'].T)
             ) @ sd[f + 'w2.weight'].T
        if cfg['use_sandwich_norm']:
            y = norm(y, sd[p + 'post_ffn_norm.weight'])
        x = x + y
    x = norm(x, sd['model.embedding_norm.weight'])
    return x @ sd['model.embed_tokens.weight'].T


def tiny_lfm3():
    '''A tiny random LFM3 with every LFM3 feature, a sliding window of 5
    over a 30-token sequence, and logits from lfm3_reference_forward().'''
    import torch
    torch.manual_seed(21)
    layer_types = ['conv', 'sliding_attention', 'full_attention', 'conv',
                   'sliding_attention', 'conv']
    cfg = dict(LFM3_BASE, model_type='lfm3',
               architectures=['Lfm3ForCausalLM'],
               num_hidden_layers=len(layer_types), layer_types=layer_types,
               use_sandwich_norm=True, sliding_window=5,
               rope_parameters={
                   'full_attention': {'rope_type': 'default',
                                      'rope_theta': 1000000.0,
                                      'partial_rotary_factor': 0.5},
                   'sliding_attention': {'rope_type': 'default',
                                         'rope_theta': 10000.0}})
    H, I, nh, nkv, hd = 48, 96, 6, 2, 8
    shapes = {'model.embed_tokens.weight': (160, H),
              'model.embedding_norm.weight': (H,)}
    for i, t in enumerate(layer_types):
        p = 'model.layers.%d.' % i
        for n in ('operator_norm', 'ffn_norm', 'post_operator_norm',
                  'post_ffn_norm'):
            shapes[p + n + '.weight'] = (H,)
        shapes.update({p + 'feed_forward.w1.weight': (I, H),
                       p + 'feed_forward.w3.weight': (I, H),
                       p + 'feed_forward.w2.weight': (H, I)})
        if t == 'conv':
            shapes.update({p + 'conv.in_proj.weight': (3 * H, H),
                           p + 'conv.conv.weight': (H, 1, 7),
                           p + 'conv.out_proj.weight': (H, H)})
        else:
            a = p + 'self_attn.'
            shapes.update({a + 'q_proj.weight': (nh * hd, H),
                           a + 'k_proj.weight': (nkv * hd, H),
                           a + 'v_proj.weight': (nkv * hd, H),
                           a + 'o_proj.weight': (H, nh * hd),
                           a + 'gate_proj.weight': (nh, H),
                           a + 'q_norm.weight': (hd,),
                           a + 'k_norm.weight': (hd,)})
    sd = {}
    for k in sorted(shapes):
        v = torch.randn(shapes[k]) * 0.3
        if 'norm' in k:
            v += 1.0
        sd[k] = v
    sd = _bf16_round(sd)
    g = torch.Generator().manual_seed(22)
    ids = torch.randint(0, 160, (30,), generator=g).tolist()
    with torch.no_grad():
        logits = lfm3_reference_forward(cfg, sd, ids).numpy().astype(
            np.float32)
    _lfm3_dump('tiny_lfm3.json', cfg, sd, ids, logits,
               'lfm3_reference_forward (tests/make_hf_fixtures.py)')


def lfm3_tokens(model_dir):
    from tokenizers import Tokenizer
    ref = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer.json'))
    texts = TEXTS + NFC_TEXTS + random_texts(500, seed=5) + [
        ' struc', '\tpubli', ' impleme', 'nclu tegory .loa',
        '1234567 89 3.14159 ٣٤٥٦',
        '<|startoftext|><|im_start|>user\nHi<|im_end|>\n<think>x</think>']
    with open(os.path.join(HERE, 'make_hf_fixtures.py'), 'rb') as fh:
        texts.append(fh.read().decode('utf-8'))
    dump('lfm3_tokens.json', {'texts': texts,
                              'ids': [ref.encode(t).ids for t in texts]})


def main():
    os.makedirs(FIX, exist_ok=True)
    args = sys.argv[1:]
    lfm3_dir = None
    for a in list(args):
        if a.startswith('--lfm3='):
            lfm3_dir = a.split('=', 1)[1]
            args.remove(a)
    if '--only-lfm3' in args:
        args.remove('--only-lfm3')
        tiny_lfm3_via_lfm2()
        tiny_lfm3()
        if lfm3_dir:
            lfm3_tokens(lfm3_dir)
        return
    safetensors_dtypes()
    base = dict(vocab_size=160, hidden_size=48, intermediate_size=96,
                num_hidden_layers=3, num_attention_heads=6,
                num_key_value_heads=2, max_position_embeddings=256,
                rms_norm_eps=1e-5, rope_theta=100000.0, hidden_act='silu',
                bos_token_id=1, eos_token_id=2)
    tiny_llama('tiny_llama_tied_bf16.json',
               dict(base, tie_word_embeddings=True), 'BF16', seed=1)
    tiny_llama('tiny_llama_untied_f16.json',
               dict(base, tie_word_embeddings=False, attention_bias=True,
                    mlp_bias=True, num_key_value_heads=3,
                    rope_scaling={'rope_type': 'llama3', 'factor': 8.0,
                                  'low_freq_factor': 1.0,
                                  'high_freq_factor': 4.0,
                                  'original_max_position_embeddings': 16}),
               'F16', seed=2)
    tiny_llama('tiny_llama_mha_f32.json',
               dict(base, tie_word_embeddings=True, num_key_value_heads=6,
                    rope_theta=10000.0), 'F32', seed=3, seq_len=70)
    tiny_llama('tiny_qwen2_bf16.json',
               dict(base, tie_word_embeddings=True, rope_theta=1000000.0,
                    rms_norm_eps=1e-6, use_sliding_window=False),
               'BF16', seed=4, kind='qwen2')
    bpe_tiny()
    bpe_tiny_qwen()
    chat_templates()
    tiny_lfm3_via_lfm2()
    tiny_lfm3()
    if len(args) > 0:
        smollm2_tokens(args[0])
        smollm2_logits(args[0])
    if len(args) > 1:
        qwen_tokens(args[1])
        qwen_logits(args[1])
    if len(args) > 2:
        qwen_instruct_chat(args[2])
    if lfm3_dir:
        lfm3_tokens(lfm3_dir)


if __name__ == '__main__':
    main()
