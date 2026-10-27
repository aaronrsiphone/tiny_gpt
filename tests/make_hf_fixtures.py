# coding: utf-8
'''
Regenerate the reference fixtures in tests/fixtures/ from the official
Hugging Face libraries. The tests themselves need only NumPy; this script
is the one place the reference implementations are used:

    pip install torch transformers tokenizers safetensors
    python tests/make_hf_fixtures.py [SmolLM2-135M-Instruct dir
                                      [Qwen2.5-Coder-1.5B dir]]

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


def main():
    os.makedirs(FIX, exist_ok=True)
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
    if len(sys.argv) > 1:
        smollm2_tokens(sys.argv[1])
        smollm2_logits(sys.argv[1])
    if len(sys.argv) > 2:
        qwen_tokens(sys.argv[2])
        qwen_logits(sys.argv[2])


if __name__ == '__main__':
    main()
