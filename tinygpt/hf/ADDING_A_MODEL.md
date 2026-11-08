# Adding a model architecture to `tinygpt/hf/`

How Llama, then Qwen2, were added here, written down as a procedure so the
next model goes the same way. The rules come from mistakes made along the
way; the [pitfalls](#11-pitfalls-that-already-happened-once) section lists
them. The [LFM3 notes](#appendix-lfm3-17b-what-is-known-so-far) at the end
are the starting point for the next target.

**Definition of done** for a new model:

1. Tokenization is **exact** against the Rust `tokenizers` library on
   thousands of texts, not approximately right.
2. Logits match `transformers` in float32 to about 1e-4 (tests allow 1e-3)
   at every prompt position, through both prefill and cached decoding, in
   both weight modes.
3. A greedy continuation (and, for an instruct model, a full chat turn) is
   token-for-token identical to `transformers`.
4. All of that is a test in `tests/test_hf_*.py`, backed by fixtures from
   `tests/make_hf_fixtures.py`. The tests pass on Python 3.13 and on
   3.10 + NumPy 1.21 (Pythonista-like), and on the iPhone via
   `hf_run.py selftest`.
5. `tinygpt/hf/README.md` lists the model and its measured speed/memory.

Plausible-looking text proves nothing. A wrong RoPE base or a missing bias
still produces fluent English.

## 0. Ground rules

- **Never execute code from a model repository.** Read `modeling_*.py`,
  `configuration_*.py` etc. as text, as the specification. The downloader's
  allowlist never fetches them, and `tools/hf_probe.py` only lists them.
- **The reference is `transformers`' own implementation**, run in a scratch
  virtualenv outside the repository, only to produce fixtures. If
  `transformers` has no native implementation and the model needs
  `trust_remote_code=True`, running the repo's code is a decision for the
  user. Ask first, and say what would run. The alternatives are a closely
  related native class (e.g. `lfm2`), or writing an independent PyTorch
  reference from the spec.
- **Downloaded files are untrusted data.** Put them in their own directory
  and run Python on them with `-I`.
- Weights are only ever loaded from **safetensors**. Pickled checkpoints
  are refused.

## 1. Environment (cloud container)

Scratch tools live in the session scratchpad, not the repo:

```
python3 -m venv $SCRATCH/refenv
$SCRATCH/refenv/bin/pip install numpy torch transformers tokenizers safetensors jinja2 huggingface_hub
uv venv -p 3.10 $SCRATCH/py310                     # Pythonista-like
VIRTUAL_ENV=$SCRATCH/py310 uv pip install "numpy==1.21.6" "jinja2==3.0.3" requests tokenizers
```

- Network: `huggingface.co` serves configs and tokenizers. The weights
  redirect to a CDN (`us.aws.cdn.hf.co` and other `*.hf.co` hosts), which
  must be allowed separately. Check with
  `curl -sSL -r 0-99 -o /dev/null -w '%{http_code}\n' https://huggingface.co/<repo>/resolve/main/model.safetensors`.
  206 means it works.
- Download real models with the project's own downloader:
  `python3 hf_run.py download --model <repo>`. They land in `hf_models/`
  (gitignored), and the tests find them there.
- Disk: a 1.7B BF16 model is about 3.5 GB, and transformers needs about
  4 bytes/param of RAM to load it in float32 for fixtures (fine on the 15 GB
  container up to about 3B).
- Linux has no Accelerate. `TINY_GPT_CBLAS=/lib/x86_64-linux-gnu/libblas.so.3`
  exercises the CBLAS ctypes bindings, and the BF16 `BNNSCopy` path is
  tested with the fake library in `tests/native16_fake_bnns.c`. Speed
  on Linux is not representative of the phone; ask the user to run
  `selftest`.

## 2. Probe the repository

```
python3 tools/hf_probe.py --remote <org>/<name>     # before downloading
python3 tools/hf_probe.py <org>/<name>              # once cached
```

It prints the files, the config (including keys `llama.py` does not read),
`generation_config.json`, every tensor name with the layer index folded
into `N` and the layers each one occurs in, the tokenizer pipeline, the
chat template's features, and whether the current code accepts the model
and tokenizer. Read it with these questions:

- **Tensor names are the architecture.** `q_norm`/`k_norm` mean QK-norm.
  `*_proj.bias` means biases (on which projections exactly?). A missing
  `lm_head` means tied embeddings. `post_*_norm` means sandwich norms.
  `conv.*`, `A_log`, `dt_proj` mean non-attention mixers. Tensors present
  only on some layers mean heterogeneous layers.
- **Parameter count × bytes/param ≈ file size?** If it is off by more than
  the header, a tensor is shared or something is missing.
- **Unread config keys** are the features you have not handled yet.
- **Memory:** float32 is 4 bytes/param. On the phone only lowmem works past
  about 0.5B params (`WEIGHTS='auto'` handles the choice).

## 3. Read the reference forward pass

Find the model's code: `transformers/models/<type>/modular_<type>.py` (best:
written as a diff against a parent such as Llama), else
`modeling_<type>.py`, else the repo's own `modeling_*.py` (as text). Read
the `forward` of the RMSNorm, rotary embedding, attention, MLP, decoder
layer, model and causal-LM classes, and write down every difference from
`modeling_llama.py`. Checklist:

| Area | Things that vary |
|------|------------------|
| Norm | RMSNorm vs LayerNorm, `w*x` vs `(1+w)*x` (Gemma), eps key name, float32 upcast, where it's applied (pre, post, sandwich), final norm's name |
| Embedding | scaled by √hidden (Gemma), tied, a norm right after embedding |
| Attention | q/k/v/o bias each separately, QK-norm (per head, before RoPE?), `head_dim` not hidden/heads, GQA grouping (`repeat_kv` order), scale (`head_dim**-0.5` or custom), logit softcap, output gate (sigmoid per head), sliding window per layer |
| RoPE | theta per layer type, partial rotary (which dims, "zero-tail" vs truncated), rotate-half vs interleaved, scaling types (linear, llama3, yarn, longrope), `attention_scaling` factor |
| MLP | activation, gate/up order (`w1`/`w3` names), fused `gate_up_proj`, biases |
| Layers | non-attention mixers (short conv, SSM), per-layer type list |
| Head | tied, final logit softcap or scale |
| Cache | what decoding needs per layer: K/V, conv state, SSM state |

Tip: transformers 5 config files differ from transformers 4 ones
(`rope_parameters` instead of `rope_theta`/`rope_scaling`, `layer_types`
instead of `use_sliding_window`). `LlamaConfig` reads both. Keep it that
way for new keys.

## 4. Implement the model

Where the code goes:

- **Llama plus flags** (biases, QK-norm, head_dim, a RoPE variant): add the
  flag to `LlamaConfig` in `tinygpt/hf/llama.py`, keyed on `model_type`
  (see how `qwen2` sets `qkv_bias`/`o_bias`). Add the type to
  `SUPPORTED_TYPES`.
- **Structurally different blocks** (sandwich norms, conv mixers, gated
  attention output): write a sibling module, e.g. `tinygpt/hf/lfm3.py`,
  that reuses the shared pieces from `llama.py`: `Linear`, `MatVec`,
  `rms_norm`, `silu_mul`, `softmax_last`, `rope_rotate`, `Sampler`,
  `generate`. Register it in `tinygpt/hf/models.py` (`_families()`), which
  `hf_run.load()` and `tools/hf_probe.py` use to dispatch on `model_type`.
  `tinygpt/hf/lfm3.py` is the worked example. The config class needs
  `n_params()`, `kv_bytes(ctx)`, `describe()`, `max_context`, `eos_ids`;
  the model needs `from_dir`, `new_cache`, `step`, `prefill`, `feed`,
  `close`, `mv`, `ctx_limit`, `weights`.

Conventions the shared code relies on:

- Weights keep the Hugging Face `(out, in)` orientation and go through
  `Linear` (fuse projections that read the same input: q/k/v, gate/up).
  `Linear` gives you both weight modes, BNNSCopy widening, the measured
  matvec/matmul backend, and lowmem slabs for free.
- Compute in float32. Read every dimension from the config.
- Implement **both** paths: `prefill(ids, cache, all_logits, chunk)` (matrix
  products, chunked) and `step(tok, cache)` (one token). They must agree
  with each other and with the reference. The tests check both.
- The cache must support `feed()`'s prefix reuse: `cache.tokens`,
  `cache.length`, `truncate(n)`. A state that can't be rewound (a conv or
  SSM state) needs either a snapshot per position or a rewind by
  re-prefilling from the start. Pick one and test the chat-turn reuse.
  (LFM3 keeps each conv layer's input for every position: one row per
  token, which is what a snapshot per position costs at minimum.)
- `ContextFull` when the cache is full. Sliding-window layers need a real
  window if it is smaller than `CONTEXT_LIMIT`: either a ring buffer, or
  (simpler, and what LFM3 does) the full cache with each query attending
  only to keys `(p - window, p]`. That is transformers' rule
  (`sliding_window_overlay`: `kv_idx > q_idx - sliding_window`). Capping
  the context at the window, as for Qwen2, is only acceptable when the
  window is large.
- Validate every tensor's shape against the config at load
  (`raw(name, shape)` does this) and raise `UnsupportedModel` with the
  name of anything missing or unexpected.

## 5. Tokenizer

Run the probe's tokenizer section first. `tinygpt/hf/tokenizer.py`
supports byte-level BPE with:

- added tokens without lstrip/rstrip/single_word;
- the NFC normalizer (with the reference's older-Unicode quirk: see
  `nfc()` and `NFC_INERT`);
- the pre-tokenizer steps Digits(individual), Split(regex, Isolated) and
  ByteLevel (with or without its own regex);
- BPE without dropout, byte fallback, prefixes/suffixes or
  `ignore_merges`.

Anything else is refused at load, and that's the to-do list.

- Translate regexes with `translate_regex()`. It refuses anything whose
  meaning differs between Oniguruma and Python `re`. Extend it, never
  bypass it.
- Character classes come from `unicode_tables.py`, which is measured from
  the reference library by `tools/gen_unicode_tables.py`. Regenerate it
  with the newest Python rather than editing by hand.
- Verify with a comparison script before writing fixtures: the reference
  `tokenizers` and ours on the edge-case list in `make_hf_fixtures.py`
  (`TEXTS`, `NFC_TEXTS`), plus a few thousand random strings, plus real
  source files. Check encode **and** decode, with and without
  `skip_special_tokens`, under both Pythons. Zero mismatches.
- Small tokenizers trained with the same pipeline
  (`bpe_tiny*.json`) make the test run without the real model.

## 6. Chat template, special tokens, sampling

- The template is rendered with jinja2 in transformers' sandbox
  configuration (`ChatFormat`). Add the repo to `TEMPLATE_REPOS` in
  `make_hf_fixtures.py` and check the renderings match byte for byte.
- Stop tokens come from `config.json`, `generation_config.json` and the
  tokenizer's `eos_token`. ChatML templates also stop at `<|im_end|>`.
- Sampling defaults follow `generation_config.json` exactly as transformers
  does. In particular, `repetition_penalty` applies even when greedy, and
  any fixture generated with `model.generate()` inherits it.
- Base vs instruct: a base model in chat mode echoes the prompt. That is
  correct behaviour. `hf_run` warns about it.

## 7. Reference fixtures (`tests/make_hf_fixtures.py`)

Add generators following the existing ones. Fixtures are JSON, with
arrays base64-encoded, because `*.npz` is gitignored and `*.safetensors` is
LFS-tracked.

- **Tiny random model** (`tiny_llama(..., kind=...)`): build the
  transformers config with small dimensions but every feature switched on.
  Seed it, use normal(0, 0.3) init and add 1 to norm weights. Store the
  weights rounded to the checkpoint dtype, and compute the reference
  logits in float32 *from the rounded weights*. One fixture per distinct
  layer mix (e.g. conv + full + sliding).
- **No library reference?** If transformers lacks the architecture, look
  for a built-in sibling and build a tiny model of the new type that
  reduces exactly to it (LFM3 → LFM2: no sandwich norms, full RoPE, gate
  weights zero so the gate is a constant 1/2, made up for by doubling
  `o_proj`; see `tiny_lfm3_via_lfm2()`). That checks the shared parts
  against library code. For the rest, transcribe the reference forward
  pass into plain whole-sequence PyTorch in the generator
  (`lfm3_reference_forward()`); it shares no code with the NumPy model, so
  it catches indexing, masking and caching mistakes, though not a misread
  of the reference.
- **Real model**: logits at every prompt position on a fixed column subset,
  plus the full last-position logits, plus a 40-token greedy continuation.
  For instruct models, add one chat turn decoded greedily with the model's
  repetition penalty.
- **Templates**: rendered on a fixed clock (`FIXED_NOW`).
- A fixture that depends on a file in the repository (the README, the
  script itself) stores that text. Otherwise editing the file silently
  turns the test into a skip.

Regenerate:
`$SCRATCH/refenv/bin/python tests/make_hf_fixtures.py [--lfm3=<lfm3 dir>] <smollm2 dir> <qwen dir> <qwen-instruct dir>`,
or only the LFM3 fixtures with `--only-lfm3 [--lfm3=<dir>]`.
Then `git diff --stat tests/fixtures`. Revert any fixture that changed
without its inputs changing (`bpe_tiny.json` training is not always
deterministic).

## 8. Tests

Mirror the existing classes:

- Add the tiny fixture to `TINY` in `test_hf_llama.py`, or write an
  equivalent class. It must cover prefill at chunk sizes 1/5/64, cached
  steps, both weight modes, and cached against uncached.
- Config tests: the flags parsed, unsupported variants refused.
- The tokenizer against its fixture; the real tokenizer and model in
  classes that skip unless cached (`tests/hf_common.py`: add a
  `<model>_dir()` helper).
- Real-model inference inside `_NoNetwork()` to prove it's offline.
- Keep the real-model greedy test short (12–16 tokens): it runs on the phone.

Run:

```
python3 -W error::ResourceWarning -m unittest discover -s tests -p 'test_hf*.py' -t .
$SCRATCH/py310/bin/python -m unittest discover -s tests -p 'test_hf*.py' -t .
python3 -c "import sys; sys.argv=['x','gradcheck-all']; from tinygpt.cli import main; main()"
```

(The last one checks the training code is untouched. Don't run
`tiny_gpt.py` with arguments: it hardcodes its own `sys.argv` and starts
training.)

## 9. When logits don't match

1. Compare the residual stream layer by layer: `model(ids,
   output_hidden_states=True).hidden_states[i]` against your own after
   layer `i`. The first layer that diverges holds the bug.
2. Inside that layer, check the operator output and the FFN output
   separately (forward hooks on the submodules).
3. Usual causes, from experience: a config key read from the wrong place
   (`rope_theta` moved into `rope_parameters` and silently defaulted to
   10000); bias on the wrong projections; GQA head grouping; rotate-half vs
   interleaved RoPE; a missing norm; the wrong eps key.
4. If prefill logits match but generations diverge, compare cached `step()`
   logits along the reference's own token path. If those match too, the
   difference is in sampling settings (`repetition_penalty` from
   `generation_config.json`), not the model. A top-2 margin under ~1e-3
   can legitimately flip greedy decoding; check the margin before
   assuming a bug.

## 10. Finish

- `hf_run.py`: dispatch, memory estimate (`n_params()`), stop tokens.
  Check `tools/hf_probe.py` now reports "OK".
- `tinygpt/hf/README.md`: the model table (architecture, weight mode,
  measured iPhone speed), the verification bullets, limitations.
- Commit messages say what was verified and how. A code-only zip for the
  user:
  `git ls-files | grep -v SmolLM2-135M-Instruct.zip` into
  `tiny_gpt/…` (they keep `hf_models/` on the phone).
- Ask the user to run `MODE='selftest'` with the new `MODEL_ID` and send
  `hf_report.txt`. That gives real speed and footprint, and runs the
  real-model tests against Accelerate.

## 11. Pitfalls that already happened once

- A patch script written as `python3 - <<'EOF'` whose replacement text
  contains `"""` breaks its own string literal. Write such blocks to a
  file first.
- `pkill -f "<pattern>"` matched the shell running it and killed the
  command.
- A dropped HTTPS stream can end without an exception. The downloader treats
  a short read as resumable, and the test needs a small `STREAM_CHUNK` to
  see it.
- `requests` error messages contain the signed CDN URL; `hub._short()`
  trims it.
- After a PR merges, GitHub may delete the branch. A later push recreates
  it.
- Pythonista keeps the interpreter alive between runs. Peak RSS is
  cumulative; use `phys_footprint` (`memory_now()`).
- iOS kills an app that exceeds its memory without raising `MemoryError`.
  Size before loading.

## Appendix: LFM3-1.7B, a worked example

`LiquidAI/LFM3-1.7B` (revision `40ebe0a`), added on 2026-10-27 in
`tinygpt/hf/lfm3.py`. These were the notes taken before writing it.

**Repository**: `model.safetensors` 3.46 GB BF16, 1,728,801,536 params
(tied embeddings). Custom code (`modeling_lfm3.py`, `lfm3_conv.py`,
`configuration_lfm3.py`, via `auto_map`). **transformers 5.20.2 has no
native `lfm3`** (it has `lfm2`, `lfm2_moe`, `lfm2_vl`). The reference
implementation therefore exists only as the repo's code.

**Config**: hidden 2048, 24 layers, 16 heads / 4 KV heads × 128,
intermediate 7680, vocab 128000, `norm_eps` 1e-5, `conv_L_cache` 7,
`conv_bias` false, `use_sandwich_norm` true, `sliding_window` 256.
`layer_types`: conv at 0,1,3,5,…,23 (13), full_attention at 2,6,10,14,18,22
(6), sliding_attention at 4,8,12,16,20 (5). RoPE per layer type:
full = theta 1e6 with `partial_rotary_factor` 0.5; sliding = theta 1e4,
full rotation.

**Per-layer structure** (from `modeling_lfm3.py`, read as text):

```
h = x + post_operator_norm( op( operator_norm(x) ) )          # sandwich
y = h + post_ffn_norm( w2( silu(w1(ffn_norm(h))) * w3(ffn_norm(h)) ) )
op = attention (attn layers) or short conv (conv layers)
final: embedding_norm(y)  -- despite the name it is the FINAL norm, applied
       after the last layer (not after the embedding); then tied lm_head
```

- **RMSNorm**: plain `w * x * rsqrt(mean(x²)+eps)` in float32, the same as
  `llama.rms_norm`.
- **Attention**: q/k/v/o without bias. QK-norm (`q_norm`/`k_norm`, RMSNorm
  over head_dim=128) is applied *before* RoPE. RoPE is rotate-half, using
  the "zero-tail" partial layout: `inv_freq` has head_dim/2 entries, the
  first `0.5*128/2 = 32` from the usual progression over the *full*
  head_dim (`theta^(-2i/128)`), the remaining 32 are **zero** (cos=1,
  sin=0, so unrotated). That is not the same as rotating only the first
  half of the vector. The scale is `head_dim**-0.5`. Then an **output gate**:
  `attn_out[h] *= sigmoid(gate_proj(x))[h]`, one scalar per head
  (`gate_proj` [16, 2048]), on the *normed* input, before `o_proj`.
  Sliding layers attend to the last 256 positions. 256 is far below
  `CONTEXT_LIMIT`, so they need a real window; capping the context isn't
  acceptable.
- **Short conv**: `B, C, x = chunk(in_proj(h), 3)` along features
  (in_proj [6144, 2048] → 3×2048, in that order). `Bx = B * x`. Then a
  causal depthwise conv over time, kernel 7 (`conv.weight` [2048, 1, 7],
  PyTorch `Conv1d` with padding 6, output truncated to the first T, so
  output[t] = Σ_k w[c,k]·Bx[t-6+k]). Then `y = C * conv_out`, then
  `out_proj` [2048, 2048]. Decode state per conv layer: the last 6 (or 7)
  `Bx` columns. Prefix reuse in `feed()` needs that state at the reuse
  point, so snapshot or recompute. (Done: Bx kept for every position.)
- **Packed-sequence handling** (`seq_idx`, padding masks): irrelevant for
  single-sequence inference.

**Tokenizer** (`tokenizer.json`, 17.9 MB): no normalizer. Split regex as
Qwen2's but with `\p{N}{1,3}` (digit groups of up to 3), then ByteLevel
(use_regex false). BPE with **`ignore_merges: true`** (a word that is itself in the model
vocabulary maps to that id directly, without running merges; it changes
the encoding of 2,459 vocabulary words, e.g. `" struc"`). 124,893 vocab entries + 140 added tokens (124 special,
16 not; no lstrip/rstrip). Decoder/post-processor are ByteLevel. BOS
`<|startoftext|>` (124894), EOS `<|im_end|>` (124900), pad `<|pad|>`.

**Chat template** (`chat_template.jinja`, 12 KB): starts with `bos_token`.
It's ChatML with `<think>…</think>` reasoning blocks, `enable_thinking`,
tools, `{% generation %}`, `namespace`. It already renders here with
jinja2. `generation_config`: do_sample, temperature 0.1, top_k 50,
repetition_penalty 1.1, eos [124900].

**How it went**:
1. Tokenizer: `ignore_merges` implemented (the lookup is in the model's
   vocabulary, not added tokens). Identical to `tokenizers` on 4,052 texts;
   the digit-group regex already translated.
2. Reference: running the repository's `modeling_lfm3.py` via
   `trust_remote_code` was blocked by the agent's permission system, so it
   wasn't run. Two substitutes: a tiny LFM3 reduced to transformers'
   built-in LFM2 (`tiny_lfm3_via_lfm2.json`), and a tiny LFM3 with every
   feature checked against a PyTorch transcription of the reference
   (`tiny_lfm3.json`). Each check fails if you shift the sliding window
   by one or replace the gate with a constant (mutation-tested).
3. `tinygpt/hf/lfm3.py`: one `_forward` serves decode and prefill. The gate
   projection is fused into q/k/v (it reads the same normed input).
   Sliding layers keep the full cache and attend over their window.
4. Real model: the parameter count matches (1,728,801,536). Greedy
   completion of "The capital of France is" gives " Paris.". In chat it
   gets 17 × 23 = 391 right, remembers a name across turns (`feed()`
   rewinds into the previous answer, because the template drops old
   `<think>` blocks), and recalls a code from 600 tokens back, past the
   sliding window. ~6 tok/s float32 on the 4-vCPU container.
5. Still open: phone `selftest` (lowmem only; 3.46 GB mapped).
