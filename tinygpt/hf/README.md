# Hugging Face inference

Run pretrained Hugging Face models (Llama / Qwen2 families) offline on NumPy
and Accelerate, from a laptop to an iPhone in Pythonista. The code lives in
this folder (`tinygpt/hf/`) and is driven by `hf_run.py` at the repository
root. It shares only the BLAS bindings in `tinygpt/backend.py` with the
training code; see the [main README](../../README.md) for that side.

`hf_run.py` downloads a pretrained model straight from the Hugging Face Hub
and runs it entirely on-device, on NumPy and Accelerate. There's no
inference API, no PyTorch and no `transformers`. This is inference only.
Nothing is trained, and the training code in the rest of the repository is
untouched.

Verified against `transformers` (exact tokenization, matching logits,
identical greedy output):

| Model | Architecture | Weights mode | iPhone decode |
|-------|--------------|--------------|---------------|
| [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Llama | float32 | ~78 tok/s |
| [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Llama | lowmem | ~21 tok/s (same tokenizer and template as 135M; logits not separately checked) |
| [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | Qwen2 | lowmem | not yet measured |
| [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) | Qwen2 | lowmem | not yet measured |

Other models with these architectures and a byte-level BPE tokenizer
should work too. For chat, use an instruct model. Base models such as
Qwen2.5-Coder-1.5B were trained only to continue text: in chat mode they
echo the message back (transformers does exactly the same), and
`hf_run.py` warns about this. Use them with `MODE = 'complete'` and the
start of some code as the prompt.

Sampling follows the model's own `generation_config.json`, as transformers
does. For example, Qwen2.5-Coder-Instruct uses temperature 0.7, top-k 20,
top-p 0.8 and repetition penalty 1.1, which transformers applies even when
decoding greedily. Setting `TEMPERATURE`, `TOP_K`, `TOP_P` or
`REPETITION_PENALTY` overrides it. Without a recommendation the defaults
are temperature 0.2 and top-p 0.9.
`MODEL_ID` also accepts a model's huggingface.co URL.

## In Pythonista

1. Open `hf_run.py`, set `MODE = 'download'` and press Run (needs a
   connection, ~270 MB). The files land in `hf_models/` next to the script.
   If `SmolLM2-135M-Instruct.zip` from this repo (Git LFS) is already on the
   device, you can skip the download. Any mode imports the zip automatically
   when the model isn't cached yet.
2. Set `MODE = 'chat'` and press Run. From here on nothing touches the
   network. Turn on airplane mode, quit and reopen Pythonista, and it still
   works.

The other settings at the top of the file are `PROMPT` (for `'ask'` /
`'complete'`), `MAX_NEW_TOKENS`, `TEMPERATURE` (0 = greedy), `TOP_K`,
`TOP_P`, `CONTEXT_LIMIT`, `WEIGHTS` and `SYSTEM_PROMPT`.

## From a shell

```
python hf_run.py download                       # once, online
python hf_run.py chat                           # offline from here on
python hf_run.py ask "Why is the sky blue?" --temp 0
python hf_run.py complete "Once upon a time" --max-new 100
python hf_run.py import SmolLM2-135M-Instruct.zip
python hf_run.py info                           # re-verify cached hashes
python hf_run.py bench                          # speed + memory -> hf_report.txt
python hf_run.py selftest                       # tests + bench -> hf_report.txt
```

```
$ python hf_run.py ask "What is the capital of France?" --temp 0
HuggingFaceTB/SmolLM2-135M-Instruct  (134,515,008 params)
30 layers, hidden 576, 9 heads / 3 kv heads x 64, mlp 1536, vocab 49152, rope theta 100000, tied embeddings
weights float32, context 1024, needs about 558 MB; Accelerate no (NumPy only)
loaded in 2.2s, peak memory 778 MB

> What is the capital of France?

The capital of France is Paris.
[37 prompt tokens in 0.14s, 7 new at 42.5 tok/s]
```

## What is in `tinygpt/hf/`

| File | Contents |
|------|----------|
| `hub.py` | Downloader using `requests` + stdlib. One call to the Hub's revision API, then streamed downloads with progress, HTTP Range resume, retries, and SHA-256 (LFS) / git SHA-1 checks. A snapshot cache works offline. Only an allowlist of files is fetched (config, tokenizer, safetensors). Paths are validated, and repos offering only pickled `*.bin`/`*.pt` weights are refused. |
| `safetensors.py` | Pure-Python reader (struct + json + mmap). Tensors are zero-copy views onto the mapped file. Every offset, length, shape and dtype is validated before bytes are read. Handles F32, F16 and BF16 (an exact 16-bit shift), plus sharded checkpoints through `model.safetensors.index.json`. |
| `tokenizer.py` | Byte-level BPE from `tokenizer.json`: added/special tokens, NFC normalization, the Digits, Split (regex translated to Python's `re`) and ByteLevel pre-tokenizers, and the same priority-queue merge order as the Rust library. Includes a streaming UTF-8 decoder and the chat template, rendered in Jinja2's sandbox exactly as transformers does it. Configurations it doesn't implement exactly are refused at load. |
| `unicode_tables.py` | `\p{L}`, `\p{N}`, `\s` and Rust `is_numeric` as codepoint ranges, measured from the reference `tokenizers` library (`tools/gen_unicode_tables.py`). It also lists the combining marks that library's older NFC tables leave alone. This keeps tokenization from drifting with Pythonista's Unicode database. |
| `llama.py` | The model (Llama and Qwen2): RMSNorm, rotate-half RoPE (including `linear`/`llama3` scaling), grouped-query attention, a preallocated KV cache, SwiGLU, tied or untied heads, and per-projection biases (Qwen2's q/k/v). All dimensions come from `config.json`, in either the transformers 4 or 5 layout. Prompts are prefilled with matrix-matrix products and decoding uses matrix-vector products. Also contains the sampler (greedy, temperature, top-k, top-p, repetition penalty). |

Design points:

- **No transposed weight copies.** Weights keep Hugging Face's `(out, in)`
  layout. Decoding is `cblas_sgemv(NoTrans)` and prefill is
  `cblas_sgemm(transB)`. q/k/v are fused into one matrix per layer, as are
  gate/up, so a layer costs four matrix-vector products.
- **Backends are measured, not assumed.** At load, each weight shape is timed
  with `cblas_sgemv`, a one-row `cblas_sgemm` and NumPy for decoding, and
  with `cblas_sgemm` and NumPy for prefill. Each candidate's output is
  checked against NumPy before it may win. `bench` prints the table.
  `TINY_GPT_MATVEC` / `TINY_GPT_MATMUL` force a choice for A/B runs.
  `backend.py` gained a `gemv()` wrapper; `gemm()` is unchanged.
- **Two weight modes, picked automatically.** `WEIGHTS = 'auto'` uses
  float32 up to ~1 GB of float32 weights and lowmem above that. On an
  iPhone, before BNNSCopy widening, float32 was 1.5x faster at 135M params
  and no faster at 360M, while using 1.8 GB more memory. Explicit float32 is refused on iOS when
  it would need over 2.5 GB, because iOS kills the app without any Python
  error. `float32` reads each tensor with a plain file read
  (no memory map, so no BF16 pages stay mapped into the process) and widens
  everything once at load (~540 MB for SmolLM2-135M). `lowmem`
  leaves the weights in the memory-mapped file in their stored dtype and
  widens a slab at a time into an 8 MB scratch buffer. That keeps resident
  weight memory to the OS's clean, droppable file cache, at roughly 1.5x (iPhone) to 4-5x (Linux)
  slower decoding before BNNSCopy widening. With it, lowmem decodes faster
  than float32 on an iPhone (see Performance), on Linux still slower. Lowmem requires the memory map. If iOS can't map the
  file, loading fails with a clear error rather than reading gigabytes
  into RAM.
- **BF16 is widened by Accelerate's `BNNSCopy`** when it is available,
  instead of NumPy's shift, with bit-identical results. Each lowmem slab's
  conversion is set up once at load, so a token costs one ctypes call per
  slab. `TINY_GPT_BF16_CONVERT=numpy` turns it off for A/B runs. Native
  16-bit products (`BNNSMatMul`, BNNS fully connected) were also measured
  and were not faster than widening, so they aren't used
  (`tools/native16_bench.py`, `docs/native16_feasibility.md`).
- **Chat reuses the KV cache.** Each turn re-renders the conversation, and
  only the tokens after the longest cached prefix are computed. When the
  context limit is reached, the oldest exchanges are dropped.

## Verification

`tests/test_hf_*.py` (plain `unittest`, NumPy only) compares everything
against fixtures produced by the official libraries
(`tests/make_hf_fixtures.py`, which needs `torch`, `transformers`,
`tokenizers` and `safetensors` on a desktop):

- **Tokenizer**: ids identical to `tokenizers` on 500+ texts (multilingual,
  emoji/ZWJ, digits, odd whitespace, special tokens, random Unicode) and on
  a snapshot of this README. Chat-template text and ids match
  `apply_chat_template` exactly.
- **Chat templates**: the real templates of SmolLM2, SmolLM3, Qwen2.5,
  Qwen3 and Phi-3.5 render byte-for-byte like transformers' renderer, over
  40 cases. These include `enable_thinking=False`, a dated system prompt
  and `{% generation %}`. Templates can't escape the sandbox, and
  `raise_exception` surfaces as a clear error.
- **Model**: three tiny random Llamas (GQA and MHA; biases; untied head;
  llama3 RoPE scaling; BF16/F16/F32) match `transformers` float32 logits to
  ~4e-5 in both weight modes. This holds for prefill at several chunk sizes
  and for cached single-token decoding, and the cached path agrees with an
  uncached recompute of every prefix.
- **Qwen2.5-Coder-1.5B**: tokenizer ids identical to `tokenizers` on
  4,000+ texts, including combining marks whose NFC handling differs
  between Unicode versions (checked under Pythonista's Unicode 13 and
  Python 3.13's 15.1). Logits match `transformers` float32 to ~5e-5, and
  a 40-token greedy completion is identical. A tiny random Qwen2 matches
  to ~3e-5 in both weight modes. The Instruct model's chat answer
  (template, tokenizer, model and repetition penalty together) matches
  transformers' greedy answer token for token.
- **SmolLM2-135M-Instruct**: logits match `transformers` float32 to <1e-4
  at every prompt position, and a 48-token greedy decode is token-for-token
  identical. Inference runs with sockets disabled to prove it's offline.
- **safetensors / BF16**: all 65,536 BF16 bit patterns, files written by the
  `safetensors` library, and a battery of malformed headers.
- **Downloader**: a fake Hub on localhost covers a dropped connection
  resumed via Range, corrupted bytes rejected, cache reuse, offline lookup,
  pickle-only repos refused and zip-import path traversal ignored.
- **BLAS bindings**: `gemv`/`gemm` against NumPy. On Linux,
  `TINY_GPT_CBLAS=/usr/lib/x86_64-linux-gnu/libblas.so.3` routes them
  through a real CBLAS, so the ctypes code is exercised without Accelerate.

```
python -m unittest discover -s tests -p 'test_hf*.py' -t .
```

The SmolLM2 tests skip unless the model is cached.

## Performance

SmolLM2-135M-Instruct, 51-token prompt, greedy decoding.

iPhone 16 Pro, iOS 27.2, Pythonista (Python 3.10, NumPy 1.22, Accelerate),
`hf_run.py selftest`:

| weights | load | first token | decode | footprint |
|---------|------|-------------|--------|-----------|
| float32 | 3.5 s  | 0.07 s | ~79 tok/s (12.7 ms/token) | 742 MB after load, 683 MB generating |
| lowmem  | 0.02 s | 0.07 s (0.93 s cold) | ~88 tok/s (11.3 ms/token) | 135 MB after load, 180 MB generating |

Both produce the same greedy text. Lowmem was ~52 tok/s (19.3 ms/token)
before BF16 widening moved from NumPy to BNNSCopy, and is now faster than
float32 at about a quarter of the memory. `tools/native16_bench.py` shows
why: one token's matrix-vector work takes 15.0 ms with NumPy widening,
8.75 ms with BNNSCopy, and 10.75 ms on float32 weights, which stream twice
the bytes from memory.

On iOS, Accelerate wins every product by 20-180x over Pythonista's
BLAS-less NumPy; `cblas_sgemv` and a one-row `cblas_sgemm` tie.

4-vCPU Linux container (NumPy + OpenBLAS, no Accelerate):

| weights | load | first token | decode | peak RSS |
|---------|------|-------------|--------|----------|
| float32 | 2.4 s | 0.15 s | ~40 tok/s | ~730 MB |
| lowmem  | 0 s   | 0.29 s | ~8.5 tok/s | ~400 MB (mostly clean file cache) |

Decoding is memory-bandwidth bound: profiling shows ~70% of each token in
the matrix-vector products, which stream all 540 MB of float32 weights.
`python hf_run.py selftest` produces these numbers on any device and writes
them to `hf_report.txt`. Its memory line reports usage before and after
load and after generating. On iOS/macOS that figure is `phys_footprint`,
which the OS memory limit uses. Peak RSS also lasts across runs in
Pythonista, because its interpreter stays alive between runs.

## Limitations

- Llama and Qwen2 architectures only (`LlamaForCausalLM`,
  `Qwen2ForCausalLM`, SiLU, no `pretraining_tp`). Others fail with a clear
  `unsupported model` error. So do tokenizers that aren't byte-level BPE
  (SentencePiece, byte fallback) or need a normalizer other than NFC.
- Qwen2 sliding-window attention isn't implemented. The context limit is
  capped at the window instead, which is exact, since the window then
  never takes effect.
- Chat templates are rendered with jinja2 (Pythonista ships it). Without
  jinja2, only plain ChatML templates like SmolLM2's work, and others are
  refused. Tool definitions (`tools=`) aren't passed to templates yet.
- Float32 compute throughout. Weights stay BF16 only in `lowmem` mode, and
  there's no int8/4-bit quantization.
- The context is a hard limit (`CONTEXT_LIMIT`, default 1024). Generation
  stops there rather than sliding, because Llama wasn't trained to slide.
