# tiny_gpt

Train tiny character-level language models from scratch — no PyTorch, no
autograd, no pretrained weights — and run them on anything from a laptop to
an iPhone in Pythonista.

`tiny_gpt.py` is the entry point to a full GPT (multi-head causal
self-attention, real backprop) implemented directly over NumPy in the
`tinygpt/` package. On Apple hardware it routes every matmul through
`cblas_sgemm` on the Accelerate framework's AMX coprocessor; everywhere else
it falls back to plain NumPy and still runs correctly, just slower.

The implementation is deliberately split into small, single-purpose files —
this is meant to be read, not just run. See [Code layout](#code-layout)
below for a map, or open `tinygpt/__init__.py`, which carries the same map
as a module docstring.

![Training tiny_gpt on an iPhone in Pythonista](images/pythonista-ios.png)
*Figure 1 — `tiny_gpt.py` training a model on-device in Pythonista on iOS.*

This guide walks through the three things you'll actually do with this repo,
in order:

1. [Build a dataset](#1-build-a-dataset) with `direct_corpus.py`
2. [Train a model](#2-train-a-model) with `tiny_gpt.py`
3. [Chat with your model](#3-chat-with-your-model) with `chat.py`

Then, if you want to change what a token *is*:
[swapping the tokenizer](#4-swapping-the-tokenizer-a-token-level-math-model).

## Requirements

- Python 3
- `numpy`
- `requests`, only for downloading Hugging Face models (`hf_run.py
  download`), and Pythonista ships it
- Nothing else. On macOS/iOS, `tiny_gpt.py` automatically detects and uses
  the system Accelerate framework for a large speedup; on every other
  platform it silently falls back to NumPy matmuls and produces identical
  results, just slower.

## Git LFS

Large model archives and weight files are tracked with Git LFS. Install Git
LFS and run `git lfs install` before cloning or adding these files; after
cloning, use `git lfs pull` if the LFS objects were not downloaded
automatically. The existing `SmolLM2-135M-Instruct.zip` is already in Git
history as a regular Git blob; converting past commits to LFS requires a
history rewrite and a coordinated force-push.

## Repository layout

| File               | Purpose                                              |
|--------------------|-------------------------------------------------------|
| `direct_corpus.py` | Downloads and assembles instruction/response datasets into a training corpus |
| `math_corpus.py`   | Generates a math worked-solution corpus for the token-level tokenizer |
| `tiny_gpt.py`      | Command-line entry point for training (see [Code layout](#code-layout)) |
| `tinygpt/`         | The implementation: model, training loop, checkpointing, diagnostics |
| `chat.py`          | Loads a trained checkpoint and generates, chats, or evaluates it |
| `hf_run.py`        | Downloads a pretrained Hugging Face model and runs it offline (see [section 5](#5-run-a-pretrained-hugging-face-model)) |

### Code layout

`tinygpt/` is split by concern, one file per piece of the system, so each
can be read (and changed) on its own instead of scrolling through one large
file:

| File                     | Contents                                                    |
|--------------------------|---------------------------------------------------------------|
| `tinygpt/backend.py`     | Accelerate/BLAS bindings — `gemm`, `bmm`, vForce wrappers, the model's working dtype |
| `tinygpt/ops.py`         | Elementwise math — activations, LayerNorm, RMSNorm, RoPE, softmax, each with a hand-written forward *and* backward |
| `tinygpt/model.py`       | `Config` and `GPT` — the forward pass, the backward pass, and both sampling paths (cached and uncached) |
| `tinygpt/optim.py`       | The Adam optimizer |
| `tinygpt/checkpoint.py`  | Saving/loading `.npz` checkpoints |
| `tinygpt/data.py`        | Corpus loading, vocabulary, batching |
| `tinygpt/tokenizer.py`   | Text ↔ ids: the tokenizer protocol and `CharTokenizer` |
| `tinygpt/math_tokenizer.py` | A fixed token-level tokenizer for a generated math corpus |
| `tinygpt/train.py`       | The training loop |
| `tinygpt/diagnostics.py` | `gradcheck`, `gradcheck_all`, `bench`, `profile` |
| `tinygpt/cli.py`         | Argument parsing and mode dispatch |
| `tinygpt/hf/`            | Pretrained-model inference, separate from the training code: Hub downloader, safetensors reader, byte-level BPE tokenizer, Llama forward pass |

A reasonable reading order: start at `model.py` to see the GPT itself, then
`diagnostics.py`'s `gradcheck()` to see how every gradient in `model.py` is
checked against an independent finite difference, then `train.py` to see
how the two are put to use.

`chat.py` only depends on one piece of this: `from tinygpt.checkpoint import
load_checkpoint`. Everything else it does — sampling, the KV cache — lives
in `GPT` itself and is reached through the loaded model object.

## 1. Build a dataset

`direct_corpus.py` downloads public instruction-tuning datasets straight
from their permanent file URLs (no Hugging Face API, no pagination, no rate
limits) and folds them into a single plain-text corpus of `U: ... / A: ...`
exchanges.

List the available sources:

```
python direct_corpus.py list
```

```
sources:

  alpaca-cleaned  52K, community-corrected fork of alpaca. CC BY-NC 4.0.
  dolly           15K human-written. CC BY-SA 3.0. Different register from alpaca.

  all             = alpaca-cleaned + dolly
```

Build a corpus from one or more sources:

```
python direct_corpus.py alpaca-cleaned dolly --out big.txt
```

![direct_corpus.py downloading and assembling a training corpus](images/dataset-build.png)
*Figure 2 — `direct_corpus.py` fetching sources, caching them on disk, and reporting corpus stats.*

A few things worth knowing before you build one:

- **Combining sources is the point.** A single dataset repeats too fast at
  typical training lengths (a 5 MB corpus at 8192 tokens/step for 16,000
  steps is ~26 epochs). Every extra source you mix in cuts the repetition
  proportionally, so `all` is a reasonable default.
- **Downloads are cached and resumable.** Each source is fetched once to a
  local file; re-running the same command reuses the cache, and a
  partial/interrupted download resumes with a `Range` request.
- **Output is ASCII-folded** (smart quotes/dashes folded to plain ASCII,
  everything outside printable ASCII + newline dropped) so the resulting
  vocabulary is always printable ASCII plus `\n` — this keeps corpora
  interchangeable for training and evaluation.
- Useful flags: `--out FILE`, `--max-in N` / `--max-out N` (length filters
  on the U/A turns), `--truncate` (truncate long turns instead of dropping
  them), `--keep-unicode` (skip the ASCII fold), `--no-shuffle`.

## 2. Train a model

`tiny_gpt.py` trains a character-level GPT on the corpus you just built and
periodically checkpoints the best validation loss to a `.npz` file.

### Important: how this script takes its arguments

Because `tiny_gpt.py` is designed to also run inside **Pythonista on iOS**,
where there is no shell and no real `argv` to pass, the bottom of the file
hardcodes its own argument list and **ignores whatever you type after
`python tiny_gpt.py`**:

```python
if __name__ == '__main__':
    sys.argv = [
    'tiny_gpt.py',
    'train',
    'big.txt',
    '--steps', '3600',
    '--B', '32',
    '--block-size', '256',
    ...
    ]
```

To configure your own training run, **edit this list directly** (corpus
file, step count, model size, architecture flags, checkpoint name) rather
than passing command-line flags — whatever you pass on the actual command
line is discarded. This is the one thing that trips people up on a first
read of the file.

The `sys.argv` overwrite happens unconditionally, before the code that reads
`sys.argv[1]` to pick a mode — so this hardcoding isn't specific to `train`,
it pins **every** mode. `python tiny_gpt.py gradcheck --rope 0` still runs
whatever is in the hardcoded list (`train` by default), not `gradcheck`. To
run `gradcheck`, `bench`, or `profile` with real command-line flags, comment
out or remove the `sys.argv = [...]` block first.

### Architecture flags

Six independently-switchable levers, all on by default in the hardcoded
block above:

| Flag                 | Effect                                                   |
|-----------------------|----------------------------------------------------------|
| `--rope 1`            | rotary position embeddings instead of a learned `wpe`    |
| `--qk-norm 1`          | non-parametric RMSNorm on q/k, applied after RoPE         |
| `--act relu2`          | squared ReLU instead of tanh-GELU                        |
| `--zero-init 1`        | output projections and the head start at zero             |
| `--value-residual 1`   | per-layer learnable shortcut back to layer-0 values       |
| `--softcap 30`         | `c*tanh(logits/c)` before cross-entropy                   |

Setting all six off exactly reproduces the original (pre-lever)
architecture, and older checkpoints load that way automatically.

### Running it

Once the hardcoded block points at your corpus and settings:

```
python tiny_gpt.py
```

Training prints a step/loss/val table and checkpoints the best validation
loss to the `--ckpt` path (default `model.npz`) every `--ckpt-every` steps,
along with a short sample generation at the end of the run.

![tiny_gpt.py training loop output](images/training-run.png)
*Figure 3 — a training run in progress: step, learning rate, train loss, validation loss, and throughput.*

Resuming a run continues the same cosine LR schedule instead of re-warming
a converged model — set `resume` to a prior checkpoint path.

Two extra modes are useful before committing to a long run — `gradcheck`
(finite-difference check of every gradient), `bench` (throughput by phase,
numpy vs. Accelerate), and `profile` (per-operation timing breakdown). Since
the hardcoded `sys.argv` block always wins, run them by editing that block
to `'gradcheck'` / `'bench'` / `'profile'` (and adjusting or removing the
`train`-only flags below it) rather than passing the mode on the command
line.

#### Example training log (placeholder)

<!-- TODO: replace with a real captured log from a full training run -->

```
step         lr      loss       val   ms/step   elapsed
0      5.3e-06    4.2891    4.2814     412.3      0.4s
250    1.3e-04    2.1043    2.0877     398.1    103.2s
500    2.7e-04    1.7822    1.7691     395.6    206.8s
750    3.9e-04    1.5964    1.6103     396.9    310.9s
1000   4.7e-04    1.4881    1.5240     397.4    414.6s
...
3600   5.3e-05    1.2703    1.2892     396.2   1493.0s

3600 steps, 7,549,747,200 tokens in 1493.0 s  (5057 tokens/s)
saved    modelv3.npz  (best val 1.2521)
```

## 3. Chat with your model

`chat.py` loads a checkpoint produced by `tiny_gpt.py` and generates from
it, using the same KV-cached generation path as training (~12x faster than
recomputing the full context per token).

```
python chat.py info model.npz                       # what's in the checkpoint
python chat.py ask model.npz "why is the sky blue?"  # single question, single answer
python chat.py chat model.npz                        # interactive REPL
python chat.py sample model.npz --n 800              # free-running generation, no prompt
python chat.py eval model.npz held_out.txt           # loss / perplexity / bits-per-char
```

Sampling knobs, available on `ask`, `chat`, and `sample`: `--temp`
(0 = greedy, below 1 sharpens, above 1 flattens), `--top-k`, `--top-p`
(nucleus), `--seed`.

Inside `chat`, `/temp 0.2`, `/topk 20`, `/seed 42`, and `/n 20` adjust those
knobs mid-conversation.

![chat.py interactive REPL session](images/chat-repl.png)
*Figure 4 — `chat.py chat model.npz`, an interactive session with a trained checkpoint.*

Set expectations accordingly: this is a character-level model, typically
well under a million parameters. It learns spelling, the `U:`/`A:` turn
structure, sentence rhythm, and answer shape — it does not learn to reason
or answer questions correctly. Judge output on whether it reads like English
dialogue, not on whether it's right.

#### Example `chat.py` session (placeholder)

<!-- TODO: replace with real output from `python chat.py chat model.npz` -->

```
$ python chat.py chat model.npz
1,847,392 params, 4 layers, 4 heads, 128 dim, block_size 256
vocab 97, trained 3600 steps
val loss 1.2521  (perplexity 3.5, 1.81 bits/char)

Type a message. Blank line or Ctrl-D to quit.
Commands: /temp 0.2  /topk 20 /seed 42  /n 20

U: why is the sky blue?
A: The sky appears blue because of the way sunlight is scattered by the
   atmosphere, with shorter blue wavelengths scattering more than other
   colors.

[241 chars in 0.31s, 778 chars/s]

U: /temp 0.3
  temp = 0.3

U: give me a one sentence summary of photosynthesis
A: Photosynthesis is the process by which plants convert sunlight, water,
   and carbon dioxide into glucose and oxygen.

[189 chars in 0.24s, 787 chars/s]
```

To measure a checkpoint quantitatively rather than by eye, run it against a
held-out text file:

```
python chat.py eval model.npz held_out.txt
```

This reports loss, perplexity, and bits-per-token, and warns if the file
contains characters outside the model's training vocabulary.

## 4. Swapping the tokenizer: a token-level math model

Everything above is character-level: one id per character, vocabulary taken
from whatever the corpus happened to contain. That is a choice, not a
constraint of the model — `GPT` only ever sees integer ids and a vocabulary
size, so a different tokenizer needs no changes in `tinygpt/model.py` at all.

`tinygpt/math_tokenizer.py` is the other extreme: a **fixed** vocabulary of
111 tokens for a generated math corpus. Whole phrases collapse into single
ids, digits stay separate, and spaces are discarded entirely:

```
U: Solve for x: 3(x - 7) - 8 = -23      ->  <u><solve_for>x:3(x-7)-8=-23
```

Every token is ASCII, which is a constraint from the keyboard rather than
the maths: this corpus gets written and read on an iOS device, where `*` is
one tap and `×` is a trip through a symbol palette. So multiplication is
`*`, dividing one fraction by another is parenthesised — `(2/3) / (4/5)` —
and "about equal" is `~=`.

`math_corpus.py` generates the corpus it expects:

```
python math_corpus.py list                     # the 22 problem kinds
```

The kinds span one-step arithmetic (signed integers, subtracting a negative,
decimals, estimation by rounding, multiplication as repeated addition)
through fraction work (common denominators, multiplying, dividing by the
reciprocal, cancelling) and multi-step algebra (two-step, collecting like
terms, variables on both sides, distributing, factoring, difference of
squares, inequalities that flip when divided by a negative) to notation that
needs more than arithmetic: factorials, summations in closed form, piecewise
branch selection, derivatives by the power rule, and definite integrals.

Generation prints which vocabulary tokens no kind reaches yet. That list is
the honest to-do list: right now only `<pad>`, `<bos>`, `<eos>` and `<unk>`
are unused, and those four are unreachable by design because the
flat-stream trainer has no sequence boundaries to mark.

### Worked example: training a tiny math GPT

End to end, four steps. The output below is from a real run on a
laptop-class container with no Accelerate — slower than an iPhone, which
gets the AMX path.

**1. Generate the corpus.** The seed defaults to 0, so these numbers are
exactly what you will see:

```
$ python math_corpus.py --n 4000 --out math.txt
kinds    decimals, derivative, distribute, divide, estimate, factorial, ...
wrote    math.txt
4000 exchanges, 702439 chars, 249345 tokens (2.82 chars/token)
vocab    107 of 111 tokens occur
unused   <pad> <bos> <eos> <unk>
```

**2. Point `tiny_gpt.py` at it.** Command-line flags are discarded, so the
hardcoded `sys.argv` block *is* the config (see
[above](#important-how-this-script-takes-its-arguments)):

```python
    sys.argv = [
    'tiny_gpt.py',
    'train',
    'math.txt',
    '--steps', '1200',
    '--B', '16',
    '--block-size', '96',
    '--n-layer', '3',
    '--n-head', '4',
    '--n-embd', '96',
    '--rope', '1',
    '--qk-norm', '1',
    '--act', 'relu2',
    '--zero-init', '1',
    '--value-residual', '1',
    '--softcap', '30',
    '--tokenizer', 'math',
    '--ckpt', 'math.npz',
    '--lr', '0.003',
    '--ckpt-every', '200',
    ]
```

`--block-size 96` on purpose: exchanges average about 62 tokens, so 96
covers a whole problem plus a little context. The 256 in the shipped block
is sized for the alpaca corpus and would be mostly wasted here.

**3. Train.**

```
$ python tiny_gpt.py
corpus   math.txt
tokens   236878 train / 12467 val   vocab 111  (math, 2.82 chars/token)
model    357026 params  (3 layers, 4 heads, 96 dim, block 96)
arch     rope=True qk_norm=True act=relu2 zero_init=True value_residual=True softcap=30
tokens/step 1536   accelerate False   bmm numpy   vForce False
checkpoint math.npz every 200 steps

step            lr      loss       val   ms/step   elapsed
0       3.0000e-05    4.7095    4.7085    2154.5      2.2s
200     2.9453e-03    0.8711    0.9461      89.4     18.0s
400     2.5341e-03    0.7088    0.6714      83.5     33.5s
600     1.8421e-03    0.4787    0.5827      81.9     49.2s
800     1.0892e-03    0.4554    0.5066      80.8     64.7s
1000    5.1431e-04    0.4408    0.4571      80.5     80.6s
1199    3.0001e-04    0.3854    0.4330      80.2     96.2s
```

Check the header before walking away: it should say `(math, 2.82
chars/token)` and `vocab 111`. If it says `char`, step 2 didn't take.

**4. Chat with it.** The tokenizer travels inside the checkpoint, so
`chat.py` needs no flag:

```
$ python chat.py info math.npz
357,026 params, 3 layers, 4 heads, 96 dim, block_size 96
vocab 111 (math tokenizer), trained 1200 steps
val loss 0.4330  (perplexity 1.5, 0.62 bits/token)

$ python chat.py ask math.npz "Solve for x: 5x + 4 = 19" --temp 0.2
A: <step>1:<subtract>4<from><both_sides>,<so>5x=15<nl>
   <step>2:<divide><both_sides><by>5,<so>x=3<nl>
   <final_answer>:x=3

$ python chat.py ask math.npz "Compute 5!" --temp 0.2
A: <step>1:5<factorial><is><the><same_as>5*4*3*2*1<nl>
   <step>2:<multiply>5*4=20<nl><step>3:<multiply>20*3=60<nl>
   <step>4:<multiply>60*2=120<nl><step>5:<multiply>120*1=120<nl>
   <final_answer>:120
```

Both of those are right, and it is worth knowing they are not
representative. The same checkpoint answers `Compute 4 * 7` with `21` and
`Compute 2/3 + 1/4` with `12/12`. Ninety-six seconds of training buys the
scaffolding, not the arithmetic — see the last bullet below.

![A token-level math model answering in chat.py](images/math-tokenizer-chat.png)
*Figure 5 — the math model's reply, shown in the tokenizer's compact form: one id per lexeme, no spaces.*

**Scaling up.** Generation is free here, which changes the sizing calculus
versus `direct_corpus.py`: there the corpus is capped by the dataset, here
another zero costs nothing. `--n 4000` is ~250k tokens, which at 4096
tokens/step for 3000 steps would be ~49 epochs of the same problems. Prefer
`--n 20000` (~1.25M tokens, ~10 epochs) with `--B 32 --block-size 128
--n-layer 4 --n-embd 128 --lr 0.002 --steps 3000`. Expect roughly half an
hour off-device at that size, considerably less on Apple silicon. Add
`'--resume', 'math.npz'` to continue a run — the cosine schedule picks up
where it stopped instead of re-warming.

**A clean eval set** is one command, since the problems are randomly
generated:

```
python math_corpus.py --n 2000 --seed 99 --out heldout.txt
python chat.py eval math.npz heldout.txt
```

Different problems from the same templates, which is worth more than the
built-in 5% tail split when comparing two checkpoints.

Three things are worth noticing when you run this:

- **The corpus and the tokenizer are one contract.** `strict=True` means the
  tokenizer raises on any word it was never taught, so `math_corpus.py`
  tokenizes the entire corpus before writing it. A template using an unknown
  word fails at generation time, naming the word, instead of silently
  becoming `<unk>` or failing mid-training. Adding a kind that reuses
  existing lexemes is therefore free; teaching the tokenizer a *new* word
  renumbers the vocabulary and invalidates every existing checkpoint.
- **Bits-per-token is not bits-per-character.** A token-level model predicts
  fewer, larger units (~2.8 chars/token here), so its loss is not comparable
  to a character model's. `chat.py eval` prints both for exactly this reason.
- **Structure is learned long before arithmetic.** After a few hundred steps
  the model reproduces every template perfectly — step numbering, turn
  markers, the shape of each solution — while the numbers inside stay wrong
  (`6*4=10`, `Round 47 to 70`). The scaffolding is a much easier distribution
  than the computation it describes. Individual facts do get memorised where
  the answer table is small — `3!` and `4!` come out right — but that is
  recall, not a rule: the same checkpoint gets `d/dx 9x^5` half right
  (`45x^3`, correct coefficient, wrong exponent) and `d/dx 2x^4` wrong.

Out-of-language input is refused rather than mangled, because a fixed
vocabulary has no honest way to represent it:

```
$ python chat.py ask math.npz "why is the sky blue?"
the math tokenizer has no rule for: whyskblue?
it only knows the language its corpus is generated in.
```

## 5. Run a pretrained Hugging Face model

`hf_run.py` downloads a pretrained model straight from the Hugging Face Hub
and runs it entirely on-device, on NumPy and Accelerate. There's no
inference API, no PyTorch and no `transformers`. This is inference only.
Nothing is trained, and the training code above is untouched.

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

### In Pythonista

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

### From a shell

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

### What is in `tinygpt/hf/`

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
  iPhone, float32 was 1.5x faster at 135M params and no faster at 360M,
  while using 1.8 GB more memory. Explicit float32 is refused on iOS when
  it would need over 2.5 GB, because iOS kills the app without any Python
  error. `float32` reads each tensor with a plain file read
  (no memory map, so no BF16 pages stay mapped into the process) and widens
  everything once at load (~540 MB for SmolLM2-135M). `lowmem`
  leaves the weights in the memory-mapped file in their stored dtype and
  widens a slab at a time into an 8 MB scratch buffer. That keeps resident
  weight memory to the OS's clean, droppable file cache, at roughly 1.5x (iPhone) to 4-5x (Linux)
  slower decoding. Lowmem requires the memory map. If iOS can't map the
  file, loading fails with a clear error rather than reading gigabytes
  into RAM.
- **Chat reuses the KV cache.** Each turn re-renders the conversation, and
  only the tokens after the longest cached prefix are computed. When the
  context limit is reached, the oldest exchanges are dropped.

### Verification

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

### Performance

SmolLM2-135M-Instruct, 51-token prompt, greedy decoding.

iPhone, Pythonista (Python 3.10, NumPy 1.22, Accelerate):

| weights | load | first token | decode | footprint (Pythonista alone ~90-120 MB) |
|---------|------|-------------|--------|-----------------------------------------|
| float32 | 3.6 s  | 0.07 s | ~78 tok/s (12.8 ms/token) | 727 MB after load, 661 MB generating |
| lowmem  | 0.01 s | 0.075 s (0.6-0.85 s cold) | ~52 tok/s (19.3 ms/token) | 119 MB after load, 133 MB generating |

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

### Limitations

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

## License

Apache License 2.0 — see [LICENSE](LICENSE).
