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
- Nothing else. On macOS/iOS, `tiny_gpt.py` automatically detects and uses
  the system Accelerate framework for a large speedup; on every other
  platform it silently falls back to NumPy matmuls and produces identical
  results, just slower.

## Repository layout

| File               | Purpose                                              |
|--------------------|-------------------------------------------------------|
| `direct_corpus.py` | Downloads and assembles instruction/response datasets into a training corpus |
| `math_corpus.py`   | Generates a math worked-solution corpus for the token-level tokenizer |
| `tiny_gpt.py`      | Command-line entry point for training (see [Code layout](#code-layout)) |
| `tinygpt/`         | The implementation: model, training loop, checkpointing, diagnostics |
| `chat.py`          | Loads a trained checkpoint and generates, chats, or evaluates it |

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

Generate a corpus and train on it:

```
python math_corpus.py list                     # the 22 problem kinds
python math_corpus.py --n 4000 --out math.txt
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

Then set `'--tokenizer', 'math'` and `'math.txt'` in `tiny_gpt.py`'s
hardcoded argv block (see [above](#important-how-this-script-takes-its-arguments))
and run it. Chatting works exactly as before — the tokenizer travels inside
the checkpoint, so `chat.py` needs no flag:

```
python chat.py ask math.npz "Solve for x: 5x + 4 = 19"
```

![A token-level math model answering in chat.py](images/math-tokenizer-chat.png)
*Figure 5 — the math model's reply, shown in the tokenizer's compact form: one id per lexeme, no spaces.*

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

#### Example math session (placeholder)

<!-- TODO: replace with real output from a trained math checkpoint -->

```
$ python chat.py info math.npz
352,418 params, 3 layers, 4 heads, 96 dim, block_size 96
vocab 111 (math tokenizer), trained 3600 steps
val loss 0.2104  (perplexity 1.2, 0.30 bits/token)

$ python chat.py ask math.npz "Solve for x: 5x + 4 = 19" --temp 0.3
A: <step>1:<subtract>4<from><both_sides>,<so>5x=15<nl>
   <step>2:<divide><both_sides><by>5,<so>x=3<nl>
   <final_answer>:x=3
```

Out-of-language input is refused rather than mangled, because a fixed
vocabulary has no honest way to represent it:

```
$ python chat.py ask math.npz "why is the sky blue?"
the math tokenizer has no rule for: whyskblue?
it only knows the language its corpus is generated in.
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
