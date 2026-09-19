# coding: utf-8
'''
train big.txt --steps 3700 --block-size 256 --n-embd 128 --n-layer 4 --seed 42 --ckpt base192.npz --rope 0 --qk-norm 0 --act gelu --zero-init 0 --value-residual 0 --softcap 0


A character-level GPT trained from scratch, on an iPhone, in Pythonista.

Real backprop. Real multi-head causal self-attention. No autograd, no
PyTorch, no pretrained weights. Every matmul goes through cblas_sgemm on
Apple's AMX coprocessor, measured at 1763 GFLOP/s on an A18 Pro.

This file is just the command-line entry point. The implementation lives
in the tinygpt/ package next to this file, split by concern -- see
tinygpt/__init__.py for a map of what's where and a suggested reading
order.

    python tiny_gpt.py gradcheck       finite-difference every gradient
    python tiny_gpt.py train [file]    train, then sample
    python tiny_gpt.py bench           throughput by phase
    python tiny_gpt.py profile         per-op timing

Six architecture levers, each independently switchable, all ON by default:

    --rope 1            rotary position embeddings instead of learned wpe
    --qk-norm 1         non-parametric RMSNorm on q and k, after RoPE
    --act relu2         squared ReLU instead of tanh-GELU
    --zero-init 1       output projections and head start at zero
    --value-residual 1  per-layer learnable shortcut to layer-0 values
    --softcap 30        c*tanh(logits/c) before cross-entropy

Set any of them to 0 (or --act gelu) to A/B a single change against the
baseline. All six off reproduces the previous architecture exactly, and
version-1 checkpoints load that way automatically.

--tokenizer picks how text becomes ids: 'char' (one id per character, the
default) or 'math' (one id per lexeme of a generated math corpus, see
math_corpus.py). The model is identical either way -- only the vocabulary
and what one id is worth change.

Falls back to numpy when Accelerate is absent, so the same file runs and
verifies off-device. Only the GEMM binding differs.

IMPORTANT -- how this file actually gets its arguments: the sys.argv
assignment below runs unconditionally, *before* tinygpt.cli.main() ever
reads sys.argv[1] to pick a mode. That means it overrides every mode
(train, gradcheck, bench, profile alike), not just the "train" example it
happens to be set to. This exists because this script also has to run
inside Pythonista on iOS, where there is no shell and no real argv to
read: to configure a run -- which mode, which corpus, how many steps,
which architecture flags -- edit the list below directly, rather than
passing flags on an actual command line, which will be silently ignored.
'''

import sys

from tinygpt.cli import main

if __name__ == '__main__':
    sys.argv = [
    'tiny_gpt.py',
    'train',
    'big.txt',
    '--steps', '3600',
    '--B', '32',
    '--block-size', '256',
    '--n-layer', '4',
    '--n-head', '4',
    '--n-embd', '128',
    '--rope', '1',
    '--qk-norm', '1',
    '--act', 'relu2',
    '--zero-init', '0',
    '--value-residual', '0',
    '--softcap', '0',
    '--tokenizer', 'char',
    '--ckpt', 'modelv3.npz',
    '--lr', '0.00053'
    ]
    main()
