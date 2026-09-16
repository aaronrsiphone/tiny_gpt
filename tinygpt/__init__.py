# coding: utf-8
'''
tinygpt -- a character-level GPT trained from scratch on top of NumPy.

Real backprop, real multi-head causal self-attention, no autograd, no
PyTorch, no pretrained weights. On Apple hardware every matmul routes
through cblas_sgemm on the Accelerate framework's AMX coprocessor;
everywhere else the same code runs, unchanged, on plain NumPy.

The implementation is split by concern so each piece can be read (and
changed) on its own:

    backend.py       Accelerate/BLAS bindings: gemm, bmm, vForce, dtype
    ops.py            Elementwise math: activations, LayerNorm, RMSNorm,
                      RoPE, softmax -- each with a hand-written forward
                      *and* backward
    model.py          Config and GPT: the forward pass, the backward pass,
                      and both sampling paths (cached and uncached)
    optim.py          The Adam optimizer
    checkpoint.py     Saving/loading .npz checkpoints
    data.py           Corpus loading, vocabulary, batching
    train.py          The training loop
    diagnostics.py    gradcheck, gradcheck_all, bench, profile
    cli.py            Argument parsing and mode dispatch

Suggested reading order for a newcomer: model.py first, to see the GPT
itself; then diagnostics.py's gradcheck(), to see how every gradient in
model.py is checked against an independent finite difference; then
train.py, to see how the two are put to use.
'''

from tinygpt.backend import HAVE_ACCELERATE, set_dtype
from tinygpt.checkpoint import CKPT_VERSION, load_checkpoint, save_checkpoint
from tinygpt.data import batcher, load_corpus, make_vocab
from tinygpt.diagnostics import bench, gradcheck, gradcheck_all, profile
from tinygpt.model import GPT, LEGACY_FLAGS, Config
from tinygpt.optim import Adam
from tinygpt.train import estimate_loss, model_min_val, train

__all__ = [
    'Config', 'GPT', 'LEGACY_FLAGS', 'Adam',
    'CKPT_VERSION', 'save_checkpoint', 'load_checkpoint',
    'load_corpus', 'make_vocab', 'batcher',
    'train', 'estimate_loss', 'model_min_val',
    'gradcheck', 'gradcheck_all', 'bench', 'profile',
    'HAVE_ACCELERATE', 'set_dtype',
]
