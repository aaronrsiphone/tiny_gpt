# coding: utf-8
'''
Command-line entry points: argument parsing and mode dispatch.

    python tiny_gpt.py gradcheck       finite-difference every gradient
    python tiny_gpt.py gradcheck-all   gradcheck the baseline plus every lever
    python tiny_gpt.py train [file]    train, then sample
    python tiny_gpt.py bench           throughput by phase
    python tiny_gpt.py profile         per-op timing

See tiny_gpt.py at the repository root for why these flags usually don't
reach this file: it hardcodes its own sys.argv for Pythonista, where there
is no shell and no real command line to read from.
'''

import sys

from tinygpt.diagnostics import bench, gradcheck, gradcheck_all, profile
from tinygpt.train import train

_STR_KEYS = ('ckpt', 'resume', 'act', 'tokenizer')


def parse_kv(args, start=0):
    """--key value pairs. 'act' holds a string that would parse as a float
    if it were not excluded: both 'gelu' and 'relu2' contain an 'e'."""
    kw = {}
    i = start
    while i < len(args) - 1:
        key = args[i].lstrip('-').replace('-', '_')
        val = args[i + 1]
        if key in _STR_KEYS:
            kw[key] = val
        else:
            kw[key] = (float(val) if '.' in val or 'e' in val.lower()
                       else int(val))
        i += 2
    return kw


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'gradcheck'
    if mode == 'gradcheck':
        ok = gradcheck(**parse_kv(sys.argv, 2))
        sys.exit(0 if ok else 1)
    elif mode == 'gradcheck-all':
        sys.exit(0 if gradcheck_all() else 1)
    elif mode == 'train':
        args = sys.argv[2:]
        path = args[0] if args and not args[0].startswith('-') else None
        train(path, **parse_kv(args, 1 if path else 0))
    elif mode == 'bench':
        bench(**parse_kv(sys.argv, 2))
    elif mode == 'profile':
        profile(**parse_kv(sys.argv, 2))
    else:
        raise SystemExit('modes: gradcheck | gradcheck-all | train [file] '
                         '| bench | profile')
