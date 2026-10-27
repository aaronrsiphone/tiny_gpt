# coding: utf-8
'''Shared helpers for the tests/test_hf_*.py suites.'''

import base64
import json
import os
import struct
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, 'fixtures')
SMOLLM2 = 'HuggingFaceTB/SmolLM2-135M-Instruct'
QWEN = 'Qwen/Qwen2.5-Coder-1.5B'
QWEN_INSTRUCT = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'


def fixture(name):
    with open(os.path.join(FIX, name), 'rb') as f:
        return json.loads(f.read().decode('utf-8'))


def b64_array(s, shape=None, dtype=np.float32):
    a = np.frombuffer(base64.b64decode(s), dtype=dtype)
    return a.reshape(shape) if shape is not None else a


def tiny_model_dir(name):
    '''Write a tiny_llama_*.json fixture out as config.json +
    model.safetensors; return (directory, fixture dict).'''
    d = fixture(name)
    td = tempfile.mkdtemp(prefix='tinygpt_hf_')
    with open(os.path.join(td, 'model.safetensors'), 'wb') as f:
        f.write(base64.b64decode(d['safetensors']))
    with open(os.path.join(td, 'config.json'), 'w') as f:
        json.dump(d['config'], f)
    return td, d


def write_safetensors(path, tensors, header_override=None):
    '''Minimal writer: tensors is {name: (dtype_str, ndarray)}. Used to
    build shards and deliberately broken files; the parser itself is
    checked against files written by the official library.'''
    header = {}
    blobs = []
    off = 0
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {'dtype': dt, 'shape': list(arr.shape),
                        'data_offsets': [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if header_override is not None:
        header = header_override
    h = json.dumps(header).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    return path


def cached(repo_id):
    '''The cached snapshot of a model, or None (its tests then skip).'''
    try:
        from tinygpt.hf import hub
        return hub.find_local(repo_id)
    except Exception:
        return None


def smollm2_dir():
    return cached(SMOLLM2)


def qwen_dir():
    return cached(QWEN)


def qwen_instruct_dir():
    return cached(QWEN_INSTRUCT)
