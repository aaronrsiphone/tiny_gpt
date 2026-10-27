# coding: utf-8
'''
A pure-Python safetensors reader: struct + json + NumPy + mmap.

The format is an 8-byte little-endian header length, a JSON header naming
each tensor's dtype, shape and [begin, end) byte range within the data
section that follows, and then the raw little-endian tensor bytes. There
is no code in the file and nothing to unpickle, which is the point.

The file is never read whole. By default it is memory-mapped and get_raw()
returns a NumPy view straight onto the mapped bytes, so asking for one
tensor touches only that tensor's pages. With use_mmap=False each tensor
is instead read into its own array, which is better when every tensor is
about to be converted and discarded anyway: nothing stays mapped into the
process afterwards. Everything in the header is validated -- dtype, shape, byte length,
bounds, overlap -- before any tensor bytes are interpreted.

BF16 has no NumPy dtype. It is exposed as uint16 raw bits, and
bf16_to_f32() widens it: a bfloat16 is exactly the top half of a float32,
so shifting the bits left by 16 is an exact conversion. Where Accelerate is
available the shift is done by BNNSCopy instead of NumPy (see backend.py),
with identical results.
'''

import json
import mmap
import os
import struct

import numpy as np

from tinygpt import backend

DTYPES = {
    'F32': np.dtype('<f4'),
    'F16': np.dtype('<f2'),
    'BF16': np.dtype('<u2'),   # raw bits; see bf16_to_f32()
}

# A real header is a few hundred KB at most. Anything claiming more is
# corrupt or hostile, and json.loads() on it would be a memory bomb.
MAX_HEADER = 64 << 20


class SafetensorsError(ValueError):
    pass


def bf16_to_f32(raw, out=None):
    '''Widen bfloat16 bits (uint16 array) to float32, exactly.'''
    raw = np.asarray(raw)
    if raw.dtype != np.uint16:
        raise TypeError('bf16_to_f32 wants uint16 bits, got %s' % raw.dtype)
    if out is None:
        out = np.empty(raw.shape, np.float32)
    elif out.shape != raw.shape or out.dtype != np.float32:
        raise ValueError('bf16_to_f32: out must be float32 %r' % (raw.shape,))
    w = backend.bf16_widener(raw, out)
    if w is not None:
        w()
        return out
    np.left_shift(raw, 16, out=out.view(np.uint32), dtype=np.uint32,
                  casting='unsafe')
    return out


def to_f32(raw, dtype, out=None):
    '''Convert a tensor as returned by get_raw() to float32.'''
    if dtype == 'BF16':
        return bf16_to_f32(raw, out)
    if out is None:
        return np.array(raw, dtype=np.float32)
    out[...] = raw
    return out


def widener(raw, dtype, out):
    '''A zero-argument function that converts raw into out (float32) every
    time it is called. For a conversion repeated on every token, as lowmem
    mode does: the BF16 fast path's setup is paid here, once.'''
    if dtype == 'BF16':
        w = backend.bf16_widener(raw, out)
        if w is not None:
            return w
    return lambda: to_f32(raw, dtype, out)


class TensorInfo(object):
    __slots__ = ('name', 'dtype', 'shape', 'begin', 'end')

    def __init__(self, name, dtype, shape, begin, end):
        self.name, self.dtype, self.shape = name, dtype, shape
        self.begin, self.end = begin, end

    @property
    def nbytes(self):
        return self.end - self.begin

    def __repr__(self):
        return 'TensorInfo(%r, %s, %r)' % (self.name, self.dtype, self.shape)


def parse_header(blob, data_len, path='<buffer>'):
    '''Validate a decoded header dict against the data section size.

    Returns (tensors: {name: TensorInfo}, metadata: dict).
    '''
    if not isinstance(blob, dict):
        raise SafetensorsError('%s: header is not a JSON object' % path)
    meta = blob.pop('__metadata__', None) or {}
    tensors = {}
    for name, d in blob.items():
        if not isinstance(d, dict):
            raise SafetensorsError('%s: entry %r is not an object'
                                   % (path, name))
        dt = d.get('dtype')
        shape = d.get('shape')
        offs = d.get('data_offsets')
        if dt not in DTYPES:
            raise SafetensorsError('%s: tensor %r has dtype %r; supported: '
                                   '%s' % (path, name, dt,
                                           ', '.join(sorted(DTYPES))))
        if (not isinstance(shape, list)
                or not all(type(s) is int and s >= 0 for s in shape)):
            raise SafetensorsError('%s: tensor %r has bad shape %r'
                                   % (path, name, shape))
        if (not isinstance(offs, list) or len(offs) != 2
                or not all(type(o) is int for o in offs)):
            raise SafetensorsError('%s: tensor %r has bad data_offsets %r'
                                   % (path, name, offs))
        b, e = offs
        n = 1
        for s in shape:
            n *= s
        want = n * DTYPES[dt].itemsize
        if not (0 <= b <= e <= data_len):
            raise SafetensorsError('%s: tensor %r bytes [%d, %d) fall outside '
                                   'the %d-byte data section'
                                   % (path, name, b, e, data_len))
        if e - b != want:
            raise SafetensorsError('%s: tensor %r is %d bytes but shape %r '
                                   '%s needs %d' % (path, name, e - b, shape,
                                                    dt, want))
        tensors[name] = TensorInfo(name, dt, tuple(shape), b, e)
    spans = sorted((t.begin, t.end, t.name) for t in tensors.values()
                   if t.end > t.begin)
    for (b0, e0, n0), (b1, e1, n1) in zip(spans, spans[1:]):
        if b1 < e0:
            raise SafetensorsError('%s: tensors %r and %r overlap'
                                   % (path, n0, n1))
    return tensors, meta


class SafeTensorsFile(object):
    '''One .safetensors file, memory-mapped read-only.

    use_mmap: True maps the file and falls back to reading tensors one by
    one if mapping fails; 'require' raises instead (lowmem mode, where a
    silent fallback would pull the whole model into RAM); False never
    maps.
    '''

    def __init__(self, path, use_mmap=True):
        self.path = path
        self._f = open(path, 'rb')
        self._mm = None
        try:
            self._open()
        except Exception:
            self.close()
            raise
        if use_mmap and self._size > 0:
            try:
                self._mm = mmap.mmap(self._f.fileno(), 0,
                                     access=mmap.ACCESS_READ)
            except (OSError, ValueError) as e:
                if use_mmap == 'require':
                    self.close()
                    raise SafetensorsError(
                        '%s: could not memory-map the %d MB file (%s); '
                        'lowmem mode needs a memory map'
                        % (path, self._size >> 20, e))
                self._mm = None   # fall back to seek+read per tensor

    def _open(self):
        path = self.path
        size = self._size = os.fstat(self._f.fileno()).st_size
        head = self._f.read(8)
        if len(head) != 8:
            raise SafetensorsError('%s: too short to be safetensors' % path)
        (hlen,) = struct.unpack('<Q', head)
        if hlen > MAX_HEADER or 8 + hlen > size:
            raise SafetensorsError('%s: header length %d is implausible for '
                                   'a %d-byte file' % (path, hlen, size))
        raw = self._f.read(hlen)
        try:
            blob = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, ValueError) as e:
            raise SafetensorsError('%s: header is not valid JSON: %s'
                                   % (path, e))
        self.data_start = 8 + hlen
        self.data_len = size - self.data_start
        self.tensors, self.metadata = parse_header(blob, self.data_len, path)

    def keys(self):
        return list(self.tensors)

    def __contains__(self, name):
        return name in self.tensors

    def info(self, name):
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError('%s: no tensor %r' % (self.path, name))

    def get_raw(self, name):
        '''The tensor in its stored dtype (BF16 as uint16 bits), shaped.

        With mmap this is a read-only view onto the file: no copy, and
        only the pages actually touched are ever read from flash.
        '''
        t = self.info(name)
        dt = DTYPES[t.dtype]
        count = t.nbytes // dt.itemsize
        if self._mm is not None:
            a = np.frombuffer(self._mm, dtype=dt, count=count,
                              offset=self.data_start + t.begin)
        else:
            a = np.empty(count, dt)
            self._f.seek(self.data_start + t.begin)
            got = self._f.readinto(memoryview(a).cast('B'))
            if got != t.nbytes:
                raise SafetensorsError('%s: short read on %r' % (self.path,
                                                                  name))
        return a.reshape(t.shape)

    def get_f32(self, name, out=None):
        '''The tensor as a new (or the given) float32 array.'''
        return to_f32(self.get_raw(name), self.info(name).dtype, out)

    def close(self):
        # Views handed out by get_raw() keep the map alive; closing it from
        # under them would crash, so just drop our reference.
        self._mm = None
        if self._f is not None:
            self._f.close()
            self._f = None


class Checkpoint(object):
    '''A model's weights: one file, or shards plus an index.json.

    Presents the union of tensors behind a single get_raw()/get_f32().
    '''

    def __init__(self, directory, use_mmap=True):
        self.files = {}
        self.where = {}
        single = os.path.join(directory, 'model.safetensors')
        index = os.path.join(directory, 'model.safetensors.index.json')
        if os.path.exists(index):
            with open(index, 'rb') as f:
                idx = json.loads(f.read().decode('utf-8'))
            wm = idx.get('weight_map')
            if not isinstance(wm, dict):
                raise SafetensorsError('%s: no weight_map' % index)
            for name, shard in wm.items():
                if (os.path.basename(shard) != shard
                        or not shard.endswith('.safetensors')):
                    raise SafetensorsError('%s: bad shard name %r'
                                           % (index, shard))
                if shard not in self.files:
                    self.files[shard] = SafeTensorsFile(
                        os.path.join(directory, shard), use_mmap)
                if name not in self.files[shard]:
                    raise SafetensorsError('%s: index says %r is in %s, '
                                           'but it is not' % (index, name,
                                                              shard))
                self.where[name] = self.files[shard]
        elif os.path.exists(single):
            st = SafeTensorsFile(single, use_mmap)
            self.files['model.safetensors'] = st
            for name in st.keys():
                self.where[name] = st
        else:
            raise SafetensorsError('%s: no model.safetensors or index'
                                   % directory)

    def keys(self):
        return list(self.where)

    def __contains__(self, name):
        return name in self.where

    def _file(self, name):
        if name not in self.where:
            raise KeyError('checkpoint has no tensor %r' % name)
        return self.where[name]

    def info(self, name):
        return self._file(name).info(name)

    def get_raw(self, name):
        return self._file(name).get_raw(name)

    def get_f32(self, name, out=None):
        return self._file(name).get_f32(name, out)

    def close(self):
        for f in self.files.values():
            f.close()
