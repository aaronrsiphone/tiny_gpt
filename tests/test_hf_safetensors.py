# coding: utf-8
'''safetensors parsing, dtype conversion and header validation.'''

import base64
import json
import os
import struct
import tempfile
import unittest

import numpy as np

from tests.hf_common import fixture, write_safetensors
from tinygpt.hf.safetensors import (Checkpoint, SafeTensorsFile,
                                    SafetensorsError, bf16_to_f32)


class DtypeTest(unittest.TestCase):
    '''Against a file written by the official `safetensors` library.'''

    @classmethod
    def setUpClass(cls):
        cls.d = fixture('safetensors_dtypes.json')
        cls.td = tempfile.mkdtemp()
        cls.path = os.path.join(cls.td, 'x.safetensors')
        with open(cls.path, 'wb') as f:
            f.write(base64.b64decode(cls.d['safetensors']))

    def test_values(self):
        for use_mmap in (True, False):
            f = SafeTensorsFile(self.path, use_mmap=use_mmap)
            for k in ('f32', 'f16', 'bf16', 'scalar', 'empty'):
                got = f.get_f32(k)
                self.assertEqual(list(got.shape), self.d['shapes'][k])
                want = np.array(self.d['values'][k], np.float32).reshape(
                    self.d['shapes'][k])
                np.testing.assert_array_equal(got, want, k)
            f.close()

    def test_bf16_specials(self):
        f = SafeTensorsFile(self.path)
        self.addCleanup(f.close)
        got = f.get_f32('bf16_special')
        want = np.array([float(v) for v in self.d['values']['bf16_special']],
                        np.float32)
        np.testing.assert_array_equal(got, want)
        self.assertTrue(np.signbit(got[1]))      # -0.0 survives

    def test_raw_is_zero_copy_view(self):
        f = SafeTensorsFile(self.path)
        self.addCleanup(f.close)
        r = f.get_raw('bf16')
        self.assertEqual(r.dtype, np.uint16)
        self.assertFalse(r.flags.writeable)


class Bf16Test(unittest.TestCase):

    def test_exhaustive(self):
        # All 65536 bit patterns: the shift must equal the definition.
        bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
        got = bf16_to_f32(bits)
        want = (bits.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(got.view(np.uint32),
                                      want.view(np.uint32))

    def test_known(self):
        b = np.array([0x3F80, 0xC020, 0x7F80, 0x0000, 0x8000, 0x3EAB],
                     np.uint16)
        np.testing.assert_array_equal(
            bf16_to_f32(b), np.array([1.0, -2.5, np.inf, 0.0, -0.0,
                                      0.333984375], np.float32))

    def test_out_buffer(self):
        b = np.array([[0x3F80, 0x4000]], np.uint16)
        out = np.zeros((1, 2), np.float32)
        self.assertIs(bf16_to_f32(b, out), out)
        np.testing.assert_array_equal(out, [[1.0, 2.0]])


class ValidationTest(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.p = os.path.join(self.td, 'bad.safetensors')

    def bad(self, header, data=b'\0' * 64):
        h = json.dumps(header).encode()
        with open(self.p, 'wb') as f:
            f.write(struct.pack('<Q', len(h)) + h + data)
        with self.assertRaises(SafetensorsError):
            SafeTensorsFile(self.p)

    def t(self, **kw):
        d = {'dtype': 'F32', 'shape': [2, 2], 'data_offsets': [0, 16]}
        d.update(kw)
        return d

    def test_rejections(self):
        self.bad({'a': self.t(dtype='I64')})
        self.bad({'a': self.t(dtype='F8_E4M3')})
        self.bad({'a': self.t(shape=[2, -2])})
        self.bad({'a': self.t(shape=[2, 2.0])})
        self.bad({'a': self.t(shape='2x2')})
        self.bad({'a': self.t(data_offsets=[0, 12])})        # wrong length
        self.bad({'a': self.t(data_offsets=[60, 76])})       # out of range
        self.bad({'a': self.t(data_offsets=[8, 0])})
        self.bad({'a': self.t(data_offsets=[0])})
        self.bad({'a': self.t(), 'b': self.t(data_offsets=[8, 24])})
        self.bad([1, 2, 3])
        self.bad({'a': 5})

    def test_bad_header_length(self):
        with open(self.p, 'wb') as f:
            f.write(struct.pack('<Q', 1 << 40) + b'{}')
        with self.assertRaises(SafetensorsError):
            SafeTensorsFile(self.p)
        with open(self.p, 'wb') as f:
            f.write(b'\1\2')
        with self.assertRaises(SafetensorsError):
            SafeTensorsFile(self.p)

    def test_bad_json(self):
        with open(self.p, 'wb') as f:
            f.write(struct.pack('<Q', 5) + b'{nope')
        with self.assertRaises(SafetensorsError):
            SafeTensorsFile(self.p)

    def test_metadata_ignored(self):
        write_safetensors(self.p, {}, header_override={
            '__metadata__': {'format': 'pt'}})
        f = SafeTensorsFile(self.p)
        self.assertEqual(f.keys(), [])
        f.close()


class ShardTest(unittest.TestCase):

    def test_sharded_checkpoint(self):
        td = tempfile.mkdtemp()
        rng = np.random.RandomState(0)
        a = rng.standard_normal((3, 4)).astype(np.float32)
        b = rng.standard_normal(5).astype(np.float16)
        write_safetensors(os.path.join(td, 'model-00001-of-00002'
                                       '.safetensors'), {'a': ('F32', a)})
        write_safetensors(os.path.join(td, 'model-00002-of-00002'
                                       '.safetensors'), {'b': ('F16', b)})
        with open(os.path.join(td, 'model.safetensors.index.json'),
                  'w') as f:
            json.dump({'weight_map': {
                'a': 'model-00001-of-00002.safetensors',
                'b': 'model-00002-of-00002.safetensors'}}, f)
        ck = Checkpoint(td)
        self.addCleanup(ck.close)
        self.assertEqual(sorted(ck.keys()), ['a', 'b'])
        np.testing.assert_array_equal(ck.get_f32('a'), a)
        np.testing.assert_array_equal(ck.get_f32('b'), b.astype(np.float32))

    def test_index_cannot_escape(self):
        td = tempfile.mkdtemp()
        with open(os.path.join(td, 'model.safetensors.index.json'),
                  'w') as f:
            json.dump({'weight_map': {'a': '../evil.safetensors'}}, f)
        with self.assertRaises(SafetensorsError):
            Checkpoint(td)


if __name__ == '__main__':
    unittest.main()
