# coding: utf-8
'''hf_run.py's memory reporting, and how the weight file gets opened.'''

import sys
import unittest

import hf_run
from tests.hf_common import tiny_model_dir
from tinygpt.hf import llama


class MemoryTest(unittest.TestCase):

    @unittest.skipUnless(sys.platform in ('linux', 'darwin', 'ios'),
                         'no memory probe on this platform')
    def test_memory_now(self):
        mb, what = hf_run.memory_now()
        self.assertIsNotNone(mb)
        self.assertGreater(mb, 1.0)
        if sys.platform in ('darwin', 'ios'):
            # phys_footprint is what the iOS memory limit is checked against.
            self.assertEqual(what, 'footprint')
        else:
            self.assertEqual(what, 'resident')
        self.assertIn('MB', hf_run.memory_line())

    @unittest.skipUnless(sys.platform in ('darwin', 'ios'), 'Mach only')
    def test_task_vm_info(self):
        info = hf_run._task_vm_info()
        self.assertGreater(info['footprint'], 0)
        self.assertGreater(info['resident'], 0)


class AutoWeightsTest(unittest.TestCase):

    def test_auto(self):
        class Cfg(object):
            def __init__(self, n):
                self.n = n

            def n_params(self):
                return self.n
        self.assertEqual(hf_run.resolve_weights('auto', Cfg(135e6)),
                         'float32')
        self.assertEqual(hf_run.resolve_weights('auto', Cfg(362e6)),
                         'lowmem')
        self.assertEqual(hf_run.resolve_weights('auto', Cfg(1.5e9)),
                         'lowmem')
        self.assertEqual(hf_run.resolve_weights('float32', Cfg(1.5e9)),
                         'float32')


class MmapRequiredTest(unittest.TestCase):
    '''lowmem must not silently read the whole model into RAM when the
    memory map fails (e.g. iOS address-space limits).'''

    def test_require(self):
        import mmap
        import os
        from tinygpt.hf import safetensors as st
        td, _ = tiny_model_dir('tiny_llama_tied_bf16.json')
        path = os.path.join(td, 'model.safetensors')
        real = mmap.mmap

        def fail(*a, **k):
            raise OSError(12, 'Cannot allocate memory')
        st.mmap.mmap = fail
        try:
            with self.assertRaises(st.SafetensorsError):
                st.SafeTensorsFile(path, use_mmap='require')
            f = st.SafeTensorsFile(path, use_mmap=True)   # falls back
            self.assertGreater(len(f.keys()), 0)
            f.close()
        finally:
            st.mmap.mmap = real


class MmapUseTest(unittest.TestCase):
    '''float32 reads weights with plain reads, so nothing of the file is
    left mapped into the process; lowmem maps it, which is its point.'''

    def opened_with(self, weights):
        seen = []
        real = llama.Checkpoint

        def spy(path, use_mmap=True):
            seen.append(use_mmap)
            return real(path, use_mmap)
        llama.Checkpoint = spy
        try:
            td, _ = tiny_model_dir('tiny_llama_tied_bf16.json')
            m = llama.LlamaModel.from_dir(td, ctx_limit=16, weights=weights)
            m.close()
        finally:
            llama.Checkpoint = real
        return seen

    def test_float32_does_not_mmap(self):
        self.assertEqual(self.opened_with('float32'), [False])

    def test_lowmem_mmaps(self):
        self.assertEqual(self.opened_with('lowmem'), ['require'])


if __name__ == '__main__':
    unittest.main()
