# coding: utf-8
'''tools/native16_diagnostics.py: report shape, helpers, and the probe's
ctypes plumbing against a fake BNNS (tests/native16_fake_bnns.c).

The fake is compiled with the system C compiler when there is one; the
probe tests skip otherwise (e.g. in Pythonista). On a Mac or iPhone the
symbols/probe run against the real Accelerate is the device diagnostic
itself, not a unit test, so it is not run here.
'''

import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), 'tools')
sys.path.insert(0, TOOLS)
import native16_diagnostics as nd  # noqa: E402
sys.path.remove(TOOLS)


def _build_fake(out_dir):
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        return None
    lib = os.path.join(out_dir, 'libfakebnns.so')
    try:
        subprocess.check_call([cc, '-O1', '-shared', '-fPIC', '-o', lib,
                               os.path.join(HERE, 'native16_fake_bnns.c')],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return lib


class HelpersTest(unittest.TestCase):

    def test_struct_layout_matches_c(self):
        for name, r in nd.struct_layout().items():
            self.assertTrue(r['ok'], name)

    def test_bf16_rounding_all_halfway_cases(self):
        # Every bf16 value, and the float32 exactly halfway above it, must
        # round to even.
        bits = np.arange(0, 0x7F80, dtype=np.uint32)
        f = (bits << 16).view(np.float32)
        self.assertTrue(np.array_equal(nd.f32_to_bf16_bits(f), bits))
        half = ((bits << 16) | 0x8000).view(np.float32)
        want = np.where(bits & 1, bits + 1, bits).astype(np.uint16)
        self.assertTrue(np.array_equal(nd.f32_to_bf16_bits(half), want))

    def test_encode_decode_exact(self):
        a = np.random.RandomState(0).standard_normal(100).astype(np.float32)
        for dt in (nd.DT_F32, nd.DT_F16, nd.DT_BF16):
            s, exact = nd.encode(a, dt)
            self.assertTrue(np.array_equal(nd.decode(s, dt), exact))

    def test_flip_sign_bits(self):
        a = np.random.RandomState(1).standard_normal(50).astype(np.float32)
        for dt in (nd.DT_F32, nd.DT_F16, nd.DT_BF16):
            s, exact = nd.encode(a, dt)
            nd.flip_sign_bits(s, dt)
            self.assertTrue(np.array_equal(nd.decode(s, dt), -exact))


class SymbolsModeTest(unittest.TestCase):

    def test_runs_and_writes_reports(self):
        d = tempfile.mkdtemp()
        try:
            res, text, base = nd.run('symbols', out_dir=d, log=lambda s: None)
            self.assertIn('native16 diagnostics', text)
            with open(base + '.json') as f:
                j = json.load(f)
            names = [s['symbol'] for s in j['symbols']]
            self.assertIn('BNNSFilterCreateLayerFullyConnected', names)
            self.assertFalse(os.path.exists(
                os.path.join(d, 'native16_diag_progress.json')))
        finally:
            shutil.rmtree(d)

    def test_previous_crash_is_reported_and_skipped(self):
        d = tempfile.mkdtemp()
        try:
            with open(os.path.join(d, 'native16_diag_progress.json'),
                      'w') as f:
                json.dump({'step': 'FC_boom'}, f)
            res, text, _ = nd.run('symbols', out_dir=d, log=lambda s: None)
            self.assertEqual(res['previous_crash'], 'FC_boom')
            self.assertIn('PREVIOUS RUN DIED DURING: FC_boom', text)
        finally:
            shutil.rmtree(d)


class ModelAlignmentTest(unittest.TestCase):

    def test_reports_tiny_model(self):
        from tests.hf_common import tiny_model_dir
        d, _ = tiny_model_dir('tiny_llama_tied_bf16.json')
        try:
            r = nd.model_alignment(d)
            self.assertNotIn('error', r)
            self.assertGreater(r['tensors'], 0)
            self.assertIn('BF16', r['dtypes'])
            self.assertEqual(nd.find_models(d), [d])
        finally:
            shutil.rmtree(d)


class FakeProbeTest(unittest.TestCase):
    '''Probe plumbing against the fake library (not real Accelerate).'''

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.lib = _build_fake(cls.dir)
        if cls.lib is None:
            raise unittest.SkipTest('no C compiler for the fake BNNS')
        cls.libs = {'Accelerate': ctypes.CDLL(cls.lib)}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir)

    def probe(self, skip=()):
        prog = nd.Progress(os.path.join(self.dir, 'progress.json'))
        r = nd.Prober(self.libs, prog, skip, lambda s: None).run()
        prog.done()
        return r

    def test_probe_results(self):
        r = self.probe()
        self.assertEqual(r['struct_layout']['status'], 'ok')
        self.assertEqual(r['BNNSNDArrayGetDataSize']['status'], 'ok')
        self.assertEqual(r['BNNSCopy_BF16_to_F32']['status'], 'ok')
        self.assertEqual(r['BNNSCopy_F16_to_F32']['status'], 'ok')
        self.assertEqual(r['vImage_Planar16F_to_PlanarF']['status'], 'ok')

        cp = r['FC_WF16_xF32_yF32_256x512_clientptr']
        self.assertEqual(cp['status'], 'ok', cp)
        self.assertIs(cp['reads_client_weights_after_create'], True)
        self.assertLess(cp['create_alloc']['bytes'], cp['weight_bytes'])

        cpy = r['FC_WF16_xF32_yF32_256x512_copy']
        self.assertEqual(cpy['status'], 'ok')
        self.assertIs(cpy['reads_client_weights_after_create'], False)
        self.assertGreaterEqual(cpy['create_alloc']['bytes'],
                                cpy['weight_bytes'])
        self.assertGreaterEqual(cpy['destroy_freed_bytes'],
                                cpy['weight_bytes'])

        self.assertEqual(r['FC_WBF16_xF32_yF32_40x72_clientptr']['status'],
                         'ok')
        self.assertEqual(r['FC_WBF16_xBF16_yF32_8x16_copy']['status'],
                         'refused')
        for tag in ('WF16_xF32_yF32', 'WBF16_xF32_yF32', 'WF16_xF16_yF16'):
            m = r['MatMul_%s_40x72' % tag]
            self.assertEqual(m['status'], 'ok', (tag, m))

    def crashed_probe(self, step):
        d = tempfile.mkdtemp(dir=self.dir)
        with open(os.path.join(d, 'progress.json'), 'w') as f:
            json.dump({'step': step}, f)
        prog = nd.Progress(os.path.join(d, 'progress.json'))
        r = nd.Prober(self.libs, prog, (), lambda s: None).run()
        prog.done()
        # The crash list persists for the next run.
        self.assertIn(step, nd.Progress(os.path.join(d, 'progress.json'))
                      .crashes)
        return r

    def test_crash_skips_whole_dtype_family(self):
        r = self.crashed_probe('FC_WBF16_xF32_yF32_40x72_copy')
        for k, v in r.items():
            if k.startswith('FC_WBF16_xF32_yF32_'):
                self.assertEqual(v['status'], 'skipped', k)
        self.assertEqual(r['FC_WF16_xF32_yF32_40x72_copy']['status'], 'ok')
        self.assertEqual(r['MatMul_WBF16_xF32_yF32_40x72']['status'], 'ok')

    def test_hook_crash_disables_hook(self):
        r = self.crashed_probe('alloc_hook_selftest')
        self.assertNotIn('alloc_hook_selftest', r)
        self.assertEqual(r['alloc_tracker_error'], 'disabled after a crash')
        fc = r['FC_WF16_xF32_yF32_256x512_clientptr']
        self.assertEqual(fc['status'], 'ok')
        self.assertIsNone(fc['create_alloc'])
        self.assertIs(fc['reads_client_weights_after_create'], True)

    def test_skip(self):
        r = self.probe(skip=['BNNSCopy_F16_to_F32'])
        self.assertEqual(r['BNNSCopy_F16_to_F32']['status'], 'skipped')


if __name__ == '__main__':
    unittest.main()
