# coding: utf-8
'''cblas_sgemv / cblas_sgemm bindings against NumPy.

On a Mac or in Pythonista these exercise Accelerate. On Linux, run with
TINY_GPT_CBLAS=/usr/lib/x86_64-linux-gnu/libblas.so.3 (any CBLAS) to
exercise the same ctypes bindings; without it only the NumPy fallback
runs and the binding-specific tests skip.
'''

import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tinygpt import backend
from tinygpt.hf import llama, safetensors


class GemvTest(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(0)

    def check(self, M, N, trans, beta):
        A = self.rng.standard_normal((M, N)).astype(np.float32)
        x = self.rng.standard_normal(M if trans else N).astype(np.float32)
        y0 = self.rng.standard_normal(N if trans else M).astype(np.float32)
        ref = (A.T if trans else A).astype(np.float64) @ x + beta * y0
        y = y0.copy()
        backend.gemv(A, x, y, trans=trans, beta=beta)
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)

    def test_shapes(self):
        for M, N in [(1, 1), (7, 3), (3, 7), (576, 576), (960, 576),
                     (576, 1536), (4099, 33)]:
            for trans in (False, True):
                for beta in (0.0, 0.5):
                    self.check(M, N, trans, beta)

    def test_noncontiguous_falls_back(self):
        A = self.rng.standard_normal((40, 30)).astype(np.float32)
        At = A.T                       # not C-contiguous
        x = self.rng.standard_normal(40).astype(np.float32)
        y = np.empty(30, np.float32)
        backend.gemv(At, x, y)
        np.testing.assert_allclose(y, A.T @ x, rtol=1e-5, atol=1e-5)

    def test_float64_falls_back(self):
        A = self.rng.standard_normal((5, 4))
        x = self.rng.standard_normal(4)
        y = np.empty(5)
        backend.gemv(A, x, y)
        np.testing.assert_allclose(y, A @ x)

    @unittest.skipUnless(backend.HAVE_SGEMV, 'no CBLAS sgemv bound')
    def test_binding_is_live(self):
        # The ctypes path, not the fallback, must produce these numbers.
        A = self.rng.standard_normal((64, 48)).astype(np.float32)
        x = self.rng.standard_normal(48).astype(np.float32)
        y = np.full(64, np.nan, np.float32)
        backend._sgemv(backend.CblasRowMajor, backend.CblasNoTrans, 64, 48,
                       1.0, backend._p(A), 48, backend._p(x), 1, 0.0,
                       backend._p(y), 1)
        np.testing.assert_allclose(y, A @ x, rtol=1e-5, atol=1e-5)

    def test_gemm_unchanged(self):
        A = self.rng.standard_normal((9, 5)).astype(np.float32)
        B = self.rng.standard_normal((7, 5)).astype(np.float32)
        C = np.empty((9, 7), np.float32)
        backend.gemm(A, B, C, transB=True)
        np.testing.assert_allclose(C, A @ B.T, rtol=1e-5, atol=1e-5)


class MatVecTuneTest(unittest.TestCase):

    def test_every_candidate_agrees(self):
        rng = np.random.RandomState(1)
        W = rng.standard_normal((300, 200)).astype(np.float32)
        x = rng.standard_normal(200).astype(np.float32)
        for name in llama.matvec_candidates():
            y = np.empty(300, np.float32)
            llama.MATVEC_IMPLS[name](W, x, y)
            np.testing.assert_allclose(y, W @ x, rtol=1e-4, atol=1e-4,
                                       err_msg=name)

    def test_wrong_candidate_rejected(self):
        saved = dict(llama.MATVEC_IMPLS)
        saved_c = llama.matvec_candidates
        try:
            def broken(W, x, y):
                np.dot(W, x, out=y)
                y[0] += 1.0
            llama.MATVEC_IMPLS['broken'] = broken
            llama.matvec_candidates = lambda: ['broken', 'numpy']
            mv = llama.MatVec()
            W = np.ones((10, 10), np.float32)
            self.assertIs(mv.tune(W), llama.MATVEC_IMPLS['numpy'])
            self.assertIn('rejected', mv.timings[(10, 10)]['broken'])
        finally:
            llama.MATVEC_IMPLS.clear()
            llama.MATVEC_IMPLS.update(saved)
            llama.matvec_candidates = saved_c


def _fake_bnns(out_dir):
    '''Build tests/native16_fake_bnns.c (a stand-in BNNSCopy), or None.'''
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        return None
    lib = os.path.join(out_dir, 'libfakebnns.so')
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'native16_fake_bnns.c')
    try:
        subprocess.check_call([cc, '-O1', '-shared', '-fPIC', '-o', lib, src],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return lib


class BF16WidenTest(unittest.TestCase):
    '''The BNNSCopy widening path against NumPy's shift. In Pythonista or
    on a Mac this binds the real Accelerate; elsewhere a fake BNNSCopy is
    compiled so the ctypes plumbing still runs.'''

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.saved = backend._bnns_copy
        if backend._bnns_copy is None:
            lib = _fake_bnns(cls.dir)
            if lib is None:
                shutil.rmtree(cls.dir)
                raise unittest.SkipTest('no BNNSCopy and no C compiler')
            backend._bnns_copy = backend._bind_bnns_copy(lib)

    @classmethod
    def tearDownClass(cls):
        backend._bnns_copy = cls.saved
        shutil.rmtree(cls.dir)

    def test_every_finite_bf16_pattern_exact(self):
        bits = np.arange(1 << 16, dtype=np.uint32)
        exp = (bits >> 7) & 0xFF
        bits = bits[exp != 0xFF].astype(np.uint16)       # drop inf / NaN
        want = (bits.astype(np.uint32) << 16).view(np.float32)
        out = np.full(bits.size, np.nan, np.float32)
        w = backend.bf16_widener(bits, out)
        self.assertIsNotNone(w)
        w()
        self.assertTrue(np.array_equal(out.view(np.uint32),
                                       want.view(np.uint32)))
        # repeated calls reuse the same descriptors
        out[:] = 0
        w()
        self.assertTrue(np.array_equal(out, want))

    def test_unsuitable_arrays_fall_back(self):
        src = np.zeros((4, 6), np.uint16)
        self.assertIsNone(backend.bf16_widener(src[:, ::2],
                                               np.zeros(12, np.float32)))
        self.assertIsNone(backend.bf16_widener(src, np.zeros(24, np.float64)))
        self.assertIsNone(backend.bf16_widener(src, np.zeros(23, np.float32)))
        # ... and safetensors still converts them, through NumPy
        out = np.empty((4, 3), np.float32)
        safetensors.widener(src[:, ::2], 'BF16', out)()
        self.assertTrue(np.array_equal(out, np.zeros((4, 3))))

    def test_bf16_to_f32_uses_it(self):
        raw = np.array([0x3F80, 0xC000, 0x0001], np.uint16)
        np.testing.assert_array_equal(safetensors.bf16_to_f32(raw),
                                      [1.0, -2.0, 9.183549615799121e-41])

    def test_lowmem_linear_matches_numpy_path(self):
        rng = np.random.RandomState(3)
        # Random BF16 bit patterns of magnitude ~1e-10..0.5, either sign.
        parts = [((rng.randint(0x3000, 0x3F00, (r, 40))
                   | rng.randint(0, 2, (r, 40)) << 15).astype(np.uint16),
                  'BF16') for r in (37, 11, 52)]
        x = rng.standard_normal(40).astype(np.float32)
        ref = np.concatenate([safetensors.bf16_to_f32(p).astype(np.float64)
                              for p, _ in parts]) @ x
        ys = []
        for native in (True, False):
            saved = backend._bnns_copy
            if not native:
                backend._bnns_copy = None
            try:
                # 600 floats of scratch: 15 rows per slab, so every part
                # spans several slabs, including ragged last ones.
                lin = llama.Linear(parts, 'lowmem', llama.MatVec('numpy'),
                                   np.empty(600, np.float32))
                kinds = set(getattr(w, 'keep', None) is not None
                            for w, _, _, _ in lin._slabs)
                self.assertEqual(kinds, {native})
                y = np.empty(100, np.float32)
                ys.append(lin.matvec(x, y).copy())
                Y = np.empty((3, 100), np.float32)
                lin.matmul(np.stack([x, x, x]), Y)
                # prefill sums in another order than decode: close, not equal
                np.testing.assert_allclose(Y[1], ys[-1], rtol=1e-5,
                                           atol=1e-6)
                ys.append(Y.copy())
            finally:
                backend._bnns_copy = saved
        # native and NumPy widening give bit-identical results
        np.testing.assert_array_equal(ys[0], ys[2])
        np.testing.assert_array_equal(ys[1], ys[3])
        np.testing.assert_allclose(ys[0], ref, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
