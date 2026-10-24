# coding: utf-8
'''cblas_sgemv / cblas_sgemm bindings against NumPy.

On a Mac or in Pythonista these exercise Accelerate. On Linux, run with
TINY_GPT_CBLAS=/usr/lib/x86_64-linux-gnu/libblas.so.3 (any CBLAS) to
exercise the same ctypes bindings; without it only the NumPy fallback
runs and the binding-specific tests skip.
'''

import unittest

import numpy as np

from tinygpt import backend
from tinygpt.hf import llama


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


if __name__ == '__main__':
    unittest.main()
