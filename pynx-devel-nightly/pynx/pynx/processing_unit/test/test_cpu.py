# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import unittest
import numpy as np
from pynx.processing_unit import ProcessingUnit


class TestFFT(unittest.TestCase):

    def test_fft_c2c(self):
        pu = ProcessingUnit()
        n = 64
        for dims in range(1, 4):
            for ndim in range(1, dims):
                for dtype in [np.complex64, np.complex128]:
                    with self.subTest(dims=dims, ndim=ndim, dtype=dtype):
                        if dtype == np.complex64:
                            tol = 1e-6
                        else:
                            tol = 1e-12

                        a0 = np.random.uniform(0, 1, [n] * dims).astype(dtype)
                        # A pure random array may not be a very good test (too random),
                        # so add a Gaussian
                        xx = [np.fft.fftshift(np.fft.fftfreq(n))] * dims
                        v = np.zeros_like(a0)
                        for x in np.meshgrid(*xx, indexing='ij'):
                            v += x ** 2
                        a0 += np.exp(-v * 0.1)

                        a = a0.copy()
                        b = np.empty_like(a)

                        # inplace transform
                        pu.fft(a, a, ndim=ndim)
                        pu.ifft(a, a, ndim=ndim)
                        self.assertTrue(np.allclose(a, a0, atol=tol, rtol=tol))

                        # in vs out-of-place
                        a = a0.copy()
                        pu.fft(a, b, ndim=ndim)
                        pu.fft(a, a, ndim=ndim)
                        self.assertTrue(np.allclose(b, a, atol=tol, rtol=tol))
                        pu.ifft(a, b, ndim=ndim)
                        pu.ifft(a, a, ndim=ndim)
                        self.assertTrue(np.allclose(b, a, atol=tol, rtol=tol))

                        # out-of-place
                        a = a0.copy()
                        pu.fft(a, b, ndim=ndim)
                        pu.ifft(b, a, ndim=ndim)
                        # print((abs(b) ** 2).sum(), (abs(a) ** 2).sum(), (abs(a0) ** 2).sum(), dims, ndim)
                        self.assertTrue(np.allclose(a, a0, atol=tol, rtol=tol))

                        # Check norm
                        a = a0.copy()
                        pu.fft(a0, a, ndim=ndim, norm=True)
                        self.assertTrue(np.isclose((abs(a0) ** 2).sum(), (abs(a) ** 2).sum()))
                        pu.ifft(a, b, ndim=ndim, norm=True)
                        self.assertTrue(np.isclose((abs(a) ** 2).sum(), (abs(b) ** 2).sum()))

                        # Check scale (norm=False)
                        a = a0.copy()
                        s = pu.fft(a0, a, ndim=ndim, return_scale=True)
                        self.assertTrue(np.isclose((abs(a0) ** 2).sum(), (abs(s * a) ** 2).sum()))
                        s = pu.ifft(a0, a, ndim=ndim, return_scale=True)
                        self.assertTrue(np.isclose((abs(a0) ** 2).sum(), (abs(s * a) ** 2).sum()))

                        # Check scale (norm=True)
                        a = a0.copy()
                        s = pu.fft(a0, a, ndim=ndim, return_scale=True, norm=True)
                        self.assertTrue(np.isclose((abs(a0) ** 2).sum(), (abs(s * a) ** 2).sum()))
                        s = pu.ifft(a0, a, ndim=ndim, return_scale=True, norm=True)
                        self.assertTrue(np.isclose((abs(a0) ** 2).sum(), (abs(s * a) ** 2).sum()))

    def test_fft_r2c(self):
        pu = ProcessingUnit()
        n = 64
        for dims in range(1, 4):
            for ndim in range(1, dims):
                for dtype in [np.complex64, np.complex128]:
                    with self.subTest(dims=dims, ndim=ndim, dtype=dtype):
                        if dtype == np.complex64:
                            tol = 1e-6
                            dtype_real = np.float32
                        else:
                            tol = 1e-12
                            dtype_real = np.float64
                        sh = [n] * dims

                        a0 = np.random.uniform(0, 1, sh).astype(dtype_real)
                        # A pure random array may not be a very good test (too random),
                        # so add a Gaussian
                        xx = [np.fft.fftshift(np.fft.fftfreq(n))] * dims
                        v = np.zeros_like(a0)
                        for x in np.meshgrid(*xx, indexing='ij'):
                            v += x ** 2
                        a0 += np.exp(-v * 0.1)

                        # Shape of the half-Hermitian array
                        sh[-1] = n // 2 + 1
                        b0 = np.empty(sh, dtype)

                        # out-of-place
                        a = a0.copy()
                        b = b0.copy()
                        pu.fft(a, b, ndim=ndim)
                        pu.ifft(b, a, ndim=ndim)
                        self.assertTrue(np.allclose(a, a0, atol=tol, rtol=tol))

                        # Check norm
                        a = a0.copy()
                        b = b0.copy()
                        pu.fft(a, b, ndim=ndim, norm=True)
                        # half-hermitian array so special comparison: last axis
                        #  has n//2+1 elements, so need to add the norm of n//2-1 elements
                        #  in the [1:-1] range for the last axis
                        na = (abs(a) ** 2).sum()
                        nb = (abs(b) ** 2).sum() + (abs(b[..., 1:-1]) ** 2).sum()
                        self.assertTrue(np.isclose(na, nb))

                        pu.ifft(b, a, ndim=ndim, norm=True)
                        na = (abs(a) ** 2).sum()
                        nb = (abs(b) ** 2).sum() + (abs(b[..., 1:-1]) ** 2).sum()
                        self.assertTrue(np.isclose(na, nb))
                        self.assertTrue(np.allclose(a, a0, atol=tol, rtol=tol))

                        # Check scale (norm=False)
                        a = a0.copy()
                        b = b0.copy()
                        s = pu.fft(a, b, ndim=ndim, return_scale=True)
                        na = (abs(a) ** 2).sum()
                        nb = (abs(s * b) ** 2).sum() + (abs(s * b[..., 1:-1]) ** 2).sum()
                        self.assertTrue(np.isclose(na, nb))

                        s = pu.ifft(b, a, ndim=ndim, return_scale=True)
                        na = (abs(s * a) ** 2).sum()
                        nb = (abs(b) ** 2).sum() + (abs(b[..., 1:-1]) ** 2).sum()
                        self.assertTrue(np.isclose(na, nb))
                        self.assertTrue(np.allclose(a, a0, atol=tol, rtol=tol))

                        # Check scale (norm=True)
                        a = a0.copy()
                        b = b0.copy()
                        s = pu.fft(a, b, ndim=ndim, return_scale=True, norm=True)
                        na = (abs(a) ** 2).sum()
                        nb = (abs(s * b) ** 2).sum() + (abs(s * b[..., 1:-1]) ** 2).sum()
                        self.assertTrue(np.isclose(na, nb))

                        s = pu.ifft(b, a, ndim=ndim, return_scale=True, norm=True)
                        na = (abs(s * a) ** 2).sum()
                        nb = (abs(b) ** 2).sum() + (abs(b[..., 1:-1]) ** 2).sum()
                        self.assertTrue(np.isclose(na, nb))
                        self.assertTrue(np.allclose(a, a0, atol=tol, rtol=tol))


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite([loadTests(TestFFT)])
    return test_suite


if __name__ == '__main__':
    res = unittest.TextTestRunner(verbosity=2, descriptions=False).run(suite())
