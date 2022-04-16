# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import sys
import os
import unittest
import traceback
import numpy as np
from pynx.processing_unit import ProcessingUnit

exclude_cuda = False
if 'PYNX_PU' in os.environ:
    if 'opencl' in os.environ['PYNX_PU'].lower() or 'opencl' in sys.argv:
        exclude_cuda = True

try:
    from pynx.processing_unit.cu_processing_unit import CUProcessingUnit, has_vkfft_cuda, cu_fft, has_skcuda
    import pycuda.gpuarray as cua
except ImportError:
    exclude_cuda = True
    has_skcuda = False


class TestFFT(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if exclude_cuda:
            raise unittest.SkipTest("CUDA tests excluded")
        cls.pu = CUProcessingUnit()
        cls.pu.select_gpu(language="cuda")
        cls.cpu = ProcessingUnit()

    @unittest.skipIf(exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(not has_skcuda and has_vkfft_cuda, "skcuda is not present but VkFFT can be used")
    def test_skcuda_fft_import(self):
        msg = "Testing for skcuda.fft (needed for CUDA FFT calculations)"
        try:
            import skcuda.fft as cu_fft
            import_ok = True
        except ImportError:
            if has_vkfft_cuda:
                raise unittest.SkipTest("skcuda import failure: ignored as pyvkfft.cuda is available")
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    @unittest.skipIf(exclude_cuda, "CUDA tests excluded")
    def test_pyvkfft_cuda_import(self):
        msg = "Testing for pyvkfft.cuda import"
        try:
            import pyvkfft.cuda
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    @unittest.skipIf(exclude_cuda, "CUDA tests excluded")
    def test_fft_c2c(self):
        pu = self.pu
        n = 64
        v_backend = []
        if cu_fft is not None:
            v_backend.append('cufft')
        if has_vkfft_cuda:
            v_backend.append('vkfft_cuda')
        for backend in v_backend:
            if backend == "vkfft_cuda":
                self.pu.use_vkfft = True
            else:
                self.pu.use_vkfft = False
            for dims in range(1, 3):
                for ndim in range(1, dims + 1):
                    for dtype in [np.complex64, np.complex128]:
                        with self.subTest(backend=backend, dims=dims, ndim=ndim, dtype=dtype):
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
                            a0 += 10 * np.exp(-v * 2)
                            a0 = cua.to_gpu(a0)

                            a = a0.copy()
                            b = cua.empty_like(a)

                            # in & out-of-place
                            a = a0.copy()
                            pu.fft(a, b, ndim=ndim)
                            pu.fft(a, a, ndim=ndim)
                            self.assertTrue(np.allclose(b.get(), a.get(), atol=tol, rtol=tol))
                            pu.ifft(a, b, ndim=ndim)
                            pu.ifft(a, a, ndim=ndim)
                            self.assertTrue(np.allclose(b.get(), a.get(), atol=tol, rtol=tol))

                            # Check norm
                            a = a0.copy()
                            pu.fft(a0, a, ndim=ndim, norm=True)
                            self.assertTrue(np.isclose((abs(a0.get()) ** 2).sum(), (abs(a.get()) ** 2).sum()))
                            pu.ifft(a, b, ndim=ndim, norm=True)
                            self.assertTrue(np.isclose((abs(a.get()) ** 2).sum(), (abs(b.get()) ** 2).sum()))
                            self.assertTrue(np.allclose(a0.get(), b.get(), atol=tol, rtol=tol))

                            # Check scale (norm=False)
                            a = a0.copy()
                            s = pu.fft(a0, a, ndim=ndim, return_scale=True)
                            self.assertTrue(np.isclose((abs(a0.get()) ** 2).sum(), (abs(s * a.get()) ** 2).sum()))
                            s = pu.ifft(a0, a, ndim=ndim, return_scale=True)
                            self.assertTrue(np.isclose((abs(a0.get()) ** 2).sum(), (abs(s * a.get()) ** 2).sum()))

                            # Check scale (norm=True)
                            a = a0.copy()
                            s = pu.fft(a0, a, ndim=ndim, return_scale=True, norm=True)
                            self.assertTrue(np.isclose((abs(a0.get()) ** 2).sum(), (abs(s * a.get()) ** 2).sum()))
                            s = pu.ifft(a0, a, ndim=ndim, return_scale=True, norm=True)
                            self.assertTrue(np.isclose((abs(a0.get()) ** 2).sum(), (abs(s * a.get()) ** 2).sum()))

                            # Check consistency with scipy. With relaxed tolerances
                            a = a0.copy()
                            ac = a.get()
                            pu.fft(a, a, ndim=ndim, norm=True)
                            self.cpu.fft(ac, ac, ndim=ndim, norm=True)
                            self.assertTrue(np.allclose(a.get(), ac, atol=tol * 40, rtol=tol * 40))
                            # self.assertTrue(np.allclose(a.get(), ac, atol=tol, rtol=tol))
                            pu.ifft(a, a, ndim=ndim, norm=True)
                            self.cpu.ifft(ac, ac, ndim=ndim, norm=True)
                            # print((abs(a.get()) ** 2).sum(), (abs(ac) ** 2).sum(), dims, ndim, dtype)
                            self.assertTrue(np.allclose(a.get(), ac, atol=tol * 20, rtol=tol * 20))
                            # self.assertTrue(np.allclose(a.get(), ac, atol=tol, rtol=tol))

                            self.pu.free_fft_plans()

    @unittest.skipIf(exclude_cuda, "CUDA tests excluded")
    def test_fft_r2c(self):
        pu = self.pu
        n = 64
        v_backend = []
        if cu_fft is not None:
            v_backend.append('cufft')
        if has_vkfft_cuda:
            v_backend.append('vkfft_cuda')
        for backend in v_backend:
            if backend == "vkfft_cuda":
                self.pu.use_vkfft = True
            else:
                self.pu.use_vkfft = False
            for dims in range(1, 3):
                for ndim in range(1, dims + 1):
                    for dtype in [np.complex64, np.complex128]:
                        with self.subTest(backend=backend, dims=dims, ndim=ndim, dtype=dtype):
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
                            a0 = cua.to_gpu(a0)

                            # Shape of the half-Hermitian array
                            sh[-1] = n // 2 + 1
                            b0 = cua.empty(sh, dtype)

                            # out-of-place
                            a = a0.copy()
                            b = b0.copy()
                            pu.fft(a, b, ndim=ndim)
                            pu.ifft(b, a, ndim=ndim)

                            # Check norm
                            a = a0.copy()
                            b = b0.copy()
                            n0 = (abs(a.get()) ** 2).sum()
                            pu.fft(a, b, ndim=ndim, norm=True)
                            # half-hermitian array so special comparison: last axis
                            #  has n//2+1 elements, so need to add the norm of n//2-1 elements
                            #  in the [1:-1] range for the last axis
                            n1 = (abs(b.get()) ** 2).sum() + (abs(b.get()[..., 1:-1]) ** 2).sum()
                            self.assertTrue(np.isclose(n0, n1))

                            pu.ifft(b, a, ndim=ndim, norm=True)
                            n2 = (abs(a.get()) ** 2).sum()

                            self.assertTrue(np.isclose(n2, n1))
                            self.assertTrue(np.allclose(a.get(), a0.get(), atol=tol, rtol=tol))

                            # Check scale (norm=False)
                            a = a0.copy()
                            b = b0.copy()
                            n0 = (abs(a.get()) ** 2).sum()
                            s1 = pu.fft(a, b, ndim=ndim, return_scale=True)
                            n1 = (abs(s1 * b.get()) ** 2).sum() + (abs(s1 * b.get()[..., 1:-1]) ** 2).sum()
                            self.assertTrue(np.isclose(n0, n1))

                            s2 = pu.ifft(b, a, ndim=ndim, return_scale=True)
                            s = s2 * s1
                            n2 = (abs(s * a.get()) ** 2).sum()
                            self.assertTrue(np.isclose(n1, n2))
                            self.assertTrue(np.allclose(s * a.get(), a0.get(), atol=tol, rtol=tol))

                            # Check scale (norm=True)
                            a = a0.copy()
                            b = b0.copy()
                            n0 = (abs(a.get()) ** 2).sum()
                            s = pu.fft(a, b, ndim=ndim, return_scale=True, norm=True)
                            n1 = (abs(s * b.get()) ** 2).sum() + (abs(s * b.get()[..., 1:-1]) ** 2).sum()
                            self.assertTrue(np.isclose(n0, n1))

                            s = pu.ifft(b, a, ndim=ndim, return_scale=True, norm=True)
                            n2 = (abs(s * a.get()) ** 2).sum()
                            self.assertTrue(np.isclose(n1, n2))
                            self.assertTrue(np.allclose(a.get(), a0.get(), atol=tol, rtol=tol))

                            # Check consistency with scipy. With relaxed tolerances
                            a = a0.copy()
                            b = b0.copy()
                            ac = a.get()
                            bc = b.get()
                            pu.fft(a, b, ndim=ndim, norm=True)
                            self.cpu.fft(ac, bc, ndim=ndim, norm=True)
                            self.assertTrue(np.allclose(b.get(), bc, atol=tol * 10, rtol=tol * 10))
                            # self.assertTrue(np.allclose(a.get(), ac, atol=tol, rtol=tol))
                            pu.ifft(b, a, ndim=ndim, norm=True)
                            self.cpu.ifft(bc, ac, ndim=ndim, norm=True)
                            self.assertTrue(np.allclose(a.get(), ac, atol=tol * 10, rtol=tol * 10))
                            # self.assertTrue(np.allclose(a.get(), ac, atol=tol, rtol=tol))

                            self.pu.free_fft_plans()


def suite():
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite([load_tests(TestFFT)])
    return test_suite


if __name__ == '__main__':
    res = unittest.TextTestRunner(verbosity=2, descriptions=False).run(suite())
