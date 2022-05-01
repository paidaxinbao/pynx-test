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

exclude_opencl = False
if 'PYNX_PU' in os.environ:
    if 'cuda' in os.environ['PYNX_PU'].lower() or 'cuda' in sys.argv:
        exclude_opencl = True

try:
    from pynx.processing_unit.cl_processing_unit import CLProcessingUnit
    from pynx.processing_unit import ProcessingUnit
    import pyopencl.array as cla
except ImportError:
    exclude_opencl = True


class TestFFT(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if exclude_opencl:
            raise unittest.SkipTest("OpenCL tests excluded")
        cls.pu = CLProcessingUnit()
        cls.pu.select_gpu(language="opencl")
        cls.pu.init_cl(test_fft=False, verbose=False)
        cls.cpu = ProcessingUnit()
        cls.dtype_complex_v = [np.complex64]
        if 'cl_khr_fp64' in cls.pu.cl_queue.device.extensions:
            cls.dtype_complex_v.append(np.complex128)

    @unittest.skipIf(exclude_opencl, "OpenCL tests excluded")
    def test_fft_c2c(self):
        pu = self.pu
        n = 64
        for dims in range(1, 4):
            for ndim in range(1, dims):
                for dtype in self.dtype_complex_v:
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
                        a0 += 10 * np.exp(-v * 2)
                        a0 = cla.to_device(self.pu.cl_queue, a0)

                        a = a0.copy()
                        b = cla.empty_like(a)

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
                        # print([(abs(v.get()) ** 2).sum() for v in [a0, a, b]], dims, ndim, dtype)
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
                        # Why do we need larger tolerances for OpenCL ?
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

    @unittest.skipIf(exclude_opencl, "OpenCL tests excluded")
    def test_fft_r2c(self):
        pu = self.pu
        n = 64
        for dims in range(1, 4):
            for ndim in range(1, dims):
                # for dtype in self.dtype_complex_v:
                for dtype in [np.complex64]:  # TODO: complex128 R2C support in gpyfft ?
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
                        a0 = cla.to_device(self.pu.cl_queue, a0)

                        # Shape of the half-Hermitian array
                        sh[-1] = n // 2 + 1
                        b0 = cla.empty(self.pu.cl_queue, tuple(sh), dtype)

                        # out-of-place
                        a = a0.copy()
                        b = b0.copy()
                        pu.fft(a, b, ndim=ndim)
                        pu.ifft(b, a, ndim=ndim)

                        # Check norm
                        a = a0.copy()
                        b = b0.copy()
                        # Out-of-place C2R with ndim>=2 destroys the input array (vkfft)
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
