# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import unittest
import traceback
import types
import numpy as np
from pynx.wavefront import wavefront
import pynx.wavefront.cpu_operator as cpuop
import pynx.wavefront.cl_operator as clop
import pynx.wavefront.cu_operator as cuop
from pynx.processing_unit import has_cuda, has_opencl


class TestWavefront(unittest.TestCase):
    def test_wavefront_create_default(self):
        msg = "testing for default wavefront creation"
        try:
            w = wavefront.Wavefront()
            ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    def test_wavefront_create(self):
        msg = "testing for custom wavefront creation"
        try:
            w = wavefront.Wavefront(d=np.ones((256, 256), dtype=np.complex64), pixel_size=1e-6, copy_d=False)
            ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    def test_wavefront_create_image(self):
        msg = "testing for wavefront creation from image bank"
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    def test_wavefront_ft_cpu(self):
        msg = "testing for wavefront FT (CPU)"
        op = cpuop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            s0 = (np.abs(w.get()) ** 2).sum()
            w = op.IFT() * w
            s1 = (np.abs(w.get()) ** 2).sum()
            w = op.FT() * w
            s2 = (np.abs(w.get()) ** 2).sum()
            ok = np.isclose(s0, s1) and np.isclose(s0, s2)
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    def test_wavefront_filter_cpu(self):
        msg = "testing for wavefront masks (CPU)"
        op = cpuop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            s0 = w.get().sum()
            w = op.RectangularMask(width=100e-6, height=50e-6) * w
            s1 = w.get().sum()
            w = op.CircularMask(25e-6) * w
            s2 = w.get().sum()
            ok = (np.isnan(w.get()).sum() == 0) and (s1 < 0.5 * s0) and (s2 < 0.8 * s1)
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    def test_wavefront_near_field_cpu(self):
        msg = "testing for wavefront near field propagation (CPU)"
        op = cpuop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.PropagateNearField(100e-6) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    def test_wavefront_near_field_cpu(self):
        msg = "testing for wavefront far field propagation (CPU)"
        op = cpuop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.PropagateFarField(10) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    def test_wavefront_fractional_cpu(self):
        msg = "testing for wavefront fractional propagation (CPU)"
        op = cpuop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.PropagateFRT(0.2) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    def test_wavefront_paganin_cpu(self):
        msg = "testing for wavefront paganin reconstruction (CPU)"
        op = cpuop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.BackPropagatePaganin() * op.PropagateNearField(100e-6) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_opencl is False, 'no opencl support')
    def test_wavefront_ft_cl(self):
        msg = "testing for wavefront FT (OpenCL)"
        op = clop
        try:
            w = wavefront.Wavefront(d="ascent", pixel_size=1e-6, copy_d=False)
            s0 = (np.abs(w.get()) ** 2).sum()
            w = op.IFT() * w
            s1 = (np.abs(w.get()) ** 2).sum()
            w = op.FT() * w
            s2 = (np.abs(w.get()) ** 2).sum()
            ok = np.isclose(s0, s1, rtol=1e-4) and np.isclose(s0, s2, rtol=1e-4)
            if not ok:
                print(s0, s1, s2)
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_opencl is False, 'no opencl support')
    def test_wavefront_filter_cl(self):
        msg = "testing for wavefront masks (OpenCL)"
        op = clop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            s0 = w.get().sum()
            w = op.RectangularMask(width=100e-6, height=50e-6) * w
            s1 = w.get().sum()
            w = op.CircularMask(25e-6) * w
            s2 = w.get().sum()
            ok = (np.isnan(w.get()).sum() == 0) and (s1 < 0.5 * s0) and (s2 < 0.8 * s1)
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_opencl is False, 'no opencl support')
    def test_wavefront_near_field_cl(self):
        msg = "testing for wavefront near field propagation (OpenCL)"
        op = clop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.PropagateNearField(100e-6) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_opencl is False, 'no opencl support')
    def test_wavefront_near_field_cl(self):
        msg = "testing for wavefront far field propagation (OpenCL)"
        op = clop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.PropagateFarField(10) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_opencl is False, 'no opencl support')
    def test_wavefront_fractional_cl(self):
        msg = "testing for wavefront fractional propagation (OpenCL)"
        op = clop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.PropagateFRT(0.2) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_opencl is False, 'no opencl support')
    def test_wavefront_paganin_cl(self):
        msg = "testing for wavefront paganin reconstruction (OpenCL)"
        op = clop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.BackPropagatePaganin() * op.PropagateNearField(100e-6) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_opencl is False, 'no opencl support')
    def test_wavefront_compare_cpu_cl(self):
        msg = "testing for wavefront CPU and OpenCL comparison"
        op = clop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w1 = w.copy()
            ok = np.allclose(w.get(), w1.get(), rtol=1e-5, atol=1e-5)
            w = cpuop.PropagateFarField(1) * cpuop.PropagateNearField(100e-6) * w
            w1 = op.PropagateFarField(1) * op.PropagateNearField(100e-6) * w1
            ok = np.allclose(w.get(), w1.get(), rtol=1e-5, atol=1e-5) and ok
            w = wavefront.Wavefront(d="ascent", pixel_size=1e-6, copy_d=False)
            w1 = w.copy()
            w = cpuop.ThinLens(3) * cpuop.RectangularMask(width=100e-6, height=50e-6) * w
            w1 = op.ThinLens(3) * op.RectangularMask(width=100e-6, height=50e-6) * w1
            ok = np.allclose(w.get(), w1.get(), rtol=1e-4, atol=1e-4) and ok
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_wavefront_ft_cu(self):
        msg = "testing for wavefront FT (CUDA)"
        op = cuop
        try:
            w = wavefront.Wavefront(d="ascent", pixel_size=1e-6, copy_d=False)
            s0 = (np.abs(w.get()) ** 2).sum()
            w = op.IFT() * w
            s1 = (np.abs(w.get()) ** 2).sum()
            w = op.FT() * w
            s2 = (np.abs(w.get()) ** 2).sum()
            ok = np.isclose(s0, s1, rtol=1e-4) and np.isclose(s0, s2, rtol=1e-4)
            if not ok:
                print(s0, s1, s2)
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_wavefront_filter_cu(self):
        msg = "testing for wavefront masks (CUDA)"
        op = cuop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            s0 = w.get().sum()
            w = op.RectangularMask(width=100e-6, height=50e-6) * w
            s1 = w.get().sum()
            w = op.CircularMask(25e-6) * w
            s2 = w.get().sum()
            ok = (np.isnan(w.get()).sum() == 0) and (s1 < 0.5 * s0) and (s2 < 0.8 * s1)
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_wavefront_near_field_cu(self):
        msg = "testing for wavefront near field propagation (CUDA)"
        op = cuop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.PropagateNearField(100e-6) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_wavefront_near_field_cu(self):
        msg = "testing for wavefront far field propagation (CUDA)"
        op = cuop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.PropagateFarField(10) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_wavefront_fractional_cu(self):
        msg = "testing for wavefront fractional propagation (CUDA)"
        op = cuop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.PropagateFRT(0.2) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)

    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_wavefront_paganin_cu(self):
        msg = "testing for wavefront paganin reconstruction (CUDA)"
        op = cuop
        try:
            w = wavefront.Wavefront(d="face", pixel_size=1e-6, copy_d=False)
            w = op.BackPropagatePaganin() * op.PropagateNearField(100e-6) * w
            ok = np.isnan(w.get()).sum() == 0
        except ImportError:
            msg += "\n" + traceback.format_exc()
            ok = False
        self.assertTrue(ok, msg=msg)


def suite():
    testsuite = unittest.TestSuite()
    for k, v in TestWavefront.__dict__.items():
        if isinstance(v, types.FunctionType):
            testsuite.addTest(TestWavefront(k))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
