# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import sys
import unittest
import traceback
import os

exclude_cuda = False
exclude_opencl = False
if 'PYNX_PU' in os.environ:
    if 'opencl' in os.environ['PYNX_PU'].lower():
        exclude_cuda = True
    elif 'cuda' in os.environ['PYNX_PU'].lower():
        exclude_opencl = True


class TestOpenCL(unittest.TestCase):
    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    def test_pyopencl(self):
        msg = "Testing for pyopencl - this can fail if you only use CUDA or CPU calculations"
        try:
            import pyopencl
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    def test_mako(self):
        msg = "Testing for mako (needed for pyopencl reduction kernels)"
        try:
            import mako
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    def test_opencl_gpu(self):
        msg = "Searching for an OpenCL GPU device"
        opencl_gpu_device = None
        try:
            import pyopencl as cl
            for p in cl.get_platforms():
                for d in p.get_devices():
                    if d.type & cl.device_type.GPU:
                        opencl_gpu_device = d
                        break
                if opencl_gpu_device is not None:
                    break
        except:
            msg += "\n" + traceback.format_exc()
        self.assertTrue(opencl_gpu_device is not None, msg=msg)

    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    def test_opencl_reduction(self):
        msg = "Testing pyopencl sum (reduction) kernel on GPU"
        kernel_sum = 0
        try:
            import numpy as np
            import pyopencl as cl
            import pyopencl.array as cla
            from pyopencl.elementwise import ElementwiseKernel as CL_ElK
            from pyopencl.reduction import ReductionKernel as CL_RedK
            opencl_gpu_device = None
            for p in cl.get_platforms():
                for d in p.get_devices():
                    if d.type & cl.device_type.GPU:
                        opencl_gpu_device = d
                        break
                if opencl_gpu_device is not None:
                    break
            from pynx.processing_unit.cl_processing_unit import CLProcessingUnit
            cl_ctx = CLProcessingUnit.get_context(opencl_gpu_device)
            cl_queue = cl.CommandQueue(cl_ctx)
            cl_psi = cla.empty(cl_queue, (100,), np.float32)
            cl_psi.fill(np.float32(1))
            cl_sum = CL_RedK(cl_ctx, np.float32, neutral="0", reduce_expr="a+b", map_expr="psi[i]",
                             arguments="__global float* psi")
            kernel_sum = cl_sum(cl_psi).get()
        except:
            msg += "\n" + traceback.format_exc()
        self.assertAlmostEqual(kernel_sum, 100, msg=msg)

    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    def test_opencl_elementwise(self):
        msg = "Testing pyopencl sum (elementwise) kernel on GPU"
        sum = 0
        try:
            import numpy as np
            import pyopencl as cl
            import pyopencl.array as cla
            from pyopencl.elementwise import ElementwiseKernel as CL_ElK
            from pyopencl.reduction import ReductionKernel as CL_RedK
            opencl_gpu_device = None
            for p in cl.get_platforms():
                for d in p.get_devices():
                    if d.type & cl.device_type.GPU:
                        opencl_gpu_device = d
                        break
                if opencl_gpu_device is not None:
                    break
            from pynx.processing_unit.cl_processing_unit import CLProcessingUnit
            cl_ctx = CLProcessingUnit.get_context(opencl_gpu_device)
            cl_queue = cl.CommandQueue(cl_ctx)
            cl_psi1 = cla.empty(cl_queue, (100,), np.float32)
            cl_psi2 = cla.empty(cl_queue, (100,), np.float32)
            cl_psi1.fill(np.float32(1))
            cl_psi2.fill(np.float32(1))
            cl_sum = CL_ElK(cl_ctx, name='cl_sum', operation="dest[i] += src[i]",
                            arguments="__global float *src, __global float *dest")
            cl_sum(cl_psi1, cl_psi2)
            sum = cl_psi2.get().sum()
        except:
            msg += "\n" + traceback.format_exc()
        self.assertAlmostEqual(sum, 200, msg=msg)

    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    def test_opencl_device_available_gpu_speed(self):
        msg = "Testing pynx.processing_unit.opencl_device.available_gpu_speed()"
        try:
            from pynx.processing_unit.opencl_device import available_gpu_speed
            tmp = available_gpu_speed(ranking='fft', fft_shape=(16, 32, 32), verbose=False)
            tmp = available_gpu_speed(ranking='bandwidth', verbose=False)
            ok = True
        except:
            msg += "\n" + traceback.format_exc()
            ok = False

        self.assertTrue(ok, msg=msg)

    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    def test_cl_processing_unit_init(self):
        msg = "Testing pynx.processing_unit.cl_processing_unit.CLProcessingUnit.init_cl"
        try:
            from pynx.processing_unit.cl_processing_unit import CLProcessingUnit
            u = CLProcessingUnit()
            u.init_cl(fft_size=(32, 64, 64), verbose=False)
            ok = True
        except:
            msg += "\n" + traceback.format_exc()
            ok = False

        self.assertTrue(ok, msg=msg)

    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    def test_processing_unit_select_gpu_cl(self):
        msg = "Testing pynx.processing_unit.ProcessingUnit.select_gpu(language='opencl')"
        try:
            from pynx.processing_unit import ProcessingUnit
            u = ProcessingUnit()
            u.set_benchmark_fft_parameters((16, 32, 32), batch=True)
            u.select_gpu(language='opencl', verbose=False)
            ok = True
        except:
            msg += "\n" + traceback.format_exc()
            ok = False

        self.assertTrue(ok, msg=msg)


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite([loadTests(TestOpenCL)])
    return test_suite


if __name__ == '__main__':
    res = unittest.TextTestRunner(verbosity=2, descriptions=False).run(suite())
