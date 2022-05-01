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


class TestCUDA(unittest.TestCase):
    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    def test_pycuda(self):
        msg = "Testing for pycuda - this can fail if you only use OpenCL or CPU calculations"
        try:
            import pycuda
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    def test_mako(self):
        msg = "Testing for mako (needed for pycuda reduction kernels)"
        try:
            import mako
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    def test_cuda_gpu_device(self):
        msg = "Number of CUDA devices should be > 0"
        import pycuda.driver as cu_drv
        nb_cuda_gpu_device = cu_drv.Device.count()
        self.assertNotEqual(nb_cuda_gpu_device, 0, msg=msg)

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    def test_cuda_processing_unit(self):
        from pynx.processing_unit import cu_processing_unit
        from pynx.processing_unit import default_processing_unit
        pu = cu_processing_unit.CUProcessingUnit()
        pu.init_cuda(gpu_name=None, verbose=False, test_fft=False)
        default_processing_unit.set_device(pu.cu_device, verbose=False, test_fft=False)

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    def test_pycuda_reduction(self):
        msg = "Testing pycuda sum (reduction) kernel on GPU"
        import numpy as np
        import pycuda.gpuarray as cua
        from pycuda.reduction import ReductionKernel as CU_RedK
        import pycuda.driver as cu_drv
        from pynx.processing_unit.cu_resources import cu_resources
        cu_ctx = cu_resources.get_context(cu_drv.Device(0))
        cu_psi = cua.empty((100,), np.float32)
        cu_psi.fill(np.float32(1))
        cu_sum = CU_RedK(np.float32, neutral="0", reduce_expr="a+b", map_expr="psi[i]", arguments="float* psi")
        kernel_sum = cu_sum(cu_psi).get()
        # cu_drv.Context.pop()
        self.assertAlmostEqual(kernel_sum, 100, msg=msg)

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    def test_cuda_elementwise(self):
        msg = "Testing pycuda sum (elementwise) kernel on GPU"
        import numpy as np
        import pycuda.gpuarray as cua
        from pycuda.elementwise import ElementwiseKernel as CU_ElK
        import pycuda.driver as cu_drv
        from pynx.processing_unit.cu_resources import cu_resources
        cu_ctx = cu_resources.get_context(cu_drv.Device(0))
        cu_psi1 = cua.empty((100,), np.float32)
        cu_psi1.fill(np.float32(1))
        cu_psi2 = cua.empty((100,), np.float32)
        cu_psi2.fill(np.float32(1))
        s = (cu_psi1.get() + cu_psi2.get()).sum()
        cu_sum = CU_ElK(name='cu_add', operation="dest[i] = src[i] + dest[i]", options=None,
                        arguments="float *src, float *dest")
        cu_sum(cu_psi1, cu_psi2)
        # cu_drv.Context.pop()
        self.assertAlmostEqual(cu_psi2.get().sum(), s, msg=msg)

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    def test_cuda_device_available_gpu_speed(self):
        msg = "Testing pynx.processing_unit.cuda_device.available_gpu_speed()"
        try:
            from pynx.processing_unit.cuda_device import available_gpu_speed
            tmp = available_gpu_speed(ranking='fft', fft_shape=(16, 32, 32), verbose=False)
            tmp = available_gpu_speed(ranking='bandwidth', verbose=False)
            ok = True
        except:
            msg += "\n" + traceback.format_exc()
            ok = False

        self.assertTrue(ok, msg=msg)

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    def test_cu_processing_unit_init(self):
        msg = "Testing pynx.processing_unit.cu_processing_unit.CUProcessingUnit.init_cuda"
        try:
            from pynx.processing_unit.cu_processing_unit import CUProcessingUnit
            u = CUProcessingUnit()
            u.init_cuda(fft_size=(32, 64, 64), verbose=False)
            ok = True
        except:
            msg += "\n" + traceback.format_exc()
            ok = False

        self.assertTrue(ok, msg=msg)

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    def test_processing_unit_select_gpu_cu(self):
        msg = "Testing pynx.processing_unit.ProcessingUnit.select_gpu(language='cuda')"
        try:
            from pynx.processing_unit import ProcessingUnit
            u = ProcessingUnit()
            u.set_benchmark_fft_parameters((16, 32, 32), batch=True)
            u.select_gpu(language='cuda', verbose=False)
            ok = True
        except:
            msg += "\n" + traceback.format_exc()
            ok = False

        self.assertTrue(ok, msg=msg)


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite([loadTests(TestCUDA)])
    return test_suite


if __name__ == '__main__':
    res = unittest.TextTestRunner(verbosity=2, descriptions=False).run(suite())
