# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import platform
import types
import timeit
import os
import warnings

import psutil
import gc
import numpy as np
from .holotomo import algo_string

from ..processing_unit.cu_processing_unit import CUProcessingUnit
import pycuda.driver as cu_drv
import pycuda.gpuarray as cua
from pycuda.elementwise import ElementwiseKernel as CU_ElK
from pycuda.reduction import ReductionKernel as CU_RedK
from pycuda.compiler import SourceModule
import pycuda.curandom as cur
import pycuda.tools as cu_tools

from ..processing_unit import default_processing_unit as main_default_processing_unit
from ..processing_unit.cu_processing_unit import CUProcessingUnit
from ..processing_unit.kernel_source import get_kernel_source as getks
from ..operator import has_attr_not_none, OperatorSum, OperatorPower, OperatorException
from . import cpu_operator as cpuop

from .holotomo import HoloTomo, HoloTomoDataStack, HoloTomoData, OperatorHoloTomo

my_float4 = cu_tools.get_or_register_dtype("my_float4",
                                           np.dtype([('a', '<f4'), ('b', '<f4'), ('c', '<f4'), ('d', '<f4')]))

# half = cu_tools.get_or_register_dtype("half", np.float16)
half = cu_tools.get_or_register_dtype("half", np.float32)


################################################################################################
# Patch HoloTomo class so that we can use 5*w to scale it.
# OK, so this might be ugly. There will definitely be issues if several types of operators
# are imported (e.g. OpenCL and CUDA)
# Solution (?): in a different sub-module, implement dynamical type-checking to decide which
# Scale() operator to call.


def patch_method(cls):
    def __rmul__(self, x):
        # Multiply object by a scalar.
        if np.isscalar(x) is False:
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s" % (str(x), str(self)))
        return Scale(x) * self

    def __mul__(self, x):
        # Multiply object by a scalar.
        if np.isscalar(x) is False:
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s" % (str(x), str(self)))
        return self * Scale(x)

    cls.__rmul__ = __rmul__
    cls.__mul__ = __mul__


patch_method(HoloTomo)


################################################################################################


class CUProcessingUnitHoloTomo(CUProcessingUnit):
    """
    Processing unit in CUDA space, for operations on HoloTomo objects.

    Handles initializing the context and kernels.
    """

    def __init__(self):
        super(CUProcessingUnitHoloTomo, self).__init__()
        # Stream for calculations (don't use default stream which can be blocking)
        self.cu_stream = None
        # Stream to copy data between host and GPU
        self.cu_stream_swap = None
        # Event recording when last swapping object in & out is finished
        self.cu_event_swap_obj = None
        # Event for calculation stream
        self.cu_event_calc = None
        # Event for swap stream
        self.cu_event_swap = None
        # Memory pool
        self.cu_mem_pool = None
        # kernels dictionaries as a function of the number of modes
        self._modes_kernels = {}
        # kernels dictionaries as a function of the number of modes and distances
        self._modes_nz_kernels = {}

    def init_cuda(self, cu_ctx=None, cu_device=None, fft_size=(1, 1024, 1024), batch=True, gpu_name=None, test_fft=True,
                  verbose=True):
        """
        Derived init_cuda function. Also creates in/out queues for parallel processing of large datasets.

        :param cu_ctx: pycuda.driver.Context. If none, a default context will be created
        :param cu_device: pycuda.driver.Device. If none, and no context is given, the fastest GPu will be used.
        :param fft_size: the fft size to be used, for benchmark purposes when selecting GPU. different fft sizes
                         can be used afterwards?
        :param batch: if True, will benchmark using a batch 2D FFT
        :param gpu_name: a (sub)string matching the name of the gpu to be used
        :param test_fft: if True, will benchmark the GPU(s)
        :param verbose: report the GPU found and their speed
        :return: nothing
        """
        super(CUProcessingUnitHoloTomo, self).init_cuda(cu_ctx=cu_ctx, cu_device=cu_device, fft_size=fft_size,
                                                        batch=batch,
                                                        gpu_name=gpu_name, test_fft=test_fft, verbose=verbose)
        # Stream for calculations (don't use the default stream which can be blocking)
        self.cu_stream = cu_drv.Stream()
        self.cu_stream_swap = cu_drv.Stream()
        self.cu_event_swap_obj = cu_drv.Event()
        self.cu_event_calc = cu_drv.Event()
        self.cu_event_swap = cu_drv.Event()

        # # Init skcuda.linalg
        # cu_linalg.init(allocator=self.cu_mem_pool.allocate)

        # Disable CUDA helf operators, which lead to errors such as:
        #   more than one instance of overloaded function "operator-" has "C" linkage
        # Unfortunately this depends on the platform/compiler, so need a test for this
        try:
            testk = CU_ElK(name='testk', operation="d[i] *= 2", preamble='#include "cuda_fp16.h"',
                           options=self.cu_options, arguments="float *d")
            cu_d = cua.empty(128, dtype=np.float32)
            testk(cu_d)
        except cu_drv.CompileError:
            print("CUProcessingUnitHoloTomo:init_cuda(): disabling CUDA half operators")
            self.cu_options.append("-D__CUDA_NO_HALF_OPERATORS__")
            self.cu_options.append("-D__CUDA_NO_HALF2_OPERATORS__")

    def cu_init_kernels(self):
        print("HoloTomo CUDA processing unit: compiling kernels...")
        t0 = timeit.default_timer()
        # Elementwise kernels
        self.cu_scale = CU_ElK(name='cu_scale',
                               operation="d[i] = complexf(d[i].real() * scale, d[i].imag() * scale )",
                               preamble=getks('cuda/complex.cu'),
                               options=self.cu_options, arguments="pycuda::complex<float> *d, const float scale")

        self.cu_sum = CU_ElK(name='cu_sum', operation="dest[i] += src[i]",
                             options=self.cu_options,
                             arguments="pycuda::complex<float> *src, pycuda::complex<float> *dest")

        self.cu_scale_complex = CU_ElK(name='cu_scale_complex',
                                       operation="d[i] = complexf(d[i].real() * s.real() - d[i].imag() * s.imag(),"
                                                 "d[i].real() * s.imag() + d[i].imag() * s.real())",
                                       preamble=getks('cuda/complex.cu'),
                                       options=self.cu_options,
                                       arguments="pycuda::complex<float> *d, const pycuda::complex<float> s")

        self.cu_iobs2psi = CU_ElK(name='cu_iobs2psi',
                                  operation="Iobs2Psi(i, iobs, iobs_empty, psi, dx, dy,"
                                            "nb_mode, nx, ny, nz, padding)",
                                  preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                           getks('holotomo/cuda/paganin_elw.cu'),
                                  options=self.cu_options,
                                  arguments="float *iobs, float *iobs_empty, pycuda::complex<float> *psi,"
                                            "float* dx, float* dy, const int nb_mode, const int nx,"
                                            "const int ny, const int nz, const int padding")

        self.cu_iobs_empty2probe = CU_ElK(name='cu_iobs_empty2probe',
                                          operation="IobsEmpty2Probe(i, iobs_empty, probe, nb_probe, nx, ny, nz,"
                                                    "padding)",
                                          preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                   getks('holotomo/cuda/paganin_elw.cu'),
                                          options=self.cu_options,
                                          arguments="float *iobs_empty, pycuda::complex<float> *probe,"
                                                    "const int nb_probe, const int nx, const int ny, const int nz,"
                                                    "const int padding")

        self.cu_quad_phase = CU_ElK(name='cu_quad_phase',
                                    operation="QuadPhase(i, d, f, forward, scale, nb_z, nb_mode, nx, ny)",
                                    preamble=getks('cuda/complex.cu') + getks('holotomo/cuda/quad_phase_elw.cu'),
                                    options=self.cu_options,
                                    arguments="pycuda::complex<float> *d, float *f, const bool forward,"
                                              "const float scale, const int nb_z, const int nb_mode,"
                                              "const int nx, const int ny")

        self.cu_calc2obs = CU_ElK(name='cu_calc2obs',
                                  operation="Calc2Obs(i, iobs, psi, nb_mode, nx, ny)",
                                  preamble=getks('cuda/complex.cu') + getks('holotomo/cuda/calc2obs_elw.cu'),
                                  options=self.cu_options,
                                  arguments="float *iobs,pycuda::complex<float> *psi, const int nb_mode, const int nx,"
                                            "const int ny")

        self.cu_obj_probez_mult = CU_ElK(name='cu_obj_probez_mult',
                                         operation="ObjectProbeZMult(i, obj, probe, psi, dx, dy, sample_flag, nproj,"
                                                   "nb_z, nb_obj, nb_probe, nx, ny)",
                                         preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                  getks('holotomo/cuda/obj_probe_mult_elw.cu'),
                                         options=self.cu_options,
                                         arguments="pycuda::complex<float>* obj, pycuda::complex<float> *probe,"
                                                   "pycuda::complex<float>* psi, float* dx, float* dy,"
                                                   "signed char* sample_flag, const int nproj, const int nb_z,"
                                                   "const int nb_obj, const int nb_probe, const int nx, const int ny")

        self.cu_obj_probez_mult_raar = \
            CU_ElK(name='cu_obj_probez_mult_raar',
                   operation="ObjectProbeZMultRAAR(i, obj, probe, psi, psiold, dx, dy,"
                             "sample_flag, nproj, nb_z, nx, ny, beta)",
                   preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                            getks('holotomo/cuda/obj_probe_mult_raar_elw.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float>* obj, pycuda::complex<float> *probe,"
                             "pycuda::complex<float>* psi, pycuda::complex<float>* psiold,"
                             "float* dx, float* dy,"
                             "signed char* sample_flag, const int nproj, const int nb_z,"
                             "const int nx, const int ny, const float beta")

        self.cu_obj_probez_mult_drap = \
            CU_ElK(name='cu_obj_probez_mult_drap',
                   operation="ObjectProbeZMultDRAP(i, obj, probe, psi, psiold, dx, dy,"
                             "sample_flag, nproj, nb_z, nx, ny)",
                   preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                            getks('holotomo/cuda/obj_probe_mult_drap_elw.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float>* obj, pycuda::complex<float> *probe,"
                             "pycuda::complex<float>* psi, pycuda::complex<float>* psiold,"
                             "float* dx, float* dy,"
                             "signed char* sample_flag, const int nproj, const int nb_z,"
                             "const int nx, const int ny")

        self.cu_obj_probecohz_mult = CU_ElK(name='cu_obj_probecohz_mult',
                                            operation="ObjectProbeCohZMult(i, obj, probe, psi, dx, dy, sample_flag, "
                                                      "probe_coeffs, nproj, nb_z, nb_probe, nx, ny)",
                                            preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                     getks('holotomo/cuda/obj_probe_mult_elw.cu'),
                                            options=self.cu_options,
                                            arguments="pycuda::complex<float>* obj, pycuda::complex<float> *probe,"
                                                      "pycuda::complex<float>* psi, float* dx, float* dy,"
                                                      "signed char* sample_flag, float* probe_coeffs,"
                                                      "const int nproj, const int nb_z,"
                                                      "const int nb_probe, const int nx, const int ny")

        self.cu_obj_probecohz_mult_raar = \
            CU_ElK(name='cu_obj_probecohz_mult_raar',
                   operation="ObjectProbeCohZMultRAAR(i, obj, probe, psi, psiold, dx, dy, sample_flag, "
                             "probe_coeffs, nproj, nb_z, nb_probe, nx, ny, beta)",
                   preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                            getks('holotomo/cuda/obj_probe_mult_raar_elw.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float>* obj, pycuda::complex<float> *probe,"
                             "pycuda::complex<float>* psi, pycuda::complex<float>* psiold,"
                             "float* dx, float* dy,"
                             "signed char* sample_flag, float* probe_coeffs,"
                             "const int nproj, const int nb_z,"
                             "const int nb_probe, const int nx, const int ny, const float beta")

        self.cu_obj_probecohz_mult_drap = \
            CU_ElK(name='cu_obj_probecohz_mult_drap',
                   operation="ObjectProbeCohZMultDRAP(i, obj, probe, psi, psiold, dx, dy, sample_flag, "
                             "probe_coeffs, nproj, nb_z, nb_probe, nx, ny)",
                   preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                            getks('holotomo/cuda/obj_probe_mult_drap_elw.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float>* obj, pycuda::complex<float> *probe,"
                             "pycuda::complex<float>* psi, pycuda::complex<float>* psiold,"
                             "float* dx, float* dy,"
                             "signed char* sample_flag, float* probe_coeffs,"
                             "const int nproj, const int nb_z,"
                             "const int nb_probe, const int nx, const int ny")

        self.cu_probe2psi = CU_ElK(name='cu_probe2psi',
                                   operation="Probe2Psi(i, probe, psi, probe_coeffs,"
                                             "nproj, nz, nb_probe, nx, ny)",
                                   preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                            getks('holotomo/cuda/obj_probe_mult_elw.cu'),
                                   options=self.cu_options,
                                   arguments="pycuda::complex<float> *probe, pycuda::complex<float>* psi,"
                                             "float* probe_coeffs, const int nproj, const int nz,"
                                             "const int nb_probe, const int nx, const int ny")

        self.cu_obj_probe2psi_dm1 = CU_ElK(name='obj_probe2psi_dm1',
                                           operation="ObjectProbe2PsiDM1(i, obj, probe, psi, dx, dy, sample_flag, nproj,"
                                                     "nb_z, nb_obj, nb_probe, nx, ny)",
                                           preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                    getks('holotomo/cuda/obj_probe_mult_dm_elw.cu'),
                                           options=self.cu_options,
                                           arguments="pycuda::complex<float>* obj, pycuda::complex<float> *probe,"
                                                     "pycuda::complex<float>* psi, float* dx, float* dy,"
                                                     "signed char* sample_flag, const int nproj, const int nb_z,"
                                                     "const int nb_obj, const int nb_probe, const int nx, const int ny")

        self.cu_obj_probe2psi_dm2 = CU_ElK(name='obj_probe2psi_dm2',
                                           operation="ObjectProbe2PsiDM2(i, obj, probe, psi, psi_old, dx, dy,"
                                                     "sample_flag, nproj, nb_z, nb_obj, nb_probe, nx, ny)",
                                           preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                    getks('holotomo/cuda/obj_probe_mult_dm_elw.cu'),
                                           options=self.cu_options,
                                           arguments="pycuda::complex<float>* obj, pycuda::complex<float> *probe,"
                                                     "pycuda::complex<float>* psi, pycuda::complex<float>* psi_old,"
                                                     "float* dx, float* dy, signed char* sample_flag,"
                                                     "const int nproj, const int nb_z, const int nb_obj,"
                                                     "const int nb_probe, const int nx, const int ny")

        self.cu_psi2obj_probe_coh = \
            CU_ElK(name="cu_psi2obj_probe_coh",
                   operation="Psi2ObjProbeCohMode(i, obj, obj_old, probe, psi, probe_new, probe_coeffs,"
                             "obj_phase0, dx, dy, sample_flag, nb_z, nb_obj, nb_probe, nx, ny,"
                             "obj_min, obj_max, reg_obj_smooth, beta_delta,"
                             "weight_empty)",
                   preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                            getks('holotomo/cuda/psi2obj_probe_elw.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float>* obj, pycuda::complex<float>* obj_old,"
                             "pycuda::complex<float> *probe, pycuda::complex<float>* psi,"
                             "pycuda::complex<float>* probe_new,float* probe_coeffs,"
                   # "half* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                             "float* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                             "const int nb_z, const int nb_obj, const int nb_probe, const int nx,"
                             "const int ny, const float obj_min, const float obj_max,"
                             "const float reg_obj_smooth,"
                             "const float beta_delta, const float weight_empty")

        self.cu_psi2probemerge = CU_ElK(name='cu_psi2probemerge',
                                        operation="Psi2ProbeMerge(i, probe, probe_new, probe_norm,"
                                                  "inertia, nb_probe, nxy, nz)",
                                        preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                 getks('holotomo/cuda/psi2obj_probe_elw.cu'),
                                        options=self.cu_options,
                                        arguments="pycuda::complex<float> *probe, pycuda::complex<float> *probe_new,"
                                                  "const float *probe_norm, const float inertia,"
                                                  "const int nb_probe, const int nxy, const int nz")

        self.cu_psi2probemerge_coh = CU_ElK(name='cu_psi2probemerge_coh',
                                            operation="Psi2ProbeMergeCoh(i, probe, probe_new, probe_norm,"
                                                      "inertia, nb_probe, nxy, nz)",
                                            preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                     getks('holotomo/cuda/psi2obj_probe_elw.cu'),
                                            options=self.cu_options,
                                            arguments="pycuda::complex<float> *probe, pycuda::complex<float> *probe_new,"
                                                      "const float *probe_norm, const float inertia,"
                                                      "const int nb_probe, const int nxy, const int nz")

        self.cu_probe_norm = CU_ElK(name='cu_probe_norm',
                                    operation="probe[i] *= sqrtf((float)nxy / norm[0])",
                                    preamble=getks('cuda/complex.cu'),
                                    options=self.cu_options,
                                    arguments="pycuda::complex<float> *probe, const float *norm, const int nxy")

        self.cu_coeff_norm = CU_ElK(name='cu_coeff_norm',
                                    operation="coeff[i * stride] *= sqrtf(norm[0] / (float)nxy)",
                                    preamble=getks('cuda/complex.cu'),
                                    options=self.cu_options,
                                    arguments="float *coeff, const float *norm, const int nxy,"
                                              "const int stride")

        self.cu_paganin_fourier = CU_ElK(name='cu_paganin_fourier',
                                         operation="paganin_fourier(i, iobs, psi, alpha, px, nb_mode, nx, ny, nz)",
                                         preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                  getks('holotomo/cuda/paganin_elw.cu'),
                                         options=self.cu_options,
                                         arguments="float *iobs, pycuda::complex<float> *psi, float* alpha,"
                                                   "const float px, const int nb_mode, const int nx, const int ny,"
                                                   "const int nz")

        self.cu_paganin_thickness = CU_ElK(name='cu_paganin_thickness',
                                           operation="paganin_thickness(i, iobs, obj, psi, obj_phase0, iz0, delta_beta,"
                                                     "nb_probe, nobj, nx, ny, nz)",
                                           preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                    getks('holotomo/cuda/paganin_elw.cu'),
                                           options=self.cu_options,
                                           arguments="float * iobs, pycuda::complex<float>* obj,"
                                                     "pycuda::complex<float> *psi, half* obj_phase0, const int iz0,"
                                                     "const float delta_beta, const int nb_probe, const int nobj,"
                                                     "const int nx, const int ny, const int nz")

        self.cu_paganin_fourier_multi = CU_ElK(name='cu_paganin_fourier_multi',
                                               operation="paganin_fourier_multi(i, psi, pilambdad, delta_beta,"
                                                         "px, nb_mode, nx, ny, nz, nb_proj, alpha)",
                                               preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                        getks('holotomo/cuda/paganin_elw.cu'),
                                               options=self.cu_options,
                                               arguments="pycuda::complex<float> *psi, float* pilambdad,"
                                                         "const float delta_beta, const float px, const int nb_mode,"
                                                         "const int nx, const int ny, const int nz,"
                                                         "const int nb_proj, const float alpha")

        self.cu_paganin2obj = CU_ElK(name='cu_paganin2obj',
                                     operation="paganin2obj(i, psi, obj, obj_phase0, delta_beta,"
                                               "nb_probe, nb_obj, nx, ny, nz, nb_proj)",
                                     preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                              getks('holotomo/cuda/paganin_elw.cu'),
                                     options=self.cu_options,
                                     arguments="pycuda::complex<float> *psi, pycuda::complex<float>* obj,"
                                               "half* obj_phase0, const float delta_beta,"
                                               "const int nb_probe, const int nb_obj, const int nx, const int ny,"
                                               "const int nz, const int nb_proj")

        self.cu_projection_amplitude = CU_ElK(name='cu_projection_amplitude',
                                              operation="ProjectionAmplitude(i, iobs, psi, nb_mode, nx, ny)",
                                              preamble=getks('cuda/complex.cu') +
                                                       getks('holotomo/cuda/projection_amplitude_elw.cu'),
                                              options=self.cu_options,
                                              arguments="float *iobs, pycuda::complex<float> *psi, const int nb_mode,"
                                                        "const int nx, const int ny")

        self.cu_subtract_mean = CU_ElK(name='cu_subtract_mean',
                                       operation="psi[i] -= sum[0] / d",
                                       preamble=getks('cuda/complex.cu'),
                                       options=self.cu_options,
                                       arguments="pycuda::complex<float> *psi, pycuda::complex<float>* sum,"
                                                 "const float d")

        self.cu_ctf_fourier = CU_ElK(name='cu_ctf_fourier',
                                     operation="ctf_fourier(i, psi, pilambdad, px, nb_mode, nx, ny, nz,"
                                               "nb_proj, alpha)",
                                     preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                              getks('holotomo/cuda/ctf_elw.cu'),
                                     options=self.cu_options,
                                     arguments="pycuda::complex<float> *psi, float* pilambdad,"
                                               "const float px, const int nb_mode, const int nx, const int ny,"
                                               "const int nz, const int nb_proj, const float alpha")

        self.cu_ctf_fourier_homogeneous = CU_ElK(name='cu_ctf_fourier_homogenous',
                                                 operation="ctf_fourier_homogeneous(i, psi, pilambdad,"
                                                           "delta_beta, px, nb_mode, nx, ny, nz, nb_proj,"
                                                           "alpha, alpha_high, sigma)",
                                                 preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                          getks('holotomo/cuda/ctf_elw.cu'),
                                                 options=self.cu_options,
                                                 arguments="pycuda::complex<float> *psi, float* pilambdad,"
                                                           "const float delta_beta, const float px, const int nb_mode,"
                                                           "const int nx, const int ny, const int nz,"
                                                           "const int nb_proj, const float alpha,"
                                                           "const float alpha_high, const float sigma")

        self.cu_ctf_phase2obj = CU_ElK(name='cu_ctf_phase2obj',
                                       operation="ctf_phase2obj(i, psi, obj, obj_phase0,"
                                                 "nb_probe, nb_obj, nx, ny, nz, nb_proj, delta_beta)",
                                       preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                getks('holotomo/cuda/ctf_elw.cu'),
                                       options=self.cu_options,
                                       arguments="pycuda::complex<float> *psi, pycuda::complex<float>* obj,"
                                                 "half* obj_phase0,"
                                                 "const int nb_probe, const int nb_obj, const int nx, const int ny,"
                                                 "const int nz, const int nb_proj, const float delta_beta")

        self.cu_psi2probe = CU_ElK(name="psi2probe",
                                   operation="Psi2Probe(i, obj, probe, psi, probe_new, probe_norm, dx,"
                                             "dy, sample_flag, nb_z, nb_obj, nb_probe, nx, ny,"
                                             "weight_empty)",
                                   preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                            getks('holotomo/cuda/psi2obj_probe_elw.cu'),
                                   options=self.cu_options,
                                   arguments="pycuda::complex<float>* obj, pycuda::complex<float> *probe,"
                                             "pycuda::complex<float>* psi, pycuda::complex<float>* probe_new,"
                                             "float* probe_norm, float* dx, float* dy, signed char* sample_flag,"
                                             "const int nb_z, const int nb_obj, const int nb_probe, const int nx,"
                                             "const int ny, const float weight_empty")

        self.cu_psi2probe_raar = \
            CU_ElK(name="psi2probe_raar",
                   operation="Psi2ProbeRAAR(i, obj, probe, psi, psiold, probe_new, probe_norm, dx,"
                             "dy, sample_flag, nb_z, nx, ny, weight_empty, beta)",
                   preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                            getks('holotomo/cuda/psi2obj_probe_raar_elw.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float>* obj, pycuda::complex<float> *probe,"
                             "pycuda::complex<float>* psi, pycuda::complex<float>* psiold,"
                             "pycuda::complex<float>* probe_new,"
                             "float* probe_norm, float* dx, float* dy, signed char* sample_flag,"
                             "const int nb_z, const int nx, const int ny,"
                             "const float weight_empty, const float beta")

        self.cu_psi2probe_drap = \
            CU_ElK(name="psi2probe_drap",
                   operation="Psi2ProbeDRAP(i, obj, probe, psi, psiold, probe_new, probe_norm, dx,"
                             "dy, sample_flag, nb_z, nx, ny, weight_empty, beta)",
                   preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                            getks('holotomo/cuda/psi2obj_probe_drap_elw.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float>* obj, pycuda::complex<float> *probe,"
                             "pycuda::complex<float>* psi, pycuda::complex<float>* psiold,"
                             "pycuda::complex<float>* probe_new,"
                             "float* probe_norm, float* dx, float* dy, signed char* sample_flag,"
                             "const int nb_z, const int nx, const int ny,"
                             "const float weight_empty, const float beta")

        self.cu_psi2reg = CU_ElK(name='cu_psi2reg',
                                 operation="psi2reg1(i, reg, psi, probe, dx, dy, sample_flag, nb_probe, nb_obj, nx, ny, nz, dn)",
                                 preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                          getks('holotomo/cuda/registration.cu') % {"blocksize": 32},
                                 options=self.cu_options,
                                 arguments="pycuda::complex<float> *reg, pycuda::complex<float> *psi,"
                                           "pycuda::complex<float> *probe,"
                                           "float *dx, float *dy, signed char *sample_flag,"
                                           "const int nb_probe, const int nb_obj, const int nx,"
                                           "const int ny, const int nz, const int dn")

        self.cu_reg_mult_conj = CU_ElK(name='cu_reg_mult_conj',
                                       operation="reg_mult_conj(i, reg, sample_flag, iz0, nz, nproj, dn)",
                                       preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                getks('holotomo/cuda/registration.cu') % {"blocksize": 32},
                                       options=self.cu_options,
                                       arguments="pycuda::complex<float> *reg, signed char *sample_flag,"
                                                 "const int iz0, const int nz, const int nproj, const int dn")

        self.cu_phase_1dfilt_highpass = \
            CU_ElK(name='cu_phase_1dfilt_highpass',
                   operation="ph_f[i] *= scale * (i % nx)",
                   preamble=getks('cuda/complex.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float> *ph_f, const float scale, const int nx")

        self.cu_phase2obj = \
            CU_ElK(name='cu_phase2obj',
                   operation="obj[i] = sqrtf(dot(obj[i],obj[i])) * complexf(cosf(ph[i]), sinf(ph[i]))",
                   preamble=getks('cuda/complex.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float> *obj, float*ph")

        self.cu_obj2phase0 = \
            CU_ElK(name='cu_obj2phase0',
                   operation="#define twopi 6.2831853071795862f\n"
                             "#define pi 3.1415926535897932f\n"
                             "const float ph0 = obj_phase0[i];"
                             "float ph = ph0 + fmodf(-atan2f(obj[i].imag(), obj[i].real()) - ph0, twopi);"
                             "if ((ph - ph0) >= pi) ph -= twopi;"
                             "else if ((ph - ph0) < -pi) ph += twopi;"
                             "obj_phase0[i] = ph",
                   preamble=getks('cuda/complex.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float> *obj, float*obj_phase0")

        self.cu_probe_mode_update = \
            CU_ElK(name='cu_probe_mode_update',
                   operation="mode[i] = fmaxf(0.0f, inertia * mode[i] + (1-inertia) * mode_new[i])",
                   options=self.cu_options,
                   arguments="float *mode, float *mode_new, const float inertia")

        # Reduction kernels
        # This will compute Poisson, Gaussian, Euclidian LLK as well as the sum of the calculated intensity
        self.cu_llk = CU_RedK(my_float4, neutral="my_float4(0,0,0,0)", reduce_expr="a+b",
                              preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') +
                                       getks('holotomo/cuda/llk_red.cu'),
                              options=self.cu_options,
                              map_expr="LLKAll(i, iobs, psi, nb_mode, nx, ny)",
                              arguments="float *iobs, pycuda::complex<float> *psi, const int nb_mode, const int nx,"
                                        "const int ny")
        # This is a reduction kernel to update each projection scale factor (is that necessary ?)
        self.cu_psi2obj_probe = CU_RedK(np.float32, neutral=0, reduce_expr="a+b", name="psi2obj_probe",
                                        map_expr="Psi2ObjProbe(i, obj, obj_old, probe, psi, probe_new, probe_norm,"
                                                 "obj_phase0, dx, dy, sample_flag, nb_z, nb_obj, nb_probe, nx, ny,"
                                                 "obj_min, obj_max, reg_obj_smooth, beta_delta,"
                                                 "weight_empty)",
                                        preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                 getks('holotomo/cuda/psi2obj_probe_elw.cu'),
                                        options=self.cu_options,
                                        arguments="pycuda::complex<float>* obj, pycuda::complex<float>* obj_old,"
                                                  "pycuda::complex<float> *probe, pycuda::complex<float>* psi,"
                                                  "pycuda::complex<float>* probe_new,float* probe_norm,"
                                        # "half* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                                                  "float* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                                                  "const int nb_z, const int nb_obj, const int nb_probe, const int nx,"
                                                  "const int ny, const float obj_min, const float obj_max,"
                                                  "const float reg_obj_smooth,"
                                                  "const float beta_delta, const float weight_empty")

        self.cu_psi2obj_probe_raar = \
            CU_RedK(np.float32, neutral=0, reduce_expr="a+b", name="psi2obj_probe_raar",
                    map_expr="Psi2ObjProbeRAAR(i, obj, obj_old, probe, psi, psiold, probe_new, probe_norm,"
                             "obj_phase0, dx, dy, sample_flag, nb_z, nx, ny,"
                             "obj_min, obj_max, reg_obj_smooth, beta_delta, weight_empty, beta)",
                    preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                             getks('holotomo/cuda/psi2obj_probe_raar_elw.cu'),
                    options=self.cu_options,
                    arguments="pycuda::complex<float>* obj, pycuda::complex<float>* obj_old,"
                              "pycuda::complex<float> *probe, pycuda::complex<float>* psi,"
                              "pycuda::complex<float>* psiold,"
                              "pycuda::complex<float>* probe_new,float* probe_norm,"
                    # "half* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                              "float* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                              "const int nb_z, const int nx,"
                              "const int ny, const float obj_min, const float obj_max,"
                              "const float reg_obj_smooth,"
                              "const float beta_delta, const float weight_empty, const float beta")

        self.cu_psi2obj_probe_drap = \
            CU_RedK(np.float32, neutral=0, reduce_expr="a+b", name="psi2obj_probe_drap",
                    map_expr="Psi2ObjProbeDRAP(i, obj, obj_old, probe, psi, psiold, probe_new, probe_norm,"
                             "obj_phase0, dx, dy, sample_flag, nb_z, nx, ny,"
                             "obj_min, obj_max, reg_obj_smooth, beta_delta, weight_empty, beta)",
                    preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                             getks('holotomo/cuda/psi2obj_probe_drap_elw.cu'),
                    options=self.cu_options,
                    arguments="pycuda::complex<float>* obj, pycuda::complex<float>* obj_old,"
                              "pycuda::complex<float> *probe, pycuda::complex<float>* psi,"
                              "pycuda::complex<float>* psiold,"
                              "pycuda::complex<float>* probe_new,float* probe_norm,"
                    # "half* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                              "float* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                              "const int nb_z, const int nx,"
                              "const int ny, const float obj_min, const float obj_max,"
                              "const float reg_obj_smooth,"
                              "const float beta_delta, const float weight_empty, const float beta")

        self.cu_scale_obs_calc = CU_RedK(np.complex64, neutral="complexf(0,0)", name='scale_obs_calc',
                                         reduce_expr="a+b", map_expr="scale_obs_calc(i, obs, calc, nx, ny, nb_mode)",
                                         preamble=getks('cuda/complex.cu') + getks('holotomo/cuda/scale_red.cu'),
                                         arguments="float *obs, pycuda::complex<float> *calc, const int nx,"
                                                   "const int ny, const int nb_mode")

        self.cu_norm_n_c = CU_RedK(np.float32, neutral=0, name='cu_norm_n_c',
                                   reduce_expr="a+b", map_expr="pow(abs(d[i]), exponent)",
                                   preamble=getks('cuda/complex.cu'),
                                   arguments="pycuda::complex<float> *d, const float exponent")

        self.cu_norm2_c = CU_RedK(np.float32, neutral=0, name='cu_norm2_c',
                                  reduce_expr="a+b", map_expr="norm(d[i])",
                                  preamble=getks('cuda/complex.cu'),
                                  arguments="pycuda::complex<float> *d")

        self.cu_psi2pos_red = CU_RedK(my_float4, neutral="my_float4(0)", reduce_expr="a+b",
                                      preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                               getks('cuda/float_n.cu') + getks('holotomo/cuda/psi2pos.cu'),
                                      map_expr="Psi2PosShift(i, psi, obj, probe, dx, dy, nx, ny, interp)",
                                      options=self.cu_options,
                                      arguments="pycuda::complex<float>* psi, pycuda::complex<float>* obj,"
                                                "pycuda::complex<float>* probe, float *dx, float *dy,"
                                                "const int nx, const int ny, const bool interp")
        # psi2posmerge isolated kernel
        # self.cu_psi2pos_merge = CU_ElK(name='cu_psi2pos_merge',
        #                                operation="Psi2PosMerge(dxy, dx, dy, max_shift, mult)",
        #                                options=self.cu_options,
        #                                preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
        #                                         getks('cuda/float_n.cu') + getks('holotomo/cuda/psi2pos.cu'),
        #                                arguments="my_float4* dxy, float* dx, float* dy,"
        #                                          "const float max_shift, const float mult")

        psi2pos_merge_mod = SourceModule(getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                         getks('cuda/float_n.cu') + getks('holotomo/cuda/psi2pos.cu'),
                                         options=self.cu_options)
        self.cu_psi2pos_merge = psi2pos_merge_mod.get_function("Psi2PosMerge")

        # Convolution kernels for support update (Gaussian)
        conv16f_mod = SourceModule(getks('cuda/complex.cu') + getks('cuda/convolution16f.cu'), options=self.cu_options)
        self.gauss_convolf_16x = conv16f_mod.get_function("gauss_convolf_16x")
        self.gauss_convolf_16y = conv16f_mod.get_function("gauss_convolf_16y")
        self.gauss_convolf_16z = conv16f_mod.get_function("gauss_convolf_16z")
        conv16c_mod = SourceModule(getks('cuda/complex.cu') + getks('cuda/convolution16c.cu'), options=self.cu_options)
        self.gauss_convolc_16x = conv16c_mod.get_function("gauss_convolc_16x")
        self.gauss_convolc_16y = conv16c_mod.get_function("gauss_convolc_16y")
        self.gauss_convolc_16z = conv16c_mod.get_function("gauss_convolc_16z")

        # Registration kernels (max find in CC map & upsampled/zoomed version)
        reg_mod = SourceModule(getks("cuda/argmax.cu") + getks("cuda/complex.cu") + getks('cuda/bilinear.cu') +
                               getks('holotomo/cuda/registration.cu') % {"blocksize": 32},
                               options=self.cu_options)
        self.cu_reg_pixel = reg_mod.get_function("cc_pixel")
        self.cu_reg_zoom = reg_mod.get_function("cc_zoom")

        print("HoloTomo CUDA processing unit: compiling kernels... Done (dt=%5.2fs)" % (timeit.default_timer() - t0))

        # Init memory pool
        self.cu_mem_pool = cu_tools.DeviceMemoryPool()

    def get_modes_kernels(self, n):
        """
        Get the kernels which depend on the number of probe modes.
        This will create the required data types and the reduction kernels.
        :param n: the number of modes
        :return: a dictionary of kernels for the given number of modes, including:
            * "vdot": computes [[np.vdot(p2, p1) for p1 in d] for p2 in d]
              to prepare the orthogonalisation of a stack of arrays.
            * "dot": computes the orthogonal modes with the v[i,j] NxN matrix:
              np.array([sum(d[i] * v[i, j] for i in range(len(d))) for j in range(len(d))])
            * "dot_red": same as "dot" but as a reduction kernel which returns the
              L2 norm of the modes, for sorting.
            * "sort": sorts the modes with descending L2 norm
        """
        if n not in self._modes_kernels:
            self._modes_kernels[n] = {}
            n2 = n ** 2
            # Create the numpy array types using a subarray
            complex_n2 = np.dtype([('x', np.complex64, (n2,))])
            float_n = np.dtype([('x', np.float32, (n,))])
            # Create the pycuda array types. The string name must correspond
            # to what is declared in the kernel source, i.e. TYPE_N
            my_complex_n = cu_tools.get_or_register_dtype("complexf_%d" % n2, complex_n2)
            my_float_n = cu_tools.get_or_register_dtype("float_%d" % n, float_n)
            self._modes_kernels[n]["vdot"] = \
                CU_RedK(complex_n2, neutral="complexf_%d(0)" % n2, reduce_expr="a+b",
                        preamble=getks('cuda/vector_type.cu') % {'TYPE': 'complexf', 'N': n2} +
                                 getks('cuda/vdot_n.cu') % {'N2': n2, 'N': n},
                        map_expr="vdot(i,d,nxy)",
                        arguments="pycuda::complex<float> *d, const unsigned int nxy")

            self._modes_kernels[n]["dot"] = \
                CU_ElK(name='cu_ortho_dot', operation="ortho_dot(i, d, v, nxy)",
                       preamble=getks("cuda/complex.cu") +
                                getks('cuda/vector_type.cu') % {'TYPE': 'float', 'N': n} +
                                getks("cuda/ortho_n.cu") % {'N': n},
                       arguments="pycuda::complex<float> *d, "
                                 "pycuda::complex<float> *v, const unsigned int nxy")

            self._modes_kernels[n]["dot_red"] = \
                CU_RedK(float_n, neutral="float_%d(0)" % n, reduce_expr="a+b",
                        name='cu_dot_red', map_expr="ortho_dot_red(i, d, v, nxy)",
                        preamble=getks("cuda/complex.cu") +
                                 getks('cuda/vector_type.cu') % {'TYPE': 'float', 'N': n} +
                                 getks("cuda/ortho_n.cu") % {'N': n},
                        arguments="pycuda::complex<float> *d, "
                                  "pycuda::complex<float> *v, const unsigned int nxy")

            self._modes_kernels[n]["sort"] = \
                CU_ElK(name='cu_ortho_sort', operation="ortho_sort(i, d, dnorm, nxy)",
                       preamble=getks("cuda/complex.cu") +
                                getks('cuda/vector_type.cu') % {'TYPE': 'float', 'N': n} +
                                getks("cuda/ortho_n.cu") % {'N': n},
                       arguments="pycuda::complex<float> *d, float *dnorm, const unsigned int nxy")

            self._modes_kernels[n]["ortho_norm"] = \
                CU_ElK(name='cu_ortho_norm', operation="ortho_norm(i, d, dnorm, nxy)",
                       preamble=getks("cuda/complex.cu") +
                                getks('cuda/vector_type.cu') % {'TYPE': 'float', 'N': n} +
                                getks("cuda/ortho_n.cu") % {'N': n},
                       arguments="pycuda::complex<float> *d, float *dnorm, const unsigned int nxy")

        return self._modes_kernels[n]

    def get_modes_nz_kernels(self, n, nz):
        """
        Get the kernels which depend on the number of probe modes and distances.
        This will create the required data types and the reduction kernels.
        :param n: the number of modes
        :param nz: the number of distances
        :return: a dictionary of kernels for the given number of modes and distances:
            * "psi2obj_probe_modes_red": computes the updated object, probe modes and
                probe modes coefficients (reduction kernel).
        """
        if (n, nz) not in self._modes_nz_kernels:
            self._modes_nz_kernels[(n, nz)] = {}
            nzn2 = n * nz * 2
            # nzn = n * nz * 2
            float_nzn2 = np.dtype([('x', np.float32, (nzn2,))])
            # float_nzn = np.dtype([('x', np.float32, (nzn,))])
            my_float_nzn2 = cu_tools.get_or_register_dtype("float_%d" % nzn2, float_nzn2)
            # my_float_nzn = cu_tools.get_or_register_dtype("float_%d" % nzn, float_nzn)
            self._modes_nz_kernels[(n, nz)]["psi2obj_probe_modes_red"] = \
                CU_RedK(float_nzn2, neutral="float_%d(0)" % nzn2, reduce_expr="a+b",
                        name='cu_psi2obj_probe_modes_red',
                        map_expr="Psi2ObjProbeRedN(i, obj, obj_old, probe, psi, probe_new, probe_norm, "
                                 "probe_coeffs, obj_phase0, dx, dy, sample_flag, nx, ny,"
                                 "obj_min, obj_max, reg_obj_smooth, beta_delta, weight_empty)",
                        preamble=getks("cuda/complex.cu") + getks('cuda/bilinear.cu') +
                                 getks('cuda/vector_type.cu') % {'TYPE': 'float', 'N': nzn2} +
                                 getks("holotomo/cuda/psi2obj_probe_red_n.cu") % {'N': n, 'NZ': nz, 'NZN2': nzn2},
                        options=self.cu_options,
                        arguments="pycuda::complex<float>* obj, pycuda::complex<float>* obj_old,"
                                  "pycuda::complex<float> *probe, pycuda::complex<float>* psi,"
                                  "pycuda::complex<float>* probe_new, float* probe_norm, float* probe_coeffs,"
                        # "half* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                                  "float* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                                  "const int nb_z, const int nb_obj, const int nb_probe, const int nx,"
                                  "const int ny, const float obj_min, const float obj_max,"
                                  "const float reg_obj_smooth,"
                                  "const float beta_delta, const float weight_empty")

            self._modes_nz_kernels[(n, nz)]["psi2obj_probe_modes_raar_red"] = \
                CU_RedK(float_nzn2, neutral="float_%d(0)" % nzn2, reduce_expr="a+b",
                        name='cu_psi2obj_probe_modes_raar_red',
                        map_expr="Psi2ObjProbeRAARRedN(i, obj, obj_old, probe, psi, psiold, probe_new, probe_norm, "
                                 "probe_coeffs, obj_phase0, dx, dy, sample_flag, nx, ny,"
                                 "obj_min, obj_max, reg_obj_smooth, beta_delta, weight_empty, beta)",
                        preamble=getks("cuda/complex.cu") + getks('cuda/bilinear.cu') +
                                 getks('cuda/vector_type.cu') % {'TYPE': 'float', 'N': nzn2} +
                                 getks("holotomo/cuda/psi2obj_probe_raar_red_n.cu") % {'N': n, 'NZ': nz, 'NZN2': nzn2},
                        options=self.cu_options,
                        arguments="pycuda::complex<float>* obj, pycuda::complex<float>* obj_old,"
                                  "pycuda::complex<float> *probe, pycuda::complex<float>* psi,"
                                  "pycuda::complex<float>* psiold,"
                                  "pycuda::complex<float>* probe_new, float* probe_norm, float* probe_coeffs,"
                        # "half* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                                  "float* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                                  "const int nb_z, const int nb_obj, const int nb_probe, const int nx,"
                                  "const int ny, const float obj_min, const float obj_max,"
                                  "const float reg_obj_smooth,"
                                  "const float beta_delta, const float weight_empty, const float beta")

            self._modes_nz_kernels[(n, nz)]["psi2obj_probe_modes_drap_red"] = \
                CU_RedK(float_nzn2, neutral="float_%d(0)" % nzn2, reduce_expr="a+b",
                        name='cu_psi2obj_probe_modes_drap_red',
                        map_expr="Psi2ObjProbeDRAPRedN(i, obj, obj_old, probe, psi, psiold, probe_new, probe_norm, "
                                 "probe_coeffs, obj_phase0, dx, dy, sample_flag, nx, ny,"
                                 "obj_min, obj_max, reg_obj_smooth, beta_delta, weight_empty, beta)",
                        preamble=getks("cuda/complex.cu") + getks('cuda/bilinear.cu') +
                                 getks('cuda/vector_type.cu') % {'TYPE': 'float', 'N': nzn2} +
                                 getks("holotomo/cuda/psi2obj_probe_drap_red_n.cu") % {'N': n, 'NZ': nz, 'NZN2': nzn2},
                        options=self.cu_options,
                        arguments="pycuda::complex<float>* obj, pycuda::complex<float>* obj_old,"
                                  "pycuda::complex<float> *probe, pycuda::complex<float>* psi,"
                                  "pycuda::complex<float>* psiold,"
                                  "pycuda::complex<float>* probe_new, float* probe_norm, float* probe_coeffs,"
                        # "half* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                                  "float* obj_phase0, float* dx, float* dy, signed char* sample_flag,"
                                  "const int nb_z, const int nb_obj, const int nb_probe, const int nx,"
                                  "const int ny, const float obj_min, const float obj_max,"
                                  "const float reg_obj_smooth,"
                                  "const float beta_delta, const float weight_empty, const float beta")

            self._modes_nz_kernels[(n, nz)]["proj2probe_mode"] = \
                CU_RedK(float_nzn2, neutral="float_%d(0)" % nzn2, reduce_expr="a+b",
                        name='cu_proj2probe_mode',
                        map_expr="Proj2ProbeMode(i, probe, psi, probe_coeffs, nx, ny)",
                        preamble=getks("cuda/complex.cu") + getks('cuda/bilinear.cu') +
                                 getks('cuda/vector_type.cu') % {'TYPE': 'float', 'N': nzn2} +
                                 getks("holotomo/cuda/psi2obj_probe_red_n.cu") % {'N': n, 'NZ': nz, 'NZN2': nzn2},
                        options=self.cu_options,
                        arguments="pycuda::complex<float> *probe, pycuda::complex<float>* psi,"
                                  "float* probe_coeffs, const int nx, const int ny")

        return self._modes_nz_kernels[(n, nz)]

    def finish(self):
        super(CUProcessingUnitHoloTomo, self).finish()
        self.cu_stream_swap.synchronize()


"""
The default processing unit 
"""
default_processing_unit = CUProcessingUnitHoloTomo()


class CUOperatorHoloTomo(OperatorHoloTomo):
    """
    Base class for a operators on HoloTomo objects using OpenCL
    """

    def __init__(self, processing_unit=None):
        super(CUOperatorHoloTomo, self).__init__()

        self.Operator = CUOperatorHoloTomo
        self.OperatorSum = CUOperatorHoloTomoSum
        self.OperatorPower = CUOperatorHoloTomoPower

        if processing_unit is None:
            self.processing_unit = default_processing_unit
        else:
            self.processing_unit = processing_unit
        if self.processing_unit.cu_ctx is None:
            # OpenCL kernels have not been prepared yet, use a default initialization
            if main_default_processing_unit.cu_device is None:
                main_default_processing_unit.use_cuda()
            self.processing_unit.init_cuda(cu_device=main_default_processing_unit.cu_device,
                                           test_fft=False, verbose=False)

    def apply_ops_mul(self, p: HoloTomo):
        """
        Apply the series of operators stored in self.ops to an object.
        In this version the operators are applied one after the other to the same object (multiplication)

        :param p: the object to which the operators will be applied.
        :return: the object, after application of all the operators in sequence
        """
        return super(CUOperatorHoloTomo, self).apply_ops_mul(p)

    def prepare_data(self, p: HoloTomo):
        stack_size, nz, ny, nx = p.data.stack_size, p.data.nz, p.data.ny, p.data.nx
        nobj, nb_probe = p.nb_obj, p.nb_probe
        pu = self.processing_unit

        # Make sure data is already in CUDA space, otherwise transfer it
        if p._timestamp_counter > p._cu_timestamp_counter or p._cu_probe is None:
            print("Copying arrays from host to GPU")
            p._cu_timestamp_counter = p._timestamp_counter

            # This will reset the contents of stacks and make sure we get the new values from host
            p._cu_stack = HoloTomoDataStack()
            p._cu_stack_swap = HoloTomoDataStack()

            p._cu_probe = cua.to_gpu_async(p._probe, allocator=pu.cu_mem_pool.allocate, stream=pu.cu_stream)
            if p.probe_mode_coeff is not None:
                p._cu_probe_mode_coeff = cua.to_gpu_async(p.probe_mode_coeff, allocator=pu.cu_mem_pool.allocate,
                                                          stream=pu.cu_stream)
            p._cu_dx = cua.to_gpu_async(p.data.dx.astype(np.float32), allocator=pu.cu_mem_pool.allocate,
                                        stream=pu.cu_stream)
            p._cu_dy = cua.to_gpu_async(p.data.dy.astype(np.float32), allocator=pu.cu_mem_pool.allocate,
                                        stream=pu.cu_stream)
            p._cu_sample_flag = cua.to_gpu_async(p.data.sample_flag.astype(np.int8), allocator=pu.cu_mem_pool.allocate,
                                                 stream=pu.cu_stream)
            p._cu_scale_factor = cua.to_gpu_async(p.data.scale_factor.astype(np.float32),
                                                  allocator=pu.cu_mem_pool.allocate, stream=pu.cu_stream)
            # Calc quadratic phase factor for near field propagation, z-dependent
            quad_f = np.pi * p.data.wavelength * p.data.detector_distance / p.data.pixel_size_detector ** 2
            p._cu_quad_f = cua.to_gpu_async(quad_f.astype(np.float32), allocator=pu.cu_mem_pool.allocate,
                                            stream=pu.cu_stream)
            for s in (p._cu_stack, p._cu_stack_swap):
                s.psi = cua.empty(shape=p._psi.shape, dtype=np.complex64,
                                  allocator=pu.cu_mem_pool.allocate)
                s.obj = cua.empty(shape=(stack_size, nobj, ny, nx), dtype=np.complex64,
                                  allocator=pu.cu_mem_pool.allocate)
                s.obj_phase0 = cua.empty(shape=(stack_size, nobj, ny, nx), dtype=half.type,
                                         allocator=pu.cu_mem_pool.allocate)
                s.iobs = cua.empty(shape=(stack_size, nz, ny, nx), dtype=np.float32, allocator=pu.cu_mem_pool.allocate)
                s.istack = None
            # Copy data for the main (computing) stack
            if len(p.data.stack_v) > 1:
                p = SwapStack(i=0, next_i=1, out=False, copy_psi=False, verbose=False) * p
            else:
                p = SwapStack(i=0, next_i=None, out=False, copy_psi=False, verbose=False) * p

    def timestamp_increment(self, p: HoloTomo):
        p._cu_timestamp_counter += 1


# The only purpose of this class is to make sure it inherits from CUOperatorHoloTomo and has a processing unit
class CUOperatorHoloTomoSum(OperatorSum, CUOperatorHoloTomo):
    def __init__(self, op1, op2):
        # TODO: should this apply to a single stack or all ?
        if np.isscalar(op1):
            op1 = Scale1(op1)
        if np.isscalar(op2):
            op2 = Scale1(op2)
        if isinstance(op1, CUOperatorHoloTomo) is False or isinstance(op2, CUOperatorHoloTomo) is False:
            raise OperatorException(
                "ERROR: cannot add a CUOperatorHoloTomo with a non-CUOperatorHoloTomo: %s + %s" % (str(op1), str(op2)))
        # We can only have a sum of two CLOperatorHoloTomo, so they must have a processing_unit attribute.
        CUOperatorHoloTomo.__init__(self, op1.processing_unit)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CUOperatorHoloTomo
        self.OperatorSum = CUOperatorHoloTomoSum
        self.OperatorPower = CUOperatorHoloTomoPower
        self.prepare_data = types.MethodType(CUOperatorHoloTomo.prepare_data, self)
        self.timestamp_increment = types.MethodType(CUOperatorHoloTomo.timestamp_increment, self)


# The only purpose of this class is to make sure it inherits from CLOperatorCDI and has a processing unit
class CUOperatorHoloTomoPower(OperatorPower, CUOperatorHoloTomo):
    def __init__(self, op, n):
        CUOperatorHoloTomo.__init__(self, op.processing_unit)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CUOperatorHoloTomo
        self.OperatorSum = CUOperatorHoloTomoSum
        self.OperatorPower = CUOperatorHoloTomoPower
        self.prepare_data = types.MethodType(CUOperatorHoloTomo.prepare_data, self)
        self.timestamp_increment = types.MethodType(CUOperatorHoloTomo.timestamp_increment, self)


class FreePU(CUOperatorHoloTomo):
    """
    Operator freeing CUDA memory.
    """

    def __init__(self, verbose=False):
        """

        :param verbose: if True, will detail all the free'd memory and a summary
        """
        super(FreePU, self).__init__()
        self.verbose = verbose

    def op(self, p: HoloTomo):
        """

        :param p: the HoloTomo object this operator applies to
        :return: the updated HoloTomo object
        """
        self.processing_unit.finish()

        p._from_pu()
        if self.verbose:
            print("FreePU:")
        bytes = 0

        for o in dir(p):
            if isinstance(p.__getattribute__(o), cua.GPUArray):
                if self.verbose:
                    print("  Freeing: %40s %10.3fMbytes" % (o, p.__getattribute__(o).nbytes / 1e6))
                    bytes += p.__getattribute__(o).nbytes
                p.__getattribute__(o).gpudata.free()
                p.__setattr__(o, None)
        for v in (p._cu_stack, p._cu_stack_swap):
            if v is not None:
                for o in dir(v):
                    if isinstance(v.__getattribute__(o), cua.GPUArray):
                        if self.verbose:
                            print("  Freeing: %40s %10.3fMbytes" % ("_cu_stack:" + o,
                                                                    v.__getattribute__(o).nbytes / 1e6))
                            bytes += v.__getattribute__(o).nbytes
                        v.__getattribute__(o).gpudata.free()
                        v.__setattr__(o, None)

        self.processing_unit.cu_mem_pool.free_held()
        gc.collect()
        if self.verbose:
            print('FreePU total: %10.3fMbytes freed' % (bytes / 1e6))
        return p

    def prepare_data(self, p: HoloTomo):
        # Overriden to avoid transferring any data to GPU
        pass

    def timestamp_increment(self, p):
        p._cu_timestamp_counter = 0


class MemUsage(CUOperatorHoloTomo):
    """
    Print memory usage of current process (RSS on host) and used GPU memory
    """

    def op(self, p: HoloTomo):
        """

        :param p: the HoloTomo object this operator applies to
        :return: the updated HoloTomo object
        """
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        gpu_mem = 0

        for o in dir(p):
            if isinstance(p.__getattribute__(o), cua.GPUArray):
                gpu_mem += p.__getattribute__(o).nbytes
        for v in (p._cu_stack, p._cu_stack_swap):
            if v is not None:
                for o in dir(v):
                    if isinstance(v.__getattribute__(o), cua.GPUArray):
                        gpu_mem += v.__getattribute__(o).nbytes

        print("Mem Usage: RSS= %6.1f Mbytes (process), GPU Mem= %6.1f Mbyts (HoloTomo object)" %
              (rss / 1024 ** 2, gpu_mem / 1024 ** 2))
        return p

    def prepare_data(self, p: HoloTomo):
        # Overriden to avoid transferring any data to GPU
        pass

    def timestamp_increment(self, p):
        # This operator does nothing
        pass


class Scale1(CUOperatorHoloTomo):
    """
    Multiply the object and/or psi by a scalar (real or complex).
    If the scale is a vector which has the size of the number of projections,
    each projection (object and/or Psi) is scaled individually

    Applies only to the current stack.
    """

    def __init__(self, x, obj=True, psi=True):
        """

        :param x: the scaling factor (can be a vector with a factor for each individual projection)
        :param obj: if True, scale the object
        :param psi: if True, scale the psi array
        """
        super(Scale1, self).__init__()
        self.x = x
        self.obj = obj
        self.psi = psi

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        if np.isscalar(self.x):
            if self.x == 1:
                return p

            if np.isreal(self.x):
                scale_k = pu.cu_scale
                x = np.float32(self.x)
            else:
                scale_k = pu.cu_scale_complex
                x = np.complex64(self.x)
            if self.obj:
                scale_k(p._cu_stack.obj, x, stream=pu.cu_stream)
            if self.psi:
                scale_k(p._cu_stack.psi, x, stream=pu.cu_stream)
        else:
            # Don't assume iproj is copied in the cu stack (probably should..)
            s = p._cu_stack
            i0 = s.iproj
            for i in range(s.nb):
                if p.data.sample_flag[i0 + i]:
                    if np.isreal(self.x[i0 + i]):
                        scale_k = pu.cu_scale
                        x = np.float32(self.x[i0 + i])
                    else:
                        scale_k = pu.cu_scale_complex
                        x = np.complex64(self.x[i0 + i])
                    if self.obj:
                        scale_k(s.obj[i], x, stream=pu.cu_stream)
                    if self.psi:
                        scale_k(s.psi[i], x, stream=pu.cu_stream)

        return p


class Scale(CUOperatorHoloTomo):
    """
    Multiply the object or probe or psi by a scalar (real or complex).
    If the scale is a vector which has the size of the number of projections,
    each projection (object and/or Psi) is scaled individually

    Will apply to all projection stacks of the HoloTomo object
    """

    def __init__(self, scale, obj=True, probe=True, psi=True):
        """

        :param scale: the scaling factor
        :param obj: if True, scale the object
        :param probe: if True, scale the probe
        :param psi: if True, scale the psi array
        """
        super(Scale, self).__init__()
        self.scale = scale
        self.obj = obj
        self.probe = probe
        self.psi = psi

    def op(self, p: HoloTomo):
        pu = self.processing_unit

        if self.probe:
            if np.isreal(self.scale):
                scale_k = pu.cu_scale
                scale = np.float32(self.scale)
            else:
                scale_k = pu.cu_scale_complex
                scale = np.complex64(self.scale)
            scale_k(p._cu_probe, scale, stream=pu.cu_stream)

        if self.obj or self.psi:
            p = LoopStack(Scale1(self.scale, obj=self.obj, psi=self.psi), copy_psi=self.psi) * p

        return p


class ScaleObjProbe1(CUOperatorHoloTomo):
    """
    Compute sum of Iobs and Icalc for 1 stack.
    """

    def __init__(self, vpsi, vobs):
        """

        :param vpsi: the array in which the psi norm sum will be stored. Should have p.data.nproj elements
        :param vobs: the array in which the observed intensity sum will be stored. Should have p.data.nproj elements
        """
        super(ScaleObjProbe1, self).__init__()
        self.vpsi = vpsi
        self.vobs = vobs

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        if p.probe_mode_coeff is None:
            nb_mode = np.int32(p.nb_probe * p.nb_obj)
        else:
            nb_mode = np.int32(p.nb_obj)
        nx = np.int32(p.data.nx)
        ny = np.int32(p.data.ny)
        nb = s.nb

        for i in range(nb):
            r = pu.cu_scale_obs_calc(s.iobs[i], s.psi[i], nx, ny, nb_mode, stream=pu.cu_stream).get()
            self.vpsi[s.iproj + i] = r.imag
            self.vobs[s.iproj + i] = r.real

        return p


class ScaleObjProbe(CUOperatorHoloTomo):
    """
    Scale object and probe to match observed intensities. The probe amplitude is scaled to match the
    average intensity in the empty beam frames, and each object projection is set to match the average
    intensity for that projection.
    """

    def __init__(self, verbose=True):
        """

        :param verbose: if True, guess what ?
        """
        super(ScaleObjProbe, self).__init__()
        self.verbose = verbose

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        # Vector of the sum of square modulus of Probe*Obj for each projection, summed over the nz distances
        vpsi = np.empty(p.data.nproj, dtype=np.float32)
        # Sum of intensities for each projection, summed over the nz distances
        vobs = np.empty(p.data.nproj, dtype=np.float32)

        # Compute all the sums. No propagation needed (Parseval)
        p = LoopStack(ScaleObjProbe1(vpsi=vpsi, vobs=vobs) * ObjProbe2Psi1()) * p
        # Find empty beam and scale probe
        nb_empty = 0
        obs_empty = 0
        psi_empty = 0
        for i in range(p.data.nproj):
            if bool(p.data.sample_flag[i]) is False:
                nb_empty += 1
                obs_empty += vobs[i]
                psi_empty += vpsi[i]
        if nb_empty == 0:
            print("No empty beam images !? Scaling probe according to average intensity (BAD!)")
            scale_probe = np.sqrt(vobs.sum() / p.data.nb_obs) \
                          / np.sqrt(pu.cu_norm_n_c(p._cu_probe, np.float32(2)).get())
        else:
            scale_probe = np.sqrt(obs_empty / psi_empty)
        if self.verbose:
            print("Scaling probe by %6.2f" % scale_probe)
        p = Scale(scale_probe, psi=False, obj=False, probe=True) * p

        # Now scale object for each projection individually
        # TODO: should we instead scale according to average ?
        scale_obj = np.sqrt(vobs / vpsi) / scale_probe
        # No real need to scale psi, but allows a consistency check
        p = Scale(scale_obj, obj=True, psi=True, probe=False) * p
        return p


class FT1(CUOperatorHoloTomo):
    """
    Forward Fourier transform.

    Applies only to the current stack.
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(FT1, self).__init__()
        self.scale = scale

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        pu.fft(s.psi, s.psi, ndim=2, norm=self.scale, stream=pu.cu_stream)
        return p


class IFT1(CUOperatorHoloTomo):
    """
    Inverse Fourier transform.

    Applies only to the current stack.
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(IFT1, self).__init__()
        self.scale = scale

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        pu.ifft(s.psi, s.psi, ndim=2, norm=self.scale, stream=pu.cu_stream)
        return p


class QuadraticPhase1(CUOperatorHoloTomo):
    """
    Operator applying a quadratic phase factor for near field propagation. The factor is different for each distance,
    based on the propagation distance stored in the HoloTomo object.

    Applies only to the current stack.
    """

    def __init__(self, forward=True, scale=1):
        """
        Application of a quadratic phase factor, and optionally a scale factor.

        The actual factor is:  :math:`scale * e^{i * factor * ((ix/nx)^2 + (iy/ny)^2)}`
        where ix and iy are the integer indices of the pixels.
        The factor is stored in the HoloTomo object.

        :param forward: if True (the default), applies the scale factor for forward propagation
        :param scale: the data will be scaled by this factor. Useful to normalize before/after a Fourier transform,
                      without accessing twice the array data.
        """
        super(QuadraticPhase1, self).__init__()
        self.scale = np.float32(scale)
        self.forward = forward

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        if p.probe_mode_coeff is None:
            nb_mode = np.int32(p.nb_obj * p.nb_probe)
        else:
            nb_mode = np.int32(p.nb_probe)
        nz = np.int32(p.data.nz)
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)
        pu.cu_quad_phase(p._cu_stack.psi, p._cu_quad_f, self.forward, self.scale,
                         nz, nb_mode, nx, ny, stream=pu.cu_stream)

        return p


class PropagateNearField1(CUOperatorHoloTomo):
    """
    Near field propagator.

    Applies only to the current stack.
    """

    def __init__(self, forward=True):
        """

        :param forward: if True (the default), perform a forward propagation based on the experimental distances.
        """
        super(PropagateNearField1, self).__init__()
        self.forward = forward

    def op(self, p: HoloTomo):
        s = self.processing_unit.fft_scale(p._psi.shape, ndim=2)
        return IFT1(scale=False) * QuadraticPhase1(forward=self.forward, scale=s[0] * s[1]) * FT1(scale=False) * p


class ObjProbe2Psi1(CUOperatorHoloTomo):
    """
    Operator multiplying object views and probe to produce the initial Psi array (before propagation)
    for all projections and distances in the stack.

    Applies only to the current stack.
    """

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = p.data.stack_v[s.istack].iproj
        nb = np.int32(p.data.stack_v[s.istack].nb)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        if p.probe_mode_coeff is None:
            pu.cu_obj_probez_mult(s.obj[0, 0], p._cu_probe, s.psi, p._cu_dx[i0:i0 + nb], p._cu_dy[i0:i0 + nb],
                                  p._cu_sample_flag[i0:i0 + nb], nb, nz, nb_obj, nb_probe, nx, ny,
                                  stream=pu.cu_stream)
        else:
            pu.cu_obj_probecohz_mult(s.obj[0, 0], p._cu_probe, s.psi, p._cu_dx[i0:i0 + nb], p._cu_dy[i0:i0 + nb],
                                     p._cu_sample_flag[i0:i0 + nb], p._cu_probe_mode_coeff[i0:i0 + nb],
                                     nb, nz, nb_probe, nx, ny, stream=pu.cu_stream)
        return p


class LLK1(CUOperatorHoloTomo):
    """
    Log-likelihood reduction kernel. Should only be used when Psi is propagated to detector space.
    This is a reduction operator - it will write llk as an argument in the HoloTomo object, and return the object.
    This operator only applies to the current stack of projections.
    If the stack number==0, the llk is re-initialized. Otherwise it is added to the current value.
    """

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        if p.probe_mode_coeff is None:
            nb_mode = np.int32(p.nb_probe * p.nb_obj)
        else:
            nb_mode = np.int32(p.nb_obj)
        nx = np.int32(p.data.nx)
        ny = np.int32(p.data.ny)
        nb = np.int32(s.nb)
        # TODO: instead of get() the result, store it on-GPU and get only the sum when all stacks are processed
        llk = pu.cu_llk(s.iobs[:nb], s.psi, nb_mode, nx, ny, stream=pu.cu_stream).get()
        if s.istack == 0:
            p.llk_poisson = llk['a']
            p.llk_gaussian = llk['b']
            p.llk_euclidian = llk['c']
            p.nb_photons_calc = llk['d']
        else:
            p.llk_poisson += llk['a']
            p.llk_gaussian += llk['b']
            p.llk_euclidian += llk['c']
            p.nb_photons_calc += llk['d']
        return p


class LLK(CUOperatorHoloTomo):
    """
    Compute the log-likelihood for the entire set of projections.
    Using this operator will loop through all the stacks and frames, project Obj*Probe and compute
    the llk for each. The history will be updated.
    This operator should not be while running a main algorithm, which can compute the LLK during their cycles.
    """

    def __init__(self, verbose=True):
        """
        Compute the log-likelihood
        :param verbose: if True, print the log-likelihood
        """
        super(LLK, self).__init__()

        self.verbose = verbose

    def op(self, p: HoloTomo):
        p = LoopStack(op=LLK1() * PropagateNearField1() * ObjProbe2Psi1(), out=False, copy_psi=False, verbose=False) * p
        p.update_history(mode='llk', update_obj=False, update_probe=False,
                         dt=0, algorithm='LLK', verbose=self.verbose)

        return p


class ApplyAmplitude1(CUOperatorHoloTomo):
    """
    Apply the magnitude from observed intensities, keep the phase. Masked pixels (marked using <0 intensities) are
    left unchanged.

    Applies only to the current stack.
    """

    def __init__(self, calc_llk=False):
        """

        :param calc_llk: if True, the log-likelihood will be calculated for this stack.
        """
        super(ApplyAmplitude1, self).__init__()
        self.calc_llk = calc_llk

    def op(self, p: HoloTomo):
        if self.calc_llk:
            # TODO: use a single-pass reduction kernel to apply the amplitude and compute the LLK
            p = LLK1() * p
        pu = self.processing_unit
        s = p._cu_stack
        if p.probe_mode_coeff is None:
            nb_mode = np.int32(p.nb_obj * p.nb_probe)
        else:
            nb_mode = np.int32(p.nb_obj)
        nx = np.int32(p.data.nx)
        ny = np.int32(p.data.ny)
        nb = np.int32(p.data.stack_v[s.istack].nb)
        pu.cu_projection_amplitude(s.iobs[:nb], s.psi, nb_mode, nx, ny, stream=pu.cu_stream)
        return p


class PropagateApplyAmplitude1(CUOperatorHoloTomo):
    """
    Propagate to detector space and apply the amplitude constraint.
    """

    def __new__(cls, calc_llk=False):
        return PropagateNearField1(forward=False) * ApplyAmplitude1(calc_llk=calc_llk) * \
               PropagateNearField1(forward=True)


class Psi2ObjProbe1(CUOperatorHoloTomo):
    """
    Operator projecting the psi arrays in sample space onto the object and probe update.
    The object can be constrained to a min and max amplitude.

    Applies only to the current stack. The probe and normalisation are stored in temporary arrays
    """

    def __init__(self, update_object=True, update_probe=True, obj_max=None, obj_min=None, reg_obj_smooth=0,
                 delta_beta=-1, weight_empty=1.0):
        """

        :param update_object: if True, update the object
        :param update_probe: if True, update the probe
        :param obj_max: the maximum amplitude for the object
        :param obj_min: the minimum amplitude for the object
        :param reg_obj_smooth: the coefficient (typically 0-1) to smooth the object update
        :param delta_beta: delta/beta ratio. Ignored if < 0
        :param weight_empty: the relative weight of empty beam images for the probe update
        """
        super(Psi2ObjProbe1, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.obj_max = obj_max
        self.obj_min = obj_min
        self.reg_obj_smooth = np.float32(reg_obj_smooth)
        if delta_beta == 0:
            self.beta_delta = np.float32(-1)
        else:
            self.beta_delta = np.float32(1 / delta_beta)
        self.weight_empty = np.float32(weight_empty)

    def op(self, p: HoloTomo):
        assert p.probe_mode_coeff is None
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = s.iproj
        nb = np.int32(s.nb)

        if self.obj_max is None:
            obj_max = np.float32(-1)
        else:
            obj_max = np.float32(self.obj_max)
        if self.obj_min is None:
            obj_min = np.float32(-1)
        else:
            obj_min = np.float32(self.obj_min)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        if s.istack == 0:
            # TODO: do not create temporary arrays here but in parent operator (AP, DM, ML...)
            p._cu_probe_new = cua.zeros(shape=p._cu_probe.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
            p._cu_probe_norm = cua.zeros(shape=(nz, ny, nx), dtype=np.float32, allocator=pu.cu_mem_pool.allocate)
            #
            p._cu_scale_new = cua.zeros(shape=p.data.nproj, dtype=np.float32, allocator=pu.cu_mem_pool.allocate)

        if self.update_object:
            # keep copy of previous object for regularisation
            obj_old = s.obj
            if self.reg_obj_smooth > 0:
                obj_old = cua.empty(shape=s.obj.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
                cu_drv.memcpy_dtod_async(src=s.obj.gpudata, dest=obj_old.gpudata, size=s.obj.nbytes,
                                         stream=pu.cu_stream)
            # TODO: as for ptycho, it would be much more efficient to avoid this python loop
            for ii in range(nb):
                r = pu.cu_psi2obj_probe(s.obj[ii, 0], obj_old[ii, 0], p._cu_probe, s.psi[ii], p._cu_probe_new,
                                        p._cu_probe_norm, s.obj_phase0[ii, 0], p._cu_dx[i0 + ii], p._cu_dy[i0 + ii],
                                        p._cu_sample_flag[i0 + ii], nz, nb_obj, nb_probe, nx, ny, obj_min, obj_max,
                                        self.reg_obj_smooth, self.beta_delta,
                                        self.weight_empty, stream=pu.cu_stream)
                # TODO: store the result directly in p._cu_scale_new[i0 + ii], like in pyOpenCL
                cu_drv.memcpy_dtod_async(src=int(r.gpudata), dest=int(p._cu_scale_new[i0 + ii].gpudata),
                                         size=r.nbytes, stream=pu.cu_stream)
        else:
            for ii in range(nb):
                pu.cu_psi2probe(s.obj[ii, 0], p._cu_probe, s.psi[ii], p._cu_probe_new, p._cu_probe_norm,
                                p._cu_dx[i0 + ii], p._cu_dy[i0 + ii], p._cu_sample_flag[i0 + ii], nz,
                                nb_obj, nb_probe, nx, ny, self.weight_empty, stream=pu.cu_stream)

        # TODO:
        # - Take into account object inertia
        # - Prepare scale factor update, by comparing each image integrated probe intensity to the average

        return p


class Psi2ObjProbeCoherent1(CUOperatorHoloTomo):
    """
    Operator projecting the psi arrays in sample space onto the object and probe update.
    The object can be constrained to a min and max amplitude.
    This operator works with coherent probe modes, each projection having a
    different linear combination of the modes.

    Applies only to the current stack.
    """

    def __init__(self, update_object=True, update_probe=True, obj_max=None, obj_min=None, reg_obj_smooth=0,
                 delta_beta=-1, weight_empty=1.0):
        """

        :param update_object: if True, update the object
        :param update_probe: if True, update the probe
        :param obj_max: the maximum amplitude for the object
        :param obj_min: the minimum amplitude for the object
        :param reg_obj_smooth: the coefficient (typically 0-1) to smooth the object update
        :param delta_beta: delta/beta ratio. Ignored if < 0
        :param weight_empty: the relative weight of empty beam images for the probe update
        """
        super(Psi2ObjProbeCoherent1, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.obj_max = obj_max
        self.obj_min = obj_min
        self.reg_obj_smooth = np.float32(reg_obj_smooth)
        self.beta_delta = np.float32(1 / delta_beta)
        self.weight_empty = np.float32(weight_empty)

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = s.iproj
        nb = np.int32(s.nb)

        if self.obj_max is None:
            obj_max = np.float32(-1)
        else:
            obj_max = np.float32(self.obj_max)
        if self.obj_min is None:
            obj_min = np.float32(-1)
        else:
            obj_min = np.float32(self.obj_min)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        if s.istack == 0:
            # TODO: do not create temporary arrays here but in parent operator (AP, DM, ML...)
            p._cu_probe_new = cua.zeros_like(p._cu_probe)
            # Normalisation factor for p._cu_probe_new
            p._cu_probe_norm = cua.zeros(shape=p._cu_probe.shape, dtype=np.float32, allocator=pu.cu_mem_pool.allocate)
            p._cu_probe_mode_coeff_new = cua.empty_like(p._cu_probe_mode_coeff)
        if True:  # self.update_object:
            # keep copy of previous object for regularisation
            obj_old = s.obj
            if self.reg_obj_smooth > 0:
                obj_old = cua.empty(shape=s.obj.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
                cu_drv.memcpy_dtod_async(src=s.obj.gpudata, dest=obj_old.gpudata, size=s.obj.nbytes,
                                         stream=pu.cu_stream)
            psi2obj_probe_modes_red = pu.get_modes_nz_kernels(nb_probe, nz)["psi2obj_probe_modes_red"]
            for ii in range(nb):
                r = psi2obj_probe_modes_red(s.obj[ii, 0], obj_old[ii, 0], p._cu_probe, s.psi[ii], p._cu_probe_new,
                                            p._cu_probe_norm, p._cu_probe_mode_coeff[i0 + ii], s.obj_phase0[ii, 0],
                                            p._cu_dx[i0 + ii], p._cu_dy[i0 + ii],
                                            p._cu_sample_flag[i0 + ii], nz, nb_obj, nb_probe, nx, ny, obj_min, obj_max,
                                            self.reg_obj_smooth, self.beta_delta,
                                            self.weight_empty, stream=pu.cu_stream)
                # print(r, r.shape)
                r = cua.GPUArray((2, nz, nb_probe), dtype=np.float32, gpudata=r.gpudata, base=r)
                # print(r, r.shape)
                p._cu_probe_mode_coeff_new[i0 + ii] = r[0] / r[1]
                # if i0 == 0:
                #    print(i0 + ii, i0, ii, r[0] / r[1], p._cu_probe_mode_coeff_new[i0 + ii])
        else:
            # TODO
            pass

        # TODO:
        # - Take into account object & probe inertia
        # update object or probe only

        return p


class Psi2ProbeMerge(CUOperatorHoloTomo):
    """
    Final update of the probe from the temporary array and the normalisation.
    """

    def __init__(self, inertia=0.01, mode_inertia=1):
        """

        :param inertia: the inertia for the probe modes update.
        :param mode_inertia: the inertia for the probe modes. 0 Will replace the probe modes
            by the new values, while 1 will keep the old values.
        """
        super(Psi2ProbeMerge, self).__init__()
        self.inertia = np.float32(inertia)
        self.mode_inertia = np.float32(mode_inertia)

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        nz = np.int32(p.data.nz)
        nb_probe = np.int32(p.nb_probe)
        nxy = np.int32(p.data.ny * p.data.nx)
        if p.probe_mode_coeff is None:
            pu.cu_psi2probemerge(p._cu_probe[0, 0], p._cu_probe_new, p._cu_probe_norm, self.inertia,
                                 nb_probe, nxy, nz, stream=pu.cu_stream)
        else:
            pu.cu_psi2probemerge_coh(p._cu_probe[0, 0], p._cu_probe_new, p._cu_probe_norm, self.inertia,
                                     nb_probe, nxy, nz, stream=pu.cu_stream)
            if self.mode_inertia < 1:
                pu.cu_probe_mode_update(p._cu_probe_mode_coeff, p._cu_probe_mode_coeff_new, self.mode_inertia)
                # TODO: Normalise probe and update coefficients ?
                #     p = ProbeNorm() * p

        # del p._cu_probe_new, p._cu_probe_norm, p._cu_scale_new
        return p


class ProbeNorm(CUOperatorHoloTomo):
    """ Operator to normalise the probe, used with coherent probe modes. Either:

    * make the average value of each mode equal to 1, and scale the
      probe mode coefficients accordingly
    * keep the ratio of probe mode coefficients constant, so any projection
      will keep a percentage of each mode contribution constant, while
      allowing for a scale factor (source intensity decrease,..)
    """

    def __init__(self, option='ratio'):
        """

        :param option: either:

            * 'probe': normalise the probe modes to 1 and scale the coefficients
              accordingly
            * 'ratio': keep the coefficients ratio constant for each projection and z.
        """
        super().__init__()
        self.option = option

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        nz = np.int32(p.data.nz)
        nproj = p.data.nproj
        nb_probe = np.int32(p.nb_probe)
        nxy = np.int32(p.data.ny * p.data.nx)
        if self.option == 'probe:':
            for iz in range(nz):
                for i in range(nb_probe):
                    n = pu.cu_norm_n_c(p._cu_probe[iz, i], np.float32(2))
                    pu.cu_probe_norm(p._cu_probe[iz, i], n, nxy)
                    if p.probe_mode_coeff is not None:
                        # shape of coeffs is (nproj, nz, nb_probe)
                        # We reshape to loop over one z and one mode coefficients
                        v = p._cu_probe_mode_coeff.reshape(nproj * nz * nb_probe)
                        stride = np.int32(nz * nb_probe)
                        pu.cu_coeff_norm(v[i + nb_probe * iz:i + nb_probe * iz + nproj], n, nxy, stride)
                        # print(iz, i, p._cu_probe_mode_coeff[0], np.sqrt(n.get() / nxy))
        elif self.option == 'ratio':
            # TODO
            pass
        return p


class Psi2PosShift1(CUOperatorHoloTomo):
    """
    Update projection shifts, by comparing the updated Psi array to object*probe, for a stack of frames.
    This can only be used if there is more than one distance, as the first one is used as a reference
    """

    def __init__(self, multiplier=1, max_shift=2, save_position_history=False):
        """

        :param multiplier: the computed displacements are multiplied by this value,
            for faster convergence
        :param max_displ: the displacements (at each iteration) are capped to this
            value (in pixels), after applying the multiplier.
        :param save_position_history: if True, save the position history
            in the HoloTomo object (slow, for debugging)
        """
        super(Psi2PosShift1, self).__init__()
        self.mult = np.float32(multiplier)
        self.max_shift = np.float32(max_shift)
        self.save_position_history = save_position_history

    def op(self, p: HoloTomo):
        """

        :param p: the HoloTomo object this operator applies to
        :return: the updated HoloTomo object
        """
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        if nz == 1:
            return p
        i0 = s.iproj
        nb = np.int32(s.nb)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        if self.save_position_history and (has_attr_not_none(p, 'position_history') is False):
            p.position_history = [[(p.cycle, p._cu_dx[i].get(),
                                    p._cu_dy[i].get())] for i in range(p.data.nproj)]

        for ii in range(nb):
            # TODO: use multiple streams to treat different projections & distances in //
            if p.data.sample_flag[ii]:
                for iz in range(1, nz):
                    r = pu.cu_psi2pos_red(s.psi[ii, iz, 0, 0], s.obj[ii, 0], p._cu_probe[0],
                                          p._cu_dx[i0 + ii, iz], p._cu_dy[i0 + ii, iz],
                                          nx, ny, False, stream=pu.cu_stream)
                    pu.cu_psi2pos_merge(r, p._cu_dx[i0 + ii, iz], p._cu_dy[i0 + ii, iz],
                                        self.max_shift, self.mult, stream=pu.cu_stream, block=(1, 1, 1))
                if self.save_position_history:
                    p.position_history[i0 + ii].append((p.cycle, p._cu_dx[i0 + ii].get(),
                                                        p._cu_dy[i0 + ii].get()))
        return p


class Psi2PosReg1(CUOperatorHoloTomo):
    """
    Update projection shifts, by comparing the updated Psi array by registration
    between the different distances, for a stack of frames.
    This can only be used if there is more than one distance, as the first one is used as a reference.
    This operator assumes that the current Psi array contains the complex images after
    amplitude projection.
    """

    def __init__(self, upsampling=10, save_position_history=False):
        """
        :param upsampling: if >1, use a 1/upsampling pixel resolution.
            Value must be even. Default=10 - the fit may be progressive with the update
            of the projection, so sub-pixel accuracy is recommended.
        :param save_position_history: if True, save the position history
            in the HoloTomo object (slower, for development)
        """
        super(Psi2PosReg1, self).__init__()
        self.upsampling = upsampling
        self.save_position_history = save_position_history

    def op(self, p: HoloTomo):
        """

        :param p: the HoloTomo object this operator applies to
        :return: the updated HoloTomo object
        """
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        if nz == 1:
            return p
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = s.iproj
        nb_proj = np.int32(s.nb)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        if self.save_position_history and (has_attr_not_none(p, 'position_history') is False):
            p.position_history = [[(p.cycle, p._cu_dx[i].get(),
                                    p._cu_dy[i].get())] for i in range(p.data.nproj)]

        # Don't use the whole image for registration
        n = int(np.floor(np.log2(min(nx - 100, ny - 100))))
        if n > 10:
            n = np.int32(1024)
        else:
            n = np.int32(2 ** n)
        # reference plane
        izreg = np.int32(0)
        # 1) compute the object views for each distance (Psi / probe). Optionally phase only ?
        #    in a new array centered on the object
        cu_reg = cua.empty(shape=(nb_proj * nz, n, n), dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
        pu.cu_psi2reg(cu_reg, s.psi, p._cu_probe[0], p._cu_dx[i0:i0 + s.nb], p._cu_dy[i0:i0 + s.nb],
                      p._cu_sample_flag[i0:],
                      nb_probe, nb_obj, nx, ny, nz, n, stream=pu.cu_stream)

        # 2) FT
        pu.fft(cu_reg, cu_reg, ndim=2, stream=pu.cu_stream)

        # 3) Compute FT(reference_image) * FT(moved_image).conj() + IFT
        pu.cu_reg_mult_conj(cu_reg[0], p._cu_sample_flag[i0:], izreg, nz, nb_proj, n, stream=pu.cu_stream)
        pu.ifft(cu_reg[:nb_proj * (nz - 1)], cu_reg[:nb_proj * (nz - 1)], ndim=2, stream=pu.cu_stream)

        # 4) Pixel registration using a custom maximum reduction kernel in parallel over
        #    all the nproj*(nz-1) images with a blocksize of 16 and a grid size of nproj*(nz-1)
        cu_dx_new = p._cu_dx[i0:i0 + s.nb].copy()
        cu_dy_new = p._cu_dy[i0:i0 + s.nb].copy()
        pu.cu_reg_pixel(cu_reg[:nb_proj * (nz - 1)], cu_dx_new, cu_dy_new, izreg, nz, nb_proj, n,
                        stream=pu.cu_stream, block=(32, 1, 1), grid=(int(nz - 1), int(nb_proj), 1))

        if self.upsampling > 1:
            dnu = np.int32(self.upsampling + (self.upsampling % 2))
            cu_ccmap = cua.empty(shape=(nb_proj * nz, dnu, dnu), dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
            # 5) FT, sub-pixel registration & shifts update
            #    again a custom reduction kernel, storing
            pu.fft(cu_reg[:nb_proj * (nz - 1)], cu_reg[:nb_proj * (nz - 1)], ndim=2, stream=pu.cu_stream)
            pu.cu_reg_zoom(cu_reg[:nb_proj * (nz - 1)], p._cu_dx[i0:], p._cu_dy[i0:],
                           cu_dx_new, cu_dy_new, izreg, nz, nb_proj, n,
                           np.float32(1.5 / dnu), dnu, cu_ccmap, stream=pu.cu_stream, block=(32, 1, 1),
                           grid=(int(nz - 1), int(nb_proj), 1))
            p.ccmap = cu_ccmap.get()  # Debugging
        else:
            p._cu_dx[i0:i0 + s.nb] += cu_dx_new
            p._cu_dy[i0:i0 + s.nb] += cu_dy_new

        p.ccmap_dx = cu_dx_new.get()
        p.ccmap_dy = cu_dy_new.get()

        if self.save_position_history:
            for ii in range(nb_proj):
                if p.data.sample_flag[ii]:
                    p.position_history[i0 + ii].append((p.cycle, p._cu_dx[i0 + ii].get(),
                                                        p._cu_dy[i0 + ii].get()))
        return p


class PhaseFilter1(CUOperatorHoloTomo):
    """ Operator to a high-pass 1D filter on the phase,
    like a sinogram filter. This will only apply to a single
    stack of projections.
    """

    def op(self, p: HoloTomo):
        """

        :param p: the HoloTomo object this filter applies to.
        :return: the updated HoloTomo object
        """
        pu = self.processing_unit
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)
        s = p._cu_stack
        shape = (p.data.stack_size, 1, ny, nx)
        shape2 = (p.data.stack_size, 1, ny, nx // 2 + 1)

        # NOTE: would it make sense to perform the filtering on the normalised complex
        # object, to avoid phase wrapping issues ? Probably not, because if we filter
        # the phase, we need to apply the result also to the unwrapped phase.
        # This probably implies the phase needs to be small ? And that this is only compatible
        # with CTF, not Paganin. If the phase before applying this filter is wrapped, it
        # likely won't be anymore after...
        # Bottom line: apply the filter to the object_phase0 array, and then use the same
        # phase for the object.

        if s.istack == 0:
            need_cu_obj_phase_f = False
            if has_attr_not_none(p, "_cu_obj_phase_f"):
                if p._cu_obj_phase_f.shape != shape2:
                    need_cu_obj_phase_f = True
            else:
                need_cu_obj_phase_f = True
            if need_cu_obj_phase_f:
                p._cu_obj_phase_f = cua.empty(shape2, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
        pu.cu_obj2phase0(s.obj, s.obj_phase0, stream=pu.cu_stream)
        pu.fft(s.obj_phase0, p._cu_obj_phase_f, ndim=1, stream=pu.cu_stream)
        scale = np.float32(0.5 / (nx * nx))
        pu.cu_phase_1dfilt_highpass(p._cu_obj_phase_f, scale, np.int32(nx // 2 + 1), stream=pu.cu_stream)
        pu.ifft(p._cu_obj_phase_f, s.obj_phase0, ndim=1, stream=pu.cu_stream)
        pu.cu_phase2obj(s.obj, s.obj_phase0, stream=pu.cu_stream)

        return p


class AP(CUOperatorHoloTomo):
    """
    Perform alternating projections between detector and object/probe space.

    This operator applies to all projections and loops over the stacks.
    """

    def __init__(self, update_object=True, update_probe=True, nb_cycle=1, calc_llk=False,
                 show_obj_probe=0, fig_num=None, obj_min=None, obj_max=None, reg_obj_smooth=0,
                 delta_beta=-1, weight_empty=1.0, update_pos=0, pos_max_shift=2, pos_mult=1,
                 pos_history=False, pos_upsampling=10, probe_inertia=0.01):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param nb_cycle: number of cycles to perform. Equivalent to AP(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
            calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
            By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        :param obj_min, obj_max: min and max amplitude for the object. Can be None
        :param reg_obj_smooth: the coefficient (typically 0-1) to smooth the object update
        :param delta_beta: delta/beta ratio (typically 1e2 to 1e3) - a <=0 value disables the constraint.
            if delta_beta>1e6 , a pure phase object (amplitude=1) is optimised.
        :param weight_empty: relative weight given to empty beam images for the probe update
        :param update_pos: positive integer, if >0, update positions every 'update_pos' cycle.
            (default=False or 0, positions are not updated).
        :param pos_max_shift: maximum allowed shift (in pixels) per scan position (default=2)
        :param pos_mult: multiply the calculated position shifts by this value. Useful since the calculated
            shifts usually are a fraction of the actual shift.
        :param pos_history: if True, save the position history (for debugging, slow)
        :param pos_upsampling=10: upsampling for registration-based position update
        :param probe_inertia=0.01: the inertia for the probe update. Should be >0 at least when
            initialising the modes for stability.
        """
        super(AP, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.nb_cycle = nb_cycle
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.obj_min = obj_min
        self.obj_max = obj_max
        self.reg_obj_smooth = reg_obj_smooth
        self.delta_beta = np.float32(delta_beta)
        self.weight_empty = weight_empty
        self.update_pos = int(update_pos)
        self.pos_max_shift = pos_max_shift
        self.pos_mult = pos_mult
        self.pos_history = pos_history
        self.pos_upsampling = pos_upsampling
        self.probe_inertia = probe_inertia

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new AP operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return AP(update_object=self.update_object, update_probe=self.update_probe, nb_cycle=self.nb_cycle * n,
                  calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe, fig_num=self.fig_num,
                  obj_min=self.obj_min, obj_max=self.obj_max, reg_obj_smooth=self.reg_obj_smooth,
                  delta_beta=self.delta_beta, update_pos=self.update_pos, pos_max_shift=self.pos_max_shift,
                  pos_mult=self.pos_mult, pos_history=self.pos_history, pos_upsampling=self.pos_upsampling,
                  probe_inertia=self.probe_inertia)

    def op(self, p: HoloTomo):
        t0 = timeit.default_timer()
        ic_dt = 0
        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            ops = PropagateApplyAmplitude1(calc_llk=calc_llk) * ObjProbe2Psi1()

            if self.update_pos:
                if ic % self.update_pos == 0:
                    ops = Psi2PosReg1(upsampling=self.pos_upsampling, save_position_history=self.pos_history) * ops

            if p.probe_mode_coeff is None:
                ops = Psi2ObjProbe1(update_object=self.update_object, update_probe=self.update_probe,
                                    obj_min=self.obj_min, obj_max=self.obj_max,
                                    reg_obj_smooth=self.reg_obj_smooth,
                                    delta_beta=self.delta_beta, weight_empty=self.weight_empty) * ops
            else:
                ops = Psi2ObjProbeCoherent1(update_object=self.update_object, update_probe=self.update_probe,
                                            obj_min=self.obj_min, obj_max=self.obj_max,
                                            reg_obj_smooth=self.reg_obj_smooth,
                                            delta_beta=self.delta_beta, weight_empty=self.weight_empty) * ops
            ops = LoopStack(ops)
            if self.update_probe:
                ops = Psi2ProbeMerge(inertia=self.probe_inertia) * ops
            p = ops * p

            if calc_llk:
                # Average time/cycle over the last N cycles
                dt = (timeit.default_timer() - t0) / (ic - ic_dt + 1)
                ic_dt = ic + 1
                t0 = timeit.default_timer()

                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_pos=self.update_pos, dt=dt, algorithm='AP', verbose=True)
            else:
                p.history.insert(p.cycle, update_obj=self.update_object, update_probe=self.update_probe,
                                 update_pos=self.update_pos, algorithm='AP', verbose=False)
            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    s = algo_string('AP', p, self.update_object, self.update_probe)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.llk_poisson / p.data.nb_obs)
                    # p = cpuop.ShowObjProbe(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1
        return p


class ObjProbe2PsiRAAR1(CUOperatorHoloTomo):
    """
    Operator multiplying object views and probe to produce the initial Psi array (before propagation)
    for all projections and distances in the stack.
    This operator performs the operation at the beginning of an RAAR cycle and computes
    Psi =  beta*obj*probe + Psi

    Applies only to the current stack.
    """

    def __init__(self, beta=0.75, zero_psi=False):
        """

        :param beta: the beta parameter for the RAAR algorithm
        :param zero_psi: if True, set the previous psi to zero. This should be used during
            the first RAAR cycle along with beta=1.
        """
        super().__init__()
        self.beta = np.float32(beta)
        self.zero_psi = zero_psi

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = p.data.stack_v[s.istack].iproj
        nb = np.int32(p.data.stack_v[s.istack].nb)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        if self.zero_psi:
            p._cu_stack.psi.fill(0, stream=pu.cu_stream)

        if p.probe_mode_coeff is None:
            pu.cu_obj_probez_mult_raar(s.obj[0, 0], p._cu_probe, s.psi, p._cu_psi_old,
                                       p._cu_dx[i0:i0 + nb], p._cu_dy[i0:i0 + nb],
                                       p._cu_sample_flag[i0:i0 + nb], nb, nz,
                                       nx, ny, self.beta, stream=pu.cu_stream)
        else:
            pu.cu_obj_probecohz_mult_raar(s.obj[0, 0], p._cu_probe, s.psi, p._cu_psi_old,
                                          p._cu_dx[i0:i0 + nb], p._cu_dy[i0:i0 + nb],
                                          p._cu_sample_flag[i0:i0 + nb], p._cu_probe_mode_coeff[i0:i0 + nb],
                                          nb, nz, nb_probe, nx, ny, self.beta, stream=pu.cu_stream)
        return p


class Psi2ObjProbeRAAR1(CUOperatorHoloTomo):
    """
    Operator projecting the psi array in sample space onto the object and probe update.
    The object and probe update is computed from 2*Psi - Psi_old,
    and Psi is updated to beta*Psi_old + (1-2beta) Psi

    Applies only to the current stack. The probe and normalisation are stored in temporary arrays.

    """

    def __init__(self, update_object=True, update_probe=True, obj_max=None, obj_min=None, reg_obj_smooth=0,
                 delta_beta=-1, weight_empty=1.0, beta=0.75):
        """

        :param update_object: if True, update the object
        :param update_probe: if True, update the probe
        :param obj_max: the maximum amplitude for the object
        :param obj_min: the minimum amplitude for the object
        :param reg_obj_smooth: the coefficient (typically 0-1) to smooth the object update
        :param delta_beta: delta/beta ratio. Ignored if < 0
        :param weight_empty: the relative weight of empty beam images for the probe update
        :param beta: the RAAR beta
        """
        super().__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.obj_max = obj_max
        self.obj_min = obj_min
        self.reg_obj_smooth = np.float32(reg_obj_smooth)
        if delta_beta == 0:
            self.beta_delta = np.float32(-1)
        else:
            self.beta_delta = np.float32(1 / delta_beta)
        self.weight_empty = np.float32(weight_empty)
        self.beta = np.float32(beta)

    def op(self, p: HoloTomo):
        assert p.probe_mode_coeff is None
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = s.iproj
        nb = np.int32(s.nb)

        if self.obj_max is None:
            obj_max = np.float32(-1)
        else:
            obj_max = np.float32(self.obj_max)
        if self.obj_min is None:
            obj_min = np.float32(-1)
        else:
            obj_min = np.float32(self.obj_min)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        if s.istack == 0:
            # TODO: do not create temporary arrays here but in parent operator (AP, DM, ML...)
            p._cu_probe_new = cua.zeros(shape=p._cu_probe.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
            p._cu_probe_norm = cua.zeros(shape=(nz, ny, nx), dtype=np.float32, allocator=pu.cu_mem_pool.allocate)
            #
            p._cu_scale_new = cua.zeros(shape=p.data.nproj, dtype=np.float32, allocator=pu.cu_mem_pool.allocate)

        if self.update_object:
            # keep copy of previous object for regularisation
            obj_old = s.obj
            if self.reg_obj_smooth > 0:
                obj_old = cua.empty(shape=s.obj.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
                cu_drv.memcpy_dtod_async(src=s.obj.gpudata, dest=obj_old.gpudata, size=s.obj.nbytes,
                                         stream=pu.cu_stream)
            # TODO: as for ptycho, it would be much more efficient to avoid this python loop
            for ii in range(nb):
                r = pu.cu_psi2obj_probe_raar(s.obj[ii, 0], obj_old[ii, 0], p._cu_probe, s.psi[ii], p._cu_psi_old[ii],
                                             p._cu_probe_new,
                                             p._cu_probe_norm, s.obj_phase0[ii, 0], p._cu_dx[i0 + ii],
                                             p._cu_dy[i0 + ii], p._cu_sample_flag[i0 + ii],
                                             nz, nx, ny, obj_min, obj_max,
                                             self.reg_obj_smooth, self.beta_delta,
                                             self.weight_empty, self.beta, stream=pu.cu_stream)
                # TODO: store the result directly in p._cu_scale_new[i0 + ii], like in pyOpenCL
                cu_drv.memcpy_dtod_async(src=int(r.gpudata), dest=int(p._cu_scale_new[i0 + ii].gpudata),
                                         size=r.nbytes, stream=pu.cu_stream)
        else:
            for ii in range(nb):
                pu.cu_psi2probe_raar(s.obj[ii, 0], p._cu_probe, s.psi[ii], p._cu_psi_old[ii],
                                     p._cu_probe_new, p._cu_probe_norm,
                                     p._cu_dx[i0 + ii], p._cu_dy[i0 + ii], p._cu_sample_flag[i0 + ii], nz,
                                     nx, ny, self.weight_empty, self.beta, stream=pu.cu_stream)

        # TODO:
        # - Take into account object inertia
        # - Prepare scale factor update, by comparing each image integrated probe intensity to the average

        return p


class Psi2ObjProbeCoherentRAAR1(CUOperatorHoloTomo):
    """
    Operator projecting the psi arrays in sample space onto the object and probe update.
    The object can be constrained to a min and max amplitude.
    This operator works with coherent probe modes, each projection having a
    different linear combination of the modes.

    Applies only to the current stack.
    """

    def __init__(self, update_object=True, update_probe=True, obj_max=None, obj_min=None, reg_obj_smooth=0,
                 delta_beta=-1, weight_empty=1.0, beta=0.75):
        """

        :param update_object: if True, update the object
        :param update_probe: if True, update the probe
        :param obj_max: the maximum amplitude for the object
        :param obj_min: the minimum amplitude for the object
        :param reg_obj_smooth: the coefficient (typically 0-1) to smooth the object update
        :param delta_beta: delta/beta ratio. Ignored if < 0
        :param weight_empty: the relative weight of empty beam images for the probe update
        """
        super().__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.obj_max = obj_max
        self.obj_min = obj_min
        self.reg_obj_smooth = np.float32(reg_obj_smooth)
        self.beta_delta = np.float32(1 / delta_beta)
        self.weight_empty = np.float32(weight_empty)
        self.beta = np.float32(beta)

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = s.iproj
        nb = np.int32(s.nb)

        if self.obj_max is None:
            obj_max = np.float32(-1)
        else:
            obj_max = np.float32(self.obj_max)
        if self.obj_min is None:
            obj_min = np.float32(-1)
        else:
            obj_min = np.float32(self.obj_min)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        if s.istack == 0:
            # TODO: do not create temporary arrays here but in parent operator (AP, DM, ML...) ?
            p._cu_probe_new = cua.zeros_like(p._cu_probe)
            # Normalisation factor for p._cu_probe_new
            p._cu_probe_norm = cua.zeros(shape=p._cu_probe.shape, dtype=np.float32, allocator=pu.cu_mem_pool.allocate)
            p._cu_probe_mode_coeff_new = cua.empty_like(p._cu_probe_mode_coeff)
        if True:  # self.update_object:
            # keep copy of previous object for regularisation
            obj_old = s.obj
            if self.reg_obj_smooth > 0:
                obj_old = cua.empty(shape=s.obj.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
                cu_drv.memcpy_dtod_async(src=s.obj.gpudata, dest=obj_old.gpudata, size=s.obj.nbytes,
                                         stream=pu.cu_stream)
            psi2obj_probe_modes_raar_red = pu.get_modes_nz_kernels(nb_probe, nz)["psi2obj_probe_modes_raar_red"]
            for ii in range(nb):
                r = psi2obj_probe_modes_raar_red(s.obj[ii, 0], obj_old[ii, 0], p._cu_probe, s.psi[ii],
                                                 p._cu_psi_old[ii], p._cu_probe_new,
                                                 p._cu_probe_norm, p._cu_probe_mode_coeff[i0 + ii], s.obj_phase0[ii, 0],
                                                 p._cu_dx[i0 + ii], p._cu_dy[i0 + ii],
                                                 p._cu_sample_flag[i0 + ii], nz, nb_obj, nb_probe, nx, ny, obj_min,
                                                 obj_max,
                                                 self.reg_obj_smooth, self.beta_delta,
                                                 self.weight_empty, self.beta, stream=pu.cu_stream)
                # print(r, r.shape)
                r = cua.GPUArray((2, nz, nb_probe), dtype=np.float32, gpudata=r.gpudata, base=r)
                # print(r, r.shape)
                p._cu_probe_mode_coeff_new[i0 + ii] = r[0] / r[1]
                # if i0 == 0:
                #    print(i0 + ii, i0, ii, r[0] / r[1], p._cu_probe_mode_coeff_new[i0 + ii])
        else:
            # TODO
            pass

        # TODO:
        # - Take into account object & probe inertia
        # update object or probe only

        return p


class RAAR(CUOperatorHoloTomo):
    """
    Perform relaxed averaged alternating reflection projections between detector and object/probe space.

    This operator applies to all projections and loops over the stacks.
    """

    def __init__(self, update_object=True, update_probe=True, beta=0.75, nb_cycle=1, calc_llk=False,
                 show_obj_probe=0, fig_num=None, obj_min=None, obj_max=None, reg_obj_smooth=0,
                 delta_beta=-1, weight_empty=1.0, update_pos=0, pos_max_shift=2, pos_mult=1,
                 pos_history=False, pos_upsampling=10, probe_inertia=0.01):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param beta: beta (between 0 and 1) for the RAAR algorithm.
            This can also be an array of beta values, one for each cycle.
        :param nb_cycle: number of cycles to perform. Equivalent to AP(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
            calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
            By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        :param obj_min, obj_max: min and max amplitude for the object. Can be None
        :param reg_obj_smooth: the coefficient (typically 0-1) to smooth the object update
        :param delta_beta: delta/beta ratio (typically 1e2 to 1e3) - a <=0 value disables the constraint.
            if delta_beta>1e6 , a pure phase object (amplitude=1) is optimised.
        :param weight_empty: relative weight given to empty beam images for the probe update
        :param update_pos: positive integer, if >0, update positions every 'update_pos' cycle.
            (default=False or 0, positions are not updated).
        :param pos_max_shift: maximum allowed shift (in pixels) per scan position (default=2)
        :param pos_mult: multiply the calculated position shifts by this value. Useful since the calculated
            shifts usually are a fraction of the actual shift.
        :param pos_history: if True, save the position history (for debugging, slow)
        :param pos_upsampling=10: upsampling for registration-based position update
        :param probe_inertia=0.01: the inertia for the probe update. Should be >0 at least when
            initialising the modes for stability.
        """
        super().__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.beta = beta
        self.nb_cycle = nb_cycle
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.obj_min = obj_min
        self.obj_max = obj_max
        self.reg_obj_smooth = reg_obj_smooth
        self.delta_beta = np.float32(delta_beta)
        self.weight_empty = weight_empty
        self.update_pos = int(update_pos)
        self.pos_max_shift = pos_max_shift
        self.pos_mult = pos_mult
        self.pos_history = pos_history
        self.pos_upsampling = pos_upsampling
        self.probe_inertia = probe_inertia

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new AP operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return RAAR(update_object=self.update_object, update_probe=self.update_probe, beta=self.beta,
                    nb_cycle=self.nb_cycle * n,
                    calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe, fig_num=self.fig_num,
                    obj_min=self.obj_min, obj_max=self.obj_max, reg_obj_smooth=self.reg_obj_smooth,
                    delta_beta=self.delta_beta, update_pos=self.update_pos, pos_max_shift=self.pos_max_shift,
                    pos_mult=self.pos_mult, pos_history=self.pos_history, pos_upsampling=self.pos_upsampling,
                    probe_inertia=self.probe_inertia)

    def op(self, p: HoloTomo):
        t0 = timeit.default_timer()
        ic_dt = 0

        pu = self.processing_unit
        p._cu_psi_old = cua.empty(shape=p._psi.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)

        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            if isinstance(self.beta, np.ndarray):
                if ic < len(self.beta):
                    beta = np.float32(self.beta[ic])
                else:
                    warnings.warn("DRAP: beta is an array but len(beta) >= nb_cycle !", stacklevel=1)
                    beta = np.float32(self.beta[-1])
            else:
                beta = np.float32(self.beta)

            if ic == 0:
                # This will fill Psi with obj*probe
                ops = PropagateApplyAmplitude1(calc_llk=calc_llk) * ObjProbe2PsiRAAR1(beta=1, zero_psi=True)
            else:
                ops = PropagateApplyAmplitude1(calc_llk=calc_llk) * ObjProbe2PsiRAAR1(beta=beta)

            if self.update_pos:
                if ic % self.update_pos == 0:
                    ops = Psi2PosReg1(upsampling=self.pos_upsampling, save_position_history=self.pos_history) * ops

            if p.probe_mode_coeff is None:
                ops = Psi2ObjProbeRAAR1(update_object=self.update_object, update_probe=self.update_probe,
                                        obj_min=self.obj_min, obj_max=self.obj_max,
                                        reg_obj_smooth=self.reg_obj_smooth, delta_beta=self.delta_beta,
                                        weight_empty=self.weight_empty, beta=beta) * ops
            else:
                ops = Psi2ObjProbeCoherentRAAR1(update_object=self.update_object, update_probe=self.update_probe,
                                                obj_min=self.obj_min, obj_max=self.obj_max,
                                                reg_obj_smooth=self.reg_obj_smooth, delta_beta=self.delta_beta,
                                                weight_empty=self.weight_empty, beta=beta) * ops
            ops = LoopStack(ops, copy_psi=True)
            if self.update_probe:
                ops = Psi2ProbeMerge(inertia=self.probe_inertia) * ops
            p = ops * p

            if calc_llk:
                # Average time/cycle over the last N cycles
                dt = (timeit.default_timer() - t0) / (ic - ic_dt + 1)
                ic_dt = ic + 1
                t0 = timeit.default_timer()

                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_pos=self.update_pos, dt=dt, algorithm='RAAR', verbose=True)
            else:
                p.history.insert(p.cycle, update_obj=self.update_object, update_probe=self.update_probe,
                                 update_pos=self.update_pos, algorithm='RAAR', verbose=False)
            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    s = algo_string('RAAR', p, self.update_object, self.update_probe)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.llk_poisson / p.data.nb_obs)
                    # p = cpuop.ShowObjProbe(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1
        # del p._cu_psi_old
        return p


class ObjProbe2PsiDRAP1(CUOperatorHoloTomo):
    """
    Operator multiplying object views and probe to produce the initial Psi array (before propagation)
    for all projections and distances in the stack.
    This operator performs the operation at the beginning of an RAAR cycle and computes
    Psi =  obj*probe + Psi

    Applies only to the current stack.
    """

    def __init__(self, zero_psi=False):
        """

        :param beta: the beta parameter for the DRAP algorithm
        :param zero_psi: if True, set the previous psi to zero. This should be used during
            the first DRAP cycle along with beta=1.
        """
        super().__init__()
        self.zero_psi = zero_psi

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = p.data.stack_v[s.istack].iproj
        nb = np.int32(p.data.stack_v[s.istack].nb)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        if self.zero_psi:
            p._cu_stack.psi.fill(0, stream=pu.cu_stream)

        if p.probe_mode_coeff is None:
            pu.cu_obj_probez_mult_drap(s.obj[0, 0], p._cu_probe, s.psi, p._cu_psi_old,
                                       p._cu_dx[i0:i0 + nb], p._cu_dy[i0:i0 + nb],
                                       p._cu_sample_flag[i0:i0 + nb], nb, nz,
                                       nx, ny, stream=pu.cu_stream)
        else:
            pu.cu_obj_probecohz_mult_drap(s.obj[0, 0], p._cu_probe, s.psi, p._cu_psi_old,
                                          p._cu_dx[i0:i0 + nb], p._cu_dy[i0:i0 + nb],
                                          p._cu_sample_flag[i0:i0 + nb], p._cu_probe_mode_coeff[i0:i0 + nb],
                                          nb, nz, nb_probe, nx, ny, stream=pu.cu_stream)
        return p


class Psi2ObjProbeDRAP1(CUOperatorHoloTomo):
    """
    Operator projecting the psi array in sample space onto the object and probe update.
    The object and probe update is computed from (1+beta)*Psi - beta*Psi_old,
    and Psi is updated to -beta*(Psi-Psi_old)

    Applies only to the current stack. The probe and normalisation are stored in temporary arrays.
    """

    def __init__(self, update_object=True, update_probe=True, obj_max=None, obj_min=None, reg_obj_smooth=0,
                 delta_beta=-1, weight_empty=1.0, beta=0.75):
        """

        :param update_object: if True, update the object
        :param update_probe: if True, update the probe
        :param obj_max: the maximum amplitude for the object
        :param obj_min: the minimum amplitude for the object
        :param reg_obj_smooth: the coefficient (typically 0-1) to smooth the object update
        :param delta_beta: delta/beta ratio. Ignored if < 0
        :param weight_empty: the relative weight of empty beam images for the probe update
        :param beta: the DRAP beta (or lambda in the Thao 2018 article)
        """
        super().__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.obj_max = obj_max
        self.obj_min = obj_min
        self.reg_obj_smooth = np.float32(reg_obj_smooth)
        if delta_beta == 0:
            self.beta_delta = np.float32(-1)
        else:
            self.beta_delta = np.float32(1 / delta_beta)
        self.weight_empty = np.float32(weight_empty)
        self.beta = np.float32(beta)

    def op(self, p: HoloTomo):
        assert p.probe_mode_coeff is None
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = s.iproj
        nb = np.int32(s.nb)

        if self.obj_max is None:
            obj_max = np.float32(-1)
        else:
            obj_max = np.float32(self.obj_max)
        if self.obj_min is None:
            obj_min = np.float32(-1)
        else:
            obj_min = np.float32(self.obj_min)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        if s.istack == 0:
            # TODO: do not create temporary arrays here but in parent operator (AP, DM, ML...)
            p._cu_probe_new = cua.zeros(shape=p._cu_probe.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
            p._cu_probe_norm = cua.zeros(shape=(nz, ny, nx), dtype=np.float32, allocator=pu.cu_mem_pool.allocate)
            #
            p._cu_scale_new = cua.zeros(shape=p.data.nproj, dtype=np.float32, allocator=pu.cu_mem_pool.allocate)

        if self.update_object:
            # keep copy of previous object for regularisation
            obj_old = s.obj
            if self.reg_obj_smooth > 0:
                obj_old = cua.empty(shape=s.obj.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
                cu_drv.memcpy_dtod_async(src=s.obj.gpudata, dest=obj_old.gpudata, size=s.obj.nbytes,
                                         stream=pu.cu_stream)
            # TODO: as for ptycho, it would be much more efficient to avoid this python loop
            for ii in range(nb):
                r = pu.cu_psi2obj_probe_drap(s.obj[ii, 0], obj_old[ii, 0], p._cu_probe, s.psi[ii], p._cu_psi_old[ii],
                                             p._cu_probe_new,
                                             p._cu_probe_norm, s.obj_phase0[ii, 0], p._cu_dx[i0 + ii],
                                             p._cu_dy[i0 + ii], p._cu_sample_flag[i0 + ii],
                                             nz, nx, ny, obj_min, obj_max,
                                             self.reg_obj_smooth, self.beta_delta,
                                             self.weight_empty, self.beta, stream=pu.cu_stream)
                # TODO: store the result directly in p._cu_scale_new[i0 + ii], like in pyOpenCL
                cu_drv.memcpy_dtod_async(src=int(r.gpudata), dest=int(p._cu_scale_new[i0 + ii].gpudata),
                                         size=r.nbytes, stream=pu.cu_stream)
        else:
            for ii in range(nb):
                pu.cu_psi2probe_drap(s.obj[ii, 0], p._cu_probe, s.psi[ii], p._cu_psi_old[ii],
                                     p._cu_probe_new, p._cu_probe_norm,
                                     p._cu_dx[i0 + ii], p._cu_dy[i0 + ii], p._cu_sample_flag[i0 + ii], nz,
                                     nx, ny, self.weight_empty, self.beta, stream=pu.cu_stream)

        # TODO:
        # - Take into account object inertia
        # - Prepare scale factor update, by comparing each image integrated probe intensity to the average

        return p


class Psi2ObjProbeCoherentDRAP1(CUOperatorHoloTomo):
    """
    Operator projecting the psi arrays in sample space onto the object and probe update.
    The object can be constrained to a min and max amplitude.
    This operator works with coherent probe modes, each projection having a
    different linear combination of the modes.

    Applies only to the current stack.
    """

    def __init__(self, update_object=True, update_probe=True, obj_max=None, obj_min=None, reg_obj_smooth=0,
                 delta_beta=-1, weight_empty=1.0, beta=0.75):
        """

        :param update_object: if True, update the object
        :param update_probe: if True, update the probe
        :param obj_max: the maximum amplitude for the object
        :param obj_min: the minimum amplitude for the object
        :param reg_obj_smooth: the coefficient (typically 0-1) to smooth the object update
        :param delta_beta: delta/beta ratio. Ignored if < 0
        :param weight_empty: the relative weight of empty beam images for the probe update
        """
        super().__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.obj_max = obj_max
        self.obj_min = obj_min
        self.reg_obj_smooth = np.float32(reg_obj_smooth)
        self.beta_delta = np.float32(1 / delta_beta)
        self.weight_empty = np.float32(weight_empty)
        self.beta = np.float32(beta)

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = s.iproj
        nb = np.int32(s.nb)

        if self.obj_max is None:
            obj_max = np.float32(-1)
        else:
            obj_max = np.float32(self.obj_max)
        if self.obj_min is None:
            obj_min = np.float32(-1)
        else:
            obj_min = np.float32(self.obj_min)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        if s.istack == 0:
            # TODO: do not create temporary arrays here but in parent operator (AP, DM, ML...) ?
            p._cu_probe_new = cua.zeros_like(p._cu_probe)
            # Normalisation factor for p._cu_probe_new
            p._cu_probe_norm = cua.zeros(shape=p._cu_probe.shape, dtype=np.float32, allocator=pu.cu_mem_pool.allocate)
            p._cu_probe_mode_coeff_new = cua.empty_like(p._cu_probe_mode_coeff)
        if True:  # self.update_object:
            # keep copy of previous object for regularisation
            obj_old = s.obj
            if self.reg_obj_smooth > 0:
                obj_old = cua.empty(shape=s.obj.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
                cu_drv.memcpy_dtod_async(src=s.obj.gpudata, dest=obj_old.gpudata, size=s.obj.nbytes,
                                         stream=pu.cu_stream)
            psi2obj_probe_modes_drap_red = pu.get_modes_nz_kernels(nb_probe, nz)["psi2obj_probe_modes_drap_red"]
            for ii in range(nb):
                r = psi2obj_probe_modes_drap_red(s.obj[ii, 0], obj_old[ii, 0], p._cu_probe, s.psi[ii],
                                                 p._cu_psi_old[ii], p._cu_probe_new,
                                                 p._cu_probe_norm, p._cu_probe_mode_coeff[i0 + ii], s.obj_phase0[ii, 0],
                                                 p._cu_dx[i0 + ii], p._cu_dy[i0 + ii],
                                                 p._cu_sample_flag[i0 + ii], nz, nb_obj, nb_probe, nx, ny, obj_min,
                                                 obj_max,
                                                 self.reg_obj_smooth, self.beta_delta,
                                                 self.weight_empty, self.beta, stream=pu.cu_stream)
                # print(r, r.shape)
                r = cua.GPUArray((2, nz, nb_probe), dtype=np.float32, gpudata=r.gpudata, base=r)
                # print(r, r.shape)
                p._cu_probe_mode_coeff_new[i0 + ii] = r[0] / r[1]
                # if i0 == 0:
                #    print(i0 + ii, i0, ii, r[0] / r[1], p._cu_probe_mode_coeff_new[i0 + ii])
        else:
            # TODO
            pass

        # TODO:
        # - Take into account object & probe inertia
        # update object or probe only

        return p


class DRAP(CUOperatorHoloTomo):
    """
    Perform Douglas-Rachford Alternating Proections between detector and object/probe space,
    according to (Thao 2018, https://doi.org/10.1007/s10589-018-9989-y and
    Hagemann 2018 https://doi.org/10.1063/1.5029927).
    This corresponds to the operator:
        DRAP = P_S ((1+ß)P_M −ßI) − ß(P_M − I)
    where P_M is the magnitude projector, and P_S the projector onto the object and probe.

    This operator applies to all projections and loops over the stacks.
    """

    def __init__(self, update_object=True, update_probe=True, beta=0.75, nb_cycle=1, calc_llk=False,
                 show_obj_probe=0, fig_num=None, obj_min=None, obj_max=None, reg_obj_smooth=0,
                 delta_beta=-1, weight_empty=1.0, update_pos=0, pos_max_shift=2, pos_mult=1,
                 pos_history=False, pos_upsampling=10, probe_inertia=0.01):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param beta: beta (between 0 and 1) for the DRAP algorithm (lambda in the Thao 2018 paper)
            Note that beta can also be an array of beta values, one for each cycle.
        :param nb_cycle: number of cycles to perform. Equivalent to AP(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
            calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
            By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        :param obj_min, obj_max: min and max amplitude for the object. Can be None
        :param reg_obj_smooth: the coefficient (typically 0-1) to smooth the object update
        :param delta_beta: delta/beta ratio (typically 1e2 to 1e3) - a <=0 value disables the constraint.
            if delta_beta>1e6 , a pure phase object (amplitude=1) is optimised.
        :param weight_empty: relative weight given to empty beam images for the probe update
        :param update_pos: positive integer, if >0, update positions every 'update_pos' cycle.
            (default=False or 0, positions are not updated).
        :param pos_max_shift: maximum allowed shift (in pixels) per scan position (default=2)
        :param pos_mult: multiply the calculated position shifts by this value. Useful since the calculated
            shifts usually are a fraction of the actual shift.
        :param pos_history: if True, save the position history (for debugging, slow)
        :param pos_upsampling=10: upsampling for registration-based position update
        :param probe_inertia=0.01: the inertia for the probe update. Should be >0 at least when
            initialising the modes for stability.
        """
        super().__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.beta = beta
        self.nb_cycle = nb_cycle
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.obj_min = obj_min
        self.obj_max = obj_max
        self.reg_obj_smooth = reg_obj_smooth
        self.delta_beta = np.float32(delta_beta)
        self.weight_empty = weight_empty
        self.update_pos = int(update_pos)
        self.pos_max_shift = pos_max_shift
        self.pos_mult = pos_mult
        self.pos_history = pos_history
        self.pos_upsampling = pos_upsampling
        self.probe_inertia = probe_inertia

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new DRAP operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return DRAP(update_object=self.update_object, update_probe=self.update_probe, beta=self.beta,
                    nb_cycle=self.nb_cycle * n,
                    calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe, fig_num=self.fig_num,
                    obj_min=self.obj_min, obj_max=self.obj_max, reg_obj_smooth=self.reg_obj_smooth,
                    delta_beta=self.delta_beta, update_pos=self.update_pos, pos_max_shift=self.pos_max_shift,
                    pos_mult=self.pos_mult, pos_history=self.pos_history, pos_upsampling=self.pos_upsampling,
                    probe_inertia=self.probe_inertia)

    def op(self, p: HoloTomo):
        t0 = timeit.default_timer()
        ic_dt = 0

        pu = self.processing_unit
        p._cu_psi_old = cua.empty(shape=p._psi.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)

        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            if isinstance(self.beta, np.ndarray):
                if ic < len(self.beta):
                    beta = np.float32(self.beta[ic])
                else:
                    warnings.warn("DRAP: beta is an array but len(beta) >= nb_cycle !", stacklevel=1)
                    beta = np.float32(self.beta[-1])
            else:
                beta = np.float32(self.beta)

            if ic == 0:
                # This will fill Psi with obj*probe
                ops = PropagateApplyAmplitude1(calc_llk=calc_llk) * ObjProbe2PsiDRAP1(zero_psi=True)
            else:
                ops = PropagateApplyAmplitude1(calc_llk=calc_llk) * ObjProbe2PsiDRAP1()

            if self.update_pos:
                if ic % self.update_pos == 0:
                    ops = Psi2PosReg1(upsampling=self.pos_upsampling, save_position_history=self.pos_history) * ops

            if p.probe_mode_coeff is None:
                ops = Psi2ObjProbeDRAP1(update_object=self.update_object, update_probe=self.update_probe,
                                        obj_min=self.obj_min, obj_max=self.obj_max,
                                        reg_obj_smooth=self.reg_obj_smooth, delta_beta=self.delta_beta,
                                        weight_empty=self.weight_empty, beta=beta) * ops
            else:
                ops = Psi2ObjProbeCoherentDRAP1(update_object=self.update_object, update_probe=self.update_probe,
                                                obj_min=self.obj_min, obj_max=self.obj_max,
                                                reg_obj_smooth=self.reg_obj_smooth, delta_beta=self.delta_beta,
                                                weight_empty=self.weight_empty, beta=beta) * ops
            ops = LoopStack(ops, copy_psi=True)
            if self.update_probe:
                ops = Psi2ProbeMerge(inertia=self.probe_inertia) * ops
            p = ops * p

            if calc_llk:
                # Average time/cycle over the last N cycles
                dt = (timeit.default_timer() - t0) / (ic - ic_dt + 1)
                ic_dt = ic + 1
                t0 = timeit.default_timer()

                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_pos=self.update_pos, dt=dt, algorithm='DRAP', verbose=True)
            else:
                p.history.insert(p.cycle, update_obj=self.update_object, update_probe=self.update_probe,
                                 update_pos=self.update_pos, algorithm='DRAP', verbose=False)
            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    s = algo_string('DRAP', p, self.update_object, self.update_probe)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.llk_poisson / p.data.nb_obs)
                    # p = cpuop.ShowObjProbe(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1
        # del p._cu_psi_old
        return p


class DM1(CUOperatorHoloTomo):
    """
    Equivalent to operator: 2 * ObjProbe2Psi1() - I.
    Also makes a copy of Psi in p._cu_psi_old

    Applies only to the current stack
    """

    def op(self, p: HoloTomo):
        """

        :param p: the HoloTomo object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = p.data.stack_v[s.istack].iproj
        nb = np.int32(p.data.stack_v[s.istack].nb)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        # Copy Psi / use memory pool so should not be wasteful
        cu_drv.memcpy_dtod_async(src=s.psi.gpudata, dest=p._cu_psi_old.gpudata, size=s.psi.nbytes, stream=pu.cu_stream)

        pu.cu_obj_probe2psi_dm1(s.obj[0, 0], p._cu_probe, s.psi, p._cu_dx[i0:i0 + nb], p._cu_dy[i0:i0 + nb],
                                p._cu_sample_flag[i0:i0 + nb], nb, nz, nb_obj, nb_probe, nx, ny,
                                stream=pu.cu_stream)

        return p


class DM2(CUOperatorHoloTomo):
    """
    # Psi(n+1) = Psi(n) - P*O + Psi_fourier

    This operator assumes that Psi_fourier is the current Psi, and that Psi(n) is in p._cu_psi_old

    Applies only to the current stack
    """

    def op(self, p: HoloTomo):
        """

        :param p: the HoloTomo object this operator applies to
        :return: the updated HoloTomo object
        """
        pu = self.processing_unit
        s = p._cu_stack
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        i0 = p.data.stack_v[s.istack].iproj
        nb = np.int32(p.data.stack_v[s.istack].nb)

        # Data size
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)

        pu.cu_obj_probe2psi_dm2(s.obj[0, 0], p._cu_probe, s.psi, p._cu_psi_old, p._cu_dx[i0:i0 + nb],
                                p._cu_dy[i0:i0 + nb], p._cu_sample_flag[i0:i0 + nb], nb, nz, nb_obj, nb_probe, nx,
                                ny, stream=pu.cu_stream)
        return p


class DM(CUOperatorHoloTomo):
    """
    Run Difference Map algorithm between detector and object/probe space.

    This operator applies to all projections and loops over the stacks.
    """

    def __init__(self, update_object=True, update_probe=True, nb_cycle=1, calc_llk=False,
                 show_obj_probe=0, fig_num=None, obj_min=None, obj_max=None, reg_obj_smooth=0,
                 delta_beta=-1, weight_empty=1.0, update_pos=0, pos_max_shift=2, pos_mult=1,
                 pos_history=False, probe_inertia=0.01):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param nb_cycle: number of cycles to perform. Equivalent to AP(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        :param obj_min, obj_max: min and max amplitude for the object. Can be None
        :param reg_obj_smooth: the coefficient (typically 0-1) to smooth the object update
        :param delta_beta: delta/beta ratio (typically 1e2 to 1e3) - a negative value disables the constraint
        :param weight_empty: relative weight given to empty beam images for the probe update
        :param update_pos: positive integer, if >0, update positions every 'update_pos' cycle.
            (default=False or 0, positions are not updated).
        :param pos_max_shift: maximum allowed shift (in pixels) per scan position (default=2)
        :param pos_mult: multiply the calculated position shifts by this value. Useful since the calculated
            shifts usually are a fraction of the actual shift.
        :param pos_history: if True, save the position history (for debugging, slow)
        """
        super(DM, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.nb_cycle = nb_cycle
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.obj_min = obj_min
        self.obj_max = obj_max
        self.reg_obj_smooth = reg_obj_smooth
        self.delta_beta = np.float32(delta_beta)
        self.weight_empty = weight_empty
        self.update_pos = int(update_pos)
        self.pos_max_shift = pos_max_shift
        self.pos_mult = pos_mult
        self.pos_history = pos_history
        self.probe_inertia = probe_inertia

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new DM operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return DM(update_object=self.update_object, update_probe=self.update_probe, nb_cycle=self.nb_cycle * n,
                  calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe, fig_num=self.fig_num,
                  obj_min=self.obj_min, obj_max=self.obj_max, reg_obj_smooth=self.reg_obj_smooth,
                  delta_beta=self.delta_beta, weight_empty=self.weight_empty, update_pos=self.update_pos,
                  pos_max_shift=self.pos_max_shift, pos_mult=self.pos_mult, pos_history=self.pos_history,
                  probe_inertia=self.probe_inertia)

    def op(self, p: HoloTomo):
        # First loop to get a starting Psi
        p = LoopStack(ObjProbe2Psi1(), copy_psi=True) * p

        pu = self.processing_unit
        p._cu_psi_old = cua.empty(shape=p._psi.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)

        t0 = timeit.default_timer()
        ic_dt = 0
        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            ops = DM2() * PropagateApplyAmplitude1(calc_llk=calc_llk) * DM1()

            if self.update_pos:
                if ic % self.update_pos == 0:
                    ops = Psi2PosShift1(multiplier=self.pos_mult, max_shift=self.pos_max_shift,
                                        save_position_history=self.pos_history) * ops

            if p.probe_mode_coeff is None:
                ops = Psi2ObjProbe1(update_object=self.update_object, update_probe=self.update_probe,
                                    obj_min=self.obj_min, obj_max=self.obj_max,
                                    reg_obj_smooth=self.reg_obj_smooth,
                                    delta_beta=self.delta_beta, weight_empty=self.weight_empty) * ops
            else:
                ops = Psi2ObjProbeCoherent1(update_object=self.update_object, update_probe=self.update_probe,
                                            obj_min=self.obj_min, obj_max=self.obj_max,
                                            reg_obj_smooth=self.reg_obj_smooth,
                                            delta_beta=self.delta_beta, weight_empty=self.weight_empty) * ops

            p = LoopStack(ops, copy_psi=True) * p
            if self.update_probe:
                p = Psi2ProbeMerge(inertia=self.probe_inertia) * p

            if calc_llk:
                # Average time/cycle over the last N cycles
                dt = (timeit.default_timer() - t0) / (ic - ic_dt + 1)
                ic_dt = ic + 1
                t0 = timeit.default_timer()

                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_pos=self.update_pos, dt=dt, algorithm='DM', verbose=True)
            else:
                p.history.insert(p.cycle, update_obj=self.update_object, update_probe=self.update_probe,
                                 update_pos=self.update_pos, algorithm='DM', verbose=False)
            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    s = algo_string('DM', p, self.update_object, self.update_probe)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.llk_poisson / p.data.nb_obs)
                    # p = cpuop.ShowObjProbe(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1

        # Cleanup
        # del p._cu_psi_old
        return p


class Calc2Obs1(CUOperatorHoloTomo):
    """
    Copy the calculated intensities to the observed ones. Can be used for simulation.
    Assumes the current Psi is already in Fourier space.

    Applies only to the current stack.
    """

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        sdata = p.data.stack_v[s.istack]
        nb = np.int32(p.data.stack_v[s.istack].nb)
        if p.probe_mode_coeff is None:
            nb_mode = np.int32(p.nb_obj * p.nb_probe)
        else:
            nb_mode = np.int32(p.nb_obj)
        nx = np.int32(p.data.nx)
        ny = np.int32(p.data.ny)
        pu.cu_calc2obs(s.iobs[:nb], s.psi, nb_mode, nx, ny, stream=pu.cu_stream)
        cu_drv.memcpy_dtoh_async(dest=sdata.iobs[:nb], src=s.iobs[:nb].gpudata, stream=pu.cu_stream)
        return p


class Calc2ObsPoisson1(CUOperatorHoloTomo):
    """
    Copy the calculated intensities to the observed ones. Can be used for simulation.
    Assumes the current Psi is already in Fourier space.

    Applies only to the current stack.
    """

    def __init__(self, scale):
        """

        :param poisson_noise: if True, will add Poisson noise to the calculated intensities
        :param nb_photon: the average number of photon per pixel to use for Poisson noise
        """
        super(Calc2ObsPoisson1, self).__init__()
        self.scale = scale

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        s0 = p.data.stack_v[s.istack]
        s0.iobs[:s.nb] = np.random.poisson(s0.iobs[:s.nb] * self.scale)
        cu_drv.memcpy_htod_async(dest=s.iobs.gpudata, src=s0.iobs, stream=pu.cu_stream)
        return p


class Calc2Obs(CUOperatorHoloTomo):
    """
    Copy the calculated intensities to the observed ones. Can be used for simulation.
    Will apply to all projection stacks of the HoloTomo object
    """

    def __init__(self, poisson_noise=True, nb_photon=1e3):
        """

        :param poisson_noise: if True, will add Poisson noise to the calculated intensities
        :param nb_photon: the average number of photon per pixel to use for Poisson noise
        """
        super(Calc2Obs, self).__init__()
        self.poisson_noise = poisson_noise
        self.nb_photon = nb_photon

    def op(self, p: HoloTomo):
        p = LoopStack(op=Calc2Obs1() * PropagateNearField1() * ObjProbe2Psi1(),
                      out=False, copy_psi=False, verbose=False) * p
        p._from_pu()
        if self.poisson_noise:
            iobs_sum = 0
            for s in p.data.stack_v:
                iobs_sum += s.iobs[:s.nb].sum()
            scalef = self.nb_photon * p.data.nb_obs / iobs_sum
            # This update of Iobs must be done in a proper LoopStack, so that
            # arrays updated are correctly sync'd between GPU and CPU, especially
            # considering the pre-fetching of the next stack..
            p = LoopStack(Calc2ObsPoisson1(scale=scalef)) * p
        return p


class BackPropagatePaganin1(CUOperatorHoloTomo):
    """ Back-propagation algorithm using Paganin's approach.
    Ref: Paganin et al., Journal of microscopy 206 (2002), 33–40. (DOI: 10.1046/j.1365-2818.2002.01010.x)

    Multi-distance operator as in Eq. 13 in Yu et al., Opt. Express 26, 11110 (2018).

    This operator uses the observed intensity to calculate a low-resolution estimate of the object, given the
    delta and beta values of its refraction index.

    The result of the transformation is the calculated object as a transmission factor, i.e. if T(r) is the
    estimated thickness of the sample, it is exp(-mu * T - 2*pi/lambda * T)

    The resulting object projection is stored in the first object mode. If the object is defined with multiple modes,
    secondary ones are set to zero.

    Applies only to the current stack.
    """

    def __init__(self, iz, delta_beta, alpha, cu_iobs_empty):
        """

        :param iz: the index of the detector distance to be taken into account (by default 0) for the propagation.
                   If None, the result from all distances will be averaged.
        :param delta_beta: delta/beta ratio, with the refraction index: n = 1 - delta + i * beta
        :param alpha: regularisation parameter
        :param cu_iobs_empty: the GPUarray with the empty beam image used for normalisation
        """
        super(BackPropagatePaganin1, self).__init__()
        self.iz = iz
        self.delta_beta = np.float32(delta_beta)
        if alpha is None:
            self.alpha = np.float32(0)
        else:
            self.alpha = np.float32(alpha)
        self.cu_iobs_empty = cu_iobs_empty

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        i0 = s.iproj
        nb_obj = np.int32(p.nb_obj)
        if p.probe_mode_coeff is None:
            nb_probe = np.int32(p.nb_probe)
        else:
            nb_probe = np.int32(1)
        nb_mode = np.int32(p.nb_obj * nb_probe)
        nx = np.int32(p.data.nx)
        ny = np.int32(p.data.ny)
        nz = np.int32(p.data.nz)
        px = np.float32(p.data.pixel_size_detector)
        nb = np.int32(s.nb)
        padding = np.int32(p.data.padding)
        nb_proj = np.int32(s.nb)

        # Note: the calculation is done on the entire stack even if actually only 1 mode and 1 z is used,
        # this is wasteful but the operator should only be run once, so it matters little

        # 0 copy iobs into psi for FT. The padded areas and masked pixels are interpolated
        pu.cu_iobs2psi(s.iobs[:nb], self.cu_iobs_empty, s.psi, p._cu_dx[i0:], p._cu_dy[i0:],
                       nb_mode, nx, ny, nz, padding, stream=pu.cu_stream)

        # 1 FT normalised observed intensity
        if False:
            # Smooth to avoid noise ?
            sigma = np.float32(3)
            nzg = np.int32(nb * nz * nb_mode)
            pu.gauss_convolc_16x(s.psi[:nb], sigma, nx, ny, nzg, block=(16, 1, 1), grid=(1, int(ny), int(nzg)),
                                 stream=pu.cu_stream)
            pu.gauss_convolc_16y(s.psi[:nb], sigma, nx, ny, nzg, block=(1, 16, 1), grid=(int(nx), 1, int(nzg)),
                                 stream=pu.cu_stream)

        p = FT1(scale=True) * p

        # 2 Paganin operator in Fourier space
        if self.iz is None:
            # Multi-distance version
            if s.istack == 0:
                print("Using Paganin multi-distance reconstruction, alpha=%8.5f" % self.alpha)
            pilambdad = np.array(p.data.detector_distance * p.data.wavelength * np.pi, dtype=np.float32)
            pilambdad = cua.to_gpu_async(pilambdad, allocator=pu.cu_mem_pool.allocate, stream=pu.cu_stream)
            pu.cu_paganin_fourier_multi(s.psi[0, 0, 0, 0], pilambdad, np.float32(self.delta_beta),
                                        px, nb_mode, nx, ny, nz, nb_proj, self.alpha, stream=pu.cu_stream)

            # 3 Back-propagate & compute object and its original phase
            p = IFT1(scale=True) * p
            pu.cu_paganin2obj(s.psi[0, 0, 0, 0], s.obj, s.obj_phase0, self.delta_beta,
                              nb_probe, nb_obj, nx, ny, nz, nb_proj, stream=pu.cu_stream)
            # print("Paganin #", s.istack, abs(s.obj.get()).sum(axis=(1, 2, 3)), s.iobs.get().sum(axis=(1, 2, 3)))
        else:
            iz = np.int32(self.iz)
            # Single-distance version
            if s.istack == 0:
                print("Using Paganin single-distance reconstruction")
            z = p.data.detector_distance
            alpha = np.array(self.delta_beta * z * p.data.wavelength / (2 * np.pi), dtype=np.float32)
            cu_alpha = cua.to_gpu_async(alpha, allocator=pu.cu_mem_pool.allocate, stream=pu.cu_stream)
            pu.cu_paganin_fourier(s.iobs[:nb], s.psi, cu_alpha, px, nb_mode, nx, ny, nz, stream=pu.cu_stream)

            # 3 Back-propagate and compute thickness and object value
            p = IFT1(scale=True) * p
            pu.cu_paganin_thickness(s.iobs[:nb], s.obj, s.psi, s.obj_phase0, iz, self.delta_beta,
                                    nb_probe, nb_obj, nx, ny, nz, stream=pu.cu_stream)

        # Copy back unwrapped object phase as it is not done in SwapStack
        sout = p.data.stack_v[s.istack]
        cu_drv.memcpy_dtoh_async(src=s.obj_phase0.gpudata, dest=sout.obj_phase0, stream=pu.cu_stream)

        return p


class BackPropagatePaganin(CUOperatorHoloTomo):
    """
    Back-propagation algorithm using the Paganin algorithm.
    Ref: Paganin et al., Journal of microscopy 206 (2002), 33–40. (DOI: 10.1046/j.1365-2818.2002.01010.x)

    Multi-distance operator as in Eq. 13 in Yu et al., Opt. Express 26, 11110 (2018).

    This operator uses the observed intensity to calculate a low-resolution estimate of the object, given the
    delta and beta values of its refraction index.

    The result of the transformation is the calculated object as a transmission factor, i.e. if T(r) is the
    estimated thickness of the sample, it is exp(-0.5*mu * T) * exp(-0.5i * delta / beta * mu * T)

    The resulting object projection is stored in the first object mode. If the object is defined with multiple modes,
    secondary ones are set to zero.

    The probe is set to a real object of modulus sqrt(iobs_empty), interpolating
    the padded areas and masked pixels. All modes are set to the same value.

    Applies to all projection stacks.
    """

    def __init__(self, iz=None, delta_beta=300, alpha=0, init_probe_modes=False):
        """

        :param iz: the index of the detector distance to be taken into account (by default 0)
            for the propagation. If None, the multi-distance formula will be used.
        :param delta_beta: delta/beta ratio, with the refraction index: n = 1 - delta + i * beta
        :param alpha: regularisation parameter, should be >=0. Used for the multi-distance approach.
        :param init_probe_modes: if True, init the coherent probe modes coefficients to 1/nb_probe.
            If False, the current values are kept.
        """
        super(BackPropagatePaganin, self).__init__()
        self.iz = iz
        self.delta_beta = delta_beta
        self.alpha = alpha
        self.init_probe_modes = init_probe_modes

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        # Find the empty beam image, move it to GPU to use it as normalisation (on CPU)
        nb_empty = 0
        iobs_empty = np.zeros((p.data.nz, p.data.ny, p.data.nx), dtype=np.float32)

        for s in p.data.stack_v:
            for iproj in range(s.nb):
                i = s.iproj + iproj
                if bool(p.data.sample_flag[i]) is False:
                    nb_empty += 1
                    iobs_empty += s.iobs[iproj]
        if nb_empty > 0:
            iobs_empty /= nb_empty
        else:
            iobs_empty[:] = 1

        cu_iobs_empty = cua.to_gpu_async(iobs_empty, allocator=self.processing_unit.cu_mem_pool.allocate,
                                         stream=pu.cu_stream)

        pu = self.processing_unit
        nx = np.int32(p.data.nx)
        ny = np.int32(p.data.ny)
        nz = np.int32(p.data.nz)
        # Set probe to one mode
        nb_probe = np.int32(p.nb_probe)
        p._cu_probe = cua.empty((nz, nb_probe, ny, nx), dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
        padding = np.int32(p.data.padding)

        # Put iobs_empty in probe
        pu.cu_iobs_empty2probe(cu_iobs_empty, p._cu_probe, nb_probe, nx, ny, nz, padding, stream=pu.cu_stream)

        if p.probe_mode_coeff is not None and self.init_probe_modes:
            # Should fill random values instead ?
            p._cu_probe_mode_coeff.fill(np.float32(1 / nb_probe), stream=pu.cu_stream)

        p = LoopStack(op=BackPropagatePaganin1(iz=self.iz, delta_beta=self.delta_beta, alpha=self.alpha,
                                               cu_iobs_empty=cu_iobs_empty),
                      out=True, copy_psi=True, verbose=True) * p

        return p


class BackPropagateCTF1(CUOperatorHoloTomo):
    """
    Back-propagation algorithm using multiple distances and a Contrast Transfer Function.
    Refs:
        * Equation 14 in: Langer, M., Cloetens, P., Guigay, J.-P. & Peyrin, F.
          Quantitative comparison of direct phase retrieval algorithms in in-line phase tomography.
          Medical Physics 35, 4556–4566 (2008).

        * Zabler, S., Cloetens, P., Guigay, J.-P., Baruchel, J. & Schlenker, M.:
          Optimization of phase contrast imaging using hard x rays,
          Review of Scientific Instruments 76, 073705 (2005).

    If delta_beta is given the homogeneous CTF approach is used instead. Ref:
        * Eq. 16 (or 22) in Yu et al., Opt. Express 26, 11110 (2018).

    This operator uses the observed intensity to calculate a low-resolution estimate of the object,
    assuming a weak phase object.

    The resulting object projection is stored in the first object mode. If the object is defined with multiple modes,
    secondary ones are set to zero.

    Applies only to the current stack.
    """

    def __init__(self, alpha, alpha_low, cu_iobs_empty, delta_beta=None, sigma=0.01):
        """

        :param alpha: regularisation factor to avoid divergence.
            When delta_beta is used the divergence does not occur at low frequencies, so
            a low alpha value can be used, with a higher alpha_high.
        :param alpha_low: if delta/beta is given, then the regularisation factor goes from
            alpha_low in the low frequencies region to alpha, the cutoff corresponding to
            the spatial frequency k=1/sqrt(lambda*z), with an erfc curve transition.
        :param delta_beta: the delta/beta ration for a CTF assuming an homogeneous materials.
            If None or <=0, the ratio is not used.
        :param cu_iobs_empty: the GPUarray with the empty beam image used for normalisation
        :param sigma: parameter to change the width of the erfc transition between
            alpha_low and alpha_high
        """
        super(BackPropagateCTF1, self).__init__()
        self.alpha = np.float32(alpha)
        self.alpha_low = np.float32(alpha_low)
        self.cu_iobs_empty = cu_iobs_empty
        self.delta_beta = delta_beta
        if self.delta_beta is not None:
            if self.delta_beta <= 0:
                self.delta_beta = None
        self.sigma = np.float32(sigma)

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        s = p._cu_stack
        i0 = s.iproj
        nb_obj = np.int32(p.nb_obj)
        if p.probe_mode_coeff is None:
            nb_probe = np.int32(p.nb_probe)
        else:
            nb_probe = np.int32(1)
        nb_mode = np.int32(p.nb_obj * nb_probe)
        nx = np.int32(p.data.nx)
        ny = np.int32(p.data.ny)
        nz = np.int32(p.data.nz)
        px = np.float32(p.data.pixel_size_detector)
        nb = np.int32(s.nb)
        padding = np.int32(p.data.padding)
        nb_proj = np.int32(s.nb)

        # 0 copy iobs into psi for FT. The padded areas and masked pixels are interpolated
        pu.cu_iobs2psi(s.iobs[:nb], self.cu_iobs_empty, s.psi, p._cu_dx[i0:], p._cu_dy[i0:],
                       nb_mode, nx, ny, nz, padding, stream=pu.cu_stream)

        # Subtract the mean of the computed I/I0 (in the literature formulas, this
        # is done by subtracting a Dirac peak in Fourier space, but this is easier
        # as it removes the need to correctly scale the Dirac peak.
        for iproj in range(nb):
            for iz in range(nz):
                psi_sum = cua.sum(s.psi[iproj, iz, 0, 0], stream=pu.cu_stream)
                pu.cu_subtract_mean(s.psi[iproj, iz, 0, 0], psi_sum, np.float32(nx * ny), stream=pu.cu_stream)

        # 2 FT normalised observed intensity
        p = FT1(scale=True) * p

        # 3 CTF operator in Fourier space
        pilambdad = np.array(p.data.detector_distance * p.data.wavelength * np.pi, dtype=np.float32)
        pilambdad = cua.to_gpu_async(pilambdad, allocator=pu.cu_mem_pool.allocate, stream=pu.cu_stream)
        if self.delta_beta is None:
            if s.istack == 0:
                print("Using CTF")
            pu.cu_ctf_fourier(s.psi[0, 0, 0, 0], pilambdad, px, nb_mode, nx, ny, nz, nb_proj,
                              self.alpha, stream=pu.cu_stream)
        else:
            if s.istack == 0:
                print("Using CTF with an homogeneous constraint")
            pu.cu_ctf_fourier_homogeneous(s.psi[0, 0, 0, 0], pilambdad, np.float32(self.delta_beta),
                                          px, nb_mode, nx, ny, nz, nb_proj, self.alpha_low, self.alpha,
                                          self.sigma, stream=pu.cu_stream)

        # 3 Back-propagate phase and compute weak phase object
        p = IFT1(scale=True) * p

        delta_beta = self.delta_beta
        if delta_beta is None:
            delta_beta = np.float32(0)

        pu.cu_ctf_phase2obj(s.psi[0, 0, 0, 0], s.obj, s.obj_phase0,
                            nb_probe, nb_obj, nx, ny, nz, nb_proj, delta_beta, stream=pu.cu_stream)

        # Copy back original object phase as it is not done in SwapStack
        # This could be removed, we assume a weak object, obj_phase0 is only used for unwrapping
        sout = p.data.stack_v[s.istack]
        cu_drv.memcpy_dtoh_async(src=s.obj_phase0.gpudata, dest=sout.obj_phase0, stream=pu.cu_stream)

        return p


class BackPropagateCTF(CUOperatorHoloTomo):
    """
    Back-propagation algorithm using multiple distances and a Contrast Transfer Function.
    Refs:
        * Equation 14 in: Langer, M., Cloetens, P., Guigay, J.-P. & Peyrin, F.
          Quantitative comparison of direct phase retrieval algorithms in in-line phase tomography.
          Medical Physics 35, 4556–4566 (2008).

        * Zabler, S., Cloetens, P., Guigay, J.-P., Baruchel, J. & Schlenker, M.:
          Optimization of phase contrast imaging using hard x rays,
          Review of Scientific Instruments 76, 073705 (2005).

    If delta_beta is given the homogeneous CTF approach is used instead. Ref:
        * Eq. 16 (or 22) in Yu et al., Opt. Express 26, 11110 (2018).

    (the two approaches are identical for infinite delta/beta, a pure phase object)

    This operator uses the observed intensity to calculate a low-resolution estimate of the object,
    assuming a weak phase object.

    The resulting object projection is stored in the first object mode. If the object is defined with multiple modes,
    secondary ones are set to zero.

    This also copies the square root of the empty beam intensity into the probe,
    interpolating the padded areas and masked pixels.

    The observed intensity is interpolated for masked and padded areas, and stored as -1-I_interp
    (so the values remain masked), except for the empty beam images for which the values
    are stored as 'observed' values to help convergence of object and probe.
    """

    def __init__(self, alpha=0.2, alpha_low=1e-5, delta_beta=None, init_probe_modes=False):
        """

        :param alpha: regularisation factor to avoid divergence.
            When delta_beta is used the divergence does not occur at low frequencies, so
            a small alpha_low value can be used, with a higher alpha.
        :param alpha_low: if delta/beta is given, then the regularisation factor goes from
            alpha_low in the low frequencies region to alpha, the cutoff corresponding to
            the spatial frequency k=1/sqrt(lambda*z), with an erfc curve transition.
        :param delta_beta: the delta/beta ration for a CTF assuming an homogeneous materials.
            Ignored if None of <=0
        :param init_probe_modes: if True, init the coherent probe modes coefficients to 1/nb_probe.
            If False, the current values are kept.
        """
        super(BackPropagateCTF, self).__init__()
        self.alpha = alpha
        self.alpha_low = alpha_low
        self.delta_beta = delta_beta
        self.init_probe_modes = init_probe_modes

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        # Find the empty beam image, move it to GPU to use it as normalisation (on CPU)
        nb_empty = 0
        iobs_empty = np.zeros((p.data.nz, p.data.ny, p.data.nx), dtype=np.float32)

        for s in p.data.stack_v:
            for iproj in range(s.nb):
                i = s.iproj + iproj
                if bool(p.data.sample_flag[i]) is False:
                    nb_empty += 1
                    iobs_empty += s.iobs[iproj]
        if nb_empty > 0:
            iobs_empty /= nb_empty
        else:
            iobs_empty[:] = 1

        cu_iobs_empty = cua.to_gpu_async(iobs_empty, allocator=self.processing_unit.cu_mem_pool.allocate,
                                         stream=pu.cu_stream)

        pu = self.processing_unit
        nx = np.int32(p.data.nx)
        ny = np.int32(p.data.ny)
        nz = np.int32(p.data.nz)
        nb_probe = np.int32(p.nb_probe)
        padding = np.int32(p.data.padding)

        # Put iobs_empty in probe
        pu.cu_iobs_empty2probe(cu_iobs_empty, p._cu_probe, nb_probe, nx, ny, nz, padding, stream=pu.cu_stream)

        if p.probe_mode_coeff is not None and self.init_probe_modes:
            # Should fill random values instead ?
            p._cu_probe_mode_coeff.fill(np.float32(1 / nb_probe), stream=pu.cu_stream)

        p = LoopStack(op=BackPropagateCTF1(alpha=self.alpha, alpha_low=self.alpha_low,
                                           cu_iobs_empty=cu_iobs_empty, delta_beta=self.delta_beta),
                      out=True, copy_psi=True, verbose=True) * p

        return p


# class OrthoProbe(CUOperatorHoloTomo):
#     """
#     Operator to orthonormalise the probe modes
#     """
#
#     def op(self, p: HoloTomo):
#         pu = self.processing_unit
#         n = p.nb_probe
#         kn = pu.get_modes_kernels(n)
#
#         for iz in range(p.data.nz):
#             if n > 1:
#                 m = kn["vdot"](p._cu_probe[iz, 0], p._cu_probe[0, 0].size)
#
#                 # We need a complex NxN array for eig, but m has a special vector type,
#                 # and mcu.view() won't work, so do the conversion manually. The 'F' order is for eig()
#                 # Specifying 'base' should do the reference counting to avoid deleting the array
#                 m = cua.GPUArray((n, n), dtype=np.complex64, gpudata=m.gpudata, base=m, order='F')
#                 v, e = cu_linalg.eig(m, 'N', 'V')
#             else:
#                 v = cua.to_gpu_async(np.ones(1, dtype=np.complex64), allocator=pu.cu_mem_pool.allocate,
#                                      stream=pu.cu_stream)
#
#             # Compute the orthonormal modes
#             # modes = np.array([sum(m[i] * v[i, j] for i in range(len(m))) for j in range(len(m))])
#             norm = kn["dot_red"](p._cu_probe[iz, 0], v, p._cu_probe[0, 0].size)
#             norm = cua.GPUArray(n, dtype=np.float32, gpudata=norm.gpudata, base=norm)
#             kn["ortho_norm"](p._cu_probe[iz, 0], norm, p._cu_probe[0, 0].size)
#         return p


class SwapStack(CUOperatorHoloTomo):
    """
    Operator to swap a stack of projections to or from GPU. Note that once this operation has been applied,
    the new Psi value may be undefined (empty array), if no previous array is copied in.
    Using this operator will automatically move the host stack arrays to pinned memory.
    """

    def __init__(self, i=None, next_i=None, out=False, copy_psi=False, verbose=False):
        """
        Select a new stack of frames, swapping data between the host and the GPU. This is done using a set of
        three buffers and three queues, used to perform in parallel 1) the GPU computing, 2) copying data to the GPU
        and 3) copying data from the GPU. High speed can only be achieved if host memory is page-locked (pinned).
        Note that the buffers used for processing and copying are swapped when using this operator.
        The buffers copied are: object(in/out), iobs (in), dx (in), dy(in), sample_flag(in),
        and optionally psi(in/out).

        :param i: the new stack to use. If it is not yet swapped in (in the current 'in' buffers), it is copied
                  to the GPU.
        :param next_i: the following stack to use, for which the copy to the GPU will be initiated in the 'in' queue.
        :param out: if True (the default) and if the HoloTomo object _cu_timestamp_counter > _timestamp_counter, the
                    data from the current stack in memory will be copied back using the 'out' queue.
        :param copy_psi: if True, also copy psi arrays if they are available. If False (the default), the psi arrays
                         are not copied and the corresponding GPU buffers are released.
        :param verbose: if True, print some information when used.
        """
        super(SwapStack, self).__init__()
        self.i = i
        self.next_i = next_i
        self.out = out
        self.copy_psi = copy_psi
        self.verbose = verbose

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        scu = p._cu_stack
        sswap = p._cu_stack_swap

        # Make sure tasks are finished in each stream before beginning a new one.
        # Use events so that the wait is done asynchronously on the GPU
        pu.cu_event_calc.record(pu.cu_stream)
        pu.cu_event_swap.record(pu.cu_stream_swap)
        pu.cu_stream.wait_for_event(pu.cu_event_swap)  # Data must have arrived before being processed
        pu.cu_stream_swap.wait_for_event(pu.cu_event_calc)  # Calc must be finished before swapping out

        # Note: *if* we run out of pinned memory, we could use PageLockedMemoryPool() to also swap
        # stacks of data to/from page-locked memory. Of course this would require another
        # parallel process working on the host to move data to/from the page-locked parts... And that one
        # would also need to be asynchronous...

        self.i %= len(p.data.stack_v)  # useful when looping and i + 1 == stack_size
        if self.next_i is not None:
            self.next_i %= len(p.data.stack_v)

        if scu.istack is None:  # This can happen once at the beginning
            self.out = False

        sout = None
        if self.out:
            sout = p.data.stack_v[scu.istack]
            SwapStack._to_pinned_memory(sout, p)

        if self.i != sswap.istack:
            # desired stack is not pre-loaded, so need to swap in data in main stream
            sin = p.data.stack_v[self.i]
            SwapStack._to_pinned_memory(sin, p)
            stream = pu.cu_stream
            if sout is not None:
                cu_drv.memcpy_dtoh_async(src=scu.obj.gpudata, dest=sout.obj, stream=stream)
                # TODO: No need to copy obj_phase0 to host, this should be constant, except during Paganin ?
                # cu_drv.memcpy_dtoh_async(src=sswap.obj_phase0.gpudata, dest=sout.obj_phase0, stream=stream)
            cu_drv.memcpy_htod_async(dest=scu.obj.gpudata, src=sin.obj, stream=stream)
            cu_drv.memcpy_htod_async(dest=scu.obj_phase0.gpudata, src=sin.obj_phase0, stream=stream)
            cu_drv.memcpy_htod_async(dest=scu.iobs.gpudata, src=sin.iobs, stream=stream)
            if self.copy_psi:
                if sout is not None:
                    cu_drv.memcpy_dtoh_async(src=scu.psi.gpudata, dest=sout.psi, stream=stream)
                cu_drv.memcpy_htod_async(dest=scu.psi.gpudata, src=sin.psi, stream=stream)
            scu.istack = self.i
            scu.iproj = sin.iproj
            scu.nb = sin.nb
            sout = None  # Copy out has been done
        else:
            # Desired stack is pre-loaded in sswap
            # Swap stacks so calculations can continue in main stack while transfers occur in p._cu_stack_swap==scu
            p._cu_stack_swap, p._cu_stack = p._cu_stack, p._cu_stack_swap
            sswap = p._cu_stack_swap

        # Desired stack is in p._cu_stack, take care of next one and out if needed, using swap stream
        stream = pu.cu_stream_swap
        sin = None
        if self.next_i is not None:
            sin = p.data.stack_v[self.next_i]
            SwapStack._to_pinned_memory(sin, p)
            sswap.istack = self.next_i
            sswap.iproj = sin.iproj
            sswap.nb = sin.nb

        # We copy object first and record an event for this - this may be used by algorithms which need the
        # next stack of objects for regularisation
        if sout is not None:
            cu_drv.memcpy_dtoh_async(src=sswap.obj.gpudata, dest=sout.obj, stream=stream)
            cu_drv.memcpy_dtoh_async(src=sswap.obj_phase0.gpudata, dest=sout.obj_phase0, stream=stream)
        if sin is not None:
            cu_drv.memcpy_htod_async(dest=sswap.obj.gpudata, src=sin.obj, stream=stream)
            cu_drv.memcpy_htod_async(dest=sswap.obj_phase0.gpudata, src=sin.obj_phase0, stream=stream)
        pu.cu_event_swap_obj.record(stream)

        # No need to copy iobs to host, this is a constant (could we ue Texture memory ?)
        if sin is not None:
            cu_drv.memcpy_htod_async(dest=sswap.iobs.gpudata, src=sin.iobs, stream=stream)
        if self.copy_psi:
            if sout is not None:
                cu_drv.memcpy_dtoh_async(src=sswap.psi.gpudata, dest=sout.psi, stream=stream)
            if sin is not None:
                cu_drv.memcpy_htod_async(dest=sswap.psi.gpudata, src=sin.psi, stream=stream)
        return p

    @staticmethod
    def _to_pinned_memory(s: HoloTomoDataStack, p: HoloTomo):
        """
        Move a given stack to pinned (pagelocked) memory, if necessary.
        :param s: the HoloTomoDataStack to be moved to pinned memory
        :return: nothing
        """
        if not s.pinned_memory:
            print("Pinning memory for stack #%2d" % s.istack)
            for o in dir(s):
                if isinstance(s.__getattribute__(o), np.ndarray):
                    old = s.__getattribute__(o)
                    # Would using the WRITECOMBINED flag be useful ? No
                    s.__setattr__(o, cu_drv.pagelocked_empty_like(old))
                    s.__getattribute__(o)[:] = old
            if p.probe_mode_coeff is None:
                psi_shape = (p.data.stack_size, p.data.nz, p.nb_obj, p.nb_probe, p.data.ny, p.data.nx)
            else:
                psi_shape = (p.data.stack_size, p.data.nz, p.nb_obj, 1, p.data.ny, p.data.nx)
            if s.psi is not None:
                # Allow to change the number of probe modes
                if s.psi.shape != psi_shape:
                    s.psi = None
            if s.psi is None:
                # Psi has not yet been initialised
                s.psi = cu_drv.pagelocked_empty(psi_shape, np.complex64)
            s.pinned_memory = True


class LoopStack(CUOperatorHoloTomo):
    """
    Loop operator to apply a given operator sequentially to the complete stack of projections of a HoloTomo object.
    This operator will take care of transferring data between CPU and GPU
    """

    def __init__(self, op, out=True, copy_psi=False, verbose=False):
        """

        :param op: the operator to apply, which can be a multiplication of operators
        :param out: if True (the default) and if the HoloTomo object _cu_timestamp_counter > _timestamp_counter, the
                    data from the current stack in memory will be copied back using the 'out' queue.
        :param copy_psi: if True, when switching between stacks, also keep psi.
        :param verbose: if True, print some information when used.
        """
        super(LoopStack, self).__init__()
        self.stack_op = op
        self.out = out
        self.copy_psi = copy_psi
        self.verbose = verbose

    def op(self, p: HoloTomo):
        if len(p.data.stack_v) == 1:
            return self.stack_op * p
        else:
            for i in range(len(p.data.stack_v)):
                p = self.stack_op * SwapStack(i, next_i=(i + 1) % len(p.data.stack_v), out=self.out,
                                              copy_psi=self.copy_psi, verbose=self.verbose) * p

        return p


class TestParallelFFT(CUOperatorHoloTomo):
    """
    Test speed on a multi-view dataset, by transferring data to/from the GPU in parallel to the FFT execution using
    concurrent queues.
    """

    def __init__(self, n_iter=5, n_stack=5, n_fft=1, psi_shape=None):
        super(TestParallelFFT, self).__init__()
        self.n_iter = n_iter
        self.n_stack = n_stack
        self.n_fft = n_fft
        self.psi_shape = psi_shape

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        if self.psi_shape is None:
            psi_shape = p._psi.shape
        else:
            psi_shape = self.psi_shape
        stack_size, nz, n_obj, nb_probe, ny, nx = psi_shape
        # Create data with pinned memory, random data to avoid any smart optimisation
        vpsi = []
        for j in range(self.n_stack):
            vpsi.append(cu_drv.pagelocked_empty(psi_shape, np.complex64))
            vpsi[-1][:] = np.random.uniform(0, 1, psi_shape)
        # Allocate 3 arrays in GPU
        cu_psi = cua.to_gpu(vpsi[0])
        cu_psi_in = cua.to_gpu(vpsi[1])
        cu_psi_out = cua.to_gpu(vpsi[2])

        # First test fft on array remaining in GPU
        pu.finish()
        t0 = timeit.default_timer()
        for i in range(self.n_iter * self.n_stack):
            for k in range(self.n_fft):
                pu.fft(cu_psi, cu_psi, ndim=2, stream=pu.cu_stream)
                pu.ifft(cu_psi, cu_psi, ndim=2, stream=pu.cu_stream)

        pu.finish()
        dt0 = timeit.default_timer() - t0
        # This measures the number of Gbyte/s for which the n_fft FFT are calculated
        gbytes = cu_psi.nbytes * 2 * self.n_iter * self.n_stack * self.n_fft * 2 * 2 / dt0 / 1024 ** 3

        print("Time for     on-GPU %d FFT of size %dx%dx%dx%dx%dx%d: %6.3fs [%8.2f Gbyte/s] [vkfft=%d]" %
              (self.n_iter * self.n_stack * self.n_fft, stack_size, nz, n_obj, nb_probe,
               ny, nx, dt0, gbytes, pu.use_vkfft))

        # test fft on array transferred sequentially to/from GPU
        pu.finish()
        t0 = timeit.default_timer()
        for i in range(self.n_iter):
            for j in range(self.n_stack):
                cu_drv.memcpy_htod_async(dest=cu_psi.gpudata, src=vpsi[j], stream=pu.cu_stream)
                for k in range(self.n_fft):
                    pu.fft(cu_psi, cu_psi, ndim=2, stream=pu.cu_stream)
                    pu.ifft(cu_psi, cu_psi, ndim=2, stream=pu.cu_stream)
                cu_drv.memcpy_dtoh_async(src=cu_psi.gpudata, dest=vpsi[j], stream=pu.cu_stream)
        pu.cu_ctx.synchronize()
        dt1 = timeit.default_timer() - t0
        gbytes = cu_psi.nbytes * 2 * self.n_iter * self.n_stack * self.n_fft * 2 * 2 / dt1 / 1024 ** 3
        print("Time for    i/o GPU %d FFT of size %dx%dx%dx%dx%dx%d: %6.3fs [%8.2f Gbyte/s] [vkfft=%d]" %
              (self.n_iter * self.n_stack * self.n_fft, stack_size, nz, n_obj, nb_probe,
               ny, nx, dt1, gbytes, pu.use_vkfft))

        # Now perform FFT while transferring in // data to and from GPU with three queues
        pu.finish()
        t0 = timeit.default_timer()
        ev_calc = cu_drv.Event(cu_drv.event_flags.DISABLE_TIMING)
        ev_swap = cu_drv.Event(cu_drv.event_flags.DISABLE_TIMING)
        for i in range(self.n_iter):
            for j in range(self.n_stack):
                ev_swap.record(pu.cu_stream_swap)
                ev_calc.record(pu.cu_stream)
                pu.cu_stream_swap.wait_for_event(ev_calc)
                pu.cu_stream.wait_for_event(ev_swap)

                cu_drv.memcpy_htod_async(dest=cu_psi_in.gpudata, src=vpsi[(j + 1) % self.n_stack],
                                         stream=pu.cu_stream_swap)
                cu_drv.memcpy_dtoh_async(src=cu_psi_out.gpudata, dest=vpsi[(j - 1) % self.n_stack],
                                         stream=pu.cu_stream_swap)

                for k in range(self.n_fft):
                    pu.fft(cu_psi, cu_psi, ndim=2, stream=pu.cu_stream)
                    pu.ifft(cu_psi, cu_psi, ndim=2, stream=pu.cu_stream)
                # Swap stacks
                cu_psi_in, cu_psi, cu_psi_out = cu_psi_out, cu_psi_in, cu_psi
        pu.finish()
        dt2 = timeit.default_timer() - t0
        gbytes = cu_psi.nbytes * 2 * self.n_iter * self.n_stack * self.n_fft * 2 * 2 / dt2 / 1024 ** 3
        print("Time for // i/o GPU %d FFT of size %dx%dx%dx%dx%dx%d: %6.3fs [%8.2f Gbyte/s] [vkfft=%d]" %
              (self.n_iter * self.n_stack * self.n_fft, stack_size, nz, n_obj, nb_probe,
               ny, nx, dt2, gbytes, pu.use_vkfft))

        return p
