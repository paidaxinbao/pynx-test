# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import os
import psutil
import types
import gc
import numpy as np
import pycuda.driver as cu_drv
import pycuda.gpuarray as cua
from pycuda.elementwise import ElementwiseKernel as CU_ElK
from pycuda.reduction import ReductionKernel as CU_RedK
from pycuda.compiler import SourceModule
import pycuda.tools as cu_tools

from ..processing_unit import default_processing_unit as main_default_processing_unit
from ..processing_unit.cu_processing_unit import CUProcessingUnit
from ..processing_unit.kernel_source import get_kernel_source as getks
from ..operator import has_attr_not_none, OperatorSum, OperatorPower, OperatorException
from .ptycho import Ptycho, OperatorPtycho, algo_string
from ..mpi import MPI

if MPI is not None:
    from .mpi.operator import ShowObjProbe
else:
    from .cpu_operator import ShowObjProbe
from .shape import get_view_coord

my_float4 = cu_tools.get_or_register_dtype("my_float4",
                                           np.dtype([('a', '<f4'), ('b', '<f4'), ('c', '<f4'), ('d', '<f4')]))


################################################################################################
# Patch Ptycho class so that we can use 5*w to scale it.
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


patch_method(Ptycho)


################################################################################################


class CUProcessingUnitPtycho(CUProcessingUnit):
    """
    Processing unit in CUDA space, for 2D Ptycho operations.

    Handles initializing the context and kernels.
    """

    def __init__(self):
        super(CUProcessingUnitPtycho, self).__init__()
        # Size of the stack size used on the GPU - can be any integer, optimal values between 10 to 30
        # Should be chosen smaller for large frame sizes.
        self.cu_stack_size = np.int32(16)

    def set_stack_size(self, s):
        """
        Change the number of frames which are stacked to perform all operations in //. If it
        is smaller than the total number of frames, operators like AP, DM, ML will loop over
        all the stacks.
        For CUDA operator which take advantage of atomic operations, it is advised to have a single stack.
        :param s: an integer number (default=16)
        :return: nothing
        """
        self.cu_stack_size = np.int32(s)

    def get_stack_size(self):
        return self.cu_stack_size

    def cu_init_kernels(self):
        """
        Initialize cuda kernels
        :return: nothing
        """
        # TODO: delay initialization, on-demand for each type of operator ?

        # Elementwise kernels
        self.cu_scale = CU_ElK(name='cu_scale',
                               operation="d[i] = complexf(d[i].real() * scale, d[i].imag() * scale )",
                               preamble=getks('cuda/complex.cu'),
                               options=self.cu_options, arguments="pycuda::complex<float> *d, const float scale")

        self.cu_scalef_mem_pos = CU_ElK(name='cu_scalef_mem',
                                        operation="d[i] = fmaxf(d[i] * scale[0], 0.0f)",
                                        options=self.cu_options, arguments="float *d, const float* scale")

        # Gauss convolution kernel in Fourier space
        self.cu_gauss_ftconv = CU_ElK(name='cu_gauss_ftconv',
                                      operation="const int ix = i % (nxy/2+1);"
                                                "int iy = i / (nxy/2+1); iy -= nxy * (iy >= nxy/2);"
                                                "const float v = -sigmaf*(ix*ix + iy*iy);"
                                                "dest[i] *= v > -50 ? expf(v): 1.9287498479639178e-22f ;",
                                      options=self.cu_options,
                                      preamble=getks('cuda/complex.cu'),
                                      arguments="pycuda::complex<float> *dest, const float sigmaf, const int nxy")

        self.cu_sum = CU_ElK(name='cu_sum',
                             operation="dest[i] += src[i]",
                             preamble=getks('cuda/complex.cu'),
                             options=self.cu_options,
                             arguments="pycuda::complex<float> *src, pycuda::complex<float> *dest")

        self.cu_sum_icalc = CU_ElK(name='cu_sum_icalc',
                                   operation="SumIcalc(i, icalc_sum, d, nxy, nz)",
                                   preamble=getks('cuda/complex.cu') + getks('ptycho/cuda/sum_intensity_elw.cu'),
                                   options=self.cu_options,
                                   arguments="float* icalc_sum, pycuda::complex<float> *d, const int nxy, const int nz")

        self.cu_sum_iobs = CU_ElK(name='cu_sum_iobs',
                                  operation="SumIobs(i, iobs_sum, obs, calc, nxy, nz, nb_mode)",
                                  preamble=getks('cuda/complex.cu') + getks('ptycho/cuda/sum_intensity_elw.cu'),
                                  options=self.cu_options,
                                  arguments="float* iobs_sum, float* obs, pycuda::complex<float> *calc,"
                                            "const int nxy, const int nz, const int nb_mode")

        self.cu_scale_complex = CU_ElK(name='cu_scale_complex',
                                       operation="d[i] = complexf(d[i].real() * s.real() - d[i].imag() * s.imag(), d[i].real() * s.imag() + d[i].imag() * s.real())",
                                       preamble=getks('cuda/complex.cu'),
                                       options=self.cu_options,
                                       arguments="pycuda::complex<float> *d, const pycuda::complex<float> s")

        self.cu_quad_phase = CU_ElK(name='cu_quad_phase',
                                    operation="QuadPhase(i, d, f, scale, nx, ny)",
                                    preamble=getks('cuda/complex.cu') + getks('ptycho/cuda/quad_phase_elw.cu'),
                                    options=self.cu_options,
                                    arguments="pycuda::complex<float> *d, const float f, const float scale, const int nx, const int ny")

        # Linear combination with 2 complex arrays and 2 float coefficients
        self.cu_linear_comb_fcfc = CU_ElK(name='cu_linear_comb_fcfc',
                                          operation="dest[i] = complexf(a * dest[i].real() + b * src[i].real(),"
                                                    "a * dest[i].imag() + b * src[i].imag())",
                                          options=self.cu_options,
                                          preamble=getks('cuda/complex.cu'),
                                          arguments="pycuda::complex<float> *dest, const float a,"
                                                    "pycuda::complex<float> *src, const float b")

        # Linear combination with 2 complex arrays and 2 float coefficients, for CG beta
        # If beta is NaN => beta=0
        self.cu_linear_comb_fcfc_beta = \
            CU_ElK(name='cu_linear_comb_fcfc_beta',
                   operation="float a = beta[0].real() / beta[0].imag();"
                             "if( !isfinite(a) || (a < 0.0f)) a=0.0f;"
                             "dest[i] = complexf(a * dest[i].real() + b * src[i].real(),"
                             "a * dest[i].imag() + b * src[i].imag())",
                   options=self.cu_options, preamble=getks('cuda/complex.cu'),
                   arguments="pycuda::complex<float> *dest, pycuda::complex<float> *beta, "
                             "pycuda::complex<float> *src, const float b")

        # Linear combination with 2 complex arrays and 2 float coefficients, for CG gamma
        # If gamma is NaN, the gradient is instead copied to the search direction
        self.cu_linear_comb_fcfc_gamma = \
            CU_ElK(name='cu_linear_comb_fcfc_gamma',
                   operation="float b = gamma[0].real()/gamma[0].imag();"
                             "if(isfinite(b)) dest[i] = complexf(a * dest[i].real() + b * src[i].real(),"
                             "a * dest[i].imag() + b * src[i].imag());"
                             "else src[i]=grad[i];",
                   options=self.cu_options, preamble=getks('cuda/complex.cu'),
                   arguments="pycuda::complex<float> *dest, const float a, "
                             "pycuda::complex<float> *src, pycuda::complex<float> *gamma,"
                             "pycuda::complex<float> *grad")

        # Linear combination with 2 float arrays and 2 float coefficients
        self.cu_linear_comb_4f = CU_ElK(name='cu_linear_comb_4f_beta',
                                        operation="dest[i] = a * dest[i] + b * src[i]",
                                        options=self.cu_options,
                                        preamble=getks('cuda/complex.cu'),
                                        arguments="float *dest, const float a, float *src, const float b")

        # Linear combination with 2 float arrays and 2 float coefficients, for CG beta
        # If beta is NaN => beta=0
        self.cu_linear_comb_4f_beta = \
            CU_ElK(name='cu_linear_comb_4f_beta',
                   operation="float a = fmaxf(0, beta[0].real()/fmaxf(beta[0].imag(), 1e-20f));"
                             "if( !isfinite(a) || a<0.0f) a=0.0f;"
                             "dest[i] = a * dest[i] + b * src[i]",
                   options=self.cu_options, preamble=getks('cuda/complex.cu'),
                   arguments=" float *dest, pycuda::complex<float> *beta, float *src, const float b")

        # Linear combination with 2 float arrays and 2 float coefficients, for CG gamma
        # If gamma is NaN, the gradient is instead copied to the search direction
        self.cu_linear_comb_4f_gamma = \
            CU_ElK(name='cu_linear_comb_4f_gamma',
                   operation="float b = gamma[0].real()/gamma[0].imag();"
                             "if(isfinite(b)) dest[i] = a * dest[i] + b * src[i]; "
                             "else src[i]=grad[i];",
                   options=self.cu_options, preamble=getks('cuda/complex.cu'),
                   arguments="float *dest, const float a, float *src,"
                             "pycuda::complex<float> *gamma, float *grad")

        # Linear combination with 2 float arrays and 2 float coefficients, for CG gamma
        # If gamma is NaN, the gradient is instead copied to the search direction
        # final value must be >=0 (for background)
        self.cu_linear_comb_4f_gamma_pos = \
            CU_ElK(name='cu_linear_comb_4f_gamma',
                   operation="float b = gamma[0].real()/gamma[0].imag();"
                             "if(isfinite(b)) dest[i] = fmaxf(a * dest[i] + b * src[i], 0.0f); "
                             "else src[i]=grad[i];",
                   options=self.cu_options, preamble=getks('cuda/complex.cu'),
                   arguments="float *dest, const float a, float *src,"
                             "pycuda::complex<float> *gamma, float *grad")

        self.cu_projection_amplitude = CU_ElK(name='cu_projection_amplitude',
                                              operation="ProjectionAmplitude(i, iobs, dcalc, background,"
                                                        "nbmode, nxy, nxystack, npsi, scale_in, scale_out)",
                                              preamble=getks('cuda/complex.cu') + getks(
                                                  'ptycho/cuda/projection_amplitude_elw.cu'),
                                              options=self.cu_options,
                                              arguments="float *iobs, pycuda::complex<float> *dcalc, float *background,"
                                                        "const int nbmode, const int nxy,"
                                                        "const int nxystack, const int npsi, const float scale_in,"
                                                        "const float scale_out")

        self.cu_projection_amplitude_background = \
            CU_ElK(name='cu_projection_amplitude_background',
                   operation="ProjectionAmplitudeBackground(i, iobs, dcalc, background,"
                             "vd, vd2, vz2, vdz2, nbmode, nxy, nxystack, npsi, first_pass, scale_in, scale_out)",
                   preamble=getks('cuda/complex.cu') +
                            getks('ptycho/cuda/projection_amplitude_elw.cu'),
                   options=self.cu_options,
                   arguments="float *iobs, pycuda::complex<float> *dcalc, float *background,"
                             "float *vd, float *vd2, float *vz2, float *vdz2, const int nbmode,"
                             "const int nxy, const int nxystack, const int npsi,"
                             "const char first_pass, const float scale_in, const float scale_out")

        self.cu_projection_amplitude_background_mode = \
            CU_ElK(name='cu_projection_amplitude_background_mode',
                   operation="ProjectionAmplitudeBackgroundMode(i, iobs, dcalc, background,"
                             "background_new, nbmode, nxy, nxystack, npsi, first_pass, scale_in, scale_out)",
                   preamble=getks('cuda/complex.cu') +
                            getks('ptycho/cuda/projection_amplitude_elw.cu'),
                   options=self.cu_options,
                   arguments="float *iobs, pycuda::complex<float> *dcalc, float *background,"
                             "float *background_new, const int nbmode,"
                             "const int nxy, const int nxystack, const int npsi,"
                             "const char first_pass, const float scale_in, const float scale_out")

        self.cu_background_update = \
            CU_ElK(name='cu_background_update',
                   operation="const float eta = fmaxf(0.8f, vdz2[i]/vd2[i]);"
                             "background[i] = fmaxf(0.0f, background[i] + (vd[i] - vz2[i] / eta) / nframes);",
                   options=self.cu_options,
                   arguments="float* background, float* vd, float* vd2, float* vz2, float* vdz2, const int nframes")

        self.cu_background_update_mode = \
            CU_ElK(name='cu_background_update_mode',
                   operation="background[i] = background_new[i] / nframes;",
                   options=self.cu_options,
                   arguments="float* background, float* background_new, const int nframes")

        self.cu_calc2obs = CU_ElK(name='cu_calc2obs',
                                  operation="Calc2Obs(i, iobs, dcalc, background, nbmode, nxy, nxystack)",
                                  preamble=getks('cuda/complex.cu') + getks('ptycho/cuda/calc2obs_elw.cu'),
                                  options=self.cu_options,
                                  arguments="float *iobs, pycuda::complex<float> *dcalc, float *background,"
                                            "const int nbmode, const int nxy, const int nxystack")

        self.cu_object_probe_mult = CU_ElK(name='cu_object_probe_mult',
                                           operation="ObjectProbeMultQuadPhase(i, psi, obj, probe, cx, cy, pixel_size,"
                                                     "f, npsi, stack_size, nx, ny, nxo, nyo, nbobj, nbprobe, interp)",
                                           preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                    getks('ptycho/cuda/obj_probe_mult_elw.cu'),
                                           options=self.cu_options,
                                           arguments="pycuda::complex<float>* psi, pycuda::complex<float> *obj, "
                                                     "pycuda::complex<float>* probe, float* cx, float* cy,"
                                                     "const float pixel_size, const float f, const int npsi,"
                                                     "const int stack_size, const int nx, const int ny, const int nxo,"
                                                     "const int nyo, const int nbobj, const int nbprobe,"
                                                     "const bool interp")

        self.cu_calc_illum = CU_ElK(name='cu_calc_illum',
                                    operation="CalcIllumination(i, probe, obj_illum, cx, cy,"
                                              "npsi, stack_size, nx, ny, nxo, nyo, nbprobe, interp, padding)",
                                    preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                             getks('ptycho/cuda/calc_illumination_elw.cu'),
                                    options=self.cu_options,
                                    arguments="pycuda::complex<float>* probe, float* obj_illum,"
                                              "float* cx, float* cy, const int npsi,"
                                              "const int stack_size, const int nx, const int ny, const int nxo,"
                                              "const int nyo, const int nbprobe, const bool interp, const int padding")

        self.cu_2object_probe_psi_dm1 = CU_ElK(name='cu_2object_probe_psi_dm1',
                                               operation="ObjectProbePsiDM1(i, psi, obj, probe, cx, cy, pixel_size, f,"
                                                         "npsi, stack_size, nx, ny, nxo, nyo, nbobj, nbprobe, interp)",
                                               preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                        getks('ptycho/cuda/obj_probe_dm_elw.cu'),
                                               options=self.cu_options,
                                               arguments="pycuda::complex<float>* psi, pycuda::complex<float> *obj,"
                                                         "pycuda::complex<float>* probe, float* cx, float* cy,"
                                                         "const float pixel_size, const float f, const int npsi,"
                                                         "const int stack_size, const int nx, const int ny,"
                                                         "const int nxo, const int nyo, const int nbobj,"
                                                         "const int nbprobe, const bool interp")

        self.cu_2object_probe_psi_dm2 = CU_ElK(name='cu_2object_probe_psi_dm2',
                                               operation="ObjectProbePsiDM2(i, psi, psi_fourier, obj, probe, cx, cy,"
                                                         "pixel_size, f, npsi, stack_size, nx, ny, nxo, nyo, nbobj,"
                                                         "nbprobe, interp)",
                                               preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                        getks('ptycho/cuda/obj_probe_dm_elw.cu'),
                                               options=self.cu_options,
                                               arguments="pycuda::complex<float>* psi,"
                                                         "pycuda::complex<float>* psi_fourier,"
                                                         "pycuda::complex<float> *obj, pycuda::complex<float>* probe,"
                                                         "float* cx, float* cy, const float pixel_size, const float f,"
                                                         "const int npsi, const int stack_size, const int nx,"
                                                         "const int ny, const int nxo, const int nyo, const int nbobj,"
                                                         "const int nbprobe, const bool interp")

        self.cu_psi_to_obj_atomic = CU_ElK(name='psi_to_obj_atomic',
                                           operation="UpdateObjQuadPhaseAtomic(i, psi, objnew, probe, objnorm, cx, cy,"
                                                     "px, f, stack_size, nx, ny, nxo, nyo, nbobj, nbprobe, npsi,"
                                                     "padding, interp)",
                                           preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                    getks('ptycho/cuda/psi_to_obj_probe_elw.cu'),
                                           options=self.cu_options,
                                           arguments="pycuda::complex<float>* psi, pycuda::complex<float> *objnew,"
                                                     "pycuda::complex<float>* probe, float* objnorm, float* cx,"
                                                     "float* cy, const float px, const float f, const int stack_size,"
                                                     "const int nx, const int ny, const int nxo, const int nyo,"
                                                     "const int nbobj, const int nbprobe, const int npsi,"
                                                     "const int padding, const bool interp")

        self.cu_sum_n = CU_ElK(name='sum_n',
                               operation="SumN(i, objnewN, objnormN, stack_size, nxyo, nbobj)",
                               preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                        getks('ptycho/cuda/psi_to_obj_probe_elw.cu'),
                               options=self.cu_options,
                               arguments="pycuda::complex<float>* objnewN, float* objnorm, float* objnormN, const int stack_size, const int nxyo, const int nbobj")

        self.cu_sum_n_norm = CU_ElK(name='sum_n_norm',
                                    operation="SumNnorm(i, objnormN, stack_size, nxyo)",
                                    preamble=getks('cuda/complex.cu') + getks('ptycho/cuda/psi_to_obj_probe_elw.cu'),
                                    options=self.cu_options,
                                    arguments="float* objnormN, const int stack_size, const int nxyo")

        self.cu_psi_to_probe = CU_ElK(name='psi_to_probe',
                                      operation="UpdateProbeQuadPhase(i, obj, probe_new, psi, probenorm, cx, cy, px, f,"
                                                "firstpass, npsi, stack_size, nx, ny, nxo, nyo, nbobj, nbprobe,"
                                                "interp)",
                                      preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                               getks('ptycho/cuda/psi_to_obj_probe_elw.cu'),
                                      options=self.cu_options,
                                      arguments="pycuda::complex<float>* psi, pycuda::complex<float> *obj,"
                                                "pycuda::complex<float>* probe_new, float* probenorm, float* cx,"
                                                "float* cy, const float px, const float f, const char firstpass,"
                                                "const int npsi, const int stack_size, const int nx, const int ny,"
                                                "const int nxo, const int nyo, const int nbobj, const int nbprobe,"
                                                "const bool interp")

        self.cu_obj_norm = CU_ElK(name='obj_norm',
                                  operation="ObjNorm(i, objnorm, obj_unnorm, obj, regmax, reg, nxyo, nbobj)",
                                  preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                           getks('ptycho/cuda/psi_to_obj_probe_elw.cu'),
                                  options=self.cu_options,
                                  arguments="float* objnorm, pycuda::complex<float>* obj_unnorm,"
                                            "pycuda::complex<float>* obj, float* regmax, const float reg,"
                                            "const int nxyo, const int nbobj")

        self.cu_obj_norm_n = CU_ElK(name='obj_norm_n',
                                    operation="ObjNormN(i, obj_norm, obj_newN, obj, regmax, reg, nxyo, nbobj, stack_size)",
                                    preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                             getks('ptycho/cuda/psi_to_obj_probe_elw.cu'),
                                    options=self.cu_options,
                                    arguments="float* obj_norm, pycuda::complex<float>* obj_newN, pycuda::complex<float>* obj, float* regmax, const float reg, const int nxyo, const int nbobj, const int stack_size")

        self.cu_obj_norm_zero_phase_mask_n = CU_ElK(name='obj_norm_zero_phase_n',
                                                    operation="ObjNormZeroPhaseMaskN(i, obj_norm, obj_newN, obj, zero_phase_mask, regmax, reg, nxyo, nbobj, stack_size)",
                                                    preamble=getks('cuda/complex.cu') +
                                                             getks('ptycho/cuda/psi_to_obj_probe_elw.cu'),
                                                    options=self.cu_options,
                                                    arguments="float* obj_norm, pycuda::complex<float>* obj_newN, pycuda::complex<float>* obj, float* zero_phase_mask, float* regmax, const float reg, const int nxyo, const int nbobj, const int stack_size")

        self.cu_grad_poisson_fourier = CU_ElK(name='grad_poisson_fourier',
                                              operation="GradPoissonFourier(i, iobs, psi, background, background_grad,"
                                                        "nbmode, nx, ny, nxystack, npsi,"
                                                        "hann_filter, scale_in, scale_out)",
                                              preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                       getks('ptycho/cuda/grad_elw.cu'),
                                              options=self.cu_options,
                                              arguments="float *iobs, pycuda::complex<float> *psi, float *background,"
                                                        "float *background_grad, const int nbmode, const int nx,"
                                                        "const int ny, const int nxystack,"
                                                        "const int npsi, const char hann_filter,"
                                                        "const float scale_in, const float scale_out")

        self.cu_psi_to_obj_grad = CU_ElK(name='psi_to_obj_grad',
                                         operation="GradObj(i, psi, obj_grad, probe, cx, cy, px, f, stack_size, nx, ny,"
                                                   "nxo, nyo, nbobj, nbprobe, interp)",
                                         preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                  getks('ptycho/cuda/grad_elw.cu'),
                                         options=self.cu_options,
                                         arguments="pycuda::complex<float>* psi, pycuda::complex<float> *obj_grad,"
                                                   "pycuda::complex<float>* probe, const float cx, const float cy,"
                                                   "const float px, const float f, const int stack_size, const int nx,"
                                                   "const int ny, const int nxo, const int nyo, const int nbobj,"
                                                   "const int nbprobe, const bool interp")

        self.cu_psi_to_obj_grad_atomic = CU_ElK(name='psi_to_obj_grad_atomic',
                                                operation="GradObjAtomic(i, psi, obj_grad, probe, cx, cy, px, f,"
                                                          "stack_size, nx, ny, nxo, nyo, nbobj, nbprobe, npsi, interp)",
                                                preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                         getks('ptycho/cuda/grad_elw.cu'),
                                                options=self.cu_options,
                                                arguments="pycuda::complex<float>* psi,"
                                                          "pycuda::complex<float> *obj_grad,"
                                                          "pycuda::complex<float>* probe, float* cx, float* cy,"
                                                          "const float px, const float f, const int stack_size,"
                                                          "const int nx, const int ny, const int nxo, const int nyo,"
                                                          "const int nbobj, const int nbprobe, const int npsi,"
                                                          "const bool interp")

        self.cu_psi_to_probe_grad = CU_ElK(name='psi_to_probe_grad',
                                           operation="GradProbe(i, psi, probe_grad, obj, cx, cy, px, f, firstpass,"
                                                     "npsi, stack_size, nx, ny, nxo, nyo, nbobj, nbprobe, interp)",
                                           preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                    getks('ptycho/cuda/grad_elw.cu'),
                                           options=self.cu_options,
                                           arguments="pycuda::complex<float>* psi, pycuda::complex<float>* probe_grad,"
                                                     "pycuda::complex<float> *obj, float* cx, float* cy,"
                                                     "const float px, const float f, const char firstpass,"
                                                     "const int npsi, const int stack_size, const int nx, const int ny,"
                                                     "const int nxo, const int nyo, const int nbobj, const int nbprobe,"
                                                     "const bool interp")

        self.cu_reg_grad = CU_ElK(name='reg_grad',
                                  operation="GradReg(i, grad, v, alpha, nx, ny)",
                                  preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                           getks('ptycho/cuda/grad_elw.cu'),
                                  options=self.cu_options,
                                  arguments="pycuda::complex<float>* grad, pycuda::complex<float>* v,"
                                            "const float alpha, const int nx, const int ny")

        self.cu_circular_shift = CU_ElK(name='cu_circular_shift',
                                        operation="circular_shift(i, source, dest, dx, dy, dz, nx, ny, nz)",
                                        preamble=getks('cuda/complex.cu') + getks('cuda/circular_shift.cu'),
                                        options=self.cu_options,
                                        arguments="pycuda::complex<float>* source, pycuda::complex<float>* dest,"
                                                  "const int dx, const int dy, const int dz,"
                                                  "const int nx, const int ny, const int nz")

        self.cu_psi2pos_merge = CU_ElK(name='cu_psi2pos_merge',
                                       operation="cx[i] += dxy[i].x - dxy0[0].real();"
                                                 "cy[i] += dxy[i].y - dxy0[0].imag();",
                                       options=self.cu_options,
                                       preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu'),
                                       arguments="my_float4* dxy, pycuda::complex<float>* dxy0, float* cx, float* cy")

        self.cu_corr_phase_ramp = CU_ElK(name='corr_phase_ramp',
                                         operation="CorrPhaseRamp2D(i, d, dx, dy, nx, ny)",
                                         preamble=getks('cuda/complex.cu') + getks('cuda/corr_phase_ramp_elw.cu'),
                                         options=self.cu_options,
                                         arguments="pycuda::complex<float> *d, const float dx, const float dy,"
                                                   "const int nx, const int ny")

        self.cu_llk_poisson_stats = CU_ElK(name='cu_llk_poisson_stats',
                                           operation="LLKPoissonStats(i, iobs, psi, background, llk_mean,"
                                                     "llk_std, llk_skew, llk_skew0, nbmode, nxy, nxystack, scale)",
                                           preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') +
                                                    getks('ptycho/cuda/llk_red.cu'),
                                           options=self.cu_options,
                                           arguments="float *iobs, pycuda::complex<float> *psi,"
                                                     "float *background, float *llk_mean,"
                                                     "float *llk_std, float *llk_skew, float *llk_skew0,"
                                                     "const int nbmode, const int nxy,"
                                                     "const int nxystack, const float scale")

        self.cu_llk_poisson_hist = CU_ElK(name='cu_llk_poisson_hist',
                                          operation="LLKPoissonHist(i, iobs, psi, background, "
                                                    "llk_sum, llk_hist, nbin, binsize,"
                                                    "nbmode, nxy, nxystack, scale)",
                                          preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') +
                                                   getks('ptycho/cuda/llk_red.cu'),
                                          options=self.cu_options,
                                          arguments="float *iobs, pycuda::complex<float> *psi,"
                                                    "float *background, float *llk_sum, short * llk_hist,"
                                                    "const int nbin, const float binsize,"
                                                    "const int nbmode, const int nxy,"
                                                    "const int nxystack, const float scale")

        self.cu_padding_interp = CU_ElK(name='padding_interp',
                                        operation="PaddingInterp(i, d, nx, ny, padding)",
                                        preamble=getks('cuda/complex.cu') + getks('ptycho/cuda/padding_interp.cu'),
                                        options=self.cu_options,
                                        arguments="pycuda::complex<float> *d, const int nx, const int ny,"
                                                  "const int padding")

        # Reduction kernels
        self.cu_max_red = CU_RedK(np.float32, neutral="-1e32", reduce_expr="max(a,b)",
                                  options=self.cu_options, arguments="float *in")

        self.cu_norm_complex_n = CU_RedK(np.float32, neutral="0", reduce_expr="a+b", name='norm_complex_n_red',
                                         map_expr="ComplexNormN(d[i], nn)",
                                         options=self.cu_options,
                                         preamble=getks('cuda/complex.cu'),
                                         arguments="pycuda::complex<float> *d, const int nn")

        # This will compute Poisson, Gaussian, Euclidian LLK as well as the sum of the calculated intensity
        self.cu_llk = CU_RedK(my_float4, neutral="my_float4(0)", reduce_expr="a+b", name='llk_red',
                              preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') +
                                       getks('ptycho/cuda/llk_red.cu'),
                              options=self.cu_options,
                              map_expr="LLKAll(i, iobs, psi, background, nbmode, nxy, nxystack, scale)",
                              arguments="float *iobs, pycuda::complex<float> *psi, float *background, const int nbmode,"
                                        "const int nxy, const int nxystack, const float scale")

        self.cu_cg_polak_ribiere_red = CU_RedK(np.complex64, neutral="complexf(0,0)", name='polak_ribiere_red',
                                               reduce_expr="a+b",
                                               map_expr="PolakRibiereComplex(grad[i], lastgrad[i])",
                                               preamble=getks('cuda/complex.cu') + getks(
                                                   'cuda/cg_polak_ribiere_red.cu'),
                                               options=self.cu_options,
                                               arguments="pycuda::complex<float> *grad, pycuda::complex<float> *lastgrad")

        self.cu_cg_polak_ribiere_redf = CU_RedK(np.complex64, neutral="0", name='polak_ribiere_redf',
                                                reduce_expr="a+b",
                                                map_expr="PolakRibiereFloat(grad[i], lastgrad[i])",
                                                preamble=getks('cuda/complex.cu') + getks(
                                                    'cuda/cg_polak_ribiere_red.cu'),
                                                options=self.cu_options,
                                                arguments="float *grad, float *lastgrad")

        self._cu_cg_poisson_gamma_red = CU_RedK(np.complex64, neutral="complexf(0,0)", name='cg_poisson_gamma_red',
                                                reduce_expr="a+b",
                                                map_expr="CG_Poisson_Gamma(i, obs, PO, PdO, dPO, dPdO, background,"
                                                         "background_dir, nxy, nxystack, nbmode, npsi, scale)",
                                                preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') + getks(
                                                    'ptycho/cuda/cg_gamma_red.cu'),
                                                options=self.cu_options,
                                                arguments="float *obs, pycuda::complex<float> *PO, "
                                                          "pycuda::complex<float> *PdO, pycuda::complex<float> *dPO,"
                                                          "pycuda::complex<float> *dPdO, float* background,"
                                                          "float* background_dir, const int nxy, const int nxystack,"
                                                          "const int nbmode, const int npsi, const float scale")

        self.cu_psi2pos_stack_red = CU_RedK(my_float4, neutral="my_float4(0)", name="psi2pos_stack_red",
                                            reduce_expr="a+b",
                                            preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                                     getks('cuda/float_n.cu') + getks('ptycho/cuda/psi2pos.cu'),
                                            map_expr="Psi2PosShift(i, psi, obj, probe, cx, cy, pixel_size, f, nx, ny,"
                                                     "nxo, nyo, ipsi, interp)",
                                            options=self.cu_options,
                                            arguments="pycuda::complex<float>* psi, pycuda::complex<float>* obj,"
                                                      "pycuda::complex<float>* probe, float* cx, float* cy,"
                                                      "const float pixel_size, const float f, const int nx,"
                                                      "const int ny, const int nxo, const int nyo,"
                                                      "const int ipsi, const bool interp")

        self.cu_psi2pos_red = CU_RedK(np.complex64, neutral="complexf(0,0)", reduce_expr="a+b",
                                      preamble=getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                               getks('cuda/float_n.cu') + getks('ptycho/cuda/psi2pos.cu'),
                                      map_expr="Psi2PosRed(i, dxy, mult, max_shift, min_shift, thres[0], nb)",
                                      options=self.cu_options,
                                      arguments="my_float4* dxy, const float mult, const float max_shift,"
                                                "const float min_shift, const float* thres, const int nb")

        self.cu_psi2pos_thres_red = CU_RedK(np.float32, neutral="0", reduce_expr="a+b",
                                            preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu'),
                                            map_expr="sqrtf(dxy[i].z*dxy[i].z + dxy[i].w*dxy[i].w) * thres / nb",
                                            options=self.cu_options,
                                            arguments="my_float4* dxy, const float thres, const int nb")

        # 4th order LLK(gamma) approximation
        self._cu_cg_poisson_gamma4_red = CU_RedK(my_float4, neutral="my_float4(0)", name='cg_poisson_gamma4_red',
                                                 reduce_expr="a+b",
                                                 map_expr="CG_Poisson_Gamma4(i, obs, PO, PdO, dPO, dPdO, nxy,"
                                                          "nxystack, nbmode, scale)",
                                                 preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') + getks(
                                                     'ptycho/cuda/cg_gamma_red.cu'),
                                                 options=self.cu_options,
                                                 arguments="float *obs, pycuda::complex<float> *PO,"
                                                           "pycuda::complex<float> *PdO, pycuda::complex<float> *dPO,"
                                                           "pycuda::complex<float> *dPdO, const int nxy,"
                                                           "const int nxystack, const int nbmode,"
                                                           "const float scale")

        self._cu_cg_gamma_reg_red = CU_RedK(np.complex64, neutral="complexf(0,0)", name='cg_gamma_reg_red',
                                            reduce_expr="a+b",
                                            map_expr="GammaReg(i, v, dv, nx, ny)",
                                            preamble=getks('cuda/complex.cu') + getks('cuda/cg_gamma_reg_red.cu'),
                                            arguments="pycuda::complex<float> *v, pycuda::complex<float> *dv,"
                                                      "const int nx, const int ny")

        self.cu_scale_intensity = CU_RedK(np.complex64, neutral="complexf(0,0)", name='scale_intensity',
                                          reduce_expr="a+b", map_expr="scale_intensity(i, obs, calc, background,"
                                                                      "nxy, nxystack, nb_mode)",
                                          preamble=getks('cuda/complex.cu') + getks('ptycho/cuda/scale_red.cu'),
                                          arguments="float *obs, pycuda::complex<float> *calc, float* background,"
                                                    "const int nxy, const int nxystack, const int nb_mode")

        self.cu_center_mass = CU_RedK(my_float4, neutral="my_float4(0)", name="cu_center_mass",
                                      reduce_expr="a+b",
                                      preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu')
                                               + getks('cuda/center_mass_red.cu'),
                                      options=self.cu_options,
                                      map_expr="center_mass_float(i, d, nx, ny, nz, power)",
                                      arguments="float *d, const int nx, const int ny, const int nz, const int power")

        self.cu_center_mass_fftshift = CU_RedK(my_float4, neutral="my_float4(0)", name="cu_center_mass_fftshift",
                                               reduce_expr="a+b",
                                               preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu')
                                                        + getks('cuda/center_mass_red.cu'),
                                               options=self.cu_options,
                                               map_expr="center_mass_fftshift_float(i, d, nx, ny, nz, power)",
                                               arguments="float *d, const int nx, const int ny,"
                                                         "const int nz, const int power")

        self.cu_center_mass_complex = CU_RedK(my_float4, neutral="my_float4(0)", name="cu_center_mass_complex",
                                              reduce_expr="a+b",
                                              preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu')
                                                       + getks('cuda/center_mass_red.cu'),
                                              options=self.cu_options,
                                              map_expr="center_mass_complex(i, d, nx, ny, nz, power)",
                                              arguments="pycuda::complex<float> *d, const int nx, const int ny,"
                                                        "const int nz, const int power")

        self.cu_center_mass_fftshift_complex = CU_RedK(my_float4, neutral="my_float4(0)", name="cu_center_mass_complex",
                                                       reduce_expr="a+b",
                                                       preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu')
                                                                + getks('cuda/center_mass_red.cu'),
                                                       options=self.cu_options,
                                                       map_expr="center_mass_fftshift_complex(i, d, nx, ny, nz, power)",
                                                       arguments="pycuda::complex<float> *d, const int nx,"
                                                                 "const int ny, const int nz, const int power")

        # custom kernels
        # Gaussian convolution kernels
        opt = "#define BLOCKSIZE 16\n#define HALFBLOCK 7\n"
        conv16_mod = SourceModule(opt + getks('cuda/complex.cu') + getks('cuda/convolution_complex.cu'),
                                  options=self.cu_options)
        self.gauss_convol_complex_16x = conv16_mod.get_function("gauss_convol_complex_x")
        self.gauss_convol_complex_16y = conv16_mod.get_function("gauss_convol_complex_y")
        self.gauss_convol_complex_16z = conv16_mod.get_function("gauss_convol_complex_z")

        opt = "#define BLOCKSIZE 32\n#define HALFBLOCK 15\n"
        conv32_mod = SourceModule(opt + getks('cuda/complex.cu') + getks('cuda/convolution_complex.cu'),
                                  options=self.cu_options)
        self.gauss_convol_complex_32x = conv32_mod.get_function("gauss_convol_complex_x")
        self.gauss_convol_complex_32y = conv32_mod.get_function("gauss_convol_complex_y")
        self.gauss_convol_complex_32z = conv32_mod.get_function("gauss_convol_complex_z")

        opt = "#define BLOCKSIZE 64\n#define HALFBLOCK 31\n"
        conv64_mod = SourceModule(opt + getks('cuda/complex.cu') + getks('cuda/convolution_complex.cu'),
                                  options=self.cu_options)
        self.gauss_convol_complex_64x = conv64_mod.get_function("gauss_convol_complex_x")
        self.gauss_convol_complex_64y = conv64_mod.get_function("gauss_convol_complex_y")
        self.gauss_convol_complex_64z = conv64_mod.get_function("gauss_convol_complex_z")

        conv16f_mod = SourceModule(getks('cuda/complex.cu') + getks('cuda/convolution16f.cu'), options=self.cu_options)
        self.gauss_convolf_16x = conv16f_mod.get_function("gauss_convolf_16x")
        self.gauss_convolf_16y = conv16f_mod.get_function("gauss_convolf_16y")
        self.gauss_convolf_16z = conv16f_mod.get_function("gauss_convolf_16z")

        # Position update kernel
        psi2pos_mod = SourceModule(getks("cuda/argmax.cu") + getks('cuda/float_n.cu') +
                                   getks('cuda/complex.cu') + getks('cuda/bilinear.cu') +
                                   getks("ptycho/cuda/psi2pos.cu"),
                                   options=["-use_fast_math"])
        self.cu_psi2pos = psi2pos_mod.get_function("Psi2Pos")
        self.cu_psi2pos_merge = psi2pos_mod.get_function("Psi2PosMerge")


"""
The default processing unit 
"""
default_processing_unit = CUProcessingUnitPtycho()


class CUObsDataStack:
    """
    Class to store a stack (e.g. 16 frames) of observed data in CUDA space
    """

    def __init__(self, cu_obs, i, npsi):
        """

        :param cu_obs: pycuda array of observed data, with N frames
        :param i: index of the first frame
        :param npsi: number of valid frames (others are filled with zeros)
        """
        self.cu_obs = cu_obs
        self.i = np.int32(i)
        self.npsi = np.int32(npsi)


class CUOperatorPtycho(OperatorPtycho):
    """
    Base class for a operators on Ptycho objects using CUDA
    """

    def __init__(self, processing_unit=None):
        super(CUOperatorPtycho, self).__init__()

        self.Operator = CUOperatorPtycho
        self.OperatorSum = CUOperatorPtychoSum
        self.OperatorPower = CUOperatorPtychoPower

        if processing_unit is None:
            self.processing_unit = default_processing_unit
        else:
            self.processing_unit = processing_unit
        if self.processing_unit.cu_ctx is None:
            # CUDA kernels have not been prepared yet, use a default initialization
            if main_default_processing_unit.cu_device is None:
                main_default_processing_unit.select_gpu(language='cuda')

            self.processing_unit.init_cuda(cu_device=main_default_processing_unit.cu_device,
                                           test_fft=False, verbose=False)

    def apply_ops_mul(self, pty):
        """
        Apply the series of operators stored in self.ops to a Ptycho2D object.
        The operators are applied one after the other.

        :param pty: the Ptycho2D to which the operators will be applied.
        :return: the Ptycho2D object, after application of all the operators in sequence
        """
        if isinstance(pty, Ptycho) is False:
            raise OperatorException(
                "ERROR: tried to apply operator:\n    %s \n  to:\n    %s\n  which is not a Ptycho object" % (
                    str(self), str(pty)))
        return super(CUOperatorPtycho, self).apply_ops_mul(pty)

    def prepare_data(self, p: Ptycho):
        """
        Make sure the data to be used is in the correct memory (host or GPU) for the operator.
        Virtual, must be derived.

        :param p: the Ptycho object the operator will be applied to.
        :return:
        """
        pu = self.processing_unit
        if has_attr_not_none(p, "_cu_obs_v") is False:
            # Assume observed intensity is immutable, so transfer only once
            self.init_cu_vobs(p)
        elif len(p._cu_obs_v[0].cu_obs) != self.processing_unit.cu_stack_size:
            # This should not happen, but if tests are being made on the speed vs the stack size, this can be useful.
            self.init_cu_vobs(p)

        if p._timestamp_counter > p._cu_timestamp_counter:
            # print("Moving object, probe, positions to CUDA GPU")
            p._cu_obj = cua.to_gpu(p._obj, allocator=pu.get_memory_pool().allocate)
            p._cu_probe = cua.to_gpu(p._probe, allocator=pu.get_memory_pool().allocate)
            p._cu_cx = cua.empty(len(p.data.posx), dtype=np.float32, allocator=pu.get_memory_pool().allocate)
            p._cu_cy = cua.empty(len(p.data.posx), dtype=np.float32, allocator=pu.get_memory_pool().allocate)
            if p._obj_zero_phase_mask is not None:
                p._cu_obj_zero_phase_mask = cua.to_gpu(p._obj_zero_phase_mask, allocator=pu.get_memory_pool().allocate)
            else:
                p._cu_obj_zero_phase_mask = None
            p._cu_timestamp_counter = p._timestamp_counter
            if p._background is None:
                p._cu_background = cua.zeros(p.data.iobs.shape[-2:], dtype=np.float32,
                                             allocator=pu.get_memory_pool().allocate)
            else:
                p._cu_background = cua.to_gpu(p._background, allocator=pu.get_memory_pool().allocate)

            # Positions (corner coordinate)
            nb_frame, ny, nx = p.data.iobs.shape
            nyo, nxo = p._obj.shape[-2:]
            cu_stack_size = self.processing_unit.cu_stack_size
            px, py = p.data.pixel_size_object()
            vcx, vcy = np.empty(nb_frame, dtype=np.float32), np.empty(nb_frame, dtype=np.float32)
            for i in range(nb_frame):
                dy, dx = p.data.posy[i] / py, p.data.posx[i] / px
                cx, cy = get_view_coord((nyo, nxo), (ny, nx), dx, dy, integer=False)
                vcx[i] = cx
                vcy[i] = cy
            cu_drv.memcpy_htod_async(dest=p._cu_cx.gpudata, src=vcx)
            cu_drv.memcpy_htod_async(dest=p._cu_cy.gpudata, src=vcy)

        need_init_psi = False
        if has_attr_not_none(p, "_cu_psi") is False:
            need_init_psi = True
        elif p._cu_psi.shape[0:3] != (len(p._obj), len(p._probe), self.processing_unit.cu_stack_size):
            need_init_psi = True
        if need_init_psi:
            ny, nx = p._probe.shape[-2:]
            p._cu_psi = cua.empty(shape=(len(p._obj), len(p._probe), self.processing_unit.cu_stack_size, ny, nx),
                                  dtype=np.complex64, allocator=pu.get_memory_pool().allocate)

        if has_attr_not_none(p, "_cu_psi_v") is False or need_init_psi:
            # _cu_psi_v is used to hold the complete copy of Psi projections for all stacks, for algorithms
            # such as DM which need them.
            p._cu_psi_v = {}

        if has_attr_not_none(p, '_cu_view') is False:
            p._cu_view = {}

    def init_cu_vobs(self, p):
        """
        Initialize observed intensity and scan positions in GPU space

        :param p: the Ptycho object the operator will be applied to.
        :return:
        """
        # print("Moving observed data to PU GPU")
        pu = self.processing_unit
        p._cu_obs_v = []
        nb_frame, ny, nx = p.data.iobs.shape
        cu_stack_size = self.processing_unit.cu_stack_size
        for i in range(0, nb_frame, cu_stack_size):
            # Positions will actually be copied in prepare_data()
            vobs = np.zeros((cu_stack_size, ny, nx), dtype=np.float32)
            for j in range(cu_stack_size):
                ij = i + j
                if ij < nb_frame:
                    vobs[j] = p.data.iobs[ij]
                else:
                    vobs[j] = np.zeros_like(vobs[0], dtype=np.float32)
            cu_vobs = cua.to_gpu(vobs, allocator=pu.get_memory_pool().allocate)
            p._cu_obs_v.append(CUObsDataStack(cu_vobs, i, np.int32(min(cu_stack_size, nb_frame - i))))
        # Initialize the size and index of current stack
        p._cu_stack_i = 0
        p._cu_stack_nb = len(p._cu_obs_v)

    def timestamp_increment(self, p):
        p._timestamp_counter += 1
        p._cu_timestamp_counter = p._timestamp_counter

    def view_register(self, obj):
        """
        Creates a new unique view key in an object. When finished with this view, it should be de-registered
        using view_purge. Note that it only reserves the key, but does not create the view.
        :return: an integer value, which corresponds to yet-unused key in the object's view.
        """
        i = 1
        while i in obj._cu_view:
            i += 1
        obj._cu_view[i] = None
        return i

    def view_copy(self, pty, i_source, i_dest):
        if i_source == 0:
            src = {'obj': pty._cu_obj, 'probe': pty._cu_probe, 'psi': pty._cu_psi, 'psi_v': pty._cu_psi_v}
        else:
            src = pty._cu_view[i_source]
        if i_dest == 0:
            pty._cu_obj = cua.empty_like(src['obj'])
            pty._cu_probe = cua.empty_like(src['probe'])
            pty._cu_psi = cua.empty_like(src['psi'])
            pty._cu_psi_v = {}
            dest = {'obj': pty._cu_obj, 'probe': pty._cu_probe, 'psi': pty._cu_psi, 'psi_v': pty._cu_psi_v}
        else:
            pty._cu_view[i_dest] = {'obj': cua.empty_like(src['obj']), 'probe': cua.empty_like(src['probe']),
                                    'psi': cua.empty_like(src['psi']), 'psi_v': {}}
            dest = pty._cu_view[i_dest]

        for i in range(len(src['psi_v'])):
            dest['psi_v'][i] = cua.empty_like(src['psi'])

        for s, d in zip([src['obj'], src['probe'], src['psi']] + [v for k, v in src['psi_v'].items()],
                        [dest['obj'], dest['probe'], dest['psi']] + [v for k, v in dest['psi_v'].items()]):
            cu_drv.memcpy_dtod(dest=d.gpudata, src=s.gpudata, size=d.nbytes)

    def view_swap(self, pty, i1, i2):
        if i1 != 0:
            if pty._cu_view[i1] is None:
                # Create dummy value, assume a copy will be made later
                pty._cu_view[i1] = {'obj': None, 'probe': None, 'psi': None, 'psi_v': None}
        if i2 != 0:
            if pty._cu_view[i2] is None:
                # Create dummy value, assume a copy will be made later
                pty._cu_view[i2] = {'obj': None, 'probe': None, 'psi': None, 'psi_v': None}
        if i1 == 0:
            pty._cu_obj, pty._cu_view[i2]['obj'] = pty._cu_view[i2]['obj'], pty._cu_obj
            pty._cu_probe, pty._cu_view[i2]['probe'] = pty._cu_view[i2]['probe'], pty._cu_probe
            pty._cu_psi, pty._cu_view[i2]['psi'] = pty._cu_view[i2]['psi'], pty._cu_psi
            pty._cu_psi_v, pty._cu_view[i2]['psi_v'] = pty._cu_view[i2]['psi_v'], pty._cu_psi_v
        elif i2 == 0:
            pty._cu_obj, pty._cu_view[i1]['obj'] = pty._cu_view[i1]['obj'], pty._cu_obj
            pty._cu_probe, pty._cu_view[i1]['probe'] = pty._cu_view[i1]['probe'], pty._cu_probe
            pty._cu_psi, pty._cu_view[i1]['psi'] = pty._cu_view[i1]['psi'], pty._cu_psi
            pty._cu_psi_v, pty._cu_view[i1]['psi_v'] = pty._cu_view[i1]['psi_v'], pty._cu_psi_v
        else:
            pty._cu_view[i1], pty._cu_view[i2] = pty._cu_view[i2], pty._cu_view[i1]
        self.timestamp_increment(pty)

    def view_sum(self, pty, i_source, i_dest):
        if i_source == 0:
            src = {'obj': pty._cu_obj, 'probe': pty._cu_probe, 'psi': pty._cu_psi, 'psi_v': pty._cu_psi_v}
        else:
            src = pty._cu_view[i_source]
        if i_dest == 0:
            dest = {'obj': pty._cu_obj, 'probe': pty._cu_probe, 'psi': pty._cu_psi, 'psi_v': pty._cu_psi_v}
        else:
            dest = pty._cu_view[i_dest]
        for s, d in zip([src['obj'], src['probe'], src['psi']] + [v for k, v in src['psi_v'].items()],
                        [dest['obj'], dest['probe'], dest['psi']] + [v for k, v in dest['psi_v'].items()]):
            self.processing_unit.cu_sum(s, d)
        self.timestamp_increment(pty)

    def view_purge(self, pty, i):
        if i is not None:
            del pty._cu_view[i]
        elif has_attr_not_none(pty, '_cu_view'):
            del pty._cu_view


# The only purpose of this class is to make sure it inherits from CUOperatorPtycho and has a processing unit
class CUOperatorPtychoSum(OperatorSum, CUOperatorPtycho):
    def __init__(self, op1, op2):
        if np.isscalar(op1):
            op1 = Scale(op1)
        if np.isscalar(op2):
            op2 = Scale(op2)
        if isinstance(op1, CUOperatorPtycho) is False or isinstance(op2, CUOperatorPtycho) is False:
            raise OperatorException(
                "ERROR: cannot add a CUOperatorPtycho with a non-CUOperatorPtycho: %s + %s" % (str(op1), str(op2)))
        # We can only have a sum of two CuOperatorPtycho, so they must have a processing_unit attribute.
        CUOperatorPtycho.__init__(self, op1.processing_unit)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CUOperatorPtycho
        self.OperatorSum = CUOperatorPtychoSum
        self.OperatorPower = CUOperatorPtychoPower
        self.prepare_data = types.MethodType(CUOperatorPtycho.prepare_data, self)
        self.timestamp_increment = types.MethodType(CUOperatorPtycho.timestamp_increment, self)
        self.view_copy = types.MethodType(CUOperatorPtycho.view_copy, self)
        self.view_swap = types.MethodType(CUOperatorPtycho.view_swap, self)
        self.view_sum = types.MethodType(CUOperatorPtycho.view_sum, self)
        self.view_purge = types.MethodType(CUOperatorPtycho.view_purge, self)


# The only purpose of this class is to make sure it inherits from CUOperatorPtycho and has a processing unit
class CUOperatorPtychoPower(OperatorPower, CUOperatorPtycho):
    def __init__(self, op, n):
        CUOperatorPtycho.__init__(self, op.processing_unit)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CUOperatorPtycho
        self.OperatorSum = CUOperatorPtychoSum
        self.OperatorPower = CUOperatorPtychoPower
        self.prepare_data = types.MethodType(CUOperatorPtycho.prepare_data, self)
        self.timestamp_increment = types.MethodType(CUOperatorPtycho.timestamp_increment, self)
        self.view_copy = types.MethodType(CUOperatorPtycho.view_copy, self)
        self.view_swap = types.MethodType(CUOperatorPtycho.view_swap, self)
        self.view_sum = types.MethodType(CUOperatorPtycho.view_sum, self)
        self.view_purge = types.MethodType(CUOperatorPtycho.view_purge, self)


class FreePU(CUOperatorPtycho):
    """
    Operator freeing CUDA memory. The gpyfft data reference in self.processing_unit is removed,
    as well as any CUDA pycuda.array.GPUArray attribute in the supplied object.
    """

    def __init__(self, verbose=False):
        """

        :param verbose: if True, will detail all the free'd memory and a summary
        """
        super(FreePU, self).__init__()
        self.verbose = verbose

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        self.processing_unit.finish()
        self.processing_unit.free_fft_plans()

        p.from_pu()
        if self.verbose:
            p.print("FreePU:")
        bytes = 0

        for o in dir(p):
            if isinstance(p.__getattribute__(o), cua.GPUArray):
                if self.verbose:
                    p.print("  Freeing: %40s %10.3fMbytes" % (o, p.__getattribute__(o).nbytes / 1e6))
                    bytes += p.__getattribute__(o).nbytes
                p.__getattribute__(o).gpudata.free()
                p.__setattr__(o, None)
        if has_attr_not_none(p, "_cu_psi_v"):
            for a in p._cu_psi_v.values():
                if self.verbose:
                    p.print("  Freeing: %40s %10.3fMbytes" % ("_cu_psi_v", a.nbytes / 1e6))
                    bytes += a.nbytes
                a.gpudata.free()
            p._cu_psi_v = {}
        for v in p._cu_obs_v:
            for o in dir(v):
                if isinstance(v.__getattribute__(o), cua.GPUArray):
                    v.__getattribute__(o).gpudata.free()
                    if self.verbose:
                        p.print("  Freeing: %40s %10.3fMbytes" % ("_cu_obs_v:" + o, v.__getattribute__(o).nbytes / 1e6))
                        bytes += v.__getattribute__(o).nbytes

        p._cu_obs_v = None
        self.processing_unit.get_memory_pool().free_held()
        gc.collect()
        if self.verbose:
            p.print('FreePU total: %10.3fMbytes freed' % (bytes / 1e6))
        return p

    def timestamp_increment(self, p):
        p._cu_timestamp_counter = 0


class MemUsage(CUOperatorPtycho):
    """
    Print memory usage of current process (RSS on host) and used GPU memory
    """

    def __init__(self, verbose=True):
        super(MemUsage, self).__init__()
        self.verbose = verbose

    def op(self, p: Ptycho):
        """

        :param p: the ptycho object this operator applies to
        :return: the updated ptycho object
        """
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        gpu_mem = 0

        for o in dir(p):
            if isinstance(p.__getattribute__(o), cua.GPUArray):
                if self.verbose:
                    print("   %40s %10.3fMbytes" % (o, p.__getattribute__(o).nbytes / 1e6))
                gpu_mem += p.__getattribute__(o).nbytes
        tmp_bytes = 0
        if has_attr_not_none(p, "_cu_psi_v"):
            for a in p._cu_psi_v.values():
                tmp_bytes += a.nbytes
            if self.verbose:
                p.print("   %40s %10.3fMbytes" % ("_cu_psi_v", tmp_bytes / 1e6))
        gpu_mem += tmp_bytes
        tmp_bytes = 0
        for v in p._cu_obs_v:
            for o in dir(v):
                if isinstance(v.__getattribute__(o), cua.GPUArray):
                    tmp_bytes += v.__getattribute__(o).nbytes
        gpu_mem += tmp_bytes
        if self.verbose:
            p.print("   %40s %10.3fMbytes" % ("_cu_obs_v", tmp_bytes / 1e6))

        d = self.processing_unit.cu_device
        p.print("GPU used: %s [%4d Mbytes]" % (d.name(), int(round(d.total_memory() // 2 ** 20))))
        p.print("Mem Usage: RSS= %6.1f Mbytes (process), GPU Mem= %6.1f Mbytes (Ptycho object)" %
                (rss / 1024 ** 2, gpu_mem / 1024 ** 2))
        return p

    def prepare_data(self, p: Ptycho):
        # Overriden to avoid transferring any data to GPU
        pass

    def timestamp_increment(self, p):
        # This operator does nothing
        pass


class Scale(CUOperatorPtycho):
    """
    Multiply the ptycho object by a scalar (real or complex).
    """

    def __init__(self, x, obj=True, probe=True, psi=True):
        """

        :param x: the scaling factor
        :param obj: if True, scale the object
        :param probe: if True, scale the probe
        :param psi: if True, scale the all the psi arrays, _cu_psi as well as _cu_psi_v
        """
        super(Scale, self).__init__()
        self.x = x
        self.obj = obj
        self.probe = probe
        self.psi = psi

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        if self.x == 1:
            return p

        if np.isreal(self.x):
            scale_k = pu.cu_scale
            x = np.float32(self.x)
        else:
            scale_k = pu.cu_scale_complex
            x = np.complex64(self.x)

        if self.obj:
            scale_k(p._cu_obj, x)
        if self.probe:
            scale_k(p._cu_probe, x)
        if self.psi:
            scale_k(p._cu_psi, x)
            for i in range(len(p._cu_psi_v)):
                scale_k(p._cu_psi_v[i], x)
        return p


class ObjProbe2Psi(CUOperatorPtycho):
    """
    Computes Psi = Obj(r) * Probe(r-r_j) for a stack of N probe positions.
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        # Multiply obj and probe with quadratic phase factor, taking into account all modes (if any)
        i = p._cu_stack_i
        npsi = p._cu_obs_v[i].npsi
        i0 = p._cu_obs_v[i].i
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        interp = p._interpolation
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))
        # p.print(i, f, p._cu_obs_v[i].npsi, self.processing_unit.cu_stack_size, nx, ny, nxo, nyo, nb_probe, nb_obj)
        # First argument is p._cu_psi[0] because the kernel will calculate the projection for all object and probe modes
        # and the full stack of frames.
        pu.cu_object_probe_mult(p._cu_psi[0, 0, 0], p._cu_obj, p._cu_probe, p._cu_cx[i0:i0 + npsi],
                                p._cu_cy[i0:i0 + npsi], p.pixel_size_object, f, npsi,
                                pu.cu_stack_size, nx, ny, nxo, nyo, nb_obj, nb_probe, interp)
        return p


class FT(CUOperatorPtycho):
    """
    Forward Fourier-transform a Psi array, i.e. a stack of N Obj*Probe views
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(FT, self).__init__()
        self.scale = scale

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        pu.fft(p._cu_psi, p._cu_psi, ndim=2, norm=self.scale)
        return p


class IFT(CUOperatorPtycho):
    """
    Backward Fourier-transform a Psi array, i.e. a stack of N Obj*Probe views
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(IFT, self).__init__()
        self.scale = scale

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        scale = pu.ifft(p._cu_psi, p._cu_psi, ndim=2, norm=self.scale)
        return p


class QuadraticPhase(CUOperatorPtycho):
    """
    Operator applying a quadratic phase factor
    """

    def __init__(self, factor, scale=1):
        """
        Application of a quadratic phase factor, and optionally a scale factor.

        The actual factor is:  :math:`scale * e^{i * factor * ((ix/nx)^2 + (iy/ny)^2)}`
        where ix and iy are the integer indices of the pixels

        :param factor: the factor for the phase calculation.
        :param scale: the data will be scaled by this factor. Useful to normalize before/after a Fourier transform,
                      without accessing twice the array data.
        """
        super(QuadraticPhase, self).__init__()
        self.scale = np.float32(scale)
        self.factor = np.float32(factor)

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        pu.cu_quad_phase(p._cu_psi, self.factor, self.scale, nx, ny)
        return p


class PropagateNearField(CUOperatorPtycho):
    """
    Near field propagator
    """

    def __init__(self, forward=True):
        """

        :param forward: if True, propagate forward, otherwise backward. The distance is taken from the ptycho data
                        this operator applies to.
        """
        super(PropagateNearField, self).__init__()
        self.forward = forward

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        f = np.float32(-np.pi * p.data.wavelength * p.data.detector_distance / p.data.pixel_size_detector ** 2)
        if self.forward is False:
            f = -f
        s = self.processing_unit.fft_scale(p._cu_psi.shape, ndim=2)
        s = s[0] * s[1]  # Compensates for FFT+iFFT scaling
        p = IFT(scale=False) * QuadraticPhase(factor=f, scale=s) * FT(scale=False) * p
        return p


class Calc2Obs1(CUOperatorPtycho):
    """
    Copy the calculated intensities to the observed ones. Can be used for simulation. Applies to a stack of N views,
    assumes the current Psi is already in Fourier space.
    """

    def __init__(self):
        """

        """
        super(Calc2Obs1, self).__init__()

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        nxy = np.int32(p._probe.shape[-2] * p._probe.shape[-1])
        nxystack = np.int32(nxy * self.processing_unit.cu_stack_size)
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        i = p._cu_stack_i
        nb_psi = p._cu_obs_v[i].npsi
        pu.cu_calc2obs(p._cu_obs_v[i].cu_obs[:nb_psi], p._cu_psi, p._cu_background, nb_mode, nxy, nxystack)
        p.data.iobs[i * pu.cu_stack_size: i * pu.cu_stack_size + nb_psi] = p._cu_obs_v[i].cu_obs[:nb_psi].get()
        return p


class Calc2Obs(CUOperatorPtycho):
    """
    Copy the calculated intensities to the observed ones, optionally with Poisson noise.
    This operator will loop other all stacks of frames, multiply object and probe and
    propagate the wavefront to the detector, and compute the new intensities.
    """

    def __init__(self, nb_photons_per_frame=None, poisson_noise=False):
        """

        :param nb_photons_per_frame: average number of photons per frame, to scale the images.
                                     If None, no scaling is performed.
        :param poisson_noise: if True, Poisson noise will be applied on the calculated frames.
                              This uses numpy.random.poisson and is not GPU-optimised.
        """
        super(Calc2Obs, self).__init__()
        self.nb_photons_per_frame = nb_photons_per_frame
        self.poisson_noise = poisson_noise

    def op(self, p: Ptycho):
        if p.data.near_field:
            prop = PropagateNearField(forward=True)
        else:
            prop = FT(scale=False)
        p = LoopStack(Calc2Obs1() * prop * ObjProbe2Psi()) * p

        p.from_pu()

        if self.nb_photons_per_frame is not None:
            p.data.iobs *= self.nb_photons_per_frame / (p.data.iobs.sum() / len(p.data.iobs))

        if self.poisson_noise:
            p.data.iobs = np.random.poisson(p.data.iobs).astype(np.float32)
        return p

    def timestamp_increment(self, p):
        # Need to force re-loading iobs data to GPU
        p._timestamp_counter += 1
        p._cu_timestamp_counter = 0


class ApplyAmplitude(CUOperatorPtycho):
    """
    Apply the magnitude from observed intensities, keep the phase. Applies to a stack of N views.
    """

    def __init__(self, calc_llk=False, update_background=False, scale_in=1, scale_out=1, background_smooth_sigma=0):
        """

        :param calc_llk: if True, the log-likelihood will be calculated for this stack.
        :param update_background: if True, update the background. The new background is
            automatically updated once the last stack is processed.
        :param scale_in: a scale factor by which the input values should be multiplied, typically because of FFT
        :param scale_out: a scale factor by which the output values should be multiplied, typically because of FFT
        :param background_smooth_sigma: sigma for gaussian smoothing of the background
        """
        super(ApplyAmplitude, self).__init__()
        self.calc_llk = calc_llk
        self.update_background = update_background
        self.scale_in = np.float32(scale_in)
        self.scale_out = np.float32(scale_out)
        self.background_smooth_sigma = np.float32(background_smooth_sigma)

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        # TODO: use a single-pass reduction kernel to apply the amplitude and compute the LLK ?
        if self.calc_llk:
            p = LLK(scale=not (np.isclose(self.scale_in, 1))) * p
        nxy = np.int32(p._probe.shape[-2] * p._probe.shape[-1])
        nxystack = np.int32(nxy * pu.cu_stack_size)
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        i = p._cu_stack_i
        nb_psi = np.int32(p._cu_obs_v[i].npsi)
        if self.update_background:
            first_pass = np.int8(p._cu_stack_i == 0)
            pu.cu_projection_amplitude_background_mode(p._cu_obs_v[i].cu_obs[0], p._cu_psi, p._cu_background,
                                                       p._cu_background_new, nb_mode, nxy, nxystack, nb_psi,
                                                       first_pass, self.scale_in, self.scale_out)

        else:
            pu.cu_projection_amplitude(p._cu_obs_v[i].cu_obs[0], p._cu_psi, p._cu_background,
                                       nb_mode, nxy, nxystack, nb_psi, self.scale_in, self.scale_out)
        # Merge background update
        if self.update_background and p._cu_stack_i == (p._cu_stack_nb - 1):
            n = np.int32(len(p.data.posx))
            pu.cu_background_update_mode(p._cu_background, p._cu_background_new, n)
            if self.background_smooth_sigma > 0:
                # Smooth background
                if self.background_smooth_sigma > 3:
                    p = BackgroundFilter(self.background_smooth_sigma) * p
                else:
                    ny, nx = np.int32(p._background.shape[-2]), np.int32(p._background.shape[-1])
                    pu.gauss_convolf_16x(p._cu_background, self.background_smooth_sigma, nx, ny, np.int32(1),
                                         block=(16, 1, 1), grid=(1, int(ny), int(1)))
                    pu.gauss_convolf_16y(p._cu_background, self.background_smooth_sigma, nx, ny, np.int32(1),
                                         block=(1, 16, 1), grid=(int(nx), 1, int(1)))
        return p


class PropagateApplyAmplitude(CUOperatorPtycho):
    """
    Propagate to the detector plane (either in far or near field, perform the magnitude projection, and propagate
    back to the object plane. This applies to a stack of frames.
    """

    def __init__(self, calc_llk=False, update_background=False, background_smooth_sigma=0):
        """

        :param calc_llk: if True, calculate llk while in the detector plane.
        :param update_background: if True, update the background.
        :param background_smooth_sigma: sigma for the gaussian smoothing of the updated background
        """
        super(PropagateApplyAmplitude, self).__init__()
        self.calc_llk = calc_llk
        self.update_background = update_background
        self.background_smooth_sigma = background_smooth_sigma

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        if p.data.near_field:
            p = PropagateNearField(forward=False) * \
                ApplyAmplitude(calc_llk=self.calc_llk, update_background=self.update_background,
                               background_smooth_sigma=self.background_smooth_sigma) \
                * PropagateNearField(forward=True) * p
        else:
            s = self.processing_unit.fft_scale(p._cu_psi.shape, ndim=2)
            p = IFT(scale=False) * \
                ApplyAmplitude(calc_llk=self.calc_llk, update_background=self.update_background,
                               scale_in=s[0], scale_out=s[1],
                               background_smooth_sigma=self.background_smooth_sigma) * \
                FT(scale=False) * p
        return p


class LLK(CUOperatorPtycho):
    """
    Log-likelihood reduction kernel. Can only be used when Psi is in diffraction space.
    This is a reduction operator - it will write llk as an argument in the Ptycho object, and return the object.
    If _cu_stack_i==0, the llk is re-initialized. Otherwise it is added to the current value.

    The LLK can be calculated directly from object and probe using: p = LoopStack(LLK() * FT() * ObjProbe2Psi()) * p
    """

    def __init__(self, scale=False):
        """

        :param scale: if True, will scale the calculated amplitude to calculate the log-likelihood. The amplitudes are
                      left unchanged.
        """
        super(LLK, self).__init__()
        self.scale = scale

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cu_stack_i
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        nb_psi = p._cu_obs_v[i].npsi
        nxy = np.int32(p._probe.shape[-2] * p._probe.shape[-1])
        nxystack = np.int32(self.processing_unit.cu_stack_size * nxy)

        s = np.float32(1)
        if self.scale and not p.data.near_field:
            s = pu.fft_scale(p._cu_psi.shape, ndim=2)[0]  # Compensates for FFT scaling

        llk = self.processing_unit.cu_llk(p._cu_obs_v[i].cu_obs[:nb_psi], p._cu_psi, p._cu_background,
                                          nb_mode, nxy, nxystack, s).get()
        if p._cu_stack_i == 0:
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


class Psi2Obj(CUOperatorPtycho):
    """
    Computes updated Obj(r) contributions from Psi and Probe(r-r_j), for a stack of N probe positions.
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cu_stack_i
        # p.print("Psi2Obj(), i=%d"%(i))
        first_pass = np.int8(i == 0)
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        npsi = np.int32(p._cu_obs_v[i].npsi)
        i0 = p._cu_obs_v[i].i
        interp = p._interpolation
        if p.data.near_field:
            f = np.float32(0)
            padding = np.int32(p.data.padding)
        else:
            f = np.float32(-np.pi / (p.data.wavelength * p.data.detector_distance))
            padding = np.int32(0)

        # Use atomic operations for object update
        if i == 0:
            if has_attr_not_none(p, '_cu_obj_new') is False:
                p._cu_obj_new = cua.zeros((nb_obj, nyo, nxo), dtype=np.complex64,
                                          allocator=pu.get_memory_pool().allocate)
            elif p._cu_obj_new.size != nb_obj * nyo * nxo:
                p._cu_obj_new = cua.zeros((nb_obj, nyo, nxo), dtype=np.complex64,
                                          allocator=pu.get_memory_pool().allocate)
            else:
                p._cu_obj_new.fill(np.complex64(0))

            if has_attr_not_none(p, '_cu_obj_norm') is False:
                p._cu_obj_norm = cua.zeros((nyo, nxo), dtype=np.float32, allocator=pu.get_memory_pool().allocate)
            elif p._cu_obj_norm.size != nyo * nxo:
                p._cu_obj_norm = cua.zeros((nyo, nxo), dtype=np.float32, allocator=pu.get_memory_pool().allocate)
            else:
                p._cu_obj_norm.fill(np.float32(0))
        pu.cu_psi_to_obj_atomic(p._cu_psi[0, 0, 0], p._cu_obj_new, p._cu_probe, p._cu_obj_norm,
                                p._cu_cx[i0:i0 + npsi], p._cu_cy[i0:i0 + npsi], p.pixel_size_object, f,
                                pu.cu_stack_size,
                                nx, ny, nxo, nyo, nb_obj, nb_probe, npsi, padding, interp)
        return p


class Psi2PosShift(CUOperatorPtycho):
    """
    Computes scan position shifts, by comparing the updated Psi array to object*probe, for a stack of frames.
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cu_stack_i
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        npsi = np.int32(p._cu_obs_v[i].npsi)
        i0 = p._cu_obs_v[i].i
        interp = np.int8(p._interpolation)
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(-np.pi / (p.data.wavelength * p.data.detector_distance))

        if i == 0:
            p._cu_dxy = cua.empty(dtype=my_float4, shape=(len(p.data.posx)))
        pu.cu_psi2pos(p._cu_psi, p._cu_obj, p._cu_probe, p._cu_cx[i0:i0 + npsi], p._cu_cy[i0:i0 + npsi],
                      p.pixel_size_object, f, nx, ny, nxo, nyo, interp, p._cu_dxy[i0:i0 + npsi], block=(128, 1, 1),
                      grid=(int(npsi), 1, 1))
        return p


class Psi2PosMerge(CUOperatorPtycho):
    """
    Merge scan position shifts, once the entire stack of frames has been processed.
    """

    def __init__(self, multiplier=1, max_displ=2, min_displ=0, threshold=0., save_position_history=False):
        """

        :param multiplier: the computed displacements are multiplied by this value
        :param max_displ: the displacements (at each iteration) are capped to this value (in pixels),
            after applying the multiplier.
        :param min_displ: the displacements (at each iteration) are ignored if smaller
            than this value (in pixels), after applying the multiplier.
        :param threshold: if the integrated grad_obj*probe along dx or dy is
            smaller than (grad_obj*probe).mean()*threshold, then the shift is ignored.
            This allows to prevent shifts where there is little contrast in
            the object. It applies independently to x and y.
        :param save_position_history: if True, save the position history in the ptycho object (slow, for debugging)
        """
        super(Psi2PosMerge, self).__init__()
        self.multiplier = np.float32(multiplier)
        self.max_displ = np.float32(max_displ)
        self.min_displ = np.float32(min_displ)
        self.threshold = np.float32(threshold)
        self.save_position_history = save_position_history

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """

        n = np.int32(len(p.data.posx))
        if self.save_position_history and (has_attr_not_none(p, 'position_history') is False):
            p.position_history = []
            x, y = p._cu_cx.get(), p._cu_cy.get()
            for i in range(n):
                p.position_history.append([(p.cycle, x[i], y[i])])

        pu = self.processing_unit

        pu.cu_psi2pos_merge(p._cu_dxy, p._cu_cx, p._cu_cy, self.multiplier,
                            self.max_displ, self.min_displ, self.threshold, n, block=(128, 1, 1),
                            grid=(1, 1, 1))

        if self.save_position_history:
            x, y = p._cu_cx.get(), p._cu_cy.get()
            for i in range(n):
                p.position_history[i].append((p.cycle, x[i], y[i]))

        return p


class Psi2Probe(CUOperatorPtycho):
    """
    Computes updated Probe contributions from Psi and Obj, for a stack of N probe positions.
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cu_stack_i
        npsi = np.int32(p._cu_obs_v[i].npsi)
        i0 = p._cu_obs_v[i].i
        first_pass = np.int8(i == 0)
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        interp = p._interpolation
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(-np.pi / (p.data.wavelength * p.data.detector_distance))

        if i == 0:
            if has_attr_not_none(p, '_cu_probe_new') is False:
                p._cu_probe_new = cua.empty((nb_probe, ny, nx), dtype=np.complex64,
                                            allocator=pu.get_memory_pool().allocate)
            elif p._cu_probe_new.size != p._cu_probe.size:
                p._cu_probe_new = cua.empty((nb_probe, ny, nx), dtype=np.complex64,
                                            allocator=pu.get_memory_pool().allocate)
            if has_attr_not_none(p, '_cu_probe_norm') is False:
                p._cu_probe_norm = cua.empty((ny, nx), dtype=np.float32, allocator=pu.get_memory_pool().allocate)
            elif p._cu_probe_norm.size != ny * nx:
                p._cu_probe_norm = cua.empty((ny, nx), dtype=np.float32, allocator=pu.get_memory_pool().allocate)

        # First argument is p._cu_psi[0] because the kernel will calculate the projection for all object and probe modes
        # and the full stack of frames.
        pu.cu_psi_to_probe(p._cu_psi[0, 0, 0], p._cu_obj, p._cu_probe_new, p._cu_probe_norm,
                           p._cu_cx[i0:i0 + npsi], p._cu_cy[i0:i0 + npsi],
                           p.pixel_size_object, f, first_pass,
                           p._cu_obs_v[i].npsi, self.processing_unit.cu_stack_size,
                           nx, ny, nxo, nyo, nb_obj, nb_probe, interp)

        return p


class Psi2ObjMerge(CUOperatorPtycho):
    """
    Call this when all stack of probe positions have been processed, and the final update of the object can
    be calculated. Temporary arrays are cleaned up
    """

    def __init__(self, inertia=1e-2, smooth_sigma=0):
        """

        :param reg: object inertia
        :param smooth_sigma: if > 0, the previous object array (used for inertia) will be convolved
                             by a gaussian with this sigma.
        """
        super(Psi2ObjMerge, self).__init__()
        self.inertia = np.float32(inertia)
        self.smooth_sigma = np.float32(smooth_sigma)

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        nb_obj = np.int32(p._obj.shape[0])
        nxo = np.int32(p._obj.shape[-1])
        nyo = np.int32(p._obj.shape[-2])
        nxyo = np.int32(nxo * nyo)
        if self.smooth_sigma > 8:
            pu.gauss_convol_complex_64x(p._cu_obj, self.smooth_sigma, nxo, nyo, nb_obj, block=(64, 1, 1),
                                        grid=(1, int(nyo), int(nb_obj)))
            pu.gauss_convol_complex_64y(p._cu_obj, self.smooth_sigma, nxo, nyo, nb_obj, block=(1, 64, 1),
                                        grid=(int(nxo), 1, int(nb_obj)))
        elif self.smooth_sigma > 4:
            pu.gauss_convol_complex_32x(p._cu_obj, self.smooth_sigma, nxo, nyo, nb_obj, block=(32, 1, 1),
                                        grid=(1, int(nyo), int(nb_obj)))
            pu.gauss_convol_complex_32y(p._cu_obj, self.smooth_sigma, nxo, nyo, nb_obj, block=(1, 32, 1),
                                        grid=(int(nxo), 1, int(nb_obj)))
        elif self.smooth_sigma > 0.1:
            pu.gauss_convol_complex_16x(p._cu_obj, self.smooth_sigma, nxo, nyo, nb_obj, block=(16, 1, 1),
                                        grid=(1, int(nyo), int(nb_obj)))
            pu.gauss_convol_complex_16y(p._cu_obj, self.smooth_sigma, nxo, nyo, nb_obj, block=(1, 16, 1),
                                        grid=(int(nxo), 1, int(nb_obj)))

        regmax = pu.cu_max_red(p._cu_obj_norm, allocator=pu.get_memory_pool().allocate)
        if p._cu_obj_zero_phase_mask is None:
            pu.cu_obj_norm(p._cu_obj_norm, p._cu_obj_new, p._cu_obj, regmax, self.inertia, nxyo, nb_obj)
        else:
            pu.cu_obj_norm_zero_phase_mask_n(p._cu_obj_norm, p._cu_obj_new, p._cu_obj, p._cu_obj_zero_phase_mask,
                                             regmax, self.inertia, nxyo, nb_obj,
                                             pu.cu_stack_size)

        # Clean up ?
        del p._cu_obj_norm, p._cu_obj_new

        return p


class Psi2ProbeMerge(CUOperatorPtycho):
    """
    Call this when all stack of probe positions have been processed, and the final update of the probe can
    be calculated. Temporary arrays are cleaned up.
    """

    def __init__(self, inertia=1e-3, smooth_sigma=0):
        """
        :param inertia: a regularisation factor to set the object inertia.
        :param smooth_sigma: if > 0, the previous object array (used for inertia) will be convolved
                             by a gaussian with this sigma.
        """
        super(Psi2ProbeMerge, self).__init__()
        self.inertia = np.float32(inertia)
        self.smooth_sigma = np.float32(smooth_sigma)

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        nb_probe = np.int32(p._probe.shape[0])
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nxy = np.int32(nx * ny)

        if self.smooth_sigma > 8:
            pu.gauss_convol_complex_64x(p._cu_probe, self.smooth_sigma, nx, ny, nb_probe, block=(64, 1, 1),
                                        grid=(1, int(ny), int(nb_probe)))
            pu.gauss_convol_complex_64y(p._cu_probe, self.smooth_sigma, nx, ny, nb_probe, block=(1, 64, 1),
                                        grid=(int(nx), 1, int(nb_probe)))
        elif self.smooth_sigma > 4:
            pu.gauss_convol_complex_32x(p._cu_probe, self.smooth_sigma, nx, ny, nb_probe, block=(32, 1, 1),
                                        grid=(1, int(ny), int(nb_probe)))
            pu.gauss_convol_complex_32y(p._cu_probe, self.smooth_sigma, nx, ny, nb_probe, block=(1, 32, 1),
                                        grid=(int(nx), 1, int(nb_probe)))
        elif self.smooth_sigma > 0.1:
            pu.gauss_convol_complex_16x(p._cu_probe, self.smooth_sigma, nx, ny, nb_probe, block=(16, 1, 1),
                                        grid=(1, int(ny), int(nb_probe)))
            pu.gauss_convol_complex_16y(p._cu_probe, self.smooth_sigma, nx, ny, nb_probe, block=(1, 16, 1),
                                        grid=(int(nx), 1, int(nb_probe)))

        # Don't get() the max value, to avoid D2H memory transfer (about 80 us faster..)
        # reg = np.float32(float(cua.max(p._cu_probe_norm).get()) * self.reg)

        # Try not to use gpuarray.max(). It re-generates the kernel ? Tiny improvement
        # regmax = cua.max(p._cu_probe_norm)

        regmax = pu.cu_max_red(p._cu_probe_norm, allocator=pu.get_memory_pool().allocate)

        pu.cu_obj_norm(p._cu_probe_norm, p._cu_probe_new, p._cu_probe, regmax, self.inertia, nxy, nb_probe)

        # Clean up ? No - there is a significant overhead
        # del p._cu_probe_norm, p._cu_probe_new
        return p


class AP(CUOperatorPtycho):
    """
    Perform a complete Alternating Projection cycle:
    - forward all object*probe views to Fourier space and apply the observed amplitude
    - back-project to object space and project onto (probe, object)
    - update background optionally
    """

    def __init__(self, update_object=True, update_probe=False, update_background=False, floating_intensity=False,
                 nb_cycle=1, calc_llk=False, show_obj_probe=False, fig_num=-1, obj_smooth_sigma=0, obj_inertia=0.01,
                 probe_smooth_sigma=0, probe_inertia=0.001, update_pos=False, pos_mult=1,
                 pos_max_shift=2, pos_min_shift=0, pos_threshold=0.05, pos_history=False, zero_phase_ramp=True,
                 background_smooth_sigma=0):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param update_background: update background ?
        :param floating_intensity: optimise floating intensity scale factor
        :param nb_cycle: number of cycles to perform. Equivalent to AP(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        :param obj_smooth_sigma: if > 0, the previous object array (used for inertia) will convoluted by a gaussian
                                 array of this standard deviation.
        :param obj_inertia: the updated object retains this relative amount of the previous object.
        :param probe_smooth_sigma: if > 0, the previous probe array (used for inertia) will convoluted by a gaussian
                                   array of this standard deviation.
        :param probe_inertia: the updated probe retains this relative amount of the previous probe.
        :param update_pos: positive integer, if >0, update positions every 'update_pos' cycle. Note that object and
                           probe are not updated during the same cycle as positions. Still experimental, recommended
                           are 5, 10 (default=False or 0, positions are not updated).
        :param pos_max_shift: maximum allowed shift (in pixels) per scan position (default=2)
        :param pos_min_shift: minimum required shift (in pixels) per scan position (default=0)
        :param pos_threshold: if the integrated grad_obj*probe along dx or dy is
            smaller than (grad_obj*probe).mean()*threshold, then the shift is ignored.
            This allows to prevent position shifts where there is little contrast in
            the object. It applies independently to x and y.
        :param pos_mult: multiply the calculated position shifts by this value. Useful since the calculated
                                shifts usually are a fraction of the actual shift.
        :param pos_history: if True, save the position history (for debugging)
        :param zero_phase_ramp: if True, the conjugate phase ramp in the object and probe will be removed
                                by centring the FT of the probe, at the end and before every display.
                                Ignored for near field.
        :param background_smooth_sigma: gaussian convolution parameter for the background update
        """
        super(AP, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.floating_intensity = floating_intensity  # TODO
        self.nb_cycle = nb_cycle
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.obj_smooth_sigma = obj_smooth_sigma
        self.obj_inertia = obj_inertia
        self.probe_smooth_sigma = probe_smooth_sigma
        self.probe_inertia = probe_inertia
        self.update_pos = update_pos
        self.pos_max_shift = pos_max_shift
        self.pos_min_shift = pos_min_shift
        self.pos_threshold = pos_threshold
        self.pos_mult = pos_mult
        self.pos_history = pos_history
        self.zero_phase_ramp = zero_phase_ramp
        self.background_smooth_sigma = np.float32(background_smooth_sigma)

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new AP operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return AP(update_object=self.update_object, update_probe=self.update_probe,
                  update_background=self.update_background, floating_intensity=self.floating_intensity,
                  nb_cycle=self.nb_cycle * n, calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe,
                  fig_num=self.fig_num, obj_smooth_sigma=self.obj_smooth_sigma, obj_inertia=self.obj_inertia,
                  probe_smooth_sigma=self.probe_smooth_sigma, probe_inertia=self.probe_inertia,
                  update_pos=self.update_pos, pos_max_shift=self.pos_max_shift, pos_min_shift=self.pos_min_shift,
                  pos_threshold=self.pos_threshold, pos_mult=self.pos_mult, pos_history=self.pos_history,
                  zero_phase_ramp=self.zero_phase_ramp,
                  background_smooth_sigma=self.background_smooth_sigma)

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        if self.update_background:
            p._cu_background_new = cua.empty_like(p._cu_background)

        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            update_pos = False
            if self.update_pos:
                if ic % int(self.update_pos) == 0:
                    update_pos = True

            ops = PropagateApplyAmplitude(calc_llk=calc_llk, update_background=self.update_background,
                                          background_smooth_sigma=self.background_smooth_sigma) * ObjProbe2Psi()
            if self.update_object:
                ops = Psi2Obj() * ops
            if self.update_probe:
                ops = Psi2Probe() * ops
            if update_pos:
                ops = Psi2PosShift() * ops

            p = LoopStack(ops) * p

            if update_pos:
                # Do we fully update the positions before object and probe update ? Given the small
                # shifts for each cycle, this should not significantly affect the object & probe update
                p = Psi2PosMerge(multiplier=self.pos_mult, max_displ=self.pos_max_shift,
                                 min_displ=self.pos_min_shift, threshold=self.pos_threshold,
                                 save_position_history=self.pos_history) * p

            if self.update_object:
                p = Psi2ObjMerge(smooth_sigma=self.obj_smooth_sigma, inertia=self.obj_inertia) * p
            if self.update_probe:
                p = Psi2ProbeMerge(smooth_sigma=self.probe_smooth_sigma, inertia=self.probe_inertia) * p

            # self.center_probe_n = 5
            # self.center_probe_max_shift = 5
            # if self.center_probe_n > 0 and p.data.near_field is False:
            #     if (ic % self.center_probe_n) == 0:
            #         p = CenterObjProbe(max_shift=self.center_probe_max_shift) * p

            if calc_llk:
                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=self.update_pos,
                                 algorithm='AP', verbose=True)
            else:
                p.history.insert(p.cycle, update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=self.update_pos, algorithm='AP',
                                 verbose=False)
            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    s = algo_string('AP', p, self.update_object, self.update_probe, self.update_background,
                                    self.update_pos)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.get_llk('poisson'))
                    if self.zero_phase_ramp:
                        p = ZeroPhaseRamp(obj=True) * p
                    p = ShowObjProbe(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1
        if self.update_background:
            del p._cu_background_new
        if self.zero_phase_ramp and not self.show_obj_probe:
            p = ZeroPhaseRamp(obj=True) * p
        return p


class DM1(CUOperatorPtycho):
    """
    Equivalent to operator: 2 * ObjProbe2Psi() - 1
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cu_stack_i
        npsi = p._cu_obs_v[i].npsi
        i0 = p._cu_obs_v[i].i
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        interp = p._interpolation
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))
        pu.cu_2object_probe_psi_dm1(p._cu_psi[0, 0, 0], p._cu_obj, p._cu_probe,
                                    p._cu_cx[i0:i0 + npsi], p._cu_cy[i0:i0 + npsi],
                                    p.pixel_size_object, f,
                                    p._cu_obs_v[i].npsi, pu.cu_stack_size,
                                    nx, ny, nxo, nyo, nb_obj, nb_probe, interp)
        return p


class DM2(CUOperatorPtycho):
    """
    # Psi(n+1) = Psi(n) - P*O + Psi_fourier

    This operator assumes that Psi_fourier is the current Psi, and that Psi(n) is in p._cu_psi_v

    On output Psi(n+1) is the current Psi, and Psi_fourier has been swapped to p._cu_psi_v
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cu_stack_i
        npsi = p._cu_obs_v[i].npsi
        i0 = p._cu_obs_v[i].i
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        interp = p._interpolation
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))
        # Swap p._cu_psi_v_copy = Psi(n) with p._cu_psi = Psi_fourier
        p._cu_psi_copy, p._cu_psi = p._cu_psi, p._cu_psi_copy
        pu.cu_2object_probe_psi_dm2(p._cu_psi[0, 0, 0], p._cu_psi_copy, p._cu_obj, p._cu_probe,
                                    p._cu_cx[i0:i0 + npsi], p._cu_cy[i0:i0 + npsi],
                                    p.pixel_size_object, f,
                                    p._cu_obs_v[i].npsi, pu.cu_stack_size,
                                    nx, ny, nxo, nyo, nb_obj, nb_probe, interp)
        return p


class DM(CUOperatorPtycho):
    """
    Operator to perform a complete Difference Map cycle, updating the Psi views for all stack of frames,
    as well as updating the object and/or probe.
    """

    def __init__(self, update_object=True, update_probe=True, update_background=False, nb_cycle=1,
                 calc_llk=False, show_obj_probe=False,
                 fig_num=-1, obj_smooth_sigma=0, obj_inertia=0.01, probe_smooth_sigma=0, probe_inertia=0.001,
                 center_probe_n=0, center_probe_max_shift=5, loop_obj_probe=1, update_pos=False, pos_mult=1,
                 pos_max_shift=2, pos_min_shift=0, pos_threshold=0.2, pos_history=False, zero_phase_ramp=True,
                 background_smooth_sigma=0):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param update_background: update background ?
        :param nb_cycle: number of cycles to perform. Equivalent to DM(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        :param obj_smooth_sigma: if > 0, the previous object array (used for inertia) will convoluted by a gaussian
                                 array of this standard deviation.
        :param obj_inertia: the updated object retains this relative amount of the previous object.
        :param probe_smooth_sigma: if > 0, the previous probe array (used for inertia) will convoluted by a gaussian
                                   array of this standard deviation.
        :param probe_inertia: the updated probe retains this relative amount of the previous probe.
        :param center_probe_n: test the probe every N cycle for deviation from the center. If deviation is larger
                               than center_probe_max_shift, probe and object are shifted to correct. If 0 (the default),
                               the probe centering is never calculated.
        :param center_probe_max_shift: maximum deviation from the center (in pixels) to trigger a position correction
        :param loop_obj_probe: when both object and probe are updated, it can be more stable to loop the object
                               and probe update for a more stable optimisation, but slower.
        :param update_pos: positive integer, if >0, update positions every 'update_pos' cycle. Note that object and
                           probe are not updated during the same cycle as positions. Still experimental, recommended
                           are 5, 10 (default=False, positions are not updated).
        :param pos_max_shift: maximum allowed shift (in pixels) per scan position (default=2)
        :param pos_min_shift: minimum required shift (in pixels) per scan position (default=0)
        :param pos_threshold: if the integrated grad_obj*probe along dx or dy is
            smaller than (grad_obj*probe).mean()*threshold, then the shift is ignored.
            This allows to prevent position shifts where there is little contrast in
            the object. It applies independently to x and y.
        :param pos_mult: multiply the calculated position shifts by this value. Useful since the calculated
                                shifts usually are a fraction of the actual shift.
        :param pos_history: if True, save the position history (for debugging)
        :param zero_phase_ramp: if True, the conjugate phase ramp in the object and probe will be removed
                                by centring the FT of the probe, at the end and before every display.
                                Ignored for near field.
        :param background_smooth_sigma: gaussian convolution parameter for the background update
        """
        super(DM, self).__init__()
        self.nb_cycle = nb_cycle
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.obj_smooth_sigma = obj_smooth_sigma
        self.obj_inertia = obj_inertia
        self.probe_smooth_sigma = probe_smooth_sigma
        self.probe_inertia = probe_inertia
        self.center_probe_n = center_probe_n
        self.center_probe_max_shift = center_probe_max_shift
        self.update_pos = update_pos
        self.pos_max_shift = pos_max_shift
        self.pos_min_shift = pos_min_shift
        self.pos_threshold = pos_threshold
        self.pos_mult = pos_mult
        self.pos_history = pos_history
        self.loop_obj_probe = loop_obj_probe
        self.zero_phase_ramp = zero_phase_ramp
        self.background_smooth_sigma = np.float32(background_smooth_sigma)

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new DM operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return DM(update_object=self.update_object, update_probe=self.update_probe,
                  update_background=self.update_background, nb_cycle=self.nb_cycle * n,
                  calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe, fig_num=self.fig_num,
                  obj_smooth_sigma=self.obj_smooth_sigma, obj_inertia=self.obj_inertia,
                  probe_smooth_sigma=self.probe_smooth_sigma, probe_inertia=self.probe_inertia,
                  center_probe_n=self.center_probe_n, center_probe_max_shift=self.center_probe_max_shift,
                  loop_obj_probe=self.loop_obj_probe, update_pos=self.update_pos,
                  pos_max_shift=self.pos_max_shift, pos_min_shift=self.pos_min_shift,
                  pos_threshold=self.pos_threshold, pos_mult=self.pos_mult,
                  pos_history=self.pos_history, zero_phase_ramp=self.zero_phase_ramp,
                  background_smooth_sigma=self.background_smooth_sigma)

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        # First loop to get a starting Psi (note that all Psi are multiplied by the quadratic phase factor)
        p = LoopStack(ObjProbe2Psi(), keep_psi=True) * p

        # We could use instead of DM1 and DM2 operators:
        # op_dm1 = 2 * ObjProbe2Psi() - 1
        # op_dm2 = 1 - ObjProbe2Psi() + FourierApplyAmplitude() * op_dm1
        # But this would use 3 copies of the whole Psi stack - too much memory ?
        # TODO: check if memory usage would be that bad, or if it's possible the psi storage only applies
        #  to the current psi array

        if self.update_background:
            p._cu_background_new = cua.empty_like(p._cu_background)

        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            update_pos = False
            if self.update_pos:
                if ic % int(self.update_pos) == 0:
                    update_pos = True

            if True:
                ops = PropagateApplyAmplitude(update_background=self.update_background,
                                              background_smooth_sigma=self.background_smooth_sigma) * DM1()
                if update_pos:
                    ops = DM2() * Psi2PosShift() * ops
                else:
                    ops = DM2() * ops
                p = LoopStack(ops, keep_psi=True, copy=True) * p

                if update_pos:
                    p = Psi2PosMerge(multiplier=self.pos_mult, max_displ=self.pos_max_shift,
                                     min_displ=self.pos_min_shift, threshold=self.pos_threshold,
                                     save_position_history=self.pos_history) * p
                elif True:
                    # Loop the object and probe update if both are done at the same time. Slow, more stable ?
                    nb_loop_update_obj_probe = 1
                    if self.update_probe and self.update_object:
                        nb_loop_update_obj_probe = self.loop_obj_probe

                    for i in range(nb_loop_update_obj_probe):
                        if self.update_object:
                            p = Psi2ObjMerge(smooth_sigma=self.obj_smooth_sigma,
                                             inertia=self.obj_inertia) * LoopStack(Psi2Obj(), keep_psi=True) * p
                        if self.update_probe:
                            p = Psi2ProbeMerge(smooth_sigma=self.probe_smooth_sigma,
                                               inertia=self.probe_inertia) * LoopStack(Psi2Probe(), keep_psi=True) * p
                else:
                    # TODO: updating probe and object at the same time does not work as in AP. Why ?
                    # Probably due to a scaling issue, as Psi is not a direct back-propagation but the result of DM2
                    ops = 1
                    if self.update_object:
                        ops = Psi2Obj() * ops
                    if self.update_probe:
                        ops = Psi2Probe() * ops

                    p = LoopStack(ops, keep_psi=True) * p

                    if self.update_object:
                        p = Psi2ObjMerge(smooth_sigma=self.obj_smooth_sigma, inertia=self.obj_inertia) * p
                    if self.update_probe:
                        p = Psi2ProbeMerge(smooth_sigma=self.probe_smooth_sigma, inertia=self.probe_inertia) * p
            else:
                # Update obj and probe immediately after back-propagation, before DM2 ?
                # Does not seem to give very good results
                ops = PropagateApplyAmplitude() * DM1()
                if self.update_object:
                    ops = Psi2Obj() * ops
                if self.update_probe:
                    ops = Psi2Probe() * ops

                p = LoopStack(DM2() * ops, keep_psi=True, copy=True) * p
                if self.update_object:
                    p = Psi2ObjMerge(smooth_sigma=self.obj_smooth_sigma, inertia=self.obj_inertia) * p
                if self.update_probe:
                    p = Psi2ProbeMerge(smooth_sigma=self.probe_smooth_sigma, inertia=self.probe_inertia) * p

            if self.center_probe_n > 0 and p.data.near_field is False:
                if (ic % self.center_probe_n) == 0:
                    p = CenterObjProbe(max_shift=self.center_probe_max_shift) * p
            if calc_llk:
                # Keep a copy of current Psi
                cu_psi0 = p._cu_psi.copy()
                # We need to perform a loop for LLK as the DM2 loop is on (2*PO-I), not the current PO estimate
                if p.data.near_field:
                    p = LoopStack(LLK() * PropagateNearField() * ObjProbe2Psi()) * p
                else:
                    p = LoopStack(LLK(scale=True) * FT(scale=False) * ObjProbe2Psi()) * p

                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background,
                                 update_pos=self.update_pos, algorithm='DM',
                                 verbose=True)
                # TODO: find a   better place to do this rescaling, only useful to avoid obj/probe divergence
                p = ScaleObjProbe(absolute=False) * p
                # Restore correct Psi
                p._cu_psi = cu_psi0
            else:
                p.history.insert(p.cycle, update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=False, algorithm='DM',
                                 verbose=False)

            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    s = algo_string('DM', p, self.update_object, self.update_probe,
                                    update_background=self.update_background, update_pos=update_pos)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.get_llk('poisson'))
                    if self.zero_phase_ramp:
                        p = ZeroPhaseRamp(obj=False) * p
                    p = ShowObjProbe(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1

        # Free some memory
        p._cu_psi_v = {}
        if self.update_background:
            del p._cu_background_new
        gc.collect()
        if self.zero_phase_ramp and not self.show_obj_probe:
            p = ZeroPhaseRamp(obj=False) * p
        return p


class _Grad(CUOperatorPtycho):
    """
    Operator to compute the object and/or probe and/or background gradient corresponding to the current stack.
    """

    def __init__(self, update_object=True, update_probe=False, update_background=False, calc_llk=False):
        """
        :param update_object: compute gradient for the object ?
        :param update_probe: compute gradient for the probe ?
        :param update_background: compute gradient for the background ?
        :param calc_llk: calculate llk while in Fourier space
        """
        super(_Grad, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.calc_llk = calc_llk

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cu_stack_i
        npsi = np.int32(p._cu_obs_v[i].npsi)
        i0 = p._cu_obs_v[i].i
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nb_mode = np.int32(nb_obj * nb_probe)
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        first_pass = np.int8(i == 0)
        nxy = np.int32(ny * nx)
        nxystack = np.int32(pu.cu_stack_size * nxy)
        hann_filter = np.int8(1)
        if p.data.near_field:
            f = np.float32(0)
            hann_filter = np.int8(0)
        else:
            f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))

        # Obj * Probe
        p = ObjProbe2Psi() * p

        s1, s2 = np.float32(1), np.float32(1)  # FFT scale, if needed
        # To detector plane
        if p.data.near_field:
            p = PropagateNearField() * p
        else:
            p = FT(scale=False) * p
            s1, s2 = pu.fft_scale(p._cu_psi.shape, ndim=2)  # Compensates for FFT scaling

        if self.calc_llk:
            p = LLK(scale=True) * p

        # Calculate Psi.conj() * (1-Iobs/I_calc) [for Poisson Gradient] & background gradient
        # TODO: different noise models
        pu.cu_grad_poisson_fourier(p._cu_obs_v[i].cu_obs[0], p._cu_psi, p._cu_background, p._cu_background_grad,
                                   nb_mode, nx, ny, nxystack, npsi, hann_filter, s1, s2)

        if p.data.near_field:
            p = PropagateNearField(forward=False) * p
        else:
            p = IFT(scale=False) * p

        if self.update_object:
            if False:  # TODO: slower, but yields better results ? No, there was a wrong sign in object grad...
                for ii in range(p._cu_obs_v[i].npsi):
                    pu.cu_psi_to_obj_grad(p._cu_psi[0, 0, ii], p._cu_obj_grad, p._cu_probe,
                                          p._cu_obs_v[i].x[ii], p._cu_obs_v[i].y[ii],
                                          p.pixel_size_object, f, pu.cu_stack_size,
                                          nx, ny, nxo, nyo, nb_obj, nb_probe, False)
            else:
                # Use atomic operations to avoid looping over frames !
                pu.cu_psi_to_obj_grad_atomic(p._cu_psi[0, 0, 0], p._cu_obj_grad, p._cu_probe, p._cu_cx[i0:i0 + npsi],
                                             p._cu_cy[i0:i0 + npsi], p.pixel_size_object, f, pu.cu_stack_size, nx, ny,
                                             nxo, nyo, nb_obj, nb_probe, npsi, p._interpolation)
        if self.update_probe:
            pu.cu_psi_to_probe_grad(p._cu_psi[0, 0, 0], p._cu_probe_grad, p._cu_obj,
                                    p._cu_cx[i0:i0 + npsi], p._cu_cy[i0:i0 + npsi],
                                    p.pixel_size_object, f, first_pass,
                                    npsi, pu.cu_stack_size,
                                    nx, ny, nxo, nyo, nb_obj, nb_probe, p._interpolation)
        return p


class Grad(CUOperatorPtycho):
    """
    Operator to compute the object and/or probe and/or background gradient. The gradient is stored
    in the ptycho object. It is assumed that the GPU gradient arrays have been already created, normally
    by the calling ML operator.
    """

    def __init__(self, update_object=True, update_probe=False, update_background=False,
                 reg_fac_obj=0, reg_fac_probe=0, calc_llk=False):
        """

        :param update_object: compute gradient for the object ?
        :param update_probe: compute gradient for the probe ?
        :param update_background: compute gradient for the background ?
        :param calc_llk: calculate llk while in Fourier space
        """
        super(Grad, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.calc_llk = calc_llk
        self.reg_fac_obj = reg_fac_obj
        self.reg_fac_probe = reg_fac_probe

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        if self.update_object:
            p._cu_obj_grad.fill(np.complex64(0))

        p = LoopStack(_Grad(update_object=self.update_object, update_probe=self.update_probe,
                            update_background=self.update_background, calc_llk=self.calc_llk)) * p

        if self.reg_fac_obj is not None:
            reg_fac_obj = np.float32(p.reg_fac_scale_obj * self.reg_fac_obj)
        else:
            reg_fac_obj = 0
        if self.reg_fac_probe is not None:
            reg_fac_probe = np.float32(p.reg_fac_scale_probe * self.reg_fac_probe)
        else:
            reg_fac_probe = 0
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        pu = self.processing_unit

        if self.update_object and reg_fac_obj > 0:
            # Regularisation contribution to the object gradient
            pu.cu_reg_grad(p._cu_obj_grad, p._cu_obj, reg_fac_obj, nxo, nyo)

        if self.update_probe and reg_fac_probe > 0:
            # Regularisation contribution to the probe gradient
            pu.cu_reg_grad(p._cu_probe_grad, p._cu_probe, reg_fac_probe, nx, ny)

        return p


class _CGGamma(CUOperatorPtycho):
    """
    Operator to compute the conjugate gradient gamma contribution to the current stack.
    """

    def __init__(self, update_background=False):
        """
        :param update_background: if updating the background ?
        """
        super(_CGGamma, self).__init__()
        self.update_background = update_background

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cu_stack_i
        npsi = p._cu_obs_v[i].npsi
        i0 = p._cu_obs_v[i].i
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nb_mode = np.int32(nb_obj * nb_probe)
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        nb_psi = np.int32(p._cu_obs_v[i].npsi)
        nxy = np.int32(ny * nx)
        nxystack = np.int32(pu.cu_stack_size * nxy)
        if p.data.near_field:
            f = np.float32(0)
            s = np.float32(1)
        else:
            f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))
            s = pu.fft_scale(p._cu_psi.shape, ndim=2)[0]

        for cupsi, cuobj, cuprobe in zip([p._cu_PO, p._cu_PdO, p._cu_dPO, p._cu_dPdO],
                                         [p._cu_obj, p._cu_obj_dir, p._cu_obj, p._cu_obj_dir],
                                         [p._cu_probe, p._cu_probe, p._cu_probe_dir, p._cu_probe_dir]):

            pu.cu_object_probe_mult(cupsi[0, 0, 0], cuobj, cuprobe,
                                    p._cu_cx[i0:i0 + npsi], p._cu_cy[i0:i0 + npsi],
                                    p.pixel_size_object, f, nb_psi, pu.cu_stack_size, nx, ny, nxo, nyo,
                                    nb_obj, nb_probe, p._interpolation)
            # switch cupsi and p._cu_psi for propagation
            cupsi, p._cu_psi = p._cu_psi, cupsi
            if p.data.near_field:
                p = PropagateNearField(forward=True) * p
            else:
                # Don't use scale here, but use scale_fft in cg_poisson_gamma_red kernel
                p = FT(scale=False) * p

        tmp = self.processing_unit._cu_cg_poisson_gamma_red(p._cu_obs_v[i].cu_obs[0], p._cu_PO, p._cu_PdO,
                                                            p._cu_dPO, p._cu_dPdO, p._cu_background,
                                                            p._cu_background_dir, nxy, nxystack, nb_mode, nb_psi,
                                                            s)
        p._cu_gamma[0] += tmp
        if False:
            tmp = self.processing_unit._cu_cg_poisson_gamma4_red(p._cu_obs_v[i].cu_obs[:nb_psi], p._cu_PO,
                                                                 p._cu_PdO, p._cu_dPO, p._cu_dPdO,
                                                                 nxy, nxystack, nb_mode, s).get()
            p._cu_cg_gamma4 += np.array((tmp['d'], tmp['c'], tmp['b'], tmp['a'], 0))

        if self.update_background:
            # TODO: use a different kernel if there is a background gradient
            pass
        return p


class ML(CUOperatorPtycho):
    """
    Operator to perform a maximum-likelihood conjugate-gradient minimization.
    """

    def __init__(self, nb_cycle=1, update_object=True, update_probe=False, update_background=False,
                 floating_intensity=False, reg_fac_obj=0, reg_fac_probe=0, calc_llk=False, show_obj_probe=False,
                 fig_num=-1, update_pos=False, pos_mult=1, pos_max_shift=2, pos_min_shift=0, pos_threshold=0.2,
                 pos_history=False, zero_phase_ramp=True):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param update_background: update background ?
        :param floating_intensity: optimise floating intensity scale factor [TODO for CUDA operators]
        :param reg_fac_obj: use this regularization factor for the object (if 0, no regularization)
        :param reg_fac_probe: use this regularization factor for the probe (if 0, no regularization)
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        :param update_pos: positive integer, if >0, update positions every 'update_pos' cycle. Note that object and
                           probe are not updated during the same cycle as positions. Still experimental, recommended
                           are 5, 10 (default=False, positions are not updated).
        :param pos_max_shift: maximum allowed shift (in pixels) per scan position (default=2)
        :param pos_min_shift: minimum required shift (in pixels) per scan position (default=0)
        :param pos_threshold: if the integrated grad_obj*probe along dx or dy is
            smaller than (grad_obj*probe).mean()*threshold, then the shift is ignored.
            This allows to prevent position shifts where there is little contrast in
            the object. It applies independently to x and y.
        :param pos_mult: multiply the calculated position shifts by this value. Useful since the calculated
                                shifts usually are a fraction of the actual shift.
        :param pos_history: if True, save the position history (for debugging)
        :param zero_phase_ramp: if True, the conjugate phase ramp in the object and probe will be removed
                                by centring the FT of the probe, at the beginning and end. Ignored for near field.
        """
        super(ML, self).__init__()
        self.nb_cycle = nb_cycle
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.reg_fac_obj = reg_fac_obj
        self.reg_fac_probe = reg_fac_probe
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.update_pos = update_pos
        self.pos_max_shift = pos_max_shift
        self.pos_min_shift = pos_min_shift
        self.pos_mult = pos_mult
        self.pos_threshold = pos_threshold
        self.pos_history = pos_history
        self.zero_phase_ramp = zero_phase_ramp

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new ML operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return ML(nb_cycle=self.nb_cycle * n, update_object=self.update_object, update_probe=self.update_probe,
                  update_background=self.update_background, reg_fac_obj=self.reg_fac_obj,
                  reg_fac_probe=self.reg_fac_probe, calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe,
                  fig_num=self.fig_num, update_pos=self.update_pos, pos_max_shift=self.pos_max_shift,
                  pos_min_shift=self.pos_min_shift, pos_threshold=self.pos_threshold, pos_mult=self.pos_mult,
                  pos_history=self.pos_history, zero_phase_ramp=self.zero_phase_ramp)

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        # First perform an AP cycle to make sure object and probe are properly scaled with respect to iobs
        p = AP(update_object=self.update_object, update_probe=self.update_probe,
               update_background=self.update_background, zero_phase_ramp=self.zero_phase_ramp) * p
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])

        # Create the necessary GPU arrays for ML
        p._cu_PO = cua.empty_like(p._cu_psi)
        p._cu_PdO = cua.empty_like(p._cu_psi)
        p._cu_dPO = cua.empty_like(p._cu_psi)
        p._cu_dPdO = cua.empty_like(p._cu_psi)
        p._cu_obj_dir = cua.zeros((nb_obj, nyo, nxo), np.complex64, allocator=pu.get_memory_pool().allocate)
        p._cu_probe_dir = cua.zeros((nb_probe, ny, nx), np.complex64, allocator=pu.get_memory_pool().allocate)
        if self.update_object:
            p._cu_obj_grad = cua.empty_like(p._cu_obj)
            p._cu_obj_grad_last = cua.empty_like(p._cu_obj)
        if self.update_probe:
            p._cu_probe_grad = cua.empty_like(p._cu_probe)
            p._cu_probe_grad_last = cua.empty_like(p._cu_probe)
        # We always need background_grad array, even if it is not used
        p._cu_background_grad = cua.zeros((ny, nx), np.float32, allocator=pu.get_memory_pool().allocate)
        p._cu_background_dir = cua.zeros((ny, nx), np.float32, allocator=pu.get_memory_pool().allocate)
        if self.update_background:
            p._cu_background_grad_last = cua.zeros((ny, nx), np.float32, allocator=pu.get_memory_pool().allocate)

        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            # Swap gradient arrays - for CG, we need the previous gradient
            if self.update_object:
                p._cu_obj_grad, p._cu_obj_grad_last = p._cu_obj_grad_last, p._cu_obj_grad
            if self.update_probe:
                p._cu_probe_grad, p._cu_probe_grad_last = p._cu_probe_grad_last, p._cu_probe_grad
            if self.update_background:
                p._cu_background_grad, p._cu_background_grad_last = p._cu_background_grad_last, p._cu_background_grad

            # 0) Update position (separate cycle)
            update_pos = False
            if self.update_pos:
                if ic % int(self.update_pos) == 0:
                    update_pos = True

            if update_pos:
                ops = PropagateApplyAmplitude(calc_llk=False, update_background=False) * ObjProbe2Psi()
                p = Psi2PosMerge(multiplier=self.pos_mult, max_displ=self.pos_max_shift, min_displ=self.pos_min_shift,
                                 threshold=self.pos_threshold, save_position_history=self.pos_history) * \
                    LoopStack(Psi2PosShift() * ops) * p
            # 1) Compute the gradients
            p = Grad(update_object=self.update_object, update_probe=self.update_probe,
                     update_background=self.update_background,
                     reg_fac_obj=self.reg_fac_obj, reg_fac_probe=self.reg_fac_probe, calc_llk=calc_llk) * p

            # 2) Search direction
            if ic == 0:
                beta = cua.zeros(1, dtype=np.complex64, allocator=pu.get_memory_pool().allocate)
                # first cycle
                if self.update_object:
                    cu_drv.memcpy_dtod(src=p._cu_obj_grad.gpudata, dest=p._cu_obj_dir.gpudata,
                                       size=p._cu_obj_dir.nbytes)
                if self.update_probe:
                    cu_drv.memcpy_dtod(src=p._cu_probe_grad.gpudata, dest=p._cu_probe_dir.gpudata,
                                       size=p._cu_probe_dir.nbytes)
                if self.update_background:
                    cu_drv.memcpy_dtod(src=p._cu_background_grad.gpudata, dest=p._cu_background_dir.gpudata,
                                       size=p._cu_background_dir.nbytes)
            else:
                # Polak-Ribière CG coefficient
                # If beta is NaN or infinite, the search direction is reset with beta=0
                cg_pr = pu.cu_cg_polak_ribiere_red
                cg_prf = pu.cu_cg_polak_ribiere_redf
                beta = cua.zeros(1, dtype=np.complex64, allocator=pu.get_memory_pool().allocate)
                if self.update_object:
                    beta[0] += cg_pr(p._cu_obj_grad, p._cu_obj_grad_last)
                if self.update_probe:
                    beta[0] += cg_pr(p._cu_probe_grad, p._cu_probe_grad_last)
                if self.update_background:
                    beta[0] += cg_prf(p._cu_background_grad, p._cu_background_grad_last)
                if self.update_object:
                    pu.cu_linear_comb_fcfc_beta(p._cu_obj_dir, beta, p._cu_obj_grad, np.float32(1))
                if self.update_probe:
                    pu.cu_linear_comb_fcfc_beta(p._cu_probe_dir, beta, p._cu_probe_grad, np.float32(1))
                if self.update_background:
                    pu.cu_linear_comb_4f_beta(p._cu_background_dir, beta, p._cu_background_grad, np.float32(1))
            # ngP = self.processing_unit.cu_norm_complex_n(p._cu_probe_grad, 2).get()
            # ngO = self.processing_unit.cu_norm_complex_n(p._cu_obj_grad, 2).get()
            # ndP = self.processing_unit.cu_norm_complex_n(p._cu_probe_dir, 2).get()
            # ndO = self.processing_unit.cu_norm_complex_n(p._cu_obj_dir, 2).get()
            # p.print('Grad: P %e O %e    Dir: P %e O %e    beta=%6.3f' % (ngP, ngO, ndP, ndO, beta))

            # 3) Line minimization
            p._cu_gamma = cua.zeros(1, dtype=np.complex64, allocator=pu.get_memory_pool().allocate)
            p = LoopStack(_CGGamma(update_background=self.update_background)) * p

            if self.update_object and self.reg_fac_obj != 0 and self.reg_fac_obj is not None:
                reg_fac_obj = np.float32(p.reg_fac_scale_obj * self.reg_fac_obj)
                nyo = np.int32(p._obj.shape[-2])
                nxo = np.int32(p._obj.shape[-1])
                tmp = self.processing_unit._cu_cg_gamma_reg_red(p._cu_obj, p._cu_obj_dir, nxo, nyo)
                tmp *= reg_fac_obj
                p._cu_gamma[0] += tmp

            if self.update_probe and self.reg_fac_probe != 0 and self.reg_fac_probe is not None:
                reg_fac_probe = np.float32(p.reg_fac_scale_probe * self.reg_fac_probe)
                ny = np.int32(p._probe.shape[-2])
                nx = np.int32(p._probe.shape[-1])
                tmp = self.processing_unit._cu_cg_gamma_reg_red(p._cu_probe, p._cu_probe_dir, nx, ny)
                tmp *= reg_fac_probe
                p._cu_gamma[0] += tmp

            if False:
                # It seems the 2nd order gamma approximation is good enough.
                gr = np.roots(p._cu_cg_gamma4)
                p.print("CG Gamma4", p._cu_cg_gamma4, "\n", gr, np.polyval(p._cu_cg_gamma4, gr))
                p.print("CG Gamma2=", gamma, "=", p._cu_cg_gamma_n, "/", p._cu_cg_gamma_d)

            # 4) Object and/or probe and/or background update
            # if gamma is NaN or infinite, the kernels will reset the search direction to the gradient
            if self.update_object:
                pu.cu_linear_comb_fcfc_gamma(p._cu_obj, np.float32(1), p._cu_obj_dir, p._cu_gamma,
                                             p._cu_obj_grad)

            if self.update_probe:
                pu.cu_linear_comb_fcfc_gamma(p._cu_probe, np.float32(1), p._cu_probe_dir, p._cu_gamma,
                                             p._cu_probe_grad)

            if self.update_background:
                pu.cu_linear_comb_4f_gamma_pos(p._cu_background, np.float32(1), p._cu_background_dir, p._cu_gamma,
                                               p._cu_background_grad)

            if calc_llk:
                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=self.update_pos,
                                 algorithm='ML', verbose=True)
            else:
                p.history.insert(p.cycle, update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=self.update_pos, algorithm='ML',
                                 verbose=False)

            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    s = algo_string('ML', p, self.update_object, self.update_probe,
                                    update_background=self.update_background, update_pos=update_pos)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.get_llk('poisson'))
                    p = ShowObjProbe(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1

        # Clean up
        del p._cu_PO, p._cu_PdO, p._cu_dPO, p._cu_dPdO, p._cu_obj_dir, p._cu_probe_dir
        if self.update_object:
            del p._cu_obj_grad, p._cu_obj_grad_last
        if self.update_probe:
            del p._cu_probe_grad, p._cu_probe_grad_last
        del p._cu_background_grad, p._cu_background_dir
        if self.update_background:
            del p._cu_background_grad_last

        gc.collect()
        if self.zero_phase_ramp:
            p = ZeroPhaseRamp(obj=True) * p
        return p


class ScaleObjProbe(CUOperatorPtycho):
    """
    Operator to scale the object and probe so that they have the same magnitude, and that the product of object*probe
    matches the observed intensity (i.e. sum(abs(obj*probe)**2) = sum(iobs))
    """

    def __init__(self, verbose=False, absolute=True):
        """

        :param verbose: print deviation if verbose=True
        :param absolute: if True, the absolute scale is computed by comparing the calculated
            intensities with the observed ones. If False, only the relative scale of object
            and probe is modified (to avoid numerical divergence), not affecting the calculated intensities.
        """
        super(ScaleObjProbe, self).__init__()
        self.verbose = verbose
        self.absolute = absolute

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        if self.absolute:
            # Compute the best scale factor
            snum, sden = 0, 0
            nxy = np.int32(p._probe.shape[-1] * p._probe.shape[-2])
            nxystack = np.int32(nxy * self.processing_unit.cu_stack_size)
            nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
            for i in range(p._cu_stack_nb):
                p = ObjProbe2Psi() * SelectStack(i) * p
                if p.data.near_field:
                    p = PropagateNearField(forward=True) * p
                else:
                    p = FT(scale=True) * p
                nb_psi = p._cu_obs_v[i].npsi
                r = pu.cu_scale_intensity(p._cu_obs_v[i].cu_obs[:nb_psi], p._cu_psi, p._cu_background,
                                          nxy, nxystack, nb_mode).get()
                snum += r.real
                sden += r.imag
            s = np.sqrt(snum / sden)
            # if not p.data.near_field:
            #    # p.print("ScaleObjProbe: not near field, compensate FFT scaling")
            #    s *= np.sqrt(p._cu_psi[0, 0, 0].size)  # Compensate for FFT scaling
        else:
            s = 1
        os = self.processing_unit.cu_norm_complex_n(p._cu_obj, np.int32(1)).get()
        ps = self.processing_unit.cu_norm_complex_n(p._cu_probe, np.int32(1)).get()
        pu.cu_scale(p._cu_probe, np.float32(np.sqrt(os / ps * s)))
        pu.cu_scale(p._cu_obj, np.float32(np.sqrt(ps / os * s)))
        if self.verbose:
            p.print("ScaleObjProbe:", ps, os, s, np.sqrt(os / ps * s), np.sqrt(ps / os * s))
        if False:
            # Check the scale factor
            snum, sden = 0, 0
            for i in range(p._cu_stack_nb):
                p = ObjProbe2Psi() * SelectStack(i) * p
                if p.data.near_field:
                    p = PropagateNearField(forward=True) * p
                else:
                    p = FT(scale=True) * p
                r = pu.cu_scale_intensity(p._cu_psi, p._cu_obs_v[i].cu_obs).get()
                snum += r.real
                sden += r.imag
            s = snum / sden
            p.print("ScaleObjProbe: now s=", s)
        return p


class CenterObjProbe(CUOperatorPtycho):
    """
    Operator to check the center of mass of the probe and shift both object and probe if necessary.
    """

    def __init__(self, max_shift=5, power=2, verbose=False):
        """

        :param max_shift: the maximum shift of the probe with respect to the center of the array, in pixels.
                          The probe and object are only translated if the shift is larger than this value.
        :param power: the center of mass is calculated on the amplitude of the array elevated at this power.
        :param verbose: print deviation if verbose=True
        """
        super(CenterObjProbe, self).__init__()
        self.max_shift = np.int32(max_shift)
        self.power = power
        self.verbose = verbose

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        nz, ny, nx = np.int32(p._probe.shape[0]), np.int32(p._probe.shape[1]), np.int32(p._probe.shape[2])
        nzo, nyo, nxo = np.int32(p._obj.shape[0]), np.int32(p._obj.shape[1]), np.int32(p._obj.shape[2])
        cm = pu.cu_center_mass_complex(p._cu_probe, nx, ny, nz, self.power).get()
        dx, dy, dz = cm['a'] / cm['d'] - nx / 2, cm['b'] / cm['d'] - ny / 2, cm['c'] / cm['d'] - nz / 2
        if self.verbose:
            p.print("CenterObjProbe(): center of mass deviation: dx=%6.2f   dy=%6.2f" % (dx, dy))
        if np.sqrt(dx ** 2 + dy ** 2) > self.max_shift:
            dx = np.int32(round(-dx))
            dy = np.int32(round(-dy))
            cu_obj = cua.empty_like(p._cu_obj)
            cu_probe = cua.empty_like(p._cu_probe)
            pu.cu_circular_shift(p._cu_probe, cu_probe, dx, dy, np.int32(0), nx, ny, nz)
            p._cu_probe = cu_probe
            pu.cu_circular_shift(p._cu_obj, cu_obj, dx, dy, np.int32(0), nxo, nyo, nzo)
            p._cu_obj = cu_obj
            # Also shift psi
            nzpsi = p._cu_psi.size // (nx * ny)
            cu_ps = cua.empty_like(p._cu_psi)
            pu.cu_circular_shift(p._cu_psi, cu_ps, dx, dy, np.int32(0), nx, ny, nzpsi)
            cu_ps, p._cu_psi = p._cu_psi, cu_ps
            if has_attr_not_none(p, "_cu_psi_v"):
                for k, cu_psi in p._cu_psi_v.items():
                    pu.cu_circular_shift(cu_psi, cu_ps, dx, dy, np.int32(0), nx, ny, nzpsi)
                    cu_ps, p._cu_psi_v[k] = p._cu_psi_v[k], cu_ps

        return p


class SumIntensity1(CUOperatorPtycho):
    """
    Operator to compute the sum of calculated and/or observed frames.
    When calculating the sum of observed intensities, masked values will be replaced
    by calculated ones.

    This operator applies to a single stack, and will perform the object*probe
    multiplication, propagate to the detector, and compute the sums.
    """

    def __init__(self, icalc=None, iobs=None):
        """

        :param icalc: the GPU array holding the sum of calculated intensities. If None, it is not calculated.
        :param iobs: the GPU array holding the sum of observed intensities. If None, it is not calculated.
        """
        super(SumIntensity1, self).__init__()
        self.icalc = icalc
        self.iobs = iobs

    def op(self, p: Ptycho):
        if self.icalc is None and self.iobs is None:
            return p
        if p.data.near_field:
            p = PropagateNearField(forward=True) * ObjProbe2Psi() * p
        else:
            p = FT(scale=False) * ObjProbe2Psi() * p

        pu = self.processing_unit
        nxy = np.int32(p._probe.shape[1] * p._probe.shape[2])
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        i = p._cu_stack_i
        nz = p._cu_obs_v[i].npsi

        if self.icalc is not None:
            pu.cu_sum_icalc(self.icalc, p._cu_psi, nxy, np.int32(nz * nb_mode))
        if self.iobs is not None:
            pu.cu_sum_iobs(self.iobs, p._cu_obs_v[i].cu_obs[:nz], p._cu_psi, nxy, nz, nb_mode)

        return p


class ApplyPhaseRamp(CUOperatorPtycho):
    """
    Apply a given phase ramp to the object and/or probe. The actual phase factor is:
    For the probe: np.exp(-2j*np.pi*(x * dx + y*dy))
    For the object: np.exp(2j*np.pi*(x * dx + y*dy))
    Where (x,y) are reduced pixel coordinates going from -.5 to .5 for the probe,
    and  +/-0.5 * nxo/nx (or nyo/ny) for the object.

    If the phase ramp is applied to both object and probe, the calculated
    intensity patterns remain unchanged.
    """

    def __init__(self, dx, dy, probe=False, obj=False):
        """

        :param dx, dy: relative shifts from the centre, calculated in reciprocal space
                       (probe array pixel coordinates)
        :param obj: if True, apply the correction to the object.
        :param probe: if True, apply the correction to the probe.
        """
        super(ApplyPhaseRamp, self).__init__()
        self.probe = probe
        self.obj = obj
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)

    def op(self, p: Ptycho):
        pu = self.processing_unit
        nz, ny, nx = np.int32(p._probe.shape[0]), np.int32(p._probe.shape[1]), np.int32(p._probe.shape[2])
        nyo, nxo = np.int32(p._obj.shape[1]), np.int32(p._obj.shape[2])

        # Corr ramp with opposite signs for object and probe
        if self.probe:
            pu.cu_corr_phase_ramp(p._cu_probe, -self.dx, -self.dy, nx, ny)
        if self.obj:
            pu.cu_corr_phase_ramp(p._cu_obj, np.float32(self.dx * nxo / nx), np.float32(self.dy * nyo / ny), nxo, nyo)
        return p


class ZeroPhaseRamp(CUOperatorPtycho):
    """
    Operator to remove the linear phase ramp on both the object and the probe, for far field ptycho.
    This first computes the center of mass of the square norm of Fourier transform of the probe,
    and then corrects both probe and object for the phase ramp corresponding to the shift
    relative to the center of the array.

    Then, a remaining phase ramp in the object can optionally be computed by:
    - computing the sum of the calculated intensity along all frames
    - computing the shift of the center of mass of that intensity w/r to the array center
    - the phase ramp parameters are stored in the ptycho object, but not applied as the calculated
    """

    def __init__(self, obj=False):
        """

        :param obj: if True, after correcting both object and probe from the probe phase ramp,
                    the object phase ramp is evaluated from the center of the calculated diffraction.
        """
        super(ZeroPhaseRamp, self).__init__()
        self.obj = obj

    def op(self, p: Ptycho):
        if p.data.near_field:
            return p
        pu = self.processing_unit
        nz, ny, nx = np.int32(p._probe.shape[0]), np.int32(p._probe.shape[1]), np.int32(p._probe.shape[2])

        cu_probe = p._cu_probe.copy()

        pu.fft(cu_probe, cu_probe, ndim=2)
        cm = pu.cu_center_mass_fftshift_complex(cu_probe, nx, ny, nz, np.int32(2)).get()
        dx = np.float32(cm['a'] / cm['d'] - nx / 2)
        dy = np.float32(cm['b'] / cm['d'] - ny / 2)
        # print("ZeroPhaseRamp(): (dx, dy)[probe] = (%6.3f, %6.3f)[obs]" % (dx, dy))

        p = ApplyPhaseRamp(dx, dy, obj=True, probe=True) * p

        if self.obj:
            # Compute the shift of the calculated frame to determine the object ramp
            icalc_sum = cua.zeros((ny, nx), dtype=np.float32)
            iobs_sum = None  # cua.zeros((ny, nx), dtype=np.float32)
            p = LoopStack(SumIntensity1(icalc=icalc_sum, iobs=iobs_sum)) * p

            # Compute shift of center of mass
            if False:
                cm = pu.cu_center_mass_fftshift(iobs_sum, nx, ny, np.int32(1), np.int32(1)).get()
                dx = np.float32(cm['a'] / cm['d'] - nx / 2)
                dy = np.float32(cm['b'] / cm['d'] - ny / 2)
                # print("ZeroPhaseRamp(): (dx, dy)[obj] = (%6.3f, %6.3f)[obs]" % (dx, dy))
            else:
                cm = pu.cu_center_mass_fftshift(icalc_sum, nx, ny, np.int32(1), np.int32(1)).get()
                dx = np.float32(cm['a'] / cm['d'] - nx / 2)
                dy = np.float32(cm['b'] / cm['d'] - ny / 2)
                # print("ZeroPhaseRamp(): (dx, dy)[obj] = (%6.3f, %6.3f)[calc]" % (dx, dy))
            p.data.phase_ramp_dx = dx
            p.data.phase_ramp_dy = dy

        return p


class CalcIllumination(CUOperatorPtycho):
    """
    Compute the integrated illumination of the object by all probe positions
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        interp = p._interpolation
        cu_obj_illum = cua.zeros((nyo, nxo), dtype=np.float32, allocator=pu.get_memory_pool().allocate)
        padding = np.int32(p.data.padding)
        for i in range(p._cu_stack_nb):
            npsi = p._cu_obs_v[i].npsi
            i0 = p._cu_obs_v[i].i
            pu.cu_calc_illum(p._cu_probe[0], cu_obj_illum, p._cu_cx[i0:i0 + npsi], p._cu_cy[i0:i0 + npsi],
                             p._cu_obs_v[i].npsi, pu.cu_stack_size, nx, ny, nxo, nyo, nb_probe, interp, padding)
        p._obj_illumination = cu_obj_illum.get()
        cu_obj_illum.gpudata.free()  # Should not be necessary, will be gc
        return p

    def timestamp_increment(self, p):
        # Is that really the correct behaviour ?
        # Object and probe etc are not modified, but ptycho object is..
        pass


class SelectStack(CUOperatorPtycho):
    """
    Operator to select a stack of observed frames to work on. Note that once this operation has been applied,
    the new Psi value may be undefined (empty array), if no previous array existed.
    """

    def __init__(self, stack_i, keep_psi=False):
        """
        Select a new stack of frames, swapping data to store the last calculated psi array in the
        corresponding, ptycho object's _cu_psi_v[i] dictionary.

        What happens is:
        * keep_psi=False: only the stack index in p is changed (p._cu_stack_i=stack_i)

        * keep_psi=True: the previous psi is stored in p._cu_psi_v[p._cu_stack_i], the new psi is swapped
                                   with p._cu_psi_v[stack_i] if it exists, otherwise initialized as an empty array.

        :param stack_i: the stack index.
        :param keep_psi: if True, when switching between stacks, store and restore psi in p._cu_psi_v.
        """
        super(SelectStack, self).__init__()
        self.stack_i = stack_i
        self.keep_psi = keep_psi

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        if self.stack_i == p._cu_stack_i:
            if self.keep_psi and self.stack_i in p._cu_psi_v:
                # This can happen if we use LoopStack(keep_psi=False) between LoopStack(keep_psi=True)
                p._cu_psi = p._cu_psi_v[self.stack_i].pop()
            return p

        if self.keep_psi:
            # Store previous Psi. This can be dangerous when starting a loop as the state of Psi may be incorrect,
            # e.g. in detector or sample space when the desired operations work in a different space...
            p._cu_psi_v[p._cu_stack_i] = p._cu_psi
            if self.stack_i in p._cu_psi_v:
                p._cu_psi = p._cu_psi_v.pop(self.stack_i)
            else:
                p._cu_psi = cua.empty_like(p._cu_psi_v[p._cu_stack_i])

        p._cu_stack_i = self.stack_i
        return p


class PurgeStacks(CUOperatorPtycho):
    """
    Operator to delete stored psi stacks in a Ptycho object's _cu_psi_v.

    This should be called for each main operator using LoopStack(), once it is finished processing, in order to avoid
    having another operator using the stored stacks, and to free memory.
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        # First make sure processing is finished, as execution is asynchronous
        self.processing_unit.finish()
        p._cu_psi_v = {}
        return p


class LoopStack(CUOperatorPtycho):
    """
    Operator to apply a given operator sequentially to the complete stack of frames of a ptycho object.

    Make sure that the current selected stack is in a correct state (i.e. in sample or detector space,...) before
    starting such a loop with keep_psi=True.
    """

    def __init__(self, op, keep_psi=False, copy=False):
        """

        :param op: the operator to apply, which can be a multiplication of operators
        :param keep_psi: if True, when switching between stacks, store psi in p._cu_psi_v.
        :param copy: make a copy of the original p._cu_psi swapped in as p._cu_psi_copy, and
                     delete it after applying the operations. This is useful for operations requiring the previous
                     value.
        """
        super(LoopStack, self).__init__()
        self.stack_op = op
        self.keep_psi = keep_psi
        self.copy = copy

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        if p._cu_stack_nb == 1:
            if self.copy:
                p._cu_psi_copy = cua.empty_like(p._cu_psi)

                cu_drv.memcpy_dtod(src=p._cu_psi.gpudata, dest=p._cu_psi_copy.gpudata, size=p._cu_psi.nbytes)
                p = self.stack_op * p

                if has_attr_not_none(p, '_cu_psi_copy'):
                    # Finished using psi copy, delete it (actual deletion will occur once GPU has finished processing)
                    p._cu_psi_copy.gpudata.free()
                    del p._cu_psi_copy
                return p
            else:
                return self.stack_op * p
        else:
            if self.copy:
                p._cu_psi_copy = cua.empty_like(p._cu_psi)

            for i in range(p._cu_stack_nb):
                p = SelectStack(i, keep_psi=self.keep_psi) * p
                if self.copy:
                    # The planned operations rely on keeping a copy of the previous Psi state...
                    cu_drv.memcpy_dtod(src=p._cu_psi.gpudata, dest=p._cu_psi_copy.gpudata, size=p._cu_psi.nbytes)
                p = self.stack_op * p

            if self.copy and has_attr_not_none(p, '_cu_psi_copy'):
                # Finished using psi copy, delete it (actual deletion will occur once GPU has finished processing)
                p._cu_psi_copy.gpudata.free()
                del p._cu_psi_copy

            if self.keep_psi:
                # Copy last stack to p._cu_psi_v
                p._cu_psi_v[p._cu_stack_i] = cua.empty_like(p._cu_psi)
                cu_drv.memcpy_dtod(src=p._cu_psi.gpudata, dest=p._cu_psi_v[p._cu_stack_i].gpudata,
                                   size=p._cu_psi.nbytes)
        return p


class LLKPoissonStats1(CUOperatorPtycho):
    """
    Compute the per-pixel Poisson LLK standard deviation and skewness.
    The Poisson LLK is calculated with a + sign if calc>obs, and a - sign otherwise.
    Applies to a single stack of frames. Normalisation by the
    total number of frames must be done outside this kernel.
    """

    def op(self, p: Ptycho):
        pu = self.processing_unit
        i = p._cu_stack_i
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        nb_psi = p._cu_obs_v[i].npsi
        nxy = np.int32(p._probe.shape[-2] * p._probe.shape[-1])
        nxystack = np.int32(pu.cu_stack_size * nxy)

        # FFT scale ?
        if p.data.near_field:
            s = np.float32(1)
        else:
            s = pu.fft_scale(p._cu_psi.shape, ndim=2)[0]

        self.processing_unit.cu_llk_poisson_stats(p._cu_obs_v[i].cu_obs[:nb_psi], p._cu_psi,
                                                  p._cu_background, p._cu_llk_mean,
                                                  p._cu_llk_std, p._cu_llk_skew, p._cu_llk_skew0,
                                                  nb_mode, nxy, nxystack, s)
        return p


class LLKPoissonHist1(CUOperatorPtycho):
    """
    Compute the per-pixel cumulated and histogram of Poisson LLK.
    Applies to a single stack of frames.
    """

    def __init__(self, nbin=20, binsize=1):
        super(LLKPoissonHist1, self).__init__()
        self.nbin = np.int32(nbin)
        self.binsize = np.float32(binsize)

    def op(self, p: Ptycho):
        pu = self.processing_unit
        i = p._cu_stack_i
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        nb_psi = p._cu_obs_v[i].npsi
        nxy = np.int32(p._probe.shape[-2] * p._probe.shape[-1])
        nxystack = np.int32(pu.cu_stack_size * nxy)

        # FFT scale ?
        if p.data.near_field:
            s = np.float32(1)
        else:
            s = pu.fft_scale(p._cu_psi.shape, ndim=2)[0]

        self.processing_unit.cu_llk_poisson_hist(p._cu_obs_v[i].cu_obs[:nb_psi], p._cu_psi,
                                                 p._cu_background, p._cu_llk_sum,
                                                 p._cu_llk_hist, self.nbin, self.binsize,
                                                 nb_mode, nxy, nxystack, s)
        # Debug
        p.iobs1 = p._cu_obs_v[i].cu_obs[:nb_psi].get()
        p.icalc1 = (abs(p._cu_psi.get()[:, :, :nb_psi]) ** 2).sum(axis=(0, 1))
        return p


class LLKPoissonStats(CUOperatorPtycho):
    """
    Compute the per-pixel Poisson LLK statistics: mean, standard
    deviation, skewness, histogram.
    The 0-centered skewness is also computed.
    Applies to a single stack of frames.
    """

    def __init__(self, scale=False, dllk=3, nbin=20, binsize=1):
        """

        :param nbin: number of bins for the histogram
        :param binsize: size of each bin. The lower coordinates of the bins
            are (np.arange(nbin)-nbin/2)*binsize
        """
        super(LLKPoissonStats, self).__init__()
        self.nbin = nbin
        self.binsize = binsize

    def op(self, p: Ptycho):
        pu = self.processing_unit
        p._cu_llk_sum = cua.zeros(p.data.iobs.shape[-2:], dtype=np.float32,
                                  allocator=pu.get_memory_pool().allocate)
        p._cu_llk_std = cua.zeros(p.data.iobs.shape[-2:], dtype=np.float32,
                                  allocator=pu.get_memory_pool().allocate)
        p._cu_llk_skew = cua.zeros(p.data.iobs.shape[-2:], dtype=np.float32,
                                   allocator=pu.get_memory_pool().allocate)
        p._cu_llk_skew0 = cua.zeros(p.data.iobs.shape[-2:], dtype=np.float32,
                                    allocator=pu.get_memory_pool().allocate)
        ny, nx = p.data.iobs.shape[-2:]
        p._cu_llk_hist = cua.zeros((self.nbin, ny, nx), dtype=np.int16,
                                   allocator=pu.get_memory_pool().allocate)
        if p.data.near_field:
            ops = PropagateNearField(forward=True) * ObjProbe2Psi()
        else:
            ops = FT(scale=False) * ObjProbe2Psi()

        # Mean & histogram
        p = LoopStack(LLKPoissonHist1(nbin=self.nbin, binsize=self.binsize) * ops) * p
        p._cu_llk_sum /= np.float32(len(p.data.iobs))
        p._cu_llk_mean = p._cu_llk_sum
        p._llk_mean = p._cu_llk_mean.get()
        p._llk_poisson_hist = p._cu_llk_hist.get()

        # Std dev & skewness
        p = LoopStack(LLKPoissonStats1() * ops) * p
        p._llk_std = p._cu_llk_std.get()
        p._llk_std = np.sqrt(p._llk_std / (len(p.data.iobs) - 1))
        p._llk_skew = p._cu_llk_skew.get() / len(p.data.iobs) / p._llk_std ** 3
        p._llk_skew0 = p._cu_llk_skew0.get() / len(p.data.iobs) / p._llk_std ** 3

        del p._cu_llk_sum, p._cu_llk_mean, p._cu_llk_std, p._cu_llk_skew, p._cu_llk_skew0, p._cu_llk_hist

        return p


class PaddingInterp(CUOperatorPtycho):
    """
    Operator to interpolate the pixels inside and near the padded areas.
    This is useful after some initial cycles, to correct for the different scaling
    between the padded and non-padded areas (induced by an incorrect object & probe
    vs Iobs scale).
    This is only useful for near field ptycho.
    """

    def __init__(self, margin_probe=32, margin_obj=128):
        """
        This operator will replace the object and
        :param margin_probe: extra pixels (in addition to the padding value) which
            will be interpolated near the border of the probe, as they can be affected
            by the nearby incorrect values
        :param margin_obj: extra pixels (in addition to the padding value) which
            will be interpolated near the border of the object, as they can be affected
            by the nearby incorrect values
        """
        super(PaddingInterp, self).__init__()
        self.margin_probe = margin_probe
        self.margin_obj = margin_obj

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        if not p.data.near_field or p.data.padding == 0:
            return p
        pu = self.processing_unit
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        mo = np.int32(p.data.padding + self.margin_obj)
        mp = np.int32(p.data.padding + self.margin_probe)

        pu.cu_padding_interp(p._cu_probe, nx, ny, mp)
        # Do we need to do this to the object too ?
        # print(nyo, nxo, mo)
        # pu.cu_padding_interp(p._cu_obj, nxo, nyo, mo)
        return p


class BackgroundFilter(CUOperatorPtycho):
    """
    Apply a Gaussian filter to the background array. This operator will perform
    a FT on the backround, multiply it by a Gaussian, and back-FT the array.
    The resulting array is normalised so that its sum is equal to the original
    array sum.
    """

    def __init__(self, sigma):
        """
        :param sigma: standard deviation for the Gaussian kernel in the detector
            space, corresponding to a convolution of the background by a Gaussian
            of FWHM=2.3548*sigma
        """
        super(BackgroundFilter, self).__init__()
        self.sigma = sigma

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        if p._background is None:
            return p
        pu = self.processing_unit
        nxy = np.int32(p._background.shape[-1])

        s = cua.sum(p._cu_background)
        cubr = cua.empty((nxy, nxy // 2 + 1), dtype=np.complex64)

        sigmaf = np.float32(2 * np.pi ** 2 * self.sigma / nxy ** 2)

        pu.fft(p._cu_background, cubr, norm=True)
        pu.cu_gauss_ftconv(cubr, sigmaf, nxy)
        pu.ifft(cubr, p._cu_background, norm=True)
        # Scale and make sure background is >=0 after FT
        pu.cu_scalef_mem_pos(p._cu_background, s / cua.sum(p._cu_background))
        return p


def get_icalc(p: Ptycho, i, shift=False):
    """
    Get the calculated intensity for frame i and a given Ptycho object.
    Note that this will reset the current psi, so should not be used during
    algorithms like DM which rely on the previous Psi value.

    :param p: the Ptycho object
    :param i: the index of the desired frame
    :param shift: if True, the array will be shifted so that the array is centered
        on the frame center (and not the corner).
    :return: the numpy array of the calculated intensity
    """
    stack_size = FT().processing_unit.get_stack_size()
    ii = i % stack_size
    p = ObjProbe2Psi() * SelectStack(i // stack_size) * p
    if p.data.near_field:
        p = PropagateNearField(forward=True) * p
    else:
        p = FT(scale=True) * p

    icalc = (abs(p._cu_psi.get()[:, :, ii]) ** 2).sum(axis=(0, 1)) + p.get_background()
    if shift:
        return np.fft.fftshift(icalc)
    return (icalc)
