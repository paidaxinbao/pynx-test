# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


import os
import warnings
import psutil
import types
import gc
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from pyopencl.elementwise import ElementwiseKernel as CL_ElK
from pyopencl.reduction import ReductionKernel as CL_RedK
from pyopencl.tools import get_or_register_dtype

from ..processing_unit import default_processing_unit as main_default_processing_unit
from ..processing_unit.cl_processing_unit import CLProcessingUnit, CLEvent
from ..processing_unit.kernel_source import get_kernel_source as getks
from ..operator import has_attr_not_none, OperatorSum, OperatorPower, OperatorException
from .ptycho import Ptycho, OperatorPtycho, algo_string
from ..mpi import MPI

if MPI is not None:
    from .mpi.operator import ShowObjProbe
else:
    from .cpu_operator import ShowObjProbe
from .shape import get_view_coord


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


class CLProcessingUnitPtycho(CLProcessingUnit):
    """
    Processing unit in OpenCL space, for 2D Ptycho operations.

    Handles initializing the context and kernels.
    """

    def __init__(self):
        super(CLProcessingUnitPtycho, self).__init__()
        # Size of the stack size used in OpenCL - can be any integer, optimal values between 10 to 30
        # Should be chosen smaller for large frame sizes.
        self.cl_stack_size = np.int32(16)

    def set_stack_size(self, s):
        """
        Change the number of frames which are stacked to perform all operations in //. If it
        is larger than the total number of frames, operators like AP, DM, ML will loop over
        all the stacks.
        :param s: an integer number (default=16)
        :return: nothing
        """
        self.cl_stack_size = np.int32(s)

    def get_stack_size(self):
        return self.cl_stack_size

    def cl_init_kernels(self):
        """
        Initialize opencl kernels
        :return: nothing
        """
        # TODO: delay initialization, on-demand for each type of operator ?

        # Elementwise kernels
        self.cl_scale = CL_ElK(self.cl_ctx, name='cl_scale',
                               operation="d[i] = (float2)(d[i].x * scale, d[i].y * scale )",
                               options=self.cl_options, arguments="__global float2 *d, const float scale")

        self.cl_scalef_mem = CL_ElK(self.cl_ctx, name='cl_scalef_mem',
                                    operation="d[i] *= scale[0]",
                                    options=self.cl_options, arguments="float *d, const float* scale")

        # Gauss convolution kernel in Fourier space
        self.cl_gauss_ftconv = CL_ElK(self.cl_ctx, name='cl_gauss_ftconv',
                                      operation="const int ix = i % (nxy/2+1);"
                                                "int iy = i / (nxy/2+1); iy -= nxy * (iy >= nxy/2);"
                                                "const float v = -sigmaf*(ix*ix + iy*iy);"
                                                "dest[i] *= v > -50 ? native_exp(v): 1.9287498479639178e-22f ;",
                                      options=self.cl_options,
                                      arguments="__global float2 *dest, const float sigmaf, const int nxy")

        self.cl_sum = CL_ElK(self.cl_ctx, name='cl_sum',
                             operation="dest[i] += src[i]",
                             options=self.cl_options, arguments="__global float2 *src, __global float2 *dest")

        self.cl_sum_icalc = CL_ElK(self.cl_ctx, name='sum_icalc',
                                   operation="SumIcalc(i, icalc_sum, d, nxy, nz)",
                                   preamble=getks('ptycho/opencl/sum_intensity_elw.cl'),
                                   options=self.cl_options,
                                   arguments="__global float* icalc_sum, __global float2 *d,"
                                             "const int nxy, const int nz")

        self.cl_sum_iobs = CL_ElK(self.cl_ctx, name='sum_iobs',
                                  operation="SumIobs(i, iobs_sum, obs, calc, nxy, nz, nb_mode)",
                                  preamble=getks('ptycho/opencl/sum_intensity_elw.cl'),
                                  options=self.cl_options,
                                  arguments="__global float* iobs_sum, __global float* obs, __global float2 *calc,"
                                            "const int nxy, const int nz, const int nb_mode")

        self.cl_floating_scale_update = CL_ElK(self.cl_ctx, name='floating_scale_update',
                                               operation="scale[i] *= iobs_sum[i] / icalc_sum[i]",
                                               options=self.cl_options,
                                               arguments="__global float *scale, __global float *iobs_sum, "
                                                         "__global float *icalc_sum")

        self.cl_floating_scale_norm = CL_ElK(self.cl_ctx, name='floating_scale_norm',
                                             operation="scale[i] /= scale_sum[0] / nb",
                                             options=self.cl_options,
                                             arguments="__global float *scale, __global float *scale_sum, const int nb")

        self.cl_scale_complex = CL_ElK(self.cl_ctx, name='cl_scale_complex',
                                       operation="d[i] = (float2)(d[i].x * s.x - d[i].y * s.y, d[i].x * s.y + d[i].y * s.x)",
                                       options=self.cl_options, arguments="__global float2 *d, const float2 s")

        self.cl_quad_phase = CL_ElK(self.cl_ctx, name='cl_quad_phase',
                                    operation="QuadPhase(i, d, f, scale, nx, ny)",
                                    preamble=getks('ptycho/opencl/quad_phase_elw.cl'), options=self.cl_options,
                                    arguments="__global float2 *d, const float f, const float scale, const int nx, const int ny")

        # Linear combination with 2 complex arrays and 2 float coefficients
        self.cl_linear_comb_fcfc = CL_ElK(self.cl_ctx, name='cl_linear_comb_fcfc',
                                          operation="dest[i] = (float2)(a * dest[i].x + b * src[i].x, a * dest[i].y + b * src[i].y)",
                                          options=self.cl_options,
                                          arguments="const float a, __global float2 *dest, const float b, __global float2 *src")

        # Linear combination with 2 float arrays and 2 float coefficients
        self.cl_linear_comb_4f = CL_ElK(self.cl_ctx, name='cl_linear_comb_4f',
                                        operation="dest[i] = a * dest[i] + b * src[i]",
                                        options=self.cl_options,
                                        arguments="const float a, __global float *dest, const float b,"
                                                  "__global float *src")

        # Linear combination with 2 float arrays and 2 float coefficients, final value must be >=0
        self.cl_linear_comb_4f_pos = CL_ElK(self.cl_ctx, name='cl_linear_comb_4f_pos',
                                            operation="dest[i] = fmax(a * dest[i] + b * src[i],0.0f);",
                                            options=self.cl_options,
                                            arguments="const float a, __global float *dest, const float b,"
                                                      "__global float *src")

        self.cl_projection_amplitude = CL_ElK(self.cl_ctx, name='cl_projection_amplitude',
                                              operation="ProjectionAmplitude(i, iobs, dcalc, background,"
                                                        "nbmode, nxy, nxystack, npsi)",
                                              preamble=getks('ptycho/opencl/projection_amplitude_elw.cl'),
                                              options=self.cl_options,
                                              arguments="__global float *iobs, __global float2 *dcalc,"
                                                        "__global float *background,"
                                                        "const int nbmode, const int nxy, const int nxystack,"
                                                        "const int npsi")

        self.cl_projection_amplitude_background = \
            CL_ElK(self.cl_ctx, name='cl_projection_amplitude_background',
                   operation="ProjectionAmplitudeBackground(i, iobs, dcalc, background,"
                             "vd, vd2, vz2, vdz2, nbmode, nxy, nxystack, npsi, first_pass)",
                   preamble=getks('ptycho/opencl/projection_amplitude_elw.cl'),
                   options=self.cl_options,
                   arguments="__global float *iobs, __global float2 *dcalc,"
                             "__global float *background, __global float *vd, __global float *vd2,"
                             "__global float *vz2, __global float *vdz2,"
                             "const int nbmode, const int nxy, const int nxystack,"
                             "const int npsi, const char first_pass")

        self.cl_projection_amplitude_background_mode = \
            CL_ElK(self.cl_ctx, name='cl_projection_amplitude_background_mode',
                   operation="ProjectionAmplitudeBackgroundMode(i, iobs, dcalc, background, background_new,"
                             "nbmode, nxy, nxystack, npsi, first_pass)",
                   preamble=getks('ptycho/opencl/projection_amplitude_elw.cl'),
                   options=self.cl_options,
                   arguments="__global float *iobs, __global float2 *dcalc,"
                             "__global float *background, __global float *background_new,"
                             "const int nbmode, const int nxy, const int nxystack,"
                             "const int npsi, const char first_pass")

        self.cl_background_update = \
            CL_ElK(self.cl_ctx, name='cl_background_update',
                   operation="const float eta = fmax(0.8f, vdz2[i]/vd2[i]);"
                             "background[i] = fmax(0.0f, background[i] + (vd[i] - vz2[i] / eta) / nframes);",
                   options=self.cl_options,
                   arguments="__global float* background, __global float* vd, __global float* vd2,"
                             "__global float* vz2, __global float* vdz2, const int nframes")

        self.cl_background_update_mode = \
            CL_ElK(self.cl_ctx, name='cl_background_update_mode',
                   operation="background[i] = background_new[i] / nframes;",
                   options=self.cl_options,
                   arguments="__global float* background, __global float* background_new, const int nframes")

        self.cl_calc2obs = CL_ElK(self.cl_ctx, name='cl_calc2obs',
                                  operation="Calc2Obs(i, iobs, dcalc, background, nbmode, nxy, nxystack)",
                                  preamble=getks('ptycho/opencl/calc2obs_elw.cl'),
                                  options=self.cl_options,
                                  arguments="__global float *iobs, __global float2 *dcalc, __global float *background,"
                                            "const int nbmode, const int nxy, const int nxystack")

        self.cl_object_probe_mult = CL_ElK(self.cl_ctx, name='cl_object_probe_mult',
                                           operation="ObjectProbeMultQuadPhase(i, psi, obj, probe, cx, cy, pixel_size, "
                                                     "f, npsi, stack_size, nx, ny, nxo, nyo, nbobj, nbprobe, scale,"
                                                     "interp)",
                                           preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                                    getks('ptycho/opencl/obj_probe_mult_elw.cl'),
                                           options=self.cl_options,
                                           arguments="__global float2* psi, __global float2 *obj, "
                                                     "__global float2* probe, __global float* cx, __global float* cy,"
                                                     "const float pixel_size, const float f, const int npsi, "
                                                     "const int stack_size, const int nx, const int ny, const int nxo,"
                                                     "const int nyo, const int nbobj, const int nbprobe,"
                                                     "__global float* scale, const char interp")

        self.cl_calc_illum = CL_ElK(self.cl_ctx, name='cl_calc_illum',
                                    operation="CalcIllumination(i, probe, obj_illum, cx, cy, npsi,"
                                              "stack_size, nx, ny, nxo, nyo, nbprobe, scale, interp, padding)",
                                    preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                             getks('ptycho/opencl/calc_illumination_elw.cl'),
                                    options=self.cl_options,
                                    arguments="__global float2* probe, __global float* obj_illum,"
                                              "__global float* cx, __global float* cy, const int npsi,"
                                              "const int stack_size, const int nx, const int ny, const int nxo,"
                                              "const int nyo, const int nbprobe, __global float* scale,"
                                              "const char interp, const int padding")

        self.cl_2object_probe_psi_dm1 = CL_ElK(self.cl_ctx, name='cl_2object_probe_psi_dm1',
                                               operation="ObjectProbePsiDM1(i, psi, obj, probe, cx, cy, pixel_size, f, "
                                                         "npsi, stack_size, nx, ny, nxo, nyo, nbobj, nbprobe, scale,"
                                                         "interp)",
                                               preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                                        getks('ptycho/opencl/obj_probe_dm_elw.cl'),
                                               options=self.cl_options,
                                               arguments="__global float2* psi, __global float2 *obj, "
                                                         "__global float2* probe, __global float* cx, "
                                                         "__global float* cy, const float pixel_size, const float f, "
                                                         "const int npsi, const int stack_size, const int nx, "
                                                         "const int ny, const int nxo, const int nyo, const int nbobj, "
                                                         "const int nbprobe, __global float* scale, const char interp")

        self.cl_2object_probe_psi_dm2 = CL_ElK(self.cl_ctx, name='cl_2object_probe_psi_dm2',
                                               operation="ObjectProbePsiDM2(i, psi, psi_fourier, obj, probe, cx, cy, "
                                                         "pixel_size, f, npsi, stack_size, nx, ny, nxo, nyo, nbobj, "
                                                         "nbprobe, scale, interp)",
                                               preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                                        getks('ptycho/opencl/obj_probe_dm_elw.cl'),
                                               options=self.cl_options,
                                               arguments="__global float2* psi, __global float2* psi_fourier,"
                                                         "__global float2 *obj, __global float2* probe,"
                                                         "__global float* cx, __global float* cy,"
                                                         "const float pixel_size, const float f, const int npsi,"
                                                         "const int stack_size, const int nx, const int ny,"
                                                         "const int nxo, const int nyo, const int nbobj, "
                                                         "const int nbprobe, __global float* scale, const char interp")

        self.cl_psi2obj_atomic = CL_ElK(self.cl_ctx, name='psi2obj_atomic',
                                        operation="UpdateObjAtomic(i, psi, objnew, probe, objnorm, cx, cy, px, f,"
                                                  "stack_size, nx, ny, nxo, nyo, nbobj, nbprobe, npsi, scale, interp)",
                                        preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                                 getks('ptycho/opencl/psi_to_obj_probe_elw.cl'),
                                        options=self.cl_options,
                                        arguments="__global float2* psi, __global float2 *objnew,"
                                                  "__global float2* probe, __global float* objnorm,"
                                                  "__global float* cx, __global float* cy, const float px,"
                                                  "const float f, const int stack_size, const int nx, const int ny,"
                                                  "const int nxo, const int nyo, const int nbobj, const int nbprobe,"
                                                  "const int npsi, __global float* scale, const char interp")

        self.cl_psi_to_probe = CL_ElK(self.cl_ctx, name='psi_to_probe',
                                      operation="UpdateProbeQuadPhase(i, obj, probe_new, psi, probenorm, cx, cy, px, f,"
                                                "firstpass, npsi, stack_size, nx, ny, nxo, nyo, nbobj, nbprobe, scale,"
                                                "interp)",
                                      preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                               getks('ptycho/opencl/psi_to_obj_probe_elw.cl'),
                                      options=self.cl_options,
                                      arguments="__global float2* psi, __global float2 *obj,"
                                                "__global float2* probe_new, __global float* probenorm,"
                                                "__global float* cx, __global float* cy, const float px, const float f,"
                                                "const char firstpass, const int npsi, const int stack_size,"
                                                "const int nx, const int ny, const int nxo, const int nyo,"
                                                "const int nbobj, const int nbprobe, __global float* scale,"
                                                "const char interp")

        self.cl_obj_norm = CL_ElK(self.cl_ctx, name='obj_norm',
                                  operation="ObjNorm(i, obj_unnorm, objnorm, obj, normmax, inertia, nxyo, nbobj)",
                                  preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                           getks('ptycho/opencl/psi_to_obj_probe_elw.cl'),
                                  options=self.cl_options,
                                  arguments="__global float2* obj_unnorm, __global float* objnorm,"
                                            "__global float2* obj, __global const float* normmax, const float inertia,"
                                            "const int nxyo, const int nbobj")

        self.cl_obj_norm_scale = CL_ElK(self.cl_ctx, name='obj_norm_scale',
                                        operation="ObjNormScale(i, obj_new, obj_norm, obj, regmax, reg, scale_sum,"
                                                  "nb_frame, nxyo, nbobj)",
                                        preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                                 getks('ptycho/opencl/psi_to_obj_probe_elw.cl'),
                                        options=self.cl_options,
                                        arguments="__global float2* obj_new, __global float* obj_norm,"
                                                  "__global float2* obj, float* regmax, const float reg,"
                                                  "__global float* scale_sum, const int nb_frame, const int nxyo,"
                                                  "const int nbobj")

        self.cl_grad_poisson_fourier = CL_ElK(self.cl_ctx, name='cl_grad_poisson_fourier',
                                              operation="GradPoissonFourier(i, iobs, psi, background, background_grad,"
                                                        "nbmode, nx, ny, nxystack, npsi, hann_filter, scale)",
                                              preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                                       getks('ptycho/opencl/grad_elw.cl'),
                                              options=self.cl_options,
                                              arguments="__global float *iobs, __global float2 *psi,"
                                                        "__global float *background, __global float *background_grad,"
                                                        "const int nbmode, const int nx, const int ny,"
                                                        "const int nxystack, const int npsi, const char hann_filter,"
                                                        "__global float* scale")

        self.cl_psi2obj_grad = CL_ElK(self.cl_ctx, name='psi_to_obj_grad',
                                      operation="GradObj(i, psi, obj_grad, probe, cx, cy, px, f, stack_size, nx, ny, "
                                                "nxo, nyo, nbobj, nbprobe, interp)",
                                      preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                               getks('ptycho/opencl/grad_elw.cl'),
                                      options=self.cl_options,
                                      arguments="__global float2* psi, __global float2 *obj_grad,"
                                                "__global float2* probe, const float cx, const float cy,"
                                                "const float px, const float f, const int stack_size, const int nx,"
                                                "const int ny, const int nxo, const int nyo, const int nbobj, "
                                                "const int nbprobe, const char interp")

        self.cl_psi_to_probe_grad = CL_ElK(self.cl_ctx, name='psi_to_probe_grad',
                                           operation="GradProbe(i, psi, probe_grad, obj, cx, cy, px, f, firstpass,"
                                                     "npsi, stack_size, nx, ny, nxo, nyo, nbobj, nbprobe, interp)",
                                           preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                                    getks('ptycho/opencl/grad_elw.cl'),
                                           options=self.cl_options,
                                           arguments="__global float2* psi, __global float2* probe_grad, "
                                                     "__global float2 *obj, __global float* cx, __global float* cy,"
                                                     "const float px, const float f, const char firstpass,"
                                                     "const int npsi, const int stack_size, const int nx, const int ny,"
                                                     "const int nxo, const int nyo, const int nbobj, const int nbprobe,"
                                                     "const char interp")

        self.cl_reg_grad = CL_ElK(self.cl_ctx, name='reg_grad',
                                  operation="GradReg(i, grad, v, alpha, nx, ny)",
                                  preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                           getks('ptycho/opencl/grad_elw.cl'),
                                  options=self.cl_options,
                                  arguments="__global float2* grad, __global float2* v, const float alpha,"
                                            "const int nx, const int ny")

        self.cl_circular_shift = CL_ElK(self.cl_ctx, name='cl_circular_shift',
                                        operation="circular_shift(i, source, dest, dx, dy, dz, nx, ny, nz)",
                                        preamble=getks('opencl/circular_shift.cl'),
                                        options=self.cl_options,
                                        arguments="__global float2* source, __global float2* dest, const int dx,"
                                                  "const int dy, const int dz, const int nx, const int ny,"
                                                  "const int nz")

        self.cl_psi2pos_merge = CL_ElK(self.cl_ctx, name='cl_psi2pos_merge',
                                       operation="cx[i] += dxy[i].x - dxy0[0].x; cy[i] += dxy[i].y - dxy0[0].y;",
                                       options=self.cl_options,
                                       arguments="__global float4* dxy, __global float2* dxy0, __global float* cx,"
                                                 "__global float* cy")

        self.cl_corr_phase_ramp = CL_ElK(self.cl_ctx, name='corr_phase_ramp',
                                         operation="CorrPhaseRamp2D(i, d, dx, dy, nx, ny)",
                                         preamble=getks('opencl/corr_phase_ramp_elw.cl'),
                                         options=self.cl_options,
                                         arguments="__global float2 *d, const float dx, const float dy,"
                                                   "const int nx, const int ny")

        # Reduction kernels
        self.cl_norm_complex_n = CL_RedK(self.cl_ctx, np.float32, neutral="0", reduce_expr="a+b",
                                         map_expr="pown(length(d[i]), nn)", options=self.cl_options,
                                         arguments="__global float2 *d, const int nn")

        # This will compute Poisson, Gaussian, Euclidian LLK as well as the sum of the calculated intensity
        self.cl_llk = CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)", reduce_expr="a+b",
                              preamble=getks('ptycho/opencl/llk_red.cl'), options=self.cl_options,
                              map_expr="LLKAll(i, iobs, psi, background, nbmode, nxy, nxystack)",
                              arguments="__global float *iobs, __global float2 *psi, __global float *background, const int nbmode, const int nxy, const int nxystack")

        self._cl_cg_polak_ribiere_complex_red = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                                        reduce_expr="a+b",
                                                        map_expr="PolakRibiereComplex(grad[i], lastgrad[i])",
                                                        preamble=getks('opencl/cg_polak_ribiere_red.cl'),
                                                        arguments="__global float2 *grad, __global float2 *lastgrad")

        self._cl_cg_polak_ribiere_complex_redf = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                                         reduce_expr="a+b",
                                                         map_expr="PolakRibiereFloat(grad[i], lastgrad[i])",
                                                         preamble=getks('opencl/cg_polak_ribiere_red.cl'),
                                                         arguments="__global float *grad, __global float *lastgrad")

        self._cl_cg_poisson_gamma_red = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                                reduce_expr="a+b",
                                                map_expr="CG_Poisson_Gamma(i, obs, PO, PdO, dPO, dPdO, background,"
                                                         "background_dir, nxy, nxystack, nbmode, npsi)",
                                                preamble=getks('ptycho/opencl/cg_gamma_red.cl'),
                                                options=self.cl_options,
                                                arguments="__global float *obs, __global float2 *PO,"
                                                          "__global float2 *PdO, __global float2 *dPO,"
                                                          "__global float2 *dPdO, __global float * background,"
                                                          "__global float * background_dir, const int nxy,"
                                                          "const int nxystack, const int nbmode, const int npsi")

        self.cl_psi2pos_stack_red = CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)", reduce_expr="a+b",
                                            preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                                     getks('ptycho/opencl/psi2pos.cl'),
                                            map_expr="Psi2PosShift(i, psi, obj, probe, cx, cy, pixel_size, f, nx, ny, "
                                                     "nxo, nyo, scale, ipsi, interp)",
                                            options=self.cl_options,
                                            arguments="__global float2* psi, __global float2 *obj,"
                                                      "__global float2* probe, __global float* cx, __global float* cy,"
                                                      "const float pixel_size, const float f, const int nx,"
                                                      "const int ny, const int nxo, const int nyo,"
                                                      "__global float* scale, const int ipsi, const char interp")

        self.cl_psi2pos_red = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)", reduce_expr="a+b",
                                      preamble=getks('opencl/complex.cl') + getks('opencl/bilinear.cl') +
                                               getks('ptycho/opencl/psi2pos.cl'),
                                      map_expr="Psi2PosRed(i, dxy, mult, max_shift, min_shift, thres[0], nb)",
                                      options=self.cl_options,
                                      arguments="__global float4* dxy, const float mult, const float max_shift,"
                                                "const float min_shift, const float* thres, const int nb")

        self.cl_psi2pos_thres_red = CL_RedK(self.cl_ctx, np.float32, neutral="0", reduce_expr="a+b",
                                            preamble=getks('opencl/complex.cl'),
                                            map_expr="native_sqrt(dxy[i].z*dxy[i].z + dxy[i].w*dxy[i].w) * thres / nb",
                                            options=self.cl_options,
                                            arguments="__global float4* dxy, const float thres, const int nb")

        # 4th order LLK(gamma) approximation
        self._cl_cg_poisson_gamma4_red = CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)",
                                                 reduce_expr="a+b",
                                                 map_expr="CG_Poisson_Gamma4(i, obs, PO, PdO, dPO, dPdO,"
                                                          "nxy, nxystack, nbmode)",
                                                 preamble=getks('ptycho/opencl/cg_gamma_red.cl'),
                                                 options=self.cl_options,
                                                 arguments="__global float *obs, __global float2 *PO,"
                                                           "__global float2 *PdO, __global float2 *dPO,"
                                                           "__global float2 *dPdO, const int nxy,"
                                                           "const int nxystack, const int nbmode")

        self._cl_cg_gamma_reg_red = CL_RedK(self.cl_ctx, cl.array.vec.float2, neutral="(float2)(0,0)",
                                            reduce_expr="a+b",
                                            map_expr="GammaReg(i, v, dv, nx, ny)",
                                            preamble=getks('opencl/cg_gamma_reg_red.cl'),
                                            arguments="__global float2 *v, __global float2 *dv,"
                                                      "const int nx, const int ny")

        self.cl_scale_intensity = CL_RedK(self.cl_ctx, cl.array.vec.float2, neutral="(float2)(0,0)",
                                          reduce_expr="a+b",
                                          map_expr="scale_intensity(i, obs, calc, background, nxy, nxystack, nb_mode)",
                                          preamble=getks('ptycho/opencl/scale_red.cl'),
                                          arguments=" __global float *obs, __global float2 *calc,"
                                                    "__global float *background, const int nxy,"
                                                    "const int nxystack, const int nb_mode")

        self.cl_center_mass = CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)",
                                      name="center_mass", reduce_expr="a+b",
                                      preamble=getks('opencl/center_mass_red.cl'),
                                      options=self.cl_options,
                                      map_expr="center_mass_float(i, d, nx, ny, nz, power)",
                                      arguments="__global float *d, const int nx, const int ny,"
                                                "const int nz, const int power")

        self.cl_center_mass_fftshift = CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)",
                                               name="center_mass_fftshift", reduce_expr="a+b",
                                               preamble=getks('opencl/center_mass_red.cl'),
                                               options=self.cl_options,
                                               map_expr="center_mass_fftshift_float(i, d, nx, ny, nz, power)",
                                               arguments="__global float *d, const int nx, const int ny,"
                                                         "const int nz, const int power")

        self.cl_center_mass_complex = CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)",
                                              reduce_expr="a+b",
                                              preamble=getks('opencl/center_mass_red.cl'), options=self.cl_options,
                                              map_expr="center_mass_complex(i, d, nx, ny, nz, power)",
                                              arguments="__global float2 *d, const int nx, const int ny, const int nz,"
                                                        "const int power")

        self.cl_center_mass_fftshift_complex = CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)",
                                                       name="center_mass_fftshift", reduce_expr="a+b",
                                                       preamble=getks('opencl/center_mass_red.cl'),
                                                       options=self.cl_options,
                                                       map_expr="center_mass_fftshift_complex(i, d, nx, ny, nz, power)",
                                                       arguments="__global float2 *d, const int nx, const int ny,"
                                                                 "const int nz, const int power")

        # Other kernels
        # Gaussian convolution kernels
        opt = self.cl_options + " -DBLOCKSIZE=16 -DHALFBLOCK=7"
        conv16_mod = cl.Program(self.cl_ctx, getks('opencl/convolution_complex.cl')).build(options=opt)
        self.gauss_convol_complex_16x = conv16_mod.gauss_convol_complex_x
        self.gauss_convol_complex_16y = conv16_mod.gauss_convol_complex_y
        self.gauss_convol_complex_16z = conv16_mod.gauss_convol_complex_z

        opt = self.cl_options + " -DBLOCKSIZE=32 -DHALFBLOCK=15"
        conv32_mod = cl.Program(self.cl_ctx, getks('opencl/convolution_complex.cl')).build(options=opt)
        self.gauss_convol_complex_32x = conv32_mod.gauss_convol_complex_x
        self.gauss_convol_complex_32y = conv32_mod.gauss_convol_complex_y
        self.gauss_convol_complex_32z = conv32_mod.gauss_convol_complex_z

        opt = self.cl_options + " -DBLOCKSIZE=64 -DHALFBLOCK=31"
        conv64_mod = cl.Program(self.cl_ctx, getks('opencl/convolution_complex.cl')).build(options=opt)
        self.gauss_convol_complex_64x = conv64_mod.gauss_convol_complex_x
        self.gauss_convol_complex_64y = conv64_mod.gauss_convol_complex_y
        self.gauss_convol_complex_64z = conv64_mod.gauss_convol_complex_z

        opt = self.cl_options + " -DBLOCKSIZE=16 -DHALFBLOCK=7"
        conv16f_mod = cl.Program(self.cl_ctx, getks('opencl/convolution16.cl')).build(options=opt)
        self.gauss_convolf_16x = conv16f_mod.gauss_convol_16x
        self.gauss_convolf_16y = conv16f_mod.gauss_convol_16y
        self.gauss_convolf_16z = conv16f_mod.gauss_convol_16z

        # Reduction kernels dictionaries which will be defined on-the-fly when the stack size will be known
        self.cl_projection_amplitude_red_v = {}
        self.cl_grad_poisson_fourier_red_v = {}

    def cl_init_kernel_n(self, n):
        """
        Init kernels specifically written for reduction with N-sized array (N being normally the stack_size)
        :param n: the size for the float_n arrays used
        :return: Nothing.
        """
        if n in self.cl_projection_amplitude_red_v:
            return
        else:
            k = CL_RedK(self.cl_ctx, get_or_register_dtype("float_n", np.dtype(('<f4', n))),
                        neutral="float_n_zero()", name='projection_amplitude_red',
                        reduce_expr="add(a,b)",
                        map_expr="ProjectionAmplitudeRed(i, iobs, dcalc, background, nbmode, nxy, nxystack, npsi)",
                        preamble=getks('opencl/float_n.cl')
                                 + getks('ptycho/opencl/projection_amplitude_red.cl'),
                        options=self.cl_options + " -DFLOAT_N_SIZE=%d " % n,
                        arguments="__global float *iobs, __global float2 *dcalc, __global float *background, "
                                  "const int nbmode, const int nxy, const int nxystack, const int npsi")
            self.cl_projection_amplitude_red_v[n] = k
        self.cl_grad_poisson_fourier_red_v[n] = CL_RedK(self.cl_ctx,
                                                        get_or_register_dtype("float_n", np.dtype(('<f4', n))),
                                                        neutral="float_n_zero()", name='grad_poisson_fourier_red',
                                                        reduce_expr="add(a,b)",
                                                        map_expr="GradPoissonFourierRed(i, iobs, psi, background,"
                                                                 "nbmode, nx, ny, nxystack, npsi, hann_filter, scale)",
                                                        preamble=getks('opencl/float_n.cl')
                                                                 + getks('ptycho/opencl/grad_red.cl'),
                                                        options=self.cl_options + " -DFLOAT_N_SIZE=%d " % n,
                                                        arguments="__global float *iobs, __global float2 *psi,"
                                                                  "__global float *background, const int nbmode, "
                                                                  "const int nx, const int ny, const int nxystack,"
                                                                  "const int npsi, const char hann_filter,"
                                                                  "__global float* scale")


"""
The default processing unit 
"""
default_processing_unit = CLProcessingUnitPtycho()


class CLObsDataStack:
    """
    Class to store a stack (e.g. 16 frames) of observed data in OpenCL space
    """

    def __init__(self, cl_obs, cl_x, cl_y, i, npsi):
        """
        
        :param cl_obs: pyopencl array of observed data, with N frames
        :param cl_x, cl_y: pyopencl arrays of the positions (in pixels) of the different frames
        :param i: index of the first frame
        :param npsi: number of valid frames (others are filled with zeros)
        """
        self.cl_obs = cl_obs
        self.cl_x = cl_x
        self.cl_y = cl_y
        self.i = np.int32(i)
        self.npsi = np.int32(npsi)
        self.x = cl_x.get()
        self.y = cl_y.get()


class CLOperatorPtycho(OperatorPtycho):
    """
    Base class for a operators on Ptycho objects using OpenCL
    """

    def __init__(self, processing_unit=None):
        super(CLOperatorPtycho, self).__init__()

        self.Operator = CLOperatorPtycho
        self.OperatorSum = CLOperatorPtychoSum
        self.OperatorPower = CLOperatorPtychoPower

        if processing_unit is None:
            self.processing_unit = default_processing_unit
        else:
            self.processing_unit = processing_unit
        if self.processing_unit.cl_ctx is None:
            # OpenCL kernels have not been prepared yet, use a default initialization
            if main_default_processing_unit.cl_device is None:
                main_default_processing_unit.select_gpu(language='opencl')

            self.processing_unit.init_cl(cl_device=main_default_processing_unit.cl_device, test_fft=False,
                                         verbose=False)

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
        return super(CLOperatorPtycho, self).apply_ops_mul(pty)

    def prepare_data(self, p):
        """
        Make sure the data to be used is in the correct memory (host or GPU) for the operator. 
        Virtual, must be derived.

        :param p: the Ptycho object the operator will be applied to.
        :return: 
        """
        if has_attr_not_none(p, "_cl_obs_v") is False:
            # Assume observed intensity is immutable, so transfer only once
            self.init_cl_vobs(p)
        elif len(p._cl_obs_v[0].cl_obs) != self.processing_unit.cl_stack_size:
            # This should not happen, but if tests are being made on the speed vs the stack size, this can be useful.
            self.init_cl_vobs(p)

        if p._timestamp_counter > p._cl_timestamp_counter:
            # print("Moving object, probe to OpenCL GPU")
            p._cl_obj = cla.to_device(self.processing_unit.cl_queue, p._obj)
            p._cl_probe = cla.to_device(self.processing_unit.cl_queue, p._probe)
            p._cl_timestamp_counter = p._timestamp_counter
            if p._background is None:
                p._cl_background = cla.zeros(self.processing_unit.cl_queue, p.data.iobs.shape[-2:], dtype=np.float32)
            else:
                p._cl_background = cla.to_device(self.processing_unit.cl_queue, p._background)

            # Also copy positions (allocation is done in init_cl_vobs())
            nb_frame, ny, nx = p.data.iobs.shape
            nyo, nxo = p._obj.shape[-2:]
            stack_size = self.processing_unit.cl_stack_size
            px, py = p.data.pixel_size_object()
            for i in range(0, nb_frame, stack_size):
                s = p._cl_obs_v[i // stack_size]
                vcx, vcy = s.x, s.y
                for j in range(stack_size):
                    ij = i + j
                    if ij < nb_frame:
                        dy, dx = p.data.posy[ij] / py, p.data.posx[ij] / px
                        cx, cy = get_view_coord((nyo, nxo), (ny, nx), dx, dy, integer=False)
                        vcx[j] = cx
                        vcy[j] = cy
                    else:
                        vcx[j] = vcx[0]
                        vcy[j] = vcy[0]
                s.cl_x = cla.to_device(self.processing_unit.cl_queue, s.x)
                s.cl_y = cla.to_device(self.processing_unit.cl_queue, s.y)

        need_init_psi = False
        if has_attr_not_none(p, "_cl_psi") is False:
            need_init_psi = True
        elif p._cl_psi.shape[0:3] != (len(p._obj), len(p._probe), self.processing_unit.cl_stack_size):
            need_init_psi = True
        if need_init_psi:
            ny, nx = p._probe.shape[-2:]
            p._cl_psi = cla.empty(self.processing_unit.cl_queue, dtype=np.complex64,
                                  shape=(len(p._obj), len(p._probe), self.processing_unit.cl_stack_size, ny, nx))

        if has_attr_not_none(p, "_cl_psi_v") is False or need_init_psi:
            # _cl_psi_v is used to hold the complete copy of Psi projections for all stacks, for algorithms
            # such as DM which need them.
            p._cl_psi_v = {}

        if has_attr_not_none(p, '_cl_view') is False:
            p._cl_view = {}

    def init_cl_vobs(self, p):
        """
        Initialize observed intensity and scan positions in OpenCL space

        :param p: the Ptycho object the operator will be applied to.
        :return: 
        """
        # print("Moving observed data to OpenCL GPU")
        p._cl_obs_v = []
        nb_frame, ny, nx = p.data.iobs.shape
        cl_stack_size = self.processing_unit.cl_stack_size
        for i in range(0, nb_frame, cl_stack_size):
            vobs = np.zeros((cl_stack_size, ny, nx), dtype=np.float32)
            for j in range(cl_stack_size):
                ij = i + j
                if ij < nb_frame:
                    vobs[j] = p.data.iobs[ij]
                else:
                    vobs[j] = np.zeros_like(vobs[0], dtype=np.float32)
            cl_vcx = cla.zeros(self.processing_unit.cl_queue, cl_stack_size, dtype=np.float32)
            cl_vcy = cla.zeros(self.processing_unit.cl_queue, cl_stack_size, dtype=np.float32)
            cl_vobs = cla.to_device(self.processing_unit.cl_queue, vobs)
            p._cl_obs_v.append(CLObsDataStack(cl_vobs, cl_vcx, cl_vcy, i, np.int32(min(cl_stack_size, nb_frame - i))))

        # Per-frame integrated intensity and scale factors, used for floating intensities
        iobs_sum = (p.data.iobs * (p.data.iobs > 0)).sum(axis=(-2, -1)).astype(np.float32)
        p._cl_iobs_sum = cla.to_device(self.processing_unit.cl_queue, iobs_sum.astype(np.float32))
        p._cl_icalc_sum = cla.zeros_like(p._cl_iobs_sum)
        p._cl_scale = cla.to_device(self.processing_unit.cl_queue, p.data.scale.astype(np.float32))

        # Size and index of current stack
        p._cl_stack_i = 0
        p._cl_stack_nb = len(p._cl_obs_v)

    def timestamp_increment(self, p):
        p._timestamp_counter += 1
        p._cl_timestamp_counter = p._timestamp_counter

    def view_register(self, obj):
        """
        Creates a new unique view key in an object. When finished with this view, it should be de-registered
        using view_purge. Note that it only reserves the key, but does not create the view.
        :return: an integer value, which corresponds to yet-unused key in the object's view.
        """
        i = 1
        while i in obj._cl_view:
            i += 1
        obj._cl_view[i] = None
        return i

    def view_copy(self, pty, i_source, i_dest):
        if i_source == 0:
            src = {'obj': pty._cl_obj, 'probe': pty._cl_probe, 'psi': pty._cl_psi, 'psi_v': pty._cl_psi_v}
        else:
            src = pty._cl_view[i_source]
        if i_dest == 0:
            pty._cl_obj = cla.empty_like(src['obj'])
            pty._cl_probe = cla.empty_like(src['probe'])
            pty._cl_psi = cla.empty_like(src['psi'])
            pty._cl_psi_v = {}
            dest = {'obj': pty._cl_obj, 'probe': pty._cl_probe, 'psi': pty._cl_psi, 'psi_v': pty._cl_psi_v}
        else:
            pty._cl_view[i_dest] = {'obj': cla.empty_like(src['obj']), 'probe': cla.empty_like(src['probe']),
                                    'psi': cla.empty_like(src['psi']), 'psi_v': {}}
            dest = pty._cl_view[i_dest]

        for i in range(len(src['psi_v'])):
            dest['psi_v'][i] = cla.empty_like(src['psi'])

        for s, d in zip([src['obj'], src['probe'], src['psi']] + [v for k, v in src['psi_v'].items()],
                        [dest['obj'], dest['probe'], dest['psi']] + [v for k, v in dest['psi_v'].items()]):
            cl.enqueue_copy(self.processing_unit.cl_queue, src=s.data, dest=d.data)

    def view_swap(self, pty, i1, i2):
        if i1 != 0:
            if pty._cl_view[i1] is None:
                # Create dummy value, assume a copy will be made later
                pty._cl_view[i1] = {'obj': None, 'probe': None, 'psi': None, 'psi_v': None}
        if i2 != 0:
            if pty._cl_view[i2] is None:
                # Create dummy value, assume a copy will be made later
                pty._cl_view[i2] = {'obj': None, 'probe': None, 'psi': None, 'psi_v': None}
        if i1 == 0:
            pty._cl_obj, pty._cl_view[i2]['obj'] = pty._cl_view[i2]['obj'], pty._cl_obj
            pty._cl_probe, pty._cl_view[i2]['probe'] = pty._cl_view[i2]['probe'], pty._cl_probe
            pty._cl_psi, pty._cl_view[i2]['psi'] = pty._cl_view[i2]['psi'], pty._cl_psi
            pty._cl_psi_v, pty._cl_view[i2]['psi_v'] = pty._cl_view[i2]['psi_v'], pty._cl_psi_v
        elif i2 == 0:
            pty._cl_obj, pty._cl_view[i1]['obj'] = pty._cl_view[i1]['obj'], pty._cl_obj
            pty._cl_probe, pty._cl_view[i1]['probe'] = pty._cl_view[i1]['probe'], pty._cl_probe
            pty._cl_psi, pty._cl_view[i1]['psi'] = pty._cl_view[i1]['psi'], pty._cl_psi
            pty._cl_psi_v, pty._cl_view[i1]['psi_v'] = pty._cl_view[i1]['psi_v'], pty._cl_psi_v
        else:
            pty._cl_view[i1], pty._cl_view[i2] = pty._cl_view[i2], pty._cl_view[i1]
        self.timestamp_increment(pty)

    def view_sum(self, pty, i_source, i_dest):
        if i_source == 0:
            src = {'obj': pty._cl_obj, 'probe': pty._cl_probe, 'psi': pty._cl_psi, 'psi_v': pty._cl_psi_v}
        else:
            src = pty._cl_view[i_source]
        if i_dest == 0:
            dest = {'obj': pty._cl_obj, 'probe': pty._cl_probe, 'psi': pty._cl_psi, 'psi_v': pty._cl_psi_v}
        else:
            dest = pty._cl_view[i_dest]
        for s, d in zip([src['obj'], src['probe'], src['psi']] + [v for k, v in src['psi_v'].items()],
                        [dest['obj'], dest['probe'], dest['psi']] + [v for k, v in dest['psi_v'].items()]):
            self.processing_unit.cl_sum(s, d)
        self.timestamp_increment(pty)

    def view_purge(self, pty, i):
        if i is not None:
            del pty._cl_view[i]
        elif has_attr_not_none(pty, '_cl_view'):
            del pty._cl_view


# The only purpose of this class is to make sure it inherits from CLOperatorPtycho and has a processing unit
class CLOperatorPtychoSum(OperatorSum, CLOperatorPtycho):
    def __init__(self, op1, op2):
        if np.isscalar(op1):
            op1 = Scale(op1)
        if np.isscalar(op2):
            op2 = Scale(op2)
        if isinstance(op1, CLOperatorPtycho) is False or isinstance(op2, CLOperatorPtycho) is False:
            raise OperatorException(
                "ERROR: cannot add a CLOperatorPtycho with a non-CLOperatorPtycho: %s + %s" % (str(op1), str(op2)))
        # We can only have a sum of two CLOperatorPtycho, so they must have a processing_unit attribute.
        CLOperatorPtycho.__init__(self, op1.processing_unit)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorPtycho
        self.OperatorSum = CLOperatorPtychoSum
        self.OperatorPower = CLOperatorPtychoPower
        self.prepare_data = types.MethodType(CLOperatorPtycho.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorPtycho.timestamp_increment, self)
        self.view_copy = types.MethodType(CLOperatorPtycho.view_copy, self)
        self.view_swap = types.MethodType(CLOperatorPtycho.view_swap, self)
        self.view_sum = types.MethodType(CLOperatorPtycho.view_sum, self)
        self.view_purge = types.MethodType(CLOperatorPtycho.view_purge, self)


# The only purpose of this class is to make sure it inherits from CLOperatorPtycho and has a processing unit
class CLOperatorPtychoPower(OperatorPower, CLOperatorPtycho):
    def __init__(self, op, n):
        CLOperatorPtycho.__init__(self, op.processing_unit)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorPtycho
        self.OperatorSum = CLOperatorPtychoSum
        self.OperatorPower = CLOperatorPtychoPower
        self.prepare_data = types.MethodType(CLOperatorPtycho.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorPtycho.timestamp_increment, self)
        self.view_copy = types.MethodType(CLOperatorPtycho.view_copy, self)
        self.view_swap = types.MethodType(CLOperatorPtycho.view_swap, self)
        self.view_sum = types.MethodType(CLOperatorPtycho.view_sum, self)
        self.view_purge = types.MethodType(CLOperatorPtycho.view_purge, self)


class FreePU(CLOperatorPtycho):
    """
    Operator freeing OpenCL memory. The fft plan/app in self.processing_unit is removed,
    as well as any OpenCL pyopencl.array.Array attribute in the supplied wavefront.
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        self.processing_unit.finish()
        p.from_pu()

        for o in dir(p):
            if isinstance(p.__getattribute__(o), cla.Array):
                p.__getattribute__(o).data.release()
                p.__setattr__(o, None)
        if has_attr_not_none(p, "_cl_psi_v"):
            for a in p._cl_psi_v.values():
                a.data.release()
            p._cl_psi_v = {}
        self.processing_unit.free_fft_plans()
        for v in p._cl_obs_v:
            for o in dir(v):
                if isinstance(v.__getattribute__(o), cla.Array):
                    v.__getattribute__(o).data.release()
        p._cl_obs_v = None
        gc.collect()
        return p

    def timestamp_increment(self, p):
        p._cl_timestamp_counter = 0


class MemUsage(CLOperatorPtycho):
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
            if isinstance(p.__getattribute__(o), cla.Array):
                if self.verbose:
                    p.print("   %40s %10.3fMbytes" % (o, p.__getattribute__(o).nbytes / 1e6))
                gpu_mem += p.__getattribute__(o).nbytes
        tmp_bytes = 0
        if has_attr_not_none(p, "_cl_psi_v"):
            for a in p._cl_psi_v.values():
                tmp_bytes += a.nbytes
            if self.verbose:
                p.print("   %40s %10.3fMbytes" % ("_cl_psi_v", tmp_bytes / 1e6))
        gpu_mem += tmp_bytes
        tmp_bytes = 0
        for v in p._cl_obs_v:
            for o in dir(v):
                if isinstance(v.__getattribute__(o), cla.Array):
                    tmp_bytes += v.__getattribute__(o).nbytes
        gpu_mem += tmp_bytes
        if self.verbose:
            p.print("   %40s %10.3fMbytes" % ("_cl_obs_v", tmp_bytes / 1e6))

        d = self.processing_unit.cl_device
        p.print("GPU used: %s [%4d Mbytes]" % (d.name, int(round(d.global_mem_size // 2 ** 20))))
        p.print("Mem Usage: RSS= %6.1f Mbytes (process), GPU Mem= %6.1f Mbytes (Ptycho object)" %
                (rss / 1024 ** 2, gpu_mem / 1024 ** 2))
        return p

    def prepare_data(self, p: Ptycho):
        # Overriden to avoid transferring any data to GPU
        pass

    def timestamp_increment(self, p):
        # This operator does nothing
        pass


class Scale(CLOperatorPtycho):
    """
    Multiply the ptycho object by a scalar (real or complex).
    """

    def __init__(self, x, obj=True, probe=True, psi=True):
        """

        :param x: the scaling factor
        :param obj: if True, scale the object
        :param probe: if True, scale the probe
        :param psi: if True, scale the all the psi arrays, _cl_psi as well as _cl_psi_v
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
            scale_k = pu.cl_scale
            x = np.float32(self.x)
        else:
            scale_k = pu.cl_scale_complex
            x = np.complex64(self.x)

        if self.obj:
            pu.ev = [scale_k(p._cl_obj, x, wait_for=pu.ev)]
        if self.probe:
            pu.ev = [scale_k(p._cl_probe, x, wait_for=pu.ev)]
        if self.psi:
            pu.ev = [scale_k(p._cl_psi, x, wait_for=pu.ev)]
            for i in range(len(p._cl_psi_v)):
                pu.ev = [scale_k(p._cl_psi_v[i], x, wait_for=pu.ev)]
        return p


class ObjProbe2Psi(CLOperatorPtycho):
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
        i = p._cl_stack_i
        i0 = p._cl_obs_v[i].i
        npsi = np.int32(p._cl_obs_v[i].npsi)
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        interp = np.int8(p._interpolation)
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))
        # print(i, f, p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size, nx, ny, nxo, nyo, nb_probe, nb_obj)
        # First argument is p._cl_psi[0] because the kernel will calculate the projection for all object and probe modes
        # and the full stack of frames.
        pu.ev = [pu.cl_object_probe_mult(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe, p._cl_obs_v[i].cl_x,
                                         p._cl_obs_v[i].cl_y, p.pixel_size_object, f, npsi, pu.cl_stack_size, nx, ny,
                                         nxo, nyo, nb_obj, nb_probe, p._cl_scale[i0:i0 + npsi], interp, wait_for=pu.ev)]
        if pu.profiling:
            if "object_probe_mult" not in pu.cl_event_profiling:
                pu.cl_event_profiling["object_probe_mult"] = []
            ev = CLEvent(pu.ev[-1], nx * ny * (12 + pu.cl_stack_size * nb_probe * nb_obj * 12),
                         nx * ny * (nb_probe * (1 + pu.cl_stack_size * nb_obj * 2)) * 8)
            pu.cl_event_profiling["object_probe_mult"].append(ev)
        return p


class FT(CLOperatorPtycho):
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
        s = pu.fft(p._cl_psi, p._cl_psi, ndim=2)

        # if pu.profiling:
        #     if "FT" not in pu.cl_event_profiling:
        #         pu.cl_event_profiling["FT"] = []
        #     nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
        #     ev = CLEvent(pu.ev[-1], 5 * nb_probe * nb_obj * nx * ny * nz * np.log2(nx * ny), p._cl_psi.nbytes * 2)
        #     pu.cl_event_profiling["FT"].append(ev)

        if self.scale:
            pu.ev = [pu.cl_scale(p._cl_psi, np.float32(s), wait_for=pu.ev)]

            if pu.profiling:
                if "FT:scale" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["FT:scale"] = []
                nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                ev = CLEvent(pu.ev[-1], nb_obj * nb_probe * nx * ny * nz * 2, p._cl_psi.nbytes * 2)
                pu.cl_event_profiling["FT:scale"].append(ev)

        return p


class IFT(CLOperatorPtycho):
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
        s = pu.ifft(p._cl_psi, p._cl_psi, ndim=2)

        # if pu.profiling:
        #     if "IFT" not in pu.cl_event_profiling:
        #         pu.cl_event_profiling["IFT"] = []
        #     nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
        #     ev = CLEvent(pu.ev[-1], 5 * nb_probe * nb_obj * nx * ny * nz * np.log2(nx * ny), p._cl_psi.nbytes * 2)
        #     pu.cl_event_profiling["IFT"].append(ev)

        if self.scale:
            pu.ev = [pu.cl_scale(p._cl_psi, np.float32(s), wait_for=pu.ev)]

            if pu.profiling:
                if "IFT:scale" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["IFT:scale"] = []
                nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                ev = CLEvent(pu.ev[-1], nb_obj * nb_probe * nx * ny * nz * 2, p._cl_psi.nbytes * 2)
                pu.cl_event_profiling["IFT:scale"].append(ev)

        return p


class QuadraticPhase(CLOperatorPtycho):
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
        pu.ev = [pu.cl_quad_phase(p._cl_psi, self.factor, self.scale, nx, ny, wait_for=pu.ev)]
        return p


class PropagateNearField(CLOperatorPtycho):
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
        p = IFT(scale=False) * QuadraticPhase(factor=f) * FT(scale=False) * p
        return p


class Calc2Obs1(CLOperatorPtycho):
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
        nxystack = np.int32(nxy * self.processing_unit.cl_stack_size)
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        i = p._cl_stack_i
        nb_psi = p._cl_obs_v[i].npsi
        pu.ev = [pu.cl_calc2obs(p._cl_obs_v[i].cl_obs[:nb_psi], p._cl_psi, p._cl_background,
                                nb_mode, nxy, nxystack, wait_for=pu.ev)]
        p.data.iobs[i * pu.cl_stack_size: i * pu.cl_stack_size + nb_psi] = p._cl_obs_v[i].cl_obs[:nb_psi].get()
        return p


class Calc2Obs(CLOperatorPtycho):
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
        p._cl_timestamp_counter = 0


class ApplyAmplitude(CLOperatorPtycho):
    """
    Apply the magnitude from observed intensities, keep the phase. Applies to a stack of N views.
    """

    def __init__(self, calc_llk=False, update_background=False, sum_icalc=False, background_smooth_sigma=0):
        """

        :param calc_llk: if True, the log-likelihood will be calculated for this stack.
        :param update_background: if True, update the background. The new background is
            automatically updated once the last stack is processed.
        :param sum_icalc: if True, will sum the calculated intensity on each frame, and store this in p._cl_icalc_sum,
                         to be used for floating intensities
        :param background_smooth_sigma: sigma for gaussian smoothing of the background
        """
        super(ApplyAmplitude, self).__init__()
        self.calc_llk = calc_llk
        self.update_background = update_background
        self.sum_icalc = sum_icalc
        self.background_smooth_sigma = np.float32(background_smooth_sigma)

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        # TODO: use a single-pass reduction kernel to apply the amplitude and compute the LLK
        if self.calc_llk:
            p = LLK() * p
        nxy = np.int32(p._probe.shape[-2] * p._probe.shape[-1])
        nxystack = np.int32(nxy * pu.cl_stack_size)
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        i = p._cl_stack_i
        nb_psi = np.int32(p._cl_obs_v[i].npsi)
        if self.update_background:
            first_pass = np.int8(p._cl_stack_i == 0)
            pu.ev = [pu.cl_projection_amplitude_background_mode(
                p._cl_obs_v[i].cl_obs[0], p._cl_psi, p._cl_background, p._cl_background_new,
                nb_mode, nxy, nxystack, nb_psi, first_pass, wait_for=pu.ev)]
            if pu.profiling:
                if "projection_amplitude_background" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["projection_amplitude_background"] = []
                # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["projection_amplitude_background"].append(ev)
        else:
            # TODO: also update background with floating intensities ?
            if self.sum_icalc:
                i0, n = p._cl_obs_v[i].i, pu.cl_stack_size
                pu.cl_init_kernel_n(n)
                res, ev = pu.cl_projection_amplitude_red_v[pu.cl_stack_size](p._cl_obs_v[i].cl_obs[0],
                                                                             p._cl_psi,
                                                                             p._cl_background,
                                                                             nb_mode, nxy, nxystack, nb_psi,
                                                                             wait_for=pu.ev,
                                                                             return_event=True,
                                                                             out=p._cl_icalc_sum[i0: i0 + n])
                pu.ev = [ev]
                if pu.profiling:
                    if "projection_amplitude_red" not in pu.cl_event_profiling:
                        pu.cl_event_profiling["projection_amplitude_red"] = []
                    ev = CLEvent(pu.ev[-1], 0, 0)
                    pu.cl_event_profiling["projection_amplitude_red"].append(ev)
            else:
                pu.ev = [pu.cl_projection_amplitude(p._cl_obs_v[i].cl_obs[0], p._cl_psi, p._cl_background,
                                                    nb_mode, nxy, nxystack, nb_psi, wait_for=pu.ev)]
                if pu.profiling:
                    if "projection_amplitude" not in pu.cl_event_profiling:
                        pu.cl_event_profiling["projection_amplitude"] = []
                    # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                    ev = CLEvent(pu.ev[-1], 0, 0)
                    pu.cl_event_profiling["projection_amplitude"].append(ev)
        # Merge background update
        if self.update_background and p._cl_stack_i == (p._cl_stack_nb - 1):
            n = np.int32(len(p.data.posx))
            pu.ev = [pu.cl_background_update_mode(p._cl_background, p._cl_background_new, n, wait_for=pu.ev)]

            if self.background_smooth_sigma > 0:
                # Smooth background ?
                if self.background_smooth_sigma > 3:
                    p = BackgroundFilter(self.background_smooth_sigma) * p
                else:
                    ny, nx = np.int32(p._background.shape[-2]), np.int32(p._background.shape[-1])
                    pu.ev = [pu.gauss_convolf_16x(pu.cl_queue, (16, ny, 1), (16, 1, 1),
                                                  p._cl_background.data, self.background_smooth_sigma,
                                                  nx, ny, np.int32(1), wait_for=pu.ev)]
                    if pu.profiling:
                        if "gauss_convolf_16x" not in pu.cl_event_profiling:
                            pu.cl_event_profiling["gauss_convolf_16x"] = []
                        ev = CLEvent(pu.ev[-1], 0, 0)
                        pu.cl_event_profiling["gauss_convolf_16x"].append(ev)
                    pu.ev = [pu.gauss_convolf_16y(pu.cl_queue, (nx, 16, 1), (1, 16, 1),
                                                  p._cl_background.data, self.background_smooth_sigma,
                                                  nx, ny, np.int32(1), wait_for=pu.ev)]
                    if pu.profiling:
                        if "gauss_convolf_16y" not in pu.cl_event_profiling:
                            pu.cl_event_profiling["gauss_convolf_16y"] = []
                        ev = CLEvent(pu.ev[-1], 0, 0)
                        pu.cl_event_profiling["gauss_convolf_16y"].append(ev)
        return p


class PropagateApplyAmplitude(CLOperatorPtycho):
    """
    Propagate to the detector plane (either in far or near field, perform the magnitude projection, and propagate
    back to the object plane. This applies to a stack of frames.
    """

    def __init__(self, calc_llk=False, update_background=False, background_smooth_sigma=0, sum_icalc=False):
        """

        :param calc_llk: if True, calculate llk while in the detector plane.
        :param update_background: if >0, update the background with the difference between
            the observed and calculated intensity (with a damping factor), averaged over all frames.
        :param background_smooth_sigma: sigma for the gaussian smoothing of the updated background
        :param sum_icalc: if True, update the background.
        """
        super(PropagateApplyAmplitude, self).__init__()
        self.calc_llk = calc_llk
        self.update_background = update_background
        self.sum_icalc = sum_icalc
        self.background_smooth_sigma = background_smooth_sigma

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        # TODO: use self.processing_unit.fft_scale() to correctly scale regardless of
        #  the FFT library and the normalisation used
        if p.data.near_field:
            p = PropagateNearField(forward=False) * \
                ApplyAmplitude(calc_llk=self.calc_llk, update_background=self.update_background,
                               background_smooth_sigma=self.background_smooth_sigma,
                               sum_icalc=self.sum_icalc) \
                * PropagateNearField(forward=True) * p
        else:
            p = IFT(scale=False) * \
                ApplyAmplitude(calc_llk=self.calc_llk, update_background=self.update_background,
                               background_smooth_sigma=self.background_smooth_sigma,
                               sum_icalc=self.sum_icalc) * FT(scale=False) * p
        return p


class LLK(CLOperatorPtycho):
    """
    Log-likelihood reduction kernel. Can only be used when Psi is in diffraction space.
    This is a reduction operator - it will write llk as an argument in the Ptycho object, and return the object.
    If _cl_stack_i==0, the llk is re-initialized. Otherwise it is added to the current value.

    The LLK can be calculated directly from object and probe using: p = LoopStack(LLK() * FT() * ObjProbe2Psi()) * p
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cl_stack_i
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        nb_psi = p._cl_obs_v[i].npsi
        nxy = np.int32(p._probe.shape[-2] * p._probe.shape[-1])
        nxystack = np.int32(self.processing_unit.cl_stack_size * nxy)
        llk = self.processing_unit.cl_llk(p._cl_obs_v[i].cl_obs[:nb_psi], p._cl_psi, p._cl_background,
                                          nb_mode, nxy, nxystack, wait_for=pu.ev).get()
        if p._cl_stack_i == 0:
            p.llk_poisson = llk['x']
            p.llk_gaussian = llk['y']
            p.llk_euclidian = llk['z']
            p.nb_photons_calc = llk['w']
        else:
            p.llk_poisson += llk['x']
            p.llk_gaussian += llk['y']
            p.llk_euclidian += llk['z']
            p.nb_photons_calc += llk['w']
        return p


class Psi2Obj(CLOperatorPtycho):
    """
    Computes updated Obj(r) contributions from Psi and Probe(r-r_j), for a stack of N probe positions.
    """

    def __init__(self, floating_intensity=False):
        """

        :param floating_intensity: if True, the intensity will be considered 'floating' from frame to frame, and
                                   a scale factor will be adjusted for each frame.
        """
        super(Psi2Obj, self).__init__()
        self.floating_intensity = floating_intensity

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cl_stack_i
        i0 = p._cl_obs_v[i].i
        # print("Psi2Obj(), i=%d"%(i))
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        npsi = np.int32(p._cl_obs_v[i].npsi)
        interp = np.int8(p._interpolation)
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(-np.pi / (p.data.wavelength * p.data.detector_distance))

        # First argument is p._cl_psi[0] because the kernel will calculate the projection for all object and probe modes
        # and the full stack of frames.
        if i == 0:
            # TODO: use a memory pool to avoid the cost of de/allocation every cycle
            p._cl_obj_new = cla.zeros(self.processing_unit.cl_queue, dtype=np.complex64, shape=(nb_obj, nyo, nxo))
            p._cl_obj_norm = cla.zeros(self.processing_unit.cl_queue, dtype=np.float32, shape=(nyo, nxo))
        if self.floating_intensity:
            pu.ev = [pu.cl_floating_scale_update(p._cl_scale[i0: i0 + npsi], p._cl_iobs_sum[i0: i0 + npsi],
                                                 p._cl_icalc_sum[i0: i0 + npsi], wait_for=pu.ev)]
            if pu.profiling:
                if "floating_scale_update" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["floating_scale_update"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["floating_scale_update"].append(ev)

        pu.ev = [pu.cl_psi2obj_atomic(p._cl_psi[0, 0, 0], p._cl_obj_new, p._cl_probe, p._cl_obj_norm,
                                      p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y, p.pixel_size_object, f,
                                      pu.cl_stack_size, nx, ny, nxo, nyo, nb_obj, nb_probe, npsi,
                                      p._cl_scale[i0: i0 + npsi], interp, wait_for=pu.ev)]
        if pu.profiling:
            if "psi2obj_atomic" not in pu.cl_event_profiling:
                pu.cl_event_profiling["psi2obj_atomic"] = []
            nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
            ev = CLEvent(pu.ev[-1], 0, 4 * nx * ny * nz * (2 * nb_probe * nb_obj + (2 + 1) * nb_obj))
            pu.cl_event_profiling["psi2obj_atomic"].append(ev)

        return p


class Psi2PosShift(CLOperatorPtycho):
    """
    Computes scan position shifts, by comparing the updated Psi array to object*probe, for a stack of frames.
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cl_stack_i
        i0 = p._cl_obs_v[i].i
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        npsi = np.int32(p._cl_obs_v[i].npsi)
        interp = np.int8(p._interpolation)
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(-np.pi / (p.data.wavelength * p.data.detector_distance))

        if i == 0:
            p._cl_dxy = cla.empty(self.processing_unit.cl_queue, dtype=cla.vec.float4, shape=(len(p.data.posx)))
        for ii in range(npsi):
            pu.ev = [pu.cl_psi2pos_stack_red(p._cl_psi[0, 0, ii], p._cl_obj, p._cl_probe, p._cl_obs_v[i].cl_x,
                                             p._cl_obs_v[i].cl_y, p.pixel_size_object, f, nx, ny, nxo, nyo,
                                             p._cl_scale, np.int32(ii), interp, out=p._cl_dxy[i0 + ii], wait_for=pu.ev,
                                             return_event=True)[1]]
        return p


class Psi2PosMerge(CLOperatorPtycho):
    """
    Merge scan position shifts, once the entire stqck of frames has been processed.
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
        self.threshold = threshold
        self.save_position_history = save_position_history

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """

        if self.save_position_history and (has_attr_not_none(p, 'position_history') is False):
            p.position_history = []
            for v in p._cl_obs_v:
                for ii in range(v.npsi):
                    p.position_history.append([(p.cycle, v.x[ii], v.y[ii])])

        pu = self.processing_unit
        n = np.int32(len(p.data.posx))

        # Average modulus of obj_grad*probe, multiplied by the relative threshold
        thres, ev = pu.cl_psi2pos_thres_red(p._cl_dxy, self.threshold, n, wait_for=pu.ev,
                                            return_event=True)

        # Shifts
        cl_dxy0, ev = pu.cl_psi2pos_red(p._cl_dxy, self.multiplier, self.max_displ, self.min_displ, thres,
                                        n, wait_for=[ev], return_event=True)
        pu.ev = [ev]
        for v in p._cl_obs_v:
            i0 = v.i
            npsi = v.npsi
            pu.ev = [pu.cl_psi2pos_merge(p._cl_dxy[i0:i0 + npsi], cl_dxy0, v.cl_x, v.cl_y, wait_for=pu.ev)]

        if self.save_position_history:
            for v in p._cl_obs_v:
                v.x = v.cl_x.get()
                v.y = v.cl_y.get()
                for ii in range(v.npsi):
                    p.position_history[v.i + ii].append((p.cycle, v.x[ii], v.y[ii]))

        return p


class Psi2Probe(CLOperatorPtycho):
    """
    Computes updated Probe contributions from Psi and Obj, for a stack of N probe positions.
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cl_stack_i
        i0 = p._cl_obs_v[i].i
        npsi = np.int32(p._cl_obs_v[i].npsi)
        first_pass = np.int8(i == 0)
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        interp = np.int8(p._interpolation)
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(-np.pi / (p.data.wavelength * p.data.detector_distance))

        if i == 0:
            p._cl_probe_new = cla.empty(pu.cl_queue, (nb_probe, ny, nx), dtype=np.complex64)
            p._cl_probe_norm = cla.empty(pu.cl_queue, (ny, nx), dtype=np.float32)

        # First argument is p._cl_psi[0] because the kernel will calculate the projection for all object and probe modes
        # and the full stack of frames.
        pu.ev = [pu.cl_psi_to_probe(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe_new, p._cl_probe_norm,
                                    p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y,
                                    p.pixel_size_object, f, first_pass, npsi, self.processing_unit.cl_stack_size,
                                    nx, ny, nxo, nyo, nb_obj, nb_probe, p._cl_scale[i0: i0 + npsi], interp,
                                    wait_for=pu.ev)]
        if pu.profiling:
            if "psi_to_probe" not in pu.cl_event_profiling:
                pu.cl_event_profiling["psi_to_probe"] = []
            nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
            ev = CLEvent(pu.ev[-1], 0, 4 * nx * ny * (2 * nz * nb_probe * nb_obj + (2 + 1) * nb_probe))
            pu.cl_event_profiling["psi_to_probe"].append(ev)

        return p


class Psi2ObjMerge(CLOperatorPtycho):
    """
    Call this when all stack of probe positions have been processed, and the final update of the object can
    be calculated. Temporary arrays are cleaned up
    """

    def __init__(self, inertia=1e-2, smooth_sigma=0, floating_intensity=False):
        """

        :param inertia: a regularisation factor to set the object inertia.
        :param floating_intensity: if True, the floating scale factors will be corrected so that the average is 1.
        :param smooth_sigma: if > 0, the previous object array (used for inertia) will be convolved
                             by a gaussian with this sigma.
        """
        super(Psi2ObjMerge, self).__init__()
        self.inertia = np.float32(inertia)
        self.floating_intensity = floating_intensity
        self.smooth_sigma = np.float32(smooth_sigma)

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        nb_obj = np.int32(p._obj.shape[0])
        nyo, nxo = p._obj.shape[-2:]
        nxyo = np.int32(nxo * nyo)

        if self.smooth_sigma > 8:
            nz, ny, nx = np.int32(nb_obj), np.int32(nyo), np.int32(nxo)
            pu.ev = [pu.gauss_convol_complex_64x(pu.cl_queue, (64, int(ny), int(nz)), (64, 1, 1), p._cl_obj.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "obj_merge_gauss_convol_complex_64x" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["obj_merge_gauss_convol_complex_64x"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["obj_merge_gauss_convol_complex_64x"].append(ev)

            pu.ev = [pu.gauss_convol_complex_64y(pu.cl_queue, (int(nx), 64, int(nz)), (1, 64, 1), p._cl_obj.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "obj_merge_gauss_convol_complex_64y" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["obj_merge_gauss_convol_complex_64y"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["obj_merge_gauss_convol_complex_64y"].append(ev)
        elif self.smooth_sigma > 4:
            nz, ny, nx = np.int32(nb_obj), np.int32(nyo), np.int32(nxo)
            pu.ev = [pu.gauss_convol_complex_32x(pu.cl_queue, (32, int(ny), int(nz)), (32, 1, 1), p._cl_obj.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "obj_merge_gauss_convol_complex_32x" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["obj_merge_gauss_convol_complex_32x"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["obj_merge_gauss_convol_complex_32x"].append(ev)

            pu.ev = [pu.gauss_convol_complex_64y(pu.cl_queue, (int(nx), 32, int(nz)), (1, 32, 1), p._cl_obj.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "obj_merge_gauss_convol_complex_32y" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["obj_merge_gauss_convol_complex_32y"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["obj_merge_gauss_convol_complex_32y"].append(ev)
        elif self.smooth_sigma > 0.1:
            nz, ny, nx = np.int32(nb_obj), np.int32(nyo), np.int32(nxo)
            pu.ev = [pu.gauss_convol_complex_16x(pu.cl_queue, (16, int(ny), int(nz)), (16, 1, 1), p._cl_obj.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "obj_merge_gauss_convol_complex_16x" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["obj_merge_gauss_convol_complex_16x"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["obj_merge_gauss_convol_complex_16x"].append(ev)

            pu.ev = [pu.gauss_convol_complex_16y(pu.cl_queue, (int(nx), 16, int(nz)), (1, 16, 1), p._cl_obj.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "obj_merge_gauss_convol_complex_16y" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["obj_merge_gauss_convol_complex_16y"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["obj_merge_gauss_convol_complex_16y"].append(ev)

        regmax = cla.max(p._cl_obj_norm)

        if self.floating_intensity:
            nb_frame = np.int32(len(p.data.iobs))
            scale_sum = cla.sum(p._cl_scale[0: nb_frame])
            pu.ev = [pu.cl_floating_scale_norm(p._cl_scale[0: nb_frame], scale_sum, nb_frame, wait_for=pu.ev)]

            pu.ev = [pu.cl_obj_norm_scale(p._cl_obj_new[0], p._cl_obj_norm, p._cl_obj, regmax, self.inertia,
                                          scale_sum, nb_frame, nxyo, nb_obj, wait_for=pu.ev)]
            if pu.profiling:
                if "obj_norm_scale" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["obj_norm_scale"] = []
                nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                ev = CLEvent(pu.ev[-1], 0, 4 * nx * ny * (2 * nz * nb_obj + 2 * nb_obj))
                pu.cl_event_profiling["obj_norm_scale"].append(ev)
        else:
            pu.ev = [pu.cl_obj_norm(p._cl_obj_new[0], p._cl_obj_norm, p._cl_obj, regmax, self.inertia, nxyo,
                                    nb_obj, wait_for=pu.ev)]

            if pu.profiling:
                if "obj_norm" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["obj_norm"] = []
                nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                ev = CLEvent(pu.ev[-1], 0, 4 * nx * ny * (2 * nz * nb_obj + 2 * nb_obj))
                pu.cl_event_profiling["obj_norm"].append(ev)

        # Clean up
        del p._cl_obj_norm, p._cl_obj_new

        return p


class Psi2ProbeMerge(CLOperatorPtycho):
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
        ny, nx = p.data.iobs.shape[-2:]
        nxy = np.int32(nx * ny)

        if self.smooth_sigma > 8:
            nz, ny, nx = np.int32(nb_probe), np.int32(ny), np.int32(nx)
            pu.ev = [pu.gauss_convol_complex_64x(pu.cl_queue, (64, int(ny), int(nz)), (64, 1, 1), p._cl_probe.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "probe_merge_gauss_convol_complex_64x" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["probe_merge_gauss_convol_complex_64x"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["probe_merge_gauss_convol_complex_64x"].append(ev)

            pu.ev = [pu.gauss_convol_complex_64y(pu.cl_queue, (int(nx), 64, int(nz)), (1, 64, 1), p._cl_probe.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "probe_merge_gauss_convol_complex_64y" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["probe_merge_gauss_convol_complex_64y"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["probe_merge_gauss_convol_complex_64y"].append(ev)
        elif self.smooth_sigma > 4:
            nz, ny, nx = np.int32(nb_probe), np.int32(ny), np.int32(nx)
            pu.ev = [pu.gauss_convol_complex_32x(pu.cl_queue, (32, int(ny), int(nz)), (32, 1, 1), p._cl_probe.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "probe_merge_gauss_convol_complex_32x" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["probe_merge_gauss_convol_complex_32x"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["probe_merge_gauss_convol_complex_32x"].append(ev)

            pu.ev = [pu.gauss_convol_complex_64y(pu.cl_queue, (int(nx), 32, int(nz)), (1, 32, 1), p._cl_probe.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "probe_merge_gauss_convol_complex_32y" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["probe_merge_gauss_convol_complex_32y"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["probe_merge_gauss_convol_complex_32y"].append(ev)
        elif self.smooth_sigma > 0.1:
            nz, ny, nx = np.int32(nb_probe), np.int32(ny), np.int32(nx)
            pu.ev = [pu.gauss_convol_complex_16x(pu.cl_queue, (16, int(ny), int(nz)), (16, 1, 1), p._cl_probe.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "probe_merge_gauss_convol_complex_16x" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["probe_merge_gauss_convol_complex_16x"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["probe_merge_gauss_convol_complex_16x"].append(ev)

            pu.ev = [pu.gauss_convol_complex_16y(pu.cl_queue, (int(nx), 16, int(nz)), (1, 16, 1), p._cl_probe.data,
                                                 self.smooth_sigma, nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "probe_merge_gauss_convol_complex_16y" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["probe_merge_gauss_convol_complex_16y"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["probe_merge_gauss_convol_complex_16y"].append(ev)

        normmax = cla.max(p._cl_probe_norm)

        pu.ev = [pu.cl_obj_norm(p._cl_probe_new[0], p._cl_probe_norm, p._cl_probe, normmax, self.inertia, nxy, nb_probe,
                                wait_for=pu.ev)]

        if pu.profiling:
            if "probe_norm" not in pu.cl_event_profiling:
                pu.cl_event_profiling["probe_norm"] = []
            # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
            ev = CLEvent(pu.ev[-1], 0, 0)
            pu.cl_event_profiling["probe_norm"].append(ev)

        # Clean up
        del p._cl_probe_norm, p._cl_probe_new

        return p


class AP(CLOperatorPtycho):
    """
    Perform a complete Alternating Projection cycle:
    - forward all object*probe views to Fourier space and apply the observed amplitude
    - back-project to object space and project onto (probe, object)
    - update background optionally
    """

    def __init__(self, update_object=True, update_probe=False, update_background=False, floating_intensity=False,
                 nb_cycle=1, calc_llk=False, show_obj_probe=False, fig_num=-1, obj_smooth_sigma=0, obj_inertia=0.01,
                 probe_smooth_sigma=0, probe_inertia=0.001, update_pos=False, pos_mult=1,
                 pos_max_shift=2, pos_min_shift=0, pos_threshold=0.2, pos_history=False, zero_phase_ramp=True,
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
        :param update_pos: if True, update positions. This automatically inhibits object and probe update
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
        self.floating_intensity = floating_intensity
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
        :return: a new DM operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return AP(update_object=self.update_object, update_probe=self.update_probe,
                  update_background=self.update_background, floating_intensity=self.floating_intensity,
                  nb_cycle=self.nb_cycle * n, calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe,
                  fig_num=self.fig_num, obj_smooth_sigma=self.obj_smooth_sigma, obj_inertia=self.obj_inertia,
                  probe_smooth_sigma=self.probe_smooth_sigma, probe_inertia=self.probe_inertia,
                  update_pos=self.update_pos, pos_max_shift=self.pos_max_shift, pos_min_shift=self.pos_min_shift,
                  pos_threshold=self.pos_threshold, pos_mult=self.pos_mult, pos_history=self.pos_history,
                  zero_phase_ramp=self.zero_phase_ramp, background_smooth_sigma=self.background_smooth_sigma)

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        if self.update_background:
            p._cl_background_new = cla.empty_like(p._cl_background)

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
                                          background_smooth_sigma=self.background_smooth_sigma,
                                          sum_icalc=self.floating_intensity) * ObjProbe2Psi()
            if self.update_object:
                ops = Psi2Obj(floating_intensity=self.floating_intensity) * ops
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
                p = Psi2ObjMerge(floating_intensity=self.floating_intensity, smooth_sigma=self.obj_smooth_sigma,
                                 inertia=self.obj_inertia) * p
            if self.update_probe:
                p = Psi2ProbeMerge(smooth_sigma=self.probe_smooth_sigma, inertia=self.probe_inertia) * p

            if calc_llk:
                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=self.update_pos,
                                 algorithm='AP', verbose=True)
            else:
                p.history.insert(p.cycle, update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=self.update_pos, algorithm='AP',
                                 verbose=False)
            p.cycle += 1

            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    s = algo_string('AP', p, self.update_object, self.update_probe, self.update_background,
                                    self.update_pos)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.get_llk('poisson'))
                    if self.zero_phase_ramp:
                        p = ZeroPhaseRamp(obj=True) * p
                    p = ShowObjProbe(fig_num=self.fig_num, title=tit) * p

        if self.update_background:
            del p._cl_background_new
        if self.zero_phase_ramp and not self.show_obj_probe:
            p = ZeroPhaseRamp(obj=True) * p
        return p


class DM1(CLOperatorPtycho):
    """
    Equivalent to operator: 2 * ObjProbe2Psi() - 1
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cl_stack_i
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        interp = np.int8(p._interpolation)
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))
        pu.ev = [pu.cl_2object_probe_psi_dm1(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe,
                                             p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y,
                                             p.pixel_size_object, f,
                                             p._cl_obs_v[i].npsi, pu.cl_stack_size,
                                             nx, ny, nxo, nyo, nb_obj, nb_probe, p._cl_scale, interp, wait_for=pu.ev)]
        if pu.profiling:
            if "2object_probe_psi_dm1" not in pu.cl_event_profiling:
                pu.cl_event_profiling["2object_probe_psi_dm1"] = []
            # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
            ev = CLEvent(pu.ev[-1], 0, 0)
            pu.cl_event_profiling["2object_probe_psi_dm1"].append(ev)
        return p


class DM2(CLOperatorPtycho):
    """
    # Psi(n+1) = Psi(n) - P*O + Psi_fourier

    This operator assumes that Psi_fourier is the current Psi, and that Psi(n) is in p._cl_psi_v

    On output Psi(n+1) is the current Psi, and Psi_fourier has been swapped to p._cl_psi_v
    """

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cl_stack_i
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        interp = np.int8(p._interpolation)
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))
        # Swap p._cl_psi_v_copy = Psi(n) with p._cl_psi = Psi_fourier
        p._cl_psi_copy, p._cl_psi = p._cl_psi, p._cl_psi_copy
        pu.ev = [pu.cl_2object_probe_psi_dm2(p._cl_psi[0, 0, 0], p._cl_psi_copy, p._cl_obj, p._cl_probe,
                                             p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y,
                                             p.pixel_size_object, f,
                                             p._cl_obs_v[i].npsi, pu.cl_stack_size,
                                             nx, ny, nxo, nyo, nb_obj, nb_probe, p._cl_scale, interp, wait_for=pu.ev)]
        if pu.profiling:
            if "2object_probe_psi_dm2" not in pu.cl_event_profiling:
                pu.cl_event_profiling["2object_probe_psi_dm2"] = []
            # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
            ev = CLEvent(pu.ev[-1], 0, 0)
            pu.cl_event_profiling["2object_probe_psi_dm2"].append(ev)

        return p


class DM(CLOperatorPtycho):
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
        self.floating_intensity = False  # Unstable with DM, so disabled
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.obj_smooth_sigma = obj_smooth_sigma
        self.obj_inertia = obj_inertia
        self.probe_smooth_sigma = probe_smooth_sigma
        self.probe_inertia = probe_inertia
        self.center_probe_n = center_probe_n
        self.center_probe_max_shift = center_probe_max_shift
        self.loop_obj_probe = loop_obj_probe
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
        # TODO: to the current psi array

        if self.update_background:
            p._cl_background_new = cla.empty_like(p._cl_background)

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
                                              background_smooth_sigma=self.background_smooth_sigma,
                                              sum_icalc=self.floating_intensity) * DM1()
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
                    # Do not update object and probe if positions are updated
                    # Loop the object and probe update if both are done at the same time. Slow, more stable ?
                    nb_loop_update_obj_probe = 1
                    if self.update_probe and self.update_object:
                        nb_loop_update_obj_probe = self.loop_obj_probe

                    for i in range(nb_loop_update_obj_probe):
                        if self.update_object:
                            p = Psi2ObjMerge(floating_intensity=self.floating_intensity,
                                             smooth_sigma=self.obj_smooth_sigma, inertia=self.obj_inertia) \
                                * LoopStack(Psi2Obj(floating_intensity=self.floating_intensity), keep_psi=True) * p
                        if self.update_probe:
                            p = Psi2ProbeMerge(smooth_sigma=self.probe_smooth_sigma,
                                               inertia=self.probe_inertia) * LoopStack(Psi2Probe(), keep_psi=True) * p
                else:
                    # TODO: updating probe and object at the same time does not work as in AP. Why ?
                    # Probably due to a scaling issue, as Psi is not a direct back-propagation but the result of DM2
                    ops = 1
                    if self.update_object:
                        ops = Psi2Obj(floating_intensity=floating_intensity) * ops
                    if self.update_probe:
                        ops = Psi2Probe() * ops

                    p = LoopStack(ops, keep_psi=True) * p

                    if self.update_object:
                        p = Psi2ObjMerge(floating_intensity=floating_intensity, smooth_sigma=self.obj_smooth_sigma,
                                         inertia=self.obj_inertia) * p
                    if self.update_probe:
                        p = Psi2ProbeMerge(smooth_sigma=self.probe_smooth_sigma, inertia=self.probe_inertia) * p
            else:
                # Update obj and probe immediately after back-propagation, before DM2 ?
                # Does not seem to give very good results
                ops = PropagateApplyAmplitude(sum_icalc=self.floating_intensity) * DM1()
                if self.update_object:
                    ops = Psi2Obj(floating_intensity=floating_intensity) * ops
                if self.update_probe:
                    ops = Psi2Probe() * ops

                p = LoopStack(DM2() * ops, keep_psi=True, copy=True) * p
                if self.update_object:
                    p = Psi2ObjMerge(floating_intensity=floating_intensity, smooth_sigma=self.obj_smooth_sigma,
                                     inertia=self.obj_inertia) * p
                if self.update_probe:
                    p = Psi2ProbeMerge(smooth_sigma=self.probe_smooth_sigma, inertia=self.probe_inertia) * p

            if self.center_probe_n > 0 and p.data.near_field is False:
                if (ic % self.center_probe_n) == 0:
                    p = CenterObjProbe(max_shift=self.center_probe_max_shift, verbose=False) * p

            if calc_llk:
                # Keep a copy of current Psi
                cl_psi0 = p._cl_psi.copy()
                # We need to perform a loop for LLK as the DM2 loop is on (2*PO-I), not the current PO estimate
                if p.data.near_field:
                    p = LoopStack(LLK() * PropagateNearField() * ObjProbe2Psi()) * p
                else:
                    p = LoopStack(LLK() * FT(scale=False) * ObjProbe2Psi()) * p

                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=self.update_pos, algorithm='DM',
                                 verbose=True)
                # TODO: find a   better place to do this rescaling, only useful to avoid obj/probe divergence
                p = ScaleObjProbe(absolute=False) * p
                # Restore correct Psi
                p._cl_psi = cl_psi0
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
        p._cl_psi_v = {}
        if self.update_background:
            del p._cl_background_new
        gc.collect()
        if self.zero_phase_ramp and not self.show_obj_probe:
            p = ZeroPhaseRamp(obj=False) * p
        return p


class _Grad(CLOperatorPtycho):
    """
    Operator to compute the object and/or probe and/or background gradient corresponding to the current stack.
    """

    def __init__(self, update_object=True, update_probe=False, update_background=False, floating_intensity=False,
                 calc_llk=False):
        """
        :param update_object: compute gradient for the object ?
        :param update_probe: compute gradient for the probe ?
        :param update_background: compute gradient for the background ?
        :param floating_intensity: if True, will sum the calculated intensity on each frame, and store this in
                                   p._cl_icalc_sum, to be used for floating intensities.
        :param calc_llk: calculate llk while in Fourier space
        """
        super(_Grad, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.floating_intensity = floating_intensity
        self.calc_llk = calc_llk

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cl_stack_i
        npsi = np.int32(p._cl_obs_v[i].npsi)
        i0 = p._cl_obs_v[i].i
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nb_mode = np.int32(nb_obj * nb_probe)
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        first_pass = np.int8(i == 0)
        nxystack = np.int32(pu.cl_stack_size * nx * ny)
        hann_filter = np.int8(1)
        interp = np.int8(p._interpolation)
        if p.data.near_field:
            f = np.float32(0)
            hann_filter = np.int8(0)
        else:
            f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))

        # Obj * Probe
        p = ObjProbe2Psi() * p

        # To detector plane
        if p.data.near_field:
            p = PropagateNearField() * p
        else:
            p = FT(scale=False) * p

        if self.calc_llk:
            p = LLK() * p

        # Calculate Psi.conj() * (1-Iobs/I_calc) [for Poisson Gradient)
        # TODO: different noise models
        if self.floating_intensity:
            # TODO: background update along floating intensity
            n = pu.cl_stack_size
            pu.cl_init_kernel_n(n)
            res, ev = pu.cl_grad_poisson_fourier_red_v[n](p._cl_obs_v[i].cl_obs[0], p._cl_psi, p._cl_background,
                                                          nb_mode, nx, ny, nxystack, npsi, hann_filter,
                                                          p._cl_scale[i0:i0 + npsi],
                                                          wait_for=pu.ev, return_event=True,
                                                          out=p._cl_icalc_sum[i0: i0 + npsi])
            pu.ev = [ev]
            if pu.profiling:
                if "grad_poisson_fourier_red" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["grad_poisson_fourier_red"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["grad_poisson_fourier_red"].append(ev)

            pu.ev = [pu.cl_floating_scale_update(p._cl_scale[i0: i0 + npsi], p._cl_iobs_sum[i0: i0 + npsi],
                                                 p._cl_icalc_sum[i0: i0 + npsi], wait_for=pu.ev)]
            if pu.profiling:
                if "floating_scale_update" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["floating_scale_update"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["floating_scale_update"].append(ev)
        else:
            pu.ev = [pu.cl_grad_poisson_fourier(p._cl_obs_v[i].cl_obs[0], p._cl_psi, p._cl_background,
                                                p._cl_background_grad, nb_mode, nx, ny, nxystack, npsi,
                                                hann_filter, p._cl_scale[i0:i0 + npsi], wait_for=pu.ev)]
            if pu.profiling:
                if "grad_poisson_fourier" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["grad_poisson_fourier"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["grad_poisson_fourier"].append(ev)

        if p.data.near_field:
            p = PropagateNearField(forward=False) * p
        else:
            p = IFT(scale=False) * p

        if self.update_object:
            for ii in range(p._cl_obs_v[i].npsi):
                pu.ev = [pu.cl_psi2obj_grad(p._cl_psi[0, 0, ii], p._cl_obj_grad, p._cl_probe,
                                            p._cl_obs_v[i].x[ii], p._cl_obs_v[i].y[ii],
                                            p.pixel_size_object, f, pu.cl_stack_size,
                                            nx, ny, nxo, nyo, nb_obj, nb_probe, interp, wait_for=pu.ev)]
                if pu.profiling:
                    if "psi_to_obj_grad" not in pu.cl_event_profiling:
                        pu.cl_event_profiling["psi_to_obj_grad"] = []
                    # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                    ev = CLEvent(pu.ev[-1], 0, 0)
                    pu.cl_event_profiling["psi_to_obj_grad"].append(ev)
        if self.update_probe:
            pu.ev = [pu.cl_psi_to_probe_grad(p._cl_psi[0, 0, 0], p._cl_probe_grad, p._cl_obj,
                                             p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y,
                                             p.pixel_size_object, f, first_pass,
                                             npsi, pu.cl_stack_size,
                                             nx, ny, nxo, nyo, nb_obj, nb_probe, interp, wait_for=pu.ev)]
            if pu.profiling:
                if "psi_to_probe_grad" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["psi_to_probe_grad"] = []
                # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["psi_to_probe_grad"].append(ev)
        return p


class Grad(CLOperatorPtycho):
    """
    Operator to compute the object and/or probe and/or background gradient. The gradient is stored
    in the ptycho object. It is assumed that the GPU gradient arrays have been already created, normally
    by the calling ML operator.
    """

    def __init__(self, update_object=True, update_probe=False, update_background=False, floating_intensity=False,
                 reg_fac_obj=0, reg_fac_probe=0, calc_llk=False):
        """

        :param update_object: compute gradient for the object ?
        :param update_probe: compute gradient for the probe ?
        :param update_background: compute gradient for the background ?
        :param floating_intensity: optimise floating intensity scale factor
        :param calc_llk: calculate llk while in Fourier space
        """
        super(Grad, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.floating_intensity = floating_intensity
        self.calc_llk = calc_llk
        self.reg_fac_obj = reg_fac_obj
        self.reg_fac_probe = reg_fac_probe

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        if self.update_object:
            p._cl_obj_grad.fill(np.complex64(0))

        p = LoopStack(_Grad(update_object=self.update_object, update_probe=self.update_probe,
                            update_background=self.update_background, floating_intensity=self.floating_intensity,
                            calc_llk=self.calc_llk)) * p
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
            pu.ev = [pu.cl_reg_grad(p._cl_obj_grad, p._cl_obj, reg_fac_obj, nxo, nyo, wait_for=pu.ev)]
            if pu.profiling:
                if "reg_grad_obj" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["reg_grad_obj"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["reg_grad_obj"].append(ev)

        if self.update_probe and reg_fac_probe > 0:
            # Regularisation contribution to the probe gradient
            pu.ev = [pu.cl_reg_grad(p._cl_probe_grad, p._cl_probe, reg_fac_probe, nx, ny, wait_for=pu.ev)]
            if pu.profiling:
                if "reg_grad_probe" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["reg_grad_probe"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["reg_grad_probe"].append(ev)

        return p


class _CGGamma(CLOperatorPtycho):
    """
    Operator to compute the conjugate gradient gamma contribution to the current stack.
    """

    def __init__(self, update_background=False):
        """
        :param update_background: if updating the background ?
        """
        super(_CGGamma, self).__init__()
        self.update_background = update_background
        # TODO: fix this scale dynamically ? Used to avoid overflows

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        i = p._cl_stack_i
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nb_mode = np.int32(nb_obj * nb_probe)
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        nb_psi = np.int32(p._cl_obs_v[i].npsi)
        nxy = np.int32(ny * nx)
        nxystack = np.int32(pu.cl_stack_size * nxy)
        interp = np.int8(p._interpolation)
        if p.data.near_field:
            f = np.float32(0)
        else:
            f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))

        for clpsi, clobj, clprobe in zip([p._cl_PO, p._cl_PdO, p._cl_dPO, p._cl_dPdO],
                                         [p._cl_obj, p._cl_obj_dir, p._cl_obj, p._cl_obj_dir],
                                         [p._cl_probe, p._cl_probe, p._cl_probe_dir, p._cl_probe_dir]):

            pu.ev = [pu.cl_object_probe_mult(clpsi[0, 0, 0], clobj, clprobe,
                                             p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y,
                                             p.pixel_size_object, f, nb_psi, pu.cl_stack_size, nx, ny, nxo, nyo,
                                             nb_obj, nb_probe, p._cl_scale, interp, wait_for=pu.ev)]
            if pu.profiling:
                if "object_probe_mult" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["object_probe_mult"] = []
                # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["object_probe_mult"].append(ev)

            # switch clpsi and p._cl_psi for propagation
            clpsi, p._cl_psi = p._cl_psi, clpsi
            if p.data.near_field:
                p = PropagateNearField(forward=True) * p
            else:
                p = FT(scale=False) * p

        # TODO: take into account background
        tmp = self.processing_unit._cl_cg_poisson_gamma_red(p._cl_obs_v[i].cl_obs[0], p._cl_PO, p._cl_PdO,
                                                            p._cl_dPO, p._cl_dPdO, p._cl_background,
                                                            p._cl_background_dir, nxy, nxystack, nb_mode, nb_psi,
                                                            wait_for=pu.ev).get()
        # if np.isnan(tmp['x'] + tmp['y']) or np.isinf(tmp['x'] + tmp['y']):
        #     nP = self.processing_unit.cl_norm_complex_n(p._cl_probe, 2).get()
        #     nO = self.processing_unit.cl_norm_complex_n(p._cl_obj, 2).get()
        #     ndP = self.processing_unit.cl_norm_complex_n(p._cl_probe_dir, 2).get()
        #     ndO = self.processing_unit.cl_norm_complex_n(p._cl_obj_dir, 2).get()
        #     nPO = self.processing_unit.cl_norm_complex_n(p._cl_PO, 2).get()
        #     ndPO = self.processing_unit.cl_norm_complex_n(p._cl_dPO, 2).get()
        #     nPdO = self.processing_unit.cl_norm_complex_n(p._cl_PdO, 2).get()
        #     ndPdO = self.processing_unit.cl_norm_complex_n(p._cl_dPdO, 2).get()
        #     p.print('_CGGamma norms: P %e O %e dP %e dO %e PO %e, PdO %e, dPO %e, dPdO %e' % (
        #         nP, nO, ndP, ndO, nPO, ndPO, nPdO, ndPdO))
        #     p.print('_CGGamma (stack #%d, NaN Gamma:)' % i, tmp['x'], tmp['y'])
        #     raise OperatorException("NaN")

        p._cl_cg_gamma_d += tmp['y']
        p._cl_cg_gamma_n += tmp['x']
        if False:
            tmp = self.processing_unit._cl_cg_poisson_gamma4_red(p._cl_obs_v[i].cl_obs[:nb_psi], p._cl_PO, p._cl_PdO,
                                                                 p._cl_dPO, p._cl_dPdO,
                                                                 nxy, nxystack, nb_mode).get()
            p._cl_cg_gamma4 += np.array((tmp['w'], tmp['z'], tmp['y'], tmp['x'], 0))

        if self.update_background:
            # TODO: use a different kernel if there is a background gradient
            pass
        return p


class ML(CLOperatorPtycho):
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
        :param floating_intensity: optimise floating intensity scale factor
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
        self.floating_intensity = floating_intensity
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
                  update_background=self.update_background, floating_intensity=self.floating_intensity,
                  reg_fac_obj=self.reg_fac_obj, reg_fac_probe=self.reg_fac_probe, calc_llk=self.calc_llk,
                  show_obj_probe=self.show_obj_probe, fig_num=self.fig_num, update_pos=self.update_pos,
                  pos_max_shift=self.pos_max_shift, pos_min_shift=self.pos_min_shift, pos_threshold=self.pos_threshold,
                  pos_mult=self.pos_mult, pos_history=self.pos_history, zero_phase_ramp=self.zero_phase_ramp)

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
        cl_queue = pu.cl_queue

        # Create the necessary GPU arrays for ML
        p._cl_PO = cla.empty_like(p._cl_psi)
        p._cl_PdO = cla.empty_like(p._cl_psi)
        p._cl_dPO = cla.empty_like(p._cl_psi)
        p._cl_dPdO = cla.empty_like(p._cl_psi)
        p._cl_obj_dir = cla.zeros(cl_queue, (nb_obj, nyo, nxo), np.complex64)
        p._cl_probe_dir = cla.zeros(cl_queue, (nb_probe, ny, nx), np.complex64)
        if self.update_object:
            p._cl_obj_grad = cla.empty(cl_queue, (nb_obj, nyo, nxo), np.complex64)
            p._cl_obj_grad_last = cla.empty(cl_queue, (nb_obj, nyo, nxo), np.complex64)
        if self.update_probe:
            p._cl_probe_grad = cla.empty(cl_queue, (nb_probe, ny, nx), np.complex64)
            p._cl_probe_grad_last = cla.empty(cl_queue, (nb_probe, ny, nx), np.complex64)
        p._cl_background_grad = cla.zeros(cl_queue, (ny, nx), np.float32)
        p._cl_background_dir = cla.zeros(cl_queue, (ny, nx), np.float32)
        if self.update_background:
            p._cl_background_grad_last = cla.zeros(cl_queue, (ny, nx), np.float32)

        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            # Swap gradient arrays - for CG, we need the previous gradient
            if self.update_object:
                p._cl_obj_grad, p._cl_obj_grad_last = p._cl_obj_grad_last, p._cl_obj_grad
            if self.update_probe:
                p._cl_probe_grad, p._cl_probe_grad_last = p._cl_probe_grad_last, p._cl_probe_grad
            if self.update_background:
                p._cl_background_grad, p._cl_background_grad_last = p._cl_background_grad_last, p._cl_background_grad

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
            # 1) Compute the gradients.
            p = Grad(update_object=self.update_object, update_probe=self.update_probe,
                     update_background=self.update_background, floating_intensity=self.floating_intensity,
                     reg_fac_obj=self.reg_fac_obj, reg_fac_probe=self.reg_fac_probe, calc_llk=calc_llk) * p

            # 2) Search direction
            beta = np.float32(0)
            if ic == 0:
                # first cycle
                if self.update_object:
                    cl.enqueue_copy(cl_queue, src=p._cl_obj_grad.data, dest=p._cl_obj_dir.data)
                    if pu.profiling:
                        if "ml_copy_grad_dir0" not in pu.cl_event_profiling:
                            pu.cl_event_profiling["ml_copy_grad_dir0"] = []
                        ev = CLEvent(pu.ev[-1], 0, p._cl_obj.nbytes * 2)
                        pu.cl_event_profiling["ml_copy_grad_dir0"].append(ev)
                if self.update_probe:
                    cl.enqueue_copy(cl_queue, src=p._cl_probe_grad.data, dest=p._cl_probe_dir.data)
                    if pu.profiling:
                        if "ml_copy_grad_dir0" not in pu.cl_event_profiling:
                            pu.cl_event_profiling["ml_copy_grad_dir0"] = []
                        ev = CLEvent(pu.ev[-1], 0, p._cl_probe.nbytes * 2)
                        pu.cl_event_profiling["ml_copy_grad_dir0"].append(ev)
                if self.update_background:
                    cl.enqueue_copy(cl_queue, src=p._cl_background_grad.data, dest=p._cl_background_dir.data)
                    if pu.profiling:
                        if "ml_copy_grad_dir0" not in pu.cl_event_profiling:
                            pu.cl_event_profiling["ml_copy_grad_dir0"] = []
                        ev = CLEvent(pu.ev[-1], 0, p._cl_background.nbytes * 2)
                        pu.cl_event_profiling["ml_copy_grad_dir0"].append(ev)
            else:
                beta_d, beta_n = 0, 0
                # Polak-Ribière CG coefficient
                cg_pr = pu._cl_cg_polak_ribiere_complex_red
                cg_prf = pu._cl_cg_polak_ribiere_complex_redf
                if self.update_object:
                    tmp = cg_pr(p._cl_obj_grad, p._cl_obj_grad_last, wait_for=pu.ev).get()
                    pu.ev = []
                    beta_n += tmp['x']
                    beta_d += tmp['y']
                if self.update_probe:
                    tmp = cg_pr(p._cl_probe_grad, p._cl_probe_grad_last, wait_for=pu.ev).get()
                    pu.ev = []
                    beta_n += tmp['x']
                    beta_d += tmp['y']
                if self.update_background:
                    tmp = cg_prf(p._cl_background_grad, p._cl_background_grad_last, wait_for=pu.ev).get()
                    pu.ev = []
                    beta_n += tmp['x']
                    beta_d += tmp['y']
                # print("Beta= %e / %e"%(beta_n, beta_d))
                # Reset direction if beta<0 => beta=0
                beta = np.float32(max(0, beta_n / max(1e-20, beta_d)))
                if np.isnan(beta_n + beta_d) or np.isinf(beta_n + beta_d):
                    warnings.warn("ML: Beta=NaN, resetting direction", stacklevel=2)
                    beta = np.float32(0)
                if self.update_object:
                    pu.ev = [pu.cl_linear_comb_fcfc(beta, p._cl_obj_dir, np.float32(1), p._cl_obj_grad,
                                                    wait_for=pu.ev)]
                    if pu.profiling:
                        if "linear_comb_fcfc_obj" not in pu.cl_event_profiling:
                            pu.cl_event_profiling["linear_comb_fcfc_obj"] = []
                        # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                        ev = CLEvent(pu.ev[-1], 0, 0)
                        pu.cl_event_profiling["linear_comb_fcfc_obj"].append(ev)
                if self.update_probe:
                    pu.ev = [pu.cl_linear_comb_fcfc(beta, p._cl_probe_dir, np.float32(1), p._cl_probe_grad,
                                                    wait_for=pu.ev)]
                    if pu.profiling:
                        if "linear_comb_fcfc_probe" not in pu.cl_event_profiling:
                            pu.cl_event_profiling["linear_comb_fcfc_probe"] = []
                        # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                        ev = CLEvent(pu.ev[-1], 0, 0)
                        pu.cl_event_profiling["linear_comb_fcfc_probe"].append(ev)
                if self.update_background:
                    pu.ev = [pu.cl_linear_comb_4f(beta, p._cl_background_dir, np.float32(1), p._cl_background_grad,
                                                  wait_for=pu.ev)]
                    if pu.profiling:
                        if "linear_comb_fcfc_background" not in pu.cl_event_profiling:
                            pu.cl_event_profiling["linear_comb_fcfc_background"] = []
                        # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                        ev = CLEvent(pu.ev[-1], 0, 0)
                        pu.cl_event_profiling["linear_comb_fcfc_background"].append(ev)

            # 3) Line minimization
            p._cl_cg_gamma_d, p._cl_cg_gamma_n = 0, 0
            if False:
                # We could use a 4th order LLK(gamma) approximation, but it does not seem to improve
                p._cl_cg_gamma4 = np.zeros(5, dtype=np.float32)

            p = LoopStack(_CGGamma(update_background=self.update_background)) * p

            if np.isnan(p._cl_cg_gamma_d + p._cl_cg_gamma_n) or np.isinf(p._cl_cg_gamma_d + p._cl_cg_gamma_n):
                warnings.warn("ML: CG_gamma = NaN. Resetting search direction")
                # TODO: Somehow the search direction is diverging. Why does this happen ?
                # As a workaround, reset the search direction
                if self.update_object:
                    cl.enqueue_copy(cl_queue, src=p._cl_obj_grad.data, dest=p._cl_obj_dir.data)
                if self.update_probe:
                    cl.enqueue_copy(cl_queue, src=p._cl_probe_grad.data, dest=p._cl_probe_dir.data)
                if self.update_background:
                    cl.enqueue_copy(cl_queue, src=p._cl_background_grad.data, dest=p._cl_background_dir.data)
                p._cl_cg_gamma_d, p._cl_cg_gamma_n = 0, 0
                p = LoopStack(_CGGamma(update_background=self.update_background)) * p

            if self.update_object and self.reg_fac_obj != 0 and self.reg_fac_obj is not None:
                reg_fac_obj = np.float32(p.reg_fac_scale_obj * self.reg_fac_obj)
                nyo = np.int32(p._obj.shape[-2])
                nxo = np.int32(p._obj.shape[-1])
                tmp = self.processing_unit._cl_cg_gamma_reg_red(p._cl_obj, p._cl_obj_dir, nxo, nyo,
                                                                wait_for=pu.ev).get()
                p._cl_cg_gamma_d += tmp['y'] * reg_fac_obj
                p._cl_cg_gamma_n += tmp['x'] * reg_fac_obj

            if self.update_probe and self.reg_fac_probe != 0 and self.reg_fac_probe is not None:
                reg_fac_probe = np.float32(p.reg_fac_scale_probe * self.reg_fac_probe)
                ny = np.int32(p._probe.shape[-2])
                nx = np.int32(p._probe.shape[-1])
                tmp = self.processing_unit._cl_cg_gamma_reg_red(p._cl_probe, p._cl_probe_dir, nx, ny,
                                                                wait_for=pu.ev).get()
                p._cl_cg_gamma_d += tmp['y'] * reg_fac_probe
                p._cl_cg_gamma_n += tmp['x'] * reg_fac_probe

            gamma = np.float32(p._cl_cg_gamma_n / p._cl_cg_gamma_d)

            if False:
                # It seems the 2nd order gamma approximation is good enough.
                gr = np.roots(p._cl_cg_gamma4)
                p.print("CG Gamma4", p._cl_cg_gamma4, "\n", gr, np.polyval(p._cl_cg_gamma4, gr))
                p.print("CG Gamma2=", gamma, "=", p._cl_cg_gamma_n, "/", p._cl_cg_gamma_d)

            # 4) Object and/or probe and/or background update
            if self.update_object:
                pu.ev = [pu.cl_linear_comb_fcfc(np.float32(1), p._cl_obj, gamma, p._cl_obj_dir, wait_for=pu.ev)]
                if pu.profiling:
                    if "linear_comb_fcfc_obj" not in pu.cl_event_profiling:
                        pu.cl_event_profiling["linear_comb_fcfc_obj"] = []
                    # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                    ev = CLEvent(pu.ev[-1], 0, 0)
                    pu.cl_event_profiling["linear_comb_fcfc_obj"].append(ev)

            if self.update_probe:
                pu.ev = [pu.cl_linear_comb_fcfc(np.float32(1), p._cl_probe, gamma, p._cl_probe_dir, wait_for=pu.ev)]
                if pu.profiling:
                    if "linear_comb_fcfc_probe" not in pu.cl_event_profiling:
                        pu.cl_event_profiling["linear_comb_fcfc_probe"] = []
                    # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                    ev = CLEvent(pu.ev[-1], 0, 0)
                    pu.cl_event_profiling["linear_comb_fcfc_probe"].append(ev)

            if self.update_background:
                pu.ev = [pu.cl_linear_comb_4f_pos(np.float32(1), p._cl_background, gamma, p._cl_background_dir,
                                                  wait_for=pu.ev)]
                if pu.profiling:
                    if "linear_comb_fcfc_background" not in pu.cl_event_profiling:
                        pu.cl_event_profiling["linear_comb_fcfc_background"] = []
                    # nb_obj, nb_probe, nz, ny, nx = p._cl_psi.shape
                    ev = CLEvent(pu.ev[-1], 0, 0)
                    pu.cl_event_profiling["linear_comb_fcfc_background"].append(ev)

            #  Update the floating intensity scale factors ?
            if self.floating_intensity:
                nb_frame = np.int32(len(p.data.iobs))
                scale_sum = cla.sum(p._cl_scale[0: nb_frame])
                pu.ev = [pu.cl_floating_scale_norm(p._cl_scale[0: nb_frame], scale_sum, nb_frame, wait_for=pu.ev)]

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
                    s = algo_string('ML', p, self.update_object, self.update_probe, self.update_background)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.get_llk('poisson'))
                    p = ShowObjProbe(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1

        # Clean up
        del p._cl_PO, p._cl_PdO, p._cl_dPO, p._cl_dPdO, p._cl_obj_dir, p._cl_probe_dir
        if self.update_object:
            del p._cl_obj_grad, p._cl_obj_grad_last
        if self.update_probe:
            del p._cl_probe_grad, p._cl_probe_grad_last
        del p._cl_background_grad, p._cl_background_dir
        if self.update_background:
            del p._cl_background_grad_last

        gc.collect()
        if self.zero_phase_ramp:
            p = ZeroPhaseRamp(obj=True) * p
        return p


class ScaleObjProbe(CLOperatorPtycho):
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
            nxystack = np.int32(nxy * self.processing_unit.cl_stack_size)
            nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
            for i in range(p._cl_stack_nb):
                p = ObjProbe2Psi() * SelectStack(i) * p
                if p.data.near_field:
                    p = PropagateNearField(forward=True) * p
                else:
                    p = FT(scale=False) * p
                nb_psi = p._cl_obs_v[i].npsi
                r = pu.cl_scale_intensity(p._cl_obs_v[i].cl_obs[:nb_psi], p._cl_psi, p._cl_background,
                                          nxy, nxystack, nb_mode, wait_for=pu.ev).get()

                snum += r['x']
                sden += r['y']
            s = np.sqrt(snum / sden)
        else:
            s = 1
        # TODO: take into account only the scanned part of the object for obj/probe relative scaling
        os = self.processing_unit.cl_norm_complex_n(p._cl_obj, np.int32(1)).get()
        ps = self.processing_unit.cl_norm_complex_n(p._cl_probe, np.int32(1)).get()
        pu.ev = [pu.cl_scale(p._cl_probe, np.float32(np.sqrt(os / ps * s)))]
        pu.ev = [pu.cl_scale(p._cl_obj, np.float32(np.sqrt(ps / os * s)), wait_for=pu.ev)]
        if self.verbose:
            p.print("ScaleObjProbe:", ps, os, s, np.sqrt(os / ps * s), np.sqrt(ps / os * s))
        if False:
            # Check the scale factor
            snum, sden = 0, 0
            for i in range(p._cl_stack_nb):
                p = ObjProbe2Psi() * SelectStack(i) * p
                if p.data.near_field:
                    p = PropagateNearField(forward=True) * p
                else:
                    p = FT(scale=False) * p
                r = pu.cl_scale_intensity(p._cl_psi, p._cl_obs_v[i].cl_obs, p._cl_background,
                                          nxy, nxystack, nb_mode, wait_for=pu.ev).get()
                snum += r['x']
                sden += r['y']
            s = snum / sden
            p.print("ScaleObjProbe: now s=", s)
        return p


class CenterObjProbe(CLOperatorPtycho):
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
        cm, ev = pu.cl_center_mass_complex(p._cl_probe, nx, ny, nz, self.power, wait_for=pu.ev, return_event=True)
        pu.ev = [ev]
        if pu.profiling:
            if "center_mass_complex_red" not in pu.cl_event_profiling:
                pu.cl_event_profiling["center_mass_complex_red"] = []
            ev = CLEvent(pu.ev[-1], 0, 0)
            pu.cl_event_profiling["center_mass_complex_red"].append(ev)
        cm = cm.get()
        dx, dy, dz = cm['x'] / cm['w'] - nx / 2, cm['y'] / cm['w'] - ny / 2, cm['z'] / cm['w'] - nz / 2
        if self.verbose:
            p.print("CenterObjProbe(): center of mass deviation: dx=%6.2f   dy=%6.2f" % (dx, dy))
        if np.sqrt(dx ** 2 + dy ** 2) > self.max_shift:
            dx = np.int32(round(-dx))
            dy = np.int32(round(-dy))
            cl_obj = cla.empty_like(p._cl_obj)
            cl_probe = cla.empty_like(p._cl_probe)
            pu.ev = [pu.cl_circular_shift(p._cl_probe, cl_probe, dx, dy, np.int32(0), nx, ny, nz, wait_for=pu.ev)]
            if pu.profiling:
                if "circular_shift_obj" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["circular_shift_obj"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["circular_shift_obj"].append(ev)
            p._cl_probe = cl_probe
            pu.ev = [pu.cl_circular_shift(p._cl_obj, cl_obj, dx, dy, np.int32(0), nxo, nyo, nzo, wait_for=pu.ev)]
            if pu.profiling:
                if "circular_shift_probe" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["circular_shift_probe"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["circular_shift_probe"].append(ev)
            p._cl_obj = cl_obj
            # Also shift psi
            nzpsi = p._cl_psi.size // (nx * ny)
            cl_ps = cla.empty_like(p._cl_psi)
            pu.ev = [pu.cl_circular_shift(p._cl_psi, cl_ps, dx, dy, np.int32(0), nx, ny, nzpsi, wait_for=pu.ev)]
            if pu.profiling:
                if "circular_shift_psi" not in pu.cl_event_profiling:
                    pu.cl_event_profiling["circular_shift_psi"] = []
                ev = CLEvent(pu.ev[-1], 0, 0)
                pu.cl_event_profiling["circular_shift_psi"].append(ev)
            cl_ps, p._cl_psi = p._cl_psi, cl_ps
            if has_attr_not_none(p, "_cl_psi_v"):
                for k, cl_psi in p._cl_psi_v.items():
                    pu.ev = [pu.cl_circular_shift(cl_psi, cl_ps, dx, dy, np.int32(0), nx, ny, nzpsi, wait_for=pu.ev)]
                    if pu.profiling:
                        if "circular_shift_psi" not in pu.cl_event_profiling:
                            pu.cl_event_profiling["circular_shift_psi"] = []
                        ev = CLEvent(pu.ev[-1], 0, 0)
                        pu.cl_event_profiling["circular_shift_psi"].append(ev)
                    cl_ps, p._cl_psi_v[k] = p._cl_psi_v[k], cl_ps

        return p


class SumIntensity1(CLOperatorPtycho):
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
        i = p._cl_stack_i
        nz = p._cl_obs_v[i].npsi

        if self.icalc is not None:
            pu.ev = [pu.cl_sum_icalc(self.icalc, p._cl_psi, nxy, np.int32(nz * nb_mode), wait_for=pu.ev)]
        if self.iobs is not None:
            pu.ev = [pu.cl_sum_iobs(self.iobs, p._cl_obs_v[i].cl_obs[:nz], p._cl_psi, nxy, nz, nb_mode, wait_for=pu.ev)]

        return p


class ApplyPhaseRamp(CLOperatorPtycho):
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
        :param obj: if True, apply the correction to the object
        :param probe: if True, apply the correction to the probe
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
            pu.ev = [pu.cl_corr_phase_ramp(p._cl_probe, -self.dx, -self.dy, nx, ny, wait_for=pu.ev)]
        if self.obj:
            pu.ev = [pu.cl_corr_phase_ramp(p._cl_obj, np.float32(self.dx * nxo / nx),
                                           np.float32(self.dy * nyo / ny), nxo, nyo, wait_for=pu.ev)]
        return p


class ZeroPhaseRamp(CLOperatorPtycho):
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

        cl_probe = p._cl_probe.copy()

        pu.fft(cl_probe, cl_probe, ndim=2)
        cm, ev = pu.cl_center_mass_fftshift_complex(cl_probe, nx, ny, nz, np.int32(2),
                                                    wait_for=pu.ev, return_event=True)
        pu.ev = [ev]
        cm = cm.get()
        dx = np.float32(cm['x'] / cm['w'] - nx / 2)
        dy = np.float32(cm['y'] / cm['w'] - ny / 2)
        # print("ZeroPhaseRamp(): (dx, dy)[probe] = (%6.3f, %6.3f)[obs]" % (dx, dy))

        p = ApplyPhaseRamp(dx, dy, obj=True, probe=True) * p

        if self.obj:
            # Compute the shift of the calculated frame to determine the object ramp
            icalc_sum = cla.zeros(pu.cl_queue, (ny, nx), dtype=np.float32)
            iobs_sum = None  # cla.zeros(pu.cl_queue, (ny, nx), dtype=np.float32)
            p = LoopStack(SumIntensity1(icalc=icalc_sum, iobs=iobs_sum)) * p

            # Compute shift of center of mass
            if False:
                cm, ev = pu.cl_center_mass_fftshift(iobs_sum, nx, ny, np.int32(1), np.int32(1),
                                                    wait_for=pu.ev, return_event=True)
                pu.ev = [ev]
                cm = cm.get()
                dx = np.float32(cm['x'] / cm['w'] - nx / 2)
                dy = np.float32(cm['y'] / cm['w'] - ny / 2)
                # print("ZeroPhaseRamp(): (dx, dy)[obj] = (%6.3f, %6.3f)[obs]" % (dx, dy))
            else:
                cm, ev = pu.cl_center_mass_fftshift(icalc_sum, nx, ny, np.int32(1), np.int32(1),
                                                    wait_for=pu.ev, return_event=True)
                pu.ev = [ev]
                cm = cm.get()
                dx = np.float32(cm['x'] / cm['w'] - nx / 2)
                dy = np.float32(cm['y'] / cm['w'] - ny / 2)
                # print("ZeroPhaseRamp(): (dx, dy)[obj] = (%6.3f, %6.3f)[calc]" % (dx, dy))
            p.data.phase_ramp_dx = dx
            p.data.phase_ramp_dy = dy

        return p


class CalcIllumination(CLOperatorPtycho):
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
        interp = np.int8(p._interpolation)
        cl_obj_illum = cla.zeros(pu.cl_queue, (nyo, nxo), dtype=np.float32)
        padding = np.int32(p.data.padding)
        for i in range(p._cl_stack_nb):
            i0 = p._cl_obs_v[i].i
            npsi = p._cl_obs_v[i].npsi
            pu.cl_calc_illum(p._cl_probe[0], cl_obj_illum, p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y,
                             npsi, pu.cl_stack_size, nx, ny, nxo, nyo, nb_probe,
                             p._cl_scale[i0:i0 + npsi], interp, padding)
        p._obj_illumination = cl_obj_illum.get()
        cl_obj_illum.data.release()  # Should not be necessary, will be gc
        return p

    def timestamp_increment(self, p):
        # Is that really the correct behaviour ?
        # Object and probe etc are not modified, but ptycho object is..
        pass


class SelectStack(CLOperatorPtycho):
    """
    Operator to select a stack of observed frames to work on. Note that once this operation has been applied,
    the new Psi value may be undefined (empty array), if no previous array existed.
    """

    def __init__(self, stack_i, keep_psi=False):
        """
        Select a new stack of frames, swapping data to store the last calculated psi array in the
        corresponding, ptycho object's _cl_psi_v[i] dictionary.

        What happens is:
        * keep_psi=False: only the stack index in p is changed (p._cl_stack_i=stack_i)

        * keep_psi=True: the previous psi is stored in p._cl_psi_v[p._cl_stack_i], the new psi is swapped
                                   with p._cl_psi_v[stack_i] if it exists, otherwise initialized as an empty array.

        :param stack_i: the stack index.
        :param keep_psi: if True, when switching between stacks, store and restore psi in p._cl_psi_v.
        """
        super(SelectStack, self).__init__()
        self.stack_i = stack_i
        self.keep_psi = keep_psi

    def op(self, p: Ptycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        if self.stack_i == p._cl_stack_i:
            if self.keep_psi and self.stack_i in p._cl_psi_v:
                # This can happen if we use LoopStack(keep_psi=False) between LoopStack(keep_psi=True)
                p._cl_psi = p._cl_psi_v[self.stack_i].pop()
            return p

        if self.keep_psi:
            # Store previous Psi. This can be dangerous when starting a loop as the state of Psi may be incorrect,
            # e.g. in detector or sample space when the desired operations work in a different space...
            p._cl_psi_v[p._cl_stack_i] = p._cl_psi
            if self.stack_i in p._cl_psi_v:
                p._cl_psi = p._cl_psi_v.pop(self.stack_i)
            else:
                p._cl_psi = cla.empty_like(p._cl_psi_v[p._cl_stack_i])

        p._cl_stack_i = self.stack_i
        return p


class PurgeStacks(CLOperatorPtycho):
    """
    Operator to delete stored psi stacks in a Ptycho object's _cl_psi_v.

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
        for a in p._cl_psi_v:
            a.data.release()
        p._cl_psi_v = {}
        return p


class LoopStack(CLOperatorPtycho):
    """
    Operator to apply a given operator sequentially to the complete stack of frames of a ptycho object.

    Make sure that the current selected stack is in a correct state (i.e. in sample or detector space,...) before
    starting such a loop with keep_psi=True.
    """

    def __init__(self, op, keep_psi=False, copy=False):
        """

        :param op: the operator to apply, which can be a multiplication of operators
        :param keep_psi: if True, when switching between stacks, store psi in p._cl_psi_v.
        :param copy: make a copy of the original p._cl_psi swapped in as p._cl_psi_copy, and
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
        pu = self.processing_unit
        if p._cl_stack_nb == 1:
            if self.copy:
                p._cl_psi_copy = cla.empty_like(p._cl_psi)
                cl.enqueue_copy(self.processing_unit.cl_queue, src=p._cl_psi.data, dest=p._cl_psi_copy.data)
                p = self.stack_op * p
                if has_attr_not_none(p, '_cl_psi_copy'):
                    # Finished using psi copy, delete it (actual deletion will occur once GPU has finished processing)
                    del p._cl_psi_copy
                return p
            else:
                return self.stack_op * p
        else:
            if self.copy:
                p._cl_psi_copy = cla.empty_like(p._cl_psi)
            for i in range(p._cl_stack_nb):
                p = SelectStack(i, keep_psi=self.keep_psi) * p
                if self.copy:
                    # The planned operations rely on keeping a copy of the previous Psi state...
                    cl.enqueue_copy(self.processing_unit.cl_queue, src=p._cl_psi.data, dest=p._cl_psi_copy.data)

                    if pu.profiling:
                        if "loopstack_copy" not in pu.cl_event_profiling:
                            pu.cl_event_profiling["loopstack_copy"] = []
                        ev = CLEvent(pu.ev[-1], 0, p._cl_psi.nbytes * 2)
                        pu.cl_event_profiling["loopstack_copy"].append(ev)

                p = self.stack_op * p
            if self.copy:
                if has_attr_not_none(p, '_cl_psi_copy'):
                    # Finished using psi copy, delete it (actual deletion will occur once GPU has finished processing)
                    del p._cl_psi_copy

            if self.keep_psi:
                # Copy last stack to p._cl_psi_v
                p._cl_psi_v[p._cl_stack_i] = cla.empty_like(p._cl_psi)
                pu.ev = [cl.enqueue_copy(self.processing_unit.cl_queue, src=p._cl_psi.data,
                                         dest=p._cl_psi_v[p._cl_stack_i].data, wait_for=pu.ev)]
                if pu.profiling:
                    if "loopstack_keep_psi_copy" not in pu.cl_event_profiling:
                        pu.cl_event_profiling["loopstack_keep_psi_copy"] = []
                    ev = CLEvent(pu.ev[-1], 0, p._cl_psi.nbytes * 2)
                    pu.cl_event_profiling["loopstack_keep_psi_copy"].append(ev)
        return p


class BackgroundFilter(CLOperatorPtycho):
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
        clbr = cla.empty(pu.cl_queue, (nxy, nxy // 2 + 1), dtype=np.complex64)

        s = cla.sum(p._cl_background)

        sigmaf = np.float32(2 * np.pi ** 2 * self.sigma / nxy ** 2)

        pu.fft(p._cl_background, clbr)

        pu.ev = [pu.cl_gauss_ftconv(clbr, sigmaf, nxy, wait_for=pu.ev)]

        pu.ifft(clbr, p._cl_background)

        pu.ev = [pu.cl_scalef_mem(p._cl_background, s / cla.sum(p._cl_background), wait_for=pu.ev)]
        return p
