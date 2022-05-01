# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


import warnings
import types
import timeit
import gc
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from pyopencl.elementwise import ElementwiseKernel as CL_ElK
from pyopencl.reduction import ReductionKernel as CL_RedK

from ...processing_unit import default_processing_unit as main_default_processing_unit
from ...processing_unit.cl_processing_unit import CLProcessingUnit, CLEvent
from ...processing_unit.kernel_source import get_kernel_source as getks
from ..operator import has_attr_not_none, OperatorSum, OperatorPower, OperatorException
from .bragg2d import Bragg2DPtycho, OperatorBragg2DPtycho, rotate
from . import cpu_operator as cpuop
from pynx.utils.matplotlib import pyplot as plt


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


patch_method(Bragg2DPtycho)


################################################################################################


class CLProcessingUnitPtycho(CLProcessingUnit):
    """
    Processing unit in OpenCL space, for 2D Ptycho operations.

    Handles initializing the context and kernels.
    """

    def __init__(self):
        super(CLProcessingUnitPtycho, self).__init__()
        # Size of the stack size used in OpenCL - can be any integer
        # TODO: test optimal value for 3D data
        self.cl_stack_size = np.int32(16)

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

        self.cl_sum = CL_ElK(self.cl_ctx, name='cl_sum',
                             operation="dest[i] += src[i]",
                             options=self.cl_options, arguments="__global float2 *src, __global float2 *dest")

        self.cl_scale_complex = CL_ElK(self.cl_ctx, name='cl_scale_complex',
                                       operation="d[i] = (float2)(d[i].x * s.x - d[i].y * s.y, d[i].x * s.y + d[i].y * s.x)",
                                       options=self.cl_options, arguments="__global float2 *d, const float2 s")

        # Linear combination with 2 complex arrays and 2 float coefficients
        self.cl_linear_comb_fcfc = CL_ElK(self.cl_ctx, name='cl_linear_comb_fcfc',
                                          operation="dest[i] = (float2)(a * dest[i].x + b * src[i].x, a * dest[i].y + b * src[i].y)",
                                          options=self.cl_options,
                                          arguments="const float a, __global float2 *dest, const float b, __global float2 *src")

        # Linear combination with 2 float arrays and 2 float coefficients
        self.cl_linear_comb_4f = CL_ElK(self.cl_ctx, name='cl_linear_comb_4f',
                                        operation="dest[i] = a * dest[i] + b * src[i]",
                                        options=self.cl_options,
                                        arguments="const float a, __global float *dest, const float b, __global float *src")

        self.cl_projection_amplitude = CL_ElK(self.cl_ctx, name='cl_projection_amplitude',
                                              operation="ProjectionAmplitude(i, iobs, dcalc, background, nbmode, nxy,"
                                                        "nxystack, npsi)",
                                              preamble=getks('ptycho/opencl/projection_amplitude_elw.cl'),
                                              options=self.cl_options,
                                              arguments="__global float *iobs, __global float2 *dcalc,"
                                                        "__global float *background, const int nbmode, const int nxy,"
                                                        "const int nxystack, const int npsi")
        self.cl_projection_amplitude_diff = CL_ElK(self.cl_ctx, name='cl_projection_amplitude_diff',
                                                   operation="ProjectionAmplitudeDiff(i, iobs, dcalc, background, "
                                                             "nbmode, nxy, nxystack, npsi)",
                                                   preamble=getks('ptycho/opencl/projection_amplitude_elw.cl'),
                                                   options=self.cl_options,
                                                   arguments="__global float *iobs, __global float2 *dcalc,"
                                                             "__global float *background, const int nbmode,"
                                                             "const int nxy, const int nxystack, const int npsi")

        self.cl_calc2obs = CL_ElK(self.cl_ctx, name='cl_calc2obs',
                                  operation="Calc2Obs(i, iobs, dcalc, nbmode, nxystack)",
                                  preamble=getks('ptycho/opencl/calc2obs_elw.cl'),
                                  options=self.cl_options,
                                  arguments="__global float *iobs, __global float2 *dcalc, const int nbmode, const int nxystack")

        self.cl_object_probe_mult = CL_ElK(self.cl_ctx, name='cl_object_probe_mult',
                                           operation="Object3DProbe2DMult(i, psi, obj, probe, support, m, cx, cy, cixo,"
                                                     "ciyo, dsx, dsy, dsz, pxo, pyo, pzo, pxp, pyp, f, npsi,"
                                                     "stack_size, nx, ny, nxo, nyo, nzo, nxp, nyp, nbobj, nbprobe)",
                                           preamble=getks('opencl/complex.cl') +
                                                    getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                    getks('ptycho/bragg2d/opencl/obj_probe_mult_elw.cl'),
                                           options=self.cl_options,
                                           arguments="__global float2* psi, __global float2 *obj,"
                                                     "__global float2* probe,  __global char* support,"
                                                     "__global float* m, __global float* cx, __global float* cy,"
                                                     "__global int* cixo, __global int* ciyo,"
                                                     "__global float* dsx, __global float* dsy, __global float* dsz,"
                                                     "const float pxo, const float pyo, const float pzo,"
                                                     "const float pxp, const float pyp, const float f, const int npsi,"
                                                     "const int stack_size, const int nx, const int ny,"
                                                     "const int nxo, const int nyo, const int nzo, const int nxp,"
                                                     "const int nyp, const int nbobj, const int nbprobe")

        self.cl_object_probe_mult_debug = CL_ElK(self.cl_ctx, name='cl_object_probe_mult',
                                                 operation="Object3DProbe2DMultDebug(i, psi, obj, probe, support, m,"
                                                           "psi3d, obj3d, probe3d, cx, cy, cixo, ciyo, dsx, dsy, dsz,"
                                                           "pxo, pyo, pzo, pxp, pyp, f, npsi, stack_size,"
                                                           "nx, ny, nxo, nyo, nzo, nxp, nyp, nbobj, nbprobe)",
                                                 preamble=getks('opencl/complex.cl') +
                                                          getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                          getks('ptycho/bragg2d/opencl/obj_probe_mult_elw.cl'),
                                                 options=self.cl_options,
                                                 arguments="__global float2* psi, __global float2 *obj,"
                                                           "__global float2* probe, __global char* support,"
                                                           "__global float* m, __global float2* psi3d,"
                                                           "__global float2* obj3d, __global float2* probe3d,"
                                                           "__global float* cx, __global float* cy,"
                                                           "__global int* cixo, __global int* ciyo,"
                                                           "__global float* dsx, __global float* dsy, __global float* dsz,"
                                                           "const float pxo, const float pyo, const float pzo,"
                                                           "const float pxp, const float pyp, const float f, const int npsi,"
                                                           "const int stack_size, const int nx, const int ny,"
                                                           "const int nxo, const int nyo, const int nzo, const int nxp,"
                                                           "const int nyp, const int nbobj, const int nbprobe")

        self.cl_2object_probe_psi_dm1 = CL_ElK(self.cl_ctx, name='cl_2object_probe_psi_dm1',
                                               operation="ObjectProbePsiDM1(i, psi, obj, probe, support, m, cx, cy,"
                                                         "cixo, ciyo, dsx, dsy, dsz, pxo, pyo, pzo, pxp, pyp, f, npsi,"
                                                         "stack_size, nx, ny, nxo, nyo, nzo, nxp, nyp, nbobj, nbprobe)",
                                               preamble=getks('opencl/complex.cl') +
                                                        getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                        getks('ptycho/bragg2d/opencl/obj_probe_dm_elw.cl'),
                                               options=self.cl_options,
                                               arguments="__global float2* psi, __global float2 *obj,"
                                                         "__global float2* probe, __global char* support,"
                                                         "__global float* m, __global float* cx, __global float* cy,"
                                                         "__global int* cixo, __global int* ciyo,"
                                                         "__global float* dsx, __global float* dsy,"
                                                         "__global float* dsz, const float pxo, const float pyo,"
                                                         "const float pzo, const float pxp, const float pyp,"
                                                         "const float f, const int npsi, const int stack_size,"
                                                         "const int nx, const int ny, const int nxo, const int nyo,"
                                                         "const int nzo, const int nxp, const int nyp, const int nbobj,"
                                                         "const int nbprobe")

        self.cl_2object_probe_psi_dm2 = CL_ElK(self.cl_ctx, name='cl_2object_probe_psi_dm2',
                                               operation="ObjectProbePsiDM2(i, psi, psi_fourier, obj, probe, support,"
                                                         "m, cx, cy, cixo, ciyo, dsx, dsy, dsz, pxo, pyo, pzo, pxp,"
                                                         "pyp,f, npsi, stack_size, nx, ny, nxo, nyo, nzo, nxp, nyp,"
                                                         "nbobj, nbprobe)",
                                               preamble=getks('opencl/complex.cl') +
                                                        getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                        getks('ptycho/bragg2d/opencl/obj_probe_dm_elw.cl'),
                                               options=self.cl_options,
                                               arguments="__global float2* psi, __global float2* psi_fourier,"
                                                         "__global float2 *obj, __global float2* probe,"
                                                         "__global char* support, __global float* m,"
                                                         "__global float* cx, __global float* cy,"
                                                         "__global int* cixo, __global int* ciyo,"
                                                         "__global float* dsx, __global float* dsy,"
                                                         "__global float* dsz, const float pxo, const float pyo,"
                                                         "const float pzo, const float pxp, const float pyp,"
                                                         "const float f, const int npsi, const int stack_size,"
                                                         "const int nx, const int ny, const int nxo, const int nyo,"
                                                         "const int nzo, const int nxp, const int nyp, const int nbobj,"
                                                         "const int nbprobe")

        # This object gradient is for object update directly from Psi during AP and DM
        self.cl_psi2obj_grad = CL_ElK(self.cl_ctx, name='psi2obj_grad',
                                      operation="Psi2ObjGrad(i, psi, obj, probe, grad, support, m, cx, cy, cixo, ciyo,"
                                                "dsx, dsy, dsz, pxo, pyo, pzo, pxp, pyp, f, stack_size, nx, ny,"
                                                "nxo, nyo, nzo, nxp, nyp, nbobj, nbprobe)",
                                      preamble=getks('opencl/complex.cl') +
                                               getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                               getks('ptycho/bragg2d/opencl/psi_to_obj_probe.cl'),
                                      options=self.cl_options,
                                      arguments="__global float2* psi, __global float2 *obj, __global float2* probe,"
                                                "__global float2 *grad, __global char* support, __global float* m,"
                                                "float cx, float cy, int cixo, int ciyo, float dsx, float dsy,"
                                                "float dsz, const float pxo, const float pyo,"
                                                "const float pzo, const float pxp, const float pyp, const float f,"
                                                "const int stack_size, const int nx, const int ny,"
                                                "const int nxo, const int nyo, const int nzo, const int nxp,"
                                                "const int nyp, const int nbobj, const int nbprobe")

        self.cl_psi2probe_grad = CL_ElK(self.cl_ctx, name='psi2probe_grad',
                                        operation="Psi2ProbeGrad(i, psi, obj, probe, grad, support, m, cx, cy, cixo,"
                                                  "ciyo, dsx, dsy, dsz, pxo, pyo, pzo, pxp, pyp, f, stack_size, nx, ny,"
                                                  "nxo, nyo, nzo, nxp, nyp, nbobj, nbprobe)",
                                        preamble=getks('opencl/complex.cl') +
                                                 getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                 getks('ptycho/bragg2d/opencl/psi_to_obj_probe.cl'),
                                        options=self.cl_options,
                                        arguments="__global float2* psi, __global float2 *obj, __global float2* probe,"
                                                  "__global float2 *grad, __global char* support, __global float* m,"
                                                  "float cx, float cy, int cixo, int ciyo, float dsx, float dsy,"
                                                  "float dsz, const float pxo, const float pyo,"
                                                  "const float pzo, const float pxp, const float pyp, const float f,"
                                                  "const int stack_size, const int nx, const int ny,"
                                                  "const int nxo, const int nyo, const int nzo, const int nxp,"
                                                  "const int nyp, const int nbobj, const int nbprobe")

        self.cl_grad_poisson_fourier = CL_ElK(self.cl_ctx, name='cl_grad_poisson_fourier',
                                              operation="GradPoissonFourier(i, iobs, psi, background, nbmode, nx, ny,"
                                                        "nxystack)",
                                              preamble=getks('opencl/complex.cl') +
                                                       getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                       getks('ptycho/bragg2d/opencl/grad_elw.cl'),
                                              options=self.cl_options,
                                              arguments="__global float *iobs, __global float2 *psi,"
                                                        "__global float *background, const int nbmode, const int nx,"
                                                        "const int ny, const int nxystack")

        # Object gradient during maximum likelihood
        self.cl_ml_psi2obj_grad = CL_ElK(self.cl_ctx, name='ml_psi2obj_grad',
                                         operation="GradObj(i, psi, obj_grad, probe, support, m, cx, cy, cixo, ciyo,"
                                                   "dsx, dsy, dsz, pxo, pyo, pzo, pxp, pyp, f, stack_size, nx, ny,"
                                                   "nxo, nyo, nzo, nxp, nyp, nbobj, nbprobe)",
                                         preamble=getks('opencl/complex.cl') +
                                                  getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                  getks('ptycho/bragg2d/opencl/grad_elw.cl'),
                                         options=self.cl_options,
                                         arguments="__global float2* psi, __global float2 *obj_grad,"
                                                   "__global float2* probe, __global char* support, __global float* m,"
                                                   "float cx, float cy, int cixo, int ciyo, float dsx, float dsy,"
                                                   "float dsz, const float pxo, const float pyo,"
                                                   "const float pzo, const float pxp, const float pyp, const float f,"
                                                   "const int stack_size, const int nx, const int ny,"
                                                   "const int nxo, const int nyo, const int nzo, const int nxp,"
                                                   "const int nyp, const int nbobj, const int nbprobe")

        # Probe gradient during maximum likelihood
        self.cl_ml_psi2probe_grad = CL_ElK(self.cl_ctx, name='ml_psi2obj_grad',
                                           operation="GradProbe(i, psi, probe_grad, obj, support, m, cx, cy, cixo,"
                                                     "ciyo, dsx, dsy, dsz, pxo, pyo, pzo, pxp, pyp, f, stack_size, nx,"
                                                     "ny, nxo, nyo, nzo, nxp, nyp, nbobj, nbprobe)",
                                           preamble=getks('opencl/complex.cl') +
                                                    getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                    getks('ptycho/bragg2d/opencl/grad_elw.cl'),
                                           options=self.cl_options,
                                           arguments="__global float2* psi, __global float2 *probe_grad,"
                                                     "__global float2* obj, __global char* support,"
                                                     "__global float* m, float cx, float cy, int cixo, int ciyo,"
                                                     "float dsx, float dsy, float dsz, const float pxo,"
                                                     "const float pyo, const float pzo, const float pxp,"
                                                     "const float pyp, const float f, const int stack_size,"
                                                     "const int nx, const int ny, const int nxo, const int nyo,"
                                                     "const int nzo, const int nxp, const int nyp, const int nbobj,"
                                                     "const int nbprobe")

        self.cl_gauss_convolve_z = CL_ElK(self.cl_ctx, name='gauss_convolve_z',
                                          operation="GaussConvolveZ(i, grad, sigma, nxyo, nzo, nbobj)",
                                          preamble=getks('opencl/complex.cl') +
                                                   getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                   getks('ptycho/bragg2d/opencl/grad_elw.cl'),
                                          options=self.cl_options,
                                          arguments="__global float2 *grad, const float sigma, const int nxyo,"
                                                    "const int nzo, const int nbobj")

        # Object update using replication of delta(psi), normalisation from probe intensity sum along z
        self.cl_psi_to_obj_diff_repz = CL_ElK(self.cl_ctx, name='psi_to_obj_rep_norm_n',
                                              operation="Psi2ObjDiffRepZ(i, dpsi, obj_diff, probe, support,"
                                                        "obj_norm, m, cx, cy, cixo, ciyo, dsx, dsy, dsz, pxo, pyo,"
                                                        "pzo, pxp, pyp, f, stack_size, nx, ny, nxo, nyo, nzo, nxp,"
                                                        "nyp, nbobj, nbprobe, probe_max_norm2[0])",
                                              preamble=getks('opencl/complex.cl') +
                                                       getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                       getks('ptycho/bragg2d/opencl/psi_to_obj_probe.cl'),
                                              options=self.cl_options,
                                              arguments="__global float2* dpsi, __global float2 *obj_diff,"
                                                        "__global float2* probe, __global char* support,"
                                                        "__global float *obj_norm,"
                                                        "__global float* m, float cx, float cy, int cixo, int ciyo,"
                                                        "float dsx, float dsy, float dsz, const float pxo,"
                                                        "const float pyo, const float pzo, const float pxp,"
                                                        "const float pyp, const float f, const int stack_size,"
                                                        "const int nx, const int ny, const int nxo, const int nyo,"
                                                        "const int nzo, const int nxp, const int nyp, const int nbobj,"
                                                        "const int nbprobe, const float* probe_max_norm2")

        # Object update using replication of delta(psi), normalisation from probe intensity sum of all illuminations
        self.cl_psi_to_obj_diff_rep1 = CL_ElK(self.cl_ctx, name='psi_to_obj_rep_norm_n',
                                              operation="Psi2ObjDiffRep1(i, dpsi, obj_diff, probe, support,"
                                                        "obj_norm, m, cx, cy, cixo, ciyo, dsx, dsy, dsz, pxo, pyo,"
                                                        "pzo, pxp, pyp, f, stack_size, nx, ny, nxo, nyo, nzo, nxp,"
                                                        "nyp, nbobj, nbprobe, probe_max_norm2[0])",
                                              preamble=getks('opencl/complex.cl') +
                                                       getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                       getks('ptycho/bragg2d/opencl/psi_to_obj_probe.cl'),
                                              options=self.cl_options,
                                              arguments="__global float2* dpsi, __global float2 *obj_diff,"
                                                        "__global float2* probe, __global char* support,"
                                                        "__global float *obj_norm,"
                                                        "__global float* m, float cx, float cy, int cixo, int ciyo,"
                                                        "float dsx, float dsy, float dsz, const float pxo,"
                                                        "const float pyo, const float pzo, const float pxp,"
                                                        "const float pyp, const float f, const int stack_size,"
                                                        "const int nx, const int ny, const int nxo, const int nyo,"
                                                        "const int nzo, const int nxp, const int nyp, const int nbobj,"
                                                        "const int nbprobe, const float* probe_max_norm2")

        # Object update using replication of delta(psi) for a single frame
        self.cl_psi_to_obj_increment1 = CL_ElK(self.cl_ctx, name='psi_to_obj_rep_norm_n',
                                               operation="Psi2ObjIncrement1(i, dpsi, obj, probe, support, m, cx, cy,"
                                                         "cixo, ciyo, dsx, dsy, dsz, pxo, pyo, pzo, pxp, pyp, f,"
                                                         "stack_size, nx, ny, nxo, nyo, nzo, nxp, nyp, nbobj, nbprobe,"
                                                         "probe_max_norm2[0], beta)",
                                               preamble=getks('opencl/complex.cl') +
                                                        getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                        getks('ptycho/bragg2d/opencl/psi_to_obj_probe.cl'),
                                               options=self.cl_options,
                                               arguments="__global float2* dpsi, __global float2 *obj,"
                                                         "__global float2* probe, __global char* support,"
                                                         "__global float* m, float cx, float cy, int cixo, int ciyo,"
                                                         "float dsx, float dsy, float dsz, const float pxo,"
                                                         "const float pyo, const float pzo, const float pxp,"
                                                         "const float pyp, const float f, const int stack_size,"
                                                         "const int nx, const int ny, const int nxo, const int nyo,"
                                                         "const int nzo, const int nxp, const int nyp, const int nbobj,"
                                                         "const int nbprobe, const float* probe_max_norm2,"
                                                         "const float beta")

        self.cl_obj_norm = CL_ElK(self.cl_ctx, name='obj_diff_norm',
                                  operation="ObjDiffNorm(i, obj_diff, obj_norm, obj, normmax, inertia, nxyzo, nbobj,"
                                            "beta)",
                                  preamble=getks('opencl/complex.cl') +
                                           getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                           getks('ptycho/bragg2d/opencl/psi_to_obj_probe.cl'),
                                  options=self.cl_options,
                                  arguments="__global float2* obj_diff, __global float* obj_norm, __global float2* obj,"
                                            "__global float* normmax, const float inertia, const int nxyzo,"
                                            "const int nbobj, const float beta")

        # self.cl_psi_to_probe_grad = CL_ElK(self.cl_ctx, name='psi_to_probe_grad',
        #                                    operation="GradProbe(i, psi, probe_grad, obj, cx, cy, cz, px, py, f, firstpass, npsi, stack_size, nx, ny, nz, nxo, nyo, nzo, nbobj, nbprobe)",
        #                                    preamble=getks('ptycho/bragg/opencl/grad_elw.cl'),
        #                                    options=self.cl_options,
        #                                    arguments="__global float2* psi, __global float2* probe_grad, __global float2 *obj, __global int* cx, __global int* cy, __global int* cz, const float px, const float f, const char firstpass, const int npsi, const int stack_size, const int nx, const int ny, const int nz, const int nxo, const int nyo, const int nzo, const int nbobj, const int nbprobe")

        self.cl_reg_smooth_complex_support_grad = CL_ElK(self.cl_ctx, name='reg_smooth_complex_grad',
                                                         operation="RegSmoothComplexSupportGrad(i, d, grad, support,"
                                                                   "nx, ny, nz, reg_fac_modulus2, reg_fac_complex)",
                                                         preamble=getks('opencl/regularization_smooth_support.cl'),
                                                         options=self.cl_options,
                                                         arguments="__global float2* d, __global float2 *grad,"
                                                                   "__global char* support, const int nx, const int ny,"
                                                                   "const int nz, const float reg_fac_modulus2,"
                                                                   "const float reg_fac_complex")

        # Reduction kernels
        self.cl_norm_complex_n = CL_RedK(self.cl_ctx, np.float32, neutral="0", reduce_expr="a+b",
                                         map_expr="pown(length(d[i]), nn)", options=self.cl_options,
                                         arguments="__global float2 *d, const int nn")

        self.cl_norm_max_complex_n_red = CL_RedK(self.cl_ctx, np.float32, neutral="0", reduce_expr="fmax(a,b)",
                                                 map_expr="pown(length(d[i]), nn)", options=self.cl_options,
                                                 arguments="__global float2 *d, const int nn")

        # This will compute Poisson, Gaussian, Euclidian LLK as well as the sum of the calculated intensity
        self.cl_llk_red = CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)", reduce_expr="a+b",
                                  preamble=getks('ptycho/opencl/llk_red.cl'), options=self.cl_options,
                                  map_expr="LLKAll(i, iobs, psi, background, nbmode, nxy, nxystack)",
                                  arguments="__global float *iobs, __global float2 *psi, __global float *background, const int nbmode, const int nxy, const int nxystack")

        self.cl_reg_smooth_complex_support_llk_red = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                                             reduce_expr="a+b",
                                                             preamble=getks('opencl/regularization_smooth_support.cl'),
                                                             options=self.cl_options,
                                                             map_expr="RegSmoothComplexSupportLLK(i, d, support, nx, ny, nz)",
                                                             arguments="__global float2* d, __global char* support, const int nx, const int ny, const int nz")

        self.cl_center_obj_probe_red = CL_RedK(self.cl_ctx, cla.vec.float8, neutral="(float8)(0,0,0,1e16,1e16,0,0,0)",
                                               reduce_expr="center_obj_probe_red(a,b)",
                                               preamble=getks('opencl/complex.cl') +
                                                        getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                        getks('ptycho/bragg2d/opencl/center_obj_probe.cl'),
                                               options=self.cl_options,
                                               map_expr="center_obj_probe(i, obj, probe, m, cx, cy, pxp, pyp,"
                                                        "nxp, nyp, nxo, nyo, nzo)",
                                               arguments="__global float2* obj,__global float2* probe,"
                                                         "__global float* m, __global float* cx, __global float* cy,"
                                                         "const float pxp, const float pyp, const int nxp,"
                                                         "const int nyp, const int nxo, const int nyo, const int nzo")

        self.cl_center_obj_probe_finish = CL_ElK(self.cl_ctx, name='cl_center_obj_probe_finish',
                                                 operation="cix[0] = (int)round(c[0].s0 / fmax(c[0].s2,1e-10f));"
                                                           "ciy[0] = (int)round(c[0].s1 / fmax(c[0].s2,1e-10f));",
                                                 options=self.cl_options,
                                                 arguments="__global int *cix, __global int *ciy, __global float8* c")

        self.cl_cg_polak_ribiere_complex_red = CL_RedK(self.cl_ctx, cl.array.vec.float2, neutral="(float2)(0,0)",
                                                       reduce_expr="a+b",
                                                       map_expr="PolakRibiereComplex(grad[i], lastgrad[i])",
                                                       preamble=getks('opencl/cg_polak_ribiere_red.cl'),
                                                       arguments="__global float2 *grad, __global float2 *lastgrad")

        self.cl_psi2obj_gamma_red = CL_RedK(self.cl_ctx, cl.array.vec.float4, neutral="(float4)(0,0,0,0)",
                                            reduce_expr="a+b",
                                            map_expr="Psi2Obj_Gamma(i, psi, obj, probe, support, dobj, m, cx, cy, cixo,"
                                                     "ciyo,  dsx, dsy, dsz, pxo, pyo, pzo, pxp, pyp, f, npsi,"
                                                     "stack_size, nx, ny, nxo, nyo, nzo, nxp, nyp, nbobj, nbprobe)",
                                            preamble=getks('opencl/complex.cl') +
                                                     getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                     getks('ptycho/bragg2d/opencl/psi_to_obj_probe.cl'),
                                            options=self.cl_options,
                                            arguments="__global float2* psi, __global float2 *obj,"
                                                      "__global float2* probe,  __global char* support,"
                                                      "__global float2* dobj, __global float* m, __global float* cx,"
                                                      "__global float* cy, __global int* cixo, __global int* ciyo,"
                                                      "__global float* dsx, __global float* dsy, __global float* dsz,"
                                                      "const float pxo, const float pyo, const float pzo,"
                                                      "const float pxp, const float pyp, const float f, const int npsi,"
                                                      "const int stack_size, const int nx, const int ny,"
                                                      "const int nxo, const int nyo, const int nzo, const int nxp,"
                                                      "const int nyp, const int nbobj, const int nbprobe")

        self.cl_psi2obj_probe_gamma_red = CL_RedK(self.cl_ctx, cl.array.vec.float4, neutral="(float4)(0,0,0,0)",
                                                  reduce_expr="a+b",
                                                  map_expr="Psi2ObjProbe_Gamma(i, psi, obj, probe, support, dobj,"
                                                           "dprobe, m, cx, cy, cixo, ciyo,  dsx, dsy, dsz, pxo, pyo,"
                                                           "pzo, pxp, pyp, f, npsi, stack_size, nx, ny, nxo, nyo, nzo,"
                                                           "nxp, nyp, nbobj, nbprobe)",
                                                  preamble=getks('opencl/complex.cl') +
                                                           getks('ptycho/bragg2d/opencl/interp_probe.cl') +
                                                           getks('ptycho/bragg2d/opencl/psi_to_obj_probe.cl'),
                                                  options=self.cl_options,
                                                  arguments="__global float2* psi, __global float2 *obj,"
                                                            "__global float2* probe,  __global char* support,"
                                                            "__global float2* dobj, __global float2* dprobe,"
                                                            "__global float* m, __global float* cx, __global float* cy,"
                                                            "__global int* cixo, __global int* ciyo,"
                                                            "__global float* dsx, __global float* dsy,"
                                                            "__global float* dsz, const float pxo, const float pyo,"
                                                            "const float pzo, const float pxp, const float pyp,"
                                                            "const float f, const int npsi, const int stack_size,"
                                                            "const int nx, const int ny, const int nxo, const int nyo,"
                                                            "const int nzo, const int nxp, const int nyp,"
                                                            "const int nbobj, const int nbprobe")

        # 2nd order LLK(gamma) approximation
        self.cl_cg_poisson_gamma_red = CL_RedK(self.cl_ctx, cl.array.vec.float2, neutral="(float2)(0,0)",
                                               reduce_expr="a+b",
                                               map_expr="CG_Poisson_Gamma(i, obs, PO, PdO, dPO, dPdO, bg, dbg,"
                                                        "nxy, nxystack, nbmode, npsi)",
                                               preamble=getks('ptycho/opencl/cg_gamma_red.cl'),
                                               options=self.cl_options,
                                               arguments="__global float *obs, __global float2 *PO, "
                                                         "__global float2 *PdO, __global float2 *dPO, "
                                                         "__global float2 *dPdO, __global float *bg,"
                                                         "__global float *dbg, const int nxy,"
                                                         "const int nxystack, const int nbmode, const int npsi")
        # 4th order LLK(gamma) approximation
        self.cl_cg_poisson_gamma4_red = CL_RedK(self.cl_ctx, cl.array.vec.float4, neutral="(float4)(0,0,0,0)",
                                                reduce_expr="a+b",
                                                map_expr="CG_Poisson_Gamma4(i, obs, PO, PdO, dPO, dPdO, nxy, nxystack,"
                                                         "nbmode)",
                                                preamble=getks('ptycho/opencl/cg_gamma_red.cl'),
                                                options=self.cl_options,
                                                arguments="__global float *obs, __global float2 *PO,"
                                                          "__global float2 *PdO, __global float2 *dPO,"
                                                          "__global float2 *dPdO, const int nxy,"
                                                          "const int nxystack, const int nbmode")

        self.cl_reg_smooth_complex_support_gamma_red = CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)",
                                                               reduce_expr="a+b",
                                                               preamble=getks(
                                                                   'opencl/regularization_smooth_support.cl'),
                                                               options=self.cl_options,
                                                               map_expr="RegSmoothComplexSupportGamma(i, d, dir, support, nx, ny, nz)",
                                                               arguments="__global float2* d, __global float2* dir, __global char* support, const int nx, const int ny, const int nz")

        self.cl_scale_intensity = CL_RedK(self.cl_ctx, cl.array.vec.float2, neutral="(float2)(0,0)", reduce_expr="a+b",
                                          map_expr="scale_intensity(i, obs, calc, background, nxy, nxystack, nb_mode)",
                                          preamble=getks('ptycho/opencl/scale_red.cl'),
                                          arguments="__global float *obs, __global float2 *calc,"
                                                    "__global float *background, const int nxy, const int nxystack,"
                                                    "const int nb_mode")

        # custom kernels
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


"""
The default processing unit 
"""
default_processing_unit = CLProcessingUnitPtycho()


class CLObsDataStack:
    """
    Class to store a stack (e.g. 16 frames) of observed 3D Bragg data in OpenCL space.
    """

    def __init__(self, cl_obs, cl_x, cl_y, cl_dsx, cl_dsy, cl_dsz, i, npsi, cl_cixo=None, cl_ciyo=None):
        """

        :param cl_obs: pyopencl array of observed data, with N frames
        :param cl_x, cl_y: pyopencl arrays of the relative shift of each frame (illumination), in the laboratory
                           reference frame, in meters.
        :param cl_dsx, cl_dsy, cl_dsz: three components of the difference of scattering vector with the average one,
                                       given in the orthonormal detector reference frame.
        :param i: index of the first frame
        :param npsi: number of valid frames (others are filled with zeros)
        :param cl_cixo, cl_ciyo: arrays of the center of the intersection of the probe and the object,
                                 for each frame, in object pixel units. These will be computed by CalcCenterObjProbe
                                 so can be initially empty. Alternatively, they can be set to the center of
                                 the object coordinates (allowing a comparison with kinematically computed scattering).
        """
        self.cl_obs = cl_obs
        self.cl_x = cl_x
        self.cl_y = cl_y
        self.i = np.int32(i)
        self.npsi = np.int32(npsi)
        self.x = cl_x.get()
        self.y = cl_y.get()
        self.cl_dsx = cl_dsx
        self.cl_dsy = cl_dsy
        self.cl_dsz = cl_dsz
        self.dsx = cl_dsx.get()
        self.dsy = cl_dsy.get()
        self.dsz = cl_dsz.get()
        # arrays of the center of the intersection of the probe and the object, for each frame, in object
        # pixel units. These will be computed by CalcCenterObjProbe so can be initially empty.
        self.cl_cixo = cl_cixo
        self.cl_ciyo = cl_ciyo
        if cl_cixo is not None:
            self.cixo = self.cl_cixo.get()
        else:
            self.cixo = None
        if cl_ciyo is not None:
            self.ciyo = self.cl_ciyo.get()
        else:
            self.ciyo = None


class CLOperatorBragg2DPtycho(OperatorBragg2DPtycho):
    """
    Base class for a operators on CDI objects using OpenCL
    """

    def __init__(self, processing_unit=None):
        super(CLOperatorBragg2DPtycho, self).__init__()

        self.Operator = CLOperatorBragg2DPtycho
        self.OperatorSum = CLOperatorBragg2DPtychoSum
        self.OperatorPower = CLOperatorBragg2DPtychoPower

        if processing_unit is None:
            self.processing_unit = default_processing_unit
        else:
            self.processing_unit = processing_unit
        if self.processing_unit.cl_ctx is None:
            # OpenCL kernels have not been prepared yet, use a default initialization
            if main_default_processing_unit.cl_device is None:
                main_default_processing_unit.select_gpu(language='opencl')
            self.processing_unit.init_cl(cl_device=main_default_processing_unit.cl_device, test_fft=False)

    def apply_ops_mul(self, pty):
        """
        Apply the series of operators stored in self.ops to a Ptycho2D object.
        The operators are applied one after the other.

        :param pty: the Ptycho2D to which the operators will be applied.
        :return: the Ptycho2D object, after application of all the operators in sequence
        """
        if isinstance(pty, Bragg2DPtycho) is False:
            raise OperatorException(
                "ERROR: tried to apply operator:\n    %s \n  to:\n    %s\n  which is not a Ptycho object" % (
                    str(self), str(pty)))
        return super(CLOperatorBragg2DPtycho, self).apply_ops_mul(pty)

    def prepare_data(self, p: Bragg2DPtycho):
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
            # print("Moving object, probe, mask, support to OpenCL GPU")
            p._cl_obj = cla.to_device(self.processing_unit.cl_queue, p._obj)
            p._cl_probe = cla.to_device(self.processing_unit.cl_queue, p._probe2d.get(shift=True))
            if p.support is None:
                p._cl_support = cla.empty(self.processing_unit.cl_queue, p._obj.shape[-3:], dtype=np.int8)
                p._cl_support.fill(np.int8(1))
            else:
                p._cl_support = cla.to_device(self.processing_unit.cl_queue, p.support.astype(np.int8))
            p._cl_timestamp_counter = p._timestamp_counter
            if p._background is None:
                p._cl_background = cla.zeros(self.processing_unit.cl_queue, p.data.iobs.shape[-2:], dtype=np.float32)
            else:
                p._cl_background = cla.to_device(self.processing_unit.cl_queue, p._background)
            p._cl_m = cla.to_device(self.processing_unit.cl_queue, p.m.astype(np.float32))

        need_init_psi = False

        if has_attr_not_none(p, "_cl_psi") is False:
            need_init_psi = True
        elif p._cl_psi.shape[0:3] != (len(p._obj), len(p._probe2d.get()), self.processing_unit.cl_stack_size):
            need_init_psi = True
        if need_init_psi:
            ny, nx = p.data.iobs.shape[-2:]
            p._cl_psi = cla.empty(self.processing_unit.cl_queue, dtype=np.complex64,
                                  shape=(len(p._obj), len(p._probe2d.get()), self.processing_unit.cl_stack_size, ny,
                                         nx))

        if has_attr_not_none(p, "_cl_psi_v") is False:
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
        nzo, nyo, nxo = p._obj.shape[1:]
        cl_stack_size = self.processing_unit.cl_stack_size
        for i in range(0, nb_frame, cl_stack_size):
            vcx = np.zeros((cl_stack_size), dtype=np.float32)
            vcy = np.zeros((cl_stack_size), dtype=np.float32)
            vdsx = np.zeros((cl_stack_size), dtype=np.float32)
            vdsy = np.zeros((cl_stack_size), dtype=np.float32)
            vdsz = np.zeros((cl_stack_size), dtype=np.float32)
            vobs = np.zeros((cl_stack_size, ny, nx), dtype=np.float32)
            if nb_frame < (i + cl_stack_size):
                # We probably want to avoid this for 3D data
                print("Number of frames is not a multiple of %d, adding %d null frames" %
                      (cl_stack_size, i + cl_stack_size - nb_frame))
            for j in range(cl_stack_size):
                ij = i + j
                if ij < nb_frame:
                    vcx[j] = p.data.posx[ij]
                    vcy[j] = p.data.posy[ij]
                    vdsx[j] = p.data.ds1[0][ij]
                    vdsy[j] = p.data.ds1[1][ij]
                    vdsz[j] = p.data.ds1[2][ij]
                    vobs[j] = p.data.iobs[ij]
                else:
                    vcx[j] = vcx[0]
                    vcy[j] = vcy[0]
                    vobs[j] = np.zeros_like(vobs[0], dtype=np.float32)
            # Shift of the probe position for each frame
            cl_vcx = cl.array.to_device(self.processing_unit.cl_queue, vcx)
            cl_vcy = cl.array.to_device(self.processing_unit.cl_queue, vcy)
            # Shift of the reciprocal space vector
            cl_vdsx = cl.array.to_device(self.processing_unit.cl_queue, vdsx)
            cl_vdsy = cl.array.to_device(self.processing_unit.cl_queue, vdsy)
            cl_vdsz = cl.array.to_device(self.processing_unit.cl_queue, vdsz)
            # Observed intensity
            cl_vobs = cl.array.to_device(self.processing_unit.cl_queue, vobs)
            # Set the default center of project frames to the center of the object. This should be corrected
            # by using CalcCenterObjProbe()
            cl_cixo = cla.zeros(self.processing_unit.cl_queue, cl_stack_size, np.int32)
            cl_ciyo = cla.empty(self.processing_unit.cl_queue, cl_stack_size, np.int32)
            p._cl_obs_v.append(CLObsDataStack(cl_vobs, cl_vcx, cl_vcy, cl_vdsx, cl_vdsy, cl_vdsz, i,
                                              np.int32(min(cl_stack_size, nb_frame - i)),
                                              cl_cixo=cl_cixo, cl_ciyo=cl_ciyo))
        # Initialize the size and index of current stack
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
            pty._cl_obj = cl.array.empty_like(src['obj'])
            pty._cl_probe = cl.array.empty_like(src['probe'])
            pty._cl_psi = cl.array.empty_like(src['psi'])
            pty._cl_psi_v = {}
            dest = {'obj': pty._cl_obj, 'probe': pty._cl_probe, 'psi': pty._cl_psi, 'psi_v': pty._cl_psi_v}
        else:
            pty._cl_view[i_dest] = {'obj': cl.array.empty_like(src['obj']), 'probe': cl.array.empty_like(src['probe']),
                                    'psi': cl.array.empty_like(src['psi']), 'psi_v': {}}
            dest = pty._cl_view[i_dest]

        for i in range(len(src['psi_v'])):
            dest['psi_v'][i] = cl.array.empty_like(src['psi'])

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


# The only purpose of this class is to make sure it inherits from CLOperatorCDI and has a processing unit
class CLOperatorBragg2DPtychoSum(OperatorSum, CLOperatorBragg2DPtycho):
    def __init__(self, op1, op2):
        if np.isscalar(op1):
            op1 = Scale(op1)
        if np.isscalar(op2):
            op2 = Scale(op2)
        if isinstance(op1, CLOperatorBragg2DPtycho) is False or isinstance(op2, CLOperatorBragg2DPtycho) is False:
            raise OperatorException(
                "ERROR: cannot add a CLOperatorCDI with a non-CLOperatorCDI: %s + %s" % (str(op1), str(op2)))
        # We can only have a sum of two CLOperatorCDI, so they must have a processing_unit attribute.
        CLOperatorBragg2DPtycho.__init__(self, op1.processing_unit)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorBragg2DPtycho
        self.OperatorSum = CLOperatorBragg2DPtychoSum
        self.OperatorPower = CLOperatorBragg2DPtychoPower
        self.prepare_data = types.MethodType(CLOperatorBragg2DPtycho.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorBragg2DPtycho.timestamp_increment, self)
        self.view_copy = types.MethodType(CLOperatorBragg2DPtycho.view_copy, self)
        self.view_swap = types.MethodType(CLOperatorBragg2DPtycho.view_swap, self)
        self.view_sum = types.MethodType(CLOperatorBragg2DPtycho.view_sum, self)
        self.view_purge = types.MethodType(CLOperatorBragg2DPtycho.view_purge, self)


# The only purpose of this class is to make sure it inherits from CLOperatorCDI and has a processing unit
class CLOperatorBragg2DPtychoPower(OperatorPower, CLOperatorBragg2DPtycho):
    def __init__(self, op, n):
        CLOperatorBragg2DPtycho.__init__(self, op.processing_unit)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorBragg2DPtycho
        self.OperatorSum = CLOperatorBragg2DPtychoSum
        self.OperatorPower = CLOperatorBragg2DPtychoPower
        self.prepare_data = types.MethodType(CLOperatorBragg2DPtycho.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorBragg2DPtycho.timestamp_increment, self)
        self.view_copy = types.MethodType(CLOperatorBragg2DPtycho.view_copy, self)
        self.view_swap = types.MethodType(CLOperatorBragg2DPtycho.view_swap, self)
        self.view_sum = types.MethodType(CLOperatorBragg2DPtycho.view_sum, self)
        self.view_purge = types.MethodType(CLOperatorBragg2DPtycho.view_purge, self)


class FreePU(CLOperatorBragg2DPtycho):
    """
    Operator freeing OpenCL memory. The gpyfft data reference in self.processing_unit is removed,
    as well as any OpenCL pyopencl.array.Array attribute in the supplied wavefront.
    """

    def op(self, p):
        for o in dir(p):
            if isinstance(p.__getattribute__(o), cla.Array):
                p.__getattribute__(o).data.release()
                p.__setattr__(o, None)
        if has_attr_not_none(p, "_cl_psi_v"):
            for a in p._cl_psi_v.values():
                a.data.release()
            p._cl_psi_v = {}
        if has_attr_not_none(self.processing_unit, 'gpyfft_plan'):
            if has_attr_not_none(self.processing_unit.gpyfft_plan, 'data'):
                self.processing_unit.free_fft_plans()
        for v in p._cl_obs_v:
            for o in dir(v):
                if isinstance(v.__getattribute__(o), cla.Array):
                    v.__getattribute__(o).data.release()
        p._cl_obs_v = None
        gc.collect()
        return p


class Scale(CLOperatorBragg2DPtycho):
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

    def op(self, p):
        if self.x == 1:
            return p

        if np.isreal(self.x):
            scale_k = self.processing_unit.cl_scale
            x = np.float32(self.x)
        else:
            scale_k = self.processing_unit.cl_scale_complex
            x = np.complex64(self.x)

        if self.obj:
            scale_k(p._cl_obj, x)
        if self.probe:
            scale_k(p._cl_probe, x)
        if self.psi:
            scale_k(p._cl_psi, x)
            for i in range(len(p._cl_psi_v)):
                scale_k(p._cl_psi_v[i], x)
        return p


class _CalcCenterObjProbe(CLOperatorBragg2DPtycho):
    """
    For each frame (illumination position), compute the center coordinates of the illuminated part of the object,
    in object pixel units, and store it in p._cl_obs_v.cl_vc{x|y}. This works on a single stack
    """

    def op(self, p):
        i = p._cl_stack_i
        nxp = np.int32(p._probe2d.get().shape[-1])
        nyp = np.int32(p._probe2d.get().shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        nyo = np.int32(p._obj.shape[-2])
        nzo = np.int32(p._obj.shape[-3])
        pxp = np.float32(p._probe2d.pixel_size)

        if p._cl_obs_v[i].cl_cixo is None:
            p._cl_obs_v[i].cl_cixo = cla.empty(self.processing_unit.cl_queue, p._cl_obs_v[i].cl_x.shape, dtype=np.int32)
            p._cl_obs_v[i].cl_ciyo = cla.empty(self.processing_unit.cl_queue, p._cl_obs_v[i].cl_x.shape, dtype=np.int32)

        for j in range(p._cl_obs_v[i].npsi):
            cl_c = self.processing_unit.cl_center_obj_probe_red(p._cl_obj[0], p._cl_probe[0], p._cl_m,
                                                                p._cl_obs_v[i].cl_x[j], p._cl_obs_v[i].cl_y[j],
                                                                pxp, pxp, nxp, nyp, nxo, nyo, nzo)

            # c = cl_c.get()
            # print(c, c['s0'] / max(c['s2'], 1e-8), c['s1'] / max(c['s2'], 1e-8))
            self.processing_unit.cl_center_obj_probe_finish(p._cl_obs_v[i].cl_cixo[j], p._cl_obs_v[i].cl_ciyo[j], cl_c)
        p._cl_obs_v[i].cixo = p._cl_obs_v[i].cl_cixo.get()
        p._cl_obs_v[i].ciyo = p._cl_obs_v[i].cl_ciyo.get()
        return p


class CalcCenterObjProbe(CLOperatorBragg2DPtycho):
    """
    For each frame (illumination position), compute the center coordinates of the illuminated part of the object,
    in object pixel units, and store it in p._cl_obs_v.cl_vc{x|y}
    """

    def __new__(cls):
        return LoopStack(_CalcCenterObjProbe())


class ObjProbe2Psi(CLOperatorBragg2DPtycho):
    """
    Computes Psi = Obj(r) * Probe(r-r_j) for a stack of N probe positions.
    """

    def op(self, p):
        # Multiply obj and probe with quadratic phase factor, taking into account all modes (if any)
        i = p._cl_stack_i
        nxp = np.int32(p._probe2d.get().shape[-1])
        nyp = np.int32(p._probe2d.get().shape[-2])
        nx = np.int32(p.data.iobs.shape[-2])
        ny = np.int32(p.data.iobs.shape[-2])
        nb_probe = np.int32(p._cl_psi.shape[1])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        pxo = np.float32(p.pxo)
        pyo = np.float32(p.pyo)
        pzo = np.float32(p.pzo)
        pxp = np.float32(p._probe2d.pixel_size)
        # cl_x, cl_y are probe positions in metric units in the xyz laboratory frame
        # cl_ix, cl_iy are the pixel center coordinates of the object volume to be integrated in the XYZ detector frame
        self.processing_unit.cl_object_probe_mult(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe, p._cl_support, p._cl_m,
                                                  p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y, p._cl_obs_v[i].cl_cixo,
                                                  p._cl_obs_v[i].cl_ciyo, p._cl_obs_v[i].cl_dsx, p._cl_obs_v[i].cl_dsy,
                                                  p._cl_obs_v[i].cl_dsz, pxo, pyo, pzo, pxp, pxp, f,
                                                  p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size, nx, ny,
                                                  nxo, nyo, nzo, nxp, nyp, nb_obj, nb_probe)
        return p


class ObjProbe2PsiDebug(CLOperatorBragg2DPtycho):
    """
    Computes Psi = Obj(r) * Probe(r-r_j) for a stack of N probe positions, and store the 3D Psi, the 3D probe in new
    arrays. This also allows to specify the probe position and and the scattering vector, without altering the
    BraggPtycho2D object and its data. This is only useful for debugging purposes.
    This can be limited to a number of frames to avoid
    """

    def __init__(self, npsi=None, positions=None, scattering_vector=None, calc_3d=True):
        """

        :param npsi: the number of positions for which to compute the 3D arrays (default=16 to limit memory usage).
                     This will be superseded by the number of positions or scattering_vector, if given.
        :param positions: (x, y, z) tuple or 2d array with ptycho probe positions in meters. The coordinates
                          must be in the laboratory reference frame, relative to the object center.
                          There cannot be more positions than the stack size.
                          [Default = None, use the positions supplied in the ptycho object]
        :param scattering_vector=(sx, sy, sz): the coordinates of the central scattering vector for all frames, taking
               into account any tilt for multi-angle Bragg projection ptychography. The scattering vector coordinates
               units should be inverse meters (without 2pi multiplication), i.e. the norm should be 2*sin(theta)/lambda,
               where theta is the Bragg angle. Note that only the difference between the scattering vector and the
               average (intensity-weighted) scattering vector is used for multi-orientation back-projection.
               [Default: None, all frames are taken with the same orientation]
        :param calc_3d: if True, will store _psi3d, _probe3d and _obj3d in the Bragg2DPtycho object as 3D
                        arrays of the Psi, object and probe before 2D integration. This can use a lot of
                        memory, so it is advised to limit npsi to a small number when using this.
        """
        super(ObjProbe2PsiDebug, self).__init__()
        self.npsi = npsi
        self.positions = positions
        self.scattering_vector = scattering_vector
        self.calc_3d = calc_3d

    def op(self, p: Bragg2DPtycho):
        # Multiply obj and probe with quadratic phase factor, taking into account all modes (if any)
        pu = self.processing_unit
        i = p._cl_stack_i
        nxp = np.int32(p._probe2d.get().shape[-1])
        nyp = np.int32(p._probe2d.get().shape[-2])
        nx = np.int32(p.data.iobs.shape[-2])
        ny = np.int32(p.data.iobs.shape[-2])
        nb_probe = np.int32(p._cl_psi.shape[1])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        pxo = np.float32(p.pxo)
        pyo = np.float32(p.pyo)
        pzo = np.float32(p.pzo)
        pxp = np.float32(p._probe2d.pixel_size)
        # cl_x, cl_y are probe positions in metric units in the xyz laboratory frame
        # cl_ix, cl_iy are the pixel center coordinates of the object volume to be integrated in the XYZ detector frame

        if self.positions is not None:
            x, y = self.positions[:2]
            assert (len(x) == len(y))
            cl_x = cla.to_device(pu.cl_queue, x.astype(np.float32))
            cl_y = cla.to_device(pu.cl_queue, y.astype(np.float32))
            cl_cixo = cla.empty(pu.cl_queue, cl_x.shape, dtype=np.int32)
            cl_ciyo = cla.empty(pu.cl_queue, cl_x.shape, dtype=np.int32)

            for j in range(len(x)):
                cl_c = self.processing_unit.cl_center_obj_probe_red(p._cl_obj[0], p._cl_probe[0], p._cl_m,
                                                                    cl_x[j], cl_y[j], pxp, pxp, nxp, nyp, nxo, nyo, nzo)

                pu.cl_center_obj_probe_finish(cl_cixo[j], cl_ciyo[j], cl_c)
        else:
            cl_x, cl_y = p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y
            cl_cixo, cl_ciyo = p._cl_obs_v[i].cl_cixo, p._cl_obs_v[i].cl_ciyo

        if self.scattering_vector is not None:
            sx, sy, sz = self.scattering_vector
            ds = sx - p.data.s0[0], sy - p.data.s0[1], sz - p.data.s0[2]
            assert (len(sx) == len(sy) == len(sz))
            if self.positions is not None:
                assert (cl_x.size == sx.size)
            ds1 = rotate(p.data.im, ds[0], ds[1], ds[2])
            cl_dsx = cla.to_device(pu.cl_queue, ds1[0].astype(np.float32))
            cl_dsy = cla.to_device(pu.cl_queue, ds1[1].astype(np.float32))
            cl_dsz = cla.to_device(pu.cl_queue, ds1[2].astype(np.float32))
        else:
            cl_dsx = p._cl_obs_v[i].cl_dsx
            cl_dsy = p._cl_obs_v[i].cl_dsy
            cl_dsz = p._cl_obs_v[i].cl_dsz

        npsi = np.int32(p._cl_obs_v[i].npsi)
        if self.npsi is not None and self.positions is None and self.scattering_vector is None:
            if npsi > self.npsi:
                npsi = self.npsi
        elif self.positions is not None or self.scattering_vector is not None:
            npsi = np.int32(cl_x.size)

        if npsi > pu.cl_stack_size:
            npsi = np.int32(pu.cl_stack_size)

        if self.calc_3d:
            _cl_psi3d = cla.zeros(self.processing_unit.cl_queue, dtype=np.complex64,
                                  shape=(len(p._obj), len(p._probe2d.get()), npsi, nzo, ny, nx))
            _cl_probe3d = cla.zeros(self.processing_unit.cl_queue, dtype=np.complex64,
                                    shape=(len(p._obj), len(p._probe2d.get()), npsi, nzo, ny, nx))
            _cl_obj3d = cla.zeros(self.processing_unit.cl_queue, dtype=np.complex64,
                                  shape=(len(p._obj), len(p._probe2d.get()), npsi, nzo, ny, nx))

            self.processing_unit.cl_object_probe_mult_debug(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe, p._cl_support,
                                                            p._cl_m, _cl_psi3d, _cl_obj3d, _cl_probe3d,
                                                            cl_x, cl_y, cl_cixo, cl_ciyo, cl_dsx, cl_dsy, cl_dsz,
                                                            pxo, pyo, pzo, pxp, pxp, f,
                                                            npsi, self.processing_unit.cl_stack_size,
                                                            nx, ny, nxo, nyo, nzo, nxp, nyp, nb_obj, nb_probe)
            p._psi3d = _cl_psi3d.get()
            p._probe3d = _cl_probe3d.get()
            p._obj3d = _cl_obj3d.get()
        else:
            self.processing_unit.cl_object_probe_mult(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe, p._cl_support,
                                                      p._cl_m, cl_x, cl_y, cl_cixo, cl_ciyo, cl_dsx, cl_dsy, cl_dsz,
                                                      pxo, pyo, pzo, pxp, pxp, f, npsi, pu.cl_stack_size, nx, ny,
                                                      nxo, nyo, nzo, nxp, nyp, nb_obj, nb_probe)

        return p


class FT(CLOperatorBragg2DPtycho):
    """
    Forward Fourier-transform a Psi array, i.e. a stack of N Obj*Probe views
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(FT, self).__init__()
        self.scale = scale

    def op(self, pty):
        plan = self.processing_unit.cl_fft_get_plan(pty._cl_psi, axes=(-1, -2), shuffle_axes=True)
        for e in plan.enqueue(forward=True):
            e.wait()  # Needed as CLFFT may use its own queues
        if self.scale:
            self.processing_unit.cl_scale(pty._cl_psi, np.float32(1 / np.sqrt(pty._cl_psi[0, 0, 0].size)))
        return pty


class IFT(CLOperatorBragg2DPtycho):
    """
    Backward Fourier-transform a Psi array, i.e. a stack of N Obj*Probe views
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(IFT, self).__init__()
        self.scale = scale

    def op(self, pty):
        plan = self.processing_unit.cl_fft_get_plan(pty._cl_psi, axes=(-1, -2), shuffle_axes=True)
        for e in plan.enqueue(forward=False):
            e.wait()  # Needed as CLFFT may use its own queues
        if self.scale:
            self.processing_unit.cl_scale(pty._cl_psi, np.float32(np.sqrt(pty._cl_psi[0, 0, 0].size)))
        return pty


class ShowPsi(CLOperatorBragg2DPtycho):
    """
    Class to display psi during an optimization.
    """

    def __init__(self, i=0, fig_num=-1, title=None, rotation=None):
        """
        :param i: the index of the Psi array to display (if the stack has several)
        :param fig_num: the matplotlib figure number. if None, a new figure will be created. if -1 (the default), the
                        current figure will be used.
        :param title: the title for the view. If None, a default title will be used.
        """
        super(ShowPsi, self).__init__()
        self.i = i
        self.fig_num = fig_num
        self.title = title
        self.rotation = rotation

    def op(self, p):
        calc = np.fft.fftshift(p._cl_psi.get()[0, 0, self.i])
        i_frame = p._cl_stack_i * self.processing_unit.cl_stack_size + self.i
        obs = np.fft.fftshift(p.data.iobs[i_frame])
        if self.fig_num == -1:
            fig = plt.figure(figsize=(10, 4))
        else:
            fig = plt.figure(self.fig_num, figsize=(10, 4))
        m = np.log10(obs.max())
        plt.clf()
        plt.subplot(121)
        plt.imshow(np.log10(abs(calc) ** 2), vmin=0, vmax=m)
        plt.title('Calc')
        plt.subplot(122)
        plt.imshow(np.log10(obs), vmin=0, vmax=m)
        plt.title('Obs')
        if self.title is not None:
            plt.suptitle(self.title)
        else:
            plt.suptitle("Frame #%d" % (i_frame))
        return p

    def timestamp_increment(self, p):
        # This display operation does not modify the data.
        pass


class Calc2Obs(CLOperatorBragg2DPtycho):
    """
    Calculate the intensities and copy them to the observed ones. Can be used for simulation.
    The new observed values are also copied to the main memory array.
    This applies to all stack of frames.
    """

    def __init__(self, poisson_noise=True, nb_photons_per_frame=None):
        """

        :param poisson_noise: if True, Poisson noise will be added to the new observed diffraction data
        :param nb_photons_per_frame: average number of photons per frame. The data will be scaled to match this. If
                                     None, the calculated number of photons is used.
        """
        super(Calc2Obs, self).__init__()
        self.poisson_noise = poisson_noise
        self.nb_photons_per_frame = nb_photons_per_frame

    def op(self, p):
        nxy = np.int32(p._cl_psi.shape[-2] * p._cl_psi.shape[-1])
        nxystack = np.int32(nxy * self.processing_unit.cl_stack_size)
        nb_mode = np.int32(p._cl_psi.shape[0] * p._cl_psi.shape[1])
        for i in range(p._cl_stack_nb):
            p = FT(scale=False) * ObjProbe2Psi() * SelectStack(i) * p
            self.processing_unit.cl_calc2obs(p._cl_obs_v[i].cl_obs, p._cl_psi, nb_mode, nxystack)
            obs = p._cl_obs_v[i].cl_obs.get()
            for j in range(self.processing_unit.cl_stack_size):
                ij = i * self.processing_unit.cl_stack_size + j
                if j < p._cl_obs_v[i].npsi:
                    p.data.iobs[ij] = obs[j]
            if i == 0:
                p.data.iobs_sum = obs.sum()
            else:
                p.data.iobs_sum += obs.sum()
        # Now scale the intensities and apply Poisson noise
        if self.nb_photons_per_frame is not None:
            p.data.iobs *= self.nb_photons_per_frame * len(p.data.iobs) / p.data.iobs_sum
        if self.poisson_noise:
            p.data.iobs = np.random.poisson(p.data.iobs)
        # Make sure we still have float32
        p.data.iobs = p.data.iobs.astype(np.float32)
        # Copy to GPU memory
        if self.nb_photons_per_frame > 0 or self.poisson_noise:
            for i in range(p._cl_stack_nb):
                ij0 = i * self.processing_unit.cl_stack_size
                cl.enqueue_copy(self.processing_unit.cl_queue, dest=p._cl_obs_v[i].cl_obs.data,
                                src=p.data.iobs[ij0:ij0 + p._cl_obs_v[i].npsi])
        return p


class ApplyAmplitude(CLOperatorBragg2DPtycho):
    """
    Apply the magnitude from observed intensities, keep the phase. Applies to a stack of N views.
    """

    def __init__(self, calc_llk=False, difference=False):
        """

        :param calc_llk: if True, the log-likelihood will be calculated for this stack.
        :param difference: if True, in return Psi will be the difference of the Fourier-constrained Psi with the
                           original array
        """
        super(ApplyAmplitude, self).__init__()
        self.calc_llk = calc_llk
        self.difference = difference

    def op(self, p):
        # TODO: use a single-pass reduction kernel to apply the amplitude and compute the LLK
        if self.calc_llk:
            p = LLK() * p
        nxy = np.int32(p._cl_psi.shape[-2] * p._cl_psi.shape[-1])
        nxystack = np.int32(nxy * self.processing_unit.cl_stack_size)
        nb_mode = np.int32(p._cl_psi.shape[0] * p._cl_psi.shape[1])
        i = p._cl_stack_i
        nb_psi = np.int32(p._cl_obs_v[i].npsi)
        if self.difference:
            self.processing_unit.cl_projection_amplitude_diff(p._cl_obs_v[i].cl_obs[0], p._cl_psi, p._cl_background,
                                                              nb_mode, nxy, nxystack, nb_psi)
        else:
            self.processing_unit.cl_projection_amplitude(p._cl_obs_v[i].cl_obs[0], p._cl_psi, p._cl_background,
                                                         nb_mode, nxy, nxystack, nb_psi)
        return p


class FourierApplyAmplitude(CLOperatorBragg2DPtycho):
    """
    Fourier magnitude operator, performing a Fourier transform, the magnitude projection, and a backward FT on a stack
    of N views.
    """

    def __new__(cls, calc_llk=False, difference=False):
        return IFT(scale=False) * ApplyAmplitude(calc_llk=calc_llk, difference=difference) * FT(scale=False)

    def __init__(self, calc_llk=False):
        super(FourierApplyAmplitude, self).__init__()


class LLK(CLOperatorBragg2DPtycho):
    """
    Log-likelihood reduction kernel. Can only be used when Psi is in diffraction space.
    This is a reduction operator - it will write llk as an argument in the Ptycho object, and return the object.
    If _cl_stack_i==0, the llk is re-initialized. Otherwise it is added to the current value.

    The LLK can be calculated directly from object and probe using: p = LoopStack(LLK() * FT() * ObjProbe2Psi()) * p
    """

    def op(self, p):
        i = p._cl_stack_i
        nb_mode = np.int32(p._cl_psi.shape[0] * p._cl_psi.shape[1])
        nb_psi = p._cl_obs_v[i].npsi
        nxy = np.int32(p._cl_psi.shape[-2] * p._cl_psi.shape[-1])
        nxystack = np.int32(self.processing_unit.cl_stack_size * nxy)
        llk = self.processing_unit.cl_llk_red(p._cl_obs_v[i].cl_obs[:nb_psi], p._cl_psi, p._cl_background,
                                              nb_mode, nxy, nxystack).get()
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


class SmoothRegLLK(CLOperatorBragg2DPtycho):
    """
    Log-likelihood reduction kernel, computing the smoothing regularization LLK terms for the object and/or probe.
    This is a reduction operator - it will write llk_reg as an argument in the Ptycho object, and return the object.
    """

    def __init__(self, reg_fac_obj_c=0, reg_fac_obj_a=0):
        """

        :param reg_fac_obj_a: scale for smoothing regularization on the object squared modulus
        :param reg_fac_obj_c: scale for smoothing regularization on the object complex values
        """
        super(SmoothRegLLK, self).__init__()
        self.reg_fac_obj_a = reg_fac_obj_a
        self.reg_fac_obj_c = reg_fac_obj_c

    def op(self, p):
        i = p._cl_stack_i
        k = 8 * p.support_sum ** 2 / (p.data.iobs.size * p.data.iobs_sum)
        if self.reg_fac_obj_a + self.reg_fac_obj_c > 0:
            nx = np.int32(p._obj.shape[-1])
            ny = np.int32(p._obj.shape[-2])
            nz = np.int32(p._obj.shape[-3])
            llk = self.processing_unit.cl_reg_smooth_complex_support_llk_red(p._cl_obj, p._cl_support, nx, ny, nz).get()
            if p._cl_stack_i == 0:
                p.llk_reg = (self.reg_fac_obj_a * llk['x'] + self.reg_fac_obj_c * llk['y']) / k
            else:
                p.llk_reg += (self.reg_fac_obj_a * llk['x'] + self.reg_fac_obj_c * llk['y']) / k

        return p


class _Psi2ObjProbeGrad(CLOperatorBragg2DPtycho):
    """
    Computes updated Obj and Probe gradients from Psi, Obj and Probe, looping over frames in a single stack.
    """

    def __init__(self, update_object=True, update_probe=False):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        """
        super(_Psi2ObjProbeGrad, self).__init__()
        self.update_object = update_object
        # TODO: implement probe update
        self.update_probe = update_probe

    def op(self, p):
        i = p._cl_stack_i
        nxp = np.int32(p._probe2d.get().shape[-1])
        nyp = np.int32(p._probe2d.get().shape[-2])
        ny = np.int32(p._cl_psi.shape[-2])
        nx = np.int32(p._cl_psi.shape[-1])
        nb_probe = np.int32(p._cl_psi.shape[1])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(-np.pi / (p.data.wavelength * p.data.detector['distance']))
        pxo = np.float32(p.pxo)
        pyo = np.float32(p.pyo)
        pzo = np.float32(p.pzo)
        pxp = np.float32(p._probe2d.pixel_size)

        if self.update_object:
            # print('_Psi2ObjProbeGrad(): computing object gradient')
            # We loop over the stack size to avoid creating an array with N=stack_size object gradient arrays
            # TODO: use atomic_add
            for ii in range(p._cl_obs_v[i].npsi):
                self.processing_unit.cl_psi2obj_grad(p._cl_psi[0, 0, ii], p._cl_obj, p._cl_probe, p._cl_obj_grad,
                                                     p._cl_support, p._cl_m, p._cl_obs_v[i].x[ii], p._cl_obs_v[i].y[ii],
                                                     p._cl_obs_v[i].cixo[ii], p._cl_obs_v[i].ciyo[ii],
                                                     p._cl_obs_v[i].dsx[ii], p._cl_obs_v[i].dsy[ii],
                                                     p._cl_obs_v[i].dsz[ii], pxo, pyo, pzo, pxp, pxp, f,
                                                     self.processing_unit.cl_stack_size, nx, ny, nxo, nyo, nzo,
                                                     nxp, nyp, nb_obj, nb_probe)

        if self.update_probe:
            for ii in range(p._cl_obs_v[i].npsi):
                self.processing_unit.cl_psi2probe_grad(p._cl_psi[0, 0, ii], p._cl_obj, p._cl_probe, p._cl_probe_grad,
                                                       p._cl_support, p._cl_m, p._cl_obs_v[i].x[ii],
                                                       p._cl_obs_v[i].y[ii], p._cl_obs_v[i].cixo[ii],
                                                       p._cl_obs_v[i].ciyo[ii], p._cl_obs_v[i].dsx[ii],
                                                       p._cl_obs_v[i].dsy[ii], p._cl_obs_v[i].dsz[ii], pxo, pyo, pzo,
                                                       pxp, pxp, f, self.processing_unit.cl_stack_size, nx, ny,
                                                       nxo, nyo, nzo, nxp, nyp, nb_obj, nb_probe)
            # print('_Psi2ObjProbeGrad(): computing probe gradient:', p._cl_probe_grad.get().sum())
        return p


class _SmoothRegGrad(CLOperatorBragg2DPtycho):
    """
    Adds smoothing regularization contribution to the object gradient. (probe: TODO)
    """

    def __init__(self, update_object=True, update_probe=False, reg_fac_obj_c=0, reg_fac_obj_a=0):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param reg_fac_obj_a: scale for smoothing regularization on the object squared modulus
        :param reg_fac_obj_c: scale for smoothing regularization on the object complex values
        """
        super(_SmoothRegGrad, self).__init__()
        self.update_object = update_object
        self.reg_fac_obj_a = np.float32(reg_fac_obj_a)
        self.reg_fac_obj_c = np.float32(reg_fac_obj_c)
        # TODO: implement probe update
        self.update_probe = update_probe

    def op(self, p):
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])

        if self.update_object:
            if self.reg_fac_obj_a + self.reg_fac_obj_c > 0:
                k = 8 * p.support_sum ** 2 / (p.data.iobs.size * p.data.iobs_sum)
                reg_fac_amplitude = np.float32(self.reg_fac_obj_a / k)
                reg_fac_complex = np.float32(self.reg_fac_obj_c / k)

                self.processing_unit.cl_reg_smooth_complex_support_grad(p._cl_obj, p._cl_obj_grad, p._cl_support,
                                                                        nxo, nyo, nzo, reg_fac_amplitude,
                                                                        reg_fac_complex)
        if self.update_probe:
            # TODO
            pass

        return p


class _Psi2ObjProbeGamma(CLOperatorBragg2DPtycho):
    """
    Computes the line minimization gamma contribution from Psi, Obj and Probe, looping over all stack of frames.
    """

    def __init__(self, update_object=True, update_probe=False):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        """
        super(_Psi2ObjProbeGamma, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe

    def op(self, p):
        i = p._cl_stack_i
        nxp = np.int32(p._probe2d.get().shape[-1])
        nyp = np.int32(p._probe2d.get().shape[-2])
        ny = np.int32(p._cl_psi.shape[-2])
        nx = np.int32(p._cl_psi.shape[-1])
        nb_probe = np.int32(p._cl_psi.shape[1])
        nb_obj = np.int32(p._cl_psi.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(-np.pi / (p.data.wavelength * p.data.detector['distance']))
        pxo = np.float32(p.pxo)
        pyo = np.float32(p.pyo)
        pzo = np.float32(p.pzo)
        pxp = np.float32(p._probe2d.pixel_size)

        if self.update_object or self.update_probe:
            tmp = self.processing_unit.cl_psi2obj_probe_gamma_red(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe,
                                                                  p._cl_support, p._cl_obj_dir, p._cl_probe_dir,
                                                                  p._cl_m, p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y,
                                                                  p._cl_obs_v[i].cl_cixo, p._cl_obs_v[i].cl_ciyo,
                                                                  p._cl_obs_v[i].cl_dsx, p._cl_obs_v[i].cl_dsy,
                                                                  p._cl_obs_v[i].cl_dsz, pxo, pyo, pzo, pxp, pxp, f,
                                                                  p._cl_obs_v[i].npsi,
                                                                  self.processing_unit.cl_stack_size, nx, ny, nxo, nyo,
                                                                  nzo, nxp, nyp, nb_obj, nb_probe).get()
            p._cl_cg_gamma_n += tmp['x']
            p._cl_cg_gamma_d += tmp['y']
            p._psi2d_to_obj3d_chi2 += tmp['z']

        return p


class _SmoothRegGamma(CLOperatorBragg2DPtycho):
    """
    Computes the line minimization gamma contribution due to the object smoothing regularization.
    """

    def __init__(self, update_object=True, update_probe=False, reg_fac_obj_a=0, reg_fac_obj_c=0):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param reg_fac_obj_a: scale for smoothing regularization on the object squared modulus
        :param reg_fac_obj_c: scale for smoothing regularization on the object complex values
        """
        super(_SmoothRegGamma, self).__init__()
        self.update_object = update_object
        # TODO: implement probe update
        self.update_probe = update_probe
        self.reg_fac_obj_a = reg_fac_obj_a
        self.reg_fac_obj_c = reg_fac_obj_c

    def op(self, p):
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])

        if self.update_object:
            if self.reg_fac_obj_a + self.reg_fac_obj_c > 0:
                tmp = self.processing_unit.cl_reg_smooth_complex_support_gamma_red(p._cl_obj, p._cl_obj_dir,
                                                                                   p._cl_support,
                                                                                   nxo, nyo, nzo).get()
                p._cl_cg_gamma_n += tmp['x'] * self.reg_fac_obj_a
                p._cl_cg_gamma_d += tmp['y'] * self.reg_fac_obj_a
                p._cl_cg_gamma_n += tmp['z'] * self.reg_fac_obj_c
                p._cl_cg_gamma_d += tmp['w'] * self.reg_fac_obj_c

        if self.update_probe:
            # TODO
            pass

        return p


class Psi2ObjProbe(CLOperatorBragg2DPtycho):
    """
    Computes updated Obj and Probe from Psi and Probe(r-r_j). Assumes the Psi values are stored
    """

    def __init__(self, nb_cycle=1, update_object=True, update_probe=False, reg_fac_obj_a=0, reg_fac_obj_c=0,
                 verbose=False):
        """

        :param nb_cycle: number of cycles for the conjugate gradient
        :param update_object: update object ?
        :param update_probe: update probe ?
        :param reg_fac_obj_a: scale for smoothing regularization on the object squared modulus
        :param reg_fac_obj_c: scale for smoothing regularization on the object complex values
        :param verbose: if True, be verbose during cycles
        """
        super(Psi2ObjProbe, self).__init__()
        self.nb_cycle = nb_cycle
        self.update_object = update_object
        self.reg_fac_obj_a = reg_fac_obj_a
        self.reg_fac_obj_c = reg_fac_obj_c
        # TODO: implement probe update with 2D regularization
        self.update_probe = update_probe
        self.verbose = verbose

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new ML operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        if n == 0:
            return 1
        if n == 1:
            return self
        return Psi2ObjProbe(nb_cycle=self.nb_cycle * n, update_object=self.update_object,
                            update_probe=self.update_probe, reg_fac_obj_a=self.reg_fac_obj_a,
                            reg_fac_obj_c=self.reg_fac_obj_c, verbose=self.verbose)

    def op(self, p):
        cl_queue = self.processing_unit.cl_queue

        # TODO: keep object gradient and direction ? => Need to use a memory pool
        # TODO: avoid zero initialization of these arrays ?

        p._cl_obj_grad = cla.zeros_like(p._cl_obj)
        p._cl_probe_grad = cla.zeros_like(p._cl_probe)
        if self.nb_cycle > 1:
            p._cl_obj_grad_last = cla.zeros_like(p._cl_obj)
            p._cl_obj_dir = cla.zeros_like(p._cl_obj)
            p._cl_probe_grad_last = cla.zeros_like(p._cl_probe)
            p._cl_probe_dir = cla.zeros_like(p._cl_probe)
        else:
            p._cl_obj_grad_last, p._cl_obj_dir = None, None
            p._cl_probe_grad_last, p._cl_probe_dir = None, None

        for ic in range(self.nb_cycle):
            t0 = timeit.default_timer()

            if self.nb_cycle > 1:
                # Swap gradient arrays - for CG, we need the previous gradient
                if self.update_object:
                    p._cl_obj_grad, p._cl_obj_grad_last = p._cl_obj_grad_last, p._cl_obj_grad
                if self.update_probe:
                    p._cl_probe_grad, p._cl_probe_grad_last = p._cl_probe_grad_last, p._cl_probe_grad

            # 1) Compute the gradients
            p._cl_obj_grad.fill(np.complex64(0))
            p = LoopStack(_Psi2ObjProbeGrad(update_object=self.update_object, update_probe=self.update_probe),
                          keep_psi=True) * p
            # Frequency-filter the gradient by a gaussian convolution along z
            nzo = np.int32(p._obj.shape[-3])
            nxyo = np.int32(p._obj.shape[-2] * p._obj.shape[-1])
            nb_obj = np.int32(p._obj.shape[0])
            sigma = np.float32(2)  # FWHM is 2.35 * sigma, convolution kernel is 15-pixels wide.
            # p._obj_grad0 = p._cl_obj_grad.get()
            self.processing_unit.cl_gauss_convolve_z(p._cl_obj_grad[0, 0], sigma, nxyo, nzo, nb_obj)
            # p._obj_grad1 = p._cl_obj_grad.get()

            # Regularisation gradient
            if self.reg_fac_obj_a + self.reg_fac_obj_c > 0:
                p = _SmoothRegGrad(reg_fac_obj_c=self.reg_fac_obj_c, reg_fac_obj_a=self.reg_fac_obj_a) * p

            # 2) Search direction
            if self.nb_cycle == 1:
                if self.update_object or self.update_probe:
                    p._cl_obj_dir = p._cl_obj_grad
                    p._cl_probe_dir = p._cl_probe_grad
            else:
                if ic == 0:
                    # first cycle
                    if self.update_object:
                        cl.enqueue_copy(cl_queue, src=p._cl_obj_grad.data, dest=p._cl_obj_dir.data)
                    if self.update_probe:
                        cl.enqueue_copy(cl_queue, src=p._cl_probe_grad.data, dest=p._cl_probe_dir.data)
                else:
                    beta_d, beta_n = 0, 0
                    # Polak-Ribière CG coefficient
                    cg_pr = self.processing_unit.cl_cg_polak_ribiere_complex_red
                    if self.update_object:
                        tmp = cg_pr(p._cl_obj_grad, p._cl_obj_grad_last).get()
                        beta_n += tmp['x']
                        beta_d += tmp['y']
                    if self.update_probe:
                        tmp = cg_pr(p._cl_probe_grad, p._cl_probe_grad_last).get()
                        beta_n += tmp['x']
                        beta_d += tmp['y']
                    # print("Beta= %e / %e" % (beta_n, beta_d))
                    # Reset direction if beta<0 => beta=0
                    beta = np.float32(max(0, beta_n / max(1e-20, beta_d)))
                    if np.isnan(beta_n + beta_d) or np.isinf(beta_n + beta_d):
                        raise OperatorException("NaN Beta")
                    if self.update_object:
                        self.processing_unit.cl_linear_comb_fcfc(beta, p._cl_obj_dir, np.float32(-1), p._cl_obj_grad)
                    if self.update_probe:
                        self.processing_unit.cl_linear_comb_fcfc(beta, p._cl_probe_dir, np.float32(-1),
                                                                 p._cl_probe_grad)

            # 3) Line minimization
            p._cl_cg_gamma_d, p._cl_cg_gamma_n, p._psi2d_to_obj3d_chi2 = 0, 0, 0

            p = LoopStack(_Psi2ObjProbeGamma(), keep_psi=True) * p

            if self.reg_fac_obj_a + self.reg_fac_obj_c > 0:
                p = _SmoothRegGamma(reg_fac_obj_a=self.reg_fac_obj_a, reg_fac_obj_c=self.reg_fac_obj_c) * p

            if np.isnan(p._cl_cg_gamma_d + p._cl_cg_gamma_n or np.isinf(p._cl_cg_gamma_d + p._cl_cg_gamma_n)):
                raise OperatorException("NaN")

            if np.isnan(p._cl_cg_gamma_d + p._cl_cg_gamma_n) or np.isinf(p._cl_cg_gamma_d + p._cl_cg_gamma_n):
                print("Gamma = NaN ! :", p._cl_cg_gamma_d, p._cl_cg_gamma_n)
            gamma = np.float32(p._cl_cg_gamma_n / p._cl_cg_gamma_d)

            # 4) Object and/or probe and/or background update
            if self.update_object:
                self.processing_unit.cl_linear_comb_fcfc(np.float32(1), p._cl_obj, gamma, p._cl_obj_dir)

            if self.update_probe:
                self.processing_unit.cl_linear_comb_fcfc(np.float32(1), p._cl_probe, gamma, p._cl_probe_dir)

            if self.verbose:
                dt = timeit.default_timer() - t0
                c = p._psi2d_to_obj3d_chi2 / p.support_sum
                r = 0
                if self.reg_fac_obj_c + self.reg_fac_obj_a > 0:
                    p = LoopStack(SmoothRegLLK(reg_fac_obj_a=self.reg_fac_obj_a, reg_fac_obj_c=self.reg_fac_obj_c)) * p
                    r = p.llk_reg / p.support_sum
                print(" %2d: Chi2(Psi2D,Obj3D)= %8.3e, reg=%8.3e (dt=%6.3fs), gamma=%12e" % (ic, c, r, dt, gamma))

        if False:
            # Clean up
            if self.update_object:
                del p._cl_obj_dir, p._cl_obj_grad, p._cl_obj_grad_last
            if self.update_probe:
                del p._cl_probe_dir, p._cl_probe_grad, p._cl_probe_grad_last

        return p


class Psi2ObjIncrement1(CLOperatorBragg2DPtycho):
    """
    Computes updated Obj from Psi and Probe(r-r_j), for a single back-propagated frame with the difference
    between the Fourier-constrained Psi and the original one.

    This should be used only with cl_stack_size = 1
    """

    def __init__(self, beta=0.9):
        """

        :param reg: a regularisation factor to dampen update in areas where the probe amplitude is smaller
                    than reg*max(abs(probe)).
        :param beta: damping parameter (0-1) for the object update
        """
        super(Psi2ObjIncrement1, self).__init__()
        self.beta = np.float32(beta)

    def op(self, p: Bragg2DPtycho):
        pu = self.processing_unit
        if pu.cl_stack_size != 1:
            raise OperatorException("Psi2ObjIncrement1: cl_stack size must be 1")
        i = p._cl_stack_i
        nxp = np.int32(p._probe2d.get().shape[-1])
        nyp = np.int32(p._probe2d.get().shape[-2])
        ny = np.int32(p._cl_psi.shape[-2])
        nx = np.int32(p._cl_psi.shape[-1])
        nb_probe = np.int32(p._cl_psi.shape[1])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(-np.pi / (p.data.wavelength * p.data.detector['distance']))
        pxo = np.float32(p.pxo)
        pyo = np.float32(p.pyo)
        pzo = np.float32(p.pzo)
        pxp = np.float32(p._probe2d.pixel_size)

        probe_norm2_max = pu.cl_norm_max_complex_n_red(p._cl_probe, np.int32(2))

        pu.ev = [pu.cl_psi_to_obj_increment1(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe, p._cl_support, p._cl_m,
                                             p._cl_obs_v[i].x[0], p._cl_obs_v[i].y[0], p._cl_obs_v[i].cixo[0],
                                             p._cl_obs_v[i].ciyo[0], p._cl_obs_v[i].dsx[0],
                                             p._cl_obs_v[i].dsy[0], p._cl_obs_v[i].dsz[0], pxo, pyo, pzo, pxp,
                                             pxp, f, self.processing_unit.cl_stack_size, nx, ny, nxo, nyo, nzo,
                                             nxp, nyp, nb_obj, nb_probe, probe_norm2_max, self.beta, wait_for=pu.ev)]
        return p


class _Psi2ObjDiff(CLOperatorBragg2DPtycho):
    """
    Computes updated Obj from Psi and Probe(r-r_j), for a one stack of back-propagated frames with the difference
    between the Fourier-constrained Psi and the original one.
    """

    def __init__(self, method="diffrepz"):
        """

        :param method: method for object update partition/normalisation, either diffrepz or diffrep1
        """

        super(_Psi2ObjDiff, self).__init__()
        self.method = method

    def op(self, p: Bragg2DPtycho):
        pu = self.processing_unit
        i = p._cl_stack_i
        nxp = np.int32(p._probe2d.get().shape[-1])
        nyp = np.int32(p._probe2d.get().shape[-2])
        ny = np.int32(p._cl_psi.shape[-2])
        nx = np.int32(p._cl_psi.shape[-1])
        nb_probe = np.int32(p._cl_psi.shape[1])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(-np.pi / (p.data.wavelength * p.data.detector['distance']))
        pxo = np.float32(p.pxo)
        pyo = np.float32(p.pyo)
        pzo = np.float32(p.pzo)
        pxp = np.float32(p._probe2d.pixel_size)

        probe_norm2_max = pu.cl_norm_max_complex_n_red(p._cl_probe, np.int32(2))
        if i == 0:
            # TODO: evaluate cost of de/allocating arrays for every cycle
            p._cl_obj_diff = cla.zeros(self.processing_unit.cl_queue, dtype=np.complex64, shape=(nb_obj, nzo, nyo, nxo))
            p._cl_obj_norm = cla.zeros(self.processing_unit.cl_queue, dtype=np.float32, shape=(nzo, nyo, nxo))

        for ii in range(pu.cl_stack_size):
            if "diffrep1" in self.method:
                pu.ev = [pu.cl_psi_to_obj_diff_rep1(p._cl_psi[0, 0, ii], p._cl_obj_diff, p._cl_probe, p._cl_support,
                                                    p._cl_obj_norm, p._cl_m, p._cl_obs_v[i].x[ii],
                                                    p._cl_obs_v[i].y[ii], p._cl_obs_v[i].cixo[ii],
                                                    p._cl_obs_v[i].ciyo[ii], p._cl_obs_v[i].dsx[ii],
                                                    p._cl_obs_v[i].dsy[ii], p._cl_obs_v[i].dsz[ii], pxo, pyo, pzo, pxp,
                                                    pxp, f, self.processing_unit.cl_stack_size, nx, ny, nxo, nyo, nzo,
                                                    nxp, nyp, nb_obj, nb_probe, probe_norm2_max, wait_for=pu.ev)]
            else:
                pu.ev = [pu.cl_psi_to_obj_diff_repz(p._cl_psi[0, 0, ii], p._cl_obj_diff, p._cl_probe, p._cl_support,
                                                    p._cl_obj_norm, p._cl_m, p._cl_obs_v[i].x[ii],
                                                    p._cl_obs_v[i].y[ii], p._cl_obs_v[i].cixo[ii],
                                                    p._cl_obs_v[i].ciyo[ii], p._cl_obs_v[i].dsx[ii],
                                                    p._cl_obs_v[i].dsy[ii], p._cl_obs_v[i].dsz[ii], pxo, pyo, pzo, pxp,
                                                    pxp, f, self.processing_unit.cl_stack_size, nx, ny, nxo, nyo, nzo,
                                                    nxp, nyp, nb_obj, nb_probe, probe_norm2_max, wait_for=pu.ev)]

        return p


class Psi2ObjDiffMerge(CLOperatorBragg2DPtycho):
    """
    Call this when all stack of probe positions have been processed, and the final update of the object can
    be calculated from the object difference. Temporary arrays are cleaned up.
    """

    def __init__(self, reg=1e-2, beta=0.9):
        """

        :param reg: a regularisation factor to dampen update in areas where the probe amplitude is smaller
                    than reg*max(abs(probe)).
        :param beta: damping parameter (0-1) for the object update
        """
        super(Psi2ObjDiffMerge, self).__init__()
        self.reg = np.float32(reg)
        self.beta = np.float32(beta)

    def op(self, p: Bragg2DPtycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        pu = self.processing_unit
        nb_obj = np.int32(p._obj.shape[0])
        nzo, nyo, nxo = p._obj.shape[-3:]
        nxyzo = np.int32(nxo * nyo * nzo)

        normmax = cla.max(p._cl_obj_norm)
        pu.ev = [pu.cl_obj_norm(p._cl_obj_diff, p._cl_obj_norm, p._cl_obj, normmax, self.reg, nxyzo, nb_obj, self.beta,
                                wait_for=pu.ev)]

        # Clean up
        # del p._cl_obj_norm, p._cl_obj_new

        return p


class Psi2ObjDiff(CLOperatorBragg2DPtycho):
    """
    Computes updated Obj from Psi and Probe(r-r_j), for a one stack of back-propagated frames
    """

    def __init__(self, reg=0.01, beta=0.9, method="diffrepz"):
        """

        :param reg: a regularisation factor to dampen update in areas where the probe amplitude is smaller
                    than reg*max(abs(probe)).
        :param beta: damping parameter (0-1) for the object update
        :param method: method for object update partition/normalisation, either diffrepz or diffrep1
        """

        super(Psi2ObjDiff, self).__init__()
        self.reg = reg
        self.beta = np.float32(beta)
        self.method = method

    def op(self, p: Bragg2DPtycho):
        return Psi2ObjDiffMerge(reg=self.reg, beta=self.beta, method=self.method) * LoopStack(_Psi2ObjDiff()) * p


class AP(CLOperatorBragg2DPtycho):
    """
    Perform a complete Alternating Projection cycle:
    - forward all object*probe views to Fourier space and apply the observed amplitude
    - back-project to object space and project onto (probe, object)
    - update background optionally
    """

    def __init__(self, update_object=True, update_probe=False, update_background=False,
                 reg_fac_obj_a=0, reg_fac_obj_c=0, nb_cycle=1, calc_llk=False,
                 show_obj_probe=False, fig_num=-1, update_method='cg', beta=0.9):
        """

        :param update_object: update object ?
        :param update_probe: update probe (TODO)
        :param reg_fac_obj_a: scale for smoothing regularization on the object squared modulus
        :param reg_fac_obj_c: scale for smoothing regularization on the object complex values
        :param nb_cycle: number of cycles to perform. Equivalent to AP(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        :param update_method: update method for the object, either 'cg' (conjugate gradient, the default), or
                              'diffrepz' (or 'diffrep1'), to use the replication of the delta(Psi)
        :param beta: damping factor when using the incremental object update (diffrep)
        """
        super(AP, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.reg_fac_obj_a = reg_fac_obj_a
        self.reg_fac_obj_c = reg_fac_obj_c
        self.nb_cycle = nb_cycle
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.update_method = update_method
        self.beta = beta

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new AP operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return AP(update_object=self.update_object, update_probe=self.update_probe, reg_fac_obj_a=self.reg_fac_obj_a,
                  reg_fac_obj_c=self.reg_fac_obj_c, nb_cycle=self.nb_cycle * n, calc_llk=self.calc_llk,
                  show_obj_probe=self.show_obj_probe, fig_num=self.fig_num, update_method=self.update_method,
                  beta=self.beta)

    def op(self, p: Bragg2DPtycho):
        if 'diff' in self.update_method:
            warnings.warn('bragg2d: AP() using update_method="diff" is experimental, use at your own risk !')

        for ic in range(self.nb_cycle):
            t0 = timeit.default_timer()
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True
            if self.update_background:
                pass  # TODO: update while in Fourier space
            if 'diff' in self.update_method:
                ops = FourierApplyAmplitude(calc_llk=calc_llk, difference=True) * ObjProbe2Psi()
            else:
                ops = FourierApplyAmplitude(calc_llk=calc_llk) * ObjProbe2Psi()

            # Compute the updated Psi
            p = LoopStack(ops, keep_psi=True) * p

            # Update object and probe by 2D->3D projection using the stored psi

            if 'diff' in self.update_method:
                if self.update_object:
                    p = Psi2ObjDiff(reg=5e-2, beta=self.beta, method=self.update_method) * p
            else:
                # This uses a conjugate gradient minimisation of Psi -> (object, probe)
                if False:
                    p = Psi2ObjProbe(update_object=self.update_object, update_probe=self.update_probe,
                                     reg_fac_obj_a=self.reg_fac_obj_a, reg_fac_obj_c=self.reg_fac_obj_c) ** 2 * p
                else:
                    # Do we need to separate object and probe update for stability ?
                    if self.update_object:
                        p = Psi2ObjProbe(update_object=True, update_probe=False,
                                         reg_fac_obj_a=self.reg_fac_obj_a, reg_fac_obj_c=self.reg_fac_obj_c) ** 2 * p
                    if self.update_probe:
                        p = Psi2ObjProbe(update_object=False, update_probe=True) ** 2 * p

            dt = timeit.default_timer() - t0
            if calc_llk:
                p.update_history(mode='llk', dt=dt, algorithm='AP', verbose=True, update_object=self.update_object,
                                 update_probe=self.update_probe)
            else:
                p.update_history(mode='algo', dt=dt, algorithm='AP', verbose=False, update_object=self.update_object,
                                 update_probe=self.update_probe)

            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    tit = "AP"
                    if self.update_object:
                        tit += "/"
                        if len(p._obj) > 1:
                            tit += "%d" % (len(p._obj))
                        tit += "o"
                    if self.update_probe:
                        tit += "/"
                        if len(p._probe) > 1:
                            tit += "%d" % (len(p._probe))
                        tit += "p"
                    tit += " #%3d, LLKn(p)=%8.3f" % (ic, p.llk_poisson / p.nb_obs)
                    p = cpuop.ShowObj(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1

        return p


class DM1(CLOperatorBragg2DPtycho):
    """
    Equivalent to operator: 2 * ObjProbe2Psi() - 1
    """

    def op(self, p):
        i = p._cl_stack_i
        nxp = np.int32(p._probe2d.get().shape[-1])
        nyp = np.int32(p._probe2d.get().shape[-2])
        nx = np.int32(p.data.iobs.shape[-2])
        ny = np.int32(p.data.iobs.shape[-2])
        nb_probe = np.int32(p._cl_psi.shape[1])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        pxo = np.float32(p.pxo)
        pyo = np.float32(p.pyo)
        pzo = np.float32(p.pzo)
        pxp = np.float32(p._probe2d.pixel_size)
        self.processing_unit.cl_2object_probe_psi_dm1(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe, p._cl_support,
                                                      p._cl_m, p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y,
                                                      p._cl_obs_v[i].cl_cixo, p._cl_obs_v[i].cl_ciyo,
                                                      p._cl_obs_v[i].cl_dsx, p._cl_obs_v[i].cl_dsy,
                                                      p._cl_obs_v[i].cl_dsz, pxo, pyo, pzo, pxp, pxp, f,
                                                      p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size, nx, ny,
                                                      nxo, nyo, nzo, nxp, nyp, nb_obj, nb_probe)
        return p


class DM2(CLOperatorBragg2DPtycho):
    """
    # Psi(n+1) = Psi(n) - P*O + Psi_fourier

    This operator assumes that Psi_fourier is the current Psi, and that Psi(n) is in p._cl_psi_v

    On output Psi(n+1) is the current Psi, and Psi_fourier has been swapped to p._cl_psi_v
    """

    # TODO: avoid access to p._cl_psi_v, which is a big kludge
    def op(self, p):
        i = p._cl_stack_i
        nxp = np.int32(p._probe2d.get().shape[-1])
        nyp = np.int32(p._probe2d.get().shape[-2])
        nx = np.int32(p.data.iobs.shape[-2])
        ny = np.int32(p.data.iobs.shape[-2])
        nb_probe = np.int32(p._cl_psi.shape[1])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        pxo = np.float32(p.pxo)
        pyo = np.float32(p.pyo)
        pzo = np.float32(p.pzo)
        pxp = np.float32(p._probe2d.pixel_size)
        # Swap p._cl_psi_v[i] = Psi(n) with p._cl_psi = Psi_fourier
        p._cl_psi_copy, p._cl_psi = p._cl_psi, p._cl_psi_copy
        self.processing_unit.cl_2object_probe_psi_dm2(p._cl_psi[0, 0, 0], p._cl_psi_copy, p._cl_obj, p._cl_probe,
                                                      p._cl_support, p._cl_m, p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y,
                                                      p._cl_obs_v[i].cl_cixo, p._cl_obs_v[i].cl_ciyo,
                                                      p._cl_obs_v[i].cl_dsx, p._cl_obs_v[i].cl_dsy,
                                                      p._cl_obs_v[i].cl_dsz, pxo, pyo, pzo, pxp, pxp, f,
                                                      p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size, nx, ny,
                                                      nxo, nyo, nzo, nxp, nyp, nb_obj, nb_probe)
        return p


class DM(CLOperatorBragg2DPtycho):
    """
    Operator to perform a complete Difference Map cycle, updating the Psi views for all stack of frames,
    as well as updating the object and/or probe.
    """

    def __init__(self, update_object=True, update_probe=False, reg_fac_obj_a=0, reg_fac_obj_c=0,
                 nb_cycle=1, calc_llk=False, show_obj_probe=False, fig_num=-1):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param reg_fac_obj_a: scale for smoothing regularization on the object squared modulus
        :param reg_fac_obj_c: scale for smoothing regularization on the object complex values
        :param nb_cycle: number of cycles to perform. Equivalent to DM(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        """
        super(DM, self).__init__()
        self.nb_cycle = nb_cycle
        self.update_object = update_object
        self.update_probe = update_probe
        self.reg_fac_obj_a = reg_fac_obj_a
        self.reg_fac_obj_c = reg_fac_obj_c
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new DM operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return DM(update_object=self.update_object, update_probe=self.update_probe,
                  reg_fac_obj_a=self.reg_fac_obj_a, reg_fac_obj_c=self.reg_fac_obj_c, nb_cycle=self.nb_cycle * n,
                  calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe, fig_num=self.fig_num)

    def op(self, p: Bragg2DPtycho):
        # Calculate starting Psi (note that all Psi are multiplied by the quadratic phase factor)
        p = LoopStack(ObjProbe2Psi(), keep_psi=True) * p

        # We could use instead of DM1 and DM2 operators:
        # op_dm1 = 2 * ObjProbe2Psi() - 1
        # op_dm2 = 1 - ObjProbe2Psi() + FourierApplyAmplitude() * op_dm1
        # But this would use 3 copies of the whole Psi stack - too much memory ?
        # TODO: check if memory usage would be that bad, or if it's possible the psi storage only applies
        # TODO: to the current psi array

        for ic in range(self.nb_cycle):
            t0 = timeit.default_timer()
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            ops = DM2() * FourierApplyAmplitude() * DM1()

            # TODO: handle the case where there is only 1 stack
            p = LoopStack(ops, keep_psi=True, copy=True) * p

            # This uses a conjugate gradient minimisation of Psi -> (object, probe)
            if self.update_object:
                p = Psi2ObjProbe(update_object=True, update_probe=False,
                                 reg_fac_obj_a=self.reg_fac_obj_a, reg_fac_obj_c=self.reg_fac_obj_c) ** 2 * p
            if self.update_probe:
                p = Psi2ObjProbe(update_object=False, update_probe=True) ** 2 * p

            dt = timeit.default_timer() - t0
            if calc_llk:
                # Keep a copy of current Psi. TODO: use copy in stack instead ?
                cl_psi0 = p._cl_psi.copy()
                cl_stack_i0 = p._cl_stack_i
                # We need to perform a loop for LLK as the DM2 loop is on (2*PO-I), not the current PO estimate
                p = LoopStack(LLK() * FT(scale=False) * ObjProbe2Psi()) * p
                # Restore correct Psi
                p._cl_psi = cl_psi0
                p._cl_stack_i = cl_stack_i0
                p.update_history(mode='llk', dt=dt, algorithm='DM', verbose=True, update_object=self.update_object,
                                 update_probe=self.update_probe)
            else:
                p.update_history(mode='algo', dt=dt, algorithm='DM', verbose=False, update_object=self.update_object,
                                 update_probe=self.update_probe)

            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    tit = "DM"
                    if self.update_object:
                        tit += "/"
                        if len(p._obj) > 1:
                            tit += "%d" % (len(p._obj))
                        tit += "o"
                    if self.update_probe:
                        tit += "/"
                        if len(p._probe) > 1:
                            tit += "%d" % (len(p._probe))
                        tit += "p"
                    tit += " #%3d, LLKn(p)=%8.3f" % (ic, p.llk_poisson / p.nb_obs)
                    p = cpuop.ShowObj(fig_num=self.fig_num, title=tit) * p

            p.cycle += 1
        return p


class _Grad(CLOperatorBragg2DPtycho):
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

    def op(self, p):
        i = p._cl_stack_i
        nxp = np.int32(p._probe2d.get().shape[-1])
        nyp = np.int32(p._probe2d.get().shape[-2])
        nx = np.int32(p.data.iobs.shape[-2])
        ny = np.int32(p.data.iobs.shape[-2])
        nb_probe = np.int32(p._cl_psi.shape[1])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        pxo = np.float32(p.pxo)
        pyo = np.float32(p.pyo)
        pzo = np.float32(p.pzo)
        pxp = np.float32(p._probe2d.pixel_size)
        nxystack = np.int32(self.processing_unit.cl_stack_size * nx * ny)
        nb_mode = np.int32(nb_obj * nb_probe)
        nb_psi = p._cl_obs_v[i].npsi

        # Calculate FT(Obj*Probe)
        p = FT(scale=False) * ObjProbe2Psi() * p
        if self.calc_llk:
            p = LLK() * p

        # Calculate Psi.conj() * (1-Iobs/I_calc) [for Poisson Gradient)
        # TODO: different noise models
        self.processing_unit.cl_grad_poisson_fourier(p._cl_obs_v[i].cl_obs[:nb_psi], p._cl_psi, p._cl_background,
                                                     nb_mode, nx, ny, nxystack)
        p = IFT(scale=False) * p

        if self.update_object:
            # We loop over the stack size to avoid creating an array with N=stack_size object gradient arrays
            for ii in range(p._cl_obs_v[i].npsi):
                self.processing_unit.cl_ml_psi2obj_grad(p._cl_psi[0, 0, ii], p._cl_obj_grad, p._cl_probe, p._cl_support,
                                                        p._cl_m, p._cl_obs_v[i].x[ii], p._cl_obs_v[i].y[ii],
                                                        p._cl_obs_v[i].cixo[ii], p._cl_obs_v[i].ciyo[ii],
                                                        p._cl_obs_v[i].dsx[ii], p._cl_obs_v[i].dsy[ii],
                                                        p._cl_obs_v[i].dsz[ii], pxo, pyo, pzo, pxp, pxp, f,
                                                        self.processing_unit.cl_stack_size, nx, ny, nxo, nyo, nzo,
                                                        nxp, nyp, nb_obj, nb_probe)
        if self.update_probe:
            # We loop over the stack size but this could be rewritten to take into account all frames at the same time
            for ii in range(p._cl_obs_v[i].npsi):
                self.processing_unit.cl_ml_psi2probe_grad(p._cl_psi[0, 0, ii], p._cl_probe_grad, p._cl_obj,
                                                          p._cl_support, p._cl_m, p._cl_obs_v[i].x[ii],
                                                          p._cl_obs_v[i].y[ii], p._cl_obs_v[i].cixo[ii],
                                                          p._cl_obs_v[i].ciyo[ii], p._cl_obs_v[i].dsx[ii],
                                                          p._cl_obs_v[i].dsy[ii], p._cl_obs_v[i].dsz[ii], pxo, pyo, pzo,
                                                          pxp, pxp, f, self.processing_unit.cl_stack_size, nx, ny, nxo,
                                                          nyo, nzo, nxp, nyp, nb_obj, nb_probe)
        if self.update_background:
            # TODO
            pass
        return p


class Grad(CLOperatorBragg2DPtycho):
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

    def op(self, p):
        if self.update_object:
            p._cl_obj_grad.fill(np.complex64(0))

        p = LoopStack(_Grad(update_object=self.update_object, update_probe=self.update_probe,
                            update_background=self.update_background, calc_llk=self.calc_llk)) * p

        # Sum the stack of object arrays
        nb_obj = np.int32(p._obj.shape[0])
        nxyzo = np.int32(p._obj.shape[-3] * p._obj.shape[-2] * p._obj.shape[-1])
        if self.reg_fac_obj != 0:
            # TODO
            # Regularization contribution to the object gradient
            pass
        if self.reg_fac_probe != 0:
            # TODO
            # Regularization contribution to the probe gradient
            pass

        return p


class _CGGamma(CLOperatorBragg2DPtycho):
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
        self.gamma_scale = np.float32(1e-10)

    def op(self, p):
        pu = self.processing_unit
        i = p._cl_stack_i
        nxp = np.int32(p._probe2d.get().shape[-1])
        nyp = np.int32(p._probe2d.get().shape[-2])
        nx = np.int32(p.data.iobs.shape[-2])
        ny = np.int32(p.data.iobs.shape[-2])
        nb_probe = np.int32(p._cl_psi.shape[1])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        pxo = np.float32(p.pxo)
        pyo = np.float32(p.pyo)
        pzo = np.float32(p.pzo)
        pxp = np.float32(p._probe2d.pixel_size)
        nxy = np.int32(nx * ny)
        nxystack = np.int32(self.processing_unit.cl_stack_size * nxy)
        nb_mode = np.int32(nb_obj * nb_probe)
        nb_psi = np.int32(p._cl_obs_v[i].npsi)

        for clpsi, clobj, clprobe in zip([p._cl_PO, p._cl_PdO, p._cl_dPO, p._cl_dPdO],
                                         [p._cl_obj, p._cl_obj_dir, p._cl_obj, p._cl_obj_dir],
                                         [p._cl_probe, p._cl_probe, p._cl_probe_dir, p._cl_probe_dir]):

            self.processing_unit.cl_object_probe_mult(clpsi[0, 0, 0], clobj, clprobe, p._cl_support, p._cl_m,
                                                      p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y, p._cl_obs_v[i].cl_cixo,
                                                      p._cl_obs_v[i].cl_ciyo, p._cl_obs_v[i].cl_dsx,
                                                      p._cl_obs_v[i].cl_dsy,
                                                      p._cl_obs_v[i].cl_dsz, pxo, pyo, pzo, pxp, pxp, f,
                                                      p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size, nx, ny,
                                                      nxo, nyo, nzo, nxp, nyp, nb_obj, nb_probe)

            plan = self.processing_unit.cl_fft_get_plan(clpsi, axes=(-1, -2), shuffle_axes=True)
            for e in plan.enqueue(forward=True):
                e.wait()

        # TODO: take into account background
        cl_bg = cla.zeros(pu.cl_queue, (ny, nx), dtype=np.float32)
        tmp = self.processing_unit.cl_cg_poisson_gamma_red(p._cl_obs_v[i].cl_obs[:nb_psi],
                                                           p._cl_PO, p._cl_PdO, p._cl_dPO, p._cl_dPdO,
                                                           cl_bg, cl_bg, nxy, nxystack, nb_mode, nb_psi).get()
        if np.isnan(tmp['x'] + tmp['y']) or np.isinf(tmp['x'] + tmp['y']):
            nP = self.processing_unit.cl_norm_complex_n(p._cl_probe, 2).get()
            nO = self.processing_unit.cl_norm_complex_n(p._cl_obj, 2).get()
            ndP = self.processing_unit.cl_norm_complex_n(p._cl_probe_dir, 2).get()
            ndO = self.processing_unit.cl_norm_complex_n(p._cl_obj_dir, 2).get()
            nPO = self.processing_unit.cl_norm_complex_n(p._cl_PO, 2).get()
            ndPO = self.processing_unit.cl_norm_complex_n(p._cl_dPO, 2).get()
            nPdO = self.processing_unit.cl_norm_complex_n(p._cl_PdO, 2).get()
            ndPdO = self.processing_unit.cl_norm_complex_n(p._cl_dPdO, 2).get()
            print('_CGGamma norms: P %e O %e dP %e dO %e PO %e, PdO %e, dPO %e, dPdO %e' %
                  (nP, nO, ndP, ndO, nPO, nPdO, ndPO, ndPdO))
            print('_CGGamma (stack #%d, NaN Gamma:)' % i, tmp['x'], tmp['y'])
            raise OperatorException("NaN")
        p._cl_cg_gamma_d += tmp['y']
        p._cl_cg_gamma_n += tmp['x']
        if False:
            tmp = self.processing_unit.cl_cg_poisson_gamma4_red(p._cl_obs_v[i].cl_obs[:nb_psi],
                                                                p._cl_PO, p._cl_PdO, p._cl_dPO, p._cl_dPdO,
                                                                self.gamma_scale, nxy, nxystack, nb_mode).get()
            p._cl_cg_gamma4 += np.array((tmp['w'], tmp['z'], tmp['y'], tmp['x'], 0))
        if self.update_background:
            # TODO: use a different kernel if there is a background gradient
            pass
        return p


class ML(CLOperatorBragg2DPtycho):
    """
    Operator to perform a maximum-likelihood conjugate-gradient minimization.
    """

    def __init__(self, nb_cycle=1, update_object=True, update_probe=False, update_background=False,
                 reg_fac_obj=0, reg_fac_probe=0, calc_llk=False, show_obj_probe=False, fig_num=-1):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param update_background: update background ?
        :param reg_fac_obj: use this regularization factor for the object (if 0, no regularization)
        :param reg_fac_probe: use this regularization factor for the probe (if 0, no regularization)
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
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

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new ML operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return ML(nb_cycle=self.nb_cycle * n, update_object=self.update_object, update_probe=self.update_probe,
                  update_background=self.update_background, reg_fac_obj=self.reg_fac_obj,
                  reg_fac_probe=self.reg_fac_probe, calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe,
                  fig_num=self.fig_num)

    def op(self, p: Bragg2DPtycho):
        nz = np.int32(p._cl_psi.shape[-3])
        ny = np.int32(p._cl_psi.shape[-2])
        nx = np.int32(p._cl_psi.shape[-1])
        nb_probe = np.int32(p._cl_psi.shape[1])
        nb_obj = np.int32(p._cl_psi.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        cl_queue = self.processing_unit.cl_queue
        stack_size = self.processing_unit.cl_stack_size

        # Create the necessary GPU arrays for ML
        p._cl_PO = cla.empty_like(p._cl_psi)
        p._cl_PdO = cla.empty_like(p._cl_psi)
        p._cl_dPO = cla.empty_like(p._cl_psi)
        p._cl_dPdO = cla.empty_like(p._cl_psi)
        p._cl_obj_dir = cla.zeros(cl_queue, (nb_obj, nzo, nyo, nxo), np.complex64)
        p._cl_probe_dir = cla.zeros(cl_queue, (nb_probe, nz, ny, nx), np.complex64)
        if self.update_object:
            p._cl_obj_grad = cla.empty(cl_queue, (nb_obj, nzo, nyo, nxo), np.complex64)
            p._cl_obj_grad_last = cla.empty(cl_queue, (nb_obj, nzo, nyo, nxo), np.complex64)
        if self.update_probe:
            p._cl_probe_grad = cla.empty(cl_queue, (nb_probe, nz, ny, nx), np.complex64)
            p._cl_probe_grad_last = cla.empty(cl_queue, (nb_probe, nz, ny, nx), np.complex64)
        if self.update_background:
            p._cl_background_grad = cla.zeros(cl_queue, (ny, nx), np.float32)
            p._cl_background_grad_last = cla.zeros(cl_queue, (ny, nx), np.float32)
            p._cl_background_dir = cla.zeros(cl_queue, (ny, nx), np.float32)
        for ic in range(self.nb_cycle):
            t0 = timeit.default_timer()
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

            # 1) Compute the gradients
            p = Grad(update_object=self.update_object, update_probe=self.update_probe,
                     update_background=self.update_background, calc_llk=calc_llk) * p

            # 2) Search direction
            if ic == 0:
                # first cycle
                if self.update_object:
                    cl.enqueue_copy(cl_queue, src=p._cl_obj_grad.data, dest=p._cl_obj_dir.data)
                if self.update_probe:
                    cl.enqueue_copy(cl_queue, src=p._cl_probe_grad.data, dest=p._cl_probe_dir.data)
                if self.update_background:
                    cl.enqueue_copy(cl_queue, src=p._cl_background_grad.data, dest=p._cl_background_dir.data)
            else:
                beta_d, beta_n = 0, 0
                # Polak-Ribière CG coefficient
                cg_pr = self.processing_unit.cl_cg_polak_ribiere_complex_red
                if self.update_object:
                    tmp = cg_pr(p._cl_obj_grad, p._cl_obj_grad_last).get()
                    beta_n += tmp['x']
                    beta_d += tmp['y']
                if self.update_probe:
                    tmp = cg_pr(p._cl_probe_grad, p._cl_probe_grad_last).get()
                    beta_n += tmp['x']
                    beta_d += tmp['y']
                if self.update_background:
                    tmp = cg_pr(p._cl_background_grad, p._cl_background_grad_last).get()
                    beta_n += tmp['x']
                    beta_d += tmp['y']
                # print("Beta= %e / %e"%(beta_n, beta_d))
                # Reset direction if beta<0 => beta=0
                beta = np.float32(max(0, beta_n / max(1e-20, beta_d)))
                if np.isnan(beta_n + beta_d) or np.isinf(beta_n + beta_d):
                    raise OperatorException("NaN")
                if self.update_object:
                    self.processing_unit.cl_linear_comb_fcfc(beta, p._cl_obj_dir, np.float32(-1), p._cl_obj_grad)
                if self.update_probe:
                    self.processing_unit.cl_linear_comb_fcfc(beta, p._cl_probe_dir, np.float32(-1), p._cl_probe_grad)
                if self.update_background:
                    self.processing_unit.cl_linear_comb_4f(beta, p._cl_background_dir,
                                                           np.float32(-1), p._cl_background_grad)
            # 3) Line minimization
            p._cl_cg_gamma_d, p._cl_cg_gamma_n = 0, 0
            if False:
                # We could use a 4th order LLK(gamma) approximation, but it does not seem to improve
                p._cl_cg_gamma4 = np.zeros(5, dtype=np.float32)
            p = LoopStack(_CGGamma(update_background=self.update_background)) * p
            if np.isnan(p._cl_cg_gamma_d + p._cl_cg_gamma_n or np.isinf(p._cl_cg_gamma_d + p._cl_cg_gamma_n)):
                raise OperatorException("NaN")

            if self.reg_fac_obj != 0:
                pass
                # TODO
            if self.reg_fac_probe != 0:
                # TODO
                pass

            if np.isnan(p._cl_cg_gamma_d + p._cl_cg_gamma_n) or np.isinf(p._cl_cg_gamma_d + p._cl_cg_gamma_n):
                print("Gamma = NaN ! :", p._cl_cg_gamma_d, p._cl_cg_gamma_n)
            gamma = np.float32(p._cl_cg_gamma_n / p._cl_cg_gamma_d)
            if False:
                # It seems the 2nd order gamma approximation is good enough.
                gr = np.roots(p._cl_cg_gamma4)
                print("CG Gamma4", p._cl_cg_gamma4, "\n", gr, np.polyval(p._cl_cg_gamma4, gr))
                print("CG Gamma2=", gamma, "=", p._cl_cg_gamma_n, "/", p._cl_cg_gamma_d)

            # 4) Object and/or probe and/or background update
            if self.update_object:
                self.processing_unit.cl_linear_comb_fcfc(np.float32(1), p._cl_obj, gamma, p._cl_obj_dir)

            if self.update_probe:
                self.processing_unit.cl_linear_comb_fcfc(np.float32(1), p._cl_probe, gamma, p._cl_probe_dir)

            if self.update_background:
                self.processing_unit.cl_linear_comb_4f(np.float32(1), p._cl_background, gamma, p._cl_background_dir)
            dt = timeit.default_timer() - t0
            if calc_llk:
                p.update_history(mode='llk', dt=dt, algorithm='ML', verbose=True, update_object=self.update_object,
                                 update_probe=self.update_probe)
            else:
                p.update_history(mode='algo', dt=dt, algorithm='ML', verbose=False, update_object=self.update_object,
                                 update_probe=self.update_probe)
            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    tit = "ML"
                    if self.update_object:
                        tit += "/"
                        if len(p._obj) > 1:
                            tit += "%d" % (len(p._obj))
                        tit += "o"
                    if self.update_probe:
                        tit += "/"
                        if len(p._probe) > 1:
                            tit += "%d" % (len(p._probe))
                        tit += "p"
                    tit += " #%3d, LLKn(p)=%8.3f" % (ic, p.llk_poisson / p.nb_obs)
                    p = cpuop.ShowObj(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1

        # Clean up
        del p._cl_PO, p._cl_PdO, p._cl_dPO, p._cl_dPdO, p._cl_obj_dir, p._cl_probe_dir
        if self.update_object:
            del p._cl_obj_grad, p._cl_obj_grad_last
        if self.update_probe:
            del p._cl_probe_grad, p._cl_probe_grad_last
        if self.update_background:
            del p._cl_background_grad, p._cl_background_grad_last, p._cl_background_dir

        return p


class ScaleObjProbe(CLOperatorBragg2DPtycho):
    """
    Operator to scale the object and probe so that they have the same magnitude, and that the product of object*probe
    matches the observed intensity (i.e. sum(abs(obj*probe)**2) = sum(iobs))
    """

    def op(self, p):
        pu = self.processing_unit
        if True:
            # Compute the best scale factor
            snum, sden = 0, 0
            nxy = np.int32(p._cl_psi.shape[-1] * p._cl_psi.shape[-2])
            nxystack = np.int32(nxy * self.processing_unit.cl_stack_size)
            nb_mode = np.int32(p._cl_psi.shape[0] * p._cl_psi.shape[1])
            for i in range(p._cl_stack_nb):
                p = FT(scale=False) * ObjProbe2Psi() * SelectStack(i) * p
                nb_psi = p._cl_obs_v[i].npsi
                bg = cla.zeros(pu.cl_queue, nxy, dtype=np.float32)
                r = pu.cl_scale_intensity(p._cl_obs_v[i].cl_obs[:nb_psi], p._cl_psi, bg, nxy, nxystack, nb_mode,
                                          wait_for=pu.ev).get()
                snum += r['x']
                sden += r['y']
            print("%f / %f" % (snum, sden))
            s = np.sqrt(snum / sden)
        else:
            nb_photons_obs = p.data.iobs_sum
            nb_photons_calc = 0
            for i in range(p._cl_stack_nb):
                p = ObjProbe2Psi() * SelectStack(i) * p
                nb_photons_calc += self.processing_unit.cl_norm_complex_n(p._cl_psi, 2, wait_for=pu.ev).get()
                pu.ev = []
            if p.data.near_field:
                s = np.sqrt(nb_photons_obs / nb_photons_calc)
            else:
                s = np.sqrt(nb_photons_obs / nb_photons_calc) / np.sqrt(p._cl_obj.size)
        # TODO: take into account only the scanned part of the object for obj/probe relative scaling
        os = self.processing_unit.cl_norm_complex_n(p._cl_obj, np.int32(1)).get()
        ps = self.processing_unit.cl_norm_complex_n(p._cl_probe, np.int32(1)).get()
        self.processing_unit.cl_scale(p._cl_probe, np.float32(np.sqrt(os / ps * s)))
        self.processing_unit.cl_scale(p._cl_obj, np.float32(np.sqrt(ps / os * s)))
        print("ScaleObjProbe:", ps, os, s, np.sqrt(os / ps * s), np.sqrt(ps / os * s))
        if False:
            # Check the scale factor
            snum, sden = 0, 0
            nxystack = np.int32(p._cl_psi.shape[-1] * p._cl_psi.shape[-2] * self.processing_unit.cl_stack_size)
            nb_mode = np.int32(p._cl_psi.shape[0] * p._cl_psi.shape[1])
            for i in range(p._cl_stack_nb):
                p = FT(scale=False) * ObjProbe2Psi() * SelectStack(i) * p
                r = pu.cl_scale_intensity(p._cl_obs_v[i].cl_obs[:nb_psi], p._cl_psi, nxystack, nb_mode,
                                          wait_for=pu.ev).get()
                snum += r['x']
                sden += r['y']
            s = snum / sden
            print("ScaleObjProbe: now s=", s)
        return p


class InitSupport(CLOperatorBragg2DPtycho):
    """
    Operator to compute the object and/or probe and/or background gradient. The gradient is stored
    in the ptycho object. It is assumed that the GPU gradient arrays have been already created, normally
    by the calling ML operator.
    """

    def __init__(self, equation, rotation_axes=None, shrink_object_around_support=False):
        """

        :param equation: an equation in x,y,z (orthonormal coordinates in the laboratory frame)
        :param rotation_axes: e.g. (('x', 0), ('y', pi/4)): optionally, the coordinates can be evaluated after a
                              rotation of the object. This is useful if the support is to be defined before being
                              rotated to be in diffraction condition. The rotation can be given as a tuple of a
                              rotation axis name (x, y or z) and a counter-clockwise rotation angle in radians.
        """
        super(InitSupport, self).__init__()
        self.rotation_axes = rotation_axes
        kernel_src = getks('opencl/random_lcg.cl') + \
                     getks('ptycho/bragg2d/opencl/init_support.cl').replace('EQUATION_INSIDE_SUPPORT', equation)
        self.cl_init_support = CL_ElK(self.processing_unit.cl_ctx, name='cl_init_support',
                                      operation="InitSupportEq(i, support, m, nxo, nyo, nzo, ix0, iy0, iz0)",
                                      preamble=kernel_src,
                                      options=self.processing_unit.cl_options,
                                      arguments="__global char *support, __global float* m, const int nxo,"
                                                "const int nyo, const int nzo, const int ix0, const int iy0,"
                                                "const int iz0")
        self.shrink_object_around_support = shrink_object_around_support

    def op(self, p: Bragg2DPtycho):
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        m = p.get_orthonormalisation_matrix(rotation_axes=self.rotation_axes)[0]
        cl_m = cla.to_device(self.processing_unit.cl_queue, m.astype(np.float32))

        # Update layer-by-layer to avoid to long kernel executions if GPU is not dedicated
        for iz0 in range(nzo):
            self.cl_init_support(p._cl_support[iz0], cl_m, nxo, nyo, nzo, np.int32(0), np.int32(0), np.int32(iz0))
        p.support = p._cl_support.get()
        if self.shrink_object_around_support:
            p.set_support(p.support, shrink_object_around_support=True)
        return p


class SelectStack(CLOperatorBragg2DPtycho):
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

    def op(self, p: Bragg2DPtycho):
        """

        :param p: the Ptycho object this operator applies to
        :return: the updated Ptycho object
        """
        if self.stack_i == p._cl_stack_i:
            if self.keep_psi and self.stack_i in p._cl_psi_v:
                # This can happen if we use LoopStack(keep_psi=False) between LoopStack(keep_psi=True)
                p._cl_psi = p._cl_psi_v.pop(self.stack_i)
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


class PurgeStacks(CLOperatorBragg2DPtycho):
    """
    Operator to delete stored psi stacks in a Ptycho object's _cl_psi_v.

    This should be called for each main operator using LoopStack(), once it is finished processing, in order to avoid
    having another operator using the stored stacks, and to free memory.
    """

    def op(self, p: Bragg2DPtycho):
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


class LoopStack(CLOperatorBragg2DPtycho):
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

    def op(self, p: Bragg2DPtycho):
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
