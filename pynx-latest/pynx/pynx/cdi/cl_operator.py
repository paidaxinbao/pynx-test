# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['default_processing_unit', 'AutoCorrelationSupport', 'FreePU', 'FT', 'IFT', 'FourierApplyAmplitude', 'ER',
           'CF', 'HIO', 'RAAR', 'GPS', 'ML', 'SupportUpdate', 'ScaleObj', 'LLK', 'LLKSupport', 'DetwinHIO',
           'DetwinRAAR', 'SupportExpand', 'ObjConvolve', 'ShowCDI', 'EstimatePSF', 'ApplyAmplitude', 'InterpIobsMask',
           'InitPSF', 'PRTF', 'UpdatePSF']

import warnings
import types
import gc
from random import randint
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.ndimage.measurements import center_of_mass

from ..processing_unit.cl_processing_unit import CLProcessingUnit
from ..processing_unit.kernel_source import get_kernel_source as getks
from ..processing_unit import default_processing_unit as main_default_processing_unit
import pyopencl as cl
import pyopencl.array as cla
from pyopencl.elementwise import ElementwiseKernel as CL_ElK
from pyopencl.reduction import ReductionKernel as CL_RedK
import pyopencl.tools as cltools

from ..operator import has_attr_not_none, OperatorException, OperatorSum, OperatorPower

from .cdi import OperatorCDI, CDI, SupportTooSmall, SupportTooLarge
from .cpu_operator import ShowCDI as ShowCDICPU
from ..utils.phase_retrieval_transfer_function import plot_prtf
from .selection import match_shape


################################################################################################
# Patch CDI class so that we can use 5*w to scale it.
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


patch_method(CDI)


################################################################################################


class CLProcessingUnitCDI(CLProcessingUnit):
    """
    Processing unit in OpenCL space, for 2D and 3D CDI operations.

    Handles initializing the context and kernels.
    """

    def __init__(self):
        super(CLProcessingUnitCDI, self).__init__()
        self.cl_mem_pool = None

    def cl_init_kernels(self):
        # Elementwise kernels
        self.cl_scale = CL_ElK(self.cl_ctx, name='cl_scale',
                               operation="d[i] = (float2)(d[i].x * scale, d[i].y * scale )",
                               options=self.cl_options, arguments="__global float2 *d, const float scale")

        self.cl_sum = CL_ElK(self.cl_ctx, name='cl_sum',
                             operation="dest[i] += src[i]",
                             options=self.cl_options, arguments="__global float2 *src, __global float2 *dest")

        self.cl_mult = CL_ElK(self.cl_ctx, name='cl_mult',
                              operation="dest[i] = (float2)(dest[i].x * src[i].x - dest[i].y * src[i].y, dest[i].x * src[i].y + dest[i].y * src[i].x)",
                              options=self.cl_options, arguments="__global float2 *src, __global float2 *dest")

        self.cl_mult_scale_complex = \
            CL_ElK(self.cl_ctx, name='cl_mult',
                   operation="dest[i] = (float2)(dest[i].x * src[i].x - dest[i].y * src[i].y,"
                             "           dest[i].x * src[i].y + dest[i].y * src[i].x) * scale",
                   options=self.cl_options, arguments="__global float2 *src, __global float2 *dest, const float scale")

        self.cl_mult_real = CL_ElK(self.cl_ctx, name='cl_mult',
                                   operation="dest[i] = (float2)(dest[i].x * src[i].x - dest[i].y * src[i].y, 0)",
                                   options=self.cl_options,
                                   arguments="__global float2 *src, __global float2 *dest")

        # # FT of the mirror: FT(a[::-1, ::-1]) == FT().conj()*np.exp(2j*np.pi*(x/nx+y/ny)) ??
        # self.cl_mult_mirror = CL_ElK(self.cl_ctx, name='cl_mult_mirror',
        #                              operation="const int ix = i%nx;"
        #                                        "const int iz = i/(nx*ny);"
        #                                        "const int iy = (i-nx*ny*iz)/nx;"
        #                                        "const float tmp = 6.283185307179586 * (ix/(float)(2*(nx-1))+iy/(float)ny+iz/(float)nz);"
        #                                        "const float s=native_sin(tmp);"
        #                                        "const float c=native_cos(tmp);"
        #                                        "const float2 a=(float2)(src[i].x * c + src[i].y * s, src[i].x * s - src[i].y * c);"
        #                                        "dest[i] = (float2)(dest[i].x * a.x - dest[i].y * a.y, dest[i].x * a.y + dest[i].y * a.x);",
        #                              options=self.cl_options,
        #                              arguments="__global float2 *src, __global float2 *dest, const int nx, const int ny, const int nz")
        #
        self.cl_div_float = CL_ElK(self.cl_ctx, name='cl_div_real',
                                   operation="dest[i] = src[i] / fmax(dest[i],1e-8f)",
                                   options=self.cl_options,
                                   arguments="__global float *src, __global float *dest")

        self.cl_scale_complex = CL_ElK(self.cl_ctx, name='cl_scale_complex',
                                       operation="d[i] = (float2)(d[i].x * s.x - d[i].y * s.y, d[i].x * s.y + d[i].y * s.x)",
                                       options=self.cl_options, arguments="__global float2 *d, const float2 s")

        self.cl_square_modulus = CL_ElK(self.cl_ctx, name='cl_square_modulus',
                                        operation="dest[i] = dot(src[i],src[i])",
                                        options=self.cl_options,
                                        arguments="__global float *dest, __global float2 *src")

        self.cl_autocorrel_iobs = CL_ElK(self.cl_ctx, name='cl_autocorrel_iobs',
                                         operation="iobsc[i] = (float2)(iobs[i]>=0 ? iobs[i]: 0, 0);",
                                         options=self.cl_options,
                                         arguments="__global float2 *iobsc, __global float *iobs")

        self.cl_apply_amplitude = CL_ElK(self.cl_ctx, name='cl_apply_amplitude',
                                         operation="ApplyAmplitude(i, iobs, dcalc, scale_in, scale_out, zero_mask,"
                                                   "confidence_interval_factor, confidence_interval_factor_mask_min,"
                                                   "confidence_interval_factor_mask_max)",
                                         preamble=getks('cdi/opencl/apply_amplitude_elw.cl'),
                                         options=self.cl_options,
                                         arguments="__global float *iobs, __global float2 *dcalc, const float scale_in,"
                                                   "const float scale_out, const char zero_mask,"
                                                   "const float confidence_interval_factor,"
                                                   "const float confidence_interval_factor_mask_min,"
                                                   "const float confidence_interval_factor_mask_max")

        self.cl_apply_amplitude_icalc = CL_ElK(self.cl_ctx, name='cl_apply_amplitude_icalc',
                                               operation="ApplyAmplitudeIcalc(i, iobs, dcalc, icalc, scale_in,"
                                                         "scale_out, zero_mask, confidence_interval_factor,"
                                                         "confidence_interval_factor_mask_min,"
                                                         "confidence_interval_factor_mask_max)",
                                               preamble=getks('cdi/opencl/apply_amplitude_elw.cl'),
                                               options=self.cl_options,
                                               arguments="__global float *iobs, __global float2 *dcalc,"
                                                         "__global float *icalc, const float scale_in,"
                                                         "const float scale_out, const char zero_mask,"
                                                         "const float confidence_interval_factor,"
                                                         "const float confidence_interval_factor_mask_min,"
                                                         "const float confidence_interval_factor_mask_max")

        self.cl_er = CL_ElK(self.cl_ctx, name='cl_er',
                            operation="ER(i, obj, support)",
                            preamble=getks('cdi/opencl/cdi_elw.cl'),
                            options=self.cl_options,
                            arguments="__global float2 *obj, __global char *support")

        self.cl_er_real = CL_ElK(self.cl_ctx, name='cl_er',
                                 operation="ER_real_pos(i, obj, support)",
                                 preamble=getks('cdi/opencl/cdi_elw.cl'),
                                 options=self.cl_options,
                                 arguments="__global float2 *obj, __global char *support")

        self.cl_hio = CL_ElK(self.cl_ctx, name='cl_hio',
                             operation="HIO(i, obj, obj_previous, support, beta)",
                             preamble=getks('cdi/opencl/cdi_elw.cl'),
                             options=self.cl_options,
                             arguments="__global float2 *obj, __global float2 *obj_previous, __global char *support, float beta")

        self.cl_hio_real = CL_ElK(self.cl_ctx, name='cl_hio_real',
                                  operation="HIO_real_pos(i, obj, obj_previous, support, beta)",
                                  preamble=getks('cdi/opencl/cdi_elw.cl'),
                                  options=self.cl_options,
                                  arguments="__global float2 *obj, __global float2 *obj_previous, __global char *support, float beta")

        self.cl_cf = CL_ElK(self.cl_ctx, name='cl_cf',
                            operation="CF(i, obj, support)",
                            preamble=getks('cdi/opencl/cdi_elw.cl'),
                            options=self.cl_options,
                            arguments="__global float2 *obj, __global char *support")

        self.cl_cf_real = CL_ElK(self.cl_ctx, name='cl_cf_real',
                                 operation="CF_real_pos(i, obj, support)",
                                 preamble=getks('cdi/opencl/cdi_elw.cl'),
                                 options=self.cl_options,
                                 arguments="__global float2 *obj, __global char *support")

        self.cl_raar = CL_ElK(self.cl_ctx, name='cl_raar',
                              operation="RAAR(i, obj, obj_previous, support, beta)",
                              preamble=getks('cdi/opencl/cdi_elw.cl'),
                              options=self.cl_options,
                              arguments="__global float2 *obj, __global float2 *obj_previous, __global char *support, float beta")

        self.cl_raar_real = CL_ElK(self.cl_ctx, name='cl_raar_real',
                                   operation="RAAR_real_pos(i, obj, obj_previous, support, beta)",
                                   preamble=getks('cdi/opencl/cdi_elw.cl'),
                                   options=self.cl_options,
                                   arguments="__global float2 *obj, __global float2 *obj_previous, __global char *support, float beta")

        # self.cl_dm1 = CL_ElK(self.cl_ctx, name='cl_dm1',
        #                      operation="DM1(i, obj, obj_previous, support)",
        #                      preamble=getks('cdi/opencl/cdi_elw.cl'), options=self.cl_options,
        #                      arguments="__global float2 *obj, __global float2 *obj_previous, __global char *support")
        #
        # self.cl_dm1_real = CL_ElK(self.cl_ctx, name='cl_dm1_real',
        #                           operation="DM1_real_pos(i, obj, obj_previous, support)",
        #                           preamble=getks('cdi/opencl/cdi_elw.cl'), options=self.cl_options,
        #                           arguments="__global float2 *obj, __global float2 *obj_previous,"
        #                                     "__global char *support")
        #
        # self.cl_dm2 = CL_ElK(self.cl_ctx, name='cl_dm1',
        #                      operation="DM2(i, obj, obj_previous, support)",
        #                      preamble=getks('cdi/opencl/cdi_elw.cl'), options=self.cl_options,
        #                      arguments="__global float2 *obj, __global float2 *obj_previous, __global char *support")
        #
        # self.cl_dm2_real = CL_ElK(self.cl_ctx, name='cl_dm1_real',
        #                           operation="DM2_real_pos(i, obj, obj_previous, support)",
        #                           preamble=getks('cdi/opencl/cdi_elw.cl'), options=self.cl_options,
        #                           arguments="__global float2 *obj, __global float2 *obj_previous,"
        #                                     "__global char *support")

        self.cl_ml_poisson_psi_gradient = CL_ElK(self.cl_ctx, name='cl_ml_poisson_psi_gradient',
                                                 operation="PsiGradient(i, psi, dpsi, iobs, nx, ny, nz)",
                                                 preamble=getks('cdi/opencl/cdi_ml_poisson_elw.cl'),
                                                 options=self.cl_options,
                                                 arguments="__global float2* psi, __global float2* dpsi,"
                                                           "__global float* iobs, const int nx, const int ny,"
                                                           "const int nz")

        self.cl_ml_poisson_reg_support_gradient = CL_ElK(self.cl_ctx, name='cl_ml_poisson_psi_gradient',
                                                         operation="RegSupportGradient(i, obj, objgrad, support, reg_fac)",
                                                         preamble=getks('cdi/opencl/cdi_ml_poisson_elw.cl'),
                                                         options=self.cl_options,
                                                         arguments="__global float2* obj, __global float2* objgrad, __global char* support, const float reg_fac")

        self.cl_ml_poisson_cg_linear = CL_ElK(self.cl_ctx, name='cl_ml_poisson_psi_gradient',
                                              operation="CG_linear(i, a, A, b, B)",
                                              preamble=getks('cdi/opencl/cdi_ml_poisson_elw.cl'),
                                              options=self.cl_options,
                                              arguments="const float a, __global float2 *A, const float b, __global float2 *B")

        self.cl_gps1 = CL_ElK(self.cl_ctx, name='cl_gps1',
                              operation="GPS1(i, obj, z, t, sigma_o, nx, ny, nz)",
                              preamble=getks('cdi/opencl/gps_elw.cl'),
                              options=self.cl_options,
                              arguments="__global float2* obj, __global float2* z, const float t, const float sigma_o,"
                                        "const int nx, const int ny, const int nz")

        self.cl_gps2 = CL_ElK(self.cl_ctx, name='cl_gps2',
                              operation="GPS2(i, obj, z, epsilon)",
                              preamble=getks('cdi/opencl/gps_elw.cl'),
                              options=self.cl_options,
                              arguments="__global float2* obj, __global float2* z, const float epsilon")

        self.cl_gps3 = CL_ElK(self.cl_ctx, name='cl_gps3',
                              operation="GPS3(i, obj, z)",
                              preamble=getks('cdi/opencl/gps_elw.cl'),
                              options=self.cl_options,
                              arguments="__global float2* obj, __global float2* z")

        self.cl_gps4 = CL_ElK(self.cl_ctx, name='cl_gps4',
                              operation="GPS4(i, obj, y, support, s, sigma_f, positivity, nx, ny, nz)",
                              preamble=getks('cdi/opencl/gps_elw.cl'),
                              options=self.cl_options,
                              arguments="__global float2* obj, __global float2* y, __global char *support,"
                                        "const float s, const float sigma_f, char positivity,"
                                        "const int nx, const int ny, const int nz")

        self.cl_mask_interp_dist = CL_ElK(self.cl_ctx, name='cl_mask_interp_dist',
                                          operation="mask_interp_dist(i, iobs, k, dist_n, nx, ny, nz)",
                                          preamble=getks('opencl/mask_interp_dist.cl'),
                                          options=self.cl_options,
                                          arguments="__global float *iobs, const int k, const int dist_n,"
                                                    "const int nx, const int ny, const int nz")

        self.cl_gaussian = CL_ElK(self.cl_ctx, name='cl_gaussian',
                                  operation="Gaussian(i, d, fwhm, nx, ny, nz)",
                                  preamble=getks('cdi/opencl/psf_elw.cl'),
                                  options=self.cl_options,
                                  arguments="__global float *d, const float fwhm,"
                                            "const int nx, const int ny, const int nz")

        self.cl_lorentzian = CL_ElK(self.cl_ctx, name='cl_lorentzian',
                                    operation="Lorentzian(i, d, fwhm, nx, ny, nz)",
                                    preamble=getks('cdi/opencl/psf_elw.cl'),
                                    options=self.cl_options,
                                    arguments="__global float *d, const float fwhm,"
                                              "const int nx, const int ny, const int nz")

        self.cl_pseudovoigt = CL_ElK(self.cl_ctx, name='cl_pseudovoigt',
                                     operation="PseudoVoigt(i, d, fwhm, eta, nx, ny, nz)",
                                     preamble=getks('cdi/opencl/psf_elw.cl'),
                                     options=self.cl_options,
                                     arguments="__global float *d, const float fwhm, const float eta,"
                                               "const int nx, const int ny, const int nz")

        self.cl_psf1 = CL_ElK(self.cl_ctx, name='cl_psf1',
                              operation="icalc[i] = iobs[i] >=0 ? iobs[i] / (icalc[i] * scale) : 1.0f",
                              # TODO: robustness ?
                              options=self.cl_options,
                              arguments="__global float *iobs, __global float *icalc, const float scale")

        # for half-hermition arrays: d1 * mirror(d2) = d1 * d2.conj()
        self.cl_psf2 = CL_ElK(self.cl_ctx, name='cl_psf2',
                              operation="d1[i] = (float2)(d1[i].x * d2[i].x + d1[i].y * d2[i].y,"
                                        "                 d1[i].y * d2[i].x - d1[i].x * d2[i].y) * scale",
                              options=self.cl_options,
                              arguments="__global float2* d1, __global float2 *d2, const float scale")

        self.cl_psf3 = CL_ElK(self.cl_ctx, name='cl_psf3',
                              operation="psf[i] *= fmax(d[i], 1e-6f) * scale",  # TODO: robustness ?
                              options=self.cl_options,
                              arguments="__global float *psf, __global float *d, const float scale")

        self.cl_psf3_hann = CL_ElK(self.cl_ctx, name='cl_psf3_hann',
                                   operation="PSF3_Hann(i, psf, d, nx, ny, nz, scale)",
                                   preamble=getks('cdi/opencl/psf_elw.cl'),
                                   options=self.cl_options,
                                   arguments="__global float *psf, __global float *d,"
                                             "const int nx, const int ny, const int nz, const float scale")

        self.cl_psf3_tukey = CL_ElK(self.cl_ctx, name='cl_psf3_tukey',
                                    operation="PSF3_Tukey(i, psf, d, alpha, nx, ny, nz, scale)",
                                    preamble=getks('cdi/opencl/psf_elw.cl'),
                                    options=self.cl_options,
                                    arguments="__global float *psf, __global float *d, const float alpha,"
                                              "const int nx, const int ny, const int nz, const float scale")

        self.cl_psf4 = CL_ElK(self.cl_ctx, name='cl_psf4',
                              operation="psf[i] /= psf_sum[0]",
                              options=self.cl_options,
                              arguments="__global float *psf, __global float *psf_sum")

        # Reduction kernels

        # We need this reduction because cla.sum will overflow over the char support array
        self.cl_nb_point_support = CL_RedK(self.cl_ctx, np.int32, neutral="0", reduce_expr="a+b",
                                           options=self.cl_options, map_expr="support[i]",
                                           arguments="__global char* support")

        # This will compute Poisson, Gaussian, Euclidian LLK as well as the sum of the calculated intensity
        self.cl_llk_red = CL_RedK(self.cl_ctx, cla.vec.float8, neutral="(float8)(0,0,0,0,0,0,0,0)", reduce_expr="a+b",
                                  preamble=getks('cdi/opencl/llk_red.cl'),
                                  options=self.cl_options,
                                  map_expr="LLKAll(i, iobs, psi, scale)",
                                  arguments="__global float *iobs, __global float2 *psi, const float scale")

        self.cl_llk_icalc_red = CL_RedK(self.cl_ctx, cla.vec.float8, neutral="(float8)(0,0,0,0,0,0,0,0)",
                                        reduce_expr="a+b",
                                        preamble=getks('cdi/opencl/llk_red.cl'),
                                        options=self.cl_options,
                                        map_expr="LLKAllIcalc(i, iobs, icalc, scale)",
                                        arguments="__global float *iobs, __global float *icalc, const float scale")

        self.cl_llk_reg_support_red = CL_RedK(self.cl_ctx, np.float32, neutral="0", reduce_expr="a+b",
                                              map_expr="LLKRegSupport(obj[i], support[i])",
                                              preamble=getks('cdi/opencl/cdi_llk_reg_support_red.cl'),
                                              options=self.cl_options,
                                              arguments="__global float2 *obj, __global char *support")

        # Polak-Ribière CG coefficient
        self.cl_cg_polak_ribiere_red = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                               reduce_expr="a+b",
                                               map_expr="PolakRibiereComplex(grad[i], lastgrad[i])",
                                               preamble=getks('opencl/cg_polak_ribiere_red.cl'),
                                               options=self.cl_options,
                                               arguments="__global float2 *grad, __global float2 *lastgrad")
        # Line minimization factor for CG
        self.cdi_ml_poisson_gamma_red = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                                reduce_expr="a+b",
                                                map_expr="Gamma(obs, psi, dpsi, i)",
                                                preamble=getks('cdi/opencl/cdi_ml_poisson_red.cl'),
                                                options=self.cl_options,
                                                arguments="__global float *obs, __global float2 *psi, __global float2 *dpsi")

        self.cdi_ml_poisson_gamma_support_red = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                                        reduce_expr="a+b",
                                                        map_expr="GammaSupport(obs, psi, dpsi, obj, dobj, support, reg_fac, i)",
                                                        preamble=getks('cdi/opencl/cdi_ml_poisson_red.cl'),
                                                        options=self.cl_options,
                                                        arguments="__global float *obs, __global float2 *psi, __global float2 *dpsi,"
                                                                  "__global float2 *obj, __global float2 *dobj, __global char *support, "
                                                                  "const float reg_fac")

        # Update support using a threshold, and return the total number of points in the support
        self.cl_support_update = CL_RedK(self.cl_ctx, np.int32, neutral="0", reduce_expr="a+b",
                                         map_expr="SupportUpdate(i, d, support, threshold, force_shrink)",
                                         preamble=getks("cdi/opencl/cdi_support_update_red.cl"),
                                         options=self.cl_options,
                                         arguments="__global float *d, __global char *support, const float threshold, const char force_shrink")

        # Update support using a threshold, for border pixels only, and return the total number of points in the support
        self.cl_support_update_border = CL_RedK(self.cl_ctx, np.int32, neutral="0", reduce_expr="a+b",
                                                map_expr="SupportUpdateBorder(i, d, support, threshold, force_shrink)",
                                                preamble=getks("cdi/opencl/cdi_support_update_red.cl"),
                                                options=self.cl_options,
                                                arguments="__global float *d, __global char *support,"
                                                          "const float threshold, const char force_shrink")

        # Init support from autocorrelation array - also return the number of points in the support
        self.cl_support_init = CL_RedK(self.cl_ctx, np.int32, neutral="0", reduce_expr="a+b",
                                       preamble=getks("cdi/opencl/cdi_support_update_red.cl"),
                                       map_expr="SupportInit(i, d, support, threshold)",
                                       options=self.cl_options,
                                       arguments="__global float2 *d, __global char *support, const float threshold")

        # Calculate the average amplitude and maximum intensity in the support (complex object)
        self.cl_average_max_red = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                          reduce_expr="(float2)(a.x+b.x, fmax(a.y,b.y))",
                                          map_expr="(float2)(native_sqrt(dot(obj[i], obj[i])),"
                                                   "dot(obj[i], obj[i])) * support[i]",
                                          options=self.cl_options,
                                          arguments="__global float2 *obj, __global char *support")

        # Calculate the average, maximum square modulus in the support, and the sum inside/outside
        self.cl_obj_support_stats_red = \
            CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)",
                    reduce_expr="(float4)(a.x+b.x, fmax(a.y,b.y), a.z+b.z, a.w+b.w)",
                    map_expr="ObjSupportStats(i, obj, support)",
                    options=self.cl_options, preamble=getks("cdi/opencl/cdi_support_update_red.cl"),
                    arguments="__global float2 *obj, __global char *support")

        # Calculate the average amplitude and maximum intensity in the support (object amplitude)
        self.cl_average_max_abs_red = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                              reduce_expr="(float2)(a.x+b.x, fmax(a.y,b.y))",
                                              map_expr="(float2)(obj[i], obj[i] * obj[i]) * support[i]",
                                              options=self.cl_options,
                                              arguments="__global float *obj, __global char *support")

        # Calculate the root mean square and maximum intensity in the support (object amplitude)
        self.cl_rms_max_abs_red = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                          reduce_expr="(float2)(a.x+b.x, fmax(a.y,b.y))",
                                          map_expr="(float2)(obj[i] * obj[i], obj[i] * obj[i]) * support[i]",
                                          options=self.cl_options,
                                          arguments="__global float *obj, __global char *support")

        self.cl_scale_amplitude = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)", reduce_expr="a+b",
                                          map_expr="ScaleAmplitude(i, iobs, calc)",
                                          preamble=getks('cdi/opencl/scale_obs_calc_red.cl'),
                                          options=self.cl_options,
                                          arguments="float * iobs, float2 *calc")

        self.cl_scale_intensity = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)", reduce_expr="a+b",
                                          map_expr="ScaleIntensity(i, iobs, calc)",
                                          preamble=getks('cdi/opencl/scale_obs_calc_red.cl'),
                                          options=self.cl_options,
                                          arguments="float * iobs, float2 *calc")

        self.cl_scale_intensity_poisson = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                                  reduce_expr="a+b",
                                                  map_expr="ScaleIntensityPoisson(i, iobs, calc)",
                                                  preamble=getks('cdi/opencl/scale_obs_calc_red.cl'),
                                                  options=self.cl_options,
                                                  arguments="float * iobs, float2 *calc")

        self.cl_scale_weighted_intensity = CL_RedK(self.cl_ctx, cla.vec.float2, neutral="(float2)(0,0)",
                                                   reduce_expr="a+b",
                                                   map_expr="ScaleWeightedIntensity(i, iobs, calc)",
                                                   preamble=getks('cdi/opencl/scale_obs_calc_red.cl'),
                                                   options=self.cl_options,
                                                   arguments="float * iobs, float2 *calc")

        # Absolute maximum of complex array
        self.cl_max_red = CL_RedK(self.cl_ctx, np.float32, neutral="0", reduce_expr="a > b ? a : b",
                                  map_expr="length(d[i])", options=self.cl_options, arguments="__global float2 *d")

        # Other kernels
        # Convolution kernels for support update (Gaussian)
        conv16_mod = cl.Program(self.cl_ctx, getks('opencl/convolution16.cl')).build(options=self.cl_options)
        self.abs_gauss_convol_16x = conv16_mod.abs_gauss_convol_16x
        self.gauss_convol_16y = conv16_mod.gauss_convol_16y
        self.gauss_convol_16z = conv16_mod.gauss_convol_16z

        # Same using a binary window
        conv16b_mod = cl.Program(self.cl_ctx, getks('opencl/convolution16b.cl')).build(options=self.cl_options)
        self.binary_window_convol_16x = conv16b_mod.binary_window_convol_16x
        self.binary_window_convol_16y = conv16b_mod.binary_window_convol_16y
        self.binary_window_convol_16z = conv16b_mod.binary_window_convol_16z
        self.binary_window_convol_16x_mask = conv16b_mod.binary_window_convol_16x_mask
        self.binary_window_convol_16y_mask = conv16b_mod.binary_window_convol_16y_mask
        self.binary_window_convol_16z_mask = conv16b_mod.binary_window_convol_16z_mask

        # Init memory pool
        self.cl_mem_pool = cltools.MemoryPool(cltools.ImmediateAllocator(self.cl_queue))


"""
The default processing unit 
"""
default_processing_unit = CLProcessingUnitCDI()


class CLOperatorCDI(OperatorCDI):
    """
    Base class for a operators on CDI objects using OpenCL
    """

    def __init__(self, processing_unit=None, lazy=False):
        super(CLOperatorCDI, self).__init__(lazy=lazy)

        self.Operator = CLOperatorCDI
        self.OperatorSum = CLOperatorCDISum
        self.OperatorPower = CLOperatorCDIPower

        if processing_unit is None:
            self.processing_unit = default_processing_unit
        else:
            self.processing_unit = processing_unit
        if self.processing_unit.cl_ctx is None:
            # OpenCL kernels have not been prepared yet, use a default initialization
            if main_default_processing_unit.cl_device is None:
                main_default_processing_unit.select_gpu(language='opencl')
            self.processing_unit.init_cl(cl_device=main_default_processing_unit.cl_device,
                                         test_fft=False, verbose=False)

    def apply_ops_mul(self, cdi: CDI):
        """
        Apply the series of operators stored in self.ops to a CDI object.
        In this version the operators are applied one after the other to the same CDI (multiplication)

        :param cdi: the CDI object to which the operators will be applied.
        :return: the CDI object, after application of all the operators in sequence
        """
        return super(CLOperatorCDI, self).apply_ops_mul(cdi)

    def prepare_data(self, cdi):
        # Make sure data is already in OpenCL space, otherwise transfer it
        if cdi._timestamp_counter > cdi._cl_timestamp_counter:
            # print("Moving data to OpenCL space")
            pu = self.processing_unit
            cdi._cl_obj = cla.to_device(self.processing_unit.cl_queue, cdi._obj, async_=False, allocator=pu.cl_mem_pool)
            cdi._cl_support = cla.to_device(self.processing_unit.cl_queue, cdi._support, async_=False,
                                            allocator=pu.cl_mem_pool)
            cdi._cl_iobs = cla.to_device(self.processing_unit.cl_queue, cdi.iobs, async_=False,
                                         allocator=pu.cl_mem_pool)
            if cdi._psf_f is None:
                cdi._cl_psf_f = None
            else:
                # We keep the Fourier Transform of the PSF convolution kernel in GPU memory (half-Hermitian array)
                if cdi._psf_f.ndim == 2:
                    axes = (-1, -2)
                else:
                    axes = (-1, -2, -3)

                cdi._cl_psf_f = cla.to_device(self.processing_unit.cl_queue, cdi._psf_f, async_=False,
                                              allocator=pu.cl_mem_pool)

            cdi._cl_timestamp_counter = cdi._timestamp_counter
        if has_attr_not_none(cdi, '_cl_obj_view') is False:
            cdi._cl_obj_view = {}

    def timestamp_increment(self, cdi):
        cdi._cl_timestamp_counter += 1

    def view_register(self, obj):
        """
        Creates a new unique view key in an object. When finished with this view, it should be de-registered
        using view_purge. Note that it only reserves the key, but does not create the view.
        :return: an integer value, which corresponds to yet-unused key in the object's view.
        """
        i = 1
        while i in obj._cl_obj_view:
            i += 1
        obj._cl_obj_view[i] = None
        return i

    def view_copy(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._cl_obj
        else:
            src = obj._cl_obj_view[i_source]
        if i_dest == 0:
            obj._cl_obj = cla.empty_like(src)
            dest = obj._cl_obj
        else:
            obj._cl_obj_view[i_dest] = cla.empty_like(src)
            dest = obj._cl_obj_view[i_dest]
        cl.enqueue_copy(self.processing_unit.cl_queue, src=src.data, dest=dest.data)

    def view_swap(self, obj, i1, i2):
        if i1 != 0:
            if i1 not in obj._cl_obj_view:
                # Create dummy value, assume a copy will be made later
                obj._cl_obj_view[i1] = None
        if i2 != 0:
            if i2 not in obj._cl_obj_view:
                # Create dummy value, assume a copy will be made later
                obj._cl_obj_view[i2] = None
        if i1 == 0:
            obj._cl_obj, obj._cl_obj_view[i2] = obj._cl_obj_view[i2], obj._cl_obj
        elif i2 == 0:
            obj._cl_obj, obj._cl_obj_view[i1] = obj._cl_obj_view[i1], obj._cl_obj
        else:
            obj._cl_obj_view[i1], obj._cl_obj_view[i2] = obj._cl_obj_view[i2], obj._cl_obj_view[i1]

    def view_sum(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._cl_obj
        else:
            src = obj._cl_obj_view[i_source]
        if i_dest == 0:
            dest = obj._cl_obj
        else:
            dest = obj._cl_obj_view[i_dest]
        self.processing_unit.cl_sum(src, dest)

    def view_purge(self, obj, i):
        if i is not None:
            del obj._cl_obj_view[i]
        elif has_attr_not_none(obj, '_cl_obj_view'):
            del obj._cl_obj_view
            self.processing_unit.cl_queue.finish()  # is this useful ?


# The only purpose of this class is to make sure it inherits from CLOperatorCDI and has a processing unit
class CLOperatorCDISum(OperatorSum, CLOperatorCDI):
    def __init__(self, op1, op2):
        if np.isscalar(op1):
            op1 = Scale(op1)
        if np.isscalar(op2):
            op2 = Scale(op2)
        if isinstance(op1, CLOperatorCDI) is False or isinstance(op2, CLOperatorCDI) is False:
            raise OperatorException(
                "ERROR: cannot add a CLOperatorCDI with a non-CLOperatorCDI: %s + %s" % (str(op1), str(op2)))
        # We can only have a sum of two CLOperatorCDI, so they must have a processing_unit attribute.
        CLOperatorCDI.__init__(self, op1.processing_unit)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorCDI
        self.OperatorSum = CLOperatorCDISum
        self.OperatorPower = CLOperatorCDIPower
        self.prepare_data = types.MethodType(CLOperatorCDI.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorCDI.timestamp_increment, self)
        self.view_copy = types.MethodType(CLOperatorCDI.view_copy, self)
        self.view_swap = types.MethodType(CLOperatorCDI.view_swap, self)
        self.view_sum = types.MethodType(CLOperatorCDI.view_sum, self)
        self.view_purge = types.MethodType(CLOperatorCDI.view_purge, self)


# The only purpose of this class is to make sure it inherits from CLOperatorCDI and has a processing unit
class CLOperatorCDIPower(OperatorPower, CLOperatorCDI):
    def __init__(self, op, n):
        CLOperatorCDI.__init__(self, op.processing_unit)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorCDI
        self.OperatorSum = CLOperatorCDISum
        self.OperatorPower = CLOperatorCDIPower
        self.prepare_data = types.MethodType(CLOperatorCDI.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorCDI.timestamp_increment, self)
        self.view_copy = types.MethodType(CLOperatorCDI.view_copy, self)
        self.view_swap = types.MethodType(CLOperatorCDI.view_swap, self)
        self.view_sum = types.MethodType(CLOperatorCDI.view_sum, self)
        self.view_purge = types.MethodType(CLOperatorCDI.view_purge, self)


class AutoCorrelationSupport(CLOperatorCDI):
    """Operator to calculate an initial support from the auto-correlation
    function of the observed intensity.
    The object will be multiplied by the resulting support.
    """

    def __init__(self, threshold=0.2, verbose=False, lazy=False, scale=False):
        """
        :param threshold: pixels above the autocorrelation maximum multiplied by the threshold will be included
                          in the support. This can either be a float, or a range tuple (min,max) between
                          which the threshold value will be randomly chosen every time the operator
                          is applied.
        :param verbose: if True, print info about the result of the auto-correlation
        :param lazy: if True, this will be queued for later execution in the cdi object
        :param scale: if True, or a string is given, will also apply ScaleObj(method=scale)
        """
        super(AutoCorrelationSupport, self).__init__(lazy=lazy)
        self.threshold = threshold
        self.verbose = verbose
        self.scale = scale

    def op(self, cdi: CDI):
        t = self.threshold
        if isinstance(t, list) or isinstance(t, tuple):
            t = np.random.uniform(t[0], t[1])

        pu = self.processing_unit
        tmp = cla.empty(pu.cl_queue, cdi._cl_iobs.shape, dtype=np.complex64, allocator=pu.cl_mem_pool)
        # Copy Iobs to complex array (we could also do a real->complex transform, C2C is easier)
        pu.cl_autocorrel_iobs(tmp, cdi._cl_iobs)
        pu.fft(tmp, tmp)
        thres = np.float32(pu.cl_max_red(tmp, wait_for=pu.ev).get() * t)
        pu.ev = []
        cdi._cl_support = cla.zeros(pu.cl_queue, cdi._cl_obj.shape, dtype=np.int8, allocator=pu.cl_mem_pool)
        cdi.nb_point_support = int(pu.cl_support_init(tmp, cdi._cl_support, thres, wait_for=pu.ev).get())
        if self.verbose:
            print('AutoCorrelation: %d pixels in support (%6.2f%%), threshold = %f (relative = %5.3f)' %
                  (cdi.nb_point_support, cdi.nb_point_support * 100 / tmp.size, thres, t))
        # Apply support to object
        self.processing_unit.cl_er(cdi._cl_obj, cdi._cl_support)

        if self.scale is True:
            return ScaleObj(method='F') * cdi
        elif isinstance(self.scale, str):
            return ScaleObj(method=self.scale) * cdi
        return cdi


class CopyToPrevious(CLOperatorCDI):
    """
    Operator which will store a copy of the cdi object as cl_obj_previous. This is used for various algorithms, such
    as difference map or RAAR
    """

    def op(self, cdi):
        pu = self.processing_unit
        if has_attr_not_none(cdi, '_cl_obj_previous') is False:
            cdi._cl_obj_previous = cla.empty_like(cdi._cl_obj)
        if cdi._cl_obj_previous.shape == cdi._cl_obj.shape:
            cdi._cl_obj_previous = cla.empty_like(cdi._cl_obj)
        pu.ev = [cl.enqueue_copy(self.processing_unit.cl_queue, cdi._cl_obj_previous.data, cdi._cl_obj.data,
                                 wait_for=pu.ev)]
        return cdi


class FromPU(CLOperatorCDI):
    """
    Operator copying back the CDI object and support data from the opencl device to numpy. The calculated complex
    amplitude is also retrieved by computing the Fourier transform of the current view of the object.
    
    DEPRECATED
    """

    def op(self, cdi):
        warnings.warn("Use of ToPU() and FromPU() operators is now deprecated. Use get() and set() to access data.")
        # cdi.obj[:] = cdi._cl_obj.get()
        # cdi._support[:] = cdi._cl_support.get()
        # cdi = FT() * cdi
        # cdi.calc = cdi._cl_obj.get()
        # cdi = IFT() * cdi
        return cdi


class ToPU(CLOperatorCDI):
    """
    Operator copying the data from numpy to the opencl device, as a complex64 array.

    DEPRECATED
    """

    def op(self, cdi):
        warnings.warn("Use of ToPU() and FromPU() operators is now deprecated. Use get() and set() to access data.")
        return cdi


class FreePU(CLOperatorCDI):
    """
    Operator freeing OpenCL memory. The FFT plan/app in self.processing_unit is removed,
    as well as any OpenCL cla.Array attribute in the supplied CDI object.
    
    The latest object and support data is retrieved from GPU memory
    """

    def op(self, cdi):
        self.processing_unit.finish()
        # Get back last object and support
        cdi.get_obj()
        # Purge all pyopencl arrays
        for o in dir(cdi):
            if isinstance(cdi.__getattribute__(o), cla.Array):
                cdi.__setattr__(o, None)
        self.processing_unit.free_fft_plans()
        self.view_purge(cdi, None)
        return cdi

    def timestamp_increment(self, cdi):
        cdi._timestamp_counter += 1


class FreeFromPU(CLOperatorCDI):
    """
    Gets back data from OpenCL and removes all OpenCL arrays.
    
    DEPRECATED
    """

    def __new__(cls):
        return FreePU() * FromPU()


class Scale(CLOperatorCDI):
    """
    Multiply the object by a scalar (real or complex).
    """

    def __init__(self, x):
        """

        :param x: the scaling factor
        """
        super(Scale, self).__init__()
        self.x = x

    def op(self, w):
        pu = self.processing_unit
        if np.isreal(self.x):
            pu.ev = [pu.cl_scale(w._cl_obj, np.float32(self.x), wait_for=pu.ev)]
        else:
            pu.ev = [pu.cl_scale_complex(w._cl_obj, np.complex64(self.x), wait_for=pu.ev)]
        return w


class FT(CLOperatorCDI):
    """
    Forward Fourier transform.
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the Fourier transform will be normalised, so that the transformed array L2 norm will
                      remain constant (by dividing the output by the square root of the object's size).
                      If False or None, the array norm will not be changed. If a scalar is given, the output array
                      is multiplied by it.
        """
        super(FT, self).__init__()
        self.scale = scale

    def op(self, cdi):
        pu = self.processing_unit
        s = pu.fft(cdi._cl_obj, cdi._cl_obj)
        if self.scale is True:
            pu.ev = [pu.cl_scale(cdi._cl_obj, np.float32(s), wait_for=pu.ev)]
        elif (self.scale is not False) and (self.scale is not None):
            cdi = Scale(self.scale) * cdi
        cdi._is_in_object_space = False
        return cdi


class IFT(CLOperatorCDI):
    """
    Inverse Fourier transform
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the Fourier transform will be normalised, so that the transformed array L2 norm will
                      remain constant (by multiplying the output by the square root of the object's size).
                      If False or None, the array norm will not be changed. If a scalar is given, the output array
                      is multiplied by it.
        """
        super(IFT, self).__init__()
        self.scale = scale

    def op(self, cdi):
        pu = self.processing_unit
        s = pu.ifft(cdi._cl_obj, cdi._cl_obj)
        if self.scale is True:
            pu.ev = [pu.cl_scale(cdi._cl_obj, np.float32(s), wait_for=pu.ev)]
        elif (self.scale is not False) and (self.scale is not None):
            cdi = Scale(self.scale) * cdi
        cdi._is_in_object_space = True
        return cdi


class Calc2Obs(CLOperatorCDI):
    """
    Copy the calculated intensities to the observed ones. Can be used for simulation.
    """

    def __init__(self):
        """

        """
        super(Calc2Obs, self).__init__()

    def op(self, cdi):
        pu = self.processing_unit
        if cdi.in_object_space():
            cdi = FT(scale=False) * cdi
            pu.ev = [pu.cl_square_modulus(cdi._cl_iobs, cdi._cl_obj, wait_for=pu.ev)]
            cdi = IFT(scale=False) * cdi
        else:
            pu.ev = [pu.cl_square_modulus(cdi._cl_iobs, cdi._cl_obj, wait_for=pu.ev)]
        return cdi


class ApplyAmplitude(CLOperatorCDI):
    """
    Apply the magnitude from an observed intensity, keep the phase.
    """

    def __init__(self, calc_llk=False, zero_mask=False, scale_in=1, scale_out=1, confidence_interval_factor=0,
                 confidence_interval_factor_mask_min=0.5, confidence_interval_factor_mask_max=1.2,
                 update_psf=False, psf_filter=None):
        """

        :param calc_llk: if true, the log-likelihood will be calculated and stored in the object
        :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to zero, otherwise the calculated
                          complex amplitude is kept with an optional scale factor.
        :param scale_in: a scale factor by which the input values should be multiplied, typically because of FFT
        :param scale_out: a scale factor by which the output values should be multiplied, typically because of FFT
        :param confidence_interval_factor: a relaxation factor, with the projection of calculated amplitude being done
                                           towards the limit of the poisson confidence interval. A value of 1
                                           corresponds to a 50% confidence interval, a value of 0 corresponds to a
                                           strict observed amplitude projection. [EXPERIMENTAL]
        :param confidence_interval_factor_mask_min, confidence_interval_factor_mask_max:
            For masked pixels where a value has been estimated (e.g. with InterpIobsMask()),
            a confidence interval can be given as a factor to be applied to the interpolated
            observed intensity. This corresponds to values stored between -1e19 and -1e38. [EXPERIMENTAL]
        :param update_psf: if True, will update the PSF convolution kernel using
            the Richard-Lucy deconvolution approach. If there is no PSF, it will be automatically
            initialised.
        :param psf_filter: either None, "hann" or "tukey": window type to filter the PSF update
        """
        super(ApplyAmplitude, self).__init__()
        self.calc_llk = calc_llk
        self.scale_in = np.float32(scale_in)
        self.scale_out = np.float32(scale_out)
        self.zero_mask = np.int8(zero_mask)
        self.confidence_interval_factor = np.float32(confidence_interval_factor)
        self.confidence_interval_factor_mask_min = np.float32(confidence_interval_factor_mask_min)
        self.confidence_interval_factor_mask_max = np.float32(confidence_interval_factor_mask_max)
        self.update_psf = update_psf
        self.psf_filter = psf_filter

    def op(self, cdi: CDI):
        # TODO: use a single-pass reduction kernel to apply the amplitude and compute the LLK
        pu = self.processing_unit
        nx = np.int32(cdi.iobs.shape[-1])
        ny = np.int32(cdi.iobs.shape[-2])
        if cdi.iobs.ndim == 3:
            nz = np.int32(cdi.iobs.shape[0])
        else:
            nz = np.int32(1)
        if cdi.get_upsample() is not None:
            uz, uy, ux = cdi.get_upsample(dim3=True)

        if self.update_psf and cdi._psf_f is None:
            cdi = InitPSF() * cdi

        if cdi._psf_f is None:
            if self.calc_llk:
                cdi = LLK(scale=self.scale_in) * cdi
            pu.ev = [pu.cl_apply_amplitude(cdi._cl_iobs, cdi._cl_obj, self.scale_in, self.scale_out,
                                           self.zero_mask, self.confidence_interval_factor,
                                           self.confidence_interval_factor_mask_min,
                                           self.confidence_interval_factor_mask_max, wait_for=pu.ev)]
        else:
            # FFT-based convolution, using half-Hermitian kernel and real->complex64 FFT
            cl_icalc = cla.empty_like(cdi._cl_iobs)  # float32
            cl_icalc_f = cla.empty_like(cdi._cl_psf_f)  # Complex64, half-Hermitian array
            s = pu.fft_scale(cdi._obj.shape)

            pu.ev = [pu.cl_square_modulus(cl_icalc, cdi._cl_obj, wait_for=pu.ev)]
            pu.fft(cl_icalc, cl_icalc_f)
            pu.ev = [pu.cl_mult_scale_complex(cdi._cl_psf_f, cl_icalc_f, s[0] * s[1], wait_for=pu.ev)]
            pu.ifft(cl_icalc_f, cl_icalc)

            DEBUG = False
            if self.calc_llk:
                llk = self.processing_unit.cl_llk_icalc_red(cdi._cl_iobs, cl_icalc, self.scale_in ** 2,
                                                            wait_for=pu.ev).get()
                pu.ev = []
                cdi.llk_poisson = llk['x']
                cdi.llk_gaussian = llk['y']
                cdi.llk_euclidian = llk['z']
                cdi.nb_photons_calc = llk['w']
                cdi.llk_poisson_free = llk['s4']
                cdi.llk_gaussian_free = llk['s5']
                cdi.llk_euclidian_free = llk['s6']
                if DEBUG:  # DEBUG
                    import matplotlib.pyplot as plt
                    from matplotlib.colors import LogNorm
                    plt.figure(501, figsize=(30, 6))

                    plt.subplot(141)
                    iobs = fftshift(cdi.iobs[0])
                    iobs[iobs < 0] = 0
                    vmin, vmax = 0.1, iobs.max()
                    plt.imshow(iobs, norm=LogNorm(vmin=vmin, vmax=vmax))
                    plt.colorbar()
                    plt.title("Iobs")

                    plt.subplot(142)
                    icalc = fftshift(abs(cdi._cl_obj.get()[0] * self.scale_in) ** 2)
                    plt.imshow(icalc, norm=LogNorm(vmin=vmin, vmax=vmax))
                    plt.colorbar()
                    plt.title("Calc**2")

                    plt.subplot(143)
                    icalc = fftshift(cl_icalc.get()[0] * self.scale_in ** 2)
                    plt.imshow(icalc, norm=LogNorm(vmin=vmin, vmax=vmax))
                    plt.colorbar()
                    plt.title("Icalc")

            pu.ev = [pu.cl_apply_amplitude_icalc(cdi._cl_iobs, cdi._cl_obj, cl_icalc, self.scale_in, self.scale_out,
                                                 self.zero_mask, self.confidence_interval_factor,
                                                 self.confidence_interval_factor_mask_min,
                                                 self.confidence_interval_factor_mask_max, wait_for=pu.ev)]

            if DEBUG and self.calc_llk:
                plt.subplot(144)
                icalc = fftshift(cl_icalc.get()[0] * self.scale_in ** 2)
                plt.imshow(icalc, norm=LogNorm(vmin=vmin, vmax=vmax))
                plt.colorbar()
                plt.title("Icalc_proj")
                # plt.tight_layout()
                plt.draw()
                plt.gcf().canvas.draw()
                plt.pause(.001)

            if self.update_psf:
                # Requires:
                # - two extra arrays of size iobs
                # - 8 r+w of arrays of size iobs, plus 4 r2c FFT
                # total (for 3D): 20 r+w of size iobs, equivalent to 10 r+w of complex object,
                # so equivalent to a full FourierApplyAmplitude without psf

                if pu.use_vkfft:
                    # Need to recompute icalc_f which is overwritten during the c2r transform,
                    pu.fft(cl_icalc, cl_icalc_f)

                # FFT scales - we try to scale the arrays as the calculations proceed
                # in order to avoid under or overflow
                s = pu.fft_scale(cdi._obj.shape)

                # iobs / convolve(icalc,psf)
                # Ideally close to 1
                pu.cl_psf1(cdi._cl_iobs, cl_icalc, self.scale_in ** 2)
                cl_icalc2_f = cla.empty_like(cdi._cl_psf_f)  # Complex64, half-Hermitian array
                pu.fft(cl_icalc, cl_icalc2_f)

                # convolve(iobs / convolve(icalc,psf), icalc_mirror)
                pu.cl_psf2(cl_icalc2_f, cl_icalc_f, s[0] ** 2 * self.scale_in ** 2, wait_for=pu.ev)
                if DEBUG:
                    print("icalc2f: %8e" % abs(cl_icalc2_f.get()).sum())
                del cl_icalc_f
                pu.ifft(cl_icalc2_f, cl_icalc)
                if DEBUG:
                    print("icalc2: %8e" % cl_icalc.get().sum())
                del cl_icalc2_f

                # psf *= convolve(iobs / convolve(icalc,psf), icalc_mirror)
                cl_psf = cla.empty_like(cdi._cl_iobs)
                pu.ifft(cdi._cl_psf_f, cl_psf)
                if self.psf_filter is None:
                    pu.cl_psf3(cl_psf, cl_icalc, s[1] ** 2)
                elif self.psf_filter.lower() == "tukey":
                    pu.cl_psf3_tukey(cl_psf, cl_icalc, np.float32(0.5), nx, ny, nz, s[1] ** 2)
                else:
                    pu.cl_psf3_hann(cl_psf, cl_icalc, nx, ny, nz, s[1] ** 2)

                # Normalise psf (and apply FFT scale)
                psf_sum = cla.sum(cl_psf)
                if DEBUG:
                    print("PSF sum: %8e" % psf_sum.get())
                pu.cl_psf4(cl_psf, psf_sum)

                # Compute & store updated psf FT
                pu.fft(cl_psf, cdi._cl_psf_f)
                del cl_psf

        return cdi


class UpdatePSF(CLOperatorCDI):
    """
    Update the PSF while in detector space. Assumes that the calculated
    """

    def __init__(self, nb_cycle=1, scale_in=1, filter=None):
        """

        :param nb_cycle: the number of cycle for the Richardson-Lucy update
        :param scale_in: a scale factor by which the input values should be multiplied, typically because of FFT
        :param filter: either None, "hann" or "tukey" - this will be used to filter the PSF update.
        """
        super().__init__()
        self.nb_cycle = nb_cycle
        self.scale_in = np.float32(scale_in)
        self.filter = filter

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new UpdatePSF operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return UpdatePSF(nb_cycle=n * self.nb_cycle, scale_in=self.scale_in, filter=self.filter)

    def op(self, cdi: CDI):
        need_back_ft = False
        if cdi.in_object_space():
            cdi = FT(scale=True) * cdi
            need_back_ft = True
            self.scale_in = np.float32(1)

        pu = self.processing_unit
        nx = np.int32(cdi.iobs.shape[-1])
        ny = np.int32(cdi.iobs.shape[-2])
        if cdi.iobs.ndim == 3:
            nz = np.int32(cdi.iobs.shape[0])
        else:
            nz = np.int32(1)
        if cdi.get_upsample() is not None:
            uz, uy, ux = cdi.get_upsample(dim3=True)
        if cdi._psf_f is None:
            cdi = InitPSF() * cdi

        # FFT-based convolution, using half-Hermitian kernel and real->complex64 FFT
        cl_icalc = cla.empty_like(cdi._cl_iobs)  # float32
        cl_icalc0 = cla.empty_like(cdi._cl_iobs)  # float32
        cl_icalc_f = cla.empty_like(cdi._cl_psf_f)  # Complex64, half-Hermitian array
        cl_icalc2_f = cla.empty_like(cdi._cl_psf_f)  # Complex64, half-Hermitian array
        cl_psf = cla.empty_like(cdi._cl_iobs)

        if cdi.get_upsample() is None:
            pu.cl_square_modulus(cl_icalc0, cdi._cl_obj)
        else:
            # TODO
            pu.cl_square_modulus_up(cl_icalc0, cdi._cl_obj, nx, ny, nz, ux, uy, uz)
        s = pu.fft_scale(cdi._obj.shape)
        DEBUG = False
        for i in range(self.nb_cycle):
            # Icalc x PSF
            pu.ev = [cl.enqueue_copy(pu.cl_queue, dest=cl_icalc.data, src=cl_icalc0.data, wait_for=pu.ev)]
            pu.fft(cl_icalc, cl_icalc_f)
            pu.ev = [pu.cl_mult_scale_complex(cdi._cl_psf_f, cl_icalc_f, s[0] * s[1], wait_for=pu.ev)]

            # Copy to avoid overwriting cl_icalc_f which is needed later for psf2
            pu.ev = [cl.enqueue_copy(pu.cl_queue, dest=cl_icalc2_f.data, src=cl_icalc_f.data, wait_for=pu.ev)]
            pu.ifft(cl_icalc2_f, cl_icalc)  # this overwrites the source array [vkFFT]

            # iobs / convolve(icalc,psf)
            pu.cl_psf1(cdi._cl_iobs, cl_icalc, self.scale_in ** 2)
            pu.fft(cl_icalc, cl_icalc2_f)

            # convolve(iobs / convolve(icalc,psf), icalc_mirror)
            pu.cl_psf2(cl_icalc2_f, cl_icalc_f, s[0] ** 2 * self.scale_in ** 2, wait_for=pu.ev)
            if DEBUG:
                print("icalc2f: %8e" % abs(cl_icalc2_f.get()).sum())
            pu.ifft(cl_icalc2_f, cl_icalc)
            if DEBUG:
                print("icalc2: %8e" % cl_icalc.get().sum())

            # psf *= convolve(iobs / convolve(icalc,psf), icalc_mirror)
            pu.ifft(cdi._cl_psf_f, cl_psf)
            if self.filter is None:
                pu.ev = [pu.cl_psf3(cl_psf, cl_icalc, np.float32(1), wait_for=pu.ev)]
            elif self.filter.lower() == "tukey":
                pu.ev = [pu.cl_psf3_tukey(cl_psf, cl_icalc, np.float32(0.5), nx, ny, nz, s[1] ** 2, wait_for=pu.ev)]
            else:
                pu.ev = [pu.cl_psf3_hann(cl_psf, cl_icalc, nx, ny, nz, s[1] ** 2, wait_for=pu.ev)]

            # Normalise psf
            # TODO: avoid normalising every cycle
            psf_sum = cla.sum(cl_psf)
            if DEBUG:
                print("PSF sum: %8e" % psf_sum.get())
            pu.cl_psf4(cl_psf, psf_sum)

            # Compute & store updated psf FT
            pu.fft(cl_psf, cdi._cl_psf_f)
        if need_back_ft:
            cdi = IFT(scale=True) * cdi

        return cdi


class FourierApplyAmplitude(CLOperatorCDI):
    """
    Fourier magnitude operator, performing a Fourier transform, the magnitude projection, and a backward FT.
    """

    def __init__(self, calc_llk=False, zero_mask=False, confidence_interval_factor=0,
                 confidence_interval_factor_mask_min=0.5, confidence_interval_factor_mask_max=1.2,
                 update_psf=False, psf_filter=None, obj_stats=False):
        """

        :param calc_llk: if True, the log-likelihood will be calculated while in diffraction space.
        :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to zero, otherwise the calculated
                          complex amplitude is kept with an optional scale factor.
        :param obj_stats: if True, will call ObjSupportStats at the end
        :param psf_filter: either None, "hann" or "tukey": window type to filter the PSF update
        """
        super(FourierApplyAmplitude, self).__init__()
        self.calc_llk = calc_llk
        self.zero_mask = zero_mask
        self.confidence_interval_factor = confidence_interval_factor
        self.confidence_interval_factor_mask_min = confidence_interval_factor_mask_min
        self.confidence_interval_factor_mask_max = confidence_interval_factor_mask_max
        self.update_psf = update_psf
        self.obj_stats = obj_stats
        self.psf_filter = psf_filter

    def op(self, cdi):
        s = self.processing_unit.fft_scale(cdi._obj.shape)  # FFT scaling
        cdi = IFT(scale=False) * ApplyAmplitude(calc_llk=self.calc_llk, zero_mask=self.zero_mask,
                                                scale_in=s[0], scale_out=s[1],
                                                confidence_interval_factor=self.confidence_interval_factor,
                                                confidence_interval_factor_mask_min=0.5,
                                                confidence_interval_factor_mask_max=1.2,
                                                update_psf=self.update_psf,
                                                psf_filter=self.psf_filter) * FT(scale=False) * cdi
        if self.obj_stats:
            cdi = ObjSupportStats() * cdi
        return cdi


class ERProj(CLOperatorCDI):
    """
    Error reduction.
    """

    def __init__(self, positivity=False):
        super(ERProj, self).__init__()
        self.positivity = positivity

    def op(self, cdi):
        pu = self.processing_unit
        if self.positivity:
            pu.ev = [pu.cl_er_real(cdi._cl_obj, cdi._cl_support, wait_for=pu.ev)]
        else:
            pu.ev = [pu.cl_er(cdi._cl_obj, cdi._cl_support, wait_for=pu.ev)]
        return cdi


class ER(CLOperatorCDI):
    """
    Error reduction cycle
    """

    def __init__(self, positivity=False, calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1, zero_mask=False,
                 confidence_interval_factor=0, confidence_interval_factor_mask_min=0.5,
                 confidence_interval_factor_mask_max=1.2, update_psf=0, psf_filter=None):
        """

        :param positivity: apply a positivity restraint
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param nb_cycle: the number of cycles to perform
        :param show_cdi: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object intensity, as for ShowCDI()
        :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to zero, otherwise the calculated
                          complex amplitude is kept with an optional scale factor.
        :param confidence_interval_factor: a relaxation factor, with the projection of calculated amplitude being done
           towards the limit of the poisson confidence interval. A value of 1
           corresponds to a 50% confidence interval, a value of 0 corresponds to a
           strict observed amplitude projection. [EXPERIMENTAL]
        :param confidence_interval_factor_mask_min, confidence_interval_factor_mask_max:
            For masked pixels where a value has been estimated (e.g. with InterpIobsMask()),
            a confidence interval can be given as a factor to be applied to the interpolated
            observed intensity. This corresponds to values stored between -1e19 and -1e38. [EXPERIMENTAL]
        :param update_psf: if >0, will update the partial coherence psf every update_psf cycles.
        :param psf_filter: either None, "hann" or "tukey": window type to filter the PSF update
        """
        super(ER, self).__init__()
        self.positivity = positivity
        self.calc_llk = calc_llk
        self.nb_cycle = nb_cycle
        self.show_cdi = show_cdi
        self.fig_num = fig_num
        self.zero_mask = zero_mask
        self.confidence_interval_factor = confidence_interval_factor
        self.confidence_interval_factor_mask_min = confidence_interval_factor_mask_min
        self.confidence_interval_factor_mask_max = confidence_interval_factor_mask_max
        self.update_psf = update_psf
        self.psf_filter = psf_filter

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new AP operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return ER(positivity=self.positivity, calc_llk=self.calc_llk, nb_cycle=self.nb_cycle * n,
                  show_cdi=self.show_cdi, fig_num=self.fig_num, zero_mask=self.zero_mask, confidence_interval_factor=0,
                  confidence_interval_factor_mask_min=self.confidence_interval_factor_mask_min,
                  confidence_interval_factor_mask_max=self.confidence_interval_factor_mask_max,
                  update_psf=self.update_psf, psf_filter=self.psf_filter)

    def op(self, cdi: CDI):
        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if cdi.cycle % self.calc_llk == 0:
                    calc_llk = True

            update_psf = False
            if self.update_psf:
                update_psf = (((cdi.cycle - 1) % self.update_psf) == 0) and \
                             ((self.nb_cycle < 5) or ((self.nb_cycle - ic) > 5))

            sup_proj = ERProj(positivity=self.positivity)

            fap = FourierApplyAmplitude(calc_llk=calc_llk, zero_mask=self.zero_mask,
                                        confidence_interval_factor=self.confidence_interval_factor,
                                        confidence_interval_factor_mask_min=self.confidence_interval_factor_mask_min,
                                        confidence_interval_factor_mask_max=self.confidence_interval_factor_mask_max,
                                        update_psf=update_psf, psf_filter=self.psf_filter, obj_stats=calc_llk)
            cdi = sup_proj * fap * cdi

            if calc_llk:
                cdi.update_history(mode='llk', algorithm='ER', verbose=True, update_psf=update_psf)
            else:
                cdi.update_history(mode='algorithm', algorithm='ER', update_psf=update_psf)

            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi

            cdi.cycle += 1
        return cdi


class CFProj(CLOperatorCDI):
    """
    Charge Flipping.
    """

    def __init__(self, positivity=False):
        super(CFProj, self).__init__()
        self.positivity = positivity

    def op(self, cdi):
        pu = self.processing_unit
        if self.positivity:
            pu.ev = [pu.cl_cf_real(cdi._cl_obj, cdi._cl_support, wait_for=pu.ev)]
        else:
            pu.ev = [pu.cl_cf(cdi._cl_obj, cdi._cl_support, wait_for=pu.ev)]
        return cdi


class CF(CLOperatorCDI):
    """
    Charge flipping cycle
    """

    def __init__(self, positivity=False, calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1, zero_mask=False,
                 confidence_interval_factor=0, confidence_interval_factor_mask_min=0.5,
                 confidence_interval_factor_mask_max=1.2, update_psf=0, psf_filter=None):
        """

        :param positivity: apply a positivity restraint
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param nb_cycle: the number of cycles to perform
        :param show_cdi: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object intensity, as for ShowCDI()
        :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to zero, otherwise the calculated
                          complex amplitude is kept with an optional scale factor.
        :param confidence_interval_factor: a relaxation factor, with the projection of calculated amplitude being done
           towards the limit of the poisson confidence interval. A value of 1
           corresponds to a 50% confidence interval, a value of 0 corresponds to a
           strict observed amplitude projection. [EXPERIMENTAL]
        :param confidence_interval_factor_mask_min, confidence_interval_factor_mask_max:
            For masked pixels where a value has been estimated (e.g. with InterpIobsMask()),
            a confidence interval can be given as a factor to be applied to the interpolated
            observed intensity. This corresponds to values stored between -1e19 and -1e38. [EXPERIMENTAL]
        :param update_psf: if >0, will update the partial coherence psf every update_psf cycles.
        :param psf_filter: either None, "hann" or "tukey": window type to filter the PSF update
        """
        super(CF, self).__init__()
        self.positivity = positivity
        self.calc_llk = calc_llk
        self.nb_cycle = nb_cycle
        self.show_cdi = show_cdi
        self.fig_num = fig_num
        self.zero_mask = zero_mask
        self.confidence_interval_factor = confidence_interval_factor
        self.confidence_interval_factor_mask_min = confidence_interval_factor_mask_min
        self.confidence_interval_factor_mask_max = confidence_interval_factor_mask_max
        self.update_psf = update_psf
        self.psf_filter = psf_filter

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new CF operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return CF(positivity=self.positivity, calc_llk=self.calc_llk, nb_cycle=self.nb_cycle * n,
                  show_cdi=self.show_cdi, fig_num=self.fig_num, zero_mask=self.zero_mask,
                  confidence_interval_factor=self.confidence_interval_factor,
                  confidence_interval_factor_mask_min=self.confidence_interval_factor_mask_min,
                  confidence_interval_factor_mask_max=self.confidence_interval_factor_mask_max,
                  update_psf=self.update_psf, psf_filter=self.psf_filter)

    def op(self, cdi: CDI):
        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if cdi.cycle % self.calc_llk == 0:
                    calc_llk = True

            update_psf = False
            if self.update_psf:
                update_psf = (((cdi.cycle - 1) % self.update_psf) == 0) and \
                             ((self.nb_cycle < 5) or ((self.nb_cycle - ic) > 5))

            sup_proj = CFProj(positivity=self.positivity)

            fap = FourierApplyAmplitude(calc_llk=calc_llk, zero_mask=self.zero_mask,
                                        confidence_interval_factor=self.confidence_interval_factor,
                                        confidence_interval_factor_mask_min=self.confidence_interval_factor_mask_min,
                                        confidence_interval_factor_mask_max=self.confidence_interval_factor_mask_max,
                                        update_psf=update_psf, psf_filter=self.psf_filter, obj_stats=calc_llk)
            cdi = sup_proj * fap * cdi

            if calc_llk:
                cdi.update_history(mode='llk', algorithm='CF', verbose=True, update_psf=update_psf)
            else:
                cdi.update_history(mode='algorithm', algorithm='CF', update_psf=update_psf)

            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi

            cdi.cycle += 1
        return cdi


class HIOProj(CLOperatorCDI):
    """
    Hybrid Input-Output.
    """

    def __init__(self, beta=0.9, positivity=False):
        super(HIOProj, self).__init__()
        self.beta = np.float32(beta)
        self.positivity = positivity

    def op(self, cdi):
        pu = self.processing_unit
        if self.positivity:
            pu.ev = [pu.cl_hio_real(cdi._cl_obj, cdi._cl_obj_previous, cdi._cl_support, self.beta, wait_for=pu.ev)]
        else:
            pu.ev = [pu.cl_hio(cdi._cl_obj, cdi._cl_obj_previous, cdi._cl_support, self.beta, wait_for=pu.ev)]
        return cdi


class HIO(CLOperatorCDI):
    """
    Hybrid Input-Output reduction cycle
    """

    def __init__(self, beta=0.9, positivity=False, calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1,
                 zero_mask=False, confidence_interval_factor=0, confidence_interval_factor_mask_min=0.5,
                 confidence_interval_factor_mask_max=1.2, update_psf=0, psf_filter=None):
        """

        :param positivity: apply a positivity restraint
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param nb_cycle: the number of cycles to perform
        :param show_cdi: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object intensity, as for ShowCDI()
        :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to zero, otherwise the calculated
                          complex amplitude is kept with an optional scale factor.
        :param confidence_interval_factor: a relaxation factor, with the projection of calculated amplitude being done
           towards the limit of the poisson confidence interval. A value of 1
           corresponds to a 50% confidence interval, a value of 0 corresponds to a
           strict observed amplitude projection. [EXPERIMENTAL]
        :param confidence_interval_factor_mask_min, confidence_interval_factor_mask_max:
            For masked pixels where a value has been estimated (e.g. with InterpIobsMask()),
            a confidence interval can be given as a factor to be applied to the interpolated
            observed intensity. This corresponds to values stored between -1e19 and -1e38. [EXPERIMENTAL]
        :param update_psf: if >0, will update the partial coherence psf every update_psf cycles.
        :param psf_filter: either None, "hann" or "tukey": window type to filter the PSF update
        """
        super(HIO, self).__init__()
        self.beta = beta
        self.positivity = positivity
        self.calc_llk = calc_llk
        self.nb_cycle = nb_cycle
        self.show_cdi = show_cdi
        self.fig_num = fig_num
        self.zero_mask = zero_mask
        self.confidence_interval_factor = confidence_interval_factor
        self.confidence_interval_factor_mask_min = confidence_interval_factor_mask_min
        self.confidence_interval_factor_mask_max = confidence_interval_factor_mask_max
        self.update_psf = update_psf
        self.psf_filter = psf_filter

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new HIO operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return HIO(beta=self.beta, positivity=self.positivity, calc_llk=self.calc_llk, nb_cycle=self.nb_cycle * n,
                   show_cdi=self.show_cdi, fig_num=self.fig_num, zero_mask=self.zero_mask,
                   confidence_interval_factor=self.confidence_interval_factor,
                   confidence_interval_factor_mask_min=self.confidence_interval_factor_mask_min,
                   confidence_interval_factor_mask_max=self.confidence_interval_factor_mask_max,
                   update_psf=self.update_psf, psf_filter=self.psf_filter)

    def op(self, cdi: CDI):
        cdi = CopyToPrevious() * cdi
        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if cdi.cycle % self.calc_llk == 0:
                    calc_llk = True

            update_psf = False
            if self.update_psf:
                update_psf = (((cdi.cycle - 1) % self.update_psf) == 0) and \
                             ((self.nb_cycle < 5) or ((self.nb_cycle - ic) > 5))

            sup_proj = HIOProj(self.beta, positivity=self.positivity)

            fap = FourierApplyAmplitude(calc_llk=calc_llk, zero_mask=self.zero_mask,
                                        confidence_interval_factor=self.confidence_interval_factor,
                                        confidence_interval_factor_mask_min=self.confidence_interval_factor_mask_min,
                                        confidence_interval_factor_mask_max=self.confidence_interval_factor_mask_max,
                                        update_psf=update_psf, psf_filter=self.psf_filter, obj_stats=calc_llk)
            cdi = sup_proj * fap * cdi

            if calc_llk:
                cdi.update_history(mode='llk', algorithm='HIO', verbose=True, update_psf=update_psf)
            else:
                cdi.update_history(mode='algorithm', algorithm='HIO', update_psf=update_psf)
            cdi.cycle += 1

        if self.show_cdi:
            if cdi.cycle % self.show_cdi == 0:
                cdi = ShowCDI(fig_num=self.fig_num) * cdi

        del cdi._cl_obj_previous
        return cdi


class RAARProj(CLOperatorCDI):
    """
    RAAR.
    """

    def __init__(self, beta=0.9, positivity=False):
        super(RAARProj, self).__init__()
        self.beta = np.float32(beta)
        self.positivity = positivity

    def op(self, cdi):
        pu = self.processing_unit
        if self.positivity:
            pu.ev = [pu.cl_raar_real(cdi._cl_obj, cdi._cl_obj_previous, cdi._cl_support, self.beta, wait_for=pu.ev)]
        else:
            pu.ev = [pu.cl_raar(cdi._cl_obj, cdi._cl_obj_previous, cdi._cl_support, self.beta, wait_for=pu.ev)]
        return cdi


class GPS(CLOperatorCDI):
    """
    GPS cycle, according to Pham et al [2019]
    """

    def __init__(self, inertia=0.05, t=1.0, s=0.9, sigma_f=0, sigma_o=0, positivity=False,
                 calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1, zero_mask=False,
                 confidence_interval_factor=0, confidence_interval_factor_mask_min=0.5,
                 confidence_interval_factor_mask_max=1.2, update_psf=0, psf_filter=None):
        """
        :param inertia: inertia parameter (sigma in original Pham2019 article)
        :param t: t parameter
        :param s: s parameter
        :param sigma_f: Fourier-space smoothing kernel width, in Fourier-space pixel units
        :param sigma_o: object-space smoothing kernel width, in object-space pixel units
        :param positivity: if True, apply a positivity restraint
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param nb_cycle: the number of cycles to perform
        :param show_cdi: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object intensity, as for ShowCDI()
        :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to zero, otherwise the calculated
                          complex amplitude is kept with an optional scale factor.
        :param confidence_interval_factor: a relaxation factor, with the projection of calculated amplitude being done
           towards the limit of the poisson confidence interval. A value of 1
           corresponds to a 50% confidence interval, a value of 0 corresponds to a
           strict observed amplitude projection. [EXPERIMENTAL]
        :param confidence_interval_factor_mask_min, confidence_interval_factor_mask_max:
            For masked pixels where a value has been estimated (e.g. with InterpIobsMask()),
            a confidence interval can be given as a factor to be applied to the interpolated
            observed intensity. This corresponds to values stored between -1e19 and -1e38. [EXPERIMENTAL]
        :param update_psf: if >0, will update the partial coherence psf every update_psf cycles.
        :param psf_filter: either None, "hann" or "tukey": window type to filter the PSF update
        """
        super(GPS, self).__init__()
        self.inertia = np.float32(inertia)
        self.t = np.float32(t)
        self.s = np.float32(s)
        self.sigma_f = np.float32(sigma_f)
        self.sigma_o = np.float32(sigma_o)
        self.positivity = positivity
        self.calc_llk = calc_llk
        self.nb_cycle = nb_cycle
        self.show_cdi = show_cdi
        self.fig_num = fig_num
        self.zero_mask = zero_mask
        self.confidence_interval_factor = confidence_interval_factor
        self.confidence_interval_factor_mask_min = confidence_interval_factor_mask_min
        self.confidence_interval_factor_mask_max = confidence_interval_factor_mask_max
        self.update_psf = int(update_psf)
        self.psf_filter = psf_filter

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new GPS operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return GPS(inertia=self.inertia, t=self.t, s=self.s, sigma_f=self.sigma_f, sigma_o=self.sigma_o,
                   positivity=self.positivity, calc_llk=self.calc_llk, nb_cycle=self.nb_cycle * n,
                   show_cdi=self.show_cdi, fig_num=self.fig_num, zero_mask=self.zero_mask,
                   confidence_interval_factor=self.confidence_interval_factor,
                   confidence_interval_factor_mask_min=self.confidence_interval_factor_mask_min,
                   confidence_interval_factor_mask_max=self.confidence_interval_factor_mask_max,
                   update_psf=self.update_psf, psf_filter=self.psf_filter)

    def op(self, cdi: CDI):
        pu = self.processing_unit

        # FFT scales
        scale_in, scale_out = pu.fft_scale(cdi._obj.shape)

        epsilon = np.float32(self.inertia / (self.inertia + self.t))

        ny, nx = np.int32(cdi._obj.shape[-2]), np.int32(cdi._obj.shape[-1])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi._obj.shape[0])
        else:
            nz = np.int32(1)

        # Make sure we have tmp copy arrays available
        if has_attr_not_none(cdi, '_cl_z') is False:
            cdi._cl_z = cla.empty_like(cdi._cl_obj)
        elif cdi._cl_z.shape != cdi._cl_obj.shape:
            cdi._cl_z = cla.empty_like(cdi._cl_obj)

        if has_attr_not_none(cdi, '_cl_y') is False:
            cdi._cl_y = cla.empty_like(cdi._cl_obj)
        elif cdi._cl_y.shape != cdi._cl_obj.shape:
            cdi._cl_y = cla.empty_like(cdi._cl_obj)

        # We start in Fourier space (obj = z_0)
        cdi = FT(scale=True) * cdi

        # z_0 = FT(obj)
        pu.ev = [cl.enqueue_copy(pu.cl_queue, dest=cdi._cl_z.data, src=cdi._cl_obj.data, wait_for=pu.ev)]

        # Start with obj = y_0 = 0
        cdi._cl_obj.fill(np.complex64(0))

        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if cdi.cycle % self.calc_llk == 0:
                    calc_llk = True

            update_psf = False
            if self.update_psf:
                update_psf = (((cdi.cycle - 1) % self.update_psf) == 0) and \
                             ((self.nb_cycle < 5) or ((self.nb_cycle - ic) > 5))

            # keep y copy
            pu.ev = [cl.enqueue_copy(pu.cl_queue, dest=cdi._cl_y.data, src=cdi._cl_obj.data, wait_for=pu.ev)]

            cdi = FT(scale=False) * cdi

            # ^z = z_k - t F(y_k)
            pu.ev = [self.processing_unit.cl_gps1(cdi._cl_obj, cdi._cl_z, self.t * scale_in, self.sigma_o, nx, ny, nz,
                                                  wait_for=pu.ev)]

            cdi = ApplyAmplitude(calc_llk=calc_llk, zero_mask=self.zero_mask,
                                 confidence_interval_factor=self.confidence_interval_factor,
                                 confidence_interval_factor_mask_min=self.confidence_interval_factor_mask_min,
                                 confidence_interval_factor_mask_max=self.confidence_interval_factor_mask_max,
                                 update_psf=update_psf, psf_filter=self.psf_filter) * cdi

            # obj = z_k+1 = (1 - epsilon) * sqrt(iobs) * exp(i * arg(^z)) + epsilon * z_k
            pu.ev = [self.processing_unit.cl_gps2(cdi._cl_obj, cdi._cl_z, epsilon, wait_for=pu.ev)]

            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = IFT(scale=True) * cdi
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi
                    cdi = FT(scale=True) * cdi

            if ic < self.nb_cycle - 1:
                # obj = 2 * z_k+1 - z_k  & store z_k+1 in z
                pu.ev = [self.processing_unit.cl_gps3(cdi._cl_obj, cdi._cl_z, wait_for=pu.ev)]

                cdi = IFT(scale=False) * cdi

                # obj = ^y = proj_support[y_k + s * obj] * G_sigma_f
                pu.ev = [self.processing_unit.cl_gps4(cdi._cl_obj, cdi._cl_y, cdi._cl_support, self.s * scale_out,
                                                      self.sigma_f, self.positivity, nx, ny, nz, wait_for=pu.ev)]
            else:
                pu.ev = [self.processing_unit.cl_scale(cdi._cl_obj, scale_out, wait_for=pu.ev)]
            if calc_llk:
                cdi = ObjSupportStats() * cdi
                cdi.update_history(mode='llk', algorithm='GPS', verbose=True, update_psf=update_psf)
            else:
                cdi.update_history(mode='algorithm', algorithm='GPS', update_psf=update_psf)
            cdi.cycle += 1

        # Free memory
        del cdi._cl_y, cdi._cl_z

        # Back to object space
        cdi = IFT(scale=False) * cdi

        return cdi


class RAAR(CLOperatorCDI):
    """
    RAAR cycle
    """

    def __init__(self, beta=0.9, positivity=False, calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1,
                 zero_mask=False, confidence_interval_factor=0, confidence_interval_factor_mask_min=0.5,
                 confidence_interval_factor_mask_max=1.2, update_psf=0, psf_filter=None):
        """

        :param positivity: apply a positivity restraint: the imaginary part inside the support is flipped.
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param nb_cycle: the number of cycles to perform
        :param show_cdi: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object intensity, as for ShowCDI()
        :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to zero, otherwise the calculated
                          complex amplitude is kept with an optional scale factor.
        :param confidence_interval_factor: a relaxation factor, with the projection of calculated amplitude being done
           towards the limit of the poisson confidence interval. A value of 1
           corresponds to a 50% confidence interval, a value of 0 corresponds to a
           strict observed amplitude projection. [EXPERIMENTAL]
        :param confidence_interval_factor_mask_min, confidence_interval_factor_mask_max:
            For masked pixels where a value has been estimated (e.g. with InterpIobsMask()),
            a confidence interval can be given as a factor to be applied to the interpolated
            observed intensity. This corresponds to values stored between -1e19 and -1e38. [EXPERIMENTAL]
        :param update_psf: if >0, will update the partial coherence psf every update_psf cycles.
        :param psf_filter: either None, "hann" or "tukey": window type to filter the PSF update
        """
        super(RAAR, self).__init__()
        self.beta = beta
        self.positivity = positivity
        self.calc_llk = calc_llk
        self.nb_cycle = nb_cycle
        self.show_cdi = show_cdi
        self.fig_num = fig_num
        self.zero_mask = zero_mask
        self.confidence_interval_factor = confidence_interval_factor
        self.confidence_interval_factor_mask_min = confidence_interval_factor_mask_min
        self.confidence_interval_factor_mask_max = confidence_interval_factor_mask_max
        self.update_psf = update_psf
        self.psf_filter = psf_filter

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new HIO operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return RAAR(beta=self.beta, positivity=self.positivity, calc_llk=self.calc_llk, nb_cycle=self.nb_cycle * n,
                    show_cdi=self.show_cdi, fig_num=self.fig_num, zero_mask=self.zero_mask,
                    confidence_interval_factor=self.confidence_interval_factor,
                    confidence_interval_factor_mask_min=self.confidence_interval_factor_mask_min,
                    confidence_interval_factor_mask_max=self.confidence_interval_factor_mask_max,
                    update_psf=self.update_psf, psf_filter=self.psf_filter)

    def op(self, cdi: CDI):
        cdi = CopyToPrevious() * cdi
        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if cdi.cycle % self.calc_llk == 0:
                    calc_llk = True

            update_psf = False
            if self.update_psf:
                update_psf = (((cdi.cycle - 1) % self.update_psf) == 0) and \
                             ((self.nb_cycle < 5) or ((self.nb_cycle - ic) > 5))

            sup_proj = RAARProj(self.beta, positivity=self.positivity)

            fap = FourierApplyAmplitude(calc_llk=calc_llk, zero_mask=self.zero_mask,
                                        confidence_interval_factor=self.confidence_interval_factor,
                                        confidence_interval_factor_mask_min=self.confidence_interval_factor_mask_min,
                                        confidence_interval_factor_mask_max=self.confidence_interval_factor_mask_max,
                                        update_psf=update_psf, psf_filter=self.psf_filter, obj_stats=calc_llk)
            cdi = sup_proj * fap * cdi

            if calc_llk:
                cdi.update_history(mode='llk', algorithm='RAAR', verbose=True, update_psf=update_psf)
            else:
                cdi.update_history(mode='algorithm', algorithm='RAAR', update_psf=update_psf)

            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi

            cdi.cycle += 1
        del cdi._cl_obj_previous
        return cdi


# class DM(CLOperatorCDI):
#     """
#     DM cycle, with fixed beta=-1 (other
#     """
#
#     def __init__(self, positivity=False, calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1,
#                  zero_mask=False):
#         """
#
#         :param positivity: apply a positivity restraint (flipping the real part inside the support when applying
#                            the support projection, if it is negative).
#         :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
#                          calculated every calc_llk cycle
#         :param nb_cycle: the number of cycles to perform
#         :param show_cdi: if a positive integer number N, the object & probe will be displayed every N cycle.
#                                By default 0 (no plot)
#         :param fig_num: the number of the figure to plot the object intensity, as for ShowCDI()
#         :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to zero, otherwise the calculated
#                           complex amplitude is kept with an optional scale factor.
#         """
#         super(DM, self).__init__()
#         self.positivity = positivity
#         self.calc_llk = calc_llk
#         self.nb_cycle = nb_cycle
#         self.show_cdi = show_cdi
#         self.fig_num = fig_num
#         self.zero_mask = zero_mask
#
#     def __pow__(self, n):
#         """
#
#         :param n: a strictly positive integer
#         :return: a new HIO operator with the number of cycles multiplied by n
#         """
#         assert isinstance(n, int) or isinstance(n, np.integer)
#         return DM(positivity=self.positivity, calc_llk=self.calc_llk, nb_cycle=self.nb_cycle * n,
#                   show_cdi=self.show_cdi, fig_num=self.fig_num, zero_mask=self.zero_mask)
#
#     def op(self, cdi: CDI):
#         pu = self.processing_unit
#         if has_attr_not_none(cdi, '_cl_obj_previous') is False:
#             cdi._cl_obj_previous = cla.empty_like(cdi._cl_obj)
#         if cdi._cl_obj_previous.shape == cdi._cl_obj.shape:
#             cdi._cl_obj_previous = cla.empty_like(cdi._cl_obj)
#
#         t0 = timeit.default_timer()
#         ic_dt = 0
#         for ic in range(self.nb_cycle):
#             calc_llk = False
#             if self.calc_llk:
#                 if cdi.cycle % self.calc_llk == 0:
#                     calc_llk = True
#
#             # Copy Psi(n) to obj_previous, and compute Psi = (2*Proj_support - I) * Psi(n)
#             if self.positivity:
#                 pu.ev = [pu.cl_dm1_real(cdi._cl_obj, cdi._cl_obj_previous, cdi._cl_support, wait_for=pu.ev)]
#             else:
#                 pu.ev = [pu.cl_dm1(cdi._cl_obj, cdi._cl_obj_previous, cdi._cl_support, wait_for=pu.ev)]
#
#             cdi = FourierApplyAmplitude(calc_llk=False, zero_mask=self.zero_mask) * cdi
#
#             # Compute Psi(n+1) = Psi(n) - (Psi - Proj_support * Psi(n))
#             if self.positivity:
#                 pu.ev = [pu.cl_dm2_real(cdi._cl_obj, cdi._cl_obj_previous, cdi._cl_support, wait_for=pu.ev)]
#             else:
#                 pu.ev = [pu.cl_dm2(cdi._cl_obj, cdi._cl_obj_previous, cdi._cl_support, wait_for=pu.ev)]
#
#             if calc_llk:
#                 cdi = IFT(scale=False) * LLK(scale=pu.fft_scale(cdi._obj.shape)[0]) * FT(scale=False) * cdi
#                 # Average time/cycle over the last N cycles
#                 dt = (timeit.default_timer() - t0) / (ic - ic_dt + 1)
#                 ic_dt = ic + 1
#                 t0 = timeit.default_timer()
#                 cdi.update_history(mode='llk', dt=dt, algorithm='DM', verbose=True)
#             else:
#                 cdi.history.insert(cdi.cycle, algorithm='DM')
#             if self.show_cdi:
#                 if cdi.cycle % self.show_cdi == 0:
#                     cdi = ShowCDI(fig_num=self.fig_num) * cdi
#             cdi.cycle += 1
#         return cdi


class ML(CLOperatorCDI):
    """
    Maximum likelihood conjugate gradient minimization
    """

    def __init__(self, reg_fac=1e-2, nb_cycle=1, calc_llk=False, show_cdi=False, fig_num=-1):
        """

        :param reg_fac:
        :param nb_cycle: the number of cycles to perform
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_cdi: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object intensity, as for ShowCDI()
        """
        super(ML, self).__init__()
        self.need_init = True
        self.reg_fac = reg_fac
        self.nb_cycle = nb_cycle
        self.calc_llk = calc_llk
        self.show_cdi = show_cdi
        self.fig_num = fig_num

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new ML operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return ML(reg_fac=self.reg_fac, nb_cycle=self.nb_cycle * n, calc_llk=self.calc_llk, show_cdi=self.show_cdi,
                  fig_num=self.fig_num)

    def op(self, cdi: CDI):
        pu = self.processing_unit
        if self.need_init is False:
            if (has_attr_not_none(cdi, '_cl_obj_dir') is False) \
                    or (has_attr_not_none(cdi, '_cl_dpsi') is False) \
                    or (has_attr_not_none(cdi, '_cl_obj_grad') is False) \
                    or (has_attr_not_none(cdi, '_cl_obj_grad_last') is False) \
                    or (has_attr_not_none(cdi, 'llk_support_reg_fac') is False):
                self.need_init = True

        if self.need_init:
            # Take into account support in regularization
            N = cdi._obj.size
            # Total number of photons
            Nph = cdi.iobs_sum
            cdi.llk_support_reg_fac = np.float32(self.reg_fac / (8 * N / Nph))
            # if cdi.llk_support_reg_fac>0:
            #    print("Regularization factor for support:", cdi.llk_support_reg_fac)

            cdi._cl_obj_dir = cla.empty(pu.cl_queue, cdi._obj.shape, np.complex64, allocator=pu.cl_mem_pool)
            cdi._cl_psi = cla.empty(pu.cl_queue, cdi._obj.shape, np.complex64, allocator=pu.cl_mem_pool)
            cdi._cl_dpsi = cla.empty(pu.cl_queue, cdi._obj.shape, np.complex64, allocator=pu.cl_mem_pool)
            cdi._cl_obj_grad = cla.empty(pu.cl_queue, cdi._obj.shape, np.complex64, allocator=pu.cl_mem_pool)
            cdi._cl_obj_gradlast = cla.empty(pu.cl_queue, cdi._obj.shape, np.complex64, allocator=pu.cl_mem_pool)
            self.need_init = False

        ny, nx = np.int32(cdi._obj.shape[-2]), np.int32(cdi._obj.shape[-1])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi._obj.shape[0])
        else:
            nz = np.int32(1)

        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if cdi.cycle % self.calc_llk == 0:
                    calc_llk = True

            cdi._cl_obj_grad, cdi._cl_obj_gradlast = cdi._cl_obj_gradlast, cdi._cl_obj_grad
            pu.ev = [cl.enqueue_copy(pu.cl_queue, dest=cdi._cl_psi.data, src=cdi._cl_obj.data, wait_for=pu.ev)]
            pu.fft(cdi._cl_psi, cdi._cl_psi)
            # TODO: move this scaling elsewhere, avoid double-accessing the array after the FFT
            # pu.ev = [pu.cl_scale(cdi._cl_psi, np.float32(1 / np.sqrt(cdi._obj.size)), wait_for=pu.ev)]

            if calc_llk:
                cdi._cl_psi, cdi._cl_obj = cdi._cl_obj, cdi._cl_psi
                cdi._is_in_object_space = False
                cdi = LLK(scale=pu.fft_scale(cdi._obj.shape)[0]) * cdi
                cdi._cl_psi, cdi._cl_obj = cdi._cl_obj, cdi._cl_psi
                cdi._is_in_object_space = True

            # This calculates the conjugate of [(1 - iobs/icalc) * psi]
            pu.ev = [pu.cl_ml_poisson_psi_gradient(cdi._cl_psi, cdi._cl_obj_grad, cdi._cl_iobs, nx, ny, nz,
                                                   wait_for=pu.ev)]

            pu.ifft(cdi._cl_obj_grad, cdi._cl_obj_grad)
            # TODO: move this scaling elsewhere, avoid double-accessing the array after the FFT
            # pu.ev = [pu.cl_scale(cdi._cl_obj_grad, np.float32(np.sqrt(cdi._obj.size)), wait_for=pu.ev)]

            if cdi.llk_support_reg_fac > 0:
                pu.ev = [pu.cl_ml_poisson_reg_support_gradient(cdi._cl_obj, cdi._cl_obj_grad, cdi._cl_support,
                                                               cdi.llk_support_reg_fac, wait_for=pu.ev)]

            if ic == 0:
                beta = 0
                pu.ev = [cl.enqueue_copy(pu.cl_queue, dest=cdi._cl_obj_dir.data, src=cdi._cl_obj_grad.data,
                                         wait_for=pu.ev)]
            else:
                # Polak-Ribière CG coefficient
                tmp = pu.cl_cg_polak_ribiere_red(cdi._cl_obj_grad, cdi._cl_obj_gradlast, wait_for=pu.ev).get()
                pu.ev = []
                if False:
                    g1 = cdi._cl_obj_grad.get()
                    g0 = cdi._cl_obj_gradlast.get()
                    A, B = (g1.real * (g1.real - g0.real) + g1.imag * (g1.imag - g0.imag)).sum(), (
                            g0.real * g0.real + g0.imag * g0.imag).sum()
                    cpubeta = A / B
                    print("betaPR: (GPU)=%8.4e  , (CPU)=%8.4e [%8.4e/%8.4e], dot(g0.g1)=%8e [%8e]" %
                          (tmp['x'] / tmp['y'], cpubeta, A, B, (g0 * g1).sum().real, (abs(g0) ** 2).sum().real))
                # Reset direction if beta<0 => beta=0
                beta = np.float32(max(0, tmp['x'] / max(1e-20, tmp['y'])))

                pu.ev = [pu.cl_ml_poisson_cg_linear(np.float32(beta), cdi._cl_obj_dir, np.float32(-1),
                                                    cdi._cl_obj_grad, wait_for=pu.ev)]
            pu.ev = [cl.enqueue_copy(pu.cl_queue, dest=cdi._cl_dpsi.data, src=cdi._cl_obj_dir.data, wait_for=pu.ev)]

            # TODO: remove need for scaling
            pu.fft(cdi._cl_dpsi, cdi._cl_dpsi, norm=True)
            if cdi.llk_support_reg_fac > 0:
                tmp = pu.cdi_ml_poisson_gamma_support_red(cdi._cl_iobs, cdi._cl_psi, cdi._cl_dpsi, cdi._cl_obj,
                                                          cdi._cl_obj_dir, cdi._cl_support,
                                                          cdi.llk_support_reg_fac, wait_for=pu.ev).get()
                pu.ev = []
                gamma_n, gamma_d = tmp['x'], tmp['y']
                gamma = np.float32(gamma_n / gamma_d)
            else:
                tmp = pu.cdi_ml_poisson_gamma_red(cdi._cl_iobs, cdi._cl_psi, cdi._cl_dpsi, wait_for=pu.ev).get()
                pu.ev = []
                gamma_n, gamma_d = tmp['x'], tmp['y']
                gamma = np.float32(gamma_n / gamma_d)

            pu.ev = [pu.cl_ml_poisson_cg_linear(np.float32(1), cdi._cl_obj, gamma, cdi._cl_obj_dir, wait_for=pu.ev)]

            if calc_llk:
                cdi = ObjSupportStats() * cdi
                cdi.update_history(mode='llk', algorithm='ML', verbose=True)
            else:
                cdi.update_history(mode='algorithm', algorithm='ML')
            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi

            cdi.cycle += 1

        return cdi


class SupportUpdate(CLOperatorCDI):
    """
    Update the support
    """

    def __init__(self, threshold_relative=0.2, smooth_width=3, force_shrink=False, method='rms',
                 post_expand=None, verbose=False, update_border_n=0, min_fraction=0, max_fraction=1,
                 lazy=False):
        """ Update support.

        Args:
            threshold_relative: must be between 0 and 1. Only points with object amplitude above a value equal to
                relative_threshold * reference_value are kept in the support.
                reference_value can either:
                - use the fact that when converged, the square norm of the object is equal to the number of
                recorded photons (normalized Fourier Transform). Then:
                reference_value = sqrt((abs(obj)**2).sum()/nb_points_support)
                - or use threshold_percentile (see below, very slow, deprecated)
            smooth_width: smooth the object amplitude using a gaussian of this width before calculating new support
                          If this is a scalar, the smooth width is fixed to this value.
                          If this is a 3-value tuple (or list or array), i.e. 'smooth_width=2,0.5,600', the smooth width
                          will vary with the number of cycles recorded in the CDI object (as cdi.cycle), varying
                          exponentially from the first to the second value over the number of cycles specified by the
                          last value.
                          With 'smooth_width=a,b,nb':
                          - smooth_width = a * exp(-cdi.cycle/nb*log(b/a)) if cdi.cycle < nb
                          - smooth_width = b if cdi.cycle >= nb
            force_shrink: if True, the support can only shrink
            method: either 'max' or 'average' or 'rms' (default), the threshold will be relative to either the maximum
                    amplitude in the object, or the average or root-mean-square amplitude (computed inside support)
            post_expand=1: after the new support has been calculated, it can be processed using the SupportExpand
                           operator, either one or multiple times, in order to 'clean' the support:
                           - 'post_expand=1' will expand the support by 1 pixel
                           - 'post_expand=-1' will shrink the support by 1 pixel
                           - 'post_expand=(-1,1)' will shrink and then expand the support by 1 pixel
                           - 'post_expand=(-2,3)' will shrink and then expand the support by respectively 2 and 3 pixels
            verbose: if True, print number of points in support
            update_border_n: if > 0, the only pixels affected by the support updated lie within +/- N pixels around the
                             outer border of the support.
            min_fraction, max_fraction: these are the minimum and maximum fraction of the support volume in
                the object. If the support volume fraction becomes smaller than min_fraction or larger
                than max_fraction, a corresponding exception will be raised.
                Example values: min_size=0.001, max_size=0.5
            lazy: if True, this will be queued for later execution in the cdi object
        Raises: SupportTooSmall or SupportTooLarge if support diverges according to {min|max}_fraction
        Returns:
            Nothing. self._support is updated
        """
        super(SupportUpdate, self).__init__(lazy=lazy)
        self.smooth_width = smooth_width
        self.threshold_relative = threshold_relative
        self.force_shrink = np.bool(force_shrink)
        self.method = method
        self.verbose = verbose
        if isinstance(post_expand, int) or isinstance(post_expand, np.integer):
            self.post_expand = (post_expand,)
        else:
            self.post_expand = post_expand
        self.update_border_n = update_border_n
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

    def op(self, cdi):
        pu = self.processing_unit
        if np.isscalar(self.smooth_width):
            smooth_width = self.smooth_width
        else:
            a, b, nb = self.smooth_width
            i = cdi.cycle
            if i < nb:
                smooth_width = a * np.exp(-i / nb * np.log(a / b))
            else:
                smooth_width = b
        # Convolve the absolute value of the object
        cdi = ObjConvolve(sigma=smooth_width) * cdi

        # Get average amplitude and maximum intensity for the object in the support (unsmoothed)
        tmp = pu.cl_average_max_red(cdi._cl_obj, cdi._cl_support, wait_for=pu.ev).get()
        pu.ev = []
        cdi._obj_max = np.sqrt(tmp['y'])

        # Actual threshold is computed on the convolved object
        if self.method == 'max' or cdi.nb_point_support == 0:
            tmp = pu.cl_average_max_abs_red(cdi._cl_obj_abs, cdi._cl_support, wait_for=pu.ev).get()
            thr = self.threshold_relative * np.float32(np.sqrt(tmp['y']))
        elif self.method == 'rms':
            tmp = pu.cl_rms_max_abs_red(cdi._cl_obj_abs, cdi._cl_support, wait_for=pu.ev).get()
            thr = self.threshold_relative * np.sqrt(np.float32(tmp['x'] / cdi.nb_point_support))
        else:
            tmp = pu.cl_average_max_abs_red(cdi._cl_obj_abs, cdi._cl_support, wait_for=pu.ev).get()
            thr = self.threshold_relative * np.float32(tmp['x'] / cdi.nb_point_support)
        pu.ev = []

        # Update support and compute the new number of points in the support
        if self.update_border_n > 0:
            # First compute the border of the support
            nx, ny = np.int32(cdi._obj.shape[-1]), np.int32(cdi._obj.shape[-2])
            if cdi._obj.ndim == 3:
                nz = np.int32(cdi._obj.shape[0])
            else:
                nz = np.int32(1)

            m1 = np.int8(2)  # Bitwise mask for expanded support
            m2 = np.int8(4)  # Bitwise mask for shrunk support

            # Convolution kernel width cannot exceed 7, so loop for larger convolutions
            for i in range(0, self.update_border_n, 7):
                n = np.int32(self.update_border_n - i) if (self.update_border_n - i) <= 7 else np.int32(7)

                # Expanded support
                m0 = m1 if i > 0 else np.int8(1)
                pu.ev = [pu.binary_window_convol_16x_mask(pu.cl_queue, (16, ny, nz), (16, 1, 1), cdi._cl_support.data,
                                                          n, nx, ny, nz, m0, m1, wait_for=pu.ev)]
                pu.ev = [pu.binary_window_convol_16y_mask(pu.cl_queue, (nx, 16, nz), (1, 16, 1), cdi._cl_support.data,
                                                          n, nx, ny, nz, m1, m1, wait_for=pu.ev)]
                if cdi._obj.ndim == 3:
                    pu.ev = [pu.binary_window_convol_16z_mask(pu.cl_queue, (nx, ny, 16), (1, 1, 16),
                                                              cdi._cl_support.data, n, nx, ny, nz, m1, m1,
                                                              wait_for=pu.ev)]
                # Shrunk support
                m0 = m2 if i > 0 else np.int8(1)
                pu.ev = [pu.binary_window_convol_16x_mask(pu.cl_queue, (16, ny, nz), (16, 1, 1), cdi._cl_support.data,
                                                          -n, nx, ny, nz, m0, m2, wait_for=pu.ev)]
                pu.ev = [pu.binary_window_convol_16y_mask(pu.cl_queue, (nx, 16, nz), (1, 16, 1), cdi._cl_support.data,
                                                          -n, nx, ny, nz, m2, m2, wait_for=pu.ev)]
                if cdi._obj.ndim == 3:
                    pu.ev = [pu.binary_window_convol_16z_mask(pu.cl_queue, (nx, ny, 16), (1, 1, 16),
                                                              cdi._cl_support.data, -n, nx, ny, nz, m2, m2,
                                                              wait_for=pu.ev)]

            nb = int(pu.cl_support_update_border(cdi._cl_obj_abs, cdi._cl_support, thr, self.force_shrink,
                                                 wait_for=pu.ev).get())
        else:
            nb = int(pu.cl_support_update(cdi._cl_obj_abs, cdi._cl_support, thr, self.force_shrink,
                                          wait_for=pu.ev).get())
        pu.ev = []

        if self.post_expand is not None:
            for n in self.post_expand:
                cdi = SupportExpand(n=n, update_nb_points_support=False) * cdi
            nb = int(pu.cl_nb_point_support(cdi._cl_support, wait_for=pu.ev).get())
            pu.ev = []

        if self.verbose:
            print("Nb points in support: %d (%6.3f%%), threshold=%8f  (%6.3f), nb photons=%10e"
                  % (nb, nb / cdi._obj.size * 100, thr, self.threshold_relative, tmp['x']))
        del cdi._cl_obj_abs  # Free memory
        cdi.nb_point_support = nb
        if nb == 0:
            av = 0
        else:
            av = np.sqrt(cdi.nb_photons_calc / nb)
        cdi.update_history(mode='support', support_update_threshold=thr)
        if cdi.nb_point_support <= self.min_fraction * cdi.iobs.size:
            raise SupportTooSmall("Too few points in support: %d (%6.3f%%)" % (nb, nb / cdi._obj.size * 100))
        elif cdi.nb_point_support >= self.max_fraction * cdi.iobs.size:
            raise SupportTooLarge("Too many points in support: %d (%6.3f%%)" % (nb, nb / cdi._obj.size * 100))
        return cdi


class ObjSupportStats(CLOperatorCDI):
    """
    Gather basic stats about the object: maximum and average amplitude inside the support,
    and percentage of square modulus outside the support.
    This should be evaluated ideally immediately after FourierApplyAmplitude. The result is stored
    in the CDI object's history.
    """

    def op(self, cdi):
        pu = self.processing_unit
        # Get average amplitude and maximum intensity for the object in the support (unsmoothed)
        tmp = pu.cl_obj_support_stats_red(cdi._cl_obj, cdi._cl_support, wait_for=pu.ev).get()
        pu.ev = []
        cdi._obj_max = np.sqrt(tmp['x'])
        # Percent of square modulus outside object
        cdi._obj2_out = tmp['w'] / (tmp['z'] + tmp['w'])
        cdi.update_history(mode='support')
        return cdi


class ScaleObj(CLOperatorCDI):
    """
    Scale the object according to the observed intensity. The scaling is either made against the amplitudes,
    the intensities, or the weighted intensities.
    This is only useful if a mask is used - the scale factor effectively only applies to masked intensities.
    :param method: 'I' (intensities), 'F' (amplitudes), 'wI' (weighted intensities), 'P' Poisson
    :return: nothing. The object is scaled to best match the intensities.
    """

    def __init__(self, method='I', verbose=False, lazy=False):
        """
        :param method: 'I' (intensities), 'F' (amplitudes), 'wI' (weighted intensities), 'P' (Poisson)
        :param verbose: if True, print the scale factor
        :param lazy: lazy evaluation
        """
        super(ScaleObj, self).__init__(lazy=lazy)
        self.method = method
        self.verbose = verbose

    def op(self, cdi):
        pu = self.processing_unit
        cdi = FT(scale=True) * cdi
        if self.method.lower() == 'f':
            # Scale the object to match Fourier amplitudes
            tmp = pu.cl_scale_amplitude(cdi._cl_iobs, cdi._cl_obj, wait_for=pu.ev).get()
            pu.ev = []
            scale = tmp['x'] / tmp['y']
        elif self.method.lower() == 'i':
            # Scale object to match Fourier intensities
            tmp = pu.cl_scale_intensity(cdi._cl_iobs, cdi._cl_obj, wait_for=pu.ev).get()
            pu.ev = []
            scale = np.sqrt(tmp['x'] / tmp['y'])
        elif self.method.lower() == 'p':
            # Scale object to match intensities with Poisson noise
            tmp = pu.cl_scale_intensity_poisson(cdi._cl_iobs, cdi._cl_obj, wait_for=pu.ev).get()
            pu.ev = []
            scale = tmp['x'] / tmp['y']
        else:
            # Scale object to match weighted intensities
            # Weight: 1 for null intensities, zero for masked pixels
            tmp = pu.cl_scale_weighted_intensity(cdi._cl_iobs, cdi._cl_obj, wait_for=pu.ev).get()
            pu.ev = []
            scale = np.sqrt(tmp['x'] / tmp['y'])
        scale *= pu.fft_scale(cdi._obj.shape)[1]
        cdi = IFT(scale=scale) * cdi
        if self.verbose:
            print("Scaled object by: %f [%s]" % (scale, self.method))
        return cdi


class LLK(CLOperatorCDI):
    """
    Log-likelihood reduction kernel. This is a reduction operator - it will write llk as an argument in the cdi object.
    If it is applied to a CDI instance in object space, a FT() and IFT() will be applied  to perform the calculation
    in diffraction space.
    This collects log-likelihood for Poisson, Gaussian and Euclidian noise models, and also computes the
    total calculated intensity (including in masked pixels).
    """

    def __init__(self, scale=1.0):
        """

        :param scale: the scale factor to be applied to the calculated amplitude before evaluating the
                      log-likelihood. The calculated amplitudes are left unmodified.
        """
        super(LLK, self).__init__()
        self.scale = np.float32(scale ** 2)

    def op(self, cdi):
        pu = self.processing_unit
        need_ft = cdi.in_object_space()

        if need_ft:
            cdi = FT() * cdi

        if cdi._psf_f is None:
            llk = pu.cl_llk_red(cdi._cl_iobs, cdi._cl_obj, self.scale, wait_for=pu.ev).get()
            pu.ev = []
        else:
            # TODO: better scaling to lower LLK ?
            # FFT-based convolution
            cl_icalc = cla.empty_like(cdi._cl_iobs)  # float32
            cl_icalc_f = cla.empty_like(cdi._cl_psf_f)  # Complex64, half-Hermitian array
            pu.ev = [pu.cl_square_modulus(cl_icalc, cdi._cl_obj)]
            pu.fft(cl_icalc, cl_icalc_f)
            pu.ev = [pu.cl_mult(cdi._cl_psf_f, cl_icalc_f, wait_for=pu.ev)]
            pu.ifft(cl_icalc_f, cl_icalc)
            llk = pu.cl_llk_icalc_red(cdi._cl_iobs, cl_icalc, self.scale, wait_for=pu.ev).get()
            cl_icalc_f.data.release()
            cl_icalc.data.release()

        cdi.llk_poisson = llk['x']
        cdi.llk_gaussian = llk['y']
        cdi.llk_euclidian = llk['z']
        cdi.nb_photons_calc = llk['w']
        cdi.llk_poisson_free = llk['s4']
        cdi.llk_gaussian_free = llk['s5']
        cdi.llk_euclidian_free = llk['s6']

        if need_ft:
            cdi = IFT() * cdi

        return cdi


class LLKSupport(CLOperatorCDI):
    """
    Support log-likelihood reduction kernel. Can only be used when cdi instance is object space.
    This is a reduction operator - it will write llk_support as an argument in the cdi object, and return cdi.
    """

    def op(self, cdi):
        pu = self.processing_unit
        llk = float(pu.cl_llk_reg_support_red(cdi._cl_obj, cdi._cl_support, wait_for=pu.ev).get())
        pu.ev = []
        cdi.llk_support = llk * cdi.llk_support_reg_fac
        return cdi


class DetwinSupport(CLOperatorCDI):
    """
    This operator can be used to halve the support (or restore the full support), in order to obtain an
    asymmetrical support function to favor one twin.
    """

    def __init__(self, restore=False, axis=0):
        """
        Constructor for the detwinning 
        :param restore: if True, the original support (stored in main memory) is copied back to the GPU
        :param axis: remove the half of the support along the given axis (default=0)
        """
        super(DetwinSupport, self).__init__()
        self.restore = restore
        self.axis = axis

    def op(self, cdi):
        pu = self.processing_unit
        if self.restore:
            cdi._cl_support = cla.to_device(pu.cl_queue, cdi._support, async_=False, allocator=pu.cl_mem_pool)
        else:
            # Get current support
            tmp = fftshift(cdi._cl_support.get())
            # Use center of mass to cut near middle
            c = center_of_mass(tmp)
            if self.axis == 0:
                tmp[int(round(c[0])):] = 0
            elif self.axis == 1 or tmp.ndim == 2:
                tmp[:, int(round(c[1])):] = 0
            else:
                tmp[:, :, int(round(c[2])):] = 0
            cdi._cl_support = cla.to_device(pu.cl_queue, fftshift(tmp), async_=False, allocator=pu.cl_mem_pool)
        return cdi


class DetwinHIO(CLOperatorCDI):
    """
    HIO cycles with a temporary halved support
    """

    def __init__(self, detwin_axis=0, nb_cycle=10, beta=0.9, positivity=False, zero_mask=False):
        """
        Constructor for the DetwinHIO operator

        :param detwin_axis: axis along which the detwinning will be performed. If None, a random axis is chosen
        :param nb_cycle: number of cycles to perform while using a halved support
        :param beta: the beta value for the HIO operator
        :param positivity: True or False
        :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to zero, otherwise the calculated
                          complex amplitude is kept with an optional scale factor.
        """
        super(DetwinHIO, self).__init__()
        self.detwin_axis = detwin_axis
        self.nb_cycle = nb_cycle
        self.beta = beta
        self.positivity = positivity
        self.zero_mask = zero_mask

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new RAAR operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return DetwinHIO(detwin_axis=self.detwin_axis, nb_cycle=self.nb_cycle * n, beta=self.beta,
                         positivity=self.positivity, zero_mask=self.zero_mask)

    def op(self, cdi: CDI):
        if self.detwin_axis is None:
            self.detwin_axis = randint(0, cdi.iobs.ndim)
        # print('Detwinning with %d HIO cycles and a half-support' % self.nb_cycle)
        return DetwinSupport(restore=True) * HIO(beta=self.beta, positivity=self.positivity,
                                                 zero_mask=self.zero_mask) ** self.nb_cycle \
               * DetwinSupport(axis=self.detwin_axis) * cdi


class DetwinRAAR(CLOperatorCDI):
    """
    RAAR cycles with a temporary halved support
    """

    def __init__(self, detwin_axis=0, nb_cycle=10, beta=0.9, positivity=False, zero_mask=False):
        """
        Constructor for the DetwinRAAR operator

        :param detwin_axis: axis along which the detwinning will be performed. If None, a random axis is chosen
        :param nb_cycle: number of cycles to perform while using a halved support
        :param beta: the beta value for the HIO operator
        :param positivity: True or False
        :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to zero, otherwise the calculated
                          complex amplitude is kept with an optional scale factor.
        """
        super(DetwinRAAR, self).__init__()
        self.detwin_axis = detwin_axis
        self.nb_cycle = nb_cycle
        self.beta = beta
        self.positivity = positivity
        self.zero_mask = zero_mask

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new RAAR operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return DetwinRAAR(detwin_axis=self.detwin_axis, nb_cycle=self.nb_cycle * n, beta=self.beta,
                          positivity=self.positivity, zero_mask=self.zero_mask)

    def op(self, cdi: CDI):
        # print('Detwinning with %d RAAR cycles and a half-support' % self.nb_cycle)
        if self.detwin_axis is None:
            self.detwin_axis = randint(0, cdi.iobs.ndim)
        return DetwinSupport(restore=True) * RAAR(beta=self.beta, positivity=self.positivity,
                                                  zero_mask=self.zero_mask) ** self.nb_cycle \
               * DetwinSupport(axis=self.detwin_axis) * cdi


class SupportExpand(CLOperatorCDI):
    """
    Expand (or shrink) the support using a binary window convolution.
    """

    def __init__(self, n=1, update_nb_points_support=True):
        """

        :param n: number of pixels to broaden the support, which will be done by a binary convolution with a
                  window size equal to 2*n+1 along all dimensions. if n is negative, the support is instead shrunk,
                  by performing the binary convolution and test on 1-support.
        :param update_nb_points_support: if True (the default), the number of points in the support will be calculated
                                         and stored in the object
        """
        super(SupportExpand, self).__init__()
        self.n = np.int32(n)
        self.update_nb_points_support = update_nb_points_support

    def op(self, cdi):
        pu = self.processing_unit
        if self.n == 0:
            return cdi
        nx, ny = np.int32(cdi._obj.shape[-1]), np.int32(cdi._obj.shape[-2])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi._obj.shape[0])
        else:
            nz = np.int32(1)
        pu.ev = [pu.binary_window_convol_16x(pu.cl_queue, (16, ny, nz), (16, 1, 1),
                                             cdi._cl_support.data, self.n, nx, ny, nz, wait_for=pu.ev)]
        pu.ev = [pu.binary_window_convol_16y(pu.cl_queue, (nx, 16, nz), (1, 16, 1),
                                             cdi._cl_support.data, self.n, nx, ny, nz, wait_for=pu.ev)]
        if cdi._obj.ndim == 3:
            pu.ev = [pu.binary_window_convol_16z(pu.cl_queue, (nx, ny, 16), (1, 1, 16),
                                                 cdi._cl_support.data, self.n, nx, ny, nz, wait_for=pu.ev)]

        if self.update_nb_points_support:
            cdi.nb_point_support = int(pu.cl_nb_point_support(cdi._cl_support, wait_for=pu.ev).get())
            pu.ev = []

        return cdi


class ObjConvolve(CLOperatorCDI):
    """
    Gaussian convolution of the object, produces a new array with the convoluted amplitude of the object.
    """

    def __init__(self, sigma=1):
        super(ObjConvolve, self).__init__()
        self.sigma = np.float32(sigma)

    def op(self, cdi):
        pu = self.processing_unit
        cdi._cl_obj_abs = cla.zeros(queue=pu.cl_queue,
                                    shape=cdi._cl_obj.shape, dtype=np.float32)
        nx, ny = np.int32(cdi._obj.shape[-1]), np.int32(cdi._obj.shape[-2])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi._obj.shape[0])
        else:
            nz = np.int32(1)
        pu.ev = [pu.abs_gauss_convol_16x(pu.cl_queue, (16, ny, nz), (16, 1, 1), cdi._cl_obj.data, cdi._cl_obj_abs.data,
                                         self.sigma, nx, ny, nz, wait_for=pu.ev)]
        pu.ev = [pu.gauss_convol_16y(pu.cl_queue, (nx, 16, nz), (1, 16, 1), cdi._cl_obj_abs.data,
                                     self.sigma, nx, ny, nz, wait_for=pu.ev)]
        if cdi._obj.ndim == 3:
            pu.ev = [pu.gauss_convol_16z(pu.cl_queue, (nx, ny, 16), (1, 1, 16), cdi._cl_obj_abs.data,
                                         self.sigma, nx, ny, nz, wait_for=pu.ev)]
        return cdi


class ShowCDI(ShowCDICPU):
    def __init__(self, fig_num=None, i=None):
        """

        :param fig_num: the matplotlib figure number. if None, a new figure will be created each time.
        :param i: if the object is 3D, display the ith plane (default: the center one)
        """
        super(ShowCDI, self).__init__(fig_num=fig_num, i=i)

    @staticmethod
    def get_icalc(cdi, i=None):
        # TODO: Take into account PSF convolution, if used
        if cdi.in_object_space():
            cdi = FT(scale=True) * cdi
            icalc = abs(cdi.get_obj()) ** 2
            cdi = IFT(scale=True) * cdi
        else:
            icalc = abs(cdi.get_obj()) ** 2
        if icalc.ndim == 3 and i is not None:
            return icalc[i]
        return icalc


class EstimatePSF(CLOperatorCDI):
    """
    Estimate the Point Spread Function. [OBSOLETE, replaced by InitPSF]
    """

    def __init__(self, *args, **kwargs):
        super(EstimatePSF, self).__init__()

    def op(self, cdi: CDI):
        warnings.warn("EstimatePSF() is obsolete. Use InitPSF() and update_psf=5 in "
                      "ER, HIO, RAAR instead, or use psf=5 in the algorithm chain",
                      DeprecationWarning, stacklevel=1)

        if cdi._psf_f is None:
            cdi = InitPSF(model="gaussian") * cdi

        return cdi


class PRTF(CLOperatorCDI):
    """Operator to compute the Phase Retrieval Transfer Function.
    When applied to a CDI object, it stores the result in it as cdi.prtf, cdi.prtf_freq, cdi.prtf_nyquist, cdi.prtf_nb
    """

    def __init__(self, fig_num=None, file_name=None, nb_shell=None, fig_title=None):
        """

        :param fig_num: the figure number to display the PRTF.
        :param file_name: if given, the PRTF figure will be saved to this file (should end in .png or .pdf).
        :param nb_shell: the number of shell in which to compute the PRTF. By default the shell thickness is 2 pixels
        :param fig_title: the figure title
        """
        super(PRTF, self).__init__()
        self.fig_num = fig_num
        self.file_name = file_name
        self.nb_shell = nb_shell
        self.fig_title = fig_title

    def op(self, cdi: CDI):
        pu = self.processing_unit
        need_ft = cdi.in_object_space()

        if need_ft:
            cdi = FT() * cdi

        sh = cdi.get_iobs().shape
        f_nyquist = np.int32(np.max(sh) / 2)
        if self.nb_shell is None:
            nb_shell = np.int32(f_nyquist / 2)
        else:
            nb_shell = np.int32(self.nb_shell)
        # print("PRTF: fnyquist=%5f  nb_shell=%4d" % (f_nyquist, nb_shell))
        ny, nx = sh[-2:]
        if len(sh) == 3:
            nz = sh[0]
        else:
            nz = 1
        cl_shell_obs = cla.zeros(pu.cl_queue, nb_shell, np.float32, allocator=pu.cl_mem_pool)
        cl_shell_calc = cla.zeros(pu.cl_queue, nb_shell, np.float32, allocator=pu.cl_mem_pool)
        cl_shell_nb = cla.zeros(pu.cl_queue, nb_shell, np.int32, allocator=pu.cl_mem_pool)
        cdi.prtf_freq = np.linspace(0, f_nyquist * (nb_shell - 1) / nb_shell, nb_shell) + f_nyquist / nb_shell / 2
        cdi.prtf_fnyquist = f_nyquist
        if cdi._psf_f is None:
            cl_prtf_k = CL_ElK(pu.cl_ctx, name='cl_prtf',
                               operation="prtf(i, obj, iobs, shell_calc, shell_obs, shell_nb, nb_shell, f_nyquist, nx, ny,"
                                         "nz)",
                               preamble=getks('opencl/complex.cl') + getks('cdi/opencl/prtf.cl'),
                               options=self.processing_unit.cl_options,
                               arguments="__global float2 *obj, __global float* iobs, __global float* shell_calc,"
                                         "__global float* shell_obs, __global int *shell_nb, const int nb_shell,"
                                         "const int f_nyquist, const int nx, const int ny, const int nz")
            pu.ev = [cl_prtf_k(cdi._cl_obj, cdi._cl_iobs, cl_shell_calc, cl_shell_obs, cl_shell_nb, nb_shell, f_nyquist,
                               nx, ny, nz, wait_for=pu.ev)]
        else:
            # FFT-based convolution, using half-Hermitian kernel and real->complex64 FFT
            cl_icalc = cla.empty_like(cdi._cl_iobs)  # float32
            cl_icalc_f = cla.empty_like(cdi._cl_psf_f)  # Complex64, half-Hermitian array

            pu.ev = [pu.cl_square_modulus(cl_icalc, cdi._cl_obj, wait_for=pu.ev)]
            pu.fft(cl_icalc, cl_icalc_f)
            pu.ev = [pu.cl_mult(cdi._cl_psf_f, cl_icalc_f, wait_for=pu.ev)]
            pu.ifft(cl_icalc_f, cl_icalc)
            cl_prtf_icalc_k = \
                CL_ElK(pu.cl_ctx, name='cl_prtf_icalc',
                       operation="prtf_icalc(i, icalc, iobs, shell_calc, shell_obs, shell_nb,"
                                 "nb_shell, f_nyquist, nx, ny, nz)",
                       preamble=getks('opencl/complex.cl') + getks('cdi/opencl/prtf.cl'),
                       options=self.processing_unit.cl_options,
                       arguments="__global float* icalc, __global float* iobs, __global float* shell_calc,"
                                 "__global float* shell_obs, __global int *shell_nb, const int nb_shell,"
                                 "const int f_nyquist, const int nx, const int ny, const int nz")
            pu.ev = [cl_prtf_icalc_k(cl_icalc, cdi._cl_iobs, cl_shell_calc, cl_shell_obs, cl_shell_nb,
                                     nb_shell, f_nyquist, nx, ny, nz, wait_for=pu.ev)]
        prtf = cl_shell_calc.get() / cl_shell_obs.get()
        nb = cl_shell_nb.get()
        prtf /= np.nanpercentile(prtf[nb > 0], 100)
        cdi.prtf = np.ma.masked_array(prtf, mask=nb == 0)
        cdi.prtf_nb = nb
        cdi.prtf_iobs = cl_shell_obs.get()

        plot_prtf(cdi.prtf_freq, f_nyquist, cdi.prtf, iobs_shell=cdi.prtf_iobs, nbiobs_shell=nb,
                  file_name=self.file_name, title=self.fig_title)

        if need_ft:
            cdi = IFT() * cdi

        return cdi


class InterpIobsMask(CLOperatorCDI):
    """
    Interpolate masked pixels observed intensity using inverse distance weighting
    """

    def __init__(self, d=8, n=4):
        """

        :param d: the half-distance of the interpolation, which will be done
            for pixel i from i-d to i+d along each dimension
        :param n: the weighting will be calculated as 1/d**n
        """
        super(InterpIobsMask, self).__init__()
        self.d = np.int32(d)
        self.n = np.int32(n)

    def op(self, cdi: CDI):
        pu = self.processing_unit
        nx, ny = np.int32(cdi.iobs.shape[-1]), np.int32(cdi.iobs.shape[-2])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi.iobs.shape[0])
        else:
            nz = np.int32(1)
        pu.ev = [self.processing_unit.cl_mask_interp_dist(cdi._cl_iobs, self.d, self.n, nx, ny, nz, wait_for=pu.ev)]
        cdi.iobs = cdi._cl_iobs.get()
        return cdi


class InitPSF(CLOperatorCDI):
    """ Initialise the point-spread function kernel to model
    partial coherence.
    """

    def __init__(self, model="pseudo-voigt", fwhm=1, eta=0.05, psf=None, filter=None):
        """
        Initialise the point-spread function to model the partial coherence,
        using either a Lorentzian, Gaussian or Pseuo-Voigt function.

        :param model: "lorentzian", "gaussian" or "pseudo-voigt".
            The default is a pseudo-Voigt, as it allows a relatively sharp peak while
            still keeping some tails which allow the psf kernel to be updated.
        :param fwhm: the full-width at half maximum, in pixels
        :param eta: eta value for the pseudo-Voigt (default 0.01)
        :param psf: an array of the PSF can be supplied. In that case the other parameters
            are ignored. The psf can be smaller than the iobs array size, and will be
            resized, normalised and transformed to the reciprocal space kernel used internally.
            The psf should be centred on the array centre, and will be fft-shifted
            automatically.
        :param filter: None, "hann" or "tukey" - filter for the initial PSF update. This is not
            used if the psf array is given as a parameter.
        :return: nothing. This initialises cdi._cl_psf_f, and copies the array to cdi._psf_f
        """
        super(InitPSF, self).__init__()
        self.model = model
        self.fwhm = np.float32(fwhm)
        self.eta = np.float32(eta)
        self.psf = psf
        self.filter = filter

    def op(self, cdi: CDI):
        pu = self.processing_unit
        nx = np.int32(cdi.iobs.shape[-1])
        ny = np.int32(cdi.iobs.shape[-2])
        if cdi.iobs.ndim == 3:
            nz = np.int32(cdi.iobs.shape[0])
        else:
            nz = np.int32(1)

        if cdi.iobs.ndim == 2:
            shape = (ny, nx)
            shape2 = (ny, nx // 2 + 1)
        else:
            shape = (nz, ny, nx)
            shape2 = (nz, ny, nx // 2 + 1)

        if self.psf is not None:
            self.psf = fftshift(match_shape([self.psf], shape=shape)[0])
            self.psf /= self.psf.sum()
            cl_psf = cla.to_device(self.processing_unit.cl_queue, self.psf.astype(np.float32), allocator=pu.cl_mem_pool)
        else:
            cl_psf = cla.empty(pu.cl_queue, shape, dtype=np.float32, allocator=pu.cl_mem_pool)
            if "gauss" in self.model.lower():
                pu.cl_gaussian(cl_psf, self.fwhm, nx, ny, nz)
            elif "lorentz" in self.model.lower():
                pu.cl_lorentzian(cl_psf, self.fwhm, nx, ny, nz)
            else:
                pu.cl_pseudovoigt(cl_psf, self.fwhm, self.eta, nx, ny, nz)

        # Normalise PSF
        s = pu.fft_scale(cdi._obj.shape)
        psf_sum = cla.sum(cl_psf)
        pu.cl_psf4(cl_psf, psf_sum)

        # Store the PSF FT
        cdi._cl_psf_f = cla.empty(pu.cl_queue, shape2, dtype=np.complex64, allocator=pu.cl_mem_pool)
        pu.fft(cl_psf, cdi._cl_psf_f)
        cdi._psf_f = cdi._cl_psf_f.get()

        if self.psf is None:
            cdi = UpdatePSF(filter=self.filter) ** 10 * cdi

        pu.ev = []
        return cdi
