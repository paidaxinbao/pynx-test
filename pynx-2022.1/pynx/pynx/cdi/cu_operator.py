# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['default_processing_unit', 'AutoCorrelationSupport', 'FreePU', 'FT', 'IFT', 'FourierApplyAmplitude', 'ER',
           'CF', 'HIO', 'RAAR', 'GPS', 'ML', 'SupportUpdate', 'ScaleObj', 'LLK', 'LLKSupport', 'DetwinHIO',
           'DetwinRAAR', 'SupportExpand', 'ObjConvolve', 'ShowCDI', 'EstimatePSF', 'InterpIobsMask', 'ApplyAmplitude',
           'InitPSF', 'PRTF', 'InitSupportShape', 'InitFreePixels', 'InitObjRandom', 'UpdatePSF', 'Zoom']

import warnings
import types
import gc
import psutil
import os
from random import randint
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift

from ..processing_unit.cu_processing_unit import CUProcessingUnit
from ..processing_unit.kernel_source import get_kernel_source as getks
from ..processing_unit import default_processing_unit as main_default_processing_unit
import pycuda.driver as cu_drv
import pycuda.gpuarray as cua
import pycuda.tools as cu_tools
from pycuda.elementwise import ElementwiseKernel as CU_ElK
from pycuda.reduction import ReductionKernel as CU_RedK
from pycuda.compiler import SourceModule
import pycuda.curandom as cur

from ..operator import has_attr_not_none, OperatorException, OperatorSum, OperatorPower

from .cdi import OperatorCDI, CDI, SupportTooSmall, SupportTooLarge
from .cpu_operator import ShowCDI as ShowCDICPU
from ..utils.phase_retrieval_transfer_function import plot_prtf
from .selection import match_shape
from ..utils.array import crop

my_float4 = cu_tools.get_or_register_dtype("my_float4",
                                           np.dtype([('a', '<f4'), ('b', '<f4'), ('c', '<f4'), ('d', '<f4')]))

my_float8 = cu_tools.get_or_register_dtype("my_float8",
                                           np.dtype([('a', '<f4'), ('b', '<f4'), ('c', '<f4'), ('d', '<f4'),
                                                     ('e', '<f4'), ('f', '<f4'), ('g', '<f4'), ('h', '<f4')]))


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
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s" % (str(self), str(x)))
        return self * Scale(x)

    cls.__rmul__ = __rmul__
    cls.__mul__ = __mul__


patch_method(CDI)


################################################################################################


class CUProcessingUnitCDI(CUProcessingUnit):
    """
    Processing unit in OpenCL space, for 2D and 3D CDI operations.

    Handles initializing the context and kernels.
    """

    def __init__(self):
        super(CUProcessingUnitCDI, self).__init__()
        self.cu_mem_pool = None

    def cu_init_kernels(self):
        # Elementwise kernels
        self.cu_scale = CU_ElK(name='cu_scale',
                               operation="d[i] = complexf(d[i].real() * s, d[i].imag() * s);",
                               preamble=getks('cuda/complex.cu'),
                               options=self.cu_options,
                               arguments="pycuda::complex<float> *d, const float s")

        self.cu_sum = CU_ElK(name='cu_sum',
                             operation="dest[i] += src[i]",
                             preamble=getks('cuda/complex.cu'),
                             options=self.cu_options,
                             arguments="pycuda::complex<float> *src, pycuda::complex<float> *dest")

        self.cu_mult = CU_ElK(name='cu_mult',
                              operation="dest[i] *= src[i]",
                              options=self.cu_options,
                              arguments="float *src, float *dest")

        self.cu_mult_complex = \
            CU_ElK(name='cu_mult_complex',
                   operation="dest[i] = complexf(dest[i].real() * src[i].real() - dest[i].imag() * src[i].imag(),"
                             "                   dest[i].real() * src[i].imag() + dest[i].imag() * src[i].real())",
                   preamble=getks('cuda/complex.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float> *src, pycuda::complex<float> *dest")

        self.cu_mult_scale_complex = \
            CU_ElK(name='cu_mult_scale_complex',
                   operation="dest[i] = complexf(dest[i].real() * src[i].real() - dest[i].imag() * src[i].imag(),"
                             "                   dest[i].real() * src[i].imag() + dest[i].imag() * src[i].real())"
                             "          * scale",
                   preamble=getks('cuda/complex.cu'),
                   options=self.cu_options,
                   arguments="pycuda::complex<float> *src, pycuda::complex<float> *dest, const float scale")

        # TODO: find a justification for the minimum numerator value
        self.cu_div_float = CU_ElK(name='cu_div_float', operation="dest[i] = src[i] / fmax(dest[i],1e-8f)",
                                   arguments="float *src, float *dest")

        self.cu_scale_complex = CU_ElK(name='cu_scale_complex',
                                       operation="d[i] = complexf(d[i].real() * s.real() - d[i].imag() * s.imag(),"
                                                 "                d[i].real() * s.imag() + d[i].imag() * s.real());",
                                       preamble=getks('cuda/complex.cu'),
                                       options=self.cu_options,
                                       arguments="pycuda::complex<float> *d, const pycuda::complex<float> s")

        self.cu_init_random = CU_ElK(name='cu_init_random',
                                     operation="  float s, c;"
                                               "__sincosf(phi[i] * phirange, &s, &c);"
                                               "const float aa = amin + a[i] * (amax-amin);"
                                               "const float re = aa * c;"
                                               "const float im = aa * s;"
                                               "d[i] = complexf(d[i].real()*re - d[i].imag()*im,"
                                               "                d[i].real()*im + d[i].imag()*re);",
                                     preamble=getks('cuda/complex.cu'),
                                     options=self.cu_options,
                                     arguments="pycuda::complex<float> *d, const float *a, const float *phi,"
                                               "const float amin, const float amax, const float phirange")

        self.cu_clear_free_pixels = CU_ElK(name='cu_clear_free_pixels',
                                           operation="const float v = iobs[i];"
                                                     "if(v<-0.5f && v>-1e19f) iobs[i] = -(v+1);",
                                           preamble=getks('cdi/cuda/init_free_pixels_elw.cu'),
                                           options=self.cu_options,
                                           arguments="float *iobs")

        self.cu_init_free_pixels = CU_ElK(name='cu_init_free_pixels',
                                          operation="init_free_pixels(i, ix, iy, iz, iobs, iobs0,"
                                                    "                 nx, ny, nz, nb, radius)",
                                          preamble=getks('cdi/cuda/init_free_pixels_elw.cu'),
                                          options=self.cu_options,
                                          arguments="const int* ix, const int* iy, const int* iz,"
                                                    "float *iobs, const float *iobs0,"
                                                    "const int nx, const int ny, const int nz,"
                                                    "const int nb, const int radius")

        self.cu_square_modulus = CU_ElK(name='cu_square_modulus',
                                        operation="dest[i] = dot(src[i],src[i]);",
                                        preamble=getks('cuda/complex.cu'),
                                        options=self.cu_options,
                                        arguments="float *dest, pycuda::complex<float> *src")

        self.cu_square_modulus_up = CU_ElK(name='cu_square_modulus_up',
                                           operation="PsiUp2Icalc(i, icalc, psi, nx, ny, nz, ux, uy, uz)",
                                           preamble=getks('cuda/complex.cu') + getks('cdi/cuda/apply_amplitude_elw.cu'),
                                           options=self.cu_options,
                                           arguments="float *icalc, pycuda::complex<float> *psi,"
                                                     "const int nx, const int ny, const int nz,"
                                                     "const int ux, const int uy, const int uz")

        self.cu_autocorrel_iobs = CU_ElK(name='cu_autocorrel_iobs',
                                         operation="iobsc[i] = complexf(iobs[i]>=0 ? iobs[i]: 0, 0);",
                                         preamble=getks('cuda/complex.cu'),
                                         options=self.cu_options,
                                         arguments="pycuda::complex<float> *iobsc, float *iobs")

        self.cu_apply_amplitude = CU_ElK(name='cu_apply_amplitude',
                                         operation="ApplyAmplitude(i, iobs, dcalc, scale_in, scale_out, zero_mask,"
                                                   "confidence_interval_factor, confidence_interval_factor_mask_min,"
                                                   "confidence_interval_factor_mask_max)",
                                         preamble=getks('cuda/complex.cu') + getks('cdi/cuda/apply_amplitude_elw.cu'),
                                         options=self.cu_options,
                                         arguments="float *iobs, pycuda::complex<float> *dcalc, const float scale_in,"
                                                   "const float scale_out, const signed char zero_mask,"
                                                   "const float confidence_interval_factor,"
                                                   "const float confidence_interval_factor_mask_min,"
                                                   "const float confidence_interval_factor_mask_max")

        self.cu_apply_amplitude_up = CU_ElK(name='cu_apply_amplitude_up',
                                            operation="ApplyAmplitudeUp(i, iobs, dcalc, scale_in, scale_out, zero_mask,"
                                                      "confidence_interval_factor, confidence_interval_factor_mask_min,"
                                                      "confidence_interval_factor_mask_max, nx, ny, nz, ux, uy, uz)",
                                            preamble=getks('cuda/complex.cu') + getks(
                                                'cdi/cuda/apply_amplitude_elw.cu'),
                                            options=self.cu_options,
                                            arguments="float *iobs, pycuda::complex<float> *dcalc,"
                                                      "const float scale_in, const float scale_out,"
                                                      "const signed char zero_mask,"
                                                      "const float confidence_interval_factor,"
                                                      "const float confidence_interval_factor_mask_min,"
                                                      "const float confidence_interval_factor_mask_max,"
                                                      "const int nx, const int ny, const int nz,"
                                                      "const int ux, const int uy, const int uz")

        self.cu_apply_amplitude_icalc = CU_ElK(name='cu_apply_amplitude_icalc',
                                               operation="ApplyAmplitudeIcalc(i, iobs, dcalc, icalc, scale_in,"
                                                         "scale_out, zero_mask, confidence_interval_factor,"
                                                         "confidence_interval_factor_mask_min,"
                                                         "confidence_interval_factor_mask_max)",
                                               preamble=getks('cuda/complex.cu') + getks(
                                                   'cdi/cuda/apply_amplitude_elw.cu'),
                                               options=self.cu_options,
                                               arguments="float *iobs, pycuda::complex<float> *dcalc, float *icalc,"
                                                         "const float scale_in, const float scale_out,"
                                                         "const signed char zero_mask,"
                                                         "const float confidence_interval_factor,"
                                                         "const float confidence_interval_factor_mask_min,"
                                                         "const float confidence_interval_factor_mask_max")

        self.cu_apply_amplitude_icalc_up = CU_ElK(name='cu_apply_amplitude_icalc_up',
                                                  operation="ApplyAmplitudeIcalcUp(i, iobs, dcalc, icalc, scale_in,"
                                                            "scale_out, zero_mask, confidence_interval_factor,"
                                                            "confidence_interval_factor_mask_min,"
                                                            "confidence_interval_factor_mask_max,"
                                                            "nx, ny, nz, ux, uy, uz)",
                                                  preamble=getks('cuda/complex.cu') + getks(
                                                      'cdi/cuda/apply_amplitude_elw.cu'),
                                                  options=self.cu_options,
                                                  arguments="float *iobs, pycuda::complex<float> *dcalc, float *icalc,"
                                                            "const float scale_in, const float scale_out,"
                                                            "const signed char zero_mask,"
                                                            "const float confidence_interval_factor,"
                                                            "const float confidence_interval_factor_mask_min,"
                                                            "const float confidence_interval_factor_mask_max,"
                                                            "const int nx, const int ny, const int nz,"
                                                            "const int ux, const int uy, const int uz")

        self.cu_er = CU_ElK(name='cu_er', operation="ER(i, obj, support)",
                            preamble=getks('cuda/complex.cu') + getks('cdi/cuda/cdi_elw.cu'),
                            options=self.cu_options,
                            arguments="pycuda::complex<float> *obj, signed char *support")

        self.cu_er_real = CU_ElK(name='cu_er', operation="ER_real_pos(i, obj, support)",
                                 preamble=getks('cuda/complex.cu') + getks('cdi/cuda/cdi_elw.cu'),
                                 options=self.cu_options,
                                 arguments="pycuda::complex<float> *obj, signed char *support")

        self.cu_hio = CU_ElK(name='cu_hio', operation="HIO(i, obj, obj_previous, support, beta)",
                             preamble=getks('cuda/complex.cu') + getks('cdi/cuda/cdi_elw.cu'),
                             options=self.cu_options,
                             arguments="pycuda::complex<float> *obj, pycuda::complex<float> *obj_previous, signed char *support, float beta")

        self.cu_hio_real = CU_ElK(name='cu_hio_real', operation="HIO_real_pos(i, obj, obj_previous, support, beta)",
                                  preamble=getks('cuda/complex.cu') + getks('cdi/cuda/cdi_elw.cu'),
                                  options=self.cu_options,
                                  arguments="pycuda::complex<float> *obj, pycuda::complex<float> *obj_previous, signed char *support, float beta")

        self.cu_cf = CU_ElK(name='cu_cf', operation="CF(i, obj, support)",
                            preamble=getks('cuda/complex.cu') + getks('cdi/cuda/cdi_elw.cu'),
                            options=self.cu_options,
                            arguments="pycuda::complex<float> *obj, signed char *support")

        self.cu_cf_real = CU_ElK(name='cu_cf_real', operation="CF_real_pos(i, obj, support)",
                                 preamble=getks('cuda/complex.cu') + getks('cdi/cuda/cdi_elw.cu'),
                                 options=self.cu_options,
                                 arguments="pycuda::complex<float> *obj, signed char *support")

        self.cu_raar = CU_ElK(name='cu_raar', operation="RAAR(i, obj, obj_previous, support, beta)",
                              preamble=getks('cuda/complex.cu') + getks('cdi/cuda/cdi_elw.cu'),
                              options=self.cu_options,
                              arguments="pycuda::complex<float> *obj, pycuda::complex<float> *obj_previous, signed char *support, float beta")

        self.cu_raar_real = CU_ElK(name='cu_raar_real', operation="RAAR_real_pos(i, obj, obj_previous, support, beta)",
                                   preamble=getks('cuda/complex.cu') + getks('cdi/cuda/cdi_elw.cu'),
                                   options=self.cu_options,
                                   arguments="pycuda::complex<float> *obj, pycuda::complex<float> *obj_previous, signed char *support, float beta")

        self.cu_ml_poisson_psi_gradient = CU_ElK(name='cu_ml_poisson_psi_gradient',
                                                 operation="PsiGradient(i, psi, dpsi, iobs, nx, ny, nz)",
                                                 preamble=getks('cuda/complex.cu') + getks(
                                                     'cdi/cuda/cdi_ml_poisson_elw.cu'),
                                                 options=self.cu_options,
                                                 arguments="pycuda::complex<float>* psi, pycuda::complex<float>* dpsi,"
                                                           "float* iobs, const int nx, const int ny, const int nz")

        self.cu_ml_poisson_reg_support_gradient = CU_ElK(name='cu_ml_poisson_psi_gradient',
                                                         operation="RegSupportGradient(i, obj, objgrad, support, reg_fac)",
                                                         preamble=getks('cuda/complex.cu') + getks(
                                                             'cdi/cuda/cdi_ml_poisson_elw.cu'),
                                                         options=self.cu_options,
                                                         arguments="pycuda::complex<float>* obj, pycuda::complex<float>* objgrad, signed char* support, const float reg_fac")

        self.cu_ml_poisson_cg_linear = CU_ElK(name='cu_ml_poisson_psi_gradient', operation="A[i] = a*A[i] + b*B[i]",
                                              preamble=getks('cuda/complex.cu'), options=self.cu_options,
                                              arguments="const float a, pycuda::complex<float> *A, const float b, pycuda::complex<float> *B")

        self.cu_gps1 = CU_ElK(name='cu_gps1',
                              operation="GPS1(i, obj, z, t, sigma_o, nx, ny, nz)",
                              preamble=getks('cuda/complex.cu') + getks('cdi/cuda/gps_elw.cu'),
                              options=self.cu_options,
                              arguments="pycuda::complex<float>* obj, pycuda::complex<float>* z,"
                                        "const float t, const float sigma_o, const int nx, const int ny, const int nz")

        self.cu_gps2 = CU_ElK(name='cu_gps2',
                              operation="GPS2(i, obj, z, epsilon)",
                              preamble=getks('cuda/complex.cu') + getks('cdi/cuda/gps_elw.cu'),
                              options=self.cu_options,
                              arguments="pycuda::complex<float>* obj, pycuda::complex<float>* z,"
                                        "const float epsilon")

        self.cu_gps3 = CU_ElK(name='cu_gps3',
                              operation="GPS3(i, obj, z)",
                              preamble=getks('cuda/complex.cu') + getks('cdi/cuda/gps_elw.cu'),
                              options=self.cu_options,
                              arguments="pycuda::complex<float>* obj, pycuda::complex<float>* z")

        self.cu_gps4 = CU_ElK(name='cu_gps4',
                              operation="GPS4(i, obj, y, support, s, sigma_f, positivity, nx, ny, nz)",
                              preamble=getks('cuda/complex.cu') + getks('cdi/cuda/gps_elw.cu'),
                              options=self.cu_options,
                              arguments="pycuda::complex<float>* obj, pycuda::complex<float>* y, signed char *support,"
                                        "const float s, const float sigma_f, signed char positivity,"
                                        "const int nx, const int ny, const int nz")

        self.cu_mask_interp_dist = CU_ElK(name='cu_mask_interp_dist',
                                          operation="mask_interp_dist(i, iobs, k, dist_n, nx, ny, nz)",
                                          preamble=getks('cuda/mask_interp_dist.cu'),
                                          options=self.cu_options,
                                          arguments="float *iobs, const int k, const int dist_n,"
                                                    "const int nx, const int ny, const int nz")

        self.cu_gaussian = CU_ElK(name='cu_gaussian',
                                  operation="Gaussian(i, d, fwhm, nx, ny, nz)",
                                  preamble=getks('cdi/cuda/psf_elw.cu'),
                                  options=self.cu_options,
                                  arguments="float *d, const float fwhm,"
                                            "const int nx, const int ny, const int nz")

        self.cu_lorentzian = CU_ElK(name='cu_lorentzian',
                                    operation="Lorentzian(i, d, fwhm, nx, ny, nz)",
                                    preamble=getks('cdi/cuda/psf_elw.cu'),
                                    options=self.cu_options,
                                    arguments="float *d, const float fwhm,"
                                              "const int nx, const int ny, const int nz")

        self.cu_pseudovoigt = CU_ElK(name='cu_pseudovoigt',
                                     operation="PseudoVoigt(i, d, fwhm, eta, nx, ny, nz)",
                                     preamble=getks('cdi/cuda/psf_elw.cu'),
                                     options=self.cu_options,
                                     arguments="float *d, const float fwhm, const float eta,"
                                               "const int nx, const int ny, const int nz")

        # Using biweight does not help. Nor does using the free pixels true intensity
        # Why set to zero instead of 1 (better stability in practice)
        self.cu_psf1 = CU_ElK(name='cu_psf1',
                              # Use 'scale' if iobs==0, equivalent to iobs=icalc
                              operation="icalc[i] = iobs[i] >=0 ? iobs[i] / (icalc[i] * scale): 1.0f",
                              # operation="const float r = iobs[i] >=0 ? iobs[i] / icalc[i] : scale;"
                              #           "icalc[i] = tukey_biweight_log10(r/scale,10)*scale;",
                              preamble=getks('cuda/biweight.cu'),
                              options=self.cu_options,
                              arguments="float *iobs, float *icalc, const float scale")

        # for half-hermition arrays: d1 * mirror(d2) = d1 * d2.conj()
        self.cu_psf2 = CU_ElK(name='cu_psf2',
                              operation="d1[i] = complexf(d1[i].real() * d2[i].real() + d1[i].imag() * d2[i].imag(),"
                                        "d1[i].imag() * d2[i].real() - d1[i].real() * d2[i].imag()) * scale",
                              preamble=getks('cuda/complex.cu'),
                              options=self.cu_options,
                              arguments="pycuda::complex<float>* d1, pycuda::complex<float> *d2, const float scale")

        self.cu_psf3 = CU_ElK(name='cu_psf3',
                              operation="psf[i] *= fmaxf(d[i], 1e-6f) * scale",  # TODO: robustness ?
                              options=self.cu_options,
                              arguments="float *psf, float *d, const float scale")

        self.cu_psf3_hann = CU_ElK(name='cu_psf3_hann',
                                   operation="PSF3_Hann(i, psf, d, nx, ny, nz, scale)",
                                   preamble=getks("cdi/cuda/psf_elw.cu"),
                                   options=self.cu_options,
                                   arguments="float* psf, float *d, const int nx, const int ny,"
                                             "const int nz, const float scale")

        self.cu_psf3_tukey = CU_ElK(name='cu_psf3_tukey',
                                    operation="PSF3_Tukey(i, psf, d, alpha, nx, ny, nz, scale)",
                                    preamble=getks("cdi/cuda/psf_elw.cu"),
                                    options=self.cu_options,
                                    arguments="float* psf, float *d, const float alpha, const int nx,"
                                              "const int ny, const int nz, const float scale")

        self.cu_psf4 = CU_ElK(name='cu_psf4',
                              operation="psf[i] /= psf_sum[0]",
                              options=self.cu_options,
                              arguments="float *psf, float *psf_sum")

        self.cu_detwinx = CU_ElK(name='cu_detwinx',
                                 operation="DetwinX(i, support, c, nx)",
                                 preamble=getks('cdi/cuda/detwin_elw.cu'),
                                 options=self.cu_options,
                                 arguments="char *support, const int c, const int nx")

        self.cu_detwiny = CU_ElK(name='cu_detwiny',
                                 operation="DetwinY(i, support, c, nx, ny)",
                                 preamble=getks('cdi/cuda/detwin_elw.cu'),
                                 options=self.cu_options,
                                 arguments="char *support, const int c, const int nx, const int ny")

        self.cu_detwinz = CU_ElK(name='cu_detwinz',
                                 operation="DetwinZ(i, support, c, nx, ny, nz)",
                                 preamble=getks('cdi/cuda/detwin_elw.cu'),
                                 options=self.cu_options,
                                 arguments="char *support, const int c, const int nx, const int ny, const int nz")

        self.cu_zoom_complex = CU_ElK(name='cu_zoom_complex',
                                      operation="zoom_complex(i, d, f, axis, center, nx, ny, nz, fill, norm)",
                                      preamble=getks('cuda/complex.cu') + getks('cuda/zoom_complex.cu'),
                                      options=self.cu_options,
                                      arguments="pycuda::complex<float> *d, const float f, const int axis,"
                                                "const int center, const int nx, const int ny, const int nz,"
                                                "const float fill, const bool norm")

        self.cu_zoom_sup = CU_ElK(name='cu_zoom_support',
                                  operation="zoom_support(i, d, f, axis, center, nx, ny, nz, fill)",
                                  preamble=getks('cuda/complex.cu') + getks('cuda/zoom_support.cu'),
                                  options=self.cu_options,
                                  arguments="signed char *d, const float f, const int axis,"
                                            "const int center, const int nx, const int ny, const int nz,"
                                            "const signed char fill")

        # Reduction kernels
        self.cu_nb_point_support = CU_RedK(np.int32, neutral="0", reduce_expr="a+b",
                                           map_expr="support[i]",
                                           options=self.cu_options,
                                           arguments="signed char *support")

        self.cu_llk_red = CU_RedK(my_float8, neutral="0",
                                  reduce_expr="a+b",
                                  map_expr="LLKAll(i, iobs, psi, scale)",
                                  preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu')
                                           + getks('cdi/cuda/llk_red.cu'),
                                  options=self.cu_options,
                                  arguments="float *iobs, pycuda::complex<float> *psi, const float scale")

        self.cu_llk_up_red = CU_RedK(my_float8, neutral="0",
                                     reduce_expr="a+b",
                                     map_expr="LLKAllUp(i, iobs, psi, scale, nx, ny, nz, ux, uy, uz)",
                                     preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu')
                                              + getks('cdi/cuda/llk_red.cu'),
                                     options=self.cu_options,
                                     arguments="float *iobs, pycuda::complex<float> *psi, const float scale,"
                                               "const int nx, const int ny, const int nz,"
                                               "const int ux, const int uy, const int uz")

        self.cu_llk_icalc_red = CU_RedK(my_float8, neutral="0",
                                        reduce_expr="a+b",
                                        map_expr="LLKAllIcalc(i, iobs, icalc, scale)",
                                        preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu')
                                                 + getks('cdi/cuda/llk_red.cu'),
                                        options=self.cu_options,
                                        arguments="float *iobs, float *icalc, const float scale")

        self.cu_llk_reg_support_red = CU_RedK(np.float32, neutral="0", reduce_expr="a+b",
                                              map_expr="LLKRegSupport(obj[i], support[i])",
                                              preamble=getks('cuda/complex.cu') + getks(
                                                  'cdi/cuda/cdi_llk_reg_support_red.cu'),
                                              options=self.cu_options,
                                              arguments="pycuda::complex<float> *obj, signed char *support")

        # Polak-Ribière CG coefficient
        self.cu_cg_polak_ribiere_red = CU_RedK(np.complex64, neutral="complexf(0,0)",
                                               reduce_expr="a+b",
                                               map_expr="PolakRibiereComplex(grad[i], lastgrad[i])",
                                               preamble=getks('cuda/complex.cu') + getks(
                                                   'cuda/cg_polak_ribiere_red.cu'),
                                               options=self.cu_options,
                                               arguments="pycuda::complex<float> *grad, pycuda::complex<float> *lastgrad")
        # Line minimization factor for CG
        self.cdi_ml_poisson_gamma_red = CU_RedK(np.complex64, neutral="complexf(0,0)",
                                                reduce_expr="a+b",
                                                map_expr="Gamma(obs, psi, dpsi, i)",
                                                preamble=getks('cuda/complex.cu') + getks(
                                                    'cdi/cuda/cdi_ml_poisson_red.cu'),
                                                options=self.cu_options,
                                                arguments="float *obs, pycuda::complex<float> *psi, pycuda::complex<float> *dpsi")

        self.cdi_ml_poisson_gamma_support_red = CU_RedK(np.complex64, neutral="complexf(0,0)",
                                                        reduce_expr="a+b",
                                                        map_expr="GammaSupport(obs, psi, dpsi, obj, dobj, support, reg_fac, i)",
                                                        preamble=getks('cuda/complex.cu') + getks(
                                                            'cdi/cuda/cdi_ml_poisson_red.cu'),
                                                        options=self.cu_options,
                                                        arguments="float *obs, pycuda::complex<float> *psi, pycuda::complex<float> *dpsi,"
                                                                  "pycuda::complex<float> *obj, pycuda::complex<float> *dobj, signed char *support, "
                                                                  "const float reg_fac")

        self.cu_support_update = CU_RedK(np.int32, neutral="0", reduce_expr="a+b",
                                         map_expr="SupportUpdate(i, d, support, threshold, force_shrink)",
                                         preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') +
                                                  getks('cdi/cuda/cdi_support_update_red.cu'),
                                         options=self.cu_options,
                                         arguments="float *d, signed char *support, const float threshold, const bool force_shrink")

        self.cu_support_update_border = CU_RedK(np.int32, neutral="0", reduce_expr="a+b",
                                                map_expr="SupportUpdateBorder(i, d, support, threshold, force_shrink)",
                                                preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') +
                                                         getks('cdi/cuda/cdi_support_update_red.cu'),
                                                options=self.cu_options,
                                                arguments="float *d, signed char *support, const float threshold,"
                                                          "const bool force_shrink")

        # Init support from autocorrelation array
        self.cu_support_init = CU_RedK(np.int32, neutral="0", reduce_expr="a+b",
                                       map_expr="SupportInit(i, d, support, threshold)",
                                       preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') +
                                                getks('cdi/cuda/cdi_support_update_red.cu'),
                                       options=self.cu_options,
                                       arguments="pycuda::complex<float> *d, signed char *support, const float threshold")

        # Calculate the average and maximum square modulus in the support (complex object)
        # TODO: avoid using direct access to _M_re and _M_im - unfortunately trying to use real() and imag()
        #  returns a compilation error, because of the volatile keyword used for reduction.
        self.cu_average_max_red = CU_RedK(np.complex64, neutral="complexf(0,0)",
                                          reduce_expr="complexf(a._M_re+b._M_re, fmaxf(a._M_im,b._M_im))",
                                          map_expr="complexf(sqrtf(dot(obj[i], obj[i])), dot(obj[i], obj[i]))"
                                                   "* (float)(support[i])",
                                          options=self.cu_options, preamble=getks('cuda/complex.cu'),
                                          arguments="pycuda::complex<float> *obj, signed char *support")

        # Calculate the average and maximum square modulus of the object in the support,
        # and the integrated square modulus inside and outside the support
        self.cu_obj_support_stats_red = \
            CU_RedK(my_float4, neutral="my_float4(0)",
                    reduce_expr="my_float4(a.x+b.x, fmaxf(a.y,b.y), a.z+b.z, a.w+b.w)",
                    map_expr="ObjSupportStats(i, obj, support)",
                    options=self.cu_options, preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') +
                                                      getks('cdi/cuda/cdi_support_update_red.cu'),
                    arguments="pycuda::complex<float> *obj, signed char *support")

        # Calculate the average amplitude and maximum intensity in the support (object amplitude)
        self.cu_average_max_abs_red = CU_RedK(np.complex64, neutral="complexf(0,0)",
                                              reduce_expr="complexf(a._M_re+b._M_re, fmaxf(a._M_im,b._M_im))",
                                              map_expr="complexf(obj[i], obj[i] * obj[i]) * (float)(support[i])",
                                              options=self.cu_options, preamble=getks('cuda/complex.cu'),
                                              arguments="float *obj, signed char *support")

        # Calculate the root mean square and maximum intensity in the support (object amplitude)
        self.cu_rms_max_abs_red = CU_RedK(np.complex64, neutral="complexf(0,0)",
                                          reduce_expr="complexf(a._M_re+b._M_re, fmaxf(a._M_im,b._M_im))",
                                          map_expr="complexf(obj[i] * obj[i], obj[i] * obj[i]) * (float)(support[i])",
                                          options=self.cu_options, preamble=getks('cuda/complex.cu'),
                                          arguments="float *obj, signed char *support")

        self.cu_scale_amplitude = CU_RedK(np.complex64, neutral="complexf(0,0)", reduce_expr="a+b",
                                          map_expr="ScaleAmplitude(i, iobs, calc)",
                                          preamble=getks('cuda/complex.cu') + getks('cdi/cuda/scale_obs_calc_red.cu'),
                                          options=self.cu_options,
                                          arguments="float * iobs, pycuda::complex<float> *calc")

        self.cu_scale_intensity = CU_RedK(np.complex64, neutral="complexf(0,0)", reduce_expr="a+b",
                                          map_expr="ScaleIntensity(i, iobs, calc)",
                                          preamble=getks('cuda/complex.cu') + getks('cdi/cuda/scale_obs_calc_red.cu'),
                                          options=self.cu_options,
                                          arguments="float * iobs, pycuda::complex<float> *calc")

        self.cu_scale_intensity_poisson = CU_RedK(np.complex64, neutral="complexf(0,0)", reduce_expr="a+b",
                                                  map_expr="ScaleIntensityPoisson(i, iobs, calc)",
                                                  preamble=getks('cuda/complex.cu') + getks(
                                                      'cdi/cuda/scale_obs_calc_red.cu'),
                                                  options=self.cu_options,
                                                  arguments="float * iobs, pycuda::complex<float> *calc")

        self.cu_scale_weighted_intensity = CU_RedK(np.complex64, neutral="complexf(0,0)", reduce_expr="a+b",
                                                   map_expr="ScaleWeightedIntensity(i, iobs, calc)",
                                                   preamble=getks('cuda/complex.cu') + getks(
                                                       'cdi/cuda/scale_obs_calc_red.cu'),
                                                   options=self.cu_options,
                                                   arguments="float * iobs, pycuda::complex<float> *calc")

        self.cu_center_mass_fftshift_complex = CU_RedK(my_float4, neutral="my_float4(0)", name="cu_center_mass_complex",
                                                       reduce_expr="a+b",
                                                       preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') +
                                                                getks('cuda/center_mass_red.cu'),
                                                       options=self.cu_options,
                                                       map_expr="center_mass_fftshift_complex(i, d, nx, ny, nz, power)",
                                                       arguments="pycuda::complex<float> *d, const int nx,"
                                                                 "const int ny, const int nz, const int power")

        # Absolute maximum of complex array
        self.cu_max_red = CU_RedK(np.float32, neutral="0", reduce_expr="a > b ? a : b", map_expr="abs(d[i])",
                                  options=self.cu_options, preamble=getks('cuda/complex.cu'),
                                  arguments="pycuda::complex<float> *d")

        # Convolution kernels for support update (Gaussian)
        conv16c2f_mod = SourceModule(getks('cuda/complex.cu') + getks('cuda/convolution16c2f.cu'),
                                     options=self.cu_options)
        self.abs_gauss_convol_16x = conv16c2f_mod.get_function("gauss_convolc2f_16x")
        conv16f_mod = SourceModule(getks('cuda/complex.cu') + getks('cuda/convolution16f.cu'), options=self.cu_options)
        self.gauss_convol_16y = conv16f_mod.get_function("gauss_convolf_16y")
        self.gauss_convol_16z = conv16f_mod.get_function("gauss_convolf_16z")

        # Same using a binary window
        conv16b_mod = SourceModule(getks('cuda/convolution16b.cu'), options=self.cu_options)
        self.binary_window_convol_16x = conv16b_mod.get_function("binary_window_convol_16x")
        self.binary_window_convol_16y = conv16b_mod.get_function("binary_window_convol_16y")
        self.binary_window_convol_16z = conv16b_mod.get_function("binary_window_convol_16z")
        self.binary_window_convol_16x_mask = conv16b_mod.get_function("binary_window_convol_16x_mask")
        self.binary_window_convol_16y_mask = conv16b_mod.get_function("binary_window_convol_16y_mask")
        self.binary_window_convol_16z_mask = conv16b_mod.get_function("binary_window_convol_16z_mask")

        # Init memory pool
        self.cu_mem_pool = cu_tools.DeviceMemoryPool()


"""
The default processing unit 
"""
default_processing_unit = CUProcessingUnitCDI()


class CUOperatorCDI(OperatorCDI):
    """
    Base class for a operators on CDI objects using OpenCL
    """

    def __init__(self, processing_unit=None, **kwargs):
        super(CUOperatorCDI, self).__init__(**kwargs)

        self.Operator = CUOperatorCDI
        self.OperatorSum = CUOperatorCDISum
        self.OperatorPower = CUOperatorCDIPower

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

    def apply_ops_mul(self, cdi: CDI):
        """
        Apply the series of operators stored in self.ops to a CDI object.
        In this version the operators are applied one after the other to the same CDI object (multiplication)

        :param cdi: the CDI object to which the operators will be applied.
        :return: the CDI object, after application of all the operators in sequence
        """
        try:
            return super(CUOperatorCDI, self).apply_ops_mul(cdi)
        except cu_drv.MemoryError as ex:
            cdi = MemUsage() * cdi
            raise OperatorException("A pycuda memory error occured") from ex

    def prepare_data(self, cdi):
        # Make sure data is already in CUDA space, otherwise transfer it
        if cdi._timestamp_counter > cdi._cu_timestamp_counter:
            obj, support, iobs = cdi._to_gpu()

            # Re-use allocated memory when possible
            if has_attr_not_none(cdi, "_cu_obj"):
                if cdi._cu_obj.shape == obj.shape:
                    cu_drv.memcpy_htod_async(dest=cdi._cu_obj.gpudata, src=obj)
                else:
                    del cdi._cu_obj
                    gc.collect()
            if has_attr_not_none(cdi, "_cu_obj") is False:
                cdi._cu_obj = cua.to_gpu(obj, allocator=self.processing_unit.cu_mem_pool.allocate)

            if has_attr_not_none(cdi, "_cu_support"):
                if cdi._cu_support.shape == support.shape:
                    cu_drv.memcpy_htod_async(dest=cdi._cu_support.gpudata, src=support)
                else:
                    del cdi._cu_support
                    gc.collect()
                    cdi._cu_support = cua.to_gpu(support, allocator=self.processing_unit.cu_mem_pool.allocate)
            else:
                cdi._cu_support = cua.to_gpu(support, allocator=self.processing_unit.cu_mem_pool.allocate)

            if has_attr_not_none(cdi, "_cu_iobs"):
                if cdi._cu_iobs.shape == iobs.shape:
                    cu_drv.memcpy_htod_async(dest=cdi._cu_iobs.gpudata, src=iobs)
                else:
                    del cdi._cu_iobs
                    gc.collect()
                    cdi._cu_iobs = cua.to_gpu(iobs, allocator=self.processing_unit.cu_mem_pool.allocate)
            else:
                cdi._cu_iobs = cua.to_gpu(iobs, allocator=self.processing_unit.cu_mem_pool.allocate)

            if cdi._psf_f is None:
                cdi._cu_psf_f = None
            else:
                # We keep the Fourier Transform of the PSF convolution kernel in GPU memory (half-Hermitian array)
                cdi._cu_psf_f = cua.to_gpu(cdi._psf_f, allocator=self.processing_unit.cu_mem_pool.allocate)
                pu = self.processing_unit

            cdi._cu_timestamp_counter = cdi._timestamp_counter
        if has_attr_not_none(cdi, '_cu_obj_view') is False:
            cdi._cu_obj_view = {}

    def timestamp_increment(self, cdi):
        cdi._cu_timestamp_counter += 1

    def view_register(self, obj):
        """
        Creates a new unique view key in an object. When finished with this view, it should be de-registered
        using view_purge. Note that it only reserves the key, but does not create the view.
        :return: an integer value, which corresponds to yet-unused key in the object's view.
        """
        i = 1
        while i in obj._cu_obj_view:
            i += 1
        obj._cu_obj_view[i] = None
        return i

    def view_copy(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._cu_obj
        else:
            src = obj._cu_obj_view[i_source]
        if i_dest == 0:
            obj._cu_obj = cua.empty_like(src)
            dest = obj._cu_obj
        else:
            obj._cu_obj_view[i_dest] = cua.empty_like(src)
            dest = obj._cu_obj_view[i_dest]
        cu_drv.memcpy_dtod(dest=dest.gpudata, src=src.gpudata, size=dest.nbytes)

    def view_swap(self, obj, i1, i2):
        if i1 != 0:
            if i1 not in obj._cu_obj_view:
                # Create dummy value, assume a copy will be made later
                obj._cu_obj_view[i1] = None
        if i2 != 0:
            if i2 not in obj._cu_obj_view:
                # Create dummy value, assume a copy will be made later
                obj._cu_obj_view[i2] = None
        if i1 == 0:
            obj._cu_obj, obj._cu_obj_view[i2] = obj._cu_obj_view[i2], obj._cu_obj
        elif i2 == 0:
            obj._cu_obj, obj._cu_obj_view[i1] = obj._cu_obj_view[i1], obj._cu_obj
        else:
            obj._cu_obj_view[i1], obj._cu_obj_view[i2] = obj._cu_obj_view[i2], obj._cu_obj_view[i1]

    def view_sum(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._cu_obj
        else:
            src = obj._cu_obj_view[i_source]
        if i_dest == 0:
            dest = obj._cu_obj
        else:
            dest = obj._cu_obj_view[i_dest]
        self.processing_unit.cu_sum(src, dest)

    def view_purge(self, obj, i):
        if i is not None:
            del obj._cu_obj_view[i]
        elif has_attr_not_none(obj, '_cu_obj_view'):
            del obj._cu_obj_view
            self.processing_unit.synchronize()  # is this useful ?


# The only purpose of this class is to make sure it inherits from CUOperatorCDI and has a processing unit
class CUOperatorCDISum(OperatorSum, CUOperatorCDI):
    def __init__(self, op1, op2):
        if np.isscalar(op1):
            op1 = Scale(op1)
        if np.isscalar(op2):
            op2 = Scale(op2)
        if isinstance(op1, CUOperatorCDI) is False or isinstance(op2, CUOperatorCDI) is False:
            raise OperatorException(
                "ERROR: cannot add a CUOperatorCDI with a non-CUOperatorCDI: %s + %s" % (str(op1), str(op2)))
        # We can only have a sum of two CLOperatorWavefront, so they must have a processing_unit attribute.
        CUOperatorCDI.__init__(self, op1.processing_unit)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CUOperatorCDI
        self.OperatorSum = CUOperatorCDISum
        self.OperatorPower = CUOperatorCDIPower
        self.prepare_data = types.MethodType(CUOperatorCDI.prepare_data, self)
        self.timestamp_increment = types.MethodType(CUOperatorCDI.timestamp_increment, self)
        self.view_copy = types.MethodType(CUOperatorCDI.view_copy, self)
        self.view_swap = types.MethodType(CUOperatorCDI.view_swap, self)
        self.view_sum = types.MethodType(CUOperatorCDI.view_sum, self)
        self.view_purge = types.MethodType(CUOperatorCDI.view_purge, self)


# The only purpose of this class is to make sure it inherits from CLOperatorWavefront and has a processing unit
class CUOperatorCDIPower(OperatorPower, CUOperatorCDI):
    def __init__(self, op, n):
        CUOperatorCDI.__init__(self, op.processing_unit)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CUOperatorCDI
        self.OperatorSum = CUOperatorCDISum
        self.OperatorPower = CUOperatorCDIPower
        self.prepare_data = types.MethodType(CUOperatorCDI.prepare_data, self)
        self.timestamp_increment = types.MethodType(CUOperatorCDI.timestamp_increment, self)
        self.view_copy = types.MethodType(CUOperatorCDI.view_copy, self)
        self.view_swap = types.MethodType(CUOperatorCDI.view_swap, self)
        self.view_sum = types.MethodType(CUOperatorCDI.view_sum, self)
        self.view_purge = types.MethodType(CUOperatorCDI.view_purge, self)


class MemUsage(CUOperatorCDI):
    """
    Print memory usage of current process (RSS on host) and used GPU memory
    """

    def __init__(self, verbose=True):
        super(MemUsage, self).__init__()
        self.verbose = verbose

    def op(self, p: CDI):
        """

        :param p: the ptycho object this operator applies to
        :return: the updated ptycho object
        """
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        gpu_mem = 0
        print("Report on memory used by the CDI object:")

        for o in dir(p):
            if isinstance(p.__getattribute__(o), cua.GPUArray):
                if self.verbose:
                    print("   %40s %10.3f Mbytes" % (o, p.__getattribute__(o).nbytes / 1e6))
                gpu_mem += p.__getattribute__(o).nbytes

        d = self.processing_unit.cu_device
        print("  GPU used: %s [%4d Mbytes]" % (d.name(), int(round(d.total_memory() // 2 ** 20))))
        print("  Total Mem Usage: RSS= %6.1f Mbytes (process), GPU Mem= %6.1f Mbytes (calculated)" %
              (rss / 1024 ** 2, gpu_mem / 1024 ** 2))
        mp = self.processing_unit.cu_mem_pool
        print("  Memory pool: %4d Mbytes (active) %4d Mbytes (managed) %d held blocks, %d active blocks" %
              (int(round(mp.managed_bytes // 2 ** 20)), int(round(mp.active_bytes // 2 ** 20)),
               mp.held_blocks, mp.active_blocks))
        return p

    def prepare_data(self, p):
        # Overriden to avoid transferring any data to GPU
        pass

    def timestamp_increment(self, p):
        # This operator does nothing
        pass


class AutoCorrelationSupport(CUOperatorCDI):
    """
    Operator to calculate an initial support from the auto-correlation function of the observed intensity.
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
        pu = self.processing_unit
        # We use temp arrays so free memory explicitly
        pu.cu_mem_pool.free_held()
        t = self.threshold
        if isinstance(t, list) or isinstance(t, tuple):
            t = np.random.uniform(t[0], t[1])

        tmp = cua.empty(cdi._cu_iobs.shape, dtype=np.complex64, allocator=pu.cu_mem_pool.allocate)
        # Copy Iobs to complex array (we could also do a real->complex transform, C2C is easier)
        pu.cu_autocorrel_iobs(tmp, cdi._cu_iobs)
        pu.fft(tmp, tmp)
        thres = np.float32(pu.cu_max_red(tmp).get() * t)
        cdi.nb_point_support = int(pu.cu_support_init(tmp, cdi._cu_support, thres).get())
        if self.verbose:
            print('AutoCorrelation: %d pixels in support (%6.2f%%), threshold = %f (relative = %5.3f)' %
                  (cdi.nb_point_support, cdi.nb_point_support * 100 / tmp.size, thres, t))
        tmp.gpudata.free()
        del tmp
        pu.cu_mem_pool.free_held()
        # Apply support to object
        self.processing_unit.cu_er(cdi._cu_obj, cdi._cu_support)

        if self.scale is True:
            return ScaleObj(method='F') * cdi
        elif isinstance(self.scale, str):
            return ScaleObj(method=self.scale) * cdi
        return cdi


class InitSupportShape(CUOperatorCDI):
    """Init the support using a description of the shape or a formula. An alternative
    to AutoCorrelationSupport when the centre of the diffraction is hidden behind
    a beamstop.
    """

    def __init__(self, shape="circle", size=None, formula=None, verbose=False, lazy=False):
        """

        :param shape: either "circle" (same a "sphere"), square (same as "cube").
            Ignored if formula is not None.
        :param size: the radius of the circle/sphere or the half-size of the square/cube.
            Ignored if formula is not None.
        :param formula: a formula giving the shape of the initial support as a function
            of x,y,z - coordinates in pixels from the center of the object array.
            This only allows: sqrt, abs, +, -, *, /   (notably ** is not accepted).
            Values not equal to zero or False will be inside the support.
            Example acceptable formulas (to be interpreted either in python, CUDA or OpenCL):
            formula="(x*x + y*y + z*z)<50"
            formula="(x*x/(20*20) + y*y/(30*30) + z*z/(10*10) )<1"
            formula="(abs(x)<20) * (abs(y)<30) * (abs(z)<25)"
        :param verbose: to be or not to be verbose, that is the parameter
        :param lazy: if True, this will be queued for later execution in the cdi object
        """
        super(InitSupportShape, self).__init__(lazy=lazy)
        if formula is None:
            assert size is not None, "InitSupportShape: a size must be given"
            if np.isscalar(size):
                sz, sy, sx = size, size, size
            elif len(size) == 2:
                sy, sx = size
                sz = 1  # won't matter, z=0
            elif len(size) == 3:
                sz, sy, sx = size
            if shape.lower() in ["circle", "sphere"]:
                formula = "(x*x/%f + y*y/%f + z*z/%f)<1" % (sx ** 2, sy ** 2, sz ** 2)
            elif shape.lower() in ["square", "cube"]:
                formula = "(abs(x)<%f) * (abs(y)<%f) * (abs(z)<%f)" % (sx, sy, sz)
            else:
                raise OperatorException("InitSupportShape: shape should be among: circle, sphere, square or cube")
        src = """
        __device__ void init_support_formula(int i, signed char *support,
                                             const int nx, const int ny, const int nz)
        {
            // x,y,z coordinates in the fft-shifted array
            const int ix = i %% nx;
            const int iy = (i %% (nx * ny)) / nx;
            const int iz = i / (nx * ny);
            const float x = ix - nx * (ix>=(nx/2));
            const float y = iy - ny * (iy>=(ny/2));
            const float z = iz - nz * (iz>=(nz/2)) * (nz>1);
            support[i] = (%s)>0;
        }
        """ % formula
        # print(formula)
        # print(src)
        self.kernel = \
            CU_ElK(name='cu_init_support_formula',
                   operation="init_support_formula(i, support, nx, ny, nz)",
                   preamble=src, options=self.processing_unit.cu_options,
                   arguments="signed char *support, const int nx, const int ny, const int nz")
        self.formula = formula
        self.verbose = verbose

    def op(self, cdi: CDI):
        if cdi.iobs.ndim == 3:
            nz, ny, nx = np.int32(cdi.iobs.shape[0]), np.int32(cdi.iobs.shape[1]), np.int32(cdi.iobs.shape[2])
        else:
            nz, ny, nx = np.int32(1), np.int32(cdi.iobs.shape[0]), np.int32(cdi.iobs.shape[1])
        self.kernel(cdi._cu_support, nx, ny, nz)
        nb = cua.sum(cdi._cu_support, dtype=np.int32).get()
        cdi.nb_point_support = nb
        if self.verbose:
            print("Init support (CUDA) using formula: %s, %d pixels in support [%6.3f%%]" %
                  (self.formula, nb, 100 * nb / cdi.iobs.size))
        return cdi


class InitObjRandom(CUOperatorCDI):
    """Set the initial value for the object using random values.
    """

    def __init__(self, src="support", amin=0, amax=1, phirange=2 * np.pi, lazy=False):
        """
        Set the parameters for the random optimisation, based on a source array (support or obj)
        The values will be set to src * a * exp(1j*phi), with:
        a = np.random.uniform(amin, amax, shape)
        phi = np.random.uniform(0, phirange, shape)
        This allows the initial array to be either based on the starting support, or
        from a starting object

        :param src: set the original array (either "support" or "obj") to scale the values. This
            can also be an array (or GPUarray) of the appropriate shape, fft-shifted so its centre is at 0.
        :param amin, amax: min and max of the random uniform values for the amplitude
        :param phirange: range of the random uniform values for the amplitude
        :param lazy: if True, this will be queued for later execution in the cdi object
        """
        super(InitObjRandom, self).__init__(lazy=lazy)
        self.src = src
        self.amin = np.float32(amin)
        self.amax = np.float32(amax)
        self.phirange = np.float32(phirange)

    def op(self, cdi: CDI):
        pu = self.processing_unit
        # We use temp array so free memory explicitly at the beginning and end.
        pu.cu_mem_pool.free_held()
        # First set the values for the object
        if isinstance(self.src, np.ndarray):
            cu_drv.memcpy_htod_async(dest=cdi._cu_obj.gpudata, src=self.src.astype(np.complex64))
        elif isinstance(self.src, cua.GPUArray):
            assert cdi._cu_obj.nbytes == self.src.nbytes
            cu_drv.memcpy_dtod(cdi._cu_obj.gpudata, self.src.gpudata, self.src.nbytes)
        elif "obj" in self.src.lower():
            pass
        else:
            src = cdi._cu_support.astype(np.complex64)
            cu_drv.memcpy_dtod(cdi._cu_obj.gpudata, src.gpudata, src.nbytes)
            del src
            gc.collect()
        # Add some random factor
        if not np.isclose(self.amin, self.amax) or not np.isclose(self.phirange, 0):
            # gen = cur.Sobol32RandomNumberGenerator()
            gen = cur.ScrambledSobol32RandomNumberGenerator()
            a = cua.empty(cdi._cu_obj.shape, dtype=np.float32, allocator=pu.cu_mem_pool.allocate)
            gen.fill_uniform(a)
            phi = cua.empty(cdi._cu_obj.shape, dtype=np.float32, allocator=pu.cu_mem_pool.allocate)
            gen.fill_uniform(phi)
            pu.cu_init_random(cdi._cu_obj, a, phi, self.amin, self.amax, self.phirange)
            a.gpudata.free()
            phi.gpudata.free()
            del a, phi
            pu.cu_mem_pool.free_held()
        return cdi


class InitFreePixels(CUOperatorCDI):
    """Operator used to init the free pixel mask by using special values in the Iobs array.
    This is used to provide an unbiased LLK indicator.
    """

    def __init__(self, ratio=5e-2, island_radius=3, exclude_zone_center=0.05, coords=None,
                 verbose=False, lazy=False):
        """

        :param ratio: (approximate) relative number of pixels to be included in the free mask
        :param island_radius: free island radius, to avoid pixel correlation due to finit object size
        :param exclude_zone_center: the relative radius of the zone to be excluded near the center
        :param coords: instead of generating random coordinates, these can be given as a tuple
            of (ix, iy[, iz]). All coordinates should be at least island_radius far from the borders,
            and these coordinates should be centred (i.e. to be applied to the centred iobs array)
        :param verbose: if True, be verbose
        :param lazy: if True, this will be queued for later execution in the cdi object
        :return: nothing. Free pixel values are modified as iobs_free = -iobs - 1
        """
        super(InitFreePixels, self).__init__(lazy=lazy)
        self.ratio = ratio
        self.island_radius = island_radius
        self.exclude_zone_center = exclude_zone_center
        self.coords = coords
        self.verbose = verbose

    def op(self, cdi: CDI):
        pu = self.processing_unit
        if cdi.iobs.ndim == 3:
            nz, ny, nx = cdi.iobs.shape
        else:
            ny, nx = cdi.iobs.shape
            nz = 1
        ratio = self.ratio
        island_radius = self.island_radius

        if cdi.get_crop() is not None:
            # Need to generate the free pixels on the original iobs array
            # TODO: find a way to do this using less memory.
            cdi._cu_iobs.gpudata.free()
            del cdi._cu_iobs
            cdi._cu_iobs = cua.to_gpu(cdi._iobs_orig, allocator=self.processing_unit.cu_mem_pool.allocate)

        if self.coords is None:
            exclude_zone_center = self.exclude_zone_center
            if cdi.iobs.ndim == 3:
                nb = int(cdi.iobs.size * ratio / (4 / 3 * 3.14 * island_radius ** 3))
                iz = np.random.randint(-nz // 2 + island_radius, nz // 2 - island_radius, nb, dtype=np.int32)
                iy = np.random.randint(-ny // 2 + island_radius, ny // 2 - island_radius, nb, dtype=np.int32)
                ix = np.random.randint(-nx // 2 + island_radius, nx // 2 - island_radius, nb, dtype=np.int32)
                idx = np.nonzero(((ix / (nx * exclude_zone_center)) ** 2 +
                                  (iy / (ny * exclude_zone_center)) ** 2 +
                                  (iz / (nz * exclude_zone_center)) ** 2) > 1)
                ix, iy, iz = np.take(ix, idx), np.take(iy, idx), np.take(iz, idx)
            else:
                nb = int(cdi.iobs.size * ratio / (np.pi * island_radius ** 2))
                iy = np.random.randint(-ny // 2 + island_radius, ny // 2 - island_radius, nb, dtype=np.int32)
                ix = np.random.randint(-nx // 2 + island_radius, nx // 2 - island_radius, nb, dtype=np.int32)
                idx = np.nonzero(((ix / (nx * exclude_zone_center)) ** 2 +
                                  (iy / (ny * exclude_zone_center)) ** 2) > 1)
                ix, iy = np.take(ix, idx), np.take(iy, idx)
                iz = np.zeros_like(ix)
        else:
            if len(self.coords) == 3:
                ix, iy, iz = self.coords
            else:
                ix, iy = self.coords
                iz = np.zeros_like(ix)
        # Clear previous free pixel mask, just in case
        pu.cu_clear_free_pixels(cdi._cu_iobs)
        # Operate on a copy of iobs to avoid collisions
        cu_iobs0 = cdi._cu_iobs.copy()
        # New pixel mask
        cu_ix = cua.to_gpu(ix.astype(np.int32))
        cu_iy = cua.to_gpu(iy.astype(np.int32))
        cu_iz = cua.to_gpu(iz.astype(np.int32))
        pu.cu_init_free_pixels(cu_ix, cu_iy, cu_iz, cdi._cu_iobs, cu_iobs0,
                               np.int32(nx), np.int32(ny), np.int32(nz),
                               np.int32(ix.size), np.int32(island_radius))
        cdi.iobs = cdi._cu_iobs.get()
        if cdi.get_crop() is not None:
            cdi._cu_iobs.gpudata.free()
            del cdi._cu_iobs
            cdi._iobs_orig = cdi.iobs
            cdi.iobs = crop(cdi._iobs_orig, margin_f=cdi.get_crop(), shift=True)
            cdi._cu_iobs = cua.to_gpu(cdi.iobs, allocator=self.processing_unit.cu_mem_pool.allocate)

        cdi.nb_free_points = np.logical_and(cdi.iobs > -1e19, cdi.iobs < -0.5).sum()
        return cdi


class CopyToPrevious(CUOperatorCDI):
    """
    Operator which will store a copy of the cdi object as cu_obj_previous. This is used for various algorithms, such
    as difference map or RAAR
    """

    def op(self, cdi):
        if has_attr_not_none(cdi, '_cu_obj_previous') is False:
            cdi._cu_obj_previous = cua.empty_like(cdi._cu_obj)
        if cdi._cu_obj_previous.shape != cdi._cu_obj.shape:
            cdi._cu_obj_previous = cua.empty_like(cdi._cu_obj)
        cu_drv.memcpy_dtod(dest=cdi._cu_obj_previous.gpudata, src=cdi._cu_obj.gpudata, size=cdi._cu_obj.nbytes)
        return cdi


class FromPU(CUOperatorCDI):
    """
    Operator copying back the CDI object and support data from the cuda device to numpy. The calculated complex
    amplitude is also retrieved by computing the Fourier transform of the current view of the object.
    """

    def op(self, cdi):
        warnings.warn("Use of ToPU() and FromPU() operators is now deprecated. Use get() and set() to access data.")
        # cdi._cu_obj.get(ary=cdi._obj)
        ## print("obj norm: ",(abs(cdi._obj)**2).sum())
        # cdi._cu_support.get(ary=cdi._support)
        ## TODO: find a more efficient way to access the calculated diffraction
        # cdi = FT() * cdi
        # cdi.calc = cdi._cu_obj.get()
        # cdi = IFT() * cdi
        return cdi


class ToPU(CUOperatorCDI):
    """
    Operator copying the data from numpy to the cuda device, as a complex64 array.
    """

    def op(self, cdi):
        warnings.warn("Use of ToPU() and FromPU() operators is now deprecated. Use get() and set() to access data.")
        # cdi._cu_obj = cua.to_gpu(cdi._obj)
        # cdi._cu_support = cua.to_gpu(cdi._support)
        # cdi._cu_iobs = cua.to_gpu(cdi.iobs)
        return cdi


class FreePU(CUOperatorCDI):
    """
    Operator freeing CUDA memory, removing any pycuda.gpuarray.Array attribute in the supplied CDI object.
    """

    def op(self, cdi):
        self.processing_unit.finish()
        self.processing_unit.free_fft_plans()
        # Get back last object and support
        cdi.get_obj()
        # Purge all GPUarray data
        for o in dir(cdi):
            if isinstance(cdi.__getattribute__(o), cua.GPUArray):
                cdi.__setattr__(o, None)
        self.view_purge(cdi, None)
        return cdi

    def timestamp_increment(self, cdi):
        cdi._timestamp_counter += 1


class FreeFromPU(CUOperatorCDI):
    """
    Gets back data from OpenCL and removes all OpenCL arrays.
    
    DEPRECATED
    """

    def __new__(cls):
        return FreePU() * FromPU()


class Scale(CUOperatorCDI):
    """
    Multiply the object by a scalar (real or complex).
    """

    def __init__(self, x):
        """

        :param x: the scaling factor
        """
        super(Scale, self).__init__()
        self.x = x

    def op(self, cdi):
        if np.isreal(self.x):
            self.processing_unit.cu_scale(cdi._cu_obj, np.float32(self.x))
        else:
            self.processing_unit.cu_scale_complex(cdi._cu_obj, np.complex64(self.x))
        return cdi


class FT(CUOperatorCDI):
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
        scale = self.processing_unit.fft(cdi._cu_obj, cdi._cu_obj)
        if self.scale is True:
            cdi = Scale(scale) * cdi
        elif (self.scale is not False) and (self.scale is not None):
            cdi = Scale(self.scale) * cdi
        cdi._is_in_object_space = False
        return cdi


class IFT(CUOperatorCDI):
    """
    Inverse Fourier transform
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the Fourier transform will be normalised, so that the transformed array L2 norm will
                      remain constant (by dividing the output by the square root of the object's size).
                      If False or None, the array norm will not be changed. If a scalar is given, the output array
                      is multiplied by it.
        """
        super(IFT, self).__init__()
        self.scale = scale

    def op(self, cdi):
        scale = self.processing_unit.ifft(cdi._cu_obj, cdi._cu_obj)
        if self.scale is True:
            cdi = Scale(scale) * cdi
        elif (self.scale is not False) and (self.scale is not None):
            cdi = Scale(self.scale) * cdi
        cdi._is_in_object_space = True
        return cdi


class Calc2Obs(CUOperatorCDI):
    """
    Copy the calculated intensities to the observed ones. Can be used for simulation.
    """

    def __init__(self):
        """

        """
        super(Calc2Obs, self).__init__()

    def op(self, cdi):
        if cdi.in_object_space():
            cdi = FT(scale=False) * cdi
            self.processing_unit.cu_square_modulus(cdi._cu_iobs, cdi._cu_obj)
            cdi = IFT(scale=False) * cdi
        else:
            self.processing_unit.cu_square_modulus(cdi._cu_iobs, cdi._cu_obj)
        return cdi


class ApplyAmplitude(CUOperatorCDI):
    """
    Apply the magnitude from an observed intensity, keep the phase. Optionally, calculate the log-likelihood before
    changing the amplitudes.
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
            the Richardson-Lucy deconvolution approach. If there is no PSF, it will be automatically
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
            if cdi.get_upsample() is None:
                pu.cu_apply_amplitude(cdi._cu_iobs, cdi._cu_obj, self.scale_in, self.scale_out,
                                      self.zero_mask, self.confidence_interval_factor,
                                      self.confidence_interval_factor_mask_min,
                                      self.confidence_interval_factor_mask_max)
            else:
                pu.cu_apply_amplitude_up(cdi._cu_iobs, cdi._cu_obj, self.scale_in, self.scale_out,
                                         self.zero_mask, self.confidence_interval_factor,
                                         self.confidence_interval_factor_mask_min,
                                         self.confidence_interval_factor_mask_max,
                                         nx, ny, nz, ux, uy, uz)
        else:
            # FFT-based convolution, using half-Hermitian kernel and real->complex64 FFT
            cu_icalc = cua.empty_like(cdi._cu_iobs)  # float32
            cu_icalc_f = cua.empty_like(cdi._cu_psf_f)  # Complex64, half-Hermitian array
            s = pu.fft_scale(cdi._obj.shape)

            if cdi.get_upsample() is None:
                pu.cu_square_modulus(cu_icalc, cdi._cu_obj)
            else:
                pu.cu_square_modulus_up(cu_icalc, cdi._cu_obj, nx, ny, nz, ux, uy, uz)

            pu.fft(cu_icalc, cu_icalc_f)
            pu.cu_mult_scale_complex(cdi._cu_psf_f, cu_icalc_f, s[0] * s[1])
            pu.ifft(cu_icalc_f, cu_icalc)

            DEBUG = False
            if self.calc_llk:
                llk = pu.cu_llk_icalc_red(cdi._cu_iobs, cu_icalc, self.scale_in ** 2).get()
                cdi.llk_poisson = llk['a']
                cdi.llk_gaussian = llk['b']
                cdi.llk_euclidian = llk['c']
                cdi.nb_photons_calc = llk['d']
                cdi.llk_poisson_free = llk['e']
                cdi.llk_gaussian_free = llk['f']
                cdi.llk_euclidian_free = llk['g']
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
                    icalc = fftshift(abs(cdi._cu_obj.get()[0] * self.scale_in) ** 2)
                    plt.imshow(icalc, norm=LogNorm(vmin=vmin, vmax=vmax))
                    plt.colorbar()
                    plt.title("Calc**2")

                    plt.subplot(143)
                    icalc = fftshift(cu_icalc.get()[0] * self.scale_in ** 2)
                    plt.imshow(icalc, norm=LogNorm(vmin=vmin, vmax=vmax))
                    plt.colorbar()
                    plt.title("Icalc")

            if cdi.get_upsample() is None:
                pu.cu_apply_amplitude_icalc(cdi._cu_iobs, cdi._cu_obj, cu_icalc,
                                            self.scale_in, self.scale_out, self.zero_mask,
                                            self.confidence_interval_factor,
                                            self.confidence_interval_factor_mask_min,
                                            self.confidence_interval_factor_mask_max)
            else:
                pu.cu_apply_amplitude_icalc_up(cdi._cu_iobs, cdi._cu_obj, cu_icalc,
                                               self.scale_in, self.scale_out, self.zero_mask,
                                               self.confidence_interval_factor,
                                               self.confidence_interval_factor_mask_min,
                                               self.confidence_interval_factor_mask_max,
                                               nx, ny, nz, ux, uy, uz)

            if DEBUG and self.calc_llk:
                plt.subplot(144)
                icalc = fftshift(cu_icalc.get()[0] * self.scale_in ** 2)
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

                # Need to recompute icalc_f which is overwritten during the c2r transform,
                # behaviour changed in cufft/cuda 11.1, and is also needed for VkFFT
                pu.fft(cu_icalc, cu_icalc_f)

                # FFT scales - we try to scale the arrays as the calculations proceed
                # in order to avoid under or overflow
                s = pu.fft_scale(cdi._obj.shape)

                # iobs / convolve(icalc,psf)
                # Ideally close to 1
                pu.cu_psf1(cdi._cu_iobs, cu_icalc, self.scale_in ** 2)
                cu_icalc2_f = cua.empty_like(cdi._cu_psf_f)  # Complex64, half-Hermitian array
                pu.fft(cu_icalc, cu_icalc2_f)

                # convolve(iobs / convolve(icalc,psf), icalc_mirror)
                pu.cu_psf2(cu_icalc2_f, cu_icalc_f, s[0] ** 2 * self.scale_in ** 2)
                if DEBUG:
                    print("icalc2f: %8e" % abs(cu_icalc2_f.get()).sum())
                del cu_icalc_f
                pu.ifft(cu_icalc2_f, cu_icalc)
                if DEBUG:
                    print("icalc2: %8e" % cu_icalc.get().sum())
                del cu_icalc2_f

                # psf *= convolve(iobs / convolve(icalc,psf), icalc_mirror)
                cu_psf = cua.empty_like(cdi._cu_iobs)
                pu.ifft(cdi._cu_psf_f, cu_psf)
                if self.psf_filter is None:
                    pu.cu_psf3(cu_psf, cu_icalc, s[1] ** 2)
                elif self.psf_filter.lower() == "tukey":
                    pu.cu_psf3_tukey(cu_psf, cu_icalc, np.float32(0.5), nx, ny, nz, s[1] ** 2)
                else:
                    pu.cu_psf3_hann(cu_psf, cu_icalc, nx, ny, nz, s[1] ** 2)

                # Normalise psf
                psf_sum = cua.sum(cu_psf)
                if DEBUG:
                    print("PSF sum: %8e" % psf_sum.get())
                pu.cu_psf4(cu_psf, psf_sum)

                # Compute & store updated psf FT
                pu.fft(cu_psf, cdi._cu_psf_f)
                del cu_psf

        return cdi


class UpdatePSF(CUOperatorCDI):
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
            self.scale_in = np.float32(1)
            need_back_ft = True

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
        cu_icalc = cua.empty_like(cdi._cu_iobs)  # float32
        cu_icalc0 = cua.empty_like(cdi._cu_iobs)  # float32
        cu_icalc_f = cua.empty_like(cdi._cu_psf_f)  # Complex64, half-Hermitian array
        cu_icalc2_f = cua.empty_like(cdi._cu_psf_f)  # Complex64, half-Hermitian array
        cu_psf = cua.empty_like(cdi._cu_iobs)

        if cdi.get_upsample() is None:
            pu.cu_square_modulus(cu_icalc0, cdi._cu_obj)
        else:
            pu.cu_square_modulus_up(cu_icalc0, cdi._cu_obj, nx, ny, nz, ux, uy, uz)
        s = pu.fft_scale(cdi._obj.shape)
        DEBUG = False
        for i in range(self.nb_cycle):
            # Icalc x PSF
            cu_drv.memcpy_dtod_async(dest=cu_icalc.gpudata, src=cu_icalc0.gpudata, size=cu_icalc.nbytes, stream=None)
            pu.fft(cu_icalc, cu_icalc_f)
            pu.cu_mult_scale_complex(cdi._cu_psf_f, cu_icalc_f, s[0] * s[1])

            # Copy to avoid overwriting cu_icalc_f which is needed later for psf2
            cu_drv.memcpy_dtod(dest=cu_icalc2_f.gpudata, src=cu_icalc_f.gpudata, size=cu_icalc_f.nbytes)
            pu.ifft(cu_icalc2_f, cu_icalc)  # this overwrites the source array [vkFFT, cuFFT]

            # iobs / convolve(icalc,psf)
            pu.cu_psf1(cdi._cu_iobs, cu_icalc, self.scale_in ** 2)
            pu.fft(cu_icalc, cu_icalc2_f)

            # convolve(iobs / convolve(icalc,psf), icalc_mirror)
            pu.cu_psf2(cu_icalc2_f, cu_icalc_f, s[0] ** 2 * self.scale_in ** 2)
            if DEBUG:
                print("icalc2f: %8e" % abs(cu_icalc2_f.get()).sum())
            pu.ifft(cu_icalc2_f, cu_icalc)
            if DEBUG:
                print("icalc2: %8e" % cu_icalc.get().sum())

            # psf *= convolve(iobs / convolve(icalc,psf), icalc_mirror)
            pu.ifft(cdi._cu_psf_f, cu_psf)
            if self.filter is None:
                pu.cu_psf3(cu_psf, cu_icalc, s[1] ** 2)
            elif self.filter.lower() == "tukey":
                pu.cu_psf3_tukey(cu_psf, cu_icalc, np.float32(0.5), nx, ny, nz, s[1] ** 2)
            else:
                pu.cu_psf3_hann(cu_psf, cu_icalc, nx, ny, nz, s[1] ** 2)

            # Normalise psf
            # TODO: avoid normalising every cycle
            psf_sum = cua.sum(cu_psf)
            if DEBUG:
                print("PSF sum: %8e" % psf_sum.get())
            pu.cu_psf4(cu_psf, psf_sum)

            # Compute & store updated psf FT
            pu.fft(cu_psf, cdi._cu_psf_f)
        if need_back_ft:
            cdi = IFT(scale=True) * cdi

        return cdi


class FourierApplyAmplitude(CUOperatorCDI):
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
        :param psf_filter: None, "hann" or "tukey" - filter for the PSF update
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


class ERProj(CUOperatorCDI):
    """
    Error reduction.
    """

    def __init__(self, positivity=False):
        super(ERProj, self).__init__()
        self.positivity = positivity

    def op(self, cdi: CDI):
        if self.positivity:
            self.processing_unit.cu_er_real(cdi._cu_obj, cdi._cu_support)
        else:
            self.processing_unit.cu_er(cdi._cu_obj, cdi._cu_support)
        return cdi


class ER(CUOperatorCDI):
    """
    Error reduction cycle
    """

    def __init__(self, positivity=False, calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1,
                 zero_mask=False, confidence_interval_factor=0,
                 confidence_interval_factor_mask_min=0.5, confidence_interval_factor_mask_max=1.2,
                 update_psf=0, psf_filter=None):
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
        :param psf_filter: None, "hann" or "tukey" - filter for the PSF update
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
        self.update_psf = int(update_psf)
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


class CFProj(CUOperatorCDI):
    """
    Charge Flipping.
    """

    def __init__(self, positivity=False):
        super(CFProj, self).__init__()
        self.positivity = positivity

    def op(self, cdi: CDI):
        if self.positivity:
            self.processing_unit.cu_cf_real(cdi._cu_obj, cdi._cu_support)
        else:
            self.processing_unit.cu_cf(cdi._cu_obj, cdi._cu_support)
        return cdi


class CF(CUOperatorCDI):
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
        :param psf_filter: None, "hann" or "tukey" - filter for the PSF update
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
        self.update_psf = int(update_psf)
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


class HIOProj(CUOperatorCDI):
    """
    Hybrid Input-Output.
    """

    def __init__(self, beta=0.9, positivity=False):
        super(HIOProj, self).__init__()
        self.beta = np.float32(beta)
        self.positivity = positivity

    def op(self, cdi):
        if self.positivity:
            self.processing_unit.cu_hio_real(cdi._cu_obj, cdi._cu_obj_previous, cdi._cu_support, self.beta)
        else:
            self.processing_unit.cu_hio(cdi._cu_obj, cdi._cu_obj_previous, cdi._cu_support, self.beta)
        return cdi


class HIO(CUOperatorCDI):
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
        :param psf_filter: None, "hann" or "tukey" - filter for the PSF update
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
        self.update_psf = int(update_psf)
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
                                        update_psf=update_psf, obj_stats=calc_llk, psf_filter=self.psf_filter)
            cdi = sup_proj * fap * cdi

            if calc_llk:
                cdi.update_history(mode='llk', algorithm='HIO', verbose=True, update_psf=update_psf)
            else:
                cdi.update_history(mode='algorithm', algorithm='HIO', update_psf=update_psf)

            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi

            cdi.cycle += 1
        del cdi._cu_obj_previous
        return cdi


class RAARProj(CUOperatorCDI):
    """
    RAAR.
    """

    def __init__(self, beta=0.9, positivity=False):
        super(RAARProj, self).__init__()
        self.beta = np.float32(beta)
        self.positivity = positivity

    def op(self, cdi):
        if self.positivity:
            self.processing_unit.cu_raar_real(cdi._cu_obj, cdi._cu_obj_previous, cdi._cu_support, self.beta)
        else:
            self.processing_unit.cu_raar(cdi._cu_obj, cdi._cu_obj_previous, cdi._cu_support, self.beta)
        return cdi


class RAAR(CUOperatorCDI):
    """
    RAAR cycle
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
        :param psf_filter: None, "hann" or "tukey" - filter for the PSF update
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
        self.update_psf = int(update_psf)
        self.psf_filter = psf_filter

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new RAAR operator with the number of cycles multiplied by n
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
                cdi.update_history(mode='algorithm', algorithm='RAAR', verbose=False, update_psf=update_psf)

            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi

            cdi.cycle += 1
        del cdi._cu_obj_previous
        return cdi


class GPS(CUOperatorCDI):
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
        :param psf_filter: None, "hann" or "tukey" - filter for the PSF update
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
        s = 1.0 / np.sqrt(cdi._cu_iobs.size)  # FFT scaling np.float32(1)

        epsilon = np.float32(self.inertia / (self.inertia + self.t))

        ny, nx = np.int32(cdi._cu_obj.shape[-2]), np.int32(cdi._cu_obj.shape[-1])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi._cu_obj.shape[0])
        else:
            nz = np.int32(1)

        # Make sure we have tmp copy arrays available
        if has_attr_not_none(cdi, '_cu_z') is False:
            cdi._cu_z = cua.empty_like(cdi._cu_obj)
        elif cdi._cu_z.shape != cdi._cu_obj.shape:
            cdi._cu_z = cua.empty_like(cdi._cu_obj)

        if has_attr_not_none(cdi, '_cu_y') is False:
            cdi._cu_y = cua.empty_like(cdi._cu_obj)
        elif cdi._cu_y.shape != cdi._cu_obj.shape:
            cdi._cu_y = cua.empty_like(cdi._cu_obj)

        # We start in Fourier space (obj = z_0)
        cdi = FT(scale=True) * cdi

        # z_0 = FT(obj)
        cu_drv.memcpy_dtod(dest=cdi._cu_z.gpudata, src=cdi._cu_obj.gpudata, size=cdi._cu_obj.nbytes)

        # Start with obj = y_0 = 0
        cdi._cu_obj.fill(np.complex64(0))

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
            cu_drv.memcpy_dtod(dest=cdi._cu_y.gpudata, src=cdi._cu_obj.gpudata, size=cdi._cu_obj.nbytes)

            cdi = FT(scale=False) * cdi

            # ^z = z_k - t F(y_k)
            self.processing_unit.cu_gps1(cdi._cu_obj, cdi._cu_z, self.t * s, self.sigma_o, nx, ny, nz)

            cdi = ApplyAmplitude(calc_llk=calc_llk, zero_mask=self.zero_mask,
                                 confidence_interval_factor=self.confidence_interval_factor,
                                 confidence_interval_factor_mask_min=self.confidence_interval_factor_mask_min,
                                 confidence_interval_factor_mask_max=self.confidence_interval_factor_mask_max,
                                 update_psf=update_psf, psf_filter=self.psf_filter) * cdi

            # obj = z_k+1 = (1 - epsilon) * sqrt(iobs) * exp(i * arg(^z)) + epsilon * z_k
            self.processing_unit.cu_gps2(cdi._cu_obj, cdi._cu_z, epsilon)

            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = IFT(scale=True) * cdi
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi
                    cdi = FT(scale=True) * cdi

            if ic < self.nb_cycle - 1:
                # obj = 2 * z_k+1 - z_k  & store z_k+1 in z
                self.processing_unit.cu_gps3(cdi._cu_obj, cdi._cu_z)

                cdi = IFT(scale=False) * cdi

                # obj = ^y = proj_support[y_k + s * obj] * G_sigma_f
                self.processing_unit.cu_gps4(cdi._cu_obj, cdi._cu_y, cdi._cu_support, self.s * s, self.sigma_f,
                                             self.positivity, nx, ny, nz)
            else:
                self.processing_unit.cu_scale(cdi._cu_obj, s)
            if calc_llk:
                cdi = ObjSupportStats() * cdi
                cdi.update_history(mode='llk', algorithm='GPS', verbose=True, update_psf=update_psf)
            else:
                cdi.update_history(mode='algorithm', algorithm='GPS', update_psf=update_psf)
            cdi.cycle += 1

        # Free memory
        cdi._cu_y.gpudata.free()
        cdi._cu_z.gpudata.free()
        del cdi._cu_y, cdi._cu_z

        # Back to object space
        cdi = IFT(scale=False) * cdi

        return cdi


class ML(CUOperatorCDI):
    """
    Maximum likelihood conjugate gradient minimization
    """

    def __init__(self, reg_fac=1e2, nb_cycle=1, calc_llk=False, show_cdi=False, fig_num=-1):
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
            if (has_attr_not_none(cdi, '_cu_obj_dir') is False) \
                    or (has_attr_not_none(cdi, '_cu_dpsi') is False) \
                    or (has_attr_not_none(cdi, '_cu_obj_grad') is False) \
                    or (has_attr_not_none(cdi, '_cu_obj_grad_last') is False) \
                    or (has_attr_not_none(cdi, 'llk_support_reg_fac') is False):
                self.need_init = True

        if self.need_init:
            # Take into account support in regularization
            N = cdi._obj.size
            # Total number of photons
            Nph = cdi.iobs_sum
            cdi.llk_support_reg_fac = np.float32(self.reg_fac / (8 * N / Nph))
            # if cdi.llk_support_reg_fac > 0:
            #    print("Regularization factor for support:", cdi.llk_support_reg_fac)

            cdi._cu_obj_dir = cua.empty(cdi._cu_obj.shape, np.complex64,
                                        allocator=self.processing_unit.cu_mem_pool.allocate)
            cdi._cu_psi = cua.empty(cdi._cu_obj.shape, np.complex64,
                                    allocator=self.processing_unit.cu_mem_pool.allocate)
            # cdi._cu_dpsi = cua.empty(cdi._cu_obj.shape, np.complex64,
            # allocator=self.processing_unit.cu_mem_pool.allocate)
            cdi._cu_obj_grad = cua.empty(cdi._cu_obj.shape, np.complex64,
                                         allocator=self.processing_unit.cu_mem_pool.allocate)
            cdi._cu_obj_gradlast = cua.empty(cdi._cu_obj.shape, np.complex64,
                                             allocator=self.processing_unit.cu_mem_pool.allocate)
            self.need_init = False

        ny, nx = np.int32(cdi._cu_obj.shape[-2]), np.int32(cdi._cu_obj.shape[-1])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi._cu_obj.shape[0])
        else:
            nz = np.int32(1)

        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if cdi.cycle % self.calc_llk == 0:
                    calc_llk = True

            cu_drv.memcpy_dtod(dest=cdi._cu_psi.gpudata, src=cdi._cu_obj.gpudata, size=cdi._cu_psi.nbytes)

            # TODO: avoid the normalisation
            pu.fft(cdi._cu_psi, cdi._cu_psi, norm=True)

            if calc_llk:
                cdi._cu_psi, cdi._cu_obj = cdi._cu_obj, cdi._cu_psi
                cdi._is_in_object_space = False
                cdi = LLK() * cdi
                cdi._cu_psi, cdi._cu_obj = cdi._cu_obj, cdi._cu_psi
                cdi._is_in_object_space = True

            # This calculates the conjugate of [(1 - iobs/icalc) * psi]
            self.processing_unit.cu_ml_poisson_psi_gradient(cdi._cu_psi, cdi._cu_obj_grad, cdi._cu_iobs, nx, ny, nz)

            # TODO: avoid the normalisation
            pu.fft(cdi._cu_obj_grad, cdi._cu_obj_grad)

            if cdi.llk_support_reg_fac > 0:
                self.processing_unit.cu_ml_poisson_reg_support_gradient(cdi._cu_obj, cdi._cu_obj_grad, cdi._cu_support,
                                                                        cdi.llk_support_reg_fac)

            if ic == 0:
                beta = np.float32(0)
                cu_drv.memcpy_dtod(dest=cdi._cu_obj_dir.gpudata, src=cdi._cu_obj_grad.gpudata,
                                   size=cdi._cu_obj_grad.nbytes)
            else:
                # Polak-Ribière CG coefficient
                tmp = self.processing_unit.cu_cg_polak_ribiere_red(cdi._cu_obj_grad, cdi._cu_obj_gradlast).get()
                if False:
                    g1 = cdi._cu_obj_grad.get()
                    g0 = cdi._cu_obj_gradlast.get()
                    A, B = (g1.real * (g1.real - g0.real) + g1.imag * (g1.imag - g0.imag)).sum(), (
                            g0.real * g0.real + g0.imag * g0.imag).sum()
                    cpubeta = A / B
                    print("betaPR: (GPU)=%8.4e  , (CPU)=%8.4e [%8.4e/%8.4e], dot(g0.g1)=%8e [%8e]" %
                          (tmp.real / tmp.imag, cpubeta, A, B, (g0 * g1).sum().real, (abs(g0) ** 2).sum().real))
                # Reset direction if beta<0 => beta=0
                beta = np.float32(max(0, tmp.real / max(1e-20, tmp.imag)))

                self.processing_unit.cu_ml_poisson_cg_linear(beta, cdi._cu_obj_dir, np.float32(-1), cdi._cu_obj_grad)

            # For next cycle
            cdi._cu_obj_grad, cdi._cu_obj_gradlast = cdi._cu_obj_gradlast, cdi._cu_obj_grad

            # Avoid using two memory allocations for obj_grad and dpsi
            cdi._cu_dpsi = cdi._cu_obj_grad

            cu_drv.memcpy_dtod(dest=cdi._cu_dpsi.gpudata, src=cdi._cu_obj_dir.gpudata, size=cdi._cu_dpsi.nbytes)

            # TODO: avoid the normalisation
            pu.fft(cdi._cu_dpsi, cdi._cu_dpsi, norm=True)

            if cdi.llk_support_reg_fac > 0:
                tmp = self.processing_unit.cdi_ml_poisson_gamma_support_red(cdi._cu_iobs, cdi._cu_psi, cdi._cu_dpsi,
                                                                            cdi._cu_obj, cdi._cu_obj_dir,
                                                                            cdi._cu_support,
                                                                            cdi.llk_support_reg_fac).get()
                gamma_n, gamma_d = tmp.real, tmp.imag
                gamma = np.float32(gamma_n / gamma_d)
            else:
                tmp = self.processing_unit.cdi_ml_poisson_gamma_red(cdi._cu_iobs, cdi._cu_psi, cdi._cu_dpsi).get()
                gamma_n, gamma_d = tmp.real, tmp.imag
                gamma = np.float32(gamma_n / gamma_d)

            self.processing_unit.cu_ml_poisson_cg_linear(np.float32(1), cdi._cu_obj, gamma, cdi._cu_obj_dir)

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


class SupportUpdate(CUOperatorCDI):
    """
    Update the support
    """

    def __init__(self, threshold_relative=0.2, smooth_width=3, force_shrink=False, method='rms',
                 post_expand=None, verbose=False, update_border_n=0, min_fraction=0, max_fraction=1,
                 lazy=False):
        """
        Update support.

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
                               smooth_width = a * exp(-cdi.cycle/nb*log(b/a)) if cdi.cycle < nb
                               smooth_width = b if cdi.cycle >= nb
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
        Raises: SupportTooSmall or SupportTooLarge if support diverges according to min_ and max_fraction
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

    def op(self, cdi: CDI):
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
        tmp = self.processing_unit.cu_average_max_red(cdi._cu_obj, cdi._cu_support).get()
        cdi._obj_max = np.sqrt(tmp.imag)

        # Actual threshold is computed on the convolved object
        if self.method == 'max' or cdi.nb_point_support == 0:
            tmp = self.processing_unit.cu_rms_max_abs_red(cdi._cu_obj_abs, cdi._cu_support).get()
            thr = self.threshold_relative * np.float32(np.sqrt(tmp.imag))
        elif self.method == 'rms':
            tmp = self.processing_unit.cu_rms_max_abs_red(cdi._cu_obj_abs, cdi._cu_support).get()
            thr = self.threshold_relative * np.sqrt(np.float32(tmp.real / cdi.nb_point_support))
        else:
            tmp = self.processing_unit.cu_average_max_abs_red(cdi._cu_obj_abs, cdi._cu_support).get()
            thr = self.threshold_relative * np.float32(tmp.real / cdi.nb_point_support)

        # Update support and compute the new number of points in the support

        if self.update_border_n > 0:
            # First compute the border of the support
            nx, ny = np.int32(cdi._cu_obj.shape[-1]), np.int32(cdi._cu_obj.shape[-2])
            if cdi._obj.ndim == 3:
                nz = np.int32(cdi._cu_obj.shape[0])
            else:
                nz = np.int32(1)

            m1 = np.int8(2)  # Bitwise mask for expanded support
            m2 = np.int8(4)  # Bitwise mask for shrunk support

            # Convolution kernel width cannot exceed 7, so loop for larger convolutions
            for i in range(0, self.update_border_n, 7):
                # Expanded support
                m0 = m1 if i > 0 else np.int8(1)
                n = np.int32(self.update_border_n - i) if (self.update_border_n - i) <= 7 else np.int32(7)

                self.processing_unit.binary_window_convol_16x_mask(cdi._cu_support, n, nx, ny, nz, m0, m1,
                                                                   block=(16, 1, 1), grid=(1, int(ny), int(nz)))
                self.processing_unit.binary_window_convol_16y_mask(cdi._cu_support, n, nx, ny, nz, m1, m1,
                                                                   block=(1, 16, 1), grid=(int(nx), 1, int(nz)))
                if cdi._obj.ndim == 3:
                    self.processing_unit.binary_window_convol_16z_mask(cdi._cu_support, n, nx, ny, nz, m1, m1,
                                                                       block=(1, 1, 16), grid=(int(nx), int(ny), 1))

                # Shrunk support
                m0 = m2 if i > 0 else np.int8(1)
                self.processing_unit.binary_window_convol_16x_mask(cdi._cu_support, -n, nx, ny, nz, m0, m2,
                                                                   block=(16, 1, 1), grid=(1, int(ny), int(nz)))
                self.processing_unit.binary_window_convol_16y_mask(cdi._cu_support, -n, nx, ny, nz, m2, m2,
                                                                   block=(1, 16, 1), grid=(int(nx), 1, int(nz)))
                if cdi._obj.ndim == 3:
                    self.processing_unit.binary_window_convol_16z_mask(cdi._cu_support, -n, nx, ny, nz, m2, m2,
                                                                       block=(1, 1, 16), grid=(int(nx), int(ny), 1))

            nb = int(self.processing_unit.cu_support_update_border(cdi._cu_obj_abs, cdi._cu_support, thr,
                                                                   self.force_shrink).get())
        else:
            nb = int(self.processing_unit.cu_support_update(cdi._cu_obj_abs, cdi._cu_support, thr,
                                                            self.force_shrink).get())

        if self.post_expand is not None:
            for n in self.post_expand:
                cdi = SupportExpand(n=n, update_nb_points_support=False) * cdi
            nb = int(self.processing_unit.cu_nb_point_support(cdi._cu_support).get())

        if self.verbose:
            print("Nb points in support: %d (%6.3f%%), threshold=%8f (%6.3f), nb photons=%10e"
                  % (nb, nb / cdi._obj.size * 100, thr, self.threshold_relative, tmp.real))
        cdi._cu_obj_abs.gpudata.free()
        del cdi._cu_obj_abs  # Free memory
        cdi.nb_point_support = nb
        cdi.update_history(mode='support', support_update_threshold=thr)
        if cdi.nb_point_support <= self.min_fraction * cdi.iobs.size:
            raise SupportTooSmall("Too few points in support: %d (%6.3f%%)" % (nb, nb / cdi._obj.size * 100))
        elif cdi.nb_point_support >= self.max_fraction * cdi.iobs.size:
            raise SupportTooLarge("Too many points in support: %d (%6.3f%%)" % (nb, nb / cdi._obj.size * 100))
        return cdi


class ObjSupportStats(CUOperatorCDI):
    """
    Gather basic stats about the object: maximum and average amplitude inside the support,
    and percentage of square modulus outside the support.
    This should be evaluated ideally immediately after FourierApplyAmplitude. The result is stored
    in the CDI object's history.
    """

    def op(self, cdi):
        # Get average amplitude and maximum intensity for the object in the support (unsmoothed)
        tmp = self.processing_unit.cu_obj_support_stats_red(cdi._cu_obj, cdi._cu_support).get()
        cdi._obj_max = np.sqrt(tmp['a'])
        # Percent of square modulus inside and outside object
        cdi._obj2_out = tmp['d'] / (tmp['c'] + tmp['d'])
        cdi.update_history(mode='support')
        return cdi


class ScaleObj(CUOperatorCDI):
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
            tmp = pu.cu_scale_amplitude(cdi._cu_iobs, cdi._cu_obj).get()
            scale = tmp.real / tmp.imag
            if False:
                tmpcalc = np.abs(cdi.get_obj()) * (cdi.iobs >= 0)
                tmpobs = np.sqrt(np.abs(cdi.iobs))
                scale_cpu = (tmpcalc * tmpobs).sum() / (tmpcalc ** 2).sum()
                print("Scaling F: scale_cpu= %8.4f, scale_gpu= %8.4f" % (scale_cpu, scale))
        elif self.method.lower() == 'i':
            # Scale object to match Fourier intensities
            tmp = pu.cu_scale_intensity(cdi._cu_iobs, cdi._cu_obj).get()
            scale = np.sqrt(tmp.real / tmp.imag)
            if False:
                tmpcalc = np.abs(cdi.get_obj()) ** 2 * (cdi.iobs >= 0)
                scale_cpu = np.sqrt((tmpcalc * cdi.iobs).sum() / (tmpcalc ** 2).sum())
                print("Scaling I: scale_cpu= %8.4f, scale_gpu= %8.4f" % (scale_cpu, scale))
        elif self.method.lower() == 'p':
            # Scale object to match intensities with Poisson noise
            tmp = pu.cu_scale_intensity_poisson(cdi._cu_iobs, cdi._cu_obj).get()
            scale = tmp.real / tmp.imag
            if False:
                tmpcalc = (np.abs(cdi.get_obj()) ** 2 * (cdi.iobs >= 0)).sum()
                tmpobs = (cdi.iobs * (cdi.iobs >= 0)).sum()
                scale_cpu = tmpobs / tmpcalc
                print("Scaling P: scale_cpu= %8.4f, scale_gpu= %8.4f" % (scale_cpu, scale))
        else:
            # Scale object to match weighted intensities
            # Weight: 1 for null intensities, zero for masked pixels
            tmp = pu.cu_scale_weighted_intensity(cdi._cu_iobs, cdi._cu_obj).get()
            scale = np.sqrt(tmp.real / tmp.imag)
            if False:
                w = (1 / (np.abs(cdi.iobs) + 1e-6) * (cdi.iobs > 1e-6) + (cdi.iobs <= 1e-6)) * (cdi.iobs >= 0)
                tmpcalc = np.abs(cdi.get_obj()) ** 2
                scale_cpu = np.sqrt((w * tmpcalc * cdi.iobs).sum() / (w * tmpcalc ** 2).sum())
                print("Scaling W: scale_cpu= %8.4f, scale_gpu= %8.4f" % (scale_cpu, scale))
        scale *= pu.fft_scale(cdi._obj.shape)[1]
        cdi = IFT(scale=scale) * cdi
        if self.verbose:
            print("Scaled object by: %f [%s]" % (scale, self.method))
        return cdi


class LLK(CUOperatorCDI):
    """
    Log-likelihood reduction kernel. This is a reduction operator - it will write llk as an argument in the cdi object.
    If it is applied to a CDI instance in object space, a FT() and IFT() will be applied  to perform the calculation
    in diffraction space.
    This collect log-likelihood for Poisson, Gaussian and Euclidian noise models, and also computes the
    total calculated intensity (including in masked pixels).
    """

    def __init__(self, scale=1.0):
        """

        :param scale: the scale factor to be applied to the calculated amplitude before evaluating the
                      log-likelihood. The calculated amplitudes are left unmodified.
        """
        super(LLK, self).__init__()
        self.scale = np.float32(scale ** 2)

    def op(self, cdi: CDI):
        pu = self.processing_unit
        need_ft = cdi.in_object_space()

        if need_ft:
            cdi = FT() * cdi

        if cdi._psf_f is None:
            if cdi.get_upsample() is None:
                llk = self.processing_unit.cu_llk_red(cdi._cu_iobs, cdi._cu_obj, self.scale).get()
            else:
                nx = np.int32(cdi.iobs.shape[-1])
                ny = np.int32(cdi.iobs.shape[-2])
                if cdi.iobs.ndim == 3:
                    nz = np.int32(cdi.iobs.shape[0])
                else:
                    nz = np.int32(1)
                uz, uy, ux = cdi.get_upsample(dim3=True)
                llk = self.processing_unit.cu_llk_up_red(cdi._cu_iobs, cdi._cu_obj, self.scale,
                                                         nx, ny, nz, ux, uy, uz).get()
        else:
            # FFT-based convolution, using half-Hermitian kernel and real->complex64 FFT
            cu_icalc = cua.empty_like(cdi._cu_iobs)  # float32
            cu_icalc_f = cua.empty_like(cdi._cu_psf_f)  # Complex64, half-Hermitian array

            pu.cu_square_modulus(cu_icalc, cdi._cu_obj)
            pu.fft(cu_icalc, cu_icalc_f)
            pu.cu_mult_complex(cdi._cu_psf_f, cu_icalc_f)
            pu.ifft(cu_icalc_f, cu_icalc)
            llk = pu.cu_llk_icalc_red(cdi._cu_iobs, cu_icalc, self.scale).get()

        cdi.llk_poisson = llk['a']
        cdi.llk_gaussian = llk['b']
        cdi.llk_euclidian = llk['c']
        cdi.nb_photons_calc = llk['d']
        cdi.llk_poisson_free = llk['e']
        cdi.llk_gaussian_free = llk['f']
        cdi.llk_euclidian_free = llk['g']

        if need_ft:
            cdi = IFT() * cdi

        return cdi


class LLKSupport(CUOperatorCDI):
    """
    Support log-likelihood reduction kernel. Can only be used when cdi instance is object space.
    This is a reduction operator - it will write llk_support as an argument in the cdi object, and return cdi.
    """

    def op(self, cdi):
        llk = float(self.processing_unit.cu_llk_reg_support_red(cdi._cu_obj, cdi._cu_support).get())
        cdi.llk_support = llk * cdi.llk_support_reg_fac
        return cdi


class DetwinSupport(CUOperatorCDI):
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
        if self.restore:
            cdi._cu_support = cdi._cu_support_detwin_tmp
            del cdi._cu_support_detwin_tmp
        else:
            # Copy current support
            cdi._cu_support_detwin_tmp = cdi._cu_support.copy()
            pu = self.processing_unit
            nx = np.int32(cdi.iobs.shape[-1])
            ny = np.int32(cdi.iobs.shape[-2])
            if cdi.iobs.ndim == 3:
                nz = np.int32(cdi.iobs.shape[0])
            else:
                nz = np.int32(1)
            cm = pu.cu_center_mass_fftshift_complex(cdi._cu_obj, nx, ny, nz, np.int32(1)).get()
            cx = np.int32(cm['a'] / cm['d'] - nx / 2)
            cy = np.int32(cm['b'] / cm['d'] - ny / 2)
            cz = np.int32(cm['c'] / cm['d'] - nz / 2)
            if self.axis < 0:
                self.axis += cdi.iobs.ndim
            if cdi.iobs.ndim == 3:
                if self.axis == 0:
                    pu.cu_detwinz(cdi._cu_support, cz, nx, ny, nz)
                elif self.axis == 1:
                    pu.cu_detwiny(cdi._cu_support, cy, nx, ny)
                else:
                    pu.cu_detwinx(cdi._cu_support, cx, nx)
            else:
                if self.axis == 0:
                    pu.cu_detwiny(cdi._cu_support, cy, nx, ny)
                else:
                    pu.cu_detwinx(cdi._cu_support, cx, nx)
        return cdi


class DetwinHIO(CUOperatorCDI):
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
        # print('Detwinning with %d HIO cycles and a half-support' % self.nb_cycle)
        if self.detwin_axis is None:
            self.detwin_axis = randint(0, cdi.iobs.ndim)
        return DetwinSupport(restore=True) * HIO(beta=self.beta, positivity=self.positivity,
                                                 zero_mask=self.zero_mask) ** self.nb_cycle \
               * DetwinSupport(axis=self.detwin_axis) * cdi


class DetwinRAAR(CUOperatorCDI):
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


class SupportExpand(CUOperatorCDI):
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
        if self.n == 0:
            return cdi
        nx, ny = np.int32(cdi._cu_obj.shape[-1]), np.int32(cdi._cu_obj.shape[-2])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi._cu_obj.shape[0])
        else:
            nz = np.int32(1)
        self.processing_unit.binary_window_convol_16x(cdi._cu_support, self.n, nx, ny, nz,
                                                      block=(16, 1, 1), grid=(1, int(ny), int(nz)))
        self.processing_unit.binary_window_convol_16y(cdi._cu_support, self.n, nx, ny, nz,
                                                      block=(1, 16, 1), grid=(int(nx), 1, int(nz)))
        if cdi._obj.ndim == 3:
            self.processing_unit.binary_window_convol_16z(cdi._cu_support, self.n, nx, ny, nz,
                                                          block=(1, 1, 16), grid=(int(nx), int(ny), 1))
        if self.update_nb_points_support:
            cdi.nb_point_support = int(self.processing_unit.cu_nb_point_support(cdi._cu_support).get())
        return cdi


class ObjConvolve(CUOperatorCDI):
    """
    3D Gaussian convolution of the object, produces a new array with the convoluted amplitude of the object.
    """

    def __init__(self, sigma=1):
        super(ObjConvolve, self).__init__()
        self.sigma = np.float32(sigma)

    def op(self, cdi):
        cdi._cu_obj_abs = cua.zeros(cdi._cu_obj.shape, dtype=np.float32,
                                    allocator=self.processing_unit.cu_mem_pool.allocate)
        nx, ny = np.int32(cdi._cu_obj.shape[-1]), np.int32(cdi._cu_obj.shape[-2])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi._cu_obj.shape[0])
        else:
            nz = np.int32(1)
        self.processing_unit.abs_gauss_convol_16x(cdi._cu_obj, cdi._cu_obj_abs, self.sigma, nx, ny, nz,
                                                  block=(16, 1, 1), grid=(1, int(ny), int(nz)))
        self.processing_unit.gauss_convol_16y(cdi._cu_obj_abs, self.sigma, nx, ny, nz,
                                              block=(1, 16, 1), grid=(int(nx), 1, int(nz)))
        if cdi._obj.ndim == 3:
            self.processing_unit.gauss_convol_16z(cdi._cu_obj_abs, self.sigma, nx, ny, nz,
                                                  block=(1, 1, 16), grid=(int(nx), int(ny), 1))
        return cdi


class ShowCDI(ShowCDICPU):
    def __init__(self, fig_num=None, i=None):
        """

        :param fig_num: the matplotlib figure number. if None, a new figure will be created each time.
        :param i: if the object is 3D, display the ith plane (default: the center one)
        """
        super(ShowCDI, self).__init__(fig_num=fig_num, i=i)

    @staticmethod
    def get_icalc(cdi: CDI, i=None):
        if cdi.in_object_space():
            cdi = FT(scale=True) * cdi
            icalc = abs(cdi.get_obj()) ** 2
            cdi = IFT(scale=True) * cdi
        else:
            icalc = abs(cdi.get_obj()) ** 2
        if icalc.ndim == 3 and i is not None:
            return icalc[i]
        return icalc


class EstimatePSF(CUOperatorCDI):
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


class PRTF(CUOperatorCDI):
    """Operator to compute the Phase Retrieval Transfer Function.
    When applied to a CDI object, it stores the result in it as
    cdi.prtf, cdi.prtf_freq, cdi.prtf_nyquist, cdi.prtf_nb
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
        cu_shell_obs = cua.zeros(nb_shell, np.float32, allocator=self.processing_unit.cu_mem_pool.allocate)
        cu_shell_calc = cua.zeros(nb_shell, np.float32, allocator=self.processing_unit.cu_mem_pool.allocate)
        cu_shell_nb = cua.zeros(nb_shell, np.int32, allocator=self.processing_unit.cu_mem_pool.allocate)
        cdi.prtf_freq = np.linspace(0, f_nyquist * (nb_shell - 1) / nb_shell, nb_shell) + f_nyquist / nb_shell / 2
        cdi.prtf_fnyquist = f_nyquist
        if cdi._psf_f is None:
            cu_prtf_k = CU_ElK(name='cu_prtf',
                               operation="prtf(i, obj, iobs, shell_calc, shell_obs, shell_nb, nb_shell, f_nyquist, nx, ny,"
                                         "nz)",
                               preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') + getks('cdi/cuda/prtf.cu'),
                               options=self.processing_unit.cu_options,
                               arguments="pycuda::complex<float>* obj, float* iobs, float* shell_calc,"
                                         "float* shell_obs, int *shell_nb, const int nb_shell,"
                                         "const int f_nyquist, const int nx, const int ny, const int nz")
            cu_prtf_k(cdi._cu_obj, cdi._cu_iobs, cu_shell_calc, cu_shell_obs, cu_shell_nb, nb_shell, f_nyquist, nx, ny,
                      nz)
        else:
            # FFT-based convolution, using half-Hermitian kernel and real->complex64 FFT
            cu_icalc = cua.empty_like(cdi._cu_iobs)  # float32
            cu_icalc_f = cua.empty_like(cdi._cu_psf_f)  # Complex64, half-Hermitian array

            if cdi.get_upsample() is None:
                pu.cu_square_modulus(cu_icalc, cdi._cu_obj)
            else:
                nx = np.int32(cdi.iobs.shape[-1])
                ny = np.int32(cdi.iobs.shape[-2])
                if cdi.iobs.ndim == 3:
                    nz = np.int32(cdi.iobs.shape[0])
                else:
                    nz = np.int32(1)
                uz, uy, ux = cdi.get_upsample(dim3=True)
                pu.cu_square_modulus_up(cu_icalc, cdi._cu_obj, nx, ny, nz, ux, uy, uz)
            pu.fft(cu_icalc, cu_icalc_f)
            pu.cu_mult_complex(cdi._cu_psf_f, cu_icalc_f)
            pu.ifft(cu_icalc_f, cu_icalc)
            cu_prtf_icalc_k = CU_ElK(name='cu_prtf_icalc',
                                     operation="prtf_icalc(i, icalc, iobs, shell_calc, shell_obs, shell_nb,"
                                               "nb_shell, f_nyquist, nx, ny, nz)",
                                     preamble=getks('cuda/complex.cu') + getks('cuda/float_n.cu') + getks(
                                         'cdi/cuda/prtf.cu'),
                                     options=self.processing_unit.cu_options,
                                     arguments="float* icalc, float* iobs, float* shell_calc,"
                                               "float* shell_obs, int *shell_nb, const int nb_shell,"
                                               "const int f_nyquist, const int nx, const int ny, const int nz")
            cu_prtf_icalc_k(cu_icalc, cdi._cu_iobs, cu_shell_calc, cu_shell_obs, cu_shell_nb,
                            nb_shell, f_nyquist, nx, ny, nz)

        prtf = cu_shell_calc.get() / cu_shell_obs.get()
        nb = cu_shell_nb.get()
        prtf /= np.nanpercentile(prtf[nb > 0], 100)
        cdi.prtf = np.ma.masked_array(prtf, mask=nb == 0)
        cdi.prtf_nb = nb
        cdi.prtf_iobs = cu_shell_obs.get()

        plot_prtf(cdi.prtf_freq, f_nyquist, cdi.prtf, iobs_shell=cdi.prtf_iobs, nbiobs_shell=nb,
                  file_name=self.file_name, title=self.fig_title)

        if need_ft:
            cdi = IFT() * cdi

        return cdi


class InterpIobsMask(CUOperatorCDI):
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
        nx, ny = np.int32(cdi._cu_iobs.shape[-1]), np.int32(cdi._cu_iobs.shape[-2])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi._cu_iobs.shape[0])
        else:
            nz = np.int32(1)
        self.processing_unit.cu_mask_interp_dist(cdi._cu_iobs, self.d, self.n, nx, ny, nz)
        cdi.iobs = cdi._cu_iobs.get()
        return cdi


class InitPSF(CUOperatorCDI):
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
        :return: nothing. This initialises cdi._cu_psf_f, and copies the array to cdi._psf_f
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
            cu_psf = cua.to_gpu(self.psf.astype(np.float32), allocator=self.processing_unit.cu_mem_pool.allocate)
        else:
            cu_psf = cua.empty(shape, dtype=np.float32, allocator=self.processing_unit.cu_mem_pool.allocate)
            if "gauss" in self.model.lower():
                pu.cu_gaussian(cu_psf, self.fwhm, nx, ny, nz)
            elif "lorentz" in self.model.lower():
                pu.cu_lorentzian(cu_psf, self.fwhm, nx, ny, nz)
            else:
                pu.cu_pseudovoigt(cu_psf, self.fwhm, self.eta, nx, ny, nz)

        # Normalise PSF
        s = pu.fft_scale(cdi._obj.shape)
        psf_sum = cua.sum(cu_psf)
        pu.cu_psf4(cu_psf, psf_sum)

        # Store the PSF FT
        cdi._cu_psf_f = cua.empty(shape2, dtype=np.complex64, allocator=self.processing_unit.cu_mem_pool.allocate)
        pu.fft(cu_psf, cdi._cu_psf_f)
        cdi._psf_f = cdi._cu_psf_f.get()

        if self.psf is None:
            cdi = UpdatePSF(filter=self.filter) ** 10 * cdi

        return cdi


class Zoom(CUOperatorCDI):
    """
    Zoom in or out the object and the support without changing the array size.
    This can be used when changing the bin and crop parameters, e.g. going
    from crop=2 to bin=2.
    """

    def __init__(self, zoom_f):
        """

        :param zoom_f: the zoom factor, either as a single floating-point value
            or a tuple/list of values far each dimension.
        """
        super().__init__()
        self.zoom_f = zoom_f

    def op(self, cdi: CDI):
        pu = self.processing_unit
        if np.isscalar(self.zoom_f):
            zoom_f = [np.float32(self.zoom_f)] * cdi.iobs.ndim
        else:
            if len(self.zoom_f) != cdi.iobs.ndim:
                raise OperatorException("CDI Zoom: array dimensions must match the number of zoom factors")
            zoom_f = np.array(self.zoom_f)
        nx, ny = np.int32(cdi._cu_iobs.shape[-1]), np.int32(cdi._cu_iobs.shape[-2])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi._cu_iobs.shape[0])
        else:
            nz = np.int32(1)

        # Get flat views of arrays to generate the elementwise kernel dimension
        obj = cdi._cu_obj.reshape(cdi._cu_obj.size)
        sup = cdi._cu_support.reshape(cdi._cu_support.size)

        if not np.isclose(zoom_f[-1], 1):
            # X
            pu.cu_zoom_complex(obj[:ny * nz], zoom_f[-1], -1, 0, nx, ny, nz, 0.0, True)
            pu.cu_zoom_sup(sup[:ny * nz], zoom_f[-1], -1, 0, nx, ny, nz, 0)
        if not np.isclose(zoom_f[-2], 1):
            # Y
            pu.cu_zoom_complex(obj[:nx * nz], zoom_f[-2], -2, 0, nx, ny, nz, 0.0, True)
            pu.cu_zoom_sup(sup[:nx * nz], zoom_f[-2], -2, 0, nx, ny, nz, 0)
        if not np.isclose(zoom_f[-3], 1) and cdi._obj.ndim == 3:
            # Z
            pu.cu_zoom_complex(obj[:nx * ny], zoom_f[-3], -3, 0, nx, ny, nz, 0.0, True)
            pu.cu_zoom_sup(sup[:nx * ny], zoom_f[-3], -3, 0, nx, ny, nz, 0)

        return cdi
