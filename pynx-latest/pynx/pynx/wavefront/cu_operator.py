# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['default_processing_unit', 'FreePU', 'QuadraticPhase', 'ThinLens', 'CircularMask',
           'RectangularMask', 'FT', 'IFT', 'PropagateFarField', 'PropagateNearField', 'PropagateFRT',
           'BackPropagatePaganin']

import warnings
import types
import numpy as np

from ..processing_unit.cu_processing_unit import CUProcessingUnit

import pycuda.driver as cu_drv
import pycuda.gpuarray as cua
from pycuda.elementwise import ElementwiseKernel as CU_ElK

from ..operator import has_attr_not_none, OperatorException, OperatorSum, OperatorPower

from .wavefront import OperatorWavefront, Wavefront, UserWarningWavefrontNearFieldPropagation
from ..processing_unit import default_processing_unit as main_default_processing_unit
from ..processing_unit.kernel_source import get_kernel_source as getks


################################################################################################
# Patch Wavefront class so that we can use 5*w to scale it.
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


patch_method(Wavefront)


################################################################################################


class CUProcessingUnitWavefront(CUProcessingUnit):
    """
    Processing unit in CUDA space, for 2D wavefront operations.

    Handles initializing the context and kernels.

    This is made to handle one array size at a time (a 2D wavefront or a stack of 2D wavefronts)
    """

    def __init__(self):
        super(CUProcessingUnitWavefront, self).__init__()

    def cu_init_kernels(self):
        self.cu_scale = CU_ElK(name='cu_scale',
                               operation="d[i] = complexf(d[i].real() * s, d[i].imag() * s)",
                               preamble=getks('cuda/complex.cu'),
                               options=self.cu_options,
                               arguments="pycuda::complex<float> *d, const float s")

        self.cu_sum = CU_ElK(name='cu_sum',
                             operation="dest[i] += src[i]",
                             preamble=getks('cuda/complex.cu'),
                             options=self.cu_options,
                             arguments="pycuda::complex<float> *src, pycuda::complex<float> *dest")

        self.cu_scale_complex = CU_ElK(name='cu_scale_complex',
                                       operation="d[i] = complexf(d[i].real() * s.real() - d[i].imag() * s.imag(), d[i].real() * s.imag() + d[i].imag() * s.real())",
                                       preamble=getks('cuda/complex.cu'),
                                       options=self.cu_options,
                                       arguments="pycuda::complex<float> *d, const pycuda::complex<float> s")

        self.cu_quad_phase_mult = CU_ElK(name='cu_quad_phase_mult',
                                         operation="QuadPhaseMult(i, d, f, scale, nx, ny)",
                                         preamble=getks('wavefront/cuda/quad_phase_mult_elw.cu'),
                                         options=self.cu_options,
                                         arguments="float2 *d, const float f, const float scale, const int nx, const int ny")

        self.cu_mask_circular = CU_ElK(name='cu_mask_circular',
                                       operation="CircularMask(i, d, radius, pixel_size, invert, nx, ny)",
                                       preamble=getks('wavefront/cuda/mask_elw.cu'),
                                       options=self.cu_options,
                                       arguments="float2 *d, const float radius, const float pixel_size, const char invert, const int nx, const int ny")

        self.cu_mask_rectangular = CU_ElK(name='cu_mask_rectangular',
                                          operation="RectangularMask(i, d, width, height, pixel_size, invert, nx, ny)",
                                          preamble=getks('wavefront/cuda/mask_elw.cu'),
                                          options=self.cu_options,
                                          arguments="float2 *d, const float width, const float height, const float pixel_size, const char invert, const int nx, const int ny")

        self.cu_square_modulus = CU_ElK(name='cu_square_modulus',
                                        operation="d[i] = make_float2( d[i].x * d[i].x + d[i].y * d[i].y, 0 );",
                                        options=self.cu_options,
                                        arguments="float2 *d")

        self.cu_paganin_transfer_function = CU_ElK(name='cu_paganin_transfer_function',
                                                   operation="paganin_transfer_function(i, d, z_delta, mu, dk, nx, ny)",
                                                   preamble=getks('wavefront/cuda/paganin_elw.cu'),
                                                   options=self.cu_options,
                                                   arguments="float2 *d, const float z_delta, const float mu, const float dk, const int nx, const int ny")

        self.cu_paganin_thickness_wavefront = CU_ElK(name='cu_paganin_thickness_wavefront',
                                                     operation="paganin_thickness_wavefront(i, d, mu, k_delta)",
                                                     preamble=getks('wavefront/cuda/paganin_elw.cu'),
                                                     options=self.cu_options,
                                                     arguments="float2 *d, const float mu, const float k_delta")


"""
The default processing unit 
"""
default_processing_unit = CUProcessingUnitWavefront()


class CUOperatorWavefront(OperatorWavefront):
    """
    Base class for a wavefront operator using CUDA
    """

    def __init__(self, processing_unit=None):
        super(CUOperatorWavefront, self).__init__()

        self.Operator = CUOperatorWavefront
        self.OperatorSum = CUOperatorWavefrontSum
        self.OperatorPower = CUOperatorWavefrontPower

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

    def apply_ops_mul(self, w):
        """
        Apply the series of operators stored in self.ops to a wavefront.
        In this version the operators are applied one after the other to the same wavefront (multiplication)

        :param w: the wavefront to which the operators will be applied.
        :return: the wavefront, after application of all the operators in sequence
        """
        return super(CUOperatorWavefront, self).apply_ops_mul(w)

    def prepare_data(self, w):
        # Make sure data is already in CUDA space, otherwise transfer it
        if w._timestamp_counter > w._cu_timestamp_counter:
            w._cu_d = cua.to_gpu(w._d)
            w._cu_timestamp_counter = w._timestamp_counter
        if has_attr_not_none(w, '_cu_d_view') is False:
            w._cu_d_view = {}

    def timestamp_increment(self, w):
        w._cu_timestamp_counter += 1

    def view_register(self, obj):
        """
        Creates a new unique view key in an object. When finished with this view, it should be de-registered
        using view_purge. Note that it only reserves the key, but does not create the view.
        :return: an integer value, which corresponds to yet-unused key in the object's view.
        """
        i = 1
        while i in obj._cu_d_view:
            i += 1
        obj._cu_d_view[i] = None
        return i

    def view_copy(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._cu_d
        else:
            src = obj._cu_d_view[i_source]
        if i_dest == 0:
            obj._cu_d = cua.empty_like(src)
            dest = obj._cu_d
        else:
            obj._cu_d_view[i_dest] = cua.empty_like(src)
            dest = obj._cu_d_view[i_dest]
        cu_drv.memcpy_dtod(dest=dest.gpudata, src=src.gpudata, size=dest.nbytes)

    def view_swap(self, obj, i1, i2):
        if i1 != 0:
            if i1 not in obj._cu_d_view:
                # Create dummy value, assume a copy will be made later
                obj._cu_d_view[i1] = None
        if i2 != 0:
            if i2 not in obj._cu_d_view:
                # Create dummy value, assume a copy will be made later
                obj._cu_d_view[i2] = None
        if i1 == 0:
            obj._cu_d, obj._cu_d_view[i2] = obj._cu_d_view[i2], obj._cu_d
        elif i2 == 0:
            obj._cu_d, obj._cu_d_view[i1] = obj._cu_d_view[i1], obj._cu_d
        else:
            obj._cu_d_view[i1], obj._cu_d_view[i2] = obj._cu_d_view[i2], obj._cu_d_view[i1]

    def view_sum(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._cu_d
        else:
            src = obj._cu_d_view[i_source]
        if i_dest == 0:
            dest = obj._cu_d
        else:
            dest = obj._cu_d_view[i_dest]
        self.processing_unit.cu_sum(src, dest)

    def view_purge(self, obj, i):
        if i is not None:
            del obj._cu_d_view[i]
        elif has_attr_not_none(obj, '_cu_d_view'):
            del obj._cu_d_view


# The only purpose of this class is to make sure it inherits from CUOperatorWavefront and has a processing unit
class CUOperatorWavefrontSum(OperatorSum, CUOperatorWavefront):
    def __init__(self, op1, op2):
        if np.isscalar(op1):
            op1 = Scale(op1)
        if np.isscalar(op2):
            op2 = Scale(op2)
        if isinstance(op1, CUOperatorWavefront) is False or isinstance(op2, CUOperatorWavefront) is False:
            raise OperatorException(
                "ERROR: cannot add a CUOperatorWavefront with a non-CUOperatorWavefront: %s + %s" % (
                    str(op1), str(op2)))
        # We can only have a sum of two CLOperatorWavefront, so they must have a processing_unit attribute.
        CUOperatorWavefront.__init__(self, op1.processing_unit)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CUOperatorWavefront
        self.OperatorSum = CUOperatorWavefrontSum
        self.OperatorPower = CUOperatorWavefrontPower
        self.prepare_data = types.MethodType(CUOperatorWavefront.prepare_data, self)
        self.timestamp_increment = types.MethodType(CUOperatorWavefront.timestamp_increment, self)
        self.view_copy = types.MethodType(CUOperatorWavefront.view_copy, self)
        self.view_swap = types.MethodType(CUOperatorWavefront.view_swap, self)
        self.view_sum = types.MethodType(CUOperatorWavefront.view_sum, self)
        self.view_purge = types.MethodType(CUOperatorWavefront.view_purge, self)


# The only purpose of this class is to make sure it inherits from CLOperatorWavefront and has a processing unit
class CUOperatorWavefrontPower(OperatorPower, CUOperatorWavefront):
    def __init__(self, op, n):
        CUOperatorWavefront.__init__(self, op.processing_unit)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CUOperatorWavefront
        self.OperatorSum = CUOperatorWavefrontSum
        self.OperatorPower = CUOperatorWavefrontPower
        self.prepare_data = types.MethodType(CUOperatorWavefront.prepare_data, self)
        self.timestamp_increment = types.MethodType(CUOperatorWavefront.timestamp_increment, self)
        self.view_copy = types.MethodType(CUOperatorWavefront.view_copy, self)
        self.view_swap = types.MethodType(CUOperatorWavefront.view_swap, self)
        self.view_sum = types.MethodType(CUOperatorWavefront.view_sum, self)
        self.view_purge = types.MethodType(CUOperatorWavefront.view_purge, self)


class CopyToPrevious(CUOperatorWavefront):
    """
    Operator which will store a copy of the wavefront as cu_d_previous. This is used for various algorithms, such
    as diffrence map or RAAR
    """

    def op(self, w):
        if has_attr_not_none(w, '_cu_d_previous') is False:
            w._cu_d_previous = cua.empty_like(w._cu_d)
        if w._cu_d_previous.shape != w._cu_d.shape:
            w._cu_d_previous = cua.empty_like(w._cu_d)
        cu_drv.memcpy_dtod(dest=w._cu_d_previous.gpudata, src=w._cu_d.gpudata, size=w._cu_d.nbytes)
        return w


class FromPU(CUOperatorWavefront):
    """
    Operator copying back the wavefront data from the opencl device to numpy.
    
    DEPRECATED
    """

    def op(self, w):
        warnings.warn("Use of ToPU() and FromPU() operators is now deprecated. Use get() and set() to access data.")
        w.get()
        # w._cu_d.get(ary=w.d)
        return w


class ToPU(CUOperatorWavefront):
    """
    Operator copying the wavefront data from numpy to the opencl device, as a complex64 array.

    DEPRECATED
    """

    def op(self, w):
        warnings.warn("Use of ToPU() and FromPU() operators is now deprecated. Use get() and set() to access data.")
        # w._cu_d = cua.to_gpu(w.d)
        return w


class FreePU(CUOperatorWavefront):
    """
    Operator freeing CUDA memory, removing any pycuda.gpuarray.Array attribute in the supplied wavefront.
    """

    def op(self, w):
        self.processing_unit.finish()
        self.processing_unit.free_fft_plans()
        w.get()  # This will copy the data to the host memory if necessary
        for o in dir(w):
            if isinstance(w.__getattribute__(o), cua.GPUArray):
                w.__setattr__(o, None)
        self.view_purge(w, None)
        return w

    def timestamp_increment(self, w):
        w._timestamp_counter += 1


class FreeFromPU(CUOperatorWavefront):
    """
    Gets back data from OpenCL and removes all OpenCL arrays.
    
    DEPRECATED
    """

    def __new__(cls):
        return FreePU() * FromPU()


class QuadraticPhase(CUOperatorWavefront):
    """
    Operator applying a quadratic phase factor
    """

    def __init__(self, factor, scale=1):
        """
        Application of a quadratic phase factor, and optionnaly a scale factor.

        The actual factor is:  :math:`scale * e^{(i * factor * (ix^2 + iy^2))}`
        where ix and iy are the integer indices of the pixels

        :param factor: the factor for the phase calculation.
        :param scale: the data will be scaled by this factor. Useful to normalize after a Fourier transform,
                      without accessing twice the array data.
        """
        super(QuadraticPhase, self).__init__()
        self.scale = np.float32(scale)
        self.factor = np.float32(factor)

    def op(self, w):
        self.processing_unit.cu_quad_phase_mult(w._cu_d, self.factor, self.scale, np.int32(w._d.shape[-1]),
                                                np.int32(w._d.shape[-2]))
        return w


class ThinLens(CUOperatorWavefront):
    """
    Multiplies the wavefront by a quadratic phase factor corresponding to a thin lens with a given focal length.
    The phase factor is: :math:`e^{-\\frac{i * \pi * (x^2 + y^2)}{\\lambda * focal\\_length}}`
    
    Note that a too short focal_length can lead to aliasing, which will occur when the phase varies
    from more than pi from one pixel to the next. A warning will be written if this occurs at half the distance
    from the center (i.e. a quarter of the array size).
    """

    def __init__(self, focal_length):
        """

        :param focal_length: focal length (in meters)
        """
        self.focal_length = focal_length
        super(ThinLens, self).__init__()

    def op(self, w):
        factor = np.float32(-np.pi * w.pixel_size ** 2 / (self.focal_length * w.wavelength))
        nx, ny = np.int32(w._d.shape[-1]), np.int32(w._d.shape[-2])
        # Calculate delta(phase) at half-distance from center zone and warn if necessary
        dphi = abs(factor * nx / 2)
        if dphi > np.pi:
            warnings.warn(
                "ThinLens Operator: d(phase factor)/pixel is %5.2f>PI (@half -distance from center)! Aliasing may occur" % dphi)
        self.processing_unit.cu_quad_phase_mult(w._cu_d, factor, np.float32(1), nx, ny)
        return w


class CircularMask(CUOperatorWavefront):
    """
    Multiplies the wavefront by a binary circular mask with a given radius
    """

    def __init__(self, radius, invert=False):
        """

        :param radius: radius of the mask (in meters)
        :param invert: if True, the inside of the circle will be masked rather than the outside
        """
        self.radius = np.float32(radius)
        self.invert = np.int8(invert)
        super(CircularMask, self).__init__()

    def op(self, w):
        nx, ny = np.int32(w._d.shape[-1]), np.int32(w._d.shape[-2])
        self.processing_unit.cu_mask_circular(w._cu_d, self.radius, w.pixel_size, self.invert, nx, ny)
        return w


class RectangularMask(CUOperatorWavefront):
    """
    Multiplies the wavefront by a rectangular mask with a given width and height
    """

    def __init__(self, width, height, invert=False):
        """

        :param width: width of the mask (in meters)
        :param height: height of the mask (in meters)
        :param invert: if True, the inside of the rectangle will be masked rather than the outside
        """
        self.width = np.float32(width)
        self.height = np.float32(height)
        self.invert = np.int8(invert)
        super(RectangularMask, self).__init__()

    def op(self, w):
        nx, ny = np.int32(w._d.shape[-1]), np.int32(w._d.shape[-2])
        self.processing_unit.cu_mask_rectangular(w._cu_d, self.width, self.height, w.pixel_size, self.invert,
                                                 nx, ny)
        return w


class Scale(CUOperatorWavefront):
    """
    Multiply the wavefront by a scalar (real or complex).
    """

    def __init__(self, x):
        """

        :param x: the scaling factor
        """
        super(Scale, self).__init__()
        self.x = x

    def op(self, w):
        if self.x == 1:
            return w
        if np.isreal(self.x):
            self.processing_unit.cu_scale(w._cu_d, np.float32(self.x))
        else:
            self.processing_unit.cu_scale_complex(w._cu_d, np.complex64(self.x))
        return w


class FT(CUOperatorWavefront):
    """
    Forward Fourier transform.
    NOTE: CUFFT returns an un-normalized FFT: (abs(fft(d))**2).sum() = (abs(d)**2).sum() * d.size
    """

    def op(self, w):
        self.processing_unit.fft(w._cu_d, w._cu_d, ndim=2, norm=True)
        return w


class IFT(CUOperatorWavefront):
    """
    Inverse Fourier transform
    NOTE: CUFFT returns an un-normalized FFT: (abs(fft(d))**2).sum() = (abs(d)**2).sum() * d.size
    """

    def op(self, w):
        self.processing_unit.ifft(w._cu_d, w._cu_d, ndim=2, norm=True)
        return w


class PropagateFarField(CUOperatorWavefront):
    """
    Far field propagator
    """

    def __init__(self, dz, forward=True, no_far_field_quadphase=True):
        """

        :param dz: propagation distance in meters
        :param forward: if True, forward propagation, otherwise backward
        :param no_far_field_quadphase: if True (default), no quadratic phase is applied in the far field
        """
        super(PropagateFarField, self).__init__()
        self.dz = np.float32(dz)
        self.no_far_field_quadphase = no_far_field_quadphase
        self.forward = forward

    def op(self, w):
        nx = w._d.shape[-1]
        pixel_size_new = w.wavelength * abs(self.dz) / nx / w.pixel_size
        if self.forward:
            pixel_size_far = pixel_size_new
        else:
            pixel_size_far = w.pixel_size
        if self.forward:
            f = np.pi / (w.wavelength * self.dz) * w.pixel_size ** 2
            w = FT() * QuadraticPhase(factor=f) * w
            if self.no_far_field_quadphase is False:
                f = np.pi / (w.wavelength * self.dz) * pixel_size_far ** 2
                w = QuadraticPhase(factor=f) * w
        else:
            if self.no_far_field_quadphase is False:
                f = -np.pi / (w.wavelength * self.dz) * pixel_size_far ** 2
                w = QuadraticPhase(factor=f) * w
            f = -np.pi / (w.wavelength * self.dz) * pixel_size_new ** 2
            w = QuadraticPhase(factor=f) * IFT() * w
        w.pixel_size = pixel_size_new
        return w


class PropagateNearField(CUOperatorWavefront):
    """
    Near field propagator
    """
    warning_near_field = True  # Will be set to False once a warning has been printed

    def __init__(self, dz, magnification=None, verbose=False):
        """

        :param dz: propagation distance (in meters)
        :param magnification: if not None, the destination pixel size will will be multiplied by this factor.
                              Note that it creates important restrictions on the validity domain of the calculation,
                              both near and far.
        :param verbose: if True, prints the propagation limit for a valid phase.
        """
        super(PropagateNearField, self).__init__()
        self.dz = np.float32(dz)
        self.magnification = magnification
        self.verbose = verbose

    def op(self, w):
        if self.magnification is None or self.magnification == 1:
            nx = w._d.shape[-1]
            max_dist = (w.pixel_size * nx) ** 2 / w.wavelength / nx
            if self.verbose:
                print("Near field propagation: |z=%f| < %f ?" % (self.dz, max_dist))
            if self.warning_near_field:
                # Calculate near field upper limit (worst case)
                if abs(self.dz) > max_dist * 2:
                    s = "WARNING: exceeding maximum near field propagation distance: z=%f > %f\n" % (self.dz, max_dist)

                    formatwarning_orig = warnings.formatwarning
                    warnings.formatwarning = lambda message, category, filename, lineno, line=None: str(message)
                    warnings.warn(s, UserWarningWavefrontNearFieldPropagation)
                    warnings.formatwarning = formatwarning_orig

                    # This only serves to avoid the test with the *same* operator
                    self.warning_near_field = False

            f = -np.pi * w.wavelength * self.dz / (nx * w.pixel_size) ** 2

            w.z += self.dz
            return IFT() * QuadraticPhase(factor=f) * FT() * w
        else:
            ny, nx = w._d.shape[-2:]
            m = self.magnification
            p = w.pixel_size
            min_dist = max(abs((m - 1) / m), abs(m - 1)) * nx * p ** 2 / w.wavelength
            max_dist = abs(m) * nx * p ** 2 / w.wavelength
            if self.verbose:
                print("Near field magnified propagation: %f < |z=%f| < %f ?" % (min_dist, self.dz, max_dist))
            if self.warning_near_field:
                if abs(self.dz) > max_dist or abs(self.dz) < min_dist:
                    s = "WARNING: outside magnified near field propagation range: %f < z=%f < %f\n" % \
                        (min_dist, self.dz, max_dist)

                    formatwarning_orig = warnings.formatwarning
                    warnings.formatwarning = lambda message, category, filename, lineno, line=None: str(message)
                    warnings.warn(s, UserWarningWavefrontNearFieldPropagation)
                    warnings.formatwarning = formatwarning_orig

                    # This only serves to avoid the test with the *same* operator
                    self.warning_near_field = False
            f1 = np.pi / (w.wavelength * self.dz) * (1 - m) * p ** 2
            f2 = -np.pi * w.wavelength * self.dz / m / (nx * p) ** 2
            f3 = np.pi / (w.wavelength * self.dz) * m * (m - 1) * p ** 2
            w = QuadraticPhase(factor=f3) * IFT() * QuadraticPhase(factor=f2) * FT() * QuadraticPhase(factor=f1) * w
            w.z += self.dz
            w.pixel_size = m * w.pixel_size
            return w


class MagnifyNearField(CUOperatorWavefront):
    """
    This operator will calculate the valid near field propagation range for a given magnification factor,
    and then perform a magnified near field propagation to the center of this range.
    The new z position and pixel size wan be obtained from the resulting wavefront
    """

    def __init__(self, magnification, verbose=False):
        """

        :param magnification: if not None, the destination pixel size will will be multiplied by this factor.
                              Note that it creates important restrictions on the validity domain of the calculation,
                              both near and far.
        :param verbose: if True, prints the propagation limit for a valid phase.
        """
        super(MagnifyNearField, self).__init__()
        self.magnification = magnification
        self.verbose = verbose

    def op(self, w):
        if self.magnification is None or self.magnification == 1:
            return w
        ny, nx = w._d.shape[-2:]
        m = self.magnification
        p = w.pixel_size
        min_dist = max(abs((m - 1) / m), abs(m - 1)) * nx * p ** 2 / w.wavelength
        max_dist = abs(m) * nx * p ** 2 / w.wavelength
        if self.verbose:
            print("Near field magnified propagation range: [%f ; %f]" % (min_dist, max_dist))
        dz = (min_dist + max_dist) / 2.
        f1 = np.pi / (w.wavelength * dz) * (1 - m) * p ** 2
        f2 = -np.pi * w.wavelength * dz / m / (nx * p) ** 2
        f3 = np.pi / (w.wavelength * dz) * m * (m - 1) * p ** 2
        w = QuadraticPhase(factor=f3) * IFT() * QuadraticPhase(factor=f2) * FT() * QuadraticPhase(factor=f1) * w
        w.z += dz
        w.pixel_size = m * w.pixel_size
        return w


class PropagateFRT(CUOperatorWavefront):
    """
    Wavefront propagator using a fractional Fourier transform
        References:

        * D. Mas, J. Garcia, C. Ferreira, L.M. Bernardo, and F. Marinho, Optics Comm 164, 233 (1999)
        * J. García, D. Mas, and R.G. Dorsch, Applied Optics 35, 7013 (1996)

        Notes:

        * the computed phase is only 'valid' for dz<N*pixel_size**2/wavelength, i.e near-to-medium field
          (I am not sure this is strictly true - the phase will vary quickly from pixel to pixel as for
          a spherical wave propagation but the calculation is still correct)
        * the amplitude remains correct even in the far field
        * only forward propagation works correctly (z>0)
    """
    warning_backward_propagation = True

    def __init__(self, dz, forward=True):
        """

        :param dz: the propagation distance
        :param forward: if True (default), forward propagation. Note that backward propagation is purely experimental
                        and not fully correct.
        """
        super(PropagateFRT, self).__init__()
        self.dz = np.float32(dz)
        self.forward = forward

    def op(self, w):
        if self.warning_backward_propagation and self.forward is False:
            self.warning_backward_propagation = False
            print("WARNING: using backward FRT propagation - this is only partially working !")
        verbose = False
        nx = w._d.shape[-1]
        if self.forward:
            z1 = w.z + self.dz
        else:
            z1 = w.z - self.dz
        phi = np.arctan(w.wavelength * self.dz / (w.pixel_size ** 2 * nx))
        if verbose:
            print("Phi=%6.2f=%6.2f°" % (phi, phi * 180 / np.pi))
        f1 = w.pixel_size ** 2 * nx / w.wavelength
        if verbose:
            print("dz=%6.3f, f1=%6.3f=%6.3f=%6.3f" %
                  (self.dz, f1, self.dz / np.tan(phi), (w.pixel_size * nx) ** 2 / (nx * w.wavelength)))
        if self.forward:
            pixel_size_z1 = w.pixel_size * np.sqrt(
                1 + w.wavelength ** 2 * nx ** 2 * self.dz ** 2 / (nx * w.pixel_size) ** 4)
        else:
            Dx1 = w.pixel_size * nx
            pback1 = np.sqrt((Dx1 ** 2 + np.sqrt(Dx1 ** 4 - 4 * w.wavelength ** 2 * nx ** 2 * self.dz ** 2)) / 2)
            pback2 = np.sqrt((Dx1 ** 2 + np.sqrt(Dx1 ** 4 - 4 * w.wavelength ** 2 * nx ** 2 * self.dz ** 2)) / 2)
            pforward1 = pback1 * np.sqrt(1 + w.wavelength ** 2 * nx ** 2 * self.dz ** 2 / (nx * pback1) ** 4)
            pforward2 = pback2 * np.sqrt(1 + w.wavelength ** 2 * nx ** 2 * self.dz ** 2 / (nx * pback2) ** 4)
            if verbose:
                print("pback1=%8.3f  pback2=%8.3f  pforward1=%8.3f  pforward2=%8.3f" % (
                    pback1 * 1e6, pback2 * 1e6, pforward1 * 1e6, pforward2 * 1e6))
            if np.isclose(w.pixel_size, pforward1):
                pixel_size_z1 = pback1
            else:
                pixel_size_z1 = pback2

        if verbose:
            if verbose:
                print("f1=%6.2fm ; pixel size=%8.2f -> (%8.2f,%8.2f,%f) nm" % (
                    f1, w.pixel_size * 1e9, w.pixel_size * np.sqrt(1 + (self.dz / f1) ** 2) * 1e9,
                    pixel_size_z1 * 1e9, np.sqrt(1 + self.dz ** 2 / f1 ** 2)))

        quad1 = -np.pi / nx * np.tan(phi / 2)
        quad2 = -np.pi / nx * np.sin(phi)
        quad3 = np.pi / nx * np.tan(phi)

        w.z = z1
        w.pixel_size = pixel_size_z1

        if self.forward:
            return QuadraticPhase(quad1 + quad3, scale=1. / w._d.size) * IFT() * QuadraticPhase(
                quad2) * FT() * QuadraticPhase(quad1) * w
        else:
            return QuadraticPhase(-quad1, scale=1. / w._d.size) * IFT() * QuadraticPhase(
                -quad2) * FT() * QuadraticPhase(-quad1 - quad3) * w


class BackPropagatePaganin(CUOperatorWavefront):
    """ Back-propagation algorithm using the single-projection approach.
    Ref: Paganin et al., Journal of microscopy 206 (2002), 33–40. (DOI: 10.1046/j.1365-2818.2002.01010.x)

    This operator is special since it will use only the intensity of the wavefront. Therefore it will first take the
    square modulus of the wavefront it is applied to, discarding any phase information.
    The result of the transformation is the calculated wavefront at the sample position, i.e. if T(r) is the
    estimated thickness of the sample, it is exp(-mu * T - 2*pi/lambda * T)
    """

    def __init__(self, dz=1, delta=1e-6, beta=1e-9):
        """

        :param dz: distance between sample and detector (meter)
        :param delta: real part of the refraction index, n = 1 - delta + i * beta
        :param beta: imaginary part of the refraction index
        """
        super(BackPropagatePaganin, self).__init__()
        self.dz = np.float32(dz)
        self.beta = np.float32(beta)
        self.delta = np.float32(delta)
        # TODO: also use the source position (magnification)

    def op(self, w):
        self.processing_unit.cu_square_modulus(w._cu_d)
        w = FT() * w
        mu = np.float32(4 * np.pi * self.beta / w.wavelength)
        dk = np.float32(2 * np.pi / (w._d.shape[-1] * w.pixel_size))
        self.processing_unit.cu_paganin_transfer_function(w._cu_d, np.float32(self.dz * self.delta), mu, dk,
                                                          np.int32(w._d.shape[-1]), np.int32(w._d.shape[-2]))
        w = IFT() * w
        k_delta = np.float32(2 * np.pi / w.wavelength * self.delta)
        self.processing_unit.cu_paganin_thickness_wavefront(w._cu_d, mu, k_delta)
        w.z -= self.dz
        return w
