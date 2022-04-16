# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['default_processing_unit', 'FreePU', 'QuadraticPhase', 'ThinLens', 'CircularMask',
           'RectangularMask', 'Scale', 'FT', 'IFT', 'PropagateFarField', 'PropagateNearField', 'PropagateFRT',
           'BackPropagatePaganin']

import warnings
import types
import numpy as np

from ..processing_unit.cl_processing_unit import CLProcessingUnit

import pyopencl as cl
import pyopencl.array
import pyopencl.elementwise
from pyopencl.elementwise import ElementwiseKernel as CL_ElK

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
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s" % (str(x), str(self)))
        return self * Scale(x)

    cls.__rmul__ = __rmul__
    cls.__mul__ = __mul__


patch_method(Wavefront)


################################################################################################


class CLProcessingUnitWavefront(CLProcessingUnit):
    """
    Processing unit in OpenCL space, for 2D wavefront operations.
    
    Handles initializing the context and kernels.
    
    This is made to handle one array size at a time (a 2D wavefront or a stack of 2D wavefronts)
    """

    def __init__(self):
        super(CLProcessingUnitWavefront, self).__init__()

    def cl_init_kernels(self):
        self.cl_quad_phase_mult = CL_ElK(self.cl_ctx, name='cl_quad_phase_mult',
                                         operation="QuadPhaseMult(i, d, f, scale, nx, ny)",
                                         preamble=getks('wavefront/opencl/quad_phase_mult_elw.cl'),
                                         options=self.cl_options,
                                         arguments="__global float2 *d, const float f, const float scale, const int nx, const int ny")

        self.cl_scale = CL_ElK(self.cl_ctx, name='cl_scale',
                               operation="d[i] = (float2)(d[i].x * scale, d[i].y * scale )",
                               options=self.cl_options, arguments="__global float2 *d, const float scale")

        self.cl_sum = CL_ElK(self.cl_ctx, name='cl_sum',
                             operation="dest[i] += src[i]",
                             options=self.cl_options, arguments="__global float2 *src, __global float2 *dest")

        self.cl_scale_complex = CL_ElK(self.cl_ctx, name='cl_scale_complex',
                                       operation="d[i] = (float2)(d[i].x * s.x - d[i].y * s.y, d[i].x * s.y + d[i].y * s.x)",
                                       options=self.cl_options, arguments="__global float2 *d, const float2 s")

        self.cl_mask_circular = CL_ElK(self.cl_ctx, name='cl_mask_circular',
                                       operation="CircularMask(i, d, radius, pixel_size, invert, nx, ny)",
                                       preamble=getks('wavefront/opencl/mask_elw.cl'),
                                       options=self.cl_options,
                                       arguments="__global float2 *d, const float radius, const float pixel_size, const char invert, const int nx, const int ny")

        self.cl_mask_rectangular = CL_ElK(self.cl_ctx, name='cl_mask_rectangular',
                                          operation="RectangularMask(i, d, width, height, pixel_size, invert, nx, ny)",
                                          preamble=getks('wavefront/opencl/mask_elw.cl'),
                                          options=self.cl_options,
                                          arguments="__global float2 *d, const float width, const float height, const float pixel_size, const char invert, const int nx, const int ny")

        self.cl_square_modulus = CL_ElK(self.cl_ctx, name='cl_square_modulus',
                                        operation="d[i] = (float2)( d[i].x * d[i].x + d[i].y * d[i].y, 0 );",
                                        options=self.cl_options, arguments="__global float2 *d")

        self.cl_paganin_transfer_function = CL_ElK(self.cl_ctx,
                                                   name='cl_paganin_transfer_function',
                                                   operation="paganin_transfer_function(i, d, z_delta, mu, dk, nx, ny)",
                                                   preamble=getks('wavefront/opencl/paganin_elw.cl'),
                                                   options=self.cl_options,
                                                   arguments="__global float2 *d, const float z_delta, const float mu, const float dk, const int nx, const int ny")

        self.cl_paganin_thickness_wavefront = CL_ElK(self.cl_ctx,
                                                     name='cl_square_modulus',
                                                     operation="paganin_thickness_wavefront(i, d, mu, k_delta)",
                                                     preamble=getks('wavefront/opencl/paganin_elw.cl'),
                                                     options=self.cl_options,
                                                     arguments="__global float2 *d, const float mu, const float k_delta")


"""
The default processing unit 
"""
default_processing_unit = CLProcessingUnitWavefront()


class CLOperatorWavefront(OperatorWavefront):
    """
    Base class for a wavefront operator using OpenCL
    """

    def __init__(self, processing_unit=None):
        super(CLOperatorWavefront, self).__init__()

        self.Operator = CLOperatorWavefront
        self.OperatorSum = CLOperatorWavefrontSum
        self.OperatorPower = CLOperatorWavefrontPower

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

    def prepare_data(self, w):
        # Make sure data is already in OpenCL space, otherwise transfer it
        if w._timestamp_counter > w._cl_timestamp_counter:
            # print("Moving data to OpenCL space")
            w._cl_d = pyopencl.array.to_device(self.processing_unit.cl_queue, w._d, async_=False)
            w._cl_timestamp_counter = w._timestamp_counter
        if has_attr_not_none(w, '_cl_d_view') is False:
            w._cl_d_view = {}

    def timestamp_increment(self, w):
        w._cl_timestamp_counter += 1

    def view_register(self, obj):
        """
        Creates a new unique view key in an object. When finished with this view, it should be de-registered
        using view_purge. Note that it only reserves the key, but does not create the view.
        :return: an integer value, which corresponds to yet-unused key in the object's view.
        """
        i = 1
        while i in obj._cl_d_view:
            i += 1
        obj._cl_d_view[i] = None
        return i

    def view_copy(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._cl_d
        else:
            src = obj._cl_d_view[i_source]
        if i_dest == 0:
            obj._cl_d = cl.array.empty_like(src)
            dest = obj._cl_d
        else:
            obj._cl_d_view[i_dest] = cl.array.empty_like(src)
            dest = obj._cl_d_view[i_dest]
        cl.enqueue_copy(self.processing_unit.cl_queue, src=src.data, dest=dest.data)

    def view_swap(self, obj, i1, i2):
        if i1 != 0:
            if i1 not in obj._cl_d_view:
                # Create dummy value, assume a copy will be made later
                obj._cl_d_view[i1] = None
        if i2 != 0:
            if i2 not in obj._cl_d_view:
                # Create dummy value, assume a copy will be made later
                obj._cl_d_view[i2] = None
        if i1 == 0:
            obj._cl_d, obj._cl_d_view[i2] = obj._cl_d_view[i2], obj._cl_d
        elif i2 == 0:
            obj._cl_d, obj._cl_d_view[i1] = obj._cl_d_view[i1], obj._cl_d
        else:
            obj._cl_d_view[i1], obj._cl_d_view[i2] = obj._cl_d_view[i2], obj._cl_d_view[i1]

    def view_sum(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._cl_d
        else:
            src = obj._cl_d_view[i_source]
        if i_dest == 0:
            dest = obj._cl_d
        else:
            dest = obj._cl_d_view[i_dest]
        self.processing_unit.cl_sum(src, dest)

    def view_purge(self, obj, i):
        if i is not None:
            del obj._cl_d_view[i]
        elif has_attr_not_none(obj, '_cl_d_view'):
            del obj._cl_d_view
            self.processing_unit.cl_queue.finish()  # is this useful ?


# The only purpose of this class is to make sure it inherits from CLOperatorWavefront and has a processing unit
class CLOperatorWavefrontSum(OperatorSum, CLOperatorWavefront):
    def __init__(self, op1, op2):
        if np.isscalar(op1):
            op1 = Scale(op1)
        if np.isscalar(op2):
            op2 = Scale(op2)
        if isinstance(op1, CLOperatorWavefront) is False or isinstance(op2, CLOperatorWavefront) is False:
            raise OperatorException(
                "ERROR: cannot add a CLOperatorWavefront with a non-CLOperatorWavefront: %s + %s" % (
                    str(op1), str(op2)))
        # We can only have a sum of two CLOperatorWavefront, so they must have a processing_unit attribute.
        CLOperatorWavefront.__init__(self, op1.processing_unit)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorWavefront
        self.OperatorSum = CLOperatorWavefrontSum
        self.OperatorPower = CLOperatorWavefrontPower
        self.prepare_data = types.MethodType(CLOperatorWavefront.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorWavefront.timestamp_increment, self)
        self.view_copy = types.MethodType(CLOperatorWavefront.view_copy, self)
        self.view_swap = types.MethodType(CLOperatorWavefront.view_swap, self)
        self.view_sum = types.MethodType(CLOperatorWavefront.view_sum, self)
        self.view_purge = types.MethodType(CLOperatorWavefront.view_purge, self)


# The only purpose of this class is to make sure it inherits from CLOperatorWavefront and has a processing unit
class CLOperatorWavefrontPower(OperatorPower, CLOperatorWavefront):
    def __init__(self, op, n):
        CLOperatorWavefront.__init__(self, op.processing_unit)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorWavefront
        self.OperatorSum = CLOperatorWavefrontSum
        self.OperatorPower = CLOperatorWavefrontPower
        self.prepare_data = types.MethodType(CLOperatorWavefront.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorWavefront.timestamp_increment, self)
        self.view_copy = types.MethodType(CLOperatorWavefront.view_copy, self)
        self.view_swap = types.MethodType(CLOperatorWavefront.view_swap, self)
        self.view_sum = types.MethodType(CLOperatorWavefront.view_sum, self)
        self.view_purge = types.MethodType(CLOperatorWavefront.view_purge, self)


class CopyToPrevious(CLOperatorWavefront):
    """
    Operator which will store a copy of the wavefront as cl_d_previous. This is used for various algorithms, such
    as difference map or RAAR
    """

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        if has_attr_not_none(w, '_cl_d_previous') is False:
            w._cl_d_previous = cl.array.empty_like(w._cl_d)
        if w._cl_d_previous.shape == w._cl_d.shape:
            w._cl_d_previous = cl.array.empty_like(w._cl_d)
        cl.enqueue_copy(self.processing_unit.cl_queue, w._cl_d_previous.data, w._cl_d.data)
        return w


class FromPU(CLOperatorWavefront):
    """
    Operator copying back the wavefront data from the opencl device to numpy.
    
    DEPRECATED
    """

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        warnings.warn("Use of ToPU() and FromPU() operators is now deprecated. Use get() and set() to access data.")
        w.get()
        # w.d[:] = w._cl_d.get()
        return w


class ToPU(CLOperatorWavefront):
    """
    Operator copying the wavefront data from numpy to the opencl device, as a complex64 array.
    
    DEPRECATED
    """

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        warnings.warn("Use of ToPU() and FromPU() operators is now deprecated. Use get() and set() to access data.")
        # w._cl_d = pyopencl.array.to_device(self.processing_unit.cl_queue, w.d, async_=False)
        return w


class FreePU(CLOperatorWavefront):
    """
    Operator freeing OpenCL memory. The gpyfft data reference in self.processing_unit is removed,
    as well as any OpenCL pyopencl.array.Array attribute in the supplied wavefront.
    """

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        w.get()  # This will copy the data to the host memory if necessary
        for o in dir(w):
            if isinstance(w.__getattribute__(o), pyopencl.array.Array):
                w.__setattr__(o, None)
        self.processing_unit.free_fft_plans()
        self.view_purge(w, None)
        return w

    def timestamp_increment(self, cdi):
        cdi._timestamp_counter += 1


class FreeFromPU(CLOperatorWavefront):
    """
    Gets back data from OpenCL and removes all OpenCL arrays.
    
    DEPRECATED
    """

    def __new__(cls):
        return FreePU() * FromPU()


class QuadraticPhase(CLOperatorWavefront):
    """
    Operator applying a quadratic phase factor
    """

    def __init__(self, factor, scale=1):
        """
        Application of a quadratic phase factor, and optionally a scale factor.
        
        The actual factor is:  :math:`scale * e^{i * factor * (ix^2 + iy^2)}`
        where ix and iy are the integer indices of the pixels
        
        :param factor: the factor for the phase calculation.
        :param scale: the data will be scaled by this factor. Useful to normalize after a Fourier transform,
                      without accessing twice the array data.
        """
        super(QuadraticPhase, self).__init__()
        self.scale = np.float32(scale)
        self.factor = np.float32(factor)

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        self.processing_unit.cl_quad_phase_mult(w._cl_d, self.factor, self.scale, np.int32(w._d.shape[-1]),
                                                np.int32(w._d.shape[-2]))
        return w


class ThinLens(CLOperatorWavefront):
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

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        factor = np.float32(-np.pi * w.pixel_size ** 2 / (self.focal_length * w.wavelength))
        nx, ny = np.int32(w._d.shape[-1]), np.int32(w._d.shape[-2])
        # Calculate delta(phase) at half-distance from center zone and warn if necessary
        dphi = abs(factor * nx / 2)
        if dphi > np.pi:
            warnings.warn(
                "ThinLens Operator: d(phase factor)/pixel is %5.2f>PI (@half -distance from center)! Aliasing may occur" % dphi)
        self.processing_unit.cl_quad_phase_mult(w._cl_d, factor, np.float32(1), nx, ny)
        return w


class CircularMask(CLOperatorWavefront):
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

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        nx, ny = np.int32(w._d.shape[-1]), np.int32(w._d.shape[-2])
        self.processing_unit.cl_mask_circular(w._cl_d, self.radius, w.pixel_size, self.invert, nx, ny)
        return w


class RectangularMask(CLOperatorWavefront):
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

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        nx, ny = np.int32(w._d.shape[-1]), np.int32(w._d.shape[-2])
        self.processing_unit.cl_mask_rectangular(w._cl_d, self.width, self.height, w.pixel_size, self.invert,
                                                 nx, ny)
        return w


class Scale(CLOperatorWavefront):
    """
    Multiply the wavefront by a scalar (real or complex).
    """

    def __init__(self, x):
        """

        :param x: the scaling factor
        """
        super(Scale, self).__init__()
        self.x = x

    def __str__(self):
        return str(self.x)

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        if self.x == 1:
            return w
        if np.isreal(self.x):
            self.processing_unit.cl_scale(w._cl_d, np.float32(self.x))
        else:
            self.processing_unit.cl_scale_complex(w._cl_d, np.complex64(self.x))
        return w


class FT(CLOperatorWavefront):
    """
    Forward Fourier transform.
    """

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        self.processing_unit.fft(w._cl_d, w._cl_d, ndim=2, norm=True)
        return w


class IFT(CLOperatorWavefront):
    """
    Inverse Fourier transform
    """

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        self.processing_unit.ifft(w._cl_d, w._cl_d, ndim=2, norm=True)
        return w


class PropagateFarField(CLOperatorWavefront):
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

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        verbose = True
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


class PropagateNearField(CLOperatorWavefront):
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

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
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


class MagnifyNearField(CLOperatorWavefront):
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

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
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


class PropagateFRT(CLOperatorWavefront):
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

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
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
        if verbose:
            print("quad1=%8.3f°, quad2=%8.3f°, quad3=%8.3f°" % (
                quad1 * 180 / np.pi, quad2 * 180 / np.pi, quad3 * 180 / np.pi))

        w.z = z1
        w.pixel_size = pixel_size_z1

        if self.forward:
            return QuadraticPhase(quad1 + quad3) * IFT() * QuadraticPhase(quad2) * FT() * QuadraticPhase(quad1) * w
        else:
            return QuadraticPhase(-quad1) * IFT() * QuadraticPhase(-quad2) * FT() * QuadraticPhase(-quad1 - quad3) * w


class BackPropagatePaganin(CLOperatorWavefront):
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

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        self.processing_unit.cl_square_modulus(w._cl_d)
        w = FT() * w
        mu = np.float32(4 * np.pi * self.beta / w.wavelength)
        dk = np.float32(2 * np.pi / (w._d.shape[-1] * w.pixel_size))
        self.processing_unit.cl_paganin_transfer_function(w._cl_d, np.float32(self.dz * self.delta), mu, dk,
                                                          np.int32(w._d.shape[-1]), np.int32(w._d.shape[-2]))
        w = IFT() * w
        k_delta = np.float32(2 * np.pi / w.wavelength * self.delta)
        self.processing_unit.cl_paganin_thickness_wavefront(w._cl_d, mu, k_delta)
        w.z -= self.dz
        return w
