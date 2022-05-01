# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['Imshow', 'ImshowRGBA', 'ImshowAbs', 'ImshowAngle', 'Rebin', 'QuadraticPhase', 'ThinLens', 'CircularMask',
           'RectangularMask', 'Scale', 'FT', 'IFT', 'PropagateFarField', 'PropagateNearField', 'PropagateFRT',
           'BackPropagatePaganin', 'MagnifyNearField', 'FreePU', 'BackPropagateCTF']

import types
import warnings
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.special import erfc
from pynx.utils.matplotlib import pyplot as plt

from ..utils.plot_utils import complex2rgbalin, complex2rgbalog, insertColorwheel, cm_phase
from ..utils.array import rebin
from ..operator import OperatorException, OperatorSum, OperatorPower, has_attr_not_none
from .wavefront import Wavefront, OperatorWavefront, UserWarningWavefrontNearFieldPropagation


################################################################################################
# Patch class so that we can use 5*w to scale it.
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


#################################################################################################################
###############################  Base CPU operator class  #######################################################
#################################################################################################################
class CPUOperatorWavefront(OperatorWavefront):
    """
    Base class for a operators on CDI objects using OpenCL
    """

    def __init__(self, processing_unit=None):
        super(CPUOperatorWavefront, self).__init__()

        self.Operator = CPUOperatorWavefront
        self.OperatorSum = CPUOperatorWavefrontSum
        self.OperatorPower = CPUOperatorWavefrontPower

    def apply_ops_mul(self, cdi):
        """
        Apply the series of operators stored in self.ops to a wavefront.
        In this version the operators are applied one after the other to the same wavefront (multiplication)

        :param w: the wavefront to which the operators will be applied.
        :return: the wavefront, after application of all the operators in sequence
        """
        return super(CPUOperatorWavefront, self).apply_ops_mul(cdi)

    def prepare_data(self, w):
        if has_attr_not_none(w, '_cpu_d_view') is False:
            w._cpu_d_view = {}

    def timestamp_increment(self, p):
        p._timestamp_counter += 1
        p._cpu_timestamp_counter = p._timestamp_counter

    def view_register(self, obj):
        """
        Creates a new unique view key in an object. When finished with this view, it should be de-registered
        using view_purge. Note that it only reserves the key, but does not create the view.
        :return: an integer value, which corresponds to yet-unused key in the object's view.
        """
        i = 1
        while i in obj._cpu_d_view:
            i += 1
        obj._cpu_d_view[i] = None
        return i

    def view_copy(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._obj
        else:
            src = obj._cpu_d_view[i_source]
        if i_dest == 0:
            obj._cpu_obj = np.empty_like(src)
            dest = obj._obj
        else:
            obj._cpu_d_view[i_dest] = np.empty_like(src)
            dest = obj._cpu_d_view[i_dest]
        dest[:] = src

    def view_swap(self, obj, i1, i2):
        if i1 != 0:
            if i1 not in obj._cpu_d_view:
                # Create dummy value, assume a copy will be made later
                obj._cpu_d_view[i1] = None
        if i2 != 0:
            if i2 not in obj._cpu_d_view:
                # Create dummy value, assume a copy will be made later
                obj._cpu_d_view[i2] = None
        if i1 == 0:
            obj._obj, obj._cpu_d_view[i2] = obj._cpu_d_view[i2], obj._cpu_obj
        elif i2 == 0:
            obj._obj, obj._cpu_d_view[i1] = obj._cpu_d_view[i1], obj._obj
        else:
            obj._cpu_d_view[i1], obj._cpu_d_view[i2] = obj._cpu_d_view[i2], obj._cpu_d_view[i1]

    def view_sum(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._obj
        else:
            src = obj._cpu_d_view[i_source]
        if i_dest == 0:
            dest = obj._obj
        else:
            dest = obj._cpu_d_view[i_dest]
        dest += src

    def view_purge(self, obj, i):
        if i is not None:
            del obj._cpu_d_view[i]
        elif has_attr_not_none(obj, '_cpu_d_view'):
            del obj._cpu_d_view


# The only purpose of this class is to make sure it inherits from CPUOperatorWavefront and has a processing unit
class CPUOperatorWavefrontSum(OperatorSum, CPUOperatorWavefront):
    def __init__(self, op1, op2):
        if np.isscalar(op1):
            op1 = Scale(op1)
        if np.isscalar(op2):
            op2 = Scale(op2)
        if isinstance(op1, CPUOperatorWavefront) is False or isinstance(op2, CPUOperatorWavefront) is False:
            raise OperatorException(
                "ERROR: cannot add a CPUOperatorWavefront with a non-CPUOperatorWavefront: %s + %s" % (
                    str(op1), str(op2)))
        CPUOperatorWavefront.__init__(self)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CPUOperatorWavefront
        self.OperatorSum = CPUOperatorWavefrontSum
        self.OperatorPower = CPUOperatorWavefrontPower
        self.prepare_data = types.MethodType(CPUOperatorWavefront.prepare_data, self)
        self.timestamp_increment = types.MethodType(CPUOperatorWavefront.timestamp_increment, self)
        self.view_copy = types.MethodType(CPUOperatorWavefront.view_copy, self)
        self.view_swap = types.MethodType(CPUOperatorWavefront.view_swap, self)
        self.view_sum = types.MethodType(CPUOperatorWavefront.view_sum, self)
        self.view_purge = types.MethodType(CPUOperatorWavefront.view_purge, self)


# The only purpose of this class is to make sure it inherits from CPUOperatorWavefront and has a processing unit
class CPUOperatorWavefrontPower(OperatorPower, CPUOperatorWavefront):
    def __init__(self, op, n):
        CPUOperatorWavefront.__init__(self)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CPUOperatorWavefront
        self.OperatorSum = CPUOperatorWavefrontSum
        self.OperatorPower = CPUOperatorWavefrontPower
        self.prepare_data = types.MethodType(CPUOperatorWavefront.prepare_data, self)
        self.timestamp_increment = types.MethodType(CPUOperatorWavefront.timestamp_increment, self)
        self.view_copy = types.MethodType(CPUOperatorWavefront.view_copy, self)
        self.view_swap = types.MethodType(CPUOperatorWavefront.view_swap, self)
        self.view_sum = types.MethodType(CPUOperatorWavefront.view_sum, self)
        self.view_purge = types.MethodType(CPUOperatorWavefront.view_purge, self)


#################################################################################################################
###############################  Exclusive CPU operators  #######################################################
#################################################################################################################

class Imshow(CPUOperatorWavefront):
    """
    Base class to display a 2D wavefront. Abstract class, must be derived to implement op()
    """

    def __init__(self, fig_num=None, i=0, title=None, axes=None):
        """

        :param fig_num: the matplotlib figure number. if None, a new figure will be created each time.
            Ignored if axes is given.
        :param i: if the wavefront is actually a stack of 2D wavefronts, display w.get()[i]
        :param title: the title for the view. If None, a default title will be used.
        :param axes: if given can be used to supply axes instead of a figure. In this
            case, the figure is not cleared and the image is draw on the given axes.
        """
        super(Imshow, self).__init__()
        self.fig_num = fig_num
        self.i = i
        self.title = title
        self.axes = axes

    def pre_imshow(self, w):
        if w.get().ndim == 2:
            d = fftshift(w.get())
        else:
            d = fftshift(w.get()[self.i])
        if self.axes is None:
            plt.figure(self.fig_num)
            plt.clf()
            self.axes = plt.gca()

        x, y = w.get_x_y()
        s = np.log10(max(abs(x).max(), abs(y).max()))
        if s < -6:
            unit_name = "nm"
            s = 1e9
        elif s < -3:
            unit_name = u"µm"
            s = 1e6
        elif s < 0:
            unit_name = "mm"
            s = 1e3
        else:
            unit_name = "m"
            s = 1
        return d, x * s, y * s, unit_name

    def post_imshow(self, w, x, y, unit_name, title_append=""):
        self.axes.set_xlabel("X (%s)" % unit_name)
        self.axes.set_ylabel("Y (%s)" % unit_name)
        if self.title is None:
            self.axes.set_title("Wavefront, z = %gm" % (w.z) + title_append)
        else:
            self.axes.set_title(self.title)
        try:
            plt.draw()
            plt.gcf().canvas.draw()
            plt.pause(.001)
        except:
            pass

    def timestamp_increment(self, w):
        pass


class ImshowRGBA(Imshow):
    """
    Display the complex wavefront (must be copied to numpy space first) using a RGBA view.
    """

    def __init__(self, fig_num=None, i=0, mode='linear', kwargs_complex2rgba=None, title=None, colorwheel=True,
                 axes=None):
        """

        :param fig_num: the matplotlib figure number. if None, a new figure will be created each time.
        :param i: if the wavefront is actually a stack of 2D wavefronts, display w.get()[i]
        :param mode: either 'linear' or 'log', the scaling using for the colour amplitude
        :param kwargs_complex2rgba: kwargs to be passed to complex2rgbalin or complex2rgbalog
        :param title: the title for the view. If None, a default title will be used.
        :param colorwheel: if True (thde default), plot the colorwheel
        """
        super(ImshowRGBA, self).__init__(fig_num=fig_num, i=i, title=title, axes=axes)
        self.mode = mode
        self.kwargs_complex2rgba = kwargs_complex2rgba
        self.colorwheel = colorwheel

    def op(self, w):
        d, x, y, unit_name = self.pre_imshow(w)
        if self.kwargs_complex2rgba is None:
            kw = {}
        else:
            kw = self.kwargs_complex2rgba
        if self.mode.lower() == 'linear':
            rgba = complex2rgbalin(d, **kw)
        else:
            rgba = complex2rgbalog(d, **kw)
        self.axes.imshow(rgba, extent=(x.min(), x.max(), y.min(), y.max()))
        self.post_imshow(w, x, y, unit_name)
        if self.colorwheel:
            insertColorwheel(left=.02, bottom=.0, width=.1, height=.1, text_col='black', fs=10)
        return w

    def timestamp_increment(self, w):
        pass


class ImshowAbs(Imshow):
    """
    Display the complex wavefront (must be copied to numpy space first) using a RGBA view.
    """

    def __init__(self, fig_num=None, i=0, mode='linear', cmap='gray', title=None, axes=None):
        """

        :param fig_num: the matplotlib figure number. if None, a new figure will be created each time.
        :param i: if the wavefront is actually a stack of 2D wavefronts, display w.get()[i]
        :param mode: either 'linear' or 'log', the scaling using for the colour amplitude. In both modes the display
                     min and max are the 0.1 and 99.9 percentile of the abs or log(abs) of the entire array.
        :param cmap: the colormap to use ('gray' by default)
        :param title: the title for the view. If None, a default title will be used.
        """
        super(ImshowAbs, self).__init__(fig_num=fig_num, i=i, title=title, axes=axes)
        self.mode = mode
        self.cmap = cmap

    def op(self, w):
        d, x, y, unit_name = self.pre_imshow(w)
        if 'log' in self.mode:
            a = np.log10(np.abs(d))
            vmin, vmax = np.percentile(a, [0.1, 99.9])
            r = self.axes.imshow(a, vmin=vmin, vmax=vmax, extent=(x.min(), x.max(), y.min(), y.max()),
                                 cmap=plt.cm.get_cmap(self.cmap))
        else:
            a = np.abs(d)
            vmin, vmax = np.percentile(a, [0.1, 99.9])
            r = self.axes.imshow(a, vmin=vmin, vmax=vmax, extent=(x.min(), x.max(), y.min(), y.max()),
                                 cmap=plt.cm.get_cmap(self.cmap))
        plt.colorbar(mappable=r, ax=self.axes)
        self.post_imshow(w, x, y, unit_name, title_append="[Amplitude]")
        return w

    def timestamp_increment(self, w):
        pass


class ImshowAngle(Imshow):
    """
    Display the complex wavefront (must be copied to numpy space first) using a RGBA view.
    """

    def __init__(self, fig_num=None, i=0, mode='linear', cmap=None, title=None, axes=None):
        """

        :param fig_num: the matplotlib figure number. if None, a new figure will be created each time.
        :param i: if the wavefront is actually a stack of 2D wavefronts, display w.get[i]
        :param title: the title for the view. If None, a default title will be used.
        """
        super(ImshowAngle, self).__init__(fig_num=fig_num, i=i, title=title, axes=axes)
        self.mode = mode
        self.cmap = cmap

    def op(self, w):
        d, x, y, unit_name = self.pre_imshow(w)
        if self.cmap is None:
            cmap = cm_phase
        else:
            cmap = plt.cm.get_cmap(self.cmap)
        r = self.axes.imshow(np.angle(d), extent=(x.min(), x.max(), y.min(), y.max()), cmap=cmap)
        plt.colorbar(mappable=r, ax=self.axes)
        self.post_imshow(w, x, y, unit_name, title_append="[Phase]")
        return w

    def timestamp_increment(self, w):
        pass


#################################################################################################################
########################  End of Exclusive CPU operators  #######################################################
#################################################################################################################

class Rebin(CPUOperatorWavefront):
    def __init__(self, rebin_factor=2):
        """
        Rebin the image by summing the pixels nxn, and correcting the pixel size accordingly
        :param rebin_factor: an integer number indicating how many pixels should be integrated along both dimensions.
        """
        super(Rebin, self).__init__()
        if not isinstance(rebin_factor, int):
            raise OperatorException("Rebin(n) attempted with n which is not an integer: %s" % (str(rebin_factor)))
        if rebin_factor < 1:
            raise OperatorException("Rebin(n) attempted with n=%s<1 !" % (str(rebin_factor)))

        self.rebin_factor = rebin_factor

    def op(self, w):
        if self.rebin_factor == 1:
            return w
        w._d = rebin(w._d, rebin_f=(1, self.rebin_factor, self.rebin_factor), scale="average")
        w.pixel_size *= self.rebin_factor
        return w


class FreePU(CPUOperatorWavefront):
    """
    Operator freeing GPU memory. The CPU version does nothing.
    """

    def op(self, w: Wavefront):
        """

        :param w: the Wavefront object this operator applies to
        :return: the updated Wavefront object
        """
        return w


class QuadraticPhase(CPUOperatorWavefront):
    """
    Operator applying a quadratic phase factor
    """

    def __init__(self, factor, scale=1):
        """
        Application of a quadratic phase factor, and optionnaly a scale factor.

        The actual factor is:  :math:`scale * e^{i * factor * (ix^2 + iy^2)}`
        where ix and iy are the integer indices of the pixels

        :param factor: the factor for the phase calculation.
        :param scale: the data will be scaled by this factor. Useful to normalize after a Fourier transform,
                      without accessing twice the array data.
        """
        super(QuadraticPhase, self).__init__()
        self.scale = np.float32(scale)
        self.factor = np.float32(factor)

    def op(self, w):
        ny, nx = w._d.shape[-2:]
        ix = fftshift(np.arange(-nx // 2, nx // 2))
        iy = fftshift(np.arange(-ny // 2, ny // 2))
        vy, vx = np.meshgrid(iy, ix, indexing='ij')
        w._d *= self.scale * np.exp(1j * (vx ** 2 + vy ** 2) * self.factor)
        return w


class ThinLens(CPUOperatorWavefront):
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
        ix = fftshift(np.arange(-nx // 2, nx // 2))
        iy = fftshift(np.arange(-ny // 2, ny // 2))
        vy, vx = np.meshgrid(iy, ix, indexing='ij')
        w._d *= np.exp(1j * (vx ** 2 + vy ** 2) * factor)
        return w


class CircularMask(CPUOperatorWavefront):
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
        ix = fftshift(np.arange(-nx // 2, nx // 2)) * w.pixel_size
        iy = fftshift(np.arange(-ny // 2, ny // 2)) * w.pixel_size
        vy, vx = np.meshgrid(iy, ix, indexing='ij')
        if self.invert:
            w._d *= (vx ** 2 + vy ** 2) > (self.radius ** 2)
        else:
            w._d *= (vx ** 2 + vy ** 2) <= (self.radius ** 2)
        return w


class RectangularMask(CPUOperatorWavefront):
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
        ix = fftshift(np.arange(-nx // 2, nx // 2)) * w.pixel_size
        iy = fftshift(np.arange(-ny // 2, ny // 2)) * w.pixel_size
        vy, vx = np.meshgrid(iy, ix, indexing='ij')
        if self.invert:
            w._d *= ((abs(vx) <= self.width / 2) * (abs(vy) <= self.height / 2)) == False
        else:
            w._d *= (abs(vx) <= self.width / 2) * (abs(vy) <= self.height / 2)
        return w


class Scale(CPUOperatorWavefront):
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

    def op(self, w):
        if self.x == 1:
            return w
        w._d *= self.x
        return w


class FT(CPUOperatorWavefront):
    """
    Forward Fourier transform.
    """

    def op(self, w):
        w._d = fftn(w._d, axes=(-2, -1)) / np.sqrt(w._d.shape[-2] * w._d.shape[-1])
        return w


class IFT(CPUOperatorWavefront):
    """
    Inverse Fourier transform
    """

    def op(self, w):
        w._d = ifftn(w._d, axes=(-2, -1)) * np.sqrt(w._d.shape[-2] * w._d.shape[-1])
        return w


class PropagateFarField(CPUOperatorWavefront):
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


class PropagateNearField(CPUOperatorWavefront):
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


class MagnifyNearField(CPUOperatorWavefront):
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


class PropagateFRT(CPUOperatorWavefront):
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
        if verbose:
            print("quad1=%8.3f°, quad2=%8.3f°, quad3=%8.3f°" % (
                quad1 * 180 / np.pi, quad2 * 180 / np.pi, quad3 * 180 / np.pi))

        w.z = z1
        w.pixel_size = pixel_size_z1

        if self.forward:
            return QuadraticPhase(quad1 + quad3) * IFT() * QuadraticPhase(quad2) * FT() * QuadraticPhase(quad1) * w
        else:
            return QuadraticPhase(-quad1) * IFT() * QuadraticPhase(-quad2) * FT() * QuadraticPhase(-quad1 - quad3) * w


class BackPropagatePaganin(CPUOperatorWavefront):
    """ Back-propagation algorithm using the single-projection approach.
    Ref: Paganin et al., Journal of microscopy 206 (2002), 33–40. (DOI: 10.1046/j.1365-2818.2002.01010.x)

    This operator is special since it will use only the intensity of the wavefront. Therefore it will first take the
    square modulus of the wavefront it is applied to, discarding any phase information.
    The result of the transformation is the calculated wavefront at the sample position, i.e. if T(r) is the
    estimated thickness of the sample, it is exp(-mu * T - 2*pi/lambda * T)
    """

    def __init__(self, dz=1, delta=1e-6, beta=1e-9, generalized_method=False, extract_thickness=True,
                 rebin_factor=None):
        """

        :param dz: distance between sample and detector (meter)
        :param delta: real part of the refraction index, n = 1 - delta + i * beta
        :param beta: imaginary part of the refraction index
        :param extract_thickness: if True, will save the reconstructed thickness as w.paganin_thickness
        :param rebin_factor: None or integer >1, to rebin the intensity before back-propagation
        """
        super(BackPropagatePaganin, self).__init__()
        self.dz = np.float32(dz)
        self.beta = np.float32(beta)
        self.delta = np.float32(delta)
        self.generalized_method = generalized_method
        self.extract_thickness = extract_thickness
        self.rebin_factor = rebin_factor

    def op(self, w):
        w._d = np.abs(w._d) ** 2
        if self.rebin_factor is not None:
            w._d = rebin(w._d, (1, self.rebin_factor, self.rebin_factor), scale="average")
            w.pixel_size *= self.rebin_factor
        w = FT() * w
        mu = np.float32(4 * np.pi * self.beta / w.wavelength)
        p = w.pixel_size
        dkx = np.float32(2 * np.pi / (w._d.shape[-1] * p))
        dky = np.float32(2 * np.pi / (w._d.shape[-2] * p))

        ny, nx = w._d.shape[-2:]
        kx = fftshift(np.arange(-nx // 2, nx // 2)) * dkx
        ky = fftshift(np.arange(-ny // 2, ny // 2)) * dky
        ky, kx = np.meshgrid(ky, kx, indexing='ij')
        dz = self.dz
        delta = self.delta
        if self.generalized_method:
            if True:
                # For debugging only
                w._paganin_filter_gen = 1 / (1 - 2 * dz * delta / mu / p ** 2 * (np.cos(kx * p) + np.cos(ky * p) - 2))
                w._paganin_filter = 1 / (dz * delta / mu * (kx ** 2 + ky ** 2) + 1)
            w._d *= w._paganin_filter_gen
        else:
            w._d /= dz * delta / mu * (kx ** 2 + ky ** 2) + 1

        w = IFT() * w
        k_delta = np.float32(2 * np.pi / w.wavelength * self.delta)

        t = -np.log(abs(w._d)) / mu
        w._d = np.exp((-0.5 * mu - 1j * k_delta) * t)
        w.z -= self.dz
        if self.extract_thickness:
            w.paganin_thickness = t
        return w


class BackPropagateCTF(CPUOperatorWavefront):
    """ Back-propagation algorithm using an homogeneous Contrast Transfer Function,
    suitable for weakly absorbing samples. See e.g. eq. 8 in:
    Yu et al, Optics Express, 26 (2018), 11110. http://dx.doi.org/10.1364/OE.26.011110

    This operator will first take the square modulus of the wavefront it is applied to,
    discarding any phase information.
    The result of the transformation is the calculated wavefront at the sample position.
    """

    def __init__(self, dz=1, delta_beta=1e-9, alpha_low=1e-5, alpha_high=0.2, sigma=0.01):
        """

        :param dz: distance between sample and detector (meter). The supplied wavefront is assumed
            to be at the detector position, to be back-propagated to the
        :param delta_beta: ratio of the real to the imaginary part of the refraction index
        :param alpha_low, alpha_high: regularisation factor to avoid divergence, with
            an alpha_low value at low frequencies (before the first zero of
            sin(pi*lambda*z*f**2), and an alpha_high value at larger frquencies. The transition
            uses an erfc function
        :param sigma: a factor to tune the width of the erfc function for the transition
            of the low to high frequencies in the alpha regularisation factor.

        """
        super().__init__()
        self.dz = np.float32(dz)
        self.delta_beta = np.float32(delta_beta)
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.sigma = sigma

    def op(self, w):
        w._d = np.abs(w._d) ** 2

        # This is the same as subtarcting the Dirac term in Fourier space,
        # but easier as we don't need to car about the scale factor due to the FT
        w._d -= 1

        w = FT() * w

        p = w.pixel_size
        dkx = np.float32(1 / (w._d.shape[-1] * p))
        dky = np.float32(1 / (w._d.shape[-2] * p))

        ny, nx = w._d.shape[-2:]
        kx = fftshift(np.arange(-nx // 2, nx // 2)) * dkx
        ky = fftshift(np.arange(-ny // 2, ny // 2)) * dky
        ky, kx = np.meshgrid(ky, kx, indexing='ij')
        kxy2 = kx ** 2 + ky ** 2

        # cutoff before first zero of sin(pi lambda z * kxy2)
        alpha_r = erfc((np.sqrt(kxy2) - 1 / np.sqrt(np.pi * w.wavelength * self.dz)) / (self.sigma / w.pixel_size / 2))
        alpha = self.alpha_low * (alpha_r) / 2 + self.alpha_high * (2 - alpha_r) / 2

        plz = np.float32(np.pi * w.wavelength * self.dz)
        f0 = np.cos(plz * kxy2) + self.delta_beta * np.sin(plz * kxy2)
        n = 2
        filt = 0.5 * self.delta_beta * f0 / (f0 ** (2 * n) + alpha ** n) ** (1 / n)
        w._d *= filt

        w = IFT() * w
        phi = w._d.real
        w._d = np.exp(phi / self.delta_beta + 1j * phi).astype(np.complex64)
        w.z -= self.dz
        # w.phi = phi[0]
        # w.alpha = alpha
        # w.filt = filt
        return w
