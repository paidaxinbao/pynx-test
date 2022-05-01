# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['default_processing_unit', 'ImshowRGBA', 'ShowCDI', 'SupportUpdate', 'ScaleObj', 'AutoCorrelationSupport',
           'FreePU', 'FT', 'IFT', 'FourierApplyAmplitude', 'ER', 'CF', 'HIO', 'RAAR', 'GPS', 'ML', 'SupportUpdate',
           'ScaleObj', 'LLK', 'LLKSupport', 'DetwinHIO', 'DetwinRAAR', 'SupportExpand', 'ObjConvolve', 'EstimatePSF',
           'InitPSF', 'InitSupportShape', 'InitObjRandom', 'InitFreePixels']

import timeit
import types
import warnings
from random import randint
import numpy as np
from skimage.restoration.deconvolution import richardson_lucy
from scipy.fftpack import fftn, ifftn, fftshift, fftfreq
from numpy.fft import rfftn, irfftn
from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import center_of_mass

from pynx.utils.matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from ..utils.plot_utils import complex2rgbalin, complex2rgbalog, insertColorwheel, cm_phase
from ..utils.math import llk_euclidian, llk_gaussian, llk_poisson
from ..operator import OperatorException, has_attr_not_none, OperatorSum, OperatorPower
from .cdi import OperatorCDI, CDI, SupportTooSmall, SupportTooLarge
from ..processing_unit import default_processing_unit


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


#################################################################################################################
###############################  Base CPU operator class  #######################################################
#################################################################################################################

class CPUOperatorCDI(OperatorCDI):
    """
    Base class for a operators on CDI objects using OpenCL
    """

    def __init__(self, processing_unit=None, lazy=False):
        super(CPUOperatorCDI, self).__init__(lazy=lazy)

        self.Operator = CPUOperatorCDI
        self.OperatorSum = CPUOperatorCDISum
        self.OperatorPower = CPUOperatorCDIPower

    def apply_ops_mul(self, cdi: CDI):
        """
        Apply the series of operators stored in self.ops to a CDI object.
        In this version the operators are applied one after the other to the same CDI object (multiplication)

        :param cdi: the CDI object to which the operators will be applied.
        :return: the CDI object, after application of all the operators in sequence
        """
        return super(CPUOperatorCDI, self).apply_ops_mul(cdi)

    def prepare_data(self, cdi):
        if has_attr_not_none(cdi, '_cpu_obj_view') is False:
            cdi._obj_view = {}

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
        while i in obj._cpu_obj_view:
            i += 1
        obj._cpu_obj_view[i] = None
        return i

    def view_copy(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._obj
        else:
            src = obj._cpu_obj_view[i_source]
        if i_dest == 0:
            obj._cpu_obj = np.empty_like(src)
            dest = obj._obj
        else:
            obj._cpu_obj_view[i_dest] = np.empty_like(src)
            dest = obj._cpu_obj_view[i_dest]
        dest[:] = src

    def view_swap(self, obj, i1, i2):
        if i1 != 0:
            if i1 not in obj._cpu_obj_view:
                # Create dummy value, assume a copy will be made later
                obj._cpu_obj_view[i1] = None
        if i2 != 0:
            if i2 not in obj._cpu_obj_view:
                # Create dummy value, assume a copy will be made later
                obj._cpu_obj_view[i2] = None
        if i1 == 0:
            obj._obj, obj._cpu_obj_view[i2] = obj._cpu_obj_view[i2], obj._cpu_obj
        elif i2 == 0:
            obj._obj, obj._cpu_obj_view[i1] = obj._cpu_obj_view[i1], obj._obj
        else:
            obj._cpu_obj_view[i1], obj._cpu_obj_view[i2] = obj._cpu_obj_view[i2], obj._cpu_obj_view[i1]

    def view_sum(self, obj, i_source, i_dest):
        if i_source == 0:
            src = obj._obj
        else:
            src = obj._cpu_obj_view[i_source]
        if i_dest == 0:
            dest = obj._obj
        else:
            dest = obj._cpu_obj_view[i_dest]
        dest += src

    def view_purge(self, obj, i):
        if i is not None:
            del obj._cpu_obj_view[i]
        elif has_attr_not_none(obj, '_cpu_obj_view'):
            del obj._cpu_obj_view


# The only purpose of this class is to make sure it inherits from CPUOperatorCDI and has a processing unit
class CPUOperatorCDISum(OperatorSum, CPUOperatorCDI):
    def __init__(self, op1, op2):
        if np.isscalar(op1):
            op1 = Scale(op1)
        if np.isscalar(op2):
            op2 = Scale(op2)
        if isinstance(op1, CPUOperatorCDI) is False or isinstance(op2, CPUOperatorCDI) is False:
            raise OperatorException(
                "ERROR: cannot add a CPUOperatorCDI with a non-CPUOperatorCDI: %s + %s" % (str(op1), str(op2)))
        CPUOperatorCDI.__init__(self)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CPUOperatorCDI
        self.OperatorSum = CPUOperatorCDISum
        self.OperatorPower = CPUOperatorCDIPower
        self.prepare_data = types.MethodType(CPUOperatorCDI.prepare_data, self)
        self.timestamp_increment = types.MethodType(CPUOperatorCDI.timestamp_increment, self)
        self.view_copy = types.MethodType(CPUOperatorCDI.view_copy, self)
        self.view_swap = types.MethodType(CPUOperatorCDI.view_swap, self)
        self.view_sum = types.MethodType(CPUOperatorCDI.view_sum, self)
        self.view_purge = types.MethodType(CPUOperatorCDI.view_purge, self)


# The only purpose of this class is to make sure it inherits from CPUOperatorCDI and has a processing unit
class CPUOperatorCDIPower(OperatorPower, CPUOperatorCDI):
    def __init__(self, op, n):
        CPUOperatorCDI.__init__(self)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CPUOperatorCDI
        self.OperatorSum = CPUOperatorCDISum
        self.OperatorPower = CPUOperatorCDIPower
        self.prepare_data = types.MethodType(CPUOperatorCDI.prepare_data, self)
        self.timestamp_increment = types.MethodType(CPUOperatorCDI.timestamp_increment, self)
        self.view_copy = types.MethodType(CPUOperatorCDI.view_copy, self)
        self.view_swap = types.MethodType(CPUOperatorCDI.view_swap, self)
        self.view_sum = types.MethodType(CPUOperatorCDI.view_sum, self)
        self.view_purge = types.MethodType(CPUOperatorCDI.view_purge, self)


#################################################################################################################
###############################  Exclusive CPU operators  #######################################################
#################################################################################################################


class ImshowRGBA(OperatorCDI):
    """
    Display the complex object (must be copied to numpy space first) using a RGBA view.
    """

    def __init__(self, fig_num=None, i=None, mode='linear', kwargs_complex2rgba=None):
        """

        :param fig_num: the matplotlib figure number. if None, a new figure will be created each time.
        :param i: if the object is 3D, display the ith plane (default: the center one)
        :param mode: either 'linear' or 'log', the scaling using for the colour amplitude
        :param kwargs_complex2rgba: kwargs to be passed to complex2rgbalin or complex2rgbalog
        """
        super(ImshowRGBA, self).__init__()
        self.fig_num = fig_num
        self.i = i
        self.mode = mode
        self.kwargs_complex2rgba = kwargs_complex2rgba

    def op(self, cdi):
        support = None
        if cdi.get_obj().ndim == 2:
            d = cdi.get_obj(shift=True)
            x, y = cdi.get_x_y()
            if cdi._support is not None:
                support = fftshift(cdi.get_support())
        else:
            if self.i is not None:
                i = self.i - len(cdi.get_obj()) // 2
            else:
                v = np.sum(abs(cdi.get_obj()), axis=(1, 2))
                i = int(round(center_of_mass(fftshift(v))[0])) - len(cdi.get_obj()) // 2
            d = fftshift(cdi.get_obj()[i])
            x, y, z = cdi.get_x_y()
            if cdi._support is not None:
                support = fftshift(cdi.get_support()[i])
        if self.kwargs_complex2rgba is None:
            kw = {}
        else:
            kw = self.kwargs_complex2rgba
        if self.mode.lower() == 'linear':
            rgba = complex2rgbalin(d, **kw)
        else:
            rgba = complex2rgbalog(d, **kw)
        plt.figure(self.fig_num)
        plt.clf()
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
        plt.imshow(rgba, extent=(x.min() * s, x.max() * s, y.min() * s, y.max() * s), origin='lower')
        plt.xlabel("X (%s)" % (unit_name))
        plt.ylabel("Y (%s)" % (unit_name))
        if support is not None:
            if support.sum():
                ix = np.nonzero(support.sum(axis=0))[0].take([0, -1])
                vx = fftshift(x.flat)[ix]
                dx = vx[1] - vx[0]
                plt.xlim(((vx[0] - dx * 0.1) * s, (vx[1] + dx * 0.1) * s))

                iy = np.nonzero(support.sum(axis=1))[0].take([0, -1])
                vy = fftshift(y.flat)[iy]
                dy = vy[1] - vy[0]
                plt.ylim(((vy[0] - dy * 0.1) * s, (vy[1] + dy * 0.1) * s))
        plt.title("Object")
        insertColorwheel(left=.02, bottom=.0, width=.1, height=.1, text_col='black', fs=10)
        try:
            plt.draw()
            plt.gcf().canvas.draw()
            plt.pause(.001)
        except:
            pass
        return cdi

    def timestamp_increment(self, w):
        pass


class ShowCDI(OperatorCDI):
    """
        Plot the current estimate of the object amplitude and phase, as well as a comparison of the calculated
        and observed intensities. For 3D data, a 2D cut is shown.
        
        NB: the object must be copied from PU space before
        NB: this is a CPU version, which will not display the calculated intensity
    """

    def __init__(self, fig_num=-1, i=None):
        """

        :param fig_num: the matplotlib figure number. if None, a new figure will be created each time.
            If -1 (the default), the current figure will be re-used.
        :param i: if the object is 3D, display the ith plane (default: the center one)
        """
        super(ShowCDI, self).__init__()
        self.fig_num = fig_num
        self.i = i

    def op(self, cdi):
        if cdi._obj.ndim == 3:
            obj = cdi.get_obj()
            support = cdi.get_support()
            if self.i is not None:
                i = self.i - len(obj) // 2
            else:
                v = np.sum(support, axis=(1, 2))
                c = center_of_mass(fftshift(v))
                i = int(round(c[0])) - len(obj) // 2
            # TODO: handle crop/bin/upsample options
            obj = obj[i]
            iobs = cdi.get_iobs()[0].copy()
            icalc = self.get_icalc(cdi, 0)
            support = support[i]
        else:
            obj = cdi.get_obj()
            iobs = cdi.get_iobs().copy()
            icalc = self.get_icalc(cdi)
            support = cdi.get_support()

        tmp = np.logical_and(iobs > -1e19, iobs < 0)
        if tmp.sum() > 0:
            # change back free pixels to their real intensity
            iobs[tmp] = -iobs[tmp] - 1
        iobs[iobs < 0] = 0

        if self.fig_num != -1:
            plt.figure(self.fig_num)
        else:
            plt.gcf()
        plt.clf()
        plt.subplot(221)
        if support is not None:
            tmp = fftshift(support)
            # Scale so that the max is at the 99 percentile relative to the number of points inside the support
            percent = 100 * (1 - 0.01 * support.sum() / support.size)
            max99 = np.percentile(abs(obj), percent)
            plt.imshow(fftshift(abs(obj)), origin='lower', vmin=0, vmax=max99, cmap=plt.cm.get_cmap('gray'))
            if tmp.sum():
                plt.xlim(np.nonzero(tmp.sum(axis=0))[0].take([0, -1]) + np.array([-10, 10]))
                plt.ylim(np.nonzero(tmp.sum(axis=1))[0].take([0, -1]) + np.array([-10, 10]))
            plt.colorbar()
        else:
            plt.imshow(fftshift(abs(obj)), origin='lower', cmap=plt.cm.get_cmap('gray'))
        plt.title('Object Amplitude')

        plt.subplot(222)
        if support is not None:
            tmp = fftshift(support)
            p = np.ma.masked_array(fftshift(np.angle(obj)), mask=(tmp == 0))
            plt.imshow(p, vmin=-np.pi, vmax=np.pi, origin='lower', cmap=cm_phase)
            if tmp.sum():
                plt.xlim(np.nonzero(tmp.sum(axis=0))[0].take([0, -1]) + np.array([-10, 10]))
                plt.ylim(np.nonzero(tmp.sum(axis=1))[0].take([0, -1]) + np.array([-10, 10]))
            plt.colorbar()
        else:
            plt.imshow(fftshift(np.angle(obj)), cmap=cm_phase)
        plt.title('Object Phase')

        mi, ma = max(iobs.min(), 0.5), iobs.max()
        plt.subplot(223)
        if icalc is not None:
            plt.imshow(fftshift(icalc), norm=LogNorm(vmin=mi, vmax=ma), origin='lower')
        plt.title('Calculated intensity')

        plt.subplot(224)
        plt.imshow(fftshift(iobs), norm=LogNorm(vmin=mi, vmax=ma), origin='lower')
        plt.title('Observed intensity')

        llk = cdi.get_llk(normalized=True)
        nbs = cdi.nb_point_support
        s = "#%3d LLK= %7.3f[free=%7.3f](p) nb_ph=%e\n" \
            "support:nb=%6d (%6.3f%%) <obj>=%10.2f max=%10.2f, out=%4.3f%%" % (
                cdi.cycle, llk[0], llk[3], cdi.nb_photons_calc,
                cdi.nb_point_support, nbs / cdi._obj.size * 100,
                np.sqrt(cdi.nb_photons_calc / nbs), cdi._obj_max, cdi._obj2_out * 100)
        plt.suptitle(s, fontsize=10)

        try:
            plt.draw()
            plt.gcf().canvas.draw()
            plt.pause(.002)
        except:
            pass
        return cdi

    @staticmethod
    def get_icalc(cdi: CDI, i=None):
        """
        This static, virtual function is used to get icalc, and should be derived depending on the GPU used.
        
        :param cdi: the cdi object from which to extract the calculated intensity
        :param i: if data is 3D, the index if the frame to extract
        :return: the calculated intensity, as a float32 numpy array, or None if the intensity could not be calculated
        """
        if cdi.in_object_space():
            cdi = FT(scale=False) * cdi
            icalc = abs(cdi.get_obj()) ** 2
            cdi = IFT(scale=False) * cdi
        else:
            icalc = abs(cdi.get_obj()) ** 2
        if icalc.ndim == 3 and i is not None:
            return icalc[i]
        return icalc

    def timestamp_increment(self, w):
        pass


#################################################################################################################
########################  End of Exclusive CPU operators  #######################################################
#################################################################################################################


class AutoCorrelationSupport(CPUOperatorCDI):
    """
    Operator to calculate an initial support from the auto-correlation function of the observed intensity.
    The object will be multiplied by the resulting support.
    """

    def __init__(self, threshold=0.2, verbose=False, lazy=False):
        """
        :param threshold: pixels above the autocorrelation maximum multiplied by the threshold will be included
                          in the support. This can either be a float, or a range tuple (min,max) between
                          which the threshold value will be randomly chosen every time the operator
                          is applied.
        :param verbose: if True, print info about the result of the auto-correlation
        :param lazy: if True, this will be queued for later execution in the cdi object
        """
        super(AutoCorrelationSupport, self).__init__(lazy=lazy)
        self.threshold = threshold
        self.verbose = verbose

    def op(self, cdi: CDI):
        t = self.threshold
        if isinstance(t, list) or isinstance(t, tuple):
            t = np.random.uniform(t[0], t[1])

        tmp = np.abs(fftn((cdi.iobs * (cdi.iobs >= 0)).astype(np.complex64)))
        thres = tmp.max() * t
        cdi._support = (tmp > thres).astype(np.int8)
        cdi.nb_point_support = cdi._support.sum()
        if self.verbose:
            print('AutoCorrelation: %d pixels in support (%6.2f%%), threshold = %f (relative = %5.3f)' %
                  (cdi.nb_point_support, cdi.nb_point_support * 100 / tmp.size, thres, t))
        cdi._obj *= cdi._support
        return cdi


class InitSupportShape(CPUOperatorCDI):
    """Init the support using a description of the shape or a formula. An alternative
    to AutoCorrelationSupport when the centre of the diffraction is hidden behind
    a beamstop.
    """

    def __init__(self, shape="circle", size=None, formula=None, verbose=False, lazy=False):
        """

        :param shape: either "circle" (same a "sphere"), square (same as "cube").
            Ignored if formula is not None.
        :param size: the radius of the circle/sphere or the half-size of the square/cube.
            This can also be a value per dimension
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
                self.formula = "(x*x/%f + y*y/%f + z*z/%f)<1" % (sx ** 2, sy ** 2, sz ** 2)
            elif shape.lower() in ["square", "cube"]:
                self.formula = "(abs(x)<%f) * (abs(y)<%f) * (abs(z)<%f)" % (sx, sy, sz)
            else:
                raise OperatorException("InitSupportShape: shape should be among: circle, sphere, square or cube")
        else:
            self.formula = formula
        self.verbose = verbose

    def op(self, cdi: CDI):
        if cdi.iobs.ndim == 3:
            nz, ny, nx = cdi.iobs.shape
            z = np.arange(-nz // 2, nz // 2, dtype=np.float32)
            y = np.arange(-ny // 2, ny // 2, dtype=np.float32)
            x = np.arange(-nx // 2, nx // 2, dtype=np.float32)
            z, y, x = np.meshgrid(z, y, x, indexing='ij')
        else:
            ny, nx = cdi.iobs.shape
            y = np.arange(-ny // 2, ny // 2, dtype=np.float32)
            x = np.arange(-nx // 2, nx // 2, dtype=np.float32)
            y, x = np.meshgrid(y, x, indexing='ij')
            z = 0
        sqrt = np.sqrt
        cdi.set_support(eval(self.formula), shift=True)
        if self.verbose:
            nb = cdi.nb_point_support
            print("Init support (CUDA) using formula: %s, %d pixels in support [%6.3f%%]" %
                  (self.formula, nb, 100 * nb / cdi.iobs.size))
        return cdi


class InitObjRandom(CPUOperatorCDI):
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
            can also be an array of the appropriate shape, fft-shifted so its centre is at 0.
        :param amin, amax: min and max of the random uniform values for the amplitude
        :param phirange: range of the random uniform values for the amplitude
        :param lazy: if True, this will be queued for later execution in the cdi object
        """
        super(InitObjRandom, self).__init__(lazy=lazy)
        self.src = src
        self.amin = amin
        self.amax = amax
        self.phirange = phirange

    def op(self, cdi: CDI):
        if isinstance(self.src, np.ndarray):
            src = self.src.copy()
        elif "obj" in self.src.lower():
            src = cdi.get_obj()
        else:
            src = cdi.get_support()
        if np.isclose(self.amin, self.amax) and np.isclose(self.phirange, 0):
            cdi.set_obj(src)
        else:
            tmp = abs(src) > 0
            idx = np.flatnonzero(tmp)
            a = np.random.uniform(self.amin, self.amax, idx.size).astype(np.complex64)
            phi = np.random.uniform(0, self.phirange, idx.size).astype(np.complex64)
            v = src.astype(np.complex64)
            v[tmp] *= a * np.exp(1j * phi)
            cdi.set_obj(v)
        return cdi


class InitFreePixels(CPUOperatorCDI):
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
        cdi.init_free_pixels(ratio=self.ratio, island_radius=self.island_radius,
                             exclude_zone_center=self.exclude_zone_center, coords=self.coords,
                             verbose=self.verbose)
        return cdi


class CopyToPrevious(CPUOperatorCDI):
    """
    Operator which will store a copy of the cdi object as _obj_previous. This is used for various algorithms, such
    as difference map or RAAR
    """

    def op(self, cdi):
        cdi._obj_previous = cdi._obj.copy()
        return cdi


class FromPU(CPUOperatorCDI):
    """
    Operator copying back the CDI object and support data from the opencl device to numpy. The calculated complex
    amplitude is also retrieved by computing the Fourier transform of the current view of the object.

    DEPRECATED
    """

    def op(self, cdi):
        warnings.warn("Use of ToPU() and FromPU() operators is now deprecated. Use get() and set() to access data.")
        return cdi


class ToPU(CPUOperatorCDI):
    """
    Operator copying the CDI data from numpy to the opencl device, as a complex64 array.

    DEPRECATED
    """

    def op(self, cdi):
        warnings.warn("Use of ToPU() and FromPU() operators is now deprecated. Use get() and set() to access data.")
        return cdi


class FreePU(CPUOperatorCDI):
    """
    Operator freeing GPU memory. For CPU operator, will only free temporary arryas.
    """

    def op(self, cdi):
        # Get back last object and support
        cdi._obj_previous = None
        self.view_purge(cdi, None)
        return cdi


class FreeFromPU(CPUOperatorCDI):
    """
    Gets back data from OpenCL and removes all OpenCL arrays.

    DEPRECATED
    """

    def __new__(cls):
        return FreePU() * FromPU()


class Scale(CPUOperatorCDI):
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
        cdi._obj *= self.x
        return cdi


class FT(CPUOperatorCDI):
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
        if self.scale is True:
            cdi._obj = fftn(cdi._obj, overwrite_x=True) / np.sqrt(cdi._obj.size)
        else:
            cdi._obj = fftn(cdi._obj, overwrite_x=True)
            if (self.scale is not False) and (self.scale is not None):
                cdi = Scale(self.scale) * cdi
        cdi._is_in_object_space = False

        return cdi


class IFT(CPUOperatorCDI):
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
        if self.scale is True:
            cdi._obj = ifftn(cdi._obj, overwrite_x=True) * np.sqrt(cdi._obj.size)
        else:
            cdi._obj = ifftn(cdi._obj, overwrite_x=True)
            if (self.scale is not False) and (self.scale is not None):
                cdi = Scale(self.scale) * cdi
        cdi._is_in_object_space = True
        return cdi


class Calc2Obs(CPUOperatorCDI):
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
            cdi.iobs = (np.abs(cdi._obj) ** 2).astype(np.float32)
            cdi = IFT(scale=False) * cdi
        else:
            cdi.iobs = (np.abs(cdi._obj) ** 2).astype(np.float32)
        return cdi


class ApplyAmplitude(CPUOperatorCDI):
    """
    Apply the magnitude from an observed intensity, keep the phase.
    """

    def __init__(self, calc_llk=False, zero_mask=False, scale_in=1, scale_out=1, confidence_interval_factor=0,
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
                                           strict observed amplitude projection. [TODO]
        :param update_psf: if True, will update the PSF convolution kernel using
            the Richard-Lucy deconvolution approach. If there is no PSF, it will be automatically
            initialised. [TODO]
        :param psf_filter: either None, "hann" or "tukey": window type to filter the PSF update
        """
        super(ApplyAmplitude, self).__init__()
        self.calc_llk = calc_llk
        self.scale_in = np.float32(scale_in)
        self.scale_out = np.float32(scale_out)
        self.zero_mask = np.int8(zero_mask)
        self.confidence_interval_factor = np.float32(confidence_interval_factor)
        self.update_psf = update_psf
        self.psf_filter = psf_filter

    def op(self, cdi):
        if self.scale_in != np.float32(1):
            cdi._obj *= self.scale_in
        if self.calc_llk:
            cdi = LLK() * cdi
        # TODO: partial coherence convolution
        calc = abs(cdi._obj)
        r = np.sqrt(np.fmax(cdi.iobs, 0)) / np.fmax(1e-12, calc)
        if self.zero_mask:
            r[cdi.iobs < 0] = 0
        else:
            r[cdi.iobs < 0] = 1
        cdi._obj *= r
        if self.scale_out is not None:
            cdi._obj *= self.scale_out

        # if self.update_psf:
        #     for i in range(5):
        #         psf_f = rfftn(psf)
        #         # convolve(icalc, psf)
        #         icalc_f = rfftn(icalc)
        #         icalc_psf_f = icalc_f * psf_f
        #         icalc_psf = np.maximum(irfftn(icalc_psf_f), 0)
        #
        #         # iobs / convolve(icalc,psf)
        #         iobs_icalc_psf = (iobs) / (icalc_psf)
        #         # iobs_icalc_psf = (iobs+2) / (icalc_psf+2)
        #         # iobs_icalc_psf = 1+tukey_bisquare((iobs) / (icalc_psf)  #tukey_bisquare:TODO
        #
        #         # convolve(iobs / convolve(icalc,psf), icalc_mirror)
        #         iobs_icalc_psf_f = rfftn(iobs_icalc_psf) * icalc_f.conj()
        #         psf = np.maximum(psf * irfftn(iobs_icalc_psf_f), np.float32(1e-20))

        return cdi


class UpdatePSF(CPUOperatorCDI):
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
        # TODO: implement PSF for CPU
        return cdi


class FourierApplyAmplitude(CPUOperatorCDI):
    """
    Fourier magnitude operator, performing a Fourier transform, the magnitude projection, and a backward FT.
    """

    def __new__(cls, calc_llk=False, zero_mask=False, update_psf=False, psf_filter=None):
        return IFT(scale=False) * ApplyAmplitude(calc_llk=calc_llk, zero_mask=zero_mask,
                                                 update_psf=update_psf, psf_filter=psf_filter) * FT(scale=False)


class ERProj(CPUOperatorCDI):
    """
    Error reduction.
    """

    def __init__(self, positivity=False):
        super(ERProj, self).__init__()
        self.positivity = positivity

    def op(self, cdi):
        if self.positivity:
            cdi._obj *= cdi._support * (cdi._obj.real > 0)
        else:
            cdi._obj *= cdi._support
        return cdi


class ER(CPUOperatorCDI):
    """
    Error reduction cycle
    """

    def __init__(self, positivity=False, calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1, zero_mask=False,
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
        :param update_psf: if >0, will update the partial coherence psf every update_psf cycles. [TODO]
        :param psf_filter: either None, "hann" or "tukey": window type to filter the PSF update
        """
        super(ER, self).__init__()
        self.positivity = positivity
        self.calc_llk = calc_llk
        self.nb_cycle = nb_cycle
        self.show_cdi = show_cdi
        self.fig_num = fig_num
        self.zero_mask = zero_mask
        self.update_psf = update_psf
        self.psf_filter = psf_filter

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new ER operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return ER(positivity=self.positivity, calc_llk=self.calc_llk, nb_cycle=self.nb_cycle * n,
                  show_cdi=self.show_cdi, fig_num=self.fig_num, zero_mask=self.zero_mask, update_psf=self.update_psf,
                  psf_filter=self.psf_filter)

    def op(self, cdi: CDI):
        t0 = timeit.default_timer()
        ic_dt = 0
        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if cdi.cycle % self.calc_llk == 0:
                    calc_llk = True

            update_psf = False
            if self.update_psf:
                update_psf = (((cdi.cycle - 1) % self.update_psf) == 0) and \
                             ((self.nb_cycle < 5) or ((self.nb_cycle - ic) > 5))

            cdi = ERProj(positivity=self.positivity) * \
                  FourierApplyAmplitude(calc_llk=calc_llk, zero_mask=self.zero_mask, update_psf=update_psf,
                                        psf_filter=self.psf_filter) * cdi

            if calc_llk:
                # Average time/cycle over the last N cycles
                dt = (timeit.default_timer() - t0) / (ic - ic_dt + 1)
                ic_dt = ic + 1
                t0 = timeit.default_timer()
                llk = cdi.get_llk()
                algo = 'ER'
                print("%4s #%3d LLK= %8.2f[%8.2f](p) %8.2f[%8.2f](g) %8.2f[%8.2f](e), nb photons=%e, "
                      "support:nb=%6d (%6.3f%%) average=%10.2f max=%10.2f, dt/cycle=%5.3fs" % (
                          algo, cdi.cycle, llk[0], llk[3], llk[1], llk[4], llk[2], llk[5], cdi.nb_photons_calc,
                          cdi.nb_point_support, cdi.nb_point_support / cdi._obj.size * 100,
                          np.sqrt(cdi.nb_photons_calc / cdi.nb_point_support), cdi._obj_max, dt))

            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi

            cdi.cycle += 1
        return cdi


class CFProj(CPUOperatorCDI):
    """
    Charge Flipping.
    """

    def __init__(self, positivity=False):
        super(CFProj, self).__init__()
        self.positivity = positivity

    def op(self, cdi):
        if self.positivity:
            cdi._obj.imag *= (2 * cdi._support * (cdi._obj.real > 0) - 1)
        else:
            cdi._obj.imag *= (2 * cdi._support - 1)
        return cdi


class CF(CPUOperatorCDI):
    """
    Charge flipping cycle
    """

    def __init__(self, positivity=False, calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1, zero_mask=False,
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
        :param update_psf: if >0, will update the partial coherence psf every update_psf cycles. [TODO]
        :param psf_filter: either None, "hann" or "tukey": window type to filter the PSF update
        """
        super(CF, self).__init__()
        self.positivity = positivity
        self.calc_llk = calc_llk
        self.nb_cycle = nb_cycle
        self.show_cdi = show_cdi
        self.fig_num = fig_num
        self.zero_mask = zero_mask
        self.update_psf = update_psf
        self.psf_filter = psf_filter

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new CF operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return CF(positivity=self.positivity, calc_llk=self.calc_llk, nb_cycle=self.nb_cycle * n,
                  show_cdi=self.show_cdi, fig_num=self.fig_num, zero_mask=self.zero_mask, update_psf=self.update_psf,
                  psf_filter=self.psf_filter)

    def op(self, cdi: CDI):
        t0 = timeit.default_timer()
        ic_dt = 0
        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if cdi.cycle % self.calc_llk == 0:
                    calc_llk = True

            update_psf = False
            if self.update_psf:
                update_psf = (((cdi.cycle - 1) % self.update_psf) == 0) and \
                             ((self.nb_cycle < 5) or ((self.nb_cycle - ic) > 5))

            cdi = CFProj(positivity=self.positivity) * \
                  FourierApplyAmplitude(calc_llk=calc_llk, zero_mask=self.zero_mask,
                                        update_psf=update_psf, psf_filter=self.psf_filter) * cdi

            if calc_llk:
                # Average time/cycle over the last N cycles
                dt = (timeit.default_timer() - t0) / (ic - ic_dt + 1)
                ic_dt = ic + 1
                t0 = timeit.default_timer()
                llk = cdi.get_llk()
                algo = 'CF'
                print("%4s #%3d LLK= %8.2f[%8.2f](p) %8.2f[%8.2f](g) %8.2f[%8.2f](e), nb photons=%e, "
                      "support:nb=%6d (%6.3f%%) average=%10.2f max=%10.2f, dt/cycle=%5.3fs" % (
                          algo, cdi.cycle, llk[0], llk[3], llk[1], llk[4], llk[2], llk[5], cdi.nb_photons_calc,
                          cdi.nb_point_support, cdi.nb_point_support / cdi._obj.size * 100,
                          np.sqrt(cdi.nb_photons_calc / cdi.nb_point_support), cdi._obj_max, dt))

            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi

            cdi.cycle += 1
        return cdi


class HIOProj(CPUOperatorCDI):
    """
    Hybrid Input-Output.
    """

    def __init__(self, beta=0.9, positivity=False):
        super(HIOProj, self).__init__()
        self.beta = np.float32(beta)
        self.positivity = positivity

    def op(self, cdi):
        if self.positivity:
            tmp = cdi._support * (cdi._obj.real > 0)
        else:
            tmp = cdi._support
        cdi._obj = (tmp == 0) * (cdi._obj_previous - self.beta * cdi._obj) + tmp * cdi._obj
        return cdi


class HIO(CPUOperatorCDI):
    """
    Hybrid Input-Output reduction cycle
    """

    def __init__(self, beta=0.9, positivity=False, calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1,
                 zero_mask=False, update_psf=0, psf_filter=None):
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
        :param update_psf: if >0, will update the partial coherence psf every update_psf cycles. [TODO]
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
                   update_psf=self.update_psf, psf_filter=self.psf_filter)

    def op(self, cdi: CDI):
        t0 = timeit.default_timer()
        ic_dt = 0
        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if cdi.cycle % self.calc_llk == 0:
                    calc_llk = True

            update_psf = False
            if self.update_psf:
                update_psf = (((cdi.cycle - 1) % self.update_psf) == 0) and \
                             ((self.nb_cycle < 5) or ((self.nb_cycle - ic) > 5))

            cdi = HIOProj(self.beta, positivity=self.positivity) * \
                  FourierApplyAmplitude(calc_llk=calc_llk, zero_mask=self.zero_mask,
                                        update_psf=update_psf, psf_filter=self.psf_filter) * CopyToPrevious() * cdi

            if calc_llk:
                # Average time/cycle over the last N cycles
                dt = (timeit.default_timer() - t0) / (ic - ic_dt + 1)
                ic_dt = ic + 1
                t0 = timeit.default_timer()
                llk = cdi.get_llk()
                algo = 'HIO'
                print("%4s #%3d LLK= %8.2f[%8.2f](p) %8.2f[%8.2f](g) %8.2f[%8.2f](e), nb photons=%e, "
                      "support:nb=%6d (%6.3f%%) average=%10.2f max=%10.2f, dt/cycle=%5.3fs" % (
                          algo, cdi.cycle, llk[0], llk[3], llk[1], llk[4], llk[2], llk[5], cdi.nb_photons_calc,
                          cdi.nb_point_support, cdi.nb_point_support / cdi._obj.size * 100,
                          np.sqrt(cdi.nb_photons_calc / cdi.nb_point_support), cdi._obj_max, dt))

            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi

            cdi.cycle += 1
        return cdi


class RAARProj(CPUOperatorCDI):
    """
    RAAR.
    """

    def __init__(self, beta=0.9, positivity=False):
        super(RAARProj, self).__init__()
        self.beta = np.float32(beta)
        self.positivity = positivity

    def op(self, cdi):
        if self.positivity:
            tmp = cdi._support * (cdi._obj.real > 0)
        else:
            tmp = cdi._support
        cdi._obj = (tmp == 0) * ((1 - 2 * self.beta) * cdi._obj + self.beta * cdi._obj_previous) + tmp * cdi._obj
        return cdi


class RAAR(CPUOperatorCDI):
    """
    RAAR cycle
    """

    def __init__(self, beta=0.9, positivity=False, calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1,
                 zero_mask=False, update_psf=0, psf_filter=None):
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
        :param update_psf: if >0, will update the partial coherence psf every update_psf cycles. [TODO]
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
        self.update_psf = update_psf
        self.psf_filter = psf_filter

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new HIO operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return RAAR(beta=self.beta, positivity=self.positivity, calc_llk=self.calc_llk, nb_cycle=self.nb_cycle * n,
                    show_cdi=self.show_cdi, fig_num=self.fig_num, zero_mask=self.zero_mask, update_psf=self.update_psf,
                    psf_filter=self.psf_filter)

    def op(self, cdi: CDI):
        t0 = timeit.default_timer()
        ic_dt = 0
        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if cdi.cycle % self.calc_llk == 0:
                    calc_llk = True

            update_psf = False
            if self.update_psf:
                update_psf = (((cdi.cycle - 1) % self.update_psf) == 0) and \
                             ((self.nb_cycle < 5) or ((self.nb_cycle - ic) > 5))

            cdi = RAARProj(self.beta, positivity=self.positivity) * \
                  FourierApplyAmplitude(calc_llk=calc_llk, zero_mask=self.zero_mask,
                                        update_psf=update_psf, psf_filter=self.psf_filter) * CopyToPrevious() * cdi

            if calc_llk:
                # Average time/cycle over the last N cycles
                dt = (timeit.default_timer() - t0) / (ic - ic_dt + 1)
                ic_dt = ic + 1
                t0 = timeit.default_timer()
                llk = cdi.get_llk()
                algo = 'RAAR'
                print("%4s #%3d LLK= %8.2f[%8.2f](p) %8.2f[%8.2f](g) %8.2f[%8.2f](e), nb photons=%e, "
                      "support:nb=%6d (%6.3f%%) average=%10.2f max=%10.2f, dt/cycle=%5.3fs" % (
                          algo, cdi.cycle, llk[0], llk[3], llk[1], llk[4], llk[2], llk[5], cdi.nb_photons_calc,
                          cdi.nb_point_support, cdi.nb_point_support / cdi._obj.size * 100,
                          np.sqrt(cdi.nb_photons_calc / cdi.nb_point_support), cdi._obj_max, dt))

            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi

            cdi.cycle += 1

        return cdi


class GPS(CPUOperatorCDI):
    """
    GPS cycle, according to Pham et al [2019]
    """

    def __init__(self, inertia=0.05, t=1.0, s=0.9, sigma_f=0, sigma_o=0, positivity=False,
                 calc_llk=False, nb_cycle=1, show_cdi=False, fig_num=-1, zero_mask=False, update_psf=0,
                 psf_filter=None):
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
        :param update_psf: if >0, will update the partial coherence psf every update_psf cycles. [TODO]
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
        self.update_psf = update_psf
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
                   update_psf=self.update_psf, psf_filter=self.psf_filter)

    def op(self, cdi: CDI):
        t0 = timeit.default_timer()
        ic_dt = 0

        epsilon = np.float32(self.inertia / (self.inertia + self.t))

        ny, nx = np.int32(cdi._obj.shape[-2]), np.int32(cdi._obj.shape[-1])
        if cdi._obj.ndim == 3:
            nz = np.int32(cdi._obj.shape[0])
        else:
            nz = np.int32(1)

        # We start in Fourier space (obj = z_0)
        cdi = FT(scale=True) * cdi

        # z_0 = FT(obj)
        z = cdi._obj.copy()

        # Start with obj = y_0 = 0
        cdi._obj.fill(np.complex64(0))

        # Gaussian smoothing arrays in object and Fourier space
        if cdi.iobs.ndim == 2:
            ny, nx = cdi.iobs.shape
            qx = fftfreq(nx).astype(np.float32)
            qy = fftfreq(ny).astype(np.float32)
            qy, qx = np.meshgrid(qy, qx, indexing='ij')
            qz = 0
        else:
            nz, ny, nx = cdi.iobs.shape
            qx = fftfreq(nx).astype(np.float32)
            qy = fftfreq(ny).astype(np.float32)
            qz = fftfreq(nz).astype(np.float32)
            qz, qy, qx = np.meshgrid(qz, qy, qx, indexing='ij')

        g_o = np.exp(-2 * np.pi ** 2 * self.sigma_o ** 2 * (qx ** 2 + qy ** 2 + qz ** 2))
        g_f = np.exp(-2 * np.pi ** 2 * self.sigma_f ** 2 * (qx ** 2 + qy ** 2 + qz ** 2))

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
            y = cdi._obj.copy()

            cdi = FT(scale=True) * cdi

            # ^z = z_k - t F(y_k)
            cdi._obj = z - self.t * g_o * cdi._obj

            cdi = ApplyAmplitude(calc_llk=calc_llk, zero_mask=self.zero_mask,
                                 update_psf=update_psf, psf_filter=self.psf_filter) * cdi

            # obj = z_k+1 = (1 - epsilon) * sqrt(iobs) * exp(i * arg(^z)) + epsilon * z_k
            cdi._obj = (1 - epsilon) * cdi._obj + epsilon * z

            if calc_llk:
                # Average time/cycle over the last N cycles
                dt = (timeit.default_timer() - t0) / (ic - ic_dt + 1)
                ic_dt = ic + 1
                t0 = timeit.default_timer()
                llk = cdi.get_llk()
                cdi.update_history(mode='llk', dt=dt, algorithm='GPS', verbose=True)
            else:
                cdi.update_history(mode='algorithm', algorithm='GPS')
            if self.show_cdi:
                if cdi.cycle % self.show_cdi == 0:
                    cdi = IFT(scale=True) * cdi
                    cdi = ShowCDI(fig_num=self.fig_num) * cdi
                    cdi = FT(scale=True) * cdi
            cdi.cycle += 1

            if ic < self.nb_cycle - 1:
                # obj = 2 * z_k+1 - z_k  & store z_k+1 in z
                cdi._obj, z = 2 * cdi._obj - z, cdi._obj

                cdi = IFT(scale=True) * cdi

                # obj = ^y = proj_support[y_k + s * obj] * G_sigma_f
                cdi._obj = g_f * (y + self.s * cdi._obj)
                if self.positivity:
                    cdi._obj *= g_f * np.logical_or(cdi._support == 0, cdi._obj.real < 0)
                else:
                    cdi._obj *= g_f * (cdi._support == 0)

        # Back to object space
        cdi = IFT(scale=True) * cdi

        return cdi


class ML(CPUOperatorCDI):
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
        obj_grad = np.empty_like(cdi._obj)
        obj_grad_last = np.empty_like(cdi._obj)
        obj_dir = np.zeros_like(cdi._obj)

        # Use support for regularization
        N = cdi._obj.size
        # Total number of photons
        Nph = cdi.iobs_sum
        cdi.llk_support_reg_fac = np.float32(self.reg_fac / (8 * N / Nph))

        # if self.reg_fac_support>0:
        #    print("Regularization factor for support:", self.reg_fac_support)

        if cdi.in_object_space() is False:
            cdi = IFT() * cdi
        for cycle in range(self.nb_cycle):
            obj_grad, obj_grad_last = obj_grad_last, obj_grad
            psi = fftn(cdi._obj) / np.sqrt(cdi._obj.size)
            icalc = abs(psi) ** 2
            iobs = cdi.iobs.copy()
            iobs[cdi.iobs < 0] = icalc[cdi.iobs < 0]

            if self.calc_llk and cycle == (self.nb_cycle - 1):
                cdi = LLK() * cdi

            # This calculates the iFT of the conjugate of [(1 - iobs/icalc) * psi]
            obj_grad = ifftn(psi.conj() * (1 - iobs / icalc)) * np.sqrt(psi.size)

            if cdi.llk_support_reg_fac > 0:
                obj_grad += cdi.llk_support_reg_fac * (1 - cdi._support) * cdi._obj

            if cycle == 0:
                beta = 0
                obj_dir = obj_grad.copy()
            else:
                # Polak-Ribière CG coefficient
                beta_n = (obj_grad.real * (obj_grad.real - obj_grad_last.real)).sum()
                beta_n += (obj_grad.imag * (obj_grad.imag - obj_grad_last.imag)).sum()
                beta_d = (abs(obj_grad_last) ** 2).sum()
                # Reset direction if beta<0 => beta=0
                beta = np.float32(max(0, beta_n / max(1e-20, beta_d)))

            obj_dir = beta * obj_dir - obj_grad
            dpsi = fftn(obj_dir) / np.sqrt(obj_dir.size)

            gamma_n = ((psi.conj() * dpsi).real * (iobs / icalc - 1)).sum()
            gamma_d = (abs(dpsi) ** 2 - iobs * (
                    abs(dpsi) ** 2 / abs(psi) ** 2 - 2 * (psi.conj() * dpsi).real ** 2 / abs(psi) ** 4)).sum()
            if cdi.llk_support_reg_fac > 0:
                gamma_n -= cdi.llk_support_reg_fac * ((1 - cdi._support) * (psi.conj() * dpsi).real).sum()
                gamma_d += cdi.llk_support_reg_fac * ((1 - cdi._support) * abs(dpsi) ** 2).sum()

            gamma = gamma_n / gamma_d

            cdi._obj += gamma * obj_dir

        return cdi


class SupportUpdate(OperatorCDI):
    """
    Update the support
    """

    def __init__(self, threshold_relative=0.2, smooth_width=3, force_shrink=False, method='rms',
                 post_expand=None, verbose=False, update_border_n=0, min_fraction=0, max_fraction=1, lazy=False):
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
                             outer border of the support. [TODO for CPU]
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

    def op(self, cdi):
        if np.isscalar(self.smooth_width):
            smooth_width = self.smooth_width
        else:
            a, b, nb = self.smooth_width
            i = cdi.cycle
            if i < nb:
                smooth_width = a * np.exp(-i / nb * np.log(a / b))
            else:
                smooth_width = b
        tmpobj = fftshift(np.abs(cdi.get_obj()).astype(np.float32))
        tmpobj = fftshift(gaussian_filter(tmpobj, smooth_width))
        support = cdi.get_support()

        # Get total number of photons and maximum intensity in the support
        tmp = (np.abs(support * cdi.get_obj()) ** 2)
        max_icalc_support = tmp.max()
        nb_ph_support = tmp.sum()
        cdi._obj_max = np.sqrt(max_icalc_support)

        # Get average amplitude and maximum intensity in the support, from the convolved amplitude
        tmp = (np.abs(support * tmpobj))
        max_abs_support = tmp.max()
        av_abs_support = tmp.sum() / cdi.nb_point_support
        rms_support = np.sqrt((tmp ** 2).sum() / cdi.nb_point_support)

        # Threshold (from average)
        if self.method == 'max':
            thr = self.threshold_relative * np.float32(max_abs_support)
        elif self.method == 'rms':
            thr = self.threshold_relative * np.float32(rms_support)
        else:
            thr = self.threshold_relative * np.float32(av_abs_support)
        # Update support and compute the new number of points in the support
        if self.force_shrink:
            support *= tmpobj > thr
        else:
            support = (tmpobj > thr).astype(np.int8)

        if self.post_expand is not None:
            for n in self.post_expand:
                cdi = SupportExpand(n=n, update_nb_points_support=False) * cdi

        nb = support.sum()
        if self.verbose:
            print("Nb points in support: %d (%6.3f%%), threshold=%8f  (%6.3f), nb photons=%10e"
                  % (nb, nb / cdi._obj.size * 100, thr, self.threshold_relative, nb_ph_support))
        cdi.nb_point_support = nb
        cdi.set_support(support)
        if cdi.nb_point_support <= self.min_fraction * cdi.iobs.size:
            raise SupportTooSmall("Too few points in support: %d (%6.3f%%)" % (nb, nb / cdi._obj.size * 100))
        elif cdi.nb_point_support >= self.max_fraction * cdi.iobs.size:
            raise SupportTooLarge("Too many points in support: %d (%6.3f%%)" % (nb, nb / cdi._obj.size * 100))
        return cdi


class ObjSupportStats(CPUOperatorCDI):
    """
    Gather basic stats about the object: maximum and average amplitude inside the support,
    and percentage of square modulus outside the support.
    This should be evaluated ideally immediately after FourierApplyAmplitude. The result is stored
    in the CDI object's history.
    """

    def op(self, cdi):
        o2 = abs(cdi._obj) ** 2
        o2in = (o2 * (cdi._support > 0)).sum()
        o2out = (o2 * (cdi._support == 0)).sum()
        cdi._obj_max = np.sqrt(o2.max())
        cdi._obj2_out = o2out / (o2in + o2out)
        cdi.update_history(mode='support')
        return cdi


class ScaleObj(OperatorCDI):
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
        if self.method.lower() == 'f':
            # Scale the object to match Fourier amplitudes
            tmpcalc = np.abs(fftn(cdi._obj)) * (cdi.iobs >= 0)
            tmpobs = np.sqrt(np.abs(cdi.iobs))
            scale = (tmpcalc * tmpobs).sum() / (tmpcalc ** 2).sum()
        elif self.method.lower() == 'i':
            # Scale object to match Fourier intensities
            tmpcalc = np.abs(fftn(cdi._obj)) ** 2 * (cdi.iobs >= 0)
            scale = np.sqrt((tmpcalc * cdi.iobs).sum() / (tmpcalc ** 2).sum())
        elif self.method.lower() == 'p':
            # Scale object to match Poisson statistics
            tmpcalc = np.abs(fftn(cdi._obj)) ** 2 * (cdi.iobs >= 0)
            scale = np.sqrt((tmpcalc * cdi.iobs).sum() / (tmpcalc ** 2).sum())
        else:
            # Scale object to match weighted intensities
            # Weight: 1 for null intensities, zero for masked pixels
            w = (1 / (np.abs(cdi.iobs) + 1e-6) * (cdi.iobs > 1e-6) + (cdi.iobs <= 1e-6)) * (cdi.iobs >= 0)
            tmpcalc = np.abs(fftn(cdi._obj)) ** 2
            scale = np.sqrt((w * tmpcalc * cdi.iobs).sum() / (w * tmpcalc ** 2).sum())
        cdi._obj *= scale
        if self.verbose:
            print("Scaled object by: %f" % (scale))
        return cdi


class LLK(CPUOperatorCDI):
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

    def op(self, cdi):
        need_ft = cdi.in_object_space()

        if need_ft:
            cdi = FT() * cdi

        # TODO: add support for convolution with partial coherence kernel
        calc = np.abs(self.scale * cdi._obj) ** 2
        cdi.nb_photons_calc = calc.sum()
        calc[cdi.iobs < 0] = cdi.iobs[cdi.iobs < 0]
        cdi.llk_poisson = llk_poisson(cdi.iobs, calc).sum()
        cdi.llk_gaussian = llk_gaussian(cdi.iobs, calc).sum()
        cdi.llk_euclidian = llk_euclidian(cdi.iobs, calc).sum()

        if need_ft:
            cdi = IFT() * cdi

        return cdi


class LLKSupport(CPUOperatorCDI):
    """
    Support log-likelihood reduction kernel. Can only be used when cdi instance is object space.
    This is a reduction operator - it will write llk_support as an argument in the cdi object, and return cdi.
    """

    def op(self, cdi):
        llk = (abs(cdi._obj) ** 2 * (cdi._support == 0)).sum()
        cdi.llk_support = llk * cdi.llk_support_reg_fac
        return cdi


class DetwinSupport(CPUOperatorCDI):
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
            cdi._support = cdi._support_tmp
            del cdi._support_tmp
        else:
            # Get current support
            cdi._support_tmp = cdi._support.copy()
            tmp = fftshift(cdi._support)
            # Use center of mass to cut near middle
            c = center_of_mass(tmp)
            if self.axis == 0:
                tmp[int(round(c[0])):] = 0
            elif self.axis == 1 or tmp.ndim == 2:
                tmp[:, int(round(c[1])):] = 0
            else:
                tmp[:, :, int(round(c[2])):] = 0
            cdi._support = fftshift(tmp)
        return cdi


class DetwinHIO(CPUOperatorCDI):
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


class DetwinRAAR(CPUOperatorCDI):
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


class SupportExpand(CPUOperatorCDI):
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
        if self.n < 0:
            tmp = 1 - cdi._support
        else:
            tmp = cdi._support

        for axis in range(cdi._obj.ndim):
            for i in range(1, abs(self.n) + 1):
                tmp += np.roll(tmp, 1, axis=axis) + np.roll(tmp, -1, axis=axis)

        if self.n < 0:
            cdi._support = tmp == 0
        else:
            cdi._support = tmp > 0

        if self.update_nb_points_support:
            cdi.nb_point_support = cdi._support.sum()
        return cdi


class ObjConvolve(CPUOperatorCDI):
    """
    Gaussian convolution of the object, produces a new array with the convoluted amplitude of the object.
    """

    def __init__(self, sigma=1):
        super(ObjConvolve, self).__init__()
        self.sigma = np.float32(sigma)

    def op(self, cdi):
        cdi._obj_abs = gaussian_filter(abs(cdi._obj), self.sigma)
        return cdi


class EstimatePSF(CPUOperatorCDI):
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


class InitPSF(CPUOperatorCDI):
    """ Initialise the point-spread function kernel to model
    partial coherence. [TODO: not yet implemented for CPU]
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
        nx = np.int32(cdi.iobs.shape[-1])
        ny = np.int32(cdi.iobs.shape[-2])
        if cdi.iobs.ndim == 3:
            nz = np.int32(cdi.iobs.shape[0])
        else:
            nz = np.int32(1)

        z, y, x = np.meshgrid(fftfreq(nz) * nz, fftfreq(ny) * ny, fftfreq(nx) * nx, indexing='ij')
        if "gauss" in self.model.lower():
            sigma = self.fwhm / 2.3548
            psf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
        elif "lorentz" in self.model.lower():
            psf = 2 / np.pi * self.fwhm / (x ** 2 + y ** 2 + z ** 2 + self.fwhm ** 2)
        else:
            sigma = self.fwhm / 2.3548
            g = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
            l = 2 / np.pi * self.fwhm / (x ** 2 + y ** 2 + z ** 2 + self.fwhm ** 2)
            psf = l * self.eta + g * (1 - self.eta)

        cdi._psf_f = rfftn(psf, norm="ortho")

        if self.psf is not None:
            cdi = UpdatePSF(filter=self.filter) ** 10 * cdi

        return cdi
