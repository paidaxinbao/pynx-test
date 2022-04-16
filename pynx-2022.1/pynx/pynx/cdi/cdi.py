# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['CDI', 'OperatorCDI', 'save_cdi_data_cxi', 'calc_throughput', 'CDIOperatorException',
           'SupportTooSmall', 'SupportTooLarge']

import time
import timeit
import sys
import warnings
import gc
from copy import deepcopy
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift, fftfreq
from numpy.fft import rfftn, irfftn
from scipy.ndimage import zoom
from ..operator import Operator, OperatorException, has_attr_not_none
from ..version import get_git_version
from ..utils.array import pad, crop, rebin
from ..utils import phase

_pynx_version = get_git_version()
from ..utils.history import History

from ..utils import h5py


def _as_dim(v, ndim=3):
    """Convert an upsample/bin/crop factor to N values

    :param v: the value (integer, None, or list/tuple/array)
    :param: the number of dimensions to return (default 3)
    :return: either None (if all values are 1 or None), or an array
        of ndim values with the factors.
    """
    if v is None:
        return None
    v3 = np.ones(ndim, dtype=np.int32)
    if isinstance(v, int) or isinstance(v, np.integer):
        for i in range(ndim):
            v3[i] = v
    else:
        for i in range(min(len(v), ndim)):
            v3[i] = v[i]
    if np.allclose(v3, 1):
        return None
    return v3


def _eq_bin_crop_upsample(v1, v2):
    """ Compare bin/crop and upsample factors. Returns True
    if factors are equivalent (both None or same values),
    False otherwise.
    """
    if v1 is None:
        if v2 is None:
            return True
        else:
            return False
    if v2 is None:
        return False
    return np.allclose(v1, v2)


class CDI:
    """
    Reconstruction class for two or three-dimensional coherent diffraction imaging data.
    """

    def __init__(self, iobs, support=None, obj=None, mask=None,
                 pixel_size_detector=None, wavelength=None, detector_distance=None,
                 crop=None, bin=None):
        """
        Constructor. All arrays are assumed to be centered in (0,0) to avoid fftshift

        :param iobs: 2D/3D observed diffraction data (intensity).
          Assumed to be corrected and following Poisson statistics, will be converted to float32.
          Dimensions should be divisible by 4 and have a prime factor decomposition up to 7.
          Internally, the following special values are used:

          * values<=-1e19 are masked. Among those, values in ]-1e38;-1e19] are estimated values,
            stored as -(iobs_est+1)*1e19, which can be used to make a loose amplitude projection.
            Values <=-1e38 are masked (no amplitude projection applied), just below the minimum
            float32 value

          * -1e19 < values <= 1 are observed but used as free pixels
            If the mask is not supplied, then it is assumed that the above special values are
            used.

        :param support: initial support in real space (1 = inside support, 0 = outside)
        :param obj: initial object. If None, it should be initialised later.
        :param mask: mask for the diffraction data (0: valid pixel, >0: masked)
        :param pixel_size_detector: detector pixel size (meters)
        :param wavelength: experiment wavelength (meters)
        :param detector_distance: detector distance (meters)
        """
        self.iobs = iobs.astype(np.float32)
        # Iobs sum, taking into account masked and free pixels (stored as -iobs-1)
        self.iobs_sum = (self.iobs * (self.iobs >= 0)).sum()
        self.iobs_sum -= (self.iobs[np.logical_and(self.iobs > -1e19, self.iobs < 0)] + 1).sum()

        if support is not None:
            self._support = support.astype(np.int8)
        else:
            # Support will be later updated
            self._support = np.ones_like(iobs, dtype=np.int8)

        if obj is None:
            # It should be initialised later
            self._obj = np.ones(iobs.shape, dtype=np.complex64)
        else:
            self._obj = obj.astype(np.complex64)

        self._is_in_object_space = True

        if mask is not None:
            # Avoid needing to store/transfer a mask in GPU. Flag masked pixels with iobs < -1e38
            self.iobs[mask > 0] = -1.05e38

        # Fourier transform of the point spread function. This is used for partial coherence correction.
        self._psf_f = None

        self.pixel_size_detector = pixel_size_detector
        self.wavelength = wavelength
        self.detector_distance = detector_distance
        self.pixel_size_object = None
        self.lambdaz = None

        # Experimental parameters
        if self.wavelength is not None and self.detector_distance is not None:
            self.lambdaz = self.wavelength * self.detector_distance
        if self.lambdaz is not None and self.pixel_size_detector is not None and iobs is not None:
            # TODO: correctly compute pixel size, at least for x and y, when number of pixels differ along directions
            self.pixel_size_object = self.lambdaz / (self.pixel_size_detector * self.iobs.shape[-1])
        # TODO: correctly compute orthonormalisation matrix in direct space if all parameters are known

        # Variables for log-likelihood statistics
        self.llk_poisson = 0
        self.llk_gaussian = 0
        self.llk_euclidian = 0
        self.llk_poisson_free = 0
        self.llk_gaussian_free = 0
        self.llk_euclidian_free = 0
        self.nb_photons_calc = 0

        self.nb_observed_points = None

        # Normally free pixels will be initialised later, but if this is a copy they may already be flagged
        self.nb_free_points = None

        self.llk_support = None  # Current value of the support log-likelihood (support regularization)
        self.llk_support_reg_fac = None  # Regularization factor for the support log-likelihood
        self.nb_point_support = None
        self._update_nb_points()

        # Max amplitude reported during support update (after smoothing)
        self._obj_max = 0
        # Fraction of integrated square modulus outside support
        self._obj2_out = 0

        # The timestamp counter record when the cdi or support data was last altered, either in the host or the
        # GPU memory.
        self._timestamp_counter = 1
        self._cl_timestamp_counter = 0
        self._cu_timestamp_counter = 0

        # Record the number of cycles (RAAR, HIO, ER, CF, etc...), which can be used to make some parameters
        # evolve, e.g. for support update
        self.cycle = 0

        # History record
        self.history = History()

        # Upsample parameter. See set_upsample() for a description
        self._upsample = None

        # Crop parameter - See set_crop() for a description
        self._crop = None

        # Bin parameter - See set_bin() for a description
        self._bin = None

        # Store original Iobs when using self._bin or self._crop
        self._iobs_orig = None

        # Operators queued for delayed execution. In practice this is used to trigger
        # the object initialisation using auto-correlation, but it could be used for
        # other tasks. This will get executed before the operations of any Operator
        self._lazy_ops = []

    def _update_nb_points(self):
        """
        Update the number of observed, free and support pi/voxels.
        :return:
        """
        self.nb_observed_points = (self.iobs > -1e19).sum()
        self.nb_free_points = ((self.iobs > -1e19) * (self.iobs < 0)).sum()
        self.nb_point_support = self._support.sum()

    def get_x_y_z(self):
        """
        Get 1D arrays of x and y (z if 3d) coordinates, taking into account the pixel size. The arrays are centered
        at (0,0) - i.e. with the origin in the corner for FFT puroposes. x is an horizontal vector and y vertical, 
        and (if 3d) z along third dimension.

        :return: a tuple (x, y) or (x, y, z) of 1D numpy arrays
        """
        if self.iobs.ndim == 2:
            ny, nx = self.iobs.shape
            x, y = np.arange(-nx // 2, nx // 2, dtype=np.float32), \
                   np.arange(-ny // 2, ny // 2, dtype=np.float32)[:, np.newaxis]
            return fftshift(x) * self.pixel_size_object, fftshift(y) * self.pixel_size_object
        else:
            nz, ny, nx = self.iobs.shape
            x, y, z = np.arange(-nx // 2, nx // 2, dtype=np.float32), \
                      np.arange(-ny // 2, ny // 2, dtype=np.float32)[:, np.newaxis], \
                      np.arange(-nz // 2, nz // 2, dtype=np.float32)[:, np.newaxis, np.newaxis]
            return fftshift(x) * self.pixel_size_object, fftshift(y) * self.pixel_size_object, \
                   fftshift(z) * self.pixel_size_object

    def get_x_y(self):
        return self.get_x_y_z()

    def copy(self, copy_history_llk=False, copy_free_pixels=False):
        """
        Creates a copy (without any reference passing) of this object, unless copy_obj is False.

        :param copy_history_llk: if True, the history, cycle number, llk, etc.. are copied to the new object.
        :param copy_free_pixels: if True, copy the distribution of free pixels to the new CDI object. Otherwise the
                                 new CDI object does not have free pixels initialised.
        :return: a copy of the object.
        """
        iobs = self.iobs.copy()
        mask = self.iobs <= -1e19

        if copy_free_pixels is False:
            tmp = np.logical_and(self.iobs > -1e19, self.iobs < 0)
            iobs[tmp] = -(self.iobs[tmp] + 1)

        cdi = CDI(iobs, support=self._support, obj=self._obj, mask=mask,
                  pixel_size_detector=self.pixel_size_detector, wavelength=self.wavelength,
                  detector_distance=self.detector_distance)

        if copy_free_pixels:
            cdi.nb_free_points = self.nb_free_points

        if copy_history_llk:
            cdi.history = deepcopy(self.history)
            cdi.llk_poisson = self.llk_poisson
            cdi.llk_gaussian = self.llk_gaussian
            cdi.llk_euclidian = self.llk_euclidian
            cdi.llk_poisson_free = self.llk_poisson_free
            cdi.llk_gaussian_free = self.llk_gaussian_free
            cdi.llk_euclidian_free = self.llk_euclidian_free
            cdi.nb_photons_calc = self.nb_photons_calc
            cdi.cycle = self.cycle
        return cdi

    def in_object_space(self):
        """

        :return: True if the current obj array is in object space, False otherwise.
        """
        return self._is_in_object_space

    def _from_gpu(self):
        """
        Internal function to get relevant arrays from GPU memory (OpenCL or CUDA).
        This always gets the object with the size, resolution and extent corresponding to the
        original Iobs array, regardless of upsample options.
        :return: Nothing
        """
        if self._timestamp_counter < self._cl_timestamp_counter:
            self._obj = self._cl_obj.get()
            self._support = self._cl_support.get()
            if has_attr_not_none(self, '_cl_psf_f'):
                self._psf_f = self._cl_psf_f.get()
            self._timestamp_counter = self._cl_timestamp_counter
            updated = True
        if self._timestamp_counter < self._cu_timestamp_counter:
            self._obj = self._cu_obj.get()
            self._support = self._cu_support.get()
            if has_attr_not_none(self, '_cu_psf_f'):
                self._psf_f = self._cu_psf_f.get()
            self._timestamp_counter = self._cu_timestamp_counter

    def _to_gpu(self):
        """
        This function is used to prepare the object and Iobs array for transfer to the
        GPU.

        Note that this does not move any array to the GPU, but prepares them for it.

        :return: a tuple (obj, support, iobs) of the arrays prepared for GPU injection. These are
            fft-shifted, with the center of the object/diffraction in the first array element.
        """
        obj = self.get_obj()
        support = self.get_support()
        iobs = self.get_iobs()
        return obj, support.astype(np.int8), iobs

    def set_upsample(self, upsample_f):
        """
        Change the upsample parameters, either as a single integer
        (same value for all dimensions) or as a tuple/list of values.
        The observed intensity array size is not affected by this parameter,
        but the extent of the object will be multiplied by this factor. The
        upsampled calculated intensities will be binned for comparison with
        the observed intensity array.
        This can be used when the oversampling ratio is not large enough
        for a reconstruction [Maddali et al, Phys. Rev. A. 99 (2019), 053838]

        This does not alter the observed intensity array, or the object array
        which is accessible by get_obj(), which will keep the dimensions
        corresponding to the original Iobs array.

        This automatically disables bin and crop parameters.

        :param upsample_f: if upsample_f is not None or 1, the object array size is
            internally extended by this factor, keeping the object pixel size
            constant.
            A list/tuple of integers can also be supplied to use a different
            upsampling for each dimension.
            This does not affect the resolution of the reconstructed object,
            but simulates an increase of the oversampling ratio by the
            upsampling factor.
        :return: nothing, the upsampling parameter is stored and
            will be used for computations.
        """
        if isinstance(upsample_f, tuple) or isinstance(upsample_f, list):
            assert len(upsample_f) == self.iobs.ndim
        upsample_f = _as_dim(upsample_f, ndim=self.iobs.ndim)
        if not _eq_bin_crop_upsample(self._upsample, upsample_f):
            self._from_gpu()
            self.set_bin(None)
            self.set_crop(None)
            if self._upsample is not None:
                # Undo the previous upsample
                self._obj = crop(self._obj, margin_f=self._upsample, shift=True)
                self._support = crop(self._support, margin_f=self._upsample, shift=True)
            if upsample_f is not None:
                # Pad object and support for upsampling
                self._obj = pad(self._obj, padding_f=upsample_f, shift=True)
                self._support = pad(self._support, padding_f=upsample_f, shift=True)
            self._timestamp_counter += 1
            self._upsample = upsample_f

    def set_bin(self, bin_f, obj_support=True, update_nb_points=True):
        """
        Change the binning parameters, either as a single integer
        (same value for all dimensions) or as a tuple/list of values.
        The observed intensity array size is then binned (and the original
        array is kept), which results in an object with the same resolution,
        but a smaller real-space extension, i.e. a lower oversampling
        (so support update should be done more carefully).
        When binning the Iobs array, the pixels tagged for free LLK
        calculation are handled.
        The only values used should be 1 or 2.

        This automatically disables crop and upsample parameters.

        :param bin_f: if bin_f is not None or 1, the iobs array size is
            binned by this factor, keeping the object pixel size
            constant but with a smaller extent.
            A list/tuple of integers can also be supplied to use a different
            binning for each dimension.
        :param obj_support: if True, the object and support arrays will be
            modified according to the bin change in the iobs array. Using
            obj_support=False allows to perform separately the object and
            support scaling using a GPU operator.
        :return: nothing, the binning parameter is stored, the iobs array
            is binned, the object array is resized.
        """
        if isinstance(bin_f, tuple) or isinstance(bin_f, list):
            assert len(bin_f) == self.iobs.ndim
        bin_f = _as_dim(bin_f, ndim=self.iobs.ndim)
        if not _eq_bin_crop_upsample(self._bin, bin_f):
            self._from_gpu()
            self.set_upsample(None)
            self.set_crop(None, obj_support=obj_support, update_nb_points=False)
            if self._bin is not None:
                # Undo previous binning
                self.iobs = self._iobs_orig
                if obj_support:
                    self._obj = pad(self._obj, padding_f=self._bin, shift=True)
                    self._support = pad(self._support, padding_f=self._bin, shift=True)
            if bin_f is not None:
                # Bin Iobs
                self._iobs_orig = self.iobs
                self.iobs = rebin(self._iobs_orig, bin_f, mask_iobs_cdi=True)
                if obj_support:
                    self._obj = crop(self._obj, margin_f=bin_f, shift=True)
                    self._support = crop(self._support, margin_f=bin_f, shift=True)
            if update_nb_points:
                self._update_nb_points()
            self._timestamp_counter += 1
            self._bin = bin_f

    def set_crop(self, crop_f, obj_support=True, update_nb_points=True):
        """
        Change the cropping parameters, either as a single integer
        (same value for all dimensions) or as a tuple/list of values.
        The observed intensity array size is then cropped (and the original
        array is kept), which results in an object with a lower resolution,
        but the same extent.
        The only values used should be 1 or 2.

        This automatically disables bin and upsample parameters.

        :param crop_f: if crop_f is not None or 1, the iobs array is cropped by
            this factor, also dividing the object size by the same amount, with
            the object pixel size multiplied by the factor.
            A list/tuple of integers can also be supplied to use a different
            binning for each dimension.
        :param obj_support: if True, the object and support arrays will be
            modified according to the bin change in the iobs array. Using
            obj_support=False allows to perform separately the object and
            support scaling using a GPU operator.
        :return: nothing, the crop parameter is stored and
            will be used for computations.
        """
        if isinstance(crop_f, tuple) or isinstance(crop_f, list):
            assert len(crop_f) == self.iobs.ndim
        crop_f = _as_dim(crop_f, ndim=self.iobs.ndim)
        if not _eq_bin_crop_upsample(self._crop, crop_f):
            self._from_gpu()
            self.set_upsample(None)
            self.set_bin(None, obj_support=obj_support, update_nb_points=False)
            if self._crop is not None:
                # Undo previous crop
                self.iobs = self._iobs_orig
                if obj_support:
                    self._obj = fftshift(zoom(fftshift(self._obj), self._crop, order=1)) / np.sqrt(np.prod(self._crop))
                    self._support = fftshift(zoom(fftshift(self._support), self._crop, order=1))
            if crop_f is not None:
                self._iobs_orig = self.iobs
                self.iobs = crop(self._iobs_orig, margin_f=crop_f, shift=True)
                if obj_support:
                    self._obj = rebin(self._obj, rebin_f=crop_f)
                    self._support = rebin(self._support, rebin_f=crop_f)
            if update_nb_points:
                self._update_nb_points()
            self._timestamp_counter += 1
            self._crop = crop_f

    def get_upsample(self, dim3=False):
        """
        Get the binning parameter as an int32 array, one value per dimension.

        :param dim3: this can b used to force the number of elements to 3, for
          convenience in GPU kernels.
        :return: the bin parameter
        """
        ndim = self.iobs.ndim
        if dim3:
            ndim = 3
        v3 = _as_dim(self._upsample, ndim=ndim)
        if v3 is None:
            return None
        return v3

    def get_bin(self, dim3=False):
        """
        Get the binning parameter as an int32 array, one value per dimension.

        :param dim3: this can b used to force the number of elements to 3, for
          convenience in GPU kernels.
        :return: the bin parameter
        """
        ndim = self.iobs.ndim
        if dim3:
            ndim = 3
        v3 = _as_dim(self._bin, ndim=ndim)
        if v3 is None:
            return None
        return v3

    def get_crop(self, dim3=False):
        """
        Get the cropping parameter as an int32 array, one value per dimension.

        :param dim3: this can b used to force the number of elements to 3, for
          convenience in GPU kernels.
        :return: the bin parameter
        """
        ndim = self.iobs.ndim
        if dim3:
            ndim = 3
        v3 = _as_dim(self._crop, ndim=ndim)
        if v3 is None:
            return None
        return v3

    def get_obj(self, shift=False):
        """
        Get the object data array. This will automatically get the latest data, either from GPU or from the host
        memory, depending where the last changes were made.

        :param shift: if True, the data array will be fft-shifted so that the center of the data is in the center
                      of the array, rather than in the corner (the default).
        :return: the 2D or 3D CDI numpy data array
        """
        self._from_gpu()
        if shift:
            return fftshift(self._obj)
        else:
            return self._obj

    def get_support(self, shift=False):
        """
        Get the support array. This will automatically get the latest data, either from GPU or from the host
        memory, depending where the last changes were made.

        :param shift: if True, the data array will be fft-shifted so that the center of the data is in the center
                      of the array, rather than in the corner (the default).
        :return: the 2D or 3D CDI numpy data array
        """
        self._from_gpu()

        if shift:
            return fftshift(self._support)
        else:
            return self._support

    def get_iobs(self, shift=False):
        """
        Get the observed intensity data array.

        :param shift: if True, the data array will be fft-shifted so that the center of the data is in the center
                      of the array, rather than in the corner (the default).
        :return: the 2D or 3D CDI numpy data array
        """
        if shift:
            return fftshift(self.iobs)
        else:
            return self.iobs

    def set_obj(self, obj, shift=False):
        """
        Set the object data array. Assumed to be in object (not Fourier) space

        :param obj: the 2D or 3D CDI numpy data array (complex64 numpy array)
        :param shift: if True, the data array will be fft-shifted so that the center of the stored data is
                      in the corner of the array. [default: the array is already shifted]
        :return: nothing
        """
        if shift:
            self._obj = fftshift(obj).astype(np.complex64)
        else:
            self._obj = obj.astype(np.complex64)
        if self._timestamp_counter <= self._cl_timestamp_counter:
            self._timestamp_counter = self._cl_timestamp_counter + 1
        if self._timestamp_counter <= self._cu_timestamp_counter:
            self._timestamp_counter = self._cu_timestamp_counter + 1
        self._is_in_object_space = True

    def set_support(self, support, shift=False):
        """
        Set the support data array.

        :param obj: the 2D or 3D CDI numpy data array (complex64 numpy array)
        :param shift: if True, the data array will be fft-shifted so that the center of the stored data is
                      in the corner of the array. [default: the array is already shifted]
        :return: nothing
        """
        if shift:
            self._support = fftshift(support).astype(np.int8)
        else:
            self._support = support.astype(np.int8)
        if self._timestamp_counter <= self._cl_timestamp_counter:
            self._timestamp_counter = self._cl_timestamp_counter + 1
        if self._timestamp_counter <= self._cu_timestamp_counter:
            self._timestamp_counter = self._cu_timestamp_counter + 1
        self.nb_point_support = self._support.sum()

    def set_iobs(self, iobs, shift=False):
        """
        Set the observed intensity data array.

        :param iobs: the 2D or 3D CDI numpy data array (float32 numpy array)
        :param shift: if True, the data array will be fft-shifted so that the center of the stored data is
                      in the corner of the array. [default: the array is already shifted]
        :return: nothing
        """
        self._from_gpu()
        if shift:
            self.iobs = fftshift(iobs).astype(np.float32)
        else:
            if iobs.dtype != np.float32:
                self.iobs = iobs.astype(np.float32)
            else:
                self.iobs = iobs
        self.iobs_sum = self.iobs[self.iobs > 0].sum()

        if self._timestamp_counter <= self._cl_timestamp_counter:
            self._timestamp_counter = self._cl_timestamp_counter + 1
        if self._timestamp_counter <= self._cu_timestamp_counter:
            self._timestamp_counter = self._cu_timestamp_counter + 1

    def set_mask(self, mask, shift=False):
        """
        Set the mask data array. Note that since the mask is stored by setting observed intensities of masked
        pixels to negative values, it is not possible to un-mask pixels.

        :param obj: the 2D or 3D CDI mask array
        :param shift: if True, the data array will be fft-shifted so that the center of the stored data is
                      in the corner of the array. [default: the array is already shifted]
        :return: nothing
        """
        if mask is None:
            return
        if shift:
            mask = fftshift(mask).astype(np.int8)
        iobs = self.get_iobs()
        iobs[mask > 0] = -1.05e38
        self.set_iobs(iobs)

    def init_psf(self, model="pseudo-voigt", fwhm=1, eta=0.1, psf_f=None):
        """
        Initialise the point-spread function to model the partial coherence,
        using either a Lorentzian, Gaussian or pseudo-Voigt function

        :param model: "lorentzian", "gaussian" or "pseudo-voigt", or None to deactivate
        :param fwhm: the full-width at half maximum, in pixels
        :param eta: the eta parameter for the pseudo-voigt
        :param psf_f: this can be used to supply an array for the PSF. This should be
          the half-Hermition which is the result of the real2complex FT of the
          detector-space PSF. It should be centered at the origin (fft-shifted).
          If this is given all other arguments are ignored.
        :return: nothing
        """
        self._from_gpu()

        if self._psf_f is not None or model is not None:
            if self._timestamp_counter <= self._cl_timestamp_counter:
                self._timestamp_counter = self._cl_timestamp_counter + 1
            if self._timestamp_counter <= self._cu_timestamp_counter:
                self._timestamp_counter = self._cu_timestamp_counter + 1

        if psf_f is not None:
            self._psf_f = psf_f.astype(np.complex64)
            return
        if model is None:
            self._psf_f = None
            return
        if self.iobs.ndim == 2:
            ny, nx = self.iobs.shape
            y, x = np.meshgrid(fftfreq(ny) * ny, fftfreq(nx) * nx, indexing='ij')
            if "gauss" in model.lower():
                sigma = fwhm / 2.3548
                psf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            elif "lorentz" in model.lower():
                psf = 2 / np.pi * fwhm / (x ** 2 + y ** 2 + fwhm ** 2)
            else:
                sigma = fwhm / 2.3548
                g = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                l = 2 / np.pi * fwhm / (x ** 2 + y ** 2 + fwhm ** 2)
                psf = l * eta + (1 - eta) * g
        else:  # ndim=3
            nz, ny, nx = self.iobs.shape
            z, y, x = np.meshgrid(fftfreq(nz) * nz, fftfreq(ny) * ny, fftfreq(nx) * nx, indexing='ij')
            if "gauss" in model.lower():
                sigma = fwhm / 2.3548
                psf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
            elif "lorentz" in model.lower():
                psf = 2 / np.pi * fwhm / (x ** 2 + y ** 2 + z ** 2 + fwhm ** 2)
            else:
                sigma = fwhm / 2.3548
                g = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
                l = 2 / np.pi * fwhm / (x ** 2 + y ** 2 + z ** 2 + fwhm ** 2)
                psf = l * eta + (1 - eta) * g
        self._psf_f = rfftn(psf).astype(np.complex64)

    def init_free_pixels(self, ratio=5e-2, island_radius=3, exclude_zone_center=0.05, verbose=False, coords=None):
        """
        Random selection of 'free' pixels in the observed intensities, to be used as unbiased indicator.
        The free pixel iobs values are modified as iobs_free = -iobs - 1

        :param ratio: (approximate) relative number of pixels to be included in the free mask
        :param island_radius: free island radius, to avoid pixel correlation due to finite object size
        :param exclude_zone_center: the relative radius of the zone to be excluded near the center
        :param verbose: if True, be verbose
        :param coords: instead of generating random coordinates, these can be given as a tuple
            of (ix, iy[, iz]). All coordinates should be at least island_radius far from the borders,
            and these coordinates are relative to the array centre, i.e.
            within [-size/2 + island_radius ; size/2 - island_radius] along each axis.
        :return:  a tuple (ix, iy[, iz]) of the pixel coordinates of the islands,
            before taking into account any mask.
        """
        # Clear previous free mask
        tmp = np.logical_and(self.iobs > -1e19, self.iobs < 0)
        self.iobs[tmp] = -(self.iobs[tmp] + 1)

        if self.iobs.ndim == 3:
            nz, ny, nx = self.iobs.shape
            if coords is None:
                nb = int(self.iobs.size * ratio / (4 / 3 * np.pi * island_radius ** 3))
                iz = np.random.randint(-nz // 2 + island_radius, nz // 2 - island_radius, nb)
                iy = np.random.randint(-ny // 2 + island_radius, ny // 2 - island_radius, nb)
                ix = np.random.randint(-nx // 2 + island_radius, nx // 2 - island_radius, nb)
                idx = np.nonzero(((ix / (nx * exclude_zone_center)) ** 2 +
                                  (iy / (ny * exclude_zone_center)) ** 2 +
                                  (iz / (nz * exclude_zone_center)) ** 2) > 1)
                ix, iy, iz = np.take(ix, idx), np.take(iy, idx), np.take(iz, idx)
            else:
                ix, iy, iz = coords[0].astype(np.int32), coords[1].astype(np.int32), coords[2].astype(np.int32)
            m = np.zeros_like(self.iobs, dtype=np.bool)
            ix += nx // 2
            iy += ny // 2
            iz += nz // 2
            # There may be faster ways to do this. But nb is small, so a sparse approach could be faster
            for dx in range(-island_radius, island_radius + 1):
                for dy in range(-island_radius, island_radius + 1):
                    for dz in range(-island_radius, island_radius + 1):
                        if (dx ** 2 + dy ** 2 + dz ** 2) <= island_radius ** 2:
                            m[iz + dz, iy + dy, ix + dx] = True
            # We return coordinates relative to diffraction centre
            coords = ix - nx // 2, iy - ny // 2, iz - nz // 2
        else:
            ny, nx = self.iobs.shape
            if coords is None:
                nb = int(self.iobs.size * ratio / (np.pi * island_radius ** 2))
                iy = np.random.randint(-ny // 2 + island_radius, ny // 2 - island_radius, nb)
                ix = np.random.randint(-nx // 2 + island_radius, nx // 2 - island_radius, nb)
                idx = np.nonzero(((ix / (nx * exclude_zone_center)) ** 2 +
                                  (iy / (ny * exclude_zone_center)) ** 2) > 1)
                ix, iy = np.take(ix, idx), np.take(iy, idx)
            else:
                ix, iy = coords[0].astype(np.int32), coords[1].astype(np.int32)
            m = np.zeros_like(self.iobs, dtype=np.bool)
            ix += nx // 2
            iy += ny // 2
            # There may be faster ways to do this. But nb is small, so a sparse approach could be faster
            for dx in range(-island_radius, island_radius + 1):
                for dy in range(-island_radius, island_radius + 1):
                    if (dx ** 2 + dy ** 2) <= island_radius ** 2:
                        m[iy + dy, ix + dx] = True
                        # print(dx, dy, m.sum())
            # We return coordinates relative to diffraction centre
            coords = ix - nx // 2, iy - ny // 2
        # fft-shift mask
        m = fftshift(m)
        # Exclude masked pixels
        m[self.iobs <= -1e19] = False

        # Flag free pixels
        self.iobs[m] = -self.iobs[m] - 1

        n = m.sum()
        del m
        gc.collect()
        self.nb_free_points = n
        if verbose:
            print('Initialized free mask with %d pixels (%6.3f%%)' % (n, 100 * n / self.iobs.size))
        self.set_iobs(self.iobs)  # This will set the timestamp
        return coords

    def set_free_pixel_mask(self, m, verbose=False, shift=False):
        """
        Set the free pixel mask. Assumes the free pixel mask is correct (excluding center, bad pixels)

        :param m: the boolean mask (1=masked, 0 otherwise)
        :param shift: if True, the mask array will be fft-shifted so that the center of the stored data is
          in the corner of the array. [default: the array is already shifted]
        :return: nothing
        """
        if shift:
            m = fftshift(m)
        # Clear previous free mask
        tmp = np.logical_and(self.iobs > -1e19, self.iobs < 0)
        self.iobs[tmp] = -(self.iobs[tmp] + 1)

        # Flag free pixels
        self.iobs[m] = -self.iobs[m] - 1
        # print("finished", timeit.default_timer() - t0)

        n = m.sum()
        self.nb_free_points = n
        if verbose:
            print('Set free mask with %d pixels (%6.3f%%)' % (n, 100 * n / m.size))
        self.set_iobs(self.iobs)  # This will set the timestamp

    def get_free_pixel_mask(self):
        return np.logical_and(self.iobs > -1e19, self.iobs < -0.5).astype(np.bool)

    def save_data_cxi(self, filename, sample_name=None, experiment_id=None, instrument=None, process_parameters=None):
        """
        Save the diffraction data (observed intensity, mask) to an HDF% CXI file.
        :param filename: the file name to save the data to
        :param sample_name: optional, the sample name
        :param experiment_id: the string identifying the experiment, e.g.: 'HC1234: Siemens star calibration tests'
        :param instrument: the string identifying the instrument, e.g.: 'ESRF id10'
        :param process_parameters: a dictionary of parameters which will be saved as a NXcollection
        :return: Nothing. a CXI file is created
        """
        # Remove mask
        iobs = self.iobs.copy()
        mask = self.iobs <= -1e19
        iobs[mask] = 0
        # Remove free mask
        tmp = iobs < 0
        iobs[tmp] = -(iobs[tmp] + 1)

        save_cdi_data_cxi(filename, iobs=iobs, wavelength=self.wavelength,
                          detector_distance=self.detector_distance,
                          pixel_size_detector=self.pixel_size_detector, mask=mask, sample_name=sample_name,
                          experiment_id=experiment_id, instrument=instrument, iobs_is_fft_shifted=True,
                          process_parameters=process_parameters)

    def save_obj_cxi(self, filename, sample_name=None, experiment_id=None, instrument=None, note=None, crop=0,
                     save_psf=True, process_notes=None, process_parameters=None, append=False,
                     shift_phase_zero=True, **kwargs):
        """
        Save the result of the optimisation (object, support) to an HDF5 CXI file.
        :param filename: the file name to save the data to
        :param sample_name: optional, the sample name
        :param experiment_id: the string identifying the experiment, e.g.: 'HC1234: Siemens star calibration tests'
        :param instrument: the string identifying the instrument, e.g.: 'ESRF id10'
        :param note: a string with a text note giving some additional information about the data, a publication...
        :param crop: integer, if >0, the object will be cropped arount the support, plus a margin of 'crop' pixels.
        :param save_psf: if True, also save the psf (if present)
        :param process_notes: a dictionary which will be saved in '/entry_N/process_1'. A dictionary entry
        can also be a 'note' as keyword and a dictionary as value - all key/values will then be saved
        as separate notes. Example: process={'program': 'PyNX', 'note':{'llk':1.056, 'nb_photons': 1e8}}
        :param process_parameters: a dictionary of parameters which will be saved as a NXcollection
        :param append: if True and the file already exists, the result will be saved in a new entry.
        :param shift_phase_zero: if True, try to center the object phase around zero inside the support
        :return: Nothing. a CXI file is created
        """

        if 'process' in kwargs:
            warnings.warn("CDI.save_obj_cxi(): parameter 'process' is deprecated,"
                          "use process_notes and process_parameters instead.")
            if process_notes is None:
                process_notes = kwargs['process']
            else:
                warnings.warn("CDI.save_obj_cxi(): parameter 'process' ignored as 'process_notes' is also given")

        if append:
            f = h5py.File(filename, "a")
            if "cxi_version" not in f:
                f.create_dataset("cxi_version", data=150)
            i = 1
            while True:
                if 'entry_%d' % i not in f:
                    break
                i += 1
            entry = f.create_group("/entry_%d" % i)
            entry_path = "/entry_%d" % i
            f.attrs['default'] = "/entry_%d" % i
            if "/entry_last" in entry:
                del entry["/entry_last"]
            entry["/entry_last"] = h5py.SoftLink("/entry_%d" % i)
        else:
            f = h5py.File(filename, "w")
            f.create_dataset("cxi_version", data=150)
            entry = f.create_group("/entry_1")
            entry_path = "/entry_1"
            f.attrs['default'] = 'entry_1'
            entry["/entry_last"] = h5py.SoftLink("/entry_1")

        f.attrs['creator'] = 'PyNX'
        # f.attrs['NeXus_version'] = '2018.5'  # Should only be used when the NeXus API has written the file
        f.attrs['HDF5_Version'] = h5py.version.hdf5_version
        f.attrs['h5py_version'] = h5py.version.version

        entry.attrs['NX_class'] = 'NXentry'
        # entry.create_dataset('title', data='1-D scan of I00 v. mr')
        entry.attrs['default'] = 'data_1'
        entry.create_dataset("program_name", data="PyNX %s" % _pynx_version)
        entry.create_dataset("start_time", data=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())))
        if experiment_id is not None:
            entry.create_dataset("experiment_id", data=experiment_id)

        if note is not None:
            note_1 = entry.create_group("note_1")
            note_1.create_dataset("data", data=note)
            note_1.create_dataset("type", data="text/plain")

        if sample_name is not None:
            sample_1 = entry.create_group("sample_1")
            sample_1.attrs['NX_class'] = 'NXsample'
            sample_1.create_dataset("name", data=sample_name)

        obj = self.get_obj(shift=True)
        sup = self.get_support(shift=True)
        if crop > 0:
            # crop around support
            if self.iobs.ndim == 3:
                l0 = np.nonzero(sup.sum(axis=(1, 2)))[0].take([0, -1]) + np.array([-crop, crop])
                if l0[0] < 0: l0[0] = 0
                if l0[1] >= sup.shape[0]: l0[1] = -1

                l1 = np.nonzero(sup.sum(axis=(0, 2)))[0].take([0, -1]) + np.array([-crop, crop])
                if l1[0] < 0: l1[0] = 0
                if l1[1] >= sup.shape[1]: l1[1] = -1

                l2 = np.nonzero(sup.sum(axis=(0, 1)))[0].take([0, -1]) + np.array([-crop, crop])
                if l2[0] < 0: l2[0] = 0
                if l2[1] >= sup.shape[2]: l2[1] = -1
                obj = obj[l0[0]:l0[1], l1[0]:l1[1], l2[0]:l2[1]]
                sup = sup[l0[0]:l0[1], l1[0]:l1[1], l2[0]:l2[1]]
            else:
                l0 = np.nonzero(sup.sum(axis=1))[0].take([0, -1]) + np.array([-crop, crop])
                if l0[0] < 0: l0[0] = 0
                if l0[1] >= sup.shape[0]: l0[1] = -1

                l1 = np.nonzero(sup.sum(axis=0))[0].take([0, -1]) + np.array([-crop, crop])
                if l1[0] < 0: l1[0] = 0
                if l1[1] >= sup.shape[1]: l1[1] = -1

                obj = obj[l0[0]:l0[1], l1[0]:l1[1]]
                sup = sup[l0[0]:l0[1], l1[0]:l1[1]]

        if shift_phase_zero:
            obj = phase.shift_phase_zero(obj, mask=sup, stack=False)

        image_1 = entry.create_group("image_1")
        image_1.create_dataset("data", data=obj, chunks=True, shuffle=True, compression="gzip")
        image_1.attrs['NX_class'] = 'NXdata'  # Is image_1 a well-formed NXdata or not ?
        image_1.attrs['signal'] = 'data'
        if obj.ndim == 3:
            image_1.attrs['interpretation'] = 'scaler'  # NeXus specs don't make sense
        else:
            image_1.attrs['interpretation'] = 'image'
        image_1.create_dataset("mask", data=sup, chunks=True, shuffle=True, compression="gzip")
        image_1["support"] = h5py.SoftLink('%s/image_1/mask' % entry_path)
        image_1.create_dataset("data_type", data="electron density")  # CXI, not NeXus
        image_1.create_dataset("data_space", data="real")  # CXI, not NeXus ?
        if self.pixel_size_object is not None:
            s = self.pixel_size_object * np.array(obj.shape)
            image_1.create_dataset("image_size", data=s)
            if False:  # TODO: x, y (z) axes description (scale), when accessible
                # See proposeition: https://www.nexusformat.org/2014_axes_and_uncertainties.html
                #
                # The following is only true if pixel size is the same along X and Y. Not always the case !
                ny, nx = obj.shape[-2:]
                x = (np.arange(nx) - nx // 2) * self.pixel_size_object
                y = (np.arange(ny) - ny // 2) * self.pixel_size_object
                image_1.create_dataset("x", data=x)
                image_1.create_dataset("y", data=y)
                image_1.attrs['axes'] = ". y x"  # How does this work ? "x y" or ['x', 'y'] ????
                # image_1.attrs['x_indices'] = [-1, ]
                # image_1.attrs['y_indices'] = [-2, ]

        instrument_1 = image_1.create_group("instrument_1")
        instrument_1.attrs['NX_class'] = 'NXinstrument'
        if instrument is not None:
            instrument_1.create_dataset("name", data=instrument)

        if self.wavelength is not None:
            nrj = 12.3984 / (self.wavelength * 1e10) * 1.60218e-16
            source_1 = instrument_1.create_group("source_1")
            source_1.attrs['NX_class'] = 'NXsource'
            source_1.attrs['note'] = 'Incident photon energy (instead of source energy), for CXI compatibility'
            source_1.create_dataset("energy", data=nrj)
            source_1["energy"].attrs['units'] = 'J'

            beam_1 = instrument_1.create_group("beam_1")
            beam_1.attrs['NX_class'] = 'NXbeam'
            beam_1.create_dataset("incident_energy", data=nrj)
            beam_1["incident_energy"].attrs['units'] = 'J'
            beam_1.create_dataset("incident_wavelength", data=self.wavelength)
            beam_1["incident_wavelength"].attrs['units'] = 'm'

        detector_1 = instrument_1.create_group("detector_1")
        detector_1.attrs['NX_class'] = 'NXdetector'
        if self.detector_distance is not None:
            detector_1.create_dataset("distance", data=self.detector_distance)
            detector_1["distance"].attrs['units'] = 'm'

        if self.pixel_size_detector is not None:
            detector_1.create_dataset("x_pixel_size", data=self.pixel_size_detector)
            detector_1["x_pixel_size"].attrs['units'] = 'm'
            detector_1.create_dataset("y_pixel_size", data=self.pixel_size_detector)
            detector_1["y_pixel_size"].attrs['units'] = 'm'

        if self._psf_f is not None and save_psf:
            # PSF for partial coherence modeling
            # Only keep the center, 48 pixels average, so rebin before fft
            s = np.array(self._psf_f.shape)
            s[-1] = (s[-1] - 1) * 2  # Correct half-Hermitian size
            r = np.int(np.round(np.prod(s // 48) ** (1 / len(s))))
            if r > 1:
                psf = fftshift(irfftn(rebin(self._psf_f, r)))
            else:
                psf = fftshift(irfftn(self._psf_f))
            detector_1.create_dataset("point_spread_function", data=psf.astype(np.float32),
                                      chunks=True, shuffle=True, compression="gzip")

        # Add shortcut to the main data
        data_1 = entry.create_group("data_1")
        data_1["data"] = h5py.SoftLink('%s/image_1/data' % entry_path)
        data_1.attrs['NX_class'] = 'NXdata'
        data_1.attrs['signal'] = 'data'
        if obj.ndim == 3:
            data_1.attrs['interpretation'] = 'scaler'  # NeXus specs don't make sense'
        else:
            data_1.attrs['interpretation'] = 'image'
        # TODO: x, y (z) axes description (scale), when accessible

        command = ""
        for arg in sys.argv:
            command += arg + " "
        process_1 = image_1.create_group("process_1")
        process_1.attrs['NX_class'] = 'NXprocess'
        process_1.create_dataset("program", data='PyNX')  # NeXus spec
        process_1.create_dataset("version", data="%s" % _pynx_version)  # NeXus spec
        process_1.create_dataset("command", data=command)  # CXI spec

        if process_notes is not None:  # Notes
            for k, v in process_notes.items():
                if isinstance(v, str):
                    if k not in process_1:
                        process_1.create_dataset(k, data=v)
                elif isinstance(v, dict) and k == 'note':
                    # Save this as notes:
                    for kk, vv in v.items():
                        i = 1
                        while True:
                            note_s = 'note_%d' % i
                            if note_s not in process_1:
                                break
                            i += 1
                        note = process_1.create_group(note_s)
                        note.attrs['NX_class'] = 'NXnote'
                        note.create_dataset("description", data=kk)
                        # TODO: also save values as floating-point if appropriate
                        note.create_dataset("data", data=str(vv))
                        note.create_dataset("type", data="text/plain")

        # Configuration & results of process: custom ESRF data policy
        # see https://gitlab.esrf.fr/sole/data_policy/blob/master/ESRF_NeXusImplementation.rst
        config = process_1.create_group("configuration")
        config.attrs['NX_class'] = 'NXcollection'
        if process_parameters is not None:
            for k, v in process_parameters.items():
                if k == 'free_pixel_mask':
                    k = 'free_pixel_mask_input'
                if v is not None:
                    if type(v) is dict:
                        # This can happen if complex configuration is passed on
                        if len(v):
                            kd = config.create_group(k)
                            kd.attrs['NX_class'] = 'NXcollection'
                            for kk, vv in v.items():
                                kd.create_dataset(kk, data=vv)
                    else:
                        config.create_dataset(k, data=v)
        if self.iobs is not None and 'iobs_shape' not in config:
            config.create_dataset('iobs_shape', data=self.iobs.shape)

        # Save the free pixel mask so that it can be re-used
        mask_free_pixel = self.get_free_pixel_mask()
        if mask_free_pixel.sum():
            config.create_dataset('free_pixel_mask', data=fftshift(mask_free_pixel.astype(np.bool)), chunks=True,
                                  shuffle=True, compression="gzip")
            config['free_pixel_mask'].attrs['note'] = "Mask of pixels used for free log-likelihood calculation"
            if mask_free_pixel.ndim == 3:
                config['free_pixel_mask'].attrs['interpretation'] = 'scaler'
            else:
                config['free_pixel_mask'].attrs['interpretation'] = 'image'

        if self.get_upsample() is not None:
            config.create_dataset('upsample_f', data=self.get_upsample())
        if self.get_bin() is not None:
            config.create_dataset('bin_f', data=self.get_bin())
        if self.get_crop() is not None:
            config.create_dataset('crop_f', data=self.get_crop())

        results = process_1.create_group("results")
        results.attrs['NX_class'] = 'NXcollection'
        llk = self.get_llk(normalized=True)
        results.create_dataset('llk_poisson', data=llk[0])
        results.create_dataset('llk_gaussian', data=llk[1])
        results.create_dataset('llk_euclidian', data=llk[2])
        results.create_dataset('free_llk_poisson', data=llk[3])
        results.create_dataset('free_llk_gaussian', data=llk[4])
        results.create_dataset('free_llk_euclidian', data=llk[5])
        results.create_dataset('nb_point_support', data=self.nb_point_support)
        h = self.history.as_numpy_record_array()
        h['time'] -= self.history.t0  # Only 4 bytes actual accuracy,
        results.create_dataset('cycle_history', data=h)
        for k in self.history.keys():
            results.create_dataset('cycle_history_%s' % k, data=self.history[k].as_numpy_record_array())

        f.close()

    def get_llkn(self):
        """
        Get the poisson normalized log-likelihood, which should converge to 1 for a statistically ideal fit
        to a Poisson-noise  dataset.

        :return: the normalized log-likelihood, for Poisson noise.
        """
        warnings.warn("cdi.get_llkn is deprecated. Use cdi.get_llk instead", DeprecationWarning)
        return self.get_llk(noise='poisson', normalized=True)

    def get_llk(self, noise=None, normalized=True):
        """
        Get the normalized log-likelihoods, which should converge to 1 for a statistically ideal fit.

        :param noise: either 'gaussian', 'poisson' or 'euclidian', will return the corresponding log-likelihood.
        :param normalized: if True, will return normalized values so that the llk from a statistically ideal model
                           should converge to 1
        :return: the log-likelihood, or if noise=None, a tuple of the three (poisson, gaussian, euclidian)
                 log-likelihoods.
        """
        n = 1
        nfree = 1
        if normalized:
            n = 1 / max((self.nb_observed_points - self.nb_point_support - self.nb_free_points), 1e-10)
            if self.nb_free_points > 0:
                nfree = 1 / self.nb_free_points
            else:
                nfree = 0

        if noise is None:
            return self.llk_poisson * n, self.llk_gaussian * n, self.llk_euclidian * n, \
                   self.llk_poisson_free * nfree, self.llk_gaussian_free * nfree, self.llk_euclidian_free * nfree
        elif 'poiss' in str(noise).lower():
            return self.llk_poisson * n
        elif 'gauss' in str(noise).lower():
            return self.llk_gaussian * n
        elif 'eucl' in str(noise).lower():
            return self.llk_euclidian * n

    def reset_history(self):
        """
        Reset history, and set current cycle to zero
        :return: nothing
        """
        self.history = History()
        self.cycle = 0

    def update_history(self, mode='algorithm', verbose=False, **kwargs):
        """
        Update the history record.
        :param mode: either 'llk' (will record new log-likelihood, nb_photons_calc, average value..),
        or 'algorithm' (will only update the algorithm) or 'support'.
        :param verbose: if True, print some info about current process
        :param kwargs: other parameters to be recorded, e.g. support_threshold=xx, dt=
        :return: nothing
        """
        if self._psf_f is not None:
            psf = 1
            psf_s = "[PSF]"
        else:
            psf = 0
            psf_s = ""

        upsample_f = self.get_upsample()
        if upsample_f is None:
            upsample_f = 1
        kwargs['upsample'] = upsample_f

        crop_f = self.get_crop()
        if crop_f is None:
            crop_f = 1
        kwargs['crop'] = crop_f

        bin_f = self.get_bin()
        if bin_f is None:
            bin_f = 1
        kwargs['bin'] = bin_f

        if mode == 'llk':
            llk = self.get_llk()
            algo = ''
            if 'algorithm' in kwargs:
                algo = kwargs['algorithm']

            # Getting a good dt is tricky, because GPU calculations are asynchronous,
            # so the history of cycles is recorded before they actually happen, except
            # when wwe get the LLK, which is synchronous
            dt = None
            if 'dt' in self.history:
                for k in reversed(self.history['dt'].keys()):
                    if k != self.cycle:
                        dt = (timeit.default_timer() - self.history['time'][k]) / (self.cycle - k)
                        break
            if dt is None:
                dt = (timeit.default_timer() - self.history.t0) / (self.cycle + 1)
            kwargs['dt'] = dt

            if verbose:
                nbs = self.nb_point_support
                print("%4s #%3d LLK= %7.3f[free=%7.3f](p), nb photons=%e, "
                      "support:nb=%6d (%6.3f%%) <obj>=%10.2f max=%10.2f,"
                      " out=%4.3f%% dt/cycle=%6.4fs %s" % (
                          algo, self.cycle, llk[0], llk[3], self.nb_photons_calc,
                          self.nb_point_support, nbs / self._obj.size * 100,
                          np.sqrt(self.nb_photons_calc / nbs), self._obj_max, self._obj2_out * 100, dt, psf_s))

            self.history.insert(self.cycle, llk_poisson=llk[0], llk_poisson_free=llk[3],
                                llk_gaussian=llk[1], llk_gaussian_free=llk[4], llk_euclidian=llk[2],
                                llk_euclidian_free=llk[2], nb_photons_calc=self.nb_photons_calc,
                                obj_average=np.sqrt(self.nb_photons_calc / self.nb_point_support), psf=psf,
                                **kwargs)
        elif mode == 'support':
            self.history.insert(self.cycle, support_size=self.nb_point_support, obj_max=self._obj_max,
                                obj_average=np.sqrt(self.nb_photons_calc / self.nb_point_support), psf=psf,
                                obj2_out=self._obj2_out, **kwargs)
        elif 'algo' in mode:
            if 'algorithm' in kwargs:
                self.history.insert(self.cycle, psf=psf, **kwargs)

    def __rmul__(self, x):
        """
        Multiply object (by a scalar).

        This is a placeholder for a function which will be replaced when importing either CUDA or OpenCL operators.
        If called before being replaced, will raise an error

        :param x: the scalar by which the wavefront will be multiplied
        :return:
        """
        if np.isscalar(x):
            raise OperatorException(
                "ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s. Did you import operators ?" % (str(x), str(self)))
        else:
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s." % (str(x), str(self)))

    def __mul__(self, x):
        """
        Multiply object (by a scalar).

        This is a placeholder for a function which will be replaced when importing either CUDA or OpenCL operators.
        If called before being replaced, will raise an error

        :param x: the scalar by which the wavefront will be multiplied
        :return:
        """
        if np.isscalar(x):
            raise OperatorException(
                "ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s. Did you import operators ?" % (str(self), str(x)))
        else:
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s." % (str(self), str(x)))

    def __str__(self):
        return "CDI"


def save_cdi_data_cxi(filename, iobs, wavelength=None, detector_distance=None, pixel_size_detector=None, mask=None,
                      sample_name=None, experiment_id=None, instrument=None, note=None, iobs_is_fft_shifted=False,
                      process_parameters=None):
    """
    Save the diffraction data (observed intensity, mask) to an HDF5 CXI file, NeXus-compatible.
    :param filename: the file name to save the data to
    :param iobs: the observed intensity
    :param wavelength: the wavelength of the experiment (in meters)
    :param detector_distance: the detector distance (in meters)
    :param pixel_size_detector: the pixel size of the detector (in meters)
    :param mask: the mask indicating valid (=0) and bad pixels (>0)
    :param sample_name: optional, the sample name
    :param experiment_id: the string identifying the experiment, e.g.: 'HC1234: Siemens star calibration tests'
    :param instrument: the string identifying the instrument, e.g.: 'ESRF id10'
    :param iobs_is_fft_shifted: if true, input iobs (and mask if any) have their origin in (0,0[,0]) and will be shifted
    back to centered-versions before being saved.
    :param process_parameters: a dictionary of parameters which will be saved as a NXcollection
    :return: Nothing. a CXI file is created
    """
    f = h5py.File(filename, "w")
    f.attrs['file_name'] = filename
    f.attrs['file_time'] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time()))
    if instrument is not None:
        f.attrs['instrument'] = instrument
    f.attrs['creator'] = 'PyNX'
    # f.attrs['NeXus_version'] = '2018.5'  # Should only be used when the NeXus API has written the file
    f.attrs['HDF5_Version'] = h5py.version.hdf5_version
    f.attrs['h5py_version'] = h5py.version.version
    f.attrs['default'] = 'entry_1'
    f.create_dataset("cxi_version", data=140)

    entry_1 = f.create_group("entry_1")
    entry_1.create_dataset("program_name", data="PyNX %s" % _pynx_version)
    entry_1.create_dataset("start_time", data=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())))
    entry_1.attrs['NX_class'] = 'NXentry'
    # entry_1.create_dataset('title', data='1-D scan of I00 v. mr')
    entry_1.attrs['default'] = 'data_1'

    if experiment_id is not None:
        entry_1.create_dataset("experiment_id", data=experiment_id)

    if note is not None:
        note_1 = entry_1.create_group("note_1")
        note_1.create_dataset("data", data=note)
        note_1.create_dataset("type", data="text/plain")

    if sample_name is not None:
        sample_1 = entry_1.create_group("sample_1")
        sample_1.create_dataset("name", data=sample_name)

    instrument_1 = entry_1.create_group("instrument_1")
    instrument_1.attrs['NX_class'] = 'NXinstrument'
    if instrument is not None:
        instrument_1.create_dataset("name", data=instrument)

    if wavelength is not None:
        nrj = 12.3984 / (wavelength * 1e10) * 1.60218e-16
        source_1 = instrument_1.create_group("source_1")
        source_1.attrs['NX_class'] = 'NXsource'
        source_1.attrs['note'] = 'Incident photon energy (instead of source energy), for CXI compatibility'
        source_1.create_dataset("energy", data=nrj)
        source_1["energy"].attrs['units'] = 'J'

        beam_1 = instrument_1.create_group("beam_1")
        beam_1.attrs['NX_class'] = 'NXbeam'
        beam_1.create_dataset("incident_energy", data=nrj)
        beam_1["incident_energy"].attrs['units'] = 'J'
        beam_1.create_dataset("incident_wavelength", data=wavelength)
        beam_1["incident_wavelength"].attrs['units'] = 'm'

    detector_1 = instrument_1.create_group("detector_1")
    detector_1.attrs['NX_class'] = 'NX_detector'

    if detector_distance is not None:
        detector_1.create_dataset("distance", data=detector_distance)
        detector_1["distance"].attrs['units'] = 'm'
    if pixel_size_detector is not None:
        detector_1.create_dataset("x_pixel_size", data=pixel_size_detector)
        detector_1["x_pixel_size"].attrs['units'] = 'm'
        detector_1.create_dataset("y_pixel_size", data=pixel_size_detector)
        detector_1["y_pixel_size"].attrs['units'] = 'm'
    if iobs_is_fft_shifted:
        detector_1.create_dataset("data", data=fftshift(iobs), chunks=True, shuffle=True,
                                  compression="gzip")
    else:
        detector_1.create_dataset("data", data=iobs, chunks=True, shuffle=True,
                                  compression="gzip")

    if mask is not None:
        if mask.sum() != 0:
            if iobs_is_fft_shifted:
                detector_1.create_dataset("mask", data=fftshift(mask), chunks=True, shuffle=True,
                                          compression="gzip")
            else:
                detector_1.create_dataset("mask", data=mask, chunks=True, shuffle=True, compression="gzip")
    if False:
        # Basis vector - this is the default CXI convention, so could be skipped
        # This corresponds to a 'top, left' origin convention
        basis_vectors = np.zeros((2, 3), dtype=np.float32)
        basis_vectors[0, 1] = -pixel_size_detector
        basis_vectors[1, 0] = -pixel_size_detector
        detector_1.create_dataset("basis_vectors", data=basis_vectors)

    data_1 = entry_1.create_group("data_1")
    data_1.attrs['NX_class'] = 'NXdata'
    data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
    data_1.attrs['signal'] = 'data'
    if iobs.ndim == 3:
        data_1.attrs['interpretation'] = 'scaler'  # NeXus specs don't make sense
    else:
        data_1.attrs['interpretation'] = 'image'

    # Remember how import was done
    command = ""
    for arg in sys.argv:
        command += arg + " "
    process_1 = data_1.create_group("process_1")
    process_1.attrs['NX_class'] = 'NXprocess'
    process_1.create_dataset("program", data='PyNX')  # NeXus spec
    process_1.create_dataset("version", data="%s" % _pynx_version)  # NeXus spec
    process_1.create_dataset("command", data=command)  # CXI spec

    # Configuration & results of process: custom ESRF data policy
    # see https://gitlab.esrf.fr/sole/data_policy/blob/master/ESRF_NeXusImplementation.rst
    if process_parameters is not None:
        config = process_1.create_group("configuration")
        config.attrs['NX_class'] = 'NXcollection'
        for k, v in process_parameters.items():
            if v is not None:
                if type(v) is dict:
                    # This can happen if complex configuration is passed on
                    if len(v):
                        kd = config.create_group(k)
                        kd.attrs['NX_class'] = 'NXcollection'
                        for kk, vv in v.items():
                            kd.create_dataset(kk, data=vv)
                else:
                    config.create_dataset(k, data=v)

    f.close()


class OperatorCDI(Operator):
    """
    Base class for an operator on CDI objects, not requiring a processing unit.
    """

    def timestamp_increment(self, cdi):
        cdi._timestamp_counter += 1


class CDIOperatorException(OperatorException):
    pass


class SupportTooSmall(CDIOperatorException):
    pass


class SupportTooLarge(CDIOperatorException):
    pass


def calc_throughput(p: CDI = None, cxi=None, verbose=False):
    """
    Analyse the throughput after a series of algorithms, either from a CDI
    object or from a CXI file.
    This does not take into account operations for support update, or partial coherence (PSF).
    :param p: the CDI object the timings are extracted from.
    :param cxi: the CXI file the history of cycles will be obtained from
    :param verbose: if True, print average throughput per algorithm steps
    :return: the average throughput in Gbyte/s
    """
    if cxi is not None:
        with h5py.File(cxi, 'r') as tmp:
            h = tmp['/entry_last/image_1/process_1/results/cycle_history'][()]
            shape = tmp['/entry_last/image_1/process_1/configuration/iobs_shape'][()]
            size = np.cumproduct(shape)[-1]
            ndim = len(shape)
    else:
        h = p.history.as_numpy_record_array()
        size = p.iobs.size
        ndim = p.iobs.ndim

    # Total number of read or write of the main array, assumed to be complex64
    # Does not take into account support update
    vnbrw = []
    for i in range(len(h)):
        algo = h['algorithm'][i].decode()

        # Upsample increases the array size except for Iobs
        # Upsample is stored in a condensed form 222 if (2,2,2)
        us = h['upsample'][i] % 10
        if h['upsample'][i] > 10:
            us *= (h['upsample'][i] // 10) % 10
        if h['upsample'][i] > 100:
            us *= h['upsample'][i] // 100

        if isinstance(algo, np.bytes_):
            algo = algo.decode('ascii')
        algo = algo.lower()
        nb_psf = 0
        if h['psf'][i]:
            nb_psf = 2 * 2 * ndim * 0.5 + 3.5
        if h['update_psf'][i]:
            # 8 r+w + 4 r2c FFT of size iobs
            nb_psf += (8 + 4 * ndim) * 0.5
        if algo == 'er':
            # ER: 2 FFT (1 read + 1 write for each dimension)
            # 2.5 r or w for amplitude projection, 2 for support constraint
            vnbrw.append((2 * 2 * ndim + 2 + 2) * us + .5 + nb_psf)
        elif algo in ['hio', 'raar']:
            # 1 extra r+w compared to er
            vnbrw.append((2 * 2 * ndim + 2 + 2 + 2) * us + .5 + nb_psf)

    vnbrw = np.array(vnbrw, dtype=np.float32)
    # Gbyte read or write per cycle
    vgb = size * 8 * vnbrw

    if verbose:
        print("Estimated memory throughput of algorithms:")
        # Average throughput per series of algorithm
        # ideally we'd need begin and end time..
        i0 = 0
        algo = h['algorithm'][0]
        if isinstance(algo, np.bytes_):
            algo = algo.decode('ascii')
        if h['psf'][0]:
            algo += "[PSF]"
        for i in range(1, len(h)):
            algo1 = h['algorithm'][i]
            if isinstance(algo1, np.bytes_):
                algo1 = algo1.decode('ascii')
            if h['psf'][i]:
                algo1 += "[PSF]"
            if algo1 != algo:
                dt = h['time'][i] - h['time'][i0]
                g = vgb[i0:i].sum() / dt / 1024 ** 3
                print("     %8s**%3d [dt=%5.1fs  <dt/cycle>=%6.4fs]: %6.1f Gbytes/s" %
                      (algo, i - i0, dt, dt / (i - i0), g))
                i0 = i
                algo = algo1
        if i > i0:  # How can i==i0 happen ?
            dt = h['time'][i] - h['time'][i0]
            g = vgb[i0:i].sum() / dt / 1024 ** 3
            print("     %8s**%3d [dt=%5.1fs  <dt/cycle>=%6.4fs]: %6.1f Gbytes/s" % (algo, i - i0, dt, dt / (i - i0), g))

    # average throughput
    gbps = vgb.sum() / (h['time'][-1] - h['time'][0]) / 1024 ** 3

    return gbps
