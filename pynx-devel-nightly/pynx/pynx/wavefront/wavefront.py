# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['Wavefront', 'UserWarningWavefrontNearFieldPropagation']

import warnings
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
from scipy import misc
from skimage import data
from ..operator import Operator, OperatorException


class Wavefront(object):
    """
    A 2D Wavefront object
    """

    def __init__(self, d=None, z=0, pixel_size=55e-6, wavelength=12398.4e-10 / 8000, copy_d=True):
        """
        Create the Wavefront object.

        Args:
            z: current position of 2D Wavefront along direction propagation, in meters
            pixel_size: current pixelsize at z, in meters
            d: the original data array at z - will be converted to complex64 if needed.
               This can either be :
               - a 2D array
               - a 3D array which will be treated as a stack of 2D wavefront for multi-views/modes propagation.
               - None: will be initialized to a 512x512 data array filled with 1.
               - a string to use one image from either scipy or scikit-image data. These will be truncated to 512x512.
               Available string values are 'ascent', 'face', 'camera', 'hubble', 'immunohistochemistry'.
               In the case of RGB images, all 3 components are loaded.

               The data should have its center at (0,0) to avoid requiring fftshifts during propagation.

               Any 2D data will be converted to 3D with a shape of (1, ny, nx)

            wavelength: X-ray wavelength in meters.
        """
        self.z = z  # Track the current position of the wavefront along propagation (in meters)
        self.pixel_size = pixel_size  # Pixel size, in meters
        self.wavelength = wavelength  # Wavelength in meters
        if d is None:
            self._d = np.ones((512, 512), dtype=np.complex64)
        else:
            if isinstance(d, str):
                if d.lower() == 'ascent':
                    self._d = misc.ascent()
                elif d.lower() == 'face':
                    self._d = misc.face()
                elif d.lower() == 'camera':
                    self._d = data.camera()
                elif d.lower() == 'hubble':
                    self._d = data.hubble_deep_field()
                elif d.lower() == 'immunohistochemistry':
                    self._d = data.immunohistochemistry()
                else:
                    warnings.warn('Wavefront: did not understand image name: %s' % (d))
                    self._d = np.ones((512, 512), dtype=np.complex64)
                # Take care of RGB images
                if self._d.ndim == 3:
                    self._d = np.moveaxis(self._d, 2, 0)
                # Crop to 512x512
                ny, nx = self._d.shape[-2:]
                if self._d.ndim == 3:
                    if ny > 512:
                        self._d = self._d[:, ny // 2 - 256:ny // 2 + 256, :]
                    if nx > 512:
                        self._d = self._d[:, :, nx // 2 - 256:nx // 2 + 256]
                else:
                    if ny > 512:
                        self._d = self._d[ny // 2 - 256:ny // 2 + 256, :]
                    if nx > 512:
                        self._d = self._d[:, nx // 2 - 256:nx // 2 + 256]
                # Convert to complex64, make sure it is contiguous
                self._d = fftshift(np.ascontiguousarray(self._d.astype(np.complex64)))

            else:
                if copy_d:
                    self._d = d.copy().astype(np.complex64)
                else:
                    self._d = d  # Assumes d already has the correct type.

        if self._d.ndim == 2:
            ny, nx = self._d.shape
            self._d = self._d.reshape((1, ny, nx))

        self._cl_d = None  # Placeholder for d array in opencl space.
        self._cu_d = None  # Placeholder for d array in cuda space

        # The timestamp counter record when the main wavefront data was last altered, either in the host or the
        # GPU memory
        self._timestamp_counter = 1
        self._cl_timestamp_counter = 0
        self._cu_timestamp_counter = 0

    def get_x_y(self):
        """
        Get 1D arrays of x and y coordinates, taking into account the pixel size. The arrays are centered
        at (0,0) - i.e. with the origin in the corner for FFT puroposes. x is an horizontal vector and y vertical.
        
        :return: a tuple (x, y) of 2D numpy arrays
        """
        ny, nx = self._d.shape[-2:]
        x, y = np.arange(-nx // 2, nx // 2, dtype=np.float32), \
               np.arange(-ny // 2, ny // 2, dtype=np.float32)[:, np.newaxis]
        return fftshift(x) * self.pixel_size, fftshift(y) * self.pixel_size

    def copy(self, copy_d=True):
        """
        Creates a copy (without any reference passing) of this object, unless copy_d is False.
        
        :param copy_d: if False, the new object will be a shallow copy, with d copied as a reference.
        :return: a copy of the object.
        """
        return Wavefront(d=self.get(), z=self.z, pixel_size=self.pixel_size, wavelength=self.wavelength, copy_d=copy_d)

    def get(self, shift=False):
        """
        Get the wavefront data array. This will automatically get the latest data, either from GPU or from the host
        memory, depending where the lat changes were made.
        
        :param shift: if True, the data array will be fft-shifted so that the center of the data is in the center
                      of the array, rather than in the corner (the default).
        :return: the 2D or 3D (stack of wavefront) numpy data array
        """
        if self._timestamp_counter < self._cl_timestamp_counter:
            self._d = self._cl_d.get()
            self._timestamp_counter = self._cl_timestamp_counter
        if self._timestamp_counter < self._cu_timestamp_counter:
            self._d = self._cu_d.get()
            self._timestamp_counter = self._cu_timestamp_counter

        if shift:
            return fftshift(self._d, axes=[-2, -1])
        else:
            return self._d

    def set(self, d, shift=False):
        """
        Set the wavefront data array.

        :param d: the data array (complex64 numpy array)
        :param shift: if True, the data array will be fft-shifted so that the center of the stored data is
                      in the corner of the array (0,0). [default: the array is already shifted]
        :return: nothing
        """
        if shift:
            self._d = fftshift(d, axes=[-2, -1]).astype(np.complex64)
        else:
            self._d = d.astype(np.complex64)
        if self._timestamp_counter <= self._cl_timestamp_counter:
            self._timestamp_counter = self._cl_timestamp_counter + 1
        if self._timestamp_counter <= self._cu_timestamp_counter:
            self._timestamp_counter = self._cu_timestamp_counter + 1

    def __rmul__(self, x):
        """
        Multiply wavefront (by a scalar).

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
        Multiply wavefront (by a scalar).

        This is a placeholder for a function which will be replaced when importing either CUDA or OpenCL operators.
        If called before being replaced, will raise an error

        :param x: the scalar by which the wavefront will be multiplied
        :return:
        """
        #  Needs access to the
        if np.isscalar(x):
            raise OperatorException(
                "ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s. Did you import operators ?" % (str(self), str(x)))
        else:
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s." % (str(self), str(x)))

    def __str__(self):
        return "Wavefront"


class OperatorWavefront(Operator):
    """
    Base class for an operator on Wavefronts, not requiring a processing unit.
    """

    def timestamp_increment(self, w):
        # By default CPU operators should increment the CPU counter. Unless they don't affect the wavefront, like
        # all display operators.
        w._timestamp_counter += 1


class UserWarningWavefrontNearFieldPropagation(UserWarning):
    pass
