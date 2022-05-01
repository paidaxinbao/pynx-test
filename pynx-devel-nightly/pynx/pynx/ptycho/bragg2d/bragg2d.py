# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['Bragg2DPtychoData', 'Bragg2DPtycho', 'OperatorBragg2DPtycho']

import warnings
import numpy as np
from ...operator import Operator, has_attr_not_none
from ...utils.rotation import rotate, rotation_matrix
from ...utils.history import History

warnings.warn("PyNX: you are importing the ptycho.bragg2d module, which is highly unstable (API will vary)."
              "Do not use it unless you know the details of current developments.")


class Bragg2DPtychoData(object):
    """Class for two-dimensional ptychographic data: observed diffraction and probe positions.
    This may include only part of the data from a larger dataset.
    """

    def __init__(self, iobs=None, positions=None, mask=None, wavelength=None, detector=None, rotation=None):
        """
        Init function. The orthonormal coordinate system to be used for the sample position follows the
        NeXus/CXI/McStas convention:
        - z parallel to the incident beam, downstream
        - y: perpendicular to the direct beam and up
        - x: perpendicular to the direct beam and towards the left when seen from the source.

        :param iobs: 3d array with (nb_frame, ny,nx) shape observed intensity (assumed to follow Poisson statistics).
                     Data is assumed to be centered on the detector, and will be fft-shifted to be centered in (0,0).
        :param positions: (x, y, z) tuple or 2d array with ptycho probe positions in meters. The coordinates
                          must be in the laboratory reference frame given above. The positions must be given
                          relative to the center of the modeled sample - in practice they should be centered around 0.
        :param mask: 2D or 3D mask (>0 means masked pixel) for the observed data. Can be None. If 2D, it is assumed
                     that the mask is the same for all frames.
        :param wavelength: wavelength of the experiment, in meters.
        :param detector: {rotation_axes:(('x', 0), ('y', pi/4)), 'pixel_size':55e-6, 'distance':1}:
               parameters for the detector as a dictionary. The rotation axes giving the detector orientation
               will be applied in order, i.e. to find the detector direction, a vector with x=y=0, z=1 is rotated
               by the axes in the given order.
               Mandatory entries are:

               * 'rotation_axes':  a tuple (or list) of tuple, and each tuple entry should be an axis rotation around
                     either x, y or z axis, e.g. ('x', pi/6). Each rotation is applied in order. The value of the
                     rotation can either be given as a floating-point value in radian, or can be an array of values.
               * 'pixel_size' : detector pixel size in meters
               * 'distance' : sample-detector distance in meters

               Optional entries are:

               * 'roi_dx' and 'roi_dy': these are the shift (in pixels) of the center of the cut region-of-interest,
                 for each frame, relative to the position of the direct beam on the detector when at rest
                 (all angles=0). These can also be arrays.

               If any of the parameters are given as an array, its size must correspond to the number of observed frames

        :param rotation: a tuple or list of rotations for the sample, e.g. (('x', 0), ('y', pi/4)), giving the
                         rotation for the sample. The entries can be given exactly as for the detector
                         'rotation_axes' entry, e.g.:
                         rotation_axes:(('y', np.linspace(-np.deg2rad(1),np.deg2rad(1),21)))

                         If several rotation axis are given, they are applied in order to set the sample in the final
                         orientation.

                         If the values are arrays, they must have the same size as the number of frames.
        """
        # Handle only nx=ny for square object pixel size
        assert iobs.shape[-2] == iobs.shape[-1]
        self.iobs = np.fft.fftshift(iobs, axes=(-2, -1)).astype(np.float32)
        # Total nb of photons is used for regularization
        self.iobs_sum = self.iobs.sum()
        if mask is not None:
            if mask.ndim == 2:
                mask = np.repeat(mask[np.newaxis, :, :], iobs.shape[0],
                                 axis=0)  # tile mask to be a 3D array with (nb_frame, ny, nx) shape
            self.mask = np.fft.fftshift(mask.astype(np.int8), axes=(-2, -1))
            self.iobs[self.mask > 0] = -100
        else:
            self.mask = None

        self.wavelength = wavelength
        self.posx, self.posy, self.posz = positions
        # if center_pos:
        #     self.posx -= self.posx.mean()
        #     self.posy -= self.posy.mean()
        #     self.posz -= self.posz.mean()
        self.detector = detector

        # Average orientation for the detector
        self.m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        for ax, angle in detector['rotation_axes']:
            if np.isscalar(angle):
                ang = angle
            else:
                ang = np.array(angle).mean()
            self.m = np.dot(rotation_matrix(ax, ang), self.m)
        self.im = np.linalg.inv(self.m)

        # TODO: take into account ROI shift (roi_dx, roi_dy)
        # Reference (average) scattering vector from detector coordinates
        self.s0 = np.array(rotate(self.m, 0, 1 / wavelength, 0))
        self.s0 -= np.array([0, 1 / wavelength, 0])
        if rotation is not None:
            for ax, angle in rotation:
                if np.isscalar(angle):
                    ang = angle
                else:
                    ang = np.array(angle).mean()
                self.s0 = np.array(rotate(rotation_matrix(ax, -ang), self.s0[0], self.s0[1], self.s0[2]))
        tth = 2 * np.arcsin(np.sqrt((self.s0 ** 2).sum()) * wavelength / 2)
        print("Average scattering vector: s=(%5.3f %5.3f %5.3f)[nm-1] (2theta=%6.2f)" %
              (self.s0[0] * 1e-9, self.s0[1] * 1e-9, self.s0[2] * 1e-9, np.rad2deg(tth)))

        # Relative shift of the scattering vector for each frame (if any)
        nb = len(self.posx)
        self.ds = np.zeros((3, nb))
        sx, sy, sz = np.zeros(nb), np.ones(nb) / wavelength, np.zeros(nb)
        for ax, angle in detector['rotation_axes']:
            sx, sy, sz = rotate(rotation_matrix(ax, angle), sx, sy, sz)
        sy -= 1 / wavelength
        if rotation is not None:
            for ax, angle in rotation:
                sx, sy, sz = rotate(rotation_matrix(ax, -angle), sx, sy, sz)
        self.ds = sx - self.s0[0], sy - self.s0[1], sz - self.s0[2]

        # Calculate ds coordinates in the detector reference frame
        self.ds1 = rotate(self.im, self.ds[0], self.ds[1], self.ds[2])

    def calc_ds(self, dix, diy):
        """
        Compute the shift of the scattering vector corresponding to a shift in pixels on the detector.
        :param dix: shift in pixels on the detector along X (detector reference frame, left to right). Can be a vector.
        :param diy: shift in pixels on the detector along Y (detector reference frame, top to bottom). Can be a vector.
        :return: the scattering vector coordinates (sx, sy, sz) in the laboratory reference frame
        """
        sx0, sy0, sz0 = self.s0  # Mean scattering vector for all frames, in laboratory reference frame
        # Try to shift by 50 pixels
        sxd = sx0 * self.detector['distance'] * self.wavelength
        syd = sy0 * self.detector['distance'] * self.wavelength
        szd = (sz0 + 1 / self.wavelength) * self.detector['distance'] * self.wavelength
        # Change in scattering vector on detector
        sxd1 = sxd + self.detector['pixel_size'] * dix
        syd1 = syd + self.detector['pixel_size'] * diy
        # New scattering vector on Ewald's sphere
        print(sxd1, syd1, szd)
        sd1n = np.sqrt(sxd1 ** 2 + syd1 ** 2 + szd ** 2)
        sx1 = sxd1 / sd1n / self.wavelength
        sy1 = syd / sd1n / self.wavelength
        sz1 = (szd / sd1n - 1) / self.wavelength
        return sx1, sy1, sz1


class Bragg2DPtycho(object):
    """ Class for 2D Bragg ptychography data: object, probe, and observed diffraction.
    This may include only part of the data from a larger dataset
    """

    def __init__(self, probe=None, data=None, support=None, background=None):
        """
        :param probe: the starting estimate of the probe, as a pynx wavefront object - can be 3D if modes are used.
        :param data: the Bragg2DPtychoData object with all observed frames, ptycho positions
        :param support: the support of the object, with values ranging from 0 (outside) to 100 (inside). Intermediate
                        values are used to smooth the object while keeping the array to int8 type. If the array type
                        is bool, True values will be converted to 100.
        """
        self._probe2d = probe  # The original 2D probe as a Wavefront object
        if self._probe2d is not None:
            if self._probe2d.get().ndim == 2:
                ny, nx = self._probe2d.get().shape
                self._probe2d.set(self._probe2d.get().reshape(1, ny, nx))
        self._obj = None
        self.support = support
        if self.support is not None:
            if self.support.dtype is np.bool:
                self.support = self.support.astype(np.int8) * 100
        self.data = data
        self._background = background

        # Matrix transformation from array indices of the array obtained by inverse Fourier Transform
        # to xyz in the laboratory frame. Different from self.data.m, which is a pure rotation matrix
        self.m = None
        # Inverse of self.m
        self.im = None

        # Voxel size of object array, in detector reference frame
        self.pxo = None
        self.pyo = None
        self.pzo = None

        # Stored variables
        # self.scan_area_obj = None
        # self.scan_area_probe = None
        # self.scan_area_points = None
        self.llk_poisson = 0
        self.llk_gaussian = 0
        self.llk_euclidian = 0
        self.nb_photons_calc = 0
        self.nb_obs = self.data.iobs.size
        if self.data.mask is not None:
            self.nb_obs *= (self.data.mask == 0).sum() / float(self.data.mask.size)

        # The timestamp counters record when the data was last altered, either in the host or the GPU memory.
        self._timestamp_counter = 1
        self._cpu_timestamp_counter = 1
        self._cl_timestamp_counter = 0
        self._cu_timestamp_counter = 0
        self.prepare()

        # Record the number of cycles (RAAR, HIO, ER, CF, etc...), which can be used to make some parameters
        # evolve, e.g. for support update
        self.cycle = 0

        # History record
        self.history = History()

    def from_pu(self):
        """
        Get all relevant arrays from processing unit, if necessary
        :return: Nothing
        """
        if self._cpu_timestamp_counter < self._timestamp_counter:
            if self._timestamp_counter == self._cl_timestamp_counter:
                if has_attr_not_none(self, '_cl_obj'):
                    self._obj = self._cl_obj.get()
                if has_attr_not_none(self, '_cl_probe'):
                    self._probe2d.set(self._cl_probe.get(), shift=True)
                if self._background is not None:
                    if has_attr_not_none(self, '_cl_background'):
                        self._background = self._cl_background.get()
            if self._timestamp_counter == self._cu_timestamp_counter:
                if has_attr_not_none(self, '_cu_obj'):
                    self._obj = self._cu_obj.get()
                if has_attr_not_none(self, '_cu_probe'):
                    self._probe2d.set(self._cu_probe.get(), shift=True)
                if self._background is not None:
                    if has_attr_not_none(self, '_cu_background'):
                        self._background = self._cu_background.get()
            self._cpu_timestamp_counter = self._timestamp_counter

    def get_probe(self):
        """
        Get the probe data array. This will automatically get the latest data, either from GPU or from the host
        memory, depending where the last changes were made.

        :return: the 3D numpy data array (nb object modes, nyo, nxo)
        """
        self.from_pu()
        return self._probe2d.get(shift=True)

    def set_probe(self, pr):
        """
        Set the probe data array. This will automatically get the latest data, either from GPU or from the host
        memory, depending where the last changes were made.

        :param pr: the probe array, either 2D or 3D (modes) - will be converted to 3D if necessary. The array is
                   assumed to be centered, and will be fft-shifted so that the probe is centered on the corners.
        """
        self.from_pu()
        if pr.ndim == 2:
            ny, nx = pr.shape
            pr = pr.reshape(1, ny, nx)
        self._probe2d.set(pr, shift=True)
        self._timestamp_counter += 1
        self._cpu_timestamp_counter = self._timestamp_counter

    def get_obj(self):
        """
        Get the object data array. This will automatically get the latest data, either from GPU or from the host
        memory, depending where the last changes were made.

        :return: the 3D numpy data array (nb object modes, nyo, nxo)
        """
        self.from_pu()
        return self._obj

    def set_obj(self, obj):
        """
        Set the object data array. This should either be a 3D array of the correct shape, or a 4D array where
        the first dimension are the object modes.

        :param obj: the object (complex64 numpy array)
        :return: nothing
        """
        if obj.ndim == 3:
            nz, ny, nx = obj.shape
            self._obj = obj.reshape((1, nz, ny, nx)).astype(np.complex64)
        else:
            self._obj = obj.astype(np.complex64)
        self._timestamp_counter += 1
        self._cpu_timestamp_counter = self._timestamp_counter

        # Test if object and support have the same dimensions
        if self.support is not None:
            assert self.support.shape == self._obj.shape[-3:]

    def set_support(self, sup, shrink_object_around_support=True):
        """
        Set the support data array. This should be a 3D array of the correct shape.

        :param sup: the support array. 0 outside support, 1 inside
        :param shrink_object_around_support: if True, will shrink the object and support array around the support
                                             volume. Note that the xyz coordinate may be shifted as a result, if
                                             the support was not centered.
        :return: nothing
        """
        if sup is not None:
            self.support = sup.astype(np.int8)
            if sup.dtype is np.bool:
                self.support *= 100

            self.support_sum = self.support.sum()
            self._timestamp_counter += 1
            self._cpu_timestamp_counter = self._timestamp_counter
            if shrink_object_around_support:
                self.shrink_object_around_support()

    def shrink_object_around_support(self):
        """
        Shrink the object around the tight support, to minimise the 3D object & support volume.
        :return:
        """
        nzo, nyo, nxo = self._obj.shape[-3:]
        z0, z1 = np.nonzero(self.support.sum(axis=(1, 2)))[0].take([0, -1])
        y0, y1 = np.nonzero(self.support.sum(axis=(0, 2)))[0].take([0, -1])
        x0, x1 = np.nonzero(self.support.sum(axis=(0, 1)))[0].take([0, -1])
        self.set_support(self.support[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1], shrink_object_around_support=False)
        if self._obj is not None:
            self.set_obj(self._obj[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1])

    def set_background(self, background):
        """
        Set the incoherent background data array.

        :param background: the background (float32 numpy array)
        :return: nothing
        """
        self._background = background.astype(np.float32)
        self._timestamp_counter += 1
        self._cpu_timestamp_counter = self._timestamp_counter

    def get_background(self, shift=False):
        """
        Get the background data array. This will automatically get the latest data, either from GPU or from the host
        memory, depending where the last changes were made.

        :param shift: if True, the data array will be fft-shifted so that the center of the data is in the center
                      of the array, rather than in the corner (the default).
        :return: the 2D numpy data array
        """
        self.from_pu()
        return self._background

    def prepare(self):
        """
        Calculate projection parameters
        :return:
        """
        self.prepare_orthonormalisation_matrix()
        self.init_obj()
        self.set_support(self.support)

    def get_orthonormalisation_matrix(self, rotation_axes=None):
        """
        Get the orthonormalisation matrix which can transforms the object pixel coordinates to meters in the laboratory
        reference frame. Optionally, add a few rotations to the transformation.
        The orthonormal (laboratory) reference frame is following the NeXus/McStas convention:
        - centered on the object center
        - z is along the direct beam propagation
        - y is vertical, upward,
        - x is horizontal so that the frame is direct, i.e. is >0 towards the left as seen from the source.

        The detector reference frame in object space, also used for the 3D probe and Psi back-propagation is such that:
        - Z is going from the object to the detector
        - X is parallel to the detector horizontal side, X>0 going right as seen from the sample
        - Y is vertical, downward, when the detector is at the origin
        - this corresponds to a detector origin which is 'top, left' as seen from the sample. The object array
        origin is thus 'top, left, source-side' as seen from the X-ray source.

        :param rotation_axes: a tuple of rotation axes, e.g. (('x', 0), ('y', pi/4)), describing the orientation of
                              the detector. The different rotations are applied in order (left-to-right) to transform
                              the pixel coordinates from a detector at origin (in the direct beam), to the coordinates
                              in the orthonormal laboratory reference frame.
        :return: a tuple of (orthonormalisation matrix, pxo, pyo, pzo), where pxo, pyo, pzo are the voxel dimensions in
                 meters
        """
        npos, ny, nx = self.data.iobs.shape
        # Pixel size. Assumes square frames from detector.
        d = self.data.detector['distance']
        lambdaz = self.data.wavelength * d
        p = self.data.detector['pixel_size']
        pyo, pxo = lambdaz / (p * ny), lambdaz / (p * nx)
        # TODO: Find method to estimate optimal pixel size along z. Would depend on probe shape and multiple angles..
        pzo = max(pxo, pyo)
        m = self.data.m.copy()
        if rotation_axes is not None:
            for ax, angle in rotation_axes:
                m = np.dot(rotation_matrix(ax, angle), m)

        # - signs following NeXus convention for axes direction for detector and laboratory reference frames
        m[:, 0] *= -pxo
        m[:, 1] *= -pyo
        m[:, 2] *= pzo
        return m, pxo, pyo, pzo

    def prepare_orthonormalisation_matrix(self):
        """
        Calculate the orthonormalisation matrix to convert probe/object array coordinates (pixel coordinates in the
        detector reference frame) to/from orthonormal ones in the laboratory reference frame.
        This also initialises the voxel sizes in object and probe space
        :return: the orthonormalisation matrix
        """
        self.m, pxo, pyo, pzo = self.get_orthonormalisation_matrix()
        self.im = np.linalg.inv(self.m)
        self.pxo = pxo
        self.pyo = pyo
        self.pzo = pzo

    def init_obj(self):
        """
        Initialize the object array
        :return: nothing. The object is created as an empty array
        """
        nzo, nyo, nxo = self.calc_obj_shape()
        print("Initialised object with %dx%dx%d voxels" % (nzo, nyo, nxo))
        self.set_obj(np.empty((1, nzo, nyo, nxo), dtype=np.complex64))

    def calc_probe_shape(self):
        """
        Calculate the probe shape, given the 2D probe and detector characteristics
        :return: the 3D probe shape (nz, ny, nx)
        """
        ny, nx = self.data.iobs.shape[-2:]
        # The number of points along z is calculated using the intersection of the back-projected 2D wavefront
        # from the detector and the propagated 2D wavefront.
        pixel_size_probe = self._probe2d.pixel_size
        # Extent of the probe along x and y in the laboratory reference frame
        nyp, nxp = self._probe2d.get().shape[-2:]
        dyp, dxp = nyp // 2 * pixel_size_probe, nxp // 2 * pixel_size_probe
        # How far must we extend the back-propagated 2D wavefront along the sample-detector axis to go beyond
        # the probe path ? Use the corners of the projected wavefront to test
        nx2 = nx / 2
        ny2 = ny / 2
        x, y, z = self.xyz_from_obj(np.array([-nx2, -nx2, nx2, nx2]), np.array([-ny2, ny2, -ny2, ny2]), 0)
        d20 = self.data.m[0, 2]
        if abs(d20) < 1e-10:
            d20 = 1e-10
        d21 = self.data.m[1, 2]
        if abs(d21) < 1e-10:
            d21 = 1e-10
        # print("d20", d20, "d21", d21)

        if d20 > 0:
            # corner with lowest x, intersection with upper x probe border: x + alpha*d20 > dxp
            i = x.argmin()
            xc, yc, zc = x[i], y[i], z[i]
            alpha = (dxp - xc) / d20
            # print("alpha_max_x: ", alpha, i)
            izmax_x = self.xyz_to_obj(xc + alpha * self.data.m[0, 2], yc + alpha * self.data.m[1, 2],
                                      zc + alpha * self.data.m[2, 2])[2]

            # corner with highest x, intersection with lower x probe border: x + alpha*d20 < -dxp
            i = x.argmax()
            xc, yc, zc = x[i], y[i], z[i]
            alpha = (-dxp - xc) / d20
            # print("alpha_min_x: ", alpha, i)
            izmin_x = self.xyz_to_obj(xc + alpha * self.data.m[0, 2], yc + alpha * self.data.m[1, 2],
                                      zc + alpha * self.data.m[2, 2])[2]
        else:
            # corner with highest x, intersection with lower x probe border: x + alpha*d20 > -dxp
            i = x.argmax()
            xc, yc, zc = x[i], y[i], z[i]
            alpha = (-dxp - xc) / d20
            # print("alpha_max_x: ", alpha, i)
            izmax_x = self.xyz_to_obj(xc + alpha * self.data.m[0, 2], yc + alpha * self.data.m[1, 2],
                                      zc + alpha * self.data.m[2, 2])[2]

            # corner with lowest x, intersection with higher x probe border: x + alpha*d20 < dxp
            i = x.argmin()
            xc, yc, zc = x[i], y[i], z[i]
            alpha = (dxp - xc) / d20
            # print("alpha_min_x: ", alpha, i)
            izmin_x = self.xyz_to_obj(xc + alpha * self.data.m[0, 2], yc + alpha * self.data.m[1, 2],
                                      zc + alpha * self.data.m[2, 2])[2]

        # Same along y
        if d21 > 0:
            # corner with lowest y, intersection with upper y probe border: y + alpha*d21 > dyp
            i = y.argmin()
            xc, yc, zc = x[i], y[i], z[i]
            alpha = (dyp - yc) / d21
            # print("alpha_max_y: ", alpha, i)
            izmax_y = self.xyz_to_obj(xc + alpha * self.data.m[0, 2], yc + alpha * self.data.m[1, 2],
                                      zc + alpha * self.data.m[2, 2])[2]

            # corner with highest y, intersection with lower y probe border: y + alpha*d21 < -dyp
            i = y.argmax()
            xc, yc, zc = x[i], y[i], z[i]
            alpha = (-dyp - yc) / d21
            # print("alpha_max_y: ", alpha, i)
            izmin_y = self.xyz_to_obj(xc + alpha * self.data.m[0, 2], yc + alpha * self.data.m[1, 2],
                                      zc + alpha * self.data.m[2, 2])[2]
        else:
            # corner with highest y, intersection with lower y probe border: y + alpha*d21 > -dyp
            i = y.argmax()
            xc, yc, zc = x[i], y[i], z[i]
            alpha = (-dyp - yc) / d21
            # print("alpha_max_y: ", alpha, i)
            izmax_y = self.xyz_to_obj(xc + alpha * self.data.m[0, 2], yc + alpha * self.data.m[1, 2],
                                      zc + alpha * self.data.m[2, 2])[2]

            # corner with lowest y, intersection with higher y probe border: y + alpha*d21 < dyp
            i = y.argmin()
            xc, yc, zc = x[i], y[i], z[i]
            alpha = (dyp - yc) / d21
            # print("alpha_min_y: ", alpha, i)
            izmin_y = self.xyz_to_obj(xc + alpha * self.data.m[0, 2], yc + alpha * self.data.m[1, 2],
                                      zc + alpha * self.data.m[2, 2])[2]
        # print("izmin_x: ", izmin_x)
        # print("izmax_x: ", izmax_x)
        # print("izmin_y: ", izmin_y)
        # print("izmax_y: ", izmax_y)
        # Final interval is smallest between x and y
        izmax = min(izmax_x, izmax_y)
        izmin = max(izmin_x, izmin_y)
        nz = izmax - izmin
        print("Calculated probe shape: ", nz, ny, nx)
        return nz, ny, nx

    def calc_obj_shape(self, margin=8, multiple=2):
        """
        Calculate the 3D object shape, given the detector, probe and scan characteristics.
        This must be called after the 3D probe has been initialized. Note that the final object shape will be shrunk
        around the support once it is given.
        :param margin: margin to extend the object area, in case the positions will change (optimization)
        :param multiple: the shape must be a multiple of that number. >=2
        :return: the 3D object shape (nzo, nyo, nxo)
        """
        probe_shape = self.calc_probe_shape()
        ix, iy, iz = self.xyz_to_obj(self.data.posx, self.data.posy, self.data.posz)

        nz = int(2 * (abs(np.ceil(iz)) + 1).max() + probe_shape[0])
        ny = int(2 * (abs(np.ceil(iy)) + 1).max() + probe_shape[1])
        nx = int(2 * (abs(np.ceil(ix)) + 1).max() + probe_shape[2])

        if margin is not None:
            nz += margin
            ny += margin
            nx += margin

        if multiple is not None:
            dz = nz % multiple
            if dz:
                nz += (multiple - dz)
            dy = ny % multiple
            if dy:
                ny += (multiple - dy)
            dx = nx % multiple
            if dx:
                nx += (multiple - dx)

        print("Calculated object shape: ", nz, ny, nx)
        return nz, ny, nx

    def xyz_to_obj(self, x, y, z):
        """
        Convert x,y,z coordinates from the laboratory reference frame to indices in the object array.
        :param x, y, z: laboratory frame coordinates in meters
        :return: (ix, iy, iz) coordinates in the array in the back-projected detector frame
        """
        ix = self.im[0, 0] * x + self.im[0, 1] * y + self.im[0, 2] * z
        iy = self.im[1, 0] * x + self.im[1, 1] * y + self.im[1, 2] * z
        iz = self.im[2, 0] * x + self.im[2, 1] * y + self.im[2, 2] * z
        return ix, iy, iz

    def xyz_from_obj(self, ix, iy, iz):
        """
        Convert x,y,z coordinates to the laboratory reference frame from indices in the 3D object array.
        :param ix, iy, iz: coordinates in the 3D object array.
        :return: (x, y, z) laboratory frame coordinates in meters
        """
        x = self.m[0, 0] * ix + self.m[0, 1] * iy + self.m[0, 2] * iz
        y = self.m[1, 0] * ix + self.m[1, 1] * iy + self.m[1, 2] * iz
        z = self.m[2, 0] * ix + self.m[2, 1] * iy + self.m[2, 2] * iz
        return x, y, z

    def get_xyz(self, rotation=None, domain='object'):
        """
        Get x,y,z orthonormal coordinates corresponding to the object grid.

        :param domain: the domain over which the xyz coordinates should be returned. It should either
                               be 'object' (the default) or the probe, the only difference being that the object
                               is extended to cover all the volume scanned by the shifted probe.
        :param rotation: ('z',np.deg2rad(-20)): optionally, the coordinates can be obtained after a rotation of the
                         object. This is useful if the object or support is to be defined as a parallelepiped, before
                         being rotated to be in diffraction condition. The rotation can be given as a tuple of a
                         rotation axis name (x, y or z) and a counter-clockwise rotation angle in radians.
        :return: a tuple of (x,y,z) coordinates, each a 3D array
        """
        if domain == "probe":
            # TODO: once probe has been calculated, use its shape rather than recalculate it ?
            nz, ny, nx = self.calc_probe_shape()
        elif domain == 'object' or domain == 'obj':
            nz, ny, nx = self._obj.shape[1:]
        else:
            raise Exception("BraggPtycho.get_xyz(): unknown domain '', should be 'object' or 'probe'" % domain)

        iz, iy, ix = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
        x, y, z = self.xyz_from_obj(ix, iy, iz)

        if rotation is not None:
            # TODO: allow multiple axis
            ax, ang = rotation
            c, s = np.cos(ang), np.sin(ang)
            if ax == 'x':
                y, z = c * y - s * z, c * z + s * y
            elif ax == 'y':
                z, x = c * z - s * x, c * x + s * z
            elif ax == 'z':
                x, y = c * x - s * y, c * y + s * x
            else:
                raise Exception("BraggPtycho.get_xyz_obj(): unknown rotation axis '%s'" % ax)

        # Assume the probe is centered on the object grid
        x -= x.mean()
        y -= y.mean()
        z -= z.mean()

        return x, y, z

    def voxel_size_object(self):
        """
        Get the object voxel size
        :return: the voxel size in meters as (pz, py, px)
        """
        tmp = self.xyz_from_obj(1, 0, 0)
        px = np.sqrt(tmp[0] ** 2 + tmp[1] ** 2 + tmp[2] ** 2)
        tmp = self.xyz_from_obj(0, 1, 0)
        py = np.sqrt(tmp[0] ** 2 + tmp[1] ** 2 + tmp[2] ** 2)
        tmp = self.xyz_from_obj(0, 0, 1)
        pz = np.sqrt(tmp[0] ** 2 + tmp[1] ** 2 + tmp[2] ** 2)
        return pz, py, px

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
        if normalized:
            n = 1 / self.nb_obs

        if noise is None:
            return self.llk_poisson * n, self.llk_gaussian * n, self.llk_euclidian * n
        elif 'poiss' in str(noise).lower():
            return self.llk_poisson * n
        elif 'gauss' in str(noise).lower():
            return self.llk_gaussian * n
        elif 'eucl' in str(noise).lower():
            return self.llk_euclidian * n

    def update_history(self, mode='llk', verbose=False, **kwargs):
        """
        Update the history record.
        :param mode: either 'llk' (will record new log-likelihood, nb_photons_calc, average value..),
        or 'algorithm' (will only update the algorithm).
        :param verbose: if True, print some info about current process
        :param kwargs: other parameters to be recorded, e.g. support_threshold=xx, dt=
        :return: nothing
        """
        if mode == 'llk':
            llk = self.get_llk()
            algo = ''
            update_object = True
            update_probe = False
            dt = 0
            if 'algorithm' in kwargs:
                algo = kwargs['algorithm']
            if 'update_object' in kwargs:
                update_object = kwargs['update_object']
            if 'update_probe' in kwargs:
                update_probe = kwargs['update_probe']
            algo = algo_string(algo, self, update_object, update_probe)
            if 'dt' in kwargs:
                dt = kwargs['dt']
            if verbose:
                print("%-8s #%3d LLK= %8.2f(p) %8.2f(g) %8.2f(e), nb photons=%e, dt/cycle=%6.4fs" % (
                    algo, self.cycle, llk[0], llk[1], llk[2], self.nb_photons_calc, dt))
            self.history.insert(self.cycle, llk_poisson=llk[0], llk_gaussian=llk[1],
                                llk_euclidian=llk[2], nb_photons_calc=self.nb_photons_calc,
                                nb_obj=len(self._obj), nb_probe=len(self._probe2d._d), **kwargs)
        elif 'algo' in mode:
            if 'algorithm' in kwargs:
                self.history.insert(self.cycle, algorithm=kwargs['algorithm'])


class OperatorBragg2DPtycho(Operator):
    """
    Base class for an operator on Ptycho2D objects.
    """

    def timestamp_increment(self, p):
        # By default CPU operators increment the CPU counter. Unless they don't affect the pty object, like
        # display operators.
        p._timestamp_counter += 1
        p._cpu_timestamp_counter = p._timestamp_counter


def algo_string(algo_base, p: Bragg2DPtycho, update_object, update_probe, update_background=False, update_pos=False):
    """
    Get a short string for the algorithm being run, e.g. 'DM/o/3p' for difference map with 1 object and 3 probe modes.

    :param algo_base: 'AP' or 'ML' or 'DM'
    :param p: the ptycho object
    :param update_object: True if updating the object
    :param update_probe: True if updating the probe
    :param update_background: True if updating the background
    :param update_pos: True if updating the positions
    :return: a short string for the algorithm
    """
    s = algo_base

    if update_object:
        s += "/"
        if len(p._obj) > 1:
            s += "%d" % (len(p._obj))
        s += "o"

    if update_probe:
        s += "/"
        if len(p._probe2d._d) > 1:
            s += "%d" % (len(p._probe))
        s += "p"

    if update_background:
        s += "/b"

    return s
