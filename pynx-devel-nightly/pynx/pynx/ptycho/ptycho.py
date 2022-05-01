# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


import os
import sys
import time
import timeit
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift, fftfreq
from scipy.spatial import ConvexHull
from scipy.signal import correlate
from skimage.draw import polygon

from ..utils import h5py
from ..operator import Operator, OperatorException, has_attr_not_none
from ..version import get_git_version

_pynx_version = get_git_version()
from ..utils import phase
from .shape import get_center_coord, get_view_coord, calc_obj_shape
from ..utils.history import History


class PtychoData:
    """Class for two-dimensional ptychographic data: observed diffraction and probe positions. 
    This may include only part of the data from a larger dataset.
    """

    def __init__(self, iobs=None, positions=None, detector_distance=None, mask=None,
                 pixel_size_detector=None, wavelength=None, near_field=False, padding=0, vidx=None):
        """
        
        :param iobs: 3d array with (nb_frame, ny,nx) shape observed intensity (assumed to follow Poisson statistics).
                     The frames will be stored fft-shifted so that the center of diffraction lies in the (0,0) corner
                     of each image. The supplied frames should have the diffraction center in the middle of the frames.
                     Intensities must be >=0. Negative values will be used to mark masked pixels.
        :param positions: (x, y, z) tuple or 2-column array with ptycho probe positions in meters.
                          For 2D data, z is ignored and can be None or missing, e.g. with (x, y)
                          The orthonormal coordinate system follows the CXI/NeXus/McStas convention, 
                          with z along the incident beam propagation direction and y vertical towards ceiling.
        :param detector_distance: detector distance in meters
        :param mask: 2D mask (>0 means masked pixel) for the observed data. Can be None. Will be fft-shifted like iobs.
                     Masked pixels are stored as negative values in the iobs array.
        :param pixel_size_detector: in meters, assuming square pixels
        :param wavelength: wavelength of the experiment, in meters.
        :param near_field: True if using near field ptycho
        :param padding: an integer value indicating the number of zero-padded pixels to be used
                        on each side of the observed frames. This can be used for near field ptychography.
                        The input iobs should already padded, the corresponding pixels will be added to the mask.
        :param vidx: array of indices of the positions. This is useful when only a subset of positions is studied,
                     e.g. when the dataset is split between parallel processes.
        """
        self.iobs = None  # full observed intensity array
        self.iobs_sum = None
        self.scale = None  # Vector of scale factors, one for each frame. Used for floating intensities.
        self.padding = padding
        self.vidx = vidx

        if self.padding > 0 and iobs is not None:
            if mask is not None:
                mask[:padding] = 1
                mask[:, :padding] = 1
                mask[-padding:] = 1
                mask[:, -padding:] = 1
            else:
                mask = np.ones(iobs.shape[-2:], dtype=np.int8)
                mask[padding:-padding, padding:-padding] = 0

        if iobs is not None:
            self.iobs = fftshift(iobs, axes=(-2, -1)).astype(np.float32)
            # This should not be necessary
            self.iobs[self.iobs < 0] = 0
            self.iobs_sum = self.iobs.sum()
            if mask is not None:
                self.mask = fftshift(mask.astype(np.int8))
                self.iobs[:, self.mask > 0] = -100
            self.scale = np.ones(len(self.iobs), dtype=np.float32)
        self.detector_distance = float(detector_distance)
        self.pixel_size_detector = float(pixel_size_detector)
        self.wavelength = float(wavelength)
        if positions is not None:
            if len(positions) == 2:
                self.posx, self.posy = (v.copy() for v in positions)
            else:
                self.posx, self.posy, self.posz = (v.copy() for v in positions)
            # Coordinates of center of positions, subtracted to posx and posy
            # TODO: WHY does using mean() make such a difference compared
            #  to min/max center for pynx-simulationpty.py ???
            # self.posx_c = (self.posx.max() + self.posx.min()) / 2
            # self.posy_c = (self.posy.max() + self.posy.min()) / 2
            self.posx_c = self.posx.mean()
            self.posy_c = self.posy.mean()
            self.posx -= self.posx_c
            self.posy -= self.posy_c
            # Keep original positions, in case they are updated.
            self.posx0 = self.posx.copy()
            self.posy0 = self.posy.copy()

        self.near_field = near_field

        # Shifts of the center of diffraction relative to the center of the array
        # This can be used to remove the a phase ramp in the final object
        # These can be calculated from the sub-pixel shift of the center of mass of
        # either the observed or the calculated diffraction patterns.
        self.phase_ramp_dx, self.phase_ramp_dy = 0, 0

    def pixel_size_object(self):
        """
        Get the x and y pixel size in object space after a FFT.
        :return: a tuple (pixel_size_x, pixel_size_y) in meters
        """
        if self.near_field:
            return self.pixel_size_detector, self.pixel_size_detector
        else:
            ny, nx = self.iobs.shape[-2:]
            pixel_size_x = self.wavelength * self.detector_distance / (nx * self.pixel_size_detector)
            pixel_size_y = self.wavelength * self.detector_distance / (ny * self.pixel_size_detector)
            return pixel_size_x, pixel_size_y

    def get_required_obj_shape(self, margin=16):
        """ Estimate the required object shape

        :param margin: number of pixels on the border to avoid stepping out of the object,
                       e.g. when refining positions.
        :return: (nyo, nxo), the 2D object shape
        """
        px, py = self.pixel_size_object()
        return calc_obj_shape(self.posx / px, self.posy / py, self.iobs.shape[-2:], margin=margin)


class Ptycho:
    """
    Class for two-dimensional ptychographic data: object, probe, and observed diffraction.
    This may include only part of the data from a larger dataset.
    """

    def __init__(self, probe=None, obj=None, background=None, data=None, nb_frame_total=None):
        """

        :param probe: the starting estimate of the probe, as a complex 2D numpy array - can be 3D if modes are used.
                      the probe should be centered in the center of the array.
        :param obj: the starting estimate of the object, as a complex 2D numpy array - can be 3D if modes are used. 
        :param background: 2D array with the incoherent background.
        :param data: the PtychoData object with all observed frames, ptycho positions
        :param nb_frame_total: total number of frames (used for normalization)
        """
        self.data = data
        self._probe = probe
        self._obj = obj
        self._background = background
        self._obj_zero_phase_mask = None
        # Use interpolation (currently bilinear, washes out details) ? Experimental.
        self._interpolation = False

        if self._probe is not None:
            self._probe = self._probe.astype(np.complex64)
            if self._probe.ndim == 2:
                ny, nx = self._probe.shape
                self._probe = self._probe.reshape((1, ny, nx))

        if self._obj is not None:
            self._obj = self._obj.astype(np.complex64)
            if self._obj.ndim == 2:
                ny, nx = self._obj.shape
                self._obj = self._obj.reshape((1, ny, nx))
        elif self.data is not None:
            nyo, nxo = self.data.get_required_obj_shape(margin=8)
            self._obj = np.ones((1, nyo, nxo), dtype=np.complex64)

        if self._background is not None:
            self._background = fftshift(self._background.astype(np.float32))
        elif data is not None:
            if data.iobs is not None:
                self._background = np.zeros(data.iobs.shape[-2:], dtype=np.float32)

        self.nb_frame_total = nb_frame_total
        if self.nb_frame_total is None and data is not None:
            self.nb_frame_total = len(data.iobs)

        # Placeholder for storage of propagated wavefronts. Only used with CPU
        self._psi = None

        # Stored variables
        if data is not None:
            self.pixel_size_object = np.float32(data.pixel_size_object()[0])
        self._scan_area_obj = None
        self._scan_area_probe = None
        self._scan_area_points = None
        self.llk_poisson = 0
        self.llk_gaussian = 0
        self.llk_euclidian = 0
        self.nb_photons_calc = 0
        if data is not None:
            self.nb_obs = (self.data.iobs >= 0).sum()
        else:
            self.nb_obs = 0

        # The timestamp counter record when the data was last altered, either in the host or the GPU memory.
        self._timestamp_counter = 1
        self._cpu_timestamp_counter = 1
        self._cl_timestamp_counter = 0
        self._cu_timestamp_counter = 0

        if self.data is not None:
            self.calc_scan_area()

        # Regularisation scale factors
        self.reg_fac_scale_obj = 0
        self.reg_fac_scale_probe = 0
        self.calc_regularisation_scale()

        # Record the number of cycles (ML, AP, DM, etc...), for history purposes
        self.cycle = 0
        # History record
        self.history = History()

        # Used with MPI, to mute output from non-master processes
        self.mpi_master = True

        self._obj_illumination = None

    def print(self, *args, **kwargs):
        """
        Print function which can be muted e.g. for non-master MPI processes
        :param args: arguments passed to the print function
        :param kwargs: keyword arguments passed to the print function
        :return: nothing
        """
        if self.mpi_master:
            print(*args, **kwargs)

    def reset_history(self):
        """
        Reset history and set cycle to 0
        :return:
        """
        self.history = History()
        self.cycle = 0

    # def init_obj_probe_mask(self):
    def calc_scan_area(self):
        """
        Compute the scan area for the object and probe, using scipy ConvexHull. The scan area for the object is
        augmented by twice the average distance between scan positions for a more realistic estimation.
        scan_area_points is also computed, corresponding to the outline of the scanned area.

        :return: Nothing. self.scan_area_probe and self.scan_area_obj are updated, as 2D arrays with the same shape 
                 as the object and probe, with False outside the scan area and True inside.
        """
        px, py = self.data.pixel_size_object()
        # These coordinates are centered on the object array
        y, x = self.data.posy, self.data.posx
        # If there are too many points, reduce to 500.
        if len(x) > 1000:
            x = x[::len(x) // 500]
            y = y[::len(y) // 500]

        # Convert x, y metric to pixel coordinates relative to the origin (top, left) corner
        nyo, nxo = self._obj.shape[-2:]
        if self.data.near_field:
            # Assume a full illumination
            vx = x / px + nxo // 2
            vy = y / py + nyo // 2
            ny, nx = self.data.iobs.shape[-2:]
            ny, nx = ny - self.data.padding, nx - self.data.padding
            vx = np.concatenate((vx - nx / 2, vx + nx / 2, vx + nx / 2, vx - nx / 2))
            vy = np.concatenate((vy + ny / 2, vy + ny / 2, vy - ny / 2, vy - ny / 2))
            points = np.array([(x, y) for x, y in zip(vx, vy)])
        else:
            points = np.array([(x / px + nxo // 2, y / py + nyo // 2) for x, y in zip(x, y)])

        c = ConvexHull(points)
        vx = np.array([points[i, 0] for i in c.vertices])  # + [points[c.vertices[0], 0]], dtype=np.float32)
        vy = np.array([points[i, 1] for i in c.vertices])  # + [points[c.vertices[0], 1]], dtype=np.float32)
        if not self.data.near_field:
            # Try to expand scan area by the average distance between points
            try:
                # Estimated average distance between points with an hexagonal model
                w = 4 / 3 / np.sqrt(3) * np.sqrt(c.volume / x.size)
                xc = vx.mean()
                yc = vy.mean()
                # Expand scan area from center by 1
                d = np.sqrt((vx - xc) ** 2 + (vy - yc) ** 2)
                vx = xc + (vx - xc) * (d + w) / d
                vy = yc + (vy - yc) * (d + w) / d
            except:
                # c.volume only supported in scipy >=0.17 (2016/02)
                pass
        # print("calc_scan_area: scan area = %8g pixels^2, center @(%6.1f, %6.1f), <d>=%6.2f)"%(c.volume, xc, yc, w))
        # Object

        rr, cc = polygon(vy, vx, (nyo, nxo))
        self._scan_area_obj = np.zeros((nyo, nxo), dtype=np.bool)
        self._scan_area_obj[rr, cc] = True

        # scan_area_points are stored relative to the center of the object
        self._scan_area_points = vx - nxo // 2, vy - nyo // 2

        if self.data.near_field:
            self._scan_area_probe = np.ones(self._probe.shape[-2:], dtype=np.bool)
        else:
            # scan_area_probe is obtained from the auto-convolution of scan_area_obj
            tmp = self._scan_area_obj.astype(np.float32)
            tmp = correlate(tmp, np.flip(tmp), mode='same') >= 1
            ny, nx = self._probe.shape[-2:]
            self._scan_area_probe = \
                tmp[(nyo - ny) // 2:(nyo + ny) // 2, (nxo - nx) // 2:(nxo + nx) // 2].astype(np.bool)

    def get_scan_area_obj(self):
        """
        Return the scan_area_obj (2D array with the object shape, True inside the
        area scanned, and False outside). It is computed if necessary.
        :return: self.scan_area_obj
        """
        if self._scan_area_obj is None:
            self.calc_scan_area()
        return self._scan_area_obj

    def get_scan_area_points(self):
        """
        Return the scan_area_points (outside polygon points delimiting the scan area).
        It is computed if necessary.
        :return: self.scan_area_points
        """
        if self._scan_area_obj is None:
            self.calc_scan_area()
        return self._scan_area_points

    def get_scan_area_probe(self):
        """
        Return the scan_area_probe (2D array with the probe shape, True inside the
        area scanned, and False outside). It is computed if necessary.
        :return: self.scan_area_probe
        """
        if self._scan_area_obj is None:
            self.calc_scan_area()
        return self._scan_area_probe

    def calc_regularisation_scale(self):
        """
        Calculate the scale factor for object and probe regularisation.
        Calculated according to Thibault & Guizar-Sicairos 2012
        :return: nothing
        """
        if self.data is not None and self._obj is not None and self._probe is not None:
            probe_size = self._probe[0].size
            obj_size = self._obj[0].size
            data_size = self.data.iobs.size
            nb_photons = self.nb_obs
            # TODO: take into account the object area actually scanned
            self.reg_fac_scale_obj = data_size * nb_photons / (8 * obj_size ** 2)
            self.reg_fac_scale_probe = data_size * nb_photons / (8 * probe_size ** 2)
            if False:
                self.print("Regularisation scale factors: object %8e probe %8e" % (self.reg_fac_scale_obj,
                                                                                   self.reg_fac_scale_probe))
        else:
            self.reg_fac_scale_obj = 0
            self.reg_fac_scale_probe = 0

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
                    self._probe = self._cl_probe.get()
                if has_attr_not_none(self, '_cl_scale'):
                    self.data.scale = self._cl_scale.get()
                if self._background is not None:
                    if has_attr_not_none(self, '_cl_background'):
                        self._background = self._cl_background.get()
                # Get back positions if they have been updated
                px, py = self.data.pixel_size_object()
                obj_shape = self._obj.shape[-2:]
                probe_shape = self._probe.shape[-2:]
                for v in self._cl_obs_v:
                    vx = v.cl_x.get()
                    vy = v.cl_y.get()
                    for ii in range(v.npsi):
                        x, y = get_center_coord(obj_shape, probe_shape, vx[ii], vy[ii], px, py)
                        if False:
                            x0, y0 = self.data.posx[v.i + ii], self.data.posy[v.i + ii]
                            self.print("Pos #%3d:  x=%6.1f dx=%6.1f  y=%6.1f dy=%6.1f" %
                                       (v.i + ii, x0 * 1e9, (x - x0) * 1e9, y0 * 1e9, (y - y0) * 1e9))
                        self.data.posx[v.i + ii] = x
                        self.data.posy[v.i + ii] = y

            if self._timestamp_counter == self._cu_timestamp_counter:
                if has_attr_not_none(self, '_cu_obj'):
                    self._obj = self._cu_obj.get()
                if has_attr_not_none(self, '_cu_probe'):
                    self._probe = self._cu_probe.get()
                if has_attr_not_none(self, '_cu_scale'):
                    self.data.scale = self._cu_scale.get()
                if self._background is not None:
                    if has_attr_not_none(self, '_cu_background'):
                        self._background = self._cu_background.get()
                # Get back positions if they have been updated
                px, py = self.data.pixel_size_object()
                obj_shape = self._obj.shape[-2:]
                probe_shape = self._probe.shape[-2:]
                vx = self._cu_cx.get()
                vy = self._cu_cy.get()
                for i in range(len(vx)):
                    x, y = get_center_coord(obj_shape, probe_shape, vx[i], vy[i], px, py)
                    if False:
                        x0, y0 = self.data.posx[i], self.data.posy[i]
                        self.print("Pos #%3d:  x=%6.1f dx=%6.1f  y=%6.1f dy=%6.1f" %
                                   (i, x0 * 1e9, (x - x0) * 1e9, y0 * 1e9, (y - y0) * 1e9))
                    self.data.posx[i] = x
                    self.data.posy[i] = y
            self._cpu_timestamp_counter = self._timestamp_counter

    def get_obj(self, remove_obj_phase_ramp=False):
        """
        Get the object data array. This will automatically get the latest data, either from GPU or from the host
        memory, depending where the last changes were made.

        :param remove_obj_phase_ramp: if True, the object will be returned after removing the phase
                                      ramp coming from the imperfect centring of the diffraction data
                                      (sub-pixel shift). Calculated diffraction patterns using such a
                                      corrected object will present a sub-pixel shift relative to the
                                      diffraction data. The ramp information comes from the PtychoData
                                      phase_ramp_d{x,y} attributes, and should have been calculated
                                      beforehand using a ZroPhaseRamp(obj=True) operator.
        :return: the 3D numpy data array (nb object modes, nyo, nxo)
        """
        self.from_pu()
        obj = self._obj
        if remove_obj_phase_ramp and (abs(self.data.phase_ramp_dx) + abs(self.data.phase_ramp_dx)) > 1e-5:
            nz, ny, nx = self._probe.shape
            nyo, nxo = self._obj.shape[-2:]
            y, x = np.meshgrid(fftshift(fftfreq(nyo, d=ny / nyo)).astype(np.float32),
                               fftshift(fftfreq(nxo, d=nx / nxo)).astype(np.float32), indexing='ij')
            obj = obj * np.exp(-2j * np.pi * (x * self.data.phase_ramp_dx + y * self.data.phase_ramp_dy))
        return obj

    def set_obj(self, obj):
        """
        Set the object data array.

        :param obj: the object (complex64 numpy array)
        :return: nothing
        """
        self.from_pu()
        self._obj = obj.astype(np.complex64)
        if self._obj.ndim == 2:
            ny, nx = self._obj.shape
            self._obj = self._obj.reshape((1, ny, nx))
        self._timestamp_counter += 1
        self._cpu_timestamp_counter = self._timestamp_counter
        self.calc_regularisation_scale()

    def get_illumination_obj(self):
        """ Get the sum of the probe intensity for all illuminations,
        which is used for object normalisation.

        :return: the array of the illumination norm, with the same 2D shape as the object
        """
        # TODO: use a timestamp for the calculation ?
        from .operator import CalcIllumination
        CalcIllumination() * self
        return self._obj_illumination

    def set_obj_zero_phase_mask(self, mask):
        """
        Set an object mask, which has the same 2D shape as the object, where values of 1 indicate that the area
        corresponds to vacuum (or air), and 0 corresponds to some material. Values between 0 and 1 can be given to
        smooth the transition.
        This mask will be used to restrain the corresponding area to a null phase, dampening the imaginary part
        at every object update.
        :param mask: a floating-point array with the same 2D shape as the object, where values of 1 indicate
        that the area corresponds to vacuum (or air), and 0 corresponds to the sample.
        :return: nothing
        """
        self._obj_zero_phase_mask = mask

    def get_probe(self):
        """
        Get the probe data array. This will automatically get the latest data, either from GPU or from the host
        memory, depending where the last changes were made.

        :return: the 3D probe numpy data array
        """
        self.from_pu()

        return self._probe

    def get_obj_coord(self):
        """
        Get the object coordinates
        :return: a tuple of two arrays corresponding to the x (columns) and y (rows coordinates)
        """
        ny, nx = self._obj.shape[-2:]
        px, py = self.data.pixel_size_object()
        yc = (np.arange(ny, dtype=np.float32) * py + self.data.posy_c - ny * py / 2)
        xc = (np.arange(nx, dtype=np.float32) * px + self.data.posx_c - nx * px / 2)
        return xc, yc

    def get_probe_coord(self):
        """
        Get the probe coordinates
        :return: a tuple of two arrays corresponding to the x (columns) and y (rows coordinates)
        """
        ny, nx = self._probe.shape[-2:]
        px, py = self.data.pixel_size_object()
        yc = (np.arange(ny, dtype=np.float32) * py - ny * py / 2)
        xc = (np.arange(nx, dtype=np.float32) * px - nx * px / 2)
        return xc, yc

    def set_background(self, background):
        """
        Set the incoherent background data array.
        It will be shifted so that the center of the diffraction image is at (0,0),
        like the stored intensity.

        :param background: the background (float32 numpy array)
        :return: nothing
        """
        self.from_pu()
        self._background = fftshift(background.astype(np.float32))
        self._timestamp_counter += 1
        self._cpu_timestamp_counter = self._timestamp_counter

    def get_background(self):
        """
        Get the background data array. This will automatically get the latest data, either from GPU or from the host
        memory, depending where the last changes were made.

        :return: the 2D numpy data array
        """
        self.from_pu()
        if self._background is not None:
            return fftshift(self._background)
        return None

    def set_probe(self, probe):
        """
        Set the probe data array.

        :param probe: the probe (complex64 numpy array)
        :return: nothing
        """
        self.from_pu()
        self._probe = probe.astype(np.complex64)
        if self._probe.ndim == 2:
            ny, nx = self._probe.shape
            self._probe = self._probe.reshape((1, ny, nx))
        self._timestamp_counter += 1
        self._cpu_timestamp_counter = self._timestamp_counter

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

    def load_obj_probe_cxi(self, filename, entry=None, verbose=True):
        """
        Load object and probe from a CXI file, result of a previous optimisation. If no data is already present in
        the current object, then the pixel size and energy/wavelength are also loaded, and a dummy (one frame) data
        object is created.

        :param filename: the CXI filename from which to load the data
        :param entry: the entry to be read. By default, the last in the file is loaded. Can be 'entry_1', 'entry_10'...
                      An integer n can also be supplied, in which case 'entry_%d' % n will be read
        :return:
        """
        f = h5py.File(filename, 'r')
        if entry is None:
            i = 1
            while True:
                if 'entry_%d' % i not in f:
                    break
                i += 1
            entry = f["entry_%d" % (i - 1)]
        elif isinstance(entry, int):
            entry = f["entry_%d" % entry]
        else:
            entry = f[entry]

        self.entry = entry  # Debug

        # Object and probe are flipped when saved to keep origin at top, left
        self.set_obj(np.flip(entry['object/data'][()], axis=-2))
        if verbose:
            self.print("CXI: Loaded object with shape: ", self.get_obj().shape)

        self.set_probe(np.flip(entry['probe/data'][()], axis=-2))
        if verbose:
            self.print("CXI: Loaded probe with shape: ", self.get_obj().shape)
        pixel_size_obj = (np.float32(entry['probe/x_pixel_size'][()])
                          + np.float32(entry['probe/y_pixel_size'][()])) / 2
        if verbose:
            self.print("CXI: object pixel size (m): ", pixel_size_obj)

        if 'mask' in entry['result_1']:
            self.scan_area_obj = np.flipud(entry['result_1/mask'][()] > 0)
            if verbose:
                self.print("CXI: Loaded scan_area_obj")

        if 'mask' in entry['result_2']:
            self.scan_area_probe = np.flipud(entry['result_2/mask'][()] > 0)
            if verbose:
                self.print("CXI: Loaded scan_area_probe")

        if 'background' in entry:
            self.set_background(entry['background/data'][()])
            if verbose:
                self.print("CXI: Loaded background")

        if self.data is None:
            ny, nx = self.get_probe().shape[-2:]
            d = np.zeros((1, ny, nx), dtype=np.float32)
            nrj = np.float32(entry['instrument_1/source_1/energy'][()])
            wavelength = 12.3984 / (nrj / 1.60218e-16) * 1e-10
            detector_distance = np.float32(entry['instrument_1/detector_1/distance'][()])
            x_pixel_size = np.float32(entry['instrument_1/detector_1/x_pixel_size'][()])
            y_pixel_size = np.float32(entry['instrument_1/detector_1/y_pixel_size'][()])
            pxy = (x_pixel_size + y_pixel_size) / 2

            # Check consistency of energy values (bug in versions priors to git 2018-10-12)
            px_obj = wavelength * detector_distance / (nx * pxy)
            if pixel_size_obj / px_obj > 1e9:
                self.print("Correcting for incorrect energy stored in file (energy 1e10 too large)")
                nrj *= 1e-10
                wavelength *= 1e10
            if verbose:
                self.print("CXI: wavelength (m): ", wavelength)
                self.print("CXI: detector pixel size (m): ", pxy)
                self.print("CXI: detector distance (m): ", detector_distance)
                self.print("CXI: created a dummy iobs with only one frame")

            self.data = PtychoData(iobs=d, positions=([0], [0], [0]), detector_distance=detector_distance,
                                   pixel_size_detector=pxy, wavelength=wavelength)
            self.pixel_size_object = np.float32(self.data.pixel_size_object()[0])
            self.nb_frame_total = 1
        self.calc_regularisation_scale()

    def save_obj_probe_cxi(self, filename, sample_name=None, experiment_id=None, instrument=None, note=None,
                           process=None, append=False, shift_phase_zero=False, params=None,
                           remove_obj_phase_ramp=False):
        """
        Save the result of the optimisation (object, probe, scan areas) to an HDF5 CXI-like file.
        
        :param filename: the file name to save the data to
        :param sample_name: optional, the sample name
        :param experiment_id: the string identifying the experiment, e.g.: 'HC1234: Siemens star calibration tests'
        :param instrument: the string identifying the instrument, e.g.: 'ESRF id10'
        :param note: a string with a text note giving some additional information about the data, a publication...
        :param process: a dictionary of strings which will be saved in '/entry_N/data_1/process_1'. A dictionary entry
                        can also be a 'note' as keyword and a dictionary as value - all key/values will then be saved
                        as separate notes. Example: process={'program': 'PyNX', 'note':{'llk':1.056, 'nb_photons': 1e8}}
        :param append: by default (append=False), any existing file will be overwritten, and the result will be saved
                       as 'entry_1'. If append==True and the file exists, a new entry_N will be saved instead.
                       This can be used to export the different steps of an optimisation.
        :param shift_phase_zero: if True, remove the linear phase ramp from the object
        :param params: a dictionary of parameters to be saved into process_1/configuration NXcollection
        :param remove_obj_phase_ramp: if True, the object will be saved after removing the phase
                                      ramp coming from the imperfect centring of the diffraction data
                                      (sub-pixel shift). Calculated diffraction patterns using such a
                                      corrected object will present a sub-pixel shift relative to the
                                      diffraction data. The ramp information comes from the PtychoData
                                      phase_ramp_d{x,y} attributes, and are not re-calculated.
        :return: Nothing. a CXI file is created
        """
        obj = self.get_obj(remove_obj_phase_ramp=remove_obj_phase_ramp)
        save_obj_probe_cxi(filename, obj, self.get_probe(), self.data.wavelength,
                           self.data.detector_distance, self.data.pixel_size_detector, self.llk_poisson / self.nb_obs,
                           self.llk_gaussian / self.nb_obs, self.llk_euclidian / self.nb_obs, self.nb_photons_calc,
                           self.history, self.data.pixel_size_object(), (self.data.posx, self.data.posy),
                           (self.data.posx_c, self.data.posy_c), (self.data.posx0, self.data.posy0),
                           scale=self.data.scale, obj_zero_phase_mask=self._obj_zero_phase_mask,
                           scan_area_obj=self.get_scan_area_obj(), scan_area_probe=self.get_scan_area_probe(),
                           obj_illumination=self.get_illumination_obj(),
                           background=self.get_background(), sample_name=sample_name, experiment_id=experiment_id,
                           instrument=instrument, note=note, process=process, append=append,
                           shift_phase_zero=shift_phase_zero, params=params,
                           obj_phase_ramp=(self.data.phase_ramp_dx, self.data.phase_ramp_dy))

    def reset_history(self):
        """
        Reset history, and set current cycle to zero
        :return: nothing
        """
        self.history = History()
        self.cycle = 0

    def get_llk(self, noise=None, norm=True):
        """ Get the log-likelihood.

        :param noise: noise model, either 'poisson', 'gaussian' or 'euclidian'.
                      If None, a dictionary is returned.
        :param norm: if True (the default), the LLK values are normalised
        :return: either a single LLK value, or a dictionary
        """
        p = self.llk_poisson
        g = self.llk_gaussian
        e = self.llk_euclidian
        n = self.nb_obs
        nph = self.nb_photons_calc
        if norm:
            p, g, e = p / n, g / n, e / n
        if noise is None:
            return {'poisson': p, 'gaussian': g, 'euclidian': e, 'nb_photons_calc': nph, 'nb_obs': n}
        if 'poiss' in noise.lower():
            return p
        if 'gauss' in noise.lower():
            return g
        return e

    def update_history(self, mode='llk', update_obj=False, update_probe=False, update_background=False,
                       update_pos=False, verbose=False, **kwargs):
        """ Update the history record.

        :param mode: either 'llk' (will record new log-likelihood and number of photons)
                     or 'algorithm' (will only update the algorithm) - for the latter case, algorithm
                     should be given as a keyword argument.
        :param verbose: if True, print some info about current process (only if mode=='llk')
        :param kwargs: other parameters to be recorded, e.g. probe_inertia=xx, algorithm='DM'
        :return: nothing
        """
        if mode == 'llk':
            d = self.get_llk(noise=None, norm=True)
            p, g, e, nph = d['poisson'], d['gaussian'], d['euclidian'], d['nb_photons_calc']
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
                s = algo_string(algo, self, update_obj, update_probe, update_background, update_pos)
                self.print("%-10s #%3d LLK= %8.2f(p) %8.2f(g) %8.2f(e), nb photons=%e, dt/cycle=%5.3fs"
                           % (s, self.cycle, p, g, e, nph, dt))

            self.history.insert(self.cycle, llk_poisson=p, llk_gaussian=g, llk_euclidian=e,
                                nb_photons_calc=nph, nb_obj=len(self._obj),
                                nb_probe=len(self._probe), **kwargs)
        elif 'algo' in mode:
            if 'algorithm' in kwargs:
                self.history.insert(self.cycle, algorithm=kwargs['algorithm'])


def save_ptycho_data_cxi(file_name, iobs, pixel_size, wavelength, detector_distance, x, y, z=None, monitor=None,
                         mask=None, dark=None, instrument="", overwrite=False, scan=None, params=None,
                         verbose=False, **kwargs):
    """
    Save the Ptychography scan data using the CXI format (see http://cxidb.org)

    :param file_name: the file name (including relative or full path) to save the data to
    :param iobs: the observed intensity, with shape (nb_frame, ny, nx)
    :param pixel_size: the detector pixel size in meters
    :param wavelength: the experiment wavelength
    :param x: the x scan positions
    :param y: the y scan positions
    :param z: the z scan positions (default=None)
    :param monitor: the monitor
    :param mask: the mask for the observed frames
    :param dark: the incoherent background
    :param instrument: a string with the name of the instrument (e.g. 'ESRF id16A')
    :param overwrite: if True, will overwrite an existing file
    :param params: a dictionary of parameters which will be saved as a NXcollection
    :param verbose: if True, print something.
    :return:
    """
    path = os.path.split(file_name)[0]
    if len(path):
        os.makedirs(path, exist_ok=True)
    if os.path.isfile(file_name) and overwrite is False:
        if verbose:
            print("CXI file already exists, not overwriting: ", file_name)
        os.system('ls -la %s' % file_name)
        return
    elif verbose:
        print('Creating CXI file: %s' % file_name)

    f = h5py.File(file_name, "w")
    f.attrs['file_name'] = file_name
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

    sample_1 = entry_1.create_group("sample_1")
    sample_1.attrs['NX_class'] = 'NXsample'

    geometry_1 = sample_1.create_group("geometry_1")
    sample_1.attrs['NX_class'] = 'NXgeometry'  # Deprecated NeXus class, move to NXtransformations
    xyz = np.zeros((3, x.size), dtype=np.float32)
    xyz[0] = x
    xyz[1] = y
    geometry_1.create_dataset("translation", data=xyz)

    data_1 = entry_1.create_group("data_1")
    data_1.attrs['NX_class'] = 'NXdata'
    data_1.attrs['signal'] = 'data'
    data_1.attrs['interpretation'] = 'image'
    data_1["translation"] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
    if monitor is not None:
        data_1.create_dataset("monitor", monitor)

    instrument_1 = entry_1.create_group("instrument_1")
    instrument_1.attrs['NX_class'] = 'NXinstrument'
    if instrument is not None:
        instrument_1.create_dataset("name", data=instrument)

    source_1 = instrument_1.create_group("source_1")
    source_1.attrs['NX_class'] = 'NXsource'
    nrj = 12384e-10 / wavelength * 1.60218e-19
    source_1.create_dataset("energy", data=nrj)  # in J
    source_1["energy"].attrs['note'] = 'Incident photon energy (instead of source energy), for CXI compatibility'

    detector_1 = instrument_1.create_group("detector_1")
    detector_1.attrs['NX_class'] = 'NX_detector'

    nz, ny, nx = iobs.shape
    detector_1.create_dataset("data", data=iobs, chunks=(1, ny, nx), shuffle=True,
                              compression="gzip")
    detector_1.create_dataset("distance", data=detector_distance)
    detector_1["distance"].attrs['units'] = 'm'
    detector_1.create_dataset("x_pixel_size", data=pixel_size)
    detector_1["x_pixel_size"].attrs['units'] = 'm'
    detector_1.create_dataset("y_pixel_size", data=pixel_size)
    detector_1["y_pixel_size"].attrs['units'] = 'm'
    if mask is not None:
        if mask.sum() != 0:
            detector_1.create_dataset("mask", data=mask, chunks=True, shuffle=True, compression="gzip")
            detector_1["mask"].attrs['note'] = "Mask of invalid pixels, applying to each frame"
    if dark is not None:
        if dark.sum() != 0:
            detector_1.create_dataset("dark", data=dark, chunks=True, shuffle=True, compression="gzip")
            detector_1["dark"].attrs['note'] = "Incoherent background (dark)"
    # Basis vector - this is the default CXI convention, so could be skipped
    # This corresponds to a 'top, left' origin convention
    basis_vectors = np.zeros((2, 3), dtype=np.float32)
    basis_vectors[0, 1] = -pixel_size
    basis_vectors[1, 0] = -pixel_size
    detector_1.create_dataset("basis_vectors", data=basis_vectors)

    detector_1["translation"] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
    data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')

    # Remember how import was done
    command = ""
    for arg in sys.argv:
        command += arg + " "
    process_1 = data_1.create_group("process_1")
    process_1.attrs['NX_class'] = 'NXprocess'
    process_1.create_dataset("program", data='PyNX')  # NeXus spec
    process_1.create_dataset("version", data="%s" % _pynx_version)  # NeXus spec
    process_1.create_dataset("command", data=command)  # CXI spec
    config = process_1.create_group("configuration")
    config.attrs['NX_class'] = 'NXcollection'
    if params is not None:
        for k, v in params.items():
            if k == 'scan' and scan is not None:
                continue
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
    if scan is not None:
        config.create_dataset('scan', data=scan)

    f.close()


def save_obj_probe_cxi(filename, obj, probe, wavelength, detector_distance, pixel_size_detector, llk_poisson,
                       llk_gaussian, llk_euclidian, nb_photons_calc, history,
                       pixel_size_object, posxy, posxy_c, posxy0, scale=None, obj_zero_phase_mask=None,
                       scan_area_obj=None, scan_area_probe=None, obj_illumination=None, background=None,
                       sample_name=None, experiment_id=None, instrument=None, note=None,
                       process=None, append=False, shift_phase_zero=False, params=None, extra_data=None,
                       obj_phase_ramp=None):
    """ Save the result of the optimisation (object, probe, scan areas) to an HDF5 CXI-like file.

    Note that object and probed are flipped (up/down) to have a (top, left) array origin.

    :param filename: the file name to save the data to
    :param obj: the object to save
    :param probe: the probe to save
    :param wavelength: the wavelength (SI unit)
    :param detector_distance: detector distance
    :param pixel_size_detector: the detector's pixel size
    :param llk_poisson, llk_gaussian, llk_euclidian: normalised log-likelihood values
    :param pixel_size_object: the object pixel size
    :param posxy: the final scanning positions (SI units)
    :param posxy0: the initial scanning positions (SI units)
    :param posxy_c: xy coordinates of the object center
    :param scale: array of per-frame scaling factir
    :param obj_zero_phase_mask: the area used for the object phase optimisation (maybe obsolete..)
    :param scan_area_obj: scan area mask on the object array
    :param scan_area_probe: scan area mask on the probe array
    :param obj_illumination: integrated incident intensity on the object area
    :param background: incoherent background
    :param sample_name: optional, the sample name
    :param experiment_id: the string identifying the experiment, e.g.: 'HC1234: Siemens star calibration tests'
    :param instrument: the string identifying the instrument, e.g.: 'ESRF id10'
    :param note: a string with a text note giving some additional information about the data, a publication...
    :param process: a dictionary of strings which will be saved in '/entry_N/data_1/process_1'. A dictionary entry
                    can also be a 'note' as keyword and a dictionary as value - all key/values will then be saved
                    as separate notes. Example: process={'program': 'PyNX', 'note':{'llk':1.056, 'nb_photons': 1e8}}
    :param append: by default (append=False), any existing file will be overwritten, and the result will be saved
                   as 'entry_1'. If append==True and the file exists, a new entry_N will be saved instead.
                   This can be used to export the different steps of an optimisation.
    :param shift_phase_zero: if True, centre the object phase around zero
    :param params: a dictionary of parameters to be saved into process_1/configuration NXcollection
    :param extra_data: a dictionary of data which will be saved as entry/_extra_data, and may be useful for debugging.
                       each value may itself be a dictionary of values to save
    :param obj_phase_ramp: the shifts (dx, dy) of the average calculated intensity from the array centre
    :return: Nothing. a CXI file is created
    """
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
    entry.attrs['default'] = 'object'

    entry.create_dataset("program_name", data="PyNX %s" % _pynx_version)
    entry.create_dataset("start_time", data=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())))
    if experiment_id is not None:
        entry.create_dataset("experiment_id", data=experiment_id)

    if note is not None:
        note_1 = entry.create_group("note_1")
        note_1.attrs['NX_class'] = 'NXnote'
        note_1.create_dataset("data", data=note)
        note_1.create_dataset("type", data="text/plain")

    if sample_name is not None:
        sample_1 = entry.create_group("sample_1")
        sample_1.attrs['NX_class'] = 'NXsample'
        sample_1.create_dataset("name", data=sample_name)

    if shift_phase_zero:
        # Get the object and center the phase around 0
        obj = phase.shift_phase_zero(obj, percent=2, origin=0, mask=scan_area_obj)

    # Store object in result_1
    if obj.ndim == 2:
        # Need this for the different layers
        obj = obj.reshape((1, obj.shape[1], obj.shape[2]))

    # Flip to have origin at top, left corner
    obj = np.flip(obj, axis=-2)
    if obj_phase_ramp is not None:
        obj_phase_ramp = obj_phase_ramp[0], -obj_phase_ramp[0]

    result_1 = entry.create_group("result_1")
    entry["object"] = h5py.SoftLink(entry_path + '/result_1')  # Unorthodox, departs from specification ?
    result_1['title'] = 'Object'
    result_1.attrs['NX_class'] = 'NXdata'
    # Save the object as a Virtual Dataset, so that we can show scan_area and illumination
    # as auxiliary signals. This requires hdf5>=1.10
    obj_layout = h5py.VirtualLayout(shape=obj.shape, dtype=obj.dtype)
    aux = []
    for i in range(len(obj)):
        ds = result_1.create_dataset("Object-layer%02d" % i, data=obj[i], chunks=True, shuffle=True, compression="gzip")
        ds.attrs['interpretation'] = 'image'
        # Explicitly using a link to the same file to avoid linking issue
        # See https://github.com/h5py/h5py/issues/1546 fixed in h5py>=3.0
        # obj_layout[i] = h5py.VirtualSource(ds)
        obj_layout[i] = h5py.VirtualSource(".", ds.name, ds.shape, ds.dtype)
        if i > 0:
            aux.append("Object-layer%02d" % i)
    result_1.create_virtual_dataset("data", obj_layout)
    result_1.attrs['signal'] = 'Object-layer00'
    result_1.create_dataset("data_type", data="electron density")
    result_1.create_dataset("data_space", data="real")
    ny, nx = obj.shape[-2:]
    px, py = pixel_size_object
    result_1.create_dataset("image_size", data=[px * nx, py * ny])
    # Store object pixel size (not in CXI specification)
    result_1.create_dataset("x_pixel_size", data=px)
    result_1.create_dataset("y_pixel_size", data=py)
    # X & Y axis data for NeXuS plotting
    unit_scale = np.log10(max(nx * px, ny * py))
    if unit_scale < -6:
        unit_name = "nm"
        unit_scale = 1e9
    elif unit_scale < -3:
        unit_name = u"µm"
        unit_scale = 1e6
    elif unit_scale < 0:
        unit_name = "mm"
        unit_scale = 1e3
    else:
        unit_name = "m"
        unit_scale = 1

    result_1.attrs['axes'] = np.array(['row_coords', 'col_coords'], dtype=h5py.special_dtype(vlen=str))
    # Flip: to have origin at top, left corner
    yc = np.flip((np.arange(ny) * py + posxy_c[1] - ny * py / 2) * unit_scale)
    result_1.create_dataset('row_coords', data=yc)
    result_1['row_coords'].attrs['units'] = unit_name
    result_1['row_coords'].attrs['long_name'] = 'Y (%s)' % unit_name
    xc = (np.arange(nx) * px + posxy_c[0] - nx * px / 2) * unit_scale
    result_1.create_dataset('col_coords', data=xc)
    result_1['col_coords'].attrs['units'] = unit_name
    result_1['col_coords'].attrs['long_name'] = 'X (%s)' % unit_name

    # Basis vector - This corresponds to a 'top, left' origin convention
    # We re-use the same field as for a detector in CXI covention
    basis_vectors = np.zeros((2, 3), dtype=np.float32)
    basis_vectors[0, 1] = -px
    basis_vectors[1, 0] = -py
    result_1.create_dataset("basis_vectors", data=basis_vectors)
    # Also add corner coordinates
    result_1.create_dataset("corner_positions", data=[xc[0] / unit_scale, yc[0] / unit_scale, 0])

    if scan_area_obj is not None:
        scan_area_obj = np.flipud(scan_area_obj)  # origin at top, left corner
        # Using 'mask' from CXI specification: 0x00001000 == CXI_PIXEL_HAS_SIGNAL 'pixel signal above background'
        s = (scan_area_obj > 0).astype(np.int) * 0x00001000
        result_1.create_dataset("mask", data=s, chunks=True, shuffle=True, compression="gzip")
        aux.append("mask")
        result_1[aux[-1]].attrs['note'] = "Calculated polygonal scan area"

    if obj_illumination is not None:
        obj_illumination = np.flipud(obj_illumination)  # origin at top, left corner
        result_1.create_dataset("obj_illumination", data=obj_illumination, chunks=True, shuffle=True,
                                compression="gzip")
        aux.append("obj_illumination")
        result_1[aux[-1]].attrs['note'] = "Integrated illumination intensity"

    result_1.attrs["auxiliary_signals"] = np.array(aux, dtype=h5py.special_dtype(vlen=str))

    # Store probe in result_2
    probe = np.flip(probe, axis=-2)  # origin at top, left corner
    result_2 = entry.create_group("result_2")
    result_2['title'] = 'Probe'
    result_2.attrs['NX_class'] = 'NXdata'
    result_2.attrs['signal'] = 'data'
    entry["probe"] = h5py.SoftLink(entry_path + '/result_2')  # Unorthodox, departs from specification ?
    result_2.create_dataset("data", data=probe, chunks=True, shuffle=True, compression="gzip")
    result_2["data"].attrs['interpretation'] = 'image'
    result_2.create_dataset("data_space", data="real")
    ny, nx = probe.shape[-2:]
    result_2.create_dataset("image_size", data=[px * nx, py * ny])
    # Store probe pixel size (not in CXI specification)
    result_2.create_dataset("x_pixel_size", data=px)
    result_2.create_dataset("y_pixel_size", data=py)
    # X & Y axis data for NeXuS plotting
    nyp, nxp = probe.shape[-2:]
    result_2.attrs['axes'] = np.array(['row_coords', 'col_coords'], dtype=h5py.special_dtype(vlen=str))
    # Flip to have origin at top, left
    yc = np.flip((np.arange(nyp) * py - nyp * py / 2) * unit_scale)
    result_2.create_dataset('row_coords', data=yc)
    result_2['row_coords'].attrs['units'] = unit_name
    result_2['row_coords'].attrs['long_name'] = 'Y (%s)' % unit_name
    xc = (np.arange(nxp) * px - nxp * px / 2) * unit_scale
    result_2.create_dataset('col_coords', data=xc)
    result_2['col_coords'].attrs['units'] = unit_name
    result_2['col_coords'].attrs['long_name'] = 'X (%s)' % unit_name

    if scan_area_probe is not None:
        scan_area_probe = np.flipud(scan_area_probe)  # origin at top, left corner
        # Using 'mask' from CXI specification: 0x00001000 == CXI_PIXEL_HAS_SIGNAL 'pixel signal above background'
        s = (scan_area_probe > 0).astype(np.int) * 0x00001000
        result_2.create_dataset("mask", data=s, chunks=True, shuffle=True, compression="gzip")

    if background is not None:
        result_3 = entry.create_group("result_3")
        result_3['title'] = 'Incoherent background'
        result_3.attrs['NX_class'] = 'NXdata'
        result_3.attrs['signal'] = 'data'
        entry["background"] = h5py.SoftLink(entry_path + '/result_3')  # Unorthodox, departs from specification ?
        result_3.create_dataset("data", data=background, chunks=True, shuffle=True,
                                compression="gzip")
        result_3["data"].attrs['interpretation'] = 'image'
        result_3.create_dataset("data_space", data="diffraction")
        result_3.create_dataset("data_type", data="intensity")

    if scale is not None:
        if np.allclose(scale, 1) is False:
            result_4 = entry.create_group("result_4")
            result_4['title'] = 'Floating intensities for each frame'
            result_4.attrs['NX_class'] = 'NXdata'
            result_4.attrs['signal'] = 'data'
            result_4.attrs['note'] = 'Floating intensities for each frame'
            entry["floating_intensity"] = h5py.SoftLink(entry_path + '/result_4')
            result_4.create_dataset("data", data=scale, compression="gzip")
            result_4["data"].attrs['interpretation'] = 'spectrum'
            result_4.create_dataset("data_space", data="diffraction")
            result_4.create_dataset("data_type", data="scale")

    # Save positions in entry_5
    result_5 = entry.create_group("result_5")
    result_5.attrs['note'] = 'Probe positions and shifts (x, y, dx, dy)'
    result_5.attrs['NX_class'] = 'NXdata'
    result_5.attrs['signal'] = 'data'
    result_5.attrs['note'] = 'Probe positions for each frame, and shift with original data: x, y, dx, dy'
    entry["positions"] = h5py.SoftLink(entry_path + '/result_5')
    s = np.empty((4, len(posxy[0])), dtype=np.float32)
    s[0] = posxy[0] + posxy_c[0]
    s[1] = posxy[1] + posxy_c[1]
    s[2] = posxy[0] - posxy0[0]
    s[3] = posxy[1] - posxy0[1]
    result_5.create_dataset("data", data=s, compression="gzip")
    result_5["data"].attrs['interpretation'] = 'spectrum'
    result_5.create_dataset("data_space", data="real")
    # result_5.create_dataset("data_type", data="translation")

    instrument_1 = entry.create_group("instrument_1")
    instrument_1.attrs['NX_class'] = 'NXinstrument'
    if instrument is not None:
        instrument_1.create_dataset("name", data=instrument)

    nrj = 12.3984 / (wavelength * 1e10)
    source_1 = instrument_1.create_group("source_1")
    source_1.attrs['NX_class'] = 'NXsource'
    source_1.attrs['note'] = 'Incident photon energy (instead of source energy), for CXI compatibility'
    source_1.create_dataset("energy", data=nrj * 1.60218e-16)  # in J
    source_1["energy"].attrs['units'] = 'J'

    beam_1 = instrument_1.create_group("beam_1")
    beam_1.attrs['NX_class'] = 'NXbeam'
    beam_1.create_dataset("incident_energy", data=nrj * 1.60218e-16)
    beam_1["incident_energy"].attrs['units'] = 'J'
    beam_1.create_dataset("incident_wavelength", data=wavelength)
    beam_1["incident_wavelength"].attrs['units'] = 'm'

    detector_1 = instrument_1.create_group("detector_1")
    detector_1.attrs['NX_class'] = 'NXdetector'
    detector_1.create_dataset("distance", data=detector_distance)
    detector_1["distance"].attrs['units'] = 'm'

    detector_1.create_dataset("x_pixel_size", data=pixel_size_detector)
    detector_1["x_pixel_size"].attrs['units'] = 'm'
    detector_1.create_dataset("y_pixel_size", data=pixel_size_detector)
    detector_1["y_pixel_size"].attrs['units'] = 'm'

    # Add shortcut to the main data as data_1 (follows CXI convention)
    data_1 = entry.create_group("data_1")
    data_1.attrs['NX_class'] = 'NXdata'
    data_1.attrs['signal'] = 'data'
    data_1.attrs['interpretation'] = 'image'
    data_1["data"] = h5py.SoftLink(entry_path + '/result_1/data')

    command = ""
    for arg in sys.argv:
        command += arg + " "
    process_1 = entry.create_group("process_1")
    process_1.attrs['NX_class'] = 'NXprocess'
    process_1.create_dataset("program", data='PyNX')  # NeXus spec
    process_1.create_dataset("version", data="%s" % _pynx_version)  # NeXus spec
    process_1.create_dataset("command", data=command)  # CXI spec

    if process is not None:
        for k, v in process.items():
            if isinstance(v, str) and k not in process_1:
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
                    note.create_dataset("data", data=str(vv))
                    note.create_dataset("description", data=kk)
                    note.create_dataset("type", data="text/plain")
    # Configuration of process: custom ESRF data policy
    # see https://gitlab.esrf.fr/sole/data_policy/blob/master/ESRF_NeXusImplementation.rst
    if params is not None or obj_zero_phase_mask is not None:
        config = process_1.create_group("configuration")
        config.attrs['NX_class'] = 'NXcollection'
        if params is not None:
            for k, v in params.items():
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

        if obj_zero_phase_mask is not None:
            config.create_dataset("obj_zero_phase_mask", data=obj_zero_phase_mask, chunks=True, shuffle=True,
                                  compression="gzip")
            config["obj_zero_phase_mask"].attrs['note'] = 'Weighted mask of region restrained to real values'

    # Configuration & results of process: custom ESRF data policy
    # see https://gitlab.esrf.fr/sole/data_policy/blob/master/ESRF_NeXusImplementation.rst
    results = process_1.create_group("results")
    results.attrs['NX_class'] = 'NXcollection'
    results.create_dataset('llk_poisson', data=llk_poisson)
    results.create_dataset('llk_gaussian', data=llk_gaussian)
    results.create_dataset('llk_euclidian', data=llk_euclidian)
    results.create_dataset('nb_photons_calc', data=nb_photons_calc)
    h = history.as_numpy_record_array()
    h['time'] -= history.t0  # Only 4 bytes actual accuracy,
    results.create_dataset('cycle_history', data=h)
    if obj_phase_ramp is not None:
        results.create_dataset('obj_phase_ramp_dxy', data=obj_phase_ramp)
        results['obj_phase_ramp_dxy'].attrs['note'] = 'Fourier shifts of calculated diffraction' \
                                                      'patterns with respect to centre. Used to' \
                                                      'correct the final object phase ramp.'
    for k in history.keys():
        # we use history[k] rather than h[k] to only list original values
        results.create_dataset('cycle_history_%s' % k, data=history[k].as_numpy_record_array())

    if extra_data is not None:
        extra = entry.create_group("_extra_data")
        extra.attrs['note'] = 'Extra data (undocumented, probably for debugging)'
        for k, v in extra_data.items():
            if isinstance(v, dict):
                d = extra.create_group(k)
                for k1, v1 in d.items():
                    d.create_dataset(k1, data=v1)
            else:
                extra.create_dataset(k, data=v)

    f.close()


class OperatorPtycho(Operator):
    """
    Base class for an operator on Ptycho2D objects.
    """

    def timestamp_increment(self, p):
        # By default CPU operators should increment the CPU counter. Unless they don't affect the pty object, like
        # all display operators.
        p._timestamp_counter += 1
        p._cpu_timestamp_counter = p._timestamp_counter


def algo_string(algo_base, p, update_object, update_probe, update_background=False, update_pos=False):
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
        if len(p._probe) > 1:
            s += "%d" % (len(p._probe))
        s += "p"

    if update_background:
        s += "/b"

    if update_pos:
        s += "/t"

    return s


def calc_throughput(p: Ptycho = None, cxi=None, verbose=False):
    """
    Analyse the throughput after a series of algorithms, either from a  Ptycho
    object or from a CXI file.
    A few things like object & probe smoothing are not taken into account,
    or the use of bilinear interpolation, background...
    :param p: the Ptycho object the timings are extracted from.
    :param cxi: the CXI file the history of cycles will be obtained from
    :param verbose: if True, print the average throughput per algorithm step
    :return: the average throughput in Gbyte/s
    """
    tit = None
    if cxi is not None:
        with h5py.File(cxi, 'r') as tmp:
            h = tmp['/entry_last/process_1/results/cycle_history'][()]
            probe_size = tmp['/entry_last/probe/data'][0].size
            iobs_size = probe_size * tmp['/entry_last/positions/data'].shape[-1]
            obj_size = tmp['/entry_last/object/data'][0].shape
            near_field = int(tmp['/entry_last/process_1/configuration/near_field'][()])
            algo = tmp['/entry_last/process_1/configuration/algorithm'][()]
    else:
        h = p.history.as_numpy_record_array()
        if h is None:
            return 0
        probe_size = p.get_probe()[0].size
        obj_size = p.get_obj()[0].size
        iobs_size = p.data.iobs.size
        near_field = int(p.data.near_field)

    # Total number of read or write of the main array, assumed to be complex64
    # Does not take into account support update
    vnbrw = []
    for i in range(len(h)):
        algo = h['algorithm'][i].decode()
        if isinstance(algo, np.bytes_):
            algo = algo.decode('ascii')
        algo = algo.lower()
        nobj, npr = h['nb_obj'][i], h['nb_probe'][i]
        nop = nobj * npr
        uo = h['update_obj'][i]
        upr = h['update_probe'][i]
        uop = uo * upr
        upos = h['update_pos'][i]
        # number of array read or write for object*probe->psi
        # 1 r + 1 w for object probe multiplication
        # (the probe is read once per stack)
        op2psi = 2 * nop
        # propag to/from detector
        # 2 FFT (1 read + 1 write for each dimension) or
        # 4 FFT (1 read + 1 write for each dimension) + 1r+w for near field
        propag = 2 * (1 + near_field) * 2 * nop + 2 * near_field * nop
        # iobs amplitude read and psi modif
        # 1 r+w for amplitude projection per mode, iobs read 0.5
        # Actually more reads are needed because the Psi array is read twice,
        # once for the Icalc calculation, and once to be updated. But it's probably cached...
        iobs2psi = 2 * nop + .5
        # psi2obj and psi2probe
        # 2 r + 1.5w for probe update (.5 for norm)
        # 1 extra read for obj update due to atomic operation
        # (for object update, the probe should be cached and thus read only once per stack..)
        psi2oobj = (2 * nop + 2 * nobj + .5) * h['update_obj'][i]
        psi2probe = (2 * nop + npr + .5) * h['update_probe'][i]
        # 3 r for position update (no modes used)
        pos = 3 * upos
        # object + probe gradient calculation: 3 reads (psi,o,pr) + 1 write each + 1 read for object (atomic)
        psi2opgrad = (3 * nop + 2 * nobj) * uo + (3 * nop + npr) * upr
        if algo == 'ap':
            vnbrw.append(op2psi + 2 * propag + iobs2psi + psi2oobj + psi2probe + pos)
        elif algo == 'dm':
            # 3 r + 1 w extra compared to AP
            vnbrw.append(op2psi + 2 * propag + iobs2psi + psi2oobj + psi2probe + pos + 4 * nop)
        elif algo == 'ml':
            # Gradient:
            #   1 r + 1 w for object probe multiplication
            #   2 FFT (1 read + 1 write for each dimension) or
            #   4 FFT (1 read + 1 write for each dimension) + 1r+w for near field
            #   1 r+w for Fourier gradient calculation, iobs read 0.5
            #   2 r+w for both object and probe gradient evaluation
            # CG=
            #   4 times (PO, PdO, dPO, dPdO):
            #     1 r + 1 w for object probe multiplication
            #     1 FFT (1 read + 1 write for each dimension) or
            #     2 FFT (1 read + 1 write for each dimension) + 1r+w for near field
            #   4 read for gamma reduction, iobs read 0.5
            vnbrw.append(op2psi + 2 * propag + iobs2psi + psi2opgrad + 4 * (op2psi + propag + nop) + 0.5)

    vnbrw = np.array(vnbrw, dtype=np.float32)
    # Gbyte read or write per cycle
    vgb = iobs_size * 8 * vnbrw
    # Take into account object & probe update (not negligible for NFP)
    # 1 read of old object, 1 write, and 0.5 read for norm
    vgb += h['update_obj'] * 8 * obj_size * (2 * h['nb_obj'] + .5)
    vgb += h['update_probe'] * 8 * probe_size * (2 * h['nb_probe'] + .5)

    if verbose:
        print("Estimated memory throughput of algorithms:")
        print("The throughput assumes each 2D FFT uses 2 read+write, which is only"
              "guaranteed for some transforms, e.g. powers of 2 up to 2048.")
        # Average throughput per series of algorithm
        # ideally we'd need begin and end time..
        i0 = 0
        algo = h['algorithm'][0]
        for i in range(1, len(h)):
            if h['algorithm'][i] != algo:
                if isinstance(algo, np.bytes_):
                    algo = algo.decode('ascii')
                dt = h['time'][i] - h['time'][i0]
                g = vgb[i0:i].sum() / dt / 1024 ** 3
                print("     %s**%3d [dt=%5.1fs  <dt/cycle>=%6.4fs]: %6.1f Gbytes/s" %
                      (algo, i - i0, dt, dt / (i - i0), g))
                i0 = i
                algo = h['algorithm'][i]
        if i > i0:  # How can i==i0 happen ?
            if isinstance(algo, np.bytes_):
                algo = algo.decode('ascii')
            dt = h['time'][i] - h['time'][i0]
            g = vgb[i0:i].sum() / dt / 1024 ** 3
            print("     %s**%3d [dt=%5.1fs  <dt/cycle>=%6.4fs]: %6.1f Gbytes/s" % (algo, i - i0, dt, dt / (i - i0), g))

    # average throughput
    gbps = vgb.sum() / (h['time'][-1] - h['time'][0]) / 1024 ** 3

    return gbps
