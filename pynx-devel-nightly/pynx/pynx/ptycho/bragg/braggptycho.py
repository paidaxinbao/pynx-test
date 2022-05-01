# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['BraggPtychoData', 'BraggPtycho', 'OperatorBraggPtycho']

import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.interpolate import RegularGridInterpolator
from ...operator import Operator


class BraggPtychoData(object):
    """Class for three-dimensional ptychographic data: observed diffraction and probe positions.
    This may include only part of the data from a larger dataset.
    """

    def __init__(self, iobs=None, positions=None, mask=None, wavelength=None, detector=None):
        """

        :param iobs: 4d array with (nb_frame, nz, ny,nx) shape observed intensity (assumed to follow Poisson statistics)
        :param positions: (x, y, z) tuple or 2d array with ptycho probe positions in meters.
                          The coordinates must follow the NeXus/CXI convention, and must already be in a
                          laboratory frame taking into account sample rotation.
        :param mask: 3D mask (>0 means masked pixel) for the observed data. Can be None.
        :param wavelength: wavelength of the experiment, in meters.
        :param detector= {'geometry':'psic', 'delta':0, 'nu':0, 'pixel_size':55e-6, 'distance':1,
                          'rotation_axis':'eta', 'rotation_step': 0.01*np.pi/180}:
               parameters for the detector as a dictionary.
        """
        if iobs is not None:
            self.iobs = fftshift(iobs, axes=(1, 2, 3)).astype(np.float32)
            if mask is not None:
                self.mask = fftshift(mask.astype(np.int8))
                self.iobs[mask > 0] = -100
            else:
                self.mask = None
        else:
            self.iobs = None

        self.wavelength = wavelength
        self.posx, self.posy, self.posz = positions
        self.posx -= self.posx.mean()
        self.posy -= self.posy.mean()
        self.posz -= self.posz.mean()
        self.detector = detector


class BraggPtycho(object):
    """ Class for 3D Bragg ptychography data: object, probe, and observed diffraction.
    This may include only part of the data from a larger dataset
    """

    def __init__(self, probe=None, data=None, support=None):
        """
        Constructor.
        :param probe: the starting estimate of the probe, as a pynx wavefront object - can be 3D if modes are used.
        :param data: the BraggPtychoData object with all observed frames, ptycho positions
        :param support: the support of the object (1 inside, 0 outside) the object will be constrained to
        """
        self._probe2d = probe  # The original 2D probe as a Wavefront object
        self._probe = None  # This will hold the probe as projected onto the 3D object
        self._obj = None
        self.support = support
        self.data = data
        self._background = None

        # Matrix transformation from array indices of the array obtained by inverse Fourier Transform
        # to xyz in the laboratory frame
        self.m = None
        # Inverse of self.m
        self.im = None

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

    def get_obj(self):
        """
        Get the object data array. This will automatically get the latest data, either from GPU or from the host
        memory, depending where the last changes were made.

        :param shift: if True, the data array will be fft-shifted so that the center of the data is in the center
                      of the array, rather than in the corner (the default).
        :return: the 3D numpy data array (nb object modes, nyo, nxo)
        """
        if self._cpu_timestamp_counter < self._timestamp_counter:
            if self._timestamp_counter == self._cl_timestamp_counter:
                self._obj = self._cl_obj.get()
                self._probe = self._cl_probe.get()
            if self._timestamp_counter == self._cu_timestamp_counter:
                self._obj = self._cu_obj.get()
                self._probe = self._cu_probe.get()
            self._cpu_timestamp_counter = self._timestamp_counter

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

    def set_support(self, sup):
        """
        Set the support data array. This should be a 3D array of the correct shape.

        :param sup: the support array. 0 outside support, 1 inside
        :return: nothing
        """
        self.support = sup.astype(np.int8)
        self._timestamp_counter += 1
        self._cpu_timestamp_counter = self._timestamp_counter

    def prepare(self):
        """
        Calculate projection parameters
        :return:
        """
        self.calc_orthonormalization_matrix()
        self.init_probe()
        self.init_obj()

    def calc_orthonormalization_matrix(self):
        """
        Calculate the orthonormalization matrix to convert probe/object array coordinates to/from orthonormal ones.
        :return:
        """
        npsi, ny, nx = self.data.iobs.shape[1:]
        if self.data.detector['geometry'] == 'psic':
            cd, sd = np.cos(self.data.detector['delta']), np.sin(self.data.detector['delta'])
            cn, sn = np.cos(self.data.detector['nu']), np.sin(self.data.detector['nu'])
            dpsi = self.data.detector['rotation_step']
            # Pixel size. Assumes square frames from detector.
            d = self.data.detector['distance']
            lambdaz = self.data.wavelength * d
            p = self.data.detector['pixel_size']
            # TODO: take into account other rotation axes
            if self.data.detector['rotation_axis'] == 'eta':
                self.b = np.array(
                    [[-p * nx * cn, -p * ny * sn * sd, 0],
                     [0, -p * ny * cd, -dpsi * npsi * d * (cd * cn - 1)],
                     [-p * nx * sn, p * ny * sd * cn, dpsi * npsi * d * sd]], dtype=np.float32)
            elif self.data.detector['rotation_axis'] == 'phi':
                self.b = np.array([[-p * nx * cn, -p * ny * sn * sd, -dpsi * npsi * d * (1 - cd * cn)],
                                   [0, -p * ny * cd, 0],
                                   [-p * nx * sn, p * ny * sd * cn, dpsi * npsi * d * sn * cd]], dtype=np.float32)
            else:
                raise Exception("BraggPtycho: only recognized rotation axis are eta or phi (PSIC geometry")

            self.m = lambdaz * np.linalg.inv(self.b).transpose()
            self.im = np.linalg.inv(self.m)
        else:
            raise Exception("BraggPtycho: only recognized geometry is psic")

    def init_obj(self):
        """
        Initialize the object array
        :return: nothing. The object is created as an empty array
        """
        nzo, nyo, nxo = self.calc_obj_shape()
        print("Initialised object with %dx%dx%d voxels" % (nzo, nyo, nxo))
        self.set_obj(np.empty((1, nzo, nyo, nxo), dtype=np.complex64))

    def calc_obj_shape(self, margin=8, multiple=2):
        """
        Calculate the 3D object shape, given the detector, probe and scan characteristics.
        This must be called after the 3D probe has been initialized:
        :param margin: margin to extend the object area, in case the positions will change (optimization)
        :param multiple: the shape must be a multiple of that number. >=2
        :return: the object shape
        """
        probe_shape = self._probe.shape[1:]
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
        :param domain='probe': the domain over which the xyz coordinates should be returned. It should either
                               be 'object' (the default) or the probe, the only difference being that the object
                               is extended to cover all the volume scanned by the shifted probe. The probe
                               has the same size as the observed 3D data.
        :param rotation=('z',np.deg2rad(-20)): optionally, the coordinates can be obtained after a rotation of the
                                               object. This is useful if the object or support is to be defined as a
                                               parallelepiped, before being rotated to be in diffraction condition.
                                               The rotation can be given as a tuple of a rotation axis name (x, y or z)
                                               and a counter-clockwise rotation angle in radians.
        :return: a tuple of (x,y,z) coordinates, each a 3D array
        """
        if domain == "probe":
            nz, ny, nx = self.data.iobs.shape[1:]
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

    def init_probe(self):
        """
        Calculate the probe other the object volume, given the 3D probe and the object volume, assuming that the probe
        is invariant along z.
        :return: Nothing. Creates self._probe
        """
        # TODO: move this to an operator and kernel for faster conversion ?
        # TODO: should the probe array have the same size as the object ? Maybe crop the probe array to exclude
        # the parts which are zeros.
        print("Calculating probe on object grid")
        x, y, z = self.get_xyz(domain='probe')
        nz, ny, nx = x.shape
        z0, z1 = z.min(), z.max()

        # Create 3D probe array in the laboratory frame
        dz = z1 - z0
        pr2d = self._probe2d.get(shift=True)
        # TODO: take into account all probe modes
        if pr2d.ndim == 3:
            pr2d = pr2d[0]
        nyp, nxp = pr2d.shape
        if np.isclose(self.data.wavelength, self._probe2d.wavelength) is False:
            raise Exception('BraggPtycho: probe and data wavelength are different !')
        pixel_size_probe = self._probe2d.pixel_size
        pr = np.empty((nz, nyp, nxp), dtype=np.complex64)
        pr[:] = pr2d

        # Original probe coordinates
        zp, yp, xp = np.arange(nz), np.arange(nyp), np.arange(nxp)
        zp = (zp - zp.mean()) * (dz / nz)
        yp = (yp - yp.mean()) * pixel_size_probe
        xp = (xp - xp.mean()) * pixel_size_probe

        # Interpolate probe to object grid
        rgi = RegularGridInterpolator((zp, yp, xp), pr, method='linear', bounds_error=False, fill_value=0)
        self._probe = rgi(np.concatenate((z.reshape(1, z.size), y.reshape(1, y.size),
                                          x.reshape(1, x.size))).transpose()).reshape((1, nz, ny, nx)).astype(
            np.complex64)
        # To check view of object:
        # pcolormesh(z[:,:,100],y[:,:,100],abs(p._probe3d[:,:,100]))
        # np.savez_compressed('probe3d.npz', probe3dobj=self._probe, probe3d=pr, probe2d=pr2d, x=x, y=y, z=z)


class OperatorBraggPtycho(Operator):
    """
    Base class for an operator on Bragg Ptycho objects.
    """

    def timestamp_increment(self, p):
        # By default CPU operators increment the CPU counter. Unless they don't affect the pty object, like
        # display operators.
        p._timestamp_counter += 1
        p._cpu_timestamp_counter = p._timestamp_counter
