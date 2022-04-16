# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import numpy as np
from ..utils.rotation import rotate, rotation_matrix


def detector2s(nx, ny, pixel_size, distance, x0, y0, rotation_axes, wavelength):
    """
    Convert the pixel positions on the detector to scattering vector coordinates
    All axes follow the NeXus convention, with an incoming beam along the z-axis.

    :param nx: number of pixels along the X detector axis
    :param ny: number of pixels along the Y detector axis
    :param pixel_size: detector pixel size (in meters)
    :param distance: sample to detector distance, in meters
    :param x0: X pixel coordinate of the direct beam on the detector, when all axes are at the origin
               (detector in the direct beam)
    :param y0: Y pixel coordinate of the direct beam on the detector, when all axes are at the origin
               (detector in the direct beam)
    :param rotation_axes: tuple of rotations to put the detector in position, e.g. (('x', 0), ('y', pi/4)):
               The rotation axes giving are be applied in order.
    :param wavelength: experiment wavelength in meters
    :return: a tuple of 3 2D arrays of scattering vector coordinates, in inverse meters. These follow the
             crystallographic convention, i.e. the norm of the vectors is 2*sin(theta)/lambda, where theta is half
             the deviation of the incident beam. The array coordinates also follow the NeXus convention, with the
             (0,0) origin being at the top left corner of the detector, as seen from the sample.
    """
    y, x = np.meshgrid(np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32), indexing='ij')
    # - signs following NeXus convention for axes direction for detector and laboratory reference frames
    y = (y0 - y) * pixel_size
    x = (x0 - x) * pixel_size
    z = np.zeros_like(x) + distance

    # xyz are in the laboratory frame coordinate with detector at origin
    # Now rotate detector
    for ax, ang in rotation_axes:
        m = rotation_matrix(ax, ang)
        x, y, z = rotate(m, x, y, z)

    # xyz are in the laboratory frame coordinate with the detector rotated, in meters
    # Now project onto Ewald's sphere and set the origin of reciprocal space
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    sx = x / (wavelength * r)
    sy = y / (wavelength * r)
    sz = z / (wavelength * r) - 1 / wavelength

    # # Compute average 2theta angle:
    # s = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    # ttheta = np.arcsin(wavelength * s / 2) * 2
    # print("Average 2theta: %8.2f°" % np.rad2deg(ttheta.mean()))

    return sx, sy, sz
