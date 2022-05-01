# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


from ..wavefront import Wavefront, PropagateNearField
import numpy as np


def simulate_probe(shape, dz, pixel_size=1e-6, wavelength=0.5e-10, nb_line_v=10, nb_line_h=10, nb_spot=10, amplitude=1):
    """
    Create a simulated probe corresponding to a HoloTomo object size, with vertical and horizontal lines and spots
    coming from optics defects, which are then propagated.

    :param shape: the 2D shape (ny, nx) of the probe
    :param dz: array of propagation distances (m) (unrelated to holo-tomo distances)
    :param pixel_size: detector pixel size (m)
    :param wavelength: the wavelength (m)
    :param nb_line_v: number of vertical lines. Width = max 5% of the probe horizontal size.
    :param nb_line_h: number of horizontal lines. Width = max 5% of the probe vertical size
    :param nb_spot: number of spots. Radius= max 5% of the probe horizontal size
    :param amplitude: the relative amplitude of the introduced optical defects (default:1).
    :return: the simulated probe, with shape (nz, nbmode=1, ny, nx)
    """
    ny, nx = shape
    nz = len(dz)
    # take convenient dimensions for the wavefront
    d = np.zeros((nz, ny, nx), dtype=np.complex64)
    for j in range(nz):
        for i in range(nb_line_v):
            w = 1 + np.random.randint(0, nx * 0.05)
            ii = np.random.randint(0, nx - w)
            t = np.random.randint(10)
            d[j, :, ii:ii + w] = t

        for i in range(nb_line_h):
            w = 1 + np.random.randint(0, ny * 0.05)
            ii = np.random.randint(0, ny - w)
            t = np.random.randint(10)
            d[j, ii:ii + w] = t
        x, y = np.meshgrid(np.arange(0, nx), np.arange(0, ny))

        for i in range(nb_spot):
            w = 1 + np.random.randint(0, nx * 0.05)
            ix = np.random.randint(w, nx - w)
            iy = np.random.randint(w, ny - w)
            r = np.sqrt((x - ix) ** 2 + (y - iy) ** 2)
            t = np.random.uniform(0,10)

            d[j] += t * (r < w)
        w = Wavefront(d=np.fft.fftshift(np.exp(1j * 1e-2 * d[j] * amplitude)), pixel_size=pixel_size, wavelength=wavelength)
        w = PropagateNearField(dz=dz[j]) * w
        d[j] = w.get(shift=True).reshape(ny, nx)
    return d.reshape(nz, 1, ny, nx)
