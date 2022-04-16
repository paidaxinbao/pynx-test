# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


import timeit
import multiprocessing
import numpy as np
from scipy.ndimage import zoom
import fabio

try:
    from skimage.registration import phase_cross_correlation as register_translation
except ImportError:
    from skimage.feature import register_translation
from ..wavefront import Wavefront, PropagateNearField


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
            t = np.random.uniform(0, 10)

            d[j] += t * (r < w)
        w = Wavefront(d=np.fft.fftshift(np.exp(1j * 1e-2 * d[j] * amplitude)), pixel_size=pixel_size,
                      wavelength=wavelength)
        w = PropagateNearField(dz=dz[j]) * w
        d[j] = w.get(shift=True).reshape(ny, nx)
    return d.reshape(nz, 1, ny, nx)


def zoom_pad_images(x, magnification, padding, nz):
    d0 = x[0]
    ny0, nx0 = np.array(d0.shape) + 2 * padding
    xz = np.empty((nz, ny0, nx0), dtype=np.float32)
    xz.fill(-1e38)  # masked value
    for iz in range(nz):
        # Zoom
        mg = magnification[0] / magnification[iz]
        if np.isclose(mg, 1):
            d = x[iz]
        else:
            d = zoom(x[iz], zoom=mg, order=1)
        ny, nx = d.shape
        if ny0 <= ny and nx0 <= nx:
            xz[iz] = d[ny // 2 - ny0 // 2:ny // 2 - ny0 // 2 + ny0, nx // 2 - nx0 // 2:nx // 2 - nx0 // 2 + nx0]
        elif ny0 > ny and nx0 <= nx:
            xz[iz, ny0 // 2 - ny // 2:ny0 // 2 - ny // 2 + ny] = d[:, nx // 2 - nx0 // 2:nx // 2 - nx0 // 2 + nx0]
        elif ny0 <= ny and nx0 > nx:
            xz[iz, :, nx0 // 2 - nx // 2:nx0 // 2 - nx // 2 + nx] = d[ny // 2 - ny0 // 2:ny // 2 - ny0 // 2 + ny0]
        else:
            xz[iz, ny0 // 2 - ny // 2:ny0 // 2 - ny // 2 + ny, nx0 // 2 - nx // 2:nx0 // 2 - nx // 2 + nx] = d
    return xz


def zoom_pad_images_kw(kwargs):
    return zoom_pad_images(**kwargs)


def align_images(x, x0, nz):
    t0 = timeit.default_timer()
    # TODO : align against previous distance instead of 0 ?
    dx = [0]
    dy = [0]
    for iz in range(1, nz):
        #         print(i,iz)
        d0 = x[0] / x0[0]
        # Align
        upsample_factor = 10
        d = x[iz] / x0[iz]
        # Use only image centres if large enough
        ny, nx = d0.shape
        # if ny > 1500:
        #     d0 = d0[ny // 2 - 512:ny // 2 + 512]
        #     d = d[ny // 2 - 512:ny // 2 + 512]
        # if nx > 1500:
        #     d0 = d0[:, nx // 2 - 512:nx // 2 + 512]
        #     d = d[:, nx // 2 - 512:nx // 2 + 512]
        pixel_shift, err, dphi = register_translation(d0, d, upsample_factor=upsample_factor)
        dx.append(pixel_shift[1])
        dy.append(pixel_shift[0])
    return dx, dy, multiprocessing.current_process().pid, timeit.default_timer() - t0


def align_images_kw(kwargs):
    return align_images(**kwargs)


def load_data(i, dark, nz, img_name):
    ny, nx = dark.shape[-2:]
    d = np.empty((nz, ny, nx), dtype=np.float32)
    for iz in range(0, nz):
        img = fabio.open(img_name % (iz + 1, iz + 1, i)).data
        d[iz] = img - dark[iz]
    return d


def load_data_kw(kwargs):
    return load_data(**kwargs)


def save_phase_edf(idx, ph, prefix_result):
    edf = fabio.edfimage.EdfImage(data=ph.astype(np.float32))
    edf.write("%s_%04d.edf" % (prefix_result, idx))


def save_phase_edf_kw(kwargs):
    return save_phase_edf(**kwargs)
