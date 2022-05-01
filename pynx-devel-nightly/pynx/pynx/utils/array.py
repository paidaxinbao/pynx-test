# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import warnings
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.fftpack import fftn, ifftn, fftshift, fftfreq

try:
    from numba import njit, prange

    has_numba = True
except ImportError:
    has_numba = False


def rebin(a, rebin_f, scale="sum", mask=None, mask_iobs_cdi=False, **kwargs):
    """
     Rebin a 2 or 3-dimensional array. If its dimensions are not a multiple of rebin_f, the array will be cropped.
     
    Args:
        a: the array to resize, which can also be a masked array
        rebin_f: the rebin factor - pixels will be summed by groups of rebin_f x rebin_f (x rebin_f). This can
                 also be a tuple/list of rebin values along each axis, e.g. rebin_f=(4,1,2) for a 3D array
                 Instead of summing/averaging the pixels over the rebin box, it is also possible to
                 select a sub-pixel by giving the shift for each dimension, e.g. with "rebin=4,1,2,0,0,1",
                 the extracted array will be a[0::4,0::1,1::2]
        scale: if "sum" (the default), the array total will be kept.
            If "average", the average pixel value will be kept.
            If "square", the array is scaled so that (abs(a)**2).sum() is kept
        mask: an array of values to be masked (0 or False, valid entries, >0 or True, should be masked).
            Alternatively, a can be a masked array.
        mask_iobs_cdi: if True, special negative Iobs values used in pynx.cdi will be correctly handled.
    Returns:
        the array after rebinning. A masked array if mask is not None.
    """
    if "normalize" in kwargs:
        scale = "average"
        warnings.warn("rebin(): normalize is deprecated, use scale='average' instead", DeprecationWarning)
    if "mode" in kwargs:
        scale = kwargs["mode"]
        warnings.warn("rebin(): mode is deprecated, use scale=XX instead", DeprecationWarning)

    if isinstance(a, np.ma.MaskedArray) and mask is None:
        a, mask = a.data, a.mask

    if mask_iobs_cdi:
        # decompose the array in the different masked areas
        m_est = np.logical_and(a <= -1e38, a > -1e19)  # True if not estimated
        m_free = np.logical_and(a > -1e19, a <= -0.5)  # True if free
        # Revert free pixels
        if m_free.sum():
            a = a.copy()
            a[m_free] = -(a[m_free] + 1)
        m_obs = a < -.5

        # Bin valid pixels
        ar_obs = rebin(a, rebin_f, mask=m_obs, scale=scale)
        # Bin interpolated pixels
        ar_int = rebin(-a / 1e19 - 1, rebin_f, mask=m_est, scale=scale)

        # Assemble final array
        ar = ar_obs.data
        # Invalid pixels
        mr_inv = np.logical_and(ar_obs.mask, ar_int.mask)
        ar[mr_inv] = -1.05e38
        # Interpolated pixels
        mr_int = np.logical_and(ar_obs.mask, ar_int.mask == 0)
        ar[mr_int] = -1e19 * (ar_int[mr_int] + 1)
        # Free pixels - we aim to keep the same number of free pixels
        # as in the un-binned version, to make sure we keep large enough islands.
        # TODO: check this is the correct approach for free pixels & re-binning
        mr_free = rebin(m_free.astype(np.float32), rebin_f, scale="sum") > 0
        ar[mr_free] = -ar[mr_free] - 1

        return ar

    ndim = a.ndim
    if isinstance(rebin_f, int) or isinstance(rebin_f, np.integer):
        rebin_f = [rebin_f] * ndim
    else:
        assert ndim == len(rebin_f) or 2 * ndim == len(rebin_f), \
            "Rebin: number of dimensions does not agree with number of rebin values:" + str(rebin_f)
    if ndim == 2:
        if len(rebin_f) == 2 * ndim:
            ry, rx, iy, ix = rebin_f
            return a[iy::ry, ix::rx]
        ny, nx = a.shape
        a = a[:ny - (ny % rebin_f[0]), :nx - (nx % rebin_f[1])]
        sh = ny // rebin_f[0], rebin_f[0], nx // rebin_f[1], rebin_f[1]
        if scale.lower() == "average":
            if mask is not None:
                b = np.ma.masked_array(a, mask).reshape(sh)
                n = (mask == 0).reshape(sh)
                return (b.sum(axis=(1, 3)) / n.sum(axis=(1, 3))).astype(a.dtype)
            return a.reshape(sh).sum(axis=(1, 3)) / np.prod(rebin_f)
        elif "sq" in scale.lower():
            if mask is not None:
                b = np.ma.masked_array(a, mask).reshape(sh).sum(axis=(1, 3))
            else:
                b = a.reshape(sh).sum(axis=(1, 3))
            return b * np.sqrt((abs(a) ** 2).sum() / (abs(b) ** 2).sum())
        else:
            if mask is not None:
                b = np.ma.masked_array(a, mask).reshape(sh)
                return b.sum(axis=(1, 3))
            return a.reshape(sh).sum(axis=(1, 3))
    elif ndim == 3:
        if len(rebin_f) == 2 * ndim:
            rz, ry, rx, iz, iy, ix = rebin_f
            return a[iz::rz, iy::ry, ix::rx]
        nz, ny, nx = a.shape
        a = a[:nz - (nz % rebin_f[0]), :ny - (ny % rebin_f[1]), :nx - (nx % rebin_f[2])]
        sh = nz // rebin_f[0], rebin_f[0], ny // rebin_f[1], rebin_f[1], nx // rebin_f[2], rebin_f[2]
        if scale.lower() == "average":
            if mask is not None:
                b = np.ma.masked_array(a, mask).reshape(sh)
                n = (mask == 0).reshape(sh)
                return (b.sum(axis=(1, 3, 5)) / n.sum(axis=(1, 3, 5))).astype(a.dtype)
            return a.reshape(sh).sum(axis=(1, 3, 5)) / np.prod(rebin_f)
        elif "sq" in scale.lower():
            if mask is not None:
                b = np.ma.masked_array(a, mask).reshape(sh).sum(axis=(1, 3, 5))
            else:
                b = a.reshape(sh).sum(axis=(1, 3, 5))
            return b * np.sqrt((abs(a) ** 2).sum() / (abs(b) ** 2).sum())
        else:
            if mask is not None:
                b = np.ma.masked_array(a, mask).reshape(sh)
                return b.sum(axis=(1, 3, 5))
            return a.reshape(sh).sum(axis=(1, 3, 5))
    elif ndim == 4:
        if len(rebin_f) == 4 * ndim:
            r3, rz, ry, rx, i3, iz, iy, ix = rebin_f
            return a[i3::r3, iz::rz, iy::ry, ix::rx]
        n3, nz, ny, nx = a.shape
        a = a[:n3 - (n3 % rebin_f[0]), :nz - (nz % rebin_f[1]), :ny - (ny % rebin_f[2]), :nx - (nx % rebin_f[3])]
        sh = n3 // rebin_f[0], rebin_f[0], nz // rebin_f[1], rebin_f[1], ny // rebin_f[2], rebin_f[2], \
             nx // rebin_f[3], rebin_f[3]
        a = a.reshape(sh)
        # print("rebin(): a.shape=", a.shape)
        if scale.lower() == "average":
            if mask is not None:
                b = np.ma.masked_array(a, mask).reshape(sh)
                n = (mask == 0).reshape(sh)
                return (b.sum(axis=(1, 3, 5, 7)) / n.sum(axis=(1, 3, 5, 7))).astype(a.dtype)
            return a.sum(axis=(1, 3, 5, 7)) / np.prod(rebin_f)
        elif "sq" in scale.lower():
            if mask is not None:
                b = np.ma.masked_array(a, mask).reshape(sh).sum(axis=(1, 3, 5, 7))
            else:
                b = a.reshape(sh).sum(axis=(1, 3, 5, 7))
            return b * np.sqrt((abs(a) ** 2).sum() / (abs(b) ** 2).sum())
        else:
            if mask is not None:
                b = np.ma.masked_array(a, mask).reshape(sh)
                return b.sum(axis=(1, 3, 5, 7))
            return a.sum(axis=(1, 3, 5, 7))
    else:
        raise Exception("pynx.utils.array.rebin() only accept arrays of dimensions 2, 3 and 4")


def center_array_2d(a, other_arrays=None, threshold=0.2, roi=None, iz=None):
    """
    Center an array in 2D so that its absolute value barycenter is in the middle.
    If the array is 3D, it is summed along the first axis to determine the barycenter, and all frames along the first
    axis are shifted.
    The array is 'rolled' so that values shifted from the right appear on the left, etc...
    Shifts are integer - no interpolation is done.

    Args:
        a: the array to be shifted, can be a floating point or complex 2D or 3D array.
        other_arrays: can be another array or a list of arrays to be shifted by the same amount as a
        threshold: only the pixels above the maximum amplitude * threshold will be used for the barycenter
        roi: tuple of (x0, x1, y0, y1) corners coordinate of ROI to calculate center of mass
        iz: if a.ndim==3, the centering will be done based on the center of mass of the absolute value summed over all
            2D stacks. If iz is given, the center of mass will be calculated just on that stack

    Returns:
        the shifted array if only one is given or a tuple of the shifted arrays.
    """
    if a.ndim == 3:
        if iz is None:
            tmp = abs(a).astype(np.float32).sum(axis=0)
        else:
            tmp = abs(a[iz]).astype(np.float32)
    else:
        tmp = abs(a).astype(np.float32)

    if threshold is not None:
        tmp *= tmp > (tmp.max() * threshold)

    y0, x0 = center_of_mass(tmp)

    if roi is not None:
        xo, x1, yo, y1 = roi
        tmproi = tmp[yo:y1, xo:x1]
        y0, x0 = center_of_mass(tmproi)
        y0 += yo
        x0 += xo

    ny, nx = tmp.shape
    dx, dy = (int(round(nx // 2 - x0)), int(round(ny // 2 - y0)))
    # print("Shifting by: dx=%6.2f dy=%6.2f" % (dx, dy))

    # Multi-axis shift is supported only in numpy version >= 1.12 (2017)
    a1 = np.roll(np.roll(a, dy, axis=-2), dx, axis=-1)
    if other_arrays is None:
        return a1
    else:
        if type(other_arrays) is list:
            v = []
            for b in other_arrays:
                v.append(np.roll(b, (dy, dx), axis=(-2, -1)))
            return a1, v
        else:
            return a1, np.roll(other_arrays, (dy, dx), axis=(-2, -1))


def crop_around_support(obj: np.ndarray, sup: np.ndarray, margin=0):
    """

    :param obj: the array to be cropped (2D or 3D)
    :param sup: the support, either a boolean or integer array, with the same dimensions as a, 0 (False) indicating
                the pixels outside the support.
    :param margin: the number or pixels to be added on all sides of the array
    :return: a tuple with (cropped array, cropped support), keeping only pixels inside the support
    """
    if obj.ndim == 3:
        l0 = np.nonzero(sup.sum(axis=(1, 2)))[0].take([0, -1]) + np.array([-margin, margin])
        if l0[0] < 0:
            l0[0] = 0
        if l0[1] >= sup.shape[0]:
            l0[1] = -1

        l1 = np.nonzero(sup.sum(axis=(0, 2)))[0].take([0, -1]) + np.array([-margin, margin])
        if l1[0] < 0:
            l1[0] = 0
        if l1[1] >= sup.shape[1]:
            l1[1] = -1

        l2 = np.nonzero(sup.sum(axis=(0, 1)))[0].take([0, -1]) + np.array([-margin, margin])
        if l2[0] < 0:
            l2[0] = 0
        if l2[1] >= sup.shape[2]:
            l2[1] = -1
        obj = obj[l0[0]:l0[1], l1[0]:l1[1], l2[0]:l2[1]]
        sup = sup[l0[0]:l0[1], l1[0]:l1[1], l2[0]:l2[1]]
    else:
        l0 = np.nonzero(sup.sum(axis=1))[0].take([0, -1]) + np.array([-margin, margin])
        if l0[0] < 0:
            l0[0] = 0
        if l0[1] >= sup.shape[0]:
            l0[1] = -1

        l1 = np.nonzero(sup.sum(axis=0))[0].take([0, -1]) + np.array([-margin, margin])
        if l1[0] < 0:
            l1[0] = 0
        if l1[1] >= sup.shape[1]:
            l1[1] = -1

        obj = obj[l0[0]:l0[1], l1[0]:l1[1]]
        sup = sup[l0[0]:l0[1], l1[0]:l1[1]]

    return obj, sup


if has_numba:
    @njit(parallel=True, cache=True)
    def interp_linear_numba(a: np.ndarray, dr, mask: np.ndarray = None):
        v = np.empty_like(a)
        m = mask
        w = np.ones_like(a, dtype=np.float32)
        if a.ndim == 2:
            ny, nx = a.shape
            y, x = dr
            y0, x0 = np.int(np.floor(y)), np.int(np.floor(x))
            dy, dx = y - y0, x - x0
            for iy in prange(ny):
                for ix in prange(nx):
                    a00 = a[(iy - y0) % ny, (ix - x0) % nx]
                    a01 = a[(iy - y0) % ny, (ix - x0 - 1) % nx]
                    a10 = a[(iy - y0 - 1) % ny, (ix - x0) % nx]
                    a11 = a[(iy - y0 - 1) % ny, (ix - x0 - 1) % nx]
                    if m is not None:
                        m00 = m[(iy - y0) % ny, (ix - x0) % nx]
                        m01 = m[(iy - y0) % ny, (ix - x0 - 1) % nx]
                        m10 = m[(iy - y0 - 1) % ny, (ix - x0) % nx]
                        m11 = m[(iy - y0 - 1) % ny, (ix - x0 - 1) % nx]

                        n = (1 - dy) * (1 - dx) * m00 + \
                            (1 - dy) * dx * m01 + \
                            dy * (1 - dx) * m10 + \
                            dy * dx * m11
                        w[iy, ix] = n
                        v[iy, ix] = (a00 * (1 - dy) * (1 - dx) * m00 +
                                     a01 * (1 - dy) * dx * m01 +
                                     a10 * dy * (1 - dx) * m10 +
                                     a11 * dy * dx * m11) / max(n, 1e-6)
                    else:
                        v[iy, ix] = a00 * (1 - dy) * (1 - dx) + \
                                    a01 * (1 - dy) * dx + \
                                    a10 * dy * (1 - dx) + \
                                    a11 * dy * dx
        elif a.ndim == 3:
            nz, ny, nx = a.shape
            z, y, x = dr
            z0, y0, x0 = np.int(np.floor(z)), np.int(np.floor(y)), np.int(np.floor(x))
            dz, dy, dx = z - z0, y - y0, x - x0
            for iz in prange(nz):
                for iy in prange(ny):
                    for ix in prange(nx):
                        a000 = a[(iz - z0) % nz, (iy - y0) % ny, (ix - x0) % nx]
                        a001 = a[(iz - z0) % nz, (iy - y0) % ny, (ix - x0 - 1) % nx]
                        a010 = a[(iz - z0) % nz, (iy - y0 - 1) % ny, (ix - x0) % nx]
                        a011 = a[(iz - z0) % nz, (iy - y0 - 1) % ny, (ix - x0 - 1) % nx]
                        a100 = a[(iz - z0 - 1) % nz, (iy - y0) % ny, (ix - x0) % nx]
                        a101 = a[(iz - z0 - 1) % nz, (iy - y0) % ny, (ix - x0 - 1) % nx]
                        a110 = a[(iz - z0 - 1) % nz, (iy - y0 - 1) % ny, (ix - x0) % nx]
                        a111 = a[(iz - z0 - 1) % nz, (iy - y0 - 1) % ny, (ix - x0 - 1) % nx]
                        if m is not None:
                            m000 = m[(iz - z0) % nz, (iy - y0) % ny, (ix - x0) % nx]
                            m001 = m[(iz - z0) % nz, (iy - y0) % ny, (ix - x0 - 1) % nx]
                            m010 = m[(iz - z0) % nz, (iy - y0 - 1) % ny, (ix - x0) % nx]
                            m011 = m[(iz - z0) % nz, (iy - y0 - 1) % ny, (ix - x0 - 1) % nx]
                            m100 = m[(iz - z0 - 1) % nz, (iy - y0) % ny, (ix - x0) % nx]
                            m101 = m[(iz - z0 - 1) % nz, (iy - y0) % ny, (ix - x0 - 1) % nx]
                            m110 = m[(iz - z0 - 1) % nz, (iy - y0 - 1) % ny, (ix - x0) % nx]
                            m111 = m[(iz - z0 - 1) % nz, (iy - y0 - 1) % ny, (ix - x0 - 1) % nx]

                            n = (1 - dz) * (1 - dy) * (1 - dx) * m000 + \
                                (1 - dz) * (1 - dy) * dx * m001 + \
                                (1 - dz) * dy * (1 - dx) * m010 + \
                                (1 - dz) * dy * dx * m011 + \
                                dz * (1 - dy) * (1 - dx) * m100 + \
                                dz * (1 - dy) * dx * m101 + \
                                dz * dy * (1 - dx) * m110 + \
                                dz * dy * dx * m111
                            w[iz, iy, ix] = n
                            v[iz, iy, ix] = (a000 * (1 - dz) * (1 - dy) * (1 - dx) * m000 +
                                             a001 * (1 - dz) * (1 - dy) * dx * m001 +
                                             a010 * (1 - dz) * dy * (1 - dx) * m010 +
                                             a011 * (1 - dz) * dy * dx * m011 +
                                             a100 * dz * (1 - dy) * (1 - dx) * m100 +
                                             a101 * dz * (1 - dy) * dx * m101 +
                                             a110 * dz * dy * (1 - dx) * m110 +
                                             a111 * dz * dy * dx * m111) / max(n, 1e-6)
                        else:
                            v[iz, iy, ix] = a000 * (1 - dz) * (1 - dy) * (1 - dx) + \
                                            a001 * (1 - dz) * (1 - dy) * dx + \
                                            a010 * (1 - dz) * dy * (1 - dx) + \
                                            a011 * (1 - dz) * dy * dx + \
                                            a100 * dz * (1 - dy) * (1 - dx) + \
                                            a101 * dz * (1 - dy) * dx + \
                                            a110 * dz * dy * (1 - dx) + \
                                            a111 * dz * dy * dx
        return v, w


def interp_linear(a: np.ndarray, dr, mask: np.ndarray = None, return_weight=False, use_numba=True):
    """ Perform a bi/tri-linear interpolation of a 2D/3D array.

    :param a: the 2D or 3D array to interpolate
    :param dr: the shift (along each dimension) of the array for the interpolation
    :param mask: the mask of values (True or >0 values are masked) to compute the interpolation
    :param return_weight: if true, also return the weight of the interpolation, which should be equal to 1 if no
                          point needed for the interpolation were masked, and a value between 0 and 1 otherwise.
    :return: a masked interpolated array
    """
    if has_numba and use_numba:
        v, w = interp_linear_numba(a, dr, mask)
        tmp = w < 1e-6
        if return_weight:
            return np.ma.masked_array(v, mask=tmp), w
        return np.ma.masked_array(v, mask=tmp)

    if mask is None:
        mask = np.ones_like(a, dtype=np.int8)
    else:
        mask = (mask == 0).astype(np.int8)

    if a.ndim == 2:
        ax = 0, 1
        y, x = dr
        y0, x0 = np.int(np.floor(y)), np.int(np.floor(x))
        dy, dx = y - y0, x - x0
        if y0 != 0 or x0 != 0:
            a = np.roll(a, [y0, x0], axis=ax)
            m = np.roll(mask, [y0, x0], axis=ax)
        else:
            m = mask
        ny, nx = a.shape
        # np.take() returns a view (faster) whereas np.roll() makes a copy
        ix01 = np.roll(np.arange(nx, dtype=np.int16), 1)
        a01, m01 = np.take(a, ix01, axis=1), np.take(m, ix01, axis=1)
        ix10 = np.roll(np.arange(ny, dtype=np.int16), 1)
        a10, m10 = np.take(a, ix10, axis=0), np.take(m, ix10, axis=0)
        a11, m11 = np.roll(a, [1, 1], axis=ax), np.roll(m, [1, 1], axis=ax)
        v = a * (1 - dy) * (1 - dx) * m \
            + a01 * (1 - dy) * dx * m01 \
            + a10 * dy * (1 - dx) * m10 \
            + a11 * dy * dx * m11
        w = (1 - dy) * (1 - dx) * m + (1 - dy) * dx * m01 \
            + dy * (1 - dx) * m10 + dy * dx * m11
    elif a.ndim == 3:
        ax = 0, 1, 2
        z, y, x = dr
        z0, y0, x0 = np.int(np.floor(z)), np.int(np.floor(y)), np.int(np.floor(x))
        dz, dy, dx = z - z0, y - y0, x - x0
        aa = np.roll(a, [z0, y0, x0], axis=ax)
        m = np.roll(mask, [z0, y0, x0], axis=ax)
        v = aa * (1 - dz) * (1 - dy) * (1 - dx) * m \
            + np.roll(aa, [0, 0, 1], axis=ax) * (1 - dz) * (1 - dy) * dx * np.roll(m, [0, 0, 1], axis=ax) \
            + np.roll(aa, [0, 1, 0], axis=ax) * (1 - dz) * dy * (1 - dx) * np.roll(m, [0, 1, 0], axis=ax) \
            + np.roll(aa, [0, 1, 1], axis=ax) * (1 - dz) * dy * dx * np.roll(m, [0, 1, 1], axis=ax) \
            + np.roll(aa, [1, 0, 0], axis=ax) * dz * (1 - dy) * (1 - dx) * np.roll(m, [1, 0, 0], axis=ax) \
            + np.roll(aa, [1, 0, 1], axis=ax) * dz * (1 - dy) * dx * np.roll(m, [1, 0, 1], axis=ax) \
            + np.roll(aa, [1, 1, 0], axis=ax) * dz * dy * (1 - dx) * np.roll(m, [1, 1, 0], axis=ax) \
            + np.roll(aa, [1, 1, 1], axis=ax) * dz * dy * dx * np.roll(m, [1, 1, 1], axis=ax)
        w = (1 - dz) * (1 - dy) * (1 - dx) * m + (1 - dz) * (1 - dy) * dx * np.roll(m, [0, 0, 1], axis=ax) \
            + (1 - dz) * dy * (1 - dx) * np.roll(m, [0, 1, 0], axis=ax) \
            + (1 - dz) * dy * dx * np.roll(m, [0, 1, 1], axis=ax) \
            + dz * (1 - dy) * (1 - dx) * np.roll(m, [1, 0, 0], axis=ax) \
            + dz * (1 - dy) * dx * np.roll(m, [1, 0, 1], axis=ax) \
            + dz * dy * (1 - dx) * np.roll(m, [1, 1, 0], axis=ax) \
            + dz * dy * dx * np.roll(m, [1, 1, 1], axis=ax)

    tmp = w < 1e-6
    if return_weight:
        return np.ma.masked_array(v / np.maximum(w, 1e-6 * tmp), mask=tmp), w
    return np.ma.masked_array(v / np.maximum(w, 1e-6 * tmp), mask=tmp)


def upsample(a: np.ndarray, bin_f, scale="sum", interp=1):
    """
     Inverse operation of rebin, for a 2 or 3-dimensional array. The array dimensions
     are multiplied by rebin_f along each dimension.
     The added values are linearly interpolated, assuming circular periodicity across boundaries. 

    Args:
        a: the array to resize
        bin_f: the bin factor - each pixel/voxel will be transformed in bin_f x bin_f (x bin_f) pixels. This can
                 also be a tuple/list of bin values along each axis, e.g. bin_f=(4,1,2) for a 3D array
        scale: if "sum" (the default), the array total will be kept.
            If "average", the average pixel value will be kept.
            If "square", (abs(a)**2).sum() is kept
        interp: if 1, a linear interpolation is used (which can be very slow). If 0, no interpolation.

    Returns: the new array with dimensions multiplied by bin_f

    """
    ndim = a.ndim
    if type(bin_f) is int:
        bin_f = [bin_f] * ndim
    else:
        assert ndim == len(bin_f), "upsample: number of dimensions does not agree with number of bin values:" + str(
            bin_f)

    assert ndim == 2 or ndim == 3, "upsample: only dimensions 2 and 3 are accepted"

    if ndim == 2:
        ny, nx = a.shape
        ny2, nx2 = bin_f[0] * ny, bin_f[1] * nx
        b = np.empty((ny2, nx2), dtype=a.dtype)
        for dy in range(bin_f[0]):
            for dx in range(bin_f[1]):
                if interp == 1:
                    # TODO: make this faster than using interp_linear
                    b[dy::bin_f[0], dx::bin_f[1]] = interp_linear(a, (-dy / bin_f[0], -dx / bin_f[1]), use_numba=False)
                else:
                    b[dy::bin_f[0], dx::bin_f[1]] = a

    else:  # ndim == 3:
        nz, ny, nx = a.shape
        nz2, ny2, nx2 = bin_f[0] * nz, bin_f[1] * ny, bin_f[1] * nx
        b = np.empty((nz2, ny2, nx2), dtype=a.dtype)
        for dz in range(bin_f[0]):
            for dy in range(bin_f[1]):
                for dx in range(bin_f[2]):
                    if interp == 1:
                        # TODO: make this faster than using interp_linear
                        b[dz::bin_f[0], dy::bin_f[1], dx::bin_f[2]] = \
                            interp_linear(a, (-dz / bin_f[0], -dy / bin_f[1], -dx / bin_f[2]), use_numba=False)
                    else:
                        b[dz::bin_f[0], dy::bin_f[1], dx::bin_f[2]] = a

    if scale.lower() == "sum":
        b = (b / np.prod(bin_f)).astype(a.dtype)
    elif "sq" in scale.lower():
        b = (b * np.sqrt((abs(a) ** 2).sum() / (abs(b) ** 2).sum())).astype(a.dtype)
    return b


def array_derivative(a, dr, mask=None, phase=False):
    """ Compute the derivative of an array along a given direction

    :param a: the complex array for which the derivative will be calculated
    :param dr: the shift in pixels (with a value along each of the array dimensions)
       the value returned will be (a(r+dr)-a[r-dr])/2/norm2(dr), or if
       one of the values is masked, e.g. (a(r+dr)-a[r])/norm2(dr)
    :param phase: if True, will return instead the derivative of the phase of
       the supplied complex array. Will return an error if the supplied
       array is not complex.
    :return: a masked array of the gradient.
    """
    dr = np.array(dr, dtype=np.float32)
    a0, w0 = np.ma.masked_array(a, mask=mask), (mask == 0).astype(np.float32)
    ap, wp = interp_linear(a, -dr, mask=mask, return_weight=True)
    am, wm = interp_linear(a, dr, mask=mask, return_weight=True)
    n = np.sqrt((dr ** 2).sum())
    if phase:
        d = np.angle(ap / am) / (2 * n) * wp * wm + np.angle(ap / a0) / n * wp * w0 * (1 - wm) + np.angle(
            a0 / am) / n * w0 * wm * (1 - wp)
    else:
        d = (ap - am) / (2 * n) * wp * wm + (ap - a0) / n * wp * w0 * (1 - wm) + (a0 - am) / n * w0 * wm * (1 - wp)
    m = (wp * wm + wp * w0 * (1 - wm) + w0 * wm * (1 - wp)) == 0
    return np.ma.masked_array(d, mask=m)


def fourier_shift(a, shift, axes=None, positivity=False):
    """ Sub-pixel shift of an array. The return type will be the same as the input.

    :param a: the array to shift, with N dimensions
    :param shift: the shift along each axis
    :param axes: a tuple of the axes along which the shift is performed.
                 If None, all axes are transformed
    :param positivity: if True, all values <0 will be set to zero for the output
    :return: the fft-shifted array
    """
    if axes is None:
        axes = range(a.ndim)
    shifts = np.zeros(a.ndim, dtype=np.float32)
    assert len(axes) == len(shift)
    for i in range(len(axes)):
        shifts[axes[i]] = shift[i]
    af = fftn(a, axes=axes)
    xi = [fftfreq(a.shape[i]) * shifts[i] for i in range(a.ndim)]
    k = np.array(np.meshgrid(*xi, indexing='ij')).sum(axis=0)
    af *= np.exp(-2j * np.pi * k)
    r = ifftn(af, axes=axes)
    if r.dtype != a.dtype:
        if a.dtype in [np.float, np.float32, np.float64]:
            r = r.real.astype(a.dtype)
            if positivity:
                r[r < 0] = 0
        else:
            r = r.astype(a.dtype)
    return r


def pad(a: np.ndarray, padding=None, padding_f=None, stack=False, value=0, shift=False):
    """

    :param a: the array to pad (2D, 2D stack, 3D)
    :param padding: the number of pixels to add on each border. if this is an integer, the same
        margin is added on each border. This can be a list/tuple with a size the number of dimensions,
        so each dimension has a different margin on each side. Finally, there can be twice the number
        of dimensions so each side of each dimension uses a different margin.
    :param padding_f: instead of giving a number of pixels to add, it is possible to give
        a factor by which the array size will be multiplied. This can either be an integer
        for all dimentions, or one value for eachi dimension.
        Ignored if padding is given.
    :param stack: if True and the array is 3D, pad it as a stack of 2D arrays. In this case,
        padding should only contain the values for the two dimensions.
    :param value: the value with which to fill the padded values
    :param shift: if True, the inpu array is assumed to be fft-shifted, and will be centred
        before padding, and fft-shifted again upon return
    :return: the padded array
    """
    if shift:
        if stack:
            a = fftshift(a, axes=(-2, -1))
        else:
            a = fftshift(a)
    if padding is None:
        s = a.shape
        padding = []
        if isinstance(padding_f, int) or isinstance(padding_f, np.integer):
            for i in range(a.ndim):
                d = s[i] * (padding_f - 1)
                padding += [d // 2, d - d // 2]
        else:
            for i in range(a.ndim):
                d = s[i] * (padding_f[i] - 1)
                padding += [d // 2, d - d // 2]
    if isinstance(padding, int) or isinstance(padding, np.integer):
        n = [padding] * (2 * (a.ndim - int(stack)))
    elif len(padding) == a.ndim - int(stack):
        n = []
        for v in padding:
            n += [v, v]
    else:
        n = padding
    assert len(n) == 2 * (a.ndim - int(stack)), "pad(): the number of dimensions does not match the array"
    if a.ndim == 2:
        ny, nx = a.shape[-2:]
        tmp = np.ones((ny + n[0] + n[1], nx + n[2] + n[3]), dtype=a.dtype) * value
        tmp[n[0]:n[0] + ny, n[2]:n[2] + nx] = a
    elif a.ndim == 3 and stack is False:
        nz, ny, nx = a.shape
        tmp = np.ones((nz + n[0] + n[1], ny + n[2] + n[3], nx + n[4] + n[5]), dtype=a.dtype) * value
        tmp[n[0]:n[0] + nz, n[2]:n[2] + ny, n[4]:n[4] + nx] = a
    else:
        ny, nx = a.shape[-2:]
        tmp = np.ones(list(a.shape[:-2]) + [ny + n[0] + n[1], nx + n[2] + n[3]], dtype=a.dtype) * value
        tmp[..., n[0]:n[0] + ny, n[2]:n[2] + nx] = a
    if shift:
        if stack:
            tmp = fftshift(tmp, axes=(-2, -1))
        else:
            tmp = fftshift(tmp)
    return tmp


def crop(a: np.ndarray, margin=None, margin_f=None, shift=False):
    """
    Crop an array by removing values on each side

    :param a: the array to crop
    :param margin: the margin to crop. This can either be:

        * an integer number (same number of pixels removed on each side and dimension)
        * a list/tuple of integers, one for each dimension. The same margin is applied
          on both sides
        * a list/tuple of integers, two for each dimension. A different margin can be
          applied on both sides.

    :param margin_f: instead of giving the margin on each side as a number of pixels,
        it is possible to give it as a integer factor by which the final size will be divided,
        e.g. if margin_f=2, the size is divided by 2, and the margin is equal to 1/4 of
        the original size, for each dimension. This can be set either as an integer, with
        the same factor for each dimension, or as a list with one value for each dimension.
        Ignored if margin is given.
    :param shift: if True, the input array is assumed to be fft-shifted  (centred on
        the first array element), and will be shifted for cropping, and fft-shifted again
        for the output.
    :return: the cropped array
    """
    if shift:
        a = fftshift(a)
    s = np.array(a.shape, dtype=np.int)
    n = []
    if margin is not None:
        if isinstance(margin, int) or isinstance(margin, np.integer):
            for v in s:
                n += [margin, v - margin]
        elif len(margin) == a.ndim:
            for i in range(a.ndim):
                n += [margin[i], s[i] - margin[i]]
        else:
            for i in range(a.ndim):
                n = [margin[2 * i], s[i] - margin[2 * i + 1]]
    else:
        if isinstance(margin_f, int) or isinstance(margin_f, np.integer):
            d = s // margin_f
        else:
            d = s // np.array(margin_f, dtype=np.int)
        n = []
        for i in range(a.ndim):
            n += [s[i] // 2 - d[i] // 2, s[i] // 2 - d[i] // 2 + d[i]]
    assert len(n) == 2 * a.ndim, "crop(): the number of dimensions does not match the array"
    if a.ndim == 2:
        a = a[n[0]:n[1], n[2]:n[3]]
    elif a.ndim == 3:
        a = a[n[0]:n[1], n[2]:n[3], n[4]:n[5]]
    elif a.ndim == 4:
        a = a[n[0]:n[1], n[2]:n[3], n[4]:n[5], n[6]:n[7]]
    else:
        raise Exception("crop(): only dimensions 2,3,4 are supported.")
    if shift:
        return fftshift(a)
    else:
        return a
