# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2013-2014 : Fondation Nanosciences, Grenoble
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#         Ondrej Mandula
from __future__ import division

__all__ = ['phase_tilt', 'grad_phase', 'phase_diff', 'minimize_grad_phase', 'shift_phase_zero', 'unwrap_phase',
           'remove_phase_ramp']

import warnings
import numpy as np
from numpy import pi
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.optimize import minimize
from scipy.signal import medfilt2d
from scipy.integrate import cumtrapz
from scipy.ndimage import fourier_shift
from skimage.restoration import unwrap_phase as sk_unwrap_phase
try:
    from skimage.registration import phase_cross_correlation as register_translation
except ImportError:
    from skimage.feature import register_translation
from skimage import __version__ as skimage_version
from packaging.version import parse as version_parse
from .array import rebin


def phase_tilt(im, grad=(0, 0), offset=0):
    """
    Multiplies the phase of the complex image by the gradient and adds phase offset.
    DEPRECATED. scipy.ndimage.fourier_shift should be used instead

    Args:
        im: input image
        grad: gradient of the phase as a multiple of 2*pi across the image
        offset: offset of the phase as a multiple of 2*pi
    """

    x = np.linspace(-pi, pi, im.shape[1]).astype(np.float32)
    y = np.linspace(-pi, pi, im.shape[0]).astype(np.float32)
    yy, xx = np.meshgrid(x, y)
    return im * np.exp(1j * (xx * grad[0] + yy * grad[1] + offset * 2 * pi))


def grad_phase(p, im):
    # oPhaseMasked = np.ma.masked_array(np.angle(im*np.exp(1j*2*np.pi*(xx*p[0]/im.shape[0]+yy*p[1]/im.shape[1]))),-mask)
    # oPhaseMasked = np.ma.masked_array(np.angle(im*np.exp(1j*(xx*p[0]+yy*p[1]))),-mask)
    oPhaseMasked = np.angle(phase_tilt(im, p))
    dx, dy = np.gradient(oPhaseMasked)
    #    return sum(abs(dx)+abs(dy))
    return np.median(abs(dx) + abs(dy))


def phase_diff(p, im, x, y):
    gx, gy = np.gradient(im - (x * p[0] + y * p[1]), 10)
    # This should allow to ignore 2pi phase jumps
    gx *= gx < 0.05
    gy *= gy < 0.05
    return (abs(gx) ** 2 + abs(gy) ** 2).sum()


def minimize_grad_phase(im, mask_thr=0.3, center_phase=None, global_min=False, mask=None, rebin_f=None):
    """
    Minimises the gradient in the phase of the input image im.

    Args:
        im: the input complex 2D image for which the phase gradient needs to be minimized
        mask_thr: only pixel amplitudes above mask_thr*abs(im).max() will be taken into account
        center_phase: after minimizing the gradient, center the phase around this value (in radians). pi/3 gives a nice contrast
        global_min: if True after a quick local minimization of the gradient, the phase differences will be minimized over the whole object.
                    Use this for flat-phased objects
        mask: a 2d mask array (True = masked points) giving the pixels which should be excluded from the optimization.
              This is cumulative with mask_thr
        rebin_f: an integer>1 can be given to compute the phase gradient on a rebinned array for faster evaluation.
    Returns:
        a tuple with (phase corrected image, correction array, mask, linear coefficients)
    """
    if mask is not None:
        mask0 = mask
        mask = np.logical_or((abs(im) / abs(im * ~mask).max()) < mask_thr, mask)
        if (~mask).sum() < 1000:
            mask = mask0
    else:
        mask = (abs(im) / abs(im).max()) < mask_thr
        if (~mask).sum() < 1000:
            mask = np.zeros(im.shape, dtype=np.bool)
    mask = mask.astype(np.bool)

    if rebin_f is not None:
        im0 = im
        mask0 = mask
        im = rebin(im, rebin_f, scale="sum")
        m = rebin((mask == 0).astype(np.int8), rebin_f, scale="sum")
        im /= m + (m == 0)
        mask = m == 0

    ny, nx = im.shape
    gradient_step = 10
    g = np.gradient(np.angle(im), gradient_step)
    maskx = np.logical_or(abs(g[1]) > 0.05, mask)
    if (~maskx).sum() < 1000:
        maskx = mask
    masky = np.logical_or(abs(g[0]) > 0.05, mask)
    if (~masky).sum() < 1000:
        masky = mask
    gx = np.percentile(g[1][~maskx], 50) * nx / (2 * pi) * gradient_step
    gy = np.percentile(g[0][~masky], 50) * ny / (2 * pi) * gradient_step
    grad = [-gy, -gx]

    if global_min:
        # Now minimize global phase gradient - only using unmasked pixels
        im2 = phase_tilt(im, grad=grad)
        im2 = np.exp(1j * np.angle(im2))
        x = np.linspace(-pi, pi, im2.shape[1]).astype(np.float32)
        y = np.linspace(-pi, pi, im2.shape[0]).astype(np.float32)
        y, x = np.meshgrid(x, y)
        im2m = np.ma.masked_array(im2, mask=mask)
        # Crop optimized area to unmasked region for faster optimization
        ix0, ix1 = np.nonzero((~mask).sum(axis=0))[0][[0, -1]]
        dx = (ix1 - ix0) // 2 + 10
        x0 = (ix1 + ix0) // 2
        iy0, iy1 = np.nonzero((~mask).sum(axis=1))[0][[0, -1]]
        dy = (iy1 - iy0) // 2 + 10
        y0 = (iy1 + iy0) // 2
        im2m = im2m[y0 - dy:y0 + dy, x0 - dx:x0 + dx]
        y = y[y0 - dy:y0 + dy, x0 - dx:x0 + dx]
        x = x[y0 - dy:y0 + dy, x0 - dx:x0 + dx]
        # Minimization
        res = minimize(phase_diff, (0, 0), args=(im2m, x, y), method='Powell', options={'xtol': 1e-18, 'disp': False})
        grad = [grad[0] + res['x'][0], grad[1] + res['x'][1]]

    if rebin_f is not None:
        im = im0
        mask = mask0
    imnew = phase_tilt(im, grad)

    corr = phase_tilt(np.ones(im.shape), grad)
    if center_phase is not None:
        imm = imnew[mask == 0]
        # Make sure we find the correct center for the phase, independent of any wrapping..
        i = np.argmin([np.std(np.angle(imm * np.exp(-1j * pi + 2j * pi * i / 8))) for i in range(0, 8)])
        imm = np.angle(imm * np.exp(-1j * pi + 2j * pi * i / 8))
        tmp = np.exp(1j * (center_phase - np.median(imm)) - 1j * np.pi + 2j * np.pi * i / 8)
        imnew *= tmp
        corr *= tmp
    return imnew, corr, mask, grad


def remove_phase_ramp(d: np.ndarray, mask_thr=None, center_phase=None, mask=None, niter=2):
    """
     Removes a linear ramp in the phase of the input array (2D or 3D).
     This uses scikit-image register_translation in Fourier space for a fast evaluation, by calculating the shift
     of FT(d) compared to FT(abs(d)). 3D correction requires scikit-image>=0.15.0.

     WARNING: This works efficiently on a pure phase ramp, with some noise, but cannot be used in general - e.g. on an
     object with a phase vortex, FT(d) and FT(abs(d)) can be quite different and cannot easily be correlated.

    :param d: the data array for which the phase ramp must be removed.
    :param mask_thr: only pixel amplitudes above mask_thr*abs(im).max() will be taken into account
    :param center_phase: after minimizing the gradient, center the phase around this value (in radians).
                         pi/3 gives a nice contrast.
    :param mask: a mask array (True = masked points) giving the pixels which should be excluded from the optimization.
                 This is cumulative with mask_thr. Masked pixels are corrected, but not used to evaluate the phase
                 gradient.
    :param niter: number of iterations to perform to remove ramp d (default=2)
    :return: a tuple with (phase corrected array, linear coefficients). The linear
             coefficients correspond to the total amplitude of the ramp along the given axis, in radians.
    """
    dm = d
    if mask is not None:
        dm = dm * (mask > 0)
    if mask_thr is not None:
        dm *= abs(dm) >= mask_thr
    upsample_factor = 20

    if dm.ndim == 3:
        assert version_parse(skimage_version) >= version_parse('0.15'), \
            'scikit-image>=0.15.0 is needed for 3D remove_phase_ramp'

    dmaf = fftshift(fftn(fftshift(np.abs(dm))))
    coeffs = np.zeros(dm.ndim, dtype=np.float32)

    for i in range(niter):
        phase_power = 20 ** (i + 1)
        dmf = fftshift(fftn(fftshift(np.abs(dm) * np.exp(1j * np.angle(dm) * phase_power))))
        dmf = dmf.astype(np.complex64)
        # Find phase ramp using register_translation in Fourier space
        s, err, dphi = register_translation(dmaf, dmf, upsample_factor=upsample_factor)
        s = -np.array(s) / phase_power
        coeffs += s
        # Correct array using fourier_shift
        dm = fftshift(fourier_shift(fftshift(dm), s))
        # Same correction without fourier_shift (in 3D):
        # nz, ny, nx = dm.shape
        # z, y, x = np.meshgrid(np.arange(nz) / nz - 0.5, np.arange(ny) / ny - 0.5, np.arange(nx) / nx - 0.5,
        #                       indexing='ij')
        # phase_corr = [(v * 2 * np.pi) for v in s]
        # dm = dm * np.exp(1j * (z * phase_corr[0] + y * phase_corr[1] + x * phase_corr[2] + dphi1))

    # Final correction
    dcorr = fftshift(fourier_shift(fftshift(d), coeffs))

    if center_phase is not None:
        dm = d
        if mask is not None:
            dm = dm * (mask > 0)
        if mask_thr is not None:
            dm *= abs(dm) >= mask_thr
        # Is that the most efficient phase centering ?
        dcorr *= np.exp(-1j * np.angle(dm.sum()))

    return dcorr.astype(d.dtype), coeffs * 2 * np.pi


def shift_phase_zero(obj, percent=5, mask=None, origin=0, stack=True, verbose=False):
    """
    Shift the phase of the given object, so that the phase range begins at 0 radians, and hopefully avoid wrapped
    phases. If the percentile range from (percent, 100-percent) cannot be lowered to less than 5.5 radians minus
    twice percent*2*pi/199, it is assumed that the object has phase wrapping and no correction can be made, so the
    object is returned unchanged.

    :param obj: the complex object for which a shift of the phase will be calculated. If the object is 3D, the
                phase shift is evaluated only on the first mode obj[0].
    :param percent: the range of the phase will be evaluated using np.percentile from 'percent' to '100-percent'.
                    This is used to avoid the influence of noise
    :param mask: if given, only the non-masked area (for which mask > 0) will be taken into account
    :param origin: the desired origin of the phase. 0 by default, but e.g. -3 can be used for a [-pi;pi] display
    :param stack: if True, treat the object as a stack of 2D object (e.g. the object modes in ptychography),
        and calculate the phase shift on the first layer, but apply to all layers.
        if True, the mask should be 2D
    :return: the object corrected by a constant phase shift, i.e. obj * exp(1j * dphi)
    """
    if obj.ndim == 3 and stack:
        p = np.angle(obj[0])
    else:
        p = np.angle(obj)
    if mask is not None:
        p = p[mask > 0]
    # Get the best phase offset, trying different origins
    vdphi = []
    vphirange = []
    for dphi in np.linspace(-np.pi, np.pi, 10):
        tmp = (p + dphi) % (2 * np.pi)
        vdphi.append([dphi, np.percentile(tmp, (percent, 100 - percent))])
        vphirange.append(vdphi[-1][1][1] - vdphi[-1][1][0])
    dphi, perc = vdphi[np.argmin(vphirange)]
    phi_range = perc[1] - perc[0]
    max_range = 5.5 - 2 * np.pi * percent / 100.
    if verbose:
        print("shift_phase_zero: dphi=%6.3f, phi range=%6.3f (<%6.3f ??)" % (dphi, phi_range, max_range))
    if phi_range < max_range:
        return obj * np.exp(1j * (dphi - perc[0] + origin))
    else:
        return obj


def unwrap_phase(d: np.array, method='skimage', medfilt_n=3):
    """
    Extract an unwrap the phase from a complex data array. Currently implemented for 2D only.

    :param d: the complex data array to be unwrapped
    :param method: either 'skimage' to use skimage.restoration.unwrap_phase, or 'gradient-x' to use the normalised
        complex array gradient to determine the smooth phase gradient and then integrate along the x-axis
        (last dimension), or 'gradient-y' to use a gradient and final integration along y.
    :param medfilt_n: size of the median kernel filtering on the initial array or phase. This is only used to
        evaluate the phase wrapping - the final returned phase array is not filtered.
    :return: the unwrapped phase in radians.
    """
    if medfilt_n is not None:
        df = (medfilt2d(d.real, medfilt_n) + 1j * medfilt2d(d.imag, medfilt_n))
        for i in range(medfilt_n):
            df[i] = df[medfilt_n]
            df[-i] = df[-medfilt_n]
        for i in range(medfilt_n):
            df[:, i] = df[:, medfilt_n]
            df[:, -i] = df[:, -medfilt_n]
    else:
        df = d
    if 'gradient' in method.lower():
        ny, nx = df.shape
        dn = df / (abs(df) + 1e-12 * (df < 1e-12))
        gy, gx = np.gradient(dn, edge_order=1) / dn
        if 'x' in method.lower():
            phx = cumtrapz(gx.imag[0], axis=0, initial=0)
            ph = cumtrapz(gy.imag, axis=0, initial=0) + phx
        else:
            phy = cumtrapz(gy.imag[:, 0], axis=0, initial=0)
            ph = cumtrapz(gx.imag, axis=1, initial=0) + phy.repeat(gx.shape[1]).reshape(gx.shape)
    else:  # method.lower() == 'skimage'
        ph = sk_unwrap_phase(np.angle(df))

    # Now compute the unfiltered unwrapped phase
    a = np.angle(d)
    a += np.round((ph - a) / (2 * np.pi)) * (2 * np.pi)
    return a
