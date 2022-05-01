#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from __future__ import division

import sys
import time
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
from matplotlib.figure import Figure
from pynx.utils.matplotlib import pyplot as plt
from pynx import wavefront
from pynx.wavefront import PropagateNearField
from pynx.utils.plot_utils import complex2rgbalin, colorwheel
from pynx.utils.math import ortho_modes, full_width
from scipy.ndimage import measurements


def probe_propagate(probe, propagation_range, pixel_size, wavelength, do_plot=True, show_plot=True, fig_size=(15, 10)):
    """
    Propagate a probe along a given range, plot it and return the probe at the focus (position where the size is minimal)
    
    Args:
        probe: the 2D complex probe to propagate
        propagation_range: either given as a range/array of dz values (in meters), or a tuple with (zmin,zmax,nb)
        pixel_size: the probe pixel size, in meters
        do_plot: plot if True (default)
        show_plot: if True, the plot will be done in a visible figure, otherwise in an offline one. 
                   Ignored unless do_plot is True
        fig_size: the figure size, default = (15,10)

    Returns:
        A tuple with:
            - the 3d array of the propagated probe along z, with size (nz, ny, nx)
            - the z coordinates along the 3d array
            - the index of the found focus point along the z direction
            - the figure if it was plotted
    """
    ny, nx = probe.shape
    if type(propagation_range) is tuple:
        vdz = np.linspace(propagation_range[0], propagation_range[1], propagation_range[2])
    else:
        vdz = propagation_range
    nz = len(vdz)
    p = np.empty((nz, ny, nx), dtype=np.complex64)
    fwhm_v = np.empty(nz, dtype=np.float32)
    fwhm_h = np.empty(nz, dtype=np.float32)
    w = wavefront.Wavefront(d=fftshift(probe.astype(np.complex64)), wavelength=wavelength, pixel_size=pixel_size)
    i = 0
    px = pixel_size * 1e6 / 2
    vx = np.arange(-nx * px, nx * px, px * 2)  # in µm, we don't care about origin yet
    vy = np.arange(-ny * px, ny * px, px * 2)

    sys.stdout.write('Propagating probe: ')
    for dz in vdz:
        if (nz - i) % 20 == 0:
            sys.stdout.write('%d ' % (nz - i))
            sys.stdout.flush()

        wz = PropagateNearField(dz) * w.copy()
        p[i] = wz.get(shift=True)
        iymax, ixmax = measurements.maximum_position(abs(p[i]))
        fwhm_h[i] = full_width(vx, abs(p[i, int(round(iymax))]) ** 2, ratio=0.2, outer=True) * 1e3
        fwhm_v[i] = full_width(vy, abs(p[i, :, int(round(ixmax))]) ** 2, ratio=0.2, outer=True) * 1e3
        i += 1
    print()

    # Find focus
    # Use maximum of variance ?
    # var = [np.var(abs(d)**2) for d in p]
    # izmax = np.argmax(var)
    # Use the 'width' of the probe calculated from a 2D gaussian model
    width_stat = np.array([2.35 * np.sqrt((abs(d) ** 2).sum() / ((abs(d) ** 2).max() * 2 * np.pi)) for d in p])
    width_statx = np.array(
        [2.35 * (abs(d) ** 2).sum() / (abs(d) ** 2).sum(axis=0).max() / np.sqrt(2 * np.pi) for d in p])
    width_staty = np.array(
        [2.35 * (abs(d) ** 2).sum() / (abs(d) ** 2).sum(axis=1).max() / np.sqrt(2 * np.pi) for d in p])
    izmax = np.argmin(width_stat)
    izmaxx = np.argmin(width_statx)
    izmaxy = np.argmin(width_staty)

    # Find peak value of phase and use this as origin
    p /= np.exp(1j * np.angle(p[izmax][abs(p[izmax]) > (abs(p[izmax]).max() * 0.8)]).mean())

    # Find XY coordinates of maximum
    # iymax, ixmax = measurements.center_of_mass(abs(p[izmax]))
    iymax, ixmax = measurements.maximum_position(abs(p[izmax]))

    ax = abs(p[izmax, int(round(iymax))]) ** 2
    ay = abs(p[izmax, :, int(round(ixmax))]) ** 2
    fwhmx, fwhmy = full_width(vx, ax) * 1e3, full_width(vy, ay) * 1e3  # in nm
    fm = max(fwhmx, fwhmy)

    if do_plot:
        if show_plot:
            try:
                fig = plt.figure(201, figsize=fig_size)
            except:
                # no GUI or $DISPLAY
                fig = Figure(figsize=fig_size)
        else:
            fig = Figure(figsize=fig_size)
        # fig.clf()  # Assume figure has already been cleared
        fig_ax = fig.add_subplot(321)
        fig_ax.imshow(complex2rgbalin(p[izmax]), aspect='equal', extent=(nx * px, -nx * px, -ny * px, ny * px))
        fig_ax.set_title(u'Probe @focus (z=%6.2f µm))' % (vdz[izmax] * 1e6))
        # Narrow display range if probe is much smaller than the field of view
        if (5 * fm / 1e3) < (nx * px):
            fig_ax.set_xlim(5 * fm / 1e3, -5 * fm / 1e3)
        if (5 * fm / 1e3) < (ny * px):
            fig_ax.set_ylim(-5 * fm / 1e3, 5 * fm / 1e3)
        fig_ax.set_xlabel(u'x (µm)')
        fig_ax.set_ylabel(u'y (µm)')

        fig_ax = fig.add_subplot(322)
        vx = np.arange(-nx, nx, 2, dtype=np.float32) * px + (nx // 2 - ixmax) * px * 2  # in µm
        vy = np.arange(-ny, ny, 2, dtype=np.float32) * px + (ny // 2 - iymax) * px * 2  # in µm
        fig_ax.plot(-vx * 1e3, ax, 'r', vy * 1e3, ay, 'b')
        fig_ax.legend(('X: FWHM=%6.0f nm' % (fwhmx), 'Y: FWHM=%6.0f nm' % (fwhmy)), fontsize=8)
        fig_ax.set_xlabel('x, y (nm)')
        fig_ax.set_ylabel('abs(probe)**2')
        fig_ax.set_xlim(-5 * fm, 5 * fm)
        fig_ax.set_title('Probe intensity profile @ focus')

        fig_ax = fig.add_subplot(323)
        fig_ax.imshow(complex2rgbalin(p[:, iymax].transpose()), aspect='auto',
                      extent=(vdz.min() * 1e6, vdz.max() * 1e6, -nx * px, nx * px))
        # fig_ax.set_xlabel('z - propagation direction (µm)')
        fig_ax.plot([vdz[izmax] * 1e6] * 2, fig_ax.get_ylim(), 'k--')
        fig_ax.plot([vdz[izmaxx] * 1e6] * 2, fig_ax.get_ylim(), 'r--')
        fig_ax.set_xlim(vdz.min() * 1e6, vdz.max() * 1e6)
        fig_ax.set_ylabel(u'x (µm)')
        fig_ax.text(.99, .99, 'Horizontal focusing', horizontalalignment='right', verticalalignment='top',
                    transform=fig_ax.transAxes)

        fig_ax = fig.add_subplot(324)
        fig_ax.plot(vdz * 1e6, fwhm_h, 'r-', vdz * 1e6, fwhm_v, 'b-')
        fig_ax.set_ylabel("Full width [abs(probe)**2] (nm)")
        fig_ax.legend(("FW@20% - horiz", "FW@20% - vert."))
        fig_ax.set_ylim(0)
        fig_ax.plot([vdz[izmax] * 1e6] * 2, fig_ax.get_ylim(), 'k--')
        fig_ax.set_xlim(vdz.min() * 1e6, vdz.max() * 1e6)

        fig_ax = fig.add_subplot(325)
        fig_ax.imshow(complex2rgbalin(p[:, :, ixmax].transpose()), aspect='auto',
                      extent=(vdz.min() * 1e6, vdz.max() * 1e6, -ny * px, ny * px))
        fig_ax.set_xlabel(u'z - propagation direction (µm)')
        fig_ax.plot([vdz[izmax] * 1e6] * 2, fig_ax.get_ylim(), 'k--')
        fig_ax.plot([vdz[izmaxy] * 1e6] * 2, fig_ax.get_ylim(), 'b--')
        fig_ax.set_xlim(vdz.min() * 1e6, vdz.max() * 1e6)
        fig_ax.set_ylabel(u'y (µm)')
        fig_ax.text(.99, .99, 'Vertical focusing', horizontalalignment='right', verticalalignment='top',
                    transform=fig_ax.transAxes)

        fig_ax = fig.add_subplot(326)
        fig_ax.plot(vdz * 1e6, width_statx * pixel_size * 1e9, 'r-')
        fig_ax.plot(vdz * 1e6, width_staty * pixel_size * 1e9, 'b-')
        fig_ax.plot(vdz * 1e6, width_stat * pixel_size * 1e9, 'k-')
        fig_ax.set_ylabel("statistical width (nm)")
        fig_ax.set_xlabel(u'z - propagation direction (µm)')
        fig_ax.legend(("Statistical FWHM[H]", "Statistical FWHM[V]", "Statistical FWHM"))
        fig_ax.set_ylim(0)
        fig_ax.plot([vdz[izmax] * 1e6] * 2, fig_ax.get_ylim(), 'k--')
        fig_ax.plot([vdz[izmaxy] * 1e6] * 2, fig_ax.get_ylim(), 'b--')
        fig_ax.plot([vdz[izmaxx] * 1e6] * 2, fig_ax.get_ylim(), 'r--')
        fig_ax.set_xlim(vdz.min() * 1e6, vdz.max() * 1e6)

        fig_ax = fig.add_axes((0.02, 0.90, 0.06, 0.06), facecolor='w')
        colorwheel(ax=fig_ax)
        return p, vdz, izmax, fig

    return p, vdz, izmax


def modes(d, pixel_size, do_plot=True, show_plot=True, verbose=False):
    """
    Determine complex modes for a given object or probe.
    
    Args:
        d: the probe or object to analyse, with the modes along the first axis/
        pixel_size: the pixel size in meters
        do_plot: plot if True (default)
        show_plot: if True, the plot will be done in a visible figure, otherwise in an offline one.
                   Ignored unless do_plot is True.

    Returns:
        a tuple of (orthogonalized modes with the same shape as the input, a vector of the relative intensities, figure)
    """
    d = ortho_modes(d)
    a = (abs(d) ** 2).sum(axis=(1, 2))
    a /= a.sum()
    if verbose:
        sys.stdout.write("Orthogonalised modes relative intensities: ")
        for aa in a:
            sys.stdout.write(" %5.2f%%" % (aa * 100))
        print("\n")

    if do_plot:
        nmode, ny, nx = d.shape
        px = pixel_size * 1e6 / 2
        nw, nh = (nmode >= 5) * 5 + (nmode < 5) * nmode, 1 + nmode // 5
        if show_plot:
            try:
                fig = plt.figure(202, figsize=(nw * 4, nh * 4))
            except:
                # no GUI or $DISPLAY
                fig = Figure(figsize=(nw * 4, nh * 4))
        else:
            fig = Figure(figsize=(nw * 4, nh * 4))
        fig.clf()
        for i in range(nmode):
            fig_ax = fig.add_subplot(nh, nw, i + 1)
            fig_ax.imshow(complex2rgbalin(d[i]), extent=(nx * px, -nx * px, -ny * px, ny * px))
            fig_ax.set_title("Mode #%d, %5.2f%% intensity" % (i, a[i] * 100))
            fig_ax.set_xlabel(u'x (µm)')
            if i == 0:
                fig_ax.set_ylabel(u'y (µm)')
        return d, a, fig
    return d, a


def probe_fwhm(probe, pixel_size, verbose=True):
    """
    Analyse probe shape and estimated FWHM using different methods:

    1. Full width at half maximum
    2. Full width at 20%
    3. Full width at half maximum using a statistical gaussian analysis - the width is the same as a gaussian
       with a maximum intensity of 1 and the same integrated intensity.
    
    Args:
        probe: the probe to analyse. If number of dimensions is 3, only the first mdoe is analysed
        pixel_size: the pixel size in meters
        verbose: if True (the default), will print the corresponding sizes

    Returns:
        ((fwhm_x, fwhm_y), (fwhm_x@20%, fwhm_y@20%), (fwhm_gauss_stat, fwhm_gauss_stat_x, fwhm_gauss_stat_y))
    """
    if probe.ndim == 2:
        p = probe
    else:
        p = probe[0]
    ny, nx = p.shape
    px = pixel_size * 1e6 / 2
    vx = np.arange(-nx * px, nx * px, px * 2)  # in µm, we don't care about origin yet
    vy = np.arange(-ny * px, ny * px, px * 2)
    iymax, ixmax = measurements.maximum_position(abs(p))

    ax = abs(p[int(round(iymax))]) ** 2
    ay = abs(p[:, int(round(ixmax))]) ** 2
    fwhmx, fwhmy = full_width(vx, ax) * 1e3, full_width(vy, ay) * 1e3  # in nm
    fwhmx20, fwhmy20 = full_width(vx, ax, ratio=0.2, outer=True) * 1e3, full_width(vy, ay, ratio=0.2,
                                                                                   outer=True) * 1e3  # in nm
    # Statistical width by comparing the maximum intensity to the integrated one, for a Gaussian
    width_stat = 2.35 * np.sqrt((abs(p) ** 2).sum() / ((abs(p) ** 2).max() * 2 * np.pi))
    px2 = (abs(p) ** 2).sum(axis=0)
    py2 = (abs(p) ** 2).sum(axis=1)
    width_stat_x = 2.35 * px2.sum() / px2.max() / np.sqrt(2 * np.pi)
    width_stat_y = 2.35 * py2.sum() / py2.max() / np.sqrt(2 * np.pi)
    if verbose:
        print("  FWHM (peak intensity): %8.2fnm(H) x%8.2fnm(V)" % (fwhmx, fwhmy))
    print("  FW @20%% intensity    : %8.2fnm(H) x%8.2fnm(V)" % (fwhmx20, fwhmy20))
    print("  FWHM (statistical)   : %8.2fnm(H) x%8.2fnm(V), average=%8.2fnm" %
          (width_stat_x * pixel_size * 1e9, width_stat_y * pixel_size * 1e9, width_stat * pixel_size * 1e9))
    return ((fwhmy * 1e-9, fwhmx * 1e-9), (fwhmy20 * 1e-9, fwhmx20 * 1e-9),
            (width_stat_x * pixel_size, width_stat_y * pixel_size, width_stat * pixel_size))
