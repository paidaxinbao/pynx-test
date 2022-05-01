# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2013-2014 : Fondation Nanosciences, Grenoble
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#         Ondrej Mandula

import warnings
import numpy as np
from numpy import pi
from pynx.utils.matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from ..utils.phase import minimize_grad_phase

cm_phase_cdict = {'red':
                      ((0.0, 0.0, 0.0),
                       (1 / 6., 0.0, 0.0),
                       (2 / 6., 1.0, 1.0),
                       (4 / 6., 1.0, 1.0),
                       (5 / 6., 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

                  'green':
                      ((0.0, 1.0, 1.0),
                       (1 / 6., 0.0, 0.0),
                       (0.5, 0.0, 0.0),
                       (4 / 6., 1.0, 1.0),
                       (1.0, 1.0, 1.0)),

                  'blue':
                      ((0.0, 1.0, 1.0),
                       (2 / 6., 1.0, 1.0),
                       (0.5, 0.0, 0.0),
                       (5 / 6., 0.0, 0.0),
                       (1.0, 1.0, 1.0)),
                  }

cm_phase = LinearSegmentedColormap('Phase', cm_phase_cdict)


def phase2rgb(s):
    """
    Crates RGB image with colour-coded phase from a complex array.
    
    Args:
        s: a complex numpy array
    
    Returns:
        an RGBA numpy array, with one additional dimension added
    """
    ph = np.angle(s)
    t = pi / 3
    rgba = np.zeros(list(s.shape) + [4])
    rgba[..., 0] = (ph < t) * (ph > -t) + (ph > t) * (ph < 2 * t) * (2 * t - ph) / t + (ph > -2 * t) * (ph < -t) * (
            ph + 2 * t) / t
    rgba[..., 1] = (ph > t) + (ph < -2 * t) * (-2 * t - ph) / t + (ph > 0) * (ph < t) * ph / t
    rgba[..., 2] = (ph < -t) + (ph > -t) * (ph < 0) * (-ph) / t + (ph > 2 * t) * (ph - 2 * t) / t
    return rgba


def complex2rgbalog(s, amin=0.5, dlogs=2, smax=None, type='uint8'):
    """
    Returns a 2D RGBA image with colour-coded phases and log10(amplitude) in brightness.
    
    Args:
        s: the 2D complex data array to be converted to RGB
        amin: the minimum value for the alpha channel
        dlogs: amplitude range displayed, in log10 units
        smax: if specified, all values above max will be displayed with an alpha value of 1
        type: either 'float': values are in the [0..1] range, or 'uint8' (0..255) (new default)

    Returns:

    """
    rgba = phase2rgb(s)
    sabs = np.abs(s)
    if smax is not None:
        sabs = smax * (sabs > smax) + sabs * (sabs <= smax)
    a = np.log10(sabs + 1e-20)
    a -= a.max() - dlogs  # display dlogs orders of magnitude
    rgba[..., 3] = amin + a / dlogs * (1 - amin) * (a > 0)
    if type == 'float':
        return rgba
    return (rgba * 255).astype(np.uint8)


def complex2rgbalin(s, gamma=1.0, smax=None, smin=None, percentile=(None, None), alpha=(0, 1), type='uint8'):
    """
    Returns RGB image with with colour-coded phase and linear amplitude in brightness.
    Optional exponent gamma is applied to the amplitude.

    Args:
        s: the complex data array (likely 2D, but can have higher dimensions)
        gamma: gamma parameter to change the brightness curve
        smax: maximum value (brightness = 1). If not supplied and percentile is not set,
              the maximum amplitude of the array is used.
        smin: minimum value(brightness = 0). If not supplied and percentile is not set,
              the maximum amplitude of the array is used.
        percentile: a tuple of two values (percent_min, percent_max) setting the percentile (between 0 and 100):
                    the smax and smin values will be  set as the percentile value in the array (see numpy.percentile).
                    These two values (when not None) supersede smax and smin.
                    Example: percentile=(0,99) to scale the brightness to 0-1 between the 1% and 99% percentile of the data amplitude.
        alpha: the minimum and maximum value for the alpha channel, normally (0,1). Useful to have different max/min
               alpha when going through slices of one object
        type: either 'float': values are in the [0..1] range, or 'uint8' (0..255) (new default)
    Returns:
        the RGBA array, with the same diemensions as the input array, plus one additional R/G/B/A dimension appended.
    """
    rgba = phase2rgb(s)
    a = np.abs(s)
    if percentile is not None:
        if percentile[0] is not None:
            smin = np.percentile(a, percentile[0])
        if percentile[1] is not None:
            smax = np.percentile(a, percentile[1])
        if smax is not None and smin is not None:
            if smin > smax:
                smin, smax = smax, smin
    if smax is not None:
        a = (a - smax) * (a <= smax) + smax
    if smin is not None:
        a = (a - smin) * (a >= smin)
    a /= a.max()
    a = a ** gamma
    rgba[..., 3] = alpha[0] + alpha[1] * a
    if type == 'float':
        return rgba
    return (rgba * 255).astype(np.uint8)


def complex2rgbalin_dark(s, gamma=1, smin=None, smax=None, percentile=None, phase_shift=0):
    """
    Returns RGB image based on a HSV projection with hue=phase, saturation=1 and value =amplitude.
    This yields a black-background image contrary to complex2rgbalin

    Args:
        s: the 2D complex data array
        gamma: gamma parameter to change the brightness curve
        smax: maximum value (value = 1). If not supplied and percentile is not set, the maximum amplitude is used.
        smin: minimum value(value = 0). If not supplied and percentile is not set, the minimum amplitude is used.
        percentile: a tuple of two values (percent_min, percent_max) setting the percentile (between 0 and 100):
                    the smax and smin values will be  set as the percentile value in the array (see numpy.percentile).
                    These two values (when not None) supersede smax and smin.
                    Example: percentile=(0,99) to scale the brightness to 0-1 between the 1% and 99% percentile
                    of the array amplitude.
        phase_shift: shift the phase origin by this amount (in radians)
    Returns:
        the RGB array, with values between 0 and 1.
    """
    h = ((np.angle(s) + np.pi) / (2 * np.pi) + phase_shift / (2 * np.pi)) % 1
    v = np.abs(s)
    if percentile is not None:
        if percentile[0] is not None:
            smin = np.percentile(v, percentile[0])
        if percentile[1] is not None:
            smax = np.percentile(v, percentile[1])
        if smax is not None and smin is not None:
            if smin > smax:
                smin, smax = smax, smin
    if smax is not None:
        v = (v - smax) * (v <= smax) + smax
    if smin is not None:
        v = (v - smin) * (v >= smin)
    v /= v.max()
    v = v ** gamma
    hsv = np.dstack((h, np.ones_like(h), v))
    return hsv_to_rgb(hsv)


def colorwheel(text_col='black', fs=16, ax=None):
    """
    Color wheel for phases in hsv colormap.
    
    Args:
        text_col: colour of text
        fs: fontsize in points
        fig: an instance of matplotlib.figure.Figure (for offline rendering) or None

    Returns:
        Nothing. Displays a colorwheel in the current or supplied figure.
    """
    xwheel = np.linspace(-1, 1, 100)
    ywheel = np.linspace(-1, 1, 100)[:, np.newaxis]
    rwheel = np.sqrt(xwheel ** 2 + ywheel ** 2)
    phiwheel = -np.arctan2(ywheel, xwheel)  # Need the - sign because imshow starts at (top,left)
    #  rhowheel=rwheel*np.exp(1j*phiwheel)
    rhowheel = 1 * np.exp(1j * phiwheel)
    if ax is None:
        ax = plt.gca()
    ax.set_axis_off()
    rgba = complex2rgbalin(rhowheel * (rwheel < 1))
    ax.imshow(rgba, aspect='equal')
    ax.text(1.1, 0.5, '$0$', fontsize=fs, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color=text_col)
    ax.text(-.1, 0.5, '$\pi$', fontsize=fs, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color=text_col)


def insertColorwheel(left=.7, bottom=.15, width=.1, height=.1, text_col='black', fs=16):
    """
    Inserts color wheel to the current axis.
    """
    plt.axes((left, bottom, width, height), facecolor='w')
    colorwheel(text_col=text_col, fs=fs)


def showCplx(im, mask=0, pixSize_um=1, showGrid=True, amplitudeLog=False, maskPhase=False, maskPhaseThr=0.01,
             cmapAmplitude='gray', cmapPhase=cm_phase, scalePhaseImg=True, suptit=None, fontSize=20, suptit_fontSize=10,
             show_what='amplitude_phase', hideTicks=False):
    """
    Displays AMPLITUDE_PHASE or REAL_IMAG ('show_what') of the complex image in two subfigures.
    """
    print(show_what.lower())
    if amplitudeLog:
        amplitude = np.log10(abs(im))
    else:
        amplitude = abs(im)
    phase = np.angle(im)
    fig = plt.figure(figsize=(8, 4))
    fig.add_subplot(121)
    if show_what.lower() == 'real_imag':
        plt.imshow(im.real, extent=(0, im.shape[1] * pixSize_um, 0, im.shape[0] * pixSize_um), cmap=cmapAmplitude,
                   interpolation='Nearest')
    else:
        plt.imshow(amplitude, extent=(0, im.shape[1] * pixSize_um, 0, im.shape[0] * pixSize_um), cmap=cmapAmplitude,
                   interpolation='Nearest')
    if showGrid:
        plt.grid(color='w')
    if pixSize_um != 1:
        plt.xlabel('microns', fontsize=fontSize)
        plt.ylabel('microns', fontsize=fontSize)
    if suptit is None:
        if show_what.lower() == 'real_imag':
            plt.title('Real', fontsize=fontSize)
        else:
            plt.title('Amplitude', fontsize=fontSize)
    if hideTicks:
        plt.xticks([])
        plt.yticks([])
    else:
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
    plt.subplot(122)
    if scalePhaseImg:
        vminPhase = -np.pi
        vmaxPhase = np.pi
    else:
        vminPhase = phase.min()
        vmaxPhase = phase.max()
    if show_what.lower() == 'real_imag':
        plt.imshow(im.imag, cmap=cmapPhase, interpolation='Nearest',
                   extent=(0, im.shape[1] * pixSize_um, 0, im.shape[0] * pixSize_um))
    else:
        plt.imshow(np.ma.masked_array(phase, mask), cmap=cmapPhase, interpolation='Nearest', vmin=vminPhase,
                   vmax=vmaxPhase, extent=(0, im.shape[1] * pixSize_um, 0, im.shape[0] * pixSize_um))
    if showGrid:
        plt.grid(color='k')

    if pixSize_um != 1:
        plt.xlabel('microns', fontsize=fontSize)
        plt.ylabel('microns', fontsize=fontSize)

    if suptit is None:
        if show_what.lower() == 'real_imag':
            plt.title('Imag', fontsize=fontSize)
        else:
            plt.title('Phase', fontsize=fontSize)

    if hideTicks:
        plt.xticks([])
        plt.yticks([])
    else:
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)

    if cmapPhase == cm_phase:
        insertColorwheel(left=.85, fs=fontSize)
    if suptit is not None:
        plt.suptitle(suptit, fontsize=suptit_fontSize)

    plt.tight_layout()
    plt.show()


def show_obj_probe(obj, probe, tit1='Object', tit2='Probe', stit=None, fig_num=100, pixel_size_object=None,
                   scan_area_obj=None, scan_area_probe=None, scan_pos=None, invert_yaxis=False, **kwargs):
    """
    Live plot of the object and probe phase and amplitude. Only the first modes are shown.

    :param obj: the object to plot
    :param probe: the probe to plot
    :param tit1: title for the object
    :param tit2: title for the probe
    :param stit: suptitle for the figure (e.g. the current algorithm, llk,...)
    :param fig_num: the figure number to use, so plotting always appears in the same one
    :param pixel_size_object: pixel size for the object
    :param scan_area_obj: the 2D array of the object scan area (1 inside, 0 outside),
        which will be sued to mask the object
    :param scan_pos: a tuple (x,y) of the scan positions in meters.
    :param invert_yaxis: if True, the plot will be inverted along the Y axis- this
        is used for near field ptychography so the view corresponds to the full field
        view, rather than using the motor axes as reference.
    """
    if 'minimize_obj_phase' in kwargs:
        warnings.warn("show_obj_probe: minimize_obj_phase is obsolete. Use ZeroPhaseRamp() operators",
                      DeprecationWarning)
    # if modes are used, only display the first mode
    if len(obj.shape) > 2:
        obj = obj[0]
    if len(probe.shape) > 2:
        probe = probe[0]
    obja = np.angle(obj)
    probea = np.angle(probe)
    if scan_area_obj is not None:
        omc = np.ma.masked_array(obj, mask=~scan_area_obj).compressed()
        objmin = np.abs(omc).min()
        objmax = np.abs(omc).max()
        obj_phi_min, obj_phi_max = np.percentile(np.angle(omc), [2, 98])
    else:
        objmin, objmax = None, None
        obj_phi_min, obj_phi_max = np.percentile(np.angle(obj), [2, 98])

    if obj_phi_max - obj_phi_min > np.pi:
        cm_phase_obj = cm_phase
        obj_phi_min, obj_phi_max = None, None
    else:
        cm_phase_obj = plt.cm.get_cmap('gray')

    if scan_area_probe is not None:
        pmc = np.ma.masked_array(probe, mask=~scan_area_probe).compressed()
        probemax = np.abs(pmc).max()
        probe_phi_min, probe_phi_max = np.percentile(np.angle(pmc), [2, 98])
    else:
        probemax = None
        probe_phi_min, probe_phi_max = np.percentile(np.angle(probe), [2, 98])

    if probe_phi_max - probe_phi_min > np.pi:
        cm_phase_probe = cm_phase
        probe_phi_min, probe_phi_max = None, None
    else:
        cm_phase_probe = plt.cm.get_cmap('gray')

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, probe.shape[0] / obj.shape[0]])
    plt.ion()
    if fig_num != -1:
        fig = plt.figure(fig_num)
    else:
        fig = plt.gcf()

    if pixel_size_object is not None:
        pix = pixel_size_object * 1e6 / 2
        scan_area_f = pixel_size_object * 1e6
    else:
        scan_area_f = 1

    if scan_pos is not None:
        scan_posx = np.array(scan_pos[0] * scan_area_f)
        scan_posx = np.append(scan_posx, scan_posx[0])
        scan_posy = np.array(scan_pos[1] * scan_area_f)
        scan_posy = np.append(scan_posy, scan_posy[0])

    ax0 = plt.subplot(gs[0])
    ny, nx = obj.shape
    if pixel_size_object is None:
        ax0.imshow(np.abs(obj), extent=(-nx / 2, nx / 2, -ny / 2, ny / 2), vmin=objmin, vmax=objmax,
                   cmap=plt.cm.get_cmap('gray'), origin='lower')
    else:
        ax0.imshow(np.abs(obj), extent=(-nx * pix, nx * pix, -ny * pix, ny * pix), vmin=objmin, vmax=objmax,
                   cmap=plt.cm.get_cmap('gray'), origin='lower')
        ax0.set_xlim(-nx * pix, nx * pix)
        ax0.set_ylim(-ny * pix, ny * pix)
        plt.xlabel(u"x(µm)", horizontalalignment='left')
        plt.ylabel(u"y(µm)")
        ax0.xaxis.set_label_coords(1.05, -.05)
    if scan_pos is not None:
        ax0.plot(scan_posx, scan_posy, 'k-', linewidth=0.5)
    plt.title(tit1 + ' modulus')

    ax1 = plt.subplot(gs[1])
    if pixel_size_object is None:
        ax1.imshow(obja, extent=(-nx / 2, nx / 2, -ny / 2, ny / 2), cmap=cm_phase_obj,
                   vmin=obj_phi_min, vmax=obj_phi_max, origin='lower')
    else:
        ax1.imshow(obja, extent=(-nx * pix, nx * pix, -ny * pix, ny * pix), cmap=cm_phase_obj,
                   vmin=obj_phi_min, vmax=obj_phi_max, origin='lower')
        ax1.set_xlim(-nx * pix, nx * pix)
        ax1.set_ylim(-ny * pix, ny * pix)
        plt.xlabel(u"x(µm)", horizontalalignment='left')
        plt.ylabel(u"y(µm)")
        ax1.xaxis.set_label_coords(1.05, -.05)
    if scan_pos is not None:
        ax1.plot(scan_posx, scan_posy, 'k-', linewidth=0.5)
    if obj_phi_min is None:
        plt.title(tit1 + ' phase')
    else:
        plt.title(tit1 + ' phase [%5.2f-%5.2f radians]' % (obj_phi_min, obj_phi_max))

    ax2 = plt.subplot(gs[2])
    ny, nx = probe.shape
    if pixel_size_object is None:
        ax2.imshow(np.abs(probe), extent=(-nx / 2, nx / 2, -ny / 2, ny / 2), vmin=0, vmax=probemax,
                   cmap=plt.cm.get_cmap('gray'), origin='lower')
    else:
        ax2.imshow(np.abs(probe), extent=(-nx * pix, nx * pix, -ny * pix, ny * pix), vmin=0,
                   vmax=probemax, cmap=plt.cm.get_cmap('gray'), origin='lower')
        ax2.set_xlim(-nx * pix, nx * pix)
        ax2.set_ylim(-ny * pix, ny * pix)
        plt.xlabel(u"x(µm)", horizontalalignment='left')
        plt.ylabel(u"y(µm)")
        ax2.xaxis.set_label_coords(1.05, -.05)
        plt.xticks(rotation=30)
    if scan_pos is not None:
        ax2.plot(scan_posx, scan_posy, 'k-', linewidth=0.5)
    plt.title(tit2 + ' modulus')

    ax3 = plt.subplot(gs[3])
    if pixel_size_object is None:
        ax3.imshow(np.angle(probe), extent=(-nx / 2, nx / 2, -ny / 2, ny / 2), cmap=cm_phase_probe,
                   vmin=probe_phi_min, vmax=probe_phi_max, origin='lower')
    else:
        ny, nx = probe.shape
        ax3.imshow(probea, extent=(-nx * pix, nx * pix, -ny * pix, ny * pix), cmap=cm_phase_probe,
                   vmin=probe_phi_min, vmax=probe_phi_max, origin='lower')
        ax3.set_xlim(-nx * pix, nx * pix)
        ax3.set_ylim(-ny * pix, ny * pix)
        plt.xlabel(u"x(µm)", horizontalalignment='left')
        plt.ylabel(u"y(µm)")
        ax3.xaxis.set_label_coords(1.05, -.05)
        plt.xticks(rotation=30)
    if scan_pos is not None:
        ax3.plot(scan_posx, scan_posy, 'k-', linewidth=0.5)
    if probe_phi_min is None:
        plt.title(tit2 + ' phase')
    else:
        plt.title(tit2 + ' phase [%5.2f-%5.2f radians]' % (probe_phi_min, probe_phi_max))

    if invert_yaxis:
        for ax in [ax0, ax1, ax2, ax3]:
            ax.invert_yaxis()

    if stit:
        txt = fig.suptitle('')
        txt.set_text(stit)

    try:
        plt.draw()
        plt.gcf().canvas.draw()
        plt.pause(.001)
    except:
        pass
