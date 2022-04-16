# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2021-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import time
import numpy as np
from scipy.ndimage import center_of_mass
from pynx.utils.matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pynx.utils.plot_utils import complex2rgbalin, colorwheel
from pynx.utils.phase import shift_phase_zero
from pynx.cdi import CDI
from pynx.version import get_git_version

_pynx_version = get_git_version()


def show_cdi(cdi: CDI, params=None, fig_num=201, save_plot=None, display_plot=False, figsize=(10, 6), crop=10,
             plot_type='rgba', title=None, subtitle=None):
    """
    Create a plot of the CDI object with different cuts, centred on the object support
    :param cdi:
    :param params:
    :param fig_num:
    :param save_plot:
    :param display_plot:
    :param figsize:
    :param crop:
    :param plot_type: either 'rgba' or 'abs'
    :return:
    """
    if save_plot is None and display_plot is False:
        return
    if display_plot:
        try:
            fig = plt.figure(fig_num, figsize=figsize)
        except:
            # no GUI or $DISPLAY
            fig = Figure(figsize=figsize)
    else:
        fig = Figure(figsize=figsize)
    fig.clf()

    # Determine the support extent
    obj = cdi.get_obj(shift=True)
    sup = cdi.get_support(shift=True)
    if crop > 0:
        # crop around support
        if obj.ndim == 3:
            l0 = np.nonzero(sup.sum(axis=(1, 2)))[0].take([0, -1]) + np.array([-crop, crop])
            if l0[0] < 0: l0[0] = 0
            if l0[1] >= sup.shape[0]: l0[1] = -1

            l1 = np.nonzero(sup.sum(axis=(0, 2)))[0].take([0, -1]) + np.array([-crop, crop])
            if l1[0] < 0: l1[0] = 0
            if l1[1] >= sup.shape[1]: l1[1] = -1

            l2 = np.nonzero(sup.sum(axis=(0, 1)))[0].take([0, -1]) + np.array([-crop, crop])
            if l2[0] < 0: l2[0] = 0
            if l2[1] >= sup.shape[2]: l2[1] = -1
            obj = obj[l0[0]:l0[1], l1[0]:l1[1], l2[0]:l2[1]]
            sup = sup[l0[0]:l0[1], l1[0]:l1[1], l2[0]:l2[1]]
        else:
            l0 = np.nonzero(sup.sum(axis=1))[0].take([0, -1]) + np.array([-crop, crop])
            if l0[0] < 0: l0[0] = 0
            if l0[1] >= sup.shape[0]: l0[1] = -1

            l1 = np.nonzero(sup.sum(axis=0))[0].take([0, -1]) + np.array([-crop, crop])
            if l1[0] < 0: l1[0] = 0
            if l1[1] >= sup.shape[1]: l1[1] = -1

            obj = obj[l0[0]:l0[1], l1[0]:l1[1]]
            sup = sup[l0[0]:l0[1], l1[0]:l1[1]]

    obj_sup = obj[sup > 0]

    obj = shift_phase_zero(obj, mask=sup, stack=False)

    # # Try randomaxes orientation to test scaling
    # n1, n2 = np.random.randint(3), np.random.randint(3)
    # print(n1, n2)
    # obj = np.swapaxes(obj, n1, n2)

    cz, cy, cx = center_of_mass(abs(obj))

    smin, smax = 0, np.percentile(abs(obj_sup), (99))
    axcmap = np.argmin(obj.shape)

    # Extent of plots
    # Should use GridSpec ? Do we need to adjust the size for the colorbar ?
    fx, fy = fig.get_size_inches()
    # Maximum height and width for subplots, in relative figure units
    maxheight, maxwidth = 0.40, 0.44
    # Largest vertical and horizontal object pixel size
    nymax = max(obj.shape[:2])
    nxmax = max(obj.shape[0], obj.shape[2])
    # Largest vertical and horizontal object physical size, without scaling
    fignymax = nymax * fy
    fignxmax = nxmax * fx
    # Scale (pixels to fraction of figure) to match vertical and horizontal constraints
    fig_scaley = maxheight / fignymax
    fig_scalex = maxwidth / fignxmax
    # Final scale - object size in fig units will be (ny, nx) * fig_scale
    fig_scale = min(fig_scalex * fx, fig_scaley * fy)

    # XY
    o = obj[int(cz)]
    ny, nx = o.shape
    xlims = [0.275 - fig_scale * nx / 2, 0.7 - fig_scale * ny / 2, fig_scale * nx, fig_scale * ny]
    if plot_type != 'rgba' and axcmap == 1:
        xlims[2] /= 0.90  # Extent for colorbart
    ax = fig.add_axes(xlims)
    if plot_type == 'rgba':
        ax.imshow(complex2rgbalin(o, smin=smin, smax=smax), origin='lower')
    else:
        cmap = None
        if plot_type != 'abs':
            cmap = plot_type
        im = ax.imshow(np.abs(o), vmin=smin, vmax=smax, origin='lower', cmap=cmap)
        if axcmap == 2:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            c = plt.colorbar(im, cax=cax)
            c.ax.tick_params(labelsize=8, direction='in')
    plt.xlabel('X', x=1, horizontalalignment='left', verticalalignment='bottom', fontsize=8, fontweight="bold")
    plt.ylabel('Y', y=1, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight="bold",
               rotation=0)
    ax.tick_params(which='both', axis='both', direction='in', pad=2, labelsize=8)

    # YZ
    o = np.swapaxes(obj[:, :, int(cx)], 0, 1)
    ny, nx = o.shape
    xlims = [0.775 - fig_scale * nx / 2, 0.7 - fig_scale * ny / 2, fig_scale * nx, fig_scale * ny]
    if plot_type != 'rgba' and axcmap == 0:
        xlims[2] /= 0.90  # Extent for colorbart
    # print(xlims)
    ax = fig.add_axes(xlims)
    if plot_type == 'rgba':
        ax.imshow(complex2rgbalin(o, smin=smin, smax=smax), origin='lower')
    else:
        cmap = None
        if plot_type != 'abs':
            cmap = plot_type
        im = ax.imshow(np.abs(o), vmin=smin, vmax=smax, origin='lower', cmap=cmap)
        if axcmap == 0:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            c = plt.colorbar(im, cax=cax)
            c.ax.tick_params(labelsize=8, direction='in')
    plt.xlabel('Z', x=1, horizontalalignment='left', verticalalignment='bottom', fontsize=8, fontweight="bold")
    plt.ylabel('Y', y=1, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight="bold",
               rotation=0)
    ax.tick_params(which='both', axis='both', direction='in', pad=2, labelsize=8)

    # XZ
    o = obj[:, int(cy)]
    ny, nx = o.shape
    xlims = [0.275 - fig_scale * nx / 2, 0.28 - fig_scale * ny / 2, fig_scale * nx, fig_scale * ny]
    if plot_type != 'rgba' and axcmap == 1:
        xlims[2] /= 0.90  # Extent for colorbart
    # print(xlims)
    ax = fig.add_axes(xlims)
    if plot_type == 'rgba':
        ax.imshow(complex2rgbalin(o, smin=smin, smax=smax), origin='lower')
        x0 = ax.get_position().xmax  # for params text left boundary
    else:
        cmap = None
        if plot_type != 'abs':
            cmap = plot_type
        im = ax.imshow(np.abs(o), vmin=smin, vmax=smax, origin='lower', cmap=cmap)
        x0 = ax.get_position().xmax  # for params text left boundary
        if axcmap == 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            c = plt.colorbar(im, cax=cax)
            c.ax.tick_params(labelsize=8, direction='in')
            x0 = c.outline.axes.get_position().xmax  # for params text left boundary
    plt.xlabel('X', x=1, horizontalalignment='left', verticalalignment='bottom', fontsize=8, fontweight="bold")
    plt.ylabel('Z', y=1, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight="bold",
               rotation=0)
    ax.tick_params(which='both', axis='both', direction='in', pad=2, labelsize=8)

    if plot_type == 'rgba':
        axpos = ax.get_position()
        ax = fig.add_axes((axpos.xmax + 0.02, axpos.ymax - 0.06, 0.04 * fy / fx, 0.04), facecolor='w')
        colorwheel(ax=ax, fs=12)
        x0 = ax.get_position().xmax  # for params text left boundary

    # Step between text lines of size 6
    dy = (6 + 1) / 72 / fig.get_size_inches()[1]

    if title is not None:
        fig.text(0.5, 1, title, horizontalalignment='center', verticalalignment='top', fontsize=10, fontweight='bold')

    if subtitle is not None:
        fig.text(0.5, 1 - dy * 10 / 6, subtitle, horizontalalignment='center',
                 verticalalignment='top', fontsize=6, fontweight='bold')

    # LLK, Support size, average, max, nb photons
    llk = cdi.get_llk(normalized=True)
    v1 = int(np.round(np.log10(cdi.nb_photons_calc)))
    v2 = cdi.nb_photons_calc / 10 ** v1
    fig.text(0.5, 1 - 2 * dy * 10 / 6, r'$LLK_p=%6.3f\ [LLK_{free}=%6.3f]\ \ \ Support: nb=%d\ \ \ '
                                       r'<\rho>=%6.2f \ \ \rho_{max}=%6.2f\ \ \ %4.2f\times 10^{%d} photons$' %
             (llk[0], llk[3], cdi.nb_point_support, np.sqrt(cdi.nb_photons_calc / cdi.nb_point_support), cdi._obj_max,
              v2, v1),
             horizontalalignment='center', verticalalignment='top', fontsize=9, fontweight='bold')
    # Params
    if params is not None:
        x0 = x0 + 0.03
        y0 = 0.49
        n = 1
        vk = [k for k in params.keys()]
        vk.sort()
        for k in vk:
            v = params[k]
            if v is not None:
                fig.text(x0, y0 - n * dy * 5 / 6, "%s = %s" % (k, str(v)), fontsize=5, horizontalalignment='left',
                         stretch='condensed', verticalalignment='top')
                n += 1

    fig.text(dy, dy, "PyNX v%s, %s" % (_pynx_version, time.strftime("%Y/%m/%d %H:%M:%S")),
             fontsize=6, horizontalalignment='left', stretch='condensed')

    if display_plot:
        try:
            plt.draw()
            plt.gcf().canvas.draw()
            plt.pause(.001)
        except:
            pass

    if save_plot:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(save_plot, dpi=150)
