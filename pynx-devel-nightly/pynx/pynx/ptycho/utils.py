# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2013-2014 : Fondation Nanosciences, Grenoble
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#         Ondrej Mandula

import numpy as np
from pynx.utils.matplotlib import pyplot as plt
from .shape import calc_obj_shape


def plot_scan(posVert, posHoriz, val=1, xlab=None, ylab=None, vlab=None, show_num=True):
    """
    Plots positions of the scan.

    Args:
        posVert, posHoriz: vertical and horizontal positions
        val: optional - one can pas e.g. val=(amplitudes**2).sum(2).sum(1) which will show each marker coloured according to integrated intesnisty
        xlab,ylab: optional - labels of the axis
        vlab: optional - label for the colorbar
        show_num: if set to True, the numbers at each point are shown
    """
    plt.figure();
    plt.scatter(posHoriz, posVert, c=val, s=40,
                cmap='jet')  # notation in Ptycho is first coordinate vertical and second horizontal...
    if show_num:
        plt.plot(posHoriz, posVert, ':')
        for i in range(len(posVert)):
            plt.annotate(str(i), xy=(posHoriz[i], posVert[i]), xytext=(5, 0), textcoords='offset points')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.grid(b=True, which='both')
    if not isinstance(val, int):
        cbr = plt.colorbar(format='%.1e')
        cbr.set_label(vlab, rotation=90)
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)


def compose_scan(frames, posVert, posHoriz, blowUpFactor=40, finalsize=None):
    """
    Creates a composite image (square) of all Ptychography frames placed at the positions [posVert, posHoriz] multiplied by blwoUpFactor.
    The size of the final image can defined by finalsize.
    """
    sf = frames.shape[1:3]
    ny, nx = calc_obj_shape(posHoriz * blowUpFactor, posVert * blowUpFactor, 2 * sf)
    nxy = max(nx, ny) + 10
    if finalsize is not None:
        q = np.ceil(nxy / (1. * finalsize))
        nxy = finalsize * q  # for rebining of integer number of pixels
    out = np.zeros((nxy, nxy))
    cx, cy = nxy // 2 - sf[0] // 2, nxy // 2 - sf[1] // 2
    for a, px, py in zip(frames, posVert, posHoriz):
        startx = cx + blowUpFactor * round(px)
        starty = cy + blowUpFactor * round(py)
        endx = startx + sf[0]
        endy = starty + sf[1]
        out[startx:endx, starty:endy] += a
    if finalsize is not None:
        out = out.reshape((finalsize, out.shape[0] // finalsize, finalsize, -1)).mean(3).mean(1)
    return out
