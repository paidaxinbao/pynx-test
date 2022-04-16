# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2013-2014 : Fondation Nanosciences, Grenoble
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#         Ondrej Mandula

from __future__ import division

import numpy as np


def calc_obj_shape(posx, posy, probe_shape, margin=16, multiple=4):
    """
    Determines the required size for the reconstructed object.

    :param posy: array of the y scan positions
    :param posx: array of the x scan positions
    :param probe_shape: shape of the probe
    :param margin: margin to extend the object area, in case the positions will change (optimization)
    :param multiple: the shape must be a multiple of that number
    :return: the shape (ny, nx) of the object
    """
    ny = int(2 * (abs(np.ceil(posy)) + 1).max() + probe_shape[0])
    nx = int(2 * (abs(np.ceil(posx)) + 1).max() + probe_shape[1])

    if margin is not None:
        ny += margin
        nx += margin

    if multiple is not None:
        dy = ny % multiple
        if dy:
            ny += (multiple - dy)
        dx = nx % multiple
        if dx:
            nx += (multiple - dx)

    return ny, nx


def get_view_coord(obj_shape, probe_shape, dx, dy, integer=True):
    """
    Get pixel coordinates of corner for the part of the object illuminated by a probe, given the shift
    of the probe relative to the object center. Object, probe and shift correspond to 2D coordinates.
    
    Args:
        obj_shape: the shape of the object
        probe_shape: the shape of the probe
        dx: the shift relative to the center of the object, along x (in pixel units)
        dy: the shift relative to the center of the object, along y (in pixel units)
        integer: if True the default), will return integer coordinates - otherwise floating point values are returned

    Returns: a tuple (cx, cy) of the corner coordinates of the illuminated object portion.

    """
    cy = (obj_shape[0] - probe_shape[0]) // 2 + dy
    cx = (obj_shape[1] - probe_shape[1]) // 2 + dx
    if integer:
        cx, cy = int(round(cx)), int(round(cy))
    msg = 'Getting outside of the object (dy=%d,dx=%d)(cy=%d,cx=%d). Consider increasing the object size.' \
          % (dy, dx, cy, cx)
    if cx <= 0:
        cx = 0
        print(msg)
    elif (cx + probe_shape[1]) >= obj_shape[1]:
        cx = obj_shape[1] - probe_shape[1]
        print(msg)
    if cy <= 0:
        cy = 0
        print(msg)
    elif (cy + probe_shape[0]) >= obj_shape[0]:
        cy = obj_shape[0] - probe_shape[0]
        print(msg)
    return cx, cy


def get_center_coord(obj_shape, probe_shape, cx, cy, px=1, py=1):
    """
    Compute the coordinates of the center of the frame, relative to the center of the object, 
    given the coordinates of the corner relative to the object array origin (0, 0). These are given in pixel
    units, or in physical unit if the pixel size is given.
    :param obj_shape: the object 2D shape
    :param probe_shape: the probe 2D shape
    :param cx: the corner coordinate of the object along x
    :param cy: the corner coordinate of the object along y
    :param px: the pixel size along x
    :param py: the pixel size along y
    :return: a tuple (dx, dy) of x and y coordinates relative to the object center
    """
    nyo, nxo = obj_shape
    ny, nx = probe_shape
    return (cx + nx / 2 - nxo / 2) * px, (cy + ny / 2 - nyo / 2) * py
