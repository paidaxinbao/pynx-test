# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['rotation_matrix', 'rotate']

import numpy as np


def rotation_matrix(axis, angle):
    """
    Creates a rotation matrix as a numpy array. The convention is the NeXus one so that a positive rotation of +pi/2:
    - with axis='x', transforms +y into z
    - with axis='y', transforms +z into x
    - with axis='z', transforms +x into y
    :param axis: the rotation axis, either 'x', 'y' or 'z'
    :param angle: the rotation angle in radians. Either a floating-point scalar or an array.
    :return: the rotation matrix, of shape (3,3) or (N,3,3) if the angle is an array of size N
    """
    c, s = np.cos(angle), np.sin(angle)
    if np.isscalar(angle):
        if axis.lower() == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        elif axis.lower() == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        else:
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    else:
        n = angle.size
        m = np.zeros((n, 3, 3))
        if axis.lower() == 'x':
            m[:, 0, 0] = 1
            m[:, 1, 1] = c
            m[:, 1, 2] = -s
            m[:, 2, 2] = c
            m[:, 2, 1] = s
        elif axis.lower() == 'y':
            m[:, 1, 1] = 1
            m[:, 0, 0] = c
            m[:, 0, 2] = s
            m[:, 2, 2] = c
            m[:, 2, 0] = -s
        else:
            m[:, 2, 2] = 1
            m[:, 0, 0] = c
            m[:, 0, 1] = -s
            m[:, 1, 1] = c
            m[:, 1, 0] = s
        return m


def rotation_matrix_dot(m1: np.ndarray, m2: np.ndarray):
    """
    Matrix multiplication of two rotation matrices. This extends the numpy.dot function to also handle the case
    where at least one matrix has a (N,3,3) size instead of (3,3)
    :param m1: the first rotation matrix, either (3,3) shape or (N,3,3)
    :param m2: the second rotation matrix, either (3,3) shape or (N,3,3)
    :return: the two matrix multiplied
    """
    if m1.ndim == 2 and m2.ndim == 2:
        return np.dot(m1, m2)
    elif m1.ndim == 2 and m2.ndim == 3:
        n = len(m2)
        m = np.empty_like(m2)
        for i in range(n):
            m[i] = np.dot(m1, m2[i])
    elif m1.ndim == 3 and m2.ndim == 2:
        n = len(m1)
        m = np.empty_like(m1)
        for i in range(n):
            m[i] = np.dot(m1[i], m2)
    else:
        n = len(m2)
        m = np.empty_like(m2)
        for i in range(n):
            m[i] = np.dot(m1[i], m2[i])
    return m


def rotate(m, x, y, z):
    """
    Perform a rotation given a rotation matrix and x, y, z coordinates (which can be arrays).
    :param m: the (3x3) rotation matrix. Alternatively, it can be an array of shape (N, 3,3)
        when created by calling rotation_matrix(axis, angle) with angle a vector with N values.
        N must coincide with the number of x,y,z elements if these are arrays.
    :param x: the array or scalar for the x coordinate
    :param y: the array or scalar for the y coordinate
    :param z: the array or scalar for the z coordinate
    :return: a tuple of x, y, z coordinated after rotation
    """
    return m[..., 0, 0] * x + m[..., 0, 1] * y + m[..., 0, 2] * z, \
           m[..., 1, 0] * x + m[..., 1, 1] * y + m[..., 1, 2] * z, \
           m[..., 2, 0] * x + m[..., 2, 1] * y + m[..., 2, 2] * z
