# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from __future__ import division

import numpy as np
from numpy import pi
from scipy import misc
import sys


def get_img(index=0):
    """
    Returns image (numpy array) from scipy.misc
    
    Args:
        index:
            0-> return scipy.misc.face()[:,:,1], cropped to 512x512
            1 -> return 512x512 scipy.misc.ascent()

    """
    if index == 0:
        return misc.face(gray=True)[100:612, 280:792]
    return misc.ascent()


def spiral_archimedes(a, n):
    """" Creates np points spiral of step a, with a between successive points
    on the spiral. Returns the x,y coordinates of the spiral points.

    This is an Archimedes spiral. the equation is:
      r=(a/2*pi)*theta
      the stepsize (radial distance between successive passes) is a
      the curved absciss is: s(theta)=(a/2*pi)*integral[t=0->theta](sqrt(1*t**2))dt
    """
    vr, vt = [0], [0]
    t = np.pi
    while len(vr) < n:
        vt.append(t)
        vr.append(a * t / (2 * np.pi))
        t += 2 * np.pi / np.sqrt(1 + t ** 2)
    vt, vr = np.array(vt), np.array(vr)
    return vr * np.cos(vt), vr * np.sin(vt)


def spiral_fermat(dmax, n):
    """"
    Creates a Fermat spiral with n points distributed in a circular area with
    diameter<= dmax. Returns the x,y coordinates of the spiral points. The average
    distance between points can be roughly estimated as 0.5*dmax/(sqrt(n/pi))

    http://en.wikipedia.org/wiki/Fermat%27s_spiral
    """
    c = 0.5 * dmax / np.sqrt(n)
    vr, vt = [], []
    t = .4
    goldenAngle = np.pi * (3 - np.sqrt(5))
    while t < n:
        vr.append(c * np.sqrt(t))
        vt.append(t * goldenAngle)
        t += 1
    vt, vr = np.array(vt), np.array(vr)
    return vr * np.cos(vt), vr * np.sin(vt)


def siemens_star(dsize=512, nb_rays=36, r_max=None, nb_rings=8, cheese_holes_nb=0, cheese_hole_max_radius=5,
                 cheese_hole_spiral_period=0):
    """
    Calculate a binary Siemens star.

    Args:
        dsize: size in pixels for the 2D array with the star data
        nb_rays: number of radial branches for the star. Must be > 0
        r_max: maximum radius for the star in pixels. If None, dsize/2 is used
        nb_rings: number of rings (the rays will have some holes between successive rings)
        cheese_holes_nb: number of cheese holes other the entire area, resulting more varied frequencies
        cheese_hole_max_radius: maximum axial radius for the holes (with random radius along x and y). If the value is
                                negative, the radius is fixed instead of random.
        cheese_hole_spiral_period: instead of randomly distributing holes, giving an integer N number for this parameter
                                   will generate holes located on an Archimedes spiral, every N pixels along the
                                   spiral, which also has a step of N pixels. The pattern of holes is aperiodic.

    Returns:
        a 2D array with the Siemens star.
    """
    if r_max is None:
        r_max = dsize // 2
    x, y = np.meshgrid(np.arange(-dsize // 2, dsize // 2, dtype=np.float32),
                       np.arange(-dsize // 2, dsize // 2, dtype=np.float32))

    a = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    am = 2 * pi / nb_rays
    d = (a % (am)) < (am / 2)
    if r_max != 0 and r_max is not None:
        d *= r < r_max
    if nb_rings != 0 and nb_rings is not None:
        if r_max is None:
            rm = dsize * np.sqrt(2) / 2 / nb_rings
        else:
            rm = r_max / nb_rings
        d *= (r % rm) < (rm * 0.9)
    if cheese_holes_nb > 0:
        if cheese_hole_spiral_period:
            cx, cy = spiral_archimedes(cheese_hole_spiral_period, cheese_holes_nb + 1)
            # remove center
            cx = cx[1:].astype(np.int32)
            cy = cy[1:].astype(np.int32)
        else:
            cx = np.random.randint(x.min(), x.max(), cheese_holes_nb)
            cy = np.random.randint(y.min(), y.max(), cheese_holes_nb)
        if cheese_hole_max_radius < 0:
            rx = (np.ones(cheese_holes_nb) * abs(cheese_hole_max_radius)).astype(np.int32)
            ry = rx
        else:
            rx = np.random.uniform(1, cheese_hole_max_radius, cheese_holes_nb)
            ry = np.random.uniform(1, cheese_hole_max_radius, cheese_holes_nb)
        for i in range(cheese_holes_nb):
            dn = int(np.ceil(max(rx[i], ry[i])))
            x0, x1 = dsize // 2 + cx[i] - dn, dsize // 2 + cx[i] + dn
            y0, y1 = dsize // 2 + cy[i] - dn, dsize // 2 + cy[i] + dn
            d[y0:y1, x0:x1] *= (((x[y0:y1, x0:x1] - cx[i]) / rx[i]) ** 2 + ((y[y0:y1, x0:x1] - cy[i]) / ry[i]) ** 2) > 1
    return d.astype(np.float32)


def fibonacci_urchin(dsize=256, nb_rays=100, r_max=64, nb_rings=8):
    """
    Calculate a binary urchin (in 3D).
    
    Args:
        dsize: size in pixels for the 2D array with the star data
        nb_rays: number of radial branches. Must be > 0
        r_max: maximum radius in pixels
        nb_rings: number of rings (the rays will have some holes between successive rings)

    Returns:
        a 3D array with the binary urchin.
    """
    tmp = np.arange(-dsize // 2, dsize // 2, dtype=np.float32)
    z, y, x = np.meshgrid(tmp, tmp, tmp, indexing='ij')
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-6

    # Generate points on a sphere of radius=1
    i = np.arange(nb_rays)
    z1 = i * 2. / nb_rays - 1 + 1. / nb_rays
    r1 = np.sqrt(1 - z1 ** 2)
    phi1 = i * np.pi * (3 - np.sqrt(5))
    x1 = np.cos(phi1) * r1
    y1 = np.sin(phi1) * r1

    # Approximate distance between points on a sphere with unit radius (approximation assuming an hexagonal packing)
    d = np.sqrt(8. / 3. / np.sqrt(3) * 4 * np.pi / nb_rays)
    # Approximate average angular distance between points
    da = d / 2

    rho = np.zeros_like(x)

    sys.stdout.write("Simulating 3d binary urchin (this WILL take a while)...")
    sys.stdout.flush()
    i = nb_rays
    for xi, yi, zi in zip(x1, y1, z1):
        # This could go MUCH faster on a GPU
        sys.stdout.write('%d ' % (i))
        sys.stdout.flush()
        rho += ((x * xi + y * yi + z * zi) / r) > np.cos(da / 2)
        i -= 1
    print("\n")

    if r_max is not None:
        rho *= r < r_max

    if nb_rings is not None:
        if r_max is None:
            rm = dsize * np.sqrt(3) / 2 / nb_rings
        else:
            rm = r_max / nb_rings
        rho *= (r % rm) < (rm * 0.8)

    return rho
