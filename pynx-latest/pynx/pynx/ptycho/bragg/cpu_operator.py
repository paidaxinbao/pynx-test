# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['ShowObj']

import numpy as np
from .braggptycho import OperatorBraggPtycho, BraggPtycho
from ...utils.plot_utils import cm_phase, insertColorwheel
from pynx.utils.matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy.interpolate import RegularGridInterpolator
from ...utils.rotation import rotation_matrix


#################################################################################################################
###############################  Exclusive CPU operators  #######################################################
#################################################################################################################

def show_3d(o, ortho_m, support=None, rotation=None, fig_num=-1, title=None, extent=None):
    """
    Display amplitude and phase along 3 cuts of a given object. Currently the 3 cuts correspond to the 3 main
    direction of the object array, not true x, y and z-cuts (TODO).
    :param o: 3d complex array of the data to be displayed
    :param ortho_m: 3x3 orthonormalization matrix, to convert array indices to orthonormal (SI units) coordinates
    :param support: 3D array mapping the relevant area to be displayed (0: outside, 1:inside), unless extent is used.
    :param rotation=('z',np.deg2rad(-20)): optionally, the object can be displayed after a rotation of the
                                           object. This is useful if the object or support is to be defined as a
                                           parallelepiped, before being rotated to be in diffraction condition.
                                           The rotation can be given as a tuple of a rotation axis name (x, y or z)
                                           and a counter-clockwise rotation angle in radians.
    :param fig_num: number of the figure to use. If -1, the last figure will be used. If None, a new one is created.
    :param title: optional title for the figure
    :param extent: a tuple with 6 values giving the extent of the display area (xmin, xmax, ymin, ymax, zmin, zmax).
                   If given, the support will be ignored.
    :return: nothing
    """
    # Work in microns
    ortho_m = ortho_m * 1e6
    if rotation is not None:
        ax, ang = rotation
        ortho_m = np.dot(rotation_matrix(axis=ax, angle=ang), ortho_m)

    # Get coordinates in original array
    nz, ny, nx = o.shape
    iz, iy, ix = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
    x = ortho_m[0, 0] * ix + ortho_m[0, 1] * iy + ortho_m[0, 2] * iz
    y = ortho_m[1, 0] * ix + ortho_m[1, 1] * iy + ortho_m[1, 2] * iz
    z = ortho_m[2, 0] * ix + ortho_m[2, 1] * iy + ortho_m[2, 2] * iz
    # Object is centered
    x -= x.mean()
    y -= y.mean()
    z -= z.mean()

    if extent is not None:
        x0, x1, y0, y1, z0, z1 = (v * 1e6 for v in extent)
    elif support is None:
        x0, x1 = x.min(), x.max()
        y0, y1 = y.min(), y.max()
        z0, z1 = z.min(), z.max()
    else:
        tmp = np.ma.masked_array(x, mask=(support == 0))
        x0, x1 = tmp.min(), tmp.max()
        x0, x1 = x0 - (x1 - x0) * 0.1, x1 + (x1 - x0) * 0.1
        tmp = np.ma.masked_array(y, mask=(support == 0))
        y0, y1 = tmp.min(), tmp.max()
        y0, y1 = y0 - (y1 - y0) * 0.1, y1 + (y1 - y0) * 0.1
        tmp = np.ma.masked_array(z, mask=(support == 0))
        z0, z1 = tmp.min(), tmp.max()
        z0, z1 = z0 - (z1 - z0) * 0.1, z1 + (z1 - z0) * 0.1

    # Only interpolate the 3 2D cuts, separate phase and amplitude to avoid interferences
    rgi_abs = RegularGridInterpolator((np.arange(-nz // 2, nz // 2), np.arange(-ny // 2, ny // 2),
                                       np.arange(-nx // 2, nx // 2)), np.abs(o), method='linear',
                                      bounds_error=False,
                                      fill_value=0)
    rgi_ang = RegularGridInterpolator((np.arange(-nz // 2, nz // 2), np.arange(-ny // 2, ny // 2),
                                       np.arange(-nx // 2, nx // 2)), np.angle(o), method='linear',
                                      bounds_error=False,
                                      fill_value=0)
    # XY Plane
    z = 0
    y, x = np.meshgrid(np.linspace(y0, y1, ny), np.linspace(x0, x1, nx), indexing='ij')
    ortho_im = np.linalg.inv(ortho_m)
    ix = ortho_im[0, 0] * x + ortho_im[0, 1] * y + ortho_im[0, 2] * z
    iy = ortho_im[1, 0] * x + ortho_im[1, 1] * y + ortho_im[1, 2] * z
    iz = ortho_im[2, 0] * x + ortho_im[2, 1] * y + ortho_im[2, 2] * z
    oabs = rgi_abs(np.concatenate((iz.reshape((1, iz.size)), iy.reshape((1, iz.size)),
                                   ix.reshape((1, iz.size)))).transpose())
    oabsxy = oabs.reshape((ny, nx)).astype(np.float32)
    oang = rgi_ang(np.concatenate((iz.reshape((1, iz.size)), iy.reshape((1, iz.size)),
                                   ix.reshape((1, iz.size)))).transpose())
    oangxy = oang.reshape((ny, nx)).astype(np.float32)

    # XZ Plane
    y = 0
    z, x = np.meshgrid(np.linspace(z0, z1, nz), np.linspace(x0, x1, nx), indexing='ij')
    ortho_im = np.linalg.inv(ortho_m)
    ix = ortho_im[0, 0] * x + ortho_im[0, 1] * y + ortho_im[0, 2] * z
    iy = ortho_im[1, 0] * x + ortho_im[1, 1] * y + ortho_im[1, 2] * z
    iz = ortho_im[2, 0] * x + ortho_im[2, 1] * y + ortho_im[2, 2] * z
    oabs = rgi_abs(np.concatenate((iz.reshape((1, iz.size)), iy.reshape((1, iz.size)),
                                   ix.reshape((1, iz.size)))).transpose())
    oabsxz = oabs.reshape((nz, nx)).astype(np.float32)
    oang = rgi_ang(np.concatenate((iz.reshape((1, iz.size)), iy.reshape((1, iz.size)),
                                   ix.reshape((1, iz.size)))).transpose())
    oangxz = oang.reshape((nz, nx)).astype(np.float32)

    # YZ Plane
    x = 0
    z, y = np.meshgrid(np.linspace(z0, z1, nz), np.linspace(y0, y1, ny), indexing='ij')
    ortho_im = np.linalg.inv(ortho_m)
    ix = ortho_im[0, 0] * x + ortho_im[0, 1] * y + ortho_im[0, 2] * z
    iy = ortho_im[1, 0] * x + ortho_im[1, 1] * y + ortho_im[1, 2] * z
    iz = ortho_im[2, 0] * x + ortho_im[2, 1] * y + ortho_im[2, 2] * z
    oabs = rgi_abs(np.concatenate((iz.reshape((1, iz.size)), iy.reshape((1, iz.size)),
                                   ix.reshape((1, iz.size)))).transpose())
    oabsyz = oabs.reshape((nz, ny)).astype(np.float32)
    oang = rgi_ang(np.concatenate((iz.reshape((1, iz.size)), iy.reshape((1, iz.size)),
                                   ix.reshape((1, iz.size)))).transpose())
    oangyz = oang.reshape((nz, ny)).astype(np.float32)

    if fig_num == -1 and len(plt.get_figlabels()):
        fig = plt.gcf()
    else:
        fig = plt.figure(fig_num, figsize=(15, 8))
    fig.clf()

    ax = fig.add_subplot(231)
    ax.imshow(oabsyz.transpose(), extent=(z0, z1, y0, y1), origin='lower')
    ax.set_xlabel('Z (um)')
    ax.set_ylabel('Y (um)')
    ax.set_aspect('equal')

    ax = fig.add_subplot(234)
    ax.imshow(oangyz.transpose(), extent=(z0, z1, y0, y1), vmin=-np.pi, vmax=np.pi,
              cmap=cm_phase,
              origin='lower')
    ax.set_xlabel('Z (um)')
    ax.set_ylabel('Y (um)')
    ax.set_aspect('equal')

    ax = fig.add_subplot(232)
    ax.imshow(oabsxz, extent=(x0, x1, z0, z1), origin='lower')
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Z (um)')
    ax.set_aspect('equal')

    ax = fig.add_subplot(235)
    ax.imshow(oangxz, extent=(x0, x1, z0, z1), vmin=-np.pi, vmax=np.pi, cmap=cm_phase,
              origin='lower')
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Z (um)')
    ax.set_aspect('equal')

    ax = fig.add_subplot(233)
    ax.imshow(oabsxy, extent=(x0, x1, y0, y1), origin='lower')
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_aspect('equal')

    ax = fig.add_subplot(236)
    ax.imshow(oangxy, extent=(x0, x1, y0, y1), vmin=-np.pi, vmax=np.pi, cmap=cm_phase, origin='lower')
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_aspect('equal')

    insertColorwheel(0.93, 0, 0.05, 0.05, fs=12)

    if title is not None:
        fig.suptitle(title)

        # Force immediate display
    try:
        plt.draw()
        plt.gcf().canvas.draw()
        plt.pause(.001)
    except:
        pass


class ShowObj(OperatorBraggPtycho):
    """
    Class to display object during an optimization.
    """

    def __init__(self, fig_num=-1, title=None, rotation=None, extent=None):
        """

        :param fig_num: the matplotlib figure number. if None, a new figure will be created. if -1 (the default), the
                        current figure will be used.
        :param title: the title for the view. If None, a default title will be used.
        :param rotation=('z',np.deg2rad(-20)): optionally, the object can be displayed after a rotation of the
                                               object. This is useful if the object or support is to be defined as a
                                               parallelepiped, before being rotated to be in diffraction condition.
                                               The rotation can be given as a tuple of a rotation axis name (x, y or z)
                                               and a counter-clockwise rotation angle in radians.
        :param extent: a tuple with 6 values giving the extent of the display area (xmin, xmax, ymin, ymax, zmin, zmax).
                       If given, the support will be ignored.
        """
        super(ShowObj, self).__init__()
        self.fig_num = fig_num
        self.title = title
        self.rotation = rotation
        self.extent = extent

    def op(self, p):
        ortho_m = p.m
        # We only show the first object mode
        o = p.get_obj()[0]
        show_3d(o, ortho_m=ortho_m, support=p.support, fig_num=self.fig_num, title=self.title, rotation=self.rotation,
                extent=self.extent)
        return p

    def timestamp_increment(self, p):
        # This display operation does not modify the data.
        pass
