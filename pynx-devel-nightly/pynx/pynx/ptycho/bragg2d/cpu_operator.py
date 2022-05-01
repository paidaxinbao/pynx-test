# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['ShowObj']

import numpy as np
from .bragg2d import OperatorBragg2DPtycho
from ..bragg.cpu_operator import show_3d


# TODO: Merge these with 3D Bragg operators

#################################################################################################################
###############################  Exclusive CPU operators  #######################################################
#################################################################################################################

class ShowObj(OperatorBragg2DPtycho):
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
