# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import numpy as np
import matplotlib.pyplot as plt
from .holotomo import HoloTomo, HoloTomoData, OperatorHoloTomo

__all__ = ['ShowObj', 'ShowPsi']

import numpy as np
import matplotlib.pyplot as plt

from ..utils.plot_utils import complex2rgbalin, complex2rgbalog, insertColorwheel, cm_phase
from .holotomo import OperatorHoloTomo


class ShowObj(OperatorHoloTomo):
    """
    Class to display a phase contrast object.
    """

    def __init__(self, fig_num=None, istack=0, i=None, type='Phase', obj_mode=0, title=None, figsize=None):
        """

        :param fig_num: the matplotlib figure number. if None, a new figure will be created each time.
        :param istack: index of the stack to display
        :param i: the index of the object to be displayed. If this is a list or array, all the listed views
                  will be shown. If None, all objects are shown.
        :param type: what to show. Can be 'phase' (the default), 'amplitude', 'rgba'.
        :param obj_mode: the mode of the object to be displayed, if more than one.
        :param title: the title for the view. If None, a default title will be used.
        :param figsize: if a new figure is created, this parameter will be passed to matplotlib
        """
        super(ShowObj, self).__init__()
        self.fig_num = fig_num
        self.istack = istack
        self.i = i
        self.type = type
        self.obj_mode = obj_mode
        self.title = title
        self.figsize = figsize

    def pre_imshow(self, pci: HoloTomo):
        pci._from_pu()
        d = pci.data.stack_v[self.istack].obj[:,self.obj_mode]
        d = d.reshape((d.shape[0], d.shape[-2], d.shape[-1]))
        if self.i is None:
            self.i = list(range(len(d)))
        if type(self.i) is int:
            d = d[self.i]
        else:
            d = d.take(self.i, axis=0)

        if self.fig_num != -1:
            plt.figure(self.fig_num, figsize=self.figsize)
        plt.clf()

        x, y = pci.get_x_y()
        s = np.log10(max(abs(x).max(), abs(y).max()))
        if s < -6:
            unit_name = "nm"
            s = 1e9
        elif s < -3:
            unit_name = u"µm"
            s = 1e6
        elif s < 0:
            unit_name = "mm"
            s = 1e3
        else:
            unit_name = "m"
            s = 1
        return d, x * s, y * s, unit_name

    def post_imshow(self, pci: HoloTomo, x, y, unit_name):
        plt.xlabel("X (%s)" % (unit_name))
        plt.ylabel("Y (%s)" % (unit_name))
        try:
            plt.draw()
            plt.gcf().canvas.draw()
            plt.pause(.001)
        except:
            pass

    def op(self, pci: HoloTomo):
        d, x, y, unit_name = self.pre_imshow(pci)
        nd = 1
        if d.ndim == 3:
            nd = len(d)
            max_cols = 3
            if nd > max_cols:
                ncols = max_cols
                nrows = nd // ncols
                if nd % ncols > 0:
                    nrows += 1
            else:
                ncols = nd
                nrows = 1
        if self.title is not None:
            plt.suptitle(self.title)
        else:
            plt.suptitle("Phase contrast object [%s]" % self.type)
        for i in range(nd):
            if d.ndim == 3:
                plt.subplot(nrows, ncols, i + 1)
                di = d[i]
            else:
                di = d
            if self.type.lower() == 'rgba':
                rgba = complex2rgbalin(di)
                plt.imshow(rgba, extent=(x.min(), x.max(), y.min(), y.max()))
                insertColorwheel(left=.02, bottom=.0, width=.1, height=.1, text_col='black', fs=10)
            elif self.type.lower() == 'amplitude':
                plt.imshow(np.abs(di), extent=(x.min(), x.max(), y.min(), y.max()), cmap=plt.cm.get_cmap('gray'))
                plt.colorbar()
            else:
                plt.imshow(np.angle(di), extent=(x.min(), x.max(), y.min(), y.max()), cmap=plt.cm.get_cmap('gray'))
                plt.colorbar()
            if d.ndim == 3:
                plt.title("i=%d" % self.i[i])
            else:
                plt.title("i=%d" % self.i)
            self.post_imshow(pci, x, y, unit_name)

        if self.type.lower() in ['rgba']:
            insertColorwheel(left=.02, bottom=.0, width=.1, height=.1, text_col='black', fs=10)
        return pci

    def timestamp_increment(self, pci):
        pass


class ShowPsi(OperatorHoloTomo):
    """
    Class to display a Psi array.
    """

    def __init__(self, fig_num=None, iproj=0, iz=0, type='phase', obj_mode=0, probe_mode=0,
                 title=None,
                 figsize=None):
        """

        :param i_stack: the index of the stack to display.
        :param fig_num: the matplotlib figure number. if None, a new figure will be created each time.
        :param iproj: the index of the projection to be displayed. If this is a list or array, all projections
                   are shown. If None, all are shown. This can only be used if there is more than 1 projection
                   in the Psi stack.
        :param iz: the index of the distance to be displayed. If this is a list or array, all listed distances
                   are shown. If None, all are shown.
        :param type: what to show. Can be 'phase' (the default), 'amplitude', 'rgba'.
        :param obj_mode: the mode of the object to be displayed, if more than one.
        :param probe_mode: the mode of the object to be displayed, if more than one.
        :param title: the title for the view. If None, a default title will be used.
        :param figsize: if a new figure is created, this will be passed to matplotlib
        """
        super(ShowPsi, self).__init__()
        self.fig_num = fig_num
        self.iproj = iproj
        self.iz = iz
        self.type = type
        self.obj_mode = obj_mode
        self.probe_mode = probe_mode
        self.title = title
        self.figsize = figsize

    def pre_imshow(self, pci: HoloTomo):
        pci._from_pu(psi=True)
        d = pci._psi[:, :, self.obj_mode, self.probe_mode]
        d = np.fft.fftshift(d, axes=(-2, -1))

        if self.iproj is None:
            self.iproj = list(range(d.shape[0]))

        if self.iz is None:
            self.iz = list(range(d.shape[1]))

        if type(self.iproj) is int:
            d = d.take((self.iproj,), axis=0)
            z = pci.data.detector_distance.take((self.iz,))
        else:
            d = d.take(self.iproj, axis=0)
            z = pci.data.detector_distance.take(self.iz)

        if type(self.iz) is int:
            d = d.take((self.iz,), axis=1)
            z = pci.data.detector_distance.take((self.iz,))
        else:
            d = d.take(self.iz, axis=1)
            z = pci.data.detector_distance.take(self.iz)

        if self.fig_num != -1:
            plt.figure(self.fig_num, figsize=self.figsize)
        plt.clf()

        x, y = pci.get_x_y()
        s = np.log10(max(abs(x).max(), abs(y).max()))
        if s < -6:
            unit_name = "nm"
            s = 1e9
        elif s < -3:
            unit_name = u"µm"
            s = 1e6
        elif s < 0:
            unit_name = "mm"
            s = 1e3
        else:
            unit_name = "m"
            s = 1
        return d, z, x * s, y * s, unit_name

    def op(self, pci: HoloTomo):
        d, z, x, y, unit_name = self.pre_imshow(pci)
        if self.title is not None:
            plt.suptitle(self.title)
        else:
            plt.suptitle("Psi [%s]" % self.type)
        nrows, ncols = d.shape[:2]
        for irow in range(nrows):
            for icol in range(ncols):
                plt.subplot(nrows, ncols, irow * ncols + icol + 1)
                di = d[irow, icol]
                if self.type.lower() == 'rgba':
                    rgba = complex2rgbalin(di)
                    plt.imshow(rgba, extent=(x.min(), x.max(), y.min(), y.max()))
                    insertColorwheel(left=.02, bottom=.0, width=.1, height=.1, text_col='black', fs=10)
                elif self.type.lower() == 'amplitude':
                    plt.imshow(np.abs(di), extent=(x.min(), x.max(), y.min(), y.max()), cmap=plt.cm.get_cmap('gray'))
                    plt.colorbar()
                else:
                    plt.imshow(np.angle(di), extent=(x.min(), x.max(), y.min(), y.max()), cmap=plt.cm.get_cmap('gray'))
                    plt.colorbar()

                if icol == 0:
                    plt.ylabel("Y (%s) [view #%d]" % (unit_name, irow))
                if irow == 0:
                    plt.title("Z = %8.5fm" % (z[icol]))
                if irow == nrows - 1:
                    plt.xlabel("X (%s)" % (unit_name))

        try:
            plt.draw()
            plt.gcf().canvas.draw()
            plt.pause(.001)
        except:
            pass

        if self.type.lower() in ['rgba']:
            insertColorwheel(left=.02, bottom=.0, width=.1, height=.1, text_col='black', fs=10)
        return pci

    def timestamp_increment(self, pci):
        pass
