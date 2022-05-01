# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2020-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import numpy as np
from scipy.fftpack import fftshift, fftfreq
from ...mpi import MPI
from .. import cpu_operator as cpuop, Ptycho
from .split import PtychoSplit
from ...utils.plot_utils import show_obj_probe


class ShowObjProbe(cpuop.ShowObjProbe):
    def __init__(self, fig_num=-1, title=None):
        super(ShowObjProbe, self).__init__(fig_num, title)

    def op(self, p: Ptycho):
        # This should not happen
        if MPI is None or not isinstance(p, PtychoSplit):
            return super(ShowObjProbe, self).op(p)

        p.stitch()
        if not p.mpi_master:
            return p

        if p.data.near_field:
            show_obj_probe(p.mpi_obj, p.get_probe(), stit=self.title, fig_num=self.fig_num,
                           pixel_size_object=p.pixel_size_object, scan_area_obj=None, scan_area_probe=None,
                           scan_pos=p.get_mpi_scan_area_points())
        else:
            obj = p.mpi_obj
            probe = p.get_probe()
            if self.remove_obj_phase_ramp:
                # print("ShowObjProbe: remove_obj_phase_ramp=(%6.3f, %6.3f)" % (p.data.phase_ramp_dx,
                #                                                               p.data.phase_ramp_dy))
                ny, nx = probe.shape[-2:]
                nyo, nxo = obj.shape[-2:]
                y, x = np.meshgrid(fftshift(fftfreq(nyo, d=ny / nyo)).astype(np.float32),
                                   fftshift(fftfreq(nxo, d=nx / nxo)).astype(np.float32),
                                   indexing='ij')
                obj = obj * np.exp(-2j * np.pi * (x * p.mpi_phase_ramp_dx + y * p.mpi_phase_ramp_dy))
            show_obj_probe(obj, probe, stit=self.title, fig_num=self.fig_num,
                           pixel_size_object=p.pixel_size_object,
                           scan_area_obj=p.get_mpi_scan_area_obj(), scan_area_probe=p.get_scan_area_probe(),
                           scan_pos=p.get_mpi_scan_area_points())
        return p


class PlotPositions(cpuop.PlotPositions):

    def __init__(self, verbose=True, show_plot=True, save_prefix=None, fig_size=(12, 6)):
        super(PlotPositions, self).__init__(verbose=verbose, show_plot=show_plot,
                                            save_prefix=save_prefix, fig_size=fig_size)

    def op(self, p: Ptycho):
        if not isinstance(p, PtychoSplit):
            return super(PlotPositions, self).op(p)
        p.stitch()
        x, y, x0, y0, x_c, y_c = p.get_mpi_pos()
        if p.mpi_master:
            return self.plot(x + x_c, y + y_c, x0 + x_c, y0 + y_c, p)
        else:
            return p


class AnalyseProbe(cpuop.AnalyseProbe):

    def __init__(self, modes=True, focus=True, verbose=True, show_plot=True, save_prefix=None):
        super(AnalyseProbe, self).__init__(modes=modes, focus=focus, verbose=verbose,
                                           show_plot=show_plot, save_prefix=save_prefix)

    def op(self, p: Ptycho):
        if not isinstance(p, PtychoSplit):
            return super(AnalyseProbe, self).op(p)
        if p.mpi_master:
            return super(AnalyseProbe, self).op(p)
        else:
            return p
