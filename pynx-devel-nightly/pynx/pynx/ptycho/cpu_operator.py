# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


import os
import psutil
import types
import time
from sys import stdout
import warnings
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift, fftfreq
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.ndimage.measurements import center_of_mass
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pynx.utils.matplotlib import pyplot as plt
from .ptycho import Ptycho, OperatorPtycho, algo_string
from ..utils.plot_utils import show_obj_probe, complex2rgbalin, colorwheel
from .shape import get_view_coord
from ..operator import has_attr_not_none, OperatorSum, OperatorPower, OperatorException
from ..utils.math import ortho_modes
from . import analysis
from ..version import get_git_version

_pynx_version = get_git_version()
from ..wavefront import Wavefront, operator, UserWarningWavefrontNearFieldPropagation


################################################################################################
# Patch Ptycho class so that we can use 5*w to scale it.
# OK, so this might be ugly. There will definitely be issues if several types of operators
# are imported (e.g. OpenCL and CUDA)
# Solution (?): in a different sub-module, implement dynamical type-checking to decide which
# Scale() operator to call.

def patch_method(cls):
    def __rmul__(self, x):
        # Multiply object by a scalar.
        if np.isscalar(x) is False:
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s" % (str(x), str(self)))
        return Scale(x) * self

    def __mul__(self, x):
        # Multiply object by a scalar.
        if np.isscalar(x) is False:
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s" % (str(x), str(self)))
        return self * Scale(x)

    cls.__rmul__ = __rmul__
    cls.__mul__ = __mul__


patch_method(Ptycho)


#################################################################################################################
###############################  Base CPU operator  #############################################################
#################################################################################################################


class CPUOperatorPtycho(OperatorPtycho):
    """
    Base class for a operators on CDI objects using CPU
    """

    def __init__(self):
        super(CPUOperatorPtycho, self).__init__()

        self.Operator = CPUOperatorPtycho
        self.OperatorSum = CPUOperatorPtychoSum
        self.OperatorPower = CPUOperatorPtychoPower

    def set_stack_size(self, s):
        """
        Change the number of frames which are stacked to perform all operations in //. If it
        is larger than the total number of frames, operators like AP, DM, ML will loop over
        all the stacks. Ignored for CPU operators
        :param s: an integer number (default=16)
        :return: nothing
        """
        pass

    def apply_ops_mul(self, pty):
        """
        Apply the series of operators stored in self.ops to a Ptycho2D object.
        The operators are applied one after the other.

        :param pty: the Ptycho2D to which the operators will be applied.
        :return: the Ptycho2D object, after application of all the operators in sequence
        """
        if isinstance(pty, Ptycho) is False:
            raise OperatorException(
                "ERROR: tried to apply operator:\n    %s \n  to:\n    %s\n  which is not a Ptycho object" % (
                    str(self), str(pty)))
        return super(CPUOperatorPtycho, self).apply_ops_mul(pty)

    def prepare_data(self, p):
        """
        Make sure the data to be used is in the correct memory (host or GPU) for the operator.
        Virtual, must be derived.

        :param p: the Ptycho object the operator will be applied to.
        :return:
        """

        if has_attr_not_none(p, "_psi_v") is False:
            # _psi_v is used to hold the complete copy of Psi projections for all stacks, for algorithms
            # such as DM which need them.
            p._psi_v = {}

        if has_attr_not_none(p, '_cpu_view') is False:
            p._cpu_view = {}

    def calc_quadratic_phase_factor(self, p: Ptycho, factor):
        """
        Calculates the quadratic phase factor and stores it in the Ptycho object
        :param factor:
        :return: Nothing
        """
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        need_calc_q = True
        if has_attr_not_none(p, '_cpu_tmp_quad_phase'):
            if np.isclose(p._cpu_tmp_quad_phase[1], abs(factor)):
                need_calc_q = False

        if need_calc_q:
            x = fftfreq(nx) * nx
            y = fftfreq(ny)[:, np.newaxis] * ny

            q = fftshift(np.exp(1j * abs(factor) * (x * x + y * y)))
            p._cpu_tmp_quad_phase = (q.astype(np.complex64), abs(factor))

    def timestamp_increment(self, p):
        p._timestamp_counter += 1
        p._cpu_timestamp_counter = p._timestamp_counter

    def view_register(self, obj):
        """
        Creates a new unique view key in an object. When finished with this view, it should be de-registered
        using view_purge. Note that it only reserves the key, but does not create the view.
        :return: an integer value, which corresponds to yet-unused key in the object's view.
        """
        i = 1
        while i in obj._cpu_view:
            i += 1
        obj._cpu_view[i] = None
        return i

    def view_copy(self, pty, i_source, i_dest):
        if i_source == 0:
            src = {'obj': pty._obj, 'probe': pty._probe, 'psi': pty._psi, 'psi_v': pty._psi_v}
        else:
            src = pty._cpu_view[i_source]
        if i_dest == 0:
            pty._obj = np.empty_like(src['obj'])
            pty._probe = np.empty_like(src['probe'])
            pty._psi = np.empty_like(src['psi'])
            pty._psi_v = {}
            dest = {'obj': pty._obj, 'probe': pty._probe, 'psi': pty._psi, 'psi_v': pty._psi_v}
        else:
            pty._cpu_view[i_dest] = {'obj': np.empty_like(src['obj']), 'probe': np.empty_like(src['probe']),
                                     'psi': np.empty_like(src['psi']), 'psi_v': {}}
            dest = pty._cpu_view[i_dest]

        for i in range(len(src['psi_v'])):
            dest['psi_v'][i] = np.empty_like(src['psi'])

        for s, d in zip([src['obj'], src['probe'], src['psi']] + [v for k, v in src['psi_v'].items()],
                        [dest['obj'], dest['probe'], dest['psi']] + [v for k, v in dest['psi_v'].items()]):
            s[:] = d[:]

    def view_swap(self, pty, i1, i2):
        if i1 != 0:
            if pty._cpu_view[i1] is None:
                # Create dummy value, assume a copy will be made later
                pty._cpu_view[i1] = {'obj': None, 'probe': None, 'psi': None, 'psi_v': None}
        if i2 != 0:
            if pty._cpu_view[i2] is None:
                # Create dummy value, assume a copy will be made later
                pty._cpu_view[i2] = {'obj': None, 'probe': None, 'psi': None, 'psi_v': None}
        if i1 == 0:
            pty._obj, pty._cpu_view[i2]['obj'] = pty._cpu_view[i2]['obj'], pty._obj
            pty._probe, pty._cpu_view[i2]['probe'] = pty._cpu_view[i2]['probe'], pty._probe
            pty._psi, pty._cpu_view[i2]['psi'] = pty._cpu_view[i2]['psi'], pty._psi
            pty._psi_v, pty._cpu_view[i2]['psi_v'] = pty._cpu_view[i2]['psi_v'], pty._psi_v
        elif i2 == 0:
            pty._obj, pty._cpu_view[i1]['obj'] = pty._cpu_view[i1]['obj'], pty._obj
            pty._probe, pty._cpu_view[i1]['probe'] = pty._cpu_view[i1]['probe'], pty._probe
            pty._psi, pty._cpu_view[i1]['psi'] = pty._cpu_view[i1]['psi'], pty._psi
            pty._psi_v, pty._cpu_view[i1]['psi_v'] = pty._cpu_view[i1]['psi_v'], pty._psi_v
        else:
            pty._cpu_view[i1], pty._cpu_view[i2] = pty._cpu_view[i2], pty._cpu_view[i1]
        self.timestamp_increment(pty)

    def view_sum(self, pty, i_source, i_dest):
        if i_source == 0:
            src = {'obj': pty._obj, 'probe': pty._probe, 'psi': pty._psi, 'psi_v': pty._psi_v}
        else:
            src = pty._cpu_view[i_source]
        if i_dest == 0:
            dest = {'obj': pty._obj, 'probe': pty._probe, 'psi': pty._psi, 'psi_v': pty._psi_v}
        else:
            dest = pty._cpu_view[i_dest]
        for s, d in zip([src['obj'], src['probe'], src['psi']] + [v for k, v in src['psi_v'].items()],
                        [dest['obj'], dest['probe'], dest['psi']] + [v for k, v in dest['psi_v'].items()]):
            s[:] += d
        self.timestamp_increment(pty)

    def view_purge(self, pty, i):
        if i is not None:
            del pty._cpu_view[i]
        elif has_attr_not_none(pty, '_cpu_view'):
            del pty._cpu_view


# The only purpose of this class is to make sure it inherits from CLOperatorCDI and has a processing unit
class CPUOperatorPtychoSum(OperatorSum, CPUOperatorPtycho):
    def __init__(self, op1, op2):
        if np.isscalar(op1):
            op1 = Scale(op1)
        if np.isscalar(op2):
            op2 = Scale(op2)
        if isinstance(op1, CPUOperatorPtycho) is False or isinstance(op2, CPUOperatorPtycho) is False:
            raise OperatorException(
                "ERROR: cannot add a CLOperatorCDI with a non-CLOperatorCDI: %s + %s" % (str(op1), str(op2)))
        # We can only have a sum of two CLOperatorCDI, so they must have a processing_unit attribute.
        CPUOperatorPtycho.__init__(self, op1.processing_unit)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CPUOperatorPtycho
        self.OperatorSum = CPUOperatorPtychoSum
        self.OperatorPower = CPUOperatorPtychoPower
        self.prepare_data = types.MethodType(CPUOperatorPtycho.prepare_data, self)
        self.timestamp_increment = types.MethodType(CPUOperatorPtycho.timestamp_increment, self)
        self.view_copy = types.MethodType(CPUOperatorPtycho.view_copy, self)
        self.view_swap = types.MethodType(CPUOperatorPtycho.view_swap, self)
        self.view_sum = types.MethodType(CPUOperatorPtycho.view_sum, self)
        self.view_purge = types.MethodType(CPUOperatorPtycho.view_purge, self)


# The only purpose of this class is to make sure it inherits from CLOperatorCDI and has a processing unit
class CPUOperatorPtychoPower(OperatorPower, CPUOperatorPtycho):
    def __init__(self, op, n):
        CPUOperatorPtycho.__init__(self, op.processing_unit)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CPUOperatorPtycho
        self.OperatorSum = CPUOperatorPtychoSum
        self.OperatorPower = CPUOperatorPtychoPower
        self.prepare_data = types.MethodType(CPUOperatorPtycho.prepare_data, self)
        self.timestamp_increment = types.MethodType(CPUOperatorPtycho.timestamp_increment, self)
        self.view_copy = types.MethodType(CPUOperatorPtycho.view_copy, self)
        self.view_swap = types.MethodType(CPUOperatorPtycho.view_swap, self)
        self.view_sum = types.MethodType(CPUOperatorPtycho.view_sum, self)
        self.view_purge = types.MethodType(CPUOperatorPtycho.view_purge, self)


#################################################################################################################
###############################  Exclusive CPU operators  #######################################################
#################################################################################################################


class ShowObjProbe(CPUOperatorPtycho):
    """
    Class to display object and probe during an optimization
    """

    def __init__(self, fig_num=-1, title=None, remove_obj_phase_ramp=True):
        """

        :param fig_num: the matplotlib figure number. if None, a new figure will be created. if -1 (the default), the
                        current figure will be used.
        :param title: the title for the view. If None, a default title will be used.
        :param remove_obj_phase_ramp: if True, the object will be displayed after removing the phase
                                      ramp coming from the imperfect centring of the diffraction data
                                      (sub-pixel shift). Calculated diffraction patterns using such a
                                      corrected object will present a sub-pixel shift relative to the
                                      diffraction data. The ramp information comes from the PtychoData
                                      phase_ramp_d{x,y} attributes, and are not re-calculated.
        """
        super(ShowObjProbe, self).__init__()
        self.fig_num = fig_num
        self.title = title
        self.remove_obj_phase_ramp = remove_obj_phase_ramp

    def op(self, p: Ptycho):
        if p.data.near_field:
            show_obj_probe(p.get_obj(), p.get_probe(), stit=self.title, fig_num=self.fig_num,
                           pixel_size_object=p.pixel_size_object, scan_area_obj=None, scan_area_probe=None,
                           scan_pos=p.get_scan_area_points())
        else:
            obj = p.get_obj(remove_obj_phase_ramp=self.remove_obj_phase_ramp)

            show_obj_probe(obj, p.get_probe(), stit=self.title, fig_num=self.fig_num,
                           pixel_size_object=p.pixel_size_object,
                           scan_area_obj=p.get_scan_area_obj(), scan_area_probe=p.get_scan_area_probe(),
                           scan_pos=p.get_scan_area_points())
        return p

    def timestamp_increment(self, p):
        # This display operation does not modify the data.
        pass


#################################################################################################################
########################  End of Exclusive CPU operators  #######################################################
#################################################################################################################

#################################################################################################################
# Operators below should be superseeded by GPU ones when available
#################################################################################################################


class FreePU(CPUOperatorPtycho):
    """
    Operator freeing GPU memory. Nothing to do for CPU.
    """

    def op(self, pty):
        return pty


class MemUsage(CPUOperatorPtycho):
    """
    Print memory usage of current process (RSS on host)
    """

    def __init__(self, verbose=True):
        super(MemUsage, self).__init__()
        self.verbose = verbose

    def op(self, p: Ptycho):
        """

        :param p: the ptycho object this operator applies to
        :return: the updated ptycho object
        """
        # TODO: also print in detail array memory used.
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss

        p.print("Mem Usage: RSS= %6.1f Mbytes (process)" % (rss / 1024 ** 2))
        return p


class Scale(CPUOperatorPtycho):
    """
    Multiply the ptycho object by a scalar (real or complex).
    """

    def __init__(self, x, obj=True, probe=True, psi=True):
        """

        :param x: the scaling factor
        :param obj: if True, scale the object
        :param probe: if True, scale the probe
        :param psi: if True, scale the all the psi arrays, _psi as well as _psi_v
        """
        super(Scale, self).__init__()
        self.x = x
        self.obj = obj
        self.probe = probe
        self.psi = psi

    def op(self, p):
        if self.x == 1:
            return p

        if np.isreal(self.x):
            x = np.float32(self.x)
        else:
            x = np.complex64(self.x)

        if self.obj:
            p._obj *= x
        if self.probe:
            p._probe *= x
        if self.psi:
            p._psi *= x
            for i in range(len(p._psi_v)):
                p.cpu_psi_v[i] *= x
        return p


class ObjProbe2Psi(CPUOperatorPtycho):
    """
    Computes Psi = Obj(r) * Probe(r-r_j) for all probe positions.
    """

    def __init__(self):
        super(ObjProbe2Psi, self).__init__()

    def op(self, p):
        # Multiply obj and probe with quadratic phase factor, taking into account all modes (if any)
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))
        self.calc_quadratic_phase_factor(p, f * p.pixel_size_object ** 2)
        obs = p.data.iobs
        n_stack = len(obs)
        p._psi = np.empty((nb_obj, nb_probe, n_stack, ny, nx), dtype=np.complex64)
        px, py = p.data.pixel_size_object()
        for iobj in range(len(p._obj)):
            for iprobe in range(len(p._probe)):
                for i in range(n_stack):
                    dy, dx = p.data.posy[i] / py, p.data.posx[i] / px
                    cx, cy = get_view_coord((nyo, nxo), (ny, nx), dx, dy)
                    op = p._probe[iprobe] * p._obj[iobj, cy:cy + ny, cx:cx + nx]
                    if p.data.near_field is False:
                        op *= p._cpu_tmp_quad_phase[0]
                    p._psi[iobj, iprobe, i] = fftshift(op)
        return p


class FT(CPUOperatorPtycho):
    """
    Forward Fourier-transform a Psi array, i.e. a stack of N=16 Obj*Probe views
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(FT, self).__init__()
        self.scale = scale

    def op(self, pty):
        if self.scale:
            pty._psi = fftn(pty._psi, axes=(-2, -1)) / np.sqrt(pty._psi.shape[-2] * pty._psi.shape[-1])
        else:
            pty._psi = fftn(pty._psi, axes=(-2, -1))
        return pty


class IFT(CPUOperatorPtycho):
    """
    Backward Fourier-transform a Psi array, i.e. a stack of N=16 Obj*Probe views
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(IFT, self).__init__()
        self.scale = scale

    def op(self, pty):
        if self.scale:
            pty._psi = ifftn(pty._psi, axes=(-2, -1)) * np.sqrt(pty._psi.shape[-2] * pty._psi.shape[-1])
        else:
            pty._psi = ifftn(pty._psi, axes=(-2, -1))
        return pty


class QuadraticPhase(CPUOperatorPtycho):
    """
    Operator applying a quadratic phase factor. The quadratic phase factor array will be cached in the object.
    """

    def __init__(self, factor, scale=1):
        """
        Application of a quadratic phase factor, and optionally a scale factor.

        The actual factor is:  :math:`scale * e^{i * factor * ((ix/nx)^2 + (iy/ny)^2)}`
        where ix and iy are the integer indices of the pixels

        :param factor: the factor for the phase calculation.
        :param scale: the data will be scaled by this factor. Useful to normalize before/after a Fourier transform,
                      without accessing twice the array data.
        """
        super(QuadraticPhase, self).__init__()
        self.scale = np.float32(scale)
        self.factor = np.float32(factor)

    def op(self, p):
        self.calc_quadratic_phase_factor(p, self.factor)
        if self.factor > 0:
            p._psi *= self.scale * p.cpu_tmp_quad[0]
        else:
            p._psi *= self.scale / p.cpu_tmp_quad[0]
        return p


class PropagateNearField(CPUOperatorPtycho):
    """
    Near field propagator
    """

    def __init__(self, forward=True):
        """

        :param forward: if True, propagate forward, otherwise backward. The distance is taken from the ptycho data
                        this operator applies to.
        """
        super(PropagateNearField, self).__init__()
        self.forward = forward

    def op(self, p):
        f = np.float32(-np.pi * p.data.wavelength * p.data.detector_distance / p.data.pixel_size_detector ** 2)
        if self.forward is False:
            f = -f
        p = IFT(scale=False) * QuadraticPhase(factor=f) * FT(scale=False) * p
        return p


class Propagate(CPUOperatorPtycho):
    """
    Propagator, either using near or far field
    """

    def __init__(self, forward=True):
        """

        :param forward: if True, propagate forward, otherwise backward. The distance and the near_field flag
                        are taken from the ptycho data this operator applies to.
        """
        super(Propagate, self).__init__()
        self.forward = forward

    def op(self, p):
        if p.data.near_field:
            p = PropagateNearField(forward=self.forward) * p
        else:
            if self.forward:
                p = FT() * p
            else:
                p = IFT() * p
        return p


class Calc2Obs(CPUOperatorPtycho):
    """
    Copy the calculated intensities to the observed ones, optionally with Poisson noise.
    This operator will loop other all stacks of frames, multiply object and probe and
    propagate the wavefront to the detector, and compute the new intensities.
    """

    def __init__(self, nb_photons_per_frame=None, poisson_noise=False):
        """

        """
        super(Calc2Obs, self).__init__()
        self.nb_photons_per_frame = nb_photons_per_frame
        self.poisson_noise = poisson_noise

    def op(self, p: Ptycho):
        p.data.iobs = (np.abs(p._psi) ** 2).sum(axis=(0, 1)) + p.get_background()

        if self.nb_photons_per_frame is not None:
            p.data.iobs *= self.nb_photons_per_frame / (p.data.iobs.sum() / len(p.data.iobs))

        if self.poisson_noise:
            p.data.iobs = np.random.poisson(p.data.iobs).astype(np.float32)

        return p


class ApplyAmplitude(CPUOperatorPtycho):
    """
    Apply the magnitude from observed intensities, keep the phase. Applies to a stack of N=16 views.
    """

    def __init__(self, calc_llk=False):
        """

        :param calc_llk: if True, the log-likelihood will be calculated for this stack.
        """
        super(ApplyAmplitude, self).__init__()
        self.calc_llk = calc_llk

    def op(self, p):
        if self.calc_llk:
            p = LLK() * p
        calc = np.sqrt((abs(p._psi) ** 2).sum(axis=(0, 1)))
        r = np.sqrt(np.fmax(p.data.iobs, 0)) / np.fmax(1e-12, calc)
        r[p.data.iobs < 0] = 1
        p._psi *= r
        return p


class PropagateApplyAmplitude(CPUOperatorPtycho):
    """
    Propagate to the detector plane (either in far or near field, perform the magnitude projection, and propagate
    back to the object plane.
    """

    def __init__(self, calc_llk=False):
        """

        :param calc_llk: if True, calculate llk while in the detector plane.
        """
        super(PropagateApplyAmplitude, self).__init__()
        self.calc_llk = calc_llk

    def op(self, p):
        if p.data.near_field:
            p = PropagateNearField(forward=False) * ApplyAmplitude(calc_llk=self.calc_llk) \
                * PropagateNearField(forward=True) * p
        else:
            p = IFT() * ApplyAmplitude(calc_llk=self.calc_llk) * FT() * p
        return p


class LLK(CPUOperatorPtycho):
    """
    Log-likelihood reduction kernel. Can only be used when Psi is in diffraction space.
    This is a reduction operator - it will write llk as an argument in the Ptycho object, and return the object.
    """

    def op(self, p: Ptycho):
        iobs = p.data.iobs.flatten()
        icalc = (np.abs(p._psi) ** 2).sum(axis=(0, 1)).flatten()

        # Poisson
        llk = np.zeros(iobs.shape, dtype=np.float32)
        idx = np.where(iobs > 0)
        llk[idx] = np.take(icalc - iobs + iobs * np.log(iobs / icalc), idx)
        idx = np.where(iobs == 0)
        llk[idx] = np.take(icalc, idx)
        p.llk_poisson = llk.sum()

        # Gaussian
        p.llk_gaussian = ((iobs - icalc) ** 2 / (iobs + 1)).sum()

        # Euclidian
        p.llk_euclidian = 4 * ((np.sqrt(abs(iobs)) - np.sqrt(icalc)) ** 2).sum()

        p.nb_photons_calc = icalc.sum()

        return p


class Psi2ObjProbe(CPUOperatorPtycho):
    """
    Computes updated Obj(r) and Probe(r-r_j) contributions from Psi.
    """

    def __init__(self, update_object=True, update_probe=False):
        """

        """
        super(Psi2ObjProbe, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe

    def op(self, p):
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector_distance))
        self.calc_quadratic_phase_factor(p, f * p.pixel_size_object ** 2)
        q = p._cpu_tmp_quad_phase[0]
        obs = p.data.iobs
        n_stack = len(obs)
        px, py = p.data.pixel_size_object()
        if self.update_object:
            obj_new = np.zeros_like(p._obj)
            obj_norm = np.zeros_like(p._obj[0], dtype=np.float32)
            for iobj in range(nb_obj):
                for iprobe in range(nb_probe):
                    pr = p._probe[iprobe]
                    for i in range(n_stack):
                        dy, dx = p.data.posy[i] / py, p.data.posx[i] / px
                        cx, cy = get_view_coord((nyo, nxo), (ny, nx), dx, dy)
                        obj_new[iobj, cy:cy + ny, cx:cx + nx] += fftshift(p._psi[iobj, iprobe, i] / q) * pr.conj()
                        obj_norm[cy:cy + ny, cx:cx + nx] += abs(pr) ** 2
            reg = obj_norm.max() * 1e-2

            p._obj = (obj_new + reg * p._obj) / np.fmax(obj_norm + reg, 1e-12)

        if self.update_probe:
            probe_norm = np.zeros_like(p._probe[0], dtype=np.float32)
            probe_old = p._probe.copy()
            p._probe[:] = 0
            for iobj in range(len(p._obj)):
                for iprobe in range(len(p._probe)):
                    for i in range(n_stack):
                        dy, dx = p.data.posy[i] / py, p.data.posx[i] / px
                        cx, cy = get_view_coord((nyo, nxo), (ny, nx), dx, dy)
                        ob = p._obj[iobj, cy:cy + ny, cx:cx + nx]
                        p._probe[iprobe] += fftshift(p._psi[iobj, iprobe, i] / q) * ob.conj()
                        probe_norm += abs(ob) ** 2
            reg = probe_norm.max() * 1e-2
            p._probe = (p._probe + reg * probe_old) / np.fmax(probe_norm + reg, 1e-12)

        return p


class AP(CPUOperatorPtycho):
    """
    Perform a complete Alternating Projection cycle:
    - forward all object*probe views to Fourier space and apply the observed amplitude
    - back-project to object space and project onto (probe, object)
    - update background optionally
    """

    def __init__(self, update_object=True, update_probe=False, update_background=False, floating_intensity=False,
                 nb_cycle=1, calc_llk=False, show_obj_probe=False, fig_num=-1, zero_phase_ramp=True,
                 background_smooth_sigma=0):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param update_background: update background ? [TODO CPU operators]
        :param floating_intensity: optimise floating intensity scale factor [TODO CPU operators]
        :param nb_cycle: number of cycles to perform. Equivalent to AP(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        :param zero_phase_ramp: if True, the conjugate phase ramp in the object and probe will be removed
                                by centring the FT of the probe, at the end and before every display.
                                Ignored for near field.
        :param background_smooth_sigma: gaussian convolution parameter for the background update
            (ignored on CPU)
        """
        super(AP, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.floating_intensity = floating_intensity  # TODO
        self.nb_cycle = nb_cycle
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.zero_phase_ramp = zero_phase_ramp

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new DM operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return AP(update_object=self.update_object, update_probe=self.update_probe,
                  floating_intensity=self.floating_intensity, nb_cycle=self.nb_cycle * n,
                  calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe, fig_num=self.fig_num,
                  zero_phase_ramp=self.zero_phase_ramp)

    def op(self, p: Ptycho):
        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True
            if self.update_background:
                pass  # TODO: update while in Fourier space
            p = PropagateApplyAmplitude(calc_llk=calc_llk) * ObjProbe2Psi() * p

            p = Psi2ObjProbe(update_object=self.update_object, update_probe=self.update_probe) * p

            if calc_llk:
                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=False, algorithm='AP',
                                 verbose=True)
            else:
                p.history.insert(p.cycle, update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=False, algorithm='AP',
                                 verbose=False)
            p.cycle += 1

            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    s = algo_string('AP', p, self.update_object, self.update_probe, self.update_background)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.get_llk('poisson'))
                    if self.zero_phase_ramp:
                        p = ZeroPhaseRamp(obj=True) * p
                    p = ShowObjProbe(fig_num=self.fig_num, title=tit) * p

        if self.zero_phase_ramp and not self.show_obj_probe:
            p = ZeroPhaseRamp(obj=True) * p
        return p


class DM(CPUOperatorPtycho):
    """
    Operator to perform a complete Difference Map cycle, updating the Psi views for all stack of frames,
    as well as updating the object and/or probe.
    """

    def __init__(self, update_object=True, update_probe=True, nb_cycle=1, calc_llk=False, show_obj_probe=False,
                 fig_num=-1, obj_smooth_sigma=0, obj_inertia=0.01, probe_smooth_sigma=0, probe_inertia=0.001,
                 center_probe_n=0, center_probe_max_shift=5, loop_obj_probe=1, zero_phase_ramp=True,
                 background_smooth_sigma=0):
        """
        TODO: Not all parameters are implemented in the CPU DM version.
        :param update_object: update object ?
        :param update_probe: update probe ?
        :param nb_cycle: number of cycles to perform. Equivalent to DM(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        :param zero_phase_ramp: if True, the conjugate phase ramp in the object and probe will be removed
                                by centring the FT of the probe, at the end and before every display.
                                Ignored for near field.
        :param background_smooth_sigma: gaussian convolution parameter for the background update
            (ignored on CPU)
        """
        super(DM, self).__init__()
        self.nb_cycle = nb_cycle
        self.update_object = update_object
        self.update_probe = update_probe
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.zero_phase_ramp = zero_phase_ramp

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new DM operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return DM(update_object=self.update_object, update_probe=self.update_probe, nb_cycle=self.nb_cycle * n,
                  calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe, fig_num=self.fig_num,
                  zero_phase_ramp=self.zero_phase_ramp)

    def op(self, p):
        # Calculate starting Psi
        p = ObjProbe2Psi() * p

        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            # 2 * ObjProbe2Psi() - 1
            psi1 = p._psi.copy()
            p = ObjProbe2Psi() * p
            psi0 = p._psi.copy()
            p._psi = 2 * p._psi - psi1

            p = PropagateApplyAmplitude() * p

            # Psi(n+1) = Psi(n) - P*O + Psi_fourier
            p._psi += psi1 - psi0

            p = Psi2ObjProbe(update_object=self.update_object, update_probe=self.update_probe) * p

            if calc_llk:
                psi1 = p._psi.copy()
                # We need to perform a loop for LLK as the DM2 loop is on (2*PO-I), not the current PO estimate
                if p.data.near_field:
                    p = LLK() * PropagateNearField(forward=True) * ObjProbe2Psi() * p
                else:
                    p = LLK() * FT() * ObjProbe2Psi() * p

                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=False, update_pos=False, algorithm='DM',
                                 verbose=True)
                # Restore correct Psi
                p._psi = psi1
            else:
                p.history.insert(p.cycle, update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=False, update_pos=False, algorithm='DM',
                                 verbose=False)

            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    s = algo_string('DM', p, self.update_object, self.update_probe)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.get_llk('poisson'))
                    if self.zero_phase_ramp:
                        p = ZeroPhaseRamp(obj=False) * p
                    p = ShowObjProbe(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1

        if self.zero_phase_ramp and not self.show_obj_probe:
            p = ZeroPhaseRamp(obj=False) * p
        return p


class ML(CPUOperatorPtycho):
    """
    Operator to perform a maximum-likelihood conjugate-gradient minimization.
    """

    def __init__(self, nb_cycle=1, update_object=True, update_probe=False, update_background=False,
                 reg_fac_obj=0, reg_fac_probe=0, calc_llk=False, show_obj_probe=False, fig_num=-1,
                 zero_phase_ramp=True):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param update_background: update background ?
        :param reg_fac_obj: use this regularization factor for the object (if 0, no regularization)
        :param reg_fac_probe: use this regularization factor for the probe (if 0, no regularization)
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        :param zero_phase_ramp: if True, the conjugate phase ramp in the object and probe will be removed
                                by centring the FT of the probe, at the beginning and end. Ignored for near field.
        """
        super(ML, self).__init__()
        self.nb_cycle = nb_cycle
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.reg_fac_obj = reg_fac_obj
        self.reg_fac_probe = reg_fac_probe
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num
        self.zero_phase_ramp = zero_phase_ramp

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new ML operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return ML(nb_cycle=self.nb_cycle * n, update_object=self.update_object, update_probe=self.update_probe,
                  update_background=self.update_background, reg_fac_obj=self.reg_fac_obj,
                  reg_fac_probe=self.reg_fac_probe, calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe,
                  fig_num=self.fig_num, zero_phase_ramp=self.zero_phase_ramp)

    def op(self, p):
        # First perform an AP cycle to make sure object and probe are properly scaled with respect to iobs
        p = AP(update_object=self.update_object, update_probe=self.update_probe,
               update_background=self.update_background, zero_phase_ramp=self.zero_phase_ramp) * p
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        obs = p.data.iobs
        n_stack = len(obs)

        # Create arrays for ML
        obj_dir = np.empty_like(p._obj)
        probe_dir = np.empty_like(p._probe)
        background_dir = np.empty_like(p._background)

        if self.update_object:
            obj_grad = np.empty_like(p._obj)
            obj_grad_last = np.empty_like(p._obj)

        if self.update_probe:
            probe_grad = np.empty_like(p._probe)
            probe_grad_last = np.empty_like(p._probe)

        if self.update_background:
            background_grad = np.empty_like(p._background)

        for ic in range(self.nb_cycle):
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            # Swap gradient arrays - for CG, we need the previous gradient
            if self.update_object:
                obj_grad, obj_grad_last = obj_grad_last, obj_grad

            if self.update_probe:
                probe_grad, probe_grad_last = probe_grad_last, probe_grad

            if self.update_background:
                background_grad, background_grad_last = background_grad_last, background_grad

            # 1) Compute the gradients (actually the gradients conjugates)
            p = Propagate(forward=True) * ObjProbe2Psi() * p
            calc = (abs(p._psi) ** 2).sum(axis=(0, 1))
            p._psi *= (1 - obs / calc) * (obs >= 0)  # with broadcasting
            p = Propagate(forward=False) * p

            px, py = p.data.pixel_size_object()
            if self.update_object:
                obj_grad[:] = 0
                for iobj in range(nb_obj):
                    for iprobe in range(nb_probe):
                        pr = p._probe[iprobe]
                        for i in range(n_stack):
                            dy, dx = p.data.posy[i] / py, p.data.posx[i] / px
                            cx, cy = get_view_coord((nyo, nxo), (ny, nx), dx, dy)
                            obj_grad[iobj, cy:cy + ny, cx:cx + nx] += fftshift(
                                p._psi[iobj, iprobe, i] / p._cpu_tmp_quad_phase[0]) * pr.conj()

            if self.update_probe:
                probe_grad[:] = 0
                for iobj in range(len(p._obj)):
                    for iprobe in range(len(p._probe)):
                        for i in range(n_stack):
                            dy, dx = p.data.posy[i] / py, p.data.posx[i] / px
                            cx, cy = get_view_coord((nyo, nxo), (ny, nx), dx, dy)
                            ob = p._obj[iobj, cy:cy + ny, cx:cx + nx]
                            probe_grad[iprobe] += fftshift(
                                p._psi[iobj, iprobe, i] / p._cpu_tmp_quad_phase[0]) * ob.conj()
            if False:
                # TODO: debug only
                if self.update_object:
                    p._obj_grad = obj_grad.copy()
                    p._obj_grad_last = obj_grad_last.copy()
                if self.update_probe:
                    p._probe_grad = probe_grad.copy()
                    p._probe_grad_last = probe_grad_last.copy()

            # 2) Search direction
            if ic == 0:
                # first cycle
                if self.update_object:
                    obj_dir = obj_grad.copy()
                if self.update_probe:
                    probe_dir = probe_grad.copy()
                if self.update_background:
                    background_dir = background_grad.copy()
            else:
                beta_d, beta_n = 0, 0
                # Polak-Ribière CG coefficient
                if self.update_object:
                    beta_n += (obj_grad.real * (obj_grad.real - obj_grad_last.real)).sum()
                    beta_n += (obj_grad.imag * (obj_grad.imag - obj_grad_last.imag)).sum()
                    beta_d += (abs(obj_grad_last) ** 2).sum()
                if self.update_probe:
                    beta_n += (probe_grad.real * (probe_grad.real - probe_grad_last.real)).sum()
                    beta_n += (probe_grad.imag * (probe_grad.imag - probe_grad_last.imag)).sum()
                    beta_d += (abs(probe_grad_last) ** 2).sum()
                if self.update_background:
                    beta_n += (background_grad.real * (background_grad.real - background_grad_last.real)).sum()
                    beta_n += (background_grad.imag * (background_grad.imag - background_grad_last.imag)).sum()
                    beta_d += (abs(background_grad_last) ** 2).sum()
                # print("Beta= %e / %e"%(beta_n, beta_d))
                # Reset direction if beta<0 => beta=0
                beta = np.float32(max(0, beta_n / max(1e-20, beta_d)))
                if np.isnan(beta_n + beta_d) or np.isinf(beta_n + beta_d):
                    raise OperatorException("CPU ML(): NaN beta")
                if self.update_object:
                    obj_dir = beta * obj_dir - obj_grad
                if self.update_probe:
                    probe_dir = beta * probe_dir - probe_grad
                if self.update_background:
                    background_dir = beta * background_dir - background_grad
                # print("Beta= %f = %f / %f" % (beta_n / beta_d, beta_n, beta_d), abs(obj_dir).mean(),
                #       abs(probe_dir).mean())

            if False:
                # TODO: debug only
                p._obj_dir = obj_dir.copy()
                p._probe_dir = probe_dir.copy()

            # 3) Line minimization
            p = Propagate(forward=True) * ObjProbe2Psi() * p
            po = p._psi.copy().astype(np.complex128)  # Change type to avoid overflows

            if self.update_object:
                p._obj, obj_dir = obj_dir, p._obj
                p = Propagate(forward=True) * ObjProbe2Psi() * p
                pdo = p._psi.copy().astype(np.complex128)
                p._obj, obj_dir = obj_dir, p._obj
            else:
                pdo = np.zeros_like(p._psi)

            if self.update_probe:
                p._probe, probe_dir = probe_dir, p._probe
                p = Propagate(forward=True) * ObjProbe2Psi() * p
                dpo = p._psi.copy().astype(np.complex128)
                p._probe, probe_dir = probe_dir, p._probe
            else:
                dpo = np.zeros_like(p._psi)

            if self.update_object and self.update_probe:
                p._obj, obj_dir = obj_dir, p._obj
                p._probe, probe_dir = probe_dir, p._probe
                p = Propagate(forward=True) * ObjProbe2Psi() * p
                dpdo = p._psi.copy().astype(np.complex128)
                p._obj, obj_dir = obj_dir, p._obj
                p._probe, probe_dir = probe_dir, p._probe
            else:
                dpdo = np.zeros_like(p._psi)

            gamma_n = ((po.conj() * (dpo + pdo)).real * (obs / calc - 1)).sum()
            tmp = ((abs(dpo + pdo) ** 2 + 2 * (po.conj() * dpdo).real)).sum(axis=(0, 1)) * (1 - obs / calc)
            gamma_d = (tmp + 2 * obs / calc ** 2 * (po.conj() * (dpo + pdo)).real.sum(axis=(0, 1)) ** 2).sum()

            # print("Gamma= %f = %f / %f" % (gamma_n / gamma_d, gamma_n, gamma_d), abs(po).mean(), abs(dpo).mean(),
            #       abs(pdo).mean(), abs(dpdo.max()))

            if self.reg_fac_obj != 0:
                pass
                # TODO
            if self.reg_fac_probe != 0:
                # TODO
                pass

            if np.isnan(gamma_d + gamma_n) or np.isinf(gamma_d + gamma_n):
                raise OperatorException("CPU ML(): Gamma = NaN ! :", gamma_d, gamma_n)
            gamma = gamma_n / gamma_d

            # 4) Object and/or probe and/or background update
            if self.update_object:
                # print("Obj change:", abs(p._obj).mean(), abs(obj_dir * gamma).mean())
                p._obj += obj_dir * gamma

            if self.update_probe:
                p._probe += probe_dir * gamma

            if self.update_background:
                p._background += background_dir * gamma

            if calc_llk:
                p = LLK() * Propagate(forward=True) * ObjProbe2Psi() * p

                p.update_history(mode='llk', update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=False, algorithm='ML',
                                 verbose=True)
            else:
                p.history.insert(p.cycle, update_obj=self.update_object, update_probe=self.update_probe,
                                 update_background=self.update_background, update_pos=False, algorithm='ML',
                                 verbose=False)

            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    s = algo_string('ML', p, self.update_object, self.update_probe, self.update_background)
                    tit = "%s #%3d, LLKn(p)=%8.3f" % (s, ic, p.get_llk('poisson'))
                    p = ShowObjProbe(fig_num=self.fig_num, title=tit) * p
            p.cycle += 1

        if self.zero_phase_ramp:
            p = ZeroPhaseRamp(obj=True) * p
        return p


class ScaleObjProbe(CPUOperatorPtycho):
    """
    Scale object and probe so that they have the same magnitude.
    """

    def __init__(self, verbose=False):
        """

        :param verbose: print deviation if verbose=True
        """
        super(ScaleObjProbe, self).__init__()
        self.verbose = verbose

    def op(self, p):
        # TODO: scale taking into account the mask ?
        nb_photons_obs = p.data.iobs.sum()
        p = ObjProbe2Psi() * p
        nb_photons_calc = (np.abs(p._psi) ** 2).sum()
        s = np.sqrt(nb_photons_obs / nb_photons_calc)
        os = np.abs(p._obj).sum()
        ps = np.abs(p._probe).sum()
        p._probe *= np.sqrt(os / ps * s)
        p._obj *= np.sqrt(ps / os * s)
        if self.verbose:
            p.print("ScaleObjProbe:", ps, os, s, np.sqrt(os / ps * s), np.sqrt(ps / os * s))
        return p


class ApplyPhaseRamp(CPUOperatorPtycho):
    """
    Apply a given phase ramp to the object and/or probe. The actual phase factor is:
    For the probe: np.exp(-2j*np.pi*(x * dx + y*dy))
    For the object: np.exp(2j*np.pi*(x * dx + y*dy))
    Where (x,y) are reduced pixel coordinates going from -.5 to .5 for the probe,
    and  +/-0.5 * nxo/nx (or nyo/ny) for the object.

    If the phase ramp is applied to both object and probe, the calculated
    intensity patterns remain unchanged.
    """

    def __init__(self, dx, dy, probe=False, obj=False):
        """

        :param dx, dy: relative shifts from the centre, calculated in reciprocal space
                       (probe array pixel coordinates)
        :param obj: if True, apply the correction to the object.
        :param probe: if True, apply the correction to the probe.
        """
        super(ApplyPhaseRamp, self).__init__()
        self.probe = probe
        self.obj = obj
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)

    def op(self, p: Ptycho):
        nz, ny, nx = np.int32(p._probe.shape[0]), np.int32(p._probe.shape[1]), np.int32(p._probe.shape[2])
        nyo, nxo = np.int32(p._obj.shape[1]), np.int32(p._obj.shape[2])

        # Corr ramp with opposite signs for object and probe
        if self.probe:
            y, x = np.meshgrid(fftshift(fftfreq(ny)), fftshift(fftfreq(nx)), indexing='ij')
            p._probe *= np.exp(-2j * np.pi * (x * self.dx + y * self.dy))
        if self.obj:
            y, x = np.meshgrid(fftshift(fftfreq(nyo, d=ny / nyo)).astype(np.float32),
                               fftshift(fftfreq(nxo, d=nx / nxo)).astype(np.float32), indexing='ij')
            p._obj *= np.exp(2j * np.pi * (x * self.dx + y * self.dy))
        return p


class ZeroPhaseRamp(CPUOperatorPtycho):
    """
    Operator to remove the linear phase ramp on both the object and the probe, for far field ptycho.
    This first computes the center of mass of the square norm of Fourier transform of the probe,
    and then corrects both probe and object for the phase ramp corresponding to the shift
    relative to the center of the array.

    Then, a remaining phase ramp in the object can optionally be computed by:
    - computing the sum of the calculated intensity along all frames
    - computing the shift of the center of mass of that intensity w/r to the array center
    - the phase ramp parameters are stored in the ptycho object, but not applied as the calculated
    """

    def __init__(self, obj=False):
        """

        :param obj: if True, after correcting both object and probe from the probe phase ramp,
                    the object phase ramp is evaluated from the center of the calculated diffraction.
        :param dx, dy: probe shifts to correct for. Should be None, and will be calculated, but
                       can be supplied for manual correction
        """
        super(ZeroPhaseRamp, self).__init__()
        self.obj = obj

    def op(self, p: Ptycho):
        if p.data.near_field:
            return p
        ny, nx = np.int32(p._probe.shape[1]), np.int32(p._probe.shape[2])

        cyx = center_of_mass((abs(fftshift(fftn(fftshift(p._probe)))) ** 2).sum(axis=0))
        dx, dy = cyx[1] - nx / 2, cyx[0] - ny / 2
        # print("ZeroPhaseRamp(): (dx, dy)[probe] = (%6.3f, %6.3f)" % (dx, dy))

        p = ApplyPhaseRamp(dx, dy, obj=True, probe=True) * p

        if self.obj:
            # Compute the shift of the calculated frame to determine the object ramp
            p = Propagate() * ObjProbe2Psi() * p
            icalc_sum = (abs(p._psi) ** 2).sum(axis=(0, 1, 2))

            # Compute shift of center of mass
            cyx = center_of_mass(fftshift(icalc_sum))
            dx, dy = cyx[1] - nx / 2, cyx[0] - ny / 2
            # print("ZeroPhaseRamp(): (dx, dy)[obj] = (%6.3f, %6.3f)[calc]" % (dx, dy))
            p.data.phase_ramp_dx = dx
            p.data.phase_ramp_dy = dy

        return p


class OrthoProbe(CPUOperatorPtycho):
    """
    Orthogonalise probe modes. If only one mode is present, nothing is done.
    """

    def __init__(self, verbose=False):
        """

        :param verbose: if True, will print the relative intensities of the different modes
        """
        super(OrthoProbe, self).__init__()
        self.verbose = verbose

    def op(self, p: Ptycho):
        if p.get_probe().shape[0] == 1:
            return p
        p.set_probe(ortho_modes((p.get_probe())))
        if self.verbose:
            stdout.write("Orthogonalised probe modes relative intensities: ")
            pr = p.get_probe()
            va = (np.abs(pr) ** 2).sum(axis=(1, 2))
            for a in va / va.sum():
                stdout.write(" %5.2f%%" % (a * 100))
            print("\n")
        return p


class AnalyseProbe(CPUOperatorPtycho):
    """
    Analyse the probe modes and focus. This can produce two plots of the orthogonal modes, and of the probe
    propagation to the focus.
    """

    def __init__(self, modes=True, focus=True, verbose=True, show_plot=True, save_prefix=None):
        """

        :param modes: if True (the default), will analyse the probe modes
        :param focus: if True (the default), will compute the probe propagation to find the focal position
        :param verbose: if True, will print some information about the modes and/or focus
        :param show_plot: if True, show the calculated plots in new figures.
        :param save_prefix: e.g. "path/to/RunAnalysis01" will save plots as "path/to/RunAnalysis01-probe-modes.png" and
                            "path/to/RunAnalysis01-probe-z.png". If None, the plots are not saved.
        """
        super(AnalyseProbe, self).__init__()
        self.modes = modes
        self.focus = focus
        self.verbose = verbose
        self.show_plot = show_plot
        self.save_prefix = save_prefix

    def op(self, p: Ptycho):
        # Analyse probe modes
        probe = p.get_probe()
        if probe.shape[0] > 1 and self.modes:
            if self.show_plot or (self.save_prefix is not None):
                d, a, fig = analysis.modes(p.get_probe(), p.data.pixel_size_object()[0],
                                           do_plot=True, show_plot=self.show_plot, verbose=self.verbose)
            else:
                d, a = analysis.modes(p.get_probe(), p.data.pixel_size_object()[0],
                                      do_plot=False, show_plot=False, verbose=self.verbose)

            probe = d[0]
            if self.save_prefix is not None or self.show_plot:
                dy = (6 + 1) / 72 / fig.get_size_inches()[1]
                fig.text(dy / 5, dy / 2,
                         "PyNX v%s, %s" % (_pynx_version, time.strftime("%Y/%m/%d %H:%M:%S")),
                         fontsize=6, horizontalalignment='left', stretch='condensed')
                canvas = FigureCanvasAgg(fig)
                if self.save_prefix is not None:
                    filename = self.save_prefix + '-probe-modes.png'
                    canvas.print_figure(filename, dpi=150)
                    print("Saving probe modes to:%s" % (filename))
                if self.show_plot:
                    try:
                        plt.draw()
                        plt.gcf().canvas.draw()
                        plt.pause(.001)
                    except:
                        pass

        else:
            probe = probe[0]
        if self.focus:
            # Analyse probe propagation of main mode
            with warnings.catch_warnings() as w:
                warnings.simplefilter('ignore', UserWarningWavefrontNearFieldPropagation)
                p3d, vdz, izmax = analysis.probe_propagate(probe, (-2e-3, 2e-3, 200), p.data.pixel_size_object()[0],
                                                           p.data.wavelength, do_plot=False)
                # Automatically adapt propagation range
                z0 = np.linspace(-2e-3, 2e-3, 200)[izmax] / 2
                dz = np.abs(2 * z0)
                if dz < 300e-6:
                    dz = 300e-6
                if self.show_plot or (self.save_prefix is not None):
                    p3d, vdz, izmax, fig = analysis.probe_propagate(probe, (z0 - dz, z0 + dz, 600),
                                                                    p.data.pixel_size_object()[0], p.data.wavelength,
                                                                    do_plot=True, show_plot=self.show_plot)
                else:
                    p3d, vdz, izmax = analysis.probe_propagate(probe, (z0 - dz, z0 + dz, 600),
                                                               p.data.pixel_size_object()[0], p.data.wavelength,
                                                               do_plot=False)

            if self.verbose:
                print("\nProbe statistics at found focus position (z=%+6.1fum):" % (vdz[izmax] * 1e6))
                analysis.probe_fwhm(p3d[izmax], p.data.pixel_size_object()[0])
            if self.save_prefix is not None or self.show_plot:
                dy = (6 + 1) / 72 / fig.get_size_inches()[1]
                fig.text(dy / 5, dy / 2,
                         "PyNX v%s, %s" % (_pynx_version, time.strftime("%Y/%m/%d %H:%M:%S")),
                         fontsize=6, horizontalalignment='left', stretch='condensed')
                if self.save_prefix is not None:
                    canvas = FigureCanvasAgg(fig)
                    filename = self.save_prefix + '-probe-z.png'
                    canvas.print_figure(filename, dpi=150)
                    print("Saving propagated probe plot to:%s" % (filename))
                if self.show_plot:
                    try:
                        plt.draw()
                        plt.gcf().canvas.draw()
                        plt.pause(.001)
                    except:
                        pass
        return p


class PlotPositions(CPUOperatorPtycho):
    """
    Plot the position shifts. If the positions have not changed from the original ones, nothing is plotted or saved.
    """

    def __init__(self, verbose=True, show_plot=True, save_prefix=None, fig_size=(12, 6)):
        """

        :param verbose: if True, will print some information about the modes and/or focus
        :param show_plot: if True, show the calculated plots in new figures.
        :param save_prefix: e.g. "path/to/RunAnalysis01" will save plots as "path/to/RunAnalysis01-probe-modes.png" and
                            "path/to/RunAnalysis01-probe-z.png". If None, the plots are not saved.
        :param fig_size: the figure size
        """
        super(PlotPositions, self).__init__()
        self.verbose = verbose
        self.show_plot = show_plot
        self.save_prefix = save_prefix
        self.fig_size = fig_size

    def op(self, p: Ptycho):
        return self.plot(p.data.posx + p.data.posx_c, p.data.posy + p.data.posy_c,
                         p.data.posx0 + p.data.posx_c, p.data.posy0 + p.data.posy_c, p)

    def plot(self, x, y, x0, y0, p: Ptycho):
        if self.show_plot is False and self.save_prefix is None:
            # Dumb !
            return p
        dx, dy = x - x0, y - y0
        r0 = np.sqrt(((x.max()) - x.min()) ** 2 + ((y.max()) - y.min()) ** 2)

        px, py = p.data.pixel_size_object()
        if np.allclose(dx / px, 0, atol=0.05) and np.allclose(dy / py, 0, atol=0.05):
            if self.verbose:
                print("Max shift in positions is too small (<0.05 pixel), not plotting")
            return p

        s = np.log10(r0)
        if s < -6:
            unit_name = "nm"
            s = 1e9
        elif s < -3:
            unit_name = "µm"
            s = 1e6
        elif s < 0:
            unit_name = "mm"
            s = 1e3
        else:
            unit_name = "m"
            s = 1

        # Estimate average distance between neighbours to scale the arrows
        xy = np.empty((len(x), 2))
        xy[:, 1] = x
        xy[:, 0] = y
        kdt = cKDTree(xy)
        dists = kdt.query(xy, 2)[0][:, 1]
        avg_dists = dists.mean()
        dr = np.sqrt(dx ** 2 + dy ** 2)
        scale = avg_dists / dr.max()
        if self.verbose:
            print("Average distance between points: %5f %s" % (avg_dists * s, unit_name))
            print("Shift in positions: mean=%f %s, max=%f%s" % (dr.mean() * s, unit_name, dr.max() * s, unit_name))
            print("Scale:", scale)
        if scale < 1:
            scale = 1
        else:
            scale = int(np.round(scale))

        if self.show_plot:
            try:
                fig = plt.figure(203, figsize=self.fig_size)
                plt.clf()
            except:
                # no GUI or $DISPLAY
                fig = Figure(figsize=self.fig_size)
        else:
            fig = Figure(figsize=self.fig_size)
        ax = fig.add_subplot(121)
        figdy = (6 + 1) / 72 / fig.get_size_inches()[1]
        ax.quiver(x0 * s, y0 * s, dx * s, dy * s, scale=1 / scale, angles='xy', scale_units='xy')

        ax.set_xlabel(u"x(%s)" % unit_name, horizontalalignment='left')
        ax.set_ylabel(u"y(%s)" % unit_name)
        fig.suptitle("Position shifts")
        ax.set_title("arrow scale: x%d, max shift=%5f%s" % (scale, dr.max() * s, unit_name), fontsize=9)
        ax.set_aspect('equal')
        # plt.gca().xaxis.set_label_coords(1.05, -.05)
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()

        ax = fig.add_subplot(122)
        # Create a heatmap of displacements
        # plt.quiver(x * s, y * s, dx * s, dy * s, scale=1 / scale, angles='xy', scale_units='xy')
        ax.scatter(x0 * s, y0 * s, 1, color='k')
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        grid_y, grid_x = np.mgrid[y1 / s:y2 / s:400j, x1 / s:x2 / s:400j]
        grid_ph = griddata((x0, y0), np.arctan2(dy, dx), (grid_x, grid_y), method='nearest')
        grid_ampl = griddata((x0, y0), dr, (grid_x, grid_y), method='cubic', fill_value=0)
        rgba = complex2rgbalin(np.ma.masked_array(grid_ampl * np.exp(1j * grid_ph), mask=np.isnan(grid_ampl)))
        ax.imshow(rgba, origin='lower', extent=(x1, x2, y1, y2))
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax = fig.add_axes((0.90, 0.20, 0.08, .08), facecolor='w')  # [left, bottom, width, height]
        colorwheel(ax=ax)

        fig.text(figdy / 5, figdy / 2, "PyNX v%s, %s" % (_pynx_version, time.strftime("%Y/%m/%d %H:%M:%S")),
                 fontsize=6, horizontalalignment='left', stretch='condensed')
        if self.save_prefix is not None:
            canvas = FigureCanvasAgg(fig)
            filename = self.save_prefix + '-positions.png'
            canvas.print_figure(filename, dpi=150)
            print("Saving positions plot to:%s" % filename)
        if self.show_plot:
            try:
                plt.draw()
                plt.gcf().canvas.draw()
                plt.pause(.001)
            except:
                pass

        return p

    def timestamp_increment(self, p):
        # This operator does not alter the ptycho object or GPU arrays
        pass


class CalcIllumination(CPUOperatorPtycho):
    """Compute the integrated illumination of the object by all probe positions
    """

    def op(self, p: Ptycho):
        pr = p._probe.copy()
        nprobe, ny, nx = pr.shape
        if p.data.padding:
            padding = p.data.padding
            pr[:, :padding] = 0
            pr[:, ny - padding:] = 0
            pr[:, :, :padding] = 0
            pr[:, :, nx - padding:] = 0
        nyo, nxo = p._obj.shape[-2:]
        px, py = p.data.pixel_size_object()
        obj_illumination = np.zeros_like(p._obj[0], dtype=np.float32)
        for i in range(len(p.data.iobs)):
            dy, dx = p.data.posy[i] / py, p.data.posx[i] / px
            cx, cy = get_view_coord((nyo, nxo), (ny, nx), dx, dy, integer=True)
            obj_illumination[cy:cy + ny, cx:cx + nx] += (abs(pr) ** 2).sum(axis=0)
        p._obj_illumination = obj_illumination
        return p


class SelectStack(OperatorPtycho):
    """
    Operator to select a stack of observed frames to work on. This operator currently does nothing on a CPU, as
    we assume there are no memory limitations on the CPU side.
    """

    def __init__(self, stack_i):
        """

        :param stack_i: the stack index.
        """
        super(SelectStack, self).__init__()

    def op(self, p):
        return p


class LoopStack(OperatorPtycho):
    """
    Operator to apply a given operator to the complete stack of a ptycho object.
    The CPU operator only applies the supplied operator once, as no loop is required.
    """

    def __new__(cls, op):
        return op
