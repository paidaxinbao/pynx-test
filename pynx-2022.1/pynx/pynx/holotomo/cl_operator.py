# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import warnings
import types
import timeit
import gc
import numpy as np
from ..operator import OperatorException

from ..processing_unit.cl_processing_unit import CLProcessingUnit
from ..processing_unit.kernel_source import get_kernel_source as getks
from ..processing_unit import default_processing_unit as main_default_processing_unit
import pyopencl as cl
import pyopencl.array as cla
from pyopencl.elementwise import ElementwiseKernel as CL_ElK
from pyopencl.reduction import ReductionKernel as CL_RedK

from ..operator import has_attr_not_none, OperatorException, OperatorSum, OperatorPower

from .cpu_operator import *
from .holotomo import HoloTomo, HoloTomoDataStack, HoloTomoData, OperatorHoloTomo


################################################################################################
# Patch HoloTomo class so that we can use 5*w to scale it.
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


patch_method(HoloTomo)


################################################################################################


class CLProcessingUnitHoloTomo(CLProcessingUnit):
    """
    Processing unit in OpenCL space, for operations on HoloTomo objects.

    Handles initializing the context and kernels.
    """

    def __init__(self):
        super(CLProcessingUnitHoloTomo, self).__init__()
        # Queue to copy data to GPU
        self.cl_queue_in = None
        # Queue to copy data from GPU
        self.cl_queue_out = None

    def init_cl(self, cl_ctx=None, cl_device=None, fft_size=(1, 1024, 1024), batch=True, gpu_name=None, test_fft=True,
                verbose=True):
        """
        Derived init_cl function. Also creates in/out queues for parallel processing of large datasets.

        :param cl_ctx: pyopencl.Context. If none, a default context will be created
        :param cl_device: pyopencl.Device. If none, and no context is given, the fastest GPU will be used.
        :param fft_size: the fft size to be used, for benchmark purposes when selecting GPU. different fft sizes
                         can be used afterwards?
        :param batch: if True, will benchmark using a batch 2D FFT
        :param gpu_name: a (sub)string matching the name of the gpu to be used
        :param test_fft: if True, will benchmark the GPU(s)
        :param verbose: report the GPU found and their speed
        :return: nothing
        """
        super(CLProcessingUnitHoloTomo, self).init_cl(cl_ctx=cl_ctx, cl_device=cl_device, fft_size=fft_size, batch=batch,
                                                 gpu_name=gpu_name, test_fft=test_fft, verbose=verbose)
        # Queue to copy data to GPU
        self.cl_queue_in = cl.CommandQueue(self.cl_ctx)
        # Queue to copy data from GPU
        self.cl_queue_out = cl.CommandQueue(self.cl_ctx)
        # List of events to be waited on by the different queues
        self.ev, self.ev_in, self.ev_out = [], [], []

    def cl_init_kernels(self):
        print("HoloTomo OpenCL processing unit: compiling kernels...")
        t0 = timeit.default_timer()
        # Elementwise kernels
        self.cl_scale = CL_ElK(self.cl_ctx, name='cl_scale',
                               operation="d[i] = (float2)(d[i].x * scale, d[i].y * scale )",
                               options=self.cl_options, arguments="__global float2 *d, const float scale")

        self.cl_sum = CL_ElK(self.cl_ctx, name='cl_sum',
                             operation="dest[i] += src[i]",
                             options=self.cl_options, arguments="__global float2 *src, __global float2 *dest")

        self.cl_scale_complex = CL_ElK(self.cl_ctx, name='cl_scale_complex',
                                       operation="d[i] = (float2)(d[i].x * s.x - d[i].y * s.y, d[i].x * s.y + d[i].y * s.x)",
                                       options=self.cl_options, arguments="__global float2 *d, const float2 s")

        self.cl_iobs2psi = CL_ElK(self.cl_ctx, name='cl_iobs2psi',
                                  operation="dest[i] = (float2)(src[i],0.0f)",
                                  options=self.cl_options, arguments="__global float *src, __global float2 *dest")

        self.cl_quad_phase = CL_ElK(self.cl_ctx, name='cl_quad_phase',
                                    operation="QuadPhase(i, d, f, scale, nproj, nb_z, nb_mode, nx, ny)",
                                    preamble=getks('holotomo/opencl/quad_phase_elw.cl'), options=self.cl_options,
                                    arguments="__global float2 *d, __global float *f, const float scale, const int nproj, const int nb_z, const int nb_mode, const int nx, const int ny")

        self.cl_calc2obs = CL_ElK(self.cl_ctx, name='cl_calc2obs',
                                  operation="Calc2Obs(i, iobs, psi, nproj, nb_z, nb_mode, nx, ny)",
                                  preamble=getks('holotomo/opencl/calc2obs_elw.cl'), options=self.cl_options,
                                  arguments="__global float *iobs,__global float2 *psi, const int nproj, const int nb_z, const int nb_mode, const int nx, const int ny")

        self.cl_obj_probe_mult = CL_ElK(self.cl_ctx, name='cl_obj_probe_mult',
                                        operation="ObjectProbeMult(i, obj, probe, psi, dxi, dyi, sample_flag, nproj, nb_z, nb_obj, nb_probe, nx, ny, nx_probe, ny_probe)",
                                        preamble=getks('holotomo/opencl/obj_probe_mult_elw.cl'),
                                        options=self.cl_options,
                                        arguments="__global float2* obj, __global float2 *probe, __global float2* psi, __global int* dxi, __global int* dyi, __global char* sample_flag, const int nproj, const int nb_z, const int nb_obj, const int nb_probe, const int nx, const int ny, const int nx_probe, const int ny_probe")

        self.cl_obj_probez_mult = CL_ElK(self.cl_ctx, name='cl_obj_probez_mult',
                                         operation="ObjectProbeZMult(i, obj, probe, psi, dxi, dyi, sample_flag, nproj, nb_z, nb_obj, nb_probe, nx, ny, nx_probe, ny_probe)",
                                         preamble=getks('holotomo/opencl/obj_probe_mult_elw.cl'),
                                         options=self.cl_options,
                                         arguments="__global float2* obj, __global float2 *probe, __global float2* psi, __global int* dxi, __global int* dyi, __global char* sample_flag, const int nproj, const int nb_z, const int nb_obj, const int nb_probe, const int nx, const int ny, const int nx_probe, const int ny_probe")

        self.cl_paganin_fourier = CL_ElK(self.cl_ctx,
                                         name='cl_paganin_fourier',
                                         operation="paganin_fourier(i, psi, iz, z_delta, mu, dk, nx, ny, nz)",
                                         preamble=getks('holotomo/opencl/paganin_elw.cl'),
                                         options=self.cl_options,
                                         arguments="__global float2 *psi, const int iz, const float z_delta, const float mu, const float dk, const int nx, const int ny, const int nz")

        self.cl_paganin_thickness = CL_ElK(self.cl_ctx,
                                           name='cl_square_modulus',
                                           operation="paganin_thickness(i, obj, psi, iz, mu, k_delta, nx, ny)",
                                           preamble=getks('holotomo/opencl/paganin_elw.cl'),
                                           options=self.cl_options,
                                           arguments="__global float2* obj, __global float2 *psi, const int iz, const float mu, const float k_delta, const int nx, const int ny")

        self.cl_projection_amplitude = CL_ElK(self.cl_ctx, name='cl_projection_amplitude',
                                              operation="ProjectionAmplitude(i, iobs, psi, nb_mode, nxy)",
                                              preamble=getks('holotomo/opencl/projection_amplitude_elw.cl'),
                                              options=self.cl_options,
                                              arguments="__global float *iobs, __global float2 *psi, const int nb_mode, const int nxy")
        # Reduction kernels
        # This will compute Poisson, Gaussian, Euclidian LLK as well as the sum of the calculated intensity
        self.cl_llk = CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)", reduce_expr="a+b",
                              preamble=getks('holotomo/opencl/llk_red.cl'), options=self.cl_options,
                              map_expr="LLKAll(i, iobs, psi, nb_mode, nxy)",
                              arguments="__global float *iobs, __global float2 *psi, const int nb_mode, const int nxy")

        print("HoloTomo OpenCL processing unit: compiling kernels... Finished (dt=%5.2fs)" % (timeit.default_timer() - t0))

    def finish(self):
        super(CLProcessingUnitHoloTomo, self).finish()
        self.cl_queue_in.finish()
        self.cl_queue_out.finish()


"""
The default processing unit 
"""
default_processing_unit = CLProcessingUnitHoloTomo()


class CLOperatorHoloTomo(OperatorHoloTomo):
    """
    Base class for a operators on HoloTomo objects using OpenCL
    """

    def __init__(self, processing_unit=None):
        super(CLOperatorHoloTomo, self).__init__()

        self.Operator = CLOperatorHoloTomo
        self.OperatorSum = CLOperatorHoloTomoSum
        self.OperatorPower = CLOperatorHoloTomoPower

        if processing_unit is None:
            self.processing_unit = default_processing_unit
        else:
            self.processing_unit = processing_unit
        if self.processing_unit.cl_ctx is None:
            # OpenCL kernels have not been prepared yet, use a default initialization
            if main_default_processing_unit.cl_device is None:
                main_default_processing_unit.use_opencl()
            self.processing_unit.init_cl(cl_device=main_default_processing_unit.cl_device,
                                         test_fft=False, verbose=False)

    def apply_ops_mul(self, pci: HoloTomo):
        """
        Apply the series of operators stored in self.ops to a wavefront.
        In this version the operators are applied one after the other to the same wavefront (multiplication)

        :param w: the wavefront to which the operators will be applied.
        :return: the wavefront, after application of all the operators in sequence
        """
        return super(CLOperatorHoloTomo, self).apply_ops_mul(pci)

    def prepare_data(self, pci: HoloTomo):
        stack_size, nz, ny, nx = pci.data.stack_size, pci.data.nz, pci.data.ny, pci.data.nx
        nobj, nprobe = pci.nb_obj, pci.nb_probe

        # TODO: make sure we use pinned/page-locked memory for better performance for all data in the host
        # Make sure data is already in OpenCL space, otherwise transfer it
        if pci._timestamp_counter > pci._cl_timestamp_counter:
            print("Creating buffers in OpenCL space")
            pci._cl_timestamp_counter = pci._timestamp_counter
            if pci._cl_stack is None:
                pci._cl_stack = HoloTomoDataStack()
            if pci._cl_stack_in is None:
                pci._cl_stack_in = HoloTomoDataStack()
            if pci._cl_stack_out is None:
                pci._cl_stack_out = HoloTomoDataStack()
            q = self.processing_unit.cl_queue_in
            pci._cl_stack.probe = cla.to_device(q, pci._probe)
            for s in (pci._cl_stack, pci._cl_stack_in, pci._cl_stack_out):
                s.psi = cla.empty(q, (stack_size, nobj, nprobe, nz, ny, nx), np.complex64)
                s.obj = cla.empty(q, (stack_size, nobj, ny, nx), np.complex64)
                s.iobs = cla.empty(q, (stack_size, nz, ny, nx), np.float32)
                s.dxi = cla.empty(q, (stack_size, nz), np.float32)
                s.dyi = cla.empty(q, (stack_size, nz), np.float32)
                s.sample_flag = cla.empty(q, (stack_size,), np.int8)
                s.scale_factor = cla.empty(q, (stack_size, nz), np.float32)
                s.i = None
            # Copy data for the main (computing) stack
            pci = SwapStack(i=pci._stack_i, next_i=None, out=False, copy_psi=False, verbose=True) * pci

        # if has_attr_not_none(holotomo, '_cl_view') is False:
        #     holotomo._cl_view = {}

    def timestamp_increment(self, pci: HoloTomo):
        pci._cl_timestamp_counter += 1

    # TODO: implement views ? This would only  work on a single stack GPU-side ?
    # def view_register(self, obj: HoloTomo):
    #     """
    #     Creates a new unique view key in an object. When finished with this view, it should be de-registered
    #     using view_purge. Note that it only reserves the key, but does not create the view.
    #     :return: an integer value, which corresponds to yet-unused key in the object's view.
    #     """
    #     i = 1
    #     while i in obj._cl_view:
    #         i += 1
    #     obj._cl_view[i] = None
    #     return i
    #
    # def view_copy(self, holotomo: HoloTomo, i_source, i_dest):
    #     if i_source == 0:
    #         src = {'obj': holotomo._cl_obj, 'probe': holotomo._cl_probe, 'psi': holotomo._cl_psi}
    #     else:
    #         src = holotomo._cl_view[i_source]
    #     if i_dest is 0:
    #         holotomo._cl_obj = cl.array.empty_like(src['obj'])
    #         holotomo._cl_probe = cl.array.empty_like(src['probe'])
    #         holotomo._cl_psi = cl.array.empty_like(src['psi'])
    #         dest = {'obj': holotomo._cl_obj, 'probe': holotomo._cl_probe, 'psi': holotomo._cl_psi}
    #     else:
    #         holotomo._cl_view[i_dest] = {'obj': cl.array.empty_like(src['obj']), 'probe': cl.array.empty_like(src['probe']),
    #                                 'psi': cl.array.empty_like(src['psi'])}
    #         dest = holotomo._cl_view[i_dest]
    #
    #     for s, d in zip([src['obj'], src['probe'], src['psi']], [dest['obj'], dest['probe'], dest['psi']]):
    #         cl.enqueue_copy(self.processing_unit.cl_queue, src=s.data, dest=d.data)
    #
    # def view_swap(self, holotomo: HoloTomo, i1, i2):
    #     if i1 != 0:
    #         if holotomo._cl_view[i1] is None:
    #             # Create dummy value, assume a copy will be made later
    #             holotomo._cl_view[i1] = {'obj': None, 'probe': None, 'psi': None}
    #     if i2 != 0:
    #         if holotomo._cl_view[i2] is None:
    #             # Create dummy value, assume a copy will be made later
    #             holotomo._cl_view[i2] = {'obj': None, 'probe': None, 'psi': None}
    #     if i1 == 0:
    #         holotomo._cl_obj, holotomo._cl_view[i2]['obj'] = holotomo._cl_view[i2]['obj'], holotomo._cl_obj
    #         holotomo._cl_probe, holotomo._cl_view[i2]['probe'] = holotomo._cl_view[i2]['probe'], holotomo._cl_probe
    #         holotomo._cl_psi, holotomo._cl_view[i2]['psi'] = holotomo._cl_view[i2]['psi'], holotomo._cl_psi
    #     elif i2 == 0:
    #         holotomo._cl_obj, holotomo._cl_view[i1]['obj'] = holotomo._cl_view[i1]['obj'], holotomo._cl_obj
    #         holotomo._cl_probe, holotomo._cl_view[i1]['probe'] = holotomo._cl_view[i1]['probe'], holotomo._cl_probe
    #         holotomo._cl_psi, holotomo._cl_view[i1]['psi'] = holotomo._cl_view[i1]['psi'], holotomo._cl_psi
    #     else:
    #         holotomo._cl_view[i1], holotomo._cl_view[i2] = holotomo._cl_view[i2], holotomo._cl_view[i1]
    #     self.timestamp_increment(holotomo)
    #
    # def view_sum(self, holotomo: HoloTomo, i_source, i_dest):
    #     if i_source == 0:
    #         src = {'obj': holotomo._cl_obj, 'probe': holotomo._cl_probe, 'psi': holotomo._cl_psi}
    #     else:
    #         src = holotomo._cl_view[i_source]
    #     if i_dest == 0:
    #         dest = {'obj': holotomo._cl_obj, 'probe': holotomo._cl_probe, 'psi': holotomo._cl_psi}
    #     else:
    #         dest = holotomo._cl_view[i_dest]
    #     for s, d in zip([src['obj'], src['probe'], src['psi']], [dest['obj'], dest['probe'], dest['psi']]):
    #         self.processing_unit.cl_sum(s, d)
    #     self.timestamp_increment(holotomo)
    #
    # def view_purge(self, holotomo: HoloTomo, i):
    #     if i is not None:
    #         del holotomo._cl_view[i]
    #     elif has_attr_not_none(holotomo, '_cl_view'):
    #         del holotomo._cl_view


# The only purpose of this class is to make sure it inherits from CLOperatorHoloTomo and has a processing unit
class CLOperatorHoloTomoSum(OperatorSum, CLOperatorHoloTomo):
    def __init__(self, op1, op2):
        # TODO: should this apply to a single stack or all ?
        if np.isscalar(op1):
            op1 = Scale1(op1)
        if np.isscalar(op2):
            op2 = Scale1(op2)
        if isinstance(op1, CLOperatorHoloTomo) is False or isinstance(op2, CLOperatorHoloTomo) is False:
            raise OperatorException(
                "ERROR: cannot add a CLOperatorHoloTomo with a non-CLOperatorHoloTomo: %s + %s" % (str(op1), str(op2)))
        # We can only have a sum of two CLOperatorHoloTomo, so they must have a processing_unit attribute.
        CLOperatorHoloTomo.__init__(self, op1.processing_unit)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorHoloTomo
        self.OperatorSum = CLOperatorHoloTomoSum
        self.OperatorPower = CLOperatorHoloTomoPower
        self.prepare_data = types.MethodType(CLOperatorHoloTomo.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorHoloTomo.timestamp_increment, self)
        # self.view_copy = types.MethodType(CLOperatorHoloTomo.view_copy, self)
        # self.view_swap = types.MethodType(CLOperatorHoloTomo.view_swap, self)
        # self.view_sum = types.MethodType(CLOperatorHoloTomo.view_sum, self)
        # self.view_purge = types.MethodType(CLOperatorHoloTomo.view_purge, self)


# The only purpose of this class is to make sure it inherits from CLOperatorCDI and has a processing unit
class CLOperatorHoloTomoPower(OperatorPower, CLOperatorHoloTomo):
    def __init__(self, op, n):
        CLOperatorHoloTomo.__init__(self, op.processing_unit)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorHoloTomo
        self.OperatorSum = CLOperatorHoloTomoSum
        self.OperatorPower = CLOperatorHoloTomoPower
        self.prepare_data = types.MethodType(CLOperatorHoloTomo.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorHoloTomo.timestamp_increment, self)
        # self.view_copy = types.MethodType(CLOperatorHoloTomo.view_copy, self)
        # self.view_swap = types.MethodType(CLOperatorHoloTomo.view_swap, self)
        # self.view_sum = types.MethodType(CLOperatorHoloTomo.view_sum, self)
        # self.view_purge = types.MethodType(CLOperatorHoloTomo.view_purge, self)


class FreePU(CLOperatorHoloTomo):
    """
    Operator freeing OpenCL memory. The gpyfft plan in self.processing_unit is removed,
    as well as any OpenCL pyopencl.array.Array attribute in the supplied wavefront.
    """

    def op(self, pci: HoloTomo):
        for s in (pci._cl_stack, pci._cl_stack_in, pci._cl_stack_out):
            for o in dir(s):
                if s.__getattribute__(o) is cla.Array:
                    s.__getattribute__(o).data.release()  # Release GPU buffer
                    s.__setattr__(o, None)
        # holotomo._cl_view = {}
        self.processing_unit.gpyfft_plan = None
        # gc.collect()
        return pci


class Scale1(CLOperatorHoloTomo):
    """
    Multiply the object or probe or psi by a scalar (real or complex).

    Applies only to the current stack.
    """

    def __init__(self, x, obj=True, probe=True, psi=True):
        """

        :param x: the scaling factor
        :param obj: if True, scale the object
        :param probe: if True, scale the probe
        :param psi: if True, scale the psi array
        """
        super(Scale1, self).__init__()
        self.x = x
        self.obj = obj
        self.probe = probe
        self.psi = psi

    def op(self, pci: HoloTomo):
        if self.x == 1:
            return pci

        if np.isreal(self.x):
            scale_k = self.processing_unit.cl_scale
            x = np.float32(self.x)
        else:
            scale_k = self.processing_unit.cl_scale_complex
            x = np.complex64(self.x)
        pu = self.processing_unit
        q = pu.cl_queue
        if self.obj:
            pu.ev = [scale_k(pci._cl_stack.obj, x, queue=q, wait_for=pu.ev)]
        if self.probe:
            pu.ev = [scale_k(pci._cl_stack.probe, x, queue=q, wait_for=pu.ev)]
        if self.psi:
            pu.ev = [scale_k(pci._cl_stack.psi, x, queue=q, wait_for=pu.ev)]
        return pci


class Scale(CLOperatorHoloTomo):
    """
    Multiply the object or probe or psi by a scalar (real or complex).

    Will apply to all projection stacks of the HoloTomo object
    """

    def __new__(cls, x, obj=True, probe=True, psi=True):
        """

        :param x: the scaling factor
        :param obj: if True, scale the object
        :param probe: if True, scale the probe
        :param psi: if True, scale the psi array
        """
        return LoopStack(op=Scale(x, obj=obj, probe=probe, psi=psi), out=True, copy_psi=psi, verbose=False)

    def __init__(self, x, obj=True, probe=True, psi=True):
        """

        :param x: the scaling factor
        :param obj: if True, scale the object
        :param probe: if True, scale the probe
        :param psi: if True, scale the psi array
        """
        super(Scale, self).__init__()


class FT1(CLOperatorHoloTomo):
    """
    Forward Fourier transform.

    Applies only to the current stack.
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(FT1, self).__init__()
        self.scale = scale

    def op(self, pci: HoloTomo):
        pu = self.processing_unit
        pu.cl_fft_set_plan(pci._cl_stack.psi, axes=(-1, -2))
        pu.ev = pu.gpyfft_plan.enqueue(forward=True, wait_for_events=pu.ev)
        if self.scale:
            q = pu.cl_queue
            pu.ev = [pu.cl_scale(pci._cl_stack.psi, np.float32(1 / np.sqrt(pci.data.nx * pci.data.ny)),
                                 queue=q, wait_for=pu.ev)]
        return pci


class IFT1(CLOperatorHoloTomo):
    """
    Inverse Fourier transform.

    Applies only to the current stack.
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(IFT1, self).__init__()
        self.scale = scale

    def op(self, pci: HoloTomo):
        pu = self.processing_unit
        pu.cl_fft_set_plan(pci._cl_stack.psi, axes=(-1, -2))
        pu.ev = self.processing_unit.gpyfft_plan.enqueue(forward=False, wait_for_events=pu.ev)
        if self.scale:
            q = pu.cl_queue
            pu.ev = [pu.cl_scale(pci._cl_stack.psi, np.float32(np.sqrt(pci.data.nx * pci.data.ny)),
                                 queue=q, wait_for=pu.ev)]
        return pci


class QuadraticPhase1(CLOperatorHoloTomo):
    """
    Operator applying a quadratic phase factor for near field propagation. The factor is different for each distance,
    based on the propagation distance stored in the HoloTomo object.

    Applies only to the current stack.
    """

    def __init__(self, forward=True, scale=1):
        """
        Application of a quadratic phase factor, and optionally a scale factor.

        The actual factor is:  :math:`scale * e^{i * factor * ((ix/nx)^2 + (iy/ny)^2)}`
        where ix and iy are the integer indices of the pixels.
        The factor is stored in the HoloTomo object.

        :param forward: if True (the default), applies the scale factor for forward propagation
        :param scale: the data will be scaled by this factor. Useful to normalize before/after a Fourier transform,
                      without accessing twice the array data.
        """
        super(QuadraticPhase1, self).__init__()
        self.scale = np.float32(scale)
        self.forward = forward

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        q = pu.cl_queue
        nb_mode = np.int32(p.nb_obj * p.nb_probe)
        stack_size = p.data.stack_size
        nz = np.int32(p.data.nz)
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)
        quad_f = np.pi * p.data.wavelength * p.data.detector_distance / p.data.pixel_size_detector ** 2
        if self.forward:
            cl_quad_f = cla.to_device(q, -quad_f.astype(np.float32), _async=True)
        else:
            cl_quad_f = cla.to_device(q, quad_f.astype(np.float32), _async=True)

        pu.ev = [pu.cl_quad_phase(p._cl_stack.psi[0, 0, 0, 0], cl_quad_f, self.scale,
                                  stack_size, nz, nb_mode, nx, ny, queue=q, wait_for=pu.ev)]
        return p


class ObjProbe2Psi1(CLOperatorHoloTomo):
    """
    Operator multiplying object views and probe to produce the initial Psi array (before propagation)
    for all projections and distances in the stack.

    Applies only to the current stack.
    """

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        q = pu.cl_queue
        stack_size = np.int32(p.data.stack_size)
        nz = np.int32(p.data.nz)
        nb_obj = np.int32(p.nb_obj)
        nb_probe = np.int32(p.nb_probe)
        nb_probez = np.int32(p._probe.shape[1])
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)
        ny_probe = np.int32(p._probe.shape[-2])
        nx_probe = np.int32(p._probe.shape[-1])

        pu.ev = [pu.cl_obj_probez_mult(p._cl_stack.obj[0, 0], p._cl_stack.probe, p._cl_stack.psi,
                                       p._cl_stack.dxi, p._cl_stack.dyi, p._cl_stack.sample_flag,
                                       stack_size, nz, nb_obj, nb_probe, nx, ny, nx_probe, ny_probe,
                                       queue=q, wait_for=pu.ev)]
        return p


class LLK1(CLOperatorHoloTomo):
    """
    Log-likelihood reduction kernel. Should only be used when Psi is propagated to detector space.
    This is a reduction operator - it will write llk as an argument in the HoloTomo object, and return the object.
    This operator only applies to the current stack of projections.
    If the stack number==0, the llk is re-initialized. Otherwise it is added to the current value.
    """

    def op(self, p):
        nb_mode = np.int32(p._cl_stack.probe.shape[1] * p._cl_stack.obj.shape[1])
        nxy = np.int32(p._probe.shape[-2] * p._probe.shape[-1])
        pu = self.processing_unit
        q = pu.cl_queue
        llk = self.processing_unit.cl_llk(p._cl_stack.iobs, p._cl_stack.psi, nb_mode, nxy,
                                          queue=q, wait_for=pu.ev).get()
        pu.ev = []
        if p._cl_stack.i == 0:
            p.llk_poisson = llk['x']
            p.llk_gaussian = llk['y']
            p.llk_euclidian = llk['z']
            p.nb_photons_calc = llk['w']
        else:
            p.llk_poisson += llk['x']
            p.llk_gaussian += llk['y']
            p.llk_euclidian += llk['z']
            p.nb_photons_calc += llk['w']
        return p


class LLK(CLOperatorHoloTomo):
    """
    Compute the log-likelihood for the entire set of projections.
    """

    def __new__(cls):
        return LoopStack(op=LLK1() * PropagateNearField1() * ObjProbe2Psi1(), out=False, copy_psi=False, verbose=True)


class ApplyAmplitude1(CLOperatorHoloTomo):
    """
    Apply the magnitude from observed intensities, keep the phase. Masked pixels (marked using <0 intensities) are
    left unchanged.

    Applies only to the current stack.
    """

    def __init__(self, calc_llk=False):
        """

        :param calc_llk: if True, the log-likelihood will be calculated for this stack.
        """
        super(ApplyAmplitude1, self).__init__()
        self.calc_llk = calc_llk

    def op(self, p):
        if self.calc_llk:
            # TODO: use a single-pass reduction kernel to apply the amplitude and compute the LLK
            p = LLK1() * p
        nb_mode = np.int32(p._cl_stack.probe.shape[1] * p._cl_stack.obj.shape[1])
        nxy = np.int32(p._probe.shape[-2] * p._probe.shape[-1])
        pu = self.processing_unit
        q = pu.cl_queue
        pu.ev = [self.processing_unit.cl_projection_amplitude(p._cl_stack.iobs, p._cl_stack.psi, nb_mode, nxy,
                                                              queue=q, wait_for=pu.ev)]
        return p


class Psi2ObjProbe1(CLOperatorHoloTomo):
    """
    Operator projecting the psi arrays in sample space onto the object projections and/or probe.

    Applies only to the current stack. The object, probe and normalisation arrays are stored in temporary arrays which
    must be created in a parent operator such as AP
    """

    def __init__(self, update_object=True, update_probe=True):
        """

        :param obj: if True( the default), update the object
        :param probe: if True (the default), update the probe
        """
        super(Psi2ObjProbe1, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe

    def op(self, p: HoloTomo):
        TODO


class AP(CLOperatorHoloTomo):
    """
    Perform alternating projections between detector and object/probe space.

    This operator applies to all projections and loops over the stacks.
    """

    def __init__(self, update_object=True, update_probe=True, nb_cycle=1, calc_llk=False,
                 show_obj_probe=0, fig_num=None):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param nb_cycle: number of cycles to perform. Equivalent to AP(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        """
        super(AP, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.nb_cycle = nb_cycle
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new DM operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return AP(update_object=self.update_object, update_probe=self.update_probe, nb_cycle=self.nb_cycle * n,
                  calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe, fig_num=self.fig_num)

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        q = pu.cl_queue
        stack_size = p.data.stack_size
        nb_obj = p.nb_obj
        nb_probe = p.nb_probe
        nz = p.data.nz
        ny = p.data.ny
        nx = p.data.nx
        # Create temporary arrays for object and probe update
        p._cl_obj_new = cla.empty(q, (stack_size, nz, nb_obj, ny, nx), np.complex64)
        p._cl_obj_norm = cla.empty(q, (stack_size, nz, nb_obj, ny, nx), np.float32)
        p._cl_probe_new = cla.empty(q, (nz, nb_probe, ny, nx), np.complex64)
        p._cl_probe_norm = cla.empty(q, (nz, nb_probe, ny, nx), np.float32)
        t0 = timeit.default_timer()
        ic_dt = 0
        for ic in range(self.nb_cycle):
            # TODO
            pass


class PropagateNearField1(CLOperatorHoloTomo):
    """
    Near field propagator.

    Applies only to the current stack.
    """

    def __init__(self, forward=True):
        """

        :param forward: if True (the default), perform a forward propagation based on the experimental distances.
        """
        super(PropagateNearField1, self).__init__()
        self.forward = forward

    def op(self, p):
        return IFT1(scale=False) * QuadraticPhase1(forward=self.forward) * FT1(scale=False) * p


class Calc2Obs1(CLOperatorHoloTomo):
    """
    Copy the calculated intensities to the observed ones. Can be used for simulation.
    Assumes the current Psi is already in Fourier space.

    Applies only to the current stack.
    """

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        q = pu.cl_queue
        stack_size = np.int32(p.data.stack_size)
        nz = np.int32(p.data.nz)
        nb_mode = np.int32(p.nb_obj * p.nb_probe)
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)
        self.processing_unit.cl_calc2obs(p._cl_stack.iobs[0, 0], p._cl_stack.psi, stack_size, nz, nb_mode,
                                         nx, ny, queue=q, wait_for=pu.ev)
        q.finish()
        p.data.stack_v[p._stack_i].iobs[:] = p._cl_stack.iobs.get(queue=q)
        pu.ev = []
        return p


class Calc2Obs(CLOperatorHoloTomo):
    """
    Copy the calculated intensities to the observed ones. Can be used for simulation.
    Will apply to all projection stacks of the HoloTomo object
    """

    def __new__(cls):
        return LoopStack(op=Calc2Obs1() * PropagateNearField1() * ObjProbe2Psi1(),
                         out=False, copy_psi=False, verbose=True)


class BackPropagatePaganin1(CLOperatorHoloTomo):
    """ Back-propagation algorithm using the single-projection approach.
    Ref: Paganin et al., Journal of microscopy 206 (2002), 33–40. (DOI: 10.1046/j.1365-2818.2002.01010.x)

    This operator uses the observed intensity to calculate a low-resolution estimate of the object, given the
    delta and beta values of its refraction index.

    The result of the transformation is the calculated object as a transmission factor, i.e. if T(r) is the
    estimated thickness of the sample, it is exp(-mu * T - 2*pi/lambda * T)

    The resulting object projection is stored in the first object mode. If the object is defined with multiple modes,
    secondary ones are set to zero.

    Applies only to the current stack.
    """

    def __init__(self, iz=0, delta=1e-6, beta=1e-9, normalize_empty_beam=True):
        """

        :param iz: the index of the detector distance to be taken into account (by default 0) for the propagation.
        :param delta: real part of the refraction index, n = 1 - delta + i * beta
        :param beta: imaginary part of the refraction index
        :param normalize_empty_beam: if True (the default), will try to find if there are observed intensity data
                                     which were recorded without a sample, and if this is the case, they will be
                                     used for normalisation of the data before applying Paganin's reconstruction.
        """
        super(BackPropagatePaganin1, self).__init__()
        self.iz = np.int32(iz)
        self.beta = np.float32(beta)
        self.delta = np.float32(delta)
        self.normalize_empty_beam = normalize_empty_beam

    def op(self, p: HoloTomo):
        pu = self.processing_unit
        q = pu.cl_queue
        # 1-normalize from empty beam (if available)
        if self.normalize_empty_beam:
            # TODO: empty beam normalisation
            pass

        # 2 FT observed intensity
        # NB: masked values are assumed to be already at suitable replacement values (zero, interpolation,...)
        pu.ev = [self.processing_unit.cl_iobs2psi(p._cl_stack.iobs, p._cl_stack.psi, queue=q, wait_for=pu.ev)]
        p = FT1(scale=False) * p

        # 3 Paganin operator in Fourier space
        mu = np.float32(4 * np.pi * self.beta / p.data.wavelength)
        dk = np.float32(2 * np.pi / (p._psi.shape[-1] * p.data.pixel_size_detector))
        nz = np.int32(p.data.iobs.nz)
        stack_size = np.int32(p.data.stack_size)
        ny = np.int32(p.data.ny)
        nx = np.int32(p.data.nx)
        dz = p.data.detector_distance[self.iz]
        pu.ev = [self.processing_unit.cl_paganin_fourier(p._cl_psi, self.iz, np.float32(dz * self.delta),
                                                         mu, dk, nx, ny, nz, queue=q, wait_for=pu.ev)]
        # 4 Back-propagate and compute thickness and object value
        p = IFT1(scale=False) * p
        k_delta = np.float32(2 * np.pi / p.data.wavelength * self.delta)
        if p._cl_obj.shape != (1, stack_size, ny, nx):
            p._cl_obj = cla.empty(self.processing_unit.cl_queue, (1, stack_size, ny, nx), dtype=np.complex64)
        # TODO: store thickness for easier unwrapping in subsequent analysis ?
        pu.ev = [self.processing_unit.cl_paganin_thickness(p._cl_obj, p._cl_psi, self.iz, mu, k_delta, nx, ny,
                                                           queue=q, wait_for=pu.ev)]
        return p


class BackPropagatePaganin(CLOperatorHoloTomo):
    """
    Back-propagation algorithm using the single-projection approach.
    Ref: Paganin et al., Journal of microscopy 206 (2002), 33–40. (DOI: 10.1046/j.1365-2818.2002.01010.x)

    This operator uses the observed intensity to calculate a low-resolution estimate of the object, given the
    delta and beta values of its refraction index.

    The result of the transformation is the calculated object as a transmission factor, i.e. if T(r) is the
    estimated thickness of the sample, it is exp(-mu * T - 2*pi/lambda * T)

    The resulting object projection is stored in the first object mode. If the object is defined with multiple modes,
    secondary ones are set to zero.

    Applies to all projection stacks.
    """

    def __new__(cls, iz=0, delta=1e-6, beta=1e-9, normalize_empty_beam=True):
        """

        :param iz: the index of the detector distance to be taken into account (by default 0) for the propagation.
        :param delta: real part of the refraction index, n = 1 - delta + i * beta
        :param beta: imaginary part of the refraction index
        :param normalize_empty_beam: if True (the default), will try to find if there are observed intensity data
                                     which were recorded without a sample, and if this is the case, they will be
                                     used for normalisation of the data before applying Paganin's reconstruction.
        """
        return LoopStack(op=BackPropagatePaganin1(iz=iz, delta=delta, beta=beta,
                                                  normalize_empty_beam=normalize_empty_beam),
                         out=True, copy_psi=False, verbose=True)

    def __init__(self, iz=0, delta=1e-6, beta=1e-9, normalize_empty_beam=True):
        """

        :param iz: the index of the detector distance to be taken into account (by default 0) for the propagation.
        :param delta: real part of the refraction index, n = 1 - delta + i * beta
        :param beta: imaginary part of the refraction index
        :param normalize_empty_beam: if True (the default), will try to find if there are observed intensity data
                                     which were recorded without a sample, and if this is the case, they will be
                                     used for normalisation of the data before applying Paganin's reconstruction.
        """
        super(BackPropagatePaganin, self).__init__()


class SwapStack(CLOperatorHoloTomo):
    """
    Operator to swap a stack of projections to or from GPU. Note that once this operation has been applied,
    the new Psi value may be undefined (empty array), if no previous array is copied in.
    """

    def __init__(self, i=None, next_i=None, out=True, copy_psi=False, verbose=False):
        """
        Select a new stack of frames, swapping data between the host and the GPU. This is done using a set of
        three buffers and three queues, used to perform in parallel 1) the GPU computing, 2) copying data to the GPU
        and 3) copying data from the GPU. High speed can only be achieved if host memory is page-locked (pinned).
        Note that the buffers used for processing and copying are swapped when using this operator.
        The buffers copied are: object(in/out), iobs (in), dxi (in), dyi(in), sample_flag(in),
        and optionally psi(in/out).

        :param i: the new stack to use. If it is not yet swapped in (in the current 'in' buffers), it is copied
                  to the GPU.
        :param next_i: the following stack to use, for which the copy to the GPU will be initiated in the 'in' queue.
        :param out: if True (the default) and if the HoloTomo object _cl_timestamp_counter > _timestamp_counter, the
                    data from the current stack in memory will be copied back using the 'out' queue.
        :param copy_psi: if True, also copy psi arrays if they are available. If False (the default), the psi arrays
                         are not copied and the corresponding GPU buffers are released.
        :param verbose: if True, print some information when used.
        """
        super(SwapStack, self).__init__()
        self.i = i
        self.next_i = next_i
        self.out = out
        self.copy_psi = copy_psi
        self.verbose = verbose

    def op(self, p: HoloTomo):
        if self.verbose:
            i = p._cl_stack.i
            if i is None:
                i = -1
            ii = p._cl_stack_in.i
            if ii is None:
                ii = -1
            io = p._cl_stack_out.i
            if io is None:
                io = -1
            ns = self.next_i
            if ns is None:
                ns = -1
            print("SwapStack(i=%2d nb=%2d s=%2d si=%2d so=%2d next=%2d)" % (self.i, len(p.data.stack_v), i, ii, io, ns))

        self.i %= len(p.data.stack_v)  # useful when looping and i + 1 == stack_size
        pu = self.processing_unit
        if p._cl_stack.i != self.i:
            # Queue data in
            self._queue_in(p, self.i)

            # Circular permutation of in, main and out stacks and the associated events
            p._cl_stack_in, p._cl_stack, p._cl_stack_out = p._cl_stack_out, p._cl_stack_in, p._cl_stack
            pu.ev_in, pu.ev, pu.ev_out = pu.ev_out, pu.ev_in, pu.ev
            p._stack_i = p._cl_stack.i  # TODO: remove this redundancy ? Do we need p._stack_i ?
            # Keep probe which is constant and only used in the main stack
            p._cl_stack.probe, p._cl_stack_out.probe = p._cl_stack_out.probe, p._cl_stack.probe

            # Queue data out
            if self.out:
                self._queue_out(p)

        if self.next_i is not None:
            self._queue_in(p, self.next_i)

        return p

    def _queue_in(self, p: HoloTomo, i):
        """
        Queue transfer from host to GPU using 'in' queue.

        :param p: the HoloTomo object where the data comes from.
        :param i : the index of the stack to transfer
        :return:
        """
        if p._cl_stack_in.i != i:
            if self.verbose:
                print('Queue in stack  #%d (copy Psi: %d)' % (i, int(self.copy_psi and p._cl_stack.psi is not None)))
            # Data was not already queued for transfer, so do it now
            pu = self.processing_unit
            qi = pu.cl_queue_in
            # Data was not already queued for transfer, so do it now
            pu.ev_in = [cl.enqueue_copy(qi, dest=p._cl_stack_in.obj.data, src=p.data.stack_v[i].obj,
                                        wait_for=pu.ev_in, is_blocking=False),
                        cl.enqueue_copy(qi, dest=p._cl_stack_in.iobs.data, src=p.data.stack_v[i].iobs,
                                        wait_for=pu.ev_in, is_blocking=False),
                        cl.enqueue_copy(qi, dest=p._cl_stack_in.dxi.data, src=p.data.stack_v[i].dxi,
                                        wait_for=pu.ev_in, is_blocking=False),
                        cl.enqueue_copy(qi, dest=p._cl_stack_in.dyi.data, src=p.data.stack_v[i].dyi,
                                        wait_for=pu.ev_in, is_blocking=False),
                        cl.enqueue_copy(qi, dest=p._cl_stack_in.sample_flag.data, src=p.data.stack_v[i].sample_flag,
                                        wait_for=pu.ev_in, is_blocking=False),
                        ]
            if self.copy_psi and p._cl_stack.psi is not None:
                pu.ev_in.append(cl.enqueue_copy(qi, dest=p._cl_stack_in.psi.data, src=p.data.stack_v[i].psi,
                                                wait_for=pu.ev_in, is_blocking=False))
            p._cl_stack_in.i = i

    def _queue_out(self, p: HoloTomo):
        """
        Queue transfer to host from GPU using 'out' queue.

        :param p: the HoloTomo object the data is exchanged with.
        :return:
        """
        if p._cl_stack_out.i != None:
            pu = self.processing_unit
            qo = pu.cl_queue_in
            i = p._cl_stack_out.i
            if self.verbose:
                print('Queue out stack #%d (copy Psi: %d)' % (i, int(self.copy_psi and p._cl_stack.psi is not None)))
            pu.ev_out = [cl.enqueue_copy(qo, dest=p.data.stack_v[i].obj, src=p._cl_stack.obj.data,
                                         wait_for=pu.ev_out, is_blocking=False),
                         cl.enqueue_copy(qo, dest=p.data.stack_v[i].iobs, src=p._cl_stack.iobs.data,
                                         wait_for=pu.ev_out, is_blocking=False),
                         cl.enqueue_copy(qo, dest=p.data.stack_v[i].dxi, src=p._cl_stack.dxi.data,
                                         wait_for=pu.ev_out, is_blocking=False),
                         cl.enqueue_copy(qo, dest=p.data.stack_v[i].dyi, src=p._cl_stack.dyi.data,
                                         wait_for=pu.ev_out, is_blocking=False),
                         cl.enqueue_copy(qo, dest=p.data.stack_v[i].sample_flag, src=p._cl_stack.sample_flag.data,
                                         wait_for=pu.ev_out, is_blocking=False)]
            if self.copy_psi and p._cl_stack.psi is not None:
                pu.ev_out.append(cl.enqueue_copy(qo, dest=p.data.stack_v[i].psi, src=p._cl_stack.psi.data,
                                                 wait_for=pu.ev_out, is_blocking=False))
            else:
                p.data.stack_v[i].psi = None


class LoopStack(CLOperatorHoloTomo):
    """
    Loop operator to apply a given operator sequentially to the complete stack of projections of a HoloTomo object.
    This operator will take care of transferring data between CPU and GPU
    """

    def __init__(self, op, out=True, copy_psi=False, verbose=False):
        """

        :param op: the operator to apply, which can be a multiplication of operators
        :param out: if True (the default) and if the HoloTomo object _cl_timestamp_counter > _timestamp_counter, the
                    data from the current stack in memory will be copied back using the 'out' queue.
        :param copy_psi: if True, when switching between stacks, also keep psi.
        :param verbose: if True, print some information when used.
        """
        super(LoopStack, self).__init__()
        self.stack_op = op
        self.out = out
        self.copy_psi = copy_psi
        self.verbose = verbose

    def op(self, p: HoloTomo):
        if len(p.data.stack_v) == 1:
            return self.stack_op * p
        else:
            for i in range(len(p.data.stack_v)):
                p = self.stack_op * SwapStack(i, next_i=(i + 1) % len(p.data.stack_v), out=self.out,
                                              copy_psi=self.copy_psi, verbose=self.verbose) * p

        return p


class TestParallelFFT(CLOperatorHoloTomo):
    """
    Test speed on a multi-view dataset, by transferring data to/from the GPU in parallel to the FFT execution using
    concurrent queues.
    """

    def op(self, pci: HoloTomo):
        cl_queue = self.processing_unit.cl_queue
        cl_queue_in = self.processing_unit.cl_queue_in
        cl_queue_out = self.processing_unit.cl_queue_out
        stack_size, n_obj, n_probe, nz, ny, nx = pci._psi.shape
        cl_psi = cla.empty(cl_queue, pci._psi.shape, dtype=np.complex64)
        cl_psi_in = cla.empty(cl_queue_in, pci._psi.shape, dtype=np.complex64)
        cl_psi_out = cla.empty(cl_queue_out, pci._psi.shape, dtype=np.complex64)
        ev_in = cl.enqueue_copy(cl_queue_in, dest=cl_psi_in.data, src=pci._psi[0, 0, :, 0, :, :])
        cl_queue.finish()
        cl_queue_in.finish()
        cl_queue_out.finish()

        # Use two arrays for in/out parallel copy (does not work with OpenCL anyway)
        psi0 = pci._psi.copy()
        psi1 = pci._psi.copy()
        #
        n_iter = 5
        n_stack = 5

        # First test fft on array remaining in GPU
        t0 = timeit.default_timer()
        self.processing_unit.cl_fft_set_plan(cl_psi, axes=(-1, -2))
        for i in range(n_iter * n_stack):
            ev_fft = self.processing_unit.gpyfft_plan.enqueue(forward=False)

        cl_queue.finish()
        dt = timeit.default_timer() - t0
        print("Time for on-GPU %d FFT of size %dx%dx%dx%dx%dx%d: %6.3fs" %
              (n_iter * n_stack, stack_size, n_obj, n_probe, nz, ny, nx, dt))

        # test fft on array transferred sequentially to/from GPU
        t0 = timeit.default_timer()
        for i in range(n_iter):
            for j in range(n_stack):
                cl.enqueue_copy(cl_queue, dest=cl_psi_in.data, src=psi0)
                self.processing_unit.cl_fft_set_plan(cl_psi, axes=(-1, -2))
                ev_fft = self.processing_unit.gpyfft_plan.enqueue(forward=False, wait_for_events=[ev_in])
                cl.enqueue_copy(cl_queue, dest=psi1, src=cl_psi.data, wait_for=ev_fft)

        cl_queue.finish()
        dt = timeit.default_timer() - t0
        print("Time for GPU %d FFT of size %dx%dx%dx%dx%dx%d with sequential data transfer: %6.3fs" %
              (n_iter, stack_size, n_obj, n_probe, nz, ny, nx, dt))

        # Now perform FFT while transferring in // data to and from GPU with three queues
        t0 = timeit.default_timer()
        ev_out = None
        for i in range(n_iter):
            for j in range(n_stack):
                ev_in = cl.enqueue_copy(cl_queue_in, dest=cl_psi_in.data, src=psi0)
                cl_psi, cl_psi_in = cl_psi_in, cl_psi
                self.processing_unit.cl_fft_set_plan(cl_psi, axes=(-1, -2))
                if ev_out is not None:
                    ev_fft = self.processing_unit.gpyfft_plan.enqueue(forward=False, wait_for_events=[ev_in, ev_out])
                else:
                    ev_fft = self.processing_unit.gpyfft_plan.enqueue(forward=False, wait_for_events=[ev_in])
                cl_psi, cl_psi_out = cl_psi_out, cl_psi
                ev_out = cl.enqueue_copy(cl_queue_out, dest=psi1, src=cl_psi_out.data, wait_for=ev_fft)
        cl_queue_out.finish()
        dt = timeit.default_timer() - t0
        print("Time for GPU %d FFT of size %dx%dx%dx%dx%dx%d with // data transfer: %6.3fs" %
              (n_iter, stack_size, n_obj, n_probe, nz, ny, nx, dt))

        return pci
