# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


import types
import timeit
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
import pyopencl as cl
import pyopencl.array as cla
from pyopencl.elementwise import ElementwiseKernel as CL_ElK
from pyopencl.reduction import ReductionKernel as CL_RedK

from ...processing_unit import default_processing_unit as main_default_processing_unit
from ...processing_unit.cl_processing_unit import CLProcessingUnit
from ...processing_unit.kernel_source import get_kernel_source as getks
from ..operator import has_attr_not_none, OperatorSum, OperatorPower, OperatorException
from .braggptycho import BraggPtycho, OperatorBraggPtycho
from . import cpu_operator as cpuop


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


patch_method(BraggPtycho)


################################################################################################


class CLProcessingUnitPtycho(CLProcessingUnit):
    """
    Processing unit in OpenCL space, for 2D Ptycho operations.

    Handles initializing the context and kernels.
    """

    def __init__(self):
        super(CLProcessingUnitPtycho, self).__init__()
        # Size of the stack size used in OpenCL - can be any integer
        # TODO: test optimal value for 3D data
        self.cl_stack_size = np.int32(1)

    def cl_init_kernels(self):
        """
        Initialize opencl kernels
        :return: nothing
        """
        # TODO: delay initialization, on-demand for each type of operator ?

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

        # Linear combination with 2 complex arrays and 2 float coefficients
        self.cl_linear_comb_fcfc = CL_ElK(self.cl_ctx, name='cl_linear_comb_fcfc',
                                          operation="dest[i] = (float2)(a * dest[i].x + b * src[i].x, a * dest[i].y + b * src[i].y)",
                                          options=self.cl_options,
                                          arguments="const float a, __global float2 *dest, const float b, __global float2 *src")

        # Linear combination with 2 float arrays and 2 float coefficients
        self.cl_linear_comb_4f = CL_ElK(self.cl_ctx, name='cl_linear_comb_4f',
                                        operation="dest[i] = a * dest[i] + b * src[i]",
                                        options=self.cl_options,
                                        arguments="const float a, __global float *dest, const float b, __global float *src")

        self.cl_projection_amplitude = CL_ElK(self.cl_ctx, name='cl_projection_amplitude',
                                              operation="ProjectionAmplitude(i, iobs, dcalc, background, nbmode, nxy, nxystack, npsi)",
                                              preamble=getks('ptycho/opencl/projection_amplitude_elw.cl'),
                                              options=self.cl_options,
                                              arguments="__global float *iobs, __global float2 *dcalc, __global float *background, const int nbmode, const int nxy, const int nxystack, const int npsi")

        self.cl_calc2obs = CL_ElK(self.cl_ctx, name='cl_calc2obs',
                                  operation="Calc2Obs(i, iobs, dcalc, nbmode, nxyzstack)",
                                  preamble=getks('ptycho/opencl/calc2obs_elw.cl'),
                                  options=self.cl_options,
                                  arguments="__global float *iobs, __global float2 *dcalc, const int nbmode, const int nxyzstack")

        self.cl_object_probe_mult = CL_ElK(self.cl_ctx, name='cl_object_probe_mult',
                                           operation="ObjectProbeMultQuadPhase(i, psi, obj, probe, cx, cy, cz, pixel_size, f, npsi, stack_size, nx, ny, nz, nxo, nyo, nzo, nbobj, nbprobe)",
                                           preamble=getks('ptycho/bragg/opencl/obj_probe_mult_elw.cl'),
                                           options=self.cl_options,
                                           arguments="__global float2* psi, __global float2 *obj, __global float2* probe, __global int* cx, __global int* cy, __global int* cz, const float pixel_size, const float f, const int npsi, const int stack_size, const int nx, const int ny, const int nz, const int nxo, const int nyo, const int nzo, const int nbobj, const int nbprobe")

        self.cl_2object_probe_psi_dm1 = CL_ElK(self.cl_ctx, name='cl_2object_probe_psi_dm1',
                                               operation="ObjectProbePsiDM1(i, psi, obj, probe, cx, cy, cz, pixel_size, f, npsi, stack_size, nx, ny, nz, nxo, nyo, nzo, nbobj, nbprobe)",
                                               preamble=getks('ptycho/bragg/opencl/obj_probe_dm_elw.cl'),
                                               options=self.cl_options,
                                               arguments="__global float2* psi, __global float2 *obj, __global float2* probe, __global int* cx, __global int* cy, __global int* cz, const float pixel_size, const float f, const int npsi, const int stack_size, const int nx, const int ny, const int nz, const int nxo, const int nyo, const int nzo, const int nbobj, const int nbprobe")

        self.cl_2object_probe_psi_dm2 = CL_ElK(self.cl_ctx, name='cl_2object_probe_psi_dm2',
                                               operation="ObjectProbePsiDM2(i, psi, psi_fourier, obj, probe, cx, cy, cz, pixel_size, f, npsi, stack_size, nx, ny, nz, nxo, nyo, nzo, nbobj, nbprobe)",
                                               preamble=getks('ptycho/bragg/opencl/obj_probe_dm_elw.cl'),
                                               options=self.cl_options,
                                               arguments="__global float2* psi, __global float2* psi_fourier, __global float2 *obj, __global float2* probe, __global int* cx, __global int* cy, __global int* cz, const float pixel_size, const float f, const int npsi, const int stack_size, const int nx, const int ny, const int nz, const int nxo, const int nyo, const int nzo, const int nbobj, const int nbprobe")

        self.cl_psi_to_obj = CL_ElK(self.cl_ctx, name='psi_to_objN',
                                    operation="UpdateObjQuadPhase(i, psi, objnew, probe, objnorm, cx, cy, cz, px, f, stack_size, nx, ny, nz, nxo, nyo, nzo, nbobj, nbprobe)",
                                    preamble=getks('ptycho/bragg/opencl/psi_to_obj_probe_elw.cl'),
                                    options=self.cl_options,
                                    arguments="__global float2* psi, __global float2 *objnew, __global float2* probe, __global float* objnorm, const int cx, const int  cy, const int cz, const float px, const float f, const int stack_size, const int nx, const int ny, const int nz, const int nxo, const int nyo, const int nzo, const int nbobj, const int nbprobe")

        self.cl_psi_to_probe = CL_ElK(self.cl_ctx, name='psi_to_probe',
                                      operation="UpdateProbeQuadPhase(i, obj, probe_new, psi, probenorm, cx, cy, cz, px, f, firstpass, npsi, stack_size, nx, ny, nz, nxo, nyo, nzo, nbobj, nbprobe)",
                                      preamble=getks('ptycho/bragg/opencl/psi_to_obj_probe_elw.cl'),
                                      options=self.cl_options,
                                      arguments="__global float2* psi, __global float2 *obj, __global float2* probe_new, __global float* probenorm, __global int* cx, __global int* cy, __global int* cz, const float px, const float f, const char firstpass, const int npsi, const int stack_size, const int nx, const int ny, const int nz, const int nxo, const int nyo, const int nzo, const int nbobj, const int nbprobe")

        self.cl_obj_norm = CL_ElK(self.cl_ctx, name='obj_norm',
                                  operation="ObjNorm(i, obj_unnorm, objnorm, obj, reg, nxyzo, nbobj)",
                                  preamble=getks('ptycho/bragg/opencl/psi_to_obj_probe_elw.cl'),
                                  options=self.cl_options,
                                  arguments="__global float2* obj_unnorm, __global float* objnorm, __global float2* obj, const float reg, const int nxyzo, const int nbobj")

        self.cl_obj_norm_support = CL_ElK(self.cl_ctx, name='obj_norm_support',
                                          operation="ObjNormSupport(i, obj_unnorm, objnorm, obj, support, reg, nxyzo, nbobj)",
                                          preamble=getks('ptycho/bragg/opencl/psi_to_obj_probe_elw.cl'),
                                          options=self.cl_options,
                                          arguments="__global float2* obj_unnorm, __global float* objnorm, __global float2* obj, __global char *support, const float reg, const int nxyzo, const int nbobj")

        self.cl_grad_poisson_fourier = CL_ElK(self.cl_ctx, name='cl_grad_poisson_fourier',
                                              operation="GradPoissonFourier(i, iobs, psi, background, npsi, nbmode, nx, ny, nz, nxyz, nxyzstack)",
                                              preamble=getks('ptycho/bragg/opencl/grad_elw.cl'),
                                              options=self.cl_options,
                                              arguments="__global float *iobs, __global float2 *psi, __global float *background, const int npsi, const int nbmode, const int nx, const int ny, const int nz, const int nxyz, const int nxyzstack")

        self.cl_psi_to_obj_grad = CL_ElK(self.cl_ctx, name='psi_to_obj_grad',
                                         operation="GradObj(i, psi, obj_grad, probe, cx, cy, cz, px, f, stack_size, nx, ny, nz, nxo, nyo, nzo, nbobj, nbprobe)",
                                         preamble=getks('ptycho/bragg/opencl/grad_elw.cl'),
                                         options=self.cl_options,
                                         arguments="__global float2* psi, __global float2 *obj_grad, __global float2* probe, const int cx, const int cy, const int cz, const float px, const float f, const int stack_size, const int nx, const int ny, const int nz, const int nxo, const int nyo, const int nzo, const int nbobj, const int nbprobe")

        self.cl_psi_to_probe_grad = CL_ElK(self.cl_ctx, name='psi_to_probe_grad',
                                           operation="GradProbe(i, psi, probe_grad, obj, cx, cy, cz, px, f, firstpass, npsi, stack_size, nx, ny, nz, nxo, nyo, nzo, nbobj, nbprobe)",
                                           preamble=getks('ptycho/bragg/opencl/grad_elw.cl'),
                                           options=self.cl_options,
                                           arguments="__global float2* psi, __global float2* probe_grad, __global float2 *obj, __global int* cx, __global int* cy, __global int* cz, const float px, const float f, const char firstpass, const int npsi, const int stack_size, const int nx, const int ny, const int nz, const int nxo, const int nyo, const int nzo, const int nbobj, const int nbprobe")

        # Reduction kernels
        self.cl_norm_complex_n = CL_RedK(self.cl_ctx, np.float32, neutral="0", reduce_expr="a+b",
                                         map_expr="pown(length(d[i]), nn)", options=self.cl_options,
                                         arguments="__global float2 *d, const int nn")

        # This will compute Poisson, Gaussian, Euclidian LLK as well as the sum of the calculated intensity
        self.cl_llk = CL_RedK(self.cl_ctx, cla.vec.float4, neutral="(float4)(0,0,0,0)", reduce_expr="a+b",
                              preamble=getks('ptycho/opencl/llk_red.cl'), options=self.cl_options,
                              map_expr="LLKAll(i, iobs, psi, background, nbmode, nxyz, nxyzstack)",
                              arguments="__global float *iobs, __global float2 *psi, __global float *background, const int nbmode, const int nxyz, const int nxyzstack")

        self._cl_cg_polak_ribiere_complex_red = CL_RedK(self.cl_ctx, cl.array.vec.float2, neutral="(float2)(0,0)",
                                                        reduce_expr="a+b",
                                                        map_expr="PolakRibiereComplex(grad[i], lastgrad[i])",
                                                        preamble=getks('opencl/cg_polak_ribiere_red.cl'),
                                                        arguments="__global float2 *grad, __global float2 *lastgrad")
        # 2nd order LLK(gamma) approximation
        self._cl_cg_poisson_gamma_red = CL_RedK(self.cl_ctx, cl.array.vec.float2, neutral="(float2)(0,0)",
                                                reduce_expr="a+b",
                                                map_expr="CG_Poisson_Gamma(i, obs, PO, PdO, dPO, dPdO, scale, nxy, nxystack, nbmode)",
                                                preamble=getks('ptycho/opencl/cg_gamma_red.cl'),
                                                options=self.cl_options,
                                                arguments="__global float *obs, __global float2 *PO, __global float2 *PdO, __global float2 *dPO, __global float2 *dPdO, const float scale, const int nxy, const int nxystack, const int nbmode")
        # 4th order LLK(gamma) approximation
        self._cl_cg_poisson_gamma4_red = CL_RedK(self.cl_ctx, cl.array.vec.float4, neutral="(float4)(0,0,0,0)",
                                                 reduce_expr="a+b",
                                                 map_expr="CG_Poisson_Gamma4(i, obs, PO, PdO, dPO, dPdO, scale, nxy, nxystack, nbmode)",
                                                 preamble=getks('ptycho/opencl/cg_gamma_red.cl'),
                                                 options=self.cl_options,
                                                 arguments="__global float *obs, __global float2 *PO, __global float2 *PdO, __global float2 *dPO, __global float2 *dPdO, const float scale, const int nxy, const int nxystack, const int nbmode")
        # custom kernels


"""
The default processing unit 
"""
default_processing_unit = CLProcessingUnitPtycho()


class CLObsDataStack:
    """
    Class to store a stack (e.g. 16 frames) of observed 3D Bragg data in OpenCL space.
    """

    def __init__(self, cl_obs, cl_x, cl_y, cl_z, i, npsi):
        """

        :param cl_obs: pyopencl array of observed data, with N frames
        :param cl_x, cl_y, cl_z: pyopencl arrays of the positions (in pixels) of the different frames
        :param i: index of the first frame
        :param npsi: number of valid frames (others are filled with zeros)
        """
        self.cl_obs = cl_obs
        self.cl_x = cl_x
        self.cl_y = cl_y
        self.cl_z = cl_z
        self.i = np.int32(i)
        self.npsi = np.int32(npsi)
        self.x = cl_x.get()
        self.y = cl_y.get()
        self.z = cl_z.get()


class CLOperatorBraggPtycho(OperatorBraggPtycho):
    """
    Base class for a operators on CDI objects using OpenCL
    """

    def __init__(self, processing_unit=None):
        super(CLOperatorBraggPtycho, self).__init__()

        self.Operator = CLOperatorBraggPtycho
        self.OperatorSum = CLOperatorBraggPtychoSum
        self.OperatorPower = CLOperatorBraggPtychoPower

        if processing_unit is None:
            self.processing_unit = default_processing_unit
        else:
            self.processing_unit = processing_unit
        if self.processing_unit.cl_ctx is None:
            # OpenCL kernels have not been prepared yet, use a default initialization
            if main_default_processing_unit.cl_device is None:
                main_default_processing_unit.select_gpu(language='opencl')
            self.processing_unit.init_cl(cl_device=main_default_processing_unit.cl_device, test_fft=False)

    def apply_ops_mul(self, pty):
        """
        Apply the series of operators stored in self.ops to a Ptycho2D object.
        The operators are applied one after the other.

        :param pty: the Ptycho2D to which the operators will be applied.
        :return: the Ptycho2D object, after application of all the operators in sequence
        """
        if isinstance(pty, BraggPtycho) is False:
            raise OperatorException(
                "ERROR: tried to apply operator:\n    %s \n  to:\n    %s\n  which is not a Ptycho object" % (
                    str(self), str(pty)))
        return super(CLOperatorBraggPtycho, self).apply_ops_mul(pty)

    def prepare_data(self, p):
        """
        Make sure the data to be used is in the correct memory (host or GPU) for the operator.
        Virtual, must be derived.

        :param p: the Ptycho object the operator will be applied to.
        :return:
        """
        if has_attr_not_none(p, "_cl_obs_v") is False:
            # Assume observed intensity is immutable, so transfer only once
            self.init_cl_vobs(p)
        elif len(p._cl_obs_v[0].cl_obs) != self.processing_unit.cl_stack_size:
            # This should not happen, but if tests are being made on the speed vs the stack size, this can be useful.
            self.init_cl_vobs(p)

        if p._timestamp_counter > p._cl_timestamp_counter:
            # print("Moving object, probe, support to OpenCL GPU")
            p._cl_obj = cla.to_device(self.processing_unit.cl_queue, p._obj)
            p._cl_probe = cla.to_device(self.processing_unit.cl_queue, p._probe)
            if p.support is None:
                p._cl_support = cla.empty(self.processing_unit.cl_queue, p._obj.shape[-3:], dtype=np.int8)
                p._cl_support.fill(np.int8(1))
            else:
                p._cl_support = cla.to_device(self.processing_unit.cl_queue, p.support.astype(np.int8))
            p._cl_timestamp_counter = p._timestamp_counter
            if p._background is None:
                p._cl_background = cla.zeros(self.processing_unit.cl_queue, p.data.iobs.shape[-3:], dtype=np.float32)
            else:
                p._cl_background = cla.to_device(self.processing_unit.cl_queue, p._background)
        need_init_psi = False

        if has_attr_not_none(p, "_cl_psi") is False:
            need_init_psi = True
        elif p._cl_psi.shape[0:3] != (len(p._obj), len(p._probe), self.processing_unit.cl_stack_size):
            need_init_psi = True
        if need_init_psi:
            nz, ny, nx = p._probe.shape[-3:]
            p._cl_psi = cla.empty(self.processing_unit.cl_queue, dtype=np.complex64,
                                  shape=(len(p._obj), len(p._probe), self.processing_unit.cl_stack_size, nz, ny, nx))

        if has_attr_not_none(p, "_cl_psi_v") is False:
            # _cl_psi_v is used to hold the complete copy of Psi projections for all stacks, for algorithms
            # such as DM which need them.
            p._cl_psi_v = {}

        if has_attr_not_none(p, '_cl_view') is False:
            p._cl_view = {}

    def init_cl_vobs(self, p):
        """
        Initialize observed intensity and scan positions in OpenCL space

        :param p: the Ptycho object the operator will be applied to.
        :return:
        """
        # print("Moving observed data to OpenCL GPU")
        p._cl_obs_v = []
        nb_frame, nz, ny, nx = p.data.iobs.shape
        nzo, nyo, nxo = p._obj.shape[1:]
        cl_stack_size = self.processing_unit.cl_stack_size
        for i in range(0, nb_frame, cl_stack_size):
            vcx = np.zeros((cl_stack_size), dtype=np.int32)
            vcy = np.zeros((cl_stack_size), dtype=np.int32)
            vcz = np.zeros((cl_stack_size), dtype=np.int32)
            vobs = np.zeros((cl_stack_size, nz, ny, nx), dtype=np.float32)
            if nb_frame < (i + cl_stack_size):
                # We probably want to avoid this for 3D data
                print("Number of frames is not a multiple of %d, adding %d null frames" %
                      (cl_stack_size, i + cl_stack_size - nb_frame))
            for j in range(cl_stack_size):
                ij = i + j
                if ij < nb_frame:
                    dz, dy, dx = p.data.posz[ij], p.data.posy[ij], p.data.posx[ij]
                    cx, cy, cz = p.xyz_to_obj(dx, dy, dz)
                    # Corner coordinates of the object
                    vcx[j] = np.int32(round(cx + (nxo - nx) / 2))
                    vcy[j] = np.int32(round(cy + (nyo - ny) / 2))
                    vcz[j] = np.int32(round(cz + (nzo - nz) / 2))
                    vobs[j] = p.data.iobs[ij]
                else:
                    vcx[j] = vcx[0]
                    vcy[j] = vcy[0]
                    vobs[j] = np.zeros_like(vobs[0], dtype=np.float32)
            cl_vcx = cl.array.to_device(self.processing_unit.cl_queue, vcx)
            cl_vcy = cl.array.to_device(self.processing_unit.cl_queue, vcy)
            cl_vcz = cl.array.to_device(self.processing_unit.cl_queue, vcz)
            cl_vobs = cl.array.to_device(self.processing_unit.cl_queue, vobs)
            p._cl_obs_v.append(CLObsDataStack(cl_vobs, cl_vcx, cl_vcy, cl_vcz, i,
                                              np.int32(min(cl_stack_size, nb_frame - i))))
        # Initialize the size and index of current stack
        p._cl_stack_i = 0
        p._cl_stack_nb = len(p._cl_obs_v)

    def timestamp_increment(self, p):
        p._timestamp_counter += 1
        p._cl_timestamp_counter = p._timestamp_counter

    def view_register(self, obj):
        """
        Creates a new unique view key in an object. When finished with this view, it should be de-registered
        using view_purge. Note that it only reserves the key, but does not create the view.
        :return: an integer value, which corresponds to yet-unused key in the object's view.
        """
        i = 1
        while i in obj._cl_view:
            i += 1
        obj._cl_view[i] = None
        return i

    def view_copy(self, pty, i_source, i_dest):
        if i_source == 0:
            src = {'obj': pty._cl_obj, 'probe': pty._cl_probe, 'psi': pty._cl_psi, 'psi_v': pty._cl_psi_v}
        else:
            src = pty._cl_view[i_source]
        if i_dest == 0:
            pty._cl_obj = cl.array.empty_like(src['obj'])
            pty._cl_probe = cl.array.empty_like(src['probe'])
            pty._cl_psi = cl.array.empty_like(src['psi'])
            pty._cl_psi_v = {}
            dest = {'obj': pty._cl_obj, 'probe': pty._cl_probe, 'psi': pty._cl_psi, 'psi_v': pty._cl_psi_v}
        else:
            pty._cl_view[i_dest] = {'obj': cl.array.empty_like(src['obj']), 'probe': cl.array.empty_like(src['probe']),
                                    'psi': cl.array.empty_like(src['psi']), 'psi_v': {}}
            dest = pty._cl_view[i_dest]

        for i in range(len(src['psi_v'])):
            dest['psi_v'][i] = cl.array.empty_like(src['psi'])

        for s, d in zip([src['obj'], src['probe'], src['psi']] + [v for k, v in src['psi_v'].items()],
                        [dest['obj'], dest['probe'], dest['psi']] + [v for k, v in dest['psi_v'].items()]):
            cl.enqueue_copy(self.processing_unit.cl_queue, src=s.data, dest=d.data)

    def view_swap(self, pty, i1, i2):
        if i1 != 0:
            if pty._cl_view[i1] is None:
                # Create dummy value, assume a copy will be made later
                pty._cl_view[i1] = {'obj': None, 'probe': None, 'psi': None, 'psi_v': None}
        if i2 != 0:
            if pty._cl_view[i2] is None:
                # Create dummy value, assume a copy will be made later
                pty._cl_view[i2] = {'obj': None, 'probe': None, 'psi': None, 'psi_v': None}
        if i1 == 0:
            pty._cl_obj, pty._cl_view[i2]['obj'] = pty._cl_view[i2]['obj'], pty._cl_obj
            pty._cl_probe, pty._cl_view[i2]['probe'] = pty._cl_view[i2]['probe'], pty._cl_probe
            pty._cl_psi, pty._cl_view[i2]['psi'] = pty._cl_view[i2]['psi'], pty._cl_psi
            pty._cl_psi_v, pty._cl_view[i2]['psi_v'] = pty._cl_view[i2]['psi_v'], pty._cl_psi_v
        elif i2 == 0:
            pty._cl_obj, pty._cl_view[i1]['obj'] = pty._cl_view[i1]['obj'], pty._cl_obj
            pty._cl_probe, pty._cl_view[i1]['probe'] = pty._cl_view[i1]['probe'], pty._cl_probe
            pty._cl_psi, pty._cl_view[i1]['psi'] = pty._cl_view[i1]['psi'], pty._cl_psi
            pty._cl_psi_v, pty._cl_view[i1]['psi_v'] = pty._cl_view[i1]['psi_v'], pty._cl_psi_v
        else:
            pty._cl_view[i1], pty._cl_view[i2] = pty._cl_view[i2], pty._cl_view[i1]
        self.timestamp_increment(pty)

    def view_sum(self, pty, i_source, i_dest):
        if i_source == 0:
            src = {'obj': pty._cl_obj, 'probe': pty._cl_probe, 'psi': pty._cl_psi, 'psi_v': pty._cl_psi_v}
        else:
            src = pty._cl_view[i_source]
        if i_dest == 0:
            dest = {'obj': pty._cl_obj, 'probe': pty._cl_probe, 'psi': pty._cl_psi, 'psi_v': pty._cl_psi_v}
        else:
            dest = pty._cl_view[i_dest]
        for s, d in zip([src['obj'], src['probe'], src['psi']] + [v for k, v in src['psi_v'].items()],
                        [dest['obj'], dest['probe'], dest['psi']] + [v for k, v in dest['psi_v'].items()]):
            self.processing_unit.cl_sum(s, d)
        self.timestamp_increment(pty)

    def view_purge(self, pty, i):
        if i is not None:
            del pty._cl_view[i]
        elif has_attr_not_none(pty, '_cl_view'):
            del pty._cl_view


# The only purpose of this class is to make sure it inherits from CLOperatorCDI and has a processing unit
class CLOperatorBraggPtychoSum(OperatorSum, CLOperatorBraggPtycho):
    def __init__(self, op1, op2):
        if np.isscalar(op1):
            op1 = Scale(op1)
        if np.isscalar(op2):
            op2 = Scale(op2)
        if isinstance(op1, CLOperatorBraggPtycho) is False or isinstance(op2, CLOperatorBraggPtycho) is False:
            raise OperatorException(
                "ERROR: cannot add a CLOperatorCDI with a non-CLOperatorCDI: %s + %s" % (str(op1), str(op2)))
        # We can only have a sum of two CLOperatorCDI, so they must have a processing_unit attribute.
        CLOperatorBraggPtycho.__init__(self, op1.processing_unit)
        OperatorSum.__init__(self, op1, op2)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorBraggPtycho
        self.OperatorSum = CLOperatorBraggPtychoSum
        self.OperatorPower = CLOperatorBraggPtychoPower
        self.prepare_data = types.MethodType(CLOperatorBraggPtycho.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorBraggPtycho.timestamp_increment, self)
        self.view_copy = types.MethodType(CLOperatorBraggPtycho.view_copy, self)
        self.view_swap = types.MethodType(CLOperatorBraggPtycho.view_swap, self)
        self.view_sum = types.MethodType(CLOperatorBraggPtycho.view_sum, self)
        self.view_purge = types.MethodType(CLOperatorBraggPtycho.view_purge, self)


# The only purpose of this class is to make sure it inherits from CLOperatorCDI and has a processing unit
class CLOperatorBraggPtychoPower(OperatorPower, CLOperatorBraggPtycho):
    def __init__(self, op, n):
        CLOperatorBraggPtycho.__init__(self, op.processing_unit)
        OperatorPower.__init__(self, op, n)

        # We need to cherry-pick some functions & attributes doubly inherited
        self.Operator = CLOperatorBraggPtycho
        self.OperatorSum = CLOperatorBraggPtychoSum
        self.OperatorPower = CLOperatorBraggPtychoPower
        self.prepare_data = types.MethodType(CLOperatorBraggPtycho.prepare_data, self)
        self.timestamp_increment = types.MethodType(CLOperatorBraggPtycho.timestamp_increment, self)
        self.view_copy = types.MethodType(CLOperatorBraggPtycho.view_copy, self)
        self.view_swap = types.MethodType(CLOperatorBraggPtycho.view_swap, self)
        self.view_sum = types.MethodType(CLOperatorBraggPtycho.view_sum, self)
        self.view_purge = types.MethodType(CLOperatorBraggPtycho.view_purge, self)


class FreePU(CLOperatorBraggPtycho):
    """
    Operator freeing OpenCL memory. The gpyfft data reference in self.processing_unit is removed,
    as well as any OpenCL pyopencl.array.Array attribute in the supplied wavefront.
    """

    def op(self, pty):
        for o in dir(pty):
            if pty.__getattribute__(o) is cla.Array:
                pty.__delattr__(o, None)
        if has_attr_not_none(pty, "_cl_psi_v"):
            pty._cl_psi_v = {}
        self.processing_unit.free_fft_plans()
        pty._cl_obs_v = None
        return pty


class Scale(CLOperatorBraggPtycho):
    """
    Multiply the ptycho object by a scalar (real or complex).
    """

    def __init__(self, x, obj=True, probe=True, psi=True):
        """

        :param x: the scaling factor
        :param obj: if True, scale the object
        :param probe: if True, scale the probe
        :param psi: if True, scale the all the psi arrays, _cl_psi as well as _cl_psi_v
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
            scale_k = self.processing_unit.cl_scale
            x = np.float32(self.x)
        else:
            scale_k = self.processing_unit.cl_scale_complex
            x = np.complex64(self.x)

        if self.obj:
            scale_k(p._cl_obj, x)
        if self.probe:
            scale_k(p._cl_probe, x)
        if self.psi:
            scale_k(p._cl_psi, x)
            for i in range(len(p._cl_psi_v)):
                scale_k(p._cl_psi_v[i], x)
        return p


class ObjProbe2Psi(CLOperatorBraggPtycho):
    """
    Computes Psi = Obj(r) * Probe(r-r_j) for a stack of N probe positions.
    """

    def op(self, p):
        # Multiply obj and probe with quadratic phase factor, taking into account all modes (if any)
        i = p._cl_stack_i
        nz = np.int32(p._probe.shape[-3])
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        # TODO: take into account different pixel size along x,y ?
        pxyz = p.voxel_size_object()
        pixel_size_object = np.sqrt(pxyz[1] * pxyz[2])
        # print(i, f, p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size, nx, ny, nxo, nyo, nb_probe, nb_obj)
        # First argument is p._cl_psi[0] because the kernel will calculate the projection for all object and probe modes
        # and the full stack of frames.
        self.processing_unit.cl_object_probe_mult(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe,
                                                  p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y, p._cl_obs_v[i].cl_z,
                                                  pixel_size_object, f,
                                                  p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size,
                                                  nx, ny, nz, nxo, nyo, nzo, nb_obj, nb_probe)
        return p


class FT(CLOperatorBraggPtycho):
    """
    Forward Fourier-transform a Psi array, i.e. a stack of N Obj*Probe views
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(FT, self).__init__()
        self.scale = scale

    def op(self, pty):
        plan = self.processing_unit.cl_fft_get_plan(pty._cl_psi, axes=None)
        for e in plan.enqueue(forward=True):
            e.wait()  # Needed as CLFFT may use its own queues
        if self.scale:
            self.processing_unit.cl_scale(pty._cl_psi, np.float32((pty._cl_psi[0, 0, 0].size) ** (-1 / 3.)))
        return pty


class IFT(CLOperatorBraggPtycho):
    """
    Backward Fourier-transform a Psi array, i.e. a stack of N Obj*Probe views
    """

    def __init__(self, scale=True):
        """

        :param scale: if True, the FFT will be normalized.
        """
        super(IFT, self).__init__()
        self.scale = scale

    def op(self, pty):
        plan = self.processing_unit.cl_fft_get_plan(pty._cl_psi, axes=None)
        for e in plan.enqueue(forward=False):
            e.wait()  # Needed as CLFFT may use its own queues
        if self.scale:
            self.processing_unit.cl_scale(pty._cl_psi, np.float32((pty._cl_psi[0, 0, 0].size) ** (1 / 3.)))
        return pty


class ShowPsi(CLOperatorBraggPtycho):
    """
    Class to display object during an optimization.
    """

    def __init__(self, i=0, fig_num=-1, title=None, rotation=None):
        """
        :param i: the index of the Psi array to display (if the stack has several)
        :param fig_num: the matplotlib figure number. if None, a new figure will be created. if -1 (the default), the
                        current figure will be used.
        :param title: the title for the view. If None, a default title will be used.
        :param rotation=('z',np.deg2rad(-20)): optionally, the object can be displayed after a rotation of the
                                               object. This is useful if the object or support is to be defined as a
                                               parallelepiped, before being rotated to be in diffraction condition.
                                               The rotation can be given as a tuple of a rotation axis name (x, y or z)
                                               and a counter-clockwise rotation angle in radians.
        """
        super(ShowPsi, self).__init__()
        self.i = i
        self.fig_num = fig_num
        self.title = title
        self.rotation = rotation

    def op(self, p):
        x, y, z = p.get_xyz(domain='probe')
        # We only show the first object mode
        o = fftshift(p._cl_psi.get()[0, 0, self.i])
        cpuop.show_3d(x, y, z, o, fig_num=self.fig_num, title=self.title, rotation=self.rotation)
        return p

    def timestamp_increment(self, p):
        # This display operation does not modify the data.
        pass


class Calc2Obs(CLOperatorBraggPtycho):
    """
    Copy the calculated intensities to the observed ones. Can be used for simulation. Applies to a stack of N views,
    assumes the current Psi is already in Fourier space.
    The new observed values are also copied to the main memory array.
    """

    def __init__(self):
        """

        """
        super(Calc2Obs, self).__init__()

    def op(self, p):
        nxyz = np.int32(p._probe.shape[-3] * p._probe.shape[-2] * p._probe.shape[-1])
        nxyzstack = np.int32(nxyz * self.processing_unit.cl_stack_size)
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        i = p._cl_stack_i
        self.processing_unit.cl_calc2obs(p._cl_obs_v[i].cl_obs, p._cl_psi, nb_mode, nxyzstack)
        obs = p._cl_obs_v[i].cl_obs.get()
        for j in range(self.processing_unit.cl_stack_size):
            ij = i * self.processing_unit.cl_stack_size + j
            if j < p._cl_obs_v[i].npsi:
                p.data.iobs[ij] = obs[j]
        return p


class ApplyAmplitude(CLOperatorBraggPtycho):
    """
    Apply the magnitude from observed intensities, keep the phase. Applies to a stack of N views.
    """

    def __init__(self, calc_llk=False):
        """

        :param calc_llk: if True, the log-likelihood will be calculated for this stack.
        """
        super(ApplyAmplitude, self).__init__()
        self.calc_llk = calc_llk

    def op(self, p):
        # TODO: use a single-pass reduction kernel to apply the amplitude and compute the LLK
        if self.calc_llk:
            p = LLK() * p
        nxyz = np.int32(p._probe.shape[-3] * p._probe.shape[-2] * p._probe.shape[-1])
        nxyzstack = np.int32(nxyz * self.processing_unit.cl_stack_size)
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        i = p._cl_stack_i
        nb_psi = np.int32(p._cl_obs_v[i].npsi)
        self.processing_unit.cl_projection_amplitude(p._cl_obs_v[i].cl_obs[0], p._cl_psi, p._cl_background,
                                                     nb_mode, nxyz, nxyzstack, nb_psi)
        return p


class FourierApplyAmplitude(CLOperatorBraggPtycho):
    """
    Fourier magnitude operator, performing a Fourier transform, the magnitude projection, and a backward FT on a stack
    of N views.
    """

    def __new__(cls, calc_llk=False):
        return IFT() * ApplyAmplitude(calc_llk=calc_llk) * FT()

    def __init__(self, calc_llk=False):
        super(FourierApplyAmplitude, self).__init__()


class LLK(CLOperatorBraggPtycho):
    """
    Log-likelihood reduction kernel. Can only be used when Psi is in diffraction space.
    This is a reduction operator - it will write llk as an argument in the Ptycho object, and return the object.
    If _cl_stack_i==0, the llk is re-initialized. Otherwise it is added to the current value.

    The LLK can be calculated directly from object and probe using: p = LoopStack(LLK() * FT() * ObjProbe2Psi()) * p
    """

    def op(self, p):
        i = p._cl_stack_i
        nb_mode = np.int32(p._probe.shape[0] * p._obj.shape[0])
        nb_psi = p._cl_obs_v[i].npsi
        nxyz = np.int32(p._probe.shape[-3] * p._probe.shape[-2] * p._probe.shape[-1])
        nxyzstack = np.int32(self.processing_unit.cl_stack_size * nxyz)
        llk = self.processing_unit.cl_llk(p._cl_obs_v[i].cl_obs[:nb_psi], p._cl_psi, p._cl_background,
                                          nb_mode, nxyz, nxyzstack).get()
        if p._cl_stack_i == 0:
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


class Psi2Obj(CLOperatorBraggPtycho):
    """
    Computes updated Obj(r) contributions from Psi and Probe(r-r_j), for a stack of N probe positions.
    """

    def op(self, p):
        i = p._cl_stack_i
        # print("Psi2Obj(), i=%d"%(i))
        nz = np.int32(p._probe.shape[-3])
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        # TODO: take into account different pixel size along x,y ?
        pxyz = p.voxel_size_object()
        pixel_size_object = np.sqrt(pxyz[1] * pxyz[2])

        if i == 0:
            p._cl_obj_new = cla.zeros(self.processing_unit.cl_queue, (nb_obj, nzo, nyo, nxo), dtype=np.complex64)
            p._cl_obj_norm = cla.zeros(self.processing_unit.cl_queue, (nzo, nyo, nxo), dtype=np.float32)

        # To avoid memory write conflicts, we must loop on the different frames which have different shifts with respect
        # to the object. This is faster that creating N=cl_stack_size objects and summing them afterwards.
        for ii in range(p._cl_obs_v[i].npsi):
            # This kernel will update the object for one probe position and all the object modes.
            self.processing_unit.cl_psi_to_obj(p._cl_psi[0, 0, ii], p._cl_obj_new, p._cl_probe, p._cl_obj_norm,
                                               p._cl_obs_v[i].x[ii], p._cl_obs_v[i].y[ii], p._cl_obs_v[i].z[ii],
                                               pixel_size_object, f, self.processing_unit.cl_stack_size,
                                               nx, ny, nz, nxo, nyo, nzo, nb_obj, nb_probe)
        return p


class Psi2ObjMerge(CLOperatorBraggPtycho):
    """
    Call this when all stack of probe positions have been processed, and the final update of the object can
    be calculated. Temporary arrays are cleaned up
    """

    def __init__(self, reg=1e-2):
        """

        """
        super(Psi2ObjMerge, self).__init__()
        self.reg = reg

    def op(self, p):
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        nxyzo = np.int32(nzo * nyo * nxo)

        reg = np.float32(float(cl.array.max(p._cl_obj_norm).get()) * self.reg)

        self.processing_unit.cl_obj_norm_support(p._cl_obj_new[0], p._cl_obj_norm, p._cl_obj, p._cl_support, reg,
                                                 nxyzo, nb_obj)

        # Clean up
        del p._cl_obj_norm, p._cl_obj_new

        return p


# TODO: implement probe update code, which would perform a 2D regularization of the probe
class Psi2Probe(CLOperatorBraggPtycho):
    """
    Computes updated Probe contributions from Psi and Obj, for a stack of N probe positions.
    """

    def op(self, p):
        i = p._cl_stack_i
        first_pass = np.int8(i == 0)
        nz = np.int32(p._probe.shape[-3])
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        # TODO: take into account different pixel size along x,y ?
        pxyz = p.voxel_size_object()
        pixel_size_object = np.sqrt(pxyz[1] * pxyz[2])

        if i == 0:
            p._cl_probe_new = cla.empty(self.processing_unit.cl_queue, (nb_probe, nz, ny, nx), dtype=np.complex64)
            p._cl_probe_norm = cla.empty(self.processing_unit.cl_queue, (nz, ny, nx), dtype=np.float32)

        # First argument is p._cl_psi[0] because the kernel will calculate the projection for all object and probe modes
        # and the full stack of frames.
        self.processing_unit.cl_psi_to_probe(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe_new, p._cl_probe_norm,
                                             p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y, p._cl_obs_v[i].cl_z,
                                             pixel_size_object, f, first_pass,
                                             p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size,
                                             nx, ny, nz, nxo, nyo, nzo, nb_obj, nb_probe)

        return p


class Psi2ProbeMerge(CLOperatorBraggPtycho):
    """
    Call this when all stack of probe positions have been processed, and the final update of the probe can
    be calculated. Temporary arrays are cleaned up.
    """

    def __init__(self, reg=1e-2):
        """

        """
        super(Psi2ProbeMerge, self).__init__()
        self.reg = reg

    def op(self, p):
        nb_probe = np.int32(p._probe.shape[0])
        nxyz = np.int32(p._probe.shape[-3] * p._probe.shape[-2] * p._probe.shape[-1])

        reg = np.float32(float(cl.array.max(p._cl_probe_norm).get()) * self.reg)

        self.processing_unit.cl_obj_norm(p._cl_probe_new[0], p._cl_probe_norm, p._cl_probe, reg, nxyz, nb_probe)

        # Clean up
        del p._cl_probe_norm, p._cl_probe_new

        return p


class AP(CLOperatorBraggPtycho):
    """
    Perform a complete Alternating Projection cycle:
    - forward all object*probe views to Fourier space and apply the observed amplitude
    - back-project to object space and project onto (probe, object)
    - update background optionally
    """

    def __init__(self, update_object=True, update_probe=False, update_background=False, nb_cycle=1, calc_llk=False,
                 show_obj_probe=False, fig_num=-1):
        """

        :param update_object: update object ?
        :param update_probe: update probe (TODO)
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
        self.update_background = update_background
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

    def op(self, p):
        for ic in range(self.nb_cycle):
            t0 = timeit.default_timer()
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True
            if self.update_background:
                pass  # TODO: update while in Fourier space
            ops = FourierApplyAmplitude(calc_llk=calc_llk) * ObjProbe2Psi()

            if self.update_object:
                ops = Psi2Obj() * ops
            if self.update_probe:
                ops = Psi2Probe() * ops

            p = LoopStack(ops) * p

            if self.update_object:
                p = Psi2ObjMerge() * p
            if self.update_probe:
                p = Psi2ProbeMerge() * p

            if calc_llk:
                dt = timeit.default_timer() - t0
                print("AP #%3d LLK= %8.2f(p) %8.2f(g) %8.2f(e), nb photons=%e, dt/cycle=%5.3fs" \
                      % (ic, p.llk_poisson / p.nb_obs, p.llk_gaussian / p.nb_obs, p.llk_euclidian / p.nb_obs,
                         p.nb_photons_calc, dt))

            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    tit = "AP"
                    if self.update_object:
                        tit += "/"
                        if len(p._obj) > 1:
                            tit += "%d" % (len(p._obj))
                        tit += "o"
                    if self.update_probe:
                        tit += "/"
                        if len(p._probe) > 1:
                            tit += "%d" % (len(p._probe))
                        tit += "p"
                    tit += " #%3d, LLKn(p)=%8.3f" % (ic, p.llk_poisson / p.nb_obs)
                    p = cpuop.ShowObj(fig_num=self.fig_num, title=tit) * p

        return p


class DM1(CLOperatorBraggPtycho):
    """
    Equivalent to operator: 2 * ObjProbe2Psi() - 1
    """

    def op(self, p):
        i = p._cl_stack_i
        nz = np.int32(p._probe.shape[-3])
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        # TODO: take into account different pixel size along x,y ?
        pxyz = p.voxel_size_object()
        pixel_size_object = np.sqrt(pxyz[1] * pxyz[2])
        self.processing_unit.cl_2object_probe_psi_dm1(p._cl_psi[0, 0, 0], p._cl_obj, p._cl_probe,
                                                      p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y, p._cl_obs_v[i].cl_z,
                                                      pixel_size_object, f,
                                                      p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size,
                                                      nx, ny, nz, nxo, nyo, nzo, nb_obj, nb_probe)
        return p


class DM2(CLOperatorBraggPtycho):
    """
    # Psi(n+1) = Psi(n) - P*O + Psi_fourier

    This operator assumes that Psi_fourier is the current Psi, and that Psi(n) is in p._cl_psi_v

    On output Psi(n+1) is the current Psi, and Psi_fourier has been swapped to p._cl_psi_v
    """

    # TODO: avoid access to p._cl_psi_v, which is a big kludge
    def op(self, p):
        i = p._cl_stack_i
        nz = np.int32(p._probe.shape[-3])
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        # TODO: take into account different pixel size along x,y ?
        pxyz = p.voxel_size_object()
        pixel_size_object = np.sqrt(pxyz[1] * pxyz[2])
        # Swap p._cl_psi_v[i] = Psi(n) with p._cl_psi = Psi_fourier
        p._cl_psi_v[i], p._cl_psi = p._cl_psi, p._cl_psi_v[i]
        self.processing_unit.cl_2object_probe_psi_dm2(p._cl_psi[0, 0, 0], p._cl_psi_v[i], p._cl_obj, p._cl_probe,
                                                      p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y, p._cl_obs_v[i].cl_z,
                                                      pixel_size_object, f,
                                                      p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size,
                                                      nx, ny, nz, nxo, nyo, nzo, nb_obj, nb_probe)
        return p


class DM(CLOperatorBraggPtycho):
    """
    Operator to perform a complete Difference Map cycle, updating the Psi views for all stack of frames,
    as well as updating the object and/or probe.
    """

    def __init__(self, update_object=True, update_probe=False, nb_cycle=1, calc_llk=False, show_obj_probe=False,
                 fig_num=-1):
        """

        :param update_object: update object ?
        :param update_probe: update probe ?
        :param nb_cycle: number of cycles to perform. Equivalent to DM(...)**nb_cycle
        :param calc_llk: if True, calculate llk while in Fourier space. If a positive integer is given, llk will be
                         calculated every calc_llk cycle
        :param show_obj_probe: if a positive integer number N, the object & probe will be displayed every N cycle.
                               By default 0 (no plot)
        :param fig_num: the number of the figure to plot the object and probe, as for ShowObjProbe()
        """
        super(DM, self).__init__()
        self.nb_cycle = nb_cycle
        self.update_object = update_object
        self.update_probe = update_probe
        self.calc_llk = calc_llk
        self.show_obj_probe = show_obj_probe
        self.fig_num = fig_num

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new DM operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return DM(update_object=self.update_object, update_probe=self.update_probe, nb_cycle=self.nb_cycle * n,
                  calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe, fig_num=self.fig_num)

    def op(self, p):
        # Calculate starting Psi (note that all Psi are multiplied by the quadratic phase factor)
        p = LoopStack(ObjProbe2Psi(), keep_psi=True) * p

        # We could use instead of DM1 and DM2 operators:
        # op_dm1 = 2 * ObjProbe2Psi() - 1
        # op_dm2 = 1 - ObjProbe2Psi() + FourierApplyAmplitude() * op_dm1
        # But this would use 3 copies of the whole Psi stack - too much memory ?
        # TODO: check if memory usage would be that bad, or if it's possible the psi storage only applies
        # TODO: to the current psi array

        for ic in range(self.nb_cycle):
            t0 = timeit.default_timer()
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            ops = DM2() * FourierApplyAmplitude() * DM1()

            # TODO: handle the case where there is only 1 stack
            p = LoopStack(ops, keep_psi=True, copy=True) * p
            if self.update_object:
                p = Psi2ObjMerge() * LoopStack(Psi2Obj(), keep_psi=True) * p
            if self.update_probe:
                p = Psi2ProbeMerge() * LoopStack(Psi2Probe(), keep_psi=True) * p

            if calc_llk:
                # We need to perform a loop for LLK as the DM2 loop is on (2*PO-I), not the current PO estimate
                p = LoopStack(LLK() * FT() * ObjProbe2Psi()) * p
                dt = timeit.default_timer() - t0
                print("DM #%3d LLK= %8.2f(p) %8.2f(g) %8.2f(e), nb photons=%e, dt/cycle=%5.3fs" \
                      % (ic, p.llk_poisson / p.nb_obs, p.llk_gaussian / p.nb_obs, p.llk_euclidian / p.nb_obs,
                         p.nb_photons_calc, dt))
                # Restore correct Psi
                cl.enqueue_copy(self.processing_unit.cl_queue, src=p._cl_psi_v[p._cl_stack_i].data, dest=p._cl_psi.data)

            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    tit = "DM"
                    if self.update_object:
                        tit += "/"
                        if len(p._obj) > 1:
                            tit += "%d" % (len(p._obj))
                        tit += "o"
                    if self.update_probe:
                        tit += "/"
                        if len(p._probe) > 1:
                            tit += "%d" % (len(p._probe))
                        tit += "p"
                    tit += " #%3d, LLKn(p)=%8.3f" % (ic, p.llk_poisson / p.nb_obs)
                    p = cpuop.ShowObj(fig_num=self.fig_num, title=tit) * p

        return p


class _Grad(CLOperatorBraggPtycho):
    """
    Operator to compute the object and/or probe and/or background gradient corresponding to the current stack.
    """

    def __init__(self, update_object=True, update_probe=False, update_background=False, calc_llk=False):
        """
        :param update_object: compute gradient for the object ?
        :param update_probe: compute gradient for the probe ?
        :param update_background: compute gradient for the background ?
        :param calc_llk: calculate llk while in Fourier space
        """
        super(_Grad, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.calc_llk = calc_llk

    def op(self, p):
        i = p._cl_stack_i
        nz = np.int32(p._probe.shape[-3])
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nb_mode = np.int32(nb_obj * nb_probe)
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        first_pass = np.int8(i == 0)
        nb_psi = p._cl_obs_v[i].npsi
        nxyz = np.int32(nz * ny * nx)
        nxyzstack = np.int32(self.processing_unit.cl_stack_size * nxyz)
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        # TODO: take into account different pixel size along x,y ?
        pxyz = p.voxel_size_object()
        pixel_size_object = np.sqrt(pxyz[1] * pxyz[2])

        # Calculate FT(Obj*Probe)
        p = FT() * ObjProbe2Psi() * p
        if self.calc_llk:
            p = LLK() * p

        # Calculate Psi.conj() * (1-Iobs/I_calc) [for Poisson Gradient)
        # TODO: different noise models
        self.processing_unit.cl_grad_poisson_fourier(p._cl_obs_v[i].cl_obs, p._cl_psi, p._cl_background,
                                                     nb_psi, nb_mode, nx, ny, nz, nxyz, nxyzstack)
        p = IFT() * p

        if self.update_object:
            # We loop over the stack size to avoid creating an array with N=stack_size object gradient arrays
            for ii in range(p._cl_obs_v[i].npsi):
                self.processing_unit.cl_psi_to_obj_grad(p._cl_psi[0, 0, ii], p._cl_obj_grad, p._cl_probe,
                                                        p._cl_obs_v[i].x[ii], p._cl_obs_v[i].y[ii],
                                                        p._cl_obs_v[i].z[ii], pixel_size_object, f,
                                                        self.processing_unit.cl_stack_size,
                                                        nx, ny, nz, nxo, nyo, nzo, nb_obj, nb_probe)
        if self.update_probe:
            self.processing_unit.cl_psi_to_probe_grad(p._cl_psi[0, 0, 0], p._cl_probe_grad, p._cl_obj,
                                                      p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y, p._cl_obs_v[i].cl_z,
                                                      pixel_size_object, f, first_pass,
                                                      p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size,
                                                      nx, ny, nz, nxo, nyo, nzo, nb_obj, nb_probe)
        if self.update_background:
            # TODO
            pass
        return p


class Grad(CLOperatorBraggPtycho):
    """
    Operator to compute the object and/or probe and/or background gradient. The gradient is stored
    in the ptycho object. It is assumed that the GPU gradient arrays have been already created, normally
    by the calling ML operator.
    """

    def __init__(self, update_object=True, update_probe=False, update_background=False,
                 reg_fac_obj=0, reg_fac_probe=0, calc_llk=False):
        """

        :param update_object: compute gradient for the object ?
        :param update_probe: compute gradient for the probe ?
        :param update_background: compute gradient for the background ?
        :param calc_llk: calculate llk while in Fourier space
        """
        super(Grad, self).__init__()
        self.update_object = update_object
        self.update_probe = update_probe
        self.update_background = update_background
        self.calc_llk = calc_llk
        self.reg_fac_obj = reg_fac_obj
        self.reg_fac_probe = reg_fac_probe

    def op(self, p):
        if self.update_object:
            p._cl_obj_grad.fill(np.complex64(0))

        p = LoopStack(_Grad(update_object=self.update_object, update_probe=self.update_probe,
                            update_background=self.update_background, calc_llk=self.calc_llk)) * p

        # Sum the stack of object arrays
        nb_obj = np.int32(p._obj.shape[0])
        nxyzo = np.int32(p._obj.shape[-3] * p._obj.shape[-2] * p._obj.shape[-1])
        if self.reg_fac_obj != 0:
            # TODO
            # Regularization contribution to the object gradient
            pass
        if self.reg_fac_probe != 0:
            # TODO
            # Regularization contribution to the probe gradient
            pass

        return p


class _CGGamma(CLOperatorBraggPtycho):
    """
    Operator to compute the conjugate gradient gamma contribution to the current stack.
    """

    def __init__(self, update_background=False):
        """
        :param update_background: if updating the background ?
        """
        super(_CGGamma, self).__init__()
        self.update_background = update_background
        # TODO: fix this scale dynamically ? Used to avoid overflows
        self.gamma_scale = np.float32(1e-10)

    def op(self, p):
        i = p._cl_stack_i
        nz = np.int32(p._probe.shape[-3])
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nb_mode = np.int32(nb_obj * nb_probe)
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        nb_psi = p._cl_obs_v[i].npsi
        nxyz = np.int32(nz * ny * nx)
        nxyzstack = np.int32(self.processing_unit.cl_stack_size * nxyz)
        f = np.float32(np.pi / (p.data.wavelength * p.data.detector['distance']))
        # TODO: take into account different pixel size along x,y ?
        pxyz = p.voxel_size_object()
        pixel_size_object = np.sqrt(pxyz[1] * pxyz[2])

        for clpsi, clobj, clprobe in zip([p._cl_PO, p._cl_PdO, p._cl_dPO, p._cl_dPdO],
                                         [p._cl_obj, p._cl_obj_dir, p._cl_obj, p._cl_obj_dir],
                                         [p._cl_probe, p._cl_probe, p._cl_probe_dir, p._cl_probe_dir]):

            self.processing_unit.cl_object_probe_mult(clpsi[0, 0, 0], clobj, clprobe,
                                                      p._cl_obs_v[i].cl_x, p._cl_obs_v[i].cl_y, p._cl_obs_v[i].cl_z,
                                                      pixel_size_object, f,
                                                      p._cl_obs_v[i].npsi, self.processing_unit.cl_stack_size,
                                                      nx, ny, nz, nxo, nyo, nzo, nb_obj, nb_probe)
            plan = self.processing_unit.cl_fft_get_plan(clpsi, axes=None)
            for e in plan.enqueue(forward=True):
                e.wait()
            # TODO: why is this scaling useful for convergence ? Numerical stability ?
            self.processing_unit.cl_scale(clpsi, np.float32((clpsi[0, 0, 0].size) ** (-1 / 3.)))

        # TODO: take into account background
        tmp = self.processing_unit._cl_cg_poisson_gamma_red(p._cl_obs_v[i].cl_obs[:nb_psi],
                                                            p._cl_PO, p._cl_PdO, p._cl_dPO, p._cl_dPdO,
                                                            self.gamma_scale, nxyz, nxyzstack, nb_mode).get()
        if np.isnan(tmp['x'] + tmp['y']) or np.isinf(tmp['x'] + tmp['y']):
            nP = self.processing_unit.cl_norm_complex_n(p._cl_probe, 2).get()
            nO = self.processing_unit.cl_norm_complex_n(p._cl_obj, 2).get()
            ndP = self.processing_unit.cl_norm_complex_n(p._cl_probe_dir, 2).get()
            ndO = self.processing_unit.cl_norm_complex_n(p._cl_obj_dir, 2).get()
            nPO = self.processing_unit.cl_norm_complex_n(p._cl_PO, 2).get()
            ndPO = self.processing_unit.cl_norm_complex_n(p._cl_dPO, 2).get()
            nPdO = self.processing_unit.cl_norm_complex_n(p._cl_PdO, 2).get()
            ndPdO = self.processing_unit.cl_norm_complex_n(p._cl_dPdO, 2).get()
            print('_CGGamma norms: P %e O %e dP %e dO %e PO %e, PdO %e, dPO %e, dPdO %e' %
                  (nP, nO, ndP, ndO, nPO, nPdO, ndPO, ndPdO))
            print('_CGGamma (stack #%d, NaN Gamma:)' % i, tmp['x'], tmp['y'])
            raise OperatorException("NaN")
        p._cl_cg_gamma_d += tmp['y']
        p._cl_cg_gamma_n += tmp['x']
        if False:
            tmp = self.processing_unit._cl_cg_poisson_gamma4_red(p._cl_obs_v[i].cl_obs[:nb_psi],
                                                                 p._cl_PO, p._cl_PdO, p._cl_dPO, p._cl_dPdO,
                                                                 self.gamma_scale, nxyz, nxyzstack, nb_mode).get()
            p._cl_cg_gamma4 += np.array((tmp['w'], tmp['z'], tmp['y'], tmp['x'], 0))
        if self.update_background:
            # TODO: use a different kernel if there is a background gradient
            pass
        return p


class ML(CLOperatorBraggPtycho):
    """
    Operator to perform a maximum-likelihood conjugate-gradient minimization.
    """

    def __init__(self, nb_cycle=1, update_object=True, update_probe=False, update_background=False,
                 reg_fac_obj=0, reg_fac_probe=0, calc_llk=False, show_obj_probe=False, fig_num=-1):
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

    def __pow__(self, n):
        """

        :param n: a strictly positive integer
        :return: a new ML operator with the number of cycles multiplied by n
        """
        assert isinstance(n, int) or isinstance(n, np.integer)
        return ML(nb_cycle=self.nb_cycle * n, update_object=self.update_object, update_probe=self.update_probe,
                  update_background=self.update_background, reg_fac_obj=self.reg_fac_obj,
                  reg_fac_probe=self.reg_fac_probe, calc_llk=self.calc_llk, show_obj_probe=self.show_obj_probe,
                  fig_num=self.fig_num)

    def op(self, p):
        nz = np.int32(p._probe.shape[-3])
        ny = np.int32(p._probe.shape[-2])
        nx = np.int32(p._probe.shape[-1])
        nb_probe = np.int32(p._probe.shape[0])
        nb_obj = np.int32(p._obj.shape[0])
        nzo = np.int32(p._obj.shape[-3])
        nyo = np.int32(p._obj.shape[-2])
        nxo = np.int32(p._obj.shape[-1])
        cl_queue = self.processing_unit.cl_queue
        stack_size = self.processing_unit.cl_stack_size

        # Create the necessary GPU arrays for ML
        p._cl_PO = cla.empty_like(p._cl_psi)
        p._cl_PdO = cla.empty_like(p._cl_psi)
        p._cl_dPO = cla.empty_like(p._cl_psi)
        p._cl_dPdO = cla.empty_like(p._cl_psi)
        p._cl_obj_dir = cla.zeros(cl_queue, (nb_obj, nzo, nyo, nxo), np.complex64)
        p._cl_probe_dir = cla.zeros(cl_queue, (nb_probe, nz, ny, nx), np.complex64)
        if self.update_object:
            p._cl_obj_grad = cla.empty(cl_queue, (nb_obj, nzo, nyo, nxo), np.complex64)
            p._cl_obj_grad_last = cla.empty(cl_queue, (nb_obj, nzo, nyo, nxo), np.complex64)
        if self.update_probe:
            p._cl_probe_grad = cla.empty(cl_queue, (nb_probe, nz, ny, nx), np.complex64)
            p._cl_probe_grad_last = cla.empty(cl_queue, (nb_probe, nz, ny, nx), np.complex64)
        if self.update_background:
            p._cl_background_grad = cla.zeros(cl_queue, (nz, ny, nx), np.float32)
            p._cl_background_grad_last = cla.zeros(cl_queue, (nz, ny, nx), np.float32)
            p._cl_background_dir = cla.zeros(cl_queue, (nz, ny, nx), np.float32)
        for ic in range(self.nb_cycle):
            t0 = timeit.default_timer()
            calc_llk = False
            if self.calc_llk:
                if ic % self.calc_llk == 0 or ic == self.nb_cycle - 1:
                    calc_llk = True

            # Swap gradient arrays - for CG, we need the previous gradient
            if self.update_object:
                p._cl_obj_grad, p._cl_obj_grad_last = p._cl_obj_grad_last, p._cl_obj_grad
            if self.update_probe:
                p._cl_probe_grad, p._cl_probe_grad_last = p._cl_probe_grad_last, p._cl_probe_grad
            if self.update_background:
                p._cl_background_grad, p._cl_background_grad_last = p._cl_background_grad_last, p._cl_background_grad

            # 1) Compute the gradients
            p = Grad(update_object=self.update_object, update_probe=self.update_probe,
                     update_background=self.update_background, calc_llk=calc_llk) * p

            # 2) Search direction
            beta = np.float32(0)
            if ic == 0:
                # first cycle
                if self.update_object:
                    cl.enqueue_copy(cl_queue, src=p._cl_obj_grad.data, dest=p._cl_obj_dir.data)
                if self.update_probe:
                    cl.enqueue_copy(cl_queue, src=p._cl_probe_grad.data, dest=p._cl_probe_dir.data)
                if self.update_background:
                    cl.enqueue_copy(cl_queue, src=p._cl_background_grad.data, dest=p._cl_background_dir.data)
            else:
                beta_d, beta_n = 0, 0
                # Polak-Ribière CG coefficient
                cg_pr = self.processing_unit._cl_cg_polak_ribiere_complex_red
                if self.update_object:
                    tmp = cg_pr(p._cl_obj_grad, p._cl_obj_grad_last).get()
                    beta_n += tmp['x']
                    beta_d += tmp['y']
                if self.update_probe:
                    tmp = cg_pr(p._cl_probe_grad, p._cl_probe_grad_last).get()
                    beta_n += tmp['x']
                    beta_d += tmp['y']
                if self.update_background:
                    tmp = cg_pr(p._cl_background_grad, p._cl_background_grad_last).get()
                    beta_n += tmp['x']
                    beta_d += tmp['y']
                # print("Beta= %e / %e"%(beta_n, beta_d))
                # Reset direction if beta<0 => beta=0
                beta = np.float32(max(0, beta_n / max(1e-20, beta_d)))
                if np.isnan(beta_n + beta_d) or np.isinf(beta_n + beta_d):
                    raise OperatorException("NaN")
                if self.update_object:
                    self.processing_unit.cl_linear_comb_fcfc(beta, p._cl_obj_dir, np.float32(-1), p._cl_obj_grad)
                if self.update_probe:
                    self.processing_unit.cl_linear_comb_fcfc(beta, p._cl_probe_dir, np.float32(-1), p._cl_probe_grad)
                if self.update_background:
                    self.processing_unit.cl_linear_comb_4f(beta, p._cl_background_dir,
                                                           np.float32(-1), p._cl_background_grad)
            # 3) Line minimization
            p._cl_cg_gamma_d, p._cl_cg_gamma_n = 0, 0
            if False:
                # We could use a 4th order LLK(gamma) approximation, but it does not seem to improve
                p._cl_cg_gamma4 = np.zeros(5, dtype=np.float32)
            p = LoopStack(_CGGamma(update_background=self.update_background)) * p
            if np.isnan(p._cl_cg_gamma_d + p._cl_cg_gamma_n or np.isinf(p._cl_cg_gamma_d + p._cl_cg_gamma_n)):
                raise OperatorException("NaN")

            if self.reg_fac_obj != 0:
                pass
                # TODO
            if self.reg_fac_probe != 0:
                # TODO
                pass

            if np.isnan(p._cl_cg_gamma_d + p._cl_cg_gamma_n) or np.isinf(p._cl_cg_gamma_d + p._cl_cg_gamma_n):
                print("Gamma = NaN ! :", p._cl_cg_gamma_d, p._cl_cg_gamma_n)
            gamma = np.float32(p._cl_cg_gamma_n / p._cl_cg_gamma_d)
            if False:
                # It seems the 2nd order gamma approximation is good enough.
                gr = np.roots(p._cl_cg_gamma4)
                print("CG Gamma4", p._cl_cg_gamma4, "\n", gr, np.polyval(p._cl_cg_gamma4, gr))
                print("CG Gamma2=", gamma, "=", p._cl_cg_gamma_n, "/", p._cl_cg_gamma_d)

            # 4) Object and/or probe and/or background update
            if self.update_object:
                self.processing_unit.cl_linear_comb_fcfc(np.float32(1), p._cl_obj, gamma, p._cl_obj_dir)

            if self.update_probe:
                self.processing_unit.cl_linear_comb_fcfc(np.float32(1), p._cl_probe, gamma, p._cl_probe_dir)

            if self.update_background:
                self.processing_unit.cl_linear_comb_4f(np.float32(1), p._cl_background, gamma, p._cl_background_dir)
            dt = timeit.default_timer() - t0
            if calc_llk:
                print("ML #%3d LLK= %8.2f(p) %8.2f(g) %8.2f(e), nb photons=%e, dt/cycle=%5.3fs" \
                      % (ic, p.llk_poisson / p.nb_obs, p.llk_gaussian / p.nb_obs, p.llk_euclidian / p.nb_obs,
                         p.nb_photons_calc, dt))
            if self.show_obj_probe:
                if ic % self.show_obj_probe == 0 or ic == self.nb_cycle - 1:
                    tit = "ML"
                    if self.update_object:
                        tit += "/"
                        if len(p._obj) > 1:
                            tit += "%d" % (len(p._obj))
                        tit += "o"
                    if self.update_probe:
                        tit += "/"
                        if len(p._probe) > 1:
                            tit += "%d" % (len(p._probe))
                        tit += "p"
                    tit += " #%3d, LLKn(p)=%8.3f" % (ic, p.llk_poisson / p.nb_obs)
                    p = cpuop.ShowObj(fig_num=self.fig_num, title=tit) * p

        # Clean up
        del p._cl_PO, p._cl_PdO, p._cl_dPO, p._cl_dPdO, p._cl_obj_dir, p._cl_probe_dir
        if self.update_object:
            del p._cl_obj_grad, p._cl_obj_grad_last
        if self.update_probe:
            del p._cl_probe_grad, p._cl_probe_grad_last
        if self.update_background:
            del p._cl_background_grad, p._cl_background_grad_last, p._cl_background_dir

        return p


class ScaleObjProbe(CLOperatorBraggPtycho):
    """
    Operator to scale the object and probe so that they have the same magnitude, and that the product of object*probe
    matches the observed intensity (i.e. sum(abs(obj*probe)**2) = sum(iobs))
    """

    def op(self, p):
        # First scale versus the observed intensity
        # TODO: scale taking into account the mask ?
        nb_photons_obs = p.data.iobs.sum()
        nb_photons_calc = 0
        for i in range(p._cl_stack_nb):
            p = ObjProbe2Psi() * SelectStack(i) * p
            nb_photons_calc += self.processing_unit.cl_norm_complex_n(p._cl_psi, 2).get()
        s = np.sqrt(nb_photons_obs / nb_photons_calc)
        os = self.processing_unit.cl_norm_complex_n(p._cl_obj, np.int32(1)).get()
        ps = self.processing_unit.cl_norm_complex_n(p._cl_probe, np.int32(1)).get()
        self.processing_unit.cl_scale(p._cl_probe, np.float32(np.sqrt(os / ps * s)))
        self.processing_unit.cl_scale(p._cl_obj, np.float32(np.sqrt(ps / os * s)))
        print("ScaleObjProbe:", ps, os, s, np.sqrt(os / ps * s), np.sqrt(ps / os * s))
        return p


class SelectStack(CLOperatorBraggPtycho):
    """
    Operator to select a stack of observed frames to work on. Note that once this operation has been applied,
    the new Psi value may be undefined (empty array), if no previous array existed.
    """

    def __init__(self, stack_i, keep_psi=False, copy=False):
        """
        Select a new stack of frames, swapping or copying data to store the last calculated psi array in the
        corresponding, ptycho object's _cl_psi_v[i] dictionnary.

        What happens is:
        * keep_psi=False: only the stack index in p is changed (p._cl_stack_i=stack_i)

        * keep_psi=True, copy=False: the previous psi is stored in p._cl_psi_v[p._cl_stack_i], the new psi is swapped
                                   with p._cl_psi_v[stack_i] if it exists, otherwise initialized as an empty array.

        * keep_psi=True, copy=True: the previous psi is stored in p._cl_psi_v[p._cl_stack_i], the new psi is copied
                                    from p._cl_psi_v[stack_i] if it exists, otherwise initialized as an empty array.

        Special case if stack_i == p._cl_stack_i: if keep_psi and copy are True, a copy is made in p._cl_psi_v[stack_i],
                                                  otherwise nothing is done.

        :param stack_i: the stack index.
        :param keep_psi: if True, when switching between stack, store psi in p._cl_psi_v.
        :param copy: by default when switching between psi stacks, the arrays are swapped, not copied to avoid memory
                     transfers. If copy=true, then the old value from the new selected stack is copied instead of
                     being swapped with the current psi.
        """
        super(SelectStack, self).__init__()
        self.stack_i = stack_i
        self.keep_psi = keep_psi
        self.copy = copy

    def op(self, p):
        if self.keep_psi:
            # TODO: this is a big KLUDGE: it allows even for a single stack, to keep an old copy... \
            # TODO: a different mechanism should be used for that purpose
            # 1) Make sure p._cl_psi_v[p._cl_stack_i] exists, it will be swapped with p._cl_psi
            need_init = False
            if p._cl_stack_i in p._cl_psi_v:
                if p._cl_psi_v[p._cl_stack_i] is None:
                    need_init = True
            else:
                need_init = True
            if need_init:
                p._cl_psi_v[p._cl_stack_i] = cl.array.empty_like(p._cl_psi)

            # 2) Swap active stack to storage
            p._cl_psi, p._cl_psi_v[p._cl_stack_i] = p._cl_psi_v[p._cl_stack_i], p._cl_psi

        if self.stack_i == p._cl_stack_i:
            if self.keep_psi and self.copy:
                cl.enqueue_copy(self.processing_unit.cl_queue, src=p._cl_psi_v[self.stack_i].data, dest=p._cl_psi.data)
            return p

        if self.keep_psi:
            # 3) Restore (swap or copy) new chosen stack if it already exists
            if self.stack_i in p._cl_psi_v:
                if p._cl_psi_v[self.stack_i] is not None:
                    if self.copy:
                        cl.enqueue_copy(self.processing_unit.cl_queue, src=p._cl_psi_v[self.stack_i].data,
                                        dest=p._cl_psi.data)
                    else:
                        p._cl_psi, p._cl_psi_v[self.stack_i] = p._cl_psi_v[self.stack_i], p._cl_psi

        p._cl_stack_i = self.stack_i
        return p


class LoopStack(CLOperatorBraggPtycho):
    """
    Operator to apply a given operator sequentially to the complete stack of frames of a ptycho object.
    """

    def __init__(self, op, keep_psi=False, copy=False):
        """

        :param op: the operator to apply, which can be a multiplication of operators
        :param keep_psi: if True, when switching between stacks, store psi in p._cl_psi_v.
        :param copy: by default when switching between psi stacks, the arrays are swapped, not copied to avoid memory
                     transfers. If copy=true, then the old value from the new selected stack is copied instead of
                     being swapped with the current psi.
        """
        super(LoopStack, self).__init__()
        self.stack_op = op
        self.keep_psi = keep_psi
        self.copy = copy

    def op(self, p):
        if p._cl_stack_nb == 1 and (self.copy is False or self.keep_psi is False):
            return self.stack_op * p
        else:
            for i in range(p._cl_stack_nb):
                p = self.stack_op * SelectStack(i, keep_psi=self.keep_psi, copy=self.copy) * p
                # print("  LoopStack(): i=%d" % (i), p._cl_psi_v)
            if self.keep_psi:
                # Copy last psi, keep stored psi copy intact.
                p = SelectStack(0, keep_psi=self.keep_psi, copy=True) * p

        return p
