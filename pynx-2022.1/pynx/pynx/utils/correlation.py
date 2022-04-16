# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import sys
import warnings
import gc
import numpy as np
from ..processing_unit import has_opencl, opencl_device, default_processing_unit as main_default_processing_unit

if has_opencl and main_default_processing_unit.pu_language not in ['cuda', 'cpu']:
    import pyopencl.array as cla
    from pyopencl.elementwise import ElementwiseKernel as CL_ElK
    from pyopencl import CompilerWarning

    warnings.simplefilter('ignore', CompilerWarning)
    from ..processing_unit.cl_processing_unit import CLProcessingUnit
    from ..processing_unit.kernel_source import get_kernel_source as getks

    pu = CLProcessingUnit()


def shell_correlation(a1, a2, mask=None, vx=None, nb_log=4, nb_shell=100, use_gpu=True, phase=False,
                      shell_correl_map=False):
    """
    Compute the cross-correlation between two arrays using opencl
    :param a1: first array (real-valued unless phase=True) to correlate (can have any dimensions)
    :param a2: second array to correlate. Must have the same shape and be aligned with a1
    :param mask: if given, mask of invalid values in both arrays (True or 1 = invalid pixels)
    :param vx: an array of range values. If None, will default to np.logspace()
    :param nb_log: if vx is not given, the spacing will use nblog range with np.logspace
    :param nb_shell: number of shells to use if vx is not given
    :param use_gpu: if True, do the computation on a GPU
    :param phase: if True, the phase of the two complex arrays will be cross-correlated
    :param shell_correl_map: if True, will also return an array with the shell correlation for each array element.
                             This is only implemented for a GPU.
    return: (x_shell_edges, v_shell, v_shell_nb) coordinates (nb_shell+1), masked values (nb_shell)
            and the number of pair points involved for each shell correlation. If shell_correl_map==True,
            an additional array is returned with the same shape as a1/a2 plus an extra dimension of size nb_shell,
            giving the shell correlation curve for each point, and another one giving the number of points
            used to calculate each shell correlation value.
    """
    if mask is None:
        a1m, a2m = a1.flat, a2.flat
        ix0 = np.arange(a1.size, dtype=np.int32)
    else:
        ix0 = np.flatnonzero(mask == False)
        a1m = a1.ravel()[ix0]
        a2m = a2.ravel()[ix0]
    if vx is None:
        if phase:
            tmp = np.angle(a1m)
            vmax = np.log10(np.percentile(tmp - min(tmp), 99.9))
        else:
            vmax = np.log10(np.percentile(a1m - min(a1m), 99.9))
        vx = np.logspace(vmax - nb_log, vmax, nb_shell + 1, dtype=np.float32)
    if phase:
        a1m = np.ravel(a1m).astype(np.complex64)
        a2m = np.ravel(a2m).astype(np.complex64)
    else:
        a1m = np.ravel(a1m).astype(np.float32)
        a2m = np.ravel(a2m).astype(np.float32)
    vx = vx.astype(np.float32)
    vy = np.zeros(nb_shell, dtype=np.float32)
    vyn1 = np.zeros(nb_shell, dtype=np.float32)
    vyn2 = np.zeros(nb_shell, dtype=np.float32)
    vynb = np.zeros(nb_shell, dtype=np.int32)
    # For each point, compute the absolute difference with all other pixels
    # in the same array, and correlate with the other array differences as a function
    # of the difference shell.
    if opencl_device.has_opencl and use_gpu:
        if pu.cl_ctx is None:
            if main_default_processing_unit.cl_device is None:
                main_default_processing_unit.select_gpu(language='opencl')
            pu.init_cl(cl_device=main_default_processing_unit.cl_device, test_fft=False, verbose=False)
        # Always re-generate kernels since nb_shell may change
        pu.cl_shell_correl = CL_ElK(pu.cl_ctx, name='cl_shell_correl',
                                    operation="shell_correl(i, v1, v2, x, y, n1, n2, ynb, nb, shell_map, shell_v_y,"
                                              "shell_v_nb)",
                                    options="-cl-mad-enable -cl-fast-relaxed-math",
                                    preamble=getks('opencl/complex.cl') +
                                             getks('utils/opencl/correlation.cl').replace('NB_SHELL', '%d' % nb_shell),
                                    arguments="__global float *v1, __global float *v2, __global float* x,"
                                              "__global float* y, __global float* n1, __global float* n2,"
                                              "__global int* ynb, const int nb, const char shell_map,"
                                              "__global float* shell_v_y, __global int* shell_v_nb")
        pu.cl_shell_correl_phase = CL_ElK(pu.cl_ctx, name='cl_shell_correl_phase',
                                          operation="shell_correl_phase(i, v1, v2, x, y, n1, n2, ynb, nb, shell_map,"
                                                    "shell_v_y, shell_v_nb)",
                                          options="-cl-mad-enable -cl-fast-relaxed-math",
                                          preamble=getks('opencl/complex.cl') +
                                                   getks('utils/opencl/correlation.cl').replace('NB_SHELL',
                                                                                                '%d' % nb_shell),
                                          arguments="__global float2 *v1, __global float2 *v2, "
                                                    "__global float* x, __global float* y, __global float* n1,"
                                                    "__global float* n2, __global int* ynb, const int nb,"
                                                    "const char shell_map, __global float* shell_v_y,"
                                                    "__global int* shell_v_nb")

        v1_cl = cla.to_device(pu.cl_queue, a1m)
        v2_cl = cla.to_device(pu.cl_queue, a2m)
        vx_cl = cla.to_device(pu.cl_queue, vx)
        vy_cl = cla.to_device(pu.cl_queue, vy)
        vyn1_cl = cla.to_device(pu.cl_queue, vyn1)
        vyn2_cl = cla.to_device(pu.cl_queue, vyn2)
        vynb_cl = cla.to_device(pu.cl_queue, vynb)
        if shell_correl_map:
            v_y_map_cl = cla.zeros(pu.cl_queue, (nb_shell, a1m.size), dtype=np.float32)
            v_nb_map_cl = cla.zeros(pu.cl_queue, (nb_shell, a1m.size), dtype=np.int32)
        else:
            # Apparently we can't use None for arrays passed to an elementwise kernel (should give a NULL pointer)
            v_y_map_cl = cla.zeros(pu.cl_queue, 1, dtype=np.float32)
            v_nb_map_cl = cla.zeros(pu.cl_queue, 1, dtype=np.int32)
        if phase:
            pu.cl_shell_correl_phase(v1_cl, v2_cl, vx_cl, vy_cl, vyn1_cl, vyn2_cl, vynb_cl, np.int32(a1m.size),
                                     np.int8(shell_correl_map), v_y_map_cl, v_nb_map_cl)
        else:
            pu.cl_shell_correl(v1_cl, v2_cl, vx_cl, vy_cl, vyn1_cl, vyn2_cl, vynb_cl, np.int32(a1m.size),
                               np.int8(shell_correl_map), v_y_map_cl, v_nb_map_cl)

        vy = vy_cl.get()
        vyn1 = vyn1_cl.get()
        vyn2 = vyn2_cl.get()
        vynb = vynb_cl.get()
        # Cleanup
        vy_cl.data.release()
        vyn1_cl.data.release()
        vyn2_cl.data.release()
        vynb_cl.data.release()
        del pu.cl_shell_correl, pu.cl_shell_correl_phase

        if shell_correl_map:
            v_y_map = v_y_map_cl.get()
            v_nb_map = v_nb_map_cl.get()
            vy_map = np.zeros([nb_shell] + list(a1.shape), dtype=np.float32)
            vynb_map = np.zeros([nb_shell] + list(a1.shape), dtype=np.int32)
            for i in range(nb_shell):
                vy_map[i].flat[ix0] = v_y_map[i]
                vynb_map[i].flat[ix0] = v_nb_map[i]
            # Cleanup
            v_y_map.data.release()
            v_nb_map.data.release()
        gc.collect()
    else:
        if shell_correl_map:
            v_y_map = np.zeros((nb_shell, a1m.size), dtype=np.float32)
            v_nb_map = np.zeros((nb_shell, a1m.size), dtype=np.int32)
        # Slow CPU version
        for i in range(len(a1m)):
            if i % int(len(a1m) / 10) == 0:
                sys.stdout.write("..%d%%" % ((len(a1m) - i) * 100 / len(a1m)))
                sys.stdout.flush()

            vyi = np.zeros(nb_shell, dtype=np.float32)
            vyin1 = np.zeros(nb_shell, dtype=np.float32)
            vyin2 = np.zeros(nb_shell, dtype=np.float32)
            d1i = abs(a1m[i] - a1m)
            d2i = abs(a2m[i] - a2m)
            for ii in range(nb_shell):
                ix = np.where(np.logical_and((d1i >= vx[ii]), (d1i < vx[ii + 1])))[0]
                if len(ix) > 0:
                    d1i_shell = np.take(d1i, ix)
                    d2i_shell = np.take(d2i, ix)
                    vyi[ii] += (d1i_shell * d2i_shell).sum()
                    vyin1[ii] += (d1i_shell ** 2).sum()
                    vyin2[ii] += (d2i_shell ** 2).sum()
                    vynb[ii] += len(ix)
                    if shell_correl_map:
                        v_y_map[ii, i] += (d1i_shell * d2i_shell).sum()
                        v_nb_map[ii, i] += len(ix)

                ix = np.where(np.logical_and((d2i >= vx[ii]), (d2i < vx[ii + 1])))[0]
                if len(ix) > 0:
                    d1i_shell = np.take(d1i, ix)
                    d2i_shell = np.take(d2i, ix)
                    vyi[ii] += (d1i_shell * d2i_shell).sum()
                    vyin1[ii] += (d1i_shell ** 2).sum()
                    vyin2[ii] += (d2i_shell ** 2).sum()
                    vynb[ii] += len(ix)
                    if shell_correl_map:
                        v_y_map[ii, i] += (d1i_shell * d2i_shell).sum()
                        v_nb_map[ii, i] += len(ix)
            vy += vyi
            vyn1 += vyin1
            vyn2 += vyin2
            tmp = np.sqrt(vyin1 * vyin2)
            vyi /= tmp + (tmp == 0) * (tmp.min() * 1e-6)
        sys.stdout.write("\n")
        sys.stdout.flush()

    tmp = np.sqrt(vyn1 * vyn2)
    vy /= tmp + (tmp == 0) * (tmp.mean() * 1e-12)
    if shell_correl_map:
        vy_map = np.zeros([nb_shell] + list(a1.shape), dtype=np.float32)
        vynb_map = np.zeros([nb_shell] + list(a1.shape), dtype=np.int32)
        for i in range(nb_shell):
            vy_map[i].flat[ix0] = v_y_map[i]
            vynb_map[i].flat[ix0] = v_nb_map[i]
        return vx, vy, vynb, vy_map, vynb_map

    return vx, vy, vynb


def std_dev_pair_correlation(a1, a2, mask=None, use_gpu=True):
    """
    Compute the per-pixel standard deviation between two arrays.
    :param a1: first array (real-valued) to correlate (can have any dimensions)
    :param a2: second array to correlate. Must have the same shape and be aligned with a1
    :param mask: if given, mask of invalid values in both arrays (True or 1 = invalid pixels)
    :param use_gpu: if True, use a GPU for calculations
    return: an array with the standard deviation from the difference between
            each pixel and all the other pixels of the same array, and computing the stndard
            deviation with the same difference in the other array.
    """
    if mask is None:
        a1m, a2m = a1.flat, a2.flat
        ix0 = np.arange(a1.size, dtype=np.int32)
    else:
        ix0 = np.flatnonzero(mask == False)
        a1m = a1.ravel()[ix0]
        a2m = a2.ravel()[ix0]
    a1m = np.ravel(a1m).astype(np.float32)
    a2m = np.ravel(a2m).astype(np.float32)
    stddev = np.zeros_like(a1m)
    if has_opencl and use_gpu:
        if pu.cl_ctx is None:
            if main_default_processing_unit.cl_device is None:
                main_default_processing_unit.select_gpu(language='opencl')
            pu.init_cl(cl_device=main_default_processing_unit.cl_device, test_fft=False, verbose=False)
        if 'cl_std_dev_pair' not in dir(pu):
            pu.cl_std_dev_pair = CL_ElK(pu.cl_ctx, name='cl_std_dev_pair',
                                        operation="std_dev_pair(i, v1, v2, stddev, nb)",
                                        options="-cl-mad-enable -cl-fast-relaxed-math",
                                        preamble=getks('opencl/complex.cl') + getks('utils/opencl/std_dev_pair_elw.cl'),
                                        arguments="__global float *v1, __global float *v2, __global float* stddev,"
                                                  "const int nb")

        v1_cl = cla.to_device(pu.cl_queue, a1m)
        v2_cl = cla.to_device(pu.cl_queue, a2m)
        stddev_cl = cla.to_device(pu.cl_queue, stddev)
        pu.cl_std_dev_pair(v1_cl, v2_cl, stddev_cl, np.int32(a1m.size))
        stddev = np.zeros_like(a1)
        stddev.flat[ix0] = stddev_cl.get()
    else:
        # CPU version
        for i in range(len(a1m)):
            d1 = np.abs(a1m[i] - a1m)
            dmean = (d1 + np.abs(a2m[i] - a2m)) / 2
            stddev[i] = np.sqrt(((d1 - dmean) ** 2).sum() / (a1m.size - 1))
        tmp = np.zeros_like(a1)
        tmp.flat[ix0] = stddev
        stddev = tmp
    return stddev
