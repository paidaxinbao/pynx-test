# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import gc
from itertools import permutations
import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cla
    from pyopencl import clrandom

    try:
        from pyvkfft.opencl import VkFFTApp
        has_vkfft_opencl = True
        try:
            # skcuda.fft import creates a context on the first available card
            import gpyfft
        except ImportError:
            gpyfft = None
    except ImportError:
        has_vkfft_opencl = False
        VkFFTApp = None

    has_opencl = True
except:
    has_opencl = False

from .cl_resources import cl_resources

import timeit


def cl_device_fft_speed(d=None, fft_shape=(16, 256, 256), axes=(-1, -2), verbose=False, nb_test=4, nb_cycle=1,
                        timing=False, shuffle_axes=False):
    """
    Compute the FFT calculation speed for a given OpenCL device.
    
    :param d: the pyopencl.Device. If not supplied, pyopencl.create_some_context() will be called, and a
              device can be chosen interactively. This will result in a new context created for each call,
              and is not efficient (the context memory cannot be freed).
    :param fft_shape: (nz,ny,nx) the shape of the complex fft transform, treated as a stack of nz 2D transforms
                                 of size nx * ny, or as a single 3D FFT, depending on the value of 'axes'
    :param axes: (1,2) the axes for the FFT. Default value is (-1,-2), which will perform a stacked 2d fft.
                       Using None will perform a 3d fft.
    :param verbose: if True, print the speed and timing for the given transform
    :param nb_test: number of time the calculations will be repeated, the best result is returned
    :param nb_cycle: each test consist of nb_cycle forward and backward FFT.
    :param timing: if True, also return the time needed for a single FFT (dt)
    :param shuffle_axes: if True, the order of axes for the transform will be shuffled to find the
                         fastest combination, and the optimal axes order will be returned.
                         Only useful for gpyfft, ignored when pyvkfft is used.
    :return: The computed speed in Gflop/s (if timing is False) or a tuple (flops, dt), and also
             with the axes if shuffle_axes is True.
    """
    destroy_ctx = False
    if d is None:
        cl_ctx = cl.create_some_context()
        destroy_ctx = True
    else:
        cl_ctx = cl_resources.get_context(d)
    cl_queue = cl.CommandQueue(cl_ctx)
    cl_psi = cla.zeros(cl_queue, fft_shape, np.complex64)

    test_3d = False
    if axes is None:
        test_3d = True
        # reverse order of axes to get the best possible speed with current clFFT
        axes = (-1, -2, -3)
    elif len(axes) == 3:
        test_3d = True

    vax = []
    if shuffle_axes and not has_vkfft_opencl:
        vax = [p for p in permutations(axes)]
    else:
        vax = [axes]

    vdt = []
    vflops = []
    for axes in vax:
        if has_vkfft_opencl:
            plan = VkFFTApp(fft_shape, cl_psi.dtype, cl_queue, axes=axes, inplace=True)
        else:
            gpyfft_plan = gpyfft.FFT(cl_ctx, cl_queue, cl_psi, None, axes=axes)
        dt = 0
        # Do N1 passes of N2 forward & backward transforms, best result returned
        for ii in range(nb_test):
            t00 = timeit.default_timer()
            ev = []
            for i in range(nb_cycle):
                if has_vkfft_opencl:
                    plan.fft(cl_psi, cl_psi)
                    plan.ifft(cl_psi, cl_psi)
                else:
                    ev += gpyfft_plan.enqueue(forward=True)
                    ev += gpyfft_plan.enqueue(forward=False)
            cl_queue.finish()
            dtnew = (timeit.default_timer() - t00) / nb_cycle
            if dt == 0:
                dt = dtnew
            elif dtnew < dt:
                dt = dtnew
        vdt.append(dt)
        if has_vkfft_opencl:
            del plan
        else:
            del gpyfft_plan
        gc.collect()
        if test_3d:
            nz, ny, nx = fft_shape
            # 3D FFT
            flop = 2 * 5 * nx * ny * nz * np.log2(nx * ny * nz)
            flops = flop / dt / 1e9
            if verbose:
                print("OpenCL FFT speed: %8.2f Gflop/s [%8.4fms per %dx%dx%d 3D transform, "
                      "axes=(%d,%d,%d)]"
                      % (flops, dt / 2 * 1000, nz, ny, nx, axes[0], axes[1], axes[2]))
        elif len(axes) == 2:
            ny, nx = fft_shape[-2:]
            if len(fft_shape) == 3:
                nz = fft_shape[0]
            else:
                nz = 1
            # 2D FFT along x and y
            flop = 2 * 5 * nx * ny * nz * np.log2(nx * ny)
            flops = flop / dt / 1e9
            if verbose:
                print(
                    "OpenCL FFT speed: %8.2f Gflop/s [%8.4fms per %dx%d 2D transform,"
                    "%8.3fms per stack of %d 2D transforms, axes=(%d,%d)]"
                    % (flops, dt / 2 / nz * 1000, ny, nx, dt / 2 * 1000, nz, axes[0], axes[1]))
        elif len(axes) == 1:
            nx = fft_shape[0]
            if len(fft_shape) == 2:
                # Batch 1D transform
                ny = fft_shape[0]
            else:
                ny = 1
            flop = 2 * 5 * nx * ny * np.log2(nx)
            flops = flop / dt / 1e9
            if verbose:
                print(
                    "OpenCL FFT speed: %8.2f Gflop/s [%8.4fms per %d 1D transform,"
                    "%8.3fms per stack of %d 1D transforms]"
                    % (flops, dt / 2 / ny * 1000, nx, dt / 2 * 1000, ny))

        vflops.append(flops)

    i = np.argmin(np.array(vdt))
    dt, axes, flops = vdt[i], vax[i], vflops[i]
    # Try to clean
    cl_psi.data.release()
    del cl_psi
    if destroy_ctx:
        del cl_ctx
    gc.collect()

    if timing:
        if shuffle_axes:
            return flops, dt, axes
        else:
            return flops, dt
    if shuffle_axes:
        return flops, axes
    return flops


def cl_device_global_mem_bandwidth(d):
    """
    Get the CUDA device global memory bandwidth
    :param d: the opencl device.
    :return: the memory bandwidth in Gbytes/s
    """
    destroy_ctx = False
    if d is None:
        cl_ctx = cl.create_some_context()
        destroy_ctx = True
    else:
        cl_ctx = cl_resources.get_context(d)
    cl_queue = cl.CommandQueue(cl_ctx)

    a = clrandom.rand(cl_queue, shape=(512, 512, 512), dtype=np.float32)
    b = cla.empty(cl_queue, shape=(512, 512, 512), dtype=np.float32)
    nb_test = 3
    nb_cycle = 10
    dt = 0
    # Do N passes, best result returned
    for ii in range(nb_test):
        t0 = timeit.default_timer()
        for j in range(nb_test):
            cl.enqueue_copy(cl_queue, dest=b.data, src=a.data)
            cl.enqueue_copy(cl_queue, dest=a.data, src=b.data)
        cl_queue.finish()
        dtnew = timeit.default_timer() - t0
        if dt == 0:
            dt = dtnew
        elif dtnew < dt:
            dt = dtnew

    if destroy_ctx:
        del cl_ctx
    gc.collect()
    return a.nbytes * 4 * nb_cycle / dt / 1024 ** 3


def available_gpu_speed(cl_platform=None, fft_shape=(16, 256, 256), axes=(-1, -2), min_gpu_mem=None, verbose=False,
                        gpu_name=None, only_gpu=True, return_dict=False, ranking='fft'):
    """
    Get a list of all available GPUs, sorted by FFT speed (Gflop/s) or bandwidth (Gbytes/s).
    
    Args:
        cl_platform: the OpenCL platform (default=None, all platform are tested)
        fft_shape: the FFT shape against which the fft speed is calculated. If None, no benchmark is performed,
                   the speed for all devices is reported as 0.
        axes: the fft axis
        min_gpu_mem: the minimum amount of gpu memory desired (bytes). Devices with less are ignored.
        verbose: if True, printout FFT speed and memory for found GPUs
        gpu_name: if given, only GPU whose name include this sub-string will be tested & reported. This can also be a
                  list of acceptable strings
        only_gpu: if True (the default), will skip non-GPU OpenCL devices
        return_dict: if True, a dictionary will be returned instead of a list, with both timing and gflops listed
        ranking: either 'fft' or 'bandwidth'.

    Returns:
        a list of tuples (GPU device, speed (Gflop/s)), ordered by decreasing speed.
        If return_dict is True, a dictionary is returned with each entry is a dictionary with gflops and dt results
    """
    if verbose:
        s = "Searching available OpenCL GPU"
        if gpu_name is not None:
            s += "[name=%s]" % str(gpu_name)
        if ranking == 'fft':
            if fft_shape is not None:
                s += "[ranking by fft, fft_shape=%s]" % (str(fft_shape))
        else:
            s += " [ranking by global memory bandwidth]"
        print(s + ":")

    if type(gpu_name) is str:
        gpu_names = [gpu_name]
    elif type(gpu_name) is list:
        gpu_names = gpu_name
    else:
        gpu_names = None

    gpu_dict = {}
    for p in cl.get_platforms():
        if cl_platform is not None:
            if type(cl_platform) == type(p):
                if p != cl_platform:
                    continue
            if type(cl_platform) == str:
                if p.name.lower().count(cl_platform) < 0:
                    continue
        elif p.name.lower().count("portable"):
            # Blacklist POCL
            if verbose:
                print("Ignoring Portable Computing Language (POCL) platform by default")
            continue
        for d in p.get_devices():
            pd_name = d.name + " [" + p.name + "]"
            if gpu_names is not None:
                skip = True
                for n in gpu_names:
                    if n.lower() in d.name.lower():
                        skip = False
                        break
                if skip:
                    continue

            if only_gpu:
                if d.type & cl.device_type.GPU is False:
                    continue

            if d.type & cl.device_type.GPU:
                mem = int(round(d.global_mem_size // 1024 ** 3))
                if min_gpu_mem is not None:
                    if d.max_mem_alloc_size < min_gpu_mem:
                        if verbose:
                            print("%60s: max memory=%5d Gb < %5d Gb - IGNORED" %
                                  (pd_name, mem, int(round(min_gpu_mem // 1024 ** 3))))
                        continue
                if ranking == 'fft':
                    if fft_shape is not None:
                        if True:  # try
                            flops, dt = cl_device_fft_speed(d, fft_shape=fft_shape, axes=axes, verbose=False,
                                                            timing=True)
                        else:
                            flops, dt = 0, 0
                    else:
                        flops, dt = -1, -1
                    if verbose:
                        if fft_shape is not None:
                            print("%60s: %4dMb [max alloc.: %3dMb],%7.2f Gflop/s"
                                  % (pd_name, int(round(d.global_mem_size // 2 ** 20)),
                                     int(round(d.max_mem_alloc_size / 2 ** 20)), flops))
                        else:
                            print("%60s: %4dMb [max alloc.: %3dMb]"
                                  % (pd_name, int(round(d.global_mem_size // 2 ** 20)),
                                     int(round(d.max_mem_alloc_size / 2 ** 20))))
                    if return_dict:
                        gpu_dict[d] = {'Gflop/s': flops, 'dt': dt}
                    else:
                        gpu_dict[d] = flops
                else:
                    gbps = int(round(cl_device_global_mem_bandwidth(d)))
                    if verbose:
                        print("%60s: %4d Gb, %5d Gbytes/s" % (pd_name, mem, gbps))
                    if return_dict:
                        gpu_dict[d] = {'Gbytes/s': gbps}
                    else:
                        gpu_dict[d] = gbps

    if return_dict:
        return gpu_dict
    else:
        return list(sorted(gpu_dict.items(), key=lambda t: -t[1]))
