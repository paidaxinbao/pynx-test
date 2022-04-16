# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="creating CUBLAS context to get version number")
try:
    import pycuda.driver as cu_drv

    cu_drv.init()
    import pycuda.gpuarray as cua
    from pycuda import curandom
    from pycuda.driver import Context

    has_cuda = True

    try:
        from pyvkfft.cuda import VkFFTApp
        has_vkfft_cuda = True
    except ImportError:
        has_vkfft_cuda = False
        VkFFTApp = None

    try:
        # skcuda.fft import creates a context on the first available card
        import skcuda.fft as cu_fft
        has_skcuda = True
    except ImportError:
        has_skcuda = False
        cu_fft = None
except:
    cu_drv = None
    has_cuda = False
    has_vkfft_cuda = False
    has_skcuda = False
    cu_fft = None


import timeit

from .cu_resources import cu_resources


def cuda_device_fft_speed(d=None, fft_shape=(16, 256, 256), batch=True, verbose=False, nb_test=4, nb_cycle=1,
                          timing=False):
    """
    Compute the FFT calculation speed for a given CUDA device.
    
    :param d: the pycuda.driver.Device. If not given, the default context will be used.
    :param fft_shape=(nz,ny,nx): the shape of the complex fft transform, treated as a stack of nz 2D transforms 
                                 of size nx * ny, or as a single 3D FFT, depending on the value of 'axes'
    :param batch: if True, will perform a batch 2D FFT. Otherwise, will perform a 3D FFT.
    :param verbose: if True, print the speed and timing for the given transform
    :param nb_test: number of time the calculations will be repeated, the best result is returned
    :param timing: if True, also return the time needed for a single FFT (dt)
    :return: The computed speed in Gflop/s (if timing is False) or a tuple (flops, dt)
    """
    if d is not None:
        ctx = cu_resources.get_context(d)

    a = curandom.rand(shape=fft_shape, dtype=np.float32).astype(np.complex64)

    if has_vkfft_cuda:
        if batch:
            plan = VkFFTApp(fft_shape, a.dtype, ndim=len(fft_shape)-1, inplace=True,)
        else:
            plan = VkFFTApp(fft_shape, a.dtype, inplace=True)
    else:
        if batch:
            plan = cu_fft.Plan(fft_shape[-2:], np.complex64, np.complex64, batch=fft_shape[0])
        else:
            plan = cu_fft.Plan(fft_shape, np.complex64, np.complex64, batch=1)

    dt = 0
    # Do N passes, best result returned
    for ii in range(nb_test):
        t00 = timeit.default_timer()
        for i in range(nb_cycle):
            if has_vkfft_cuda:
                plan.fft(a, a)
                plan.ifft(a, a)
            else:
                cu_fft.fft(a, a, plan)
                cu_fft.ifft(a, a, plan)
        Context.synchronize()
        dtnew = (timeit.default_timer() - t00) / nb_cycle
        if dt == 0:
            dt = dtnew
        elif dtnew < dt:
            dt = dtnew
    nz, ny, nx = fft_shape
    if batch:
        # 2D FFT along x and y
        flop = 2 * 5 * nx * ny * nz * np.log2(nx * ny)
        flops = flop / dt / 1e9
        if verbose:
            print("CUFFT speed: %8.2f Gflop/s [%8.4fms per %dx%d 2D transform, %8.3fms per stack of %d 2D transforms]"
                  % (flops, dt / 2 / nz * 1000, ny, nx, dt / 2 * 1000, nz))
    else:
        # 3D FFT
        flop = 2 * 5 * nx * ny * nz * np.log2(nx * ny * nz)
        flops = flop / dt / 1e9
        if verbose:
            print("CUFFT speed: %8.2f Gflop/s [%8.4fms per %dx%dx%d 3D transform]"
                  % (flops, dt / 2 * 1000, nz, ny, nx))
    if d is not None:
        # We created the context, so destroy it
        ctx.pop()
    if timing:
        return flops, dt
    return flops


def cuda_device_global_mem_bandwidth(d, measured=False):
    """
    Get the CUDA device global memory bandwidth
    :param d: the CUDA device.
    :param measured: if True, measure the bandwidth
    :return: the memory bandwidth in Gbytes/s
    """
    if not measured:
        # MEMORY_CLOCK_RATE is in kHz, * 2 due to double data rate (DDR) memory
        return d.get_attribute(cu_drv.device_attribute.MEMORY_CLOCK_RATE) * \
               d.get_attribute(cu_drv.device_attribute.GLOBAL_MEMORY_BUS_WIDTH) * 2 * 1000 / 8 / 1024 ** 3

    # Make sure a context is active
    ctx = cu_resources.get_context(d)

    a = curandom.rand(shape=(512, 512, 512), dtype=np.float32)
    b = cua.empty(shape=(512, 512, 512), dtype=np.float32)
    nb_test = 3
    nb_cycle = 10
    dt = 0
    ev_begin = cu_drv.Event()
    ev_end = cu_drv.Event()
    # Do N passes, best result returned
    for ii in range(nb_test):
        ev_begin.record()
        for i in range(nb_cycle):
            cu_drv.memcpy_dtod_async(dest=b.gpudata, src=a.gpudata, size=a.nbytes)
            cu_drv.memcpy_dtod_async(dest=a.gpudata, src=b.gpudata, size=a.nbytes)
        ev_end.record()
        ev_end.synchronize()
        dtnew = ev_end.time_since(ev_begin) / 1000
        if dt == 0:
            dt = dtnew
        elif dtnew < dt:
            dt = dtnew
    return a.nbytes * 4 * nb_cycle / dt / 1024 ** 3


def available_gpu_speed(fft_shape=(16, 256, 256), batch=True, min_gpu_mem=None, verbose=False, gpu_name=None,
                        return_dict=False, ranking='fft'):
    """
    Get a list of all available GPUs, sorted by FFT speed (Gflop/s) or memory bandwidth (Gbytes/s).
    
    Args:
        fft_shape: the FFT shape against which the fft speed is calculated
        batch: if True, perform a batch 2D FFT rather than a 3D one
        min_gpu_mem: the minimum amount of gpu memory desired (bytes). Devices with less are ignored.
        verbose: if True, printout speed and memory for found GPUs
        gpu_name: if given, only GPU whose name include this sub-string will be tested & reported. This can also be a
                  list of acceptable strings
        return_dict: if True, a dictionary will be returned instead of a list, with both timing and gflops listed
        ranking: either 'fft' or 'bandwidth'.

    Returns:
        a list of tuples (GPU device, speed (Gflop/s) or memory bandwidth), ordered by decreasing values.
        If return_dict is True, a dictionary is returned with each entry is a dictionary with gflops and dt results
    """
    if verbose:
        s = "Computing speed for available CUDA GPU"
        if gpu_name is not None:
            s += "[name=%s]" % (gpu_name)
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
    for i in range(cu_drv.Device.count()):
        d = cu_drv.Device(i)
        if gpu_names is not None:
            skip = True
            for n in gpu_names:
                if n.lower() in d.name().lower():
                    skip = False
                    break
            if skip:
                continue
        mem = int(round(d.total_memory() // 1024 ** 3))
        if min_gpu_mem is not None:
            if mem < min_gpu_mem / 1024 ** 3:
                continue
        if ranking == 'fft':
            if fft_shape is not None:
                try:
                    flops, dt = cuda_device_fft_speed(d, fft_shape=fft_shape, batch=batch, verbose=False, timing=True)
                except:
                    flops, dt = 0, 0
            else:
                flops, dt = -1, -1
            if verbose:
                if fft_shape is not None:
                    print("%60s: %4d Gb, %7.2f Gflop/s"
                          % (d.name(), mem, flops))
                else:
                    print("%60s: %4d Gb" % (d.name(), mem))
            if return_dict:
                gpu_dict[d] = {'Gflop/s': flops, 'dt': dt, 'mem': mem}
            else:
                gpu_dict[d] = flops
        else:
            gbps = int(round(cuda_device_global_mem_bandwidth(d, measured=True)))
            if verbose:
                print("%60s: %4d Gb, %5d Gbytes/s" % (d.name(), mem, gbps))
            if return_dict:
                gpu_dict[d] = {'Gbytes/s': gbps}
            else:
                gpu_dict[d] = gbps

    if return_dict:
        return gpu_dict
    else:
        return list(sorted(gpu_dict.items(), key=lambda t: -t[1]))
