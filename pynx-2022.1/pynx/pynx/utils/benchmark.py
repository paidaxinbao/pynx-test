# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import numpy as np
from pynx.utils.matplotlib import pyplot as plt

from ..processing_unit import cuda_device, opencl_device
from .math import test_smaller_primes


def benchmark_fft(device, do_plot=True, ndim=2, batch=1, max_size=4096, step_size=16, verbose=True):
    """
    Calculate FFT speed  for one or several CUDA or OpenCL device(s)
    
    :param device: the device to be tested. This can be a pycuda or pyopencl device, or a string to match against
                   available devices names. It can also be a list/tuple of the two pycuda/pyopencl devices
    :param do_plot: if True, plot using matplotlib
    :param ndim: the number of dimensions for the FFT - either 2 or 3
    :param batch: the batch number for the FFT,e.g. to perform 2D FFT on a 3D stack of 'batch' depth
    :param max_size: the maximum (inclusive) size (along each FFT dimension) of the FFT.
                     benchmark will stop increasing stop when an exception (likely memory allocation) is caught.
    :return: a dictionary with the name of the device + CLFFT or CUFFT as key, and two vectors with fft size and gflops
    """
    results = {}
    min_size = 64
    if opencl_device.has_opencl:
        cl = opencl_device.cl
        cl_device = None
        if type(device) is cl.Device:
            cl_device = device
        elif type(device) is str:
            for p in cl.get_platforms():
                for d in p.get_devices():
                    if d.name.lower().find(device.lower()) < 0:
                        continue
                    cl_device = d
                    break
                if cl_device is not None:
                    break
        elif type(device) is tuple or device is list:
            for d in device:
                if type(d) is cl.Device:
                    cl_device = device
                    break
        if cl_device is not None:
            max_prime = 13
            vn = []
            vgflops = []
            vdt = []
            dt = 0
            for n in range(min_size, max_size, step_size):
                if test_smaller_primes(n, max_prime, required_dividers=None):
                    try:
                        if ndim == 2:
                            gflops = opencl_device.cl_device_fft_speed(cl_device, fft_shape=(batch, n, n),
                                                                       axes=(-1, -2), verbose=verbose)
                        else:
                            gflops = opencl_device.cl_device_fft_speed(cl_device, fft_shape=(n, n, n), axes=None,
                                                                       verbose=verbose)
                        vn.append(n)
                        vgflops.append(gflops)
                        vdt.append(dt)
                    except:
                        break
            results["CLFFT: %s" % cl_device.name] = np.array(vn), np.array(vgflops), np.array(vdt)

        else:
            print("No pyopencl device matching")

    if cuda_device.has_cuda:
        cu_drv = cuda_device.cu_drv
        cu_device = None
        if type(device) is cu_drv.Device:
            cu_device = device
        elif type(device) is str:
            for i in range(cu_drv.Device.count()):
                if cu_drv.Device(i).name().lower().find(device.lower()) < 0:
                    continue
                cu_device = cu_drv.Device(i)
                break
        elif type(device) is tuple or device is list:
            for d in device:
                if type(d) is cu_drv.Device:
                    cu_device = device
                    break
        if cu_device is not None:
            cu_ctx = cu_device.make_context()
            cu_ctx.push()  # Is that needed after make_context() ?
            max_prime = 7
            vn = []
            vgflops = []
            vdt = []
            dt = 0
            for n in range(min_size, max_size + 1, step_size):
                if test_smaller_primes(n, max_prime, required_dividers=None):
                    try:
                        if ndim == 2:
                            gflops = cuda_device.cuda_device_fft_speed(cu_ctx, fft_shape=(batch, n, n), batch=True,
                                                                       verbose=verbose)
                        else:
                            gflops = cuda_device.cuda_device_fft_speed(cu_ctx, fft_shape=(n, n, n), batch=False,
                                                                       verbose=verbose)
                        vn.append(n)
                        vgflops.append(gflops)
                        vdt.append(dt)
                    except:
                        break
            results["CUFFT: %s" % cu_device.name()] = np.array(vn), np.array(vgflops), np.array(vdt)
            cu_ctx.pop()

        else:
            print("No pyopencl device matching")

    if do_plot:
        plt.figure()
        vx, vy = [], []
        for k, v in results.items():
            vn, vgflops, vdt = v
            col = 'b'
            if 'CUFFT' in k:
                col = 'g'
            plt.semilogx(vn, vgflops, '%s-' % col, label=k)
            plt.semilogx(vn, vgflops, '%s.' % col)
            vx += vn.tolist()
            vy += vgflops.tolist()
        plt.ylabel('Gflop/s')
        plt.xlabel('%dD FFT size' % (ndim))
        plt.title('%dD FFT, batch=%d [in-place complex64 transform]' % (ndim, batch))
        plt.legend(loc='upper left', fontsize=10)
        x = 2 ** np.arange(np.log2(min_size), np.log2(max(vx)), dtype=np.int)
        plt.xticks(x, x)
        plt.xlim(min(vx), max(vx))
        plt.ylim(min(vy), max(vy))
        plt.grid()

    return results
