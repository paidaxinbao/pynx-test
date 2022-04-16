# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import pyopencl as cl
from pynx.processing_unit.opencl_device import cl_device_fft_speed
from pynx.processing_unit.cuda_device import cuda_device_fft_speed, cu_drv

ctx = cl.create_some_context()
d = ctx.devices[0]
cl_device_fft_speed(d, fft_shape=(256, 256, 256), axes=None, verbose=True)
cl_device_fft_speed(d, fft_shape=(512, 512, 512), axes=None, verbose=True)
cl_device_fft_speed(d, fft_shape=(1024, 1024, 1024), axes=None, verbose=True)
cl_device_fft_speed(d, fft_shape=(16, 1024, 1024), axes=(-1, -2), verbose=True)
cl_device_fft_speed(d, fft_shape=(16, 2048, 2048), axes=(-1, -2), verbose=True)
# cl_device_fft_speed(d,fft_shape=(16,4096,4096),axes=(-1,-2),verbose=True)


d = cu_drv.Device(0)
cuda_device_fft_speed(d, fft_shape=(256, 256, 256), batch=False, verbose=True)
cuda_device_fft_speed(d, fft_shape=(512, 512, 512), batch=False, verbose=True)
# cuda_device_fft_speed(d,fft_shape=(1024,1024,1024),batch=False,verbose=True)
cuda_device_fft_speed(d, fft_shape=(16, 1024, 1024), batch=True, verbose=True)
cuda_device_fft_speed(d, fft_shape=(16, 2048, 2048), batch=True, verbose=True)
cuda_device_fft_speed(d, fft_shape=(16, 4096, 4096), batch=True, verbose=True)
