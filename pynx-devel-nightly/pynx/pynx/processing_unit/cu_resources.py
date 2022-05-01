# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import atexit
try:
    import pycuda.driver as cu_drv
    import pycuda.tools as cu_tools

    has_cuda = True
except ImportError:
    has_cuda = False


class CUResources(object):
    """
    This class handles CUDA resources (e.g. CUDA context) to avoid wasting GPU memory.
    """

    def __init__(self):
        self._device_context_dict = {}
        self.cu_mem_pool  = None

    def get_context(self, device):
        """
        Method to get a context, using the static device context dictionary to avoid creating new contexts,
        which will use up the GPU memory.
        :param device: the pyCUDA device for which a context is desired
        """
        if device in self._device_context_dict:
            ctx = self._device_context_dict[device]
            ctx.push()
            return ctx
        # Create a new context
        ctx = device.make_context()
        self._device_context_dict[device] = ctx
        return ctx

    def get_memory_pool(self):
        if self.cu_mem_pool is None:
            self.cu_mem_pool = cu_tools.DeviceMemoryPool()
        return self.cu_mem_pool


cu_resources = CUResources()

def cleanup_cu_ctx():
    # Is that really clean ?
    if has_cuda:
        if cu_resources.cu_mem_pool is not None:
            # See https://github.com/inducer/pycuda/issues/74
            # Need to sotop memory pool before context deletion
            cu_resources.cu_mem_pool.stop_holding()
        if cu_drv.Context is not None:
            while cu_drv.Context.get_current() is not None:
                cu_drv.Context.pop()


atexit.register(cleanup_cu_ctx)
