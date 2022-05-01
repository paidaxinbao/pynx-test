# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

try:
    import pyopencl as cl

except ImportError:
    has_opencl = False


class CLResources(object):
    """
    This class handles OpenCL resources (e.g. OpenCL context) to avoid wasting GPU memory.
    """

    def __init__(self):
        self._device_context_dict = {}

    def get_context(self, device):
        """
        Static method to get a context, using the static device context dictionary to avoid creating new contexts,
        which will use up the GPU memory.
        :param device: the pyOpenCL device for which a context is desired
        """
        if device in self._device_context_dict:
            return self._device_context_dict[device]
        # Create a new context
        ctx = cl.Context([device])
        self._device_context_dict[device] = ctx
        return ctx


cl_resources = CLResources()
