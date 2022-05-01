# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import os


def get_kernel_source(relpath):
    """
    Get the source code of an OpenCL or CUDA kernel, from the given path relative to the root PyNX directory.
    
    Args:
        relpath: relative path for the kernel, e.g. "opencl/cg_polak_ribiere.cl"

    Returns:

    """
    # Using with... avoids some faulty warnings during tests
    with open(os.path.join(os.path.dirname(__file__), "../", relpath)) as d:
        return d.read()

