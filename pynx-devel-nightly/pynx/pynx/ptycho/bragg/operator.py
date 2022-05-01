# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


from ...processing_unit import has_cuda, has_opencl
from .cpu_operator import *

# Import CUDA operators if possible, otherwise OpenCL
if False: #has_cuda:
    from .cu_operator import *
elif has_opencl:
    from .cl_operator import *
