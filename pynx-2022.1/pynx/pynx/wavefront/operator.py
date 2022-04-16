# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from ..processing_unit import has_cuda, has_opencl
from ..processing_unit import default_processing_unit as main_default_processing_unit
from .cpu_operator import *

# Import CUDA operators if possible, otherwise OpenCL, unless a language has already been set through the default
# main processing unit
if has_cuda and main_default_processing_unit.pu_language not in ['opencl', 'cpu']:
    from .cu_operator import *
elif has_opencl and main_default_processing_unit.pu_language not in ['cuda', 'cpu']:
    from .cl_operator import *
