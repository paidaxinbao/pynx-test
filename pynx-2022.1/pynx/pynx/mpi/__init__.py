# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2020-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

try:
    from mpi4py import MPI

    mpic = MPI.COMM_WORLD
except ImportError:
    MPI = None
