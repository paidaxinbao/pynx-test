#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

"""
This package provides access to the processing unit test suite.
"""

import unittest

from pynx.processing_unit.test.test_opencl import suite as test_opencl_suite
from pynx.processing_unit.test.test_cuda import suite as test_cuda_suite
from pynx.processing_unit.test.test_cpu import suite as test_cpu_suite
from pynx.processing_unit.test.test_cuda_fft import suite as test_cuda_fft_suite
from pynx.processing_unit.test.test_opencl_fft import suite as test_opencl_fft_suite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_cpu_suite())
    test_suite.addTest(test_opencl_suite())
    test_suite.addTest(test_cuda_suite())
    test_suite.addTest(test_opencl_fft_suite())
    test_suite.addTest(test_cuda_fft_suite())
    return test_suite
