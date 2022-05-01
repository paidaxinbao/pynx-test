#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

"""
This package includes tests for the CDI python API.
"""

import unittest

from .test_cdi import suite as test_cdi_suite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_cdi_suite())
    return test_suite
