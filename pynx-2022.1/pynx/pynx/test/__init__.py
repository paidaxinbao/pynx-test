#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

"""
This package provides access to the full PyNX test suite.
"""

import unittest
import sys
import io
from pynx.test.test_imports import suite as imports_suite
from pynx.processing_unit.test import suite as processing_unit_suite
from pynx.cdi.test import suite as cdi_suite
from pynx.cdi.runner.test import suite as cdi_runner_suite
from pynx.ptycho.test import suite as ptycho_suite
from pynx.ptycho.runner.test import suite as ptycho_runner_suite
from pynx.scattering.test import suite as scattering_suite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(imports_suite())
    test_suite.addTest(processing_unit_suite())
    test_suite.addTest(cdi_suite())
    test_suite.addTest(cdi_runner_suite())
    test_suite.addTest(ptycho_suite())
    test_suite.addTest(ptycho_runner_suite())
    test_suite.addTest(scattering_suite())
    return test_suite


if __name__ == '__main__':
    # Suppressing stdout should not be necessary
    # TODO: use logging to redirect all tests output, and get more details about failures
    sys.stdout = io.StringIO()
    res = unittest.TextTestRunner(verbosity=2, descriptions=False).run(suite())
