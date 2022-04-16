# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import unittest
import traceback
import types


class TestImports(unittest.TestCase):

    @unittest.expectedFailure
    def test_expected_failure(self):
        # Dummy test
        msg = "Dummy test for expected failures"
        try:
            import does_not_exist
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    def test_numpy(self):
        msg = "Testing for numpy"
        try:
            import numpy as np
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    def test_scipy(self):
        msg = "Testing for scipy"
        try:
            import scipy
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    def test_silx(self):
        msg = "Testing for silx"
        try:
            import silx
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    def test_fabio(self):
        msg = "Testing for fabio"
        try:
            import silx
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    def test_h5py(self):
        msg = "Testing for h5py and hdf5plugin"
        try:
            import hdf5plugin
            import h5py
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    def test_skimage(self):
        msg = "Testing for scikit-image"
        try:
            import skimage
            import_ok = True
        except ImportError:
            msg += "\n" + traceback.format_exc()
            import_ok = False
        self.assertTrue(import_ok, msg=msg)


def suite():
    test_suite = unittest.TestSuite()
    for k, v in TestImports.__dict__.items():
        if isinstance(v, types.FunctionType):
            test_suite.addTest(TestImports(k))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
