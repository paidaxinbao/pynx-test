#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

"""
This package includes tests for the CDI command-line scripts.
"""

import os
import sys
import platform
import subprocess
import unittest
import tempfile
import shutil
# import warnings
# from functools import wraps
from pynx.cdi.test.test_cdi import make_cdi_data_file, make_cdi_support_file, \
    make_cdi_mask_file, make_cdi_flatfield_file
from pynx.processing_unit import has_opencl, has_cuda

has_mpi = False
try:
    from mpi4py import MPI

    if shutil.which('mpiexec') is not None:
        has_mpi = True
except ImportError:
    pass

exclude_cuda = False
exclude_opencl = False
if 'PYNX_PU' in os.environ:
    if 'opencl' in os.environ['PYNX_PU'].lower():
        exclude_cuda = True
    elif 'cuda' in os.environ['PYNX_PU'].lower():
        exclude_opencl = True


# def ignore_warnings(func):
#     @wraps(func)
#     def inner(self, *args, **kwargs):
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             res = func(self, *args, **kwargs)
#         return res
#     return inner


class TestCDIRunner(unittest.TestCase):
    """
    Class for tests of the CDI runner scripts
    """

    @classmethod
    def setUpClass(cls):
        # Directory contents will automatically get cleaned up on deletion
        cls.tmp_dir_obj = tempfile.TemporaryDirectory()
        cls.tmp_dir = cls.tmp_dir_obj.name

    # @ignore_warnings
    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_cdi_runner_id01_3d_cxi_opencl(self):
        my_env = os.environ.copy()
        my_env["PYNX_PU"] = "opencl"
        path = make_cdi_data_file(shape=(128, 128, 128), file_type='cxi', dir=self.tmp_dir)
        exe_suffix = '.py'
        if platform.system() == "Windows":
            exe_suffix = ''
        args = ['pynx-id01cdi' + exe_suffix, 'data=%s' % path, "nb_raar=50", "nb_hio=50",
                "nb_er=50", "support_update_period=20"]
        # Test suquentially several options. Could also try combined options..
        algo1 = "(Sup*ER**10)**2,psf=5,psf_filter=None, DetwinRAAR**10*Sup*RAAR**50"
        algo2 = "(Sup*ER**10)**2,psf_init=gaussian@1, DetwinRAAR**10*Sup*RAAR**50"
        for options in [[], ["roi=16,112,16,112,16,112"], ["positivity"], ["detwin=0"], ["output_format=npz"],
                        ["support_size=50"], ["support_only_shrink"], ["support_update_border_n=3"],
                        ["rebin=2"], ["rebin=2,1,2"], ["mask_interp=16,2"], ["psf=pseudo-voigt,1,0.05,10"],
                        ["psf=gaussian,1,10", "psf_filter=tukey"],
                        ['algo="%s"' % algo1], ['algo="%s"' % algo2]]:
            option_string = ""
            for s in options:
                option_string += s + " "
            with self.subTest(option=option_string, command=args + options):
                with subprocess.Popen(args + options, stderr=subprocess.PIPE,
                                      stdout=subprocess.PIPE, cwd=self.tmp_dir, env=my_env) as p:
                    stdout, stderr = p.communicate(timeout=200)
                    res = p.returncode
                    msg = "Failed command-line:"
                    for a in p.args:
                        msg += " " + a
                    msg += "\n"
                    self.assertFalse(res, msg=msg + stderr.decode())

    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    @unittest.skipUnless('live_plot' in sys.argv or 'liveplot' in sys.argv, "live plot tests skipped")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_cdi_runner_id01_3d_cxi_liveplot_opencl(self):
        my_env = os.environ.copy()
        my_env["PYNX_PU"] = "opencl"
        path = make_cdi_data_file(shape=(128, 128, 128), file_type='cxi', dir=self.tmp_dir)
        exe_suffix = '.py'
        if platform.system() == "Windows":
            exe_suffix = ''
        with subprocess.Popen(['pynx-id01cdi' + exe_suffix, 'data=%s' % path, 'live_plot'], stderr=subprocess.PIPE,
                              stdout=subprocess.PIPE, cwd=self.tmp_dir, env=my_env) as p:
            stdout, stderr = p.communicate(timeout=200)
            res = p.returncode
            self.assertFalse(res, msg=stderr.decode())

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_cdi_runner_id01_3d_cxi_cuda(self):
        my_env = os.environ.copy()
        my_env["PYNX_PU"] = "cuda"
        path = make_cdi_data_file(shape=(128, 128, 128), file_type='cxi', dir=self.tmp_dir)
        # test npy, npz ahd h5 for mask (filename will used a MiXed case)
        mask_path1 = make_cdi_mask_file(shape=(128, 128, 128), file_type='npy', dir=self.tmp_dir)
        mask_path2 = make_cdi_mask_file(shape=(128, 128, 128), file_type='npz', dir=self.tmp_dir)
        mask_path3 = make_cdi_mask_file(shape=(128, 128, 128), file_type='h5', dir=self.tmp_dir)
        # Test npy, npz and h5 for flatfield (2D and 3D)
        flat_path1 = make_cdi_flatfield_file(shape=(128, 128), file_type='npy', dir=self.tmp_dir)
        flat_path2 = make_cdi_flatfield_file(shape=(128, 128, 128), file_type='npz', dir=self.tmp_dir)
        flat_path3 = make_cdi_flatfield_file(shape=(128, 128), file_type='h5', dir=self.tmp_dir)
        # Test support, with different sizes
        support_path1 = make_cdi_support_file(shape=(128, 128, 128), dir=self.tmp_dir)
        support_path2 = make_cdi_support_file(shape=(50, 50, 50), dir=self.tmp_dir)
        support_path3 = make_cdi_support_file(shape=(130, 130, 130), dir=self.tmp_dir)
        exe_suffix = '.py'
        if platform.system() == "Windows":
            exe_suffix = ''
        args = ['pynx-id01cdi' + exe_suffix, 'data=%s' % path, "nb_raar=50", "nb_hio=50",
                "nb_er=50", "support_update_period=20"]
        # Test sequentially several options. Could also try combined options..
        algo1 = "(Sup*ER**10)**2,psf=5,psf_filter=None, DetwinRAAR**10*Sup*RAAR**50"
        algo2 = "(Sup*ER**10)**2,psf_init=gaussian@1, DetwinRAAR**10*Sup*RAAR**50"
        for options in [[], ["roi=16,112,16,112,16,112"], ["positivity"], ["detwin=0"], ["output_format=npz"],
                        ["support_size=50"], ["support_size=50 support_type=sphere"],
                        ["support_size=50 support_type=cube"], ["support_only_shrink"], ["support_update_border_n=3"],
                        ["rebin=2"], ["rebin=2,1,2"], ["mask_interp=16,2"], ["mask=%s" % mask_path1],
                        ["mask=%s" % mask_path2], ["mask=%s" % mask_path3], ["psf=pseudo-voigt,1,0.05,10"],
                        ["psf=gaussian,1,10", "psf_filter=tukey"],
                        ['algo="%s"' % algo1], ['algo="%s"' % algo2],
                        ["flatfield=%s" % flat_path1], ["flatfield=%s" % flat_path2], ["flatfield=%s" % flat_path3],
                        ["support=%s" % support_path1], ["support=%s" % support_path2], ["support=%s" % support_path3]]:
            option_string = ""
            for s in options:
                option_string += s + " "
            with self.subTest(option=option_string, command=args + options):
                with subprocess.Popen(args + options, stderr=subprocess.PIPE,
                                      stdout=subprocess.PIPE, cwd=self.tmp_dir, env=my_env) as p:
                    stdout, stderr = p.communicate(timeout=200)
                    res = p.returncode
                    msg = "Failed command-line:"
                    for a in p.args:
                        msg += " " + a
                    msg += "\n"
                    self.assertFalse(res, msg=msg + stdout.decode() + stderr.decode())

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipUnless('live_plot' in sys.argv or 'liveplot' in sys.argv, "live plot tests skipped")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_cdi_runner_id01_3d_cxi_liveplot_cuda(self):
        my_env = os.environ.copy()
        my_env["PYNX_PU"] = "cuda"
        path = make_cdi_data_file(shape=(128, 128, 128), file_type='cxi', dir=self.tmp_dir)
        exe_suffix = '.py'
        if platform.system() == "Windows":
            exe_suffix = ''
        args = ['pynx-id01cdi' + exe_suffix, 'data=%s' % path, 'live_plot']
        option_string = ""
        for options in [[]]:
            for s in options:
                option_string += s + " "
            with self.subTest(option=option_string, command=args + options):
                with subprocess.Popen(args + options, stderr=subprocess.PIPE,
                                      stdout=subprocess.PIPE, cwd=self.tmp_dir, env=my_env) as p:
                    stdout, stderr = p.communicate(timeout=200)
                    res = p.returncode
                    self.assertFalse(res, msg=stderr.decode())

    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_cdi_runner_id01_2d_cxi_opencl(self):
        my_env = os.environ.copy()
        my_env["PYNX_PU"] = "opencl"
        path = make_cdi_data_file(shape=(128, 128), file_type='cxi', dir=self.tmp_dir)
        exe_suffix = '.py'
        if platform.system() == "Windows":
            exe_suffix = ''
        args = ['pynx-id01cdi' + exe_suffix, 'data=%s' % path, "nb_raar=50", "nb_hio=50",
                "nb_er=50", "support_update_period=20"]
        # Test sequentially several options. Could also try combined options..
        for options in [[], ["roi=16,112,16,112"], ["positivity"], ["detwin=0"], ["output_format=npz"],
                        ["support_size=50"], ["support_only_shrink"], ["support_update_border_n=3"],
                        ["rebin=2"], ["rebin=2,1"], ["mask_interp=16,2"]]:
            option_string = ""
            for s in options:
                option_string += s + " "
            with self.subTest(option=option_string, command=args + options):
                with subprocess.Popen(args + options, stderr=subprocess.PIPE,
                                      stdout=subprocess.PIPE, cwd=self.tmp_dir, env=my_env) as p:
                    stdout, stderr = p.communicate(timeout=200)
                    res = p.returncode
                    msg = "Failed command-line:"
                    for a in p.args:
                        msg += " " + a
                    msg += "\n"
                    self.assertFalse(res, msg=msg + stderr.decode())

    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    @unittest.skipUnless('live_plot' in sys.argv or 'liveplot' in sys.argv, "live plot tests skipped")
    def test_cdi_runner_id01_2d_cxi_liveplot_opencl(self):
        my_env = os.environ.copy()
        my_env["PYNX_PU"] = "opencl"
        path = make_cdi_data_file(shape=(128, 128), file_type='cxi', dir=self.tmp_dir)
        exe_suffix = '.py'
        if platform.system() == "Windows":
            exe_suffix = ''
        with subprocess.Popen(['pynx-id01cdi' + exe_suffix, 'data=%s' % path, 'live_plot'], stderr=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              cwd=self.tmp_dir, env=my_env) as p:
            stdout, stderr = p.communicate(timeout=200)
            res = p.returncode
            self.assertFalse(res, msg=stderr.decode())

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_cdi_runner_id01_2d_cxi_cuda(self):
        my_env = os.environ.copy()
        my_env["PYNX_PU"] = "cuda"
        path = make_cdi_data_file(shape=(128, 128), file_type='cxi', dir=self.tmp_dir)
        exe_suffix = '.py'
        if platform.system() == "Windows":
            exe_suffix = ''
        args = ['pynx-id01cdi' + exe_suffix, 'data=%s' % path, "nb_raar=50", "nb_hio=50",
                "nb_er=50", "support_update_period=20"]
        # Test support, with different sizes
        support_path1 = make_cdi_support_file(shape=(128, 128), dir=self.tmp_dir)
        support_path2 = make_cdi_support_file(shape=(50, 50), dir=self.tmp_dir)
        support_path3 = make_cdi_support_file(shape=(130, 130), dir=self.tmp_dir)
        # Test sequentially several options. Could also try combined options..
        for options in [[], ["roi=16,112,16,112"], ["positivity"], ["detwin=0"], ["output_format=npz"],
                        ["support_size=50"], ["support_size=50 support_type=circle"],
                        ["support_size=50 support_type=square"], ["support_only_shrink"], ["support_update_border_n=3"],
                        ["rebin=2"], ["rebin=2,1"], ["nb_run=3", "nb_run_keep=1"], ["mask_interp=16,2"],
                        ["support=%s" % support_path1], ["support=%s" % support_path2], ["support=%s" % support_path3]]:
            option_string = ""
            for s in options:
                option_string += s + " "
            with self.subTest(option=option_string, command=args + options):
                with subprocess.Popen(args + options, stderr=subprocess.PIPE,
                                      stdout=subprocess.PIPE, cwd=self.tmp_dir, env=my_env) as p:
                    stdout, stderr = p.communicate(timeout=200)
                    res = p.returncode
                    msg = "Failed command-line:"
                    for a in p.args:
                        msg += " " + a
                    msg += "\n"
                    self.assertFalse(res, msg=msg + stderr.decode())

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipUnless('live_plot' in sys.argv or 'liveplot' in sys.argv, "live plot tests skipped")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_cdi_runner_id01_2d_cxi_liveplot_cuda(self):
        my_env = os.environ.copy()
        my_env["PYNX_PU"] = "cuda"
        path = make_cdi_data_file(shape=(128, 128), file_type='cxi', dir=self.tmp_dir)
        exe_suffix = '.py'
        if platform.system() == "Windows":
            exe_suffix = ''
        with subprocess.Popen(['pynx-id01cdi' + exe_suffix, 'data=%s' % path, 'live_plot'], stderr=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              cwd=self.tmp_dir, env=my_env) as p:
            stdout, stderr = p.communicate(timeout=200)
            res = p.returncode
            self.assertFalse(res, msg=stderr.decode())

    @unittest.skipIf(has_mpi is False, 'no MPI support')
    def test_cdi_runner_id01_2d_cxi_mpi_run(self):
        my_env = os.environ.copy()
        path = make_cdi_data_file(shape=(128, 128), file_type='cxi', dir=self.tmp_dir)
        exe_suffix = '.py'
        if platform.system() == "Windows":
            exe_suffix = ''
        args = ['timeout', '60', 'mpiexec', '-n', '2', 'pynx-id01cdi' + exe_suffix, 'data=%s' % path, "nb_raar=50",
                "nb_hio=50", "nb_er=50", "support_update_period=20", "mpi=run",
                "nb_run=4", "nb_run_keep=2"]
        # Test several options. Could also try combined options..
        for options in [[]]:
            with subprocess.Popen(args + options, stderr=subprocess.PIPE,
                                  stdout=subprocess.PIPE, cwd=self.tmp_dir, env=my_env) as p:
                stdout, stderr = p.communicate(timeout=200)
                res = p.returncode
                msg = "Failed command-line:"
                for a in p.args:
                    msg += " " + a
                msg += "\n"
                self.assertFalse(res, msg=msg + stderr.decode())

    @unittest.skipIf(has_mpi is False, 'no MPI support')
    def test_cdi_runner_id01_2d_cxi_mpi_scan(self):
        my_env = os.environ.copy()
        path = make_cdi_data_file(shape=(128, 128), file_type='cxi', dir=self.tmp_dir)
        for i in range(2):
            os.system("ln -sf %s %s/scan%02d.cxi" % (path, self.tmp_dir, i))
        path = self.tmp_dir + "/scan%02d.cxi"
        exe_suffix = '.py'
        if platform.system() == "Windows":
            exe_suffix = ''
        args = ['timeout', '60', 'mpiexec', '-n', '2', 'pynx-id01cdi' + exe_suffix, 'data=' + path, "nb_raar=50",
                "nb_hio=50", "nb_er=50", "support_update_period=20", "mpi=scan", "scan=0,1"]
        # Test several options. Could also try combined options..
        for options in [[]]:
            with subprocess.Popen(args + options, stderr=subprocess.PIPE,
                                  stdout=subprocess.PIPE, cwd=self.tmp_dir, env=my_env) as p:
                stdout, stderr = p.communicate(timeout=200)
                res = p.returncode
                msg = "Failed command-line:"
                for a in p.args:
                    msg += " " + a
                msg += "\n"
                self.assertFalse(res, msg=msg + stderr.decode())


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite([loadTests(TestCDIRunner)])
    return test_suite


if __name__ == '__main__':
    res = unittest.TextTestRunner(verbosity=2, descriptions=False).run(suite())
