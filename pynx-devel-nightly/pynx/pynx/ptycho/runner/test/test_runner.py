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
import platform
import sys
import subprocess
import unittest
import tempfile
import shutil
import warnings

has_mpi = False
try:
    from mpi4py import MPI
    import shutil

    if shutil.which('mpiexec') is not None:
        has_mpi = True
except ImportError:
    pass

from pynx.ptycho.test.test_ptycho import make_ptycho_data, make_ptycho_data_cxi
from pynx.processing_unit import has_cuda, has_opencl

exclude_cuda = False
exclude_opencl = False
if 'PYNX_PU' in os.environ:
    if 'opencl' in os.environ['PYNX_PU'].lower():
        exclude_cuda = True
    elif 'cuda' in os.environ['PYNX_PU'].lower():
        exclude_opencl = True

if 'cuda' in sys.argv or not has_opencl:
        exclude_opencl = True
if 'opencl' in sys.argv or not has_cuda:
        exclude_cuda = True


class TestPtychoRunner(unittest.TestCase):
    """
    Class for tests of the Ptycho runner scripts
    """

    @classmethod
    def setUpClass(cls):
        # Directory contents will automatically get cleaned up on deletion
        cls.tmp_dir_obj = tempfile.TemporaryDirectory()
        cls.tmp_dir = cls.tmp_dir_obj.name

    def test_ptycho_runner(self):
        path = make_ptycho_data_cxi(dsize=160, nb_frame=40, nb_photons=1e9, dir_name=self.tmp_dir)
        exe_suffix = '.py'
        if platform.system() == "Windows":
            exe_suffix = ''
        options = {'cxi': ['pynx-cxipty' + exe_suffix, 'data=%s' % path, 'probe=gaussian,400e-9x400e-9',
                           'algorithm=ML**20,DM**20,nbprobe=2,DM**20,nbprobe=1,probe=1',
                           'verbose=10', 'save=all', 'saveplot'],
                   'cxi_mask_flat': ['pynx-cxipty' + exe_suffix, 'data=%s' % path, 'probe=gaussian,400e-9x400e-9',
                                     'algorithm=ML**20,DM**20,nbprobe=2,DM**20,nbprobe=1,probe=1',
                                     'verbose=10', 'save=all', 'saveplot', 'mask=mask.npz', 'flatfield=flatfield.npz'],
                   'cxi_mpi': ['timeout', '120', 'mpiexec', '-n', '2', 'pynx-cxipty' + exe_suffix,
                               'data=%s' % path, 'probe=gaussian,400e-9x400e-9', 'mpi=split',
                               'algorithm=analysis,ML**20,DM**20,nbprobe=2,DM**20,nbprobe=1,probe=1',
                               'verbose=10', 'save=all', 'saveplot'],
                   'simulation': ['pynx-simulationpty' + exe_suffix, 'frame_nb=64', 'frame_size=128',
                                  'algorithm=analysis,ML**20,AP**20,DM**20,background=1,DM**20,nbprobe=1,probe=1',
                                  'verbose=10', 'save=all', 'saveplot', 'simul_background=0.05']}
        for k, v in options.items():
            langs = []
            if not exclude_cuda:
                langs += ['cuda']
            if not exclude_opencl:
                langs += ['opencl']
            for lang in langs:
                opt_liveplot = ['']
                if 'live_plot' in sys.argv or 'liveplot' in sys.argv:
                    opt_liveplot += ['liveplot']
                for opt_live in opt_liveplot:
                    if 'mpi' in k and (opt_live == 'liveplot' or not has_mpi):
                        continue
                    # print(k, lang, opt_live)
                    with self.subTest(k, config="%s %s" % (lang, opt_live), command=v + [opt_live]):
                        my_env = os.environ.copy()
                        my_env["PYNX_PU"] = lang
                        with subprocess.Popen(v + [opt_live], stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                                              cwd=self.tmp_dir, env=my_env) as p:
                            stdout, stderr = p.communicate(timeout=200)
                            res = p.returncode
                            self.assertFalse(res, msg=stderr.decode())


def suite():
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite([load_tests(TestPtychoRunner)])
    return test_suite


if __name__ == '__main__':
    # sys.stdout = io.StringIO()
    warnings.simplefilter('ignore')
    res = unittest.TextTestRunner(verbosity=2, descriptions=False).run(suite())
