# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2021-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

"""
This package includes unit tests using experimental datasets, comparing the final results
with standard ones.
"""
import unittest

import os
import sys
import subprocess
import unittest
import tempfile
import shutil
from urllib.parse import urlparse
import numpy as np
from pynx.processing_unit import has_opencl, has_cuda
from pynx.cdi.selection import match2
from pynx.utils.math import ortho_modes

from pynx.utils import h5py

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


class PtychoTestData():
    """
    Class to run tests & compare the result with a previous reference result.
    """

    def __init__(self, data_url, result_ref_url, command_url):
        """

        :param data_url: the url or path to the dataset to be tested
        :param result_ref_url: the url or path to the result (CXI) file which will be compared to
            the obtained file.
        :param command_url: the url or path to a text file with the *single* command-line
            which should be run to produce the result.
            The 'saveprefix' will be added automatically so need not be included.
            If '{data_name}' is included in the command, it will be replaced by the name
            of the data file
        """
        self.data_url = urlparse(data_url)
        self.result_ref_url = urlparse(result_ref_url)
        self.command_url = urlparse(command_url)
        self.data = None
        self.result_ref = None
        self.command = None

    def get_data(self):
        """
        Get the data, reference results and command from the url
        :return: nothing
        """
        if 'http' in self.data_url.scheme:
            os.system('curl -O %s' % self.data_url.geturl())
            self.data = os.path.split(self.data_url.path)[-1]
        else:
            self.data = self.data_url.geturl()

        if 'http' in self.result_ref_url.scheme:
            os.system('curl -O %s' % self.result_ref_url.geturl())
            self.result_ref = os.path.split(self.result_ref_url.path)[-1]
        else:
            self.result_ref = self.result_ref_url.geturl()

        if 'http' in self.command_url.scheme:
            os.system('curl -O %s' % self.command_url.geturl())
            self.command = open(os.path.split(self.result_ref_url.path)[-1]).readline()
        else:
            with open(self.command_url.geturl()) as tmp:
                self.command = tmp.read()

        if '{data_name}' in self.command:
            self.command = self.command.format(data_name=self.data)

    def run(self):
        """
        Run the tests
        """
        self.get_data()
        with subprocess.Popen(self.command.split() + ['saveprefix=result_scan%02d_run%02d'],
                              stderr=subprocess.PIPE, stdout=subprocess.PIPE) as p:
            stdout, stderr = p.communicate(timeout=600)
            res = p.returncode
            assert not res, stderr.decode()
        self.compare_llk()
        self.compare_obj()
        self.compare_probe()

    def compare_llk(self):
        """
        Compare the results log-likelihood
        :return: nothing. An exception will be raised if the results don't match
        """
        print(self.result_ref)
        ref = h5py.File(self.result_ref, 'r')['entry_last/process_1/results/llk_poisson'][()]
        res = h5py.File('latest.cxi', 'r')['entry_last/process_1/results/llk_poisson'][()]
        assert np.isclose(ref, res, rtol=0.1)

    def compare_obj(self):
        """
        Compare the object with the reference result
        :return: nothing. An exception will be raised if the results don't match
        """
        ref = h5py.File(self.result_ref, 'r')['entry_last/object/data'][()]
        ref_illum = h5py.File(self.result_ref, 'r')['entry_last/object/obj_illumination'][()]
        res = h5py.File('latest.cxi', 'r')['entry_last/object/data'][()]
        res_illum = h5py.File('latest.cxi', 'r')['entry_last/object/obj_illumination'][()]
        assert res.ndim == ref.ndim, "The objects dimensions do not match"
        assert np.isclose(res.size, ref.size, rtol=0.01), "The objects size do not match"
        r = match2(ref[0] * ref_illum, res[0] * res_illum, match_phase_ramp=True,
                   match_orientation=False, match_scale=True)[-1]
        assert r < 0.3, "The objects do not match (R=%5.3f > 0.15)" % r

    def compare_probe(self):
        """
        Compare the object with the reference result
        :return: nothing. An exception will be raised if the results don't match
        """
        ref = h5py.File(self.result_ref, 'r')['entry_last/probe/data'][()]
        res = h5py.File('latest.cxi', 'r')['entry_last/probe/data'][()]
        assert res.ndim == ref.ndim, "The probe dimensions do not match"
        assert np.isclose(res.size, ref.size, rtol=0.01), "The probe sizes do not match"
        r = match2(ref[0], res[0], match_phase_ramp=True, match_orientation=False, match_scale=True)[-1]
        assert r < 0.3, "The probes do not match (R=%5.3f > 0.15)" % r
        if ref.shape[0] > 1:
            w1 = ortho_modes(ref, return_weights=True)[-1]
            w2 = ortho_modes(res, return_weights=True)[-1]
            assert np.allclose(w1, w2, atol=0.03, rtol=0.03)


class TestCDIRunnerDatasets(unittest.TestCase):

    def test_ptycho_datasets(self):
        refs = [('/Users/vincent/data/201606id01-FZP-S0013.cxi', '/Users/vincent/data/ResultsScan0000/Run0060.cxi',
                 '/Users/vincent/data/ResultsScan0000/command')]
        for r in refs:
            with self.subTest("Ptycho dataset:", data_url=r[0], result_ref_url=r[1], command_url=r[2]):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    os.chdir(tmp_dir)

                    p = PtychoTestData(data_url=r[0], result_ref_url=r[1], command_url=r[2])
                    p.run()


if __name__ == '__main__':
    unittest.main()
