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


class ExpDataTest:
    """
    Class to run tests on experimental data and compare the results
    with a previous reference result.
    """

    def __init__(self, data_url, result_ref_url, verbose=True):
        """

        :param data_url: the url or path to the dataset to be tested. If it is an archive (.zip,
            .tar, .tar.gz or .tar.bz2), it will be uncompressed first.
        :param result_ref_url: the url or path to the result (CXI) file which will be compared to
            the obtained file. The command used to perform the analysis will be extracted
            from /entry_1/process_1/command. The command should work if the data is downloaded
            (and optionally extracted) in the working directory.
        """
        self.data_url = urlparse(data_url)
        self.result_ref_url = urlparse(result_ref_url)
        self.data = None
        self.result_ref = None
        self.command = None
        # In case there are two steps to perform - e.g. CDI reconstruction + modes analysis
        self.command2 = None
        self.verbose = verbose

    def get_data(self):
        """
        Get the data, reference results and command from the url
        :return: nothing
        """
        if 'http' in self.data_url.scheme or 'ftp' in self.data_url.scheme:
            self.data = os.path.split(self.data_url.path)[-1]
            if self.verbose:
                print("Getting data: ", self.data)
            if not os.path.exists(self.data):
                os.system('curl -O %s' % self.data_url.geturl())
        else:
            if self.verbose:
                print("Data: ", self.data)
            self.data = self.data_url.geturl()

        # Uncompress if archive
        s0, s1 = os.path.splitext(self.data)
        if s1 == ".zip":
            os.system("unzip -n -qq %s" % self.data)
        elif s1 == ".tar":
            os.system("tar -xf %s" % self.data)
        elif s1 == ".gz" and os.path.splitext(s0)[-1] == ".tar":
            os.system("tar -xzf %s" % self.data)
        elif s1 == ".bz2" and os.path.splitext(s0)[-1] == ".tar":
            os.system("tar -xjf %s" % self.data)

        if 'http' in self.result_ref_url.scheme or 'ftp' in self.result_ref_url.scheme:
            if self.verbose:
                print("Getting result: ", self.result_ref)
            self.result_ref = os.path.split(self.result_ref_url.path)[-1]
            if not os.path.exists(self.result_ref):
                os.system('curl -O %s' % self.result_ref_url.geturl())
        else:
            if self.verbose:
                print("Result file: ", self.data)
            self.result_ref = self.result_ref_url.geturl()

        # Get command from cxi file
        with h5py.File(self.result_ref) as h:
            if '/entry_last/process_1/command' in h:
                self.command = h['/entry_last/process_1/command'][()].decode()
            elif '/entry_1/process_1/command' in h:
                self.command = h['/entry_1/process_1/command'][()].decode()
            elif '/entry_last/image_1/process_1/command' in h:
                self.command = h['/entry_last/image_1/process_1/command'][()].decode()
            elif '/entry_1/image_1/process_1/command' in h:
                self.command = h['/entry_1/image_1/process_1/command'][()].decode()

            # Is there another command to run first ?
            print('/entry_last/image_1/process_2/command' in h, '/entry_1/image_1/process_2/command' in h)
            if '/entry_last/image_1/process_2/command' in h:
                self.command2 = h['/entry_last/image_1/process_2/command'][()].decode()
            elif '/entry_1/image_1/process_2/command' in h:
                self.command2 = h['/entry_1/image_1/process_2/command'][()].decode()

        # Remove prefix from command
        # We assume there is no space in the executable name
        if self.command2 is not None:
            c = self.command2.split()[0]
            self.command2 = self.command2.replace(c, os.path.split(c)[-1])
            if self.verbose:
                print("Command to run first: ", self.command2)
        c = self.command.split()[0]
        self.command = self.command.replace(c, os.path.split(c)[-1])

        # Special case for CDI modes analysis - we need to replace the names
        # the produced CXI files by a wildcard
        if self.command2 is not None and 'pynx-cdi-analysis' in self.command:
            s = ''
            for v in self.command.split():
                if os.path.splitext(v)[-1] == '.cxi':
                    if "*.cxi" not in s:
                        s += '*.cxi '
                else:
                    s += v + ' '
        if self.verbose:
            print("Command to run: ", self.command)


class ExpDataTestPtycho(ExpDataTest):
    """
    Class to run tests on experimental ptychography data and compare the results
    with a previous reference result.
    """

    def run(self):
        """
        Run the tests
        """
        self.get_data()
        c = self.command.split() + ['saveprefix=result_scan%02d_run%02d']
        with subprocess.Popen(c, stderr=subprocess.PIPE, stdout=subprocess.PIPE) as p:
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
        if self.verbose:
            print("LLK comparison: ref=%10.3f res=%10.3f" % (ref, res))

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
        if self.verbose:
            print("Matching objects: R_match=%5.3f < 0.3" % r)
        assert r < 0.3, "The objects do not match (R=%5.3f > 0.3)" % r

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
        if self.verbose:
            print("Matching probes: R_match=%5.3f < 0.3" % r)
        assert r < 0.3, "The probes do not match (R=%5.3f > 0.3)" % r
        if ref.shape[0] > 1:
            w1 = ortho_modes(ref, return_weights=True)[-1]
            w2 = ortho_modes(res, return_weights=True)[-1]
            if self.verbose:
                print("Matching probe modes ratios: ref=", w1, " res=", w2)
            assert np.allclose(w1, w2, atol=0.05, rtol=0.05)


class ExpDataTestCDI(ExpDataTest):
    """
    Class to run tests on experimental CDI data and compare the results
    with a previous reference result.
    """

    def run(self):
        """
        Run the tests
        """
        self.get_data()
        c = self.command.split() + ['saveprefix=result_scan%02d_run%02d']
        with subprocess.Popen(c, stderr=subprocess.PIPE, stdout=subprocess.PIPE) as p:
            stdout, stderr = p.communicate(timeout=600)
            res = p.returncode
            assert not res, stderr.decode()
        self.compare_llk()
        self.compare_obj()

    def compare_llk(self):
        """
        Compare the results log-likelihood
        :return: nothing. An exception will be raised if the results don't match
        """
        print(self.result_ref)
        ref = h5py.File(self.result_ref, 'r')['entry_last/image_1/process_1/results/llk_poisson'][()]
        res = h5py.File('latest.cxi', 'r')['entry_last/image_1/process_1/results/llk_poisson'][()]
        if self.verbose:
            print("LLK comparison: ref=%10.3f res=%10.3f" % (ref, res))

        assert np.isclose(ref, res, rtol=0.1)

    def compare_obj(self):
        """
        Compare the object with the reference result
        :return: nothing. An exception will be raised if the results don't match
        """
        ref_nsup = h5py.File(self.result_ref, 'r')['/entry_last/image_1/process_1/results/nb_point_support'][()]
        res_nsup = h5py.File('latest.cxi', 'r')['/entry_last/image_1/process_1/results/nb_point_support'][()]
        ref = h5py.File(self.result_ref, 'r')['entry_last/image_1/data'][()]
        res = h5py.File('latest.cxi', 'r')['entry_last/image_1/data'][()]
        assert np.isclose(ref_nsup, res_nsup, rtol=0.1), "The objects size do not match"
        r = match2(ref, res, match_phase_ramp=False, match_orientation=True, match_scale=False)[-1]
        if self.verbose:
            print("Matching objects: R_match=%5.3f < 0.3" % r)
        assert r < 0.3, "The objects do not match (R=%5.3f > 0.3)" % r


class TestRunnerDatasets(unittest.TestCase):

    def test_ptycho_datasets(self):
        refs = [('http://ftp.esrf.fr/pub/scisoft/PyNX/data/201606id01-FZP-S013.cxi',
                 'http://ftp.esrf.fr/pub/scisoft/PyNX/data/201606id01-FZP-S013-result.cxi')]
        for r in refs:
            with self.subTest("Ptycho dataset:", data_url=r[0], result_ref_url=r[1]):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    os.chdir(tmp_dir)
                    p = ExpDataTestPtycho(data_url=r[0], result_ref_url=r[1], verbose=True)
                    p.run()

    def test_cdi_datasets(self):
        refs = [('http://ftp.esrf.fr/pub/scisoft/PyNX/data/CDI-id01-201702-Pt-raw-data.tar.bz2',
                 'http://ftp.esrf.fr/pub/scisoft/PyNX/data/CDI-id01-201702-Pt-raw-result.cxi'),
                # TODO: use modes analysis for comparison
                # ('http://ftp.esrf.fr/pub/scisoft/PyNX/data/CDI-id01-201702-Pt-data.cxi',
                #  'http://ftp.esrf.fr/pub/scisoft/PyNX/data/CDI-id01-201702-Pt-result-modes.h5')
                ]
        for r in refs:
            with self.subTest("CDI dataset:", data_url=r[0], result_ref_url=r[1]):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    os.chdir(tmp_dir)
                    p = ExpDataTestCDI(data_url=r[0], result_ref_url=r[1], verbose=True)
                    p.run()


def suite():
    test_suite = unittest.TestSuite()
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(load_tests(TestRunnerDatasets))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
