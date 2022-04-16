#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

"""
This file includes tests for the Ptycho python API.
"""

import os
import unittest
import tempfile
import shutil
import sys
import io
import warnings
import gc
import numpy as np
from pynx.processing_unit import has_cuda, has_opencl
from pynx.ptycho import simulation, shape
from pynx.ptycho import save_ptycho_data_cxi
from pynx.ptycho import *

if has_cuda:
    import pynx.ptycho.cu_operator as cuop
if has_opencl:
    import pynx.ptycho.cl_operator as clop
    from pyopencl import CompilerWarning
    import warnings

    warnings.simplefilter('ignore', CompilerWarning)

import pynx.ptycho.cpu_operator as cpuop

exclude_cuda = False
exclude_opencl = False
if 'PYNX_PU' in os.environ:
    if 'opencl' in os.environ['PYNX_PU'].lower():
        exclude_cuda = True
    elif 'cuda' in os.environ['PYNX_PU'].lower():
        exclude_opencl = True


def make_ptycho_data(dsize=256, nb_frame=100, nb_photons=1e9):
    pixel_size_detector = 55e-6
    wavelength = 1.5e-10
    detector_distance = 1
    obj_info = {'type': 'phase_ampl', 'phase_stretch': np.pi / 2, 'alpha_win': .2}
    probe_info = {'type': 'focus', 'aperture': (200e-6, 200e-6), 'focal_length': .08, 'defocus': 450e-6,
                  'shape': (dsize, dsize)}

    # 50 scan positions correspond to 4 turns, 78 to 5 turns, 113 to 6 turns
    scan_info = {'type': 'spiral', 'scan_step_pix': 30, 'n_scans': nb_frame}
    data_info = {'num_phot_max': nb_photons, 'bg': 0, 'wavelength': wavelength, 'detector_distance': detector_distance,
                 'detector_pixel_size': pixel_size_detector, 'noise': 'poisson'}

    # Initialisation of the simulation with specified parameters
    s = simulation.Simulation(obj_info=obj_info, probe_info=probe_info, scan_info=scan_info, data_info=data_info,
                              verbose=False)
    s.make_data()

    # Positions from simulation are given in pixels
    x, y = s.scan.values
    px = wavelength * detector_distance / pixel_size_detector / dsize

    iobs = s.amplitude.values ** 2

    return iobs, pixel_size_detector, wavelength, detector_distance, x * px, y * px


def make_ptycho_data_cxi(dsize=256, nb_frame=100, nb_photons=1e9, dir_name=None):
    iobs, pixel_size_detector, wavelength, detector_distance, x, y = make_ptycho_data(dsize, nb_frame, nb_photons)
    f, path = tempfile.mkstemp(suffix='.cxi', prefix="TestPtycho", dir=dir_name)
    save_ptycho_data_cxi(path, iobs, pixel_size_detector, wavelength, detector_distance, x, y, z=None,
                         monitor=None, mask=None, instrument="Simulation", overwrite=True)
    # Also make a mask and flatfield
    mask = np.random.uniform(0, 1, (dsize, dsize)) < 0.05
    flatfield = np.random.uniform(0.98, 1.02, (dsize, dsize))
    np.savez_compressed(os.path.join(dir_name, 'mask.npz'), mask=mask)
    np.savez_compressed(os.path.join(dir_name, 'flatfield.npz'), flatfield=flatfield)
    return path


class TestPtycho(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Directory contents will automatically get cleaned up on deletion
        cls.tmp_dir_obj = tempfile.TemporaryDirectory()
        cls.tmp_dir = cls.tmp_dir_obj.name
        # Ptycho data and main object, created in make_ptycho_data and make_ptycho_obj
        cls.data = None
        cls.p = None
        cls.obj0 = None
        cls.illum_mask = None  # Object illumination
        cls.probe0 = None
        cls.ops = []
        if 'cuda' not in sys.argv and 'opencl' not in sys.argv:
            cls.ops.append(cpuop)
        if has_cuda and 'opencl' not in sys.argv:
            cls.ops.append(cuop)
        if has_opencl and 'cuda' not in sys.argv:
            cls.ops.append(clop)

    @classmethod
    def tearDownClass(cls):
        if cls.data is not None:
            del cls.data
        if cls.p is not None:
            del cls.p
        gc.collect()

    def make_ptycho_data(self):
        """
        Make ptycho data, if it does not already exist.
        :return: Nothing
        """
        if self.data is None:
            iobs, pixel_size_detector, wavelength, detector_distance, x, y = make_ptycho_data(128, 35, 1e9)
            self.data = PtychoData(iobs, positions=(x, y), detector_distance=detector_distance,
                                   pixel_size_detector=pixel_size_detector, wavelength=wavelength)

    def make_ptycho_obj(self):
        """
        Make ptycho obj, if it does not already exist.
        :return: Nothing
        """
        if self.p is None:
            self.make_ptycho_data()
            n = self.data.iobs.shape[-1]
            px = self.data.pixel_size_object()[0]
            pxd = self.data.pixel_size_detector
            nyo, nxo = shape.calc_obj_shape(self.data.posx / px, self.data.posy / px, self.data.iobs.shape[1:])

            obj_init_info = {'type': 'random', 'range': (0, 1, 0, 0.5), 'shape': (nyo, nxo)}
            probe_init_info = {'type': 'focus', 'aperture': (230e-6, 230e-6), 'focal_length': .08, 'defocus': 460e-6,
                               'shape': (n, n)}
            data_info = {'wavelength': self.data.wavelength, 'detector_distance': self.data.detector_distance,
                         'detector_pixel_size': pxd}
            init = simulation.Simulation(obj_info=obj_init_info, probe_info=probe_init_info, data_info=data_info,
                                         verbose=False)
            init.make_obj()
            init.make_probe()

            self.p = Ptycho(probe=init.probe.values, obj=init.obj.values, data=self.data, background=None)
            self.p = ScaleObjProbe(verbose=False) * self.p

            # 100 cycles of DM to get a minimum of convergence
            self.p = FreePU() * DM(update_object=True, update_probe=True, calc_llk=0) ** 100 * self.p
            # Keep these object and probe as a reference
            self.obj0 = self.p.get_obj()
            self.probe0 = self.p.get_probe()
            illum = self.p.get_illumination_obj()
            self.illum_mask = illum > (illum.max() * 0.1)

    def test_00_make_ptycho_data(self):
        self.make_ptycho_data()

    def test_01_make_ptycho_obj(self):
        self.make_ptycho_obj()

    def test_make_ptycho_cxi(self):
        path = make_ptycho_data_cxi(dir_name=self.tmp_dir)

    def test_cxi_reload(self):
        self.make_ptycho_obj()
        o = self.p.get_obj()
        pr = self.p.get_probe()
        f, path = tempfile.mkstemp(suffix='.cxi', prefix="TestPtycho", dir=self.tmp_dir)
        self.p.save_obj_probe_cxi(path)
        self.p.load_obj_probe_cxi(path, verbose=False)
        self.o = o
        self.pr = pr
        self.assertTrue(np.allclose(pr, self.p.get_probe()))
        self.assertTrue(np.allclose(o, self.p.get_obj()))

    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_AP_opencl(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        # Compute LLK before and after
        self.p = clop.LoopStack(clop.PropagateApplyAmplitude(calc_llk=True) * clop.ObjProbe2Psi()) * self.p
        llk0 = self.p.llk_poisson / self.p.nb_obs
        self.p = clop.AP(update_object=True, update_probe=False, calc_llk=0) ** 10 * self.p
        self.p = clop.AP(update_object=True, update_probe=True, calc_llk=0) ** 10 * self.p
        self.p = clop.AP(update_object=True, update_probe=True, update_background=True, calc_llk=0) ** 10 * self.p
        self.p = clop.LoopStack(clop.PropagateApplyAmplitude(calc_llk=True) * clop.ObjProbe2Psi()) * self.p
        llk1 = self.p.llk_poisson / self.p.nb_obs
        # print("AP**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
        self.p = clop.FreePU() * self.p

    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_DM_opencl(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        # Compute LLK before and after
        self.p = clop.LoopStack(clop.PropagateApplyAmplitude(calc_llk=True) * clop.ObjProbe2Psi()) * self.p
        llk0 = self.p.llk_poisson / self.p.nb_obs
        self.p = clop.DM(update_object=True, update_probe=False, calc_llk=0) ** 10 * self.p
        self.p = clop.DM(update_object=True, update_probe=True, calc_llk=0) ** 10 * self.p
        self.p = clop.DM(update_object=True, update_probe=True, update_background=True, calc_llk=0) ** 10 * self.p
        self.p = clop.LoopStack(clop.PropagateApplyAmplitude(calc_llk=True) * clop.ObjProbe2Psi()) * self.p
        llk1 = self.p.llk_poisson / self.p.nb_obs
        # print("DM**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
        self.p = clop.FreePU() * self.p

    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_ML_opencl(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        # Compute LLK before and after
        self.p = clop.LoopStack(clop.PropagateApplyAmplitude(calc_llk=True) * clop.ObjProbe2Psi()) * self.p
        llk0 = self.p.llk_poisson / self.p.nb_obs
        self.p = clop.ML(update_object=True, update_probe=False, calc_llk=0) ** 10 * self.p
        self.p = clop.ML(update_object=True, update_probe=True, calc_llk=0) ** 10 * self.p
        self.p = clop.ML(update_object=True, update_probe=True, update_background=True, calc_llk=0) ** 10 * self.p
        self.p = clop.LoopStack(clop.PropagateApplyAmplitude(calc_llk=True) * clop.ObjProbe2Psi()) * self.p
        llk1 = self.p.llk_poisson / self.p.nb_obs
        # print("ML**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
        self.p = clop.FreePU() * self.p

    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_ML_regul_opencl(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        # Compute LLK before and after
        self.p = clop.LoopStack(clop.PropagateApplyAmplitude(calc_llk=True) * clop.ObjProbe2Psi()) * self.p
        llk0 = self.p.llk_poisson / self.p.nb_obs
        self.p = clop.ML(update_object=True, update_probe=True, calc_llk=0, reg_fac_probe=0.1) ** 10 * self.p
        self.p = clop.ML(update_object=True, update_probe=True, calc_llk=0, reg_fac_obj=0.1) ** 10 * self.p
        self.p = clop.ML(update_object=True, update_probe=True, calc_llk=0,
                         reg_fac_obj=0.1, reg_fac_probe=0.1) ** 10 * self.p
        self.p = clop.LoopStack(clop.PropagateApplyAmplitude(calc_llk=True) * clop.ObjProbe2Psi()) * self.p
        llk1 = self.p.llk_poisson / self.p.nb_obs
        # print("ML**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
        self.p = clop.FreePU() * self.p

    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_background_AP_ML_opencl(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        b0 = self.p.get_background().copy()
        # Compute LLK before and after
        self.p = clop.LoopStack(clop.PropagateApplyAmplitude(calc_llk=True) * clop.ObjProbe2Psi()) * self.p
        llk0 = self.p.llk_poisson / self.p.nb_obs
        self.p = clop.AP(update_object=True, update_probe=True, update_background=True, calc_llk=0) ** 10 * self.p
        self.p = clop.ML(update_object=True, update_probe=True, update_background=True, calc_llk=0) ** 10 * self.p
        self.p = clop.LoopStack(clop.PropagateApplyAmplitude(calc_llk=True) * clop.ObjProbe2Psi()) * self.p
        llk1 = self.p.llk_poisson / self.p.nb_obs
        # print("ML**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
        self.p = clop.FreePU() * self.p
        self.p.set_background(b0)

    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_update_positions_opencl(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = clop.FreePU() * self.p
        posx0 = self.p.data.posx.copy()
        posy0 = self.p.data.posy.copy()
        n = len(posx0)
        for i in np.random.randint(0, n, 10):
            self.p.data.posx[i] += np.random.uniform(-5, 5) * self.p.data.pixel_size_object()[0]
            self.p.data.posy[i] += np.random.uniform(-5, 5) * self.p.data.pixel_size_object()[0]
        # dx0 = (self.p.data.posx - posx0) / self.p.data.pixel_size_object()[0]
        # dy0 = (self.p.data.posy - posy0) / self.p.data.pixel_size_object()[1]
        self.p._interpolation = True
        self.p = clop.AP(update_object=True, update_probe=True, calc_llk=0) ** 2 * self.p
        self.p = clop.AP(update_object=True, update_probe=True, update_pos=True, calc_llk=0) ** 100 * self.p
        self.p = clop.ML(update_object=True, update_probe=True, update_pos=True, calc_llk=0) ** 100 * self.p
        self.p._interpolation = False
        self.p = clop.FreePU() * self.p
        # TODO: Compare calculated and real shifts, test optimisation worked
        # dx = (self.p.data.posx - posx0) / self.p.data.pixel_size_object()[0]
        # dy = (self.p.data.posy - posy0) / self.p.data.pixel_size_object()[1]
        # print("dr: %6.2f -> %6.2f" % (np.sqrt(dx0 ** 2 + dy0 ** 2).mean(), np.sqrt(dx ** 2 + dy ** 2).mean()))
        # PlotPositions() * self.p
        # Put back original positions
        self.p.data.posx = posx0
        self.p.data.posy = posy0

    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_fourier_scale_opencl(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = clop.ObjProbe2Psi() * self.p
        psi0 = self.p._cl_psi.get()
        self.p = clop.FT(scale=False) * self.p
        s = clop.default_processing_unit.fft_scale(self.p._cl_psi.shape, ndim=2)
        psi1 = self.p._cl_psi.get() * s[0]
        self.p = clop.IFT(scale=False) * self.p
        psi2 = self.p._cl_psi.get() * s[0] * s[1]
        self.p = clop.FreePU() * self.p
        # Check that L2 norms have the expected behaviour
        npsi0 = (np.abs(psi0) ** 2).sum()
        npsi1 = (np.abs(psi1) ** 2).sum()
        npsi2 = (np.abs(psi2) ** 2).sum()
        message = "Checking FT scale: %7.3f %7.3f" % (npsi1 / npsi0, npsi2 / npsi1)
        self.assertAlmostEqual(npsi1 / npsi0, 1, msg=message, delta=1e-5)
        message = "Checking IFT scale: %7.3f %7.3f" % (npsi1 / npsi0, npsi2 / npsi1)
        self.assertAlmostEqual(npsi1 / npsi2, 1, msg=message, delta=1e-5)

    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_phase_ramp_opencl(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = clop.ZeroPhaseRamp(obj=True) * clop.ApplyPhaseRamp(0.34, -0.45, obj=False, probe=True) * self.p
        # Tolerance is high due to low resolution during test (128 pixels..)
        tmp = np.allclose([self.p.data.phase_ramp_dx, self.p.data.phase_ramp_dy], [-.34, .45], atol=0.2)
        self.assertTrue(tmp, msg="CUDA: calculated object phase ramp is incorrect")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_AP_cuda(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        # Compute LLK before and after
        self.p = cuop.LoopStack(cuop.PropagateApplyAmplitude(calc_llk=True) * cuop.ObjProbe2Psi()) * self.p
        llk0 = self.p.llk_poisson / self.p.nb_obs
        self.p = cuop.AP(update_object=True, update_probe=False, calc_llk=0) ** 10 * self.p
        self.p = cuop.AP(update_object=True, update_probe=True, calc_llk=0) ** 10 * self.p
        self.p = cuop.LoopStack(cuop.PropagateApplyAmplitude(calc_llk=True) * cuop.ObjProbe2Psi()) * self.p
        llk1 = self.p.llk_poisson / self.p.nb_obs
        # print("AP**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
        self.p = cuop.FreePU() * self.p

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_DM_cuda(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        # Compute LLK before and after
        self.p = cuop.LoopStack(cuop.PropagateApplyAmplitude(calc_llk=True) * cuop.ObjProbe2Psi()) * self.p
        llk0 = self.p.llk_poisson / self.p.nb_obs
        self.p = cuop.DM(update_object=True, update_probe=False, calc_llk=0) ** 10 * self.p
        self.p = cuop.DM(update_object=True, update_probe=True, calc_llk=0) ** 10 * self.p
        self.p = cuop.DM(update_object=True, update_probe=True, update_background=True, calc_llk=0) ** 10 * self.p
        self.p = cuop.LoopStack(cuop.PropagateApplyAmplitude(calc_llk=True) * cuop.ObjProbe2Psi()) * self.p
        llk1 = self.p.llk_poisson / self.p.nb_obs
        # print("DM**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
        self.p = cuop.FreePU() * self.p

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_ML_cuda(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        # Compute LLK before and after
        self.p = cuop.LoopStack(cuop.PropagateApplyAmplitude(calc_llk=True) * cuop.ObjProbe2Psi()) * self.p
        llk0 = self.p.llk_poisson / self.p.nb_obs
        self.p = cuop.ML(update_object=True, update_probe=True, calc_llk=0) ** 10 * self.p
        self.p = cuop.LoopStack(cuop.PropagateApplyAmplitude(calc_llk=True) * cuop.ObjProbe2Psi()) * self.p
        llk1 = self.p.llk_poisson / self.p.nb_obs
        # print("ML**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
        self.p = cuop.FreePU() * self.p

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_ML_regul_cuda(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        # Compute LLK before and after
        self.p = cuop.LoopStack(cuop.PropagateApplyAmplitude(calc_llk=True) * cuop.ObjProbe2Psi()) * self.p
        llk0 = self.p.llk_poisson / self.p.nb_obs
        self.p = cuop.ML(update_object=True, update_probe=True, calc_llk=0, reg_fac_probe=0.1) ** 10 * self.p
        self.p = cuop.ML(update_object=True, update_probe=True, calc_llk=0, reg_fac_obj=0.1) ** 10 * self.p
        self.p = cuop.ML(update_object=True, update_probe=True, calc_llk=0,
                         reg_fac_obj=0.1, reg_fac_probe=0.1) ** 10 * self.p
        self.p = cuop.LoopStack(cuop.PropagateApplyAmplitude(calc_llk=True) * cuop.ObjProbe2Psi()) * self.p
        llk1 = self.p.llk_poisson / self.p.nb_obs
        # print("ML**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
        self.p = cuop.FreePU() * self.p

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_background_AP_ML_cuda(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        b0 = self.p.get_background().copy()
        # Compute LLK before and after
        self.p = cuop.LoopStack(cuop.PropagateApplyAmplitude(calc_llk=True) * cuop.ObjProbe2Psi()) * self.p
        llk0 = self.p.llk_poisson / self.p.nb_obs
        self.p = cuop.AP(update_object=True, update_probe=True, update_background=True, calc_llk=0) ** 10 * self.p
        self.p = cuop.ML(update_object=True, update_probe=True, update_background=True, calc_llk=0) ** 10 * self.p
        self.p = cuop.LoopStack(cuop.PropagateApplyAmplitude(calc_llk=True) * cuop.ObjProbe2Psi()) * self.p
        llk1 = self.p.llk_poisson / self.p.nb_obs
        # print("ML**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
        self.p = cuop.FreePU() * self.p
        self.p.set_background(b0)

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_update_positions_cuda(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = cuop.FreePU() * self.p
        posx0 = self.p.data.posx.copy()
        posy0 = self.p.data.posy.copy()
        n = len(posx0)
        for i in np.random.randint(0, n, 10):
            self.p.data.posx[i] += np.random.uniform(-5, 5) * self.p.data.pixel_size_object()[0]
            self.p.data.posy[i] += np.random.uniform(-5, 5) * self.p.data.pixel_size_object()[0]
        # dx0 = (self.p.data.posx - posx0) / self.p.data.pixel_size_object()[0]
        # dy0 = (self.p.data.posy - posy0) / self.p.data.pixel_size_object()[1]
        self.p._interpolation = True
        self.p = cuop.AP(update_object=True, update_probe=True, calc_llk=0) ** 2 * self.p
        self.p = cuop.AP(update_object=True, update_probe=True, update_pos=True, calc_llk=0) ** 100 * self.p
        self.p = cuop.ML(update_object=True, update_probe=True, update_pos=True, calc_llk=0) ** 100 * self.p
        self.p._interpolation = False
        self.p = cuop.FreePU() * self.p
        # TODO: Compare calculated and real shifts, test optimisation worked
        # dx = (self.p.data.posx - posx0) / self.p.data.pixel_size_object()[0]
        # dy = (self.p.data.posy - posy0) / self.p.data.pixel_size_object()[1]
        # print("dr: %6.2f -> %6.2f" % (np.sqrt(dx0 ** 2 + dy0 ** 2).mean(), np.sqrt(dx ** 2 + dy ** 2).mean()))
        # PlotPositions() * self.p
        # Put back original positions
        self.p.data.posx = posx0
        self.p.data.posy = posy0

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_fourier_scale_cuda(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = cuop.ObjProbe2Psi() * self.p
        psi0 = self.p._cu_psi.get()
        self.p = cuop.FT(scale=False) * self.p
        s = cuop.default_processing_unit.fft_scale(self.p._cu_psi.shape, ndim=2)
        psi1 = self.p._cu_psi.get() * s[0]
        self.p = cuop.IFT(scale=False) * self.p
        psi2 = self.p._cu_psi.get() * s[0] * s[1]
        self.p = cuop.FreePU() * self.p
        # Check that L2 norms have the expected behaviour
        npsi0 = (np.abs(psi0) ** 2).sum()
        npsi1 = (np.abs(psi1) ** 2).sum()
        npsi2 = (np.abs(psi2) ** 2).sum()
        message = "Checking FT scale: %7.3f %7.3f" % (npsi1 / npsi0, npsi2 / npsi1)
        self.assertAlmostEqual(npsi1 / npsi0, 1, msg=message, delta=1e-5)
        message = "Checking IFT scale: %7.3f %7.3f" % (npsi1 / npsi0, npsi2 / npsi1)
        self.assertAlmostEqual(npsi2 / npsi1, 1, msg=message, delta=1e-5)

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_phase_ramp_cuda(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = cuop.ZeroPhaseRamp(obj=True) * cuop.ApplyPhaseRamp(0.34, -0.45, obj=False, probe=True) * self.p
        # Tolerance is high due to low resolution during test (128 pixels..)
        tmp = np.allclose([self.p.data.phase_ramp_dx, self.p.data.phase_ramp_dy], [-.34, .45], atol=0.2)
        self.assertTrue(tmp, msg="CUDA: calculated object phase ramp is incorrect")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_consistency_opencl_cuda_ObjProbe2Psi(self):
        i = 0
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = clop.ObjProbe2Psi() * clop.SelectStack(i) * self.p
        psicl = self.p._cl_psi.get()[0, 0, 0]
        self.p = cuop.ObjProbe2Psi() * cuop.SelectStack(i) * self.p
        psicu = self.p._cu_psi.get()[0, 0, 0]
        atol = np.abs(psicl).max() * 1e-4
        self.p = clop.FreePU() * self.p
        self.p = cuop.FreePU() * self.p
        self.assertTrue(np.allclose(psicl, psicu, rtol=1e-4, atol=atol),
                        msg="CUDA and OpenCL Psi2ObjProbe must give close results")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_consistency_cuda_cpu_ObjProbe2Psi(self):
        i = 0
        self.make_ptycho_obj()
        for near_field in [False, True]:
            with self.subTest(near_field=near_field):
                self.p.set_obj(self.obj0)
                self.p.set_probe(self.probe0)
                self.p = cpuop.ObjProbe2Psi() * self.p
                psicpu = self.p._psi[0, 0, 0]
                self.p = cuop.ObjProbe2Psi() * cuop.SelectStack(i) * self.p
                psicu = self.p._cu_psi.get()[0, 0, 0]
                atol = np.abs(psicpu).max() * 1e-4
                self.p = cuop.FreePU() * self.p
                self.assertTrue(np.allclose(psicpu, psicu, rtol=1e-4, atol=atol),
                                msg="CUDA and CPU Psi2ObjProbe must give close results")
        self.p.data.near_field = False

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_consistency_opencl_cuda_FT_ObjProbe2Psi(self):
        i = 0
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = clop.FT(scale=False) * clop.ObjProbe2Psi() * clop.SelectStack(i) * self.p
        psicl = self.p._cl_psi.get()[0, 0, 0]
        self.p = cuop.FT(scale=False) * cuop.ObjProbe2Psi() * cuop.SelectStack(i) * self.p
        psicu = self.p._cu_psi.get()[0, 0, 0]
        atol = np.abs(psicl).max() * 1e-4
        self.p = clop.FreePU() * self.p
        self.p = cuop.FreePU() * self.p
        self.assertTrue(np.allclose(psicl, psicu, rtol=1e-4, atol=atol),
                        msg="CUDA and OpenCL FT*Psi2ObjProbe must give close results")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_consistency_cuda_cpu_FT_ObjProbe2Psi(self):
        i = 0
        self.make_ptycho_obj()
        for near_field in [False, True]:
            with self.subTest(near_field=near_field):
                # KLUDGE: is it safe to change near_field that way ?
                self.p.data.near_field = near_field
                self.p.set_obj(self.obj0)
                self.p.set_probe(self.probe0)
                self.p = cpuop.FT(scale=True) * cpuop.ObjProbe2Psi() * self.p
                psicpu = self.p._psi[0, 0, 0]
                self.p = cuop.FT(scale=True) * cuop.ObjProbe2Psi() * cuop.SelectStack(i) * self.p
                psicu = self.p._cu_psi.get()[0, 0, 0]
                atol = np.abs(psicpu).max() * 1e-4
                self.p = cuop.FreePU() * self.p
                self.assertTrue(np.allclose(psicpu, psicu, rtol=1e-4, atol=atol),
                                msg="CUDA and CPU FT*Psi2ObjProbe must give close results")
        self.p.data.near_field = False

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_consistency_opencl_cuda_PropagateApplyAmplitude(self):
        i = 0
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = clop.PropagateApplyAmplitude() * clop.ObjProbe2Psi() * clop.SelectStack(i) * self.p
        psicl = self.p._cl_psi.get()[0, 0, 0]
        self.p = cuop.PropagateApplyAmplitude() * cuop.ObjProbe2Psi() * cuop.SelectStack(i) * self.p
        psicu = self.p._cu_psi.get()[0, 0, 0]
        atol = np.abs(psicl).max() * 1e-4
        self.p = clop.FreePU() * self.p
        self.p = cuop.FreePU() * self.p
        self.assertTrue(np.allclose(psicl, psicu, rtol=1e-4, atol=atol),
                        msg="CUDA and OpenCL PropagateApplyAmplitude must give close results")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_consistency_cuda_cpu_ApplyAmplitude(self):
        i = 0
        self.make_ptycho_obj()
        for near_field in [False, True]:
            with self.subTest(near_field=near_field):
                self.p.set_obj(self.obj0)
                self.p.set_probe(self.probe0)
                self.p = cpuop.ApplyAmplitude() * cpuop.FT(scale=True) * cpuop.ObjProbe2Psi() * self.p
                psicpu = self.p._psi[0, 0, 0]
                self.p = cuop.ApplyAmplitude() * cuop.FT(scale=True) * \
                         cuop.ObjProbe2Psi() * cuop.SelectStack(i) * self.p
                psicu = self.p._cu_psi.get()[0, 0, 0]
                atol = np.abs(psicpu).max() * 1e-3
                self.p = cuop.FreePU() * self.p
                self.assertTrue(np.allclose(psicpu, psicu, rtol=1e-3, atol=atol),
                                msg="CUDA and CPU PropagateApplyAmplitude must give close results")
        self.p.data.near_field = False

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_consistency_cuda_cpu_PropagateApplyAmplitude(self):
        i = 0
        self.make_ptycho_obj()
        for near_field in [False, True]:
            with self.subTest(near_field=near_field):
                self.p.set_obj(self.obj0)
                self.p.set_probe(self.probe0)
                self.p = cpuop.PropagateApplyAmplitude() * cpuop.ObjProbe2Psi() * self.p
                psicpu = self.p._psi[0, 0, 0]
                self.p = cuop.PropagateApplyAmplitude() * cuop.ObjProbe2Psi() * cuop.SelectStack(i) * self.p
                psicu = self.p._cu_psi.get()[0, 0, 0]
                atol = np.abs(psicpu).max() * 1e-3
                self.p = cuop.FreePU() * self.p
                self.assertTrue(np.allclose(psicpu, psicu, rtol=1e-3, atol=atol),
                                msg="CUDA and CPU PropagateApplyAmplitude must give close results")
        self.p.data.near_field = False

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_consistency_opencl_cuda_AP(self, rtol=1e-4):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = clop.AP(update_probe=True, update_object=True) ** 5 * self.p
        objcl = self.p.get_obj()
        probecl = self.p.get_probe()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = cuop.AP(update_probe=True, update_object=True) ** 5 * self.p
        objcu = self.p.get_obj()
        probecu = self.p.get_probe()
        atolo = np.abs(objcl).max() * rtol
        atolp = np.abs(probecl).max() * rtol
        self.p = clop.FreePU() * self.p
        self.p = cuop.FreePU() * self.p
        self.assertTrue(np.allclose(objcl, objcu, rtol=rtol, atol=atolo)
                        and np.allclose(probecl, probecu, rtol=rtol, atol=atolp),
                        msg="CUDA and OpenCL AP**5 must give close results")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_consistency_cuda_cpu_AP(self, rtol=1e-3):
        self.make_ptycho_obj()
        for near_field in [False, True]:
            with self.subTest(near_field=near_field):
                self.p.set_obj(self.obj0.copy())
                self.p.set_probe(self.probe0.copy())
                self.p = cpuop.AP(update_probe=True, update_object=True) ** 5 * self.p
                objcpu = self.p.get_obj().copy()
                probecpu = self.p.get_probe().copy()
                self.p.set_obj(self.obj0.copy())
                self.p.set_probe(self.probe0.copy())
                self.p = cuop.AP(update_probe=True, update_object=True) ** 5 * self.p
                objcu = self.p.get_obj()
                probecu = self.p.get_probe()
                atolo = np.abs(objcpu).max() * rtol
                atolp = np.abs(probecpu).max() * rtol
                self.p = cuop.FreePU() * self.p
                self.assertTrue(np.allclose(objcpu, objcu, rtol=rtol, atol=atolo),
                                msg="CUDA and CPU AP must give close objects")
                self.assertTrue(np.allclose(probecpu, probecu, rtol=rtol, atol=atolp),
                                msg="CUDA and CPU AP must give close probes")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_consistency_opencl_cuda_DM(self, rtol=1e-3):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = clop.DM(update_probe=True, update_object=True) ** 5 * self.p
        objcl = self.p.get_obj()
        probecl = self.p.get_probe()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = cuop.DM(update_probe=True, update_object=True) ** 5 * self.p
        objcu = self.p.get_obj()
        probecu = self.p.get_probe()
        atolo = np.abs(objcl).max() * rtol
        atolp = np.abs(probecl).max() * rtol
        self.p = clop.FreePU() * self.p
        self.p = cuop.FreePU() * self.p
        self.assertTrue(np.allclose(objcl, objcu, rtol=rtol, atol=atolo)
                        and np.allclose(probecl, probecu, rtol=rtol, atol=atolp),
                        msg="CUDA and OpenCL DM**5 must give close results")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_consistency_cuda_cpu_DM(self, rtol=1e-3):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0.copy())
        self.p.set_probe(self.probe0.copy())
        self.p = cpuop.DM(update_probe=True, update_object=True, zero_phase_ramp=False) ** 5 * self.p
        objcpu = self.p.get_obj().copy()[0] * self.illum_mask
        probecpu = self.p.get_probe().copy()
        self.p.set_obj(self.obj0.copy())
        self.p.set_probe(self.probe0.copy())
        self.p = cuop.DM(update_probe=True, update_object=True, zero_phase_ramp=False) ** 5 * self.p
        objcu = self.p.get_obj()[0] * self.illum_mask
        probecu = self.p.get_probe()
        atolo = np.abs(objcpu).max() * rtol
        atolp = np.abs(probecpu).max() * rtol
        self.p = cuop.FreePU() * self.p
        self.assertTrue(np.allclose(objcpu, objcu, rtol=rtol, atol=atolo),
                        msg="CUDA and CPU DM**5 must give close objects")
        self.assertTrue(np.allclose(probecpu, probecu, rtol=rtol, atol=atolp),
                        msg="CUDA and CPU DM**5 must give close probes")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_consistency_opencl_cuda_Grad(self, rtol=5e-3):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p._cl_obj_grad = clop.cla.empty(clop.default_processing_unit.cl_queue, self.p._obj.shape, np.complex64)
        self.p._cl_probe_grad = clop.cla.empty(clop.default_processing_unit.cl_queue, self.p._probe.shape, np.complex64)
        self.p._cl_background_grad = clop.cla.zeros(clop.default_processing_unit.cl_queue, self.p._background.shape,
                                                    np.float32)
        self.p._cl_background_dir = clop.cla.zeros(clop.default_processing_unit.cl_queue, self.p._background.shape,
                                                   np.float32)
        self.p = clop.Grad(update_probe=True, update_object=True) * self.p
        illum = self.p.get_illumination_obj()
        objcl = self.p._cl_obj_grad.get() * illum
        probecl = self.p._cl_probe_grad.get()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p._cu_obj_grad = cuop.cua.empty(self.p._obj.shape, dtype=np.complex64)
        self.p._cu_probe_grad = cuop.cua.empty(self.p._probe.shape, dtype=np.complex64)
        self.p._cu_background_grad = cua.zeros(self.p._background.shape, dtype=np.float32)
        self.p._cu_background_dir = cua.zeros(self.p._background.shape, dtype=np.float32)
        self.p = cuop.Grad(update_probe=True, update_object=True) * self.p
        objcu = self.p._cu_obj_grad.get() * illum
        probecu = self.p._cu_probe_grad.get()
        atolo = np.abs(objcl).max() * rtol
        atolp = np.abs(probecl).max() * rtol
        self.p = clop.FreePU() * self.p
        self.p = cuop.FreePU() * self.p
        self.assertTrue(np.allclose(objcl, objcu, rtol=rtol, atol=atolo)
                        and np.allclose(probecl, probecu, rtol=rtol, atol=atolp),
                        msg="CUDA and OpenCL Grad must give close results")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_consistency_opencl_cuda_ML(self, rtol=1e-3):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = clop.ML(update_probe=True, update_object=True) ** 2 * self.p
        objcl = self.p.get_obj()
        probecl = self.p.get_probe()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = cuop.ML(update_probe=True, update_object=True) ** 2 * self.p
        objcu = self.p.get_obj()
        probecu = self.p.get_probe()
        # Relax tolerance for ML
        atolo = np.abs(objcl).max() * rtol
        atolp = np.abs(probecl).max() * rtol
        self.p = clop.FreePU() * self.p
        self.p = cuop.FreePU() * self.p
        self.assertTrue(np.allclose(objcl, objcu, rtol=rtol, atol=atolo)
                        and np.allclose(probecl, probecu, rtol=rtol, atol=atolp),
                        msg="CUDA and OpenCL ML**2 must give close results")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_consistency_opencl_cuda_background(self, rtol=1e-3):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        b0 = self.p.get_background().copy()
        self.p = clop.AP(update_probe=True, update_object=True, update_background=True) ** 2 * self.p
        # self.p = clop.ML(update_probe=True, update_object=True,update_background=True) ** 2 * self.p
        objcl = self.p.get_obj()
        probecl = self.p.get_probe()
        backgcl = self.p.get_background()
        self.p = clop.FreePU() * self.p

        self.p.set_background(b0)
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        self.p = cuop.AP(update_probe=True, update_object=True, update_background=True) ** 2 * self.p
        # self.p = cuop.ML(update_probe=True, update_object=True, update_background=True) ** 2 * self.p
        objcu = self.p.get_obj()
        probecu = self.p.get_probe()
        backgcu = self.p.get_background()

        # Relax tolerance for background (why?)
        atolo = np.abs(objcl).max() * rtol
        atolp = np.abs(probecl).max() * rtol
        atolb = np.abs(backgcl).max() * rtol
        self.p = cuop.FreePU() * self.p
        self.p.set_background(b0)
        # TODO: fix this test
        # self.assertTrue(np.allclose(objcl, objcu, rtol=rtol, atol=atolo)
        #                 and np.allclose(probecl, probecu, rtol=rtol, atol=atolp)
        #                 and np.allclose(backgcl, backgcu, rtol=rtol, atol=atolb),
        #                 msg="CUDA and OpenCL AP(o/p/b)**2 must give close results")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    @unittest.skipIf('cuda' in sys.argv or exclude_cuda, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_consistency_opencl_cuda_illumination(self, rtol=1e-5):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)

        self.p = cuop.CalcIllumination() * self.p
        cu_n = self.p._obj_illumination.copy()
        atol = cu_n.max() * rtol

        self.p = clop.CalcIllumination() * self.p
        cl_n = self.p._obj_illumination.copy()
        self.assertTrue(np.allclose(cu_n, cl_n, rtol=rtol, atol=atol),
                        msg="CUDA and OpenCL CalcIllumination must give close results")

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_consistency_cuda_cpu_illumination(self, rtol=1e-5):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)

        self.p = cuop.CalcIllumination() * self.p
        cu_n = self.p._obj_illumination.copy()
        atol = cu_n.max() * rtol

        self.p = cpuop.CalcIllumination() * self.p
        cpu_n = self.p._obj_illumination.copy()
        self.assertTrue(np.allclose(cu_n, cpu_n, rtol=rtol, atol=atol),
                        msg="CUDA and CPU CalcIllumination must give close results")

    def test_get_iobs_icalc(self):
        self.make_ptycho_obj()
        self.p.set_obj(self.obj0)
        self.p.set_probe(self.probe0)
        for op in self.ops:
            with self.subTest(op=op):
                for i in [0, len(self.p.data.iobs) - 1]:
                    iobs = self.p.get_iobs(i)
                    iobs = self.p.get_iobs(i, shift=True)
                    icalc = op.get_icalc(self.p, i)
                    icalc = op.get_icalc(self.p, i, shift=True)


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestPtycho))
    return test_suite


if __name__ == '__main__':
    sys.stdout = io.StringIO()
    warnings.simplefilter('ignore')
    res = unittest.TextTestRunner(verbosity=2, descriptions=False).run(suite())
