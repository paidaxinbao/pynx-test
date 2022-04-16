#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

"""
This file includes tests for the CDI python API.
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
import h5py as h5
from scipy.fftpack import fftn, ifftn, fftshift
from pynx.utils.pattern import siemens_star, fibonacci_urchin
from pynx.cdi import *

if has_cuda:
    import pynx.cdi.cu_operator as cuop
else:
    cuop = None

if has_opencl:
    import pynx.cdi.cl_operator as clop
    from pyopencl import CompilerWarning
    import warnings

    warnings.simplefilter('ignore', CompilerWarning)
else:
    clop = None

import pynx.cdi.cpu_operator as cpuop

exclude_cuda = False
exclude_opencl = False
if 'PYNX_PU' in os.environ:
    if 'opencl' in os.environ['PYNX_PU'].lower():
        exclude_cuda = True
    elif 'cuda' in os.environ['PYNX_PU'].lower():
        exclude_opencl = True


def make_cdi_data(shape=(128, 128, 128), obj_shape='rectangle', nb_photons=1e9):
    """
    Create CDI data.
    :param shape: the shape of the data file, either 2D or 3D.
    :param obj_shape: the object shape, either 'rectangle' (by default the lateral size is 1/4 of the array shape),
                      or 'circle' or 'sphere' or 'star' (a Siemens star)
    :param nb_photons: the total number of photons in the data array
    :return: a tuple with the simulated (object, observed intensity)
    """
    ndim = len(shape)
    assert (ndim in [2, 3])
    if ndim == 2:
        ny, nx = shape
        y, x = np.meshgrid(np.arange(ny) - ny // 2, np.arange(nx) - nx // 2, indexing='ij')
        z = 0
        nz = 1
    else:
        nz, ny, nx = shape
        z, y, x = np.meshgrid(np.arange(nz) - nz // 2, np.arange(ny) - ny // 2, np.arange(nx) - nx // 2, indexing='ij')

    if obj_shape == 'star':
        if ndim == 2:
            nxy = min(nx, ny)
            a = siemens_star(dsize=nxy, nb_rays=7, r_max=nxy / 4, nb_rings=3)
            d = np.zeros((ny, nx))
            d[ny // 2 - nxy // 2:ny // 2 + nxy // 2, nx // 2 - nxy // 2:nx // 2 + nxy // 2] = a
        else:
            nxy = min(nx, ny, nz)
            a = fibonacci_urchin(dsize=nxy, nb_rays=20, r_max=nxy / 4, nb_rings=8)
            d = np.zeros((nz, ny, nx))
            d[nz // 2 - nxy // 2:nz // 2 + nxy // 2, ny // 2 - nxy // 2:ny // 2 + nxy // 2,
            nx // 2 - nxy // 2:nx // 2 + nxy // 2] = a
    elif obj_shape in ['circle', 'sphere']:
        r = min(x, y) / 8
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) <= r
    else:
        # 'rectangle'
        d = (abs(x) <= (nx // 8)) * (abs(y) <= (ny // 8)) * (abs(z) <= (nz // 8))

    obj = d
    d = fftshift(np.abs(fftn((d.astype(np.complex64))))) ** 2
    d *= nb_photons / d.sum()

    return obj.astype(np.complex64), d


def make_cdi_data_file(shape=(128, 128, 128), obj_shape='rectangle', file_type='cxi', nb_photons=1e9, dir=None,
                       mask_fraction=0.02):
    """
    Create a CDI data file.
    :param shape: the shape of the data file, either 2D or 3D.
    :param obj_shape: the object shape, either 'rectangle' (by default the lateral size is 1/4 of the array shape),
                      or 'circle' or 'sphere' or 'star' (a Siemens star)
    :param file_type: either npz or cxi
    :param nb_photons: the total number of photons in the data array
    :param dir: the directory where the file will be created
    :param mask_fraction: fraction of masked pixels (for CXI only)
    :return: the file name
    """
    obj, d = make_cdi_data(shape=shape, obj_shape=obj_shape, nb_photons=nb_photons)
    # Mask 2% of pixels
    mask = np.zeros(d.shape, dtype=np.bool)
    mask[np.random.uniform(0, 1, mask.shape) < mask_fraction] = True

    if file_type == 'cxi':
        f, path = tempfile.mkstemp(suffix='.cxi', prefix="TestCDI", dir=dir)
        save_cdi_data_cxi(path, d, wavelength=1.5e-10, detector_distance=1, pixel_size_detector=55e-6, mask=mask,
                          sample_name=None, experiment_id=None, instrument=None, note=None, iobs_is_fft_shifted=False)
    else:
        # npz
        f, path = tempfile.mkstemp(suffix='.npz', dir=dir)
        np.savez_compressed(path, d=d)

    return path


def make_cdi_support_file(shape=(128, 128, 128), obj_shape='rectangle', dir=None):
    """
    Create a CDI support file.
    :param shape: the shape of the support, either 2D or 3D.
    :param obj_shape: the object shape, either 'rectangle' (by default the lateral size is 1/4 of the array shape),
                      or 'circle' or 'sphere' or 'star' (a Siemens star)
    :param file_type: either npz or cxi
    :param dir: the directory where the file will be created
    :return: the file name
    """
    obj, d = make_cdi_data(shape=shape, obj_shape=obj_shape, nb_photons=1e9)
    s = abs(obj)
    s = s > s.max() / 10
    f, path = tempfile.mkstemp(suffix='.npz', dir=dir)
    np.savez_compressed(path, support=s)

    return path


def make_cdi_mask_file(shape=(128, 128, 128), file_type='npz', dir=None, fraction_mask=0.01):
    """
    Create a mask file.
    :param shape: the shape of the data file, either 2D or 3D.
    :param file_type: can be npz, npy.
    :param dir: the directory where the file will be created
    :param fraction_mask: fraction of masked pixels, which will be randomly distributed
    :return: the file name
    """
    mask = np.random.uniform(0, 1, shape) < fraction_mask
    # Use mixed case for the name to test no lowercase is enforced
    f, path = tempfile.mkstemp(suffix='.' + file_type, prefix="TestMask", dir=dir)
    if file_type == "npy":
        np.save(path, mask, allow_pickle=False)
    elif file_type == "npz":
        np.savez_compressed(path, mask=mask)
    else:
        with h5.File(path, "w") as h:
            entry_1 = h.create_group("entry_1")
            entry_1.create_dataset("mask", data=mask, chunks=True, compression="gzip")
    return path


def make_cdi_flatfield_file(shape=(128, 128, 128), file_type='npz', dir=None):
    """
    Create a flatfield file.
    :param shape: the shape of the data file, either 2D or 3D.
    :param file_type: can be npz, npy, h5
    :param dir: the directory where the file will be created
    :return: the file name
    """
    flatfield = np.random.uniform(0.99, 1.01, shape).astype(np.float32)
    # Use mixed case for the name to test no lowercase is enforced
    f, path = tempfile.mkstemp(suffix='.' + file_type, prefix="TestFlat", dir=dir)
    if file_type == "npy":
        np.save(path, flatfield, allow_pickle=False)
    elif file_type == "npz":
        np.savez_compressed(path, flatfield=flatfield)
    else:
        with h5.File(path, "w") as h:
            entry_1 = h.create_group("entry_1")
            entry_1.create_dataset("flat", data=flatfield, chunks=True, compression="gzip")
    return path


class TestCDI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Directory contents will automatically get cleaned up on deletion
        cls.tmp_dir_obj = tempfile.TemporaryDirectory()
        cls.tmp_dir = cls.tmp_dir_obj.name
        # cdi objects, created in make_cdi_obj* for the 2D and 3D cases
        cls.cdi_2d = None
        cls.obj0_2d = None
        cls.support0_2d = None
        cls.cdi_3d = None
        cls.obj0_3d = None
        cls.support0_3d = None

    def make_cdi_obj2d(self):
        """
        Make 2D cdi obj, if it does not already exist.
        :return: Nothing
        """
        if self.cdi_2d is None:
            obj2d, d2d = make_cdi_data(shape=(256, 256))
            # Mask 2% of pixels
            mask = np.zeros(d2d.shape, dtype=np.bool)
            mask[np.random.uniform(0, 1, mask.shape) < 0.02] = True
            sup = obj2d > obj2d.max() * 0.1
            obj2d *= np.random.uniform(0.95, 1.05, obj2d.shape)
            self.cdi_2d = CDI(obj=fftshift(obj2d), pixel_size_detector=55e-6, iobs=fftshift(d2d),
                              support=fftshift(sup), wavelength=1.5e-10, detector_distance=1, mask=mask)
            self.cdi_2d = FourierApplyAmplitude() * self.cdi_2d
            self.obj0_2d = self.cdi_2d.get_obj().copy()
            self.support0_2d = self.cdi_2d.get_support().copy()
            self.iobs0_2d = self.cdi_2d.get_iobs().copy()
            self.cdi_2d = FreePU() * self.cdi_2d

    def make_cdi_obj3d(self):
        """
        Make 3D cdi obj, if it does not already exist.
        :return: Nothing
        """
        if self.cdi_3d is None:
            obj3d, d3d = make_cdi_data(shape=(128, 128, 128))
            # Mask 2% of pixels
            mask = np.zeros(d3d.shape, dtype=np.bool)
            mask[np.random.uniform(0, 1, mask.shape) < 0.02] = True
            sup = obj3d > obj3d.max() * 0.1
            obj3d *= np.random.uniform(0.95, 1.05, obj3d.shape)
            self.cdi_3d = CDI(obj=fftshift(obj3d), pixel_size_detector=55e-6, iobs=fftshift(d3d),
                              support=fftshift(sup), wavelength=1.5e-10, detector_distance=1)
            self.cdi_3d = FourierApplyAmplitude() * self.cdi_3d
            self.obj0_3d = self.cdi_3d.get_obj().copy()
            self.support0_3d = self.cdi_3d.get_support().copy()
            self.iobs0_3d = self.cdi_3d.get_iobs().copy()
            self.cdi_3d = FreePU() * self.cdi_3d

    def test_make_cdi_cxi(self):
        path = make_cdi_data_file(file_type='cxi', dir=self.tmp_dir)

    def test_make_cdi_npz(self):
        path = make_cdi_data_file(file_type='npz', dir=self.tmp_dir)

    def test_00_make_cdi_obj2d(self):
        self.make_cdi_obj2d()

    def test_00_make_cdi_obj3d(self):
        self.make_cdi_obj3d()

    def test_InitSupport(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = [cpuop]
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        # if 'cuda' not in sys.argv and not exclude_opencl:
        #     ops.append(clop)
        for op in ops:
            for cdi, obj0, support0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d),
                                        (self.cdi_3d, self.obj0_3d, self.support0_3d)]:
                with self.subTest(op=op, ndim=obj0.ndim):
                    cdi.set_obj(obj0)
                    cdi = op.InitSupportShape(shape="circle", size=20) * cdi
                    self.assertGreater(cdi.nb_point_support, 10)
                    cdi = op.InitSupportShape(shape="sphere", size=20) * cdi
                    self.assertGreater(cdi.nb_point_support, 10)
                    cdi = op.InitSupportShape(shape="square", size=20) * cdi
                    self.assertGreater(cdi.nb_point_support, 10)
                    cdi = op.InitSupportShape(shape="cube", size=20) * cdi
                    self.assertGreater(cdi.nb_point_support, 10)
                    if cdi.iobs.ndim == 2:
                        cdi = op.InitSupportShape(shape="circle", size=(20, 10)) * cdi
                        self.assertGreater(cdi.nb_point_support, 10)
                        cdi = op.InitSupportShape(shape="sphere", size=(20, 10)) * cdi
                        self.assertGreater(cdi.nb_point_support, 10)
                        cdi = op.InitSupportShape(shape="square", size=(20, 10)) * cdi
                        self.assertGreater(cdi.nb_point_support, 10)
                        cdi = op.InitSupportShape(shape="cube", size=(20, 10)) * cdi
                        self.assertGreater(cdi.nb_point_support, 10)
                    else:
                        cdi = op.InitSupportShape(shape="circle", size=(20, 10, 10)) * cdi
                        self.assertGreater(cdi.nb_point_support, 10)
                        cdi = op.InitSupportShape(shape="sphere", size=(20, 10, 10)) * cdi
                        self.assertGreater(cdi.nb_point_support, 10)
                        cdi = op.InitSupportShape(shape="square", size=(20, 10, 10)) * cdi
                        self.assertGreater(cdi.nb_point_support, 10)
                        cdi = op.InitSupportShape(shape="cube", size=(20, 10, 10)) * cdi
                        self.assertGreater(cdi.nb_point_support, 10)

    def test_InitObj(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = [cpuop]
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        # if 'cuda' not in sys.argv and not exclude_opencl:
        #     ops.append(clop)
        for op in ops:
            for cdi, obj0, support0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d),
                                        (self.cdi_3d, self.obj0_3d, self.support0_3d)]:
                with self.subTest(op=op, ndim=obj0.ndim):
                    cdi.set_obj(obj0)
                    cdi.set_support(support0)
                    cdi = op.InitObjRandom(src="support") * cdi
                    cdi = op.InitObjRandom(src="support", amin=0.9, amax=1, phirange=1) * cdi
                    cdi = op.InitObjRandom(src="obj", amin=0.9, amax=1, phirange=1) * cdi
                    cdi = op.InitObjRandom(src="obj") * cdi
                    cdi = op.InitObjRandom(src=obj0, amin=0.9, amax=1, phirange=1) * cdi

    def test_InitFreePixels_cpu(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        op = cpuop
        for cdi, obj0, support0, iobs0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d, self.iobs0_2d),
                                           (self.cdi_3d, self.obj0_3d, self.support0_3d, self.iobs0_3d)]:
            cdi.set_obj(obj0)
            cdi.set_support(support0)
            cdi.set_iobs(iobs0.copy())

            ratio = 5e-2
            radius = 3
            xzc = 0.05

            coords = cdi.init_free_pixels(ratio=ratio, island_radius=radius, exclude_zone_center=xzc)
            n0 = cdi.nb_free_points
            self.assertTrue(n0 > iobs0.size * ratio / 2)

            coords = cdi.init_free_pixels(coords=coords, island_radius=radius, exclude_zone_center=xzc)
            self.assertTrue(n0 == cdi.nb_free_points)

            cdi = op.InitFreePixels(coords=coords, island_radius=radius, exclude_zone_center=xzc) * cdi
            self.assertTrue(n0 == cdi.nb_free_points)

            cdi = op.InitFreePixels(ratio=ratio, island_radius=radius, exclude_zone_center=xzc) * cdi
            self.assertTrue(abs(cdi.nb_free_points - n0) < 0.3 * max(n0, cdi.nb_free_points))

            cdi.set_iobs(iobs0.copy())

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    def test_InitFreePixels_cuda(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        op = cuop
        for cdi, obj0, support0, iobs0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d, self.iobs0_2d),
                                           (self.cdi_3d, self.obj0_3d, self.support0_3d, self.iobs0_3d)]:
            cdi.set_obj(obj0)
            cdi.set_support(support0)
            cdi.set_iobs(iobs0.copy())

            ratio = 5e-2
            radius = 3
            xzc = 0.05

            coords = cdi.init_free_pixels(ratio=ratio, island_radius=radius, exclude_zone_center=xzc)
            n0 = cdi.nb_free_points
            self.assertTrue(abs(cdi.nb_free_points - iobs0.size * ratio) / (iobs0.size * ratio) < 0.2)

            coords = cdi.init_free_pixels(coords=coords, island_radius=radius, exclude_zone_center=xzc)
            self.assertTrue(n0 == cdi.nb_free_points)

            cdi = op.InitFreePixels(coords=coords, island_radius=radius, exclude_zone_center=xzc) * cdi
            self.assertTrue(n0 == cdi.nb_free_points)

            cdi = op.InitFreePixels(ratio=ratio, island_radius=radius, exclude_zone_center=xzc) * cdi
            self.assertTrue(abs(cdi.nb_free_points - iobs0.size * ratio) / (iobs0.size * ratio) < 0.2)
            self.assertTrue(abs(cdi.nb_free_points - n0) < 0.3 * max(n0, cdi.nb_free_points))

            cdi.set_iobs(iobs0.copy())

    def test_FourierApplyAmplitude(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = [cpuop]
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        if 'cuda' not in sys.argv and not exclude_opencl:
            ops.append(clop)
        for op in ops:
            for cdi, obj0, support0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d),
                                        (self.cdi_3d, self.obj0_3d, self.support0_3d)]:
                with self.subTest(op=op, ndim=obj0.ndim):
                    cdi.set_obj(obj0)
                    cdi.set_support(support0)
                    cdi = op.FourierApplyAmplitude() * cdi
                    cdi = FourierApplyAmplitude(obj_stats=True) * cdi
                    cdi = FreePU() * cdi

    def test_HIO(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = [cpuop]
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        if 'cuda' not in sys.argv and not exclude_opencl:
            ops.append(clop)
        for op in ops:
            for cdi, obj0, support0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d),
                                        (self.cdi_3d, self.obj0_3d, self.support0_3d)]:
                with self.subTest(op=op, ndim=obj0.ndim):
                    cdi.set_obj(obj0)
                    cdi.set_support(support0)
                    # Compute LLK before and after
                    cdi = op.LLK() * cdi
                    llk0 = cdi.get_llkn()
                    cdi = op.LLK() * op.HIO(positivity=True, calc_llk=0) ** 10 * cdi
                    llk1 = cdi.get_llkn()
                    # print("HIO**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
                    cdi = op.FreePU() * cdi

    def test_ER(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = [cpuop]
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        if 'cuda' not in sys.argv and not exclude_opencl:
            ops.append(clop)
        for op in ops:
            for cdi, obj0, support0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d),
                                        (self.cdi_3d, self.obj0_3d, self.support0_3d)]:
                with self.subTest(op=op, ndim=obj0.ndim):
                    cdi.set_obj(obj0)
                    cdi.set_support(support0)
                    # Compute LLK before and after
                    cdi = op.LLK() * cdi
                    llk0 = cdi.get_llkn()
                    cdi = op.LLK() * op.ER(positivity=True, calc_llk=0) ** 10 * cdi
                    llk1 = cdi.get_llkn()
                    # print("HIO**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
                    cdi = op.FreePU() * cdi

    def test_ER_psf(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = [cpuop]
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        if 'cuda' not in sys.argv and not exclude_opencl:
            ops.append(clop)
        for op in ops:
            for cdi, obj0, support0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d),
                                        (self.cdi_3d, self.obj0_3d, self.support0_3d)]:
                with self.subTest(op=op, ndim=obj0.ndim):
                    cdi.set_obj(obj0)
                    cdi.set_support(support0)
                    # Compute LLK before and after
                    cdi = op.LLK() * cdi
                    llk0 = cdi.get_llkn()
                    self.p = op.ER(positivity=True, calc_llk=0) ** 10 * cdi
                    self.p = op.LLK() * op.ER(positivity=True, calc_llk=0) ** 10 * op.InitPSF() * cdi
                    llk1 = cdi.get_llkn()
                    # print("HIO**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
                    cdi = op.FreePU() * cdi
                    cdi._psf_f = None

    def test_RAAR(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = [cpuop]
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        if 'cuda' not in sys.argv and not exclude_opencl:
            ops.append(clop)
        for op in ops:
            for cdi, obj0, support0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d),
                                        (self.cdi_3d, self.obj0_3d, self.support0_3d)]:
                with self.subTest(op=op, ndim=obj0.ndim):
                    cdi.set_obj(obj0)
                    cdi.set_support(support0)
                    # Compute LLK before and after
                    cdi = op.LLK() * cdi
                    llk0 = cdi.get_llkn()
                    cdi = op.LLK() * op.RAAR(positivity=True, calc_llk=0) ** 10 * cdi
                    llk1 = cdi.get_llkn()
                    # print("HIO**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
                    cdi = op.FreePU() * cdi

    def test_SupportUpdate(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = [cpuop]
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        if 'cuda' not in sys.argv and not exclude_opencl:
            ops.append(clop)
        for op in ops:
            for cdi, obj0, support0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d),
                                        (self.cdi_3d, self.obj0_3d, self.support0_3d)]:
                with self.subTest(op=op, ndim=obj0.ndim):
                    cdi.set_obj(obj0)
                    cdi.set_support(support0)
                    # Compute LLK before and after
                    cdi = op.LLK() * cdi
                    llk0 = cdi.get_llkn()
                    ra = op.RAAR(positivity=True, calc_llk=0) ** 10
                    cdi = op.SupportUpdate(force_shrink=True) * ra * cdi
                    cdi = op.SupportUpdate() * ra * cdi
                    cdi = op.SupportUpdate(update_border_n=2) * ra * cdi
                    cdi = op.SupportUpdate(method='max') * ra * cdi
                    cdi = op.LLK() * cdi
                    llk1 = cdi.get_llkn()
                    # print("HIO**10: LLK %8.2f -> %8.2f" %(llk0, llk1))
                    cdi = op.FreePU() * cdi

    def test_SupportUpdateExceptions(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = [cpuop]
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        if 'cuda' not in sys.argv and not exclude_opencl:
            ops.append(clop)
        for op in ops:
            for cdi, obj0, support0, iobs0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d, self.iobs0_2d),
                                               (self.cdi_3d, self.obj0_3d, self.support0_3d, self.iobs0_3d)]:
                cdi.set_obj(obj0)
                cdi.set_support(support0)
                try:
                    cdi = op.SupportUpdate(min_fraction=0.9) * cdi
                except SupportTooSmall:
                    pass
                else:
                    raise Exception("SupportTooSmall exception was not raised")
                try:
                    cdi = op.SupportUpdate(max_fraction=0.001) * cdi
                except SupportTooLarge:
                    pass
                else:
                    raise Exception("SupportTooLarge exception was not raised")

    def test_InterpIobsMask(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = []  # cpuop
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        if 'cuda' not in sys.argv and not exclude_opencl:
            ops.append(clop)
        for op in ops:
            for cdi, obj0, support0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d),
                                        (self.cdi_3d, self.obj0_3d, self.support0_3d)]:
                with self.subTest(op=op, ndim=obj0.ndim):
                    cdi.set_obj(obj0)
                    cdi.set_support(support0)
                    iobs = cdi.get_iobs().copy()
                    cdi = op.InterpIobsMask() * cdi
                    cdi.set_iobs(iobs)
                    cdi = op.FreePU() * cdi

    def test_PSF(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = [cpuop]
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        if 'cuda' not in sys.argv and not exclude_opencl:
            ops.append(clop)
        for op in ops:
            for cdi, obj0, support0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d),
                                        (self.cdi_3d, self.obj0_3d, self.support0_3d)]:
                for model in ["pseudo-voigt", "lorentzian", "gaussian"]:
                    for filter in [None, "hann", "tukey"]:
                        with self.subTest(op=op, ndim=obj0.ndim, model=model, filter=filter):
                            cdi = op.InitPSF(model=model, filter=filter) * cdi
                            cdi = op.RAAR(positivity=True, calc_llk=0, update_psf=5, psf_filter=filter) ** 10 * cdi
                            cdi = op.HIO(positivity=True, calc_llk=0, update_psf=5, psf_filter=filter) ** 10 * cdi
                            cdi = op.ER(positivity=True, calc_llk=0, update_psf=5, psf_filter=None) ** 10 * cdi
                            cdi = op.FreePU() * cdi
                            cdi._psf_f = None

    def test_AutoCorrelationSupport_lazy(self):
        self.make_cdi_obj2d()
        self.make_cdi_obj3d()
        ops = [cpuop]
        if 'opencl' not in sys.argv and not exclude_cuda:
            ops.append(cuop)
        if 'cuda' not in sys.argv and not exclude_opencl:
            ops.append(clop)
        for op in ops:
            for cdi, obj0, support0 in [(self.cdi_2d, self.obj0_2d, self.support0_2d),
                                        (self.cdi_3d, self.obj0_3d, self.support0_3d)]:
                with self.subTest(op=op, ndim=obj0.ndim):
                    cdi.set_obj(obj0)
                    cdi.set_support(support0 * 0)
                    cdi = op.AutoCorrelationSupport(threshold=0.1, lazy=True) * cdi
                    self.assertTrue(cdi.get_support().sum() == 0)
                    cdi = op.ER(positivity=True, calc_llk=0) ** 10 * cdi
                    self.assertTrue(cdi.get_support().sum() > 10)
                    cdi = op.FreePU() * cdi

    def run_algorithm_consistency(self, op, mod_op1=clop, mod_op2=cuop, atol=1e-5, rtol=1e-4,
                                  rtol_llk=1e-2, nbtol=10, nbfractol=0.005, ndim=2):
        """
        Test the consistency of the calculation using e.g. OpenCL and CUDA, for a given operator
        :param op: the operator to be used, written in a generic way such as 'op.Operator()', where 'op.' will be 
                   replaced successively by 'clop.' and 'cuop.', e.g.: 'op.FT(scale=False)', 'op.LLK(),
                   'op.FourierApplyAmplitude()', 'op.SupportUpdate() * op.FourierApplyAmplitude()', etc...
        :param mod_op1: first operator module, either clop (OpenCL), cuop (CUDA) or cpuop (CPU)
        :param mod_op2: second operator module, either clop (OpenCL), cuop (CUDA) or cpuop (CPU)
        :param atol: absolute tolerance, relative to the maximum of the object (or FT'd object) array.
        :param rtol: relative tolerance for the object (or FT'd object) array.
        :param rtol_llk: relative tolerance for the calculated LLK.
        :param nbtol: number of points which can be above the numerical tolerance
        :param nbfractol: fraction of points which can differ in the support array.
        :param ndim: number of dimensions. Either 2 or 3.
        :return: nothing
        """
        if ndim == 2:
            self.make_cdi_obj2d()
            cdi, obj0, support0 = self.cdi_2d, self.obj0_2d, self.support0_2d
        else:
            self.make_cdi_obj3d()
            cdi, obj0, support0 = self.cdi_3d, self.obj0_3d, self.support0_3d

        if mod_op1 == clop:
            s_op1 = 'clop.'
            pu1 = 'OpenCL'
        elif mod_op1 == cpuop:
            s_op1 = 'cpuop.'
            pu1 = 'CPU'
        else:
            s_op1 = 'cuop.'
            pu1 = 'CUDA'
        cdi.set_obj(obj0)
        cdi.set_support(support0)
        cdi = eval(op.replace('op.', s_op1)) * cdi
        objcl = cdi.get_obj()
        supcl = cdi.get_support()
        llkcl = cdi.get_llkn()

        if mod_op2 == clop:
            s_op2 = 'clop.'
            pu2 = 'OpenCL'
        elif mod_op2 == cpuop:
            s_op2 = 'cpuop.'
            pu2 = 'CPU'
        else:
            s_op2 = 'cuop.'
            pu2 = 'CUDA'
        cdi.set_obj(obj0)
        cdi.set_support(support0)
        cdi = eval(("op.FreePU() * " + op).replace('op.', s_op2)) * cdi
        objcu = cdi.get_obj()
        supcu = cdi.get_support()
        llkcu = cdi.get_llkn()

        atol = np.abs(objcl).max() * atol
        nb = (supcl != supcu).sum()
        nbf = nb / obj0.size * 100
        self.assertTrue((nb < nbfractol * obj0.size),
                        msg="%s and %s operation (%dD): '%s' must give close results (dSupport=%d [%4.2f%%])" %
                            (pu1, pu2, ndim, op, nb, nbf))
        nb = (abs(objcu - objcl) > (rtol * abs(objcl) + atol)).sum()
        self.assertTrue(nb < nbtol,
                        msg="%s and %s operation (%dD): '%s' must give close results (Psi, nbdiff=%d)" %
                            (pu1, pu2, ndim, op, nb))
        self.assertTrue((llkcl - llkcu) < (llkcl * rtol_llk + 1e-4),
                        msg="%s and %s operation (%dD): '%s' must give close results (LLK= %8g, %8g)" % (
                            pu1, pu2, ndim, op, llkcl, llkcu))

    @unittest.skipIf('opencl' in sys.argv or exclude_cuda, "CUDA tests excluded")
    @unittest.skipIf(has_cuda is False, 'no CUDA support')
    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_consistency_opencl_cuda(self):
        for ndim in (2, 3):
            for op in ['op.FT(scale=True)', 'op.IFT(scale=True)', 'op.FourierApplyAmplitude()',
                       'op.SupportUpdate(method="rms", threshold_relative=0.3)',
                       'op.SupportUpdate(method="max")',
                       'op.SupportUpdate(method="average", threshold_relative=0.3)',
                       'op.ER()', 'op.ER(positivity=True)']:
                self.run_algorithm_consistency(op, mod_op1=clop, mod_op2=cuop, ndim=ndim)

    @unittest.expectedFailure
    @unittest.skipIf('cuda' in sys.argv or exclude_opencl, "OpenCL tests excluded")
    @unittest.skipIf(has_opencl is False, 'no OpenCL support')
    def test_consistency_opencl_cpu(self):
        for ndim in (2,):
            for op in ['op.FT(scale=True)', 'op.IFT(scale=True)', 'op.FourierApplyAmplitude()', 'op.SupportUpdate()',
                       'op.ER()', 'op.ER(positivity=True)']:
                self.run_algorithm_consistency(op, mod_op1=clop, mod_op2=cpuop, ndim=ndim)


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestCDI))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
