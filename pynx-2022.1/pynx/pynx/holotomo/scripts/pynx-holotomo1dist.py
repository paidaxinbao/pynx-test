#! /sware/exp/pynx/devel.p9/bin/python

# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
import os
import sys

os.environ['PYNX_PU'] = 'cuda.0'
import time
import psutil

nproc = len(os.sched_getaffinity(0)) * psutil.cpu_count(logical=False) // psutil.cpu_count(logical=True)
print("Number of available processors: ", nproc)
from multiprocessing import Pool
import numpy as np
import hdf5plugin
import h5py as h5
import fabio
from pynx.utils.phase import unwrap_phase
from pynx.holotomo import *
from pynx.utils.array import rebin, pad
from nabu.reconstruction.fbp import Backprojector

t00 = time.time()

################################################################
# Experiment parameters - should be laoded from a file
################################################################

save_phase_chunks = False
save_edf = False
save_fbp_vol = False
padding = 0  # default value if not in in parameter file

if False:
    data_dir = "/data/id16b/inhouse3/for_vincent/"
    dark_name = "AZ31_K_tensileHR_001_1_/AZ31_K_tensileHR_001_1_/dark.edf"
    ref_name = "AZ31_K_tensileHR_001_1_/AZ31_K_tensileHR_001_1_/refHST0000.edf"
    img_prefix = "AZ31_K_tensileHR_001_1_/AZ31_K_tensileHR_001_1_%04d.edf"

    delta_beta = 280
    wavelength = 12.3984e-10 / 17.5

    nb_proj = 721  # number of images loaded, including the empty one
    nb_proj_total = 720  # Total number of available projections

    # Experimental parameters (ID16B)
    # Note that the only parameters used in the end are pixel_size
    # and detector_distance (stored in vz)

    # sx = 44.2438  # sample motor position. In mm ? relative to what ? focus ?
    # sx0h = 1.02869 ;
    # sx0v = 1.04543 ;
    # z1_2 = 0.560974  # Total distance focus-detector (in m?)

    z1h = 0.0430511;  # Focus-sample distance
    z1v = 0.0430344;
    z2h = 0.517923;  # Sample-detector distance
    z2v = 0.517939;
    # pixel_size_orig = 2*6.5e-07  # real detector pixel size, no magnification
    pixel_size = 9.97278e-08  # Pixel size (magnified)
    # distance = 517.923  # Corrected distance sample-detector in mm ??

    # effective propagation distances z1*z2/(z1+z2)
    z1 = 0.5 * (z1h + z1v)
    z2 = 0.5 * (z2h + z2v)
    detector_distance = z1 * z2 / (z1 + z2)
    print("Effective propagation distance:", detector_distance)
    # print("Magnification:", z1_2 / z1)
    # print("Calculated pixel size:", pixel_size_orig / (z1_2 / z1))

    rebin_n = 1
else:
    # Look for the first "*.par" file given as argument
    par_nok = True
    for v in sys.argv:
        if len(v) < 4:
            continue
        if v[-4:] == '.par':
            print("Loading parameters from: %s" % v)
            exec(open(v).read())
            par_nok = False
        if 'projection_range' in v:
            print("Using range from command-line: %s" % v)
            projection_range = eval(v.split("=")[-1])

    if par_nok:
        print("You must supply a .par file")
        sys.exit(1)

# Indices for the used projections
if len(projection_range) == 2:
    idx = np.arange(projection_range[0], projection_range[1], dtype=np.int16)
else:
    idx = np.arange(projection_range[0], projection_range[1], projection_range[2], dtype=np.int16)

nb_proj_total = len(idx)
vz = np.array([detector_distance], dtype=np.float32)
nb_proj = nb_proj_total + 1  # number of images loaded, including the empty one

# Only one distance in this script
nz = 1

if rebin_n > 1:
    pixel_size *= rebin_n
    try:
        tomo_rot_center /= rebin_n
    except:
        pass
if len(projection_range) == 3:
    try:
        tomo_angular_step *= projection_range[2]
    except:
        pass

print("################################################################")
print(" Loading data in //")
print("################################################################")
t0 = time.time()
dark = fabio.open(data_dir + "/" + dark_name).data
ref = fabio.open(data_dir + "" + ref_name).data - dark

if rebin_n > 1:
    dark = rebin(dark, (rebin_n, rebin_n))
    ref = rebin(ref, (rebin_n, rebin_n))
    pixel_size *= rebin_n
    try:
        tomo_rot_center /= rebin_n
    except:
        pass

ny, nx = dark.shape


def load_data(i):
    img = fabio.open(data_dir + "/" + img_prefix % i).data
    if rebin_n > 1:
        img = rebin(img, rebin_n)
    return img - dark


pool = Pool(nproc)  # Pool(nproc if nproc < 20 else 20)
res = pool.map(load_data, idx) + [ref]
del pool

iobs = np.empty((nb_proj, nz, ny, nx), dtype=np.float32)
for i in range(nb_proj):
    iobs[i] = res[i]

iobs[-1] = ref
sample_flag = np.ones(nb_proj, dtype=np.bool)
sample_flag[-1] = False

print("Data size (including reference frame): %dx%dx%d (%6.3fGbyte)" \
      % (iobs.shape[0], iobs.shape[2], iobs.shape[3], iobs.size * 4 / 1024 ** 3))
print("Time to load data: %4.1fs" % (time.time() - t0))

print("################################################################")
print("Create PCIData & PCI Object")
print("################################################################")

# Pad data
if padding:
    iobs = pad(iobs, padding=padding, stack=True)
    ny += 2 * padding
    nx += 2 * padding

# Create PCIData
data = HoloTomoData(iobs, pixel_size_detector=pixel_size, wavelength=wavelength, detector_distance=vz,
                    stack_size=stack_size, sample_flag=sample_flag, idx=list(idx) + [-1], padding=padding)

# Create PCI object
p = HoloTomo(data=data, obj=None, probe=None)

dt = time.time() - t00
print("Elapsed time since beginning:  %4.1fs" % dt)

print("################################################################")
print(" Algorithms")
print("################################################################")

print("\nPaganin reconstruction & scaling\n")
p = BackPropagatePaganin(delta_beta=delta_beta) * p
p.set_probe(np.ones((nz, 1, ny, nx)))
p = ScaleObjProbe() * p

dt = time.time() - t00
print("\nElapsed time since beginning:  %4.1fs\n" % dt)
p = AP(update_object=False, update_probe=True, calc_llk=10,
       delta_beta=delta_beta, reg_obj_smooth=ap_smooth, weight_empty=10) ** 5 * p

if nb_dm:
    p = DM(update_object=True, update_probe=True, calc_llk=10,
           delta_beta=delta_beta, reg_obj_smooth=dm_smooth, weight_empty=10) ** nb_dm * p
if nb_ap:
    p = AP(update_object=True, update_probe=True, calc_llk=10,
           delta_beta=delta_beta, reg_obj_smooth=ap_smooth, weight_empty=1) ** nb_ap * p

# Free GPU memory
p = FreePU() * p

dt = time.time() - t00
print("Elapsed time since beginning:  %4.1fs" % dt)

if save_phase_chunks:
    print("################################################################")
    print(" Saving phased projections to hdf5 file [slow]")
    print("################################################################")
    t0 = time.time()
    p.save_obj_probe_chunk(chunk_prefix=prefix_result + "pynx_phase_%04d", save_obj_phase=True,
                           save_obj_complex=False, save_probe=True, dtype=np.float16,
                           verbose=True, crop_padding=True)

    dt = time.time() - t0
    print("Time to save phases:  %4.1fs" % dt)
    dt = time.time() - t00
    print("Elapsed time since beginning:  %4.1fs" % dt)

if save_edf:
    print("################################################################")
    print(" Saving phased images to %scalc_images/%s_N.edf in //" % (prefix_result, prefix))
    print("################################################################")
    os.makedirs(prefix_result + 'calc_images', exist_ok=True)

    # idx, obj_phase = p.get_obj_phase_unwrapped(crop_padding=True, dtype=np.float32)
    # for i, o in zip(idx, obj_phase):
    #    edf = fabio.edfimage.EdfImage(data=o.astype(np.float32))
    #    edf.write("%scalc_images/%s_%04d.edf" % (prefix_result, prefix, i))

    def save_edf(i):
        idxjunk, obj_phase = p.get_obj_phase_unwrapped(crop_padding=True, dtype=np.float32, idx=i)
        edf = fabio.edfimage.EdfImage(data=obj_phase[0].astype(np.float32))
        edf.write("%scalc_images/%s_%04d.edf" % (prefix_result, prefix, i))

    pool = Pool(nproc)  # Pool(nproc if nproc < 20 else 20)
    res = pool.map(save_edf, p.data.idx[:-1])
    del pool

if save_fbp_vol:
    print("################################################################")
    print(" Aggregate reconstructed projections in memory")
    print("################################################################")

    t0 = time.time()
    nproj, ny, nx = nb_proj_total, p.data.stack_v[0].obj.shape[-2], p.data.stack_v[0].obj.shape[-1]
    ny, nx = ny - 2 * padding, nx - 2 * padding
    obj3d_phase = np.empty((nproj, ny, nx), dtype=np.float32)


    # In parallel - careful not to use too much memory !
    # Numba here may be more efficient
    def calc_obj3d_phase(i):
        s = p.data.stack_v[i // p.data.stack_size]
        obj_phase0 = s.obj_phase0[i % p.data.stack_size, 0]
        obj = s.obj[i % p.data.stack_size, 0]
        op = obj_phase0 + ((-np.angle(obj) - obj_phase0) % (2 * np.pi))
        op -= ((op - obj_phase0) >= np.pi) * 2 * np.pi
        op += ((op - obj_phase0) < -np.pi) * 2 * np.pi
        return op


    pool = Pool(nproc if nproc < 20 else 20)

    res = pool.map(calc_obj3d_phase, range(nb_proj_total)) + [ref]
    del pool

    for i in range(nb_proj_total):
        if padding:
            obj3d_phase[i] = res[i][padding:-padding, padding:-padding]
        else:
            obj3d_phase[i] = res[i]

    # dt = time.time() - t0
    # print("Time to compute 3D phase object:  %4.1fs" % dt)

    obj3d_phase -= obj3d_phase.mean(axis=0)  # Poor man's ring removal

    dt = time.time() - t0
    print("Time to compute 3D phase object & remove average:  %4.1fs" % dt)

    dt = time.time() - t00
    print("Elapsed time since beginning:  %4.1fs" % dt)

    print("################################################################")
    print(" FBP using Nabu")
    print("################################################################")
    t0 = time.time()

    nz, ny, nx = obj3d_phase.shape
    vol = np.empty((ny, nx, nx))
    print(obj3d_phase.shape)
    # TODO: tune dn as a function of the frame size
    dn = 20
    for i in range(0, obj3d_phase.shape[1] - 100, dn):
        # pynx result
        multi_sino = np.swapaxes(-obj3d_phase[:, i:i + dn, :], 0, 1).copy()
        print(i, multi_sino.shape)
        B = Backprojector(multi_sino.shape, rot_center=tomo_rot_center,
                          angles=np.deg2rad(np.arange(multi_sino.shape[-2]) * tomo_angular_step),
                          filter_name=None)
        res = B.fbp(multi_sino)

        for ii in range(len(res)):
            vol[i + ii] = res[ii]

    dt = time.time() - t0
    print("Time to perform FBP reconstruction:  %4.1fs" % dt)
    dt = time.time() - t00
    print("Elapsed time since beginning:  %4.1fs" % dt)

    print("################################################################")
    print(" Saving hdf5 of FBP volume as a float16 array [SLOW]")
    print("################################################################")
    t0 = time.time()
    filename = prefix_result + "pynx_vol.h5"
    f = h5.File(filename, "w")
    f.attrs['creator'] = 'PyNX'
    # f.attrs['NeXus_version'] = '2018.5'  # Should only be used when the NeXus API has written the file
    f.attrs['HDF5_Version'] = h5.version.hdf5_version
    f.attrs['h5py_version'] = h5.version.version
    f.attrs['default'] = 'entry_1'

    entry_1 = f.create_group("entry_1")
    entry_1.attrs['NX_class'] = 'NXentry'
    entry_1.attrs['default'] = 'data_1'
    data_1 = entry_1.create_group("data_1")
    data_1.attrs['NX_class'] = 'NXdata'
    data_1.attrs['signal'] = 'data'
    data_1.attrs['interpretation'] = 'image'
    data_1['title'] = 'PyNX (FBP:Nabu)'
    nz, ny, nx = vol.shape
    data_1.create_dataset("data", data=vol.astype(np.float16), chunks=(1, ny, nx), shuffle=True, compression="gzip")
    f.close()
    print("Finished saving %s" % filename)

    dt = time.time() - t0
    print("Time to save hdf5 volume:  %4.1fs" % dt)

if save_fbp_vol_ht:
    print("################################################################")
    print(" Loading holotomo reconstructed projections in memory")
    print("################################################################")

    t0 = time.time()


    def load_ht(i):
        return fabio.open(data_dir + "/" + img_prefix_ht % i).data


    pool = Pool(nproc)
    res = pool.map(load_ht, idx[:-1])
    del pool

    recons_holotomo = np.empty((len(res), ny, nx), dtype=np.float32)
    for i in range(len(res)):
        recons_holotomo[i] = res[i]

    # Load reconstructed images from holotomo_slave
    print(recons_holotomo.shape)
    recons_holotomo -= recons_holotomo.mean(axis=0)  # Poor man's ring removal

    dt = time.time() - t0
    print("Time to load ht results & remove average:  %4.1fs" % dt)

    dt = time.time() - t00
    print("Elapsed time since beginning:  %4.1fs" % dt)

    print("################################################################")
    print(" FBP(ht) using Nabu")
    print("################################################################")
    t0 = time.time()

    print(recons_holotomo.shape)
    nz, ny, nx = recons_holotomo.shape
    volht = np.empty((ny, nx, nx))
    dn = 108
    for i in range(0, recons_holotomo.shape[1] - 100, dn):
        # pynx result
        multi_sino = np.swapaxes(recons_holotomo[:, i:i + dn, :], 0, 1).copy()
        print(i, multi_sino.shape)
        B = Backprojector(multi_sino.shape, rot_center=tomo_rot_center,
                          angles=np.deg2rad(np.arange(multi_sino.shape[-2]) * tomo_angular_step),
                          filter_name=None)
        res = B.fbp(multi_sino)

        for ii in range(len(res)):
            volht[i + ii] = res[ii]

    dt = time.time() - t0
    print("Time to perform FBP(ht) reconstruction:  %4.1fs" % dt)
    dt = time.time() - t00
    print("Elapsed time since beginning:  %4.1fs" % dt)

    print("################################################################")
    print(" Saving hdf5 of FBP(ht) volume as a float16 array [SLOW]")
    print("################################################################")
    t0 = time.time()
    filename = prefix_result + "ht_vol.h5"
    f = h5.File(filename, "w")
    f.attrs['creator'] = 'PyNX'
    # f.attrs['NeXus_version'] = '2018.5'  # Should only be used when the NeXus API has written the file
    f.attrs['HDF5_Version'] = h5.version.hdf5_version
    f.attrs['h5py_version'] = h5.version.version
    f.attrs['default'] = 'entry_1'

    entry_1 = f.create_group("entry_1")
    entry_1.attrs['NX_class'] = 'NXentry'
    entry_1.attrs['default'] = 'data_1'
    data_1 = entry_1.create_group("data_1")
    data_1.attrs['NX_class'] = 'NXdata'
    data_1.attrs['signal'] = 'data'
    data_1.attrs['interpretation'] = 'image'
    data_1['title'] = 'holotomo (FBP:Nabu)'
    nz, ny, nx = volht.shape
    data_1.create_dataset("data", data=volht.astype(np.float16), chunks=(1, ny, nx), shuffle=True, compression="gzip")
    f.close()
    print("Finished saving %s" % filename)

    dt = time.time() - t0
    print("Time to save hdf5 volume:  %4.1fs" % dt)

dt = time.time() - t00
print("################################################################")
print("Finished - Elapsed time since beginning:  %4.1fs" % dt)
print("################################################################")
