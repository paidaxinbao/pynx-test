# -*- coding: utf-8 -*-

########################################################################
#
# Example of 2D Bragg ptycho reconstruction from simulated data
# (c) ESRF 2019-present
# Authors: Vincent Favre-Nicolin <favre@esrf.fr>
#
########################################################################
import os
import timeit
import numpy as np

# Use only OpenCL - this must be done before PyNX imports
os.environ['PYNX_PU'] = 'opencl'  # Force using OpenCL only (even for Wavefront)

from pynx.ptycho.bragg2d import *
from pynx.ptycho.bragg.cpu_operator import show_3d
from pynx.ptycho.simulation import spiral_archimedes
from pynx.wavefront import Wavefront, ImshowRGBA, ImshowAbs, PropagateFarField, PropagateNearField
import matplotlib.pyplot as plt

# Experiment parameters
wavelength = 1.5e-10
delta = np.deg2rad(64)
eta = delta / 2
nu = np.deg2rad(0)
pixel_size_detector = 55e-6
ny, nx = (192, 192)
detector_distance = 1

# Spiralscan positions
nb = 128
default_processing_unit.cl_stack_size = nb  # 16
xs, ys = spiral_archimedes(100e-9, nb)
zs = np.zeros_like(xs)

# Rotate positions according to eta (piezo motors in the sample frame are rotated by eta)
ce, se = np.cos(eta), np.sin(eta)
ys, zs = ce * ys + se * zs, ce * zs - se * ys

# Project the positions along z onto the average sample position, to avoid non-centered Psi
zs += ys / np.tan(eta)

# detector parameters
detector = {'rotation_axes': (('x', -delta), ('y', -nu)), 'pixel_size': pixel_size_detector,
            'distance': detector_distance}

# Create empty data
data = Bragg2DPtychoData(iobs=np.empty((nb, ny, nx), dtype=np.float32), positions=(xs, ys, zs), mask=None,
                         wavelength=wavelength, detector=detector)

if False:
    # Import existing probe from 2D ptycho
    d = np.load("/Users/favre/Analyse/201606id01-FZP/ResultsScan0013/latest.npz")
    # d = np.load("/Users/vincent/Analyse/201606id01-FZP/ResultsScan0000/latest.npz")
    pr = Wavefront(d=np.fft.fftshift(d['probe'], axes=(-2, -1)), z=0, pixel_size=d['pixelsize'], wavelength=wavelength)
else:
    # Simulate probe from a focused aperture, with some defocusing
    pixel_size_focus = wavelength * detector_distance / (nx * pixel_size_detector)
    focal_length = 0.09
    defocus = 200e-6
    widthy = 300e-6
    widthx = 60e-6
    pixel_size_aperture = wavelength * focal_length / (nx * pixel_size_focus)
    pr = Wavefront(d=np.ones((ny, nx)), wavelength=wavelength, pixel_size=pixel_size_aperture)
    x, y = pr.get_x_y()
    print(x.min(), x.max(), y.min(), y.max(), wavelength, pixel_size_aperture)
    pr.set((abs(y) < (widthy / 2)) * (abs(x) < (widthx / 2)))
    pr = PropagateNearField(dz=defocus) * PropagateFarField(focal_length, forward=False) * pr

print('Probe pixel size: %6.2fnm' % (pr.pixel_size * 1e9))
# pr = ImshowRGBA()*pr  # Display 2D probe

# Create main Bragg Ptycho object
p = Bragg2DPtycho(probe=pr, data=data, support=None)
pxyz = p.voxel_size_object()
print(wavelength * detector_distance / (pixel_size_detector * nx) * 1e9)
print("Object voxel size: %6.2fnm x %6.2fnm x %6.2fnm" % (pxyz[0] * 1e9, pxyz[1] * 1e9, pxyz[2] * 1e9))
print(p.m)

#  ####################################  Create Object & Support  ###############################################
# Base parallelepiped object
x0, x1, y0, y1, z0, z1 = -1e-6, 1e-6, -200e-9, 200e-9, -400e-9, 400e-9
# Create a support. Larger than the object, or not...
rs = 1.0
x, y, z = p.get_xyz(domain='object', rotation=('x', eta))
sup = (x >= rs * x0) * (x <= rs * x1) * (y >= rs * y0) * (y <= rs * y1) * (z >= rs * z0) * (z <= rs * z1)
p.set_support(sup, shrink_object_around_support=False)

if False:
    plt.figure(figsize=(9, 4))
    show_3d(p.support, ortho_m=p.m, rotation=('x', eta))

p.set_support(sup, shrink_object_around_support=True)

if False:
    plt.figure(figsize=(9, 4))
    show_3d(p.support, ortho_m=p.m, rotation=('x', eta))

x, y, z = p.get_xyz(domain='object', rotation=('x', eta))
obj0 = (x >= x0) * (x <= x1) * (y >= y0) * (y <= y1) * (z >= z0) * (z <= z1)
obj1 = obj0.copy()
if False:
    # Add some strain
    obj1 = obj0 * np.exp(1j * 8 * np.exp(-(x ** 2 + z ** 2) / 200e-9 ** 2))
if False:
    # a few random twin domains
    nb_domain = 20
    cx = np.random.uniform(x0, x1, nb_domain)
    cz = np.random.uniform(z0, z1, nb_domain)
    c = (np.random.uniform(0, 1, nb_domain) > 0.5).astype(np.float32)
    # distance of eqch domain
    dist2 = np.ones_like(obj0, dtype=np.float32)
    ph = np.zeros_like(obj0, dtype=np.float32)
    for i in range(nb_domain):
        d2 = (x - cx[i]) ** 2 + (z - cz[i]) ** 2
        ph = ph * (d2 >= dist2) + c[i] * (d2 < dist2)
        dist2 = dist2 * (d2 >= dist2) + d2 * (d2 < dist2)
    obj1 = obj0 * (2 * ph - 1)  # +/-1
    # obj1 = obj0 * np.exp(1j * np.pi / 2 * ph)  # 0 or pi/2
if True:
    # Put hole to get an idea about the shape & orientation of pixels
    nzo, nyo, nxo = x.shape
    obj1[nzo // 2 - 4:nzo // 2 + 4, nyo // 2 - 4:nyo // 2 + 4, nxo // 2 - 4:nxo // 2 + 4] = 0
    # obj1[nzo//2-20:nzo//2+20,nyo//2-2:nyo//2+2,nxo//2-2:nxo//2+2]=0

p.set_obj(obj1)

# plt.figure(figsize=(9,4))
# p = ShowObj(rotation=None, title='object rotated back to eta=0, no support') * p #, extent=(x0,x1,y0,y1,z0,z1)

#  #################################### Simulate data #############################################
p = Calc2Obs(poisson_noise=True, nb_photons_per_frame=1e9) * FT() * ObjProbe2Psi() * p

#  #################################### Set object starting point #############################################
p.set_obj(obj0 * np.random.uniform(.9, 1.0, obj0.shape))
# plt.figure(figsize=(9, 4))
# p = ShowObj(rotation=('x', eta)) * p

# Scale object and probe with observed intensity before any optimisation
p = ScaleObjProbe() * p

#  #################################### Reconstruct object #############################################

plt.figure(figsize=(9, 4))
p = DM(calc_llk=10, show_obj_probe=10, reg_fac_obj_a=1e-2, reg_fac_obj_c=1e-2, back_projection_method='cg') ** 40 * p
# p = DM(calc_llk=1, show_obj_probe=10, obj_inertia=0., obj_smooth_sigma=0, back_projection_method='rep') ** 40 * p
p = AP(calc_llk=10, show_obj_probe=20, reg_fac_obj_a=0, reg_fac_obj_c=0, back_projection_method='cg') ** 200 * p
# p = AP(calc_llk=1, show_obj_probe=10, obj_inertia=0.00, obj_smooth_sigma=1, back_projection_method='rep') ** 200 * p
p = ML(calc_llk=10, show_obj_probe=10) ** 40 * p
# np.savez('bragg.npz', probe3d=p.get_probe(), obj=p.get_obj(), probe2d=pr.get(shift=True), x=x, y=y, z=z)
