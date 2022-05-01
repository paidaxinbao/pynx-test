########################################################################
#
# Example of 3D Bragg ptycho reconstruction from simulated data
# (c) ESRF 2017-present
# Authors: Vincent Favre-Nicolin <favre@esrf.fr>
#
########################################################################

import timeit
import numpy as np
from pynx.ptycho.bragg import *
from pynx.ptycho.simulation import spiral_archimedes
from pynx.wavefront import Wavefront
import matplotlib.pyplot as plt
from pynx.processing_unit import default_processing_unit as main_default_processing_unit

main_default_processing_unit.select_gpu()

# Experiment parameters
wavelength = 1.5e-10
delta = np.deg2rad(60)
nu = 0
pixel_size_detector = 55e-6
ny, nx = (160, 160)
npsi = 80
psi_step = np.deg2rad(0.01)
detector_distance = 1

# Spiralscan positions
nb = 10
xs, ys = spiral_archimedes(200e-9, nb)
zs = np.zeros_like(xs)

# This will for using only one stack - if memory allows it !
default_processing_unit.cl_stack_size = nb

# detector parameters
detector = {'geometry': 'psic', 'delta': delta, 'nu': nu, 'pixel_size': pixel_size_detector,
            'distance': detector_distance, 'rotation_axis': 'eta', 'rotation_step': psi_step}

# Create empty data
data = BraggPtychoData(iobs=np.empty((nb, npsi, ny, nx), dtype=np.float32), positions=(xs, ys, zs), mask=None,
                       wavelength=wavelength, detector=detector)

# Import existing probe from 2D ptycho
d = np.load("/Users/favre/Analyse/201606id01-FZP/ResultsScan0013/latest.npz")
#d = np.load("/Users/vincent/Analyse/201606id01-FZP/ResultsScan0000/latest.npz")
pr = Wavefront(d=fftshift(d['probe']), z=0, pixel_size=d['pixelsize'], wavelength=wavelength)

# Create main Bragg Ptycho object
p = BraggPtycho(probe=pr, data=data, support=None)
pxyz = p.voxel_size_object()
print("Object voxel size: %6.2fnm x %6.2fnm x %6.2fnm" % (pxyz[0] * 1e9, pxyz[1] * 1e9, pxyz[2] * 1e9))

# Create parallelepiped object
x0, x1, y0, y1, z0, z1 = -1e-6, 1e-6, -100e-9, 100e-9, -200e-9, 200e-9
x, y, z = p.get_xyz(domain='object', rotation=('x', delta / 2))
obj0 = (x >= x0) * (x <= x1) * (y >= y0) * (y <= y1) * (z >= z0) * (z <= z1)
if True:
    # Add some strain
    obj1 = obj0 * np.exp(1j * 8 * np.exp(-(x ** 2 + z ** 2) / 200e-9 ** 2))
    p.set_obj(obj1)
else:
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
    p.set_obj(obj1)
    # plt.imshow((np.ma.masked_array(obj1, mask=obj0 < 0.5)).sum(axis=1))

p = ShowObj() * p

# Calculate the observed intensity and copy it to the observed ones
p = Calc2Obs() * FT() * ObjProbe2Psi() * p
# Apply a scale factor and use Poisson noise
p.data.iobs = np.random.poisson(p.data.iobs * 1e10 / p.data.iobs.sum())
# KLUDGE: we clear GPU data to make sure the new iobs will be used
p = FreePU() * p

# Set a support larger than the object
sup = (x >= 1.2 * x0) * (x <= 1.2 * x1) * (y >= 1.2 * y0) * (y <= 1.2 * y1) * (z >= 1.2 * z0) * (z <= 1.2 * z1)
p.set_support(sup)

# Go back to the unstrained object as a starting point
p.set_obj(obj0)

# Solve this
p = DM(calc_llk=1, show_obj_probe=10) ** 80 * p
p = AP(calc_llk=10, show_obj_probe=10) ** 40 * p
p = ML(calc_llk=1, show_obj_probe=10) ** 40 * p


# np.savez('/Users/favre/tmp/bragg.npz', probe3d=p._probe3d, obj=p.get_obj(), probe2d=pr.get(shift=True), x=x, y=y, z=z)
