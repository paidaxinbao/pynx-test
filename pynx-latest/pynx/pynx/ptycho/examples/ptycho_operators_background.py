########################################################################
#
# Example of the ptychograpic reconstruction using OpenCL on simulated data and incoherent background
# (c) ESRF 2017-present
# Authors: Vincent Favre-Nicolin <favre@esrf.fr>
#
########################################################################

import timeit
from pylab import *
from pynx.ptycho import simulation, shape

# This will import the base Ptycho object as well as operators (CPU, CUDA or OpenCL chosen automatically)
from pynx.ptycho import *

##################
# Simulation of the ptychographic data:
n = 256
pixel_size_detector = 55e-6
wavelength = 1.5e-10
detector_distance = 1
obj_info = {'type': 'phase_ampl', 'phase_stretch': pi / 2, 'alpha_win': .2}
probe_info = {'type': 'focus', 'aperture': (60e-6, 200e-6), 'focal_length': .08, 'defocus': 100e-6, 'shape': (n, n)}
#probe_info = {'type': 'gauss', 'sigma_pix': (40, 40), 'defocus': 100e-6, 'shape': (n, n)}

# 50 scan positions correspond to 4 turns, 78 to 5 turns, 113 to 6 turns
scan_info = {'type': 'spiral', 'scan_step_pix': 30, 'n_scans': 120}
data_info = {'num_phot_max': 1e9, 'bg': 0, 'wavelength': wavelength, 'detector_distance': detector_distance,
             'detector_pixel_size': pixel_size_detector, 'noise': 'poisson'}

# Initialisation of the simulation with specified parameters
s = simulation.Simulation(obj_info=obj_info, probe_info=probe_info, scan_info=scan_info, data_info=data_info)
s.make_data()

# Positions from simulation are given in pixels
posx, posy = s.scan.values

iobs = s.amplitude.values**2  # square root of the measured diffraction pattern intensity

# Add some background
iobs_mean = iobs.sum(axis=0).mean()
background = simulation.gauss2D((n, n), sigma=(n / 4, n / 4))
background += background.max() / 4
background *= iobs_mean * 0.05 / background.sum()
# Use Poisson statistics for the detector
iobs = np.random.poisson(iobs + background)

pixel_size_object = wavelength * detector_distance / pixel_size_detector / n

##################
# Size of the reconstructed object (obj)
nyo, nxo = shape.calc_obj_shape(posx, posy, iobs.shape[1:])

# Initial object
# obj_init_info = {'type':'flat','shape':(nx,ny)}
obj_init_info = {'type': 'random', 'range': (0, 1, 0, 0.5), 'shape': (nyo, nxo)}
# Initial probe
probe_init_info = {'type': 'focus', 'aperture': (20e-6, 20e-6), 'focal_length': .08, 'defocus': 50e-6, 'shape': (n, n)}
data_info = {'wavelength': wavelength, 'detector_distance': detector_distance,
             'detector_pixel_size': pixel_size_detector}
init = simulation.Simulation(obj_info=obj_init_info, probe_info=probe_init_info, data_info=data_info)

init.make_obj()
init.make_probe()

data = PtychoData(iobs=iobs, positions=(posx * pixel_size_object, posy * pixel_size_object), detector_distance=1,
                  mask=None, pixel_size_detector=55e-6, wavelength=1.5e-10)

p = Ptycho(probe=s.probe.values, obj=init.obj.values, data=data, background=None)

# Initial scaling is important to avoid overflows during ML
show = 20
p = ScaleObjProbe(verbose=True) * p
p = DM(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=show) ** 100 * p
p = AP(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=show) ** 100 * p
p = AP(update_object=True, update_probe=True, update_background=True, calc_llk=20, show_obj_probe=show) ** 100 * p
p = ML(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=show) ** 40 * p
p = AP(update_object=True, update_probe=True, update_background=True, calc_llk=20, show_obj_probe=show) ** 100 * p
p = ML(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=show) ** 40 * p
