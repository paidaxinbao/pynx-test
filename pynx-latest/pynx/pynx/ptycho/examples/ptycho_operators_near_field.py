########################################################################
#
# Example of the ptychograpic reconstruction using OpenCL on simulated data
# (c) ESRF 2017-present
# Authors: Vincent Favre-Nicolin <favre@esrf.fr>
#
########################################################################

import timeit
from pylab import *
from pynx.ptycho import simulation, shape

# from pynx.ptycho import *
from pynx.ptycho.ptycho import *
from pynx.ptycho.cpu_operator import *  # Use only CPU operators ?
from pynx.ptycho.operator import *  # Use CUDA > OpenCL > CPU operators, as available
import pynx.ptycho.cpu_operator as cpuop
import pynx.ptycho.cl_operator as clop

##################
detector_distance = 1.5
wavelength = 1.5e-10
pixel_size_detector = 1e-6
# Simulation of the ptychographic data:
n = 256
obj_info = {'type': 'phase_ampl', 'phase_stretch': pi / 2, 'alpha_win': .2}
probe_info = {'type': 'near_field', 'aperture': (80e-6, 80e-6), 'defocus': 0.5, 'shape': (n, n)}

# 50 scan positions correspond to 4 turns, 78 to 5 turns, 113 to 6 turns
scan_info = {'type': 'spiral', 'scan_step_pix': 20, 'n_scans': 120}
data_info = {'num_phot_max': 1e9, 'bg': 0, 'wavelength': wavelength, 'detector_distance': detector_distance,
             'detector_pixel_size': pixel_size_detector, 'noise': 'poisson', 'near_field': True}

# Initialisation of the simulation
s = simulation.Simulation(obj_info=obj_info, probe_info=probe_info, scan_info=scan_info, data_info=data_info)
s.make_data()

# Positions from simulation are given in pixels
posx, posy = s.scan.values
ampl = s.amplitude.values  # square root of the measured diffraction pattern intensity

##################
# Size of the reconstructed object (obj)
nyo, nxo = shape.calc_obj_shape(posx, posy, ampl.shape[1:])

# Initial object
# obj_init_info = {'type':'flat','shape':(nx,ny)}
obj_init_info = {'type': 'random', 'range': (0, 1, 0, 0.5), 'shape': (nyo, nxo)}
# Initial probe
probe_init_info = {'type': 'near_field', 'aperture': (90e-6, 90e-6), 'defocus': 0.3, 'shape': (n, n)}
init = simulation.Simulation(obj_info=obj_init_info, probe_info=probe_init_info, data_info=data_info)

init.make_obj()
init.make_probe()

data = PtychoData(iobs=ampl ** 2, positions=(posx * pixel_size_detector, posy * pixel_size_detector),
                  detector_distance=detector_distance, mask=None,
                  pixel_size_detector=pixel_size_detector, wavelength=wavelength, near_field=True)

p = Ptycho(probe=init.probe.values, obj=init.obj.values, data=data, background=None)

# Initial scaling is important to avoid overflows during ML
p = ScaleObjProbe(verbose=True) * p
p = DM(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20) ** 100 * p
p = AP(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20) ** 40 * p
p = ML(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20) ** 40 * p


if False:
    # Timing vs stack size
    n = 50
    vx = []
    vy = []
    for stack_size in range(1, 32 + 1, 1):
        default_processing_unit.set_stack_size(stack_size)
        p = DM(update_object=True, update_probe=True) ** 10 * p
        t0 = timeit.default_timer()
        p = DM(update_object=True, update_probe=True) ** n * p
        dt = (timeit.default_timer() - t0) / n
        print("DM dt/cycle=%5.3fs [stack_size=%2d]" % (dt, stack_size))
        vx.append(stack_size)
        vy.append(dt)
    figure()
    plot(vx, vy, '-')
    xlabel('stack size')
    ylabel('Time for a DM cycle (s)')
