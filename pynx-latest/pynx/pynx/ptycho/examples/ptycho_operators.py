########################################################################
#
# Example of the ptychograpic reconstruction using OpenCL on simulated data
# (c) ESRF 2017-present
# Authors: Vincent Favre-Nicolin <favre@esrf.fr>
#
########################################################################

import timeit
from pylab import *

if False:
    from pynx.processing_unit import default_processing_unit

    # This can be used to select the GPU and/or the language, and must be called before other pynx imports
    # Otherwise a default GPU will be used according to its speed, which is usually sufficient
    default_processing_unit.select_gpu(language='OpenCL', gpu_name='R9')

# The following statement will import either CUDA or OpenCL operators
from pynx.ptycho import *
from pynx.ptycho import simulation

##################
# Simulation of the ptychographic data:
n = 256
pixel_size_detector = 55e-6
wavelength = 1.5e-10
detector_distance = 1
obj_info = {'type': 'phase_ampl', 'phase_stretch': pi / 2, 'alpha_win': .2}
# probe_info = {'type': 'focus', 'aperture': (30e-6, 30e-6), 'focal_length': .08, 'defocus': 100e-6, 'shape': (n, n)}
probe_info = {'type': 'gauss', 'sigma_pix': (40, 40), 'defocus': 100e-6, 'shape': (n, n)}

# 50 scan positions correspond to 4 turns, 78 to 5 turns, 113 to 6 turns
scan_info = {'type': 'spiral', 'scan_step_pix': 30, 'n_scans': 120}
data_info = {'num_phot_max': 1e9, 'bg': 0, 'wavelength': wavelength, 'detector_distance': detector_distance,
             'detector_pixel_size': pixel_size_detector, 'noise': 'poisson'}

# Initialisation of the simulation with specified parameters
s = simulation.Simulation(obj_info=obj_info, probe_info=probe_info, scan_info=scan_info, data_info=data_info)
s.make_data()

# Positions from simulation are given in pixels
posx, posy = s.scan.values

ampl = s.amplitude.values  # square root of the measured diffraction pattern intensity

pixel_size_object = wavelength * detector_distance / pixel_size_detector / n

##################
# Size of the reconstructed object (obj)
nyo, nxo = shape.calc_obj_shape(posx, posy, ampl.shape[1:])

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

data = PtychoData(iobs=ampl ** 2, positions=(posx * pixel_size_object, posy * pixel_size_object), detector_distance=1,
                  mask=None, pixel_size_detector=55e-6, wavelength=1.5e-10)

p = Ptycho(probe=s.probe.values, obj=init.obj.values, data=data, background=None)

# Initial scaling is important to avoid overflows during ML
p = ScaleObjProbe(verbose=True) * p
p = DM(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20) ** 100 * p
p = ML(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20) ** 40 * p

if True:
    print("###########################################################################################################")
    # Add probe modes
    pr = p.get_probe()
    ny, nx = pr.shape[-2:]
    nb_probe = 3
    pr1 = np.empty((nb_probe, ny, nx), dtype=np.complex64)
    pr1[0] = pr[0]
    for i in range(1, nb_probe):
        n = abs(pr).mean()
        pr1[i] = np.random.uniform(0, n, (ny, nx)) * exp(1j * np.random.uniform(0, 2 * np.pi, (ny, nx)))

    p.set_probe(pr1)

    p = DM(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20) ** 100 * p
    p = AP(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20) ** 40 * p
    p = ML(update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20) ** 100 * p

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

if False:
    # Export data and object, probe to CXI files
    p.save_obj_probe_cxi('obj_probe.cxi')
    save_ptycho_data_cxi('data.cxi', ampl ** 2, pixel_size_detector, wavelength, detector_distance,
                         posx * pixel_size_object, posy * pixel_size_object, z=None, monitor=None,
                         mask=None, instrument='simulation', overwrite=True)
