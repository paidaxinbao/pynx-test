#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
from __future__ import division

import timeit
import numpy as np
from ...ptycho import simulation, Ptycho, PtychoData, Calc2Obs, AP, DM, ScaleObjProbe, save_ptycho_data_cxi
from ..shape import calc_obj_shape
from ...mpi import MPI

if MPI is not None:
    from ..mpi import PtychoSplit, ShowObjProbe
from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, params_generic

helptext_beamline = """
Script to perform a ptychography analysis on a simulated dataset

Examples:
    pynx-simulationpty.py frame_nb=128 frame_size=256 algorithm=analysis,ML**100,DM**200,nbprobe=2,probe=1 
                          saveplot liveplot

command-line arguments:
    frame_nb=128: number of simulated frames (will be generated along a spiral)
    
    frame_size=256: size along x and y of each frame
    
    siemens: if given on the command-line, a siemens star will be used for the object, instead of using two
             different 512x512 zoomed base images for the amplitude and phase.

    logo: if given on the command-line, a logo with 'PyNX' will be used for the object.
    
    asym: if given on the command-line, all but 1 scan positions in the lower right quadrant
          (x>x.mean(), y<y.mean()) will be removed. Positions are added in the other
          parts to keep the number of frames as expected. The probe used for simulation
          will also have the lower right quadrant set to zero.
    
    simul_background=0.1: add incoherent (gaussian-shaped) background, with an integrated intensity
        equal to the given factor multiplied by the simulated intensity. Reasonable values up to 0.2
        [default:0 - no background]
"""

# NOTE: scripts to test absolute orientation:
# mpiexec -n 4 pynx-simulationpty.py logo asym liveplot saveplot mpi=split
#  frame_nb=400 frame_size=256 algorithm=analysis,AP**20,positions=1,AP**400,probe=1
# pynx-simulationpty.py logo asym liveplot saveplot frame_nb=200 frame_size=256
#   algorithm=analysis,AP**20,positions=1,AP**400,probe=1
#
# In both cases compare:
# - positions plot with lone illumination position
# - absolute orientation of object and probe in saved plot, silx view of object, illumination and probe


params_beamline = {'frame_nb': 128, 'frame_size': 256, 'probe': 'auto', 'siemens': False,
                   'logo': False, 'asym': False, 'defocus': 100e-6,
                   'algorithm': 'ML**100,AP**200,DM**200,probe=1', 'instrument': 'simulation',
                   'nrj': 8, 'detectordistance': 1, 'pixelsize': 55e-6, 'maxsize': None,
                   'roi': 'full', 'autocenter': False, 'saveprefix': 'none', 'simul_background': 0}

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanSimul(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanSimul, self).__init__(params, scan, timings=timings)

    def load_scan(self):
        nb = self.params['frame_nb']
        n = self.params['frame_size']
        pixel_size_detector = self.params['pixelsize']
        wavelength = 12.3984e-10 / self.params['nrj']
        detector_distance = self.params['detectordistance']
        # 50 scan positions correspond to 4 turns, 78 to 5 turns, 113 to 6 turns
        scan_info = {'type': 'spiral', 'scan_step_pix': 0.1 * n, 'n_scans': nb}

        s = simulation.Simulation(obj_info=None, probe_info=None, scan_info=scan_info, data_info=None,
                                  verbose=self.mpi_master)
        s.make_scan()

        if self.params['near_field']:
            pixel_size_object = pixel_size_detector
        else:
            pixel_size_object = wavelength * detector_distance / pixel_size_detector / n
        self.x, self.y = s.scan.values[0] * pixel_size_object, s.scan.values[1] * pixel_size_object
        self.imgn = np.arange(nb, dtype=np.int32)
        if self.params['asym']:
            idx = np.where(np.logical_or(self.x <= self.x.mean(), self.y >= self.y.mean()))[0]
            # Keep one illumination farthest in the corner
            idx = np.append(idx, [np.argmax(self.x - self.y)])
            self.x, self.y = np.take(self.x, idx), np.take(self.y, idx)
            # Append existing positions to keep total number of frames
            dn = nb - len(self.x)
            idx = np.random.randint(0, len(self.x), dn)
            self.x = np.append(self.x, self.x[idx])
            self.y = np.append(self.y, self.y[idx])

            # Also off-center positions to check absolute position
            self.x += 2 * np.abs(self.x).max()
            self.y += 3 * np.abs(self.y).max()

    def load_data(self):
        t0 = timeit.default_timer()
        n = self.params['frame_size']
        pixel_size_detector = self.params['pixelsize']
        wavelength = 12.3984e-10 / self.params['nrj']
        detector_distance = self.params['detectordistance']
        if self.params['siemens']:
            obj_info = {'type': 'siemens', 'alpha_win': .2}
        elif self.params['logo']:
            obj_info = {'type': 'logo', 'phase_stretch': 1, 'ampl_range': (0.9, 0.1)}
        else:
            obj_info = {'type': 'phase_ampl', 'phase_stretch': np.pi / 2, 'alpha_win': .2}

        near_field = self.params['near_field']
        if near_field:
            pixel_size_object = pixel_size_detector
        else:
            # The probe is calculated so that it will defocus to about 40% of the object frame size,
            # and the step size is adapted accordingly
            pixel_size_object = wavelength * detector_distance / (n * pixel_size_detector)
        aperture = 400e-6
        focal_length = 0.1
        defocus = 0.4 * pixel_size_object * n / aperture * focal_length
        if near_field:
            ap = pixel_size_detector * n * 0.7
            probe_info = {'type': 'near_field', 'aperture': (ap, ap), 'defocus': 0.3, 'shape': (n, n)}
        else:
            probe_info = {'type': 'focus', 'aperture': (aperture, aperture), 'focal_length': focal_length,
                          'defocus': defocus, 'shape': (n, n)}
            if self.params['probe'] == 'auto':
                # Put some randomness in the starting probe
                r = np.random.uniform(0.9, 1.1, 4)
                self.params['probe'] = 'focus,%ex%e,%e' % (aperture * r[0], aperture * r[1], focal_length * r[2])
                self.params['defocus'] = defocus * r[3]

        # 50 scan positions correspond to 4 turns, 78 to 5 turns, 113 to 6 turns
        scan_info = {'type': 'custom', 'x': self.x / pixel_size_object, 'y': self.y / pixel_size_object}
        data_info = {'num_phot_max': 1e6, 'bg': 0, 'wavelength': wavelength, 'detector_distance': detector_distance,
                     'detector_pixel_size': pixel_size_detector, 'noise': 'poisson', 'near_field': near_field}

        s = simulation.Simulation(obj_info=obj_info, probe_info=probe_info, scan_info=scan_info,
                                  data_info=data_info, verbose=self.mpi_master)
        s.make_probe()
        probe0 = s.probe.values

        t1 = timeit.default_timer()
        if self.timings is not None:
            dt = t1 - t0
            if "load_data_simul_probe" in self.timings:
                self.timings["load_data_simul_probe"] += dt
            else:
                self.timings["load_data_simul_probe"] = dt

        data = PtychoData(iobs=np.ones((len(self.x), n, n), dtype=np.float32), positions=(self.x, self.y),
                          detector_distance=detector_distance, mask=None, pixel_size_detector=pixel_size_detector,
                          wavelength=wavelength, near_field=near_field)

        if 'split' not in self.params['mpi']:
            s.make_obj()
            s.make_obj_true(data.get_required_obj_shape(margin=2))
            p = Ptycho(probe=probe0, obj=s.obj.values, data=data, background=None)
        else:
            p = PtychoSplit(probe=probe0, obj=None, data=data, background=None,
                            mpi_neighbour_xy=self.mpi_neighbour_xy)
            if self.mpi_master:
                s.make_obj()
                # The object shape actually needed is computed in PtychoSplit.init_mpi_obj()
                s.make_obj_true(p.mpi_obj.shape[-2:])
            p.set_mpi_obj(s.obj.values)

        if 'asym' in self.params:
            if self.params['asym']:
                # Use asymmetry to check the absolute orientation of the reconstruction
                x, y = p.get_probe_coord()
                y, x = np.meshgrid(y, x, indexing='ij')
                px, py = data.pixel_size_object()
                tmp = np.logical_or(x <= 0, y >= 0).astype(np.int8)
                probe0 *= tmp + (1 - tmp) * np.exp(np.maximum(y, -x) / (px * len(x) / 32))
                p.set_probe(probe0)

        t2 = timeit.default_timer()
        if self.timings is not None:
            dt = t2 - t1
            if "load_data_simul_obj" in self.timings:
                self.timings["load_data_simul_obj"] += dt
            else:
                self.timings["load_data_simul_obj"] = dt

        p = Calc2Obs(nb_photons_per_frame=1e8) * p
        iobs = np.fft.fftshift(p.data.iobs, axes=(1, 2))

        dark = None
        if self.params['simul_background'] > 0:
            x, y = p.get_probe_coord()
            dx, dy = np.meshgrid((x - x.mean()) / (x.max() - x.min()), (y - y.mean()) / (y.max() - y.min()))
            tmp = np.exp(-5 * (dx ** 2 + dy ** 2))
            dark = iobs.sum() / (tmp.sum() * len(iobs)) * self.params['simul_background'] * tmp
            iobs += dark

        t3 = timeit.default_timer()
        if self.timings is not None:
            dt = t3 - t2
            if "load_data_simul_iobs" in self.timings:
                self.timings["load_data_simul_iobs"] += dt
            else:
                self.timings["load_data_simul_iobs"] = dt

        if 'split' in self.params['mpi']:
            vnb = self.mpic.gather(len(iobs), root=0)
            viobs_sum = self.mpic.gather(iobs.sum(), root=0)
            if self.mpi_master:
                print(vnb, viobs_sum)
                scale = self.mpic.bcast(np.array(viobs_sum).sum() / np.array(vnb).sum(), root=0)
            else:
                scale = self.mpic.bcast(None, root=0)
        else:
            scale = iobs.sum() / len(iobs)

        # 1e8 photons on average
        self.raw_data = np.random.poisson(iobs / scale * 1e8)
        if dark is not None:
            dark = dark * (1e8 / scale)
            print("Including dark with <dark>=%6.0f ph/pixel, <Iobs>=%6.0f" % (dark.mean(), self.raw_data.mean()))
            self.dark = dark

        t4 = timeit.default_timer()
        if self.timings is not None:
            dt = t4 - t3
            if "load_data_simul_noise" in self.timings:
                self.timings["load_data_simul_noise"] += dt
            else:
                self.timings["load_data_simul_noise"] = dt

        # if self.mpi_master:
        #     print("RunnerScanSimul: elapsed time for load_data()= %5.2fs" % (time.time()-t0))
        self.load_data_post_process()


class PtychoRunnerSimul(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(PtychoRunnerSimul, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        This method should be superseded in a beamline/instrument-specific child class
        Returns:
            True if the argument is interpreted, false otherwise
        """
        if k in ['frame_nb', 'frame_size', 'padding']:
            self.params[k] = int(v)
            return True
        elif k in ['siemens', 'logo', 'asym', 'near_field']:
            self.params[k] = True
            return True
        elif k in ['simul_background']:
            self.params[k] = float(v)
            return True
        return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['pixelsize'] == 55e-6 and self.params['near_field']:
            self.params['pixelsize'] = 100e-9
        pass
