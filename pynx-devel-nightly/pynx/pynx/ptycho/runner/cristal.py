#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from __future__ import division

import sys
import os
import time
import locale
import timeit

from ...utils import h5py
import numpy as np

import silx

from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, params_generic

helptext_beamline = """
Script to perform a ptychography analysis on data from cristal@Soleil

Example:
    pynx-cristalpty.py data=my_nexus_file.nxs h5data=/a/scan_data/data_04 
      ptychomotors=/a/scan_data/trajectory_1_1,/a/scan_data/trajectory_1_2 
      detectordistance=1.3 nrj=8 xy=x*1e-12,y*1e-12 probe=60e-6x200e-6,0.09 
      algorithm=analysis,ML**100,DM**200,nbprobe=3,probe=1 verbose=10 save=all saveplot liveplot

Command-line arguments (beamline-specific):
    data=/some/dir/to/data.nxs: path to nexus data file [mandatory]

    h5data=/a/scan_data/data_04: hdf5 path to stack of 2d images inside nexus file [default=/a/scan_data/data_04]
    
    ptychomotors=/a/scan_data/trajectory_1_1,/a/scan_data/trajectory_1_2: hdf5 paths to the motor positions of the
          ptychography scan. Note that the orientation and scaling of these motors must be given by the "xy" parameter
          [default:ptychomotors=/a/scan_data/trajectory_1_1,/a/scan_data/trajectory_1_2]

    nrj=8: energy in keV [mandatory]

    detectordistance=1.3: detector distance in meters [mandatory]

    pixelsize=55e-6: pixel size on detector in meters [default: 55e-6]

    monitor=/a/scan_data/monitor: hdf5 paths to the monitor value. The frames will be normalized by the ratio of the 
                                  counter value divided by the median value of the counter over the entire scan 
                                  (so as to remain close to Poisson statistics). A monitor intensity lower than 10% of 
                                  the median value will be interpreted as an image taken without beam and will be 
                                  skipped.
                                  [default = None]
"""

params_beamline = {'data': None, 'h5data': '/a/scan_data/data_04', 'nrj': None, 'detectordistance': None,
                   'pixelsize': 55e-6, 'monitor': None,
                   'ptychomotors': '/a/scan_data/trajectory_1_1,/a/scan_data/trajectory_1_2',
                   'instrument': 'Cristal@Soleil'}

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanCristal(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanCristal, self).__init__(params, scan, timings=timings)

    def load_scan(self):
        h5 = h5py.File(self.params['data'], 'r')

        xmot, ymot = self.params['ptychomotors'].split(',')[0:2]

        self.x, self.y = np.array(h5[xmot][()]), np.array(h5[ymot][()])

        imgn = np.arange(len(self.x), dtype=np.int)

        if self.params['monitor'] is not None:
            mon = np.array(h5[self.params['monitor']][()])
            mon0 = np.median(mon)
            mon /= mon0
            self.validframes = np.where(mon > 0.1)
            if len(self.validframes) != len(mon):
                print(
                    'WARNING: The following frames have a monitor value < 0.1 the median value and will be ignored (no beam ?)')
                print(np.where(mon <= (mon0 * 0.1)))
            self.x = np.take(self.x, self.validframes)
            self.y = np.take(self.y, self.validframes)
            imgn = np.take(imgn, self.validframes)
        else:
            mon = None

        if self.params['moduloframe'] is not None:
            n1, n2 = self.params['moduloframe']
            idx = np.where(imgn % n1 == n2)[0]
            imgn = imgn.take(idx)
            self.x = self.x.take(idx)
            self.y = self.y.take(idx)

        if self.params['maxframe'] is not None:
            N = self.params['maxframe']
            if len(imgn) > N:
                print("MAXFRAME: only using first %d frames" % (N))
                imgn = imgn[:N]
                self.x = self.x[:N]
                self.y = self.y[:N]
        h5.close()
        self.imgn = imgn

    def load_data(self):
        h5 = h5py.File(self.params['data'], 'r')
        # Load all frames
        imgn = self.imgn
        t0 = timeit.default_timer()
        vimg = None
        d0 = 0

        sys.stdout.write('Reading HDF5 frames: ')
        # frames are grouped in different subentries
        i0 = 0
        entry0 = 1
        h5entry = self.params["h5data"]
        print("\nReading h5 data entry: %s" % (h5entry))
        h5d = np.array(h5[h5entry][()])
        ii = 0
        for i in imgn:
            if (i - imgn[0]) % 20 == 0:
                sys.stdout.write('%d ' % (i - imgn[0]))
                sys.stdout.flush()
            # Load all frames
            if i >= (i0 + len(h5d)):
                # Read next data pack
                i0 += len(h5d)
                entry0 += 1
                h5entry = self.params["h5data"] % entry0
                print("\nReading h5 data entry: %s" % (h5entry))
                h5d = np.array(h5[h5entry][()])
            frame = h5d[i - i0]
            if vimg is None:
                vimg = np.empty((len(imgn), frame.shape[0], frame.shape[1]), dtype=frame.dtype)
            vimg[ii] = frame
            d0 += frame
            ii += 1
        print("\n")
        dt = timeit.default_timer() - t0
        print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, d0.size * len(vimg) / 1e6 / dt))
        if self.raw_mask is not None:
            if self.raw_mask.sum() > 0:
                print("\nMASKING %d pixels from detector flags" % (self.raw_mask.sum()))

        self.raw_data = vimg
        self.load_data_post_process()


class PtychoRunnerCristal(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(PtychoRunnerCristal, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        This method should be superseded in a beamline/instrument-specific child class
        Returns:
            True if the argument is interpreted, false otherwise
        """
        if k == 'data' or k == 'h5data' or k == 'ptychomotors':
            self.params[k] = v
            return True
        elif k == 'nrj' or k == 'detectordistance' or k == 'pixelsize':
            self.params[k] = float(v)
            return True
        return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['data'] is None:
            raise PtychoRunnerException('Missing argument: no data given')
        if self.params['h5data'] is None:
            raise PtychoRunnerException('Missing argument: no h5data given')
        if self.params['detectordistance'] is None:
            raise PtychoRunnerException('Missing argument: detectordistance')
        if self.params['nrj'] is None:
            raise PtychoRunnerException('Missing argument: nrj (keV)')
        if self.params['ptychomotors'] is None:
            raise PtychoRunnerException('Missing argument: ptychomotors')
        # Set default fileprefix
        if self.params['saveprefix'] is None:
            s = os.path.splitext(os.path.split(self.params['data'])[-1])[0]
            self.params['saveprefix'] = s + "-%04d/Run%04d"
            print("No saveprefix given, using: ", self.params['saveprefix'])

