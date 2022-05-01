#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from __future__ import division

import sys
import time
import locale
import timeit

from ...utils import h5py
import numpy as np

from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, params_generic

helptext_beamline = """
Script to perform a ptychography analysis on data from 25A@TPS

Example:
    pynx-tps25apty.py h5file=pty_41_61_master.h5 specfile=scan.dat probe=gauss,3e-6x3e-6
      algorithm=analysis,ML**100,DM**200,nbprobe=3,probe=1 verbose=10 save=all saveplot liveplot

Command-line arguments (beamline-specific):
    scanfile=/some/dir/to/scan.txt: path to text file with motor positions [mandatory]

    data=path/to/data.h5: path to hdf5 data file, with images, wavelength, detector distance [mandatory]
    
    h5data=entry/data/data_%06d: relative path to data inside hdf5 file [default=entry/data/data_%06d]
"""

params_beamline = {'scanfile': None, 'data': None, 'object': 'random,0.9,1,-.2,.2', 'instrument': 'TPS 25A',
                   'h5data': 'entry/data/data_%06d'}

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanTPS25A(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanTPS25A, self).__init__(params, scan, timings=timings)

    def load_scan(self):
        print("reading motor positions from: ", self.params['scanfile'])
        self.x, self.y = np.loadtxt('scan_2670.txt', delimiter=',', skiprows=1, unpack=True, usecols=(1, 2))

        self.imgn = np.arange(len(self.x), dtype=np.int)

        read_all_frames = True
        if self.params['moduloframe'] is not None:
            n1, n2 = self.params['moduloframe']
            idx = np.where(self.imgn % n1 == n2)[0]
            self.imgn = self.imgn.take(idx)
            self.x = self.x.take(idx)
            self.y = self.y.take(idx)

        if self.params['maxframe'] is not None:
            N = self.params['maxframe']
            if len(self.imgn) > N:
                print("MAXFRAME: only using first %d frames" % (N))
                self.imgn = self.imgn[:N]
                self.x = self.x[:N]
                self.y = self.y[:N]

    def load_data(self):
        self.h5 = h5py.File(self.params['data'], 'r')

        if self.params['pixelsize'] is None:
            self.params['pixelsize'] = np.array(self.h5.get('entry/instrument/detector/x_pixel_size'))
            print("Pixelsize?", self.params['pixelsize'])

        self.params['detectordistance'] = np.array(self.h5.get('entry/instrument/detector/detector_distance'))

        if self.params['nrj'] is None:
            assert (self.h5.get("entry/instrument/beam/incident_wavelength")[()] > 0.1)
            self.params['nrj'] = 12.3984 / self.h5.get("entry/instrument/beam/incident_wavelength")[()]

        if len(self.x) < 4:
            raise PtychoRunnerException("Less than 4 scan positions, is this a ptycho scan ?")
        # Spec values are in microns, convert to meters
        self.x *= 1e-6
        self.y *= 1e-6

        # Load all frames
        t0 = timeit.default_timer()

        self.print('Reading HDF5 frames: ')
        # frames are grouped in different subentries
        entry0 = 1
        ct = 0
        i0 = 0
        vimg = None
        # NB: it may be faster/easier to create a virtual dataset ?
        while True:
            h5entry = self.params["h5data"] % entry0
            if h5entry not in self.h5:
                break
            h5d = self.h5[h5entry]
            nb = len(h5d)

            # Read all suitable frames
            idx = np.where(np.logical_and(self.imgn >= i0, self.imgn < (i0 + nb)))[0]
            if len(idx):
                self.print("Reading h5 data entry: %s [%d frames]" % (h5entry, len(idx)))
                if vimg is None:
                    # Init array
                    frame = np.array(h5d[0])
                    vimg = np.empty((len(self.imgn), frame.shape[0], frame.shape[1]), dtype=frame.dtype)
                vimg[idx] = h5d[self.imgn[idx] - i0]
                ct += len(idx)

            entry0 += 1
            i0 += nb
        if ct != len(self.imgn):
            raise PtychoRunnerException("Could not read the expected number of frames: %d < %d" % (ct, len(self.imgn)))

        dt = timeit.default_timer() - t0
        print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, vimg.size / 1e6 / dt))

        # Build mask. Values of 2**32-1 and -2 are invalid (module gaps or invalidated pixels)
        # TODO: mask independently the different frames
        self.raw_mask = ((vimg > (2 ** 32 - 3)).sum(axis=0) > 0).astype(np.int8)
        if self.raw_mask.sum() == 0:
            self.raw_mask = None
        else:
            self.print("\nMASKING %d pixels from detector flags" % (self.raw_mask.sum()))

        self.raw_data = vimg
        self.load_data_post_process()


class PtychoRunnerTPS25A(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(PtychoRunnerTPS25A, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        This method should be superseded in a beamline/instrument-specific child class
        Returns:
            True if the argument is interpreted, false otherwise
        """
        if k in ['scanfile', 'data', 'h5data']:
            self.params[k] = v
            return True
        return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['scanfile'] is None:
            raise PtychoRunnerException('Missing argument: no scanfile given')
        if self.params['data'] is None:
            raise PtychoRunnerException('Missing argument: no data (hdf5 master) given')
