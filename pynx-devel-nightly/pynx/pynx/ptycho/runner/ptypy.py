#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
from __future__ import division

import sys
import timeit

from ...utils import h5py
import numpy as np

from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, params_generic

helptext_beamline = """
Script to perform a ptychography analysis on data recorded in ptypy (*.ptyd) format (in test !)

Example:
    pynx-ptypy.py data=data.ptyd probe=focus,60e-6x200e-6,0.09 
      algorithm=analysis,ML**100,DM**200,nbprobe=3,probe=1 verbose=10 save=all saveplot liveplot

command-line arguments:
    data=/some/dir/to/data.ptyd: path to ptypy data [mandatory]

    scan=56: scan number 
           [optional: this will just be used to name the output directory accordingly, otherwise 0]
    
    gpu=Titan: GPU name for OpenCL calculation
               [default = will be auto-selected from available GPUs]

    xy=y,x: order and expression to be used for the XY positions (e.g. '-x,y',...). 
            Mathematical operations can also
            be used, e.g.: xy=0.5*x+0.732*y,0.732*x-0.5*y
            [default: 'y,x' will follow ptypy convention]
"""

params_beamline = {'xy': 'y,x'}

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanPtyPy(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanPtyPy, self).__init__(params, scan, timings=timings)

    def load_scan(self):
        h5data = h5py.File(self.params['data'], 'r')
        # TODO: handle multiple chunks
        tmp = h5data['/chunks/0/positions'][()]

        if True:
            self.x, self.y = tmp[:, 0], tmp[:, 1]
            do_transpose, do_flipud, do_fliplr = False, False, False

        if len(self.x) < 4:
            raise PtychoRunnerException("Less than 4 scan positions, is this a ptycho scan ?")

        imgn = np.arange(len(self.x), dtype=np.int)

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
        self.imgn = imgn

    def load_data(self):
        imgn = self.imgn
        h5data = h5py.File(self.params['data'], 'r')
        self.params['instrument'] = self.params['data']
        self.params['nrj'] = h5data['/info/energy'][()]  # Energy in keV
        self.params['detectordistance'] = h5data['/info/distance'][()]

        self.params['pixelsize'] = h5data['/info/psize'][()]
        if np.isscalar(h5data['/info/rebin'][()]):
            self.params['pixelsize'] *= h5data['/info/rebin'][()]

        # Load all frames
        print("Reading frames from PtyPy file:")
        ii = 0
        t0 = timeit.default_timer()
        vimg = h5data['/chunks/0/data'][imgn.tolist()]

        dt = timeit.default_timer() - t0
        print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, vimg.size / 1e6 / dt))

        self.raw_data = vimg
        self.load_data_post_process()


class PtychoRunnerPtyPy(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(PtychoRunnerPtyPy, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        This method should be superseded in a beamline/instrument-specific child class
        Returns:
            True if the argument is interpreted, false otherwise
        """
        if k in ['data']:
            self.params[k] = v
            return True
        return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['data'] is None:
            raise PtychoRunnerException('Missing argument: no data file [data=my_data_file.ptyd given')
