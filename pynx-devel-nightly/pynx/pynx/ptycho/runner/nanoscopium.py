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
Script to perform a ptychography analysis on data recorded on nanoscopium@Soleil (*.nxs format)

Example:
    pynx-nanoscopiumpty.py data=flyscan_1061-0001.nxs probe=focus,700e-6,0.63 defocus=4 detectordistance=3.265 
                algorithm=analysis,ML**100,AP**400,DM**100,nprobe=3,DM**200,probe=1 saveplot mask=maxipix liveplot

command-line arguments:
    data=/some/dir/to/data.nxs: path to nxs data [mandatory]
        If the data string includes one or several '%', it is assumed these are fields
        to be replaced by the scan number, e.g. "data=flyscan_%04d-0001.nxs". This allows
        to process series of scans using scan=6642,6643 or scan="range(13,27+1)" (note the quotes
        which are necessary when using parenthesis in a parameter)

"""

params_beamline = {'instrument': 'nanoscopium@Soleil', 'detectordistance': None, 'pixelsize': 55e-6, 'mask': 'maxipix',
                   'object': 'random,0.95,1,0,0.1', 'obj_smooth': 0.5, 'obj_inertia': 0.1, 'probe_smooth': 0.2,
                   'probe_inertia': 0.01}

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanNanoscopium(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanNanoscopium, self).__init__(params, scan, timings=timings)

    def load_scan(self):
        filename = self.params['data']
        if '%' in filename:
            filename = filename % self.scan
            self.print('Loading data:', filename)
        h5data = h5py.File(filename, 'r')
        # Get the first entry and hope for the best
        entry = [v for v in h5data.values()][0]

        if 'scan_data/sample_Piezo_TX' in entry:
            self.x = entry['scan_data/sample_Piezo_TX'][()].flatten() * -1e-3
        else:
            self.x = entry['scan_data/sample_Piezo_Tx'][()].flatten() * -1e-3
        if 'scan_data/sample_Piezo_Tz' in entry:
            self.y = entry['scan_data/sample_Piezo_Tz'][()].flatten() * 1e-3
        else:
            self.y = entry['scan_data/sample_Piezo_TZ'][()].flatten() * 1e-3

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
        filename = self.params['data']
        if '%' in filename:
            filename = filename % self.scan
        h5data = h5py.File(filename, 'r')
        # Get the first entry and hope for the best
        entry = [v for v in h5data.values()][0]
        if self.params['nrj'] is None:
            # Energy in keV must be corrected (ask Kadda !)
            self.params['nrj'] = entry['NANOSCOPIUM/Monochromator/energy'][0] * 1.2371 - 1.3456

        # Load all frames
        print("Reading frames from nxs file:")
        t0 = timeit.default_timer()
        h5d = entry['scan_data/Image_merlin_image']
        vimg = None
        if len(h5d.shape) == 4:
            imgn = np.array(self.imgn, dtype=np.int32)
            if vimg is None:
                vimg = np.empty((self.x.size, h5d.shape[-2], h5d.shape[-1]))
            # flyscan data is 2D..
            n1, n2 = h5d.shape[:2]
            for i in range(n1):
                i0 = n2 * i
                idx = np.where(np.logical_and(imgn >= i0, imgn < (i0 + n2)))[0]
                if len(idx):
                    vimg[idx] = h5d[i, imgn[idx] - i0]
        else:
            vimg = h5d[self.imgn]

        dt = timeit.default_timer() - t0
        print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, vimg.size / 1e6 / dt))

        # Add gaps
        n, ny, nx = vimg.shape
        self.raw_data = np.empty((n, 516, 516))
        self.raw_data[:, :256, :256] = vimg[:, :256, :256]
        self.raw_data[:, 260:, :256] = vimg[:, 256:, :256]
        self.raw_data[:, :256, 260:] = vimg[:, :256, 256:]
        self.raw_data[:, 260:, 260:] = vimg[:, 256:, 256:]
        # TODO: check the following values - pixels should be masked anyway...
        self.raw_data[:, 256] *= 2.5
        self.raw_data[:, :, 256] *= 2.5
        self.raw_data[:, 260] *= 2.5
        self.raw_data[:, :, 260] *= 2.5

        print(self.x.shape, self.y.shape, self.raw_data.shape)

        self.load_data_post_process()


class PtychoRunnerNanoscopium(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(PtychoRunnerNanoscopium, self).__init__(argv, params, ptycho_runner_scan_class)
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
        elif k == 'nrj' or k == 'detectordistance':
            self.params[k] = float(v)
            return True
        return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['data'] is None:
            raise PtychoRunnerException('Missing argument: no data file [data=my_data_file.nxs] given')
        if self.params['detectordistance'] is None:
            raise PtychoRunnerException('Missing argument: detectordistance=xx (in meters)')
        if self.params['nrj'] is None:
            raise PtychoRunnerException('Missing argument: nrj=xx (in kev)')
