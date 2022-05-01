#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from __future__ import division

import sys
import os
import glob
import time
import locale
import timeit

from ...utils import h5py
import numpy as np

from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, params_generic

helptext_beamline = """
Script to perform a near-field ptychography analysis on data from id16(a)@ESRF

Example:
    pynx-id16pty-nfp.py data=data.h5 meta=meta.h5 probe=60e-6x60e-6,0.09
      algorithm=analysis,ML**100,DM**200,nbprobe=3,probe=1

Command-line arguments (beamline-specific):
    data=data.h5: hdf5 filename with the scan data. The scan number, energy, 
                    detector distance will be extracted 
                    [mandatory - if a single *.h5 file is in the working directory,
                     it will be used by default]

    meta=meta.h5: hdf5 filename with the scan metadata. The scan number, energy, 
                    detector distance will be extracted 
                    [mandatory - if a single *.h5 file is in the working directory,
                     it will be used by default]

    ptychomotors=spy,spz,-x,y: names of the two motors used for ptychography, optionally
                               followed by a mathematical expression to
                               calculate the actual motor positions (axis convention, angle..)
                               in SI units (meters).
                               Values will be extracted from the data file 
                               as /entry_0000/instrument/Frelon/header/motor_name.
                               Note that if the 'xy=-y,x' command-line argument is used, it is 
                               applied after this, using 'ptychomotors=spy,spz,-x,y' 
                               is equivalent to 'ptychomotors=spy,spz xy=-x,y'
                               [optional, default=spy,spz,-x*1e-6,y*1e-6]
    padding=100: padding to be added on the borders of the iobs array [default:0]
"""

# NB: for id16 we start from a flat object (high energy, high transmission)
params_beamline = {'meta': None, 'pixelsize': 55e-6, 'ptychomotors': 'spy,spz,-x*1e-6,y*1e-6',
                   'data': None, 'detectordistance': None, 'object': 'random,0.95,1,0,0.1',
                   'instrument': 'ESRF id16a', 'near_field': True, 'roi': 'full', 'maxsize': 10000,
                   'autocenter': False, 'remove_obj_phase_ramp': False, 'padding': 0}

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanID16aNF(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanID16aNF, self).__init__(params, scan, timings=timings)

    def load_scan(self):
        data_filename = self.params['data']
        if '%' in data_filename:
            data_filename = data_filename % tuple(self.scan for i in range(data_filename.count('%')))
            print('data filename for scan #%d: %s' % (self.scan, data_filename))
        data = h5py.File(data_filename, mode='r', enable_file_locking=False)
        m = self.params['ptychomotors'].split(',')
        xmot, ymot = m[0:2]
        self.x = np.array([v for v in data['entry_0000/instrument/Frelon/header/' + xmot][()].split()],
                          dtype=np.float32)
        self.y = np.array([v for v in data['entry_0000/instrument/Frelon/header/' + ymot][()].split()],
                          dtype=np.float32)

        imgn = np.arange(len(self.x), dtype=np.int32)
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

        imgn = np.array(imgn)

        if len(m) >= 3:
            x, y = self.x, self.y
            self.x, self.y = eval(m[-2]), eval(m[-1])

        if len(self.x) < 4:
            raise PtychoRunnerException("Less than 4 scan positions, is this a ptycho scan ?")

        self.imgn = imgn

    def load_data(self):
        data_filename = self.params['data']
        if '%' in data_filename:
            data_filename = data_filename % tuple(self.scan for i in range(data_filename.count('%')))
        data = h5py.File(data_filename, mode='r', enable_file_locking=False)

        meta = h5py.File(self.params['meta'], mode='r', enable_file_locking=False)
        # Entry name is obfuscated, so just take the first entry with 'entry' in the name
        meta_entry = None
        for k, v in meta.items():
            if 'entry' in k:
                meta_entry = v
                break

        if False:
            date_string = data["entry_0000/start_time"][()]  # '2020-12-12T15:29:07Z'
            if 'Z' == date_string[-1]:
                date_string = date_string[:-1]
                # TODO: for python 3.7+, use datetime.isoformat()
            if sys.version_info > (3,) and isinstance(date_string, bytes):
                date_string = date_string.decode('utf-8')  # Could also use ASCII in this case
            else:
                date_string = str(date_string)
            pattern = '%Y-%m-%dT%H:%M:%S'

            try:
                lc = locale._setlocale(locale.LC_ALL)
                locale._setlocale(locale.LC_ALL, 'C')
                epoch = int(time.mktime(time.strptime(date_string, pattern)))
                locale._setlocale(locale.LC_ALL, lc)
            except ValueError:
                print("Could not extract time from spec header, unrecognized format: %s, expected: %s" % (
                    date_string, pattern))

        self.params['nrj'] = float(meta_entry["instrument/monochromator/energy"][()])

        self.params['pixelsize'] = float(meta_entry["TOMO/pixelSize"][()]) * 1e-6
        print("Effective pixel size: %12.3fnm" % (self.params['pixelsize'] * 1e9))

        if self.scan is None:
            self.scan = 0

        # Raw positioners positions, almost guaranteed not to be SI
        positioners = {}
        tmpn = meta_entry["sample/positioners/name"][()]
        tmpv = meta_entry["sample/positioners/value"][()]
        if isinstance(tmpn, np.bytes_):
            tmpn = tmpn.decode('ascii')
            tmpv = tmpv.decode('ascii')

        for k, v in zip(tmpn.split(), tmpv.split()):
            positioners[k] = float(v)

        if self.params['detectordistance'] is None:
            sx = 1e-3 * positioners['sx']
            sx0 = 1e-3 * float(meta_entry["TOMO/sx0"][()])
            z1 = sx - sx0
            z12 = 1e-3 * float(meta_entry["PTYCHO/focusToDetectorDistance"][()])  # z1+z2
            z2 = z12 - z1
            print(sx * 1e3, sx0 * 1e3, z1 * 1e3, z12 * 1e3, z2 * 1e3)
            self.params['detectordistance'] = z1 * z2 / z12
            print("Effective propagation distance: %12.8fm" % self.params['detectordistance'])

        # read all frames
        imgn = self.imgn
        t0 = timeit.default_timer()
        print("Reading frames:")
        self.raw_data = data["/entry_0000/measurement/data"][imgn.tolist()]
        dt = timeit.default_timer() - t0
        print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' %
              (dt, self.raw_data[0].size * len(self.raw_data) / 1e6 / dt))

        self.load_data_post_process()


class PtychoRunnerID16aNF(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(PtychoRunnerID16aNF, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        Returns:
            True if the argument is interpreted, false otherwise
        """
        if k in ['meta', 'ptychomotors', 'data']:
            self.params[k] = v
            return True
        elif k in ['padding']:
            self.params[k] = int(v)
            return True
        elif k in ['pixelsize', 'detectordistance']:
            self.params[k] = float(v)
            return True
        return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['liveplot'] is True:
            # KLUDGE ?
            self.params['liveplot'] = 'object_phase'
        if self.params['meta'] is None:
            raise PtychoRunnerException('Missing argument: no h5metadata file given (meta=..)')
        if self.params['data'] is None:
            # TODO: use scan # and filename in metadata
            raise PtychoRunnerException('Missing argument: data')
        if self.params['ptychomotors'] is None:
            raise PtychoRunnerException('Missing argument: ptychomotors')
