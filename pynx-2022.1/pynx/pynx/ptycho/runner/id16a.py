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

import fabio
from ...utils import h5py
import numpy as np

from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, params_generic

helptext_beamline = """
Script to perform a ptychography analysis on data from id16(a)@ESRF

Example:
    pynx-id16pty.py h5meta=data.h5 probe=60e-6x60e-6,0.09
      algorithm=analysis,ML**100,DM**200,nbprobe=3,probe=1

Command-line arguments (beamline-specific):
    h5meta=data.h5: hdf5 filename with the scan metadata. The scan number, energy, 
                    detector distance will be extracted 
                    [mandatory - if a single *.h5 file is in the working directory,
                     it will be used by default]

    pixelsize=55e-6: pixel size on detector in meters
                     [default: 55e-6 or will be read frommetadata or hdf5 data]

    imgname=/dir/to/images/prefix%05d.edf: images location with mask. Images will be read starting
                                           from 0 or 1, until no more frames are found, or maxframe
                                           has been reached.
                                           To process several scans, several % fields can be added, the last
                                           will be interpreted as the image number and the others replaced by the
                                           scan number. 
                                           Examples:
                                              /dir/to/scan_%04d/prefix_scan%04d_img%05d.edf 
                                              /dir/to/scan_%04d/prefix__img%05d.edf 
                                              /dir/to/images/prefix_img%05d.edf
                                           [mandatory if h5data is not supplied]
                                           
    
    h5data=/path/to/hdf5/data/file.nxs: path to nexus data file (lambda detector).
                                     If one or several % field are included in the name, they will be replaced by
                                     the scan number.
                                     Examples: 
                                        h5data=/path/to/hdf5/data/scan%d/scan%d.nxs
                                        h5data=/path/to/hdf5/data/scan%d.nxs
                                        h5data=/path/to/hdf5/data/data.nxs
                                     [mandatory unless imgname is given, will default 
                                      to the only *.nxs file in directory]

    ptychomotors=spy,spz,-x,y: name of the two motors used for ptychography, or the motor 
                               positions file, optionally followed by a mathematical expression to
                               be used to calculate the actual motor positions (axis convention, angle..).
                               Values are either:
                                   - extracted from the edf files, and are assumed to be in mm.
                                   - or extracted from the 'mot_pos.txt' file, and assumed to 
                                     be in microns..
                               Example 1: ptychomotors=spy,spz
                               Example 2: ptychomotors=spy,spz,-x,y
                               Example 3: ptychomotors=mot_pos.txt
                               Note that if the 'xy=-y,x' command-line argument is used, it is 
                               applied after this, using 'ptychomotors=spy,spz,-x,y' 
                               is equivalent to 'ptychomotors=spy,spz xy=-x,y'
                               [Mandatory]

    detectordistance=1.3: detector distance in meters [default: will be read from hdf5 metadata]

    gpu=Titan: GPU name for OpenCL calculation [Mandatory]
"""

# NB: for id16 we start from a flat object (high energy, high transmission)
params_beamline = {'h5meta': None, 'pixelsize': 55e-6, 'monitor': None, 'ptychomotors': None, 'imgname': None,
                   'h5data': None, 'object': 'random,0.95,1,0,0.1',
                   'instrument': 'ESRF id16a', 'gpu': None}

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanID16a(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanID16a, self).__init__(params, scan, timings=timings)

    def load_scan(self):
        m = self.params['ptychomotors'].split(',')
        if len(m) in [1, 3]:
            print("reading motors from file:", m[0])
            # ptycho motor positions from single file, e.g. mot_pos.txt
            self.x, self.y = np.loadtxt(m[0], unpack=True)
            # Motor positions are in microns, convert to meters
            self.x *= 1e-6
            self.y *= 1e-6
        else:
            # ptycho motors from edf file
            xmot, ymot = m[0:2]

        if self.params['h5data'] is not None:
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
        else:
            # Load all available frames, as well as motor positions from frame headers (...)
            i = 0  # Start at 0 or 1 ?
            if self.params['imgname'].count('%') == 2:
                if os.path.isfile(self.params['imgname'] % (self.scan, i)) is False:
                    i = 1
            elif self.params['imgname'].count('%') == 3:
                if os.path.isfile(self.params['imgname'] % (self.scan, self.scan, i)) is False:
                    i = 1
            else:
                if os.path.isfile(self.params['imgname'] % i) is False:
                    i = 1

            vimg = []  # We don't know how many images there are, so start with a list...
            imgn = []
            x, y = [], []  # motor positions
            if self.params['monitor'] is not None:
                mon = []

            while True:
                if self.params['moduloframe'] is not None:
                    n1, n2 = self.params['moduloframe']
                    if i % n1 != n2:
                        i += 1
                        continue

                if i % 20 == 0:
                    sys.stdout.write('%d ' % i)
                    sys.stdout.flush()

                imgname = self.params['imgname']
                if imgname.count('%') == 2:
                    imgname = imgname % (self.scan, i)
                elif imgname.count('%') == 3:
                    imgname = imgname % (self.scan, self.scan, i)
                else:
                    imgname = imgname % i

                if os.path.isfile(imgname):
                    d = fabio.open(imgname)
                    vimg.append(d.data)
                    motors = {}
                    for k, v in zip(d.header['motor_mne'].split(), d.header['motor_pos'].split()):
                        motors[k] = float(v)
                    x.append(motors[xmot])  # Motor positions in microns
                    y.append(motors[ymot])
                    imgn.append(i)
                    if self.params['monitor'] is not None:
                        counters = {}
                        for k, v in zip(d.header['counter_mne'].split(), d.header['counter_pos'].split()):
                            counters[k] = float(v)
                        mon.append(counters[self.params['monitor']])
                    if self.params['maxframe'] is not None:
                        if len(vimg) == self.params['maxframe']:
                            print("MAXFRAME: only using first %d frames" % (len(vimg)))
                            break
                    i += 1
                else:
                    break
            self.x = np.array(x) * 1e-6
            self.y = np.array(y) * 1e-6
            imgn = np.array(imgn)

        if len(m) >= 3:
            x, y = self.x, self.y
            self.x, self.y = eval(m[-2]), eval(m[-1])

        if len(self.x) < 4:
            raise PtychoRunnerException("Less than 4 scan positions, is this a ptycho scan ?")

        if self.params['monitor'] is not None:
            mon0 = np.median(mon)
            mon /= mon0
            self.validframes = np.where(mon > 0.1)
            if len(self.validframes) != len(mon):
                print('WARNING: The following frames have a monitor value < 0.1 x '
                      'the median value and will be ignored (no beam ?)')
                print(np.where(mon <= (mon0 * 0.1)))
            self.x = np.take(self.x, self.validframes)
            self.y = np.take(self.y, self.validframes)
            imgn = np.take(imgn, self.validframes)

        self.imgn = imgn

    def load_data(self):
        h5meta_filename = self.params['h5meta']
        if '%' in h5meta_filename:
            h5meta_filename = h5meta_filename % tuple(self.scan for i in range(h5meta_filename.count('%')))
            print('h5meta filename for scan #%d: %s' % (self.scan, h5meta_filename))
        h5meta = h5py.File(h5meta_filename, 'r')

        # Assume there's only a single entry, of which we don't know the name...
        for v in h5meta.values():
            break

        h5prefix = v.name
        # DEBUG
        self.h5prefix = h5prefix
        self.h5meta = h5meta

        date_string = h5meta[h5prefix + "/start_time"][()]  # '2015-03-11T23:56:58.390629'
        if sys.version_info > (3,) and isinstance(date_string, bytes):
            date_string = date_string.decode('utf-8')  # Could also use ASCII in this case
        else:
            date_string = str(date_string)
        pattern = '%Y-%m-%dT%H:%M:%S.%f'
        try:
            lc = locale._setlocale(locale.LC_ALL)
            locale._setlocale(locale.LC_ALL, 'C')
            epoch = int(time.mktime(time.strptime(date_string, pattern)))
            locale._setlocale(locale.LC_ALL, lc)
        except ValueError:
            print("Could not extract time from spec header, unrecognized format: %s, expected: %s" % (
            date_string, pattern))

        self.params['nrj'] = h5meta.get(h5prefix + "/measurement/initial/energy")[()]
        if self.scan is None:
            self.scan = int(h5meta.get(h5prefix + "/scan_number")[()])
        if self.params['detectordistance'] is None:
            # self.params['detectordistance'] = h5meta.get(h5prefix+"/PTYCHO/focusToDetectorDistance")[()]
            tmp = eval(h5meta.get(h5prefix + "/PTYCHO/parameters")[()])
            self.params['detectordistance'] = float(tmp['focusToDetectorDistance'])  # Seriously ??
            print("Detector distance from h5 metadata: %6.3fm" % self.params['detectordistance'])

        # read all frames
        imgn = self.imgn
        t0 = timeit.default_timer()
        sys.stdout.write("Reading frames:")
        sys.stdout.flush()
        if self.params['h5data'] is not None:
            # all images from a single hdf5/nexus file
            # Here assuming lambda detector pseudo-nexus files
            h5data_filename = self.params['h5data']
            if h5data_filename.count('%') == 1:
                h5data_filename = h5data_filename % tuple(self.scan for i in range(h5data_filename.count('%')))
            print('h5data filename for scan #%d: %s' % (self.scan, h5data_filename))
            self.h5data = h5py.File(h5data_filename, 'r')
            vimg = self.h5data.get("/entry/instrument/detector/data")[imgn.tolist()]
        else:
            # Load all available frames, as well as motor positions from frame headers (...)
            i = 0  # Start at 0 or 1 ?
            if self.params['imgname'].count('%') == 2:
                if os.path.isfile(self.params['imgname'] % (self.scan, i)) is False:
                    i = 1
            elif self.params['imgname'].count('%') == 3:
                if os.path.isfile(self.params['imgname'] % (self.scan, self.scan, i)) is False:
                    i = 1
            else:
                if os.path.isfile(self.params['imgname'] % i) is False:
                    i = 1

            vimg = []

            for i in imgn:
                if i % 20 == 0:
                    sys.stdout.write('%d ' % i)
                    sys.stdout.flush()

                imgname = self.params['imgname']
                if imgname.count('%') == 2:
                    imgname = imgname % (self.scan, i)
                elif imgname.count('%') == 3:
                    imgname = imgname % (self.scan, self.scan, i)
                else:
                    imgname = imgname % i

                d = fabio.open(imgname)
                vimg.append(d.data)

        dt = timeit.default_timer() - t0
        print("\n")
        print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, vimg[0].size * len(vimg) / 1e6 / dt))

        # Convert to numpy arrays
        d = np.empty((len(vimg), vimg[0].shape[0], vimg[0].shape[1]), dtype=vimg[0].dtype)
        for i in range(len(vimg)):
            d[i] = vimg[i]

        self.raw_data = d
        self.load_data_post_process()


class PtychoRunnerID16a(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(PtychoRunnerID16a, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        Returns:
            True if the argument is interpreted, false otherwise
        """
        if k == 'h5meta' or k == 'ptychomotors' or k == 'imgname' or k == 'h5data':
            self.params[k] = v
            return True
        elif k == 'pixelsize':
            self.params[k] = float(v)
            return True
        return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['h5meta'] is None:
            # Try to find a *.h5 file in the current directory
            g = glob.glob("*.h5")
            if len(g) == 1:
                self.params['h5meta'] = g[0]
                print('no h5meta given, using only .h5 in current directory: %s' % g[0])
            else:
                raise PtychoRunnerException('Missing argument: no h5metadata file given (h5meta=..)')
        if self.params['imgname'] is None and self.params['h5data'] is None:
            g = glob.glob("*.nxs")
            if len(g) == 1:
                self.params['h5data'] = g[0]
                print('no h5data given, using only .nxs in current directory: %s' % g[0])
            else:
                raise PtychoRunnerException('Missing argument: imgname or h5data')
        if self.params['ptychomotors'] is None:
            raise PtychoRunnerException('Missing argument: ptychomotors')
