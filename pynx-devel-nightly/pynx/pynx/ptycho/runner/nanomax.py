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
from ..mpi import MPI

from ...utils import h5py
import numpy as np

from ...utils.array import rebin
from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, params_generic

helptext_beamline = """
Script to perform a ptychography analysis on data from nanomax@MaxIV

Example:
    pynx-nanomaxpty.py h5meta=data.h5 data=scan_%04d_pil100k_0000.hdf5 scan=9 probe=60e-6x60e-6,0.09
      algorithm=analysis,ML**100,DM**200,nbprobe=3,probe=1

Command-line arguments (beamline-specific):
    h5meta=data.h5: hdf5 filename with the scan metadata and motor positions. The energy, 
                    detector distance will be extracted 
                    [mandatory - if a single *.h5 file is in the working directory, it will be used by default]
                    
    scan=5: scan number [mandatory]

    pixelsize=55e-6: pixel size on detector in meters
                     [will be read from data if not given]                                          
    
    data=/path/to/hdf5/data/file.hdf5: path to nexus data file (lambda detector).
                                     If one or several % field are included in the name, they will be replaced by
                                     the scan number.
                                     Examples: 
                                        data=/path/to/hdf5/data/scan%d/scan%d.hdf5
                                        data=/path/to/hdf5/data/scan%d.hdf5
                                        data=/path/to/hdf5/data/data.hdf5
                                     [mandatory]

    ptychomotors=pix,piz,-x,y: name of the two motors used for ptychography, optionally followed 
                               by a mathematical expression to be used to calculate the actual
                               motor positions (axis convention, angle..). Values will be extracted
                               from the spec files, and are assumed to be in microns.
                               Example 1: ptychomotors=pix,piz
                               Example 2: ptychomotors=pix,piz,-x,y
                               Note that if the 'xy=-y,x' command-line argument is used, 
                               it is applied after this, using 'ptychomotors=pix,piz,-x,y' 
                               is equivalent to 'ptychomotors=pix,piz xy=-x,y'
                               Note that the data for the motors may be cropped if the motor positions array
                               is larger than the number of images available.
                               [Mandatory]


    detectordistance=1.3: detector distance in meters [mandatory]

    gpu=Titan: GPU name for calculations [Mandatory]
"""

# NB: for id16 we start from a flat object (high energy, high transmission)
params_beamline = {'h5meta': None, 'pixelsize': None, 'monitor': None, 'ptychomotors': 'samx,samy',
                   'data': None, 'detectordistance': None, 'object': 'random,0.95,1,0,0.1',
                   'instrument': 'NanoMAX@MaxIV', 'gpu': None, 'scan': None, 'map_rebin': None}

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanNanoMAX(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanNanoMAX, self).__init__(params, scan, timings=timings)

        self.fast_scan_2d, self.fast_scan_nx, self.fast_scan_ny = False, None, None

    def load_scan(self):
        h5meta_filename = self.params['h5meta']
        h5meta = h5py.File(h5meta_filename, 'r')
        h5prefix = "entry%d" % self.scan

        if self.params['map_rebin'] is not None and 'split' in self.params['mpi']:
            raise PtychoRunnerException("map_rebin is not supported with mpi=split. Use CXI instead.")

        # Motor positions
        mx, my = self.params['ptychomotors'].split(',')
        mx = mx.strip()
        my = my.strip()
        print(self.params['ptychomotors'], mx, my)
        self.x = h5meta.get(h5prefix + "/measurement/%s" % mx)[()] * 1e-6
        self.y = h5meta.get(h5prefix + "/measurement/%s" % my)[()] * 1e-6
        # KLUDGE, to handle fast scans, where the stored buffer is larger than the actual data...
        fast_scan_2d, fast_scan_ny, fast_scan_nx = False, None, None
        if '_buff' in mx and '_buff' in my:
            self.fast_scan_2d = True
            print('Motor names suggest this was a fast scan, cropping empty columns')
            colmax = np.nonzero(self.x.sum(axis=0) + self.y.sum(axis=0))[0][-1]
            self.x = self.x[:, :colmax + 1]
            self.y = self.y[:, :colmax + 1]
            self.fast_scan_ny, self.fast_scan_nx = self.x.shape
            print('Fast scan shape:', self.x.shape)

        imgn = np.arange(self.x.size, dtype=np.int32)
        if self.params['moduloframe'] is not None:
            if self.params['map_rebin'] is not None and fast_scan_2d:
                raise PtychoRunnerException("'map_rebin' cannot be used with maxframe or moduloframe for 2D maps")

            n1, n2 = self.params['moduloframe']
            idx = np.where(imgn % n1 == n2)[0]
            imgn = imgn.take(idx)
            self.x = self.x.flatten()
            self.y = self.y.flatten()
            self.x = self.x.take(idx)
            self.y = self.y.take(idx)

        if self.params['maxframe'] is not None:
            if self.params['map_rebin'] is not None and fast_scan_2d:
                raise PtychoRunnerException("'map_rebin' cannot be used with maxframe or moduloframe for 2D maps")
            N = self.params['maxframe']
            if len(imgn) > N:
                print("MAXFRAME: only using first %d frames" % N)
                imgn = imgn[:N]
                self.x = self.x.flatten()
                self.y = self.y.flatten()
                self.x = self.x[:N]
                self.y = self.y[:N]
        self.imgn = imgn

    def load_data(self):
        h5meta_filename = self.params['h5meta']
        h5meta = h5py.File(h5meta_filename, 'r')

        h5prefix = "entry%d" % self.scan

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

        # Energy in eV
        self.params['nrj'] = h5meta.get(h5prefix + "/measurement/energy")[()] * 1e-3

        # KLUDGE, to handle fast scans, where the stored buffer is larger than the actual data...
        fast_scan_2d = self.fast_scan_2d
        fast_scan_nx = self.fast_scan_nx
        fast_scan_ny = self.fast_scan_ny

        # read all frames
        t0 = timeit.default_timer()
        sys.stdout.write("Reading frames:")
        sys.stdout.flush()
        imgn = self.imgn

        # Read all images from a single hdf5/nexus file, here assuming Pilatus formatting
        h5data_filename = self.params['data']
        if h5data_filename.count('%') == 1:
            h5data_filename = h5data_filename % tuple(self.scan for i in range(h5data_filename.count('%')))
        print('h5data filename for scan #%d: %s' % (self.scan, h5data_filename))
        self.h5data = h5py.File(h5data_filename, 'r')

        vimg = None
        if 'entry_0000/instrument/Pilatus' in self.h5data:
            # Pilatus data
            if self.params['pixelsize'] is None:
                self.params['pixelsize'] = \
                    float(self.h5data['entry_0000/instrument/Pilatus/detector_information/pixel_size/xsize'][()])
                print("Read pixel size from Pilatus detector file: %6.1f microns" % (self.params['pixelsize'] * 1e6))
            ii = 0
            for i in imgn:
                if (i - imgn[0]) % 20 == 0:
                    sys.stdout.write('%d ' % (i - imgn[0]))
                    sys.stdout.flush()
                frame = self.h5data['entry_%04d/measurement/Pilatus/data' % ii][()]
                if vimg is None:
                    vimg = np.empty((len(imgn), frame.shape[-2], frame.shape[-1]), dtype=frame.dtype)
                vimg[ii] = frame
                ii += 1
        elif 'entry_0000/instrument/Merlin' in self.h5data:
            # Merlin detector data, possibly fast scan
            if self.params['pixelsize'] is None:
                self.params['pixelsize'] = \
                    float(self.h5data['entry_0000/instrument/Merlin/detector_information/pixel_size/xsize'][()]
                          * 1e-6)
                print("Read pixel size from Merlin detector file: %6.1f microns" % (self.params['pixelsize'] * 1e6))
            if fast_scan_2d:
                # This is a 2D map (likely a fast scan)
                ii = 0
                for iy in range(fast_scan_ny):
                    if iy % 2 == 0:
                        sys.stdout.write('%d ' % ii)
                        sys.stdout.flush()
                    frame = self.h5data['entry_%04d/measurement/Merlin/data' % iy][()]
                    if vimg is None:
                        vimg = np.empty(list(n for n in self.x.shape) + [frame.shape[-2], frame.shape[-1]],
                                        dtype=frame.dtype)
                    vimg[iy] = frame
                    ii += fast_scan_nx
            else:
                ii = 0
                for i in imgn:
                    if (i - imgn[0]) % 20 == 0:
                        sys.stdout.write('%d ' % (i - imgn[0]))
                        sys.stdout.flush()
                    frame = self.h5data['entry_%04d/measurement/Merlin/data' % ii][()]
                    if vimg is None:
                        vimg = np.empty((len(imgn), frame.shape[-2], frame.shape[-1]), dtype=frame.dtype)
                    vimg[ii] = frame
                    ii += 1

        else:
            raise PtychoRunnerException("Did not recognise either Pilatus or Merlin data, new detector ?")

        if self.params['map_rebin'] is not None:
            rebinf = [int(n) for n in self.params['map_rebin'].split(',')]
            print("Rebinning scan map, rebinf: ", rebinf)
            self.x = rebin(self.x, rebinf, scale="average")
            self.y = rebin(self.y, rebinf, scale="average")
            vimg = rebin(vimg, rebinf + [1, 1], scale="sum")

        self.x = self.x.flatten()
        self.y = self.y.flatten()
        if vimg.ndim == 4:
            vimg = vimg.reshape((vimg.shape[0] * vimg.shape[1], vimg.shape[2], vimg.shape[3]))

        dt = timeit.default_timer() - t0
        print("\n")
        print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, vimg[0].size * len(vimg) / 1e6 / dt))

        if len(self.x) < 4:
            raise PtychoRunnerException("Less than 4 scan positions, is this a ptycho scan ?")

        if self.params['monitor'] is not None:
            mon = h5meta.get(h5prefix + "/measurement/%s" % self.params['monitor'])[()]
            mon0 = np.median(mon)
            mon /= mon0
            self.validframes = np.where(mon > 0.1)
            if len(self.validframes) != len(mon):
                print('WARNING: The following frames have a monitor value < 0.1 the median value '
                      'and will be ignored (no beam ?)')
                print(np.where(mon <= (mon0 * 0.1)))
            self.x = np.take(self.x, self.validframes)
            self.y = np.take(self.y, self.validframes)
            # imgn = np.take(imgn, self.validframes)
        else:
            mon = None

        self.raw_data = vimg
        self.load_data_post_process()


class PtychoRunnerNanoMAX(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(PtychoRunnerNanoMAX, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        Returns:
            True if the argument is interpreted, false otherwise
        """
        if k in ['h5meta', 'data', 'ptychomotors', 'map_rebin']:
            self.params[k] = v
            return True
        elif k == 'pixelsize' or k == 'detectordistance':
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
        if self.params['scan'] is None:
            raise PtychoRunnerException('Missing argument: scan number')
        if self.params['detectordistance'] is None:
            raise PtychoRunnerException('Missing argument: detectordistance')
        if self.params['data'] is None:
            raise PtychoRunnerException('Missing argument: data')
