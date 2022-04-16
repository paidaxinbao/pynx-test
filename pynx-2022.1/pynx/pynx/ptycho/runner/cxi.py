#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
from __future__ import division

import sys
import os
import timeit

from ...utils import h5py
import numpy as np

from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, params_generic

helptext_beamline = """
Script to perform a ptychography analysis on data recorded in CXI format (http://cxidb.org/cxi.html)

Examples:
    pynx-cxipty.py data=data.cxi probe=focus,60e-6x200e-6,0.09 
      algorithm=analysis,ML**100,DM**200,nbprobe=3,probe=1 verbose=10 save=all saveplot liveplot

command-line arguments:
    cxifile=/some/dir/to/data.cxi or data=some/dir/to/data.cxi: path to CXI data [mandatory]

    scan=56: scan number [optional: this is used to name the output directory, otherwise 0].
             Several scans can also processed by giving a generic cxi name and scan numbers, 
             e.g.: data=data%03d.cx scan=13,45,78

    gpu=Titan: GPU name for OpenCL calculation 
               [default = will be auto-selected from available GPUs]
    
    xyrange=-2e-6,2e-6,-5e-6,-2e-6: range where the data points will be taken into 
             account (xmin, xmax, ymin, ymax). All scan positions outside this range are ignored.
             This must be given in original coordinates, in meters.
             [default=No spatial restrictions]
"""

params_beamline = {}

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanCXI(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanCXI, self).__init__(params, scan, timings=timings)

    def load_scan(self):
        if os.path.isfile(self.params['cxifile']) is False:
            raise PtychoRunnerException("CXI file does not exist: %s" % (self.params['cxifile']))
        cxi = h5py.File(self.params['cxifile'], 'r')
        x, y = cxi['/entry_1/sample_1/geometry_1/translation'][0:2]

        if len(x) < 4:
            raise PtychoRunnerException("Less than 4 scan positions, is this a ptycho scan ?")

        imgn = np.arange(len(x), dtype=np.int)

        if '/entry_1/data_1/monitor' in cxi:
            mon = cxi['/entry_1/data_1/monitor'][()]
            mon0 = np.median(mon)
            mon /= mon0
            self.validframes = np.where(mon > 0.1)
            if len(self.validframes) != len(mon):
                print('WARNING: The following frames have a monitor value < 0.1'
                      ' the median value and will be ignored (no beam ?)')
                print(np.where(mon <= (mon0 * 0.1)))
            x = np.take(x, self.validframes)
            y = np.take(y, self.validframes)
            imgn = np.take(imgn, self.validframes)
        else:
            mon = None

        if 'xyrange' in self.params:
            xmin, xmax, ymin, ymax = self.params['xyrange']
            print("Restricting scan positions to %f < x < %f and %f < y < %f" % (xmin, xmax, ymin, ymax))
            idx = np.where((x >= xmin) * (x <= xmax) * (y >= ymin) * (y <= ymax))[0]
            if len(idx) < 10:
                raise PtychoRunnerException("Only %d points remaining after applying the xyrange "
                                            "constraint. original range: %5e<x<%5e %5e<y<%5e"
                                            % (len(idx), x.min(), x.max(), y.min(), y.max()))
            else:
                print("   ... %d/%d remaining positions" % (len(idx), len(x)))
            imgn = imgn.take(idx)
            x = x.take(idx)
            y = y.take(idx)

        if self.params['moduloframe'] is not None:
            n1, n2 = self.params['moduloframe']
            idx = np.where(imgn % n1 == n2)[0]
            imgn = imgn.take(idx)
            x = x.take(idx)
            y = y.take(idx)

        if self.params['maxframe'] is not None:
            N = self.params['maxframe']
            if len(imgn) > N:
                print("MAXFRAME: only using first %d frames" % (N))
                imgn = imgn[:N]
                x = x[:N]
                y = y[:N]
        self.x, self.y, self.imgn = x, y, imgn

    def load_data(self):
        imgn = self.imgn
        if imgn is None:
            raise PtychoRunnerException("load_data(): imgn is None. Did you call load_scan() before ?")
        if os.path.isfile(self.params['cxifile']) is False:
            raise PtychoRunnerException("CXI file does not exist: %s" % (self.params['cxifile']))
        cxi = h5py.File(self.params['cxifile'], 'r')
        self.params['instrument'] = cxi['/entry_1/instrument_1/name'][()]
        if self.params['nrj'] is None:
            self.params['nrj'] = cxi['/entry_1/instrument_1/source_1/energy'][()] / 1.60218e-16
        self.params['detectordistance'] = cxi['/entry_1/instrument_1/detector_1/distance'][()]
        self.params['pixelsize'] = cxi['/entry_1/instrument_1/detector_1/x_pixel_size'][()]

        if self.params['scan'] is None and '/entry_1/data_1/process_1/configuration/scan' in cxi:
            try:
                self.scan = int(cxi['/entry_1/data_1/process_1/configuration/scan'][()])
                print('CXI: read scan number=%d' % self.scan)
            except:
                pass

        # Load all frames
        sys.stdout.write("Reading %d frames from CXI-HDF5 file: 0" % (len(imgn)))
        sys.stdout.flush()
        vrange = np.arange(0, len(imgn), 1)
        t0 = timeit.default_timer()
        vimg = None
        for i in range(len(vrange)):
            i1 = vrange[i]
            if i1 == len(vrange) - 1:
                i2 = len(imgn)
            else:
                i2 = vrange[i + 1]
            # print(imgn[i1:i2].tolist())
            tmp = cxi['/entry_1/data_1/data'][imgn[i1:i2].tolist()]
            if vimg is None:
                vimg = np.empty((len(imgn), tmp.shape[-2], tmp.shape[-1]), dtype=tmp.dtype)
            vimg[i1:i2] = tmp
            if i2 % 20 == 0:
                sys.stdout.write('.%d' % (i2))
                sys.stdout.flush()
        print()
        d0 = vimg.sum(axis=0)
        dt = timeit.default_timer() - t0
        print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, d0.size * len(vimg) / 1e6 / dt))

        if '/entry_1/instrument_1/detector_1/mask' in cxi:
            print("Loaded mask from CXI data: /entry_1/instrument_1/detector_1/mask")
            self.raw_mask = cxi['/entry_1/instrument_1/detector_1/mask'][()]
            self.raw_mask = (self.raw_mask != 0).astype(np.int8)
        if '/entry_1/instrument_1/detector_1/dark' in cxi:
            print("Loaded dark from CXI data: /entry_1/instrument_1/detector_1/dark")
            self.dark = cxi['/entry_1/instrument_1/detector_1/dark'][()]
        self.raw_data = vimg
        self.load_data_post_process()


class PtychoRunnerCXI(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(PtychoRunnerCXI, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        This method should be superseded in a beamline/instrument-specific child class
        Returns:
            True if the argument is interpreted, false otherwise
        """
        if k == 'data':
            k = 'cxifile'
        if k == 'cxifile':
            self.params[k] = v
            return True
        if k == 'xyrange':
            self.params[k] = [float(xy) for xy in v.split(',')]
            return True
        return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['cxifile'] is None:
            raise PtychoRunnerException('Missing argument: no cxifile=*.cxi (or data=*.cxi) given')
