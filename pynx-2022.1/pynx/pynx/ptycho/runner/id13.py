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
import warnings

import fabio
from ...utils import h5py
import numpy as np
import silx
from silx.io.specfile import SpecFile

from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, params_generic

helptext_beamline = """
Script to perform a ptychography analysis on data from id13@ESRF.
This can handle by default 2020-type data (all scans data in hdf5 and eiger images
in a per-scan hdf5 file).

Example:
    pynx-id13pty.py data=scans.h5 scan=41 detectordistance=1.3 
      probe=60e-6x200e-6,0.09 verbose=10 save=all saveplot liveplot
      algorithm=analysis,ML**100,DM**200,nbprobe=3,probe=1 

Command-line arguments (beamline-specific):
    data=/some/dir/to/data.h5: path to hdf5 scan data [mandatory, unless specfile is given]
    
    specfile=/some/dir/to/specfile.spec: path to specfile [obsolete format]

    data_detector=other/path/to/eiger/%04d/scan_%04d.h5: filename for the hdf5 data
        If there are several %d fields in the name, every one but the last
        will be replaced by the scan number.
    
    scan=56: scan number in data or specfile [mandatory].
             Alternatively a list or range of scans can be given:
                scan=12,23,45 or scan="range(12,25)" (note the quotes)

    h5data=entry/data/data_%6d: generic hdf5 path to the stack of 2d images inside hdf5 file 
                                [default=entry_0000/measurement/data]

    nrj=8: energy in keV [mandatory, but should be in hdf5 file]

    pixelsize=75e-6: pixel size on detector in meters 
                     [default: 75e-6]

    ptychomotors=nnp2,nnp3,-x,y: name of the two motors used for ptychography, optionally followed
                                 by a mathematical expression to be used to calculate the actual 
                                 motor positions (axis convention, angle..). Values will be 
                                 extracted from the edf files, and are assumed to be in microns.
                                 Example 1: ptychomotors=nnp2,nnp3
                                 Example 2: ptychomotors=nnp2,nnp3,-x,y
                                 Note that if the 'xy=-y,x' command-line argument is used, it is 
                                 applied after this, using 'ptychomotors=nnp2,nnp3,-x,y' is 
                                 equivalent to 'ptychomotors=nnp2,nnp3 xy=-x,y'
                                 [default:nnp2_position,nnp3_position xy=-x,y]

    monitor=opt2: spec name for the monitor counter. The frames will be normalised by the ratio of
                  the counter value divided by the median value of the counter over the entire scan
                  (so as to remain close to Poisson statistics). A monitor intensity lower than 10%
                  of the median value will be interpreted as an image taken without beam and will be 
                  skipped.
                  [default = None]

    kmapfile=detector/kmap/kmap_00000.edf.gz: if images are saved in a multiframe data file.
                                              This superseeds imgname=...
"""

# NB: for id13 we start from a flat object
params_beamline = {'specfile': None, 'h5file': None, 'h5data': "entry_0000/measurement/data", 'nrj': None,
                   'data_detector': None,
                   'pixelsize': 75e-6, 'kmapfile': None, 'monitor': None,
                   'object': 'random,0.95,1,0,0.1', 'instrument': 'ESRF id13',
                   'ptychomotors': 'nnp2,nnp3', 'xy': '-x,y'}

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanID13(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanID13, self).__init__(params, scan, timings=timings)

    def load_scan(self):
        if self.params['data'] is not None:
            # Scans are recorded in an hdf5 file
            self.h = h5py.File(self.params['data'], mode='r')
            entry = self.h['%d.1' % self.scan]
            m = self.params['ptychomotors'].split(',')
            xmot, ymot = m[0:2]
            if xmot not in entry['measurement'] or ymot not in entry['measurement']:
                raise PtychoRunnerException(
                    'Ptycho motors (%s, %s) not found in scan #%d of data file:%s' % (
                        xmot, ymot, self.scan, self.params['data']))

            self.x, self.y = entry['measurement/%s' % xmot][()], entry['measurement/%s' % ymot][()]
            if len(m) == 4:
                x, y = self.x, self.y
                self.x, self.y = eval(m[2]), eval(m[3])
            if len(self.x) < 4:
                raise PtychoRunnerException("Less than 4 scan positions, is this a ptycho scan ?")
            # Spec values are in microns, convert to meters
            self.x *= 1e-6
            self.y *= 1e-6
            imgn = np.arange(len(self.x), dtype=np.int)
            if self.params['monitor'] is not None:
                mon = entry['measurement/%s' % self.params['monitor']][()]
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
            else:
                mon = None


        else:
            # pre-2020 data with a specfile
            self.s = SpecFile(self.params['specfile'])['%d.1' % (self.scan)]
            self.h = self.s.scan_header_dict
            print('Read scan:', self.h['S'])

            date_string = self.h['D']  # 'Wed Mar 23 14:41:56 2016'
            date_string = date_string[date_string.find(' ') + 1:]
            pattern = '%b %d %H:%M:%S %Y'
            try:
                lc = locale._setlocale(locale.LC_ALL)
                locale._setlocale(locale.LC_ALL, 'C')
                epoch = int(time.mktime(time.strptime(date_string, pattern)))
                locale._setlocale(locale.LC_ALL, lc)
            except ValueError:
                print(
                    "Could not extract time from spec header, unrecognized format: %s, expected:" % (
                        date_string) + pattern)

            m = self.params['ptychomotors'].split(',')
            xmot, ymot = m[0:2]

            if self.s.labels.count(xmot) == 0 or self.s.labels.count(ymot) == 0:
                raise PtychoRunnerException(
                    'Ptycho motors (%s, %s) not found in scan #%d of specfile:%s' % (
                        xmot, ymot, self.scan, self.params['specfile']))

            self.x, self.y = self.s.data_column_by_name(xmot), self.s.data_column_by_name(ymot)

            if len(m) == 4:
                x, y = self.x, self.y
                self.x, self.y = eval(m[2]), eval(m[3])
            if len(self.x) < 4:
                raise PtychoRunnerException("Less than 4 scan positions, is this a ptycho scan ?")
            # Spec values are in microns, convert to meters
            self.x *= 1e-6
            self.y *= 1e-6

            imgn = np.arange(len(self.x), dtype=np.int)

            if self.params['monitor'] is not None:
                if silx is not None:
                    mon = self.s.data_column_by_name(self.params['monitor'])
                else:
                    mon = self.d[self.params['monitor']]
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

        self.imgn = imgn

    def load_data(self):
        imgn = np.sort(np.array(self.imgn))

        if self.params['data'] is not None:
            # 2020 id13 data format
            t0 = timeit.default_timer()
            if False:
                # One virtual dataset compiles all data
                h = h5py.File(self.params['data'], mode='r')
                print("Reading entry: %d.1" % self.scan)
                entry = self.h['%d.1' % self.scan]
                print("Reading %d frames from %s" % (len(imgn), self.params[data_detector]))
                vimg = entry['measurement/eiger'][imgn]
                print(vimg.sum())
                print(vimg.sum(axis=(1, 2)))
            else:
                prefix = self.params['data_detector']
                nbs = prefix.count('%')
                scans = [self.scan] * (nbs - 1)

                self.print("Reading %d frames from %s" % (len(imgn), self.params["data_detector"]))
                # frames are grouped in different files
                ifile = 0
                ct = 0
                i0 = 0
                vimg = None
                # NB: it would be faster to use a virtual dataset...
                while True:
                    # Read next hdf5 file
                    s = self.params["data_detector"] % tuple(scans + [ifile])
                    if not os.path.exists(s):
                        break
                    hd = h5py.File(s, mode='r')
                    hdd = hd['entry_0000/measurement/data']
                    nb = len(hdd)

                    # Read all suitable frames
                    idx = np.where(np.logical_and(imgn >= i0, imgn < (i0 + nb)))[0]
                    if len(idx):
                        self.print("Reading h5 data file: %s [%d frames]" % (s, len(idx)))
                        if vimg is None:
                            # Init array
                            frame = np.array(hdd[0])
                            vimg = np.empty((len(imgn), frame.shape[0], frame.shape[1]), dtype=frame.dtype)
                            if self.params['pixelsize'] is None:
                                self.params['pixelsize'] = \
                                    hd['entry_0000/instrument/eiger/detector_information/pixel_size/xsize'][()]
                                self.print("Pixelsize?", self.params['pixelsize'])
                        # np.put(vimg, idx, hdd[imgn[idx] - i0])  # real slow
                        try:
                            vimg[idx] = hdd[imgn[idx] - i0]  # Does not work on h5py <= 2.10:
                        except TypeError:
                            #   PointSelection __getitem__ only works with bool arrays in h5py <= 2.10
                            warnings.warn("id13 data read: error reading h5pydata[idx] - "
                                          "You should install h5py>=2.10.0")
                            tmp = hdd[imgn[idx] - i0]
                            for i in range(len(idx)):
                                vimg[idx[i]] = tmp[i]
                        ct += len(idx)

                    ifile += 1
                    i0 += nb
                if ct != len(imgn):
                    raise PtychoRunnerException(
                        "Could not read the expected number of frames: %d < %d" % (ct, len(imgn)))

                    if vimg is None:
                        ny, nx = hd['entry_0000/measurement/data'][0].shape
                        vimg = np.empty((len(self.x), ny, nx), dtype=np.float32)
                    vimg[nb0:nb0 + len(idx)] = hd['entry_0000/measurement/data'][idx]

            dt = timeit.default_timer() - t0
            self.print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, vimg.size / 1e6 / dt))
        else:
            self.h5 = h5py.File(self.params['h5file'], 'r')

            if self.params['pixelsize'] is None:
                self.params['pixelsize'] = np.array(self.h5.get('entry/instrument/detector/x_pixel_size'))
                self.print("Pixelsize?", self.params['pixelsize'])

            if self.params['nrj'] is None:
                assert (self.h5.get("entry/instrument/beam/incident_wavelength")[()] > 0.1)
                self.params['nrj'] = 12.3984 / self.h5.get("entry/instrument/beam/incident_wavelength")[()]

            # Load all frames
            t0 = timeit.default_timer()
            vimg = None
            entry0 = 1
            d0 = 0
            if self.params['kmapfile'] is not None:
                sys.stdout.write("Reading frames from KMAP file (this WILL take a while)...")
                sys.stdout.flush()
                kfile = fabio.open(self.params['kmapfile'])
                if kfile.getNbFrames() < len(imgn):
                    raise PtychoRunnerException("KMAP: only %d frames instead of %d in data file (%s) !"
                                                "Did you save all frames in a single file ?" %
                                                (kfile.getNbFrames(), len(imgn), self.params['kmapfile']))
                ii = 0
                for i in imgn:
                    if (i - imgn[0]) % 20 == 0:
                        sys.stdout.write('%d ' % (i - imgn[0]))
                        sys.stdout.flush()
                    frame = kfile.getframe(i).data
                    if vimg is None:
                        vimg = np.empty((len(imgn), frame.shape[0], frame.shape[1]), dtype=frame.dtype)
                    vimg[ii] = frame
                    d0 += frame
                    ii += 1
            else:
                self.print('Reading HDF5 frames: ')
                # frames are grouped in different subentries
                entry0 = 1
                ct = 0
                i0 = 0
                # NB: it would be faster to create a virtual dataset...
                while True:
                    # Read all entries in hdf5 file
                    h5entry = self.params["h5data"] % entry0
                    if h5entry not in self.h5:
                        break
                    h5d = self.h5[h5entry]
                    nb = len(h5d)

                    # Read all suitable frames
                    idx = np.where(np.logical_and(imgn >= i0, imgn < (i0 + nb)))[0]
                    if len(idx):
                        self.print("Reading h5 data entry: %s [%d frames]" % (h5entry, len(idx)))
                        if vimg is None:
                            # Init array
                            frame = np.array(h5d[0])
                            vimg = np.empty((len(imgn), frame.shape[0], frame.shape[1]), dtype=frame.dtype)
                        vimg[idx] = h5d[imgn[idx] - i0]
                        ct += len(idx)

                    entry0 += 1
                    i0 += nb
                if ct != len(imgn):
                    raise PtychoRunnerException(
                        "Could not read the expected number of frames: %d < %d" % (ct, len(imgn)))
            dt = timeit.default_timer() - t0
            self.print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, vimg.size / 1e6 / dt))

        # Build mask. Values of 2**32-1 and -2 are invalid (module gaps or invalidated pixels)
        # TODO: mask independently the different frames
        self.raw_mask = ((vimg > (2 ** 32 - 3)).sum(axis=0) > 0).astype(np.int8)  # very slow !
        if self.raw_mask.sum() == 0:
            self.raw_mask = None
        else:
            self.print("\nMASKING %d pixels from detector flags" % (self.raw_mask.sum()))

        self.raw_data = vimg
        self.load_data_post_process()


class PtychoRunnerID13(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(PtychoRunnerID13, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        This method should be superseded in a beamline/instrument-specific child class
        Returns:
            True if the argument is interpreted, false otherwise
        """
        if k == 'h5file':
            k = 'data_detector'
        if k in ['specfile', 'h5file', 'kmapfile', 'ptychomotors', 'data', 'data_detector']:
            self.params[k] = v
            return True
        elif k in ['nrj', 'pixelsize']:
            self.params[k] = float(v)
            return True
        return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['scan'] is None:
            raise PtychoRunnerException('Missing argument: no scan number given')
        if self.params['specfile'] is None and self.params['data'] is None:
            raise PtychoRunnerException('Missing argument: no data (or specfile for pre-2020 experiments) given')
        if self.params['data_detector'] is None:
            raise PtychoRunnerException('Missing argument: no data_detector file name prefix given')
        if self.params['detectordistance'] is None:
            raise PtychoRunnerException('Missing argument: detector distance')
        if self.params['ptychomotors'] is None:
            raise PtychoRunnerException('Missing argument: ptychomotors')
        if self.params['data'] is not None and self.params['nrj'] is None:
            raise PtychoRunnerException('Missing argument: nrj= (in keV) is required')
        if self.params['kmapfile'] is not None and self.params['nrj'] is None:
            raise PtychoRunnerException('Missing argument: for KMAP data, nrj= (in keV) is required')
