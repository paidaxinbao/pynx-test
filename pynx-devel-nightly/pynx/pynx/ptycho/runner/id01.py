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
import timeit
import locale

import fabio
import numpy as np

from silx.io.specfile import SpecFile

from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, params_generic

helptext_beamline = """
Script to perform a ptychography analysis on data from id01@ESRF

Example:
    pynx-id01pty.py specfile=siemens.spec scan=57 detectordistance=1.3 ptychomotors=pix,piz,-x,y
      probe=60e-6x200e-6,0.09 algorithm=analysis,ML**100,DM**200,nbprobe=3,probe=1 
      loadmask=maxipix verbose=10 save=all saveplot liveplot

command-line arguments:
    specfile=/some/dir/to/specfile.spec: path to specfile [mandatory]

    scan=56: scan number in specfile [mandatory].
             Alternatively a list or range of scans can be given:
                scan=12,23,45 or scan="range(12,25)" (note the quotes)

    imgcounter=mpx4inr: spec counter name for image number 
                        [default=auto, will use mpx4inr or ei2minr]

    imgname=/dir/to/images/prefix%05d.edf.gz: images location with mask 
        [default: will be extracted from ULIMA_mpx4 or ULIMA_eiger2M the spec scan header]

    nrj=8: energy in keV [default: will be extracted from UMONO spec scan header,  mononrj=x.xxkeV]

    detectordistance=1.3: detector distance in meters 
                          [mandatory, unless given in spec header UDETCALIB]

    pixelsize=55e-6: pixel size on detector in meters [default:55e-6]

    ptychomotors=pix,piz,-x,y: name of the two motors used for ptychography, optionally followed 
                               by a mathematical expression to be used to calculate the actual
                               motor positions (axis convention, angle..). Values will be extracted
                               from the spec files, and are assumed to be in microns.
                               Example 1: ptychomotors=pix,piz
                               Example 2: ptychomotors=pix,piz,-x,y
                               Note that if the 'xy=-y,x' command-line argument is used, 
                               it is applied after this, using 'ptychomotors=pix,piz,-x,y' 
                               is equivalent to 'ptychomotors=pix,piz xy=-x,y'
                               [Mandatory]

    xyrange=-2e-6,2e-6,-5e-6,-2e-6: range where the data points will be taken into 
             account (xmin, xmax, ymin, ymax). All scan positions outside this range are ignored.
             This must be given in original coordinates, in meters.
             [default=No spatial restrictions]

    monitor=opt2: spec name for the monitor counter. The frames will be normalized by the ratio
                  of the counter value divided by the median value of the counter over the entire
                  scan (so as to remain close to Poisson statistics). A monitor intensity lower 
                  than 10% of the median value will be interpreted as an image taken without beam
                  and will be skipped.
                  [default = None]

    kmapfile=detector/kmap/kmap_00000.edf.gz: if images are saved in a multiframe data file 
                                              This superseeds imgname=...

    livescan: if set, this keyword will trigger the analysis of all scans of a given type 
              (by default 'spiralscan') in the supplied spec file. If scan=NN is also given,
              the analysis will start after scan #NN.
              Once all scans are processed, the script will wait for new scans to arrive.
"""

params_beamline = {'specfile': None, 'imgcounter': 'auto', 'imgname': None, 'nrj': None, 'detectordistance': None,
                   'pixelsize': 55e-6, 'kmapfile': None, 'monitor': None, 'ptychomotors': None,
                   'instrument': 'ESRF id01'}

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanID01(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanID01, self).__init__(params, scan, timings=timings)

    def load_scan(self):
        self.s = SpecFile(self.params['specfile'])['%d.1' % self.scan]
        self.h = self.s.scan_header_dict
        self.print('Read scan:', self.h['S'])

        if self.params['imgcounter'] == 'auto':
            if 'ei2minr' in self.s.labels:
                self.params['imgcounter'] = 'ei2minr'
            elif 'mpx4inr' in self.s.labels:
                self.params['imgcounter'] = 'mpx4inr'
            self.print("Using image counter: %s" % (self.params['imgcounter']))

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

        if self.params['kmapfile'] is None and self.params['imgcounter'] != 'auto':
            # Ignore image numbers if we read a kmap file, assume the images are stored sequentially in a single file
            # If imgcounter == 'auto', assume images are numbered from zero if mpx4inr or ei2minr were not found
            imgn = self.s.data_column_by_name(self.params['imgcounter']).astype(np.int)
        else:
            imgn = np.arange(len(self.x), dtype=np.int)

        if self.params['monitor'] is not None:
            mon = self.s.data_column_by_name(self.params['monitor'])
            if self.params['data2cxi']:
                self.raw_data_monitor = mon
            mon0 = np.median(mon)
            mon /= mon0
            self.validframes = np.where(mon > 0.1)
            if len(self.validframes) != len(mon):
                self.print('WARNING: The following frames have a monitor value < 0.1'
                           ' the median value and will be ignored (no beam ?)')
                self.print(np.where(mon <= (mon0 * 0.1)))
            self.x = np.take(self.x, self.validframes)
            self.y = np.take(self.y, self.validframes)
            imgn = np.take(imgn, self.validframes)
        else:
            mon = None

        if 'xyrange' in self.params:
            xmin, xmax, ymin, ymax = self.params['xyrange']
            print("Restricting scan positions to %f < x < %f and %f < y < %f" % (xmin, xmax, ymin, ymax))
            idx = np.where((self.x >= xmin) * (self.x <= xmax) * (self.y >= ymin) * (self.y <= ymax))[0]
            if len(idx) < 10:
                raise PtychoRunnerException("Only %d points remaining after applying xyrange"
                                            "constraint. original range: %5e<x<%5e %5e<y<%5e"
                                            % (len(idx), self.x.min(), self.x.max(), self.y.min(), self.y.max()))
            else:
                print("   ... %d/%d remaining positions" % (len(idx), len(x)))
            imgn = imgn.take(idx)
            self.x = self.x.take(idx)
            self.y = self.y.take(idx)

        if self.params['moduloframe'] is not None:
            n1, n2 = self.params['moduloframe']
            idx = np.where(imgn % n1 == n2)[0]
            imgn = imgn.take(idx)
            self.x = self.x.take(idx)
            self.y = self.y.take(idx)

        if self.params['maxframe'] is not None:
            N = self.params['maxframe']
            if len(imgn) > N:
                self.print("MAXFRAME: only using first %d frames" % (N))
                imgn = imgn[:N]
                self.x = self.x[:N]
                self.y = self.y[:N]

        # Check we really have frames available (save is enabled)
        if int(round(imgn.max() - imgn.min()) + 1) < len(imgn):
            if imgn.min() == imgn.max():
                raise PtychoRunnerException(
                    "Frame range (%d-%d) does not match expected number of frames (%d). Were frames SAVED ?"
                    % (imgn.min(), imgn.max(), len(imgn)))
            else:
                raise PtychoRunnerException("Frame range (%d-%d) does not match expected number of frames (%d)." % (
                    imgn.min(), imgn.max(), len(imgn)))

        self.imgn = imgn

    def load_data(self):
        self.s = SpecFile(self.params['specfile'])['%d.1' % (self.scan)]
        self.h = self.s.scan_header_dict

        date_string = self.h['D']  # 'Wed Mar 23 14:41:56 2016'
        date_string = date_string[date_string.find(' ') + 1:]
        pattern = '%b %d %H:%M:%S %Y'
        try:
            lc = locale._setlocale(locale.LC_ALL)
            locale._setlocale(locale.LC_ALL, 'C')
            epoch = int(time.mktime(time.strptime(date_string, pattern)))
            locale._setlocale(locale.LC_ALL, lc)
            self.print("SPEC date: %s -> %s " % (date_string, time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                                                            time.localtime(epoch))), epoch)
        except ValueError:
            self.print("Could not extract time from spec header,"
                       " unrecognized format: %s, expected:" % (date_string) + pattern)

        if self.params['nrj'] is None:
            self.print("Reading nrj", self.h['UMONO'])
            self.params['nrj'] = float(self.h['UMONO'].split('mononrj=')[1].split('ke')[0])

        if self.params['detectordistance'] is None and 'UDETCALIB' in self.h:
            if 'det_distance_CC=' in self.h['UDETCALIB']:
                # UDETCALIB cen_pix_x=18.347,cen_pix_y=278.971,pixperdeg=445.001,det_distance_CC=1402.175,det_distance_COM=1401.096,timestamp=20170926...
                self.params['detectordistance'] = float(self.h['UDETCALIB'].split('stance_CC=')[1].split(',')[0])
                self.print("Reading detector distance from spec data: det_distance_CC=%6.3fm"
                           % self.params['detectordistance'])

        if self.params['detectordistance'] is None:
            raise PtychoRunnerException('Missing argument: no detectordistance given or from scpec header (UDETCALIB)')

        # Load all frames
        imgn = self.imgn
        vimg = None
        d0 = 0
        t0 = timeit.default_timer()
        if self.params['kmapfile'] is not None:
            sys.stdout.write("Reading frames from KMAP file (this WILL take a while)...")
            sys.stdout.flush()
            kfile = fabio.open(self.params['kmapfile'])
            if kfile.getNbFrames() < len(imgn):
                raise PtychoRunnerException(
                    "KMAP: only %d frames instead of %d in data file (%s) ! Did you save all frames in a single file ?" %
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
            if self.params['imgname'] is None:
                if 'ULIMA_eiger2M' in self.h:
                    imgname = self.h['ULIMA_eiger2M'].strip().split('_eiger2M_')[0] + '_eiger2M_%05d.edf.gz'
                    self.print("Using Eiger 2M detector images: %s" % (imgname))
                else:
                    imgname = self.h['ULIMA_mpx4'].strip().split('_mpx4_')[0] + '_mpx4_%05d.edf.gz'
                    self.print("Using Maxipix mpx4 detector images: %s" % (imgname))
            else:
                imgname = self.params['imgname']
            if self.mpi_master:
                sys.stdout.write('Reading frames: ')
            ii = 0
            for i in imgn:
                if (i - imgn[0]) % 20 == 0 and self.mpi_master:
                    sys.stdout.write('%d ' % (i - imgn[0]))
                    sys.stdout.flush()
                frame = fabio.open(imgname % i).data
                if vimg is None:
                    vimg = np.empty((len(imgn), frame.shape[0], frame.shape[1]), dtype=frame.dtype)
                vimg[ii] = frame
                d0 += frame
                ii += 1
        self.print("\n")
        dt = timeit.default_timer() - t0
        print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, d0.size * len(vimg) / 1e6 / dt))

        self.raw_data = vimg
        self.load_data_post_process()


class PtychoRunnerID01(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(PtychoRunnerID01, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        This method should be superseded in a beamline/instrument-specific child class
        Returns:
            True if the argument is interpreted, false otherwise
        """
        if k == 'specfile' or k == 'imgcounter' or k == 'imgname' or k == 'kmapfile' or k == 'ptychomotors':
            self.params[k] = v
            return True
        elif k == 'nrj' or k == 'detectordistance' or k == 'pixelsize':
            self.params[k] = float(v)
            return True
        elif k == 'livescan':
            if v == None:
                self.params[k] = True
            else:
                self.params[k] = v
            return True
        elif k == 'xyrange':
            self.params[k] = [float(xy) for xy in v.split(',')]
        return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['specfile'] is None:
            raise PtychoRunnerException('Missing argument: no specfile given')
        if self.params['scan'] is None and self.params['livescan'] is False:
            raise PtychoRunnerException('Missing argument: no scan number given')
        if self.params['ptychomotors'] is None:
            raise PtychoRunnerException('Missing argument: ptychomotors')
        if self.params['kmapfile'] is not None and self.params['nrj'] is None:
            raise PtychoRunnerException('Missing argument: for KMAP data, nrj= (in keV) is required')
        if True:
            if self.params['imgname'] is not None:
                imgname = self.params['imgname']
                # NOTE: filling field with a numpy integer does not raise a:
                # "TypeError: not all arguments converted during string formatting"
                # But this error is raised if using a python integer as argument...
                if imgname % np.int64(1) == imgname % np.int64(2):
                    raise PtychoRunnerException('Error with imgname=%s : a field (e.g. %%05d) must be part of the name'
                                                % (self.params['imgname']))

    def process_scans(self):
        """
        Run all the analysis on the supplied scan list. This derived function handles the livescan option.

        :return: Nothing
        """
        if self.params['livescan'] is False:
            super(PtychoRunnerID01, self).process_scans()
        else:
            # Type of scan which will be searched for ptycho data
            scan_type = 'spiralscan'
            if self.params['livescan'] is not True:
                scan_type = self.params['livescan'].lower()
            iscan = -1
            if self.params['scan'] is not None:
                iscan = int(self.params['scan']) - 1
            while True:
                sf = SpecFile(self.params['specfile'])
                if len(sf.keys()) > iscan + 1:
                    for iscan in range(iscan + 1, len(sf.keys())):
                        s = sf[iscan]
                        h = s.scan_header_dict
                        if scan_type in h['S']:  # Match scan title ?
                            scan = sf.list()[iscan]
                            if iscan == len(sf.keys()) - 1:
                                # Last scan ? Wait 20s for new data
                                while True:
                                    nb = s.data.shape[1]
                                    print('#            LIVESCAN: scan #%d: %s (waiting 20s for new data)' %
                                          (sf.list()[iscan], h['S']))
                                    time.sleep(20)
                                    sf = SpecFile(self.params['specfile'])
                                    s = sf[iscan]
                                    if s.data.shape[1] == nb:
                                        break

                            nb = s.data.shape[1]
                            nb_min = 16
                            if nb < nb_min:
                                print('#' * 100)
                                print('#            LIVESCAN: skipping scan #%d (only %d<%d data points): %s' %
                                      (sf.list()[iscan], nb, nb_min, h['S']))
                                print('#' * 100)
                                continue

                            if self.params['data2cxi']:
                                filename = os.path.join(os.path.dirname(self.params['saveprefix'] % (scan, 0)),
                                                        'data.cxi')
                                if os.path.isfile(filename):
                                    print('#' * 100)
                                    print('#            LIVESCAN: data2cxi: skip scan #%d (no overwriting): %s' %
                                          (sf.list()[iscan], h['S']))
                                    print('#' * 100)
                                    continue

                            print('#' * 100)
                            print('#            LIVESCAN: reading scan #%d: %s' % (sf.list()[iscan], h['S']))
                            print('#' * 100)
                            try:
                                self.ws = self.PtychoRunnerScan(self.params, scan)
                                self.ws.prepare_processing_unit()
                                self.ws.load_data()
                                if self.params['data2cxi']:
                                    if self.params['data2cxi'] == 'crop':
                                        self.ws.center_crop_data()
                                        self.ws.ƒ(crop=True, verbose=True)
                                    else:
                                        self.ws.save_data_cxi(crop=False, verbose=True)
                                else:
                                    self.ws.center_crop_data()
                                    self.ws.prepare()
                                    self.ws.run()
                            except PtychoRunnerException as ex:
                                print(self.help_text)
                                print('\n\n Caught exception for scan %d: %s    \n' % (scan, str(ex)))
                                sys.exit(1)
                print("#            LIVESCAN: waiting for more scans (type='%s')..." % scan_type)
                time.sleep(10)
