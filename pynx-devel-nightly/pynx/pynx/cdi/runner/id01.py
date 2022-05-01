#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import sys
import timeit
import numpy as np
import fabio
from silx.io.specfile import SpecFile
from .runner import CDIRunner, CDIRunnerException, CDIRunnerScan, params_generic
from pynx.cdi import *

params_beamline = {'detwin': True, 'support_size': None, 'nb_raar': 600, 'nb_hio': 0, 'nb_er': 200, 'nb_ml': 0,
                   'instrument': 'ESRF id01', 'specfile': None, 'imgcounter': 'auto', 'imgname': None, 'scan': None,
                   'auto_center_resize': True, 'zero_mask': 'auto'}

helptext_beamline = """
Script to perform a CDI reconstruction of data from id01@ESRF.
command-line/file parameters arguments: (all keywords are case-insensitive):

    specfile=/some/dir/to/specfile.spec: path to specfile [mandatory, unless data= is used instead]

    scan=56: scan number in specfile [mandatory].
             Alternatively a list or range of scans can be given:
                scan=12,23,45 or scan="range(12,25)" (note the quotes)

    imgcounter=mpx4inr: spec counter name for image number
                        [default='auto', will use either 'mpx4inr' or 'ei2mint']

    imgname=/dir/to/images/prefix%05d.edf.gz: images location with prefix 
            [default: will be extracted from the ULIMA_mpx4 entry in the spec scan header]
            
    Specific defaults for this script:
        auto_center_resize = True
        detwin = True
        nb_raar = 600
        nb_hio = 0
        nb_er = 200
        nb_ml = 0
        support_size = None
        zero_mask = auto
"""

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class CDIRunnerScanID01(CDIRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(CDIRunnerScanID01, self).__init__(params, scan, timings=timings)

    def load_data(self):
        """
        Loads data. If no id01-specific keywords have been supplied, use the default data loading.

        """
        if self.params['specfile'] is None or self.scan is None:
            super(CDIRunnerScanID01, self).load_data()
        else:
            if isinstance(self.scan, str):
                # do we combine several scans ?
                vs = self.scan.split('+')
            else:
                vs = [self.scan]
            imgn = None
            scan_motor_last_value = None
            for scan in vs:
                if scan is None:
                    scan = 0
                else:
                    scan = int(scan)
                s = SpecFile(self.params['specfile'])['%d.1' % (scan)]
                h = s.scan_header_dict

                if self.params['imgcounter'] == 'auto':
                    if 'ei2minr' in s.labels:
                        self.params['imgcounter'] = 'ei2minr'
                    elif 'mpx4inr' in s.labels:
                        self.params['imgcounter'] = 'mpx4inr'
                    print("Using image counter: %s" % (self.params['imgcounter']))

                if self.params['wavelength'] is None and 'UMONO' in h:
                    nrj = float(h['UMONO'].split('mononrj=')[1].split('ke')[0])
                    w = 12.384 / nrj
                    self.params['wavelength'] = w
                    print("Reading nrj from spec data: nrj=%6.3fkeV, wavelength=%6.3fA" % (nrj, w))

                if 'UDETCALIB' in h:
                    # UDETCALIB cen_pix_x=18.347,cen_pix_y=278.971,pixperdeg=445.001,det_distance_CC=1402.175,
                    # det_distance_COM=1401.096,timestamp=20170926...
                    if self.params['detector_distance'] is None:
                        if 'det_distance_CC=' in h['UDETCALIB']:
                            self.params['detector_distance'] = float(
                                h['UDETCALIB'].split('stance_CC=')[1].split(',')[0])
                            print("Reading detector distance from spec data: %6.3fm" % self.params['detector_distance'])
                        else:
                            print(
                                'No detector distance given. No det_distance_CC in UDETCALIB ??: %s' % (h['UDETCALIB']))
                # Also save specfile parameters as a 'user_' param
                self.params['user_id01_specfile_header'] = s.file_header_dict
                self.params['user_id01_specfile_scan_header'] = s.scan_header_dict

                # Read images
                if imgn is None:
                    imgn = s.data_column_by_name(self.params['imgcounter']).astype(np.int)
                else:
                    # try to be smart: exclude first image if first motor position is at the end of last scan
                    if scan_motor_last_value == s.data[0, 0]:
                        i0 = 0
                    else:
                        print("Scan %d: excluding first image at same position as previous one" % (scan))
                        i0 = 1
                    imgn = np.append(imgn, s.data_column_by_name(self.params['imgcounter'])[i0:].astype(np.int))
                scan_motor_last_value = s.data[0, 0]
            self.iobs = None
            t0 = timeit.default_timer()
            if self.params['imgname'] == None:
                if 'ULIMA_eiger2M' in h:
                    imgname = h['ULIMA_eiger2M'].strip().split('_eiger2M_')[0] + '_eiger2M_%05d.edf.gz'
                    print("Using Eiger 2M detector images: %s" % (imgname))
                else:
                    imgname = h['ULIMA_mpx4'].strip().split('_mpx4_')[0] + '_mpx4_%05d.edf.gz'
                    print("Using Maxipix mpx4 detector images: %s" % (imgname))
            else:
                imgname = self.params['imgname']
            sys.stdout.write('Reading frames: ')
            ii = 0
            for i in imgn:
                if (i - imgn[0]) % 20 == 0:
                    sys.stdout.write('%d ' % (i - imgn[0]))
                    sys.stdout.flush()
                frame = fabio.open(imgname % i).data
                if self.iobs is None:
                    self.iobs = np.empty((len(imgn), frame.shape[0], frame.shape[1]), dtype=frame.dtype)
                self.iobs[ii] = frame
                ii += 1
            print("\n")
            dt = timeit.default_timer() - t0
            print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, self.iobs.size / 1e6 / dt))

    def prepare_cdi(self):
        """
        Prepare CDI object from input data.

        :return: nothing. Creates or updates self.cdi object.
        """
        super(CDIRunnerScanID01, self).prepare_cdi()
        # Scale initial object (unnecessary if auto-correlation is used)
        if self.params['support'] != 'auto':
            self.cdi = ScaleObj(method='F', lazy=True) * self.cdi


class CDIRunnerID01(CDIRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(CDIRunnerID01, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        This method should be superseded in a beamline/instrument-specific child class
        Returns:
            True if the argument is interpreted, false otherwise
        """
        if k in ['specfile', 'imgcounter', 'imgname', 'scan']:
            self.params[k] = v
            return True
        return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        print()
        if self.params['data'] is None and (self.params['specfile'] is None or self.params['scan'] is None):
            raise CDIRunnerException('No data provided. Need at least data=, or specfile= and scan=')
