#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import timeit
import numpy as np

from .runner import CDIRunner, CDIRunnerException, CDIRunnerScan, params_generic
from pynx.cdi import *

params_beamline = {'auto_center_resize': False, 'support_type': 'circle', 'detwin': False, 'support_size': 50,
                   'nb_raar': 0, 'nb_hio': 600, 'nb_er': 200, 'nb_ml': 0, 'instrument': 'ESRF id10', 'mask': 'zero',
                   'zero_mask': 0, 'positivity': True}

helptext_beamline = """
Script to perform a CDI reconstruction of data from id10@ESRF.
command-line/file parameters arguments: (all keywords are case-insensitive):

    Specific defaults for this script:
        auto_center_resize=True
        detwin = False
        mask = zero  # use 'mask=no' from the command-line to avoid using a mask
        nb_raar = 600
        nb_hio = 0
        nb_er = 200
        nb_ml = 0
        support_size = 50
        support_type = circle
        zero_mask = 0
        positivity = True
"""

params = params_generic.copy()
for k, v in params_beamline.items():
    params[k] = v


class CDIRunnerScanID10(CDIRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(CDIRunnerScanID10, self).__init__(params, scan, timings=timings)

    def prepare_cdi(self):
        """
        Prepare CDI object from input data.

        :return: nothing. Creates or updates self.cdi object.
        """
        super(CDIRunnerScanID10, self).prepare_cdi()
        # Scale initial object (unnecessary if auto-correlation is used)
        if self.params['support'] != 'auto':
            self.cdi = ScaleObj(method='F', lazy=True) * self.cdi


class CDIRunnerID10(CDIRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        super(CDIRunnerID10, self).__init__(argv, params, ptycho_runner_scan_class)
        self.help_text += helptext_beamline

    # def parse_arg_beamline(self, k, v):
    #     """
    #     Parse argument in a beamline-specific way. This function only parses single arguments.
    #     If an argument is recognized and interpreted, the corresponding value is added to self.params
    #
    #     This method should be superseded in a beamline/instrument-specific child class
    #     Returns:
    #         True if the argument is interpreted, false otherwise
    #     """
    #     if k in []:
    #         self.params[k] = v
    #         return True
    #     elif k in []:
    #         self.params[k] = float(v)
    #         return True
    #     elif k in []:
    #         if v is None:
    #             self.params[k] = True
    #             return True
    #         elif type(v) is bool:
    #             self.params[k] = v
    #             return True
    #         elif type(v) is str:
    #             if v.lower() == 'true' or v == '1':
    #                 self.params[k] = True
    #                 return True
    #             else:
    #                 self.params[k] = False
    #                 return True
    #         else:
    #             return False
    #     elif k in []:
    #         self.params[k] = int(v)
    #         return True
    #     return False

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['data'] is None:
            raise CDIRunnerException('No data provided. Need at least data=..., or a parameters input file')
