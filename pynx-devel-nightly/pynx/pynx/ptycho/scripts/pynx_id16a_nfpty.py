#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from __future__ import division

import sys
from pynx.ptycho.runner import helptext_generic, PtychoRunnerException
from pynx.ptycho.runner.id16a_nfp import PtychoRunnerID16aNF, PtychoRunnerScanID16aNF, helptext_beamline, params

help_text = helptext_generic + helptext_beamline


def main():
    try:
        w = PtychoRunnerID16aNF(sys.argv, params, PtychoRunnerScanID16aNF)
        w.process_scans()
    except PtychoRunnerException as ex:
        print(help_text)
        print('\n\n Caught exception: %s    \n' % (str(ex)))
        sys.exit(1)


if __name__ == '__main__':
    main()
