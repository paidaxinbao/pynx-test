#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from __future__ import division

import sys, os

from pynx.ptycho.runner import PtychoRunnerException, helptext_generic, params_generic
from pynx.ptycho.runner.hermes import PtychoRunnerHermes, PtychoRunnerScanHermes, helptext_beamline, params_beamline, \
    create_params

help_text = helptext_generic + helptext_beamline
params = create_params(params_generic, params_beamline)


def main():
    try:
        w = PtychoRunnerHermes(sys.argv, params, PtychoRunnerScanHermes)
        w.process_scans()
    except PtychoRunnerException as ex:
        print(help_text)
        print("\n\n Caught exception: %s    \n" % (str(ex)))
        sys.exit(1)

    print()
    print("Finished")
    print("####################################\n####################################\n")


if __name__ == "__main__":
    main()
