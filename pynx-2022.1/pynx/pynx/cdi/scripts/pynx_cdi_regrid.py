#!/usr/bin/env python3
# coding: utf-8

"""
Rebuild the 3D reciprocal space
by projecting a set of 2d speckle SAXS pattern taken at various rotation angles
into a 3D regular volume
"""

__author__ = "Jérôme Kieffer"
__copyright__ = "2020 ESRF"
__license__ = "MIT"
__version__ = "0.9"
__date__ = "17/12/2020"

import sys
from pynx.cdi.runner.regrid import main

if __name__ == "__main__":
    result = main()
    sys.exit(result)
