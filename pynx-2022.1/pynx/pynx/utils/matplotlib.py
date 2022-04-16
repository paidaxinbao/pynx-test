# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


# Just a workaround to switch back to Agg if the default backend (Tk) is not available, when importing pyplot
try:
    import matplotlib.pyplot as pyplot

except ImportError:
    import warnings
    import os

    warnings.warn("Failed importing matplotlib.pyplot. Is tkinter missing ? Trying now to switch to Agg..")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as pyplot
    except ImportError:
        import sys
        t, v, tb = sys.exc_info()
        raise t(v).with_traceback(tb)
