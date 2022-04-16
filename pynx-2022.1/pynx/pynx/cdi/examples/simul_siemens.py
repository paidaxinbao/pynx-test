# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import numpy as np
from scipy.fftpack import ifftshift, fftshift, fft2

# NB: It is possible to use the PYNX_PU environment variable to choose the GPU or language,
# e.g. using PYNX_PU=opencl or PYNX_PU=cuda or PYNX_PU=cpu or PYNX_PU=Titan, etc..

from pynx.utils.pattern import siemens_star
from pynx.cdi import *

# Note that this is a moderately difficult example: it is easy to get the outline of
# the siemens star, but the many voids within the structure can be difficult to
# reconstruct, especially when using the option to remove intensity in the centre.
# With this beamstop this needs to be run a number of time to get a really good result.

# Test on a simulated pattern (2D)
n = 512

# Siemens-Star object
obj0 = siemens_star(n, nb_rays=18, r_max=60, nb_rings=3, cheese_holes_nb=40)

# Start from a slightly loose disc support
x, y = np.meshgrid(np.arange(-n // 2, n // 2, dtype=np.float32), np.arange(-n // 2, n // 2, dtype=np.float32))
r = np.sqrt(x ** 2 + y ** 2)
support = r < 65

iobs = abs(ifftshift(fft2(fftshift(obj0.astype(np.complex64))))) ** 2
iobs = np.random.poisson(iobs * 1e10 / iobs.sum())
mask = np.zeros_like(iobs, dtype=np.int16)
if True:
    # Mask some values in the central beam
    print("Removing %6.3f%% intensity (more difficult, retry to get a good result" %
          (iobs[255:257, 255:257].sum() / iobs.sum() * 100))
    iobs[255:257, 255:257] = 0
    mask[255:257, 255:257] = 1

cdi = CDI(fftshift(iobs), obj=None, support=fftshift(support), mask=fftshift(mask), wavelength=1e-10,
          pixel_size_detector=55e-10)

cdi.init_free_pixels()

# Init real object from the chosen support
cdi = InitObjRandom(src="support", amin=0.8, amax=1, phirange=0) * cdi

# Initial scaling of the object [ only useful if there are masked pixels !]
cdi = ScaleObj(method='F') * cdi

show = 40

# Do 100 cycles of RAAR
cdi = RAAR(calc_llk=20) ** 100 * cdi

if show > 0:
    cdi = ShowCDI(fig_num=1) * cdi

sup = SupportUpdate(threshold_relative=0.3, smooth_width=(2.0, 0.5, 800), force_shrink=False)

cdi = (sup * ER(calc_llk=20, show_cdi=show, fig_num=1) ** 5 * RAAR(calc_llk=20, show_cdi=show,
                                                                   fig_num=1) ** 40) ** 5 * cdi
cdi = DetwinRAAR(nb_cycle=10) * cdi
cdi = (sup * ER(calc_llk=20, show_cdi=show, fig_num=1) ** 5 * RAAR(calc_llk=20, show_cdi=show,
                                                                   fig_num=1) ** 40) ** 20 * cdi

# Finish with ML or ER
# cdi = ML(reg_fac=1e-2, calc_llk=5, show_cdi=show, fig_num=1) ** 100 * cdi
cdi = ER(calc_llk=20, show_cdi=show, fig_num=1) ** 100 * cdi

cdi = ShowCDI(fig_num=1) * cdi
