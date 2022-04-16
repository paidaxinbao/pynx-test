# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fftshift
import fabio
from matplotlib import pyplot as plt

from pynx.cdi import *

# CDI example from an experimental data set from id10 (courtesy of Yuriy Chushkin)

iobs = np.flipud(fabio.open("data/logo5mu_20sec.edf").data)
support = np.flipud(fabio.open("data/mask_logo5mu_20sec.edf").data)
n = len(iobs)
x, y = np.meshgrid(np.arange(0, n, dtype=np.float32), np.arange(0, n, dtype=np.float32))

# Mask specific to this dataset (from beamstop, after symmetrization of observed data)
mask = np.logical_or(iobs != 0, np.logical_or(abs(x - 258) > 30, abs(y - 258) > 30))
mask *= np.logical_or(iobs != 0, np.logical_or(abs(x - 246) > 30, abs(y) < 495))
mask *= np.logical_or(iobs != 0, np.logical_or(abs(x - 266) > 30, abs(y) > 20))
mask *= np.logical_or(iobs != 0, np.logical_or(abs(x - 10) > 30, abs(y - 270) > 5))
mask *= np.logical_or(iobs != 0, np.logical_or(abs(x - 498) > 30, abs(y - 240) > 5))
mask *= np.logical_or(iobs != 0, np.logical_or(abs(x) > 30, abs(y) > 20))
mask *= np.logical_or(iobs != 0, np.logical_or(abs(x - 510) > 30, abs(y - 510) > 20))
mask = (mask == 0)  # 0: OK, 1: masked

plt.figure(1, figsize=(8, 8))

# ========================= Try first from the known support (easy !) =============================
cdi = CDI(fftshift(iobs), obj=None, support=fftshift(support), mask=fftshift(mask), wavelength=1e-10,
          pixel_size_detector=55e-6)

# Set free mask for unbiased likelihood estimation
cdi.init_free_pixels()

# Init real object from the supplied support
cdi = InitObjRandom(src="support", amin=0.8, amax=1, phirange=0) * cdi

# Initial scaling, required by mask
cdi = ScaleObj(method='F') * cdi

# Do 4 * (50 cycles of HIO + 20 of ER), displaying object after each group of cycle
cdi = (ShowCDI(fig_num=1) * ER(calc_llk=10) ** 20 * HIO(calc_llk=20) ** 50) ** 4 * cdi

# cdi = (ShowCDI(fig_num=1) * ER(calc_llk=10) ** 20 * CF(calc_llk=20) ** 50) ** 4 * cdi

# cdi = (ShowCDI(fig_num=1) * ER(calc_llk=10) ** 20 * RAAR(calc_llk=20) ** 50) ** 4 * cdi

print("\n======================================================================\n")
print("This was too easy - start again from a loose support !\n")

# ========================= Try again from a loose support ========================================
support = np.flipud(fabio.open("data/mask_logo5mu_20sec.edf").data)
support = gaussian_filter(support.astype(np.float32), 4) > 0.2
cdi = CDI(fftshift(iobs), obj=None, support=fftshift(support), mask=fftshift(mask), wavelength=1e-10,
          pixel_size_detector=55e-6)

# Set free mask for unbiased likelihood estimation
cdi.init_free_pixels()

# Init real object from the chosen support
cdi = InitObjRandom(src="support", amin=0.8, amax=1, phirange=0) * cdi

# Initial scaling, required by mask
cdi = ScaleObj(method='F') * cdi

# Display every N cycles
show = 100

# Do 200 cycles of HIO, displaying object every N cycle and log-likelihood every 20 cycle
cdi = HIO(calc_llk=20, show_cdi=show, fig_num=2) ** 200 * cdi

# Support update operator
sup = SupportUpdate(threshold_relative=0.25, smooth_width=(2, 0.5, 800), force_shrink=False)

if True:
    # Do 40 cycles of HIO, then 5 of ER, update support, repeat
    cdi = (sup * ER(calc_llk=20, show_cdi=show, fig_num=2) ** 5
           * HIO(calc_llk=20, show_cdi=show, fig_num=2) ** 40) ** 20 * cdi
else:
    # Do 40 cycles of HIO, update support, repeat
    cdi = (sup * HIO(calc_llk=20, show_cdi=show, fig_num=2) ** 40) ** 20 * cdi

# Finish with ML or ER
# cdi = ML(reg_fac=1e-2, calc_llk=20, show_cdi=show, fig_num=1) ** 100 * cdi
cdi = ER(calc_llk=20, show_cdi=show, fig_num=2) ** 100 * cdi
