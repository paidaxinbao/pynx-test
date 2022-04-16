# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import numpy as np
from scipy.fftpack import ifftshift, fftshift, fft2
from matplotlib import pyplot as plt
from pynx.utils.pattern import fibonacci_urchin
from pynx.cdi import *

# Test on a simulated pattern
n = 128

# Object coordinates
tmp = np.arange(-n // 2, n // 2, dtype=np.float32)
z, y, x = np.meshgrid(tmp, tmp, tmp, indexing='ij')
r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

support = None
if False:
    # 'Urchin' object (3d analog to a Siemens star
    obj0 = fibonacci_urchin(n, nb_rays=12, r_max=30, nb_rings=2)
    # Start from a slightly loose support
    support = r < 35
else:
    # Parallelepiped - should be trivial to reconstruct !
    obj0 = (abs(x) < 12) * (abs(y) < 10) * (abs(z) < 16)
    # Start from a slightly loose support
    support = (abs(x) < 20) * (abs(y) < 20) * (abs(z) < 25)

iobs = abs(ifftshift(fft2(fftshift(obj0.astype(np.complex64))))) ** 2
iobs = np.random.poisson(iobs * 1e10 / iobs.sum())
mask = np.zeros_like(iobs, dtype=np.int8)
if False:
    # Try masking some values in the central beam - more difficult !!
    iobs[255:257, 255:257, 255:257] = 0
    mask[255:257, 255:257, 255:257] = 1

plt.figure(figsize=(8, 8))

cdi = CDI(fftshift(iobs), obj=None, support=fftshift(support), mask=fftshift(mask), wavelength=1e-10,
          pixel_size_detector=55e-6)

# Init real object from the chosen support
cdi = InitObjRandom(src="support", amin=0.8, amax=1, phirange=0) * cdi

# Do 50*4 cycles of HIO, displaying object every 50 cycle
cdi = (RAAR(calc_llk=50, show_cdi=50, positivity=True) ** 50) ** 4 * cdi

# Support update operator
sup = SupportUpdate(threshold_relative=0.4, smooth_width=(2, 0.25, 600), force_shrink=False)

# Cycles with the support update
cdi = (ER(positivity=True, calc_llk=40, show_cdi=40) ** 5 * RAAR(positivity=True, calc_llk=40) ** 40) ** 20 * cdi

# Finish with ER then ML
cdi = ER(positivity=True, calc_llk=40, show_cdi=40) ** 200 * cdi
