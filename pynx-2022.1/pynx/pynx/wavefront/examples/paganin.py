# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import numpy as np
from scipy import misc
from pylab import *
# The following import will use CUDA if available, otherwise OpenCL
from pynx.wavefront import *

################################################################################################################
# Create a wavefront as a simple transmission through a rectangular object
w = Wavefront(d=np.zeros((512, 512), dtype=np.complex64), pixel_size=1e-6, wavelength=1.5e-10)
a, b = 100e-6 / 2, 200e-6 / 2
x, y = w.get_x_y()
d = ((abs(y) < a) * (abs(x) < b))
delta = 1e-6
beta = 1e-9
thickness = 1e-6
mu = 4 * np.pi * beta / w.wavelength
k = 2 * np.pi / w.wavelength
print("       mu * t = %f\nk * delta * t = %f" % (mu * thickness, k * delta * thickness))
w.set(exp(1j * k * (-delta + 1j * beta) * thickness * d))

# Display the amplitude
w = ImshowAbs(fig_num=1) * w

# Propagate and display amplitude
w = ImshowAbs(fig_num=2) * PropagateNearField(0.5) * w

# Reconstruct original wavefield using Paganin's equation (only using propagated intensity, phase is discarded)
w = BackPropagatePaganin(dz=0.5) * w

# Display reconstructed Amplitude, phase
w = ImshowAngle(fig_num=3) * ImshowAbs(fig_num=4) * w

# Display the calculated thickness from the reconstructed phase
figure(5)
imshow(fftshift(-np.angle(w.get())/(k*delta)), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
title("Back-propagation using Paganin's approach")
xlabel(u'X (µm)')
ylabel(u'Y (µm)')
c=colorbar()

################################################################################################################
# Slightly more complicated example
w = Wavefront(d=np.zeros((512, 512), dtype=np.complex64), pixel_size=1e-6, wavelength=1.5e-10)
delta = 1e-6
beta = 1e-9
mu = 4 * np.pi * beta / w.wavelength
k = 2 * np.pi / w.wavelength
print("       mu * t = %f\nk * delta * t = %f" % (mu * thickness, k * delta * thickness))
w.set(exp(1j * k * (-delta + 1j * beta) * fftshift(misc.ascent()) * 1e-7))
d1 = w.get(shift=True).copy()
# Propagate and reconstruct
dz = 0.5
w = PropagateNearField(dz) * w
d2 = w.get(shift=True).copy()
w = BackPropagatePaganin(dz=dz) * w
d3 = w.get(shift=True).copy()
# Display
figure(6, figsize=(10, 8))
subplot(221)
imshow(abs(d1))
title("Amplitude (z=0)")
colorbar()
subplot(222)
imshow(abs(d2))
title("Amplitude (z=%5.2fm)" % dz)
colorbar()
subplot(223)
imshow(abs(d3))
title("Reconstructed Amplitude (Paganin)")
colorbar()
subplot(224)
imshow(np.angle(d3))
title("Reconstructed Phase (Paganin)")
colorbar()
