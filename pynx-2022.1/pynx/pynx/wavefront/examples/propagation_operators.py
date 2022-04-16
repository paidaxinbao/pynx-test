# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import timeit
from pylab import *
# The following import will use CUDA if available, otherwise OpenCL
from pynx.wavefront import *


if True:
    # Near field propagation of a simple 20x200 microns slit
    w = Wavefront(d=np.zeros((512, 512), dtype=np.complex64), pixel_size=1e-6, wavelength=1.5e-10)
    a = 20e-6 / 2
    x, y = w.get_x_y()
    w.set((abs(y) < a) * (abs(x) < 100e-6))
    w = PropagateNearField(0.5) * w
    # w = PropagateFRT(3) * w
    w = ImshowRGBA(fig_num=1, title="Near field propagation (0.5m) of a 20x200 microns aperture") * w

if True:
    # Near field propagation of a simple 40x200 microns slit, displaying the propagated wavefront by steps
    # of 0.2 m propagation
    w = Wavefront(d=np.zeros((512, 512), dtype=np.complex64), pixel_size=1e-6, wavelength=1.5e-10)
    a = 40e-6 / 2
    x, y = w.get_x_y()
    w.set((abs(y) < a) * (abs(x) < 100e-6))
    # Perform 15 near field propagation of 0.2m steps, displaying the complex wavefront each time
    # the **15 expression allows to repeat the series of operators 15 times.
    w = (ImshowRGBA(fig_num=2) * PropagateNearField(0.2))**15 * w

if True:
    w = Wavefront(d=np.zeros((1024, 1024), dtype=np.complex64), pixel_size=1.3e-6, wavelength=1.e-10)
    a = 43e-6 / 2
    x, y = w.get_x_y()
    w.set((abs(y) < a) * (abs(x) < 100e-6))
    # w = PropagateNearField(3) * w
    w = PropagateFRT(3) * w
    # w = PropagateFarField(20) * w
    # w = PropagateNearField(-3) * PropagateNearField(3) * w
    # w = PropagateFRT(3, forward=False) * PropagateFRT(3) * w
    # w = PropagateFarField(100, forward=False) * PropagateFarField(100) * w
    w = ImshowRGBA(fig_num=3, title="Fractional Fourier transform propagation (3m) of a 43x200 microns slit") * w

if True:
    # Jacques et al 2012 single slit setup - here with simulated 1 micron pixel
    # Compare with figure 7 for a=43,88,142,82 microns
    figure(figsize=(15, 10))
    for a in np.array([22, 43, 88, 142, 182]) * 1e-6 / 2:
        w = Wavefront(d=np.zeros((1024, 1024), dtype=np.complex64), wavelength=1e-10, pixel_size=1.3e-6)
        x, y = w.get_x_y()
        w.set(abs(y) < (a) + x * 0)  # +x*0 needed to make sure the array is 2D
        # w = PropagateNearField(3) * w
        w = PropagateFRT(3) * w
        # w = PropagateFarField(3) * w
        icalc = fftshift(abs(w.get())).mean(axis=1) ** 2
        x, y = w.get_x_y()
        plot(fftshift(y) * 1e6, icalc, label=u'a=%5.1f µm' % (a * 2e6))
        text(0, icalc[len(icalc) // 2], u'a=%5.1f µm' % (a * 2e6))
        print('a=%5.1fum, dark spot at a^2/(2pi*lambda)=%5.2fm, I[0]=%5.2f' % (
            2 * a * 1e6, (2 * a) ** 2 / (2 * pi * w.wavelength), icalc[len(icalc) // 2]))
    title("Propagation of a slit at 3 meters, wavelength= 0.1nm")
    legend()
    xlim(-100, 100)
    xlabel(u'X (µm)')

if True:
    # propagation of a stack of A x 200 microns apertures, varying A
    w = Wavefront(d=np.zeros((16, 512, 512), dtype=np.complex64), pixel_size=1e-6, wavelength=1.5e-10)
    x, y = w.get_x_y()
    d = w.get()
    for i in range(16):
        a = 5e-6 / 2 * (i + 1)
        d[i] = ((abs(y) < a) * (abs(x) < 100e-6))
    w.set(d)
    w = PropagateFRT(1.2) * w
    figure(figsize=(15, 10))
    x, y = w.get_x_y()
    x *= 1e6
    y *= 1e6
    for i in range(16):
        subplot(4, 4, i + 1)
        imshow(abs(fftshift(w.get()[i])), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
        title(u"A=%dµm" % (10 * (i + 1)))
        if i >= 12:
            xlabel(u'X (µm)')
        if i % 4 == 0:
            ylabel(u'Y (µm)')
        xlim(-150, 150)
        ylim(-100, 100)
    suptitle(u"Fractional Fourier propagation (0.5m) of a A x 200 µm aperture")

if True:
    # Using a lens plus near field propagation to focus a 40x80 microns^ wavefront
    w = Wavefront(d=np.zeros((512, 512), dtype=np.complex64), pixel_size=1e-6, wavelength=1.5e-10)
    x, y = w.get_x_y()
    w.set((abs(y) < 20e-6) * (abs(x) < 40e-6))
    w = PropagateNearField(1.5) * ThinLens(focal_length=2) * w
    w = ImshowAbs(fig_num=6, title="1.5m propagation after a f=2m lens") * w

if True:
    # Time propagation of stacks of 1024x1024 wavefronts
    for nz in [1, 1, 10, 50, 100, 200]:  # First size is repeated to avoid counting initializations
        t0 = timeit.default_timer()
        d = np.zeros((nz, 512, 512), dtype=np.complex64)
        w = Wavefront(d=d, pixel_size=1e-6, wavelength=1.5e-10)
        print("####################### Stack size: %4d x %4d *%4d ################" % w.get().shape)
        x, y = w.get_x_y()
        a = 20e-6 / 2
        d[:] = (abs(y) < a) * (abs(x) < 100e-6)
        w.set(d)
        t1 = timeit.default_timer()
        print("%30s: dt=%6.2fms" % ("Wavefront creation (CPU)", (t1 - t0) * 1000))
        w = PropagateFRT(1.2) * w
        t2 = timeit.default_timer()
        print("%30s: dt=%6.2fms" % ("Copy to GPU and propagation", (t2 - t1) * 1000))
        w = PropagateFRT(1.2) * w
        t3 = timeit.default_timer()
        print("%30s: dt=%6.2fms" % ("Propagation", (t3 - t2) * 1000))
        w = FreePU() * w  # We use FreePU() to make sure we release GPU memory as fast as possible
        t4 = timeit.default_timer()
        print("%30s: dt=%6.2fms" % ("Copy from GPU", (t4 - t3) * 1000))
