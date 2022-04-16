# -*- coding: utf-8 -*-
from pynx.wavefront import fzp

from pynx.utils.plot_utils import complex2rgbalin, colorwheel
from pylab import *
import numpy as np

delta_min = 70e-9
wavelength = np.float32(12398.4 / 7000 * 1e-10)
rmax = np.float32(150e-6)  # radius of FZP
focal_length = 2 * rmax * delta_min / wavelength
print("FZP: diameter= %6.2fum, focal length=%6.2fcm" % (rmax * 2 * 1e6, focal_length * 100))
nr, ntheta = np.int32(1024), np.int32(512)  # number of points for integration on FZP
r_cs = np.float32(25e-6)  # Central stop radius
osa_z, osa_r = np.float32(focal_length - .021), np.float32(30e-6)  # OSA position and radius
sourcex = np.float32(0e-6)  # Source position
sourcey = np.float32(0e-6)
sourcez = np.float32(-90)
focal_point = 1 / (1 / focal_length - 1 / abs(sourcez))

gpu_name = "gpu" # will combine available gpu (experimental - it is safer to put part of the name of the gpu you want)

if True:
    # rectangular illumination ?
    xmin, xmax, ymin, ymax = 50e-6, 110e-6, -100e-6, 100e-6
    fzp_nx, fzp_ny = 512, 512
else:
    xmin, xmax, ymin, ymax = False, False, False, False
    fzp_nx, fzp_ny = None, None

if True:
    x = linspace(-.8e-6, .8e-6, 256)
    y = linspace(-.8e-6, .8e-6, 256)[:, newaxis]
    z = focal_point
else:
    x = linspace(-.8e-6, .8e-6, 256)[:, newaxis]
    y = 0
    z = focal_point + linspace(-2e-3, 2e-3, 256)

x = (x + (y + z) * 0).astype(float32)
y = (y + (x + z) * 0).astype(float32)
z = (z + (x + y) * 0).astype(float32)
nxyz = len(x.flat)

a_real = (x * 0).astype(np.float32)
a_imag = (x * 0).astype(np.float32)

a, dt, flop = fzp.FZP_thread(x, y, z, sourcex=sourcex, sourcey=sourcey, sourcez=sourcez, wavelength=wavelength,
                             focal_length=focal_length, rmax=rmax,
                             r_cs=r_cs, osa_z=osa_z, osa_r=osa_r, nr=nr, ntheta=ntheta,
                             fzp_xmin=xmin, fzp_xmax=xmax, fzp_ymin=ymin, fzp_ymax=ymax, fzp_nx=fzp_nx, fzp_ny=fzp_ny,
                             gpu_name=gpu_name, verbose=True)
print("clFZP dt=%9.5fs, %8.2f Gflop/s" % (dt, flop / 1e9 / dt))

if xmin is not None and xmax is not None:
  print("Correct phase for off-axis illumination")
  a *= exp(2j * pi * x * (xmin + xmax) / 2 / focal_length / wavelength)

clf()
zz = z - focal_point
if (x.max() - x.min()) <= 1e-8:
  # pylab.imshow(complex2rgba(a,amin=0.0,dlogs=2),extent=(z.min()*1e6,z.max()*1e6,y.min()*1e9,y.max()*1e9),aspect='equal',origin='lower')
  imshow(complex2rgbalin(a), extent=(zz.min() * 1e6, zz.max() * 1e6, y.min() * 1e9, y.max() * 1e9), aspect='equal', origin='lower')
  xlabel(r"$z\ (\mu m)$", fontsize=16)
  ylabel(r"$y\ (nm)$", fontsize=16)
elif (z.max() - z.min()) <= 1e-8:
  # pylab.imshow(complex2rgba(a,amin=0.0,dlogs=2),extent=(x.min()*1e6,x.max()*1e6,y.min()*1e6,y.max()*1e6),aspect='equal',origin='lower')
  imshow(complex2rgbalin(a), extent=(x.min() * 1e9, x.max() * 1e9, y.min() * 1e9, y.max() * 1e9), aspect='equal', origin='lower')
  xlabel(r"$x\ (nm)$", fontsize=16)
  ylabel(r"$y\ (nm)$", fontsize=16)
else:
  # pylab.imshow(complex2rgba(a,amin=0.0,dlogs=2),extent=(z.min()*1e3,z.max()*1e3,x.min()*1e6,x.max()*1e6),aspect='equal',origin='lower')
  imshow(complex2rgbalin(a), extent=(zz.min() * 1e6, zz.max() * 1e6, x.min() * 1e9, x.max() * 1e9), aspect='equal', origin='lower')
  xlabel(r"$z\ (\mu m)$", fontsize=16)
  ylabel(r"$x\ (nm)$", fontsize=16)

ax = axes((0.22, 0.76, 0.12, .12), facecolor='w')  # [left, bottom, width, height]
colorwheel()
