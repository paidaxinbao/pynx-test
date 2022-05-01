# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2016-present : Univ. Grenoble Alpes, CEA/INAC/SP2M
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       author: Vincent Favre-Nicolin vincent.favre-nicolin@univ-grenoble-alpes.fr, favre@esrf.fr

import time
import sys
import unittest
from numpy import arange, sin, exp, pi, newaxis, float32, random, ones, isnan, sqrt
from pynx import scattering


def fhkl(gpu_name, nx=40, ny=40, nz=40, nh=40, nk=40, nl=40, verbose=False,
         language="OpenCL", cl_platform="", return_array=False):
    # Create array of 3D coordinates, 50x50x50 cells
    x = arange(0, nx, dtype=float32)
    y = arange(0, ny, dtype=float32)[:, newaxis]
    z = arange(0, nz, dtype=float32)[:, newaxis, newaxis]
    # HKL coordinates as a 2D array
    h = random.uniform(.01, .5, nh)
    k = random.uniform(.01, .5, nk)[:, newaxis]
    l = random.uniform(2.01, 2.5, nl)[:, newaxis, newaxis]
    # The actual computation (done twice - first time to compile kernel)
    fhkl_, dt = scattering.Fhkl_thread(h, k, l, x, y, z, gpu_name=gpu_name, verbose=verbose,
                                       language=language, cl_platform=cl_platform)
    fhkl_, dt = scattering.Fhkl_thread(h, k, l, x, y, z, gpu_name=gpu_name, verbose=verbose,
                                       language=language, cl_platform=cl_platform)
    # Compare to analytical formula
    fhkl_gold = (exp(-1j * pi * h * (nx - 1)) * sin(pi * h * nx) / sin(pi * h)) * (
            exp(-1j * pi * k * (ny - 1)) * sin(pi * k * ny) / sin(pi * k)) * (
                        exp(-1j * pi * l * (nz - 1)) * sin(pi * l * nz) / sin(pi * l))
    diff = abs(fhkl_ - fhkl_gold)
    if return_array:
        return diff
    tmp = diff.mean() / abs(fhkl_gold).mean()
    s = "PASS"
    if tmp > 0.01 or isnan(tmp):
        s = "FAIL"
    print("%20s: %5d 10^3 reflections, %5d 10^3 atoms, speed=%7.3f 10^9 reflections.atoms/s  "
          "=>   <|GPU-analytical|> / <|analytical|>=%7.5f, %10s" %
          ("fhkl", nh * nk * nl // 1000, nx * ny * nz // 1000, nx * ny * nz * nh * nk * nl / dt / 1e9, tmp, s))
    return tmp < 0.01


def fhklo(gpu_name, nx=40, ny=40, nz=40, nh=40, nk=40, nl=40, verbose=False, language="OpenCL",
          cl_platform="", return_array=False):
    # Create array of 3D coordinates, 50x50x50 cells
    x = arange(0, nx, dtype=float32)
    y = arange(0, ny, dtype=float32)[:, newaxis]
    z = arange(0, nz, dtype=float32)[:, newaxis, newaxis]  # z negative, atoms below the surface
    occ = ones((x + y + z).shape) * 0.4
    # HKL coordinates as a 2D array
    h = random.uniform(.01, .5, nh)
    k = random.uniform(.01, .5, nk)[:, newaxis]
    l = random.uniform(2.01, 2.5, nl)[:, newaxis, newaxis]
    # The actual computation (done twice - first time to compile kernel)
    fhkl_, dt = scattering.Fhkl_thread(h, k, l, x, y, z, occ, gpu_name=gpu_name, verbose=verbose,
                                       language=language, cl_platform=cl_platform)
    fhkl_, dt = scattering.Fhkl_thread(h, k, l, x, y, z, occ, gpu_name=gpu_name, verbose=verbose,
                                       language=language, cl_platform=cl_platform)
    # Compare to analytical formula
    fhkl_gold = occ.mean() * (exp(-1j * pi * h * (nx - 1)) * sin(pi * h * nx) / sin(pi * h)) * (
            exp(-1j * pi * k * (ny - 1)) * sin(pi * k * ny) / sin(pi * k)) * (
                        exp(-1j * pi * l * (nz - 1)) * sin(pi * l * nz) / sin(pi * l))
    diff = abs(fhkl_ - fhkl_gold)
    tmp = diff.mean() / abs(fhkl_gold).mean()
    s = "PASS"
    if tmp > 0.01 or isnan(tmp): s = "FAIL"
    print("%20s: %5d 10^3 reflections, %5d 10^3 atoms, speed=%7.3f 10^9 reflections.atoms/s  "
          "=>   <|GPU-analytical|> / <|analytical|>=%7.5f, %10s" % (
              "fhklo", nh * nk * nl // 1000, nx * ny * nz // 1000, nx * ny * nz * nh * nk * nl / dt / 1e9, tmp, s))
    if return_array:
        return diff
    return tmp < 0.01


def fhklo_graz(gpu_name, nx=40, ny=40, nz=40, nh=40, nk=40, nl=40, verbose=False,
               language="OpenCL", cl_platform="", return_array=False):
    # Create array of 3D coordinates, 50x50x50 cells
    x = arange(0, nx, dtype=float32)
    y = arange(0, ny, dtype=float32)[:, newaxis]
    z = -arange(0, nz, dtype=float32)[:, newaxis, newaxis]  # z negative, atoms below the surface
    occ = ones((x + y + z).shape) * 0.4
    # HKL coordinates as a 2D array
    h = random.uniform(.01, .5, nh)
    k = random.uniform(.01, .5, nk)[:, newaxis]
    l = random.uniform(2.01, 2.5, nl)[:, newaxis, newaxis]
    sz_imag = ones(l.shape) * 0.01
    # The actual computation (done twice - first time to compile kernel)
    fhkl_, dt = scattering.Fhkl_thread(h, k, l, x, y, z, occ, gpu_name=gpu_name, sz_imag=sz_imag,
                                       verbose=verbose, language=language, cl_platform=cl_platform)
    fhkl_, dt = scattering.Fhkl_thread(h, k, l, x, y, z, occ, gpu_name=gpu_name, sz_imag=sz_imag,
                                       verbose=verbose, language=language, cl_platform=cl_platform)
    # Compare to analytical formula
    fhkl_gold = occ.mean() * (exp(-1j * pi * h * (nx - 1)) * sin(pi * h * nx) / sin(pi * h)) * \
                (exp(-1j * pi * k * (ny - 1)) * sin(pi * k * ny) / sin(pi * k)) * \
                (exp(2 * pi * (1j * l + sz_imag) * nz) - 1) / (exp(2 * pi * (1j * l + sz_imag)) - 1)
    diff = abs(fhkl_ - fhkl_gold)
    tmp = diff.mean() / abs(fhkl_gold).mean()
    s = "PASS"
    if tmp > 0.01 or isnan(tmp): s = "FAIL"
    print("%20s: %5d 10^3 reflections, %5d 10^3 atoms, speed=%7.3f 10^9 reflections.atoms/s  "
          "=>   <|GPU-analytical|> / <|analytical|>=%7.5f, %10s" %
          ("fhklo_graz", nh * nk * nl // 1000, nx * ny * nz // 1000,
           nx * ny * nz * nh * nk * nl / dt / 1e9, tmp, s))
    if return_array:
        return diff
    return tmp < 0.01


def fhkl_graz(gpu_name, nx=40, ny=40, nz=40, nh=40, nk=40, nl=40, verbose=False, language="OpenCL",
              cl_platform="", return_array=False):
    # Create array of 3D coordinates, 50x50x50 cells
    x = arange(0, nx, dtype=float32)
    y = arange(0, ny, dtype=float32)[:, newaxis]
    z = -arange(0, nz, dtype=float32)[:, newaxis, newaxis]  # z negative, atoms below the surface
    # HKL coordinates as a 2D array
    h = random.uniform(.01, .5, nh)
    k = random.uniform(.01, .5, nk)[:, newaxis]
    l = random.uniform(2.01, 2.5, nl)[:, newaxis, newaxis]
    sz_imag = ones(l.shape) * 0.01
    # The actual computation (done twice - first time to compile kernel)
    fhkl_, dt = scattering.Fhkl_thread(h, k, l, x, y, z, gpu_name=gpu_name, sz_imag=sz_imag,
                                       verbose=verbose, language=language, cl_platform=cl_platform)
    fhkl_, dt = scattering.Fhkl_thread(h, k, l, x, y, z, gpu_name=gpu_name, sz_imag=sz_imag,
                                       verbose=verbose, language=language, cl_platform=cl_platform)
    # Compare to analytical formula
    fhkl_gold = (exp(-1j * pi * h * (nx - 1)) * sin(pi * h * nx) / sin(pi * h)) * \
                (exp(-1j * pi * k * (ny - 1)) * sin(pi * k * ny) / sin(pi * k)) * \
                (exp(2 * pi * (1j * l + sz_imag) * nz) - 1) / (exp(2 * pi * (1j * l + sz_imag)) - 1)
    diff = abs(fhkl_ - fhkl_gold)
    tmp = diff.mean() / abs(fhkl_gold).mean()
    s = "PASS"
    if tmp > 0.01 or isnan(tmp): s = "FAIL"
    print("%20s: %5d 10^3 reflections, %5d 10^3 atoms, speed=%7.3f 10^9 reflections.atoms/s  "
          "=>   <|GPU-analytical|> / <|analytical|>=%7.5f, %10s" %
          ("fhkl_graz", nh * nk * nl // 1000, nx * ny * nz // 1000,
           nx * ny * nz * nh * nk * nl / dt / 1e9, tmp, s))
    if return_array:
        return diff
    return tmp < 0.01


def dwba4(gpu_name, show_plot=True, verbose=True, language="OpenCL", cl_platform="",
          return_array=False):
    from numpy import mgrid, flatnonzero, take, append, sqrt, linspace, float32, abs
    from pynx.scattering import fthomson, gid

    # Energy (eV), grazing-incidence angle
    nrj = 10000.
    wavelength = 12398.4 / nrj
    alphai = 0.15 * pi / 180
    a = 5.4309

    # Quantum dot as a truncated sphere
    tmp = mgrid[-50:50, -50:50, 0:20]
    x0, y0, z0 = tmp[0], tmp[1], tmp[2]
    idx = flatnonzero(sqrt(x0 ** 2 + y0 ** 2 + (z0 + 20) ** 2) < 50)
    x0 = take(x0.ravel(), idx)
    y0 = take(y0.ravel(), idx)
    z0 = take(z0.ravel(), idx)

    # Add all diamond sites
    x = append(x0, x0 + .5)
    y = append(y0, y0 + .5)
    z = append(z0, z0)

    x = append(x, x0 + .5)
    y = append(y, y0)
    z = append(z, z0 + .5)

    x = append(x, x0)
    y = append(y, y0 + .5)
    z = append(z, z0 + .5)

    x = append(x, x + .25)
    y = append(y, y + .25)
    z = append(z, z + .25)

    # Ge occupancy in Ge_xSi_(1-x) dot
    occ = 0.2 + 0.6 * z / z.max()

    # Ux and Uy displacements (relative to Si)
    ux = x * (0.005 + z * 0.001 * (1 + sqrt(x ** 2 + y ** 2) / 50))
    uy = y * (0.005 + z * 0.001 * (1 + sqrt(x ** 2 + y ** 2) / 50))
    if verbose: print("Simulation with epsilon_xx=epsilon_yy going from %4.2f%% to %4.2f%%" % (
        0.5, (0.005 + z * 0.001 * (1 + sqrt(x ** 2 + y ** 2) / 50)).max() * 100))

    # Compute refraction index for substrate
    si = gid.Scatterer("Si", (0., 0., 0.), 1.0, 1.0 / (8 * pi ** 2), nrj)
    substrate = gid.Crystal((a, a, a, 90, 90, 90), "Fd3m:1", (si,))
    if verbose:
        print("Si refraction index:  delta=%6e  beta=%6e" % substrate.GetRefractionIndexDeltaBeta())
        print("Si critical angle=%5.3f° @ %6.0f eV" % (substrate.GetCriticalAngle() * 180 / pi, si.nrj))
        print("Si atomic density (atoms/m^3):", substrate.GetAtomDensity())
        print("Si density: %7.1f kg/m^3:" % (substrate.GetDensity()))
        print("Si linear absorption coefficient=%5.3g m^-1" % (substrate.GetLinearAbsorptionCoeff()))

    # Reciprocal space coordinates
    h = linspace(3.85, 4.02, 150)
    k = float32(0)
    alphaf = linspace(0, 0.35 * pi / 180, 200)[:, newaxis]
    l = a * sin(alphaf + alphai) / wavelength

    fhklge = gid.fhkl_dwba4(x + ux, y + uy, z, h, k, l=None, occ=occ, alphai=alphai, alphaf=alphaf,
                            substrate=substrate, wavelength=wavelength,
                            e_par=0., e_perp=1.0, gpu_name="GTX", language=language, cl_platform=cl_platform)

    fhklsi = gid.fhkl_dwba4(x + ux, y + uy, z, h, k, l=None, occ=1 - occ, alphai=alphai, alphaf=alphaf,
                            substrate=substrate, wavelength=wavelength,
                            e_par=1., e_perp=0., gpu_name="GTX", language=language, cl_platform=cl_platform)

    # Scattering factors
    s = a / sqrt(h ** 2 + k ** 2 + l ** 2)
    fGe = fthomson.f_thomson(s, "Ge")
    fSi = fthomson.f_thomson(s, "Si")

    # Sum scattering
    fhkl_ = fhklge * fGe + fhklsi * fSi
    if show_plot:
        # plot versus H and alpha_f
        from pylab import imshow, cm, xlabel, ylabel, colorbar
        imshow(abs(fhkl_) ** 2, aspect='auto', origin='lower', extent=(h.min(), h.max(), 0, alphaf.max() * 180 / pi),
               cmap=cm.jet)
        xlabel("$H\ (r.l.u.)$", fontsize=18)
        ylabel(r"$\alpha_f\ (^\circ)$", fontsize=18)
        colorbar()

    s = "PASS"
    tmp = abs(fhkl_).mean()
    if tmp < 1 or isnan(tmp): s = "FAIL"
    print("%20s: simple DWBA test, 4 paths, using cctbx for refraction index calculations "
          "(no strict check)                                         %10s" %
          ("pynx.gid.FhklDWBA4", s))
    if return_array:
        return fhkl
    return abs(fhkl_).mean() > 1


def dwba5(gpu_name, show_plot=True, verbose=True, language="OpenCL", cl_platform="", return_array=False):
    from numpy import mgrid, flatnonzero, take, append, linspace, float32, abs, log10
    try:
        from pynx.scattering import fthomson, gid
    except:
        s = "FAIL"
        print("%20s: simple DWBA test, 5 paths, using cctbx                                      "
              "                            (MISSING cctbx ?) ===>     %10s" %
              ("pynx.gid.FhklDWBA5", s))
        return False
    # Simple scattering, small cube above a substrate (larger cube).
    nrj = 10000.
    wavelength = 12398.4 / nrj
    alphai = 0.15 * pi / 180
    a = 5.4309

    tmp = mgrid[-30:30, -30:30, 0:20] * 1.02
    x, y, z = tmp[0], tmp[1], tmp[2]
    tmp = mgrid[-30:30, -30:30, -100:0]
    x = append(x, tmp[0].ravel())
    y = append(y, tmp[1].ravel())
    z = append(z, tmp[2].ravel())
    h = linspace(3.85, 4.02, 85)
    k = float32(0)
    alphaf = linspace(0, 0.5 * pi / 180, 100)[:, newaxis]
    l = a * (sin(alphaf) + sin(alphai)) / wavelength

    si = gid.Scatterer("Si", (0., 0., 0.), 1.0, 1.0 / (8 * pi ** 2), nrj)
    substrate = gid.Crystal((a, a, a, 90, 90, 90), "Fd3m:1", (si,))
    fhkl_ = gid.fhkl_dwba5(x, y, z, h, k, l=None, occ=None, alphai=alphai, alphaf=alphaf,
                           substrate=substrate, wavelength=wavelength,
                           e_par=0., e_perp=1.0, gpu_name="GTX", verbose=verbose, language=language,
                           cl_platform=cl_platform)
    if show_plot:
        # plot versus H and alpha_f
        from pylab import imshow, cm, xlabel, ylabel, colorbar, figure
        figure(1)
        imshow(log10(abs(fhkl_) ** 2), aspect='auto', origin='lower',
               extent=(h.min(), h.max(), 0, alphaf.max() * 180 / pi), cmap=cm.jet, vmin=7)
        xlabel(r"$H\ (r.l.u.)$", fontsize=18)
        ylabel(r"$\alpha_f\ (^\circ)$", fontsize=18)
        colorbar()
        figure(2)
        imshow(log10(abs(fhkl_) ** 2), aspect='auto', origin='lower', extent=(h.min(), h.max(), l.min(), l.max()),
               cmap=cm.jet, vmin=7)
        xlabel(r"$H\ (r.l.u.)$", fontsize=18)
        ylabel(r"$L\ (r.l.u.)$", fontsize=18)
        colorbar()
    s = "PASS"
    tmp = abs(fhkl_).mean()
    if tmp < 1 or isnan(tmp): s = "FAIL"
    print("%20s: simple DWBA test, 5 paths, using cctbx for refraction index calculations "
          "(no strict check)                                         %10s" %
          ("pynx.gid.FhklDWBA5", s))
    if return_array:
        return fhkl
    return abs(fhkl_).mean() > 1


def mrats(nhkl, natoms, gpu_name="GTX", verbose=False, language="OpenCL", cl_platform="", timing=False):
    """
    Benchmark the scattering calculation.

    :param nhkl: number of points in reciprocal space
    :param natoms: number of atoms
    :param gpu_name: name of gpu
    :param verbose: if True, print the result of the speed test
    :param language: the language to use, 'cuda' or 'opencl' or 'cpu'
    :param cl_platform: the OpenCL platform (optional)
    :param timing: if True, also return the time spent for the calculation
    :return: the number of million of atoms multiplied by the number of reflections, divided by the
             computing time, divided by 1e6. MRAtS = Million Reflection Atoms / Second
             if timing is True, return a tuple (MRAtS, dt)
    """
    h = random.uniform(-8, 8, nhkl)
    k = random.uniform(-8, 8, nhkl)
    l = random.uniform(-8, 8, nhkl)

    x = random.uniform(-8, 8, natoms)
    y = random.uniform(-8, 8, natoms)
    z = random.uniform(-8, 8, natoms)

    if True:
        # Add a little disorder
        x = x + 0 * (y + z)
        y = y + 0 * (x + z)
        z = z + 0 * (x + y)
        x += random.normal(0, .02, size=x.shape)
        y += random.normal(0, .02, size=x.shape)
        z += random.normal(0, .02, size=x.shape)

    fhkl_, dt = scattering.Fhkl_thread(h, k, l, x, y, z, verbose=False, gpu_name=gpu_name,
                                       nbCPUthread=None, language=language, cl_platform=cl_platform)
    MRAtS = nhkl * float(natoms) / dt / 1e6
    fhkl_gold, dt_gold = None, None
    if verbose:
        print("%7d reflections, %8d atoms, dt=%7.5fs , %10.3f MAtoms.reflections/s (%s)" %
              (nhkl, natoms, dt, MRAtS, gpu_name))

    if timing:
        return MRAtS, dt
    return MRAtS


def speed(gpu_name, do_plot=False, language="OpenCL", cl_platform="", plot_style='k-o'):
    """Test F(hkl) speed as a function of the number of atoms and reflections.
    """
    nhkl = 100
    vnhkl = []
    dt0 = 0
    d = {}
    while dt0 < 2 and nhkl < 1e7:
        vnhkl.append(nhkl)
        vnat = []
        vMRAtS = []
        iatoms = 2
        dt = 0
        while dt < 2:
            natoms = int(10 ** iatoms)
            vnat.append(natoms)
            t0 = time.time()
            vMRAtS.append(
                1e6 * mrats(nhkl, natoms, gpu_name=gpu_name, verbose=True, language=language, cl_platform=cl_platform))
            dt = time.time() - t0
            if iatoms == 2: dt0 = dt
            iatoms += 0.2
        d[nhkl] = (vnat, vMRAtS)
        nhkl *= 10
    if do_plot:
        from pylab import loglog, text, xlabel, ylabel, xlim, log10, show
        for k, v in d.items():
            loglog(v[0], v[1], plot_style, markersize=3)
            text(v[0][0] / 2.5, v[1][0], "$10^{%1d}$" % (log10(k)), fontsize=18, weight='bold',
                 verticalalignment='center')

        text(70, 3e9, "$N_{refl}$", fontsize=18, weight='bold', horizontalalignment='center',
             verticalalignment='center')
        text(1e5, 2e9, "$GPU$", fontsize=18, weight='bold', horizontalalignment='center', verticalalignment='center')

        ylabel("$reflections\cdot atoms\cdot s^{-1}$", fontsize=18)
        xlabel("$nb\ atoms$", fontsize=18)
        xlim(30, 1e7)
        show()


def run_all(nx=40, ny=40, nz=40, nh=40, nk=40, nl=40, verbose=False, gpu_name=None, language=None, cl_platform=None):
    test_cpu = (language is None or language == "CPU") and (scattering.fhkl.has_cython or scattering.fhkl.has_weave)
    if gpu_name is not None:
        if gpu_name.lower() != 'cpu':
            test_cpu = False

    if test_cpu:
        print("######## PyNX: testing for device: CPU (C++/SSE)")
        junk = fhkl("CPU", nx, ny, nz, nh, nk, nl, verbose=verbose, language="CPU")
        junk = fhklo("CPU", nx, ny, nz, nh, nk, nl, verbose=verbose, language="CPU")
        junk = fhklo_graz("CPU", nx, ny, nz, nh, nk, nl, verbose=verbose, language="CPU")
        junk = fhkl_graz("CPU", nx, ny, nz, nh, nk, nl, verbose=verbose, language="CPU")
        junk = dwba5("CPU", show_plot=False, verbose=verbose, language="CPU")

    test_cuda = scattering.fhkl.drv is not None
    if language is not None:
        if language.lower() != 'cuda':
            test_cuda = False

    if test_cuda:
        scattering.fhkl.drv.init()
        for i in range(scattering.fhkl.drv.Device.count()):
            d = scattering.fhkl.drv.Device(i).name()
            if scattering is not None:
                if d.lower().find(gpu_name.lower()) < 0:
                    continue
            print("######## PyNX: testing for device (CUDA): ", d)
            junk = fhkl(d, nx, ny, nz, nh, nk, nl, verbose=verbose, language='CUDA')
            junk = fhklo(d, nx, ny, nz, nh, nk, nl, verbose=verbose, language='CUDA')
            junk = fhklo_graz(d, nx, ny, nz, nh, nk, nl, verbose=verbose, language='CUDA')
            junk = fhkl_graz(d, nx, ny, nz, nh, nk, nl, verbose=verbose, language='CUDA')
            junk = dwba5(d, show_plot=False, verbose=verbose, language='CUDA')

    test_opencl = scattering.fhkl.cl is not None
    if language is not None:
        if language.lower() != 'opencl':
            test_opencl = False

    if test_opencl:
        for p in scattering.fhkl.cl.get_platforms():
            if cl_platform is not None:
                if p.name.lower().find(cl_platform.lower()) < 0:
                    continue
            for d in p.get_devices():
                if gpu_name is not None:
                    if d.name.lower().find(gpu_name.lower()) < 0:
                        continue
                print("######## PyNX: testing for device (OpenCL: platform=%s): %s" % (p.name, d.name))
                junk = fhkl(d.name, nx, ny, nz, nh, nk, nl, verbose=verbose, language='OpenCL', cl_platform=p.name)
                junk = fhklo(d.name, nx, ny, nz, nh, nk, nl, verbose=verbose, language='OpenCL',
                             cl_platform=p.name)
                junk = fhklo_graz(d.name, nx, ny, nz, nh, nk, nl, verbose=verbose, language='OpenCL',
                                  cl_platform=p.name)
                junk = fhkl_graz(d.name, nx, ny, nz, nh, nk, nl, verbose=verbose, language='OpenCL',
                                 cl_platform=p.name)
                junk = dwba5(d.name, show_plot=False, verbose=verbose, language='OpenCL', cl_platform=p.name)


# TODO dynamically add all tests for all available computing units and platforms
class TestScattering(unittest.TestCase):
    def test_Cython(self):
        msg = "Testing for cython"
        try:
            import cython
            import_ok = True
        except ImportError:
            import_ok = False
        self.assertTrue(import_ok, msg=msg)

    @unittest.skipIf(scattering.fhkl.has_cython == False, "Cython is not available")
    def test_fhkl_cpu(self):
        n = 20

        # CPU
        if scattering.fhkl.has_cython or scattering.fhkl.has_weave:
            if scattering.fhkl.has_weave:
                s = 'scipy.weave'
            else:
                s = 'cython'
            msg = "Testing fhkl calculations: platform=CPU, language=C/SSE, method=%s" % s
            res = fhkl('CPU', n, n, n, n, n, n, verbose=False, language='CPU', return_array=True)
            self.assertLess(res.mean(), 1, msg=msg)

    @unittest.skipIf(scattering.fhkl.drv is None, "PyCUDA is not available")
    def test_fhkl_cuda(self):
        n = 20
        # CUDA
        scattering.fhkl.drv.init()
        for i in range(scattering.fhkl.drv.Device.count()):
            d = scattering.fhkl.drv.Device(i).name()
            msg = "Testing fhkl calculations: platform=CUDA, device=%s" % d
            res = fhkl(d, n, n, n, n, n, n, verbose=False, language='CUDA', return_array=True)
            self.assertLess(res.mean(), 1, msg=msg)

    @unittest.skipIf(scattering.fhkl.drv is None, "PyCUDA is not available")
    def test_fhklo_cuda(self):
        n = 20
        for p in scattering.fhkl.cl.get_platforms():
            for d in p.get_devices():
                msg = "Testing fhklo calculations: platform=OpenCL, device=%s" % d
                res = fhklo(d.name, n, n, n, n, n, n, verbose=False, language='CUDA', return_array=True)
                self.assertLess(res.mean(), 1, msg=msg)

    @unittest.skipIf(scattering.fhkl.drv is None, "PyCUDA is not available")
    def test_fhkl_graz_cuda(self):
        n = 20
        for p in scattering.fhkl.cl.get_platforms():
            for d in p.get_devices():
                msg = "Testing fhkl_graz calculations: platform=OpenCL, device=%s" % d
                res = fhkl_graz(d.name, n, n, n, n, n, n, verbose=False, language='CUDA', return_array=True)
                self.assertLess(res.mean(), 1, msg=msg)

    @unittest.skipIf(scattering.fhkl.drv is None, "PyCUDA is not available")
    def test_fhklo_graz_cuda(self):
        n = 20
        for p in scattering.fhkl.cl.get_platforms():
            for d in p.get_devices():
                msg = "Testing fhklo_graz calculations: platform=OpenCL, device=%s" % d
                res = fhklo_graz(d.name, n, n, n, n, n, n, verbose=False, language='CUDA', return_array=True)
                self.assertLess(res.mean(), 1, msg=msg)

    @unittest.skipIf(scattering.fhkl.cl is None, "PyOpenCL is not available")
    def test_fhkl_opencl(self):
        n = 20
        for p in scattering.fhkl.cl.get_platforms():
            for d in p.get_devices():
                # if d.type & cl.device_type.GPU is False:
                #    continue
                msg = "Testing fhkl calculations: platform=OpenCL, device=%s" % d
                res = fhkl(d.name, n, n, n, n, n, n, verbose=False, language='OpenCL', return_array=True)
                self.assertLess(res.mean(), 1, msg=msg)

    @unittest.skipIf(scattering.fhkl.cl is None, "PyOpenCL is not available")
    def test_fhklo_opencl(self):
        n = 20
        for p in scattering.fhkl.cl.get_platforms():
            for d in p.get_devices():
                msg = "Testing fhklo calculations: platform=OpenCL, device=%s" % d
                res = fhklo(d.name, n, n, n, n, n, n, verbose=False, language='OpenCL', return_array=True)
                self.assertLess(res.mean(), 1, msg=msg)

    @unittest.skipIf(scattering.fhkl.cl is None, "PyOpenCL is not available")
    def test_fhkl_graz_opencl(self):
        n = 20
        for p in scattering.fhkl.cl.get_platforms():
            for d in p.get_devices():
                msg = "Testing fhkl_graz calculations: platform=OpenCL, device=%s" % d
                res = fhkl_graz(d.name, n, n, n, n, n, n, verbose=False, language='OpenCL', return_array=True)
                self.assertLess(res.mean(), 1, msg=msg)

    @unittest.skipIf(scattering.fhkl.cl is None, "PyOpenCL is not available")
    def test_fhklo_graz_opencl(self):
        n = 20
        for p in scattering.fhkl.cl.get_platforms():
            for d in p.get_devices():
                msg = "Testing fhklo_graz calculations: platform=OpenCL, device=%s" % d
                res = fhklo_graz(d.name, n, n, n, n, n, n, verbose=False, language='OpenCL', return_array=True)
                self.assertLess(res.mean(), 1, msg=msg)


def suite():
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite([load_tests(TestScattering)])
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
