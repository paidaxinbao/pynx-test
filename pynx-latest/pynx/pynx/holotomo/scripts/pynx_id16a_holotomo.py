#!/home/esrf/favre/dev/pynx-py38-power9-env/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2020-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import os
import sys
import timeit
import multiprocessing

os.environ['PYNX_PU'] = 'cuda.0'
import numpy as np
import psutil

try:
    # Get the real number of processor cores available
    # os.sched_getaffinity is only available on some *nix platforms
    nproc = len(os.sched_getaffinity(0)) * psutil.cpu_count(logical=False) // psutil.cpu_count(logical=True)
except AttributeError:
    nproc = os.cpu_count()
print("Number of available processors: ", nproc)
from multiprocessing import Pool
import matplotlib.pyplot as plt

import fabio
from pynx.holotomo import *
from pynx.holotomo.utils import load_data_kw, zoom_pad_images_kw, align_images_kw, save_phase_edf, save_phase_edf_kw
from pynx.utils.math import primes, test_smaller_primes


def main():
    t00 = timeit.default_timer()
    ################################################################
    # Experiment parameters - should be loaded from a file
    ################################################################
    # All these will be loaded from the parameters file
    params = {'i0': None,
              'data_dir': None,
              'prefix': None,
              'dark_name': None,
              'ref_name': None,
              'img_name': None,
              'prefix_result': None,
              'nz': None,
              'reference_plane': None,
              'padding': None,
              'projection_range': None,
              'delta_beta': None,
              'wavelength': None,
              'detector_distance': None,
              'magnification': None,
              'algorithm': None,
              'stack_size': None,
              'save_phase_chunks': True,
              'save_edf': False,
              'align_rhapp': None,
              'obj_smooth': 0,
              'obj_min': None,
              'obj_max': None,
              # Beta coefficient for the RAAR or DRAP algorithm. Not to be mistaken
              # with the refraction index beta
              'beta': 0.9,
              # Number of coherent probe modes. If >1, the (constant) probe mode coefficients
              # will linearly vary as a function of the projection index
              'nb_probe': 1,
              # Effective Pixel size in meters (it will be multiplied by rebin_n)
              'pixel_size': None}

    ################################################################
    # Read parameters - should be loaded from a file
    ################################################################
    # Look for the first "*.par" file given as argument
    par_nok = True
    for v in sys.argv:
        if len(v) < 4:
            continue
        if v[-4:] == '.par':
            print("Loading parameters from: %s" % v)
            dic = {}
            exec(open(v).read(), globals(), dic)
            # There must be a better way to do this ??
            for k, v in dic.items():
                if k in params.keys():
                    params[k] = v
            par_nok = False
            break
    if par_nok:
        print("You must supply a .par file")
        sys.exit(1)

    # load parameters from command-line args, at least i0=N
    for arg in sys.argv:
        if '=' in arg:
            k, v = arg.split('=')
            if k == 'i0':
                params['i0'] = eval(v)
                print(arg)
            elif k == 'projection_range':  # [istart, iend[ , step
                params['projection_range'] = eval(v)
                print(arg)

    print("Parameters:")
    for k, v in params.items():
        print("     %s: " % k, v)

    i0 = params['i0']
    data_dir = params['data_dir']
    prefix = params['prefix']
    dark_name = params['dark_name']
    ref_name = params['ref_name']
    img_name = params['img_name']
    prefix_result = params['prefix_result']
    nz = params['nz']
    reference_plane = params['reference_plane']
    padding = params['padding']
    projection_range = params['projection_range']
    delta_beta = params['delta_beta']
    wavelength = params['wavelength']
    detector_distance = params['detector_distance']
    magnification = params['magnification']
    algorithm = params['algorithm']
    stack_size = params['stack_size']
    save_phase_chunks = params['save_phase_chunks']
    save_edf = params['save_edf']
    align_rhapp = params['align_rhapp']
    obj_smooth = params['obj_smooth']
    obj_min = params['obj_min']
    obj_max = params['obj_max']
    beta = params['beta']
    nb_probe = params['nb_probe']
    pixel_size = params['pixel_size']

    if i0 is None:
        print("You must set i0=N from the command line")
        sys.exit(1)
    else:
        print("i0=%d" % i0)

    if projection_range is None:
        print("You must set projection_range=i0,i1,step from the command line")
        sys.exit(1)
    else:
        print("projection_range: ", projection_range)

    proj_idx = np.arange(i0 + projection_range[0], projection_range[1], projection_range[2])
    proj_idx = np.append(proj_idx, -1)  # Add one image for the empty beam
    nb_proj = len(proj_idx)  # number of images loaded, including the empty one
    print("nb_proj=%d" % nb_proj)

    magnification = np.array(magnification)
    detector_distance = np.array(detector_distance)

    ################################################################
    # Prepare for output
    ################################################################

    path = os.path.split(prefix_result)[0]
    if len(path):
        os.makedirs(path, exist_ok=True)

    ################################################################
    # Load data in //
    ################################################################
    t0 = timeit.default_timer()

    dark = None
    for iz in range(0, nz):
        d = fabio.open(dark_name % (iz + 1)).data
        ny, nx = d.shape
        if dark is None:
            dark = np.empty((nz, ny, nx), dtype=np.float32)
        dark[iz] = d

    ny, nx = dark.shape[-2:]
    print("Frame size: %d x %d" % (ny, nx))

    # Test if we have a radix transform
    primesy, primesx = primes(ny + 2 * padding), primes(nx + 2 * padding)
    if max(primesx + primesy) > 13:
        padup = padding
        while not test_smaller_primes(ny + 2 * padup, required_dividers=[2]) or \
                not test_smaller_primes(nx + 2 * padup, required_dividers=[2]):
            padup += 1
        paddown = padding
        while not test_smaller_primes(ny + 2 * paddown, required_dividers=[2]) or \
                not test_smaller_primes(nx + 2 * paddown, required_dividers=[2]):
            paddown -= 1
        raise RuntimeError("The dimensions (with padding=%d) are incompatible with a radix FFT:\n"
                           "  ny=%d primes=%s  nx=%d primes=%s (should be <=13)\n"
                           "  Closest acceptable padding values: %d or %d" %
                           (padding, ny + 2 * padding, str(primesy),
                            nx + 2 * padding, str(primesx), paddown, padup))

    ref = np.empty_like(dark)
    for iz in range(0, nz):
        d = fabio.open(ref_name % (iz + 1)).data
        ref[iz] = d - dark[iz]

    vkw = [{'i': i, 'dark': dark, 'nz': nz, 'img_name': img_name} for i in proj_idx[:-1]]
    load_data_kw(vkw[0])
    res = []
    with multiprocessing.Pool(nproc) as pool:
        results = pool.imap(load_data_kw, vkw)  # , chunksize=1
        for i in range(len(vkw)):
            r = results.next(timeout=20)
            res.append(r)
    res.append(ref)

    iobs = np.empty((nb_proj, nz, ny, nx), dtype=np.float32)
    for i in range(nb_proj):
        iobs[i] = res[i]
    del res

    sample_flag = np.ones(nb_proj, dtype=np.bool)
    sample_flag[-1] = False
    dt = timeit.default_timer() - t0
    print("Time to load & uncompress data: %4.1fs [%6.2f Mbytes/s]" %
          (dt, (iobs.nbytes + dark.nbytes + ref.nbytes) / dt / 1024 ** 2))

    ################################################################
    # Zoom & register images, keep first distance pixel size and size
    # Do this using multiple process to speedup
    ################################################################

    # TODO: zoom (linear interp) & registration could be done on the GPU once the images are loaded ?

    print("Magnification relative to iz=0: ", magnification / magnification[0])
    t0 = timeit.default_timer()

    vkw = [{'x': iobs[i], 'magnification': magnification, 'padding': padding, 'nz': nz} for i in range(nb_proj)]
    zoom_pad_images_kw(vkw[0])
    res = []
    with multiprocessing.Pool(nproc) as pool:
        results = pool.imap(zoom_pad_images_kw, vkw)
        for i in range(nb_proj):
            r = results.next(timeout=20)
            res.append(r)

    iobs_align = np.array(res, dtype=np.float32)
    ny, nx = iobs_align.shape[-2:]

    del res
    print("Zoom & pad images: dt = %6.2fs" % (timeit.default_timer() - t0))

    print("Pixel size after magnification: %6.3fnm" % (pixel_size * 1e9))

    if nz > 1:
        # Align images
        t0 = timeit.default_timer()

        if align_rhapp is None:
            print("Aligning images")

            # This can sometimes (<1 in 10) fail (hang indefinitely). Why ?
            # res = pool.map(align_images, range(nb_proj))
            if padding:
                vkw = [{'x': iobs_align[i, :, padding:-padding, padding:-padding],
                        'x0': iobs_align[-1, :, padding:-padding, padding:-padding], 'nz': nz} for i in range(nb_proj)]
            else:
                vkw = [{'x': iobs_align[i,], 'x0': iobs_align[-1], 'nz': nz} for i in range(nb_proj)]
            align_images_kw(vkw[0])
            align_ok = False
            nb_nok = 0
            while not align_ok:
                if nb_nok >= 4:
                    print("4 failures, bailing out")
                    sys.exit(1)
                try:
                    res = []
                    with multiprocessing.Pool(nproc) as pool:
                        results = pool.imap(align_images_kw, vkw, chunksize=1)
                        for i in range(nb_proj):
                            r = results.next(timeout=20)
                            res.append(r)
                    align_ok = True
                    print("align OK", len(res))
                except:
                    print("Timeout, re-trying")
                    nb_nok += 1

            # print(res)

            dx = np.zeros((nb_proj, nz))
            dy = np.zeros((nb_proj, nz))
            for i in range(len(iobs)):
                dx[i] = res[i][0]
                dy[i] = res[i][1]

            # Use polyfit to smooth shifts #####################################
            # TODO: use shift corrections common to all parallel optimisations with a prior determination ?
            dx, dy, dxraw, dyraw = dx.copy(), dy.copy(), dx, dy
            for iz in range(1, nz):
                polx = np.polynomial.polynomial.polyfit(proj_idx[:-1], dx[:-1, iz], 6)
                poly = np.polynomial.polynomial.polyfit(proj_idx[:-1], dy[:-1, iz], 6)
                dx[:, iz] = np.polynomial.polynomial.polyval(proj_idx, polx)
                dy[:, iz] = np.polynomial.polynomial.polyval(proj_idx, poly)

            if True:
                # Plot shift of images vs first distance
                plt.figure(figsize=(12.5, 4))
                for iz in range(1, nz):
                    plt.subplot(1, nz - 1, iz)
                    plt.plot(proj_idx[:-1], dxraw[:-1, iz], 'r.', label='x')
                    plt.plot(proj_idx[:-1], dyraw[:-1, iz], 'b.', label='y')
                    plt.plot(proj_idx[:-1], dx[:-1, iz], 'r', label='x')
                    plt.plot(proj_idx[:-1], dy[:-1, iz], 'b', label='y')

                    plt.title("Alignment iz=%d vs iz=0 [PyNX]" % (iz))
                    plt.legend()
                plt.tight_layout()
                plt.savefig(prefix_result + '_i0=%04d_shifts.png' % i0)
                np.savez_compressed(prefix_result + '_i0=%04d_shifts.npz' % i0, dx=dx, dy=dy, dxraw=dxraw, dyraw=dyraw)
        else:
            print("Aligning images: using shift imported from holoCT")
            # Load alignment shifts from rhapp (holoCT)
            nb = np.loadtxt(align_rhapp, skiprows=4, max_rows=1, dtype=np.int)[2]
            m = np.loadtxt(align_rhapp, skiprows=5, max_rows=nb * 8, dtype=np.float32).reshape((nb, 4, 2))
            dx = m[..., 1]
            dy = m[..., 0]
            tmp_idx = proj_idx
            tmp_idx[-1] = 0

            dx = np.take(dx.copy(), tmp_idx, axis=0)
            dy = np.take(dy.copy(), tmp_idx, axis=0)

        # Switch reference plane:
        if reference_plane:
            for iz in range(nz):
                if iz != reference_plane:
                    dx[:, iz] -= dx[:, reference_plane]
                    dy[:, iz] -= dy[:, reference_plane]
            dx[:, reference_plane] = 0
            dy[:, reference_plane] = 0

        print("Align images: dt = %6.2fs" % (timeit.default_timer() - t0))
    else:
        dx, dy = None, None

    ################################################################
    # Use coherent probe modes ?
    ################################################################
    if nb_probe > 1:
        # Use linear ramps for the probe mode coefficients
        coherent_probe_modes = np.zeros((nb_proj, nz, nb_probe))
        dn = nb_proj // (nb_probe - 1)
        for iz in range(nz):
            for i in range(nb_probe - 1):
                coherent_probe_modes[i * dn:(i + 1) * dn, iz, i] = np.linspace(1, 0, dn)
                coherent_probe_modes[i * dn:(i + 1) * dn, iz, i + 1] = np.linspace(0, 1, dn)
    else:
        coherent_probe_modes = False

    ################################################################
    # Create HoloTomoData and HoloTomo objects
    ################################################################
    data = HoloTomoData(iobs_align, pixel_size_detector=pixel_size,
                        wavelength=wavelength, detector_distance=detector_distance,
                        stack_size=stack_size, sample_flag=sample_flag,
                        dx=dx, dy=dy, idx=proj_idx, padding=padding)
    # Create PCI object
    p = HoloTomo(data=data, obj=None, probe=None, coherent_probe_modes=coherent_probe_modes)
    dt = timeit.default_timer() - t00
    print("Elapsed time since beginning:  %4.1fs" % dt)

    ################################################################
    # Algorithms
    ################################################################
    t0 = timeit.default_timer()

    p.set_probe(np.ones((nz, 1, ny, nx)))
    p = ScaleObjProbe() * p

    db = delta_beta
    update_obj = True
    update_probe = True
    for algo in algorithm.split(",")[::-1]:
        if "=" in algo:
            print("Changing parameter? %s" % algo)
            k, v = algo.split("=")
            if k == "delta_beta":
                db = eval(v)
                if db == 0:
                    db = None
                elif db == 1:
                    db = delta_beta
                else:
                    delta_beta = db
            elif k == "beta":
                beta = eval(v)
            elif k == "obj_smooth":
                obj_smooth = eval(v)
            elif k == "obj_min":
                obj_min = eval(v)
            elif k == "obj_max":
                obj_max = eval(v)
            elif k == "probe":
                update_probe = eval(v)
            elif k == "obj":
                update_obj = eval(v)
            else:
                print("WARNING: did not understand algorithm step: %s" % algo)
        elif "paganin" in algo.lower():
            print("Paganin back-projection")
            p = BackPropagatePaganin(delta_beta=delta_beta) * p
            p.set_probe(np.ones((nz, 1, ny, nx)))
            p = ScaleObjProbe() * p
        elif "ctf" in algo.lower():
            print("CTF back-projection")
            p = BackPropagateCTF(delta_beta=delta_beta) * p
            # p.set_probe(np.ones((nz, 1, ny, nx)))
            p = ScaleObjProbe() * p
        else:
            print("Algorithm step: %s" % algo)
            dm = DM(update_object=update_obj, update_probe=update_probe,
                    calc_llk=10, delta_beta=db, obj_min=obj_min, obj_max=obj_max,
                    reg_obj_smooth=obj_smooth, weight_empty=1)
            ap = AP(update_object=update_obj, update_probe=update_probe,
                    calc_llk=10, delta_beta=db, obj_min=obj_min, obj_max=obj_max,
                    reg_obj_smooth=obj_smooth, weight_empty=1)
            raar = RAAR(update_object=update_obj, update_probe=update_probe, beta=beta,
                        calc_llk=10, delta_beta=db, obj_min=obj_min, obj_max=obj_max,
                        reg_obj_smooth=obj_smooth, weight_empty=1)
            drap = DRAP(update_object=update_obj, update_probe=update_probe, beta=beta,
                        calc_llk=10, delta_beta=db, obj_min=obj_min, obj_max=obj_max,
                        reg_obj_smooth=obj_smooth, weight_empty=1)
            p = eval(algo.lower()) * p

    p = FreePU() * p
    print("Algorithms: dt = %6.2fs" % (timeit.default_timer() - t0))

    if save_phase_chunks:
        filename_prefix = prefix_result + '_i0=%04d'
        print("################################################################")
        print(" Saving phased projections to hdf5 file: " + filename_prefix)
        print("################################################################")
        t0 = timeit.default_timer()
        p.save_obj_probe_chunk(chunk_prefix=filename_prefix, save_obj_phase=True,
                               save_obj_complex=False, save_probe=True, dtype=np.float16,
                               verbose=True, crop_padding=True)

        dt = timeit.default_timer() - t0
        print("Time to save phases:  %4.1fs" % dt)

    if save_edf:
        print("################################################################")
        print(" Saving phased images to edf files: " + prefix_result + "_%04d.edf")
        print("################################################################")
        t0 = timeit.default_timer()

        # Somehow getting the unwrapped phases in // using multiprocessing does not work
        # ... the process seems to hang, even with just a fork... Side effect of pinned memory ?
        ph = p.get_obj_phase_unwrapped(crop_padding=True, dtype=np.float32, idx=proj_idx[:-1])[1]
        print("Got unwrapped phases in %4.1fs" % (timeit.default_timer() - t0))
        vkw = [{'idx': proj_idx[i], 'ph': ph[i], 'prefix_result': prefix_result} for i in range(nb_proj - 1)]
        with multiprocessing.Pool(nproc) as pool:
            pool.map(save_phase_edf_kw, vkw)

        dt = timeit.default_timer() - t0
        print("Time to unwrap & save phases:  %4.1fs" % dt)

    dt = timeit.default_timer() - t00
    print("Elapsed time since beginning:  %4.1fs" % dt)


if __name__ == '__main__':
    main()
