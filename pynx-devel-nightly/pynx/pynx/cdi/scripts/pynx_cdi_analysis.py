#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
from __future__ import division

import sys
import os
import platform
import glob
import timeit
import time
from multiprocessing import Pool
import functools
import psutil
import numpy as np
from numpy.fft import fftshift
import matplotlib.pyplot as plt
from pynx.utils import h5py as h5
from pynx.cdi.selection import match_shape, match2, array_cen
from pynx.utils.math import ortho_modes
from pynx.version import get_git_version

_pynx_version = get_git_version()
from pynx.cdi import CDI, FT
from pynx.utils.phase_retrieval_transfer_function import prtf, plot_prtf
from pynx.processing_unit import has_opencl, has_cuda
from pynx.utils.string import longest_common_prefix
from ..runner import CDIRunnerException
from ...mpi import MPI

if has_cuda:
    from pynx.cdi.cu_operator import PRTF, InitPSF
elif has_opencl:
    from pynx.cdi.cl_operator import PRTF, InitPSF

params = {'modes': False, 'modes_output': None, 'movie': False, 'movie.type': 'complex', 'movie.cmap': 'viridis',
          'subpixel': False, 'average': False, 'prtf': False, 'modes_type': 'complex', 'modes_crop': 'auto',
          'phase_ramp': False}

helptext = """
pynx-cdi-analysis: script to analyse a series of CDI reconstructions (mode decomposition, PRTF,..)

    Note that this cannot use MPI - an error will be raised if used.

Example:
    pynx-cdi-analysis *LLK*.cxi modes
    pynx-cdi-analysis *LLK*.cxi modes movie
    pynx-cdi-analysis solution1.cxi solution2.cxi movie
    pynx-cdi-analysis data.cxi solution1.cxi solution2.cxi modes prtf

command-line arguments:
    path/to/Run.cxi: path to cxi file with the reconstructions to analyse. Several can be supplied [mandatory]

    modes: if used, analyse the CDI reconstructions and decompose them into eigen-values. The first mode should
           represent most of the intensity. If modes=N (N integer) is given, only the first N modes are saved.
           The objects are first aligned (the R_match R-factor indicates how good individual object fit,
           lower values are better), and 
           
    modes_output=sample_modes.h5: supply a filename for the output of the modes analysis
        [default: the longest common prefix to the input files + "-modes.h5"]

    modes_type=abs: can be 'abs' or 'amplitude' for real-valued objects, or 'complex' [default: complex]
    
    modes_crop=no: can be 'auto' (the default, will crop around the support if available), or 'no', or an
        integer value to crop around the support plus a number of pixels.
    
    average: if this keyword is given, the average of reconstructions will also be saved in the modes.h5 file.
    
    subpixel: if this keyword is given, subpixel registration is used for object alignment
        for the modes analysis.
    
    phase_ramp: if this keyword is used, the phase ramp will be matched between solutions.
        This was previously the default, but since all solutions correspond to the same Fourier Transform,
        it should not be necessary.
    
    movie: if used with 3D input data, a movie will be made, either from the single CXI or h5 (modes) data file,
           or from the first two files (if the .h5 mode file is listed, it is always considered first).
           Some options can be given:
           - movie=complex: to display the complex 3d data (the default)
           - movie=amplitude: to display the amplitude
           - movie=amplitude,grey: to display the amplitude using a grayscale rather than the default colormap.
                                   possible options are 'grey' and 'grey_r'. Otherwise viridis is used
    prtf: if given, will compute the Phase Retrieval Transfer Function. This requires including the experimental
          data CXI file among the input file (it will be automatically recognised from 'result' CXI files) and
          using the 'modes' keyword or giving the result modes hdf5 file as input. At least one CXI result file
          must be given so that the actual ROI used is known.
          If 'modes' is also used, the PRTF is saved in the modes hdf5 file

"""


def load_cxi(s):
    print("    Loading: %s" % s)
    sup = None
    obj = None
    psf = None
    llk = None
    roi_final = None
    if s[-4:] == '.npz':
        for k, v in np.load(s).items():
            obj = v
            if v.size > 1000:
                break
    else:
        h = h5.File(s, 'r')
        if '/entry_1/instrument_1/detector_1/data' in h:
            # This is iobs data, not a result object
            iobs = h['/entry_1/instrument_1/detector_1/data'][()]
            iobs_mask = None
            if '/entry_1/instrument_1/detector_1/mask' in h:
                iobs_mask = h['/entry_1/instrument_1/detector_1/mask'][()]
            return {'file_name': s, 'iobs': iobs, 'iobs_mask': iobs_mask}
        else:
            obj = h['entry_1/image_1/data'][()]
            sup = h['entry_1/image_1/support'][()]
            if 'entry_1/image_1/process_1/configuration/roi_final' in h:
                roi_final = h['entry_1/image_1/process_1/configuration/roi_final'][()]
            if 'entry_1/image_1/process_1/results/free_llk_poisson' in h:
                llk = float(h['entry_1/image_1/process_1/results/free_llk_poisson'][()])
            else:
                llk = None
            if '/entry_1/image_1/instrument_1/detector_1/point_spread_function' in h:
                psf = h['/entry_1/image_1/instrument_1/detector_1/point_spread_function'][()]
    if 'complex' not in params['modes_type']:
        obj = np.abs(obj)

    if obj.size > 1e8 and params['modes_crop'] != "no":
        # Try cropping
        d = obj
        shape0 = d.shape
        if sup is None:
            threshold = 0.15
            ad = np.abs(d)
            sup = (ad > (ad.max() * threshold)).astype(np.int16)
            del ad
        margin = 2
        if params['modes_crop'] != "auto":
            margin = int(params['modes_crop'])
        if d.ndim == 3:
            l0 = np.nonzero(sup.sum(axis=(1, 2)))[0].take([0, -1]) + np.array([-margin, margin])
            if l0[0] < 0: l0[0] = 0
            if l0[1] >= sup.shape[0]: l0[1] = -1

            l1 = np.nonzero(sup.sum(axis=(0, 2)))[0].take([0, -1]) + np.array([-margin, margin])
            if l1[0] < 0: l1[0] = 0
            if l1[1] >= sup.shape[1]: l1[1] = -1

            l2 = np.nonzero(sup.sum(axis=(0, 1)))[0].take([0, -1]) + np.array([-margin, margin])
            if l2[0] < 0: l2[0] = 0
            if l2[1] >= sup.shape[2]: l2[1] = -1
            d = d[l0[0]:l0[1], l1[0]:l1[1], l2[0]:l2[1]]
        else:
            l0 = np.nonzero(sup.sum(axis=1))[0].take([0, -1]) + np.array([-margin, margin])
            if l0[0] < 0: l0[0] = 0
            if l0[1] >= sup.shape[0]: l0[1] = -1

            l1 = np.nonzero(sup.sum(axis=0))[0].take([0, -1]) + np.array([-margin, margin])
            if l1[0] < 0: l1[0] = 0
            if l1[1] >= sup.shape[1]: l1[1] = -1

            d = d[l0[0]:l0[1], l1[0]:l1[1]]
        shape1 = d.shape
        print("    %s: Cropping around support with a margin of %d pixels: " % (s, margin), shape0, " -> ", shape1)
        obj = d
        if params['modes_crop'] != "no":
            # Centre array
            obj = array_cen(obj, thres=0.1, decomposed=True)
    return {'file_name': s, 'obj': obj, 'psf': psf, 'roi_final': roi_final, 'llk': llk}


def match_obj(i, v, match_phase_ramp, upsample_factor):
    d = v[i]
    d1c, d2c, r = match2(v[0], d, match_phase_ramp=match_phase_ramp, match_scale=False,
                         upsample_factor=upsample_factor, verbose=False, match_orientation='center',
                         use_gpu=True)
    print("    R_match(%s) = %6.3f%%" % (i, r * 100))
    return d2c


def main():
    # Make sure MPI is *not* used, otherwise bail out.
    # Note that this only works if at least 2 MPI tasks are used,
    # if 'mpiexec -n 1 ...' is used, the program will fail when using multiprocessing
    if MPI is not None:
        mpic = MPI.COMM_WORLD
        mpi_size = mpic.Get_size()
        if mpi_size > 1:
            raise CDIRunnerException("pynx-cdi-analysis cannot be used with MPI: mpi_size=%d>1 !" % mpi_size)

    try:
        # Get the real number of processor cores available
        # os.sched_getaffinity is only available on some *nix platforms
        nproc = len(os.sched_getaffinity(0)) * psutil.cpu_count(logical=False) // psutil.cpu_count(logical=True)
    except AttributeError:
        nproc = os.cpu_count()

    t_start = time.time()
    # CXI results of reconstruction
    cxi_files = []
    # hdf5 file with mode analysis
    h5_input = None
    # Observed intensity for PRTF
    cxi_iobs, iobs, iobs_mask = None, None, None
    prtf_ring_thick = 2

    if platform.system() == "Windows":
        # Need to expand wildcards manually...
        sys_argv = []
        for arg in sys.argv:
            if '*' in arg:
                sys_argv += glob.glob(arg)
            else:
                sys_argv.append(arg)
        print(sys_argv)
    else:
        sys_argv = sys.argv

    for arg in sys_argv:
        if arg == 'help':
            print(helptext)
        elif arg in ['subpixel', 'average', 'phase_ramp']:
            params[arg] = True
        elif 'prtf' in arg:
            params['prtf'] = True
        elif 'modes_output' in arg:
            params[arg] = True
            if 'modes.h5' in arg:
                params['modes_output'] = 'modes.h5'
            else:
                params['modes_output'] = arg.split("=")[1]
        elif 'modes_type' in arg:
            params['modes_type'] = arg.split("=")[1]
        elif 'modes_crop' in arg:
            params['modes_crop'] = arg.split("=")[1]
        elif 'modes' in arg and arg != 'modes.h5':
            params['modes'] = True
            if '=' in arg:
                # Number of modes to output
                try:
                    params['modes'] = int(arg.split('=')[-1])
                except Exception:
                    print("Did not understand: '%s' - all modes will be extracted" % arg)
        elif 'movie' in arg:
            params['movie'] = True
            if 'complex' in arg:
                params['movie.type'] = 'complex'
            elif 'amplitude' in arg:
                params['movie.type'] = 'amplitude'

            if 'gray_r' in arg or 'grey_r' in arg:
                params['movie.cmap'] = 'gray_r'
            elif 'gray' in arg or 'grey' in arg:
                params['movie.cmap'] = 'gray'
        else:
            if len(arg) > 4:
                if arg[-4:] in ['.cxi', '.npz']:
                    cxi_files.append(arg)
            if len(arg) > 3:
                if arg[-3:] == '.h5':
                    h5_input = arg

    if params['movie'] is False and params['modes'] is False and params['prtf'] is False:
        print(helptext)
        print("\nNo 'movie', 'prtf' or 'modes' keyword given, nothing to do !")
        sys.exit(0)

    print("Importing data files")

    print("Loading %d files in // [%d proc]" % (len(cxi_files), nproc))
    p = Pool(nproc)
    res = p.map(load_cxi, cxi_files)
    del p

    # Extract iobs data if present
    iobs = None
    iobs_mask = None
    for v in res:
        if 'iobs' in v:
            iobs = v['iobs']
            iobs_mask = v['iobs_mask']
            res.remove(v)
            break

    # Sort objects by llk, if not None
    if res[0]['llk'] is not None:  # Assume none or all have llk
        res.sort(key=lambda x: x['llk'])

    # Common prefix
    prefix = longest_common_prefix([v['file_name'] for v in res])
    if params['modes']:
        print('Calculating modes from the imported objects')
        if params['modes_output'] is None:
            # Find common prefix for files
            params['modes_output'] = prefix + "-modes.h5"
            print("Will save the modes to: ", params['modes_output'])
        print('Matching arrays against the first one [%s] - this may take a while' % res[0]['file_name'])
        v = [vv['obj'] for vv in res]
        v0 = v
        if params['modes_crop'] == "no":
            v = match_shape(v, method='max', cen=False)
        else:
            v = match_shape(v, method='median')
        if params['subpixel']:
            upsample_factor = 20
        else:
            upsample_factor = 1
        v2 = [v[0]]
        t0 = timeit.default_timer()

        if 'complex' in params['modes_type']:
            match_phase_ramp = params['phase_ramp']
        else:
            match_phase_ramp = False

        if match_phase_ramp:
            print("Matching arrays and phase ramp")
            for i in range(1, len(v)):
                v2.append(match_obj(i, v, match_phase_ramp, upsample_factor))
        else:
            print("Matching arrays in //")
            p = Pool(nproc)  # nproc if nproc < 20 else 20
            v2 += p.map(functools.partial(match_obj, v=v, match_phase_ramp=match_phase_ramp,
                                          upsample_factor=upsample_factor), range(1, len(v)))
            del p

        dt = timeit.default_timer() - t0
        print('Elapsed time: %6.1fs' % dt)
        print("Analysing modes")
        nb_mode = params['modes']
        if nb_mode is True or isinstance(nb_mode, int) is False:
            nb_mode = None
        vo, modes_weights = ortho_modes(v2, nb_mode=nb_mode, return_weights=True)
        print("First mode represents %6.3f%%" % (modes_weights[0] * 100))

        # Also compute average
        vave = v2[0].copy()
        for i in range(1, len(v)):
            vave += v2[i]
        vave /= len(v)

        # Save result to hdf5
        h5_out_filename = params['modes_output']
        h5_input = h5_out_filename
        print('Saving modes analysis to: %s' % h5_out_filename)
        f = h5.File(h5_out_filename, "w")

        # NeXus
        f.attrs['default'] = 'entry_1'
        f.attrs['creator'] = 'PyNX'
        f.attrs['HDF5_Version'] = h5.version.hdf5_version
        f.attrs['h5py_version'] = h5.version.version

        entry_1 = f.create_group("entry_1")
        entry_1.create_dataset("program_name", data="PyNX %s" % _pynx_version)
        entry_1.create_dataset("start_time", data=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_start)))
        entry_1.attrs['NX_class'] = 'NXentry'
        entry_1.attrs['default'] = 'data_1'

        image_1 = entry_1.create_group("image_1")
        image_1.create_dataset("data", data=vo, chunks=True, shuffle=True, compression="gzip")
        image_1.attrs['NX_class'] = 'NXdata'  # Is image_1 a well-formed NXdata or not ?
        image_1.attrs['signal'] = 'data'
        image_1.attrs['interpretation'] = 'image'
        image_1.attrs['label'] = 'modes'

        command = ""
        for arg in sys.argv:
            command += arg + " "
        data_1 = f['/entry_1/image_1']
        process_1 = data_1.create_group("process_1")
        process_1.attrs['NX_class'] = 'NXprocess'
        process_1.attrs['label'] = 'Process information about the modes analysis'
        process_1.create_dataset("program", data='PyNX')
        process_1.create_dataset("version", data="%s" % _pynx_version)
        process_1.create_dataset("command", data=command)
        process_1.create_dataset("file_names", data=[r['file_name'] for r in res])
        if res[0]['file_name'][-4:] == '.cxi':
            # Try to copy the original process information, including
            # all initial solving parameters
            with h5.File(res[0]['file_name'], 'r') as h0:
                if '/entry_last/image_1/process_1' in h0:
                    h0.copy('/entry_last/image_1/process_1', data_1, name="process_2")
                    data_1['process_2'].attrs['label'] = \
                        'Process information for the Original CDI reconstruction (best solution)'

        # Add shortcut to the main data
        data_1 = entry_1.create_group("data_1")
        data_1["data"] = h5.SoftLink('/entry_1/image_1/data')
        data_1.attrs['NX_class'] = 'NXdata'  # Is image_1 a well-formed NXdata or not ?
        data_1.attrs['signal'] = 'data'
        data_1.attrs['interpretation'] = 'image'
        data_1.attrs['label'] = 'modes'

        if params['average']:
            image_2 = entry_1.create_group("image_2")
            image_2.create_dataset("data", data=vave, chunks=True, shuffle=True, compression="gzip")
            image_2.attrs['NX_class'] = 'NXdata'  # Is image_1 a well-formed NXdata or not ?
            image_2.attrs['signal'] = 'data'
            image_2.attrs['label'] = 'average of solutions'

        # Add weights
        data_2 = entry_1.create_group("data_2")
        ds = data_2.create_dataset("data", data=modes_weights)
        ds.attrs['long_name'] = "Relative weights of modes"
        data_2.attrs['NX_class'] = 'NXdata'  # Is image_1 a well-formed NXdata or not ?
        data_2.attrs['signal'] = 'data'
        data_2.attrs['interpretation'] = 'spectrum'
        data_2.attrs['label'] = 'modes relative weights'

        f.close()

    if params['movie']:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as manimation
        from pynx.utils.plot_utils import complex2rgbalin, insertColorwheel

        # Make a movie going through 3d slices, comparing two objects if at least 2 are listed
        print("Make movie from file(s):")
        if h5_input is not None:
            # Use the first mode
            o1 = h5.File(h5_input, mode='r')['entry_1/image_1/data'][0]
            o1n = h5_input
            print("        %s" % h5_input)
            if len(res):
                o2 = res[0]['obj']
                o2n = res[0]['file_name']
                print("        %s" % o2n)
            else:
                o2 = None
        else:
            o1 = res[0]['obj']
            o1n = res[0]['file_name']
            print("        %s" % o1n)
            if len(res) > 1:
                o2 = res[1]['obj']
                o2n = res[1]['file_name']
                print("        %s" % o2n)
            else:
                o2 = None
        print("Movie type: %s" % params['movie.type'])
        if o1.ndim != 3:
            print('Movie generation from CXI data only supported for 3D objects')
            exit()

        try:
            FFMpegWriter = manimation.writers['ffmpeg']
        except:
            print("Could not import FFMpeg writer for movie generation")
            exit()

        metadata = dict(title='3D CDI slices', artist='PyNX')
        writer = FFMpegWriter(fps=5, metadata=metadata)
        fontsize = 10
        if o2 is None:
            fig = plt.figure(figsize=(6, 5))
            o1m = np.abs(o1).max()
            with writer.saving(fig, "%s.mp4" % prefix, dpi=100):
                for i in range(len(o1)):
                    if (i % 10) == 0:
                        print(i)
                    plt.clf()
                    plt.title("%s - #%3d" % (o1n, i), fontsize=fontsize)
                    if params['movie.type'] == 'amplitude':
                        plt.imshow(abs(o1[i]), vmin=0, vmax=o1m, cmap=params['movie.cmap'])
                    else:
                        plt.imshow(complex2rgbalin(o1[i], smin=0, alpha=(0, np.abs(o1[i]).max() / o1m)))
                        insertColorwheel(left=0.85, bottom=.0, width=.1, height=.1, text_col='black', fs=10)
                    writer.grab_frame()
        else:
            print('Matching shape and orientation of objects for 3D CDI movie')
            print('(Lower R-values indicate a better match)')
            o1, o2 = match_shape([o1, o2], method='median')
            o1, o2, r = match2(o1, o2, match_phase_ramp=False, verbose=False)
            print("R_match = %6.3f%% " % (r * 100))

            fig = plt.figure(figsize=(12, 5))

            o1m = np.abs(o1).max()
            o2m = np.abs(o2).max()

            with writer.saving(fig, "%s.mp4" % prefix, dpi=100):
                for i in range(len(o1)):
                    if (i % 10) == 0:
                        print(i)
                    plt.clf()
                    plt.subplot(121)
                    plt.title("%s" % o1n, fontsize=fontsize)
                    if params['movie.type'] == 'amplitude':
                        plt.imshow(abs(o1[i]), vmin=0, vmax=o1m, cmap=params['movie.cmap'])
                    else:
                        plt.imshow(complex2rgbalin(o1[i], smin=0, alpha=(0, np.abs(o1[i]).max() / o1m)))

                    plt.subplot(122)
                    plt.title("%s" % o2n, fontsize=fontsize)
                    plt.suptitle("%3d" % i)
                    if params['movie.type'] == 'amplitude':
                        plt.imshow(abs(o2[i]), vmin=0, vmax=o2m, cmap=params['movie.cmap'])
                    else:
                        plt.imshow(complex2rgbalin(o2[i], smin=0, alpha=(0, np.abs(o2[i]).max() / o2m)))
                        insertColorwheel(left=0.90, bottom=.0, width=.1, height=.1, text_col='black', fs=10)
                    writer.grab_frame()

    if params['prtf']:
        if iobs is None:
            print("PRTF asked, but no CXI data (iobs) given on input")
            sys.exit(0)
        if len(cxi_files) == 0 and h5_input is None:
            print("PRTF asked, but no CXI result or modes.h5 file given")
            sys.exit(0)
        if h5_input is not None:
            print("PRTF: using first mode")
            # Use the first mode
            o1 = h5.File(h5_input, mode='r')['entry_1/image_1/data'][0]
            o1n = h5_input
        else:
            print("PRTF: using best object")
            o1 = res[0]['obj']
            o1n = res[0]['file_name']
        roi_final = res[0]['roi_final']
        if roi_final is not None:
            print('PRTF: cropping iobs to extent used for reconstruction:', roi_final)
            if len(roi_final) == 4:
                iy0, iy1, ix0, ix1 = roi_final
                iobs = iobs[iy0:iy1, ix0:ix1]
                if iobs_mask is not None:
                    iobs_mask = iobs_mask[iy0:iy1, ix0:ix1]
            elif len(roi_final) == 6:
                iz0, iz1, iy0, iy1, ix0, ix1 = roi_final
                iobs = iobs[iz0:iz1, iy0:iy1, ix0:ix1]
                if iobs_mask is not None:
                    iobs_mask = iobs_mask[iz0:iz1, iy0:iy1, ix0:ix1]

        obj = o1
        if o1.shape != iobs.shape:
            obj = np.zeros(iobs.shape, dtype=np.complex64)
            # Assume object shape is smaller than iobs
            if iobs.ndim == 2:
                ny, nx = iobs.shape
                nyo, nxo = o1.shape
                obj[ny // 2 - nyo // 2:ny // 2 - nyo // 2 + nyo, nx // 2 - nxo // 2:nx // 2 - nxo // 2 + nxo] = o1
            else:
                nz, ny, nx = iobs.shape
                nzo, nyo, nxo = o1.shape
                obj[nz // 2 - nzo // 2:nz // 2 - nzo // 2 + nzo, ny // 2 - nyo // 2:ny // 2 - nyo // 2 + nyo,
                nx // 2 - nxo // 2:nx // 2 - nxo // 2 + nxo] = o1
        title = "%s - %s" % (cxi_iobs, o1n)
        if iobs_mask is not None:
            iobs_mask = fftshift(iobs_mask)

        if has_cuda or has_opencl:
            cdi = CDI(fftshift(iobs), obj=None, support=None, mask=iobs_mask, wavelength=None,
                      pixel_size_detector=None)
            cdi.set_obj(obj)
            psf = res[0]['psf']
            if psf is not None:
                print("PRTF: will use psf from best model")
                cdi = InitPSF(psf=psf) * cdi
            cdi = PRTF(file_name='%s-PRTF.png' % prefix, fig_title=title) * cdi
            freq = cdi.prtf_freq
            fnyquist = cdi.prtf_fnyquist
            pr = cdi.prtf
        else:
            print("PRTF: CPU calc...")
            cdi = CDI(fftshift(iobs), obj=None, support=None, mask=fftshift(iobs_mask), wavelength=None,
                      pixel_size_detector=None)
            cdi.set_obj(obj)
            cdi = FT() * cdi
            icalc = np.abs(cdi.get_obj(shift=True)) ** 2
            freq, fnyquist, pr, prtf_iobs = prtf(icalc, iobs, mask=iobs_mask, ring_thick=prtf_ring_thick)

            plot_prtf(freq, fnyquist, pr, iobs_shell=prtf_iobs, file_name='%s-PRTF.png' % prefix, title=title)

        if params['modes']:
            print('Saving PRTF also to: %s' % params['modes_output'])
            f = h5.File(params['modes_output'], "r+")
            entry_1 = f['entry_1']
            data_3 = entry_1.create_group("data_3")
            ds = data_3.create_dataset("PRTF", data=pr)
            ds.attrs['long_name'] = "Phase Retrieval Transfer Function"
            ds = data_3.create_dataset("freq", data=freq / fnyquist)
            ds.attrs['long_name'] = "f / f_Nyquist"
            data_3.attrs['NX_class'] = 'NXdata'
            data_3.attrs['signal'] = 'PRTF'
            data_3.attrs['axes'] = 'freq'
            data_3.attrs['interpretation'] = 'spectrum'
            data_3.attrs['label'] = 'Phase Retrieval Transfer Function'

        print('Saved phase retrieval transfer function plot to: %s-PRTF.png' % prefix)


if __name__ == '__main__':
    main()
