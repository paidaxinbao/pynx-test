#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from __future__ import division

import warnings
from sys import stdout
import os
import sys
import time
import timeit
import copy
import traceback
from PIL import Image
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
from scipy.signal import fftconvolve, medfilt2d
from scipy.io import loadmat
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pynx.utils.matplotlib import pyplot as plt

try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from ...utils import h5py

import fabio

from pynx.utils import plot_utils, phase
from pynx import wavefront
from pynx.wavefront import PropagateNearField as PropagateNearField_Wavefront
from pynx.version import get_git_version

_pynx_version = get_git_version()
from pynx.utils.math import smaller_primes
from pynx.ptycho import *
from pynx.ptycho import simulation, shape
from pynx.utils.array import rebin, center_array_2d
from pynx.ptycho import analysis
from ...mpi import MPI

if MPI is not None:
    from pynx.ptycho.mpi import PtychoSplit, PlotPositions, ShowObjProbe, AnalyseProbe

# Generic help text, to be completed with beamline/instrument-specific help text

helptext_generic = """
generic (not beamline-specific) command-line arguments: (all keywords are case-insensitive)
    scan=56: scan number (e.g. in specfile) [mandatory, unless cxifile is used as input].
             Alternatively a list or range of scans can be given:
                scan=12,23,45 or scan="range(12,25)" (note the necessary quotes when using range)

    maxframe=128: limit the total number of frames used to the first N
                  [default=None, all frames are used]

    maxsize=256: the frames are automatically cropped to the largest possible size while keeping 
                 the center of gravity of diffraction in the frame center. Use this to limit the 
                 maximum frame size, for tests or limiting memory use [default=512]
    
    detectordistance=1.3: detector distance in meters. 'distance=1.3' is also accepted.
        This parameter is mandatory unless the file format embeds that information.

    obj_max_pix=10000: the maximum allowed size in pixels for the object (by default 8000). This
                       allows to avoid mistake when scan coordinates are e.g. in meters instead
                       of microns, resulting in a very large object.
    
    obj_margin=32: margin (in pixels) around the calculated object area. This is useful
                   when refining positions, to avoid getting outside the object area.
                   [default: 32]

    moduloframe=n1,n2: instead of using all sequential frames of the scan, only take one in n1.
                       if n2<n1 is also given, then the frame numbers taken will be the n for which
                       n % n1 == n2. This can be used to perform two reconstructions with half of 
                       the frames, and then analyse the resolution using Fourier ring correlation.
                       If both moduloframe and maxframe are given, the total number of frames taken 
                       is still maxframe.
                       [default: take all frames]

    algorithm=ML**50,DM**100,probe=1: algorithm for the optimization: [default='ML**50,DM**100']
    
        The algorithm used is:
        - divided in 'steps' separated by commas, e.g: ML**50,DM**100
        - interpreted from right to left, as a mathematical operator to an object on the 
          right-hand side
        - should not contain any space, unless it is given between quotes ('')
        
        - the first type of commands can change basic parameters or perform some analysis:
          - probe=1 or 0: activate or deactivate the probe optimisation (by default only the 
            object is optimised)
          - object=1 or 0: activate or deactivate the object optimisation
          - background=1 or 0: activate or deactivate the background optimisation.
            When set to 1, this will initialise the background to at least 1e-2 to
            enable the background optimisation.
          - background_smooth=3: gaussian sigma for smoothing the updated background
            [default:3, large values are possible and will use FFT convolution]
          - position=N or 0: activate or deactivate (0) position optimisation every N cycles 
                             (preferably for AP or ML, some prior convergence is needed. 
                             Can also work with DM but is not recommended).
                             Recommended value is every 5 cycles
          - pos_mult=1: multiplier for the calculated position shift. Can be used to accelerate
                        convergence or make it more cautious. (default:5, suitable
                        for an object with reasonable contrast)
          - pos_max_shift: maximum shift of position update in pixels between iterations (default:2)
          - pos_min_shift: minimum shift of position update in pixels between iterations (default:0)
          - pos_threshold: if the integrated norm of the object gradient multiplied by the probe is lower
                           than the average value (for all positions) multiplied by this threshold,
                           the position is not changed. This allows to avoid updating positions
                           in area where the object is flat, and sensitivity to shifts is low.
                           [default:0.2]
          - nbprobe=3: change the number of modes for the probe (can go up or down)
          - regularization=1e-4: setting the regularization parameter for the object, to penalize 
                                 local variations in ML runs and smooth the solution
          - obj_smooth=1.5 or probe_smooth=1.0: these parameters will partially smooth the object and probe inertia,
            softening the resulting arrays. This applies to DM and AP algorithms.
          - obj_inertia=0.01 or obj_probe=0.001: these parameters set the inertia of the object and/or probe update,
            yielding more stable result. This applies to DM and AP algorithms.
          - ortho: will perform orthogonalisation of the probe modes. The modes are sorted by 
            decreasing intensity.
          - analysis: perform an analysis of the probe (propagation, modes). Useful combined 
            with 'saveplot' to save the analysis plots
        
        - the second type are operators which will be applied to the Ptycho object:
          - AP: alternate projections. Slow but converging algorithm
          - DM: difference map. Fast early convergence, oscillating after.
          - ML: maximum likelihood conjugate gradient (Poisson-noise). Robust, converging,
            for final optimization.
          These operators can be combined mathematically, e.g.:
          - DM**100: corresponds to 100 cycles of difference map
          - ML**40*DM**100: 100 cycles of DM followed by 40 cycles of ML (note the order)


        Example algorithms chains:
          - algorithm=ML**40,DM**100,probe=1: activate probe optimisation, 
            then 100 DM and 40 ML (quick)
          - algorithm=ML**100,DM**200,nbprobe=3,ML**40,DM**100,probe=1,DM**100: first DM with 
            object update only,  then 100 DM also updating the probe, then use 3 probe modes 
            and do 100 DM followed by 40 ML
          - algorithm=ML**100*AP**200*DM**200,probe=1: 200 DM then 200 AP then 100 ML (one step)
          - algorithm='(ML**10*DM**20)**5,probe=1': 
            repeat 5 times [20 cycles of DM followed by 5 cycles of ML]
            (note the quotes necessary for the parenthesis)

    nbrun=10: number of optimizations to perform [default=1]

    run0=10: number for the first run (can be used to overwrite previous run results) 
             [default: after previous results or 1]

    liveplot: liveplot during optimization [default: no display]

    saveplot: will save plot at the end of the optimization (png file) [default= not saved].
              Optionally this can also specify if only the object phase should be plotted, e.g.:
              saveplot=object_phase: will display the object phase
              saveplot=object_rgba: will use RGBA to display both amplitude and phase.

    saveprefix=ResultsScan%04d/Run%04d: prefix to save the optimized object and probe 
              (as a .cxi or .npz file) and optionally image (png).
              Use "saveprefix=none" to disable saving (e.g. for tests)
              This can be used without a sub-directory, e.g. saveprefix=scan%04d_run%04d
              [default='ResultsScan%04d/Run%04d']

    output_format='cxi': choose the output format for the final object and support.
                         Possible choices: 'cxi', 'npz'
                         [Default='cxi']

    remove_obj_phase_ramp=0 or 1: if 1 (True, the default), the final object will be saved after
                                  removing the phase ramp estimated from the imperfect
                                  centring of the diffraction data (sub-pixel shift). Calculated
                                  diffraction patterns using such a corrected object will present
                                  a sub-pixel shift relative to the diffraction data.

    save=all: either 'final' or 'all' this keyword will activate saving after each optimization 
              step (ptycho, ML) of the algorithm in any given run [default=final]

    load=Results0057/Run0001.cxi (or .npz): load object and probe from previous optimization. Note that the
               object and probe will be scaled if the number of pixels is different for the probe.
              [default: start from a random object, simulate probe]


    loadprobe=Results0057/Run0001.npz (or .cxi): load only probe from previous optimization 
                                       [default: simulate probe]

    loadpixelsize=8.6e-9: specify the pixel size (in meters) from a loaded probe 
                          (and possibly object). If the pixel size is different,
                          the loaded arrays will be scaled to match the new pixel size.
                          [default: when loading previous files, object/probe pixel size is 
                          calculated from the size of the probe array, assuming same detector 
                          distance and pixel size]

    probe=focus,60e-6x200e-6,0.09: define the starting probe, either using:
                                  focus,60e-6x200e-6,0.09: slits size (horizontal x vertical),
                                                           focal distance (all in meters)
                                  focus,200e-6,0.09: radius of the circular aperture, 
                                                     focal distance (all in meters)
                                  gaussian,100e-9x200e-9: gaussian type with horizontal x vertical
                                                          FWHM, both given in meters
                                  disc,100e-9: disc-shape, with diameter given in meters
                                  [mandatory, ignored if 'load' or 'loadprobe' is used, or for near-field]

    defocus=1e-6: defocused position (+: towards detector). The initial probe is propagated
                  by this distance before being used. This is true both for calculated probes
                  (using 'probe=...')  and for probes loaded from a previous file.

    rotate=30: rotate the probe (either simulated or loaded) by X degrees [default: no rotation]

    object=random,0.9,1,0,6: specify the original object values. The object will be initialised
                             over the entire area using random values: random,0-1,0-6.28 : random 
                             amplitudes between 0 and 1, random phases between 0 and 6.28.
                             For high energy small-angle ptycho (i.e. high transmission), 
                             recommended value is: random,0.9,1,0,0
                             [default: random,0,1,0,6.28]

    verbose=20: print evolution of llk (and display plot if 'liveplot' is set) every N cycle 
                [default=50]

    data2cxi: if set, the raw data will be saved in CXI format (http://cxidb.org/cxi.html), 
              will all the required information for a ptychography experiment (energy, detector 
              distance, scan number, translation axis are all required). if 'data2cxi=crop' 
              is used, the data will be saved after centering and cropping (default is to save 
              the raw data). If this keyword is present, the processing stops after exporting the data.

    mask= or loadmask=mask.npy: the mask to be used for detector data, which should have the same 2D
                       shape as the raw detector data.
                       This should be a boolean or integer array with good pixels=0 and bad ones>0
                       (values are expected to follow the CXI convention)
                       Acceptable formats:
                       - mask.npy, mask.npz (the first data array will be used)
                       - mask.edf or mask.edf.gz (a single 2D array is expected)
                       - "mask.h5:/entry_1/path/to/mask" hdf5 format with the full path to the 
                         2D array. 'hdf5' is also accepted as extension.
                       - 'maxipix': if this special name is entered, the masked pixels will be rows 
                         and columns multiples of 258+/-3

    roi=xmin,xmax,ymin,ymax: the region-of-interest to be used for actual inversion. The area is taken 
                             with python conventions, i.e. pixels with indices xmin<= x < xmax and 
                             ymin<= y < ymax.
                             Additionally, the shape of the area must be square, and 
                             n=xmax-xmin=ymax-ymin must also be a suitable integer number
                             for OpenCL or CUDA FFT, i.e. it must be a multiple of 2 and the largest number in
                             its prime factor decomposition must be less or equal to the largest value
                             acceptable by clFFT (<=13 as of November 2016) or cuFFT (<=7).
                             If n does not fulfill these constraints,
                             it will be reduced using the largest possible integer smaller than n.
                             This option supersedes 'maxsize' unless roi='auto'.
                             Other possible values:
                             - 'auto': automatically selects the roi from the center of mass 
                                       and the maximum possible size. [default]
                             - 'all' or 'full': use the entire, uncentered frames. Only useful for pre-processed
                                      data. Cropping may still be performed to get a square and 
                                      FFT-friendly size.

    rebin=2: the experimental images can be rebinned (i.e. a group of n x n pixels is replaced by a
             single one whose intensity is equal to the sum of all the pixels). This 'rebin' is 
             performed last: the ROI, mask, background, pixel size should all correspond to 
             full (non-rebinned) frames. 
             [default: no rebin]

    autocenter=0: by default, the object and probe are re-centered automatically after each 
                  optimisation step, to avoid drifts. This can be used to deactivate this behaviour
                  [default=True]
    
    center_probe_n, center_probe_max_shift: during DM, the probe can shift. The probe center of mass  will be checked
                                            every center_probe_n cycles, and the object and probe position will be
                                            corrected if the center deviates by more than center_probe_max_shift pixels.
                                            This is ignored for near field ptycho.
                                            [default: center_probe_n=5, center_probe_max_shift=5]
    
    dm_loop_obj_probe=3: during DM, when both object and probe are updated, it can be more stable to loop the object
                         and probe update for a more stable optimisation, but slower. [default: 2]
                  
    detector_orientation=1,0,0: three flags which, if True, will do in this order: 
                                array transpose (x/y exchange), flipud, fliplr [default: no change]
                                The changes also apply to the mask.
    
    xy=y,x: order and expression to be used for the XY positions (e.g. '-x,y',...). Mathematical 
            operations can also be used, e.g.: xy=0.5*x+0.732*y,0.732*x-0.5*y
            [default: None- may be superseded in some scripts e.g. for ptypy]
    
    flatfield=flat.npy: the flatfield correction to be applied to the detector data. The array must
                        have the same shape as the frames, which will be multiplied by this 
                        correction.
                        Acceptable formats:
                        - flat.npy, flat.npz (the first data array will be used)
                        - flat.edf or flat.edf.gz (a single 2D array is expected)
                        - "flat.h5:/entry_1/path/to/flat" hdf5 format with the full path to the 
                          2D array. 'hdf5' is also accepted as extension.
                        - flat.mat: from a matlab file. The first array found is loaded
                       [default: no flatfield correction is applied]
    
    dark=dark.npy: the dark correction  (incoherent background). The array must have 
                       the same shape as the frames. This will be taken into account during
                       the optimisation, and not subtracted from the observed data, unless
                       substract_dark is used.
                       Acceptable formats:
                       - dark.npy, dark.npz (the first data array will be used)
                       - dark.edf or dark.edf.gz (a single 2D array is expected)
                       - "dark.h5:/entry_1/path/to/dark" hdf5 format with the full path to the 
                         2D array. 'hdf5' is also accepted as extension.
                       - dark.mat: from a matlab file. The first array found is loaded
                       [default: no dark correction is applied]
    
    dark_subtract or dark_subtract=0.9: use this to subtract the dark from the observed
        intensity. This is normally discouraged, as it will mess up the Poisson statistics
        of the observed data, but it can be useful in the case of very high background.
        If given as a simple keyword, the dark is subtracted. If given with a float,
        the dark will be multiplied by this factor before subtraction.
        [default: False, dark is not subtracted]
    
    orientation_round_robin: will test all possible combinations of xy and detector_orientation 
                             to find the correct detector configuration.
    
    mpi=multiscan or split: when launching the script using mpiexec, this tells the script to
              either distribute the list of scans to different processes (multiscan, the default),
              or split a large scan in different parts, which are automatically aligned and
              stitched. Examples:
              mpiexec -n 2 pynx-cxipty.py scan=11,12,13,14 data=scan%02d.cxi mpi=multiscan
                                           probe=focus,120e-6x120e-6,0.1 defocus=100e-6 verbose=50
                                           algorithm=analysis,ML**100,DM**200,nbprobe=2,probe=1 
              mpiexec -n 4 pynx-cxipty.py  mpi=split data=scan151.cxi verbose=50
                                           probe=focus,120e-6x120e-6,0.1 defocus=100e-6
                                           algorithm=analysis,ML**100,DM**200,nbprobe=2,probe=1 
    
    mpi_split_nb_overlap=2: depth of shared neighbours between overlapping regions when using
                            mpi=split [default:1] (experimental, may change)
    
    mpi_split_nb_neighbour=20: target number of shared neighbours for each set for overlap 
                            [default:20]  (experimental, may change)

    multiscan_reuse_ptycho: if used as a keyword, successive scans will re-start from the previous
        ptycho object and probe, and skip initialisation steps. Useful for ptycho-tomo.
        This can also be used to supply a shorter algorithm chain which will be used after the first
        scan, e.g. with:
            algorithm=ML**100,AP**100,DM**1000,nbprobe=2,probe=1
            multiscan_reuse_ptycho=ML**100,AP**100,probe=1,AP**50,probe=0
        In the above example, the probe would be re-used so there is no need to use 'nbprobe=2',
        and the first step (AP**50) would only optimise the object.
        Note that if you want to re-process the first scan with this short algorithm chain,
        it is possible to list the first scan twice, e.g.: scan=12,12,13,14,15
        [default: False, every scan starts from a new object and probe]
==================================================================================================
                                   End of common help text
                       Instructions for the specific script are given below
==================================================================================================
"""

# This must be defined in in beamline/instrument-specific scripts
helptext_beamline = ""

params_generic = {'scan': None, 'algorithm': 'ML**50,DM**100', 'nbrun': 1, 'run0': None, 'liveplot': False,
                  'saveplot': False, 'saveprefix': 'ResultsScan%04d/Run%04d', 'livescan': False, 'load': None,
                  'loadprobe': None, 'probe': None, 'defocus': None, 'gpu': None, 'regularization': 0, 'save': 'final',
                  'loadpixelsize': None, 'rotate': None, 'maxframe': None, 'maxsize': 512, 'output_format': 'cxi',
                  'object': 'random,0.9,1,0,0.5', 'verbose': 50, 'moduloframe': None, 'data2cxi': False, 'nrj': None,
                  'pixelsize': None, 'instrument': None, 'epoch': time.time(), 'cxifile': None,
                  'detector_orientation': None, 'xy': None, 'loadmask': None, 'roi': 'auto',
                  'rebin': None, 'autocenter': True, 'data': None, 'flatfield': None, 'dark': None,
                  'orientation_round_robin': False, 'fig_num': 100, 'profiling': False, 'stack_size': None,
                  'obj_smooth': 0, 'obj_inertia': 0.05, 'probe_smooth': 0, 'probe_inertia': 0.005, 'movie': None,
                  'center_probe_n': 5, 'center_probe_max_shift': 5, 'dm_loop_obj_probe': 1,
                  'obj_max_pix': 8000, 'mpi': 'multiscan', 'mpi_split_nb_overlap': 1, 'remove_obj_phase_ramp': True,
                  'pos_mult': 5, 'pos_max_shift': 1, 'pos_min_shift': 0, 'pos_threshold': 0.2, 'obj_margin': 32,
                  'mpi_split_nb_neighbour': 20, 'near_field': False, 'multiscan_reuse_ptycho': False,
                  'padding': 0, 'dark_subtract': 0, 'background_smooth': 3, 'detectordistance': None}


class PtychoRunnerException(Exception):
    pass


class PtychoRunnerScan(object):
    """
    Abstract class to handle ptychographic data. Must be derived to be used.
    Only the load_scan() and load_data() functions need be derived.
    """

    def __init__(self, params, scan, mpi_comm=None, timings=None):
        self.params = params
        self.scan = scan
        self.defocus_done = False
        self.raw_data_monitor = None
        self.raw_mask = None  # Original mask (uncropped, etc..)
        self.mask = None  # mask for running algorithm
        self.rebinf = 1
        self._run = None
        self.processing_unit = None
        self.p = None  # Ptycho object
        self.data = None  # PtychoData object
        self.flatfield = None
        self.dark = None
        self.raw_dark = None
        self.iobs = None
        self.raw_x, self.raw_y, self.x, self.y = None, None, None, None
        self.imgn = None
        # Keep the complete list of positions for simulation
        self.mpi_x = None
        self.mpi_y = None

        self.probe0 = None
        self.obj0 = None

        # Coordinates (x,y) of neighbour points with each MPI process
        self.mpi_neighbour_xy = None

        # Default parameters for optimization
        self.update_object = True
        self.update_probe = False
        self.update_background = 0
        self.update_position = False
        self.floating_intensity = False
        self.mpi_master = True  # True also if MPI is not used
        if MPI is not None:
            self.mpic = mpi_comm
            if self.mpic is None:
                self.mpic = MPI.COMM_WORLD
            # This may not be a very clean method to differentiate 'split' and 'multi' MPI...
            self.mpi_master = (self.mpic.Get_rank() == 0) or ('split' not in self.params['mpi'])
            self.mpi_size = self.mpic.Get_size()
            self.mpi_rank = self.mpic.Get_rank()

        # Total time spent on algorithms
        self.timings = timings

    def print(self, *args, **kwargs):
        """
        MPI-aware print function. Non-master processes will be muted
        :param args: args passed to print
        :param kwargs: kwrags passed to print
        :return: nothing
        """
        if self.mpi_master:
            print(*args, **kwargs)

    def load_scan(self):
        """
        Loads scan positions, using beamline-specific parameters. Abstract function, must be derived.
        This also filters the set of scan positions according to parameters (xyrange, monitor values, maxframe,...)

        If MPI is used, only the master reads the scan positions. This automatically calls mpi_scan_split() at the
        end to split the scan if necessary (when MPI is used and mpi='splitscan' is used).

        Returns: Nothing. The  scan positions, and the scan position indices ()
                 to be loaded for that runner are stored in self.x, self.y, self.imgn.
                 If MPI is used, only the master should call this,
                 and then call mpi_scan_split(). This is handled in Runner:process_scan().
        """
        raise PtychoRunnerException("You should not call pure virtual PtychoRunnerScan.load_scan() ? "
                                    "It must be superseded in a child class for each instrument/beamline")

    def mpi_scan_split(self):
        """
        This function is called after load_scan(), and will split the scan among all the MPI process.
        If MPI is not used, it does nothing and just passes the (x, y, imgn) values.

        :return: nothing. The x, y, imgn attributes are updated if necessary after splitting among MPI processes
        """
        if 'split' not in self.params['mpi'] or MPI is None:
            return
        if self.mpi_size == 1:
            return

        x_orig, y_orig, imgn_orig = self.x, self.y, self.imgn
        x, y, imgn = self.x, self.y, self.imgn
        if self.mpi_master:
            self.mpi_x = self.x.copy()
            self.mpi_y = self.y.copy()
            # Split the scan positions in mpi_size subsets, and broadcast the data
            k_means = KMeans(init='k-means++', n_clusters=self.mpi_size, n_init=10)
            X = np.stack((self.x, self.y)).transpose()
            k_means.fit(X)
            vidx = []
            for i in range(self.mpi_size):
                vidx.append(np.where(k_means.labels_ == i)[0])

            # Average nearest neighbour distance
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
            # We assume that spacing between points is reasonably regular
            max_dist = np.percentile(distances[:, 1], 90) * (self.params['mpi_split_nb_overlap'] + 0.5)

            #############################################################################
            # The sets of points are not generally equally-sized, so adjust that
            def get_first_neighbours(i, vidx, max_dist):
                """ Get the list of neighbouring points around a set"""
                vn = {}
                nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X[vidx[i]])
                for ii in range(len(vidx)):
                    if ii != i:
                        distances, indices = nbrs.radius_neighbors(X[vidx[ii]], max_dist, sort_results=True)
                        vn[ii] = set()
                        for iii in range(len(vidx[ii])):
                            if len(indices[iii]):
                                vn[ii].add(iii)
                    else:
                        vn[ii] = []
                return vn

            # Size of all sets of points
            vnb = np.array([len(idx) for idx in vidx], dtype=np.int)
            # print(vnb)
            nb = len(self.x)
            nb0 = nb / self.mpi_size  # Ideal size for all sets
            for iter_ in range(10):
                # print(iter_, abs(vnb - nb0).max(), max(0.02 * nb0, 5))
                if abs(vnb - nb0).max() < max(0.01 * nb0, 2):
                    break
                for i in np.argsort(vnb):
                    dn = nb0 - vnb[i]
                    if dn <= 2:
                        break

                    # Grab points from neighbour regions, if some are in excess
                    vn = get_first_neighbours(i, vidx, max_dist)
                    vnb0 = np.array([len(vn[k]) for k in range(self.mpi_size)])
                    # print(i, vnb0, dn)
                    # Only exchange points with sets which have a minimum of 10 neighbours
                    tmp = (vnb - (nb0 + vnb[i]) / 2) * (vnb0 > 10)
                    tmp = np.floor(tmp * dn / tmp.sum()).astype(np.int)
                    for j in range(self.mpi_size):
                        if tmp[j] > 0:
                            # Need to grab tmp[j] points into i from j
                            # These points need to be the closest from the center of i
                            # AND need to be among listed neighbours
                            x0, y0 = x[vidx[i]].mean(), y[vidx[i]].mean()
                            vnj = list(vn[j])
                            tmpx, tmpy = x[vidx[j][vnj]], y[vidx[j][vnj]]

                            idx = np.argsort((tmpx - x0) ** 2 + (tmpy - y0) ** 2)[:tmp[j]]

                            idx1, idx2 = set(vidx[i]), set(vidx[j])
                            # print("%2d<-%2d  %2d" % (i, j, tmp[j]))  # , np.array(vnj)[idx]
                            for k in idx:
                                ii = vidx[j][vnj[k]]
                                idx1.add(ii)
                                idx2.remove(ii)
                            vidx[i] = np.array(list(idx1), dtype=np.int)
                            vidx[j] = np.array(list(idx2), dtype=np.int)
                            vnb[i] = len(idx1)
                            vnb[j] = len(idx2)

            #############################################################################
            vc = {}
            if self.params['liveplot'] and self.mpi_master:
                from matplotlib.pyplot import cm
                plt.figure(figsize=(12, 6))
                plt.subplot(121)
                plt.scatter(x, y, s=6)
                dx, dy = x.max() - x.min(), y.max() - y.min()
                plt.xlim(x.min() - .05 * dx, x.max() + .05 * dx)
                plt.ylim(y.min() - .05 * dy, y.max() + .05 * dy)
                plt.gca().set_aspect(1)
                plt.subplot(122)
                color = iter(cm.rainbow(np.linspace(0, 1, self.mpi_size)))
                for i in range(self.mpi_size):
                    vc[i] = next(color)
                    idx = vidx[i]
                    plt.scatter(x[idx], y[idx], s=8, color=vc[i])
                    x0, y0 = k_means.cluster_centers_[i]
                    plt.text(x0, y0, "%d\n[%d]" % (i, len(idx)), fontsize=16, horizontalalignment='center',
                             verticalalignment='center', weight='bold')
                plt.xlim(x.min() - .05 * dx, x.max() + .05 * dx)
                plt.ylim(y.min() - .05 * dy, y.max() + .05 * dy)
                plt.gca().set_aspect(1)
                plt.draw()
                plt.gcf().canvas.draw()
                plt.pause(.01)

            # Determine the neighbours between subsets
            v_neighbours = {}

            for i1 in range(self.mpi_size):
                idx1 = vidx[i1]
                if i1 not in v_neighbours:
                    v_neighbours[i1] = {}
                for i2 in range(i1 + 1, self.mpi_size):
                    if i2 not in v_neighbours:
                        v_neighbours[i2] = {}
                    v_neighbours[i1][i2] = set()  # i2 points neighbours of i1
                    v_neighbours[i2][i1] = set()  # v_neighbours[i1][i2]  # Symmetric
                    idx2 = vidx[i2]

                    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X[idx1])
                    distances, indices = nbrs.radius_neighbors(X[idx2], max_dist, sort_results=True)
                    # print(i1, i2, v_neighbours[i1][i2], "\n\n", indices)
                    for i in range(len(idx2)):
                        d, idx = distances[i], indices[i],
                        if len(idx):
                            # i1 has idx2[i] in i2 as neighbour
                            v_neighbours[i1][i2].add(idx2[i])
                            for ne in idx:
                                # i2 has idx1[ne] in i1 as neighbour
                                v_neighbours[i2][i1].add(idx1[ne])

                    # print("neighbours[%d,%d]=%3d" % (i1, i2, len(v_neighbours[i1][i2])))
                    # print("neighbours[%d,%d]=%3d" % (i2, i1, len(v_neighbours[i2][i1])))

            # Original points unique to each set
            vpoints = {}
            # Points in each set after adding neighbours
            vpoints_neigh = {}
            # Final list of shared points after re-arranging neighbours
            v_neighbours_final = {}

            # Matrix of the number of neighbour points per subset
            mn = np.zeros((self.mpi_size, self.mpi_size), dtype=np.int32)
            for i1 in range(self.mpi_size):
                # Number of points in the set
                # mn[i1, i1] = len(vidx[i1])
                for i2 in range(self.mpi_size):
                    if i1 != i2:
                        mn[i1, i2] = len(v_neighbours[i1][i2])
            # Choose a maximum number of neighbour points to add for synchronisation
            vnb = np.array([len(v) for v in vidx], dtype=np.int32)
            mnb = mn.sum(axis=-1)
            # Reach about nb points after adding neighbours, if possible
            nb = min((vnb + mnb).min(), vnb.max() + self.params['mpi_split_nb_neighbour'])
            # print("Number of points per set with neighbours: %3d -> %3d" % (vnb.mean(), nb))
            rng = np.random.default_rng()
            for i1 in range(self.mpi_size):
                vpoints[i1] = set(vidx[i1])
                vpoints_neigh[i1] = set(vidx[i1])
                tmp = np.array([len(v) for v in v_neighbours[i1].values()], dtype=np.int32)
                scale_nb = (nb - vnb[i1]) / tmp.sum()
                # print(i1, np.round(tmp * scale_nb), np.round(tmp * scale_nb).sum() + vnb[i1])
                v_neighbours_final[i1] = {}
                for i2, v in v_neighbours[i1].items():
                    n = int(np.ceil(scale_nb * len(v)))
                    if n > 0:
                        idx = np.array(list(v), dtype=np.int32)
                        idx = rng.choice(idx, n, replace=False, shuffle=False)
                        v_neighbours_final[i1][i2] = idx
                        n0 = len(vpoints_neigh[i1])
                        vpoints_neigh[i1] = vpoints_neigh[i1].union(set(idx.tolist()))
                        # print("  %d: add %2d neighbours from %d" % (i1, len(vpoints_neigh[i1]) - n0, i2), n)
                        if self.params['liveplot']:
                            plt.scatter(x[idx], y[idx], s=2, c=[vc[i1]])

            for i1 in range(self.mpi_size):
                print("MPI subset %2d: %4d points (%4d with neighbours)" %
                      (i1, len(vpoints[i1]), len(vpoints_neigh[i1])))

                if i1 == 0:
                    imgn = imgn_orig[list(vpoints_neigh[i1])].astype(np.int32)
                    x = x_orig[list(vpoints_neigh[i1])].astype(np.float32)
                    y = y_orig[list(vpoints_neigh[i1])].astype(np.float32)
                else:
                    # print("MPI #%d sending %d points to #%d" % (0, len(vpoints_neigh[i1]), i1))
                    self.mpic.send(len(vpoints_neigh[i1]), dest=i1, tag=10)
                    self.mpic.Send(imgn_orig[list(vpoints_neigh[i1])].astype(np.int32), dest=i1, tag=11)
                    self.mpic.Send(x_orig[list(vpoints_neigh[i1])].astype(np.float32), dest=i1, tag=12)
                    self.mpic.Send(y_orig[list(vpoints_neigh[i1])].astype(np.float32), dest=i1, tag=13)
                # Send list of neighbour coordinates for each pair of object
                mpi_neighbour_xy = {}
                for i2 in range(self.mpi_size):
                    if i2 == i1:
                        continue
                    if i2 in v_neighbours_final[i1]:
                        # print("v_neighbours_final[%d, %d]:"%(i1,i2))
                        # print("   ", list(v_neighbours_final[i1][i2]), len(v_neighbours_final[i1][i2]))
                        # print("   ", list(v_neighbours_final[i2][i1]), len(v_neighbours_final[i2][i1]))
                        idx = set(list(v_neighbours_final[i1][i2]))
                        idx = list(idx.union(list(v_neighbours_final[i2][i1])))
                        tmpx = x_orig[idx].astype(np.float32)
                        tmpy = y_orig[idx].astype(np.float32)
                        if len(tmpx):
                            mpi_neighbour_xy[i2] = (tmpx, tmpy)
                if i1 == 0:
                    self.mpi_neighbour_xy = mpi_neighbour_xy
                else:
                    self.mpic.send(mpi_neighbour_xy, dest=i1, tag=14)

            if self.params['liveplot']:
                plt.draw()
                plt.gcf().canvas.draw()
                plt.pause(0.01)
                if self.params['saveplot']:
                    run0 = self.init_run_number()
                    sf = self.params['saveprefix'] % (self.scan, run0) + '-split.png'
                    print("Saving split positions to: %s" % sf)
                    plt.savefig(sf)
                    if os.path.isfile(sf):
                        sf = os.path.split(sf)
                        os.system('ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest-split.png')))

        else:
            nb = self.mpic.recv(source=0, tag=10)
            imgn = np.empty(nb, dtype=np.int32)
            self.mpic.Recv(imgn, source=0, tag=11)
            x = np.empty(nb, dtype=np.float32)
            self.mpic.Recv(x, source=0, tag=12)
            y = np.empty(nb, dtype=np.float32)
            self.mpic.Recv(y, source=0, tag=13)
            self.mpi_neighbour_xy = self.mpic.recv(source=0, tag=14)

            # print("MPI #%d received %d points from #%d" % (self.mpi_rank, nb, 0))
        # for k, v in self.mpi_neighbour_xy.items():
        #     print("MPI #%d: %d neighbour points with #%d" % (self.mpi_rank, len(v[0]), k))

        # print("mpi_scan_split(): #%d, pos_c = (%7.2f, %7.2f)" % (self.mpi_rank, x.mean() * 1e6, y.mean() * 1e6))
        self.x, self.y, self.imgn = x, y, imgn

    def load_data(self):
        """
        Loads data, using beamline-specific parameters. Abstract function, must be derived

        Returns:

        """
        raise PtychoRunnerException("You should not call pure virtual PtychoRunnerScan.load_data(). "
                                    "It should be superseded in a child class for each instrument/beamline")

    def prepare_processing_unit(self):
        """
        Prepare processing unit (CUDA, OpenCL, or CPU). This must be called after load_scan so that the size of the
        dataset is known

        Returns: nothing. Creates self.processing_unit, and adapts the stack size for CUDA

        """
        if default_processing_unit.cu_device is None and default_processing_unit.cl_device is None:
            s = "Ptycho runner: preparing processing unit"
            if self.params['gpu'] is not None:
                s += " [given GPU name: %s]" % str(self.params['gpu'])
            self.print(s)
            try:
                default_processing_unit.select_gpu(gpu_name=self.params['gpu'], verbose=True)
            except Exception as ex:
                s0 = "\n  original error: " + str(ex)
                if self.params['gpu'] is not None:
                    s = "Failed initialising GPU. Please check GPU name [%s] or CUDA/OpenCL installation"
                    raise PtychoRunnerException(s % str(self.params['gpu']) + s0)
                else:
                    raise PtychoRunnerException(
                        "Failed initialising GPU. Please check GPU name or CUDA/OpenCL installation" + s0)

        if default_processing_unit.pu_language == 'cpu':
            raise PtychoRunnerException("CUDA or OpenCL or GPU not available - you need a GPU to use pynx.ptycho !")

        if self.params['stack_size'] is not None:
            default_processing_unit.set_stack_size(self.params['stack_size'])
            self.print('Set GPU stack size to: ', self.params['stack_size'])
        else:
            if default_processing_unit.pu_language == 'cuda':
                # Set stack size to size of data
                n = len(self.x)
                # Avoid too large stack sizes
                if n > 128:
                    n = int(np.ceil(n / np.ceil(n / 128)))

                default_processing_unit.set_stack_size(n)
                self.print("Using CUDA GPU=> setting large stack size (%d) (override with stack_size=N)" % n)

        self.processing_unit = default_processing_unit

    def load_data_post_process(self):
        """
        Applies some post-processing to the input data, according to parameters. Also loads the mask.
        User-supplied mask is loaded if necessary.

        This must be called at the end of load_data()
        :return:
        """
        self._init_mask(self.raw_data[0].shape)
        self._load_flat_field()
        self._load_dark()

        if self.raw_mask is not None:
            if self.raw_mask.shape != self.raw_data[0].shape:
                raise PtychoRunnerException("Mask and raw data shape are not identical !")

        if self.flatfield is not None:
            if self.flatfield.shape != self.raw_data[0].shape:
                raise PtychoRunnerException("flatfield and raw data shapes are not identical !")

        if self.dark is not None:
            if self.dark.shape != self.raw_data[0].shape:
                print(self.dark.shape, self.raw_data.shape)
                raise PtychoRunnerException("dark and raw data shapes are not identical !")

        if not self.params['near_field']:
            self.params['padding'] = 0
        padding = self.params['padding']

        if padding:
            ny, nx = self.raw_data.shape[-2:]
            tmp = np.zeros((len(self.raw_data), ny + 2 * padding, nx + 2 * padding), dtype=np.float32)
            tmp[:, padding:-padding, padding:-padding] = self.raw_data
            self.raw_data = tmp

            if self.raw_mask is not None:
                tmp = np.zeros((ny + 2 * padding, nx + 2 * padding), dtype=np.float32)
                tmp[padding:-padding, padding:-padding] = self.raw_mask
                self.raw_mask = tmp

            if self.flatfield is not None:
                tmp = np.ones((ny + 2 * padding, nx + 2 * padding), dtype=np.float32)
                tmp[padding:-padding, padding:-padding] = self.flatfield
                self.flatfield = tmp

            if self.dark is not None:
                tmp = np.zeros((ny + 2 * padding, nx + 2 * padding), dtype=np.float32)
                tmp[padding:-padding, padding:-padding] = self.dark
                self.dark = tmp

        # Store original x,y in case we use self.params['xy']
        self.raw_x, self.raw_y = self.x, self.y

    def _init_mask(self, shape):
        """
        Load mask if the corresponding parameter has been set, or just initialize an array of 0.
        This is called after raw data has been loaded by load_data()
        Note that a mask may already exist if pixels were flagged by the detector

        Args:
            shape: the 2D shape of the raw data
        Returns:
            Nothing
        """
        mask_user = None
        if self.params['loadmask'] is not None:
            if self.params['loadmask'].find('.h5:') > 0 or self.params['loadmask'].find('.hdf5:') > 0:
                # hdf5 file with path to mask
                s = self.params['loadmask'].split(':')
                h5f = h5py.File(s[0], 'r')
                if s[1] not in h5f:
                    raise PtychoRunnerException(
                        "Error extracting mask from hdf5file: path %s not found in %s" % (s[1], s[0]))
                mask_user = h5f[s[1]][()]
                h5f.close()
            elif self.params['loadmask'] == 'maxipix':
                mask_user = np.zeros(shape, dtype=np.int8)
                ny, nx = shape
                for i in range(258, ny, 258):
                    mask_user[i - 3:i + 3] = 1
                for i in range(258, nx, 258):
                    mask_user[:, i - 3:i + 3] = 1
            else:
                filename = self.params['loadmask']
                ext = os.path.splitext(filename)[-1]
                if ext == '.edf' or ext == '.gz':
                    mask_user = fabio.open(filename).data
                elif ext == '.npy':
                    mask_user = np.load(filename)
                elif ext == '.npz':
                    for v in np.load(filename).items():
                        mask_user = v[1]
                        break
                elif ext == '.tif' or ext == '.tiff':
                    mask_user = np.array(Image.open(filename)) > 0
                else:
                    self.print(ext)
                    self.print("What is this mask extension: %s ??" % (ext))
            self.print("Loaded MASK from: %s with % d pixels masked (%5.3f%%)"
                       % (self.params['loadmask'], mask_user.sum(), mask_user.sum() * 100 / mask_user.size))
        if self.raw_mask is None:
            if mask_user is not None:
                self.raw_mask = mask_user.astype(np.int8)
        elif mask_user is not None:
            self.raw_mask += mask_user.astype(np.int8)
        if self.raw_mask is not None:
            s = self.raw_mask.sum()
            if s:
                self.print("Initialized mask with %d (%6.3f%%) bad pixels" % (s, s * 100 / self.raw_mask.size))

    def _load_flat_field(self):
        """
        Load flat field if the corresponding parameter has been set.

        Returns:
            Nothing
        """
        flatfield = None
        if self.params['flatfield'] is not None:
            if self.params['flatfield'].find('.h5:') > 0 or self.params['flatfield'].find('.hdf5:') > 0:
                # hdf5 file with path to flatfield
                s = self.params['flatfield'].split(':')
                h5f = h5py.File(s[0], 'r')
                if s[1] not in h5f:
                    raise PtychoRunnerException(
                        "Error extracting flatfield from hdf5file: path %s not found in %s" % (s[1], s[0]))
                flatfield = h5f[s[1]][()]
                h5f.close()
            else:
                filename = self.params['flatfield']
                ext = os.path.splitext(filename)[-1]
                if ext == '.edf' or ext == '.gz':
                    flatfield = fabio.open(filename).data
                elif ext == '.npy':
                    flatfield = np.load(filename)
                elif ext == '.npz':
                    # Just grab the first array larger than 1000 values
                    for k, v in np.load(filename).items():
                        if np.size(v) > 1000:
                            flatfield = v
                            break
                elif ext == '.mat':
                    a = list(loadmat(filename).values())
                    for v in a:
                        if np.size(v) > 1000:
                            # Avoid matlab strings and attributes, and get the array
                            flatfield = np.array(v)
                            break
                else:
                    self.print(ext)
                    self.print("What is this flatfield extension: %s ??" % (ext))
            self.print("Loaded FLATFIELD from: ", self.params['flatfield'])
        if flatfield is not None:
            self.flatfield = flatfield.astype(np.float32)
            self.flatfield /= self.flatfield.mean()

    def _load_dark(self):
        """
        Load dark if the corresponding parameter has been set.

        Returns:
            Nothing
        """
        dark = None
        if self.params['dark'] is not None:
            if self.params['dark'].find('.h5:') > 0 or self.params['dark'].find('.hdf5:') > 0:
                # hdf5 file with path to dark
                s = self.params['dark'].split(':')
                h5f = h5py.File(s[0], 'r')
                if s[1] not in h5f:
                    raise PtychoRunnerException(
                        "Error extracting dark from hdf5file: path %s not found in %s" % (s[1], s[0]))
                dark = h5f[s[1]][()]
                if dark.ndim == 3:
                    dark = dark.astype(np.float32).mean(axis=0)
                h5f.close()
            else:
                filename = self.params['dark']
                ext = os.path.splitext(filename)[-1]
                if ext == '.edf' or ext == '.gz':
                    dark = fabio.open(filename).data
                elif ext == '.npy':
                    dark = np.load(filename)
                elif ext == '.npz':
                    # Just grab the first array larger than 1000 values
                    for k, v in np.load(filename).items():
                        if np.size(v) > 1000:
                            dark = v
                            break
                elif ext == '.mat':
                    a = list(loadmat(filename).values())
                    for v in a:
                        if np.size(v) > 1000:
                            # Avoid matlab strings and attributes, and get the array
                            dark = np.array(v)
                            break
                else:
                    self.print(ext)
                    self.print("What is this dark extension: %s ??" % (ext))
            self.print("Loaded DARK from: ", self.params['dark'])
        if dark is not None:
            self.dark = dark.astype(np.float32)
            self.raw_dark = dark

    def center_crop_data(self):
        """
        Once the data has been loaded in self.load_data() [overloaded in child classes), this function can be called at the end of load_data
        to take care of centering the data (finding the center of diffraction) and cropping it with a size suitable for clFFT.
        Rebin is performed if necessary.

        Returns:
            Nothing. self.iobs and self.dsize are updated. self.raw_data holds the raw data if needed for CXI export
        """
        if self.params['xy'] is not None:
            # TODO: move this to load_data_post_process ?
            x, y = self.raw_x, self.raw_y
            self.x, self.y = eval(self.params['xy'])
        else:
            self.x, self.y = self.raw_x, self.raw_y

        raw_data = self.raw_data

        mask = self.raw_mask

        if self.flatfield is not None:
            flatfield = self.flatfield
        else:
            flatfield = 1

        if self.dark is not None:
            dark = self.dark
        else:
            dark = 0

        if self.params['detector_orientation'] is not None:
            # TODO: move this to load_data_post_process ?
            # User-supplied change of orientation
            do_transpose, do_flipud, do_fliplr = eval(self.params['detector_orientation'])

            if do_fliplr or do_flipud or do_transpose:
                self.print("Transpose: %d, flipud: %d, fliplr: %d" % (do_transpose, do_flipud, do_fliplr))
                if do_transpose:
                    raw_data = self.raw_data.transpose((0, -1, -2))
                if do_flipud:
                    raw_data = self.raw_data[..., ::-1, :]
                if do_fliplr:
                    raw_data = self.raw_data[..., ::-1]

                if mask is not None:
                    if do_transpose:
                        mask = self.raw_mask.transpose((-1, -2))
                    if do_flipud:
                        mask = self.raw_mask[..., ::-1, :]
                    if do_fliplr:
                        mask = self.raw_mask[..., ::-1]

                if self.flatfield is not None:
                    if do_transpose:
                        flatfield = self.flatfield.transpose((-1, -2))
                    if do_flipud:
                        flatfield = self.flatfield[..., ::-1, :]
                    if do_fliplr:
                        flatfield = self.flatfield[..., ::-1]

                if self.dark is not None:
                    if do_transpose:
                        dark = self.dark.transpose((-1, -2))
                    if do_flipud:
                        dark = self.dark[..., ::-1, :]
                    if do_fliplr:
                        dark = self.dark[..., ::-1]

        if self.mpi_master:
            # Only the master MPI process prepares the centering & cropping
            raw_data_sum = raw_data.sum(axis=0) - dark * len(raw_data)
            # Find image center
            ny, nx = raw_data_sum.shape

            raw_data_sum0 = raw_data_sum
            # Did user set the ROI to use ?
            if self.params['roi'] == 'auto':
                X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
                if self.raw_mask is not None:
                    raw_data_sum[mask > 0] = 0

                # Try to remove hot pixels, but not if too much intensity is removed
                tmp = medfilt2d(raw_data_sum.astype(np.float32), 3)  # Should remove hot pixels
                if tmp.sum() > 0.3 * raw_data_sum.sum():
                    raw_data_sum = tmp

                # tmp = raw_data_sum > np.percentile(raw_data_sum, 95)
                # if tmp.astype(np.int32).sum() > 100:
                #     raw_data_sum *= tmp

                # We don't use center_of_mass which does not work with masked arrays
                if mask is not None:
                    raw_data_sum = np.ma.masked_array(raw_data_sum, mask)
                x0, y0 = (raw_data_sum * X).sum() / raw_data_sum.sum(), (raw_data_sum * Y).sum() / raw_data_sum.sum()
                self.print("Center of diffraction: X=%6.2f Y=%6.2f" % (x0, y0))
                x0n, y0n = (int(round(x0)), int(round(y0)))

                # Maximum window size
                if self.params['maxsize'] is not None:
                    self.dsize = int(min(x0n, y0n, nx - x0n, ny - y0n, self.params['maxsize'] // 2)) * 2
                else:
                    self.dsize = int(min(x0n, y0n, nx - x0n, ny - y0n)) * 2
            elif self.params['roi'] in ['all', 'full']:
                xmin, xmax, ymin, ymax = 0, raw_data.shape[-1], 0, raw_data.shape[-2]
                x0n = int((xmax + xmin) // 2)
                y0n = int((ymax + ymin) // 2)
                self.dsize = min((xmax - xmin, ymax - ymin))
            else:
                vs = self.params['roi'].split(',')
                xmin, xmax, ymin, ymax = int(eval(vs[0])), int(eval(vs[1])), int(eval(vs[2])), int(eval(vs[3]))
                # Handle case when limits go beyond array size, indicating a circular shift of the data is needed
                if xmin < 0 or xmax >= nx:
                    if xmin < 0:
                        dx = -xmin
                    else:
                        dx = nx - xmax
                    self.print("Circular shift of diffraction frames (and mask,..) along x by %d" % dx)
                    self.raw_data = np.roll(self.raw_data, dx, axis=-1)
                    raw_data = self.raw_data
                    if self.raw_mask is not None:
                        self.raw_mask = np.roll(self.raw_mask, dx, axis=-1)
                    mask = self.raw_mask
                    raw_data_sum0 = np.roll(raw_data_sum0, dx, axis=-1)
                    if type(flatfield) == type(raw_data):
                        flatfield = np.roll(flatfield, dx, axis=-1)
                    if type(dark) == type(raw_data):
                        dark = np.roll(dark, dx, axis=-1)
                    xmin += dx
                    xmax += dx

                if ymin < 0 or ymax >= ny:
                    if ymin < 0:
                        dy = -ymin
                    else:
                        dy = ny - ymax
                    self.print("Circular shift of diffraction frames (and mask,..) along y by %d" % dy)
                    self.raw_data = np.roll(self.raw_data, dy, axis=-2)
                    raw_data = self.raw_data
                    if self.raw_mask is not None:
                        self.raw_mask = np.roll(self.raw_mask, dy, axis=-2)
                    mask = self.raw_mask
                    raw_data_sum0 = np.roll(raw_data_sum0, dy, axis=-2)
                    if type(flatfield) == type(raw_data):
                        flatfield = np.roll(flatfield, dy, axis=-2)
                    if type(dark) == type(raw_data):
                        dark = np.roll(dark, dy, axis=-2)
                    ymin += dy
                    ymax += dy

                x0n = int((xmax + xmin) // 2)
                y0n = int((ymax + ymin) // 2)
                self.dsize = min((xmax - xmin, ymax - ymin))

            # Rebin ?
            if self.params['rebin'] is not None:
                self.rebinf = self.params['rebin']
                self.dsize = self.dsize // self.rebinf

            # Compute acceptable size, depending on cuFFT or clFFT version, with both dimensions even
            prime_fft = self.processing_unit.max_prime_fft_radix()
            self.print("Largest prime number acceptable for FFT size:", prime_fft)
            assert (self.rebinf <= prime_fft)
            self.dsize = smaller_primes(self.dsize, prime_fft, [2])

            ds2r = self.dsize // 2 * self.rebinf

            if self.params['liveplot'] and self.mpi_master:
                # Plot crop area and highlight masked pixels
                plt.figure(99)
                plt.clf()
                vmax = np.log10(np.percentile(raw_data_sum0, 99.9))
                v = np.log10(raw_data_sum0 + 1e-6)
                if mask is not None:
                    v = v * (mask == 0) + vmax * 1.2 * (mask != 0)
                plt.imshow(v, vmin=0, vmax=vmax * 1.2, cmap=plt.cm.jet)
                plt.plot([x0n - ds2r, x0n + ds2r, x0n + ds2r, x0n - ds2r, x0n - ds2r],
                         [y0n - ds2r, y0n - ds2r, y0n + ds2r, y0n + ds2r, y0n - ds2r], 'r')
                plt.colorbar()
                plt.title("Sum of raw data [log scale 0-99.9%%, cutoff=1], crop area and masked pixels")
                plt.xlim(0, nx)
                plt.ylim(0, ny)
                try:
                    plt.draw()
                    plt.gcf().canvas.draw()
                    plt.pause(.002)
                except:
                    pass
                    # Don't close window here, Tk + ipython --pylab crashes on this (somehow GUI update out of main loop).
                    # plt.close()
            crop_params = self.dsize, y0n, x0n, ds2r, self.rebinf
        else:
            crop_params = None

        if MPI is not None:
            self.dsize, y0n, x0n, ds2r, self.rebinf = self.mpic.bcast(crop_params, root=0)

        nb = len(raw_data)

        self.iobs = np.zeros((nb, self.dsize, self.dsize))
        for jj in range(nb):
            if self.params['dark_subtract']:
                img = (raw_data[jj] - dark * self.params['dark_subtract']) * flatfield
                self.dark = None
                self.raw_dark = None
            else:
                img = raw_data[jj] * flatfield

            if self.rebinf > 1:
                self.iobs[jj] = rebin(img[y0n - ds2r:y0n + ds2r, x0n - ds2r:x0n + ds2r], self.rebinf)
            else:
                self.iobs[jj] = img[y0n - ds2r:y0n + ds2r, x0n - ds2r:x0n + ds2r]
        self.iobs = np.maximum(self.iobs, 0).astype(np.float32)  # If some dark was incorrectly subtracted

        if mask is not None:
            self.mask = mask[y0n - ds2r:y0n + ds2r, x0n - ds2r:x0n + ds2r]
            if self.rebinf > 1:
                self.mask = rebin(self.mask, self.rebinf)

            # Set masked pixels to 0
            self.iobs *= (self.mask == 0)

        if self.dark is not None:
            self.dark = dark[y0n - ds2r:y0n + ds2r, x0n - ds2r:x0n + ds2r]
            if self.rebinf > 1:
                self.dark = rebin(self.dark, self.rebinf)

        self.params['roi_actual'] = [x0n - ds2r, x0n + ds2r, y0n - ds2r, y0n + ds2r]
        self.print('Final iobs data size after cropping / centering / rebinning:', self.iobs.shape)

        if self.params['algorithm'] != 'manual' and self.params['orientation_round_robin'] is False:
            # Free memory
            self.raw_data = None

    def prepare(self):
        """
        Prepare object and probe.

        Returns: nothing. Adds self.obj0 and self.probe0

        """
        z = self.params['detectordistance']
        pixelsize = self.params['pixelsize'] * self.rebinf
        self.wavelength = 12.3984 / self.params['nrj'] * 1e-10
        if self.wavelength > 1e-5:
            raise PtychoRunnerException("Wavelength is larger than 10 micron. Is the energy correct ?")

        if self.params['near_field']:
            pix_size_direct = pixelsize
        else:
            angle_rad_per_pixel = pixelsize / z
            # pix size in reciprocal space
            pix_size_reciprocal = 1 / self.wavelength * angle_rad_per_pixel

            # pix size in direct space
            pix_size_direct = 1. / pix_size_reciprocal / self.dsize

        # scan positions in pixels, relative to center. X and Y from scan data, in meters
        xpix = (self.x - self.x.mean()) // pix_size_direct
        ypix = (self.y - self.y.mean()) // pix_size_direct

        self.print("E=%6.3fkeV, zdetector=%6.3fm, pixel size=%6.2fum, pixel size(object)=%6.1fnm"
                   % (self.params['nrj'], z, pixelsize * 1e6, pix_size_direct * 1e9))

        # Compute the size of the reconstructed object (obj)
        self.ny, self.nx = shape.calc_obj_shape(xpix, ypix, (self.dsize, self.dsize),
                                                margin=self.params['obj_margin'])

        if max(self.ny, self.nx) > self.params['obj_max_pix']:
            raise PtychoRunnerException("Calculated object size is to large: (%d, %d) pixels > %d!!"
                                        "Are scan positions in meters and not microns ?"
                                        "If not, you can override this limit with obj_max_pix=..." %
                                        (self.ny, self.nx, self.params['obj_max_pix']))

        self.probe_init_info = None
        oldpixelsize = None  # only needed if we load an old object or probe
        if self.params['load'] is None:
            # Initial object parameters
            s = self.params['object'].split(',')
            if s[0].lower() == 'random':
                a0, a1, p0, p1 = float(s[1]), float(s[2]), float(s[3]), float(s[4])
                self.obj_init_info = {'type': 'random', 'range': (a0, a1, p0, p1), 'shape': (self.ny, self.nx)}
                self.print("Using random object type with amplitude range: %5.2f-%5.2f and phase range: %5.2f-%5.2f" % (
                    a0, a1, p0, p1))
            else:
                raise PtychoRunnerException("Could not understand starting object type: %s", self.params['object'])
            self.data_info = {'wavelength': self.wavelength, 'detector_distance': z, 'detector_pixel_size': pixelsize}

            if self.params['loadprobe'] is None:
                if not self.params['near_field']:
                    # Initial probe
                    s = self.params['probe'].split(',')
                    if s[0] == 'focus':
                        s6 = s[1].split('x')
                        if len(s6) == 1:
                            s6r = float(s6[0])
                            self.probe_init_info = {'type': 'focus', 'aperture': (s6r,), 'focal_length': float(s[2]),
                                                    'shape': (self.dsize, self.dsize)}
                        else:
                            s6h, s6v = float(s6[0]), float(s6[1])
                            self.probe_init_info = {'type': 'focus', 'aperture': (s6h, s6v),
                                                    'focal_length': float(s[2]),
                                                    'shape': (self.dsize, self.dsize)}
                    elif s[0] == 'disc':
                        s6 = float(s[1])
                        self.probe_init_info = {'type': 'disc', 'radius_pix': (s6 / 2 / pix_size_direct),
                                                'shape': (self.dsize, self.dsize)}
                    elif s[0] == 'gaussian' or s[0] == 'gauss':
                        s6 = s[1].split('x')
                        s6h, s6v = float(s6[0]) / (pix_size_direct * 2.3548), float(s6[1]) / (
                                pix_size_direct * 2.3548)
                        self.probe_init_info = {'type': 'gauss', 'sigma_pix': (s6h, s6v),
                                                'shape': (self.dsize, self.dsize)}
                    else:
                        # Focused rectangular opening, without initial 'focus' keyword (DEPRECATED)
                        s6 = s[0].split('x')
                        s6h, s6v = float(s6[0]), float(s6[1])
                        self.probe_init_info = {'type': 'focus', 'aperture': (s6h, s6v), 'focal_length': float(s[1]),
                                                'shape': (self.dsize, self.dsize)}
            else:
                self.params['probe'] = None
                ext = os.path.splitext(self.params['loadprobe'])[-1]
                if ext == '.npz':
                    tmp = np.load(self.params['loadprobe'])
                    self.probe0 = tmp['probe']
                    if self.params['loadpixelsize'] is not None:
                        oldpixelsize = self.params['loadpixelsize']
                    elif 'pixelsize' in tmp.keys():
                        # TODO: take into account separate x and y pixel size
                        if np.isscalar(tmp['pixelsize']):
                            oldpixelsize = float(tmp['pixelsize'])
                        else:
                            oldpixelsize = tmp['pixelsize'].mean()
                    else:
                        oldpixelsize = pix_size_direct * self.dsize / self.probe0.shape[-1]
                else:
                    h = h5py.File(self.params['loadprobe'], 'r')
                    # Find last entry in file
                    i = 1
                    while True:
                        if 'entry_%d' % i not in h:
                            break
                        i += 1
                    entry = h['entry_%d' % (i - 1)]
                    self.probe0 = entry['probe/data'][()]
                    # TODO: Is this a kludge due to the way the probe is saved in the CXI file, or OK ?
                    self.probe0 = np.flip(self.probe0, axis=-2)  # origin at top, left corner
                    if self.params['loadpixelsize'] is not None:
                        oldpixelsize = self.params['loadpixelsize']
                    else:
                        oldpixelsize = (entry['probe/x_pixel_size'][()] + entry['probe/y_pixel_size'][()]) / 2
        else:
            # TODO: also import background if available
            self.params['loadprobe'] = None
            self.params['probe'] = None
            ext = os.path.splitext(self.params['load'])[-1]
            if ext == '.npz':
                self.objprobe = np.load(self.params['load'])
                self.obj0 = self.objprobe['obj']
                self.probe0 = self.objprobe['probe']
                if self.params['loadpixelsize'] is not None:
                    oldpixelsize = self.params['loadpixelsize']
                elif 'pixelsize' in self.objprobe.keys():
                    # TODO: take into account separate x and y pixel size
                    if np.isscalar(self.objprobe['pixelsize']):
                        oldpixelsize = float(self.objprobe['pixelsize'])
                    else:
                        oldpixelsize = self.objprobe['pixelsize'].mean()
                else:
                    oldpixelsize = pix_size_direct * self.dsize / self.probe0.shape[-1]
            else:
                h = h5py.File(self.params['load'], 'r')
                # Find last entry in file
                i = 1
                while True:
                    if 'entry_%d' % i not in h:
                        break
                    i += 1
                entry = h['entry_%d' % (i - 1)]
                self.probe0 = entry['probe/data'][()]
                self.obj0 = entry['object/data'][()]
                # TODO: Is this a kludge due to the way the probe is saved in the CXI file, or OK ?
                self.probe0 = np.flip(self.probe0, axis=-2)  # origin at top, left corner
                self.obj0 = np.flip(self.obj0, axis=-2)  # origin at top, left corner
                if self.params['loadpixelsize'] is not None:
                    oldpixelsize = self.params['loadpixelsize']
                else:
                    oldpixelsize = (entry['probe/x_pixel_size'][()] + entry['probe/y_pixel_size'][()]) / 2

        if self.probe0 is not None and not self.params['near_field']:
            # TODO: clean up this , separate scaling & reshaping probe ?
            nold = self.probe0.shape[-1]
            if oldpixelsize is not None or not np.isclose(nold, self.dsize):
                if oldpixelsize is None:
                    oldpixelsize = pix_size_direct
                # We loaded a probe and/or object, need to scale ?
                if not np.isclose(oldpixelsize, pix_size_direct, 1e-3, 0) or not np.isclose(nold, self.dsize):
                    scale = oldpixelsize / pix_size_direct
                    self.print("Probe: rescaling (x%5.3f) and reshaping (%d->%d pixels)" % (scale, nold, self.dsize))
                    # resize probe
                    nz = 1
                    if self.probe0.ndim == 3:
                        nz = self.probe0.shape[0]
                    oldprobe = self.probe0.reshape((nz, nold, nold))
                    oldprobe = zoom(oldprobe.real, (1, scale, scale)) + 1j * zoom(oldprobe.imag, (1, scale, scale))
                    self.probe0 = np.zeros((nz, self.dsize, self.dsize), dtype=np.complex64)
                    nold = oldprobe.shape[-1]
                    if nold % 2:
                        oldprobe = oldprobe[:, :-1, :-1]
                        nold = oldprobe.shape[-1]
                    if nold < self.dsize:
                        self.probe0[:, self.dsize // 2 - nold // 2: self.dsize // 2 + nold // 2,
                        self.dsize // 2 - nold // 2:self.dsize // 2 + nold // 2] = oldprobe
                    else:
                        self.probe0 = oldprobe[:, nold // 2 - self.dsize // 2: nold // 2 + self.dsize // 2,
                                      nold // 2 - self.dsize // 2:nold // 2 + self.dsize // 2]

                    if nz == 1:
                        self.probe0 = self.probe0.reshape((self.dsize, self.dsize))

                    if self.params['load'] is not None:
                        # Resize object as well
                        oldobj = self.obj0
                        nzo = 1
                        if oldobj.ndim == 3:
                            nzo = oldobj.shape[0]
                        nyo, nxo = oldobj.shape[-2:]
                        oldobj = oldobj.reshape((nzo, nyo, nxo))
                        self.obj0 = np.zeros((nzo, self.ny, self.nx), dtype=np.complex64)
                        oldobj = zoom(oldobj.real, (1, scale, scale)) + 1j * zoom(oldobj.imag, (1, scale, scale))
                        nyo, nxo = oldobj.shape[-2:]
                        if nyo % 2 == 1:
                            if oldobj.ndim == 2:
                                oldobj = oldobj[:-1]
                            else:
                                oldobj = oldobj[:, :-1]
                            nyo -= 1
                        if nxo % 2 == 1:
                            if oldobj.ndim == 2:
                                oldobj = oldobj[:, :-1]
                            else:
                                oldobj = oldobj[:, :, :-1]
                            nxo -= 1
                        if (nyo + nxo) < (self.ny + self.nx):
                            self.obj0[:, self.ny // 2 - nyo // 2: self.ny // 2 + nyo // 2,
                            self.nx // 2 - nxo // 2:self.nx // 2 + nxo // 2] = oldobj
                        else:
                            self.obj0 = oldobj[:, nyo // 2 - self.ny // 2: nyo // 2 + self.ny // 2,
                                        nxo // 2 - self.nx // 2:nxo // 2 + self.nx // 2]

                        if nzo == 1:
                            self.obj0 = self.obj0.reshape((self.ny, self.nx))

    def init_run_number(self):
        """
        Determine the current run number, depending on existing files
        :return: run0
        """
        run0 = 0
        if self.mpi_master and self.params['saveprefix'].lower() != 'none':
            # Create directory to save files
            path = os.path.split(self.params['saveprefix'])[0]
            if '%' in path:
                path = path % self.scan
            if len(path):
                os.makedirs(path, exist_ok=True)
            if self.params['run0'] is None:
                # Look for existing saved files
                run0 = 1
                while True:
                    testfile1 = self.params['saveprefix'] % (self.scan, run0) + ".npz"
                    testfile2 = self.params['saveprefix'] % (self.scan, run0) + "-00.npz"
                    testfile3 = self.params['saveprefix'] % (self.scan, run0) + ".cxi"
                    if os.path.isfile(testfile1) or os.path.isfile(testfile2) or os.path.isfile(testfile3):
                        run0 += 1
                    else:
                        break
            else:
                run0 = self.params['run0']
        return run0

    def run(self, reuse_ptycho=False):
        """
        Main work function, will run according to the set of algorithms specified in self.params.

        :param reuse_ptycho: if True, will reuse the previous Ptycho and PtychoData objects
            and skip some initialisation steps

        :return:
        """
        run0 = self.init_run_number()
        if MPI is not None:
            run0 = self.mpic.bcast(run0, root=0)
        # Enable profiling ?
        if self.params['profiling']:
            self.processing_unit.enable_profiling(True)

        for run in range(run0, run0 + self.params['nbrun']):
            t0 = timeit.default_timer()
            self._run = run
            if self.mpi_master:
                self.print("\n", "#" * 100, "\n#", "\n# Scan: %3d Run: %g\n#\n" % (self.scan, run), "#" * 100)

            # self.print("PtychoRunner.run(): #%d, pos_c = (%7.2f, %7.2f)" % (self.mpi_rank,
            #                                                            self.x.mean() * 1e6, self.y.mean() * 1e6))
            self.data = PtychoData(iobs=self.iobs, positions=(self.x, self.y),
                                   detector_distance=self.params['detectordistance'],
                                   mask=self.mask, pixel_size_detector=self.params['pixelsize'],
                                   wavelength=self.wavelength, vidx=self.imgn, near_field=self.params['near_field'],
                                   padding=self.params['padding'])

            if not reuse_ptycho:
                # Init object and probe according to parameters
                if self.params['load'] is None:
                    init = simulation.Simulation(obj_info=self.obj_init_info, probe_info=self.probe_init_info,
                                                 data_info=self.data_info, verbose=self.mpi_master)
                    init.make_obj()
                    init.make_obj_true(self.data.get_required_obj_shape(margin=self.params['obj_margin']))
                    self.obj0 = init.obj.values
                    self.print("Making obj:", self.obj0.shape, self.ny, self.nx)

                    if self.params['loadprobe'] is None:
                        if self.params['near_field']:
                            self.probe0 = np.ones(self.iobs[0].shape, dtype=np.complex64)
                        else:
                            init.make_probe()
                            self.probe0 = init.probe.values

                if self.params['defocus'] is not None and self.params['defocus'] != 0 and self.defocus_done is False:
                    self.defocus_done = True  # Don't defocus again for multiple runs of the same scan
                    if len(self.probe0.shape) == 2:
                        self.w = wavefront.Wavefront(d=fftshift(self.probe0.astype(np.complex64)),
                                                     wavelength=self.wavelength,
                                                     pixel_size=self.data.pixel_size_object()[0])
                        self.w = PropagateNearField_Wavefront(self.params['defocus']) * self.w

                        self.probe0 = self.w.get(shift=True)
                    else:
                        # Propagate all modes
                        for i in range(len(self.probe0)):
                            self.w = wavefront.Wavefront(d=fftshift(self.probe0[i].astype(np.complex64)),
                                                         wavelength=self.wavelength,
                                                         pixel_size=self.data.pixel_size_object()[0])
                            self.w = PropagateNearField_Wavefront(self.params['defocus']) * self.w
                            self.probe0[i] = self.w.get(shift=True)

                t1 = timeit.default_timer()
                if self.timings is not None:
                    dt = t1 - t0
                    if "scan_run_prepare_obj_probe" in self.timings:
                        self.timings["scan_run_prepare_obj_probe"] += dt
                    else:
                        self.timings["scan_run_prepare_obj_probe"] = dt

                if self.params['rotate'] is not None:
                    # Rotate probe
                    self.print("ROTATING probe by %6.2f degrees" % (self.params['rotate']))
                    re, im = self.probe0.real, self.probe0.imag
                    self.probe0 = rotate(re, self.params['rotate'], reshape=False,
                                         axes=(-2, -1)) + 1j * rotate(im, self.params['rotate'], reshape=False,
                                                                      axes=(-2, -1))

                # mpi_neighbour_xy is only initialised if MPI size is > 0
                if MPI is not None and 'split' in self.params['mpi'] and self.mpi_neighbour_xy is not None:
                    self.p = PtychoSplit(probe=self.probe0, obj=self.obj0, data=self.data, background=self.dark,
                                         mpi_neighbour_xy=self.mpi_neighbour_xy)
                else:
                    self.p = Ptycho(probe=self.probe0, obj=self.obj0, data=self.data, background=self.dark)
                self.p = ScaleObjProbe(verbose=True) * self.p
            else:
                self.p = FreePU() * self.p
                self.p.data = self.data

            # Staring point of the history & timing of the algorithm
            self.p.reset_history()

            # TODO: also handle memory error here (start of GPU allocation)

            if self.timings is not None:
                dt = timeit.default_timer() - t0
                if "scan_run_prepare" in self.timings:
                    self.timings["scan_run_prepare"] += dt
                else:
                    self.timings["scan_run_prepare"] = dt

            self._algo_s = ""
            self._stepnum = 0
            if self.params['algorithm'].lower() == 'manual':
                return
            algo = self.params['algorithm']
            self.run_algorithm(algo)

        if self.params['movie'] is not None:
            self.save_movie()

    def run_algorithm(self, algo_string):
        """
        Run a single or suite of algorithms in a given run.

        :param algo_string: a single or suite of algorithm steps to use, e.g. 'ML**100' or
                           'analysis,ML**100,DM**100,nbprobe=3,DM**100'
                           or 'analysis,ML**100*DM**100,nbprobe=3,DM**100'
        :return: Nothing
        """
        use_old_algo_string = False
        if '*' not in algo_string:
            for s in ['0ap', 'ap0', '0dm', 'dm0', '0ml', 'ml0']:
                if s in algo_string.lower():
                    use_old_algo_string = True
                    break

        if use_old_algo_string:
            self.print("\n", "#" * 100, "\n#",
                       "\n# WARNING: You are using the old algorithm strings, which are DEPRECATED\n#"
                       "\n#      5s sleep - remember to read the updated command-line help !\n"
                       "# If you were writing: algorithm=probe=1,nbprobe=3,100DM,40ML,analysis\n"
                       "# You should now use:  algorithm=analysis,ML**40,DM**100,nbprobe=3,probe=1 (order right-to-left !)\n"
                       "# Or alternatively:    algorithm=analysis,ML**40*DM**100,nbprobe=3,probe=1\n"
                       + "#" * 100)
            time.sleep(5)
            for algo in algo_string.split(','):
                if self._algo_s == "":
                    self._algo_s += algo
                else:
                    self._algo_s += ',' + algo
                self.print("\n", "#" * 100, "\n#", "\n# Scan: %3d Run: %g , Algorithm: %s\n#\n" %
                           (self.scan, self._run, algo), "#" * 100)
                realoptim = False  # Is this a real optimization (ptycho or ML), or just a change of parameter ?
                show_obj_probe = self.params['liveplot']
                if show_obj_probe:
                    show_obj_probe = self.params['verbose']
                if algo.lower().find('ap') >= 0:
                    realoptim = True
                    s = algo.lower().split('ap')
                    if len(s[0].strip()) >= 1:
                        nbcycle = int(s[0])
                    elif len(s[1].strip()):
                        nbcycle = int(s[1])
                    self.p = AP(update_object=self.update_object, update_probe=self.update_probe,
                                update_background=self.update_background,
                                show_obj_probe=show_obj_probe, update_pos=self.update_position,
                                calc_llk=self.params['verbose'], fig_num=100) ** nbcycle * self.p
                elif algo.lower().find('dm') >= 0:
                    realoptim = True
                    s = algo.lower().split('dm')
                    if len(s[0].strip()) >= 1:
                        nbcycle = int(s[0])
                    elif len(s[1].strip()):
                        nbcycle = int(s[1])
                    self.p = DM(update_object=self.update_object, update_probe=self.update_probe,
                                show_obj_probe=show_obj_probe, calc_llk=self.params['verbose'], fig_num=100,
                                center_probe_max_shift=self.params['center_probe_max_shift'],
                                center_probe_n=self.params['center_probe_n'],
                                loop_obj_probe=self.params['dm_loop_obj_probe']) ** nbcycle * self.p
                elif algo.lower().find('ml') >= 0:
                    realoptim = True
                    s = algo.lower().split('ml')
                    if len(s[0].strip()) >= 1:
                        nbcycle = int(s[0])
                    elif len(s[1].strip()):
                        nbcycle = int(s[1])
                    self.p = ML(update_object=self.update_object, update_probe=self.update_probe,
                                update_pos=self.update_position,
                                show_obj_probe=show_obj_probe, reg_fac_obj=self.params['regularization'],
                                calc_llk=self.params['verbose'], fig_num=100) ** nbcycle * self.p
                elif algo.lower().find('ortho') >= 0:
                    self.p = OrthoProbe(verbose=True) * self.p
                elif algo.lower().find('nbprobe=') >= 0:
                    nb_probe = int(algo.lower().split('nbprobe=')[-1])

                    pr = self.p.get_probe()
                    nz, ny, nx = pr.shape
                    if nb_probe == nz:
                        continue

                    pr1 = np.empty((nb_probe, ny, nx), dtype=np.complex64)
                    for i in range(min(nz, nb_probe)):
                        pr1[i] = pr[i]
                    for i in range(nz, nb_probe):
                        n = abs(pr[0]) * np.random.uniform(0, 0.04, (ny, nx))
                        pr1[i] = n * np.exp(1j * np.random.uniform(0, 2 * np.pi, (ny, nx)))

                    self.p.set_probe(pr1)

                elif algo.lower().find('object=') >= 0:
                    self.update_object = int(algo.lower().split('object=')[-1])
                elif algo.lower().find('probe=') >= 0:
                    self.update_probe = int(algo.lower().split('probe=')[-1])
                elif algo.lower().find('background=') >= 0:
                    self.update_background = int(algo.lower().split('background=')[-1])
                elif algo.lower().find('position=') >= 0:
                    self.update_position = int(algo.lower().split('position=')[-1])
                elif algo.lower().find('positions=') >= 0:
                    self.update_position = int(algo.lower().split('positions=')[-1])
                elif algo.lower().find('regularization=') >= 0:
                    self.params['regularization'] = float(algo.lower().split('regularization=')[-1])
                elif algo.lower().find('analyze') >= 0 or algo.lower().find('analysis') >= 0:
                    probe = self.p.get_probe()
                    if self.params['saveplot'] and self.params['saveprefix'].lower() != 'none':
                        if self._stepnum > 0:
                            steps = "-%02d" % (self._stepnum - 1)
                        else:
                            # Need when self.params['save'] is not 'all'
                            steps = ""
                        save_prefix = self.params['saveprefix'] % (self.scan, self._run) + steps
                        self.p = AnalyseProbe(modes=True, focus=True, verbose=True,
                                              save_prefix=save_prefix, show_plot=False) * self.p
                        self.print("Plot positions ?")
                        self.p = PlotPositions(verbose=True, save_prefix=save_prefix, show_plot=False) * self.p
                        if os.name == 'posix' and self.mpi_master:
                            try:
                                if probe.shape[0] > 1:
                                    sf = os.path.split(save_prefix + '-probe-modes.png')
                                    os.system(
                                        'ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest-probe-modes.png')))
                                sf = os.path.split(save_prefix + '-probe-z.png')
                                os.system('ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest-probe-z.png')))
                                sf = save_prefix + '-positions.png'
                                if os.path.isfile(sf):
                                    sf = os.path.split(sf)
                                    os.system('ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest-positions.png')))
                            except:
                                pass
                else:
                    self.print("ERROR: did not understand algorithm step:", algo)

                if realoptim and self.params['autocenter'] and not self.params['near_field']:
                    pr = self.p.get_probe()
                    obj = self.p.get_obj()
                    pr, obj = center_array_2d(pr, other_arrays=obj, iz=0)
                    self.p.set_obj(obj)
                    self.p.set_probe(pr)

                if self.params['save'] == 'all' and realoptim:
                    self.save(self._run, self._stepnum, self._algo_s)
                    self._stepnum += 1
        else:
            # Using new operator-based algorithm
            algo_split = algo_string.split(',')
            algo_split.reverse()
            t0 = timeit.default_timer()
            for algo in algo_split:
                t0algo = timeit.default_timer()
                if self._algo_s == "":
                    self._algo_s = algo + self._algo_s
                else:
                    self._algo_s = algo + ',' + self._algo_s
                self.print("\n", "#" * 100, "\n#", "\n# Scan: %3d Run: %g , Algorithm: %s\n#\n" %
                           (self.scan, self._run, algo), "#" * 100)
                realoptim = False  # Is this a real optimization (ptycho or ML), or just a change of parameter ?

                if algo.lower().find('ortho') >= 0:
                    self.p = OrthoProbe(verbose=True) * self.p
                elif algo.lower().find('nbprobe=') >= 0:
                    nb_probe = int(algo.lower().split('nbprobe=')[-1])

                    pr = self.p.get_probe()
                    nz, ny, nx = pr.shape
                    if nb_probe == nz:
                        continue

                    pr1 = np.empty((nb_probe, ny, nx), dtype=np.complex64)
                    for i in range(min(nz, nb_probe)):
                        pr1[i] = pr[i]
                    for i in range(nz, nb_probe):
                        n = abs(pr[0]) * np.random.uniform(0, 0.04, (ny, nx))
                        pr1[i] = n * np.exp(1j * np.random.uniform(0, 2 * np.pi, (ny, nx)))

                    self.p.set_probe(pr1)
                elif algo.lower().find('nbobject=') >= 0 or algo.lower().find('nbobj=') >= 0:
                    nb_obj = int(algo.lower().split('=')[-1])

                    o = self.p.get_obj()
                    nz, ny, nx = o.shape
                    if nb_obj == nz:
                        continue

                    o1 = np.empty((nb_obj, ny, nx), dtype=np.complex64)
                    for i in range(min(nz, nb_obj)):
                        o1[i] = o[i]
                    for i in range(nz, nb_obj):
                        n = abs(o[0]) * np.random.uniform(0, 0.04, (ny, nx))
                        o1[i] = n * np.exp(1j * np.random.uniform(0, 2 * np.pi, (ny, nx)))

                    self.p.set_obj(o1)
                elif algo.lower().find('padding') >= 0 and algo.lower().find('interp') >= 0:
                    self.p = PaddingInterp() * self.p

                elif algo.lower().find('object=') >= 0:
                    self.update_object = bool(int(algo.lower().split('object=')[-1]))
                elif algo.lower().find('probe=') >= 0:
                    self.update_probe = bool(int(algo.lower().split('probe=')[-1]))
                elif algo.lower().find('background=') >= 0:
                    ubg0 = self.update_background
                    self.update_background = bool(int(algo.lower().split('background=')[-1]))
                    if self.update_background and not ubg0:
                        # make sure background is >0 for background update to work
                        b = self.p.get_background()
                        b += 1e-2 * (b < 1e-4)
                        self.p.set_background(b)
                elif algo.lower().find('background_smooth=') >= 0:
                    self.params['background_smooth'] = float(algo.lower().split('background_smooth=')[-1])
                elif algo.lower().find('position=') >= 0:
                    self.update_position = int(algo.lower().split('position=')[-1])
                elif algo.lower().find('positions=') >= 0:
                    self.update_position = int(algo.lower().split('positions=')[-1])
                elif algo.lower().find('pos_mult=') >= 0:
                    self.params['pos_mult'] = float(algo.lower().split('pos_mult=')[-1])
                elif algo.lower().find('pos_max_shift=') >= 0:
                    self.params['pos_max_shift'] = float(algo.lower().split('pos_max_shift=')[-1])
                elif algo.lower().find('pos_min_shift=') >= 0:
                    self.params['pos_min_shift'] = float(algo.lower().split('pos_min_shift=')[-1])
                elif algo.lower().find('pos_threshold=') >= 0:
                    self.params['pos_threshold'] = float(algo.lower().split('pos_threshold=')[-1])
                elif algo.lower().find('floating_intensity=') >= 0:
                    self.floating_intensity = bool(int(algo.lower().split('floating_intensity=')[-1]))
                elif algo.lower().find('regularization=') >= 0:
                    self.params['regularization'] = float(algo.lower().split('regularization=')[-1])
                elif algo.lower().find('obj_smooth=') >= 0:
                    self.params['obj_smooth'] = float(algo.lower().split('obj_smooth=')[-1])
                elif algo.lower().find('obj_inertia=') >= 0:
                    self.params['obj_inertia'] = float(algo.lower().split('obj_inertia=')[-1])
                elif algo.lower().find('probe_smooth=') >= 0:
                    self.params['probe_smooth'] = float(algo.lower().split('probe_smooth=')[-1])
                elif algo.lower().find('probe_inertia=') >= 0:
                    self.params['probe_inertia'] = float(algo.lower().split('probe_inertia=')[-1])
                elif algo.lower().find('dm_loop_obj_probe=') >= 0:
                    self.params['dm_loop_obj_probe'] = int(algo.lower().split('dm_loop_obj_probe=')[-1])
                elif algo.lower().find('analyze') >= 0 or algo.lower().find('analysis') >= 0:
                    t1 = timeit.default_timer()
                    if self.params['saveplot'] and self.params['saveprefix'].lower() != 'none':
                        probe = self.p.get_probe()
                        if self._stepnum > 0:
                            steps = "-%02d" % (self._stepnum - 1)
                        else:
                            # Need when self.params['save'] is not 'all'
                            steps = ""
                        save_prefix = self.params['saveprefix'] % (self.scan, self._run) + steps
                        self.p = AnalyseProbe(modes=True, focus=not self.params['near_field'], verbose=True,
                                              save_prefix=save_prefix, show_plot=False) * self.p
                        self.print("Plot positions ?")
                        self.p = PlotPositions(verbose=True, save_prefix=save_prefix, show_plot=False) * self.p
                        if os.name == 'posix' and self.mpi_master:
                            try:
                                if probe.shape[0] > 1:
                                    sf = os.path.split(save_prefix + '-probe-modes.png')
                                    os.system(
                                        'ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest-probe-modes.png')))
                                sf = os.path.split(save_prefix + '-probe-z.png')
                                os.system('ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest-probe-z.png')))
                                sf = save_prefix + '-positions.png'
                                if os.path.isfile(sf):
                                    sf = os.path.split(sf)
                                    os.system('ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest-positions.png')))
                            except:
                                pass
                    if self.timings is not None:
                        dt = timeit.default_timer() - t1
                        if "analysis" in self.timings:
                            self.timings["analysis"] += dt
                        else:
                            self.timings["analysis"] = dt

                elif algo.lower().find('verbose=') >= 0:
                    self.params['verbose'] = int(algo.lower().split('verbose=')[-1])
                elif algo.lower().find('live_plot=') >= 0:
                    self.params['liveplot'] = int(algo.lower().split('live_plot=')[-1])
                elif algo.lower().find('fig_num=') >= 0:
                    self.params['fig_num'] = int(algo.lower().split('fig_num=')[-1])
                else:
                    # This should be an operator string to apply
                    realoptim = True
                    show_obj_probe = self.params['liveplot']
                    if show_obj_probe:
                        show_obj_probe = self.params['verbose']
                    fig_num = self.params['fig_num']

                    # First create basic operators
                    ap = AP(update_object=self.update_object, update_probe=self.update_probe,
                            update_background=self.update_background, floating_intensity=self.floating_intensity,
                            show_obj_probe=show_obj_probe, update_pos=self.update_position,
                            calc_llk=self.params['verbose'], fig_num=fig_num,
                            obj_smooth_sigma=self.params['obj_smooth'], obj_inertia=self.params['obj_inertia'],
                            probe_smooth_sigma=self.params['probe_smooth'], probe_inertia=self.params['probe_inertia'],
                            pos_mult=self.params['pos_mult'], pos_max_shift=self.params['pos_max_shift'],
                            pos_min_shift=self.params['pos_min_shift'], pos_threshold=self.params['pos_threshold'],
                            background_smooth_sigma=self.params['background_smooth'])
                    dm = DM(update_object=self.update_object, update_probe=self.update_probe,
                            update_background=self.update_background,
                            show_obj_probe=show_obj_probe, update_pos=self.update_position,
                            calc_llk=self.params['verbose'], fig_num=fig_num,
                            obj_smooth_sigma=self.params['obj_smooth'], obj_inertia=self.params['obj_inertia'],
                            probe_smooth_sigma=self.params['probe_smooth'], probe_inertia=self.params['probe_inertia'],
                            center_probe_max_shift=self.params['center_probe_max_shift'],
                            center_probe_n=self.params['center_probe_n'],
                            loop_obj_probe=self.params['dm_loop_obj_probe'],
                            pos_mult=self.params['pos_mult'], pos_max_shift=self.params['pos_max_shift'],
                            pos_min_shift=self.params['pos_min_shift'], pos_threshold=self.params['pos_threshold'],
                            background_smooth_sigma=self.params['background_smooth'])
                    ml = ML(update_object=self.update_object, update_probe=self.update_probe,
                            update_background=self.update_background,
                            floating_intensity=self.floating_intensity, update_pos=self.update_position,
                            show_obj_probe=show_obj_probe, reg_fac_obj=self.params['regularization'],
                            calc_llk=self.params['verbose'], fig_num=fig_num,
                            pos_mult=self.params['pos_mult'], pos_max_shift=self.params['pos_max_shift'],
                            pos_min_shift=self.params['pos_min_shift'], pos_threshold=self.params['pos_threshold'])

                    showobjprobe = ShowObjProbe(fig_num=fig_num)

                    try:
                        ops = eval(algo.lower())
                        self.p = ops * self.p
                    except Exception as ex:
                        # self.print(self.help_text)
                        self.print('\n\n Caught exception for scan %d: %s    \n' % (self.scan, str(ex)))
                        self.print(traceback.format_exc())
                        if has_cuda:
                            # Probably shouldn't do an import here...
                            from pycuda.driver import MemoryError
                            if isinstance(ex, MemoryError):
                                self.print("A GPU memory error was encountered")
                                self.p = MemUsage(verbose=True) * self.p
                                sys.exit(1)
                        elif has_opencl:
                            from pyopencl import MemoryError
                            if isinstance(ex, MemoryError):
                                self.print("A GPU memory error was encountered")
                                self.p = MemUsage(verbose=True) * self.p
                                sys.exit(1)

                        self.print('Could not interpret operator-based algorithm (see above error): ', algo)
                        # TODO: print valid examples of algorithms
                        sys.exit(1)

                if realoptim and self.params['autocenter'] and not self.params['near_field']:
                    pr = self.p.get_probe()
                    obj = self.p.get_obj()
                    pr, obj = center_array_2d(pr, other_arrays=obj, iz=0)
                    self.p.set_obj(obj)
                    self.p.set_probe(pr)

                if realoptim and self.timings is not None:
                    dt = timeit.default_timer() - t0algo
                    if "algorithm" in self.timings:
                        self.timings["algorithm"] += dt
                    else:
                        self.timings["algorithm"] = dt

                if self.params['save'] == 'all' and realoptim:
                    self.save(self._run, self._stepnum, self._algo_s)
                    self._stepnum += 1
            if 'split' in self.params['mpi']:
                t1 = timeit.default_timer()
                self.p.stitch(verbose=False)
                if self.timings is not None:
                    dt = timeit.default_timer() - t1
                    if "mpi_stitch_end" in self.timings:
                        self.timings["mpi_stitch_end"] += dt
                    else:
                        self.timings["mpi_stitch_end"] = dt

            if self.mpi_master:
                dt = timeit.default_timer() - t0
                self.print("\nTotal elapsed time for algorithms: %8.2fs " % dt)
                calc_throughput(self.p, verbose=True)

        if self.params['save'] != 'all' and self.params['algorithm'].lower() not in ['analyze', 'analysis', 'manual']:
            self.save(self._run, algostring=self._algo_s)

        # Free GPU memory
        self.p = FreePU() * self.p

        if self.params['profiling'] and 'cl_event_profiling' in dir(self.processing_unit):
            # Profiling can only work with OpenCL
            self.print("\n", "#" * 100, "\n#", "\n#         Profiling info\n#\n", "#" * 100)
            dt = 0
            vv = []
            for s in self.processing_unit.cl_event_profiling:
                v = np.array([(e.event.profile.end - e.event.profile.start) for e in
                              self.processing_unit.cl_event_profiling[s]])
                gf = np.array([e.gflops() for e in self.processing_unit.cl_event_profiling[s]])
                gb = np.array([e.gbs() for e in self.processing_unit.cl_event_profiling[s]])
                vv.append((s, v.mean() * 1e-3, len(v), v.sum() * 1e-6, gf.mean(), gb.mean()))
                dt += v.sum() * 1e-6
            vv.sort(key=lambda x: x[3], reverse=True)
            for v in vv:
                self.print("dt(%25s)=%9.3f µs , nb=%6d, dt_sum=%10.3f ms [%4.1f%%], %8.3f Gflop/s, %8.3f Gb/s"
                           % (v[0], v[1], v[2], v[3], v[3] / dt * 100, v[4], v[5]))
            self.print("                                                    Total: dt=%11.3f ms" % (dt))
        self.print_probe_fwhm()

    def print_probe_fwhm(self):
        """
        Analyse probe shape and print estimated FWHM. Ignored for near field ptycho.

        Returns:
            Nothing
        """
        if self.params['near_field']:
            return
        if self.mpi_master:
            self.print("\n", "#" * 100, "\n")
            self.print("Probe statistics at sample position:")
            analysis.probe_fwhm(self.p.get_probe(), self.data.pixel_size_object()[0])

    def save_data_cxi(self, crop=True, verbose=False, **kwargs):
        """
        Save the scan data using the CXI format (see http://cxidb.org)
        Args:
            crop: if True, only the already-cropped data will be save. Otherwise, the original raw data is saved.
        Returns:

        """
        path = os.path.dirname(self.params['saveprefix'] % (self.scan, 0))
        os.makedirs(path, exist_ok=True)
        file_name = os.path.join(path, "data.cxi")

        wavelength = 12.3984e-10 / self.params['nrj']

        if (self.iobs is not None) and crop:
            iobs = self.iobs
        else:
            iobs = self.raw_data

        if crop:
            pixel_size = self.params['pixelsize'] * self.rebinf
        else:
            pixel_size = self.params['pixelsize']

        mask = None
        if (self.mask is not None) and crop:
            if self.mask.sum() != 0:
                mask = self.mask
        elif self.raw_mask is not None:
            if self.raw_mask.sum() != 0:
                mask = self.raw_mask

        dark = None
        if (self.dark is not None) and crop:
            if self.dark.sum() != 0:
                dark = self.dark
        elif self.raw_dark is not None:
            if self.raw_dark.sum() != 0:
                dark = self.raw_dark

        detector_distance = self.params['detectordistance']

        save_ptycho_data_cxi(file_name, iobs, pixel_size, wavelength, detector_distance, x=self.x, y=self.y, z=None,
                             monitor=self.raw_data_monitor, mask=mask, dark=dark,
                             instrument=self.params['instrument'],
                             overwrite=False, scan=self.scan, params=self.params, verbose=verbose)

    def save(self, run, stepnum=None, algostring=None):
        """
        Save the result of the optimization, and (if  self.params['saveplot'] is True) the corresponding plot.
        This is an internal function.

        :param run:  the run number (integer)
        :param stepnum: the step number in the set of algorithm steps
        :param algostring: the string corresponding to all the algorithms ran so far, e.g. '100DM,100AP,100ML'
        :return:
        """
        t0 = timeit.default_timer()
        if self.params['saveprefix'].lower() == 'none':
            return
        if stepnum is None:
            steps = ""
        else:
            steps = "-%02d" % stepnum

        if 'npz' in self.params['output_format'].lower():
            filename = self.params['saveprefix'] % (self.scan, run) + steps + ".npz"
            self.print("\n", "#" * 100, "\n#",
                       "\n#         Saving object and probe to: %s\n#\n" % filename, "#" * 100)

            # Shift back the phase range to [0...], if object phase is not wrapped.
            # TODO: handle objects with multiple modes
            obj = phase.shift_phase_zero(self.p.get_obj()[0], percent=2, origin=0, mask=self.p.get_scan_area_obj())
            kwargs = {'obj': obj, 'probe': self.p.get_probe(), 'pixelsize': self.data.pixel_size_object(),
                      'scan_area_obj': self.p.get_scan_area_obj(), 'scan_area_probe': self.p.get_scan_area_probe()}

            if self.p.get_background() is not None:
                if self.p.get_background().sum() > 0:
                    kwargs['background'] = self.p.get_background()

            np.savez_compressed(filename, **kwargs)
            if os.name == 'posix':
                try:
                    sf = os.path.split(filename)
                    os.system('ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest.npz')))
                except:
                    pass
        else:
            # Save as CXI file
            filename = self.params['saveprefix'] % (self.scan, run) + ".cxi"
            self.print("\n", "#" * 100, "\n#",
                       "\n#         Saving object and probe to: %s\n#\n" % filename, "#" * 100)

            command = ""
            for arg in sys.argv:
                command += arg + " "
            process = {"command": command}

            if algostring is not None:
                process["algorithm"] = algostring

            params_string = ""
            for p in self.params.items():
                k, v = p
                if v is not None and k not in ['output_format']:
                    params_string += "%s = %s\n" % (k, str(v))

            process["parameters_all"] = params_string
            process["program"] = "PyNX"
            process["version"] = _pynx_version

            self.p.save_obj_probe_cxi(filename, sample_name=None, experiment_id=None,
                                      instrument=self.params['instrument'], note=None,
                                      process=process, append=True, params=self.params,
                                      remove_obj_phase_ramp=self.params['remove_obj_phase_ramp'])
            if os.name == 'posix' and (self.mpi_master or 'split' not in self.params['mpi']):
                try:
                    sf = os.path.split(filename)
                    os.system('ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest.cxi')))
                except:
                    pass
        if self.timings is not None:
            dt = timeit.default_timer() - t0
            if "saving" in self.timings:
                self.timings["saving"] += dt
            else:
                self.timings["saving"] = dt

        if self.params['saveplot']:
            self.save_plot(run, stepnum, algostring)

    def save_plot(self, run, stepnum=None, algostring=None, display_plot=False):
        """
        Save the plot to a png file.

        :param run:  the run number (integer)
        :param stepnum: the step number in the set of algorithm steps
        :param algostring: the string corresponding to all the algorithms ran so far, e.g. '100DM,100AP,100ML'
        :param display_plot: if True, the saved plot will also be displayed
        :return:
        """
        t0 = timeit.default_timer()
        if stepnum is None:
            steps = ""
        else:
            steps = "-%02d" % stepnum
        if algostring is None:
            algostring = self.params['algorithm']

        if 'split' in self.params['mpi']:
            self.p.stitch(sync=True)
            obj = self.p.mpi_obj
            scan_area_obj = self.p.get_mpi_scan_area_obj()
            scan_area_points = self.p.get_mpi_scan_area_points()
            if not self.mpi_master:
                return
        else:
            obj = self.p.get_obj()
            scan_area_obj = self.p.get_scan_area_obj()
            scan_area_points = self.p.get_scan_area_points()
        scan_area_probe = self.p.get_scan_area_probe()

        if self.p.data.near_field or not self.params['remove_obj_phase_ramp']:
            obj = obj[0]
            probe = self.p.get_probe()[0]
        else:
            obj = phase.minimize_grad_phase(obj[0], center_phase=0, global_min=False,
                                            mask=~scan_area_obj, rebin_f=2)[0]
            probe = phase.minimize_grad_phase(self.p.get_probe()[0], center_phase=0, global_min=False,
                                              mask=~scan_area_probe, rebin_f=2)[0]

        if display_plot:
            try:
                fig = plt.figure(101, figsize=(10, 6))
            except:
                # no GUI or $DISPLAY
                fig = Figure(figsize=(10, 6))
        else:
            fig = Figure(figsize=(10, 6))
        fig.clf()

        # ======================== Plot Object ===================================
        nyo, nxo = obj.shape[-2:]
        # Manually compute axes size to get correct aspect and allow for twinx & twiny. Use gridspec ?
        fx, fy = fig.get_size_inches()
        maxwx, maxwy = 0.37, 0.7
        if (nxo / (maxwx * fx)) > (nyo / (maxwy * fy)):
            tmp = maxwx * nyo / nxo * fx / fy / 2
            fig_scale_o = maxwx * fx / nxo
            ax = fig.add_axes((0.06, 0.5 - tmp, maxwx, 2 * tmp))
        else:
            tmp = maxwy * nxo / nyo * fy / fx / 2
            fig_scale_o = maxwy * fy / nyo
            ax = fig.add_axes((0.2 - tmp, 0.1, 2 * tmp, maxwy))

        # Coordinates centered on object
        if 'split' in self.params['mpi']:
            tmpx, tmpy = self.p.get_mpi_obj_coord()
        else:
            tmpx, tmpy = self.p.get_obj_coord()
        tmpx, tmpy = (tmpx.min() * 1e6, tmpx.max() * 1e6), (tmpy.min() * 1e6, tmpy.max() * 1e6)
        self.print(tmpx, tmpy)
        if self.params['saveplot'] == 'object_phase':
            # Show only object phase
            obja = np.angle(obj)
            smin, smax = np.percentile(np.ma.masked_array(obja, ~scan_area_obj).compressed(), (2, 98))
            if smax - smin < np.pi:
                cm_phase = plt.cm.get_cmap('gray')
            else:
                smin, smax = 0, 2 * np.pi
                cm_phase = plot_utils.cm_phase
            mp = ax.imshow(obja, vmax=smax, vmin=smin, extent=(tmpx[0], tmpx[1], tmpy[0], tmpy[1]),
                           cmap=cm_phase, origin='lower')
            # if smax - smin < np.pi:
            #    fig.colorbar(mp, ax=ax)
            ax.set_title("Object phase [%5.2f-%5.2f radians]" % (smin, smax), fontsize=9)
        else:
            # Show object as RGBA/HSV
            smin, smax = 0, np.ma.masked_array(abs(obj), ~scan_area_obj).max()
            # Need to use aspect='auto' to have two twin axes
            ax.imshow(plot_utils.complex2rgbalin(obj, smax=smax, smin=smin),
                      extent=(tmpx[0], tmpx[1], tmpy[0], tmpy[1]), aspect='auto', origin='lower')
            ax.set_title("Object amplitude & phase", fontsize=9)
            if smin is not None and smax is not None:
                ax.text(0.002, 0.99, "brightness scaling: 0-max=[%5.2f-%5.2f]" % (smin, smax),
                        horizontalalignment='left', verticalalignment='top',
                        fontsize=6, transform=ax.transAxes)

        # Plot scan area
        scan_area_pointsx = scan_area_points[0] * self.data.pixel_size_object()[0] * 1e6
        scan_area_pointsy = scan_area_points[1] * self.data.pixel_size_object()[1] * 1e6
        scan_area_pointsx = np.append(scan_area_pointsx, scan_area_pointsx[0]) + self.data.posx_c * 1e6
        scan_area_pointsy = np.append(scan_area_pointsy, scan_area_pointsy[0]) + self.data.posy_c * 1e6
        ax.plot(scan_area_pointsx, scan_area_pointsy, 'k-', linewidth=0.3)

        axtx = ax.twinx()
        axty = ax.twiny()
        ax.tick_params(axis='both', labelsize=8)
        axtx.tick_params(axis='both', labelsize=8)
        axty.tick_params(axis='both', labelsize=8)
        axtx.set_xlabel(u"dx(µm)", fontsize=8)
        axty.set_ylabel(u"dy(µm)", fontsize=8)
        ax.set_xlabel(u"absolute x(µm)", fontsize=8)
        ax.set_ylabel(u"absolute y(µm)", fontsize=8)

        def convert_ax_twin_to_relative(ax0):
            """
            Update twin axes with centered coordinates
            """
            y1, y2 = ax0.get_ylim()
            axtx.set_ylim((y1 - y2) / 2, (y2 - y1) / 2)
            x1, x2 = ax0.get_xlim()
            axty.set_xlim((x1 - x2) / 2, (x2 - x1) / 2)
            try:
                # axty.figure.canvas.draw()
                axtx.figure.canvas.draw()
            except:
                pass

        ax.callbacks.connect("ylim_changed", convert_ax_twin_to_relative)
        ax.callbacks.connect("xlim_changed", convert_ax_twin_to_relative)

        ax.set_xlim(tmpx[0], tmpx[1])
        ax.set_ylim(tmpy[0], tmpy[1])

        if self.p.data.near_field:
            ax.invert_yaxis()

        # Step between text lines of size 6
        dy = (6 + 1) / 72 / fig.get_size_inches()[1]

        # fig.text(0.25, 2 * dy, "Absolute coordinates of object center: x=%8.6e y=%8.6e" %
        #          (self.p.data.posx_c, self.p.data.posy_c),
        #          fontsize=6, horizontalalignment='center', stretch='condensed')

        # ======================== Plot Probe ===================================
        rx, ry = probe.shape[1] * fig_scale_o / fx, probe.shape[0] * fig_scale_o / fy
        ax = fig.add_axes((0.95 - rx, 0.1, rx, ry))
        ny, nx = probe.shape[-2:]
        tmpx = nx / 2 * self.data.pixel_size_object()[0] * 1e6
        tmpy = ny / 2 * self.data.pixel_size_object()[1] * 1e6
        smax = abs(probe * scan_area_probe).max()

        ax.imshow(plot_utils.complex2rgbalin(probe, smax=smax), extent=(tmpx, -tmpx, -tmpy, tmpy), origin='lower')
        ax.set_xlabel(u"x(µm)", fontsize=8)
        ax.set_ylabel(u"y(µm)", fontsize=8)
        ax.set_title("Probe amplitude & phase", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        if self.p.data.near_field:
            ax.invert_yaxis()

        # ptycho.insertColorwheel(left=0.47, bottom=.03, width=.06, height=.06)
        ax = fig.add_axes((0.47, 0.01, 0.06, 0.06), facecolor='w')
        plot_utils.colorwheel(ax=ax, fs=12)

        fig.suptitle("Scan #%d, %d frames, pixelsize=%5.1fnm, LLK=%8.3f\n algo=%s" %
                     (self.scan, len(self.x), self.data.pixel_size_object()[0] * 1e9,
                      self.p.llk_poisson / self.p.nb_obs, algostring), fontsize=9)
        y0 = 0.95 - 1.5 * dy
        n = 1
        vk = [k for k in self.params.keys()]
        vk.sort()
        for k in vk:
            v = self.params[k]
            if v is not None and k not in ['liveplot', 'livescan', 'saveplot', 'scan', 'algorithm', 'save',
                                           'saveprefix', 'nbrun', 'run0',
                                           'pixelsize', 'imgcounter', 'epoch', 'data2cxi', 'verbose']:
                fig.text(0.505, y0 - n * dy, "%s = %s" % (k, str(v)), fontsize=6, horizontalalignment='left',
                         stretch='condensed')
                n += 1
        fig.text(dy, dy, "PyNX v%s, finished at %s" % (_pynx_version, time.strftime("%Y/%m/%d %H:%M:%S")),
                 fontsize=6, horizontalalignment='left', stretch='condensed')

        if not self.params['near_field']:
            # Add probe full width information
            fwhmyx, fw20yx, fws = analysis.probe_fwhm(self.p.get_probe(), self.data.pixel_size_object()[0],
                                                      verbose=False)
            fig.text(0.6, dy, "FWHM : %7.2f(H)x%7.2f(V) nm**2 [peak]" % (fwhmyx[1] * 1e9, fwhmyx[0] * 1e9),
                     fontsize=6, horizontalalignment='left', stretch='condensed')
            fig.text(0.6, 2 * dy, "FW20%%: %7.2f(H)x%7.2f(V)nm**2 [extended]" % (fw20yx[1] * 1e9, fw20yx[0] * 1e9),
                     fontsize=6,
                     horizontalalignment='left', stretch='condensed')
            fig.text(0.6, 3 * dy, "FW (stat):  %7.2f [%7.2f(H)x%7.2f(V)] nm" %
                     (fws[0] * 1e9, fws[1] * 1e9, fws[2] * 1e9),
                     fontsize=6, horizontalalignment='left', stretch='condensed')

        # Add beam direction
        ax = fig.add_axes((0.55, 0.2, 0.05, 0.05), facecolor='w')
        ax.set_axis_off()
        ax.text(0, 0, 'x\n Beam\n(// z)', horizontalalignment='center', verticalalignment='center')  # fontweight='bold'
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if display_plot:
            try:
                plt.draw()
                plt.gcf().canvas.draw()
                plt.pause(.001)
            except:
                pass

        canvas = FigureCanvasAgg(fig)
        filename = self.params['saveprefix'] % (self.scan, run) + steps + '.png'
        canvas.print_figure(filename, dpi=150)
        if os.name == 'posix':
            try:
                sf = os.path.split(filename)
                os.system('ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest.png')))
            except:
                pass
        if self.timings is not None:
            dt = timeit.default_timer() - t0
            if "save_plot" in self.timings:
                self.timings["save_plot"] += dt
            else:
                self.timings["save_plot"] = dt

    def save_movie(self):
        """
        Create a movie of the scan with all scan positions and diffraction frames. Requires matplotlib and ffmpeg.
        :return:
        """
        self.print("\n", "#" * 100, "\n#", "\n#         Creating movie of scan \n#\n", "#" * 100)
        import matplotlib
        mpl_backend = matplotlib.get_backend()
        if True:  # try:
            matplotlib.use("Agg", warn=False)
            import matplotlib.pyplot as plt
            import matplotlib.animation as manimation
            from matplotlib.colors import LogNorm
            FFMpegWriter = manimation.writers['ffmpeg']

            # Get starting obj and probe
            obj, probe, data, scan_area_obj = self.obj0, self.probe0, None, None
            if obj is None:
                init = simulation.Simulation(obj_info=self.obj_init_info, probe_info=self.probe_init_info,
                                             data_info=self.data_info, verbose=self.mpi_master)
                init.make_obj()
                obj = init.obj.values

            if probe is None:
                init = simulation.Simulation(obj_info=self.obj_init_info, probe_info=self.probe_init_info,
                                             data_info=self.data_info, verbose=self.mpi_master)
                init.make_probe()
                probe = init.probe.values

            if self.p is not None:
                obj = self.p.get_obj()
                probe = self.p.get_probe()
                data = self.p.data
                scan_area_obj = self.p.get_scan_area_obj()
            else:
                data = PtychoData(iobs=self.iobs, positions=(self.x, self.y),
                                  detector_distance=self.params['detectordistance'],
                                  mask=None, pixel_size_detector=self.params['pixelsize'],
                                  wavelength=self.wavelength, vidx=self.imgn)

            if obj.ndim == 3:
                obj = obj[0]
            px, py = data.pixel_size_object()
            nyo, nxo = obj.shape[-2:]
            ny, nx = probe.shape[-2:]
            rx = nxo / 2 * self.data.pixel_size_object()[0] * 1e6
            ry = nyo / 2 * self.data.pixel_size_object()[1] * 1e6

            # Compute total illumination
            illum0 = (np.abs(probe) ** 2).sum(axis=0)
            illum_sum = np.zeros(obj.shape[-2:], dtype=np.float32)
            for i in range(len(self.iobs)):
                dy, dx = data.posy[i] / py, data.posx[i] / px
                cx, cy = get_view_coord((nyo, nxo), (ny, nx), dx, dy)
                illum_sum[cy:cy + ny, cx:cx + nx] += illum0

            # Movie type
            movie_type = 'illumination'
            fps = 10
            step = 1
            if type(self.params['movie']) is str:
                if 'object' in self.params['movie'].lower():
                    movie_type = 'object'
                if 'fps=' in self.params['movie'].lower():
                    fps = int(self.params['movie'].lower().split('fps=')[-1].split(',')[0])
                if 'step=' in self.params['movie'].lower():
                    step = int(self.params['movie'].lower().split('step=')[-1].split(',')[0])

            if movie_type == 'object':
                smin, smax = 0, np.ma.masked_array(abs(obj), ~scan_area_obj).max()

            self.print('Movie type: %s' % movie_type)

            metadata = dict(title='Ptycho scan', artist='PyNX')
            writer = FFMpegWriter(fps=fps, metadata=metadata)
            fontsize = 12

            path = os.path.dirname(self.params['saveprefix'] % (self.scan, 0))
            os.makedirs(path, exist_ok=True)
            filename = os.path.join(path, "scan-movie.mp4")

            fig = plt.figure(figsize=(10, 5))
            m = np.abs(illum_sum).max()
            mm = np.percentile(self.iobs, 99.999)
            illum = np.zeros_like(illum_sum)
            sys.stdout.write("Generating movie frames...")
            sys.stdout.flush()
            with writer.saving(fig, filename, dpi=100):
                for i in range(len(self.iobs)):
                    if (len(self.iobs) - i) % 10 == 0:
                        sys.stdout.write('%d ' % (len(self.iobs) - i))
                        sys.stdout.flush()
                    plt.clf()

                    plt.subplot(121)
                    dy, dx = data.posy[i] / py, data.posx[i] / px
                    cx, cy = get_view_coord((nyo, nxo), (ny, nx), dx, dy)
                    illum[cy:cy + ny, cx:cx + nx] += illum0
                    if i % step:
                        # Do not show all frames
                        continue
                    if movie_type == 'object':
                        smin, smax = 0, np.ma.masked_array(abs(obj), ~scan_area_obj).max()
                        o = obj * illum / (illum_sum + 1e-8 * m)
                        plt.imshow(plot_utils.complex2rgbalin(o, smax=smax, smin=smin), extent=(rx, -rx, -ry, ry))
                        plt.title("Object amplitude & phase")
                        plt.plot(rx - (cx + nx // 2) * px * 1e6, ry - (cy + ny // 2) * py * 1e6, 'ro')
                        plt.xlabel(u"x(µm)")
                        plt.ylabel(u"y(µm)")
                        plt.xlim(rx, -rx)
                        plt.ylim(-ry, ry)
                    else:  # movie_type == 'illumination':
                        plt.imshow(illum, vmin=0, vmax=m)
                        plt.plot((cx + nx // 2,), (cy + ny // 2,), 'ro')
                        plt.title('Cumulated illumination')

                    plt.subplot(122)
                    plt.imshow(self.iobs[i], norm=LogNorm(vmin=0.5, vmax=mm))
                    # plt.colorbar()
                    plt.title('Diffracted intensity')
                    plt.suptitle("%s - #%3d" % (self.params['saveprefix'] % (self.scan, 0), i), fontsize=fontsize)
                    writer.grab_frame()
        self.print('\nSaved movie to: %s' % filename)
        # except:
        #    self.print("PyNX ptycho runner. Could not create movie. Is FFMPEG installed ?")
        matplotlib.use(mpl_backend, warn=False)


class PtychoRunner:
    """
    Class to process a series of scans with a series of algorithms, given from the command-line.
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):
        """

        :param argv: the command-line parameters
        :param params: parameters for the optimization, with some default values.
        :param ptycho_runner_scan_class: the class to use to run the analysis.
        """
        self.timings = {}
        self.t0 = time.time()
        self.params = copy.deepcopy(params)
        self.argv = argv
        self.PtychoRunnerScan = ptycho_runner_scan_class
        self.help_text = helptext_generic

        self.mpi_master = True  # True even if MPI is not used
        if MPI is not None:
            self.mpic = MPI.COMM_WORLD
            self.mpi_master = self.mpic.Get_rank() == 0
            self.mpi_size = self.mpic.Get_size()
            self.mpi_rank = self.mpic.Get_rank()

        self.parse_arg()

        if 'help' not in self.argv and '--help' not in self.argv and self.mpi_master:
            self.check_params()

        self.ws = None

    def print(self, *args, **kwargs):
        """
        MPI-aware print function. Non-master processes will be muted
        :param args: args passed to print
        :param kwargs: kwrags passed to print
        :return: nothing
        """
        if self.mpi_master:
            print(*args, **kwargs)

    def parse_arg(self):
        """
        Parses the arguments given on a command line

        Returns: nothing

        """
        t0 = time.time()
        if self.mpi_master:
            for arg in self.argv:
                if arg.lower() in ['liveplot', 'livescan', 'data2cxi', 'orientation_round_robin', 'profiling',
                                   'saveplot', 'movie', 'multiscan_reuse_ptycho']:
                    self.params[arg.lower()] = True
                elif arg.lower() in ['dark_subtract']:
                    self.params[arg.lower()] = 1
                else:
                    s = arg.find('=')
                    if 0 < s < (len(arg) - 1):
                        k = arg[:s].lower()
                        v = arg[s + 1:]
                        self.print(k, v)
                        if k == 'mpi':
                            if MPI is None:
                                raise PtychoRunnerException('mpi=%s was given but MPI is not available' % v)
                            elif self.mpi_size == 1:
                                raise PtychoRunnerException('mpi=%s was given but MPI rank = 1' % v)
                        # NB: 'scan' is kept as a string, to be able to interpret e.g. scan="range(12,22)"
                        if k == 'mask':
                            k = 'loadmask'
                        if k in ['algorithm', 'saveprefix', 'load', 'object', 'probe', 'loadprobe', 'save', 'monitor',
                                 'scan', 'loadmask', 'detector_orientation', 'xy', 'flatfield', 'roi', 'data2cxi',
                                 'saveplot', 'dark', 'output_format', 'movie', 'mpi', 'multiscan_reuse_ptycho']:
                            self.params[k] = v
                        elif k in ['gpu']:
                            # Allows several GPU (sub)strings to be listed
                            g = v.split(',')
                            # Use either a string or a list, to check if both cases are correctly processed
                            if len(g) == 1:
                                self.params[k] = g[0]
                            else:
                                self.params[k] = g
                        elif k in ['nbrun', 'run0', 'maxframe', 'maxsize', 'verbose', 'rebin', 'stack_size',
                                   'center_probe_n', 'center_probe_max_shift', 'dm_loop_obj_probe', 'obj_max_pix',
                                   'mpi_split_nb_overlap', 'obj_margin', 'mpi_split_nb_neighbour']:
                            self.params[k] = int(v)
                        elif k in ['defocus', 'loadpixelsize', 'rotate', 'nrj', 'energy', 'dark_subtract',
                                   'detectordistance', 'detectordistance', 'distance']:
                            if k == 'energy':
                                k = 'nrj'
                            if k in ['distance', 'detector_distance']:
                                k = 'detectordistance'
                            self.params[k] = float(v)
                        elif k in ['autocenter', 'remove_obj_phase_ramp']:
                            if v.lower() in ['false', '0']:
                                self.params[k] = False
                        elif k == 'moduloframe':
                            vs = v.split(',')
                            n1 = int(vs[0])
                            if len(vs) == 1:
                                self.params[k] = (n1, 0)
                            else:
                                n2 = int(vs[1])
                                if n2 >= n1 or n2 < 0:
                                    raise PtychoRunnerException(
                                        'Parameter moduloframe: n1=%d, n2=%d must satisfy 0<=n2<n1')
                                self.params[k] = (n1, n2)
                        elif not (self.parse_arg_beamline(k, v)):
                            if self.mpi_master:
                                self.print("WARNING: argument not interpreted: %s=%s" % (k, v))
                    else:
                        if arg.find('.cxi') >= 0:
                            self.params['cxifile'] = arg
                        elif arg.find('pynx-') < 0 and self.parse_arg_beamline(arg.lower(), None) is False:
                            if self.mpi_master:
                                self.print("WARNING: argument not interpreted: %s" % arg)
        self.timings["parse_arg"] = time.time() - t0
        t0 = time.time()
        if MPI is not None:
            self.params = self.mpic.bcast(self.params, root=0)
        self.timings["parse_arg_mpi_bcast"] = time.time() - t0

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        This method should be superseded in a beamline/instrument-specific child class.

        Returns:
            True if the argument is interpreted, false otherwise
        """
        return False

    def check_params(self):
        """
        Check if self.params includes a minimal set of valid parameters

        Returns: Nothing. Will raise an exception if necessary
        """
        self.check_params_beamline()
        if self.params['probe'] is None and self.params['load'] is None and self.params['loadprobe'] is None and \
                self.params['data2cxi'] is False and self.params['movie'] is None and not self.params['near_field']:
            raise PtychoRunnerException('Missing argument: either probe=, load= or loadprobe= is required')
        if self.params['scan'] is None and self.params['cxifile'] is None and self.params[
            'data'] is None and 'h5meta' not in self.params and self.params['livescan'] is None:
            raise PtychoRunnerException('Missing argument: no scan # given')
        # if self.params['gpu'] is None :
        #    raise PtychoRunnerException('Missing argument: no gpu name given (e.g. gpu=Titan)')
        if 'split' in self.params['mpi'] and 'npz' in self.params['output_format']:
            raise PtychoRunnerException("output_format=npz is not supported for mpi=splitscan")
        if self.params['mpi_split_nb_overlap'] < 1 or self.params['mpi_split_nb_overlap'] > 10:
            raise PtychoRunnerException("mpi_split_nb_overlap=%d must be between 1 and 10 (1-2 recommended)"
                                        % self.params['mpi_split_nb_overlap'])

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamline.
        Derived implementations can also set default values when appropriate.

        Returns: Nothing. Will raise an exception if necessary
        """
        pass

    def process_scans(self):
        """
        Run all the analysis on the supplied scan list, unless 'help' is given as a
        command-line argument.

        :return: Nothing
        """
        self.timings["process_scans_load_scan"] = 0
        self.timings["process_scans_mpi_scan_split"] = 0
        self.timings["process_scans_load_data"] = 0
        self.timings["process_scans_prepare_processing_unit"] = 0
        self.timings["process_scans_center_crop_data"] = 0
        self.timings["process_scans_prepare"] = 0
        self.timings["process_scans_algorithm"] = 0
        self.timings["process_scans_run"] = 0
        self.timings["process_scans_all"] = 0
        if 'help' in self.argv or '--help' in self.argv:
            self.print(self.help_text)
            sys.exit(0)

        if (self.params['cxifile'] is not None or self.params['data'] is not None or 'h5meta' in self.params or
            self.params['instrument'] == 'simulation') and self.params['scan'] is None:
            # Only when reading a CXI, ptypy or a single hdf5 metadata file (from id16) we accept a dummy scan value
            vscan = [0]
        else:
            vscan = eval(self.params['scan'])
            if type(vscan) is int:
                vscan = [vscan]

        cxifile0 = self.params['cxifile']
        if MPI is not None:
            if 'multi' in self.params['mpi'] and self.mpi_size > 1:
                # Distribute the scan among the different cients, independently
                vscan = vscan[self.mpi_rank::self.mpi_size]
                print("MPI #%2d analysing scans:" % self.mpi_rank, vscan)
            elif 'split' in self.params['mpi'] and self.mpi_master:
                self.print("Using MPI: %s" % self.params['mpi'])
        else:
            self.params['mpi'] = "no"
        for scan in vscan:
            reuse_ptycho = False
            if self.params['multiscan_reuse_ptycho'] is not False and self.ws is not None:
                reuse_ptycho = True
                if isinstance(self.params['multiscan_reuse_ptycho'], str):
                    self.params['algorithm0'] = self.params['algorithm']
                    self.params['algorithm'] = self.params['multiscan_reuse_ptycho']
            try:
                t00 = time.time()
                if cxifile0 is not None:
                    if '%' in cxifile0:
                        self.params['cxifile'] = cxifile0 % scan
                        self.print('Loading CXIFile:', self.params['cxifile'])
                if reuse_ptycho:
                    self.ws.scan = scan
                else:
                    self.ws = self.PtychoRunnerScan(self.params, scan, timings=self.timings)
                t0 = time.time()
                if self.mpi_master or 'split' not in self.params['mpi']:
                    self.ws.load_scan()
                self.timings["process_scans_load_scan"] += time.time() - t0
                t0 = time.time()
                self.ws.mpi_scan_split()
                self.timings["process_scans_mpi_scan_split"] += time.time() - t0
                t0 = time.time()
                if not reuse_ptycho:
                    self.ws.prepare_processing_unit()
                self.timings["process_scans_prepare_processing_unit"] += time.time() - t0
                t0 = time.time()
                self.ws.load_data()
                self.timings["process_scans_load_data"] += time.time() - t0
                if self.params['data2cxi']:
                    if self.params['data2cxi'] == 'crop':
                        self.ws.center_crop_data()
                        self.ws.save_data_cxi(crop=True, verbose=True)
                    else:
                        self.ws.save_data_cxi(verbose=True)
                else:
                    if self.params['orientation_round_robin']:
                        for xy in ['x,y', 'x,-y', '-x,y', '-x,-y', 'y,x', 'y,-x', '-y,x', '-y, -x']:
                            self.params['xy'] = xy
                            for transp in range(2):
                                for flipud in range(2):
                                    for fliplr in range(2):
                                        self.params['detector_orientation'] = '%d,%d,%d' % (transp, flipud, fliplr)
                                        self.ws.center_crop_data()
                                        self.ws.prepare()
                                        self.ws.run()

                    else:
                        t0 = time.time()
                        self.ws.center_crop_data()
                        self.timings["process_scans_center_crop_data"] += time.time() - t0
                        t0 = time.time()
                        if not reuse_ptycho:
                            self.ws.prepare()
                        self.timings["process_scans_prepare"] += time.time() - t0
                        t0 = time.time()
                        self.ws.run(reuse_ptycho=reuse_ptycho)
                        self.timings["process_scans_run"] += time.time() - t0
                self.timings["process_scans_all"] += time.time() - t00
            except PtychoRunnerException as ex:
                self.print(traceback.format_exc())
                self.print(self.help_text)
                self.print('\n\n Caught exception for scan %d: %s    \n' % (scan, str(ex)))
                sys.exit(1)
        self.params['cxifile'] = cxifile0

        self.print("Timings:")
        for k, v in self.timings.items():
            if v > 1e-6:
                if MPI is not None:
                    self.print("MPI #%2d: %40s :%6.2fs" % (self.mpi_rank, k, v))
                else:
                    self.print("         %40s :%6.2fs" % (k, v))
