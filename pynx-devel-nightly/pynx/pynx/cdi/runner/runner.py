#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import os
import sys
import gc
import copy
import time
import timeit
import warnings
import traceback
from PIL import Image

from ...utils import h5py
import fabio
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.ndimage.measurements import center_of_mass
from scipy.io import loadmat
from .. import *
from ...utils.math import smaller_primes
from ...utils.array import rebin
from ...mpi import MPI
from ..plot import show_cdi

helptext_generic = """
generic (not beamline-specific) command-line/file parameters arguments: (all case-insensitive)

    data=data.npy: name of the data file including the 3D observed intensity.
                   recognized formats include .npy, .npz (if several arrays are included iobs, 
                   should be named 'data' or 'iobs'), .tif or .tiff 
                   (assumes a multiframe tiff image), or .cxi (hdf5).
                   An hdf5 file can be supplied with a path, 
                   e.g. data=path/to/data.h5:entry/path/data
                   Multiple files can be processed by combining with scan, e.g.:
                   * data=my_data%04d.cxi scan=23,24,25
                   * data=my_data%sd.cxi scan=room_temperature,
                   [mandatory unless another beamline-specific method to import data is used]
                   
    detector_distance=0.7: detector distance in meters
    
    pixel_size_data=55e-6: deprecated, use pixel_size_detector instead

    pixel_size_detector=55e-6: pixel size of the supplied data (detector pixel size)
    
    wavelength=1.5e-10: wavelength in meters
    
    verbose=20: the run will print and optionally plot every 'verbose' cycles
    
    live_plot: if used as keyword (or live_plot=True in a parameters file), a live plot 
               will be shown  every 'verbose' cycle. If an integer number N is given, 
               display every N cycle.
    save_plot: if used as keyword (or save_plot=True in a parameters file), a
        final plot will be saved in a png file with 3 cuts of the object, some statistics
        and the optimisation parameters.
        The default is to show the object cuts as RGBA, or the absolute if positivity
        is used. The type of plot can be changed, e.g.:
         * save_plot=rgba : the default
         * save_plot=abs : absolute value (default with positivity)
         * save_plot=gray : specify a matplotlib colormat name
    
    gpu=Titan: name of the gpu to use [optional, by default the fastest available will be used]
    
    auto_center_resize: if used (command-line keyword) or =True, the input data will be centered 
                        and cropped  so that the size of the array is compatible with the (GPU) 
                        FFT library used. If 'roi' is used, centering is based on ROI. 
                        [default=False]
    
    roi=0,120,0,235,0,270: set the region-of-interest to be used, with min,max given along each 
                           dimension in python fashion (min included, max excluded), for the 2 or 3
                           axis. ROI coordinates should be indicated before any rebin is done.
                           Note that if 'auto_center_resize' is used, the ROI may still be shrunk
                           to obtain an array size compatible with the FFT library used. Similarly 
                           it will be shrunk if 'max_size' is used but ROI size is larger.
                           Other example: roi=0,-1,300-356,300+256,500-256,500+256
                           [default=None]
                        
    nb_run=1: number of times to run the optimization [default=1]
    
    nb_run_keep: number of best run results to keep, according to likelihood statistics. This is only useful
                 associated with nb_run [default: keep all run results]
    
    nb_run_keep_max_obj2_out: when performing multiple runs with nb_run_keep, if the fraction
        of the object square modulus outside the object support is larger than this value,
        the solution will be rejected regardless of the free log-likelihood.
        [default:0.1]
    
    data2cxi: if used as keyword (or data2cxi=True in a parameters file), convert the original 
              data to CXI(HDF5)  format. Will be saved to file 'data.cxi', or if a data file
              has been supplied (e.g. data57.npz), to the same file with extension .cxi.
              
    output_format='cxi': choose the output format for the final object and support.
                         Other possible choice: 'npz', 'none'
                         [Default='cxi']
    
    note='This dataset was measure... Citation: Journal of coherent imaging (2018), 729...':
         Optional text note which will be saved as a note in the output CXI file 
         (and also for data2cxi).
         
    instrument='ESRF-idxx': the name of the beamline/instrument used for data collection
                            [default: depends on the script actually called]
                            
    sample_name='GaN nanowire': optional name for the sample
    
    flatfield=flat.npz: filename with a flatfield array by which the observed intensity
        will be multiplied. In order to preserve Poisson statistics, the flat correction
        should be close to 1.
        Accepted file formats: .npy, .npz, .edf, .mat, .tif, .h5, .cxi.
        The first available array will be used if multiple are present.
        For an hdf5 file (h5 or cxi), the path can be supplied e.g. using 
        data=path/to/flatfield.h5:entry/path/flat - if the hdf5 path is not given,
        the first array with a correct shape and with 'flat' in the path is used.
        If the flatfield is 2D and the observed intensity 3D, the flatfield is repeated
        along the depth (first/slowest axis).
        Note that if no flatfield is given and mask=maxipix1 or maxipix6, a correction is
        automatically to the gap pixels (see mask documentation below).
        [default=None, no flatfield correction]

    mask=zero: mask for the diffraction data. If 'zero', all pixels with iobs <= 0 will be masked.
              If 'negative', all pixels with iobs < 0 will be masked. 
              If 'maxipix', the maxipix gaps will be masked.
              If 'maxipix1', the maxipix gaps will be masked, except for the large pixel one ach side of the gap,
              for which the intensity will be divided by 3.
              If 'maxipix6', the maxipix gaps will be linearly interpolated between the gap large pixels.
              NaN-valued pixels in the observed intensity are always masked.
              
              If a filename is given (accepted formats: .npy, .npz, .edf, .mat, .tif, .h5, .cxi):
              (the first available array will be used if multiple are present) file. For an hdf5 file (.h5 or .cxi),
              the path can be supplied e.g. using data=path/to/mask.h5:entry/path/mask - if the hdf5 path
              is not given, the first array with a correct shape and with 'mask' in the path is used.
              Pixels = 0 are valid, > 0 are masked. If the mask is 2D
              and the data 3D, the mask is repeated for all frames along the first dimension (depth).
              Several masks can be combined, separated by a comma, e.g. mask=maxipix,dead.npz
              [default=None, no mask]
    
    free_pixel_mask=result.cxi: pixel mask for free log-likelihood calculation. By default the free pixel mask is
                                randomly initialised, but this can be used to load a mask from a file for
                                consistency between runs. It can be loaded from a .npy, .edf, .tif, .npz or 
                                .mat file - if .npz or .mat are used, the first array matching the iobs shape is used. 
                                A .cxi filename can also be used as input, in which case it is assumed that this
                                is the result of a previous optimisation, and that the mask can be loaded from
                                '/entry_last/image_1/process_1/configuration/free_pixel_mask'
    
    iobs_saturation=1e6: saturation value for the observed intensity. Any pixel above this intensity will be masked
                         [default: no saturation value]
    
    zero_mask: by default masked pixels are free and keep the calculated intensities during HIO, RAAR, ER and CF cycles.
               Setting this flag will force all masked pixels to zero intensity. This can be more stable with a large 
               number of masked pixels and low intensity diffraction data.
               If a value is supplied the following options can be used:
               zero_mask=0: masked pixels are free and keep the calculated complex amplitudes
               zero_mask=1: masked pixels are set to zero
               zero_mask=auto: this is only meaningful when using a 'standard' algorithm below. The masked pixels will
                               be set to zero during the first 60% of the HIO/RAAR cycles, and will be free during the 
                               last 40% and ER, ML ones.

    mask_interp=16,2: interpolate masked pixels from surrounding pixels, using an inverse distance
        weighting. The first number N indicates that the pixels used for interpolation range
        from i-N to i+N for pixel i around all dimensions. The second number n that the weight
        is equal to 1/d**n for pixels with at a distance n.
        The interpolated values iobs_m are stored in memory as -1e19*(iobs_m+1) so that the
        algorithm knows these are not trul observations, and are applied with a large
        confidence interval.
        [default:None, no interpolation is done for masked pixels]
    
    confidence_interval_factor_mask=0.5,1.2:
        When masked pixels are interpolated using mask_interp, the calculated values iobs_m are
        not observed and the amplitude projection is done for those with a confidence interval,
        equal to: iobs_m*confidence_interval_factor_mask
        Two values (min/max) must be given, normally around 1 [default:0.5,1.2]

    object=obj.npy: starting object. Import object from .npy, .npz, .mat (the first available array 
          will  be used if multiple are present), CXI or hdf5 modes file.
          It is also possible to supply the random range for both amplitude and phase, using:
          * object=support,0.9,1,0.5: this will use random values over the initial support,
            with random amplitudes between 0.9 and 1 and phases with a 0.5 radians range
          * object=obj.npy,0.9,1,0.5: same but the random values will be multiplied by
                the loaded object.
          [default=None, object will be defined as random values inside the support area]
    
    support=sup.cxi: starting support. Different options are available:
        * support=sup.cxi: Import support from .npy, .npz, .edf, .mat (the first 
          available array will be used if multiple are present) or CXI/hdf5 file. Pixels > 0
          are in the support, 0 outside. If the support shape is different than the Iobs
          array, it will be cropped or padded accordingly.
          If a CXI filename is given, the support will be searched in entry_last/image_1/mask
          or entry_1/image_1/mask, e.g. to load support from a previous result file.
          For CXI and h5 files, the hdf5 path can also be given:
          support=path/to/data.h5:entry_1/image_1/data
        * support=auto: support will be estimated using the intensity
        * support=circle: or 'square', the support will be initialised to a 
          circle (sphere in 3d), or a square (cube).
        * support=object: the support will be initialised from the object
          using the given support threshold and smoothing parameters - this requires
          that the object itself is loaded from a file. The applied threshold is always
          relative to the max in this case.
        * support=object,0.1,1.0: same as above, but 0.1 will be used as the relative
          threshold and 1.0 the smoothing width just for this initialisation.
        All those are ignored if support_formula is given.
        [default='auto', support will be defined by auto-correlation]
    
    support_size=50: size (radius or half-size) for the initial support, to be used in 
                     combination with 'support_type'. The size is given in pixel units.
                     Alternatively one value can be given for each dimension, i.e. 
                     support_size=20,40 for 2D data, and support_size=20,40,60 for 3D data. 
                     This will result in an initial support which is a rectangle/parallelepiped
                     or ellipsis/ellipsoid.  Ignored if support_formula is given.
                     [if not given, this will trigger the use of auto-correlation 
                      to estimate the initial support]

    support_formula: formula to compute the support. This should be an equation using ix,iy,iz and ir pixel coordinates
                     (all centered on the array) which can be evaluated to produce an initial support (1/True inside, 
                     0/False outside). Mathematical functions should use the np. prefix (np.sqrt, ..). Examples:
                     support_formula="ir<100": sphere or circle of radius 100 pixels
                     support_formula="(np.sqrt(ix**2+iy**2)<50)*(np.abs(iz)<100)": cylinder of radius 50 and height 200

    support_autocorrelation_threshold=0.1: if no support is given, it will be estimated 
                                           from the intensity autocorrelation, with this relative 
                                           threshold.
                                           [default value: 0.1]

    support_threshold=0.25: threshold for the support update. Alternatively two values can be given, and the threshold
                            will be randomly chosen in the interval given by two values: support_threshold=0.20,0.28.
                            This is mostly useful in combination with nb_run.
                            [default=0.25]
    
    support_threshold_method=max: method used to determine the absolute threshold for the 
                                  support update. Either:'max' or 'average' or 'rms' (the default) values,
                                  taken over the support area/volume, after smoothing.
                                  Note that 'rms' and 'average' use the modulus or root-mean-square value normalised 
                                  over the support area, so when the support shrinks, the threshold tends to increase.
                                  In contrast, the 'max' value tends to diminish as the optimisation
                                  progresses. rms is varying more slowly than average and is thus more stable.
                                  In practice, using rms or average with a too low threshold can lead to the divergence 
                                  of the support (too large), if the initial support 

    support_only_shrink: if set or support_only_shrink=True, the support will only shrink 
                         (default: the support can grow again)
                         
    support_smooth_width_begin=2
    support_smooth_width_end=0.25: during support update, the object amplitude is convoluted by a
                                   gaussian with a size
                                   (sigma) exponentially decreasing from support_smooth_width_begin
                                   to support_smooth_width_end from the first to the last RAAR or 
                                   HIO cycle.
                                   [default values: 2 and 0.5]
    
    support_smooth_width_relax_n: the number of cycles over which the support smooth width will
                                  exponentially decrease from support_smooth_width_begin to 
                                  support_smooth_width_end, and then stay constant. 
                                  This is ignored if nb_hio, nb_raar, nb_er are used, 
                                  and the number of cycles used
                                  is the total number of HIO+RAAR cycles [default:500]
    
    support_post_expand=1: after the support has been updated using a threshold,  it can be shrunk 
                           or expanded by a few pixels, either one or multiple times, e.g. in order
                           to 'clean' the support:
                           - support_post_expand=1 will expand the support by 1 pixel
                           - support_post_expand=-1 will shrink the support by 1 pixel
                           - support_post_expand=-1,1 will shrink and then expand the support 
                             by 1 pixel
                           - support_post_expand=-2,3 will shrink and then expand the support 
                             by 2 and 3 pixels
                           - support_post_expand=2,-4,2 will expand/shrink/expand the support 
                             by 2, 4 and 2 pixels
                           - etc..
                           [default=None, no shrinking or expansion]

    support_update_border_n: if > 0, the only pixels affected by the support updated lie within +/- N pixels around the
                             outer border of the support.

    support_update_period=50: during RAAR/HIO, update support every N cycles.
                              If 0, support is never updated.

    support_fraction_min=0.001: if the fraction of points inside the support falls below 
        this value, the run is aborted, and a few tries will be made by dividing the
        support_threshold by support_threshold_auto_tune_factor.
        [default:0.0001]

    support_fraction_max=0.5: if the fraction of points inside the support becomes larger than
        this value, the run is aborted, and a few tries will be made by multiplying the
        support_threshold by support_threshold_auto_tune_factor.
        [default:0.7]
    
    support_threshold_auto_tune_factor: the factor by which the support threshold will be
        changed if the support diverges
        [default: 1.1]

    positivity: if set or positivity=True, the algorithms will be biased towards a real, positive
                object. Object is still complex-valued, but random start will begin with real 
                values. [default=False]
    
    beta=0.9: beta value for the HIO/RAAR algorithm [default=0.9]
    
    crop_output=0: if >0 (default:4), the output data will be cropped around the final
                   support plus 'crop_output' pixels. If 0, no cropping is performed.
    
    rebin=2: the experimental data can be rebinned (i.e. a group of n x n (x n) pixels is
             replaced by a single one whose intensity is equal to the sum of all the pixels).
             Both iobs and mask (if any) will be rebinned, but the support (if any) should
             correspond to the new size. The supplied pixel_size_detector should correspond
             to the original size. The rebin factor can also be supplied as one value per
             dimension, e.g. "rebin=4,1,2".
             Also, instead of summing pixels with the given rebin size, it is possible
             to select a single sub-pixel by skipping instead of binning: 
             e.g. using "rebin=4,1,2,0,0,1", the extracted pixel will use 
             array slicing data[0::4,0::1,1::2].
             [default: no rebin]
    
    max_size=256: maximum size for the array used for analysis, along all dimensions. The data
                  will be cropped to this value after rebinning and centering. [default: no maximum size]
                  
    user_config*=*: this can be used to store a custom configuration parameter which will be ignored by the 
                    algorithm, but will be stored among configuration parameters in the CXI file (data and output).
                    e.g.: user_config_temperature=268K  user_config_comment="Vibrations during measurement" etc...

    ############# ALGORITHMS: standard version, using RAAR, then HIO, then ER and ML

    nb_raar=600: number of relaxed averaged alternating reflections cycles, which the 
                 algorithm will use first. During RAAR and HIO, the support is updated regularly

    nb_hio=0: number of hybrid input/output cycles, which the algorithm will use after RAAR. 
                During RAAR and HIO, the support is updated regularly

    nb_er=200: number of error reduction cycles, performed after HIO, without support update

    nb_ml=20: number of maximum-likelihood conjugate gradient to perform after ER

    detwin: if set (command-line) or if detwin=True (parameters file), 10 cycles will be performed
            at 25% of the total number of RAAR or HIO cycles, with a support cut in half to bias
            towards one twin image

    psf=pseudo-voigt,1,0.05,10: this will trigger the activation of a point-spread function
        kernel to model partial coherence (and detector psf), after 66% of the total number
        of HIO and RAAR have been reached. 
        The following options are possible for the psf:
        psf=pseudo-voigt,1,0.05,10: use a pseudo-Voigt profile with FWHM 1 pixel and eta=0.05,
            i.e. 5% Lorentzian and 95% Gaussian, then update every 5 cycles.
        psf=gaussian,1.5,10: start with a Gaussian of FWHM 1.5 pixel, and update it every 10 cycles 
        psf=lorentzian,0.6,10: start with a Lorentzian of FWHM 0.6 pixel, and update it every 10 cycles 
        psf=path/to/file.npz: load the PSF array from an npz file. The array loaded will either
            be named 'psf' or 'PSF', or the first found array will be loaded. The PSF must be centred
            in the array, and will be resized (so it can be cropped). The PSF will be updated
            every 5 cycles
        psf=/path/to/file.cxi: load the PSF from a previous result from a CXI file.The PSF will be updated
            every 5 cycles
        psf=/path/to/file.cxi,10: same as before, update the PSF every 10 cycles
        psf=/path/to/file.cxi,0: same as before, do not update the PSF
        
        Recommended values:
        - for highly coherent datasets: either no PSF, or psf=gaussian,1,20
        - with significant partial coherence: psf=pseudo-voigt,1,0.05,20
    
    psf_filter=hann: the PSF update can also be filtered to avoid divergence. Possible values
        include "hann", "tukey" or "none" (the default, equivalent to False)
        
    ############# ALGORITHMS: customized version 
    
    algorithm="ER**50,(Sup*ER**5*HIO**50)**10": give a specific sequence of algorithms and/or 
              parameters to be 
              used for the optimisation (note: this string is case-insensitive).
              Important: 
              1) when supplied from the command line, there should be NO SPACE in the expression !
              And if there are parenthesis in the expression, quotes are required around the 
              algorithm string
              2) the string and operators are applied from right to left

              Valid changes of individual parameters include (see detailed description above):
                positivity = 0 or 1
                support_only_shrink = 0 or 1
                beta = 0.7
                live_plot = 0 (no display) or an integer number N to trigger plotting every N cycle
                support_update_period = 0 (no update) or a positive integer number
                support_smooth_width_begin = 2.0
                support_smooth_width_end = 0.5
                support_smooth_width_relax_n = 500
                support_threshold = 0.25
                support_threshold_mult = 0.7  (multiply the threshold by 0.7)
                support_threshold_method=max or average or rms
                support_post_expand=-1#2 (in this case the commas are replaced by # for parsing)
                zero_mask = 0 or 1
                psf = 5: will update the PSF every 5 cycles (and init PSF if necessary with the default
                    pseudo-Voigt of FWHM 1 pixels and eta=0.1)
                psf_init=gaussian@1.5: initialise the PSF with a Gaussian of FWHM 1.5 pixels
                psf_init=lorentzian@0.5: initialise the PSF with a Lorentzian of FWHM 0.5 pixels
                psf_init=pseudo-voigt@1@0.1: initialise the PSF with a pseudo-Voigt of FWHM 1 pixels
                    and eta=0.1
                    Note that using psf_init automatically triggers updating the PSF every 5 cycles,
                    unless it has already been set using 'psf=...'
                verbose=20
                fig_num=1: change the figure number for plotting
                
              Valid basic operators include:
                ER: Error Reduction
                HIO: Hybrid Input/Output
                RAAR: Relaxed Averaged Alternating Reflections
                DetwinHIO: HIO with a half-support (along first dimension)
                DetwinHIO1: HIO with a half-support (along second dimension)
                DetwinHIO2: HIO with a half-support (along third dimension)
                DetwinRAAR: RAAR with a half-support (along first dimension)
                DetwinRAAR1: RAAR with a half-support (along second dimension)
                DetwinRAAR2: RAAR with a half-support (along third dimension)
                CF: Charge Flipping
                ML: Maximum Likelihood conjugate gradient (incompatible with partial coherence PSF)
                FAP: FourierApplyAmplitude- Fourier to detector space, apply observed amplitudes,
                    and back to object space.
                Sup or SupportUpdate: update the support according to the support_* parameters
                ShowCDI: display of the object and calculated/observed intensity. This can be used
                         to trigger this plot at specific steps, instead of regularly using 
                         live_plot=N. This is thus best used using live_plot=0
              
              Examples of algorithm strings, where steps are separated with commas (and NO SPACE!),
              and are applied from right to left. Operations in a given step will be applied
              mathematically, also from right to left, and **N means repeating N tymes (N cycles) 
              the  operation on the left of the exponent:
                algorithm=HIO : single HIO cycle
                
                algorithm=ER**100 : 100 cycles of ER
                
                algorithm=ER**50*HIO**100 : 100 cycles of HIO, followed by 50 cycles of ER
                
                algorithm=ER**50,HIO**100 : 100 cycles of HIO, followed by 50 cycles of ER
                    The result is the same as the previous example, the difference between using *
                    and "," when switching from HIO to ER is mostly cosmetic as the process will
                    separate the two algorithmic steps explicitly when using a ",", which
                    can be slightly slower.
                    Moreover, when using "save=all", the different steps will be saved as
                    different entries in the CXI file.
                
                algorithm="ER**50,(Sup*ER**5*HIO**50)**10" : 
                    10 times [50 HIO + 5 ER + Support update], followed by 50 ER
                
                algorithm="ER**50,verbose=1,(Sup*ER**5*HIO**50)**10,verbose=100,HIO**100":
                    change the periodicity of verbose output
                
                algorithm="ER**50,(Sup*ER**5*HIO**50)**10,support_post_expand=1,
                           (Sup*ER**5*HIO**50)**10,support_post_expand=-1#2,HIO**100"
                    same but change the post-expand (wrap) method
                
                algorithm="ER**50,(Sup*ER**5*HIO**50)**5,psf=5,(Sup*ER**5*HIO**50)**10,HIO**100"
                    activate partial correlation after a first series of algorithms
                
                algorithm="ER**50,(Sup*HIO**50)**4,psf=5,(Sup*HIO**50)**8"
                    typical algorithm steps with partial coherence
                
                algorithm="ER**50,(Sup*HIO**50)**4,(Sup*HIO**50)**4,positivity=0,
                          (Sup*HIO**50)**8,positivity=1"
                    same as previous but starting with positivity constraint, removed at the end.

                

            [default: use nb_raar, nb_hio, nb_er and nb_ml to perform the sequence of algorithms]     

    save=all: either 'final' or 'all' this keyword will activate saving after each optimisation 
              step (comma-separated) of the algorithm in any given run. Note that the name of
              the save file including the LLK ank LLK_free values will be decided by the first
              step and not the final one.
              [default=final]

    ############# MPI: distribute computing 

    mpi=scan or run: when launching the script using mpiexec, this tells the script to
              either distribute the list of scans to different processes (mpi=scan, the default),
              or (mpi=run) split the runs to the different processes. If nb_run_keep is used,
              the results are merged before selecting the best results.
    

==================================================================================================
                                   End of common help text
                       Instructions for the specific script are given below
==================================================================================================
"""

# This must be defined in in beamline/instrument-specific scripts
helptext_beamline = ""

params_generic = {'data': None, 'detector_distance': None, 'pixel_size_detector': None, 'wavelength': None,
                  'verbose': 50, 'live_plot': False, 'gpu': None, 'auto_center_resize': False, 'roi_user': None,
                  'roi_final': None, 'nb_run': 1,
                  'max_size': None, 'data2cxi': False, 'output_format': 'cxi', 'mask': None, 'support': 'auto',
                  'support_autocorrelation_threshold': 0.1, 'support_only_shrink': False, 'object': None,
                  'support_update_period': 50, 'support_smooth_width_begin': 2, 'support_smooth_width_end': 0.5,
                  'support_smooth_width_relax_n': 500, 'support_size': None, 'support_threshold': 0.25,
                  'positivity': False, 'beta': 0.9, 'crop_output': 4, 'rebin': None, 'support_update_border_n': 0,
                  'support_threshold_method': 'rms', 'support_post_expand': None, 'psf': False, 'note': None,
                  'instrument': None, 'sample_name': None, 'fig_num': 1, 'algorithm': None, 'zero_mask': False,
                  'nb_run_keep': None, 'save': 'final', 'gps_inertia': 0.05, 'gps_t': 1.0, 'gps_s': 0.9,
                  'gps_sigma_f': 0, 'gps_sigma_o': 0, 'iobs_saturation': None, 'free_pixel_mask': None,
                  'support_formula': None, 'mpi': 'splitscan', 'mask_interp': None,
                  'confidence_interval_factor_mask_min': 0.5,
                  'confidence_interval_factor_mask_max': 1.2,
                  'save_plot': False, 'support_fraction_min': 1e-4, 'support_fraction_max': 0.7,
                  'support_threshold_auto_tune_factor': 1.1, 'nb_run_keep_max_obj2_out': 0.10,
                  'flatfield': None, 'psf_filter': None}


class CDIRunnerException(Exception):
    pass


class CDIRunnerScan:
    """
    Abstract class to handle CDI data. Must be derived to be used.
    """

    def __init__(self, params, scan, timings=None):
        self.params = params
        self.scan = scan
        self.mask = None
        self.rebinf = 1
        self.cdi = None
        self.iobs = None
        self.support = None
        self.processing_unit = None
        self._algo_s = ""
        # Keep track of saved results, for later sorting & merging
        # Entries are dictionaries with file name and properties including log-likelihood,...
        self.saved_results = []
        if timings is not None:
            self.timings = timings
        else:
            self.timings = {}

    def load_data(self):
        """
        Loads data. This function only loads data (2D or 3D) from generic files, and should be derived for
        beamline-specific imports (from spec+individual images, etc..)

        Returns: nothing. Loads data in self.iobs, and initializes mask and initial support.

        """
        t0 = timeit.default_timer()
        filename = self.params['data']
        print('Loading data: ', filename)

        # Extract hdf5path if supplied as 'filename:path/to/support'
        if ':' in os.path.splitdrive(filename)[1] and ('h5' in filename or 'cxi' in filename):
            # Looks like an hdf5 file with the hdf5 path
            drive, filename = os.path.splitdrive(filename)
            filename, h5paths = filename.split(':')
            filename = drive + filename
            h5paths = [h5paths]
        else:
            h5paths = ['entry_1/instrument_1/detector_1/data', 'entry_1/image_1/data']

        ext = os.path.splitext(filename)[-1]
        if ext == '.npy':
            self.iobs = np.load(filename).astype(np.float32)
        elif ext == '.edf':
            self.iobs = fabio.open(filename).data.astype(np.float32)
        elif ext == '.tif' or ext == '.tiff':
            frames = Image.open(filename)
            nz, ny, nx = frames.n_frames, frames.height, frames.width
            self.iobs = np.empty((nz, ny, nx), dtype=np.float32)
            for i in range(nz):
                frames.seek(i)
                self.iobs[i] = np.array(frames)
        elif ext == '.npz':
            d = np.load(filename)
            if 'data' in d.keys():
                self.iobs = d['data'].astype(np.float32)
            elif 'iobs' in d.keys():
                self.iobs = d['iobs'].astype(np.float32)
            else:
                # Assume only the data is in the datafile
                for k, v in d.items():
                    self.iobs = v.astype(np.float32)
                    break
        elif ext == '.h5':
            with h5py.File(filename, 'r') as h:
                import_nok = True
                for hp in h5paths:
                    if hp in h:
                        self.iobs = h[hp][()].astype(np.float32)
                        import_nok = False
                        break
                if import_nok:
                    raise CDIRunnerException("Could not find data=%s" % self.params['data'])
        elif ext == '.cxi':
            cxi = h5py.File(filename, 'r')
            if '/entry_1/instrument_1/source_1/energy' in cxi:
                nrj = cxi['/entry_1/instrument_1/source_1/energy'][()] / 1.60218e-16
                self.params['wavelength'] = 12.384 / nrj * 1e-10
                print("  CXI input: Energy = %8.2fkeV" % nrj)
            if '/entry_1/instrument_1/detector_1/distance' in cxi:
                self.params['detector_distance'] = cxi['/entry_1/instrument_1/detector_1/distance'][()]
                print("  CXI input: detector distance = %8.2fm" % self.params['detector_distance'])
            if '/entry_1/instrument_1/detector_1/x_pixel_size' in cxi:
                self.params['pixel_size_detector'] = cxi['/entry_1/instrument_1/detector_1/x_pixel_size'][()]
                print("  CXI input: detector pixel size = %8.2fum" % (self.params['pixel_size_detector'] * 1e6))
            print("  CXI input: loading iobs")
            if 'entry_1/instrument_1/detector_1/data' in cxi:
                self.iobs = cxi['entry_1/instrument_1/detector_1/data'][()].astype(np.float32)
            elif 'entry_1/image_1/data' in cxi:
                self.iobs = cxi['entry_1/image_1/data'][()].astype(np.float32)
                if 'entry_1/image_1/is_fft_shifted' in cxi:
                    if cxi['entry_1/image_1/is_fft_shifted'][()] > 0:
                        self.iobs = fftshift(self.iobs)
            else:
                self.iobs = cxi['entry_1/data_1/data'][()].astype(np.float32)
            if 'entry_1/instrument_1/detector_1/mask' in cxi:
                self.mask = cxi['entry_1/instrument_1/detector_1/mask'][()].astype(np.int8)
                nb = self.mask.sum()
                print("  CXI input: loading mask, with %d pixels masked (%6.3f%%)" % (nb, nb * 100 / self.mask.size))

            if 'entry_1/data_1/process_1/configuration' in cxi:
                # Load specific parameters from CXI file and carry on some parameters
                # TODO: this is a bit of a kludge, find a more maintainable way to decide which parameters to keep..
                for k, v in cxi['entry_1/data_1/process_1/configuration'].items():
                    vv = None
                    if k in self.params:
                        vv = self.params[k]
                    if (k not in self.params.keys() or (k in ['instrument', 'sample_name', 'scan', 'imgname']
                                                        and vv is None)) and v is not None:
                        if type(v) is h5py.Group:
                            self.params[k] = {}
                            for dk, dv in v.items():
                                self.params[k][dk] = dv[...]
                        else:
                            self.params[k] = v[...]

        print("Finished loading iobs data, with size:", self.iobs.size)

        self.timings['load_data'] = timeit.default_timer() - t0

    def rebin_data(self):
        """
        Will rebin the data (iobs and mask) and update pixel_size_detector according to self.params['rebin'].
        this must be called at the end of load_data, after init_mask. If a ROI has been given, cropping
        will occur here.
        :return:
        """
        if self.params['roi_user'] is not None:
            print("Cropping to user ROI: ", self.params['roi_user'])
            if self.iobs.ndim == 3:
                izmin, izmax, iymin, iymax, ixmin, ixmax = self.params['roi_user']
                self.iobs = self.iobs[izmin:izmax, iymin:iymax, ixmin:ixmax]
                if self.mask is not None:
                    self.mask = self.mask[izmin:izmax, iymin:iymax, ixmin:ixmax]
            elif self.iobs.ndim == 2:
                iymin, iymax, ixmin, ixmax = self.params['roi_user']
                self.iobs = self.iobs[iymin:iymax, ixmin:ixmax]
                if self.mask is not None:
                    self.mask = self.mask[iymin:iymax, ixmin:ixmax]

        if self.params['rebin'] is not None:
            print("Rebinning Iobs with rebin=(%s)" % self.params['rebin'])
            rs = self.params['rebin'].split(',')
            if len(rs) == 1:
                self.params['rebin'] = int(rs[0])
                if self.params['rebin'] == 1:
                    print('Ignoring rebin=1')
                    return
            else:
                self.params['rebin'] = []
                for rr in rs:
                    self.params['rebin'].append(int(rr))
                if self.params['rebin'][:self.iobs.ndim] == [1] * self.iobs.ndim:
                    print('Ignoring rebin=1')
                    return
            self.iobs = rebin(self.iobs, self.params['rebin'])
            if self.mask is not None:
                self.mask = rebin(self.mask, self.params['rebin'])

    def init_mask(self):
        """
        Load mask if the corresponding parameter has been set, or just initialize an array of 0.
        This must be called after iobs has been imported, and before cropping, so at the end of load_data()

        Returns:
            Nothing. self.mask is created.
        """
        if self.params['mask'] is None or self.params['mask'] in ['no', 'nan', 'NaN']:
            if self.mask is None:
                # Else the mask has probably been pre-loaded, e.g. from a CXI file
                self.mask = np.zeros_like(self.iobs, dtype=np.int8)

        self.mask = np.isnan(self.iobs).astype(np.int8)
        if self.params['mask'] is not None and self.params['mask'] not in ['no', 'nan', 'NaN']:
            # Allow to add multiple masks
            for mask_string in self.params['mask'].split(','):
                if mask_string.lower() == 'zero':
                    # All zero-valued pixels are masked
                    print('Masking all pixels with iobs <= 0')
                    self.mask += (self.iobs <= 0).astype(np.int8)
                elif mask_string.lower() == 'negative':
                    # All pixels <0 are masked
                    print('Masking all pixels with iobs < 0')
                    self.mask += (self.iobs < 0).astype(np.int8)
                elif mask_string.lower() == 'tomo_beamstop':
                    # Mask all half-lines along all directions which are always equal to zero
                    print('Masking all zero-valued half-lines')
                    self.mask += np.zeros_like(self.iobs, dtype=np.int8)
                    nz, ny, nx = self.iobs.shape
                    # Along z
                    iy, ix = np.nonzero(self.iobs[:nz // 2].sum(axis=0) == 0)
                    self.mask[:nz // 2, iy, ix] += 1
                    iy, ix = np.nonzero(self.iobs[nz // 2:].sum(axis=0) == 0)
                    self.mask[nz // 2:, iy, ix] += 1
                    # Along y
                    iz, ix = np.nonzero(self.iobs[:, :ny // 2].sum(axis=1) == 0)
                    self.mask[iz, :ny // 2, ix] += 1
                    iz, ix = np.nonzero(self.iobs[:, ny // 2:].sum(axis=1) == 0)
                    self.mask[iz, ny // 2:, ix] += 1
                    # Along x
                    iz, iy = np.nonzero(self.iobs[:, :, :nx // 2].sum(axis=2) == 0)
                    self.mask[iz, iy, :nx // 2] += 1
                    iz, iy = np.nonzero(self.iobs[:, :, nx // 2:].sum(axis=2) == 0)
                    self.mask[iz, iy, nx // 2:] += 1
                    # Also mask pixels around center which are equal to zero
                    self.mask[3 * nz // 8:5 * nz // 8, 3 * ny // 8:5 * ny // 8, 3 * nx // 8:5 * nx // 8] += \
                        self.iobs[3 * nz // 8:5 * nz // 8, 3 * ny // 8:5 * ny // 8, 3 * nx // 8:5 * nx // 8] <= 0

                elif 'maxipix' in mask_string.lower():
                    if (self.iobs.shape[-2] % 258 != 0) or (self.iobs.shape[-1] % 258 != 0):
                        raise CDIRunnerException(
                            "ERROR: used mask= maxipix, but x and y dimensions are not multiple of 258.")
                    print('Initializing MAXIPIX mask')
                    ny, nx = self.iobs.shape[-2:]
                    dn = 3
                    if mask_string.lower() == 'maxipix1':
                        # Use the big gap pixel (3x surface) intensity in a single pixel,
                        # normalised in corr_flat_field()
                        dn = 2
                    elif mask_string.lower() == 'maxipix6':
                        # The big pixel intensity will be spread over the entire gap in corr_flat_field()
                        dn = 0
                    if dn > 0:
                        if self.iobs.ndim == 2:
                            for i in range(258, ny, 258):
                                self.mask[i - dn:i + dn] += 1
                            for i in range(258, nx, 258):
                                self.mask[:, i - dn:i + dn] += 1
                        else:
                            for i in range(258, ny, 258):
                                self.mask[:, i - dn:i + dn] += 1
                            for i in range(258, nx, 258):
                                self.mask[:, :, i - dn:i + dn] += 1
                else:
                    filename = mask_string
                    h5path = None
                    if ':' in os.path.splitdrive(filename)[1] and ('h5' in filename or 'cxi' in filename):
                        # Looks like an hdf5 file with the hdf5 path
                        drive, filename = os.path.splitdrive(filename)
                        filename, h5path = filename.split(':')
                        filename = drive + filename

                    print('Loading mask from: ', filename)
                    ext = os.path.splitext(filename)[-1].lower()
                    if ext == '.npy':
                        self.mask += np.load(filename).astype(np.int8)
                    elif ext == '.tif' or ext == '.tiff':
                        frames = Image.open(filename)
                        frames.seek(0)
                        self.mask += np.array(frames).astype(np.int8)
                    elif ext == '.mat':
                        a = list(loadmat(filename).values())
                        for v in a:
                            if np.size(v) > 1000:
                                # Avoid matlab strings and attributes, and get the array
                                self.mask += np.array(v).astype(np.int8)
                                break
                    elif ext == '.npz':
                        d = np.load(filename)
                        if 'mask' in d.keys():
                            self.mask += d['mask'].astype(np.int8)
                        else:
                            # Assume only the mask is in the datafile
                            for k, v in d.items():
                                self.mask += v.astype(np.int8)
                                break
                    elif ext == '.edf':
                        self.mask += fabio.open(filename).data.astype(np.int8)
                    elif ext in ['.h5', '.cxi']:
                        with h5py.File(filename, 'r') as h:
                            if h5path is None:
                                # Try to find a dataset with 'mask' in the name and an adequate shape
                                def find_mask(name, obj):
                                    if 'mask' in name.lower():
                                        if isinstance(obj, h5py.Dataset):
                                            print(obj.shape, self.iobs.shape)
                                            if obj.shape == self.iobs.shape or \
                                                    (self.iobs.ndim == 3 and obj.shape == self.iobs[0].shape):
                                                return name

                                h5path = h.visititems(find_mask)
                            if False:
                                # Too risky
                                if h5path is None:
                                    # Find the first dataset with an adequate shape
                                    def find_mask2(name, obj):
                                        if isinstance(obj, h5py.Dataset):
                                            if obj.shape == self.iobs.shape or \
                                                    (self.iobs.ndim == 3 and obj.shape == self.iobs[0].shape):
                                                return name

                                    h5path = h.visititems(find_mask2)
                            if h5path is None:
                                raise CDIRunnerException("Supplied mask=%s, but could not find a suitable mask"
                                                         "dataset in the hdf5 file." % filename)
                            print("Loading mask from hdf5 file %s at path: %s" % (filename, h5path))
                            self.mask += h[h5path][()]
                    else:
                        raise CDIRunnerException(
                            "Supplied mask=%s, but format not recognized (recognized .npy,"
                            ".npz, .tif, .edf, .mat, .h5, .cxi)"
                            % filename)

        if self.params['iobs_saturation'] is not None:
            if self.params['iobs_saturation'] > 0:
                m = self.iobs > self.params['iobs_saturation']
                ms = m.sum()
                if ms > 0:
                    print('Masking %d saturated pixels (%6.3f%%)' % (ms, ms / self.iobs.size))
                    self.mask = ((self.mask + m) > 0).astype(np.int8)

        # Keep only 1 and 0
        self.mask = self.mask > 0

        nb = self.mask.sum()
        p = nb * 100 / self.mask.size
        if p > 20:
            print('WARNING: large number of masked pixels !')
        print("Initialized mask, with %d pixels masked (%6.3f%%)" % (nb, p))
        if self.iobs.ndim == 3 and self.mask.ndim == 2:
            print("Got a 2D mask, expanding to 3D")
            self.mask = np.tile(self.mask, (len(self.iobs), 1, 1))
        if self.iobs.shape != self.mask.shape:
            raise CDIRunnerException('Iobs', self.iobs.shape, ' and mask', self.mask.shape, ' must have the same shape')
        # Set to zero masked pixels (need a bool array for this)
        self.iobs[self.mask] = 0

        if self.params['mask'] not in ['zero', 'negative']:
            # Make sure no intensity is < 0, which could come from background subtraction
            tmp = self.iobs < 0
            nb = tmp.sum()
            if nb:
                print("CDI runner: setting %d pixels with iobs < 0 to zero" % nb)
                self.iobs[tmp] = 0

        # Keep mask as int8
        self.mask = self.mask.astype(np.int8)

    def corr_flat_field(self):
        """
        Correct the image by a known flat field. Currently only applies correction for maxipix large pixels
        :return:
        """
        if self.params['flatfield'] is not None:
            filename = self.params['flatfield']
            print('Loading flatfield from: ', filename)
            h5path = None
            if ':' in os.path.splitdrive(filename)[1] and ('h5' in filename or 'cxi' in filename):
                # Looks like an hdf5 file with the hdf5 path
                drive, filename = os.path.splitdrive(filename)
                filename, h5path = filename.split(':')
                filename = drive + filename
            ext = os.path.splitext(filename)[-1]
            flatfield = None
            if ext == '.npy':
                flatfield = np.load(filename).astype(np.float32)
            elif ext == '.edf':
                flatfield = fabio.open(filename).data.astype(np.float32)
            elif ext == '.tif' or ext == '.tiff':
                frames = Image.open(filename)
                nz, ny, nx = frames.n_frames, frames.height, frames.width
                flatfield = np.empty((nz, ny, nx), dtype=np.float32)
                for i in range(nz):
                    frames.seek(i)
                    flatfield[i] = np.array(frames)
            elif ext == '.npz':
                d = np.load(filename)
                if 'flatfield' in d.keys():
                    flatfield = d['flatfield'].astype(np.float32)
                elif 'flat' in d.keys():
                    flatfield = d['flat'].astype(np.float32)
                elif 'data' in d.keys():
                    flatfield = d['data'].astype(np.float32)
                else:
                    # Assume only the flatfield is in the datafile and hope for the best...
                    for k, v in d.items():
                        flatfield = v.astype(np.float32)
                        break
            elif ext in ['.h5', '.cxi']:
                with h5py.File(filename, 'r') as h:
                    if h5path is None:
                        # Try to find a dataset with 'flat' in the name and an adequate shape
                        def find_flat(name, obj):
                            print(name)
                            if 'flat' in name.lower():
                                if isinstance(obj, h5py.Dataset):
                                    print(obj.shape, self.iobs.shape)
                                    if obj.shape == self.iobs.shape or \
                                            (self.iobs.ndim == 3 and obj.shape == self.iobs[0].shape):
                                        return name

                        h5path = h.visititems(find_flat)
                    if h5path is None:
                        raise CDIRunnerException("Supplied flatfield=%s, but could not find a suitable flatfield"
                                                 "dataset in the hdf5 file." % filename)
                    print("Applying flatfield from hdf5 file %s at path: %s" % (filename, h5path))
                    flatfield = h[h5path][()]
            if flatfield is None:
                raise CDIRunnerException(
                    "Supplied flatfield=%s, but format not recognized (recognized .npy, "
                    ".npz, .tif, .edf, .mat, .h5, .cxi)"
                    % filename)
            if flatfield.shape != self.iobs.shape and (self.iobs.ndim != 3 or flatfield.shape != self.iobs[0].shape):
                raise CDIRunnerException(
                    "Supplied flatfield=%s does not have the correct shape:" % filename,
                    flatfield.shape, self.iobs.shape)
            self.iobs *= flatfield.astype(np.float32)
        elif self.params['mask'] is not None:
            if 'maxipix1' in self.params['mask']:
                ny, nx = self.iobs.shape[-2:]
                self.iobs = self.iobs.astype(np.float32)  # Avoid integer types
                if self.iobs.ndim == 2:
                    for i in range(258, ny, 258):
                        self.iobs[i - 3] /= 3
                        self.iobs[i + 2] /= 3
                    for i in range(258, nx, 258):
                        self.iobs[:, i - 3] /= 3
                        self.iobs[:, i + 2] /= 3
                else:
                    for i in range(258, ny, 258):
                        self.iobs[:, i - 3] /= 3
                        self.iobs[:, i + 2] /= 3
                    for i in range(258, nx, 258):
                        self.iobs[:, :, i - 3] /= 3
                        self.iobs[:, :, i + 2] /= 3
            elif 'maxipix6' in self.params['mask']:
                ny, nx = self.iobs.shape[-2:]
                if self.iobs.ndim == 2:
                    for i in range(258, ny, 258):
                        v0, v1 = self.iobs[i - 3] / 3, self.iobs[i + 2] / 3
                        m = self.mask[i - 3] + self.mask[i + 2]
                        for j in range(-3, 3):
                            self.iobs[i + j] = (v0 * (6 - j + 3) + v1 * (j + 3)) / 6
                            self.mask[i + j] += m
                    for i in range(258, nx, 258):
                        v0, v1 = self.iobs[:, i - 3] / 3, self.iobs[:, i + 2] / 3
                        m = self.mask[:, i - 3] + self.mask[:, i + 2]
                        for j in range(-3, 3):
                            self.iobs[:, i + j] = (v0 * (6 - j + 3) + v1 * (j + 3)) / 6
                            self.mask[:, i + j] += m
                else:
                    for i in range(258, ny, 258):
                        v0, v1 = self.iobs[:, i - 3] / 3, self.iobs[:, i + 2] / 3
                        m = self.mask[:, i - 3] + self.mask[:, i + 2]
                        for j in range(-3, 3):
                            self.iobs[:, i + j] = (v0 * (6 - j + 3) + v1 * (j + 3)) / 6
                            self.mask[:, i + j] += m
                    for i in range(258, nx, 258):
                        v0, v1 = self.iobs[:, :, i - 3] / 3, self.iobs[:, :, i + 2] / 3
                        m = self.mask[:, :, i - 3] + self.mask[:, :, i + 2]
                        for j in range(-3, 3):
                            self.iobs[:, :, i + j] = (v0 * (6 - j + 3) + v1 * (j + 3)) / 6
                            self.mask[:, :, i + j] += m
                # Keep only 1 and 0 for the mask
                self.mask = (self.mask > 0).astype(np.int8)

            # np.savez_compressed('test_maxipix_flat_mask.npz', mask=self.mask, iobs2=self.iobs.sum(axis=0))

    def init_support(self):
        """
        Prepare the initial support. Note that the mask should be initialized first.
        if self.params['support'] == 'auto', nothing is done, and the support will be created after the cdi object,
        to use GPU functions.
        :return: Nothing. self.support is created
        """
        if self.params['support_formula'] is not None:
            self.support = InitSupportShape(formula=self.params['support_formula'], lazy=True)
            return

        elif self.params['support'] in ['square', 'cube', 'circle', 'sphere'] and \
                self.params['support_size'] is not None:
            s = eval(self.params['support_size'])
            self.support = InitSupportShape(shape=self.params['support'], size=s, lazy=True)
            return

        elif type(self.params['support']) is str:

            if self.params['support'].lower().split(',')[0] in ['object', 'obj']:
                if self.params['object'] is None:
                    raise CDIRunnerException("Error: support=object was used but no object was given")
                if isinstance(self.params['object'], str):
                    if self.params['object'] == "support" or self.params['object'].startswith("support,"):
                        raise CDIRunnerException("Error: support=object was used but no object was given")
                t = self.params['support_threshold']
                s = self.params['support_smooth_width_begin']
                if ',' in self.params['support']:
                    tmp = self.params['support'].split(',')
                    t = float(tmp[1])
                    if len(tmp) > 2:
                        s = float(tmp[2])
                self.support = SupportUpdate(threshold_relative=t, smooth_width=s, method='max', lazy=True,
                                             verbose=True)
                print("support=object: will create the initial support from the supplied object "
                      "using: relative threshold=%4.3f, method=max, smooth=%4.2f" % (t, s))
                return

            if self.params['support'].lower() == 'auto':
                print("No support given. Will use autocorrelation to estimate initial support")
                r = self.params['support_autocorrelation_threshold']
                self.support = AutoCorrelationSupport(r, lazy=True, verbose=True)
                return

            filename = self.params['support']
            print('Loading support from: ', filename)

            # Extract hdf5path if supplied as 'filename:path/to/support'
            if ':' in os.path.splitdrive(filename)[1] and ('h5' in filename or 'cxi' in filename):
                # Looks like an hdf5 file with the hdf5 path
                drive, filename = os.path.splitdrive(filename)
                filename, h5paths = filename.split(':')
                filename = drive + filename
                h5paths = [h5paths]
            else:
                h5paths = ['entry_last/image_1/mask', 'entry_1/image_1/mask']

            ext = os.path.splitext(filename)[-1]
            if ext == '.npy':
                self.support = np.load(filename).astype(np.int8)
            elif ext == '.tif' or ext == '.tiff':
                frames = Image.open(filename)
                frames.seek(0)
                self.support = np.array(frames).astype(np.int8)
            elif ext == '.mat':
                a = list(loadmat(filename).values())
                for v in a:
                    if np.size(v) > 1000:
                        # Avoid matlab strings and attributes, and get the array
                        self.support = np.array(v).astype(np.int8)
                        break
            elif ext == '.npz':
                d = np.load(filename)
                if 'support' in d.keys():
                    self.support = d['support'].astype(np.int8)
                else:
                    # Assume only the support is in the datafile
                    for k, v in d.items():
                        self.support = v.astype(np.int8)
                        break
            elif ext == '.edf':
                self.support = fabio.open(filename).data.astype(np.int8)
            elif ext == '.cxi' or ext == '.h5':
                # this may be a CXI result file, try to find the mask
                with h5py.File(filename, 'r') as h:
                    import_nok = True
                    for hp in h5paths:
                        if hp in h:
                            self.support = h[hp][()].astype(np.int8)
                            print("Loaded support from %s:%s" % (filename, hp))
                            import_nok = False
                            break
                    if import_nok:
                        raise CDIRunnerException(
                            "Supplied support=%s, could not find support e.g. in entry_1/image_1/mask)" % filename)
            else:
                raise CDIRunnerException(
                    "Supplied support=%s, but format not recognized"
                    "(not .npy, .npz, .tif, .edf, .mat, .cxi)" % filename)
            if self.iobs.ndim == 3 and self.support.ndim == 2:
                print("Got a 2D support, expanding to 3D")
                self.support = np.tile(self.support, (len(self.iobs), 1, 1))

            # We can accept a support with a size different from the data, just pad with zeros
            # if smaller than iobs, or crop (assuming it is centered) otherwise
            if self.iobs.shape != self.support.shape:
                print("Reshaping supplied support to fit iobs size",
                      self.support.shape, "->", self.iobs.shape)
                sup = np.zeros_like(self.iobs, dtype=np.int8)
                if self.iobs.ndim == 2:
                    ny, nx = self.iobs.shape
                    nys, nxs = self.support.shape
                    # Crop
                    if ny < nys:
                        self.support = self.support[(nys-ny) // 2:(nys - ny) // 2 + ny]
                    if nx < nxs:
                        self.support = self.support[:, (nxs - nx) // 2:(nxs - nx) // 2 + nx]
                    # Pad
                    nys, nxs = self.support.shape
                    sup[(ny - nys) // 2:(ny - nys) // 2 + nys, (nx - nxs) // 2:(nx - nxs) // 2 + nxs] = self.support
                if self.iobs.ndim == 3:
                    nz, ny, nx = self.iobs.shape
                    nzs, nys, nxs = self.support.shape
                    # Crop
                    if nz < nzs:
                        self.support = self.support[(nzs - nz) // 2:(nzs - nz) // 2 + nz]
                    if ny < nys:
                        self.support = self.support[:, (nys - ny) // 2:(nys - ny) // 2 + ny]
                    if nx < nxs:
                        self.support = self.support[:, :, (nxs - nx) // 2:(nxs - nx) // 2 + nx]
                    # Pad
                    nzs, nys, nxs = self.support.shape
                    sup[(nz - nzs) // 2:(nz - nzs) // 2 + nzs, (ny - nys) // 2:(ny - nys) // 2 + nys,
                    (nx - nxs) // 2:(nx - nxs) // 2 + nxs] = self.support
                self.support = sup
        else:
            print("No support given. Will use autocorrelation to estimate initial support")
            r = self.params['support_autocorrelation_threshold']
            self.support = AutoCorrelationSupport(r, lazy=True, verbose=True)
            return

        nb = self.support.sum()
        p = nb * 100 / self.support.size
        print("Initialized support ", self.support.shape, ", with %d pixels (%6.3f%%)" % (nb, p))
        if p > 20:
            print('WARNING: large number of support pixels !')

    def prepare_data(self, init_mask=True, rebin_data=True, init_support=True, center_crop_data=True,
                     corr_flat_field=True):
        """
        Prepare CDI data for processing.

        :param init_mask: if True (the default), initialize the mask
        :param rebin_data: if True (the default), rebin the data
        :param init_support: if True (the default), initialize the support
        :param center_crop_data: if True (the default), center & crop data
        :param corr_flat_field: if True (the default), correct for the flat field
        :return:
        """
        if init_mask:
            t0 = timeit.default_timer()
            self.init_mask()
            self.timings['init_mask'] = timeit.default_timer() - t0
        if corr_flat_field:
            t0 = timeit.default_timer()
            self.corr_flat_field()
            self.timings['corr_flat_field'] = timeit.default_timer() - t0
        if rebin_data:
            t0 = timeit.default_timer()
            self.rebin_data()
            self.timings['rebin_data'] = timeit.default_timer() - t0
        if init_support:
            t0 = timeit.default_timer()
            self.init_support()
            self.timings['init_support'] = timeit.default_timer() - t0
        if center_crop_data:
            t0 = timeit.default_timer()
            self.center_crop_data()
            self.timings['center_crop_data'] = timeit.default_timer() - t0

    def center_crop_data(self):
        """
        Auto-center diffraction pattern, and resize according to allowed FFT prime decomposition.
        Takes into account ROI if supplied.
        :return: 
        """
        self.params['iobs_shape_orig'] = self.iobs.shape
        if self.iobs.ndim == 3:
            nz0, ny0, nx0 = self.iobs.shape
            if self.params['auto_center_resize'] and self.params['roi_user'] is None:
                # Find center of mass
                z0, y0, x0 = center_of_mass(self.iobs)
                print("Center of mass at:", z0, y0, x0)
                iz0, iy0, ix0 = int(round(z0)), int(round(y0)), int(round(x0))
            else:
                iz0, iy0, ix0 = int(nz0 // 2), int(ny0 // 2), int(nx0 // 2)
            # Max symmetrical box around center of mass
            nx = 2 * min(ix0, nx0 - ix0)
            ny = 2 * min(iy0, ny0 - iy0)
            nz = 2 * min(iz0, nz0 - iz0)
            if self.params['max_size'] is not None:
                m = self.params['max_size']
                if nx > m:
                    nx = m
                if ny > m:
                    ny = m
                if nz > m:
                    nz = m
            # Crop data to fulfill FFT size requirements
            nz1, ny1, nx1 = smaller_primes((nz, ny, nx), maxprime=self.processing_unit.max_prime_fft_radix(),
                                           required_dividers=(2,))

            print("Centering & reshaping data: (%d, %d, %d) -> (%d, %d, %d)" % (nz0, ny0, nx0, nz1, ny1, nx1))
            self.iobs = self.iobs[iz0 - nz1 // 2:iz0 + nz1 // 2, iy0 - ny1 // 2:iy0 + ny1 // 2,
                        ix0 - nx1 // 2:ix0 + nx1 // 2]
            if self.mask is not None:
                self.mask = self.mask[iz0 - nz1 // 2:iz0 + nz1 // 2, iy0 - ny1 // 2:iy0 + ny1 // 2,
                            ix0 - nx1 // 2:ix0 + nx1 // 2]
            if isinstance(self.support, np.ndarray):
                self.support = self.support[nz0 // 2 - nz1 // 2:nz0 // 2 + nz1 // 2,
                               ny0 // 2 - ny1 // 2:ny0 // 2 + ny1 // 2,
                               nx0 // 2 - nx1 // 2:nx0 // 2 + nx1 // 2]

            # Store the final ROI relative to the uncropped, non-rebinned array, for CXI export
            r = [1, 1, 1]
            if self.params['rebin'] is not None:
                # 'rebin' parameter has already been pre-processed in rebin_data()
                if type(self.params['rebin']) is int:
                    r = [self.params['rebin']] * 3
                elif len(self.params['rebin']) == 1:
                    r = [self.params['rebin']] * 3
                else:
                    r = self.params['rebin']
            izmin, izmax, iymin, iymax, ixmin, ixmax = 0, 0, 0, 0, 0, 0
            if self.params['roi_user'] is not None:
                izmin, izmax, iymin, iymax, ixmin, ixmax = self.params['roi_user']
            self.params['roi_final'] = ((iz0 - nz1 // 2) * r[0] + izmin, (iz0 + nz1 // 2) * r[0] + izmin,
                                        (iy0 - ny1 // 2) * r[1] + iymin, (iy0 + ny1 // 2) * r[1] + iymin,
                                        (ix0 - nx1 // 2) * r[2] + ixmin, (ix0 + nx1 // 2) * r[2] + ixmin)
        else:
            ny0, nx0 = self.iobs.shape
            if self.params['auto_center_resize'] and self.params['roi_user'] is None:
                # Find center of mass
                y0, x0 = center_of_mass(self.iobs)
                print("Center of mass at:", y0, x0)
                iy0, ix0 = int(round(y0)), int(round(x0))
            else:
                iy0, ix0 = int(ny0 // 2), int(nx0 // 2)
            # Max symmetrical box around center of mass
            nx = 2 * min(ix0, nx0 - ix0)
            ny = 2 * min(iy0, ny0 - iy0)
            if self.params['max_size'] is not None:
                m = self.params['max_size']
                if nx > m:
                    nx = m
                if ny > m:
                    ny = m
            # Crop data to fulfill FFT size requirements
            ny1, nx1 = smaller_primes((ny, nx), maxprime=self.processing_unit.max_prime_fft_radix(),
                                      required_dividers=(2,))

            print("Centering & reshaping data: (%d, %d) -> (%d, %d)" % (ny0, nx0, ny1, nx1))
            self.iobs = self.iobs[iy0 - ny1 // 2:iy0 + ny1 // 2, ix0 - nx1 // 2:ix0 + nx1 // 2]
            if self.mask is not None:
                self.mask = self.mask[iy0 - ny1 // 2:iy0 + ny1 // 2, ix0 - nx1 // 2:ix0 + nx1 // 2]
            if isinstance(self.support, np.ndarray):
                self.support = self.support[ny0 // 2 - ny1 // 2:ny0 // 2 + ny1 // 2,
                               nx0 // 2 - nx1 // 2:nx0 // 2 + nx1 // 2]
                print(self.iobs.shape, self.mask.shape, self.support.shape)
            # Store the final ROI relative to the uncropped, non-rebinned array, for CXI export
            r = [1, 1]
            if self.params['rebin'] is not None:
                # 'rebin' parameter has already been pre-processed in rebin_data()
                if type(self.params['rebin']) is int:
                    r = [self.params['rebin']] * 2
                elif len(self.params['rebin']) == 1:
                    r = [self.params['rebin']] * 2
                else:
                    r = self.params['rebin']
            iymin, iymax, ixmin, ixmax = 0, 0, 0, 0
            if self.params['roi_user'] is not None:
                iymin, iymax, ixmin, ixmax = self.params['roi_user']
            self.params['roi_final'] = ((iy0 - ny1 // 2) * r[0] + iymin, (iy0 + ny1 // 2) * r[0] + iymin,
                                        (ix0 - nx1 // 2) * r[1] + ixmin, (ix0 + nx1 // 2) * r[1] + ixmin)

    def init_object(self):
        """
        Prepare the initial object. It will be random unless a file was supplied for input.
        :return: the initialised object, or an operator which can be used for a lazy initialisation
        """
        if self.iobs is not None:
            shape = self.iobs.shape
        else:
            # Once the cdi object is created, we delete self.iobs to save memory
            shape = self.cdi.iobs.shape

        if self.params['object'] is None:
            # Initialize random object
            if self.params['positivity']:
                obj = InitObjRandom(src="support", amin=0, amax=1, phirange=0)
            else:
                obj = InitObjRandom(src="support", amin=0, amax=1, phirange=2 * np.pi)
            return obj

        elif type(self.params['object']) is str:
            if ',' in self.params['object']:
                filename = self.params['object'].split(',')[0]
                if filename == 'support':
                    # Special case: object=support,0.9,1,0,0.5
                    s = [float(v) for v in self.params['object'].split(',')[1:]]
                    obj = InitObjRandom(src="support", amin=s[0], amax=s[1], phirange=s[2])
                    return obj
            else:
                filename = self.params['object']
            print('Loading object from: ', filename)
            ext = os.path.splitext(filename)[-1]
            if ext == '.npy':
                obj = np.load(filename).astype(np.complex64)
            elif ext == '.mat':
                a = list(loadmat(filename).values())
                for v in a:
                    if np.size(v) > 1000:
                        # Avoid matlab strings and attributes, and get the array
                        obj = np.array(v).astype(np.complex64)
                        break
            elif ext == '.npz':
                d = np.load(filename)
                if 'object' in d.keys():
                    obj = d['object'].astype(np.complex64)
                elif 'obj' in d.keys():
                    obj = d['obj'].astype(np.complex64)
                elif 'data' in d.keys():
                    obj = d['data'].astype(np.complex64)
                else:
                    # Assume only the object is in the datafile
                    for k, v in d.items():
                        obj = v.astype(np.complex64)
                        break
            elif ext == '.cxi':
                # TODO: allow to supply the hdf5 data path
                cxi = h5py.File(filename, 'r')
                if '/entry_1/image_1/data' in cxi:
                    obj = cxi['/entry_1/image_1/data'][()].astype(np.complex64)
                elif '/entry_1/data1/data' in cxi:
                    obj = cxi['/entry_1/data1/data'][()].astype(np.complex64)
                else:
                    raise CDIRunnerException(
                        "Supplied object=%s, but did not find either /entry_1/image_1/data"
                        "or /entry_1/data1/data in the CXI file" % filename)
            elif ext == '.h5':
                # TODO: allow to supply the hdf5 data path
                h = h5py.File(filename, 'r')
                if '/entry_1/data_1/data' in h:
                    obj = h['/entry_1/image_1/data']
                elif '/entry_1/data1/data' in h:
                    obj = h['/entry_1/data1/data']
                else:
                    raise CDIRunnerException(
                        "Supplied object=%s, but did not find either /entry_1/image_1/data"
                        "or /entry_1/data1/data in the CXI file" % filename)
                if obj.ndim == len(shape):
                    obj = obj[()]
                elif obj.ndim == len(shape) + 1:
                    # Likely a modes file, take the first one
                    obj = obj[0]
                else:
                    raise CDIRunnerException(
                        "Supplied object=%s, but the dimensions of the object don't match" % filename)
            else:
                raise CDIRunnerException(
                    "Supplied object=%s, but format not recognized (not .cxi, .npy, .npz, .mat)" % filename)

        # We can accept a smaller object than the data, just expand it as necessary
        if shape != obj.shape:
            for i in range(len(shape)):
                if shape[i] < obj.shape[i]:
                    raise CDIRunnerException('Object', obj.shape, ' must have a <= shape than iobs',
                                             shape, '(smaller objects will be padded with zeros)')
            print("Reshaping object to fit iobs size", obj.shape, "->", shape)
            tmp = np.zeros(shape, dtype=np.complex64)
            if len(shape) == 2:
                ny, nx = shape
                nys, nxs = obj.shape
                tmp[(ny - nys) // 2:(ny - nys) // 2 + nys, (nx - nxs) // 2:(nx - nxs) // 2 + nxs] = obj
            if len(shape) == 3:
                nz, ny, nx = shape
                nzs, nys, nxs = obj.shape
                tmp[(nz - nzs) // 2:(nz - nzs) // 2 + nzs, (ny - nys) // 2:(ny - nys) // 2 + nys,
                (nx - nxs) // 2:(nx - nxs) // 2 + nxs] = obj
            obj = tmp
        if self.cdi is not None:
            self.cdi.set_obj(obj, shift=True)
        if type(self.params['object']) is str:
            if ',' in self.params['object']:
                # Random parameters where supplied in addition to a starting object
                s = [float(v) for v in self.params['object'].split(',')[1:]]
                if self.cdi is not None:
                    obj = InitObjRandom(src="obj", amin=s[0], amax=s[1], phirange=s[2])
                else:
                    obj = InitObjRandom(src=fftshift(obj), amin=s[0], amax=s[1], phirange=s[2])
        print("Finished initializing object ")
        return obj

    def prepare_cdi(self):
        """
        Prepare CDI object from input parameters. If self.cdi already exists, it is re-used, and
        only the initial object and support are updated. To save memory, self.iobs and self.mask
        are set to None.

        :return: nothing. Creates or updates self.cdi object.
        """
        t0 = timeit.default_timer()
        obj = self.init_object()
        self.timings['prepare_cdi: init_object'] = timeit.default_timer() - t0
        t1 = timeit.default_timer()

        sup = None
        if isinstance(self.support, np.ndarray):
            sup = fftshift(self.support)

        if self.cdi is not None:
            # If the object is supplied as an array, the support may be initialised
            # from the support.
            if isinstance(obj, np.ndarray):
                self.cdi.set_obj(obj)

            if isinstance(sup, np.ndarray):
                self.cdi.set_support(sup)
            elif isinstance(self.support, OperatorCDI):
                self.cdi = self.support * self.cdi

            # The object may be initialised from the support
            if isinstance(obj, OperatorCDI):
                self.cdi = obj * self.cdi

            # Reset the PSF
            self.cdi.init_psf(model=None)
        else:
            pix = self.params['pixel_size_detector']
            d = self.params['detector_distance']
            w = self.params['wavelength']
            m = self.mask
            if m is not None:
                m = fftshift(m)
                self.mask = None
            iobs = fftshift(self.iobs)
            self.iobs = None
            gc.collect()
            self.cdi = CDI(iobs=iobs, support=sup, mask=m, pixel_size_detector=pix,
                           wavelength=w, detector_distance=d, obj=None)
            self._algo_s = ""

            # If the object is supplied as an array, the support may be initialised
            # from the support.
            if isinstance(obj, np.ndarray):
                self.cdi.set_obj(obj)

            if isinstance(self.support, OperatorCDI):
                self.cdi = self.support * self.cdi

            # The object may be initialised from the support
            if isinstance(obj, OperatorCDI):
                self.cdi = obj * self.cdi

            gc.collect()
            self.timings['prepare_cdi: create CDI'] = timeit.default_timer() - t1

            t1 = timeit.default_timer()
            if self.params['free_pixel_mask'] is not None:
                filename = self.params['free_pixel_mask']
                print('Loading free pixel mask from: ', filename)
                ext = os.path.splitext(filename)[-1]
                if ext == '.npy':
                    free_pixel_mask = np.load(filename).astype(np.bool)
                elif ext == '.tif' or ext == '.tiff':
                    frames = Image.open(filename)
                    frames.seek(0)
                    free_pixel_mask = np.array(frames).astype(np.bool)
                elif ext == '.mat':
                    a = list(loadmat(filename).values())
                    for v in a:
                        if np.size(v) == iobs.size:
                            # Avoid matlab strings and attributes, and get the array
                            free_pixel_mask = np.array(v).astype(np.bool)
                            break
                elif ext == '.npz':
                    d = np.load(filename)
                    if 'free_pixel_mask' in d.keys():
                        free_pixel_mask = d['free_pixel_mask'].astype(np.bool)
                    else:
                        # Assume only the mask is in the datafile
                        for k, v in d.items():
                            if v.shape == iobs.shape:
                                free_pixel_mask = v.astype(np.bool)
                            break
                elif ext == '.edf':
                    free_pixel_mask = fabio.open(filename).data.astype(np.bool)
                elif ext == '.cxi':
                    fh5 = h5py.File(filename, mode='r')
                    if '/entry_last/image_1/process_1/configuration' in fh5:
                        free_pixel_mask = fh5['/entry_last/image_1/process_1/configuration/free_pixel_mask'][()]
                    else:
                        raise CDIRunnerException("Supplied free_pixel_mask=%s, but /entry_last/image_1/process_1/"
                                                 "configuration/free_pixel_mask not found !" % filename)
                else:
                    raise CDIRunnerException("Supplied mask=%s, but format not recognized (recognized: .npy,"
                                             ".npz, .tif, .edf, .mat)" % filename)
                self.cdi.set_free_pixel_mask(free_pixel_mask, verbose=True, shift=True)
            else:
                self.cdi = InitFreePixels(ratio=0.05, island_radius=3, exclude_zone_center=0.05,
                                          verbose=True, lazy=True) * self.cdi
            self.timings['prepare_cdi: init_free_pixels'] = timeit.default_timer() - t1

            if self.params['mask_interp'] is not None:
                t1 = timeit.default_timer()
                d, n = self.params['mask_interp']
                print("Interpolating masked pixels with InterpIobsMask(%d, %d)" % (d, n))
                self.cdi = InterpIobsMask(d, n) * self.cdi
                self.timings['prepare_cdi: mask_interp'] = timeit.default_timer() - t1

        self.timings['prepare_cdi'] = timeit.default_timer() - t0

    def prepare_processing_unit(self):
        """
        Prepare processing unit (CUDA, OpenCL, or CPU).

        Returns: nothing. Creates self.processing_unit

        """
        # Avoid calling again select_gpu()
        if default_processing_unit.cu_device is None and default_processing_unit.cl_device is None:
            s = "CDI runner: preparing processing unit"
            if self.params['gpu'] is not None:
                s += " [given GPU name: %s]" % str(self.params['gpu'])
            print(s)
            try:
                default_processing_unit.select_gpu(gpu_name=self.params['gpu'], verbose=True)
            except Exception as ex:
                s0 = "\n  original error: " + str(ex)
                if self.params['gpu'] is not None:
                    s = "Failed initialising GPU. Please check GPU name [%s] or CUDA/OpenCL installation"
                    raise CDIRunnerException(s % str(self.params['gpu']) + s0)
                else:
                    raise CDIRunnerException(
                        "Failed initialising GPU. Please check GPU name or CUDA/OpenCL installation" + s0)
        if default_processing_unit.pu_language == 'cpu':
            raise CDIRunnerException("CUDA or OpenCL or GPU not available - you need a GPU to use pynx.CDI !")

        self.processing_unit = default_processing_unit

    def get_psf(self):
        """
        Get the PSF parameters from self.params['psf']

        :return: a tuple (model, fwhm, eta, update_psf) with the PSF parameters,
          or (psf, update_psf) if the PSF array was loaded from a file.
        """
        if self.params['psf'] is False:
            False
        elif self.params['psf'] is True:
            return "pseudo-voigt", 1, 0.05, 5
        v = self.params['psf'].split(',')
        update_psf = 5
        if len(v[0]) > 4:
            psf = None
            if v[0][-4:].lower() == '.npz':
                d = np.load(v[0])
                if 'psf' in d.keys():
                    psf = d['psf']
                elif 'PSF' in d.keys():
                    psf = d['PSF']
                else:
                    # Assume only the PSF is in the datafile
                    for k, v in d.items():
                        psf = v
                        if isinstance(psf, np.ndarray):
                            break
                print("Loaded PSF from %s with shape:" % v[0], psf.shape)
                if psf is None:
                    raise CDIRunnerException("Could not find PSF array in: ", v[0])
                if len(v) > 1:
                    update_psf = int(v[1])
                return psf, update_psf
            elif v[0][-4:].lower() == '.cxi':
                h = h5py.File(v[0], mode='r')
                for e in ['entry_last/image_1/instrument_1/detector_1/point_spread_function0',
                          'entry_1/image_1/instrument_1/detector_1/point_spread_function0']:
                    if e in h:
                        psf = h[e][()]
                print("Loaded PSF from %s with shape:" % v[0], psf.shape)
                if psf is None:
                    raise CDIRunnerException("Could not find PSF array in: ", v[0])
                if len(v) > 1:
                    update_psf = int(v[1])
                return psf, update_psf
        # We got there, so no PSF was imported from a file, use a model
        model = v[0]
        fwhm = float(v[1])
        eta = 0.05
        if 'voigt' in model.lower() or 'pseudo' in model.lower():
            eta = float(v[2])
            if len(v) == 4:
                update_psf = int(v[3])
        elif len(v) == 3:
            update_psf = int(v[2])
        if 'voigt' in model.lower() or 'pseudo' in model.lower():
            print("Chosen PSF model: %s, fwhm=%5.2f")
        return model, fwhm, eta, update_psf

    def run(self, file_name=None, run_n=1):
        """
        Main work function. Will run selected algorithms according to parameters

        :param file_name: output file_name. If None, the result is not saved.
        :param run_n: the run number
        :return:
        """
        # Starting point of the history of algorithm & timings
        self.cdi.reset_history()
        if self.params['algorithm'] is not None:
            self.run_algorithm(self.params['algorithm'], file_name=file_name, run_n=run_n)
            return

        # We did not get an algorithm string, so create the chain of operators from individual parameters
        nb_raar = self.params['nb_raar']
        nb_hio = self.params['nb_hio']
        nb_er = self.params['nb_er']
        nb_ml = self.params['nb_ml']
        positivity = self.params['positivity']
        support_only_shrink = self.params['support_only_shrink']
        beta = self.params['beta']
        detwin = self.params['detwin']
        psf = self.params['psf']
        live_plot = self.params['live_plot']
        support_update_period = self.params['support_update_period']
        support_smooth_width_begin = self.params['support_smooth_width_begin']
        support_smooth_width_end = self.params['support_smooth_width_end']
        support_threshold = self.params['support_threshold']
        support_threshold_method = self.params['support_threshold_method']
        support_post_expand = self.params['support_post_expand']
        support_update_border_n = self.params['support_update_border_n']
        confmin = self.params['confidence_interval_factor_mask_min']
        confmax = self.params['confidence_interval_factor_mask_max']
        verbose = self.params['verbose']
        zero_mask = self.params['zero_mask']
        min_fraction = self.params['support_fraction_min']
        max_fraction = self.params['support_fraction_max']
        if live_plot is True:
            live_plot = verbose
        elif live_plot is False:
            live_plot = 0

        if nb_ml > 0 and psf:
            print("ML deactivated - PSF is unimplemented in ML")
            nb_ml = 0

        print('No algorithm chain supplied. Proceeding with the following parameters:')
        print(' %30s = ' % 'nb_hio', nb_hio)
        print(' %30s = ' % 'nb_raar', nb_raar)
        print(' %30s = ' % 'nb_er', nb_er)
        print(' %30s = ' % 'nb_ml', nb_ml)
        print(' %30s = ' % 'positivity', positivity)
        print(' %30s = ' % 'support_only_shrink', support_only_shrink)
        print(' %30s = ' % 'beta', beta)
        print(' %30s = ' % 'detwin', detwin)
        print(' %30s = ' % 'live_plot', live_plot)
        print(' %30s = ' % 'support_update_period', support_update_period)
        print(' %30s = ' % 'support_smooth_width_begin', support_smooth_width_begin)
        print(' %30s = ' % 'support_smooth_width_end', support_smooth_width_end)
        print(' %30s = ' % 'support_threshold', support_threshold)
        print(' %30s = ' % 'support_threshold_method', support_threshold_method)
        print(' %30s = ' % 'support_post_expand', support_post_expand)
        if support_update_border_n:
            print(' %30s = ' % 'support_update_border_n', support_update_border_n)
        print(' %30s = ' % 'support_fraction_min/max', min_fraction, max_fraction)
        print(' %30s = ' % 'confidence_interval_factor_mask_min', confmin)
        print(' %30s = ' % 'confidence_interval_factor_mask_max', confmax)
        print(' %30s = ' % 'zero_mask', zero_mask)
        print(' %30s = ' % 'verbose', verbose)
        print(' %30s = ' % 'psf', psf)

        # Write an algorithm string based on the given parameters
        # Begin by listing all cycles for which there is a new operator applied
        cycles = [nb_hio + nb_raar + nb_er + nb_ml]
        if nb_hio > 0:
            cycles.append(nb_hio)
        if nb_raar > 0:
            cycles.append(nb_hio + nb_raar)
        if nb_er > 0:
            cycles.append(nb_hio + nb_raar + nb_er)
        if nb_ml > 0:
            cycles.append(nb_hio + nb_raar + nb_er + nb_ml)
        if support_update_period > 0:
            cycles += list(range(0, nb_hio + nb_raar + nb_er, support_update_period))
        if detwin:
            detwin_cycles = int((nb_hio + nb_raar) // 4)
            cycles.append(detwin_cycles)
        if psf:
            psf_0 = int((nb_raar + nb_hio) * 0.66)
            cycles += [psf_0]

        if type(zero_mask) is str:
            if zero_mask.lower() == 'auto':
                zm = 1
                zm_cycle = int((nb_hio + nb_raar) * 0.6)
                cycles.append(zm_cycle)
            else:
                zm_cycle = None
                if zero_mask.lower() == 'true' or zero_mask.lower() == '1':
                    zm = True
                else:
                    zm = False
        else:
            zm_cycle = None
            zm = zero_mask  # Can be 0, 1, True, False..

        cycles.sort()

        algo = 1  # Start with unity operator
        algo_str = []
        while len(cycles) > 1:
            algo1 = 1
            i0, i1 = cycles[:2]
            n = i1 - i0
            cycles.pop(0)

            if i0 == i1:
                continue
            update_psf = 0
            psf_filter = self.params['psf_filter']
            if psf:
                if i1 == psf_0:
                    psf = self.get_psf()
                    if len(psf) == 2:
                        algo1 = InitPSF(psf=psf[0]) * algo1
                    else:
                        algo1 = InitPSF(model=psf[0], fwhm=psf[1], eta=psf[2])
                    print('Activating PSF after %d cycles, updated every %d cycles' % (psf_0, psf[-1]))
                if i1 >= psf_0:
                    update_psf = psf[-1]

            if zm_cycle is not None:
                if i1 == zm_cycle:
                    # Free masked pixels
                    zm = 0
                    print('Switching from zero_mask=1 to 0 after %d cycles' % zm_cycle)
            if i1 <= nb_hio:
                algo1 = HIO(beta=beta, positivity=positivity, calc_llk=verbose, show_cdi=live_plot,
                            zero_mask=zm, confidence_interval_factor_mask_min=confmin,
                            confidence_interval_factor_mask_max=confmax, update_psf=update_psf,
                            psf_filter=psf_filter) ** n * algo1
            elif nb_raar > 0 and i1 <= (nb_hio + nb_raar):
                algo1 = RAAR(beta=beta, positivity=positivity, calc_llk=verbose, show_cdi=live_plot,
                             zero_mask=zm, confidence_interval_factor_mask_min=confmin,
                             confidence_interval_factor_mask_max=confmax, update_psf=update_psf,
                             psf_filter=psf_filter) ** n * algo1
            elif nb_er > 0 and i1 <= (nb_hio + nb_raar + nb_er):
                algo1 = ER(positivity=positivity, calc_llk=verbose, show_cdi=live_plot, zero_mask=zm,
                           confidence_interval_factor_mask_min=confmin,
                           confidence_interval_factor_mask_max=confmax, update_psf=update_psf,
                           psf_filter=psf_filter) ** n * algo1
            elif nb_ml > 0 and i1 <= (nb_hio + nb_raar + nb_er + nb_ml):
                algo1 = ML(calc_llk=verbose, show_cdi=live_plot) ** n * FourierApplyAmplitude(calc_llk=True,
                                                                                              zero_mask=zm) * algo1

            if support_update_period > 0:
                if i1 % support_update_period == 0 and len(cycles) > 1:
                    s = support_smooth_width_begin, support_smooth_width_end, nb_raar + nb_hio
                    algo1 = SupportUpdate(threshold_relative=support_threshold, smooth_width=s,
                                          force_shrink=support_only_shrink, method=support_threshold_method,
                                          post_expand=support_post_expand,
                                          update_border_n=support_update_border_n, min_fraction=min_fraction,
                                          max_fraction=max_fraction) * algo1

            if detwin:
                if i1 == detwin_cycles:
                    if i1 <= nb_hio:
                        algo1 = DetwinHIO(beta=beta, positivity=False, nb_cycle=10, zero_mask=zm) * algo1
                    else:
                        algo1 = DetwinRAAR(beta=beta, positivity=False, nb_cycle=10, zero_mask=zm) * algo1

            algo_str.append(str(algo1))
            if algo is None:
                algo = algo1
            else:
                algo = algo1 * algo

        # Finish by FourierApplyAmplitude
        confmin = self.params['confidence_interval_factor_mask_min']
        confmax = self.params['confidence_interval_factor_mask_max']
        algo = FourierApplyAmplitude(calc_llk=True, zero_mask=zm,
                                     confidence_interval_factor_mask_max=confmax,
                                     confidence_interval_factor_mask_min=confmin, obj_stats=True) * algo

        # algo = PRTF(file_name='%s-PRTF.png' % os.path.splitext(file_name)[0], fig_title=file_name) * algo

        # print("Algorithm chain: ", algo)

        # More pretty printing:
        s = ''
        while len(algo_str):
            s0 = algo_str.pop(0)
            n = 1
            while len(algo_str):
                if algo_str[0] == s0:
                    del algo_str[0]
                    n += 1
                else:
                    break
            if s != '':
                s = ' * ' + s
            if n > 1:
                s = '(%s)**%d' % (s0, n) + s
            else:
                s = s0 + s

        # Reformat chain so that it can be re-used from the command-line
        s = s.replace('()', '')
        s = s.replace('SupportUpdate', 'Sup')
        if psf:
            s = s.replace('*InitPSF *', ',psf=%d,' % psf[-1])

        print("Algorithm chain: ", s)

        # Now execute the recipe !
        t0 = timeit.default_timer()
        self.cdi = algo * self.cdi

        print("\nTotal elapsed time for algorithms: %8.2fs" % (timeit.default_timer() - t0))
        calc_throughput(self.cdi, verbose=True)
        if file_name is not None:
            self.save_result(file_name)
            if self.params['save_plot']:
                self.save_plot(os.path.splitext(file_name)[0] + '.png', algo_string=s)

    def run_algorithm(self, algo_string, file_name=None, run_n=1):
        """
        Run a single or suite of algorithms in a given run.

        :param algo_string: a single or suite of algorithm steps to use, which can either correspond to a change
                            of parameters, i.e. 'beta=0.5', 'support_threshold=0.3', 'positivity=1',
                            or operators which should be applied to the cdi object, such as:
                            'hio**20': 20 cycles of HIO
                            'er': a single cycle or error reduction
                            'detwinhio**20': 20 cycles of hio after halving the support
                            'er**20*hio**100': 100 cycles of hio followed by 20 cycles of ER
                            '(sup*er**10*raar**100)**10': 10 times [100 cycles of RAAR + 10 ER + Support update]
        :param file_name: output file_name. If None, the result is not saved.
        :param run_n: the run number
        """
        algo_split = algo_string.split(',')
        algo_split.reverse()
        print(algo_split)
        t0 = timeit.default_timer()
        update_psf = 0
        for algo in algo_split:
            if self._algo_s == "":
                self._algo_s += algo
            else:
                self._algo_s += ',' + algo
            print("\n", "#" * 100, "\n#", "\n#         Run: %g , Algorithm: %s\n#\n" % (run_n, algo), "#" * 100)
            realoptim = False  # Is this a real optimization (HIO, ER, ML, RAAR,...), or just a change of parameter ?
            if algo.lower().find('beta=') >= 0:
                self.params['beta'] = float(algo.lower().split('beta=')[-1])
            elif algo.lower().find('support_threshold=') >= 0:
                self.params['support_threshold'] = float(algo.lower().split('support_threshold=')[-1])
            elif algo.lower().find('support_threshold_mult=') >= 0:
                self.params['support_threshold'] *= float(algo.lower().split('support_threshold_mult=')[-1])
            elif algo.lower().find('gps_inertia=') >= 0:
                self.params['gps_inertia'] = float(algo.lower().split('gps_inertia=')[-1])
            elif algo.lower().find('gps_t=') >= 0:
                self.params['gps_t'] = float(algo.lower().split('gps_t=')[-1])
            elif algo.lower().find('gps_s=') >= 0:
                self.params['gps_s'] = float(algo.lower().split('gps_s=')[-1])
            elif algo.lower().find('gps_sigma_o=') >= 0:
                self.params['gps_sigma_o'] = float(algo.lower().split('gps_sigma_o=')[-1])
            elif algo.lower().find('gps_sigma_f=') >= 0:
                self.params['gps_sigma_f'] = float(algo.lower().split('gps_sigma_f=')[-1])
            elif algo.lower().find('positivity=') >= 0:
                self.params['positivity'] = int(algo.lower().split('positivity=')[-1])
            elif algo.lower().find('zero_mask=') >= 0:
                self.params['zero_mask'] = int(algo.lower().split('zero_mask=')[-1])
            elif algo.lower().find('support_only_shrink=') >= 0:
                self.params['support_only_shrink'] = int(algo.lower().split('support_only_shrink=')[-1])
            elif algo.lower().find('positivity=') >= 0:
                self.params['positivity'] = float(algo.lower().split('positivity=')[-1])
            elif algo.lower().find('support_smooth_width_begin=') >= 0:
                self.params['support_smooth_width_begin'] = float(algo.lower().split('support_smooth_width_begin=')[-1])
            elif algo.lower().find('support_smooth_width_end=') >= 0:
                self.params['support_smooth_width_end'] = float(algo.lower().split('positivity=')[-1])
            elif algo.lower().find('support_post_expand=') >= 0:
                s = algo.lower().split('support_post_expand=')[-1].replace('#', ',')
                self.params['support_post_expand'] = eval(s)
            elif algo.lower().find('support_threshold_method=') >= 0:
                self.params['support_threshold_method'] = algo.lower().split('support_post_expand=')[-1]
            elif algo.lower().find('support_update_border_n=') >= 0:
                self.params['support_update_border_n'] = int(algo.lower().split('support_update_border_n=')[-1])
            elif algo.lower().find('confidence_interval_factor_mask_min=') >= 0:
                self.params['confidence_interval_factor_mask_min'] = \
                    algo.lower().split('confidence_interval_factor_mask_min=')[-1]
            elif algo.lower().find('confidence_interval_factor_mask_max=') >= 0:
                self.params['confidence_interval_factor_mask_max'] = \
                    algo.lower().split('confidence_interval_factor_mask_max=')[-1]
            # elif algo.lower().find('crop=') >= 0:
            #     b = algo.lower().split('crop=')[-1]
            #     if len(b) > 1:
            #         # parameters per axis have been given, e.g. crop=122
            #         b = [int(v) for v in b]
            #     else:
            #         b = int(b)
            #     self.cdi.set_crop(b)
            #     # self.cdi = ScaleObj(method="F", verbose=True) * self.cdi
            # elif algo.lower().find('bin=') >= 0:
            #     b = algo.lower().split('bin=')[-1]
            #     if len(b) > 1:
            #         # parameters per axis have been given, e.g. bin=122
            #         b = [int(v) for v in b]
            #     else:
            #         b = int(b)
            #     self.cdi.set_bin(b)
            #     # TODO: rescale object to take into account bin/skip
            #     # self.cdi = ScaleObj(method="F", verbose=True) * self.cdi
            # elif algo.lower().find('skip=') >= 0:
            #     b = algo.lower().split('skip=')[-1]
            #     if len(b) > 1:
            #         # parameters per axis have been given, e.g. skip=122
            #         b = [int(v) for v in b]
            #         if len(b) == self.cdi.iobs.ndim:
            #             # Skip is used, but the shifts were not given select them randomly
            #             for i in range(self.cdi.iobs.ndim):
            #                 b.append(np.random.randint(0, b[i]))
            #     else:
            #         b = int(b)
            #     print("Using bin(skip) parameters:", b)
            #     self.cdi.set_bin(b)
            #     # TODO: rescale object to take into account bin/skip
            #     # self.cdi = ScaleObj(method="F", verbose=True) * self.cdi
            elif algo.lower().find('upsample=') >= 0:
                self.cdi.set_upsample(eval(algo.lower().split('upsample=')[-1]))
                # self.cdi = ScaleObj(method="F", verbose=True) * self.cdi
            elif algo.lower().find('bin=') >= 0:
                self.cdi.set_bin(eval(algo.lower().split('bin=')[-1]))
            elif algo.lower().find('crop=') >= 0:
                self.cdi.set_crop(eval(algo.lower().split('crop=')[-1]))
            elif 'psf_init=' in algo.lower():
                tmp = algo.lower().split('psf_init=')[-1].split("@")
                fwhm = float(eval(tmp[1]))
                if len(tmp) > 2:
                    eta = float(eval(tmp[2]))
                    print("Init PSF with model = %s, fwhm=%5.2f pixels" % (tmp[0], fwhm))
                else:
                    eta = 0.05
                    print("Init PSF with model = %s, fwhm=%5.2f pixels, eta = %5.3f" % (tmp[0], fwhm, eta))
                self.cdi = InitPSF(model=tmp[0], fwhm=fwhm, eta=eta) * self.cdi
                if update_psf == 0:
                    update_psf = 5
                    print("Will update PSF every %d cycles" % update_psf)
            elif 'psf=' in algo.lower():
                update_psf = int(eval(algo.lower().split('psf=')[-1]))
            elif 'psf_filter' in algo.lower():
                psf_filter = algo.lower().split('psf_filter=')[-1]
                if psf_filter in ['hann', 'tukey']:
                    self.params['psf_filter'] = psf_filter
                else:
                    self.params['psf_filter'] = None
            elif algo.lower().find('verbose=') >= 0:
                self.params['verbose'] = int(algo.lower().split('verbose=')[-1])
            elif algo.lower().find('live_plot=') >= 0:
                self.params['live_plot'] = int(algo.lower().split('live_plot=')[-1])
            elif algo.lower().find('fig_num=') >= 0:
                self.params['fig_num'] = int(algo.lower().split('fig_num=')[-1])
            elif algo.lower().find('manual') >= 0:
                pass  # Dummy type of algorithm
            else:
                # Not a keyword, so this must be an algorithm to run. Finally !
                realoptim = True
                positivity = self.params['positivity']
                support_only_shrink = self.params['support_only_shrink']
                beta = self.params['beta']
                support_smooth_width_begin = self.params['support_smooth_width_begin']
                support_smooth_width_end = self.params['support_smooth_width_end']
                support_smooth_width_relax_n = self.params['support_smooth_width_relax_n']
                smooth_width = support_smooth_width_begin, support_smooth_width_end, support_smooth_width_relax_n
                support_threshold = self.params['support_threshold']
                gps_inertia = self.params['gps_inertia']
                gps_t = self.params['gps_t']
                gps_s = self.params['gps_s']
                gps_sigma_o = self.params['gps_sigma_o']
                gps_sigma_f = self.params['gps_sigma_f']
                support_threshold_method = self.params['support_threshold_method']
                support_post_expand = self.params['support_post_expand']
                support_update_border_n = self.params['support_update_border_n']
                min_fraction = self.params['support_fraction_min']
                max_fraction = self.params['support_fraction_max']
                confmin = self.params['confidence_interval_factor_mask_min']
                confmax = self.params['confidence_interval_factor_mask_max']
                verbose = int(self.params['verbose'])
                live_plot = self.params['live_plot']
                fig_num = self.params['fig_num']
                zm = self.params['zero_mask']
                psf_filter = self.params['psf_filter']
                if type(zm) is str:
                    if zm.lower() == 'auto':
                        # TODO: better handle zero_mask='auto' for custom algorithms ?
                        zm = False
                    else:
                        if zm.lower() == 'true' or zm.lower() == '1':
                            zm = True
                        else:
                            zm = False
                if int(live_plot) == 1:  # That's for live_plot=True
                    live_plot = verbose

                # Create elementary operators
                fap = FourierApplyAmplitude(calc_llk=True, zero_mask=zm,
                                            confidence_interval_factor_mask_max=confmax,
                                            confidence_interval_factor_mask_min=confmin)
                er = ER(positivity=positivity, calc_llk=verbose, show_cdi=live_plot, fig_num=fig_num, zero_mask=zm,
                        confidence_interval_factor_mask_min=confmin,
                        confidence_interval_factor_mask_max=confmax, update_psf=update_psf, psf_filter=psf_filter)
                hio = HIO(beta=beta, positivity=positivity, calc_llk=verbose, show_cdi=live_plot, fig_num=fig_num,
                          zero_mask=zm, confidence_interval_factor_mask_min=confmin,
                          confidence_interval_factor_mask_max=confmax, update_psf=update_psf, psf_filter=psf_filter)
                raar = RAAR(beta=beta, positivity=positivity, calc_llk=verbose, show_cdi=live_plot, fig_num=fig_num,
                            zero_mask=zm, confidence_interval_factor_mask_min=confmin,
                            confidence_interval_factor_mask_max=confmax, update_psf=update_psf, psf_filter=psf_filter)
                gps = GPS(inertia=gps_inertia, t=gps_t, s=gps_s, sigma_f=gps_sigma_f, sigma_o=gps_sigma_o,
                          positivity=positivity, calc_llk=verbose, show_cdi=live_plot, fig_num=fig_num, zero_mask=zm,
                          confidence_interval_factor_mask_min=confmin,
                          confidence_interval_factor_mask_max=confmax, update_psf=update_psf, psf_filter=psf_filter)
                detwinhio = DetwinHIO(detwin_axis=0, beta=beta, positivity=positivity, zero_mask=zm)
                detwinhio1 = DetwinHIO(detwin_axis=1, beta=beta, positivity=positivity, zero_mask=zm)
                detwinhio2 = DetwinHIO(detwin_axis=2, beta=beta, positivity=positivity, zero_mask=zm)
                detwinraar = DetwinRAAR(detwin_axis=0, beta=beta, positivity=positivity, zero_mask=zm)
                detwinraar1 = DetwinRAAR(detwin_axis=1, beta=beta, positivity=positivity, zero_mask=zm)
                detwinraar2 = DetwinRAAR(detwin_axis=2, beta=beta, positivity=positivity, zero_mask=zm)
                cf = CF(positivity=positivity, calc_llk=verbose, show_cdi=live_plot, fig_num=fig_num, zero_mask=zm,
                        update_psf=update_psf, psf_filter=psf_filter)
                ml = ML(calc_llk=verbose, show_cdi=live_plot, fig_num=fig_num) * fap
                initpsf = InitPSF()
                sup = SupportUpdate(threshold_relative=support_threshold, smooth_width=smooth_width,
                                    force_shrink=support_only_shrink, post_expand=support_post_expand,
                                    method=support_threshold_method, update_border_n=support_update_border_n,
                                    min_fraction=min_fraction, max_fraction=max_fraction)
                supportupdate = sup
                showcdi = ShowCDI(fig_num=fig_num)

                try:
                    ops = eval(algo.lower())
                    self.cdi = ops * self.cdi
                except SupportTooSmall:
                    raise
                except SupportTooLarge:
                    raise
                except Exception as ex:
                    print(traceback.format_exc())
                    print(self.help_text)
                    print('\n\n Caught exception for scan %d: %s    \n' % (self.scan, str(ex)))
                    print('Could not interpret operator-based algorithm: ', algo)
                    # TODO: print valid examples of algorithms
                    sys.exit(1)
            if self.params['save'] == 'all' and realoptim and file_name is not None:
                # Finish with FourierApplyAmplitude
                zm = self.params['zero_mask']
                if type(zm) is str:
                    if zm.lower() == 'auto':
                        zm = False
                    else:
                        if zm.lower() == 'true' or zm.lower() == '1':
                            zm = True
                        else:
                            zm = False
                confmin = self.params['confidence_interval_factor_mask_min']
                confmax = self.params['confidence_interval_factor_mask_max']
                self.cdi = FourierApplyAmplitude(calc_llk=True, zero_mask=zm,
                                                 confidence_interval_factor_mask_max=confmax,
                                                 confidence_interval_factor_mask_min=confmin) * self.cdi
                self.save_result(file_name)

        self.timings['run_algorithm (algorithms)'] = timeit.default_timer() - t0
        t1 = timeit.default_timer()
        if self.params['save'] != 'all' and file_name is not None:
            # Finish with FourierApplyAmplitude
            zm = self.params['zero_mask']
            if type(zm) is str:
                if zm.lower() == 'auto':
                    zm = False
                else:
                    if zm.lower() == 'true' or zm.lower() == '1':
                        zm = True
                    else:
                        zm = False
            confmin = self.params['confidence_interval_factor_mask_min']
            confmax = self.params['confidence_interval_factor_mask_max']
            self.cdi = FourierApplyAmplitude(calc_llk=True, zero_mask=zm,
                                             confidence_interval_factor_mask_max=confmax,
                                             confidence_interval_factor_mask_min=confmin) * self.cdi
            self.save_result(file_name)

        print("\nTotal elapsed time for algorithms & saving: %8.2fs" % (timeit.default_timer() - t0))

        calc_throughput(self.cdi, verbose=True)
        if file_name is not None and self.params['save_plot']:
            self.save_plot(os.path.splitext(file_name)[0] + '.png', algo_string=algo_string)
        self.timings['run_algorithm (final save)'] = timeit.default_timer() - t1

    def save_result(self, file_name=None, verbose=True):
        """ Save the results (object, support,...) at the end of a run

        :param file_name: the filename to save the data to. If its extension is either npz or cxi, this will
                          override params['output_format']. Otherwise, the extension will be replaced. Note that
                          the full filename will be appended with the llk value.
        :return:
        """
        if self.params['output_format'].lower() == 'none':
            return

        if file_name is None:
            file_name = "Result.%s" % self.params['output_format']

        filename, ext = os.path.splitext(os.path.split(file_name)[1])

        if len(ext) < 2:
            ext = "." + self.params['output_format']

        llk = self.cdi.get_llk(normalized=True)

        if llk is not None and self.params['save'].lower() != 'all':
            filename += "_LLKf%08.4f_LLK%08.4f" % (llk[3], llk[0])

        filename += "_SupportThreshold%7.5f" % self.params['support_threshold']

        if ext not in ['.npz', '.cxi']:
            warnings.warn("CDI.save_result(): output format (%s) not understood, defaulting to CXI" % ext)
            ext = '.cxi'
        filename += ext

        if os.path.isfile(filename) and self.params['save'].lower() != 'all':
            warnings.warn("CDI.save_result(): output file already exists, not overwriting: %s" % filename)
            # raise CDIRunnerException("ERROR: output file already exists, no overwriting: %s" % filename)
            filename = filename[:-4]
            nn = 1
            while os.path.isfile("%s_%02d%s" % (filename, nn, ext)):
                nn += 1
            filename = "%s_%02d%s" % (filename, nn, ext)

        if verbose:
            print("Saving result to: %s" % filename)

        if ext == '.npz':
            np.savez_compressed(filename, obj=self.cdi.get_obj(shift=True))
        else:
            process = {}

            if self.params['algorithm'] is not None:
                process["algorithm"] = self.params['algorithm']

            params_string = ""
            for p in self.params.items():
                k, v = p
                if v is not None and k not in ['output_format']:
                    params_string += "%s = %s\n" % (k, str(v))

            process["note"] = {'configuration parameters used for phasing': params_string}

            append = False
            if self.params['save'].lower() == 'all':
                append = True

            self.cdi.save_obj_cxi(filename, sample_name=self.params['sample_name'], experiment_id=None,
                                  instrument=self.params['instrument'], note=self.params['note'],
                                  crop=int(self.params['crop_output']), process_notes=process,
                                  process_parameters=self.params, append=append)
            if os.name == 'posix':
                try:
                    sf = os.path.split(filename)
                    os.system('ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest.cxi')))
                except:
                    pass

        llk = self.cdi.get_llk()
        d = {'file_name': filename, 'llk_poisson': llk[0], 'llk_gaussian': llk[1], 'llk_euclidian': llk[2],
             'llk_poisson_free': llk[3], 'llk_gaussian_free': llk[4], 'llk_euclidian_free': llk[5],
             'nb_point_support': self.cdi.nb_point_support, 'obj2_out': self.cdi._obj2_out}
        replaced = False
        for n, v in enumerate(self.saved_results):
            if v['file_name'] == filename:
                self.saved_results[n] = d
                replaced = True
                break
        if replaced is False:
            self.saved_results.append(d)
        if verbose:
            print("To view the result file (HDF5/CXI or npz), use: silx view %s" % filename)

    def save_plot(self, file_name, algo_string=None):
        """
        Save a final plot for the object (only supported for 3D data)
        :param file_name: filename for the plot
        :param algo_string: will be used as subtitle
        :return: nothing
        """
        if self.cdi is None:
            return
        if self.cdi.iobs.ndim != 3:
            print("save_plot is only supported for 3D CDI")
            return
        print("Saving plot to: ", file_name)
        p = {'cwd': os.getcwd()}
        for k, v in self.params.items():
            if v is not None and k not in ['live_plot', 'verbose', 'data2cxi', 'output_format', 'save_plot',
                                           'fig_num']:
                p[k] = v
        if isinstance(self.params['save_plot'], str):
            plot_type = self.params['save_plot']
        else:
            if self.params['positivity']:
                plot_type = 'abs'
            else:
                plot_type = 'rgba'
        title = self.params['data']
        if 'specfile' in self.params:
            if self.params['specfile'] is not None and self.scan is not None:
                title = ("%s  #%d" % (self.params['specfile'], self.scan))

        show_cdi(self.cdi, params=p, save_plot=file_name, display_plot=False, figsize=(10, 10),
                 crop=2, plot_type=plot_type, title=title, subtitle=algo_string)

        if os.name == 'posix':
            try:
                sf = os.path.split(file_name)
                os.system('ln -sf "%s" %s' % (sf[1], os.path.join(sf[0], 'latest.png')))
            except:
                pass


class CDIRunner:
    """
    Class to process CDI data, with parameters from the command-line or from a text file.
    """

    def __init__(self, argv, params, cdi_runner_scan_class):
        """

        :param argv: the command-line parameters
        :param params: parameters for the optimization, with some default values.
        :param ptycho_runner_scan_class: the class to use to run the analysis.
        """
        self.params = copy.deepcopy(params)
        self.argv = argv
        self.CDIRunnerScan = cdi_runner_scan_class
        self.help_text = helptext_generic
        self.parameters_file_name = None
        self.timings = {}

        self.mpi_master = True  # True even if MPI is not used
        if MPI is not None:
            self.mpic = MPI.COMM_WORLD
            self.mpi_master = self.mpic.Get_rank() == 0
            self.mpi_size = self.mpic.Get_size()
            self.mpi_rank = self.mpic.Get_rank()

        self.parse_argv()
        if 'help' not in self.argv and '--help' not in self.argv:
            self.check_params()

    def print(self, *args, **kwargs):
        """
        MPI-aware print function. Non-master processes will be muted
        :param args: args passed to print
        :param kwargs: kwrags passed to print
        :return: nothing
        """
        if self.mpi_master:
            print(*args, **kwargs)

    def parse_argv(self):
        """
        Parses all the arguments given on a command line,

        Returns: nothing

        """
        for arg in self.argv:
            s = arg.find('=')
            if s > 0 and s < (len(arg) - 1):
                k = arg[:s].lower()
                v = arg[s + 1:]
                if self.parse_arg(k, v) is False:
                    print("WARNING: argument not interpreted: %s=%s" % (k, v))
            elif arg.find('.txt') > 0:
                print('Using parameters file: ', arg)
                self.parse_parameters_file(arg)
            elif arg.find('pynx-') > 0:
                continue
            elif self.parse_arg(arg.lower()) is False:
                print("WARNING: argument not interpreted: %s" % (arg))

    def parse_parameters_file(self, filename):
        """
        Read parameters from a text file, written with one parameter per line as a python script
        :return: nothing. The parameters are accepted if understood, and stored in self.params
        """
        self.parameters_file_name = filename
        ll = open(filename).readlines()
        for l in ll:
            i = l.find('#')
            if i >= 0:
                l = l[:i]
            if len(l.strip()) < 4:
                continue
            if l.strip()[0] == '#':
                continue
            s = l.find('=')
            if s > 0 and s < (len(l) - 1):
                k = l[:s].lower().strip()
                v = l[s + 1:].strip()
                v = v.replace("'", "")
                v = v.replace('"', '')
                if self.parse_arg(k, v) is False:
                    print("WARNING: argument not interpreted: %s=%s" % (k, v))
            else:
                print("WARNING: argument not interpreted: %s" % (l))

    def parse_arg(self, k, v=None):
        """
        Interprets one parameter. Will
        :param k: string with the name of the parameter
        :param v: value of the parameter, or None if a keyword parameter
        :return: True if parameter could be interpreted, False otherwise
        """
        print(k, v)
        if k == 'liveplot':
            k = 'live_plot'
        if k == 'saveplot':
            k = 'save_plot'
        if k == 'maxsize':
            k = 'max_size'
        if k == 'roi':
            k = 'roi_user'
        if k == 'pixel_size_data':
            k = 'pixel_size_detector'
        if k == 'support_type':
            warnings.warn("'support_type=' is deprecated, use 'support=' instead", DeprecationWarning)
            k = 'support'
        if k in ['live_plot', 'psf', 'save_plot']:
            if v is None:
                self.params[k] = True
                return True
            elif type(v) is bool:
                self.params[k] = v
                return True
            elif type(v) is str:
                if v.lower() == 'true' or v.lower() == '1':
                    self.params[k] = True
                    return True
                elif v.lower() == 'false' or v.lower() == '0':
                    self.params[k] = False
                    return True
                elif k in ['live_plot']:
                    self.params[k] = int(v)
                    return True
                else:  # k in ['psf']:
                    self.params[k] = v
                    return True
            else:
                return False
        elif k in ['auto_center_resize', 'data2cxi', 'positivity', 'support_only_shrink', 'detwin', 'crop_output']:
            if v is None:
                self.params[k] = True
                return True
            elif type(v) is bool:
                self.params[k] = v
                return True
            elif type(v) is str:
                if v.lower() == 'true' or v.lower() == '1':
                    self.params[k] = True
                    return True
                else:
                    self.params[k] = False
                    return True
            else:
                return False
        elif k in ['detector_distance', 'pixel_size_data', 'pixel_size_detector', 'wavelength',
                   'support_smooth_width_begin', 'support_smooth_width_end', 'beta',
                   'support_autocorrelation_threshold', 'iobs_saturation', 'support_fraction_min',
                   'support_fraction_max', 'support_threshold_auto_tune_factor',
                   'nb_run_keep_max_obj2_out']:
            self.params[k] = float(v)
            return True
        elif k in ['support_threshold']:
            if ',' in v:
                self.params[k] = [float(val) for val in v.split(',')]  # a range is given
                self.params[k].sort()
            else:
                self.params[k] = float(v)  # single value
            return True
        elif k in ['verbose', 'nb_run', 'nb_run_keep', 'max_size', 'support_update_period', 'nb_raar', 'nb_hio',
                   'nb_er', 'nb_ml', 'support_smooth_width_relax_n', 'support_update_border_n']:
            self.params[k] = int(v)
            return True
        elif k in ['data', 'mask', 'support', 'object', 'rebin', 'support_threshold_method', 'save',
                   'output_format', 'support_size', 'note', 'instrument', 'sample_name', 'algorithm', 'zero_mask',
                   'free_pixel_mask', 'support_formula', 'mpi', 'crop', 'flatfield']:
            self.params[k] = v
            return True
        elif k in ['gpu']:
            # Allows several GPU (sub)strings to be listed
            g = v.split(',')
            # Use either a string or a list, to check if both cases are correctly processed
            if len(g) == 1:
                self.params[k] = g[0]
            else:
                self.params[k] = g
        elif k in "psf_filter":
            if v.lower() in ["hann", "tukey"]:
                self.params[k] = v.lower()
            else:
                self.params[k] = None
        elif k == 'confidence_interval_factor_mask':
            self.params['confidence_interval_factor_mask_min'], \
            self.params['confidence_interval_factor_mask_max'] = eval(v)
        elif k in ['support_post_expand', 'roi_user', 'mask_interp']:
            self.params[k] = eval(v)
            return True
        elif 'user_config' in k:
            # These parameters are only used to store some information about the process, and will be stored
            # in the cxi output
            self.params[k] = v
        else:
            return self.parse_arg_beamline(k, v)

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
        If 'help' is given as command-line parameter, the help text is printed and the program exits.

        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['psf'] is True:
            raise CDIRunnerException(
                'PSF: using "psf" (on the command-line or psf=True (input file) is not longer accepted. '
                'You must specify a model using e.g. psf=gaussian,0.5,10 (command-line) '
                'or  psf="pseudo-voigt,0.5,0.05,10" (using quotes in an input file). '
                'See the helptext for details and recommended values depending on the level of coherence.')

        self.check_params_beamline()

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamline

        Returns: Nothing. Will raise an exception if necessary
        """
        pass

    def process_scans(self):
        """
        Run all the analysis on the supplied scan list, unless 'help' is given as a
        command-line argument.

        :return: Nothing
        """
        t0 = timeit.default_timer()
        if 'help' in self.argv or '--help' in self.argv:
            self.print(self.help_text)
            sys.exit(0)

        if 'scan' in self.params:
            if isinstance(self.params['scan'], str):
                if 'range' in self.params['scan']:
                    # use range() function to specify a series of scans
                    vscan = eval(self.params['scan'])
                else:
                    # + is used to combine several scans in one, but there may still be a series of scans, e.g. '1+2,3+4'
                    vscan = self.params['scan'].split(',')
            else:
                # This could be None, the default value for self.params['scan'], to load a CXI, numpy, tiff file, etc...
                vscan = [self.params['scan']]
        else:
            vscan = [None]

        if MPI is not None:
            if 'scan' in self.params['mpi'] and self.mpi_size > 1:
                # Distribute the scan among the different cients, independently
                vscan = vscan[self.mpi_rank::self.mpi_size]
                print("MPI #%2d analysing scans:" % self.mpi_rank, vscan)
            elif 'run' in self.params['mpi'] and self.mpi_master:
                self.print("Using MPI: %s" % self.params['mpi'])

        warn_liveplot = False
        if self.params['live_plot'] and self.params['nb_run'] > 1:
            warn_liveplot = True
        if MPI is not None:
            if self.params['live_plot'] and self.mpi_size > 1:
                warn_liveplot = True
        if warn_liveplot:
            self.print("\n", "#" * 100, "\n#",
                       "\n# WARNING: you are using the LIVE_PLOT option with multiple runs or MPI \n#"
                       "\n#      this will slow down the optimisation !\n"
                       "\n#      Please remember to enable live_plot only for tests \n"
                       + "#" * 100)

        for scan in vscan:
            if scan is not None:
                self.print("\n", "#" * 100, "\n#", "\n#  CDI Scan: ", scan, "\n#\n", "#" * 100)
            try:
                # We can alter data filename with scan number
                data_orig = self.params['data']
                # Get prefix to save CXI and output file
                file_prefix = None
                if 'specfile' in self.params:
                    if self.params['specfile'] is not None:
                        self.print(self.params['specfile'])
                        file_prefix = os.path.splitext(os.path.split(self.params['specfile'])[1])[0]

                if file_prefix is None and 'data' in self.params:
                    if '%' in self.params['data']:
                        # Inject scan number in data filename
                        try:
                            self.params['data'] = self.params['data'] % int(scan)
                        except:
                            try:
                                self.params['data'] = self.params['data'] % scan
                            except:
                                print("Failed to replace %d or %s in data string by: ", scan)
                    if self.params['data'] is not None:
                        file_prefix = os.path.splitext(os.path.split(self.params['data'])[1])[0]

                if file_prefix is None:
                    file_prefix = 'data'

                if isinstance(scan, str):
                    try:
                        s = int(scan)
                        file_prefix = file_prefix + '_S%04d' % s
                    except ValueError:
                        file_prefix = file_prefix + '_S%s' % scan
                elif isinstance(scan, int):
                    file_prefix = file_prefix + '_S%04d' % scan

                self.ws = self.CDIRunnerScan(self.params, scan, timings=self.timings)
                self.ws.load_data()
                need_init_mask = True

                # data2cxi shall only be done by master for mpi=run jobs
                do_data2cxi = False
                if self.params['data2cxi']:
                    do_data2cxi = True
                    if MPI is not None:
                        if 'run' in self.params['mpi'] and not self.mpi_master:
                            do_data2cxi = False
                if do_data2cxi:
                    t1 = timeit.default_timer()
                    # This will save the original data, before cropping & centering
                    file_name = file_prefix + '.cxi'
                    if os.path.isfile(file_name):
                        self.print("Data CXI file already exists, not overwriting: ", file_name)
                    else:
                        self.ws.init_mask()
                        self.ws.corr_flat_field()
                        need_init_mask = False
                        self.print("Saving data to CXI file: ", file_name)
                        save_cdi_data_cxi(file_name, self.ws.iobs, wavelength=self.params['wavelength'],
                                          detector_distance=self.params['detector_distance'],
                                          pixel_size_detector=self.params['pixel_size_detector'],
                                          sample_name=self.params['sample_name'], mask=self.ws.mask,
                                          instrument=self.params['instrument'], note=self.params['note'],
                                          process_parameters=self.params)
                    self.timings['data2cxi'] = timeit.default_timer() - t1

                t1 = timeit.default_timer()
                self.ws.prepare_processing_unit()
                self.timings['prepare_processing_unit'] = timeit.default_timer() - t1
                self.ws.prepare_data(init_mask=need_init_mask, corr_flat_field=need_init_mask)

                support_threshold = self.params['support_threshold']

                vrun = np.arange(1, self.params['nb_run'] + 1)

                if MPI is not None:
                    if 'run' in self.params['mpi'] and self.mpi_size > 1:
                        # Distribute the runs among the different cients, independently
                        vrun = vrun[self.mpi_rank::self.mpi_size]
                        print("MPI #%2d: performing %d runs" % (self.mpi_rank, len(vrun)))

                for run in vrun:
                    # KLUDGE: modifying the support threshold in params may not be the best way ?
                    st = ""
                    if isinstance(support_threshold, list) or isinstance(support_threshold, tuple):
                        self.params['support_threshold'] = np.random.uniform(support_threshold[0], support_threshold[1])
                        st = "support_threshold = %5.3f" % self.params['support_threshold']
                    nbtry = 5
                    ok = False
                    while not ok and nbtry > 0:
                        nbtry -= 1
                        try:
                            self.print("\n", "#" * 100, "\n#", "\n#  CDI Run: %g/%g  %s\n#\n" %
                                       (run, self.params['nb_run'], st), "#" * 100)
                            s = time.strftime("-%Y-%m-%dT%H-%M-%S", time.localtime())
                            if self.params['nb_run'] > 1:
                                s += "_Run%04d" % run
                            self.ws.prepare_cdi()
                            self.ws.run(file_name=file_prefix + s, run_n=run)
                            ok = True
                        except SupportTooSmall as ex:
                            s = self.params['support_threshold_auto_tune_factor']
                            if nbtry:
                                s2 = "# ... trying to divide support threshold by %4.3f " \
                                     "(test %d/5)\n#\n" % (s, 5 - nbtry + 1)
                            else:
                                s2 = "# ... giving up\n#\n"
                            self.print("\n", "#" * 100, "\n#", "\n#  %s\n" % str(ex), s2, "#" * 100)
                            self.params['support_threshold'] /= s
                            st = "support_threshold = %5.3f" % self.params['support_threshold']
                        except SupportTooLarge as ex:
                            s = self.params['support_threshold_auto_tune_factor']
                            if nbtry:
                                s2 = "# ... trying to multiply support threshold by %4.3f " \
                                     "(test %d/5)\n#\n" % (s, 5 - nbtry + 1)
                            else:
                                s2 = "# ... giving up\n#\n"
                            self.print("\n", "#" * 100, "\n#", "\n#  %s\n" % str(ex), s2, "#" * 100)
                            self.params['support_threshold'] *= s
                            st = "support_threshold = %5.3f" % self.params['support_threshold']

                # Free GPU memory
                self.ws.cdi = FreePU() * self.ws.cdi

                if self.params['nb_run'] > 1 and self.params['nb_run_keep'] is not None:
                    # Keep only some of the best runs, delete others
                    if self.params['nb_run'] > self.params['nb_run_keep']:
                        res = self.ws.saved_results
                        if MPI is not None:
                            if 'run' in self.params['mpi']:
                                vres = self.mpic.gather(self.ws.saved_results, root=0)
                                if self.mpi_master:
                                    for i in range(1, self.mpi_size):
                                        res += vres[i]
                        if self.mpi_master or 'run' not in self.params['mpi']:
                            m = self.params['nb_run_keep_max_obj2_out']
                            if m > 0:
                                # Remove solutions with too large amplitude outside the support
                                res.sort(key=lambda x: x['obj2_out'])
                                if res[-1]['obj2_out'] > m:
                                    print("Removing solutions with too large amplitude outside the support"
                                          " (obj2_out > nb_run_keep_max_obj2_out=%4.3f):" % m)
                                while len(res) > 0 and res[-1]['obj2_out'] > m:
                                    v = res.pop()
                                    self.print("Removing: %s" % v['file_name'])
                                    fn = v['file_name']
                                    fnpng = fn.split("_LLKf")[0] + '.png'
                                    os.system('rm -f "%s" "%s"' % (fn, fnpng))
                            # Keep remaining solutions based on LLK_free
                            res.sort(key=lambda x: x['llk_poisson_free'])
                            if len(res) > self.params['nb_run_keep']:
                                print("Keeping %d solutions with the smallest Poisson free LLK"
                                      % self.params['nb_run_keep'])
                                for i in range(self.params['nb_run_keep'], len(res)):
                                    self.print("Removing: %s" % res[i]['file_name'])
                                    fn = res[i]['file_name']
                                    fnpng = fn.split("_LLKf")[0] + '.png'
                                    os.system('rm -f "%s" "%s"' % (fn, fnpng))
                                self.ws.saved_results = res[:self.params['nb_run_keep']]
                            os.system('rm -f latest.cxi latest.png')
                    else:
                        self.print('CDI runner: nb_run=%d <= nb_run_keep=%d !! Keeping all results' % (
                            self.params['nb_run'], self.params['nb_run_keep']))
                self.params['data'] = data_orig
            except Exception as ex:
                print(traceback.format_exc())
                print(self.help_text)
                print('\n\n Caught exception for scan %s:\n   %s    \n' % (str(scan), str(ex)))
                sys.exit(1)

        self.print("Timings:")
        for k, v in self.timings.items():
            if v > 1e-6:
                if MPI is not None:
                    self.print("MPI #%2d: %40s :%6.2fs" % (self.mpi_rank, k, v))
                else:
                    self.print("         %40s :%6.2fs" % (k, v))

        if warn_liveplot:
            self.print("\n", "#" * 100, "\n#",
                       "\n# WARNING: you are using the LIVE_PLOT option with multiple runs or MPI \n#"
                       "\n#      this will slow down the optimisation !\n"
                       "\n#      Please remember to enable live_plot only for tests \n"
                       + "#" * 100)
