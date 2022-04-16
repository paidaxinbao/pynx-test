Version 2022.1 (2022-02-06)
---------------------------
Major changes
^^^^^^^^^^^^^

* simplified installation: e.g. to get both cuda and OpenCL backends,
  this can be done using just 'pip install pynx.tar.bz2[cuda]',
  including all dependencies.
* Ptycho: incoherent background optimisation with a much more stable
  approach for AP and DM.
* Ptycho: add near field runner script (ESRF-id16A), also enabled for simulation
* Ptycho: much faster position optimisation (cuda). Change defaults in runner
  for positions optimisation (mult=5 and advise update every N cycles)
* Ptycho: add 'multiscan_reuse_ptycho' option to runner scripts to chain the
  analysis of multiple scans (e.g. for ptycho-tomo) by re-using the
  previous object and probe, avoiding extra initialisations,
  and allowing to use a shorter algorithm chain after the first scan
* Ptycho: the FFT scaling has been unified between CUDA, OpenCL and CPU
  operators which are now coherent. The amplitudes of the object and
  probe are each increased (compared to pre-12/2021 versions) by a factor,
  the product of which is equal the Fourier window size.
* CDI: improved PSF calculation, with several options for the
  partial coherence kernel, including periodic update of PSF managed
  from the algorithm chain, loading previous PSF, etc..
  The initial PSF model must be specified with the CDI runner, e.g.
  using psf=pseudo-voigt,0.5,0.05,20 (see doc)
* CDI: optimisations for large data sizes, with faster calculations
  and less memory used. Now 960**3 datasets require ~28 GB GPU memory
* CDI: in a python script, the object must be initialised using the
  InitObj operator, unless an object has been supplied.
* CDI: add a Regrid3D class and a pynx-cdi-regrid.py script to
  transform small-angle CDI frames to a 3D dataset (Jérôme Kieffer)
* CDI: the support update can raise exceptions if the support is too
  small or too large. This is handled automatically in the runner
  to restart while changing a little the threshold.
* CDI runner: when using nb_run and nb_run_keep, also reject solutions
  which have a too large fraction of square modulus outside the support.
* CDI analysis script: faster default options without phase ramp match,
  and parallel processing for loading an modes analysis. Copy
  original process information into the modes file.
* Now requires Python>=3.7. Source installation must use 'pip install'
  and not 'python setup.py install'
* Switch default library for FFT to VkFFT (using pyvkfft), for both
  OpenCL and CUDA. scikit-cuda and gpyfft are still supported.
  This improves performance for opencl, and also for cuda with
  large FFT dimensions (>1k). Memory requirements are lowered
  for cuda when nx<=8192 and ny,nz<=4096

Minor changes & bugfixes
^^^^^^^^^^^^^^^^^^^^^^^^

* Ptycho: correct auto-scaling of object & probe which altered convergence
  for DM when printing LLK
* Ptycho: minor speed improvement (cuda) with fully asynchronous ML
* Ptycho runner: saveprefix does not require to use a sub-directory
* Ptycho & CDI will now report average bandwidth per algorithm step
* CDI: improve memory allocation which was causing a slowdown when using PSF
* CDI: add lazy operators for support and random object initialisation,
  free pixels generation.
* CDI: added a ObjSupportStats operator which gathers the the fraction
  of the object's square modulus outside the support (obj2_out)
* CDI runner: correct loading previous object, allow to alter loaded
  object by random amplitude and/or phase factors.
* CDI runner: only perform data2cxi with the master job for mpi=run
* CDI runner: allow to select a subpixel with 'rebin' option instead
  of summing over the rebin range (i.e. skip rather than bin)
* CDI runner: correctly take into account mask with a user-supplied ROI.
* Ptycho: added unit test class to compare analysis to reference results
* GPU auto-selection now based on bandwidth, not FFT
* Operators now support lazy operations, to store an operator and apply
  it only when the next non-lazy operator is applied (useful when the
  CDI or Ptycho object initialisation is delayed)
* FFT: added a generic interface to perform FFT with the correct scaling
  in ProcessingUnit-derived classes
* MPI: when using N>1 tasks with N GPUs per node, the ranking of the GPU
  is now tweaked so that each MPI task will use a different GPU, with
  a workaround the unpredictable MPI rank distribution per node, and
  the fact that GPU device order can be different even for two process
  on the same node.
* pycuda>=2021.1 is required to handle array larger than 4 GB (>812**3)
  (https://github.com/inducer/pycuda/issues/282)
* [BUG]: correct a CUDA 'invalid resource handle' bug when analysing
  successive scans with multiple GPUs.

Version 2020.2.2 (2020-12-12)
-----------------------------
* CDI modes analysis: add CUDA version for accelerated phase matching
* Ptycho: in the CXI output, enforce using a local link for the object
  virtual dataset, to avoid a broken link.
  Workaround for https://github.com/h5py/h5py/issues/1546 in h5py<3.0

Version 2020.2.1 (2020-12-05)
-----------------------------
* CDI runner: correct bug where mask file with uppercase characters
  was not correctly imported
* Add cuFFT workaround for CUDA>=11 (see https://github.com/lebedov/scikit-cuda/issues/308)

Version 2020.2 (2020-10-23)
---------------------------
* Ptycho-MPI: add ability to distribute computing on several GPU and/or nodes,
  using MPI. This can be used to distribute independent scans  (mpi=multiscan)
  or to split a large scan (mpi=split) in multiple parts which are automatically
  stitched (aligned in phase and positions)
* Ptycho: systematic and reproducible phase ramp removal (far field)
* Ptycho: keep track of original, absolute coordinates in object
* Ptycho: use NeXus formatting to automatically display object absolute coordinates
* Ptycho simulation script: better scale the probe, step size and object according
  to the frame size. Allow using a siemens star as an object. Use GPU
  calculations for simulation.
* Ptycho: add a threshold filter for the position correction, to avoid updating
  shifts where the object gradient is too small.
* Ptycho runner: add obj_max_pix parameter for the maximum object size
* Ptycho: the absolute position of object & probe is corrected in save plots
  as well as the CXI file.
* Ptycho: rename pynx-analyzeprobe.py to pynx-ptycho-analysis.py, and
  allow using CXI as input files.
* CDI: add interpolation scheme for masked pixels (e.g. in gaps), using
  inverse distance weighted interpolation, and a large confidence interval
  for amplitude projection during optimisation.
* CDI-MPI: add ability to distribute multiple scans or runs using MPI
* CDI & Ptycho runner: access help text using 'help' or '--help' keywords
* CDI runner: allow processing multiple scans in a more generic way
* CDI runner: allow importing the initial support from a previous
  CXI result file, or from an hdf5 file, optionally with the hdf5
  path to the support data array.
* CDI: add iobs plot to PRTF plots.
* CDI: save the correct un-masked iobs in CXI files
* CDI, Ptycho: more accurate reporting of average dt/cycle,
  including support update, graphical displays.
* pynx-cdi-analysis: save PRTF in modes.h5 when possible
* Documentation: now include notebooks in the html docs (using nbsphinx)
* HDF5: set by default the HDF5_USE_FILE_LOCKING environment variable
  to FALSE when opening a data file which may also be written by another
  process simultaneously. This is inhibited if HDF5_USE_FILE_LOCKING
  is already set.

Version 2020.1 (2020-02-02)
---------------------------
* CDI runner: enable combining several masks and interpolating gap maxipix masks
* CDI runner: enable setting initial support based on command-line equation
* CDI: Faster cdi array matching and pynx-cdi-analysis using OpenCL
* CDI: add phase retrieval transfer function (PRTF) plotting code
* CDI & ptychography: more automatic tests
* Ptycho: enable position/translation corrections
* [BUG] Ptycho: correct gradient calculation for maximum likelihood/conjugate gradient algorithm
* [BUG] Ptycho runner: correctly reshape and rescale probe as needed when loading a previous probe
* All: use safe import for matplotlib.pyplot in case tk is not available, switching backend to agg
* pynx-test.py: add option for automatic tests & email reporting
* More efficient memory usage, especially for tests
* [INCOMPATIBLE CHANGE] Scattering: change sign in Fhkl_thread, now computing
  F(hkl)=SUM_i exp(-2j*pi*(h*x_i + k*y_i + l*z_i)) instead of F(hkl)=SUM_i exp(+2j*pi*(h*x_i + k*y_i + l*z_i))

Version 2019.2.6  (2019-06-19)
------------------------------
* [BUG] Ptycho: Correct CUDA ML operator, which prevented correct minimisation
* CDI: keep the free pixel mask during successive runs (nb_run=N)

Version 2019.2.5  (2019-06-02)
------------------------------
* Ptycho: large speedup when using CUDA by increasing stack size (needed for fast, recent GPU) and atomic operations
* Ptycho: store history of figures of merit and cycle parameters. Export to CXI file.
* Ptycho: add pynx-simulationpy runner for tests
* Ptycho: add nanoscopium runner script
* Ptycho: add dm_loop_obj_probe parameter to control looping over object+probe update
* Improve pynx-test.py output
* Improve documentation
* Improve pynx.test.speed to test for large pinned memory allocations.
* [BUG] CDI: fix CDI Calc2Obs operator
* [BUG] CDI & Ptycho: correct nps file import
* [BUG] Ptycho: correct wavelength calculation for CXI export
* [BUG] Ptycho CXI runner: correct xyrange parameter interpretation

Version 2019.2  (2019-05-20)
----------------------------
* CDI & Ptychography: CXI output files follow the NeXus standard, allowing direct display when opened with silx view.
* CDI: record history of indicators (log-likelihood, support size and levels, ...) in CXI output
* CDI runner: add save=all option to save several steps in the algorithm chain
* CDI: support update has been improved to avoid diverging, affecting threshold levels to be used.
* CDI: allow updating support only around the border of the support (support_update_border_n)
* CDI: add GPS operator
* CDI: export a more complete set of configuration parameters to CXI files
* CDI: correct scaling (ML operator, initial scaling)
* CDI: correct examples
* CDI runner: add save=all options to export solved object after each step
* Ptychography: CUDA operators are now preferred to OpenCL (significantly faster for large frame sizes)
* Ptychography: improve near field algorithm, allowing to specify mask with a zero-phase restraint (vacuum)
* Ptycho runner: allow to roll (circular shift) data instead of cropping
* Ptycho: add NanoMAX (MaxIV) runner script
* Utils: add phase retrieval transfer function estimation (for CDI)

Version 2019.1  (2019-02-07)
----------------------------
* CDI: add 'free' log-likelihood figure-of-merit.
* CDI: allow to give a range for the support threshold, when performing multiple runs.
* CDI: allow to keep only the best solutions when performing multiple runs
* CDI: id01 and id10 scripts will now print the algorithm chain used, when it is not user-supplied
* CDI: add pynx-cdi-analysis script to analyse proposed solutions.
* Ptycho: enable CUDA operators, 2x speed improvements, especially for large frame sizes
* Ptycho: correct probe and object orientation and axis in plots, so that both are seen from the source
* Ptycho: auto-correct probe centering when necessary (DM)
* Ptycho: better handling of plots for near field Ptycho
* Ptycho scripts: add ability to create a movie of the scan
* CDI & Ptycho: improved speed of calculations from GPU profiling.
* Add test suite
* Support for Python 3.7
* Python 3.4 is deprecated
* [Incompatible change] Ptycho: now all API functions using x,y(,z) coordinates as input or output
  will use them in alphabetical order. The inverse order is only used for shapes e.g. (ny, nx). This affects
  notably declaration of PtychoData, as well as get_view_coord(), calc_obj_shape(), Simulation.scan.values
* [BUG] CDI: correct handling of smooth parameter in OpenCL SupportUpdate() operator
* [BUG] CDI: correct handling of masked pixels when using auto-correlation to init the support (OpenCL)
* [BUG] Ptycho: correct taking into account of mask when using a command-lien script
* [BUG] Ptycho script: correct taking into account mask in some circumstances

Version 2018.2.0  (2018-07-17)
------------------------------
* CDI: enable using partial coherence (GPU-optimised)
* CDI runner: use algorithm steps based on operators, e.g. algorithm='ER**50,(Sup*ER**5*HIO**50)**10'
* CDI id01 runner: allow batch processing data from a spec file + scan numbers
* CDI runner: use the scan number to save CXI files (data and output)
* Ptycho: switch completely to the new operator-based API
* Ptycho: switch scripts output to hdf5/CXI file format
* Ptycho: add id16A runner (lambda detector)
* Ptycho runner: add ability to substract a dark image
* Ptycho runner: add orientation_round_robin option
* Ptycho runner: use 'mask=' instead of 'loadmask='
* Ptycho CXI runner: use 'data=' instead of 'cxifile='
* Ptycho CXI runner: allow analysing several CXI data files using a generic manne: 'data=data%05d.cxi scan=13,67,89'
* Ptycho: improve display of phase
* Ptycho API: add AnalyseProbe and OrthoProbe operators
* Ptycho: plot 'up' correctly (flip up/down plotting with respect with previous version)

Version 3.6.3  (2018-03-21)
---------------------------
* CDI: sample name, instrument and a note can be saved to CXI files
* CDI: change FFT-scaling approach (lower noise from masked high-frequency pixels ?)
* Ptycho id01 runner: read detector distance from UDETCALIB if available
* [BUG] Ptycho: correct reading mask from hdf5
* Wavefront: default to filling the wavefront with 1 instead of 0.
* Wavefront: Add ability to start from a photo/image from scipy or skimage
* Add benchmark module (pynx.test.speed)

Version 3.6.2  (2018-01-25)
---------------------------
* Ptycho: id01 runner: add 'livescan' option to search for new data when analysing a given spec data file.
* Ptycho runner: data2cxi will now export raw data, unless data2cxi=crop was used (corrected bug)
* Use PYNX_PU environment variable to set language (CUDA/OpenCL/CPU) and/or gpu name and/or gpu rank
* Ptycho and CDI: add CPU API (not yest accessible for ptycho runner scripts, only with new python API)

Version 3.6.1  (2017-12-19)
---------------------------
* CDI runner: add roi= keyword to manually supply the region-of-interest.
* CDI: add option to update the support based on the maximum value, instead of the average
* CDI runner: add 'support_post_expand' keyword to shrink and/or expand the support by a few pixels after update
* CDI: handle <0 observed intensities during initial scaling of object
* CDI runner scripts: report poisson, gaussian and euclidian llk
* CDI id01 runner script: add support for the Eiger detector
* CDI: update examples
* CDI runner: correctly take into account output_format keyword
* CDI: correct some bugs with the OpenCL implementation
* Ptycho: add operator-based python API (not yet used for command-line scripts)
* Ptycho: add operator-based near field ptychography
* Processing Unit API: allow to centrally select a GPU language and/or a device
* Remove official support for Python 2.7. Now supporting Python>=3.4

Version 3.5.0  (2017-10-09)
---------------------------
* CDI: use auto-correlation to estimate initial support, if none is supplied: AutoCorrelationSupport() operator
* CDI: add pynx-id01cdi.py runner script, allows to perform CDI analysis directly from spec and images files
* CDI: better initial object scaling. ScaleObj operators implemented in CUDA and OpenCL
* CDI scripts: allow rebinning input data, with different rebin values for each axis
* CDI scripts: allow loading initial object from a file (npy, npz, mat, cxi)
* CDI scripts: enable multiple runs
* CDI and Wavefront Operators: enable sum of operators (experimental)
* CDI: ML() Operator. Default nb cycles=1, allow using power (ML()**n) to change the number of cycles
* Ptycho scripts: check for existence of CXI file if supplied.
* Ptycho scripts: Bring object phase origin to zero if possible, before final save.
* Ptycho scripts: add ability to export cropped data as CXI file, to save space.

Version 3.4.0  (2017-09-21)
---------------------------
* CDI scripts: allow importing matlab mask files.
* CDI scripts: if the mask is 2D and iobs 3D, apply the same mask to all frames
* CDI scripts: handle 2D data
* CDI: allow importing a support from a file.
* CDI: allow to disable support update.
* CDI: use auto-correlation to estimate initial support if none is supplied. Default threshold is 10%.
* [Incompatible] CDI API: remove pixel_size_object and lambdaz arguments to CDI constructor

Version 3.3.4  (2017-09-14)
---------------------------
* CDI: add import/export of diffraction data to/from CXI/hdf5 file format
* CDI: add export of final phased object to CXI/hdf5 format (by default)
* CDI: add import of mask from file
* CDI scripts: add keywords data2cxi and output_format
* CDI: add OpenCL version of SupportUpdate() operator
* [BUG] For all operators, make sure Op1 * Op2 returns a new operator, to avoid altering Op1. Same for Op1**N
* Ptycho: handle case when resized probe from previous result had an odd size.
* [BUG] Ptycho: fix incorrect hdf5 softlink path to translation in CXI files

Version 3.3.1  (2017-07-19)
---------------------------
* Ptycho: Correct probe simulation using new wavefront operator API
* [BUG] error in CUDA context destruction

Version 3.3.0 (2017-07-19)
--------------------------
* CDI, Ptycho: either CUDA or OpenCL operators are automatically loaded using e.g. 'from cdi.operator import * '
* CDI, Ptycho: automatically transfer data to/from GPU memory, using timestamps
* Ptycho: using new wavefront operator API for Ptycho probe simulation
* Wavefront: update examples
* CDI: update id10 runner script
* CDI: update examples
* [Incompatible] FreePU(), ToPU() and FreeFromPU() operators are obsoleted

Version 3.2.2 (2017-07-04)
--------------------------
* CDI: Fix CUDA ObjConvolve for 2D objects

Version 3.2.1 (2017-07-04)
--------------------------
* Ptycho: Add generic handling of detector orientation and ptychography positions handling. Add correct default values for ptypy files.

Version 3.2.0 (2017-07-03)
--------------------------
* CDI, wavefront: update examples using operators
* CDI: add CUDA SupportUpdate and ObjConvolve operators
* CDI id10 script: add the ability to use RAAR before HIO. Defaults to non-mandatory-shrinking
* CDI: add HIO and RAAR detwin operators
* CDI: add HIO, ER, RAAR and CF with positivity constraint
* Ptycho: automatically adapt probe proagation range during analysis
* Ptycho: add the ability to start from a focused circular aperture
* Wavefront: add Thinlens, circular and rectangular mask operators
* Wavefront: correct sign of final quadratic phase factor for FRT and Near Field propagation


Version 3.1.0 (2017-05-18)
--------------------------
* CDI: update example
* Ptycho: add pynx-ptypy.py runner script to handle PtyPy data format
* Ptycho: Save scan_area_probe/obj along object and probe
* Utils: add pynx-resolution-FSC.py script for resolution analysis using Fourier Shell Correlation, courtesy of J.C. da Silva

Version 3.0.0 (2017-05-03)
--------------------------
* [Incompatible] **CDI and Wavefront: complete rewrite of  API using operators**, using either CUDA or OpenCL

Version 2.8.0 (2017-04-27)
--------------------------
* Update documentation
* Code reorganization with the addition of ``pynx.processing_unit``
* CDI: add specific HIO and ER kernels to bias towards real positive components. Add RAAR kernel. Improve 3D FFT speed
* Utils: Add pynx.utils.benchmark with FFT benchmark (cufft, clfft) function

Version 2.7.0 (2017-02-05)
--------------------------
* Update installation instructions
* Ptycho: update display of object and probe
* Ptycho: More verbose output at the beginning of each optimisation

Version 2.6.0 (2017-02-03)
--------------------------
* Ptycho: reorganize runner (script) classes in a runner module
* Ptycho: add 'latest-...' symbolic links at the end of every run

Version 2.5.X (2017-02-01)
--------------------------
* [BUG] Ptycho: correctly transform x,y scan positions in meters, also for export to CXI file
* CDI: add ESRF 2D logo example
* Ptycho: deactivate multi-GPU support.
* Ptycho: only report the normalized log-likelihood
* Ptycho: id13 script, improve reading from Eiger data files (masked pixels,..)
* Ptycho: add ability to trigger object/probe/background optimization during multi-step optimization

Version 2.4.0-2.4.2 (2017-01-30)
--------------------------------
* Ptycho: auto-selection of fastest GPU
* Ptycho scripts: add keyword loadmask=maxipix to handle gap pixels
* Ptycho: Add background optimization to alternating projection algorithm
* Ptycho: multiple improvements to plots

Versions 1.3.0-2.3.2 (2016)
---------------------------
* Ptycho: complete rewrite of the ptycho module, with pure OpenCL calculations including multiple modes,
  maximum likelihood conjugate gradient,...
* CDI: new module for CDI using OpenCL
