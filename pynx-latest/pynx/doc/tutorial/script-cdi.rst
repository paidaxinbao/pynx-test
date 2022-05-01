.. api_reference:

CDI analysis using a command-line script
========================================
The simplest way to analyse a CDI dataset using PyNX is to use a
command-line script.

There are two main scripts which can be currently used:
`pynx-id01cdi.py` and `pynx-id10cdi.py`, which differ only by
a few default options (HIO vs RAAR algorithm, and for id10: `positivity`
and `mask=zero` options).

A simple data analysis can be done, when reading a CXI file, using:

.. code-block:: bash

  pynx-id01cdi.py data=data.cxi

This will simply run the analysis with default parameter. The initial
support will be determined using auto-correlation (which is usually
fine for Bragg CDI where no beamstop is used), no positivity...

A more detailed example using a Vaterite dataset (`Cherkas et al.,
Crystal Growth & Design 17, 4183–4188 (2017)
<http://dx.doi.org/10.1021/acs.cgd.7b00476>`_):

.. code-block:: bash

  # If necessary, activate your python environment with PyNX
  source /path/to/my/python/environment/bin activate

  # Download the example dataset
  curl -O http://ftp.esrf.fr/pub/scisoft/PyNX/data/T25_60_3D.cxi

  # View the CXI file using the silx viewer:
  silx view T25_60_3D.cxi

  # Run the PyNX analysis script
  pynx-id10cdi.py data=T25_60_3D.cxi support=circle support_size=70\
                  nb_raar=800 nb_hio=0 nb_er=200 verbose=50\
                  support_smooth_width_begin=3 support_smooth_width_end=1\
                  positivity support_threshold=0.2 max_size=512\
                  support_threshold_method=max liveplot

  # View the result from the output CXI file using the silx viewer
  silx view latest.cxi

  # Note that you can see all the parameters used for the optimisation
  # in entry_last/image_1/process_1/

  # You can tune the threshold if you want to improve the solution,
  # or try HIO instead of RAAR (nb_hio=800 nb_raar=0)


To perform a more complete analysis, it is advised to use multiple runs,
select the best from the free log-likelihood, and combine them:

.. code-block:: bash

  # Perform 10 runs and combine the 5 best ones (takes a little longer,
  # 30s per run on a V100 GPU, liveplot is disabled)
  # Note support_threshold=0.1,0.2 means threshold is randomly chosen
  # between 0.1 and 0.2 for each run

  rm -f *LLK*.cxi  # remove previous results

  pynx-id10cdi.py data=T25_60_3D.cxi support=circle support_size=70\
                  nb_raar=800 nb_hio=0 nb_er=200 verbose=50\
                  support_smooth_width_begin=3 support_smooth_width_end=1\
                  positivity support_threshold=0.2 max_size=512\
                  support_threshold_method=max nb_run=10 nb_run_keep=5

  # Perform a modes analysis and produce a movie (requires ffmpeg)
  pynx-cdi-analysis.py *LLK*.cxi modes movie

  # Look at modes analysis
  silx view modes.h5

  # See movie of slices from modes analysis
  vlc cdi-3d-slices.mp4

More information
----------------

The full documentation for the command-line scripts can be obtained by using
the `help` keyword from the command-line, e.g.:

.. code-block:: bash

  pynx-id01cdi.py help

For more information, please read the
:ref:`online documentation on CDI scripts <cdi_scripts>`
