.. PyNX documentation master file, created by
   sphinx-quickstart on Tue Sep 20 11:03:38 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    tutorial/*
    scripts/*
    modules/*
    *


.. include:: ../README.rst
   :end-before: Pynx.cdi

Automated testing
=================
To automatically test PyNX after installation, you can run the `pynx-test.py` script, which will run a
series of tests and can help diagnose issues specific to GPU languages (OpenCL, CUDA), dependencies
or applications (CDI, Ptycho..). Alternatively you can run `pytest` (if installed) from the root
of the PyNX source directory.

Beginner tutorials
==================
To begin using PyNX, you can read the following :doc:`tutorial/index`:

 * Use **command-line-scripts** for data analysis:

   * :doc:`Ptychography scripts tutorial <tutorial/script-ptycho>`
   * :doc:`CDI scripts tutorial <tutorial/script-cdi>`

 * **Python API tutorial notebooks** for:

   * :ref:`tutorial_cdi`
   * :ref:`tutorial_ptychography`
   * :ref:`tutorial_scattering`
   * :ref:`tutorial_wavefront`

Command-line scripts
====================
:doc:`scripts/index`
    Documentation of scripts included in *PyNX*

API Documentation
=================
:doc:`modules/index`
    Documentation of modules included in *PyNX*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

