Installation
============
*PyNX* supports python version 3.7 and above

You should follow `Full installation in a python virtualenv`_, which can be used on any system which supports a python
virtual environment. Installation has been tested on linux (e.g. debian >=8) systems as well as macOS computers.
See below the notes regarding :ref:`installation on windows computers <windows_install>`.

If you are using an nVidia card and already have CUDA development tools, you can also use the
:ref:`quick install <quick_install>` which only uses ``pip``.

Generic Instructions
====================
*PyNX* is focused on using Graphical Processing Units (GPU) for faster calculations, so you will need:

* a GPU (which can be an integrated GPU)
* an OpenCL installation (drivers and libraries)
* and/or CUDA (which gives better performance), which requires CUDA drivers and development tools (nvcc)

PyNX should still work on a CPU only, but without any optimisation, and will therefore be very slow, especially for
3D CDI and Ptychography algorithms. Also, not all features are available in the CPU operators,
so the CPU version is mostly intended for basic testing and educational purposes.

.. _quick_install:

Quick installation
------------------
This allows to install PyNX quickly, assuming that you already have:

 * a python (>=3.7) installation
 * pip for python package installation
 * already-installed CUDA development tools (nvcc)

It is advised to use a python virtual environment (or a conda one)

You can install PyNX and the dependencies using the following commands:

.. code-block:: bash

  pip install --upgrade pip
  pip install setuptools wheel --upgrade
  pip install numpy cython scipy matplotlib ipython notebook scikit-image ipywidgets ipympl
  pip install h5py hdf5plugin h5glance silx fabio
  curl -O http://ftp.esrf.fr/pub/scisoft/PyNX/pynx-latest.tar.bz2
  pip install pynx-latest.tar.bz2[cuda]

Note you can remove the ``[cuda]`` if you do not want cuda but only OpenCL support,
and you can use instead ``[cuda,gui]`` if you want also gui components, or
even ``[cuda,gui,mpi]`` if you want to also have MPI support (if you have >=2 GPU).

The ``[cuda]`` option does not change anything to the PyNX installation, but only adds
specific dependencies. So you can also install ``pycuda`` afterwards.

Once this has been run, you can :ref:`test pynx <testing>`.

Full installation in a python virtualenv
----------------------------------------
The following script should work on any POSIX (Linux, MacOS X) system, and requires:

 * a python (>=3.7) installation
 * pip for python package installation
 * GPU dependencies, for CUDA and/or OpenCL (at least one should be present, both can be used):

   * OpenCL libraries (out-of-the box on MacOS X, using nvidia/AMD drivers on Linux)
   * CUDA libraries and development tools
 * git, cmake and standard development tools (for C/C++/python, depending on the operating system)

The script can be found in the source code as 'install_scripts/install-pynx-venv.sh',
or can be downloaded from http://ftp.esrf.fr/pub/scisoft/PyNX/install-scripts/install-pynx-venv.sh

Once this has been run, you can :ref:`test pynx <testing>`

Full installation in a conda environment
----------------------------------------
You can also install PyNX in a conda environment in the same way as a python virtual environment.
You should preferably install all packages available through conda or conda-forge, before
using pip to install PyNX.

You can use the installation script found in the source code as 'install_scripts/install-pynx-conda.sh',


.. _windows_install:

Windows installation on a PC with an nVidia card
------------------------------------------------
Windows is *not officially supported*, as it can be complex to get the correct environment tools
to make PyNX work. You are therefore encouraged to use PyNX on a linux workstation or cluster node,
or on a macOS machine.

Nevertheless the following was found to work on a PC with Windows 10 and an nVidia GTX 1080:

* Install CUDA 11.2 tools & drivers from
  https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64
* Install Visual studio 2019 from https://visualstudio.microsoft.com/fr/downloads/ .
  With the installation of: C++ desktop tools, command-line interface tools (CLI), C++ clang
* add C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29333\bin\Hostx64\x64
  to the windows PATH environment variable
* Install Anaconda and create an environment with all pynx dependencies installed from conda,
  except for the following installed with pip (using the conda terminal): pycuda, pyvkfft, fabio, pynx

With all this PyNX runs correctly with CUDA in a notebook or ipython console.

Notes:

* command-line scripts can also be used but **without the ``.py`` extension**, e.g.
  to test PynX you should run ``pynx-info`` and ``pynx-test`` from the anaconda console.
* on-the-fly compilation of GPU kernels is significantly slower on windows than on
  linux and macOS, so when running a script or notebook, the first time the kernels are
  compiled (which happens transparently), the process may seem to hang. Once this has been
  done the optimisation will run at the same speed as on other platforms, depending on the GPU.

Development version
-------------------
If you want to live on the wild side, you can install the (public) development version (updated nightly) using:

.. code-block:: bash

  curl -O http://ftp.esrf.fr/pub/scisoft/PyNX/pynx-latest.tar.bz2
  pip install pynx-devel-nightly.tar.bz2[cuda]

.. _testing:

Testing the installation
------------------------
Once installed, you can use:

.. code-block:: bash

  pynx-info.py

which will tell you the version of pynx installed, and if you have
support for cuda and/or opencl.

You can then test pynx from the console by using:

.. code-block:: bash

  pynx-test.py

To also test live-plotting, you can run:

.. code-block:: bash

  pynx-test.py live_plot

You can also run more specific tests using command-line keywords (combinations are possible):
 * ``pynx-test.py processing_unit`` : only run basic OpenCL and CUDA tests
 * ``pynx-test.py cdi`` : only run CDI tests
 * ``pynx-test.py cdi_runner`` : only run CDI runner tests
 * ``pynx-test.py ptycho`` : only run ptychography tests
 * ``pynx-test.py ptycho_runner`` : only run ptychography runner tests
 * ``pynx-test.py cuda`` : only run CUDA tests
 * ``pynx-test.py opencl`` : only run opencl tests

Dependencies
------------
Requirements:

* git, cmake and standard development tools (compilers, headers...)

* Python packages (all installable using pip):

 * numpy, scipy, matplotlib
 * cython
 * scikit-image, scikit-learn
 * h5py hdf5plugin
 * silx fabio
 * psutil
 * pyvkfft (fft using cuda and opencl)

* Recommended:

 * ipython, notebook
 * pyqt5, pyopengl (for silx viewer)
 * sphinx and nbsphinx to generate documentation

* For OpenCL

 * pyopencl, mako

* For CUDA:

 * CUDA development tools (nvcc)
 * pycuda

* Optionally:

 * the cctbx library, if you want to use the *pynx.scattering.gid* module for grazing incidence scattering.
   This is a bit complex to install, so it should probably be installed first, before all other packages.

 * pandoc for sphinx documentation generation

 * scikit-cuda or gpyfft if you want to use those libraries rather than pyvkfft
