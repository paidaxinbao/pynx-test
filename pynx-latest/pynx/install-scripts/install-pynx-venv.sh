#!/bin/bash

# This is a complete installation script for a python virtual environment with PyNX.
# This has been tested on debian 8 and macOS computers.
# It assumes you already have installed :
# - python>=3.4 (>=3.5 recommended)
# - git, cmake, compilers (Xcode with developer tools on macOS)
# - mpi if you want to use pynx with MPI
# - opencl headers and drivers (native on macOS, libraries needed on linux)
# - cuda development tools and drivers (optional)
# (note that you can also elect to use only opencl or cuda)

echo $1
if [ -z $2 ];
then
  echo "No directory or python executable given for installation !";
  echo "Usage: install-pynx-venv.sh DIRNAME PYTHON_EXE PYNX_VERSION"
  echo "   with: DIRNAME the name of the new directory to create the python virtual environement, e.g. pynx-env"
  echo "         PYTHON_EXE the name of the python executable, e.g. python3.9"
  echo "         PYNX_VERSION (optional) the git tag for the pynx version to be installed"
  echo "example: install-pynx-venv.sh pynx-env python3.8"
  echo "example: install-pynx-venv.sh pynx-2021.1-py39 python3.9 2021.1"
  exit
fi


echo
echo "#############################################################################################"
echo " Creating & the destination directory"
echo "#############################################################################################"
echo


if [ -d "$PWD/$1" ]; then
    echo "ERROR: directory $PWD/$1 already exists !"
    echo " Please remove the target directory first."
    exit
fi

if mkdir -p $1;
then
  echo "Installing in: " $1
else
  echo "Cannot install in: " $1
  echo "Exiting"
  exit
fi

cd $1
export BASEDIR=$PWD
echo $BASEDIR

echo
echo "#############################################################################################"
echo " Creating the python virtual environment"
echo "#############################################################################################"
echo
# Create the python virtual environment, without system packages
cd $BASEDIR
if [[ "$OSTYPE" == "darwin"* ]]; then
  # See https://matplotlib.org/faq/osx_framework.html
  if $2 -m venv ./ ; then
    echo "Created virtual environment"
  else
       echo "Failed to create virtual environment using python - missing venv module ?"
    exit
  fi
else
  if virtualenv -p $2 ./ ; then
    echo "Created virtual environment"
  elif $2 -m venv ./ ; then
    echo "Created virtual environment using 'python -m venv' (instead of virtualenv)"
  else
    echo "Failed to create virtual environment. Did you install python-venv or virtualenv ?"
    exit
  fi
fi

source bin/activate

echo
echo "#############################################################################################"
echo " Installing python packages"
echo "#############################################################################################"
echo
# install requirements
pip install --upgrade pip
pip install setuptools wheel --upgrade
pip install numpy cython scipy matplotlib ipython notebook h5py psutil pyvkfft --upgrade
pip install scikit-image hdf5plugin h5glance silx fabio ipywidgets ipympl scikit-learn --upgrade
# lxml and pillow are only necessary to avoid fabio warning
# pip install pillow lxml

# pip install jupyterhub jupyterlab --upgrade

echo
echo "#############################################################################################"
echo " Installing mpi4py (if possible)"
echo "#############################################################################################"
echo
pip install mpi4py --upgrade || echo "###Could not install mpi4py - mpicc not in PATH ?###"


echo
echo "#############################################################################################"
echo " Installing pyopencl packages"
echo "#############################################################################################"
echo

pip install pybind11 mako
if pip install pyopencl --upgrade ; then
    has_pyopencl=1
else
    echo "pyopencl installation failed - OpenCL WILL NOT BE AVAILABLE !" ;
    echo "  OpenCL is needed for most applications - Check your opencl headers and your internet access if pip failed" ;
    echo "  You can proceed if you do not have a GPU and wish only to make tests using the CPU (much slower)" ;
    echo ;
    read -p  "Press Ctrl-C to abort, or RETURN to continue" ;
    has_pyopencl=0
fi

mkdir -p $BASEDIR/dev


echo
echo "#############################################################################################"
echo "Installing pyCUDA (optional)"
echo "#############################################################################################"
echo
# To also have CUDA support - use wheels if missing cuda.h for compilation
# Note: recent CUDA (>=8.0) require more recent scikit-cuda installation (from git), but older CUDA are incompatible
# with current scikit-cuda git...
# If scikit-cuda (old, from pip) gives an error, try pip install git+https://github.com/lebedov/scikit-cuda.git
if [[ $(command -v nvcc ) ]] ;
then
     pip install pycuda --upgrade || echo "###\nCould not install pycuda - CUDA probably missing ?\n###\n"
fi

cd $BASEDIR
if [[ $(hostname -f) == debian8-devel.esrf.fr ]];
then
    # For debian8-devel, install SIP from source, pyqt5 and pyopengl
    echo
    echo "#############################################################################################"
    echo "Installing SIP from source and pyqt5, pyopengl"
    echo "#############################################################################################"
    echo
    # SIP must be installed from source, and as it installs a header in ..env/include/python3.4m , that include dir cannot be a symlink to the original python include dir
    cd $BASEDIR/include
    mv python3.4m python3.4m-lnk
    mkdir python3.4m
    cp -Rf python3.4m-lnk/* python3.4m/
    rm -Rf python3.4m-lnk
    cd $BASEDIR/dev
    wget https://downloads.sourceforge.net/project/pyqt/sip/sip-4.19.13/sip-4.19.13.tar.gz
    tar -xzf sip-4.19.13.tar.gz
    cd sip-4.19.13/
    python configure.py
    make install

    pip install pyqt5 --no-deps --upgrade
    pip install pyopengl --upgrade
fi


echo
echo "#############################################################################################"
echo " Installing PyNX"
echo "#############################################################################################"
echo
cd $BASEDIR/dev
echo "Select method to download PyNX: git (if you have a https://gitlab.esrf.fr account) or ftp (no login required):"
select yn in "ftp" "git" "manual"; do
    case $yn in
        ftp ) curl -O http://ftp.esrf.fr/pub/scisoft/PyNX/pynx-latest.tar.bz2 ; tar -xjf pynx-latest.tar.bz2 ; break;;
        git ) git clone https://gitlab.esrf.fr/favre/PyNX.git pynx ; break;;
        manual ) echo "PyNX installation skipped-sould be manually installed (local git copy, etc..)" ; break;;
    esac
done

if [ -d pynx ]; then
    echo "Installing PyNX..."
    cd pynx

    if [ -z $3 ];
        then echo "No tag given - using git pynx master head"
    else
        git checkout tags/$3
    fi

    pip install .
fi


echo
echo "#############################################################################################"
echo "Finished installation !"
echo "#############################################################################################"
echo
echo "To use the PyNX environment, use source $BASEDIR/bin/activate"
echo
if [[ $(hostname -f) != debian8-devel.esrf.fr ]];
then
    echo "If you wish to also use the SILX viewer (recommended), you should also install pyqt5 and pyopengl:"
    echo " source $BASEDIR/bin/activate ; pip install pyqt5 pyopengl"
fi
