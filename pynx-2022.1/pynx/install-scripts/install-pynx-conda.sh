#!/usr/bin/env bash

# This script was written to install PyNX using conda - tested on MacOS 10.13.6 with miniconda

#####################################################################################################
## The following can be used to install conda
############################################# MacOs
#curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
#chmod +x Miniconda3-latest-MacOSX-x86_64.sh
#
## Conda install: accept license, choose directory to install, accept conda setup in ~/.bash_profile
#./Miniconda3-latest-MacOSX-x86_64.sh
#
#source ~/.bash_profile    # This may be different under linux, only needed once
#
############################################# Linux
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#chmod +x Miniconda3-latest-Linux-x86_64.sh
#
## Conda install: accept license, choose directory to install, accept conda setup in ~/.bash_profile
#./Miniconda3-latest-Linux-x86_64.sh
#
#source ~/.bash_profile    # This may be different under linux, only needed once

echo $1

if [ -z $2 ];
then
  echo "No directory or python executable given for installation !";
  echo "Usage: install-pynx-conda.sh ENVNAME PYTHON_VERSION PYNX_VERSION"
  echo "   with: ENVNAME the name of the python virtual environement, e.g. pynx-env"
  echo "         PYTHON_VERSION the python version, e.g. 3.9"
  echo "         PYNX_VERSION (optional) the git tag for the pynx version to be installed"
  echo "example: install-pynx-conda.sh pynx-env 3.9"
  echo "example: install-pynx-conda.sh pynx-2021.1-py38 3.8 v2021.1"
  exit
fi

echo
echo "#############################################################################################"
echo " Creating conda virtual environment"
echo "#############################################################################################"
echo

# create the conda virtual environement with necessary packages
conda create --yes -n $1 python=$2 pip
if [ $? -ne 0 ];
then
  echo "Conda environment creation failed."
  echo $?
  exit 1
fi

# Activate conda environment (see https://github.com/conda/conda/issues/7980)
eval "$(conda shell.bash hook)"
conda activate $1
if [ $? -ne 0 ];
then
  echo "Conda environment activation failed. Maybe 'conda init' is needed (see messages) ?"
  exit 1
fi

# Add conda-forge for this environment (NB: could use --append instead to give conda-forge less priority)
conda config --env --add channels conda-forge
if [ $? -ne 0 ];
then
  echo "Adding conda-forge failed."
  exit 1
fi
conda config --env --set channel_priority strict

# Add packages from conda-forge
if [[ "$OSTYPE" != "darwin"* ]]; then
  conda install --yes -n $1 ocl-icd-system
fi

conda install --yes -n $1 numpy cython scipy matplotlib fabio ipympl pyopencl pytools scikit-image silx \
      ipython notebook h5py hdf5plugin ipywidgets psutil scikit-learn
if [ $? -ne 0 ];
then
  echo "Conda environment scientific packages installation failed."
  exit 1
fi

# Install remaining packages using pip
pip install h5glance pyvkfft

# Is this a stable way to get the path to the environment ?
BASEDIR=`conda info | grep 'active env location' | awk '{print $5}'`
mkdir -p $BASEDIR/dev

has_pyopencl=1

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
cd pynx

if [ -z $3 ];
    then echo "No tag given - using git pynx master head"
else
    git checkout tags/$3
fi

pip install .


echo
echo "#############################################################################################"
echo "Installing pyCUDA (optional)"
echo "#############################################################################################"
echo
# To also have CUDA support - use wheels if missing cuda.h for compilation
# Note: recent CUDA (>=8.0) require more recent scikit-cuda installation (from git), but older CUDA are incompatible
# with current scikit-cuda git...
# If scikit-cuda (old, from pip or conda) gives an error, try pip install git+https://github.com/lebedov/scikit-cuda.git
if [[ $(command -v nvcc ) ]] ;
then
     pip install pycuda --upgrade || echo "###\nCould not install pycuda - CUDA probably missing ?\n###\n"
fi

echo
echo "#############################################################################################"
echo "Finished installation !"
echo "#############################################################################################"
echo
echo "To use the silx viewer in 3D, also install pyopengl with:"
echo "     conda install --yes -n $1 pyopengl"
echo
echo "To use the PyNX environment, use 'conda activate $1'"
echo
echo "To test pynx installation, run 'pynx-test.py"


# Notes to also install paraview and ipyparaview:
#  conda install --yes -n $1 paraview nodejs jupyterlab
#  cd $BASEDIR/dev
#  git clone https://github.com/NVIDIA/ipyparaview.git
#  cd ipyparaview
#  ./build.sh
#  jupyter nbextension enable ipyparaview --py --sys-prefix
