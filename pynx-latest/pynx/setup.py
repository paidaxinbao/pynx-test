# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#         Ondrej Mandula


import platform
import os
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.install_lib import install_lib as su_install_lib
from setuptools.command.sdist import sdist as su_sdist
from pynx.version import __version__, get_git_version, get_git_date
from setuptools.command.bdist_egg import bdist_egg

cmdclass = {}
setup_requires = ['setuptools', 'wheel']
if 'x86_64' in platform.machine():
    try:
        from Cython.Distutils import build_ext
        import numpy

        cython_modules = [
            Extension("pynx.scattering.cpu", sources=["pynx/scattering/cpu.pyx", "pynx/scattering/c_cpu.c"],
                      include_dirs=[numpy.get_include()],
                      extra_compile_args=['-O3', '-ffast-math', '-msse', '-msse2', '-mssse3', '-msse4.1',
                                          '-march=native', '-mfpmath=sse', '-fstrict-aliasing', '-pipe',
                                          '-fomit-frame-pointer', '-funroll-loops', '-ftree-vectorize'])]
        cmdclass['build_ext'] = build_ext
        setup_requires.append('cython')
    except:
        cython_modules = []
else:
    cython_modules = []


class pynx_sdist(su_sdist):
    def make_release_tree(self, base_dir, files):
        super(pynx_sdist, self).make_release_tree(base_dir, files)
        try:
            # Replace git_version_placeholder by real git version
            version_file = os.path.join(base_dir, "pynx/version.py")
            vers = open(version_file).read()
            os.remove(version_file)
            with open(version_file, "w") as fh:
                print(get_git_version())
                print(get_git_date())
                vers = vers.replace("git_version_placeholder", get_git_version())
                vers = vers.replace("git_date_placeholder", get_git_date())
                fh.write(vers)
        except:
            print("sdist: replacing git_version_placeholder failed")


class pynx_install_lib(su_install_lib):
    def run(self):
        super(pynx_install_lib, self).run()
        try:
            # print(self.install_dir, self.build_dir)
            # Replace git_version_placeholder by real git version
            version_file = os.path.join(self.install_dir, "pynx/version.py")
            vers = open(version_file).read()
            os.remove(version_file)
            with open(version_file, "w") as fh:
                print(get_git_version())
                print(get_git_date())
                vers = vers.replace("git_version_placeholder", get_git_version())
                vers = vers.replace("git_date_placeholder", get_git_date())
                fh.write(vers)
        except:
            print("install_lib: replacing git_version_placeholder failed")


cmdclass['sdist'] = pynx_sdist
cmdclass['install_lib'] = pynx_install_lib

# Converts scripts to console_scripts
scripts = ['pynx/cdi/scripts/pynx_id10cdi.py',
           'pynx/cdi/scripts/pynx_id01cdi.py',
           'pynx/cdi/scripts/pynx_cdi_analysis.py',
           'pynx/cdi/scripts/pynx_cdi_regrid.py',
           'pynx/holotomo/scripts/pynx_id16a_holotomo.py',
           'pynx/ptycho/scripts/pynx_cristalpty.py',
           'pynx/ptycho/scripts/pynx_cxipty.py',
           'pynx/ptycho/scripts/pynx_hermespty.py',
           'pynx/ptycho/scripts/pynx_id01pty.py',
           'pynx/ptycho/scripts/pynx_id13pty.py',
           'pynx/ptycho/scripts/pynx_id16apty.py',
           'pynx/ptycho/scripts/pynx_id16a_nfpty.py',
           'pynx/ptycho/scripts/pynx_nanomaxpty.py',
           'pynx/ptycho/scripts/pynx_nanoscopiumpty.py',
           'pynx/ptycho/scripts/pynx_ptypy.py',
           'pynx/ptycho/scripts/pynx_ptycho_analysis.py',
           'pynx/ptycho/scripts/pynx_simulationpty.py',
           'pynx/ptycho/scripts/pynx_tps25apty.py',
           'pynx/scripts/pynx_info.py',
           'pynx/scripts/pynx_test.py',
           'pynx/utils/scripts/pynx_resolution_FSC.py'
           ]

console_scripts = []
for s in scripts:
    s1 = os.path.split(s)[1]
    s0 = os.path.splitext(s)[0]
    console_scripts.append("%s = %s:main" % (s1.replace('_', '-'), s0.replace('/', '.')))


class bdist_egg_disabled(bdist_egg):
    """ Disabled bdist_egg, to prevent use of 'python setup.py install' """

    def run(self):
        sys.exit("Aborting building of eggs. Please use `pip install .` to install from source.")


cmdclass['bdist_egg'] = bdist_egg if 'bdist_egg' in sys.argv else bdist_egg_disabled

setup(
    name="PyNX",
    version=__version__,
    packages=find_packages(),
    entry_points={'console_scripts': console_scripts,
                  # 'gui_scripts': [],
                  },
    cmdclass=cmdclass,
    ext_modules=cython_modules,
    python_requires='>=3.7',
    setup_requires=setup_requires,
    install_requires=['numpy>=1.12', 'scipy>=1.6.0', 'matplotlib', 'scikit-image',
                      'h5py>=2.9', 'hdf5plugin', 'fabio', 'silx', 'packaging', 'psutil', 'scikit-learn',
                      'mako', 'pyopencl', 'pyvkfft>=2022.1'],
    extras_require={'cuda': ['pycuda>=2021.1'], 'gid': ["cctbx"], 'mpi': ['mpi4py'],
                    'doc': ['sphinx', 'nbsphinx', 'nbsphinx-link', 'sphinx-argparse'],
                    'gui': ["ipywidgets", "jupyter", "pyqt5", "ipympl", "pyopengl"]},
    include_package_data=True,
    data_files=[],

    # metadata for upload to PyPI
    author="vincefn",
    author_email="favre@esrf.fr",
    description="PyNX - GPU-accelerated python toolkit for coherent X-ray imaging and nano-crystallography",
    license="CeCILL-B",
    keywords="PyNX GPU OpenCL CUDA crystallography diffraction scattering coherent X-rays Imaging ptychography CDI",
    # Should also run under windows, as long as PyOpenCL and/or PyCUDA are properly installed
    platforms=["MacOS", "POSIX", "Linux"],
    url="http://ftp.esrf.fr/pub/scisoft/PyNX/",
    long_description=
    "PyNX provides Python tools for coherent X-ray imaging and nano-crystallography: \
      1) to compute the X-ray scattering for nano-structures, using GPU (CUDA or OpenCL)"
    " acceleration, including in grazing incidence conditions.\
      2) X-rays scattering atomic scattering factors\
      3) 2D X-ray wavefront propagation\
      4) tools for ptychography reconstruction\
      5) Coherent Diffraction Imaging (2D and 3D) reconstruction algorithms",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: License :: CEA CNRS Inria Logiciel Libre License B (CeCILL-B)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Physics',
        'Environment :: GPU :: NVIDIA CUDA'
    ],
)
