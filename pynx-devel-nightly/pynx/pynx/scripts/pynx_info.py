#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2021-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from pynx.version import get_git_version, get_git_date, __version__, __copyright__


def main():
    print("PyNX version: %s" % get_git_version())
    print("Last git commit date: %s" % get_git_date())
    print("Copyright: %s" % __copyright__)
    print()
    try:
        from ..processing_unit import has_cuda, has_opencl
        print("  cuda support: ", has_cuda)
        print("opencl support: ", has_opencl)
    except:
        print("Error testing for CUDA and OpenCL support")
    try:
        from pyvkfft.version import __version__
        print("pyvkfft: version %s " % __version__)
    except ImportError:
        print("pyvkfft: not installed")
    print("\nTo run pynx tests, use pynx-test.py")


if __name__ == '__main__':
    main()
