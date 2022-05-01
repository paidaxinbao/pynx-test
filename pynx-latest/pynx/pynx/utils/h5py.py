# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2020-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
# This file can be used to open hdf5 files in read mode while disabling
# file locking. This allows another process/computer to write or append
# the hdf5 file while this process reads it.
# This is a KLUDGE, and only works as long as the environment variable
# is checked by the hdf5 library *when opening the file*.
# See https://gitlab.esrf.fr/bliss/bliss/-/blob/master/nexus_writer_service/io/nexus.py#L912
# for the implementation in BLISS

import os
try:
    # We must import hdf5plugin before h5py, even if it is not used here.
    import hdf5plugin
except:
    pass
# We need to import everything to behave like h5py, and just patch h5py.File
import h5py
from h5py import *


class File(h5py.File):
    """
    This File class overloads h5py.File to allow disabling file locking.
    At ESRF when reading data files which may be accessed by another process,
    the HDF5_USE_FILE_LOCKING will be set to FALSE temporarily
    to avoid locking any other writing process.
    """

    def __init__(self, filename, mode="r", enable_file_locking=None, swmr=None, **kwargs):
        """

        :param filename: the hdf5 file to open
        :param mode: the opening mode. See h5py documentation
        :param enable_file_locking: if True or False, the environment variable
            HDF5_USE_FILE_LOCKING will be set to TRUE or FALSE.
            If None and the environment variable exists, it is unchanged.
            If None and no HDF5_USE_FILE_LOCKING environment variable exists,
            it is set to FALSE when the mode is 'r'.
        :param swmr: swmr option passed to h5py
        """
        if "HDF5_USE_FILE_LOCKING" in os.environ:
            env_HDF5_USE_FILE_LOCKING = os.environ["HDF5_USE_FILE_LOCKING"]
        else:
            env_HDF5_USE_FILE_LOCKING = None

        if enable_file_locking:
            os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
        elif enable_file_locking is False:
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        elif env_HDF5_USE_FILE_LOCKING is None and mode == 'r':
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

        super(File, self).__init__(filename, mode=mode, swmr=swmr, **kwargs)

        if env_HDF5_USE_FILE_LOCKING is None:
            if mode == 'r':
                del os.environ["HDF5_USE_FILE_LOCKING"]
        else:
            os.environ["HDF5_USE_FILE_LOCKING"] = env_HDF5_USE_FILE_LOCKING
