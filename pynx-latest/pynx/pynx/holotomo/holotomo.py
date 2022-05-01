# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['HoloTomoData', 'HoloTomo', 'OperatorHoloTomo', 'algo_string']

import time
import warnings
import multiprocessing
import numpy as np
from ..utils import h5py
from ..operator import Operator, OperatorException
from ..utils.history import History
from ..version import get_git_version

_pynx_version = get_git_version()

warnings.warn("PyNX: you are importing the holotomo module, which is still unstable (API will vary)."
              "Do not use it unless you know the details of current developments.")


class HoloTomoDataStack:
    """
    Class to hold HoloTomo data arrays for a stack of projections in a GPU. Several are needed in order to achieve optimal
    throughput using parallel processing as well as data transfers to and from the GPU.
    Some objects may be left empty because they are not needed (e.g. probe which is the same for all projections, or
    psi which is only kept for some algorithms).
    This can also be used to store data in the host computer for optimised transfer using pinned/page-locked memory.
    """

    def __init__(self, istack=None, iproj=None, iobs=None, obj=None, psi=None, nb=None, obj_phase0=None):
        """

        :param istack: the index for this stack
        :param iproj: the index of the first projection in this stack
        :param iobs: observed intensity, with the center in (0,0) for FFT purposes. Shape: (stack_size, nb_z, ny, nx)
        :param obj: the object. Shape: (stack_size, nb_obj, ny, nx)
        :param psi: the current psi array, which has to be stored for some algorithms.
                    Shape: (stack_size, nb_z, nb_obj, nb_probe, ny, nx)
        :param nb: number of used projections in this stack, in case the total number of projections is not
                   a multiple of stack_size
        :param obj_phase0: the initial estimate for the phase, usually obtained with the Paganin operator. This
                           is used to determine the phase wrapping in ulterior estimates of the object, when
                           applying a delta/beta constraint. This has the same structure as obj, but is stored
                           with a float16|32 type. The phase should be positive, i.e. the object argument is -obj_phase0
        """
        self.istack = istack
        self.iproj = iproj
        self.iobs = iobs
        self.obj = obj
        self.psi = psi
        self.nb = nb
        # True if arrays are currently using pinned memory
        self.pinned_memory = False
        self.obj_phase0 = obj_phase0
        if self.obj_phase0 is not None:
            # self.obj_phase0 = self.obj_phase0.astype(np.float16)
            self.obj_phase0 = self.obj_phase0.astype(np.float32)


class HoloTomoData:
    """
    Class to hold phase contrast imaging data collected at several propagation distances, and several projections.
    The input data will be reorganised to optimise memory transfers between GPU and host.
    """

    def __init__(self, iobs, pixel_size_detector, wavelength, detector_distance, dx=None, dy=None, sample_flag=None,
                 scale_factor=None, stack_size=1, idx=None, padding=0):
        """
        Init function.
        :param iobs: array of observed intensities (scaled to the same pixel size and aligned) with 4 dimensions:
                     - nb_proj different projections along the first axis (dimension can be =1). Note that 'empty'
                       images should be included, so the size is actually nb_proj+nb_empty
                     - the nb_z different distances along the second axis (dimension can be =1)
                     - the 2D images along the last two axis. These may be cropped to be adapted to suitable sizes
                       for GPU FFT.
                     - values < 0 are masked (and stored as -1e38, or -1-I_interp if interpolated)
                     - the input iobs array should be centered on the 2D arrays, as measured
                     Notes:
                     - the iobs data should already be padded, since different propagation distances will have
                        different padded regions. Those should be masked using negative values (-1e38).
        :param pixel_size_detector: the detector pixel size in meters, which must be the same for all images.
        :param wavelength: the experiment wavelength in meters.
        :param detector_distance: the detector distance as an array or list of nb_z distance in meters
        :param dx, dy: shifts of the object with respect to the illumination, in pixels. The shape of the shifts must
                       be (nb_proj, nb_z).
        :param sample_flag: boolean array with nb_proj values, indicating whether there was a sample (True or 1)
                            or if the direct beam was measured (False or 0).
        :param scale_factor: float array with a scale factor for each image. Shape (nb_proj, nb_dist)
        :param stack_size: data will be processed by stacks including one or several projections, and all
                           propagation distances. This is done to manage the GPU memory footprint of the algorithms.
                           stack_size is the number of projections per stack (default 1).
        :param idx: the vector with the indices of the projections analysed. If None, range(nb_proj) will be used
            A negative value can be used for empty beam images, but they will be ignored.
        :param padding: the number of pixels used for padding on each border. This value is only used for
            configuring the interpolation of missing data, and to crop the final output.
        """
        self.nproj, self.nz, self.ny, self.nx = iobs.shape
        if idx is None:
            self.idx = np.arange(self.nproj, dtype=np.int16)
        else:
            self.idx = np.array(idx).astype(np.int16)
        self.stack_size = stack_size
        self.pixel_size_detector = pixel_size_detector
        # TODO: take into account non-monochromaticity ?
        self.wavelength = wavelength
        self.detector_distance = np.array(detector_distance, dtype=np.float32)

        if dx is None or dy is None:
            self.dx = np.zeros((self.nproj, self.nz), dtype=np.float32)
            self.dy = np.zeros((self.nproj, self.nz), dtype=np.float32)
        else:
            assert dx.shape == dy.shape
            self.dx, self.dy = dx.astype(np.float32), dy.astype(np.float32)

        if sample_flag is None:
            self.sample_flag = np.ones(self.nproj, dtype=np.int8)
        else:
            self.sample_flag = sample_flag.astype(np.int8)
        if scale_factor is None:
            self.scale_factor = np.ones(self.nproj, dtype=np.float32)
        else:
            self.scale_factor = scale_factor.astype(np.float32)

        # Padding used on each border these values will be masked.
        self.padding = padding

        # Reorganize data in a list of data stack, which can be later optimized for faster memory transfers
        self.stack_v = []
        n = stack_size
        for i in range(0, self.nproj, n):
            # Numpy is smart enough to only take the available data, even if i+n > len(iobs)
            tmp = iobs[i:i + n]
            self.stack_v.append(HoloTomoDataStack(istack=i // n, iproj=i, iobs=tmp, nb=len(tmp)))

        self.nb_obs = iobs.size


class HoloTomo:
    """ Reconstruction class for two-dimensional phase coherent imaging data, using multiple propagation
    distances and multiple projections, i.e. holo-tomography.
    This class is designed to handle large datasets which cannot fit on a GPU. Complex operations will loop
    over projections by continuously swapping data between host and GPU.
    """

    def __init__(self, data: HoloTomoData, obj: np.array = None, probe: np.array = None, coherent_probe_modes=False):
        """

        :param data: a HoloTomoData object with observed data
        :param obj: an object which will be fitted to the data, which should have 4 dimensions:
                    - projections (considered independent) along the first axis
                    - object modes along the second axis
                    - xy along the last two dimensions, with the same size as the data
                    The data center should be the center of the array.
        :param probe: the illumination corresponding to the data, which should have 4 dimensions:
                    - one illumination per detector distance along the first axis, or just 1 if it is the same
                    - illumination/probe modes along the first axis (>=1)
                    - illumination 2D data along the last two dimensions
                    Since images can be taken with a shifted illumination, the 2D probe size should be larger than
                    the observed intensity and object arrays. This will be automatically corrected, and the
                    dx and dy shifts will also be recentered so that the middle dx and dy
                    values are equal to zero.
                    The data center should be the center of the array.
        :param coherent_probe_modes: if True, the probe modes will be applied coherently, as in the
            "orthogonal probe relaxation" method: for each projection, only one coherent illumination
            which is a linear combination of orthogonal probe modes is used.
            If False, then each probe mode is incoherently illuminating the object, and independently
            propagated to the detector, after which their intensity is summed before comparison to
            the observed data.
            Alternatively an array of shape (nproj, nz, nprobe) of coefficients can be supplied
            to give the coefficients of each mode.
        """
        # TODO: add incoherent background, flat field ?
        self.data = data

        # Empty objects to hold GPU data. This will be HoloTomoDataStack objects when used
        # The 'in' and 'out' stacks are used for memory transfers in // to calculations with the main stack.
        # The probe GPU array (identical for all projections) is kept in the active stack
        self._cl_stack, self._cl_stack_in, self._cl_stack_out = None, None, None
        self._cu_stack, self._cu_stack_in, self._cu_stack_out = None, None, None

        # The timestamp counters record when the holotomo data was last altered, either in the host or GPU memory.
        self._timestamp_counter = 1
        self._cl_timestamp_counter = 0
        self._cu_timestamp_counter = 0

        # Holds the Psi array for the current stack of projection(s)
        self._psi = None

        # Probe array
        self._probe = None

        # Default number of object modes
        self.nb_obj = 1
        # Default number of probe modes
        self.nb_probe = 1
        # Coherent probe modes coefficients. If coherent_probe_modes=True, this will hold
        # a 3-dimensional array (nb_proj, nb_z, nb_mode) holding the coefficients
        # of the linear combination of probe modes for each projection and distance
        self.probe_mode_coeff = None

        self._init_obj(obj)
        self._init_probe(probe, coherent_probe_modes)
        self._init_psi()

        self.llk_poisson = 0
        self.llk_gaussian = 0
        self.llk_euclidian = 0
        self.nb_photons_calc = 0

        # Record the number of cycles (ML, AP, DM, etc...), for history purposes
        self.cycle = 0
        # History record
        self.history = History()

    def _init_obj(self, obj):
        """
        Init the object array by storing it in the HoloTomoData stack. We do not keep a complete 3D array of the object
        in a single array. The shape of each stack is (stack_size, nb_obj, nyo, nxo), where nb_obj
        is the number of object modes (currently 1).

        :return:
        """
        # TODO: use calc_obj_shape to take into account lateral shifts and expand object size
        # For now, assume object and probe have the same size
        n = self.data.stack_size
        if obj is None:
            self.nb_obj = 1
            s = n, 1, self.data.ny, self.data.nx
            for i in range(len(self.data.stack_v)):
                self.data.stack_v[i].obj = np.ones(s, dtype=np.complex64)
        else:
            assert obj.shape[-2] == self.data.ny
            assert obj.shape[-1] == self.data.nx
            self.nb_obj = obj.shape[1]
            for i in range(len(self.data.stack_v)):
                if (i + 1) * n <= len(obj):
                    self.data.stack_v[i].obj = obj[i * n:(i + 1) * n]
                else:
                    dn = len(obj) - i * n
                    self.data.stack_v[i].obj[:dn] = obj[i * n:]
        # Init obj_phase0 to dummy values
        for i in range(len(self.data.stack_v)):
            s = n, 1, self.data.ny, self.data.nx
            # self.data.stack_v[i].obj_phase0 = -np.ones(s, dtype=np.float16) * 1000
            self.data.stack_v[i].obj_phase0 = np.ones(s, dtype=np.float32) * 1000

    def _init_probe(self, probe, coherent_probe_modes):
        """
        Init & resize (if necessary) the probe according to the minimum size calculated for the HoloTomoData.

        :return: nothing.
        """
        if probe is None:
            self.nb_probe = 1
            self._probe = np.ones((self.data.nz, 1, self.data.ny,
                                   self.data.nx), dtype=np.complex64)
        else:
            self.nb_probe = probe.shape[1]
            self._probe = probe.astype(np.complex64)
        assert self._probe.shape == (self.data.nz, self.nb_probe, self.data.ny, self.data.nx)
        if isinstance(coherent_probe_modes, np.ndarray):
            self.probe_mode_coeff = coherent_probe_modes.astype(np.float32)
        elif coherent_probe_modes:
            self.probe_mode_coeff = np.ones((self.data.nproj, self.data.nz, self.nb_probe), dtype=np.float32)
        else:
            self.probe_mode_coeff = None

    def _init_psi(self):
        """
        Init the psi array
        :return: nothing.
        """
        if self.probe_mode_coeff is not None:
            nb_illum = 1
        else:
            nb_illum = self.nb_probe
        self._psi = np.ones((self.data.stack_size, self.data.nz, self.nb_obj, nb_illum,
                             self.data.ny, self.data.nx), dtype=np.complex64)

    def get_x_y(self):
        """
        Get 1D arrays of x and y coordinates, taking into account the pixel size.
        x is an horizontal vector and y vertical.

        :return: a tuple (x, y) of 1D numpy arrays
        """
        ny, nx = self.data.ny, self.data.nx
        px = self.data.pixel_size_detector
        x, y = np.arange(-nx // 2, nx // 2, dtype=np.float32), \
               np.arange(-ny // 2, ny // 2, dtype=np.float32)[:, np.newaxis]
        return x * px, y * px

    def set_probe(self, probe, probe_mode_coefficients=None):
        """
        Give a new value for the probe (illumination) array.

        :param probe: the 3D array to be used
        :param probe_mode_coefficients: if None, coherent probe modes will not be used.
            Otherwise an array of shape (nproj, nz, nb_probe) can be supplied.
        :return: nothing
        """
        self._from_pu()
        self._init_probe(probe, probe_mode_coefficients)
        self._timestamp_counter += 1

    def get_probe(self):
        """
        Get the probe
        :return: the complex array of the probe, with 3 dimensions (1 for the modes)
        """
        self._from_pu()
        return self._probe

    def set_obj(self, obj):
        """
        Give a new value for the object projections. The supplied array shape should be: (nb_proj, ny, nx)

        :param obj: the 3D array to be used
        :return: nothing
        """
        self._from_pu()
        nz, ny, nx = obj.shape
        assert nx == self.data.nx and ny == self.data.ny and nz == self.data.nproj
        for s in self.data.stack_v:
            # We use s.obj[:] because the array may be using pinned memory
            s.obj[:] = obj[s.iproj:s.iproj + s.nb].reshape((s.nb, 1, ny, nx)).astype(np.complex64)
        self._timestamp_counter += 1

    def set_positions(self, dx, dy):
        """
        Set the positions fot all projections and distances
        :param dx, dy: shifts of the object projections with respect to the illumination, in pixels.
            The shape of the shifts must be (nb_proj, nb_z).
        :return:
        """
        self._from_pu()
        self.data.dx = dx.astype(np.float32)
        self.data.dy = dy.astype(np.float32)
        self._timestamp_counter += 1

    def _from_pu(self, psi=False):
        """
        Get object, probe, and optionally psi array from opencl or CUDA memory to host memory for the current stack,
        only if they are more recent than the arrays in the host memory.

        Access to psi is only for development purposes

        :return: nothing
        """
        # TODO: should this rather use an imported FromPU operator ? Would be cleaner, but would importing it..

        # Careful when copying back data: arrays may already be allocated with pinned memory, so we must not
        # change the location of the data - hence the [:] copy
        if self._timestamp_counter < self._cl_timestamp_counter and self._cl_stack is not None:
            from .cl_operator import default_processing_unit
            default_processing_unit.finish()
            self.data.stack_v[self._cl_stack.istack].obj[:] = self._cl_stack.obj.get()
            self._probe = self._cl_stack.probe.get()
            self._timestamp_counter = self._cl_timestamp_counter

        if self._timestamp_counter < self._cu_timestamp_counter and self._cu_stack is not None:
            from .cu_operator import default_processing_unit
            default_processing_unit.finish()
            self.data.stack_v[self._cu_stack.istack].obj[:] = self._cu_stack.obj.get()
            self._probe[:] = self._cu_probe.get()
            if self.probe_mode_coeff is not None:
                self.probe_mode_coeff = self._cu_probe_mode_coeff.get()
            self._timestamp_counter = self._cu_timestamp_counter

        if psi:
            if self._cl_stack is not None:
                if self._cl_stack.psi is not None:
                    self._psi = self._cl_stack.psi.get()
            if self._cu_stack is not None:
                if self._cu_stack.psi is not None:
                    self._psi = self._cu_stack.psi.get()

    def __rmul__(self, x):
        """
        Multiply object (by a scalar).

        This is a placeholder for a function which will be replaced when importing either CUDA or OpenCL operators.
        If called before being replaced, will raise an error

        :param x: the scalar by which the wavefront will be multiplied
        :return:
        """
        if np.isscalar(x):
            raise OperatorException(
                "ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s. Did you import operators ?" % (str(x), str(self)))
        else:
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s." % (str(x), str(self)))

    def __mul__(self, x):
        """
        Multiply object (by a scalar).

        This is a placeholder for a function which will be replaced when importing either CUDA or OpenCL operators.
        If called before being replaced, will raise an error

        :param x: the scalar by which the wavefront will be multiplied
        :return:
        """
        if np.isscalar(x):
            raise OperatorException(
                "ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s. Did you import operators ?" % (str(self), str(x)))
        else:
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s." % (str(self), str(x)))

    def __str__(self):
        return "HoloTomo"

    def reset_history(self):
        """
        Reset history, and set current cycle to zero
        :return: nothing
        """
        self.history = History()
        self.cycle = 0

    def update_history(self, mode='llk', update_obj=False, update_probe=False, update_backgroung=False,
                       update_pos=False, verbose=False, **kwargs):
        """
        Update the history record.
        :param mode: either 'llk' (will record new log-likelihood and number of photons)
                     or 'algorithm' (will only update the algorithm) - for the latter case, algorithm
                     should be given as a keyword argument
        :param verbose: if True, print some info about current process (only if mode=='llk')
        :param kwargs: other parameters to be recorded, e.g. probe_inertia=xx, dt=xx, algorithm='DM'
        :return: nothing
        """
        if mode == 'llk':
            algo = ''
            dt = 0
            if 'algorithm' in kwargs:
                algo = kwargs['algorithm']
            if 'dt' in kwargs:
                dt = kwargs['dt']
            if verbose:
                s = algo_string(algo, self, update_obj, update_probe, update_backgroung, update_pos)
                print("%-10s #%3d LLK= %8.3f(p) %8.3f(g) %8.3f(e), nb photons=%e, dt/cycle=%5.3fs"
                      % (s, self.cycle, self.llk_poisson / self.data.nb_obs, self.llk_gaussian / self.data.nb_obs,
                         self.llk_euclidian / self.data.nb_obs, self.nb_photons_calc, dt))

            self.history.insert(self.cycle, llk_poisson=self.llk_poisson / self.data.nb_obs,
                                llk_gaussian=self.llk_gaussian / self.data.nb_obs,
                                llk_euclidian=self.llk_euclidian / self.data.nb_obs,
                                nb_photons_calc=self.nb_photons_calc, nb_obj=self.nb_obj, nb_probe=self.nb_probe,
                                **kwargs)
        elif 'algo' in mode:
            if 'algorithm' in kwargs:
                self.history.insert(self.cycle, algorithm=kwargs['algorithm'])

    def save_obj_probe_chunk(self, chunk_prefix, save_obj_phase=True, save_obj_complex=False,
                             save_probe=True, dtype=np.float16, verbose=False, crop_padding=True):
        """
        Save the chunk (the projections included in this object) in an hdf5 file
        :param chunk_prefix: the prefix, e.g. "my_sample_%04d" for the filename the data will be saved to.
            The %04d field will be replaced by the first projection. A '.h5' prefix will be appended
            if no "%" is included in the prefix, only '.h5' will be appended to the filename.
        :param save_obj_phase: if True, save the object phase
        :param save_obj_complex: if True, also save the different complex projections of the object
        :param save_probe: if True, save the complex probe (illumination)
        :param dtype: the floating point dtype to save the phase
        :param verbose: if True, will print some information.
        :param crop_padding: if True (the default), the padding area is not saved
        :return:
        """
        if self.data.padding == 0:
            crop_padding = False
        fname = chunk_prefix + ".h5"
        if '%' in fname:
            fname = fname % self.data.idx[0]
        if verbose:
            print("Saving holotomo projections chunk & illumination to: %s" % fname)
        f = h5py.File(fname, "w")
        # f.create_dataset("cxi_version", data=150)
        entry = f.create_group("/entry_1")
        entry_path = "/entry_1"
        f.attrs['default'] = 'entry_1'
        f.attrs['creator'] = 'PyNX'
        # f.attrs['NeXus_version'] = '2018.5'  # Should only be used when the NeXus API has written the file
        f.attrs['HDF5_Version'] = h5py.version.hdf5_version
        f.attrs['h5py_version'] = h5py.version.version
        entry.attrs['NX_class'] = 'NXentry'

        entry.create_dataset("program_name", data="PyNX %s" % _pynx_version)
        entry.create_dataset("start_time", data=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())))

        # X & Y axis data for NeXuS plotting
        padding = self.data.padding
        ny, nx = self.data.stack_v[0].obj.shape[-2:]
        if crop_padding:
            ny, nx = ny - 2 * padding, nx - 2 * padding
        px = self.data.pixel_size_detector
        unit_scale = np.log10(max(nx * px, ny * px))
        if unit_scale < -6:
            unit_name = "nm"
            unit_scale = 1e9
        elif unit_scale < -3:
            unit_name = u"µm"
            unit_scale = 1e6
        elif unit_scale < 0:
            unit_name = "mm"
            unit_scale = 1e3
        else:
            unit_name = "m"
            unit_scale = 1

        if save_obj_phase:
            entry.attrs['default'] = 'object_phase'
            result_1 = entry.create_group("result_1")
            entry["object_phase"] = h5py.SoftLink(entry_path + '/result_1')
            result_1['title'] = 'Object Phase'
            result_1.attrs['NX_class'] = 'NXdata'
            result_1.attrs['signal'] = 'data'
            # Assemble object phase
            nproj = self.data.sample_flag.sum()
            obj_phase = np.empty((nproj, ny, nx), dtype=dtype)
            i = 0
            idx = []
            for s in self.data.stack_v:
                for ii in range(s.nb):
                    if self.data.sample_flag[s.iproj + ii]:
                        idx.append(self.data.idx[s.iproj + ii])
                        obj_phase0 = s.obj_phase0[ii, 0]
                        if crop_padding:
                            op = np.angle(s.obj[ii, 0, padding:-padding, padding:-padding])
                            obj_phase0 = obj_phase0[padding:-padding, padding:-padding]
                        else:
                            op = np.angle(s.obj[ii, 0])
                        op = obj_phase0 + ((op - obj_phase0) % (2 * np.pi))
                        op -= ((op - obj_phase0) // np.pi) * 2 * np.pi
                        obj_phase[i] = op
                        i += 1

            result_1.create_dataset("projection_idx", data=np.array(idx, dtype=np.int16), chunks=True, shuffle=True,
                                    compression="gzip")
            result_1.create_dataset("data", data=obj_phase, chunks=(1, ny, nx), shuffle=True, compression="gzip")
            result_1["data"].attrs['interpretation'] = 'image'
            result_1.create_dataset("data_space", data="real")
            result_1.create_dataset("image_size", data=[px * nx, px * ny])
            # Store probe pixel size (not in CXI specification)
            result_1.create_dataset("x_pixel_size", data=px)
            result_1.create_dataset("y_pixel_size", data=px)
            # X & Y axis data for NeXuS plotting
            result_1.attrs['axes'] = np.array(['row_coords', 'col_coords'], dtype=h5py.special_dtype(vlen=str))
            # Flip to have origin at top, left
            yc = np.flip((np.arange(ny) * px - ny * px / 2) * unit_scale)
            result_1.create_dataset('row_coords', data=yc)
            result_1['row_coords'].attrs['units'] = unit_name
            result_1['row_coords'].attrs['long_name'] = 'Y (%s)' % unit_name
            xc = (np.arange(nx) * px - nx * px / 2) * unit_scale
            result_1.create_dataset('col_coords', data=xc)
            result_1['col_coords'].attrs['units'] = unit_name
            result_1['col_coords'].attrs['long_name'] = 'X (%s)' % unit_name

        if save_obj_complex:
            if not save_obj_phase:
                entry.attrs['default'] = 'object'
            result_2 = entry.create_group("result_2")
            entry["object"] = h5py.SoftLink(entry_path + '/result_2')
            result_2['title'] = 'Object (complex)'
            result_2.attrs['NX_class'] = 'NXdata'
            result_2.attrs['signal'] = 'data'
            # Assemble object phase
            nproj = self.data.sample_flag.sum()
            dtype_cplx = np.complex64
            obj = np.empty((nproj, ny, nx), dtype=dtype_cplx)
            i = 0
            idx = []
            for s in self.data.stack_v:
                for ii in range(s.nb):
                    if self.data.sample_flag[s.iproj + ii]:
                        idx.append(self.data.idx[s.iproj + ii])
                        ob = s.obj[ii, 0]
                        if crop_padding:
                            ob = ob[..., padding:-padding, padding:-padding]
                        obj[i] = ob
                        i += 1

            result_2.create_dataset("projection_idx", data=np.array(idx, dtype=np.int16), chunks=True, shuffle=True,
                                    compression="gzip")
            result_2.create_dataset("data", data=obj, chunks=True, shuffle=True, compression="gzip")
            result_2["data"].attrs['interpretation'] = 'image'
            result_2.create_dataset("data_space", data="real")
            result_2.create_dataset("image_size", data=[px * nx, px * ny])
            # Store probe pixel size (not in CXI specification)
            result_2.create_dataset("x_pixel_size", data=px)
            result_2.create_dataset("y_pixel_size", data=px)
            # X & Y axis data for NeXuS plotting
            result_2.attrs['axes'] = np.array(['row_coords', 'col_coords'], dtype=h5py.special_dtype(vlen=str))
            # Flip to have origin at top, left
            yc = np.flip((np.arange(ny) * px - ny * px / 2) * unit_scale)
            result_2.create_dataset('row_coords', data=yc)
            result_2['row_coords'].attrs['units'] = unit_name
            result_2['row_coords'].attrs['long_name'] = 'Y (%s)' % unit_name
            xc = (np.arange(nx) * px - nx * px / 2) * unit_scale
            result_2.create_dataset('col_coords', data=xc)
            result_2['col_coords'].attrs['units'] = unit_name
            result_2['col_coords'].attrs['long_name'] = 'X (%s)' % unit_name

        if save_probe:
            result_3 = entry.create_group("result_3")
            entry["probe"] = h5py.SoftLink(entry_path + '/result_3')
            result_3['title'] = 'Probe'
            result_3.attrs['NX_class'] = 'NXdata'
            result_3.attrs['signal'] = 'data'
            if crop_padding:
                pr = self.get_probe()[..., padding:-padding, padding:-padding]
            else:
                pr = self.get_probe()
            result_3.create_dataset("data", data=pr, chunks=True, shuffle=True, compression="gzip")
            result_3["data"].attrs['interpretation'] = 'image'
            result_3.create_dataset("data_space", data="real")
            result_3.create_dataset("image_size", data=[px * nx, px * ny])
            # Store probe pixel size (not in CXI specification)
            result_3.create_dataset("x_pixel_size", data=px)
            result_3.create_dataset("y_pixel_size", data=px)
            # X & Y axis data for NeXuS plotting
            result_3.attrs['axes'] = np.array(['row_coords', 'col_coords'], dtype=h5py.special_dtype(vlen=str))
            # Flip to have origin at top, left
            yc = np.flip((np.arange(ny) * px - ny * px / 2) * unit_scale)
            result_3.create_dataset('row_coords', data=yc)
            result_3['row_coords'].attrs['units'] = unit_name
            result_3['row_coords'].attrs['long_name'] = 'Y (%s)' % unit_name
            xc = (np.arange(nx) * px - nx * px / 2) * unit_scale
            result_3.create_dataset('col_coords', data=xc)
            result_3['col_coords'].attrs['units'] = unit_name
            result_3['col_coords'].attrs['long_name'] = 'X (%s)' % unit_name
            if self.probe_mode_coeff is not None:
                result_3.create_dataset('probe_mode_coeff', data=self.probe_mode_coeff)

    def get_obj_phase_unwrapped(self, crop_padding=True, dtype=np.float16, idx=None, nproc=1):
        """
        Get an array of the object phase, unwrapped based on the initial phase from Paganin/CTF
        :param crop_padding: if True (the default), the padded area is not saved
        :param dtype: the numpy dtype to use for the phase (defaults to float16)
        :param idx: if None, all the projections are returned. If idx is a number or
            a list/array of projections, only those are returned. These are
            the projections index, as given to the HoloTomoData object, i.e.
            HoloTomoData.idx .
        :return: idx, ph where idx is the index of the analysed frames, and ph, a 3D array
            with the phases for each projection
        """
        padding = self.data.padding
        if padding == 0:
            crop_padding = False
        ny, nx = self.data.stack_v[0].obj.shape[-2:]
        if isinstance(idx, int) or isinstance(idx, np.integer):
            idx = [idx]
        elif idx is None:
            idx = self.data.idx
        nproj = len(idx)
        if crop_padding:
            ny, nx = ny - 2 * padding, nx - 2 * padding
        obj_phase = np.empty((nproj, ny, nx), dtype=dtype)
        i = 0
        vidx = []
        for s in self.data.stack_v:
            for ii in range(s.nb):
                if self.data.sample_flag[s.iproj + ii]:
                    idxtmp = self.data.idx[s.iproj + ii]
                    if idxtmp not in idx:
                        continue
                    vidx.append(idxtmp)
                    if s.obj_phase0 is not None:
                        obj_phase0 = s.obj_phase0[ii, 0]
                        if crop_padding:
                            op = np.angle(s.obj[ii, 0, padding:-padding, padding:-padding])
                            obj_phase0 = obj_phase0[padding:-padding, padding:-padding]
                        else:
                            op = np.angle(s.obj[ii, 0])
                        op = obj_phase0 + ((op - obj_phase0) % (2 * np.pi))
                        tmp = op - obj_phase0
                        op[tmp >= np.pi] -= 2 * np.pi
                        op[tmp < -np.pi] += 2 * np.pi
                    else:
                        if crop_padding:
                            op = np.angle(s.obj[ii, 0, padding:-padding, padding:-padding])
                        else:
                            op = np.angle(s.obj[ii, 0])
                    obj_phase[i] = op
                    i += 1
        return vidx, obj_phase


class OperatorHoloTomo(Operator):
    """
    Base class for an operator on CDI objects, not requiring a processing unit.
    """

    def timestamp_increment(self, pci: HoloTomo):
        pci._timestamp_counter += 1


def algo_string(algo_base, p: HoloTomo, update_object, update_probe, update_background=False, update_pos=False):
    """
    Get a short string for the algorithm being run, e.g. 'DM/o/3p' for difference map with 1 object and 3 probe modes.

    :param algo_base: 'AP' or 'ML' or 'DM'
    :param p: the holotomo object
    :param update_object: True if updating the object
    :param update_probe: True if updating the probe
    :param update_background: True if updating the background
    :param update_pos: True if updating the positions
    :return: a short string for the algorithm
    """
    s = algo_base

    if update_object:
        s += "/"
        if p.nb_obj > 1:
            s += "%d" % p.nb_obj
        s += "o"

    if update_probe:
        s += "/"
        if p.nb_probe > 1:
            s += "%d" % p.nb_probe
        s += "p"

    if update_background:
        s += "/b"

    if update_pos:
        s += "/t"

    return s
