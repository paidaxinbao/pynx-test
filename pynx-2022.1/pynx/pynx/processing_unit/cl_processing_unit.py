# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


import warnings
import timeit
import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cla
    import pyopencl.elementwise
    import gpyfft

    has_opencl = True
    has_gpyfft = True

    gpyfft_version = gpyfft.version.__version__.split('.')
    if int(gpyfft_version[0]) == 0 and int(gpyfft_version[1]) < 7:
        warnings.warn("WARNING: gpyfft version earlier than 0.7.0. Upgrade may be required", gpyfft_version)
    # else:
    #     print("gpyfft version:", gpyfft.version.__version__)

except ImportError:
    has_gpyfft = False
    has_opencl = False

try:
    import pyopencl as cl
    import pyopencl.array as cla
    import pyopencl.elementwise
    from pyvkfft.opencl import VkFFTApp

    has_opencl = True
    has_vkfft_opencl = True
except ImportError:
    has_vkfft_opencl = False

if has_opencl:
    cl_event = cl.Event
else:
    cl_event = None

from . import ProcessingUnit, ProcessingUnitException
from .cl_resources import cl_resources
from .opencl_device import cl_device_fft_speed

from pynx.utils.math import test_smaller_primes


class CLEvent(object):
    """
    Record an event, and optionally the number of floating-point operations and the number of bytes transferred
    during the event. This is only useful when profiling is used.
    """

    def __init__(self, event: cl_event, nflop: int = 0, nbyte: int = 0):
        self.event = event
        self.nflop = nflop
        self.nbytes = nbyte

    def gflops(self):
        """
        Gflop/s

        :return: the estimated computed Gflop/s
        """
        return self.nflop / max(self.event.profile.end - self.event.profile.start, 1)

    def gbs(self):
        """
        Transfer speed in Gb/s. Depending on the event, this can be host to device or device to device speed.

        :return: the transfer speed in Gb/s
        """
        return self.nbytes / max(self.event.profile.end - self.event.profile.start, 1)


class CLProcessingUnit(ProcessingUnit):
    """
    Processing unit in OpenCL space.

    Handles initializing the context. Kernel initilization must be done in derived classes.
    """

    def __init__(self):
        super(CLProcessingUnit, self).__init__()
        self.cl_ctx = None  # OpenCL context
        self.cl_queue = None  # OpenCL queue
        self.cl_options = None  # OpenCL compile options
        self.gpyfft_plan = None  # gpyfft plan
        self.gpyfft_shape = None  # gpyfft data shape
        self.gpyfft_axes = None  # gpyfft axes
        self.ev = []  # List of openCL events which are queued
        self.profiling = False  # profiling enabled ?
        # Keep track of events for profiling
        self.cl_event_profiling = {}
        # gpyfft plans
        self._gpyfft_plan_v = {}
        # Switch to vkfft instead of gpyfft
        self.use_vkfft = has_vkfft_opencl
        # if self.use_vkfft:
        #     print("Using pyVkFFT instead of gpyfft/clFFT")
        self._vkfft_plan_v = {}
        # We can either use norm=0 - which corresponds to a normalising
        # fft scale of 1/sqrt(transform_size) for each transform,
        # or norm=1, where the norm would be sqrt(transform_size) and
        # 1/sqrt(transform_size) for forward and backward transforms
        # norm=0 is the same as cufft and norm=1 the same as gpyfft
        #
        # We keep the gpyfft choice for consistency, but this could
        # change in the future if fft_scale is systematically used.
        self._vkfft_norm = 1

    def init_cl(self, cl_ctx=None, cl_device=None, fft_size=(1, 1024, 1024), batch=True, gpu_name=None, test_fft=True,
                verbose=True, profiling=None):
        """
        Initialize the OpenCL context and creates an associated command queue

        :param cl_ctx: pyopencl.Context. If none, a default context will be created
        :param cl_device: pyopencl.Device. If none, and no context is given, the fastest GPu will be used.
        :param fft_size: the fft size to be used, for benchmark purposes when selecting GPU. different fft sizes
                         can be used afterwards?
        :param batch: if True, will benchmark using a batch 2D FFT
        :param gpu_name: a (sub)string matching the name of the gpu to be used
        :param test_fft: if True, will benchmark the GPU(s)
        :param verbose: report the GPU found and their speed
        :param profiling: if True, enable profiling for the command queue. If None, current value of self.profiling
                          is used
        :return: nothing
        """
        self.set_benchmark_fft_parameters(fft_size=fft_size, batch=batch)
        self.use_opencl(gpu_name=gpu_name, cl_ctx=cl_ctx, cl_device=cl_device, test_fft=test_fft, verbose=verbose)

        assert test_smaller_primes(fft_size[-1], self.max_prime_fft_radix(), required_dividers=(2,)) \
               and test_smaller_primes(fft_size[-2], self.max_prime_fft_radix(), required_dividers=(2,))

        self.cl_ctx = CLProcessingUnit.get_context(self.cl_device)
        if profiling is not None:
            self.profiling = profiling
        if self.profiling:
            self.cl_queue = cl.CommandQueue(self.cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.cl_queue = cl.CommandQueue(self.cl_ctx)
        self.cl_options = "-cl-mad-enable -cl-fast-relaxed-math"
        # Workaround OpenCL intel Iris Pro wrong calculation of lgamma()...
        if self.cl_ctx.devices[0].name.find('Iris Pro') >= 0:
            self.cl_options += " -DIRISPROBUG=1 "
        self.cl_init_kernels()

    def select_gpu(self, gpu_name=None, gpu_rank=0, ranking="bandwidth", language=None, verbose=False):
        super().select_gpu(gpu_name=gpu_name, gpu_rank=gpu_rank, ranking=ranking, language=None, verbose=verbose)
        # Grab the correct context, as it could have changed when testing
        # for multiple devices...
        if self.cl_device is not None:
            self.get_context(self.cl_device)

    @classmethod
    def get_context(cls, device):
        """
        Static method to get a context, using the static device context dictionary to avoid creating new contexts,
        which will use up the GPU memory.
        :param device: the pyOpenCL device for which a context is desired
        """
        return cl_resources.get_context(device)

    def enable_profiling(self, profiling=True):
        """
        Enable profiling for the main OpenCL queue
        :param profiling: True to enable (the default)
        :return:
        """
        self.profiling = profiling
        if self.cl_ctx is not None:
            # Context already exists, update queue. Otherwise, queue will be initialised when init_cl is called.
            if self.profiling:
                self.cl_queue = cl.CommandQueue(self.cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
            else:
                self.cl_queue = cl.CommandQueue(self.cl_ctx)

    def cl_init_kernels(self):
        """
        Initialize kernels. Virtual function, must be derived.

        :return:
        """

    def cl_fft_set_plan(self, cl_data, axes=(-1, -2)):
        """
        Creates FFT plan, or updates it if the shape of the data or the axes have changed.

        .. deprecated:: 2020.2
        Use :func:`cl_fft_get_plan` instead.

        :param cl_data: the pyopencl.array.Array data
        :param axes: the FFT axes. If None, this will be set to (-2, -1) for 2D and (-3, -2, -1) for 3D
        :return: nothing
        """
        warnings.warn("CLProcessingUnit.cl_fft_set_plan() is deprecated, use cl_fft_get_plan() instead",
                      DeprecationWarning)
        self.gpyfft_plan = self.cl_fft_get_plan(cl_data, axes=axes)

    def cl_fft_get_plan(self, cl_data, axes=(-1, -2), shuffle_axes=False, out=None):
        """
        Get a FFT plan (gpyfft/clfft) according to the given parameters.
        If the plan already exists, it is re-used.

        :param cl_data: the pyopencl.array.Array data
        :param axes: the FFT axes. If None, this will be set to (-2, -1) for 2D and (-3, -2, -1) for 3D
        :param shuffle_axes: if True, will try permutations on the axes order to get the optimal
                             speed.
        :param out: the opencl array for the output. This is only needed for complex->real transform
        :return: the fft plan
        """
        if self.use_vkfft:
            raise RuntimeError("Calling cl_fft_get_plan when use_vkfft=True. Use vkfft_get_plan or (better) "
                               "directly fft() or ifft()")
        if axes is None:
            # Use default values providing good speed with gpyfft
            # TODO: update gpyfft which should now handle this
            axes = tuple(reversed(list(range(cl_data.ndim))))
        elif len(axes) < cl_data.ndim - 1:
            # Kludge to work around "data layout not supported (only single non-transformed axis allowed)" in gpyfft

            # This requires that transform axes indices are negative, so change that first
            axes = list(axes)
            for i in range(len(axes)):
                if axes[i] > 0:
                    axes[i] = -(cl_data.ndim - axes[i])
            axes = tuple(axes)

            s = list(cl_data.shape)
            n = cl_data.ndim
            for i in range(1, cl_data.ndim):
                if i - 1 not in axes and i not in axes and i - n not in axes and i - 1 - n not in axes:
                    # print(i - 1, i, i - n, i - 1 - n, axes)
                    # collapse axes
                    n0 = s.pop(0)
                    s[0] *= n0
                else:
                    break
            s = tuple(s)
            if s != cl_data.shape:
                # print("cl_fft_set_plan: collapsing data shape from ", cl_data.shape, " to ", s, '. Axes:', axes)
                cl_data = cl_data.reshape(s)
                if out is not None:
                    s = list(s)
                    # We keep the last dimension in case of an R2C transform
                    s[-1] = out.shape[-1]
                    out = out.reshape(tuple(s))

        if out is None:
            k = (cl_data.shape, cl_data.dtype, axes, None)
        else:
            k = (cl_data.shape, cl_data.dtype, axes, out.dtype)

        if k in self._gpyfft_plan_v:
            # Note that changing the data and result in the plan is only necessary when
            # using enqueue, but not when using enqueue_arrays()
            self._gpyfft_plan_v[k].data = cl_data
            if out is not None:
                self._gpyfft_plan_v[k].result = out
            return self._gpyfft_plan_v[k]

        real = False
        if out is not None:
            if out.dtype == np.float32:
                real = True  # C2R transform

        if shuffle_axes:
            if real or cl_data.dtype == np.float32:
                warnings.warn("cl_fft_get_plan(): shuffle_axes not supported for C<->R transforms")
            else:
                # Change the axes if necessary, but keep the original ones as key
                flops, dt, axes = cl_device_fft_speed(self.cl_device, cl_data.shape, axes,
                                                      timing=True, shuffle_axes=True)

        self._gpyfft_plan_v[k] = gpyfft.FFT(self.cl_ctx, self.cl_queue, cl_data, out, axes=axes, real=real)

        return self._gpyfft_plan_v[k]

    def vkfft_get_plan(self, cl_data, ndim=None, out=None):
        """
        Get a FFT plan according to the given parameters. if the plan already exists, it is re-used

        :param cl_data: the pyopencl.array.Array data
        :param ndim: the number of dimensions to use for the FFT. By default,
            uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
            array to perform a batched 3D FFT on all the layers. The FFT
            is always performed along the last axes if the array's number
            of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
            on the x and y axes for ndim=2.
        :param out: the opencl array for the output. If None, or if out==cl+data,
            the transform will be inplace.
        :return: the VkFFTApp
        """
        if ndim is None:
            ndim = cl_data.ndim
        if out is None:
            inplace = True
        else:
            if out.data.int_ptr == cl_data.data.int_ptr:
                inplace = True
            else:
                inplace = False
        r2c = cl_data.dtype in [np.float16, np.float32, np.float64]
        k = (cl_data.shape, ndim, cl_data.dtype, inplace)
        if k not in self._vkfft_plan_v:
            self._vkfft_plan_v[k] = VkFFTApp(cl_data.shape, cl_data.dtype, self.cl_queue, ndim,
                                             inplace=inplace, norm=self._vkfft_norm, r2c=r2c)
        return self._vkfft_plan_v[k]

    def cl_fft_free_plans(self):
        """
        .. deprecated:: 2021.1
        Use :func:`free_fft_plans` instead.

        Delete the gpyfft plans from memory. Actual deletion may only occur later (gc...)
        :return: nothing
        """
        warnings.warn("CLProcessingUnit.cl_fft_free_plans() is deprecated, use free_fft_plans() instead",
                      DeprecationWarning)
        self.finish()
        self._gpyfft_plan_v = {}

    def finish(self):
        for e in self.ev:
            e.wait()
        self.cl_queue.finish()

    def fft(self, src, dest, ndim=None, norm=False, return_scale=True, return_dest=False, **kwargs):
        """
        Perform a FFT transform, in or out-of-place. This will automatically
        create a FFT plan as needed using the appropriate backend.
        This supports both C2C and R2C transforms according to the type of
        the src and dest arrays.
        For R2C, if the last dimension of the source is n, the last dimension
        of the half-Hermitian result transform is n//2+1, when out-of-place.
        If an inplace R2C transform is used (only using VkFFT backend), the
        source array must have the last dimension padded by two values, which are
        ignored in the source, and used to keep the same number of bytes in the output.

        :param src: the source array
        :param dest: the destination array. can be the same as src for an in-place transform.
        :param ndim: number of dimensions for the transform, the FFT is done only for
            the last ndim axes. If None, transform applies to all dimensions.
        :param norm: if True, the FFT will keep the L2 norm of the array (abs(d)**2).sum(),
            like the norm='ortho' of scipy and numpy.
        :param return_scale: if True, returns the scale factor by which the dest array must
            be multiplied so that the L2 norm is the same as the src array.
        :param return_dest: if True, return the destination array. This is useful for
            R2C/C2R inplace transforms where the returned array will be a view of
            the source array (for vkfft only).
        :param kwargs: some extra parameters can be given depending on the backend:
            nproc=4: number of parallel process (CPU only, will default to available CPU)
            stream=s: specifying a CUDA stream for the transform
        :return: nothing, or the scale to keep the L2 norm if return_scale is True,
            or the destination array if return_dest is True, or (scale, dest) if
            both are True.
        """
        if self.use_vkfft:
            app = self.vkfft_get_plan(cl_data=src, ndim=ndim, out=dest)
            dest = app.fft(src, dest)
            s = app.get_fft_scale()
        else:
            # Need the reversed axis list for R2C/C2R transform so the x-axis changes dimensions
            if ndim is None:
                ax = tuple(reversed(list(range(src.ndim))))
            else:
                ax = tuple(reversed(list(range(src.ndim))[-ndim:]))
            if src.dtype == np.float64:
                # TODO: support float64<->complex128 R2C transform (upgrade gpyfft ?)
                raise ProcessingUnitException("R2C/C2R transforms not supported for double precision")
            shuffle = np.iscomplexobj(src) and np.iscomplexobj(dest)
            plan = self.cl_fft_get_plan(src, axes=ax, shuffle_axes=shuffle, out=dest)
            self.ev = plan.enqueue(forward=True, wait_for_events=self.ev)
            s = self.fft_scale(src, ndim=ndim)[0]
        if norm:
            dest *= dest.dtype.type(s)
            s = np.float32(1)
        if return_scale:
            if return_dest:
                return s, dest
            else:
                return s
        elif return_dest:
            return dest

    def ifft(self, src, dest, ndim=None, norm=False, return_scale=True, return_dest=False, **kwargs):
        """
        Perform a FFT transform, in or out-of-place. This will automatically
        create a FFT plan as needed using the appropriate backend.
        This supports both C2C and C2R transforms according to the type of
        the src and dest arrays.
        For C2R, if the last dimension of the destination is n, the last dimension
        of the half-Hermitian source is n//2+1, when out-of-place.
        If an inplace C2R transform is used (only using VkFFT backend), the
        destination array will have the last dimension padded by two values,
        to keep the same number of bytes as in the source.

        :param src: the source array
        :param dest: the destination array. can be the same as src for an in-place transform.
        :param ndim: number of dimensions for the transform, the FFT is done only for
            the last ndim axes. If None, transform applies to all dimensions.
        :param norm: if True, the FFT will keep the L2 norm of the array (abs(d)**2).sum(),
            like the norm='ortho' of scipy and numpy.
        :param return_scale: if True, returns the scale factor by which the dest array must
            be multiplied so that the L2 norm is the same as the src array.
        :param return_dest: if True, return the destination array. This is useful for
            R2C/C2R inplace transforms where the returned array will be a view of
            the source array (for vkfft only).
        :param kwargs: some extra parameters can be given depending on the backend:
            nproc=4: number of parallel process (CPU only, will default to available CPU)
            stream=s: specifying a CUDA stream for the transform
        :return: nothing, or the scale to keep the L2 norm if return_scale is True,
            or the destination array if return_dest is True, or (scale, dest) if
            both are True.
        """
        if self.use_vkfft:
            app = self.vkfft_get_plan(cl_data=dest, ndim=ndim, out=src)
            dest = app.ifft(src, dest)
            s = app.get_ifft_scale()
        else:
            # Use gpyfft
            # Need the reversed axis list for R2C/C2R transform so the x-axis changes dimensions
            if ndim is None:
                ax = tuple(reversed(list(range(src.ndim))))
            else:
                ax = tuple(reversed(list(range(src.ndim))[-ndim:]))
            if dest.dtype == np.float64:
                # TODO: support float64<->complex128 R2C transform (upgrade gpyfft ?)
                raise ProcessingUnitException("R2C/C2R transforms not supported for double precision")
            shuffle = np.iscomplexobj(src) and np.iscomplexobj(dest)
            plan = self.cl_fft_get_plan(src, axes=ax, shuffle_axes=shuffle, out=dest)
            self.ev = plan.enqueue(forward=False, wait_for_events=self.ev)
            s = self.fft_scale(dest, ndim=ndim)[1]
        if norm:
            dest *= dest.dtype.type(s)
            s = np.float32(1)

        if return_scale:
            if return_dest:
                return s, dest
            else:
                return s
        elif return_dest:
            return dest

    def fft_scale(self, src, ndim=None, axes=None):
        """
        Get the forward and backward FFT scales by which the destination array
        must be multiplied to keep the L2 norm of the input array.
        :param src: the source (for the forward transform) array or its shape.
            For a R2C/C2R transform, this must be the real array.
        :param ndim: number of dimensions for the transform, the FFT is done only for
            the last ndim axes. If None, transform applies to all dimensions.
        :param axes: reserved for future use
        :return: a tuple with the forward and backward scales
        """
        if isinstance(src, tuple) or isinstance(src, list):
            sh = src
        else:
            sh = src.shape
        if ndim is None:
            s = np.sqrt(np.prod(sh))
        else:
            s = np.sqrt(np.prod(sh[-ndim:]))
        if self.use_vkfft:
            if self._vkfft_norm == 0:
                return 1 / s, 1 / s
        # For gpyfft and self._vkfft_norm=1
        return 1 / s, s

    def free_fft_plans(self):
        """
        If memory was allocated for FFT plans, free it if possible
        :return: nothing
        """
        self.finish()
        self._gpyfft_plan_v = {}
        self._vkfft_plan_v = {}

    def fft_benchmark(self, shape, dtype, ndim, inplace=False, nb=None, nb_test=3, free_plan=True):
        """
        Benchmark a couple of FT and IFT transform
        :param shape: shape of the benchmarked transform
        :param dtype: the dtype for the transform. If it is real, a R2C/C2R transform is performed
        :param ndim: number of dimensions for the transform
        :param inplace: if True, an inplace transform is used
        :param nb: the number of couple of transforms used for testing. If None, this
            will be automatically selected for accuracy.
        :param nb_test: number of tests performed, the best result is returned
        :param free_plan: if True, free plans at the end of the test
        :return: a tuple (dt, throughput) with dt the time for a couple of FT+IFT transforms,
            throughput is the memory throughput computed by assuming that each transform axis
            uses exactly one read and one write. dt in seconds, throughput in Gbytes/s
        """
        a = cla.zeros(self.cl_queue, shape, dtype=dtype)
        a.fill(2.0)
        if dtype in [np.complex64, np.complex128]:
            if inplace:
                b = cla.empty_like(a)
            else:
                b = a
        self.fft(a, b, ndim=ndim, norm=False)
        self.finish()
        # print(100 * 1024 ** 3 / (ndim * 2 * 2 * a.nbytes))
        if nb is None:
            # Target about 0.1s test, minimum 100GB/s
            nb = 10 * 1024 ** 3 / (ndim * 2 * 2 * a.nbytes)
            nb = max(int(nb), 1)
            nb = min(nb, 1000)
        dt_best = 0
        for i in range(nb_test):
            t0 = timeit.default_timer()
            for i in range(nb):
                self.fft(a, b, ndim=ndim, norm=False)
                self.fft(b, a, ndim=ndim, norm=False)
            self.finish()
            dt = timeit.default_timer() - t0
            if dt_best == 0 or dt < dt_best:
                dt_best = dt
        dt_best /= nb
        thr = ndim * 2 * 2 * a.nbytes / dt_best / 1024 ** 3
        if free_plan:
            self.free_fft_plans()
        return dt_best, thr

    def max_prime_fft_radix(self):
        """Return the largest prime number accepted for radix-based FFT transforms.
        Larger prime numbers in the array size decomposition may be possible but
        using a slower Bluestein algorithm"""
        return 13


if has_opencl:
    # This is mostly used during tests while avoiding to destroy & create FFT plans
    default_processing_unit = CLProcessingUnit()
else:
    default_processing_unit = None
