# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import warnings
import timeit

warnings.filterwarnings("ignore", message="creating CUBLAS context to get version number")
import numpy as np

from .cuda_device import cu_drv, cua, has_cuda, has_vkfft_cuda, VkFFTApp, cu_fft
from . import ProcessingUnit, ProcessingUnitException, ProcessingUnitWarning
from .cu_resources import cu_resources
from pynx.utils.math import test_smaller_primes


# does.not.work.
# warnings.filterwarnings('once', "cu_fft_get_plan: the 'batch' parameter has been deprecated. Use ndim instead.")


class CUProcessingUnit(ProcessingUnit):
    """
    Processing unit in CUDA space.

    Handles initializing the context and fft plan. Kernel initialization must be done in derived classes.
    """

    def __init__(self):
        super(CUProcessingUnit, self).__init__()
        self.cu_ctx = None  # CUDA context
        self.cu_options = None  # CUDA compile options
        self.cu_arch = None
        self.profiling = False
        self._cufft_plan_v = {}
        self._vkfft_app_v = {}
        self.cu_memory_pool = None
        self.warn = True
        # Switch to vkfft instead of cufft
        self.use_vkfft = has_vkfft_cuda
        # if self.use_vkfft:
        #     print("Using pyVkFFT instead of cuFFT")

    def init_cuda(self, cu_ctx=None, cu_device=None, fft_size=(1, 1024, 1024), batch=True, gpu_name=None, test_fft=True,
                  verbose=True):
        """
        Initialize the OpenCL context and creates an associated command queue

        :param cu_ctx: pycuda.driver.Context. If none, a default context will be created
        :param cu_device: pycuda.driver.Device. If none, and no context is given, the fastest GPu will be used.
        :param fft_size: the fft size to be used, for benchmark purposes when selecting GPU. different fft sizes
                         can be used afterwards?
        :param batch: if True, will benchmark using a batch 2D FFT
        :param gpu_name: a (sub)string matching the name of the gpu to be used
        :param test_fft: if True, will benchmark the GPU(s)
        :param verbose: report the GPU found and their speed
        :return: nothing
        """
        self.set_benchmark_fft_parameters(fft_size=fft_size, batch=batch)
        self.use_cuda(gpu_name=gpu_name, cu_ctx=cu_ctx, cu_device=cu_device, test_fft=test_fft, verbose=verbose)

        assert test_smaller_primes(fft_size[-1], self.max_prime_fft_radix(), required_dividers=(2,)) \
               and test_smaller_primes(fft_size[-2], self.max_prime_fft_radix(), required_dividers=(2,))

        if self.cu_ctx is None:
            self.cu_ctx = CUProcessingUnit.get_context(self.cu_device)

        self.cu_options = ["-use_fast_math"]
        # TODO: KLUDGE. Add a workaround if the card (Pascal) is more recent than the driver...
        # if cu_drv.get_version() == (7,5,0) and self.cu_device.compute_capability() == (6,1):
        #    print("WARNING: using a Pascal card (compute capability 6.1) with driver .5: forcing arch=sm_52")
        #    self.cu_arch = 'sm_50'
        self.cu_init_kernels()

    def select_gpu(self, gpu_name=None, gpu_rank=0, ranking="bandwidth", language=None, verbose=False):
        super().select_gpu(gpu_name=gpu_name, gpu_rank=gpu_rank, ranking=ranking, language=None, verbose=verbose)
        # Grab the correct context, as it could have changed when testing
        # for multiple devices...
        if self.cu_device is not None:
            self.get_context(self.cu_device)

    @classmethod
    def get_context(cls, device):
        """
        Static method to get a context, using the static device context dictionary to avoid creating new contexts,
        which will use up the GPU memory.
        :param device: the pyCUDA device for which a context is desired
        """
        return cu_resources.get_context(device)

    def get_memory_pool(cls):
        """
        Get the global memory pool
        :return:
        """
        return cu_resources.get_memory_pool()

    def cu_init_kernels(self):
        """
        Initialize kernels. Virtual function, must be derived.

        :return: nothing
        """

    def cu_fft_set_plan(self, cu_data, batch=True, stream=None):
        """
        Creates FFT plan, or updates it if the shape of the data or the axes have changed.

        .. deprecated:: 2020.2
        Use :func:`cu_fft_get_plan` instead.

        :param cu_data: an array from which the FFT shape will be extracted
        :param batch: if True, perform a 2D batch FFT on an n-dimensional array (n>2). Ignored if data is 2D.
                      The FFT is computed over the last two dimensions.
        :param stream: the cuda stream to be associated with this plan
        :return: nothing
        """
        warnings.warn("CUProcessingUnit.cu_fft_set_plan() is deprecated, use cu_fft_get_plan() instead",
                      DeprecationWarning)
        self.cufft_plan = self.cu_fft_get_plan(cu_data, batch, stream)

    def cu_fft_get_plan(self, cu_data, ndim=None, stream=None, dtype_dest=None, dtype_src=None,
                        inplace=False, **kwargs):
        """
        Get a FFT plan (cuFFT of pyvkfft app) according to the given parameters.
        If the plan already exists, it is re-used

        :param cu_data: an array from which the FFT shape will be extracted. Alternatively
            this can be the shape of the array, but dtype_src and dtype_dest must be given
        :param ndim: number of dimensions for the transform, the FFT is done only for
            the last ndim axes. If None, the transform applies to all dimensions.
        :param stream: the cuda stream to be associated with this plan
        :param dtype_dest: if None, the destination array dtype is taken from cu_data.
            This can be used for complex<->real transforms
        :param dtype_src: if None, the source array dtype is taken from cu_data.
            This can be used for complex<->real transforms
        :param inplace: if False (the default), perform and out-of-place transform
        :return: the cufft plan
        """
        if isinstance(ndim, bool):
            raise ProcessingUnitException("ndim is given as a bool, which probably means it "
                                          "was intended for the old 'batch' parameter. "
                                          "Please use ndim instead (e.g. ndim=2)")
        if 'batch' in kwargs:
            if self.warn:
                # Why do filters not work (still gives 8 warnings during a CDI analysis with PSF)
                warnings.warn("cu_fft_get_plan: the 'batch' parameter has been deprecated. Use ndim instead.",
                              stacklevel=2)
                self.warn = False
            if kwargs['batch']:
                if ndim is None:
                    ndim = 2
                elif ndim != 2:
                    raise ProcessingUnitException("Both batch and ndim have been given. Please only use ndim.")
        if isinstance(cu_data, tuple) or isinstance(cu_data, list):
            fft_shape = cu_data
        else:
            fft_shape = cu_data.shape

        if dtype_dest is None:
            dtype_dest = cu_data.dtype

        if dtype_src is None:
            dtype_src = cu_data.dtype

        if ndim is None:
            ndim = len(fft_shape)

        if self.use_vkfft:
            if not has_vkfft_cuda:
                raise ProcessingUnitException("use_vkfft is True but has_vkfft_cuda is False !")

            k = (fft_shape, dtype_src, ndim, stream, inplace)
            if k not in self._vkfft_app_v:
                if dtype_src in [np.float32, np.float64]:
                    r2c = True
                else:
                    r2c = False
                self._vkfft_app_v[k] = VkFFTApp(fft_shape, dtype_src, ndim=ndim, inplace=inplace,
                                                stream=stream, norm=0, r2c=r2c)
            return self._vkfft_app_v[k]
        else:
            k = (fft_shape, dtype_src, ndim, stream)
            if k not in self._cufft_plan_v:
                self._cufft_plan_v[k] = cu_fft.Plan(fft_shape[-ndim:], dtype_src, dtype_dest,
                                                    batch=int(np.product(fft_shape[:-ndim])), stream=stream)

        return self._cufft_plan_v[k]

    def cu_fft_free_plans(self):
        """
        .. deprecated:: 2021.1
        Use :func:`free_fft_plans` instead.

        Delete the cufft/pyvkfft plans from memory. Actual deletion may only occur later (gc...)
        :return: nothing
        """
        warnings.warn("CUProcessingUnit.cu_fft_free_plans() is deprecated, use free_fft_plans() instead",
                      DeprecationWarning)
        self.free_fft_plans()

    def enable_profiling(self, profiling=True):
        """
        Enable profiling
        :param profiling: True to enable (the default)
        :return:
        """
        if profiling and self.profiling is False:
            cu_drv.start_profiler()
        elif self.profiling and profiling is False:
            cu_drv.stop_profiler()
        self.profiling = profiling

    def finish(self):
        self.cu_ctx.synchronize()

    def fft(self, src, dest, ndim=None, norm=False, return_scale=True, **kwargs):
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
        :param kwargs: some extra parameters can be given depending on the backend:
            nproc=4: number of parallel process (CPU only, will default to available CPU)
            stream=s: specifying a CUDA stream for the transform
        :return: nothing, or the scale to keep the L2 norm if return_scale is True.
        """
        if 'stream' in kwargs:
            stream = kwargs['stream']
        else:
            stream = None
        if np.iscomplexobj(src) and np.isrealobj(dest):
            raise ProcessingUnitException("C2R transforms are only allowed for backward (ifft) transforms")
        if self.use_vkfft:
            if src.gpudata == dest.gpudata:
                inplace = True
            else:
                inplace = False
            plan = self.cu_fft_get_plan(src.shape, ndim=ndim, stream=stream, dtype_src=src.dtype,
                                        dtype_dest=dest.dtype, inplace=inplace)
        else:
            plan = self.cu_fft_get_plan(src.shape, ndim=ndim, stream=stream, dtype_src=src.dtype, dtype_dest=dest.dtype)
        if self.use_vkfft:
            plan.fft(src, dest)
        else:
            cu_fft.fft(src, dest, plan, scale=False)
        s = self.fft_scale(src, ndim=ndim)[0]
        if norm:
            dest *= s

        if return_scale:
            if norm:
                return 1
            else:
                return s

    def ifft(self, src, dest, ndim=None, norm=False, return_scale=True, **kwargs):
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
        :param kwargs: some extra parameters can be given depending on the backend:
            nproc=4: number of parallel process (CPU only, will default to available CPU)
            stream=s: specifying a CUDA stream for the transform
        :return: nothing, or the scale to keep the L2 norm if return_scale is True.
        """
        if 'stream' in kwargs:
            stream = kwargs['stream']
        else:
            stream = None
        if np.isrealobj(src) and np.iscomplexobj(dest):
            raise ProcessingUnitException("R2C transforms are only allowed for forward (ifft) transforms")
        # Use the destination shape in case this is a C2R transform as the real size is the reference
        if self.use_vkfft:
            if src.gpudata == dest.gpudata:
                inplace = True
            else:
                inplace = False
            # The plan goes both ways for R2C/C2R with vkfft
            plan = self.cu_fft_get_plan(dest.shape, ndim=ndim, stream=stream, dtype_src=dest.dtype,
                                        dtype_dest=src.dtype, inplace=inplace)
        else:
            plan = self.cu_fft_get_plan(dest.shape, ndim=ndim, stream=stream, dtype_src=src.dtype,
                                        dtype_dest=dest.dtype)
        if self.use_vkfft:
            plan.ifft(src, dest)
        else:
            cu_fft.ifft(src, dest, plan, scale=False)
        s = self.fft_scale(dest, ndim=ndim)[1]
        if norm:
            dest *= s

        if return_scale:
            if norm:
                return 1
            else:
                return s

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
        return 1 / s, 1 / s

    def free_fft_plans(self):
        """
        If memory was allocated for FFT plans, free it if possible
        :return: nothing
        """
        if cu_fft is not None:
            if cu_fft.cufft.cufftGetVersion() >= 10200:
                with warnings.catch_warnings():
                    txt = "WARNING: cuFFT plan destruction inhibited as a workaround for " \
                          "an issue with CUDA>=11.0. See https://github.com/lebedov/scikit-cuda/issues/308"
                    warnings.filterwarnings('once', message=txt, append=True)
                    warnings.warn(txt, UserWarning)
            else:
                self._cufft_plan_v = {}
        self._vkfft_app_v = {}

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
        a = cua.zeros(shape, dtype=dtype)
        a.fill(2.0)
        if dtype in [np.complex64, np.complex128]:
            if inplace:
                b = cua.empty_like(a)
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
        if self.use_vkfft:
            return 13
        else:
            # Some transforms with larger primes (11 notably) also use efficient
            # radix transforms, but coverage detail is undocumented in cuFFT
            return 7


if has_cuda:
    # This is mostly used during tests while avoiding to destroy & create FFT plans
    default_processing_unit = CUProcessingUnit()
else:
    default_processing_unit = None
