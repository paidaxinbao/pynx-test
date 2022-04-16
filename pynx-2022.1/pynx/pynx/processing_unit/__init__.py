# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import os
import platform
import warnings
import timeit
from psutil import cpu_count
from .cuda_device import has_cuda
from .opencl_device import has_opencl
import numpy as np
# scipy.fft must be explicitly imported
import scipy.fft
from scipy.fft import fftn, ifftn, rfftn, irfftn

try:
    from mpi4py import MPI

    mpi = MPI.COMM_WORLD
    if mpi.Get_size() == 1:
        mpi = None
except ImportError:
    mpi = None


class ProcessingUnitException(Exception):
    pass


class ProcessingUnitWarning(UserWarning):
    pass


# Exclude languages based on PYNX_PU environment variable
if 'PYNX_PU' in os.environ:
    pynx_py_env = os.environ['PYNX_PU'].lower().split('.')
    if 'cuda' in pynx_py_env[0]:
        if has_cuda is False:
            raise ProcessingUnitException('PyNX processing unit: CUDA language is selected from the PYNX_PU '
                                          'environment variable, but CUDA is not available !')
        has_opencl = False
        print("PyNX processing unit: select CUDA language from PYNX_PU environment variable")
    elif 'opencl' in pynx_py_env[0]:
        if has_opencl is False:
            raise ProcessingUnitException('PyNX processing unit: OpenCL language is selected from the PYNX_PU '
                                          'environment variable, but OpenCL is not available !')
        has_cuda = False
        print("PyNX processing unit: select OpenCL language from PYNX_PU environment variable")
    elif 'cpu' in pynx_py_env[0]:
        has_opencl = False
        has_cuda = False
        print("PyNX processing unit: select CPU from PYNX_PU environment variable - OpenCL and CUDA will be ignored")


class ProcessingUnit(object):

    def __init__(self):
        # Generic parameters
        self.pu_language = None  # 'cpu', 'opencl' or 'cuda' - None if un-initialised
        # OpenCL device
        self.cl_device = None
        # CUDA device
        self.cu_device = None

        # Default fft size for benchmarking
        self.benchmark_fft_size = (16, 400, 400)
        # if True, will perform a batch FFT (e.g. 16 2D FFT of size 256x256 with the default parameters)
        self.benchmark_fft_batch = True
        try:
            # Get the real number of processor cores available
            # os.sched_getaffinity is only available on some *nix platforms
            self.nproc = len(os.sched_getaffinity(0)) * cpu_count(logical=False) // cpu_count(logical=True)
        except AttributeError:
            self.nproc = os.cpu_count()

    def enable_profiling(self, profiling=True):
        """
        Enable profiling (for OpenCL only)
        :param profiling: True to enable (the default)
        :return:
        """
        print('Profiling can only be enabled when using OpenCL')

    def set_benchmark_fft_parameters(self, fft_size, batch):
        """
        Set FFT size and axis for benchmarking processing units. fft_size=(16, 400, 400) and batch=True is the
        default and will perform a batch 2D FFT on 16 2D arrays.
        :param benchmark_fft_size: FFT size (3D or stacked 2D) for benchmarking.
        :param batch: if True, will perform a stacked 2D FFT. Otherwise a 3D FFT will be performed
        :return: nothing
        """
        self.benchmark_fft_size = fft_size
        self.benchmark_fft_batch = batch

    def set_device(self, d=None, verbose=True, test_fft=True):
        """
        Set the computing device to be used. A quick FFT test is also performed.

        :param d: either a pyopencl.Device or pycuda.Driver.Device. Otherwise the language is set to 'cpu'
        :param verbose: if True, will print information about the used device
        :param test_fft: if True (the default), the FFT speed is evaluated.
        :return: True if the FFT calculation was correctly achieved.
        """
        if has_opencl:
            from . import opencl_device
            if self.benchmark_fft_batch:
                fft_axis = (-2, -1)
            else:
                fft_axis = None
            if type(d) is opencl_device.cl.Device:
                if test_fft:
                    opencl_device.cl_device_fft_speed(d, self.benchmark_fft_size, fft_axis, True)
                self.cl_device = d
                self.pu_language = 'opencl'

                if verbose:
                    print("Using OpenCL GPU: %s" % (self.cl_device.name))
                return True
        if has_cuda:
            from . import cuda_device
            if type(d) is cuda_device.cu_drv.Device:
                self.cu_device = d
                self.pu_language = 'cuda'
                if verbose:
                    print("Using CUDA GPU: %s" % (self.cu_device.name()))
                return True
        if verbose:
            print("Using CPU")
        self.pu_language = 'cpu'
        self.max_prime_fft = 2 ** 16 + 1

    def use_opencl(self, gpu_name=None, platform=None, cl_ctx=None, cl_device=None, test_fft=True, verbose=True):
        """
        Use an OpenCL device for computing. This method should only be called to more selectively choose an OpenCL
        device from a known context or platform. select_gpu() should be preferred.

        :param gpu_name: a (sub)string matching the name of the gpu to be used
        :param platform: a (sub)string matching the name of the opencl platform to be used
        :param cl_ctx: if already know, a pyopencl context can be supplied
        :param cl_device: if already know, a pyopencl device can be supplied
        :param test_fft: if True, will benchmark the devices using FFT
        :return: True if an opencl device was found and FFT calculations are working (using gpyfft & clFFT) with it
        """
        if has_opencl is False:
            return self.set_device(test_fft=test_fft, verbose=verbose)
        if cl_ctx is not None and type(cl_ctx) is opencl_device.cl.Context:
            return self.set_device(cl_ctx.devices[0], test_fft=test_fft, verbose=verbose)
        if cl_device is not None and type(cl_device) is opencl_device.cl.Device:
            return self.set_device(cl_device, test_fft=test_fft, verbose=verbose)
        # Try to find the fastest GPU
        if self.benchmark_fft_batch:
            fft_axis = (-2, -1)
        else:
            fft_axis = None
        if test_fft:
            gpu_speed = opencl_device.available_gpu_speed(cl_platform=platform, fft_shape=self.benchmark_fft_size,
                                                          axes=fft_axis, min_gpu_mem=None, verbose=verbose,
                                                          gpu_name=gpu_name)
        else:
            gpu_speed = opencl_device.available_gpu_speed(cl_platform=platform, fft_shape=None, axes=None,
                                                          min_gpu_mem=None, verbose=verbose, gpu_name=gpu_name)
        if len(gpu_speed) == 0:
            return self.set_device(verbose=verbose)
        if gpu_name is not None:
            found_gpu = False
            for g in gpu_speed:
                if g[0].name.lower().count(gpu_name.lower()) > 0:
                    return self.set_device(g[0], test_fft=False, verbose=verbose)
            return found_gpu
        return self.set_device(gpu_speed[0][0], test_fft=False, verbose=verbose)

    def use_cuda(self, gpu_name=None, cu_ctx=None, cu_device=None, test_fft=False, verbose=True):
        """
        Use a CUDA device for computing. This method should only be called to more selectively choose an CUDA
        device from a known context or device. select_gpu() should be preferred.

        :param gpu_name: a (sub)string matching the name of the gpu to be used
        :param cu_ctx: if already known, a pycuda context can be supplied
        :param cu_device: if already know, a pycuda device can be supplied
        :param test_fft: if True, will benchmark the devices using FFT
        :return: True if an cuda device was found and FFT calculations are working with it
        """
        if has_cuda is False:
            return self.set_device(test_fft=test_fft, verbose=verbose)
        if cu_ctx is not None and type(cu_ctx) is cuda_device.cu_drv.Context:
            return self.set_device(cu_ctx.get_device(), test_fft=test_fft, verbose=verbose)
        if cu_device is not None and type(cu_device) is cuda_device.cu_drv.Device:
            return self.set_device(cu_device, test_fft=test_fft, verbose=verbose)
        # Try to find the fastest GPU
        if test_fft:
            gpu_speed = cuda_device.available_gpu_speed(fft_shape=self.benchmark_fft_size,
                                                        batch=self.benchmark_fft_batch, verbose=verbose)
        else:
            gpu_speed = cuda_device.available_gpu_speed(fft_shape=None, batch=None, verbose=verbose)

        if len(gpu_speed) == 0:
            return False
        if gpu_name is not None:
            for g in gpu_speed:
                if g[0].name().lower().count(gpu_name.lower()) > 0:
                    return self.set_device(g[0], test_fft=False, verbose=verbose)
            return self.set_device()
        return self.set_device(gpu_speed[0][0], test_fft=False, verbose=verbose)

    def use_gpu(self, gpu_name=None, gpu_rank=0, ranking="fft_speed", language=None):
        warnings.warn('pynx.ProcessingUnit.use_gpu() is deprecated, please use select_gpu() instead',
                      DeprecationWarning)
        return self.select_gpu(gpu_name=gpu_name, gpu_rank=gpu_rank, ranking=ranking, language=language)

    def select_gpu(self, gpu_name=None, gpu_rank=0, ranking="bandwidth", language=None, verbose=False):

        """
        Automatically select the fastest GPU, either from CUDA or OpenCL.
        If no gpu name or language is given, but there is an environment variable 'PYNX_PU', that variable will
        be used to determine the processing unit used. This variable (case-insensitive) can correspond to a language
        (OpenCL, CUDA or CPU) or match a gpu name (e.g. 'Titan'), or be a combination separated by a dot,
        e.g. 'OpenCL.Titan', or be a language and a rank, e.g. 'CUDA.2' to select the 3rd CUDA device - note that in
        this case, the rank always corresponds to the order of the device, and not the fft speed. Last case, the first
        part can match the gpu name, and the second part can be a rank (if using a device with multiple identical GPU).

        If MPI is used and multiple devices are available, the selection is made according to the MPI rank,
        after the gpu name filtering is taken into account.

        This will call set_device() with the selected device.

        :param gpu_name: a (sub)string matching the name of the gpu to be used. This can also be a list of
                         acceptable gpu names.
        :param gpu_rank: the rank of the GPU to use. If ranking is 'fft_speed', this will select from the fastest
                         GPU. If ranking is 'order', it will just find the GPU matching the name, in the order
                         they are found.
        :param ranking: can be 'fft_speed' (or 'fft'), all GPU are tested for FFT speed, and chosen from that
                        ranking. If ranking is 'order', they will be listed as they are found by CUDA/OpenCL.
                        If 'bandwidth' (default), the on-device memory transfer speed is used as benchmark.
                        This is useful to select a GPU just based on its name, without the overhead of the FFT test.
                        Ignored if using MPI (the ranking will use the GPU's PCI id (cuda) or pointer (opencl),
                        distributed by MPI order to the process on the same node. Note that it will be
                        cleaner if each MPI process only sees a single GPU).
        :param language: either 'opencl', 'cuda', 'cpu', or None. If None, the preferred language is cuda>opencl>cpu
        :param verbose: True or False (default) for verbosity
        :return: True if a GPU (or CPU) could be selected. An exception is returned if no GPU was found,
                 unless language was 'cpu'.
        :raise Exception: if no GPU could be selected, and the language was not 'cpu'. The message will indicate
                          if a gpu_name/language/gpu_rank was specified, but could not be selected.
        """

        benchmark_results = []

        if 'PYNX_PU' in os.environ:
            pynx_py_env = os.environ['PYNX_PU'].lower().split('.')
            if pynx_py_env[0] not in ['opencl', 'cuda', 'cpu']:
                if len(pynx_py_env[0]) > 0:
                    gpu_name = pynx_py_env[0]

            if len(pynx_py_env) == 2:
                if len(pynx_py_env[1]) > 0:
                    s = pynx_py_env[1]
                    try:
                        gpu_rank = int(s)
                        if gpu_rank < 100:
                            ranking = 'order'
                            if verbose:
                                print("PyNX processing unit: use #%d device from "
                                      "PYNX_PU environment variable" % gpu_rank)
                        else:
                            # that's for the case where the card number (e.g. 1080) is given...
                            gpu_name = s
                            if verbose:
                                print("PyNX processing unit: searching '%s' GPU from PYNX_PU environment variable" % s)

                    except ValueError:
                        gpu_name = s
                        if verbose:
                            print("PyNX processing unit: searching '%s' GPU from PYNX_PU environment variable" % s)

        if mpi is not None:
            ranking = 'mpi'

        if ranking not in ['fft_speed', 'fft']:
            # This will deactivate FFT speed, just reporting found GPU devices
            fft_shape = None
        else:
            ranking = 'fft'
            fft_shape = self.benchmark_fft_size

        test_cuda = False
        if language is None:
            test_cuda = True
        elif 'cuda' in language.lower():
            test_cuda = True

        if test_cuda:
            # Test first CUDA - if selection is by name, CUDA devices will be reported before OpenCL
            if has_cuda:
                benchmark_results += cuda_device.available_gpu_speed(fft_shape=fft_shape,
                                                                     batch=self.benchmark_fft_batch, min_gpu_mem=None,
                                                                     verbose=verbose, gpu_name=gpu_name,
                                                                     ranking=ranking)
            if language is not None:
                if 'cuda' in language.lower():
                    if len(benchmark_results) == 0:
                        s = 'Desired GPU language is CUDA, but no device found !'
                        if gpu_name is not None:
                            s += ' [gpu_name=%s]' % str(gpu_name)
                        raise Exception(s)

        if len(benchmark_results):
            # We have found at least 1 CUDA device, so use it
            if mpi is not None:
                nb = len(benchmark_results)
                r = mpi.Get_rank()
                if nb > 1:
                    # TODO: evaluate how reliable this mpi hook is to select GPUs
                    # If we have more than 1 CUDA GPU available, use MPI rank to select it
                    # We cannot assume that the rank is sequential on each node, so we need
                    # to determine a sequential rank on each node.
                    # This assumes that all MPI tasks on each node see the same devices.
                    nodes = mpi.gather(platform.node(), root=0)
                    nodes_ct = {}
                    local_ranks = []
                    if mpi.Get_rank() == 0:
                        for n in nodes:
                            if n in nodes_ct:
                                nodes_ct[n] += 1
                            else:
                                nodes_ct[n] = 0
                            local_ranks.append(nodes_ct[n])
                    r = mpi.scatter(local_ranks, root=0)
                    # Before assigning the GPU, sort the found devices by their PCI id, because the device
                    # order can be different on two process on the same node (!)
                    benchmark_results = [(r[0], r[1], r[0].pci_bus_id()) for r in benchmark_results]
                    benchmark_results = list(sorted(benchmark_results, key=lambda t: t[2]))
                    if verbose:
                        print("select_gpu using MPI: node=%s mpi_rank=%d, using GPU #%d/%d PCI:" %
                              (platform.node(), mpi.Get_rank(), r % nb, nb), benchmark_results[r % nb][0].pci_bus_id())
                return self.set_device(benchmark_results[r % nb][0], test_fft=False, verbose=verbose)
            else:
                return self.set_device(benchmark_results[gpu_rank][0], test_fft=False, verbose=verbose)

        test_opencl = False
        if language is None:
            test_opencl = True
        elif 'opencl' in language.lower():
            test_opencl = True
        if test_opencl:
            if has_opencl:
                if self.benchmark_fft_batch:
                    fft_axis = (-2, -1)
                else:
                    fft_axis = None
                benchmark_results += opencl_device.available_gpu_speed(fft_shape=fft_shape, axes=fft_axis,
                                                                       min_gpu_mem=None, verbose=verbose,
                                                                       gpu_name=gpu_name, ranking=ranking)
            if language is not None:
                if 'opencl' in language.lower():
                    if len(benchmark_results) == 0:
                        s = 'Desired GPU language is OpenCL, but no device found !'
                        if gpu_name is not None:
                            s += ' [gpu_name=%s]' % str(gpu_name)
                        raise Exception(s)

        if language is not None:
            if 'cpu' in language.lower():
                return self.set_device(test_fft=False, verbose=verbose)

        if gpu_name is not None:
            if 'cpu' in str(gpu_name).lower():
                return self.set_device(test_fft=False, verbose=verbose)

        if len(benchmark_results):
            if mpi is not None:
                # if we have more than 1 GPU available, use MPI rank to select it
                # This requires that a language be selected, or each device will be listed twice for cuda+opencl
                nb = len(benchmark_results)
                r = mpi.Get_rank()
                if nb > 1:
                    # TODO: evaluate how reliable this mpi hook is to select GPUs
                    # If we have more than 1 CUDA GPU available, use MPI rank to select it
                    # We cannot assume that the rank is sequential on each node, so we need
                    # to determine a sequential rank on each node.
                    # This assumes that all MPI tasks on each node see the same devices.
                    nodes = mpi.gather(platform.node(), root=0)
                    nodes_ct = {}
                    local_ranks = []
                    if mpi.Get_rank() == 0:
                        for n in nodes:
                            if n in nodes_ct:
                                nodes_ct[n] += 1
                            else:
                                nodes_ct[n] = 0
                            local_ranks.append(nodes_ct[n])
                    r = mpi.scatter(local_ranks, root=0)
                    # Before assigning the GPU, sort the found devices by their address, because the device
                    # order can be different on two process on the same node (is that also true for opencl?)
                    benchmark_results = [(r[0], r[1], r[0].int_ptr) for r in benchmark_results]
                    benchmark_results = list(sorted(benchmark_results, key=lambda t: t[2]))
                    if verbose:
                        print("select_gpu using MPI: node=%s mpi_rank=%d, using GPU #%d/%d ptr:" %
                              (platform.node(), mpi.Get_rank(), r % nb, nb), benchmark_results[r % nb][0].int_ptr)
                return self.set_device(benchmark_results[gpu_rank][r % nb], test_fft=False, verbose=verbose)

            if ranking in ['fft', 'bandwidth']:
                benchmark_results = sorted(benchmark_results, key=lambda t: -t[1])
            if len(benchmark_results) <= gpu_rank:
                if gpu_name is not None:
                    s = ' [gpu_name=%s]' % str(gpu_name)
                else:
                    s = ''
                raise Exception('Desired GPU%s rank is %d, but only %d GPU found !' % (s, len(benchmark_results),
                                                                                       gpu_rank))
            return self.set_device(benchmark_results[gpu_rank][0], test_fft=False, verbose=verbose)

        s = "Could not find a suitable GPU. Please check GPU name or CUDA/OpenCL installation"

        if gpu_name is not None:
            s += " [name=%s]" % str(gpu_name)
        raise Exception(s)

    @classmethod
    def get_context(cls, device):
        """
        Static method to get a context, using the static device context dictionary to avoid creating new contexts,
        which will use up the GPU memory.
        This abstract function is implemented in CUProcessingUnit and CLProcessingUnit classes, the base version should
        never be called.
        :param device: the pyCUDA or pyOpenCL device for which a context is desired
        :return: the OpenCL or
        """
        raise Exception('ProcessingUnit.get_context() is an abstract function, which should not be called. '
                        'You probably want to use a CLProcessingUnit or CUProcessingUnit.')

    def finish(self):
        """
        Wait till all processing unit calculations are finished

        Virtual method, should be derived for CUDA or OpenCL
        :return: Nothing
        """
        pass

    def synchronize(self):
        """
        Wait till all processing unit calculations are finished

        Virtual method, should be derived for CUDA or OpenCL
        :return: Nothing
        """
        warnings.warn("ProcessingUnit.synchronize() is deprecated. Use finish() instead.", DeprecationWarning)
        return self.finish()

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
        if 'nproc' in kwargs:
            nproc = kwargs['nproc']
        else:
            nproc = self.nproc
        if ndim is None:
            ax = tuple(list(range(src.ndim)))
        else:
            ax = tuple(list(range(src.ndim))[-ndim:])
        if np.iscomplexobj(src) and np.iscomplexobj(dest):
            assert src.shape == dest.shape
            if norm:
                dest[:] = fftn(src, axes=ax, workers=nproc, norm="ortho")
            else:
                dest[:] = fftn(src, axes=ax, workers=nproc)
        elif np.isrealobj(src) and np.iscomplexobj(dest):
            assert src.shape[:-1] == dest.shape[:-1]
            assert src.shape[-1] // 2 + 1 == dest.shape[-1]
            if norm:
                dest[:] = rfftn(src, axes=ax, workers=nproc, norm="ortho")
            else:
                dest[:] = rfftn(src, axes=ax, workers=nproc)
        else:
            raise ProcessingUnitException("ProcessingUnit.fft(): must be a C2C or R2C transform")
        if return_scale:
            if norm:
                return 1
            else:
                return self.fft_scale(src, ndim=ndim)[0]

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
        if 'nproc' in kwargs:
            nproc = kwargs['nproc']
        else:
            nproc = self.nproc
        if ndim is None:
            ax = tuple(list(range(src.ndim)))
        else:
            ax = tuple(list(range(src.ndim))[-ndim:])
        if np.iscomplexobj(src) and np.iscomplexobj(dest):
            assert src.shape == dest.shape
            if norm:
                dest[:] = ifftn(src, axes=ax, workers=nproc, norm="ortho")
            else:
                dest[:] = ifftn(src, axes=ax, workers=nproc)
        elif np.iscomplexobj(src) and np.isrealobj(dest):
            assert src.shape[:-1] == dest.shape[:-1]
            assert dest.shape[-1] // 2 + 1 == src.shape[-1]
            if norm:
                dest[:] = irfftn(src, axes=ax, workers=nproc, norm="ortho")
            else:
                dest[:] = irfftn(src, axes=ax, workers=nproc)
        else:
            raise ProcessingUnitException("ProcessingUnit.ifft(): must be a C2C or C2R transform")
        if return_scale:
            if norm:
                return 1
            else:
                return self.fft_scale(dest, ndim=ndim)[1]

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
        return 1 / s, s

    def free_fft_plans(self):
        """
        If memory was allocated for FFT plans, free it if possible
        :return: nothing
        """
        pass

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
        a = np.zeros(shape, dtype=dtype)
        a.fill(2.0)
        if dtype in [np.complex64, np.complex128]:
            if inplace:
                b = np.empty_like(a)
            else:
                b = a
        self.fft(a, b, ndim=ndim, norm=False)
        self.finish()
        # print(100 * 1024 ** 3 / (ndim * 2 * 2 * a.nbytes))
        if nb is None:
            # Target about 0.1s test, minimum 10GB/s
            nb = 0.1 * 1 * 1024 ** 3 / (ndim * 2 * 2 * a.nbytes)
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

# The default processing unit
default_processing_unit = ProcessingUnit()
