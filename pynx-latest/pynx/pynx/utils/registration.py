# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import timeit
import numpy as np
from scipy.ndimage import fourier_shift

from scipy import log10, sqrt, pi, exp
from scipy.fftpack import ifftshift, fftshift, fft2, ifft2, fftfreq, fftn, ifftn

from pynx import scattering
from ..processing_unit import has_cuda, has_opencl

if has_cuda:
    from ..processing_unit.kernel_source import get_kernel_source as getks
    from ..processing_unit.cu_processing_unit import CUProcessingUnit
    import pycuda.gpuarray as cua
    from pycuda.reduction import ReductionKernel as CU_RedK
    from pycuda.elementwise import ElementwiseKernel as CU_Elk
    import pycuda.tools as cu_tools
    from pycuda.compiler import SourceModule
    import skcuda.fft as cu_fft

    argmax_dtype = np.dtype([("idx", np.int32), ("cur_max", np.float32)])
    cu_tools.get_or_register_dtype("idx_max", argmax_dtype)


# if has_opencl:  # TODO: OpenCL registration


def shift(img, v):
    """
    Shift an image 2D or 3D, if necessary using a subpixel (FFT-based) approach. Values are wrapped around the array borders.

    Alternative methods:
        for real data: see scipy.ndimage.interpolation.shift (uses spline interpolation, complex not supported)
        for complex data: scipy.ndimage.fourier_shift
    
    Args:
        img: the image (2D or 3D) to be shifted, which will be treated as complex-valued data.
        v: the shift values (2 or 3 values according to array dimensions)

    Returns:
        the shifted array
    """
    assert (img.ndim == len(v))
    if all([(x % 1 == 0) for x in v]):
        # integer shifting
        simg = np.roll(img, v[-1], axis=-1)
        for i in range(img.ndim - 1):
            simg = np.roll(simg, v[i], axis=i)
        return simg
    else:
        if img.dtype == np.float64 or img.dtype == np.complex128:
            dtype = np.complex128
        else:
            dtype = np.complex64
        return ifftn(fourier_shift(fftn(img.astype(dtype)), v))


def register_translation_old(ref_img, img, upsampling=1, verbose=False, gpu='CPU'):
    """
    Calculate the translation shift between two images.

    Also see in scikit-image: skimage.feature.register_translation, for a more complete implementation
    
    Args:
        ref_img: the reference image
        img: the image to be translated to match ref_img
        upsampling: integer value - the pixel resolution of the computed shift will be equal to 1/upsampling

    Returns:
        the shift value along each axis of the image
    """
    assert (ref_img.shape == img.shape)
    assert (img.ndim == 2)  # TODO: 3D images
    t00 = timeit.default_timer()
    ny, nx = img.shape
    ref_img_f = fftn(np.array(ref_img, dtype=np.complex64, copy=False))
    img_f = fftn(np.array(img, dtype=np.complex64, copy=False))

    cc_f = img_f * ref_img_f.conj()

    # maxshift = 8 #try to limit the range to search for the peak (only activated useful with a _very_ small range => deactivated)

    # if maxshift is None:
    #    firstfft = True
    # elif maxshift < np.sqrt(img.size)/4:
    #    firstfft = False
    # else:
    #    firstfft = True
    if True:
        # integer pixel registration
        nflop = 5 * nx * ny * (np.log2(nx) + np.log2(ny))
        t0 = timeit.default_timer()
        icc = ifftn(cc_f)

        theshift = np.array(np.unravel_index(np.argmax(abs(icc)), cc_f.shape))
        for i in range(len(theshift)):
            theshift[i] -= (theshift[i] > cc_f.shape[i] // 2) * cc_f.shape[i]
        dt = timeit.default_timer() - t0
        if verbose:
            print("integer shift using FFT (%5.2fs, nGflop=%6.3f)" % (dt, nflop / 1e9), theshift)
    # else:
    #    # integer pixel registration, assuming the shift is smaller than maxshift, using a DFT
    #    w = 2 * maxshift
    #    nflop = 2 * 8 * nx * ny * w ** 2
    #    t0 = timeit.default_timer()
    #    z, y, x= np.meshgrid(0, fftshift(np.arange(-.5, .5, 1 / ny)), fftshift(np.arange(-.5, .5, 1 / nx)), indexing='ij')
    #    hk = np.arange(-w / 2, w / 2, 1 / w)
    #    l, k, h= np.meshgrid(0, hk, hk, indexing='ij')

    #    cc = scattering.Fhkl_thread(h,k,l,x,y,z,occ=cc_f.real,gpu_name='Iris')[0] + 1j*scattering.Fhkl_thread(h,k,l,x,y,z,occ=cc_f.imag,
    #                                                                                                        gpu_name='Iris')[0]
    #    theshift = np.array(np.unravel_index(np.argmax(abs(cc)), cc.shape[-2:]))
    #    theshift = [k[0,int(np.round(theshift[0])),0],h[0,0,int(np.round(theshift[1]))]]
    #    dt = timeit.default_timer() - t0
    #    print("theshift using DFT (%5.2fs, nGflop=%6.3f)"%(dt, nflop/1e9), theshift)

    if gpu.lower() != 'CPU':
        language = 'opencl'
    else:
        language = 'cpu'
    if upsampling > 1:
        # subpixel registration
        k1 = np.sqrt(upsampling)
        # for uw, us in [(1.5, 1 / upsampling)]: # one-step optimization
        for uw, us in [(1.5, 1.5 / k1), (3 / k1, 1 / upsampling)]:  # two-step optimization à la Guizar-Sicairos
            # uw: width of upsampled region
            # us: step size in upsampled region
            t0 = timeit.default_timer()

            z, y, x = np.meshgrid(0, fftshift(np.arange(-.5, .5, 1 / ny)), fftshift(np.arange(-.5, .5, 1 / nx)),
                                  indexing='ij')
            h = np.arange(-uw / 2, uw / 2, us) + theshift[-1]
            k = np.arange(-uw / 2, uw / 2, us) + theshift[-2]
            l, k, h = np.meshgrid(0, k, h, indexing='ij')

            cc = scattering.Fhkl_thread(h, k, l, x, y, z, occ=cc_f.real, gpu_name=gpu, language=language)[0] \
                 + 1j * scattering.Fhkl_thread(h, k, l, x, y, z, occ=cc_f.imag, gpu_name=gpu, language=language)[0]
            theshift = np.array(np.unravel_index(np.argmax(abs(cc)), cc.shape[-2:]))
            theshift = [k[0, int(np.round(theshift[0])), 0], h[0, 0, int(np.round(theshift[1]))]]
            dt = timeit.default_timer() - t0
            if verbose:
                print("subpixel shift using DFT (%5.2fs)" % (dt), theshift)
    if verbose:
        print("Final shift (%5.2fs)" % (timeit.default_timer() - t00), theshift)
    return theshift


def register_translation_cuda(ref_img, img, upsampling=10, processing_unit=None, overwrite=False, blocksize=None):
    """
    CUDA image registration, with or without subpixel precision. Sub-pixel
    implementation is currently very slow (slower than scikit-image).
    :param ref_img, img: the images to be registered, either as numpy array or as 
        a pycuda.gpuarray.GPUArray.
    :param upsampling: the upsampling factor (integer >=1), for subpixel registration
    :param processing_unit: the processing unit to be used for the calculation.
        Should already be initialised. If None, the default one will be used.
    :param overwrite: if True and the input images are pycuda GPUArrays, they will be
        overwritten.
    :param blocksize: the CUDA blocksize for the subpixel registration.
        If None, will be automatically set to 32. Larger values
        do not seem to bring improvements. Must be smaller than upsampling**2
    :return: the computed shift as a tuple (dy, dx)
    """
    pu = processing_unit
    if pu is None:
        pu = CUProcessingUnit()
        pu.init_cuda(test_fft=False, verbose=False)

    if blocksize is None:
        blocksize = 32
    if blocksize > upsampling ** 2:
        blocksize = upsampling ** 2

    if not isinstance(ref_img, cua.GPUArray):
        ref_img = cua.to_gpu(ref_img.astype(np.complex64))
    else:
        ref_img = ref_img.astype(np.complex64)

    if not isinstance(img, cua.GPUArray):
        img = cua.to_gpu(img.astype(np.complex64))
    else:
        img = img.astype(np.complex64)

    if register_translation_cuda.cu_argmax_c_red is None:
        register_translation_cuda.cu_argmax_c_red = \
            CU_RedK(argmax_dtype, neutral="idx_max(0,0.0f)", name='argmax_c',
                    reduce_expr="argmax_reduce(a,b)",
                    map_expr="idx_max(i, abs(d[i]))",
                    preamble=getks("cuda/argmax.cu"),
                    options=["-use_fast_math"],
                    arguments="pycuda::complex<float> *d")

    if register_translation_cuda.cu_argmax_f_red is None:
        register_translation_cuda.cu_argmax_f_red = \
            CU_RedK(argmax_dtype, neutral="idx_max(0,0.0f)", name='argmax_f',
                    reduce_expr="argmax_reduce(a,b)",
                    map_expr="idx_max(i, d[i])",
                    preamble=getks("cuda/argmax.cu"),
                    options=["-use_fast_math"],
                    arguments="float *d")

    fft_plan = pu.cu_fft_get_plan(ref_img, ndim=2)
    if overwrite is False:
        d1f, d2f = cua.empty_like(ref_img), cua.empty_like(ref_img)
        # out-of-place FFT of the two images
        cu_fft.fft(ref_img, d1f, fft_plan, scale=True)
        cu_fft.fft(img, d2f, fft_plan, scale=True)
    else:
        d1f, d2f = ref_img, img
        cu_fft.fft(d1f, d1f, fft_plan, scale=True)
        cu_fft.fft(d2f, d2f, fft_plan, scale=True)

    # Pixel registration
    cc0 = d1f * d2f.conj()
    cu_fft.ifft(cc0, d1f, fft_plan, scale=False)
    idx = register_translation_cuda.cu_argmax_c_red(d1f).get()["idx"]

    ny, nx = d1f.shape
    shift0 = [idx // nx, idx % nx]
    shift0[0] -= (shift0[0] > ny / 2) * ny
    shift0[1] -= (shift0[1] > nx / 2) * nx
    if upsampling == 1:
        return shift0

    # Sub-pixel registration
    if blocksize not in register_translation_cuda.cu_cc_zoom:
        reg_mod = SourceModule(getks("cuda/argmax.cu") +
                               getks("utils/cuda/registration.cu") % {"blocksize": blocksize},
                               options=["-use_fast_math"])
        register_translation_cuda.cu_cc_zoom[blocksize] = reg_mod.get_function("cc_zoom")
    cu_cc_zoom = register_translation_cuda.cu_cc_zoom[blocksize]

    upsample_range = 1.5

    nxy1 = np.int32(upsampling * upsample_range)
    y0 = np.float32(shift0[0] - (nxy1 // 2) / upsampling)
    x0 = np.float32(shift0[1] - (nxy1 // 2) / upsampling)
    z0 = np.float32(0)
    dxy = np.float32(1 / upsampling)
    nyu, nxu = np.int32(d1f.shape[-2]), np.int32(d1f.shape[-1])
    nzu = np.int32(1)
    cc1_cu = cua.empty((nxy1, nxy1), dtype=np.float32)

    cu_cc_zoom(cc1_cu, cc0, x0, y0, z0, dxy, dxy, dxy, nxy1, nxy1, nxu, nyu, nzu,
               block=(int(blocksize), 1, 1), grid=(int(nxy1), int(nxy1), 1))
    idx1 = register_translation_cuda.cu_argmax_f_red(cc1_cu).get()["idx"]

    ix1 = idx1 % nxy1
    iy1 = idx1 // nxy1
    x1 = x0 + dxy * ix1
    y1 = y0 + dxy * iy1

    return y1, x1


def register_translation_cuda_n(ref_img, img, upsampling=10, processing_unit=None, overwrite=False,
                                blocksize=None):
    """
    CUDA image registration, with or without subpixel precision. Sub-pixel
    implementation is currently very slow (slower than scikit-image).
    This version allows to register a stack of N*M images against a stack of N reference images.
    Performance improvement compared with looping individual registration is
    mostly obtained for subpixel accuracy, with ideally N*M a multiple of the
    number of GPU multi-processors.

    :param ref_img, img: the stack of images to be registered, either as numpy arrays or as
        a pycuda.gpuarray.GPUArray.
    :param upsampling: the upsampling factor (integer >=1), for subpixel registration
    :param processing_unit: the processing unit to be used for the calculation.
        Should already be initialised. If None, the default one will be used.
    :param overwrite: if True and the input images are pycuda GPUArrays, they will be
        overwritten.
    :param blocksize: the CUDA blocksize for the subpixel registration.
        If None, will be automatically set to min(upsampling**2, 64). Larger values
        do not seem to bring improvements.
        Check that register_translation_cuda_n.cu_cc_zoomN[blocksize].shared_size_bytes
        does not exceed 48kb.
    :return: the computed shift as a tuple of arrays (dy, dx)
    """
    pu = processing_unit
    if pu is None:
        pu = CUProcessingUnit()
        pu.init_cuda(test_fft=False, verbose=False)

    if blocksize is None:
        blocksize = min(upsampling ** 2, 512)

    if not isinstance(ref_img, cua.GPUArray):
        ref_img = cua.to_gpu(ref_img.astype(np.complex64))
    else:
        ref_img = ref_img.astype(np.complex64)

    if not isinstance(img, cua.GPUArray):
        img = cua.to_gpu(img.astype(np.complex64))
    else:
        img = img.astype(np.complex64)

    if img.ndim == 2:
        img = img.reshape((1, img.shape[0], img.shape[1]))
    if ref_img.ndim == 2:
        ref_img = ref_img.reshape((1, ref_img.shape[0], ref_img.shape[1]))

    if register_translation_cuda.cu_argmax_c_red is None:
        register_translation_cuda.cu_argmax_c_red = \
            CU_RedK(argmax_dtype, neutral="idx_max(0,0.0f)", name='argmax_c',
                    reduce_expr="argmax_reduce(a,b)",
                    map_expr="idx_max(i, abs(d[i]))",
                    preamble=getks("cuda/argmax.cu"),
                    options=["-use_fast_math"],
                    arguments="pycuda::complex<float> *d")

    fft_plan1 = pu.cu_fft_get_plan(ref_img, ndim=2)
    fft_plan2 = pu.cu_fft_get_plan(img, ndim=2)
    fft_plan0 = pu.cu_fft_get_plan(img[0], ndim=2)
    if overwrite is False:
        d1f, d2f = cua.empty_like(ref_img), cua.empty_like(img)
        # out-of-place FFT of the two images
        cu_fft.fft(ref_img, d1f, fft_plan1, scale=True)
        cu_fft.fft(img, d2f, fft_plan2, scale=True)
    else:
        d1f, d2f = ref_img, img
        cu_fft.fft(d1f, d1f, fft_plan1, scale=True)
        cu_fft.fft(d2f, d2f, fft_plan2, scale=True)

    # Pixel registration
    nzref, ny, nx = ref_img.shape
    nz, ny, nx = img.shape
    dy, dx = np.empty(nz, dtype=np.float32), np.empty(nz, dtype=np.float32)
    cutmp = cua.empty_like(d2f[0])
    # This may go faster by treating all images simultaneously ?
    for i in range(nz):
        i1 = i * nzref // nz
        d2f[i] = d1f[i1] * d2f[i].conj()
        cu_fft.ifft(d2f[i], cutmp, fft_plan0, scale=False)
        idx = register_translation_cuda.cu_argmax_c_red(cutmp).get()["idx"]
        dy0, dx0 = idx // nx, idx % nx
        dy[i] = dy0 - (dy0 > ny / 2) * ny
        dx[i] = dx0 - (dx0 > nx / 2) * nx
    if upsampling == 1:
        return dy, dx

    # Sub-pixel registration
    if blocksize not in register_translation_cuda_n.cu_cc_zoomN:
        reg_mod = SourceModule(getks("cuda/argmax.cu") +
                               getks("utils/cuda/registration.cu") % {"blocksize": blocksize},
                               options=["-use_fast_math"])
        register_translation_cuda_n.cu_cc_zoomN[blocksize] = reg_mod.get_function("cc_zoomN")
    cu_cc_zoom_n = register_translation_cuda_n.cu_cc_zoomN[blocksize]

    upsample_range = 1.5

    nxy1 = np.int32(upsampling * upsample_range)
    y0 = np.float32(dy - (nxy1 // 2) / upsampling)
    x0 = np.float32(dx - (nxy1 // 2) / upsampling)
    dxy = np.float32(1 / upsampling)
    nyu, nxu = np.int32(ny), np.int32(nx)
    x0_cu = cua.to_gpu(x0.astype(np.float32))
    y0_cu = cua.to_gpu(y0.astype(np.float32))
    cu_cc_zoom_n(d2f, x0_cu, y0_cu, dxy, dxy, nxy1, nxy1, nxu, nyu,
                 block=(int(blocksize), 1, 1), grid=(nz, 1, 1))

    return y0_cu.get(), x0_cu.get()


register_translation_cuda.cu_argmax_c_red = None
register_translation_cuda.cu_argmax_f_red = None
register_translation_cuda.cu_cc_zoom = {}
register_translation_cuda_n.cu_cc_zoomN = {}
