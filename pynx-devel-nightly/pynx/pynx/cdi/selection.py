# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
"""
This module includes functions to:
- compare and match solutions (objects) from CDI optimisations
- provide figures-of-merit for optimised objects
- combine and solutions
"""

import timeit
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import fourier_shift
from scipy.ndimage.measurements import center_of_mass
from scipy.fftpack import fftn, ifftn, fftshift

try:
    from skimage.registration import phase_cross_correlation as register_translation
except ImportError:
    from skimage.feature import register_translation
from ..processing_unit import has_cuda, has_opencl
from ..processing_unit import default_processing_unit as main_default_processing_unit
import warnings

# if has_cuda and main_default_processing_unit.pu_language not in ['opencl', 'cpu']:
#     import pycuda.driver as cu_drv
#     import pycuda.gpuarray as cua
#     from pycuda.elementwise import ElementwiseKernel as CU_ElK
#     from pycuda.reduction import ReductionKernel as CU_RedK

if has_cuda and main_default_processing_unit.pu_language not in ['opencl', 'cpu']:
    import pycuda.gpuarray as cua
    from pycuda.reduction import ReductionKernel as CU_RedK
    from ..processing_unit.cu_processing_unit import CUProcessingUnit
    from ..processing_unit.kernel_source import get_kernel_source as getks
elif has_opencl and main_default_processing_unit.pu_language not in ['cuda', 'cpu']:
    import pyopencl.array as cla
    from pyopencl.reduction import ReductionKernel as CL_RedK
    from pyopencl import CompilerWarning

    warnings.simplefilter('ignore', CompilerWarning)
    from ..processing_unit.cl_processing_unit import CLProcessingUnit
    from ..processing_unit.kernel_source import get_kernel_source as getks


def array_cen(a: np.ndarray, thres=None, decomposed=False):
    """
    Move the centre of mass of the array to its centre.
    :param a: the array to be centered
    :param thres: if a fraction (between 0 and 1) is given, the center of mass is
        calculated on an array whare all values are capped to abs(a).max*thres,
        so that a few intense pixels do not have an exaggerated weight.
    :param decomposed: if True, the centre of mass and thresholding is done by projecting
        independently against each axis. This is better at avoiding that an object with
        an asymmetric shape (e.g. a triangle) gets rolled out on the sides of the array.
        Ignored if thres is None.
    :return: the centred array
    """
    if thres is None:
        c = center_of_mass(abs(a))
    else:
        if decomposed:
            c = []
            for i in range(a.ndim):
                ax = list(range(a.ndim))
                ax.pop(i)
                aa = abs(a).sum(axis=tuple(ax))
                m = aa.max() * thres
                aa[aa > m] = m
                c.append(center_of_mass(aa)[0])
        else:
            aa = abs(a)
            m = aa.max() * thres
            aa[aa > m] = m
            c = center_of_mass(aa)
    dr = np.array(a.shape).astype(np.float32) / 2 - np.array(c)
    dr = np.round(dr).astype(np.int32)
    if np.allclose(dr, 0):
        return a
    return np.roll(a, dr, range(a.ndim))


def match_shape(arrays, method='min', shape=None, cen=True):
    """
    Match the shape of two or more arrays by cropping the borders, or zero-padding if necessary
    :param arrays: a list or tuple of arrays, all either 2D or 3D
    :param method: either 'min' (use the largest size along each dimension which is smaller than all array sizes)
        or 'max' (largest size along each dimension among all the arrays)
        or 'median': use the median value for the size along each dimension. The 'median' option is better when
        matching more than 2 arrays, when one may be an outlier with incorrect dimensions.
    :param shape: if given, all arrays shapes will be set to the given tuple, and 'method' is ignored.
    :param cen: if True, the arrays will be centred (using a 0.1 threshold and decomposition) before matching
    :return: a list of the arrays cropped or zero-padded to the same shape. The data type of each individual array
        is preserved
    """
    d1 = arrays[0]
    v = []
    ndim = d1.ndim
    if shape is not None:
        if len(shape) == 3:
            nz, ny, nx = shape
        elif len(shape) == 2:
            ny, nx = shape
    else:
        if method == 'median':
            nx = int(np.median(list(d.shape[-1] for d in arrays)))
            ny = int(np.median(list(d.shape[-2] for d in arrays)))
            if ndim == 3:
                nz = int(np.median(list(d.shape[-3] for d in arrays)))
        elif method == 'max':
            nx = max((d.shape[-1] for d in arrays))
            ny = max((d.shape[-2] for d in arrays))
            if ndim == 3:
                nz = max((d.shape[-3] for d in arrays))
        else:
            nx = min((d.shape[-1] for d in arrays))
            ny = min((d.shape[-2] for d in arrays))
            if ndim == 3:
                nz = min((d.shape[-3] for d in arrays))

    for d in arrays:
        if cen:
            d = array_cen(d, thres=0.1, decomposed=True)
        if ndim == 3:
            nz1, ny1, nx1 = d.shape
            tmp = np.zeros((nz, ny, nx), dtype=d.dtype)
        else:
            ny1, nx1 = d.shape
            tmp = np.zeros((ny, nx), dtype=d.dtype)

        n, n1 = nx, nx1
        if n <= n1:
            d = d[..., n1 // 2 - n // 2:n1 // 2 - n // 2 + n]

        n, n1 = ny, ny1
        if n <= n1:
            d = d[..., n1 // 2 - n // 2:n1 // 2 - n // 2 + n, :]

        if ndim >= 3:
            n, n1 = nz, nz1
            if n <= n1:
                d = d[..., n1 // 2 - n // 2:n1 // 2 - n // 2 + n, :, :]

        if ndim == 3:
            nz1, ny1, nx1 = d.shape
            tmp[nz // 2 - nz1 // 2:nz // 2 - nz1 // 2 + nz1, ny // 2 - ny1 // 2:ny // 2 - ny1 // 2 + ny1,
            nx // 2 - nx1 // 2:nx // 2 - nx1 // 2 + nx1] = d
        else:
            ny1, nx1 = d.shape
            tmp[ny // 2 - ny1 // 2:ny // 2 - ny1 // 2 + ny1, nx // 2 - nx1 // 2:nx // 2 - nx1 // 2 + nx1] = d

        v.append(tmp)

    return v


def flipn(d: np.ndarray, flip_axes):
    """
    Flip an array along any combination of axes
    :param d: the array to manipulate
    :param flip_axes: an iterable list of d.ndim values with True/False values indicating if the array must be flipped
                      along each axis. If flip_axes=None, flip is dona along all axes.
                      An extra value can be added to optionally return the conjugate of the array.
    :return: a copy of the array after modification
    """
    if flip_axes is None:
        return np.flip(d)
    for i in range(d.ndim):
        if flip_axes[i]:
            d = np.flip(d, i)
    if len(flip_axes) > d.ndim:
        if flip_axes[d.ndim]:
            return d.conj()
    return d


def corr_phase(pars, d: np.ndarray):
    """
    Apply a linear phase shift to a complex array, either 2D or 3D
    :param pars: the shift parameters for the phase. The correction applied is a multiplication by exp(-1j * dphi),
        with dphi = pars[0] + pars[1] * x0 + pars[2] * x1 {+ pars[3] * x2}, where x0, x1, x2 are coordinates array
        covering [0 ; 1[ along each axis.
    :param d: the array for which the phase will be shifted
    :return: a copy of the array after phase-shifting
    """
    if d.ndim == 3:
        nz, ny, nx = d.shape
        iz, iy, ix = np.meshgrid(np.arange(nz) / nz, np.arange(ny) / ny, np.arange(nx) / nx, indexing='ij')
        return (d * np.exp(-1j * (pars[0] + pars[1] * iz + pars[2] * iy + pars[3] * ix))).astype(d.dtype)
    else:
        ny, nx = d.shape
        iy, ix = np.meshgrid(np.arange(ny) / ny, np.arange(nx) / nx, indexing='ij')
        return (d * np.exp(-1j * (pars[0] + pars[1] * iy + pars[2] * ix))).astype(d.dtype)


def fit_phase2(pars, phi1: np.ndarray, phi2: np.ndarray, a1: np.ndarray, a2: np.ndarray, xyz: tuple):
    """
    Fit function to match the phase between two arrays, with linear phase shift parameters.
    :param pars: a list of ndim+1 linear phase shift parameters (one constant plus one along each axis)
    :param phi1: the phase (angle) of the first array, assumed to be within [-pi, pi]
    :param phi2: the phase (angle) of the second array, assumed to be within [-pi, pi]
    :param a1: the amplitude (or weight) of the first array. Should be >=0
    :param a2:  the amplitude (or weight) of the second array. Should be >=0
    :param xyz: a tuple with of ndim arrays with the same shape as the arrays, giving a set of [0;1[ coordinates along
        each axis.
    :return: a floating point figure of merit, ((a1 + a2) * delta_phi ** 2).sum()
    """
    dphi = phi2 - pars[0]
    for i in range(len(xyz)):
        dphi -= pars[i + 1] * xyz[i]
    dphi = np.abs(phi1 - dphi)
    dphi = np.minimum(dphi, 2 * np.pi - dphi)
    return (a1 * a2 * dphi ** 2).sum()


def fit_phase2_cl(pars, a1, a2, cl_func, cl_queue):
    """
    Fit function to match the phase between two arrays, with linear phase shift parameters. OpenCL version
    :param pars: a list of ndim+1 linear phase shift parameters (one constant plus one along each axis)
    :param a1: the amplitude (or weight) of the first array. Should be >=0
    :param a2:  the amplitude (or weight) of the second array. Should be >=0
    :return: a floating point figure of merit, ((a1 + a2) * delta_phi ** 2).sum()
    """
    nx = np.int32(a1.shape[-1])
    ny = np.int32(a1.shape[-2])
    if a1.ndim == 3:
        nz = np.int32(a1.shape[0])
        r = cl_func(a1, a2, cla.to_device(cl_queue, np.array(pars).astype(np.float32)), nx, ny, nz).get()
    else:
        r = cl_func(a1, a2, cla.to_device(cl_queue, np.array(pars).astype(np.float32)), nx, ny).get()
    return r


def fit_phase2_cu(pars, a1, a2, cu_func):
    """
    Fit function to match the phase between two arrays, with linear phase shift parameters. OpenCL version
    :param pars: a list of ndim+1 linear phase shift parameters (one constant plus one along each axis)
    :param a1: the amplitude (or weight) of the first array. Should be >=0
    :param a2:  the amplitude (or weight) of the second array. Should be >=0
    :return: a floating point figure of merit, ((a1 + a2) * delta_phi ** 2).sum()
    """
    nx = np.int32(a1.shape[-1])
    ny = np.int32(a1.shape[-2])
    if a1.ndim == 3:
        nz = np.int32(a1.shape[0])
        r = cu_func(a1, a2, cua.to_gpu(np.array(pars).astype(np.float32)), nx, ny, nz).get()
    else:
        r = cu_func(a1, a2, cua.to_gpu(np.array(pars).astype(np.float32)), nx, ny).get()
    return r


def r_match(d1: np.ndarray, d2: np.ndarray, percent=99, threshold=0.05):
    """
    Compute an unweighted R-factor between two arrays (complex or real)
    :param d1: the first array
    :param d2: the second array
    :param percent: a percent value between 0 and 100. If used, the R factor will only be calculated over the
        data points above the nth percentile multiplied by the threshold value in either arrays
    :param threshold: the R factor will only be calculated over the data points above the maximum or
        nth percentile multiplied by the threshold value
    :return: the R-factor calculated as sqrt(sum(abs(d1-d2)**2)/(0.5*sum(abs(d1)**2+abs(d2)**2)))
    """
    a1, a2 = abs(d1), abs(d2)
    idx = np.logical_or(a1 > (threshold * np.percentile(a1, percent)), (a2 > (threshold * np.percentile(a2, percent))))
    return np.sqrt(2 * (abs(d1[idx] - d2[idx]) ** 2).sum() / (abs(d1[idx]) ** 2 + abs(d2[idx]) ** 2).sum())


def match2(d1: np.ndarray, d2: np.ndarray, match_phase_ramp=False, match_orientation='center', match_scale=True,
           verbose=False, upsample_factor=1, use_gpu=True, return_params=False):
    """
    Match array d2 against array d1, by flipping it along one or several axis and/or calculating its conjugate,
    translation registration, and matching amplitudes. Both arrays should be 2D or 3D.
    :param d1: the first array
    :param d2: the second array
    :param match_phase_ramp: if True (the default), the phase ramps will be matched as well. This is slower. The
                             phase shift is always fitted, unless both arrays are real.
    :param match_orientation: either 'center' or 'all', to test only the center-of-symmetry inversion or a flip along
                              all axes
    :param match_scale: if True (the default), multiply d2 by a scale factor to match the amplitude of d1
    :param verbose: if True, print some info while matching arrays. If verbose>=2, also print timings, >=3: verbose fit
    :param upsample_factor: upsampling factor for subpixel registration (default: 1 - no subpixel). Good values
                            are 10 or 20.
    :param use_gpu: if True, will use a GPU if available
    :param return_params: if True, will also return flip, conjugate, shift and phase slope parameters
    :return: (d1, d2, r) the two arrays after matching their shape, orientation, translation and (optionally) phase.
        The first array is only cropped if necessary, but otherwise unmodified.
        r is the result of r_match(d1,d2, percent=99).
        If return_params is True, will return (d1, d2, r, flip, conj, pixel_shift, phase_shift),
        where flip is a tuple indicating whether each axis of d2 has been flipped, conj indicates if the conjugate
        of d2 was used, pixel_shift is the shift in pixels of d2, and phase_shift are the phase correction using
        corr_phase. The correction can be applied manually (assuming the shapes of the arrays already matched) using:
            d2c = flipn(d2, flip)
            d2c = ifftn(fourier_shift(fftn(d2), pixel_shift))
            if conj:
                d2c = d2c.conj()
            d2c = corr_phase(phase_shift, d2c)
    """
    # Match the two arrays shape
    t0 = timeit.default_timer()
    if d1.shape != d2.shape:
        d1, d2 = match_shape((d1, d2), cen=False)
    t1 = timeit.default_timer()
    if verbose >= 2:
        print("Match2: dt[match_shape]=%6.3fs" % (t1 - t0))

    # Match the orientation and translation based on the amplitudes
    a1 = abs(d1)
    a2 = abs(d2)
    errmin = 1e6
    flip_min = None
    if match_orientation == 'all':
        if d1.ndim == 3:
            for flipx in [0, 1]:
                for flipy in [0, 1]:
                    for flipz in [0, 1]:
                        s, err, dphi = register_translation(a1, flipn(a2, [flipz, flipy, flipx]))
                        if errmin > err:
                            flip_min = [flipz, flipy, flipx]
                            errmin = err
                        # print("Matching orientation: %d %d %d:" % (flipz, flipy, flipx), s, err, dphi)
            if verbose:
                print('Best orientation match: flipz=%d flipy=%d flipx=%d error=%6.3f' % (
                    flip_min[0], flip_min[1], flip_min[2], errmin))
        else:
            for flipx in [0, 1]:
                for flipy in [0, 1]:
                    s, err, dphi = register_translation(a1, flipn(a2, [flipy, flipx]))
                    if errmin > err:
                        flip_min = [flipy, flipx]
                        errmin = err
                    # print("Matching orientation: %d %d:" % (flipy, flipx), s, err, dphi)
            if verbose:
                print('Best orientation match: flipy=%d flipx=%d error=%6.3f' % (
                    flip_min[0], flip_min[1], errmin))
    else:
        if d1.ndim == 3:
            for flip in [0, 1]:
                s, err, dphi = register_translation(a1, flipn(a2, [flip, flip, flip]))
                if errmin > err:
                    flip_min = [flip, flip, flip]
                    errmin = err
                # print("Matching orientation: %d %d %d:" % (flipz, flipy, flipx), s, err, dphi)
            if verbose:
                print('Best orientation match: flipz=%d flipy=%d flipx=%d error=%6.3f' % (
                    flip_min[0], flip_min[1], flip_min[2], errmin))
        else:
            for flip in [0, 1]:
                s, err, dphi = register_translation(a1, flipn(a2, [flip, flip]))
                if errmin > err:
                    flip_min = [flip, flip]
                    errmin = err
                # print("Matching orientation: %d %d:" % (flipy, flipx), s, err, dphi)
            if verbose:
                print('Best orientation match: flipy=%d flipx=%d error=%6.3f' % (
                    flip_min[0], flip_min[1], errmin))

    d2 = flipn(d2, flip_min)
    t2 = timeit.default_timer()
    if verbose >= 2:
        print("Match2: dt[match_orientation]=%6.3fs" % (t2 - t1))
    conj = False
    phase_shift = None
    if match_phase_ramp:
        pixel_shift, err, dphi = register_translation(a1, abs(d2), upsample_factor=upsample_factor)
        # Roll can be used for pixel registration
        # d2 = np.roll(d2, [int(round(v)) for v in s], range(d2.ndim)) * np.exp(1j * dphi)
        d2 = ifftn(fourier_shift(fftn(d2), pixel_shift))
        t3 = timeit.default_timer()
        if verbose >= 2:
            print("Match2: dt[register_translation]=%6.3fs" % (t3 - t2))

        # Match phase shift and ramp, testing both conjugates
        if has_cuda and use_gpu:
            if verbose:
                print("Match2: match_phase_ramp using CUDA")
            if main_default_processing_unit.cu_device is None:
                main_default_processing_unit.select_gpu(language='cuda')
            if match2.pu is None:
                match2.pu = CUProcessingUnit()
                match2.pu.init_cuda(cu_device=main_default_processing_unit.cu_device, test_fft=False, verbose=False)
            if d1.ndim == 2:
                if match2.cu_dphi_red2 is None:
                    match2.cu_dphi_red2 = CU_RedK(np.float32, neutral="0", reduce_expr="a+b",
                                                  map_expr="DeltaPhi2(i, a1, a2, c, nx, ny)",
                                                  preamble=getks('cuda/complex.cu') +
                                                           getks("cdi/cuda/delta_phi_red.cu"),
                                                  options=["-use_fast_math"],
                                                  arguments="pycuda::complex<float> *a1, pycuda::complex<float>  *a2,"
                                                            "float *c,  const int nx, const int ny")
                cu_dphi_red = match2.cu_dphi_red2
            else:
                if match2.cu_dphi_red3 is None:
                    match2.cu_dphi_red3 = CU_RedK(np.float32, neutral="0", reduce_expr="a+b",
                                                  map_expr="DeltaPhi3(i, a1, a2, c, nx, ny, nz)",
                                                  preamble=getks('cuda/complex.cu') +
                                                           getks("cdi/cuda/delta_phi_red.cu"),
                                                  options=["-use_fast_math"],
                                                  arguments="pycuda::complex<float> *a1, pycuda::complex<float>  *a2,"
                                                            "float *c,  const int nx, const int ny, const int nz")
                cu_dphi_red = match2.cu_dphi_red3

            cu_d1 = cua.to_gpu(d1.astype(np.complex64))
            cu_d2 = cua.to_gpu(d2.astype(np.complex64))

            p0 = np.zeros(1 + d1.ndim)
            p0[0] = dphi
            p0 = minimize(fit_phase2_cu, p0, args=(cu_d1, cu_d2, cu_dphi_red), method='powell')
            if verbose >= 3:
                print(p0)
            cu_d2 = cua.to_gpu(d2.conj().astype(np.complex64))
            p1 = minimize(fit_phase2_cu, np.zeros(1 + d1.ndim), args=(cu_d1, cu_d2, cu_dphi_red), method='powell')
            if verbose >= 3:
                print(p1)
            del cu_d1, cu_d2
        elif has_opencl and use_gpu:
            if verbose:
                print("Match2: match_phase_ramp using OpenCL")
            if main_default_processing_unit.cl_device is None:
                main_default_processing_unit.select_gpu(language='opencl')
            if match2.pu is None:
                match2.pu = CLProcessingUnit()
                match2.pu.init_cl(cl_device=main_default_processing_unit.cl_device, test_fft=False, verbose=False)
            if d1.ndim == 2:
                if match2.cl_dphi_red2 is None:
                    match2.cl_dphi_red2 = CL_RedK(match2.pu.cl_ctx, np.float32, neutral="0", reduce_expr="a+b",
                                                  map_expr="DeltaPhi2(i, a1, a2, c, nx, ny)",
                                                  preamble=getks("cdi/opencl/delta_phi_red.cl"),
                                                  options="-cl-mad-enable -cl-fast-relaxed-math",
                                                  arguments="__global float2 *a1, __global float2 *a2,"
                                                            "__global float *c,  const int nx, const int ny")
                cl_dphi_red = match2.cl_dphi_red2
            else:
                if match2.cl_dphi_red3 is None:
                    match2.cl_dphi_red3 = CL_RedK(match2.pu.cl_ctx, np.float32, neutral="0", reduce_expr="a+b",
                                                  map_expr="DeltaPhi3(i, a1, a2, c, nx, ny, nz)",
                                                  preamble=getks("cdi/opencl/delta_phi_red.cl"),
                                                  options="-cl-mad-enable -cl-fast-relaxed-math",
                                                  arguments="__global float2 *a1, __global float2 *a2,"
                                                            "__global float *c, const int nx, const int ny,"
                                                            "const int nz")
                cl_dphi_red = match2.cl_dphi_red3

            cl_d1 = cla.to_device(match2.pu.cl_queue, d1.astype(np.complex64), async_=False)
            cl_d2 = cla.to_device(match2.pu.cl_queue, d2.astype(np.complex64), async_=False)

            p0 = np.zeros(1 + d1.ndim)
            p0[0] = dphi
            p0 = minimize(fit_phase2_cl, p0, args=(cl_d1, cl_d2, cl_dphi_red, match2.pu.cl_queue), method='powell')
            if verbose >= 3:
                print(p0)
            cl_d2 = cla.to_device(match2.pu.cl_queue, d2.conj().astype(np.complex64), async_=False)
            p1 = minimize(fit_phase2_cl, np.zeros(1 + d1.ndim),
                          args=(cl_d1, cl_d2, cl_dphi_red, match2.pu.cl_queue), method='powell')
            if verbose >= 3:
                print(p1)
            del cl_d1, cl_d2
        else:
            if verbose:
                print("Match2: match_phase_ramp using CPU:", has_opencl, use_gpu)
            phi1, a1 = np.angle(d1), abs(d1)
            phi2, a2 = np.angle(d2), abs(d2)
            if d1.ndim == 3:
                nz, ny, nx = d1.shape
                xyz = np.meshgrid(np.arange(nz) / nz, np.arange(ny) / ny, np.arange(nx) / nx, indexing='ij')
            else:
                ny, nx = d1.shape
                xyz = np.meshgrid(np.arange(ny) / ny, np.arange(nx) / nx, indexing='ij')

            p0 = np.zeros(1 + d1.ndim)
            p0[0] = dphi
            options = {'xtol': 1e-3, 'ftol': 1e-3}
            p0 = minimize(fit_phase2, p0, args=(phi1, phi2, a1, a2, xyz), method='powell', options=options)
            if verbose >= 3:
                print(p0)
            p1 = np.zeros(1 + d1.ndim)
            p1[0] = -dphi
            p1 = minimize(fit_phase2, p1, args=(phi1, -phi2, a1, a2, xyz), method='powell', options=options)
            if verbose >= 3:
                print(p1)
        if p0['fun'] < p1['fun']:
            phase_shift = p0['x']
            d2 = corr_phase(phase_shift, d2)
        else:
            phase_shift = p1['x']
            d2 = corr_phase(phase_shift, d2.conj())
            conj = True
        t4 = timeit.default_timer()
        if verbose >= 2:
            print("Match2: dt[match_phase_ramp]=%6.3fs" % (t4 - t3), "ramp: ", phase_shift)
    else:
        if np.sum(flip_min) % 2 and np.iscomplexobj(d1):
            d2 = d2.conj()
        pixel_shift, err, dphi = register_translation(d1, d2, upsample_factor=upsample_factor)
        if verbose >= 2:
            print("Match2: ", pixel_shift, err, dphi)
        phase_shift = np.zeros(d1.ndim + 1)
        phase_shift[0] = -dphi
        # Roll can be used for pixel registration
        # d2 = np.roll(d2, [int(round(v)) for v in s], range(d2.ndim)) * np.exp(1j * dphi)
        if np.isrealobj(d1) and np.isrealobj(d2):
            d2 = abs(ifftn(fourier_shift(fftn(d2), pixel_shift))).astype(np.float32)
        else:
            d2 = ifftn(fourier_shift(fftn(d2), pixel_shift)) * np.exp(1j * dphi)
        t4 = t3 = timeit.default_timer()
        if verbose >= 2:
            print("Match2: dt[register_translation]=%6.3fs" % (t3 - t2))

    if verbose:
        print(register_translation(a1, abs(d2)))

    if match_scale:
        # Match amplitudes to minimise (abs(a1-S*a2)**2).sum()
        a1, a2 = abs(d1), abs(d2)
        d2 *= (a1 * a2).sum() / (a2 ** 2).sum()

    r = r_match(d1, d2, percent=99, threshold=0.05)

    tn = timeit.default_timer()
    if verbose:
        print("Final R_match between arrays: R=%6.3f%% [dt=%8.3fs]" % (r * 100, tn - t0))

    if return_params:
        return d1, d2, r, flip_min, conj, pixel_shift, phase_shift
    return d1, d2, r


# Use this to avoid re-initialising context and kernel when calling the function multiple times
match2.pu = None
match2.cl_dphi_red2 = None
match2.cl_dphi_red3 = None
match2.cu_dphi_red2 = None
match2.cu_dphi_red3 = None
