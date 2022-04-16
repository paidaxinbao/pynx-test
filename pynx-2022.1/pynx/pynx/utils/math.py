# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2013-2014 : Fondation Nanosciences, Grenoble
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#         Ondrej Mandula

# import multiprocessing as mp
import numpy as np
from scipy.linalg import svd


def primes(n):
    """ Returns the prime decomposition of n as a list
    """
    v = [1]
    assert (n > 0)
    i = 2
    while i * i <= n:
        while n % i == 0:
            v.append(i)
            n //= i
        i += 1
    if n > 1:
        v.append(n)
    return v


def test_smaller_primes(n, maxprime=13, required_dividers=(4,)):
    """
    Test if the largest prime divider is <=maxprime, and optionally includes some dividers.
    
    Args:
        n: the integer number for which the prime decomposition will be tested
        maxprime: the maximum acceptable prime number. This defaults to the largest integer accepted by the clFFT library for OpenCL GPU FFT.
        required_dividers: list of required dividers in the prime decomposition. If None, this test is skipped.
    Returns:
        True if the conditions are met.
    """
    p = primes(n)
    if max(p) > maxprime:
        return False
    if required_dividers is not None:
        for k in required_dividers:
            if n % k != 0:
                return False
    return True


def smaller_primes(n, maxprime=13, required_dividers=(4,), decrease=True):
    """ Find the closest integer <= or >=n (or list/array of integers), for which the largest prime divider
    is <=maxprime, and has to include some dividers.
    The default values for maxprime is the largest integer accepted by the clFFT library for OpenCL GPU FFT.
    
    Args:
        n: the integer number
        maxprime: the largest prime factor acceptable
        required_dividers: a list of required dividers for the returned integer.
        decrease: if True (thed default), the integer returned will be <=n, otherwise it will be >=n.
    Returns:
        the integer (or list/array of integers) fulfilling the requirements
    """
    if (type(n) is list) or (type(n) is tuple) or (type(n) is np.ndarray):
        vn = [smaller_primes(i, maxprime=maxprime, required_dividers=required_dividers, decrease=decrease) for i in n]
        if type(n) is np.ndarray:
            return np.array(vn)
        else:
            return type(n)(tuple(vn))
    else:
        if maxprime < n:
            assert (n > 1)
            while test_smaller_primes(n, maxprime=maxprime, required_dividers=required_dividers) is False:
                if decrease:
                    n = n - 1
                    if n == 0:
                        # TODO: should raise an exception
                        return 0
                else:
                    n = n + 1
        return n


def ortho_modes(m, method='eig', verbose=False, return_matrix=False, nb_mode=None, return_weights=False):
    """
    Orthogonalize modes from a N+1 dimensional array or a list/tuple of N-dimensional arrays.
    The decomposition is such that the total intensity (i.e. (abs(m)**2).sum()) is conserved, and
    the modes are orthogonal, i.e. (mo[i]*mo[j].conj()).sum()=0 for i!=j

    Args:
        m: the stack of modes to orthogonalize along the first dimension.
        method: either 'eig' to use eigenvalue decomposition,
            or 'svd' to use singular value decomposition.
        verbose: if True, the matrix of coefficients will be printed
        return_matrix: if True, return the linear combination matrix
        nb_mode: the maximum number of modes to be returned. If None, all are returned.
        return_weights: if True, will also return the relative weights of all the modes. This is useful if nb_mode
                         is used, and only a partial list of modes is returned.
    Returns:
        an array (mo) with the same shape as given in input, but with orthogonal modes sorted by decreasing norm.
        Then if return_matrix is True, the matrix of linear coefficients is returned
        Then if return_weights is True, an array with the
    """
    if 'eig' in method:
        # Eigen analysis à la ptypy
        mm = np.array([[np.vdot(p2, p1) for p1 in m] for p2 in m])
        s, v = np.linalg.eig(mm)
    else:  # if 'svd' in method:
        # Singular value decomposition
        mm = np.reshape(m, (m.shape[0], m.size // m.shape[0]))
        v, s, vh = svd(mm, compute_uv=True, full_matrices=False)  # , lapack_driver='gesvd'
    e = (-s).argsort()
    v = v[:, e]
    # Make sure the largest coefficient is >0 to get same results for SVD and eig..
    for j in range(len(e)):
        if v[abs(v[:, j]).argmax(), j].real < 0:
            v[:, j] *= -1
    modes = np.array([sum(m[i] * v[i, j] for i in range(len(m))) for j in range(len(m))])
    if verbose:
        print("Orthonormal decomposition coefficients (rows)")
        print(np.array2string((v.transpose()), threshold=10, precision=3,
                              floatmode='fixed', suppress_small=True))
    if nb_mode is not None:
        nb_mode = min(len(m), nb_mode)
    else:
        nb_mode = len(m)

    if return_weights and return_matrix:
        w = np.array([(abs(modes[i]) ** 2).sum() for i in range(len(m))]) / (abs(modes) ** 2).sum()
        return modes[:nb_mode], v, w
    if return_matrix:
        return modes[:nb_mode], v
    if return_weights:
        w = np.array([(abs(modes[i]) ** 2).sum() for i in range(len(m))]) / (abs(modes) ** 2).sum()
        return modes[:nb_mode], w
    return modes[:nb_mode]


def full_width(x, y, ratio=0.5, outer=False):
    """
    Determine the full-width from an XY dataset.
    
    Args:
        x: the abscissa (1d array) of the data to be fitted. Assumed to be in monotonic ascending order
        y: the y-data as a 1d array. The absolute value of the array will be taken if it is complex.
        ratio: the fraction (default=0.5 for FWHM) at which the width should be measured
        outer: if True, the width will be measured by taking the outermost points which fall below the ratio to the maiximum.
               Otherwise, the width will be taken as the width around the maximum (regardless of secondary peak which may be above maw*ratio)

    Returns:
        the full width
    """
    n = len(y)
    if np.iscomplexobj(y):
        y = abs(y)
    imax = y.argmax()
    if imax == y.size:
        imax -= 1
    ay2 = y[imax] * ratio
    if outer:
        # Find the first and last values below max/2, in the entire array.
        ix1 = np.where(y[:imax + 1] > ay2)[0][0] - 1
        ix2 = np.where(y[imax:] > ay2)[0][-1] + imax + 1
    else:
        # Find the first values below max/2, left and right of the peak.
        # This allows FWHM to be found even if secondary peaks are > max/2
        try:
            ix1 = np.where(y[:imax] <= ay2)[0][-1]
        except IndexError:
            ix1 = 0
        try:
            ix2 = np.where(y[imax:] <= ay2)[0][0] + imax
        except IndexError:
            ix2 = n-1

    if ix1 >= 0:
        v0, v1 = y[ix1], y[ix1 + 1]
        xleft = (x[ix1 + 1] * (ay2 - v0) + x[ix1] * (v1 - ay2)) / (v1 - v0)
    else:
        xleft = x[0]

    if ix2 <= (n - 1):
        v2, v3 = y[ix2 - 1], y[ix2]
        xright = (x[ix2 - 1] * (ay2 - v3) + x[ix2] * (v2 - ay2)) / (v2 - v3)
    else:
        xright = x[n - 1]
    # print(ix1,imax,ix2, xleft, xright)
    return xright - xleft


def llk_poisson(iobs, imodel, imodel_min=0.1):
    """
    Compute the Poisson log-likelihood for a calculated intensity, given observed values.
    The value computed is normalized so that its asymptotic value (for a large number of points) is equal
    to the number of observed points

    Args:
        iobs: the observed intensities
        imodel: the calculated/model intensity
        imodel_min: the minimum accepted value for imodel (to avoid infinite llk when imodel=0 and iobs>0)

    Returns:
        The negative log-likelihood.
    """
    if np.isscalar(iobs):
        if iobs > 0:
            llk = imodel - iobs + iobs * np.log(iobs / imodel)
        else:
            llk = imodel
    else:
        imodel
        llk = np.empty(iobs.shape, dtype=np.float32)
        idx = np.where(iobs.flat > 0)
        llk.flat[idx] = np.take(
            (imodel - iobs + iobs * np.log(iobs / (imodel + imodel_min * (imodel < imodel_min)))).flat, idx)
        idx = np.where(iobs.flat == 0)
        llk.flat[idx] = np.take(imodel.flat, idx)
        # Negative iobs are masked
        idx = np.where(iobs.flat < 0)
        llk.flat[idx] = 0
    return 2 * llk


def llk_gaussian(iobs, imodel):
    """
    Compute the Gaussian log-likelihood for a calculated intensity, given observed values.
    The value computed is normalized so that its asymptotic value (for a large number of points) is equal
    to the number of observed points.

    Args:
        iobs: the observed intensities
        imodel: the calculated/model intensity

    Returns:
        The negative log-likelihood.
    """
    return (imodel - iobs) ** 2 / (np.abs(iobs) + 1) * (iobs >= 0)


def llk_euclidian(iobs, imodel):
    """
    Compute the Eucldian log-likelihood for a calculated intensity, given observed values.
    The value computed is normalized so that its asymptotic value (for a large number of points) is equal
    to the number of observed points. This model is valid if obs and calc are reasonably close.

    Args:
        iobs: the observed intensities
        imodel: the calculated/model intensity

    Returns:
        The negative log-likelihood.
    """
    return 4 * (np.sqrt(np.abs(imodel)) - np.sqrt(np.abs(iobs))) ** 2 * (iobs >= 0)


# # This does not perform as expected. Better get a cuda implementation
# def multiprocessing_random_poisson(d, nprocs=None):
#     """
#     Compute numpy.random.poisson(d) by distributing the task over available CPU cores.
#     :param d: the array for which np.random.poisson will be called
#     :param nprocs: number of parallel process to use. If None, uses multiprocessing.cpu_count()
#     :return: the updated value of d with Poisson-distributed values
#     """
#
#     if nprocs is None:
#         nprocs = mp.cpu_count()
#     dflat = d.flat
#     n, n1 = len(dflat), len(dflat) // nprocs
#     slices = [(i, min(n, i + n1)) for i in range(0, n, n1)]
#     with mp.Pool(processes=nprocs) as p:
#         res = [p.apply_async(np.random.poisson, args=(dflat[s[0]:s[1]],)) for s in slices]
#
#         for i in range(nprocs):
#             dflat[slices[i][0]:slices[i][1]] = res[i].get()
#     return d


if __name__ == '__main__':
    from pylab import *

    rc('text', usetex=True)
    if True:
        # Testing asymptotic values for the Poisson log-likelihood, for observed data following Poisson statistics.
        figure()
        nb = 2 ** 20
        vim = np.arange(0, 16, dtype=np.float32)
        vsum = np.zeros_like(vim)
        vsumllk = np.zeros_like(vim)
        for i in range(vim.size):
            m = 2 ** vim[i]
            print(m)
            imodel = np.random.uniform(0, m, nb)
            iobs = np.random.poisson(imodel)
            vsum[i] = iobs.sum()
            vsumllk[i] = llk_poisson(iobs, imodel).sum()
        semilogx(vsum / nb, vsumllk / nb, 'k.', label='Uniform')

        vsum = np.zeros_like(vim)
        vsumllk = np.zeros_like(vim)
        for i in range(vim.size):
            m = 2 ** vim[i]
            print(m)
            # imodel = np.random.pareto(2, nb) * m
            imodel = np.random.exponential(m, nb)
            iobs = np.random.poisson(imodel)
            vsum[i] = iobs.sum()
            vsumllk[i] = llk_poisson(iobs, imodel).sum()
        semilogx(vsum / nb, vsumllk / nb, 'r.', label='Exponential')
        legend()
        xlabel("$<I_{obs}>$")
        ylabel("$<LLK>$")
        title(r"$<LLK_{Poisson}>=\frac{2}{N_{obs}}\left\{\displaystyle\sum_{I_{obs}>0}\left[I_{obs}"
              r"*ln(\frac{I_{obs}}{I_{calc}}) + I_{calc} -I_{obs}\right]+ \sum_{I_{obs}=0}I_{calc}\right\}$")

    if True:
        # Testing asymptotic values for the Gaussian log-likelihood, for observed data following Poisson statistics.
        figure()
        nb = 2 ** 20
        vim = np.arange(0, 16, dtype=np.float32)
        vsum = np.zeros_like(vim)
        vsumllk = np.zeros_like(vim)
        for i in range(vim.size):
            m = 2 ** vim[i]
            print(m)
            imodel = np.random.uniform(0, m, nb)
            iobs = np.random.poisson(imodel)
            vsum[i] = iobs.sum()
            vsumllk[i] = llk_gaussian(iobs, imodel).sum()
        semilogx(vsum / nb, vsumllk / nb, 'k.', label='Uniform')

        vsum = np.zeros_like(vim)
        vsumllk = np.zeros_like(vim)
        for i in range(vim.size):
            m = 2 ** vim[i]
            print(m)
            # imodel = np.random.pareto(2, nb) * m
            imodel = np.random.exponential(m, nb)
            iobs = np.random.poisson(imodel)
            vsum[i] = iobs.sum()
            vsumllk[i] = llk_gaussian(iobs, imodel).sum()
        semilogx(vsum / nb, vsumllk / nb, 'r.', label='Exponential')
        legend()
        xlabel("$<I_{obs}>$")
        ylabel("$<LLK>$")
        title(r"$<LLK_{Gaussian}>=\frac{1}{N_{obs}}\displaystyle\sum\frac{(I_{obs}-I_{calc})^2}{I_{obs}+1}$")

    if True:
        # Testing asymptotic values for the Euclidian log-likelihood, for observed data following Poisson statistics.
        figure()
        nb = 2 ** 20
        vim = np.arange(0, 16, dtype=np.float32)
        vsum = np.zeros_like(vim)
        vsumllk = np.zeros_like(vim)
        for i in range(vim.size):
            m = 2 ** vim[i]
            print(m)
            imodel = np.random.uniform(0, m, nb)
            iobs = np.random.poisson(imodel)
            vsum[i] = iobs.sum()
            vsumllk[i] = llk_euclidian(iobs, imodel).sum()
        semilogx(vsum / nb, vsumllk / nb, 'k.', label='Uniform')

        vsum = np.zeros_like(vim)
        vsumllk = np.zeros_like(vim)
        for i in range(vim.size):
            m = 2 ** vim[i]
            print(m)
            # imodel = np.random.pareto(2, nb) * m
            imodel = np.random.exponential(m, nb)
            iobs = np.random.poisson(imodel)
            vsum[i] = iobs.sum()
            vsumllk[i] = llk_euclidian(iobs, imodel).sum()
        semilogx(vsum / nb, vsumllk / nb, 'r.', label='Exponential')
        legend()
        xlabel("$<I_{obs}>$")
        ylabel("$<LLK>$")
        title(r"$<LLK_{Euclidian}>=\frac{4}{N_{obs}}\displaystyle\sum(\sqrt{I_{obs}}-\sqrt{I_{calc}})^2$")
