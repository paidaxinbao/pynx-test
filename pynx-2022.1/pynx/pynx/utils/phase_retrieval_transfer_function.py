# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#         Julio Cesar da Silva (mailto: jdasilva@esrf.fr) (nyquist and ring_thickness code)


import numpy as np


def ring_thickness(shape):
    """
    Define ring_thick
    """
    n = shape
    nmax = np.max(n)
    x = np.arange(-np.fix(n[1] / 2.0), np.ceil(n[1] / 2.0)) * np.floor(nmax / 2.0) / np.floor(n[1] / 2.0)
    y = np.arange(-np.fix(n[0] / 2.0), np.ceil(n[0] / 2.0)) * np.floor(nmax / 2.0) / np.floor(n[0] / 2.0)
    if len(shape) == 3:
        z = np.arange(-np.fix(n[2] / 2.0), np.ceil(n[2] / 2.0)) * np.floor(nmax / 2.0) / np.floor(n[2] / 2.0)
        xx = np.meshgrid(x, y, z)
    elif len(shape) == 2:
        xx = np.meshgrid(x, y)
    else:
        print('Number of dimensions is different from 2 or 3.Exiting...')
        raise SystemExit('Number of dimensions is different from 2 or 3.Exiting...')
    sumsquares = np.zeros_like(xx[-1])
    for ii in np.arange(0, len(shape)):
        sumsquares += xx[ii] ** 2
    index = np.round(np.sqrt(sumsquares))
    return index


def nyquist(shape):
    """
    Evaluate the Nyquist Frequency
    """
    nmax = np.max(shape)
    fnyquist = np.floor(nmax / 2.0)
    freq = np.arange(0, fnyquist + 1)
    return freq, fnyquist


def prtf(icalc, iobs, mask=None, ring_thick=5, shell_averaging_method='before', norm_percentile=100, scale=False):
    """
    Compute the phase retrieval transfer function, given calculated and observed intensity. Note that this
    function assumes that calc and obs are orthonormal arrays.

    :param icalc: the calculated intensity array (origin at array center), either 2D or 3D
    :param iobs: the observed intensity (origin at array center)
    :param mask: the mask, values > 0 indicating bad pixels in the observed intensity array
    :param ring_thick: the thickness of each shell or ring, in pixel units
    :param shell_averaging_method: by default ('before') the amplitudes are averaged over the shell before
                                   the PRTF=<calc>/<obs> is computed. If 'after' is given, then the ratio of calc
                                   and observed amplitudes will first be calculated (excluding zero-valued observed
                                   pixels), and then averaged to compute the PRTF=<calc/obs>.
    :param norm_percentile: the output PRTF is normalised so that the 'norm_percentile' percentile value is 1. If None,
                            no normalisation is performed.
    :param scale: if True (default=False), the calculated array is scaled so that the non-masked integrated intensity
                 are equal. Should only be used with norm_percentile=None
    :return: a tuple with the (frequency, frequency_nyquist, prtf, iobs), where frequency, prtf  are masked arrays,
             fnyquist the Nyquist frequency, and iobs the integrated observed intensity per shell
    """

    # TODO assumes uniform grid i.e pixel same size in all dimensions and no curvature 
    #  -  need reciprocal space coords / an ortho-normalisation matrix
    if scale:
        s = (iobs * (mask == 0)).sum() / (icalc * (mask == 0)).sum()
    else:
        s = 1
    calc = np.sqrt(icalc * s)
    obs = np.sqrt(abs(iobs))
    prtf, nb, iobs_shell = [], [], []
    index = ring_thickness(iobs.shape)
    freq, fnyquist = nyquist(iobs.shape)
    if mask is None:
        mask = np.zeros(iobs.shape, dtype=np.int8)
    for ii in freq:
        tmp = np.where(np.logical_and(index == ii, mask == 0))
        nb.append(len(tmp[0]))
        if len(tmp[0]):
            if ring_thick == 0:
                tmpcalc = calc[tmp]
                tmpobs = obs[tmp]
            else:
                r2 = ring_thick / 2
                tmpcalc = calc[(np.where((index >= (ii - r2)) & (index <= (ii + r2)) & (mask == 0)))]
                tmpobs = obs[(np.where((index >= (ii - r2)) & (index <= (ii + r2)) & (mask == 0)))]
            # TODO: check how PRTF should be calculated, after or before cumming intensities in ring thickness
            if 'after' in shell_averaging_method.lower():
                # Average Icalc/Iobs
                arrtmpcalc = np.array(tmpcalc)
                arrtmpobs = np.array(tmpobs)
                nbvox = (arrtmpobs > 0).sum()
                prtfcalc = np.divide(arrtmpcalc, arrtmpobs, out=np.zeros_like(arrtmpcalc), where=arrtmpobs != 0)
                prtf.append(prtfcalc.sum() / nbvox)
                iobs_shell.append(arrtmpobs.sum())
            else:
                # Average Icalc.sum(ring) / Iobs.sum(ring) - more optimistic
                iobs_shell.append(tmpobs.sum())
                prtf.append(tmpcalc.sum() / iobs_shell[-1])
        else:
            # All values are masked (central stop ?)
            # print(ii)
            prtf.append(0)
            iobs_shell.append(0)
    prtf = np.array(prtf, dtype=np.float32)
    nb = np.array(nb, dtype=np.int)
    if norm_percentile is not None:
        prtf /= np.percentile(prtf[nb > 0], norm_percentile)
    return np.ma.masked_array(freq, mask=(nb == 0)), fnyquist, np.ma.masked_array(prtf, mask=(nb == 0)), iobs_shell


def plot_prtf(freq, fnyquist, prtf, iobs_shell=None, nbiobs_shell=None, pixel_size=None, title=None, file_name=None,
              fig_num=101):
    """
    Plot the phase retrieval transfer function

    :param freq: the frequencies for which the phase retrieval transfer function was calculated
    :param fnyquist: the nyquist frequency
    :param prtf: the phase retrieval transfer function
    :param iobs_shell: the integrated observed intensity per frequency shell (same size as prtf), which will be plotted
                       against a second y-axis and can give some information about the amount of scattering.
    :param nbiobs_shell: number of points per iobs_shell (to be able to compute the average Iobs per pixel)
    :param pixel_size: the pixel size in metres, for resolution axis
    :param title: the figure title
    :param file_name: if given, the plot will be saved to this file ('*.png' or '*.pdf')
    """
    from pynx.utils.matplotlib import pyplot as plt
    plt.figure(fig_num, figsize=(8, 4))
    plt.clf()
    ax1 = plt.gca()
    ax1.plot(freq / fnyquist, prtf, label='PRTF')
    ax1.grid()
    ax1.set_xlabel(r"relative spatial frequency ($f/f_{Nyquist}$)")
    ax1.set_ylabel("$PRTF$")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    # ax1.hlines(1 / np.exp(1), 0, 1, 'r', '--', label="Threshold (1/e)")
    ax1.legend(loc='center left', framealpha=0.5)

    if pixel_size is not None:
        s = np.log10(pixel_size)
        # Add secondary X-axis with resolution in metric units
        if s < -6:
            unit_name = "nm"
            s = 1e9
        elif s < -3:
            unit_name = "$\mu m$"
            s = 1e6
        elif s < 0:
            unit_name = "mm"
            s = 1e3
        else:
            unit_name = "m"
            s = 1
        ax2 = ax1.twiny()
        x = plt.xticks()[0][1:]
        x2 = pixel_size * s / x
        ax2.set_xticks(x)
        ax2.set_xticklabels(["%.1f" % xx for xx in x2])
        ax2.set_xlabel(r"Resolution in %s" % unit_name)

    if iobs_shell is not None:
        ax3 = ax1.twinx()
        ax3.semilogy(freq / fnyquist, iobs_shell, 'r.', alpha=0.1, label=r'$\Sigma I_{obs}$')
        if nbiobs_shell is not None:
            ax3.semilogy(freq / fnyquist, iobs_shell / (nbiobs_shell+1e-6), 'g.', alpha=0.1, label=r'$<I_{obs}>$')
        ax3.legend(loc='center right', framealpha=0.5)
        plt.ylabel(r'$I_{obs}$', color='red')

    if title is not None:
        if len(title) > 30:
            plt.title(title, fontsize=9)
        else:
            plt.title(title)
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name)
