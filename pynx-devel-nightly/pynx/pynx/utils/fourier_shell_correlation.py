# -*- coding: utf-8 -*-
# Computes the Fourier Shell Correlation between image1 and image2, and computes
# the threshold funcion T of 1 or 1/2 bit.
#
#   (c) 2015-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Julio Cesar da Silva (mailto: jdasilva@esrf.fr)
#         (Vincent Favre-Nicolin, favre@esrf.fr : small changes for PyNX incorporation)

from __future__ import division, print_function
import numpy as np
from numpy import meshgrid
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from pynx.utils.matplotlib import pyplot as plt

__all__ = ['FourierShellCorr', 'FSCPlot', 'HannApod']


def _radtap(X, Y, tappix, zerorad):
    """
    Creates a central cosine tapering. 
    It receives the X and Y coordinates, tappix is the extent of
    tapering, zerorad is the radius with no data (zeros).
    """
    tau = 2 * tappix  # period of cosine function (only half a period is used)

    R = np.sqrt(X ** 2 + Y ** 2)
    taperfunc = 0.5 * (1 + np.cos(2 * np.pi * (R - zerorad - tau / 2.) / tau))
    taperfunc = (R > zerorad + tau / 2.) * 1.0 + taperfunc * (R <= zerorad + tau / 2)
    taperfunc = taperfunc * (R >= zerorad)
    return taperfunc


class HannApod:
    def __init__(self, outputdim, filterdim, unmodsize):
        self.outputdim = outputdim
        self.unmodsize = unmodsize
        self.filterdim = filterdim

    def fract_hanning(self):  # outputdim,unmodsize):
        """
        fract_hanning(outputdim,unmodsize)
        out = Square array containing a fractional separable Hanning window with
        DC in upper left corner.
        outputdim = size of the output array
        unmodsize = Size of the central array containing no modulation.
        Creates a square hanning window if unmodsize = 0 (or ommited), otherwise the output array 
        will contain an array of ones in the center and cosine modulation on the
        edges, the array of ones will have DC in upper left corner.
        """
        N = np.arange(0, self.outputdim)
        Nc, Nr = np.meshgrid(N, N)
        if self.unmodsize == 0:
            out = (1. + np.cos(2 * np.pi * Nc / self.outputdim)) * (1. + np.cos(2 * np.pi * Nr / self.outputdim)) / 4.
        else:
            # columns modulation
            outc = (1. + np.cos(
                2 * np.pi * (Nc - np.floor((self.unmodsize - 1) / 2)) / (self.outputdim + 1 - self.unmodsize))) / 2.
            if np.floor((self.unmodsize - 1) / 2.) > 0:
                outc[:, :np.floor((self.unmodsize - 1) / 2.)] = 1
            outc[:, np.floor((self.unmodsize - 1) / 2) + self.outputdim + 3 - self.unmodsize:len(N)] = 1
            # row modulation
            outr = (1. + np.cos(
                2 * np.pi * (Nr - np.floor((self.unmodsize - 1) / 2)) / (self.outputdim + 1 - self.unmodsize))) / 2.
            if np.floor((self.unmodsize - 1) / 2.) > 0:
                outr[:np.floor((self.unmodsize - 1) / 2.), :] = 1
            outr[np.floor((self.unmodsize - 1) / 2) + self.outputdim + 3 - self.unmodsize:len(N), :] = 1

            out = outc * outr

        return out

    def fract_hanning_pad(self):  # outputdim,filterdim,unmodsize):#(N,N,np.round(N*(1-filtertomo))):
        """    
        fract_hanning_pad(outputdim,filterdim,unmodsize)
        out = Square array containing a fractional separable Hanning window with
        DC in upper left corner.
        outputdim = size of the output array
        filterdim = size of filter (it will zero pad if filterdim<outputdim)
        unmodsize = Size of the central array containing no modulation.
        Creates a square hanning window if unmodsize = 0 (or ommited), otherwise the output array 
        will contain an array of ones in the center and cosine modulation on the
        edges, the array of ones will have DC in upper left corner.
        """
        if self.outputdim < self.unmodsize:
            raise SystemExit('Output dimension must be smaller or equal to size of unmodulated window')
        if self.outputdim < self.filterdim:
            raise SystemExit('Filter cannot be larger than output size')
        if self.unmodsize < 0:
            self.unmodsize = 0
            print('Specified unmodsize<0, setting unmodsize = 0')
        out = np.zeros((self.outputdim, self.outputdim))
        auxindini = int(np.round(self.outputdim / 2. - self.filterdim / 2.))
        auxindend = int(np.round(self.outputdim / 2. + self.filterdim / 2.))
        hanning_window = self.fract_hanning()
        out[auxindini:auxindend, auxindini:auxindend] = fftshift(hanning_window)
        # out[auxindini:auxindend, auxindini:auxindend]=np.fft.fftshift(self.fract_hanning(filterdim,unmodsize))
        # return np.fft.fftshift(out)
        return out


class FourierShellCorr(HannApod):
    """
    Computes the Fourier Shell Correlation between image1 and image2, and computes
    the threshold funcion T of 1 or 1/2 bit. 
    It can handle non-cube arrays, but it assumes that the voxel is isotropic.
    It applies a Hanning window of the size of the data to the data before the 
    Fourier transform calculations to attenuate the border effects.
    
    Reference: M. van Heel, M. Schatz, "Fourier shell correlation threshold criteria",
    Journal of Structural Biology 151, 250-262 (2005) https://doi.org/10.1016/j.jsb.2005.05.009
    
    @author: Julio Cesar da Silva (jdasilva@esrf.fr) 
    """

    def __init__(self, img1, img2, snrt=0.2071, ring_thick=0, rad_apod=60, axial_apod=20):
        """

        :param img1: first image (2D or 3D)
        :param img2: second image (2D or 3D)
        :param snrt: power SNR for threshold computation. Options:
          SNRt = 0.5 -> 1 bit threshold for average
          SNRt = 0.2071 -> 1/2 bit threshold for average
        :param ring_thick: thickness (in pixel units) of the frequency rings.  Normally the pixels get
          assigned to the closest integer pixel ring in Fourier Domain.
          With ring_thick, each ring gets more pixels and  more statistics.
        :param rad_apod: radial apodisation width
        :param axial_apod: axial apodisation width
        """
        self.snrt = snrt
        self.ring_thick = ring_thick
        self.img1 = np.array(img1)
        self.img2 = np.array(img2)
        self.rad_apod = rad_apod
        self.axial_apod = axial_apod
        print('Input images have {} dimensions'.format(self.img1.ndim))
        if self.img1.shape != self.img2.shape:
            print("Images must have the same size")
            raise SystemExit
        if ring_thick != 0:
            print('Using ring_thick = {}'.format(ring_thick))
        print('Using SNRt =', snrt)

    def nyquist(self):
        """
        Evaluate the Nyquist Frequency
        """
        nmax = np.max(self.img1.shape)
        fnyquist = np.floor(nmax / 2.0)
        f = np.arange(0, fnyquist + 1)
        return f, fnyquist

    def ringthickness(self):
        """
        Define ring_thick
        """
        n = self.img1.shape
        nmax = np.max(n)
        x = np.arange(-np.fix(n[1] / 2.0), np.ceil(n[1] / 2.0)) * np.floor(nmax / 2.0) / np.floor(n[1] / 2.0)
        y = np.arange(-np.fix(n[0] / 2.0), np.ceil(n[0] / 2.0)) * np.floor(nmax / 2.0) / np.floor(n[0] / 2.0)
        if self.img1.ndim == 3:
            z = np.arange(-np.fix(n[2] / 2.0), np.ceil(n[2] / 2.0)) * np.floor(nmax / 2.0) / np.floor(n[2] / 2.0)
            X = meshgrid(x, y, z)
        elif self.img1.ndim == 2:
            X = np.meshgrid(x, y)
        else:
            print('Number of dimensions is different from 2 or 3.Exiting...')
            raise SystemExit('Number of dimensions is different from 2 or 3.Exiting...')
        sumsquares = np.zeros_like(X[-1])
        for ii in np.arange(0, self.img1.ndim):
            sumsquares += X[ii] ** 2
        index = np.round(np.sqrt(sumsquares))
        return index

    def apodization(self):
        """
        Compute the Hanning window of the size of the data for the apodization
        """
        n = self.img1.shape
        if self.img1.ndim == 2:
            window = np.outer(np.hanning(n[0]), np.hanning(n[1]))
        elif self.img1.ndim == 3:
            window1 = np.hanning(n[0])
            window2 = np.hanning(n[1])
            window3 = np.hanning(n[2])
            windowaxial = np.outer(window2, window3)
            windowsag = np.array([window1 for ii in range(n[1])]).swapaxes(0, 1)
            # win2d = np.rollaxis(np.array([np.tile(windowaxial,(1,1)) for ii in range(n[0])]),1).swapaxes(1,2)
            win2d = np.array([np.tile(windowaxial, (1, 1)) for ii in range(n[0])])
            window = np.array([np.squeeze(win2d[:, :, ii]) * windowsag for ii in range(n[2])]).swapaxes(0, 1).swapaxes(
                1, 2)
        else:
            print('Number of dimensions is different from 2 or 3. Exiting...')
            raise SystemExit('Number of dimensions is different from 2 or 3. Exiting...')
        return window

    def circle(self):
        if self.img1.ndim == 2:
            shape_x = self.img1.shape[1]
            shape_y = self.img1.shape[0]
        elif self.img1.ndim == 3:
            shape_x = self.img1.shape[2]
            shape_y = self.img1.shape[1]
        x_array = np.arange(0, shape_x)
        y_array = np.arange(0, shape_y)
        self.X, self.Y = np.meshgrid(x_array - np.round(shape_x / 2.), y_array - np.round(shape_y / 2.))
        circular_region = 1 - _radtap(self.X, self.Y, self.rad_apod, np.round(shape_x / 2.) - self.rad_apod)
        return circular_region

    def transverse_apodization(self):
        """
        Compute the Hanning window of the size of the data for the apodization
        """
        print('Calculating the transverse apodization')
        n = self.img1.shape
        HannApod.__init__(self, n[0], n[0], n[0] - 2 * self.axial_apod)
        filters = HannApod.fract_hanning_pad(self)
        window1d = filters[:, int(filters.shape[0] / 2)]
        window2d = np.array([window1d for ii in range(n[1])]).swapaxes(0, 1)
        return window2d

    def fouriercorr(self):
        """
        Compute FSC and threshold
        """
        # Apodization
        n = self.img1.shape
        circular_region = self.circle()
        if self.img1.ndim == 2:
            self.window = circular_region
        elif self.img1.ndim == 3:
            window2D = self.transverse_apodization()
            circle3D = np.asarray([circular_region for ii in range(n[0])])
            self.window = np.array([np.squeeze(circle3D[:, :, ii]) * window2D for ii in range(n[2])]) \
                .swapaxes(0, 1).swapaxes(1, 2)

        # FSC computation
        F1 = ifftshift(fftn(fftshift(self.img1 * self.window)))
        # F1 = ifftshift(fftn(fftshift(self.img1)))
        # print(F1.shape)
        F2 = ifftshift(fftn(fftshift(self.img2 * self.window)))
        # F2 = ifftshift(fftn(fftshift(self.img2)))
        C, C1, C2, npts = [[], [], [], []]
        index = self.ringthickness()
        f, fnyquist = self.nyquist()
        for ii in f:
            if self.ring_thick == 0:
                auxF1 = F1[np.where(index == ii)]
                auxF2 = F2[np.where(index == ii)]
            else:
                auxF1 = F1[(np.where((index >= (ii - self.ring_thick / 2)) & (index <= (ii + self.ring_thick / 2))))]
                auxF2 = F2[(np.where((index >= (ii - self.ring_thick / 2)) & (index <= (ii + self.ring_thick / 2))))]
            C.append(np.sum(auxF1 * np.conj(auxF2)))
            C1.append(np.sum(auxF1 * np.conj(auxF1)))
            C2.append(np.sum(auxF2 * np.conj(auxF2)))
            npts.append(auxF1.shape[0])
            # The correlation  
        FSC = np.abs(np.asarray(C)) / (np.sqrt(np.asarray(C1) * np.asarray(C2)))

        npts = np.asarray(npts)
        # Threshold computation
        if hasattr(self.snrt, '__len__'):
            T = []
            for snrt in self.snrt:
                Tnum = (snrt + (2 * np.sqrt(snrt) / np.sqrt(npts + np.spacing(1))) + 1 / np.sqrt(npts))
                Tden = (snrt + (2 * np.sqrt(snrt) / np.sqrt(npts + np.spacing(1))) + 1)
                T.append(Tnum / Tden)
        else:
            Tnum = (self.snrt + (2 * np.sqrt(self.snrt) / np.sqrt(npts + np.spacing(1))) + 1 / np.sqrt(npts))
            Tden = (self.snrt + (2 * np.sqrt(self.snrt) / np.sqrt(npts + np.spacing(1))) + 1)
            T = Tnum / Tden

        return FSC, T


class FSCPlot(FourierShellCorr):
    """
    Plot the FSC and threshold curves
    """

    def __init__(self, img1, img2, snrt=[0.2071, 0.5], ring_thick=0, rad_apod=60, axial_apod=20, pixel_size=None):
        FourierShellCorr.__init__(self, img1, img2, snrt, ring_thick, rad_apod, axial_apod)
        self.FSC, self.T = FourierShellCorr.fouriercorr(self)
        self.f, self.fnyquist = FourierShellCorr.nyquist(self)
        self.pixel_size = pixel_size
        self.img1 = img1
        self.img2 = img2

    def plot(self, save_plot=None, plot_images=False, cmap=None, figsize=None):
        """

        :param save_plot: if given (string), save the plot to the given file name
        :param plot_images: if True, the images will be plotted along the FSC plot
        :param cmap: the colormap to be used. Useful for phase maps
        :param figsize: thf igure size to use. If None, a default is used.
        :return: Nothing. A dictionary with the plotted coordinates for all curves and coordinates is saved as self.d

        """
        self.d = {}
        if plot_images:
            if figsize is None:
                figsize = 12, 4
            plt.figure(figsize=figsize)
            ax1 = plt.subplot(141)
            ax1.imshow(self.img1, interpolation='none', cmap=cmap)
            ax1.set_axis_off()
            ax1.set_title('Image 1 (reference)')
            ax2 = plt.subplot(142)
            ax2.imshow(self.img2, interpolation='none', cmap=cmap)
            ax2.set_axis_off()
            ax2.set_title('Image 2')
            plt.subplot(122)
        else:
            plt.figure(figsize=figsize)
            plt.clf()
        self.d['f_nyquist'] = self.f / self.fnyquist
        self.d['fsc'] = self.FSC
        plt.plot(self.f / self.fnyquist, self.FSC, '-k', label='FSC')
        plt.legend()
        if hasattr(self.snrt, '__len__') is False:
            vsnrt = [self.snrt]
            vT = [self.T]
        else:
            vsnrt = self.snrt
            vT = self.T
        for i in range(len(vsnrt)):
            snrt = vsnrt[i]
            T = vT[i]
            if np.isclose(snrt, 0.2071):
                self.d['1/2 bit threshold'] = T
                plt.plot(self.f / self.fnyquist, T, '--b', label='1/2 bit threshold')
                plt.legend()
            elif np.isclose(snrt, 0.5):
                self.d['1 bit threshold'] = T
                plt.plot(self.f / self.fnyquist, T, '--r', label='1 bit threshold')
                plt.legend()
            else:
                self.d['Threshold SNR = %g ' % self.snrt] = T
                plotT = plt.plot(self.f / self.fnyquist, T)
                plt.legend(plotT, 'Threshold SNR = %g ' % self.snrt, loc='center')
        plt.xlim(0, 1)
        plt.ylim(0, 1.1)
        plt.xlabel('Spatial frequency/Nyquist')
        plt.ylabel('Magnitude')
        plt.grid()
        if self.pixel_size is not None:
            s = np.log10(self.pixel_size)
            # Add secondary X-axis with resolution in metric units
            if s < -6:
                unit_name = "nm"
                s = 1e9
            elif s < -3:
                unit_name = u"µm"
                s = 1e6
            elif s < 0:
                unit_name = "mm"
                s = 1e3
            else:
                unit_name = "m"
                s = 1

            ax1 = plt.gca()
            ax2 = ax1.twiny()
            x = plt.xticks()[0][1:]
            x2 = self.pixel_size * s / x
            ax2.set_xticks(x)
            ax2.set_xticklabels(["%.1f" % xx for xx in x2])
            ax2.set_xlabel(r"Resolution in %s" % (unit_name))
            self.d['nm'] = x2

        if save_plot is not None:
            plt.savefig(save_plot)
        plt.show()
