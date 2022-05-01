# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2013-2014 : Fondation Nanosciences, Grenoble
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#         Ondrej Mandula

from __future__ import division

import sys
import warnings
import numpy as np
from numpy import pi
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import shift
from pynx.utils.matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from ..utils.plot_utils import showCplx
from ..utils.pattern import get_img, spiral_fermat, spiral_archimedes, siemens_star
from .shape import get_view_coord, calc_obj_shape
from pynx.utils.array import rebin
from .. import wavefront
from .. import ptycho


def gauss2D(im_size=(64, 64), mu=(0, 0), sigma=(1, 1), rotation=0):
    """
    2D gaussian rotated by rotation (in degrees) angle. L1 normalisation to 1.
    """
    ny, nx = im_size
    v = np.array(sigma) ** 2
    mu = np.array(mu)
    rotation = np.deg2rad(rotation)

    x = np.linspace(-nx // 2, nx // 2, nx)
    y = np.linspace(-ny // 2, ny // 2, ny)
    xx, yy = np.meshgrid(x, y)
    xxr = xx * np.cos(rotation) - yy * np.sin(rotation)
    yyr = xx * np.sin(rotation) + yy * np.cos(rotation)

    mu[0] = mu[0] * np.cos(rotation) - mu[1] * np.sin(rotation)
    mu[1] = mu[0] * np.sin(rotation) + mu[1] * np.cos(rotation)

    g = np.exp(-((xxr - mu[0]) ** 2 / (2 * v[0]) + (yyr - mu[1]) ** 2 / (2 * v[1])))
    return g / g.sum()


def sinc2d(im_size=(64, 64), center=(0, 0), width=(1, 1), rotation=0):
    """
    2D squared sinc function (with + and - sign) rotated by rotation (in degrees) angle. L1 normalisation to 1.
    """
    ny, nx = im_size
    mu = np.array(center)
    rotation = np.deg2rad(rotation)

    x = np.linspace(-nx // 2, nx // 2, nx)
    y = np.linspace(-ny // 2, ny // 2, ny)
    xx, yy = np.meshgrid(x, y)
    xxr = xx * np.cos(rotation) - yy * np.sin(rotation)
    yyr = xx * np.sin(rotation) + yy * np.cos(rotation)

    mu[0] = mu[0] * np.cos(rotation) - mu[1] * np.sin(rotation)
    mu[1] = mu[0] * np.sin(rotation) + mu[1] * np.cos(rotation)

    x = xxr - mu[0]
    y = yyr - mu[1]

    g = np.sinc(x / width[0]) * np.sinc(y / width[1])
    g = g ** 2 * np.sign(g)
    return g / abs(g).sum()


def disc(im_size=(64, 64), center=(0, 0), radius=10):
    """
    Disc of given radius.
    """
    ny, nx = im_size
    x = np.linspace(-nx // 2, nx // 2, nx)
    y = np.linspace(-ny // 2, ny // 2, ny)
    xx, yy = np.meshgrid(x, y)  # the first coordinate is vertical, second is horizontal!
    g = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius ** 2
    return 1.0 * g


def make_beam_stop(im_size=(256, 256), radius=10):
    nx, ny = im_size
    x = np.linspace(-nx // 2, nx // 2, nx)
    y = np.linspace(-ny // 2, ny // 2, ny)
    yy, xx = np.meshgrid(y, x)  # the first coordinate is vetical, second is horizontal!
    return xx ** 2 + yy ** 2 >= radius ** 2


def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.

    Source: https://github.com/berceanu/gp-linear-response/blob/master/opo.py

    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length)  # rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)

    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x < alpha / 2
    w[first_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[first_condition] - alpha / 2)))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x >= (1 - alpha / 2)
    w[third_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[third_condition] - 1 + alpha / 2)))

    return w


def tukeywin2D(window_size, alpha=0.5):
    w1 = tukeywin(window_size[0], alpha)
    w2 = tukeywin(window_size[1], alpha)
    return np.outer(w1, w2)


def psi(obj, probe, x, y, border):
    cx, cy = get_view_coord(obj.shape, probe.shape, x, y, integer=False)
    ny, nx = probe.shape
    dx, dy = x % 1, y % 1
    cx, cy = int(round(cx - dx)), int(round(cy - dy))
    if np.isclose(dx, 0) and np.isclose(dy, 0):
        return probe * obj[cy:cy + ny, cx:cx + nx]
    else:
        o = obj[cy:cy + ny, cx:cx + nx]
        return probe * (shift(o.real, (-dy, -dx), mode='wrap', order=2) + shift(o.imag, (-dy, -dx), mode='wrap',
                                                                                order=2) * 1j)
        # fourier_shift creates some diffraction from the object discontinuity on the borders
        # return probe * fourier_shift(o, (-dy, -dx))


class Im(object):
    def __init__(self, values=None, info=None):
        self.values = values
        self.info = info
        if self.info is None:
            self.info = {}
        if values is not None:
            self.info.update({'type': 'Custom'})

    def make_even(self):
        """Ensures an even shape of the images along x and y by removing one pixel when necessary"""
        if self.values.ndim >= 2:  # Should work on 2D images as well as stack of 2D images
            psy, psx = self.values.shape[-2:]
            if psx % 2:
                psx -= 1
            if psy % 2:
                psy -= 1
            if (psy, psx) != self.values.shape[-2:]:
                self.values = self.values[..., :psy, :psx]
                self.info['shape'] = self.values.shape
                self.info['type'] += 'Even'
                print("Changing shape to even values: (%g,%g)" % self.values.shape)

    def show(self):
        if len(self.values) == 2:  # scan positions
            plt.figure()
            plt.plot(self.values[1], self.values[0], '-x')
            plt.axis('equal')
            plt.gca().invert_yaxis()
            plt.xlabel('pos_hor [pixels]')
            plt.ylabel('pos_ver [pixels]')
            plt.grid(b=True)
        elif np.ndim(self.values) < 3:  # for 2D complex images
            showCplx(self.values)
        else:
            pass


class Simulation(object):
    """
    Simulation of ptychographic data (object, probe, diffraction data).

    EXAMPLES:
    d = ptycho.Simulation() # Initialises simulation object with default values of probe, object and scan

    or

    # Initialises ptychographic dataset with specifice obj (complex numpy array), probe (complex numpy array) and scan (list of two numpy arrays specifing verical and horizontal position of the can in pixels)
    d = ptycho.Simulation(obj=obj, probe=probe, scan=scan)

    or

    # Initialises simulation object with specific values of obj and probe and scan, missing parameters are filled with default values
    d = ptycho.Simulation(obj_info={'num_phot_max':10**3}, probe_info={'type':'Gauss','sigma_pix':(50,30)}, scan_info={'type':'spiral','n_scans':77}) # Initialises simulation object with specific values of obj and probe and scan, missing parameters are filled with default values

    d.make_data() # creates obj, probe, scan and ptychographic dataset
    d.make_obj() # creates obj only
    d.make_probe() # creates probe only
    d.print_info() # prints all the parameters of the simulation

    or

    # Only simulate probe
    d = ptycho.Simulation(probe_info={'type':'focus', 'aperture':(60e-6,200e-6), 'focal_length': 8e-2, 'defocus': 0},
                          data_info={'wavelength':1.5e-10, 'detector_distance': 1, 'detector_pixel_size': 55e-6})
    d.make_probe() # creates probe only


    Note that tne created data, object or probe are centered in the generated arrays.
    """

    def __init__(self, obj=None, obj_info=None, probe=None, probe_info=None, scan=None, scan_info=None, data_info=None,
                 verbose=1, prec='double'):
        """
        obj_info: dictionary with   obj_info['type'] = ('real','real_imag','ampl_phase','flat','random') # type of the object
                                    obj_info['phase_stretch'] = ps #  specify the stretch of the phase image (default is ps=2pi)
                                    obj_info['range'] = 0,1,0,2*pi #  range for random object with amplitude range then phase range in radians this
                                                                      supersedes obj_info['phase_stretch']
                                    obj_info['alpha_win'] = a # ratio of the obj edge to be dimmed down by tukeywin2D. This is to artefact from sharp edges of the object.
                                    defaults set in update_default_obj

        obj: specific value of obj can be passed (2D complex numpy array). obj_info can be passed as empty dictionary (default).

        probe_info: dictionary with probe_info['type'] = ('flat', 'Gauss', 'sinc', 'focus', 'near_field') # type of the probe
                                    probe_info['size'] = (sizex,sizey) # size of the probe in pixels (for 'Gauss', 'sinc')
                                    probe_info['sigma_pix'] = value # sigma (in pixels) for 'Gauss', 'sinc' probes
                                    probe_info['aperture'] = (widthx,widthy) # width of the rectangular aperture (before focusing or near field propagation) in meters
                                                                             # Alternatively can be a tuple with a single value (circular aperture)
                                    probe_info['focal_length'] = focal_length: for 'focused aperture', focal length of optics (in meters)
                                    probe_info['defocus'] = defocus: in meters, used for near field, or to defocus any other probe
                                    defaults set in update_default_probe

        probe: specific value of probe can be passed (2D complex numpy array). probe_info can be passed as empty dicitionary (default).

        scan_info: dictionary with  scan_info['type'] = ('spiral' ,'raster', 'custom') # type of the scan
                                    scan_info['scan_step_pix'] = value # step in the scan in [nm]
                                    scan_info['n_scans'] = value # number of scan positions
                                    scan_info['integer_values'] = True # Require integer positions
                                    scan_info['x']=value  # pixel coordinates for a custom scan (same for y)
                                    defaults set in update_default_scan

        scan: specific value of scan can be passed (list/tuple of two arrays (posx, posy)). scan_info can be passed as empty dictionary (default).

        data_info: dictionary with data_info['pix_size_direct_nm'] = value # pixels size in the object space [nm]
                                    data_info['nb_photons_per_frame'] = value # average nb photons per frame
                                    data_info['bg'] = value # uniform background added to each frame
                                    data_info['beam_stop_transparency'] = value
                                    data_info['noise'] = 'poisson' will create Poisson noise in the data. None will make noise-free data.
                                    data_info['wavelength'] = wavelength in meters. Mandatory for probe_info['type'] == 'focused aperture'
                                    data_info['detector_distance'] = detector distance in meters
                                    data_info['detector_pixel_size'] = detector pixel size in meters
                                    data_info['near_field'] = if True, uses near field propagation
                                    defaults set in update_default_data

        """
        if prec == 'single':  # precision of the simulation
            self.DTYPE_REAL = np.float32  # dtype of the real arrays
            self.DTYPE_CPLX = np.complex64  # dtype of the complex arrays
        elif prec == 'double':
            self.DTYPE_REAL = np.float64
            self.DTYPE_CPLX = np.complex128
        # print('Using %s precision for simulation.' % prec)

        if obj_info is None:
            obj_info = {}
        if probe_info is None:
            probe_info = {}
        if scan_info is None:
            scan_info = {}
        if data_info is None:
            data_info = {}
        else:
            if 'detector_dist' in data_info and 'detector_distance' not in data_info:
                warnings.warn("'detector_dist' is deprecated, use 'detector_distance'", DeprecationWarning)
                data_info['detector_distance'] = data_info['detector_dist']

        self.obj = Im(obj, obj_info)
        self.probe = Im(probe, probe_info)
        self.scan = Im(scan, scan_info)
        self.data_info = data_info
        self.verbose = verbose
        if 'wavelength' in data_info.keys():
            self.wavelength = data_info['wavelength']
        else:
            self.wavelength = None

    def update_default_data(self):
        # print('Updating defaults values for simulation.')
        default_data_info = {'pix_size_direct_nm': 10,
                             'num_phot_max': None,  # Deprecated
                             'nb_photons_per_frame': 1e8,
                             'bg': 0,
                             'beam_stop_transparency': 0,
                             'noise': 'poisson'}

        default_data_info.update(self.data_info)
        self.data_info = default_data_info.copy()

    def update_default_scan(self):
        # print('Updating defaults values for scan.')
        default_scan_info = {'type': 'spiral',
                             'scan_step_pix': 30,
                             'n_scans': 50,
                             'integer_values': True}
        default_scan_info.update(self.scan.info)
        self.scan.info = default_scan_info.copy()

    def update_default_obj(self):
        # print('Updating defaults values for object.')
        default_obj_info = {'type': 'ampl_phase'}
        default_obj_info.update(self.obj.info)
        self.obj.info = default_obj_info.copy()

    def update_default_probe(self):
        # print('Updating defaults values for probe.')
        default_probe_info = {'type': 'gauss',
                              'shape': (256, 256),
                              'sigma_pix': (50, 50),  # in pixels
                              'rotation': 0}
        default_probe_info.update(self.probe.info)
        self.probe.info = default_probe_info.copy()

    def print_info(self):
        print("Parameters of the simulation:")
        print("Data info: %s" % self.data_info)
        print("Scan info: %s" % self.scan.info)
        print("Object info: %s" % self.obj.info)
        print("Probe info: %s" % self.probe.info)

    def make_data(self):

        self.update_default_data()

        if self.obj.values is None:
            self.make_obj()

        if self.probe.values is None:
            self.make_probe()

        if self.scan.values is None:
            self.make_scan()

        self.obj.make_even()
        self.probe.make_even()

        posx, posy = self.scan.values
        posy_max, posx_max = np.ceil(abs(posy).max()), np.ceil(abs(posx).max())
        n = len(posx)  # number of scan positions

        if self.verbose:
            print("Simulating ptychographic data [%d frames]." % (n))

        s_v = self.probe.values.shape

        # TODO: allow different x and y size
        assert s_v[0] == s_v[1]

        wavelength = self.data_info['wavelength']
        detector_distance = self.data_info['detector_distance']
        pixel_size_detector = self.data_info['detector_pixel_size']
        if 'near_field' in self.data_info:
            near_field = self.data_info['near_field']
            pixel_size_object = pixel_size_detector
        else:
            near_field = False
            pixel_size_object = wavelength * detector_distance / (s_v[0] * pixel_size_detector)

        if 'rebin_factor' in self.data_info:
            rf = self.data_info['rebin_factor']
        else:
            rf = 1
        s_a = int(s_v[0] // rf), int(s_v[1] // rf)

        # Make the real object, zoom to fill if necessary
        nyo, nxo = calc_obj_shape(posx, posy, probe_shape=self.probe.values.shape)
        self.make_obj_true((nyo, nxo))

        self.psi = Im(np.zeros((n, s_v[0], s_v[1]), dtype=self.DTYPE_CPLX))
        intensity = np.ones((n, s_a[0], s_a[1]), dtype=self.DTYPE_REAL)

        if 'beam_stop_radius' in self.data_info:
            if 'beam_stop_transparency' not in self.data_info:
                self.data_info['beam_stop_transparency'] = 0
            if self.verbose: print("Beam stop with transparency %s" % self.data_info['beam_stop_transparency'])
            beam_stop_tmp = disc(self.probe.values.shape, self.data_info['beam_stop_radius'])
            beam_stop = self.data_info['beam_stop_transparency'] * (beam_stop_tmp) + (1. - beam_stop_tmp)
        else:
            beam_stop = np.ones((s_v[0], s_v[1]), dtype=self.DTYPE_REAL)

        nb_photons_per_frame = self.data_info['nb_photons_per_frame']

        # Use Ptycho code for simulation
        pd = ptycho.PtychoData(iobs=intensity, positions=(posx * pixel_size_object, posy * pixel_size_object),
                               detector_distance=detector_distance, wavelength=wavelength, near_field=near_field,
                               mask=np.isclose(beam_stop, 0), pixel_size_detector=pixel_size_detector)
        p = ptycho.Ptycho(probe=self.probe.values, obj=self.obj.values, data=pd)
        p = ptycho.Calc2Obs(nb_photons_per_frame=nb_photons_per_frame,
                            poisson_noise=(self.data_info['noise'] == 'poisson')) * p
        intensity = np.fft.fftshift(p.data.iobs, axes=(1, 2)) * beam_stop
        if rf > 1:  # rebin_factor
            intensity = rebin(intensity, (1, rf, rf), scale="sum")
            if self.verbose:
                print("\nBinning data by %s" % rf)

        if self.verbose:
            print("\n")

        if self.data_info['num_phot_max'] is not None:
            intensity /= intensity.max()
            intensity *= self.data_info['num_phot_max']
        intensity += self.data_info['bg']

        self.amplitude = Im(np.sqrt(intensity))

        if self.verbose:
            self.print_info()

    def make_obj_true(self, obj_shape):
        """
        Enlarge object according to real object shape. Replace previous object.
        :param obj_shape: maximum absolute displacement along x-axis
        :return:
        """
        o_s = self.obj.values.shape
        if self.obj.info['type'].lower() == 'siemens':
            mxy = max(obj_shape)
            tmp = siemens_star(dsize=mxy, nb_rays=36, r_max=0.9 * mxy, nb_rings=8,
                               cheese_hole_spiral_period=40, cheese_holes_nb=500 * (mxy // 500) ** 2,
                               cheese_hole_max_radius=6)[:obj_shape[0], :obj_shape[1]]
            # Make this a transmission object
            tmp = np.exp((-0.05 + 1j) * tmp).astype(self.obj.values.dtype)
        else:
            if True:
                # Zoom the object to fill the area
                oa, op = np.abs(self.obj.values), np.angle(self.obj.values)
                # Zoom image
                tmpa = zoom(oa, (obj_shape[0] / o_s[0], obj_shape[1] / o_s[1]), order=0)
                tmpp = zoom(op, (obj_shape[0] / o_s[0], obj_shape[1] / o_s[1]), order=0)
                tmp = (tmpa * np.exp(1j * tmpp)).astype(self.obj.values.dtype)
            else:
                # Tile the object to fill the area - deactivated as the original image is 512 pixels, so
                # a size likely similar to the probe, so may produce weird periodic artefacts
                ny, nx = obj_shape
                ny0, nx0 = self.obj.values.shape
                tmp = np.tile(self.obj.values, (int(np.ceil(ny / ny0)), int(np.ceil(nx / nx0))))
                ny1, nx1 = tmp.shape
                tmp = tmp[ny1 // 2 - ny // 2:ny1 // 2 - ny // 2 + ny,
                      nx1 // 2 - nx // 2:nx1 // 2 - nx // 2 + nx]

        if 'alpha_win' in self.obj.info:
            s = tmp.shape
            mi = abs(tmp).min()
            w = tukeywin2D(s, self.obj.info['alpha_win'])
            tmp = mi + w * (tmp - mi)  # tuckey window to smooth edges (alpha_win)

        self.obj = Im(tmp, self.obj.info)

    def make_obj(self):
        self.update_default_obj()
        info = self.obj.info
        obj_type = info['type'].lower()
        if self.verbose:
            print("Simulating object: %s" % obj_type)

        if obj_type == 'custom':
            obj = self.obj.values

        elif ('ampl' in obj_type) and ('phase' in obj_type):
            im0 = np.flipud(self.DTYPE_REAL(get_img(0)))
            im1 = np.flipud(self.DTYPE_REAL(get_img(1)))
            # Stretch the phase to interval (-phase_stretch/2, +phase_stretch/2)
            phase0 = im1 - im1.min()
            ps = 2 * np.pi
            phase_stretch = ps * phase0 / self.DTYPE_REAL(phase0.max()) - ps / 2.
            # Let's limit the amplitude extent to 0.8-1
            im0 = 1.25 ** (-im0.astype(np.float32) / im0.max())
            obj = im0 * np.exp(1j * phase_stretch)

        elif ('real' in obj_type) and ('imag' in obj_type):
            im0 = np.flipud(self.DTYPE_REAL(get_img(0)))
            im1 = np.flipud(self.DTYPE_REAL(get_img(1)))
            obj = im0 + 1j * im1

        elif obj_type.lower() == 'random':
            if 'range' not in info.keys():
                if 'phase_stretch' in info.keys():
                    dp = info['phase_stretch']
                    info['range'] = (0, 1, 0, dp)
                else:
                    info['range'] = (0, 1, 0, 2 * pi)
            s = info['range']
            if len(s) == 2:
                a0, a1 = 0, 1
                p0, p1 = s[0], s[1]
            else:
                a0, a1 = s[0], s[1]
                p0, p1 = s[2], s[3]
            s = info['shape']
            rand_phase = np.random.uniform(p0, p1, s)
            obj = np.random.uniform(a0, a1, s) * np.exp(1j * rand_phase)

        elif obj_type.lower() == 'flat':
            obj = np.ones(info['shape']).astype(self.DTYPE_CPLX)

        elif obj_type.lower() == 'siemens':
            obj = siemens_star(dsize=512, nb_rays=36, r_max=450, nb_rings=8, cheese_hole_spiral_period=200)
        elif obj_type.lower() == 'logo':
            fig = Figure(figsize=(6, 4), dpi=128)
            canvas = FigureCanvas(fig)
            ax = fig.subplots()
            t = ax.text(0.5, 0.5, "PyNX", fontsize=140, fontweight='heavy', ha='center', va='center')
            ax.axis('off')
            canvas.draw()
            a = np.array(canvas.renderer.buffer_rgba())[:, :, 0]

            fig = Figure(figsize=(6, 4), dpi=128)
            canvas = FigureCanvas(fig)
            ax = fig.subplots()
            t = ax.text(0.5, 0.5, "PyNX", fontsize=150, fontweight='heavy', ha='center', va='center')
            ax.axis('off')
            canvas.draw()
            ph = np.array(canvas.renderer.buffer_rgba())[:, :, 0]
            ph = (ph.max() - ph) / ph.max()

            obj = np.flipud((a.max() - a) * np.exp(-1j * ph).astype(self.DTYPE_CPLX))

        else:
            msg = "Unknown object type:", self.obj.info['type']
            raise NameError(msg)

        if 'phase_stretch' in info and not 'range' in info:
            phase = np.angle(obj)
            phase0 = phase - phase.min()
            if phase0.any():
                ps = info['phase_stretch']
                phase_stretch = ps * phase0 / phase0.max() - ps / 2.
                obj = abs(obj) * np.exp(1j * phase_stretch)

        if 'ampl_range' in info and not 'range' in info:
            a0, a1 = info['ampl_range']
            obj = (a0 + a1 * np.abs(obj) / np.abs(obj).max()) * np.exp(1j * np.angle(obj))

        if 'alpha_win' in info:
            s = obj.shape
            mi = abs(obj).min()
            w = tukeywin2D(s, info['alpha_win'])
            obj = mi + w * (obj - mi)  # tuckey window to smooth edges (alpha_win)

        self.obj.values = self.DTYPE_CPLX(obj)

    def make_probe(self):
        """
        Simulates the beam.
        """
        self.update_default_probe()
        info = self.probe.info
        if self.verbose:
            print("Simulating probe: %s" % info['type'])
        probe_type = info['type'].lower()

        if probe_type == 'custom':
            pass
        elif probe_type == 'flat':
            self.probe.values = self.DTYPE_CPLX(np.ones(info['shape']))

        elif probe_type == 'gauss':
            self.probe.values = self.DTYPE_CPLX(
                gauss2D(info['shape'], mu=(0, 0), sigma=info['sigma_pix'], rotation=info['rotation']))

        elif probe_type == 'disc':
            self.probe.values = self.DTYPE_CPLX(disc(info['shape'], center=(0, 0), radius=info['radius_pix']))

        elif probe_type == 'sinc':
            self.probe.values = self.DTYPE_CPLX(sinc2d(info['shape'], center=(0, 0), width=info['width_pix']))

        elif probe_type == 'focus':
            focal_length = info['focal_length']
            nx, ny = self.probe.info['shape']
            assert nx == ny
            # Pixel size at focus
            pixel_size_focus = None
            if 'detector_distance' in self.data_info.keys() and 'detector_pixel_size' in self.data_info.keys():
                pixel_size_focus = self.wavelength * self.data_info['detector_distance'] / (
                        nx * self.data_info['detector_pixel_size'])
            elif 'pix_size_direct_nm' in self.data_info.keys():
                pixel_size_focus = self.data_info['pix_size_direct_nm'] * 1e-9
            assert pixel_size_focus is not None, "Could not determine pixel size at focus to calculate probe from focused aperture"

            assert self.wavelength is not None, "Wavelength is None !?"

            # Pixel size at aperture
            pixel_size_aperture = self.wavelength * focal_length / (nx * pixel_size_focus)
            d = np.zeros((nx, nx), dtype=np.complex64)
            self.w = wavefront.Wavefront(d=d, wavelength=self.wavelength, pixel_size=pixel_size_aperture)
            x, y = self.w.get_x_y()
            if len(info['aperture']) == 1:
                # Circular aperture
                radius = info['aperture'][0]
                r = np.sqrt(x ** 2 + y ** 2)
                self.w.set(r < radius)
            else:
                # Rectangular slit
                widthx, widthy = info['aperture']
                self.w.set((abs(y) < (widthy / 2)) * (abs(x) < (widthx / 2)))
            w = wavefront.PropagateFarField(focal_length, forward=False) * self.w
            if 'defocus' in info.keys():
                w = wavefront.PropagateNearField(info['defocus']) * self.w
            self.probe.values = self.w.get(shift=True)
            w = wavefront.FreePU() * w

        elif probe_type == 'near_field':
            defocus = self.probe.info['defocus']
            nx, ny = self.probe.info['shape']
            assert nx == ny
            # Pixel size at focus
            pixel_size_detector = self.data_info['detector_pixel_size']

            assert self.wavelength is not None, "Wavelength is None !?"

            # Pixel size at aperture
            d = np.zeros((nx, nx), dtype=np.complex64)
            self.w = wavefront.Wavefront(d=d, wavelength=self.wavelength, pixel_size=pixel_size_detector)
            x, y = self.w.get_x_y()
            if len(info['aperture']) == 1:
                # Circular aperture
                radius = info['aperture'][0]
                r = np.sqrt(x ** 2 + y ** 2)
                self.w.set(r < radius)
            else:
                # Rectangular slit
                widthx, widthy = info['aperture']
                self.w.set((abs(y) < (widthy / 2)) * (abs(x) < (widthx / 2)))
            w = wavefront.PropagateNearField(dz=defocus) * self.w
            self.probe.values = self.w.get(shift=True)
            w = wavefront.FreePU() * w

        else:
            msg = "Unknown probe type:", self.probe.info['type']
            raise NameError(msg)

    def make_scan(self):
        self.update_default_scan()
        info = self.scan.info
        if self.verbose:
            print("Simulating scan: %s" % info['type'])
        scan_type = info['type'].lower()
        if scan_type in ['rect', 'raster']:
            posy = np.arange(0, self.obj.values.shape[-2], info['scan_step_pix'])
            posx = np.arange(0, self.obj.values.shape[-1], info['scan_step_pix'])
            posy, posx = np.meshgrid(posy, posx, indexing='ij')
            posy, posx = posy.ravel(), posx.ravel()
            posx -= posx.mean()
            posy -= posy.mean()

        elif scan_type == 'spiral':
            posy, posx = spiral_archimedes(info['scan_step_pix'], info['n_scans'])
        elif scan_type in ['spiralfermat', 'spiral_fermat']:
            n = info['n_scans']
            dmax = 2 * np.sqrt(n / np.pi) * info['scan_step_pix']
            posy, posx = spiral_fermat(dmax, n)
        elif scan_type == 'custom':
            posx = info['x']
            posy = info['y']
        else:
            msg = "Unknown scan type:", self.scan.info['type']
            raise NameError(msg)

        if info['integer_values']:
            # print("Integer values of the scan!")
            self.scan.values = (posx.round(), posy.round())
        else:
            self.scan.values = (posx, posy)

    def show_illumination_sum(self, log_values=False):
        posx, posy = self.scan.values
        illum = np.zeros_like(self.obj.values)
        nix, niy = illum.shape
        npx, npy = self.probe.values.shape
        for sy, sx in zip(posy, posx):
            startx = nix / 2 + sx - npx / 2
            starty = niy / 2 + sy - npy / 2
            illum[startx:startx + npx, starty:starty + npy] += self.probe.values
        if log_values:
            im2show = np.log10(abs(illum))
        else:
            im2show = abs(illum)
        plt.imshow(im2show, interpolation='Nearest', extent=(0, illum.shape[1], 0, illum.shape[0]))
        plt.plot(posy + niy / 2, -posx + nix / 2, '*:k')  # for "plot" the vetical axis is inverted compared to "imshow"
        plt.plot(posy + niy / 2, -posx + nix / 2, 'xw')
        return illum
