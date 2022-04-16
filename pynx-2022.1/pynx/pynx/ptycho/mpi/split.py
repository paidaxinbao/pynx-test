# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2020-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


import timeit
import numpy as np
from scipy.fftpack import fftshift, fftfreq
from scipy.spatial import ConvexHull
from scipy import optimize
from skimage.draw import polygon

try:
    from skimage.registration import phase_cross_correlation as register_translation
except ImportError:
    from skimage.feature import register_translation
from ..ptycho import Ptycho as PtychoBase, save_obj_probe_cxi, algo_string
from .operator import *
from ...mpi import *

assert MPI is not None


class PtychoSplit(PtychoBase):
    """
    Class to for a ptycho object and data distributed over several MPI process.
    """

    def __init__(self, probe=None, obj=None, background=None, data=None, nb_frame_total=None, mpi_neighbour_xy=None,
                 mpi_comm=None):
        super(PtychoSplit, self).__init__(probe=probe, obj=obj, background=background, data=data,
                                          nb_frame_total=nb_frame_total)
        self.mpi_master = True
        self.mpic = mpi_comm
        if self.mpic is None:
            self.mpic = MPI.COMM_WORLD
        self.mpi_rank = self.mpic.Get_rank()
        self.mpi_size = self.mpic.Get_size()
        self.mpi_master = self.mpi_rank == 0
        # Stitched version of attributes in either Ptycho or PtychoData
        self.mpi_posx = None
        self.mpi_posy = None
        self.mpi_posx0 = None
        self.mpi_posy0 = None
        self.mpi_posx_c = None
        self.mpi_posy_c = None
        self.mpi_scale = None
        self.mpi_obj = None
        self.mpi_illum_norm = None
        self.mpi_obj_coords = None  # left, bottom, width, height in pixels
        self._mpi_scan_area_obj = None
        self._mpi_scan_area_points = None
        self.mpi_phase_ramp_dx = 0
        self.mpi_phase_ramp_dy = 0

        # List of overlapping points between different process for phase & position matching
        self.mpi_neighbour_xy = mpi_neighbour_xy  # Metric
        self.mpi_neighbour_ixy = None  # Pixel coordinates relative to local object

        # Timestamp counter for the last stitch of the object.
        self._stitch_timestamp_counter = 0

        self.init_mpi()

    def save_obj_probe_cxi(self, filename, sample_name=None, experiment_id=None, instrument=None, note=None,
                           process=None, append=False, shift_phase_zero=False, params=None,
                           remove_obj_phase_ramp=False):
        self.stitch()  # Will be done only if necessary
        # Gather the different object parts for output
        # Avoid gather since large objects can lead to error like:
        #  SystemError: Negative size passed to PyBytes_FromStringAndSize
        vprobe, vobj, villum = [], [], []
        if self.mpi_master:
            for i in range(0, self.mpi_size):
                if i == 0:
                    pr, o, n = self._probe, self._obj, self._obj_illumination
                else:
                    ix, iy, nxo, nyo = self.mpi_obj_coords[i]
                    o = np.empty((len(self._obj), nyo, nxo), dtype=np.complex64)
                    n = np.empty((nyo, nxo), dtype=np.float32)
                    pr = np.empty_like(self._probe)
                    self.mpic.Recv(pr, source=i, tag=150)
                    self.mpic.Recv(o, source=i, tag=151)
                    self.mpic.Recv(n, source=i, tag=152)
                vprobe.append(pr)
                vobj.append(o)
                villum.append(n)
        else:
            self.mpic.Send(self._probe, dest=0, tag=150)
            self.mpic.Send(self._obj, dest=0, tag=151)
            self.mpic.Send(self._obj_illumination, dest=0, tag=152)

        # Create an array with all positions information
        pos = np.empty((5, len(self.data.vidx)), dtype=np.float32)
        pos[0] = self.data.vidx
        pos[1] = self.data.posx + self.data.posx_c
        pos[2] = self.data.posy + self.data.posy_c
        pos[3] = self.data.posx - self.data.posx0
        pos[4] = self.data.posy - self.data.posy0
        vpos = self.mpic.gather(pos, root=0)

        posx, posy, posx0, posy0, posx_c, posy_c = self.get_mpi_pos()
        llk = self.get_llk(noise=None, norm=True)
        if not self.mpi_master:
            return

        extra_data = {}
        extra_data['nb_part'] = self.mpi_size
        for i in range(self.mpi_size):
            extra_data['Probe MPI#%02d' % i] = vprobe[i]
            extra_data['Obj MPI#%02d' % i] = vobj[i]
            extra_data['Illum MPI#%02d' % i] = villum[i]
            extra_data['Positions MPI#%02d' % i] = vpos[i]
            ix, iy, nx, ny = self.mpi_obj_coords[i]
            extra_data['ix iy MPI#%02d' % i] = (ix, iy)

        obj = self.mpi_obj
        probe = self.get_probe()
        if remove_obj_phase_ramp and (abs(self.mpi_phase_ramp_dx) + abs(self.mpi_phase_ramp_dy)) > 1e-5:
            ny, nx = probe.shape[-2:]
            nyo, nxo = obj.shape[-2:]
            y, x = np.meshgrid(fftshift(fftfreq(nyo, d=ny / nyo)).astype(np.float32),
                               fftshift(fftfreq(nxo, d=nx / nxo)).astype(np.float32), indexing='ij')
            obj = obj * np.exp(-2j * np.pi * (x * self.mpi_phase_ramp_dx + y * self.mpi_phase_ramp_dy))

        save_obj_probe_cxi(filename, obj, probe, self.data.wavelength,
                           self.data.detector_distance, self.data.pixel_size_detector, llk['poisson'],
                           llk['gaussian'], llk['euclidian'], llk['nb_photons_calc'],
                           self.history, self.data.pixel_size_object(), (posx, posy), (posx_c, posy_c), (posx0, posy0),
                           scale=self.data.scale, obj_zero_phase_mask=self._obj_zero_phase_mask,
                           scan_area_obj=self.get_mpi_scan_area_obj(), scan_area_probe=self.get_scan_area_probe(),
                           background=self._background, sample_name=sample_name, experiment_id=experiment_id,
                           instrument=instrument, note=note, process=process, append=append,
                           shift_phase_zero=shift_phase_zero, params=params, obj_illumination=self.mpi_illum_norm,
                           extra_data=extra_data, obj_phase_ramp=(self.mpi_phase_ramp_dx, self.mpi_phase_ramp_dy))

    def init_mpi(self):
        """
        Initialise MPI-specific attributes (only for the master, gathering info from other processes).
        :return: nothing
        """
        if self.mpi_master:
            self.mpi_posx = [self.data.posx]
            self.mpi_posy = [self.data.posy]
            self.mpi_posx0 = [self.data.posx0]
            self.mpi_posy0 = [self.data.posy0]
            self.mpi_posx_c = [self.data.posx_c]
            self.mpi_posy_c = [self.data.posy_c]
            self.mpi_scale = [self.data.scale]
            for i in range(1, self.mpi_size):
                v = self.mpi_posx, self.mpi_posy, self.mpi_posx0, self.mpi_posy0, self.mpi_posx_c, \
                    self.mpi_posy_c, self.mpi_scale
                for iv in range(len(v)):
                    v[iv].append(self.mpic.recv(source=i, tag=20 + iv))
            # print("PtychoStitch(): finished gathering positions and scale from all MPI process")
        else:
            v = self.data.posx, self.data.posy, self.data.posx0, self.data.posy0, self.data.posx_c, \
                self.data.posy_c, self.data.scale
            for iv in range(len(v)):
                self.mpic.send(v[iv], dest=0, tag=20 + iv)
        self.init_mpi_obj()

    def get_mpi_pos(self):
        """
        Synchronise and get the full list of positions
        :return: (posx, posy, posx0, posy0, posx_c, posy_c): the positions along x and y,
                 with current and original values, relative to the center given as _c
        """
        vimgn = self.mpic.gather(self.data.vidx, root=0)
        vpos = self.mpic.gather((self.data.posx, self.data.posy), root=0)
        vpos0 = self.mpic.gather((self.data.posx0, self.data.posy0), root=0)
        if not self.mpi_master:
            return [None] * 6
        # Merge positions
        vposx, vposy, vposx0, vposy0, vnb = {}, {}, {}, {}, {}
        for i in range(self.mpi_size):
            for posx, posy, posx0, posy0, idx in zip(vpos[i][0], vpos[i][1], vpos0[i][0], vpos0[i][1], vimgn[i]):
                if idx not in vposx:
                    vposx[idx] = posx + self.mpi_posx_c[i]
                    vposy[idx] = posy + self.mpi_posy_c[i]
                    vposx0[idx] = posx0 + self.mpi_posx_c[i]
                    vposy0[idx] = posy0 + self.mpi_posy_c[i]
                    vnb[idx] = 1
                else:
                    vposx[idx] += posx + self.mpi_posx_c[i]
                    vposy[idx] += posy + self.mpi_posy_c[i]
                    vnb[idx] += 1
        posx, posy = np.empty(len(vposx), dtype=np.float32), np.empty(len(vposx), dtype=np.float32)
        posx0, posy0 = np.empty(len(vposx), dtype=np.float32), np.empty(len(vposx), dtype=np.float32)
        k = sorted(vposx.keys())
        for i in range(len(k)):
            posx[i] = vposx[k[i]] / vnb[k[i]]
            posy[i] = vposy[k[i]] / vnb[k[i]]
            posx0[i] = vposx0[k[i]]
            posy0[i] = vposy0[k[i]]

        px, py = self.data.pixel_size_object()
        ix, iy, nx, ny = self.mpi_obj_coords[0]
        nyo, nxo = self.mpi_obj.shape[-2:]
        posx_c = self.data.posx_c - px * ((nx - nxo) / 2 + ix)
        posy_c = self.data.posy_c - py * ((ny - nyo) / 2 + iy)

        # for i in range(len(self.mpi_obj_coords)):
        #     ix, iy, nx, ny = self.mpi_obj_coords[i]
        #     x0 = self.mpi_posx_c[i] - px * ((nx - nxo) / 2 + ix)
        #     y0 = self.mpi_posy_c[i] - py * ((ny - nyo) / 2 + iy)
        #     print("get_mpi_pos: %10.3f %10.3f" % (x0 * 1e6, y0 * 1e6))

        return posx - posx_c, posy - posy_c, posx0 - posx_c, posy0 - posy_c, posx_c, posy_c

    def get_mpi_obj_coord(self):
        """
        Get the object coordinates
        :return: a tuple of two arrays corresponding to the x (columns) and y (rows coordinates)
        """
        px, py = self.data.pixel_size_object()
        ix, iy, nx, ny = self.mpi_obj_coords[0]
        nyo, nxo = self.mpi_obj.shape[-2:]
        posx_c = self.data.posx_c - px * ((nx - nxo) / 2 + ix)
        posy_c = self.data.posy_c - py * ((ny - nyo) / 2 + iy)
        xc = np.arange(nxo, dtype=np.float32) * px + posx_c - nxo * px / 2
        yc = np.arange(nyo, dtype=np.float32) * py + posy_c - nyo * py / 2
        return xc, yc

    def calc_mpi_scan_area(self):
        """
        Compute the scan area for the object and probe, using scipy ConvexHull. The scan area for the object is
        augmented by twice the average distance between scan positions for a more realistic estimation.
        scan_area_points is also computed, corresponding to the outline of the scanned area.

        :return: Nothing. mpi_scan_area_points and mpi_scan_area_obj are updated, the latter
                 as a 2D arrays with the same shape as the object, with False outside the scan area and True inside.
        """
        if not self.mpi_master:
            return
        px, py = self.data.pixel_size_object()
        x, y = self.mpi_posx[0] + self.mpi_posx_c[0], self.mpi_posy[0] + self.mpi_posy_c[0]
        # There will be some duplicate points (overlap) but that does not matter
        for i in range(self.mpi_size):
            x = np.append(x, self.mpi_posx[i] + self.mpi_posx_c[i])
            y = np.append(y, self.mpi_posy[i] + self.mpi_posy_c[i])
        # If there are too many points, reduce to 500.
        if len(x) > 1000:
            x = x[::len(x) // 500]
            y = y[::len(y) // 500]

        # Convert x, y metric to pixel coordinates relative to the origin (top, left) corner
        ix, iy, nx, ny = self.mpi_obj_coords[0]
        x0, y0 = self.mpi_posx_c[0] - (nx // 2 + ix) * px, self.mpi_posy_c[0] - (ny // 2 + iy) * py
        points = np.array([((x - x0) / px, (y - y0) / py) for x, y in zip(x, y)])

        c = ConvexHull(points)
        vx = np.array([points[i, 0] for i in c.vertices])  # + [points[c.vertices[0], 0]], dtype=np.float32)
        vy = np.array([points[i, 1] for i in c.vertices])  # + [points[c.vertices[0], 1]], dtype=np.float32)
        # Try to expand scan area by the average distance between points
        try:
            # Estimated average distance between points with an hexagonal model
            w = 4 / 3 / np.sqrt(3) * np.sqrt(c.volume / x.size)
            xc = vx.mean()
            yc = vy.mean()
            # Expand scan area from center by 1
            d = np.sqrt((vx - xc) ** 2 + (vy - yc) ** 2)
            vx = xc + (vx - xc) * (d + w) / d
            vy = yc + (vy - yc) * (d + w) / d
        except:
            # c.volume only supported in scipy >=0.17 (2016/02)
            pass
        # print("calc_scan_area: scan area = %8g pixels^2, center @(%6.1f, %6.1f), <d>=%6.2f)"%(c.volume, xc, yc, w))
        # Object

        rr, cc = polygon(vy, vx, self.mpi_obj.shape[-2:])
        self._mpi_scan_area_obj = np.zeros(self.mpi_obj.shape[-2:], dtype=np.bool)
        self._mpi_scan_area_obj[rr, cc] = True

        # scan_area_points are relative to the center of the object
        self._mpi_scan_area_points = vx - self.mpi_obj.shape[-1] // 2, vy - self.mpi_obj.shape[-2] // 2

    def get_mpi_scan_area_obj(self):
        """
        Return the mpi_scan_area_obj. It is computed if necessary.
        :return: scan_area_obj, a 2D array with the object shape, True inside the
                 area scanned, and False outside
        """
        if self._mpi_scan_area_obj is None:
            self.calc_mpi_scan_area()
        return self._mpi_scan_area_obj

    def get_mpi_scan_area_points(self):
        """
        Return the mpi_scan_area_points.
        It is computed if necessary.
        :return: scan_area_points, a tuple (vx, vy) of polygon points delimiting the scan area
        """
        if self._mpi_scan_area_obj is None:
            self.calc_mpi_scan_area()
        return self._mpi_scan_area_points

    def init_mpi_obj(self):
        """
        Initialise the (empty) array for the stitched object
        :return: nothing. self.mpi_obj, self.mpi_obj_xy are created, self.mpi_obj_coords is
        """
        # Get the size of the different parts of the object, and their corner positions relative to master object
        nobj, ny0, nx0 = self.get_obj().shape
        if self.mpi_master:
            self.mpi_obj_coords = [[0, 0, nx0, ny0]]
            px, py = self.data.pixel_size_object()
            for i in range(1, self.mpi_size):
                ny, nx = self.mpic.recv(source=i, tag=30)
                # Pixel shifts of the different object parts
                dxp = int(np.round((self.mpi_posx_c[i] - self.mpi_posx_c[0]) / px))
                dyp = int(np.round((self.mpi_posy_c[i] - self.mpi_posy_c[0]) / py))
                # print(i, dxp, dyp, nx - nx0, ny - ny0)
                self.mpi_obj_coords.append([dxp + (nx0 - nx) // 2, dyp + (ny0 - ny) // 2, nx, ny])
            # Compute stitched object shape
            cx0m = min(v[0] for v in self.mpi_obj_coords)
            cy0m = min(v[1] for v in self.mpi_obj_coords)
            cx1m = max(v[0] + v[2] for v in self.mpi_obj_coords)
            cy1m = max(v[1] + v[3] for v in self.mpi_obj_coords)
            nxo, nyo = cx1m - cx0m, cy1m - cy0m
            self.mpi_obj = np.empty((nobj, nyo, nxo), dtype=np.complex64)
            self.mpi_illum_norm = np.empty((nyo, nxo), dtype=np.float32)
            print("PtychoSplit.init_mpi_obj(): final object size:", self.mpi_obj.shape)
            for i in range(0, self.mpi_size):
                self.mpi_obj_coords[i][0] -= cx0m
                self.mpi_obj_coords[i][1] -= cy0m
        else:
            self.mpic.send(self.get_obj().shape[-2:], dest=0, tag=30)

        # Compute the coordinates of neighbouring points which will be used for synchronisation
        self.mpi_neighbour_ixy = {}
        px, py = self.data.pixel_size_object()
        nyo, nxo = self.get_obj().shape[-2:]
        x0, y0 = self.data.posx_c - nxo // 2 * px, self.data.posy_c - nyo // 2 * px
        if self.mpi_neighbour_xy is not None:
            for k, v in self.mpi_neighbour_xy.items():
                x = np.array(v[0], dtype=np.float32)
                y = np.array(v[1], dtype=np.float32)
                self.mpi_neighbour_ixy[k] = np.round((x - x0) / px).astype(np.int32), \
                                            np.round((y - y0) / py).astype(np.int32)
                # print("MPI%2d mpi_neighbour_ixy[%d]:" % (self.mpi_rank, k), self.mpi_neighbour_ixy[k],
                #       np.alltrue(self.mpi_neighbour_ixy[k][0] < nxo), np.alltrue(self.mpi_neighbour_ixy[k][1] < nyo))

    def sync(self, nbpix=16, verbose=False):
        """
        Synchronise the phase and relative positions of all parts of the object
        :param nbpix: half-size of the areas shared between processes for synchronisation
        :param verbose: if True, print timings & info
        :return: nothing
        """
        t0 = timeit.default_timer()
        self.from_pu()
        t1 = timeit.default_timer()
        if verbose:
            self.print("sync: dt[%20s]=%6.2fs" % ("from_pu", t1 - t0))

        # First register probe translations (done on the first mode)
        # TODO: sub-pixel alignment ?
        if self.mpi_master:
            v = (abs(self._probe) ** 2).sum(axis=0)
            v1 = np.empty_like(v)
            for i in range(1, self.mpi_size):
                self.mpic.Recv(v1, source=i, tag=200)
                r = register_translation(v, v1, upsample_factor=1, space='real')
                self.mpic.send(r, dest=i, tag=201)
        else:
            self.mpic.Send((abs(self._probe) ** 2).sum(axis=0), dest=0, tag=200)
            # self.mpic.send(self._probe[0], dest=0, tag=200)
            r = self.mpic.recv(source=0, tag=201)
            if max(abs(r[0][0]), abs(r[0][1])) >= 1:
                # !!! shifts must be integer or process can *silently* hang !
                self._probe = np.roll(self._probe, np.round(r[0]).astype(np.int32), axis=(-2, -1))
                self._obj = np.roll(self._obj, np.round(r[0]).astype(np.int32), axis=(-2, -1))
                # print("MPI #%02d: shift=" % self.mpi_rank, r[0])

        t2 = timeit.default_timer()
        if verbose:
            self.print("sync: dt[%20s]=%6.2fs" % ("probe_translation", t2 - t1))

        # Check that positions shifts are coherent
        vpos = self.mpic.gather((self.data.vidx, self.data.posx + self.data.posx_c,
                                 self.data.posy + self.data.posy_c), root=0)

        t3 = timeit.default_timer()
        if verbose:
            self.print("sync: dt[%20s]=%6.2fs" % ("pos_gather", t3 - t2))

        if self.mpi_master:
            # Get shift of positions
            vdxy = {}
            nb = 0
            for i in range(self.mpi_size):
                ii = vpos[i][0]
                xi, yi = vpos[i][1], vpos[i][2]
                for j in range(i + 1, self.mpi_size):
                    ij = vpos[j][0]
                    comm, icomm, jcomm = np.intersect1d(ii, ij, return_indices=True, assume_unique=True)
                    if len(comm) > 4:
                        xj, yj = vpos[j][1], vpos[j][2]
                        if i not in vdxy:
                            vdxy[i] = {}
                        dx, dy = xj[jcomm] - xi[icomm], yj[jcomm] - yi[icomm]
                        if True:
                            # Remove outliers to be more robust
                            idx = np.argsort(dx ** 2 + dy ** 2)
                            vdxy[i][j] = dx[idx[2:-2]], dy[idx[2:-2]]
                        else:
                            vdxy[i][j] = dx, dy
                        nb += len(vdxy[i][j][0])
                        if False:
                            dx = np.median(xi[icomm] - xj[jcomm])
                            dy = np.median(yi[icomm] - yj[jcomm])
                            print("%d common points between %d and %d. Median shift (nm): (%6.2f, %6.2f)" % (
                                len(comm), i, j, dx * 1e9, dy * 1e9))
            # Optimise correction shifts
            par = np.zeros(2 * (self.mpi_size - 1), dtype=np.float32)
            if nb > 10:
                scale = (abs(vpos[0][1]) + abs(vpos[0][2])).mean()

                def min_dxy(p, vdxy, scale):
                    d = 0
                    for i in vdxy.keys():
                        for j in vdxy[i].keys():
                            dx, dy = vdxy[i][j]
                            dx = dx / scale
                            dy = dy / scale
                            if i > 0:
                                dx -= p[(i - 1) * 2]
                                dy -= p[(i - 1) * 2 + 1]
                            dx += p[(j - 1) * 2]
                            dy += p[(j - 1) * 2 + 1]
                            d += (dx ** 2 + dy ** 2).sum()
                    return d

                res = optimize.minimize(min_dxy, par, args=(vdxy, scale), method='Powell')
                par = np.array(res.x) * scale

            # TODO: check if there is a global drift (in root object) & correct

            # print(par * 1e9)
            for i in range(1, self.mpi_size):
                dx, dy = par[(i - 1) * 2], par[(i - 1) * 2 + 1]
                self.mpic.send((dx, dy), dest=i, tag=210)
                # print("Global pos shift (MPI #%02d): (%8.2f, %8.2f) nm" % (i, dx * 1e9, dy * 1e9))
        else:
            # get back the shifts, shift positions & object
            dx, dy = self.mpic.recv(source=0, tag=210)
            # print("Got pos shift    (MPI #%02d): (%8.2f, %8.2f) nm" % (self.mpi_rank, dx * 1e9, dy * 1e9))
            px, py = self.data.pixel_size_object()
            dix, diy = np.int(np.round(dx / px)), np.int(np.round(dy / py))
            if (abs(dix) + abs(diy)) > 0:
                self._obj = np.roll(self._obj, (dix, diy), axis=(-1, -2))
                # TODO: Should we round the pos shifts at pixel resolution ?
                self.data.posx += dx
                self.data.posy += dy

        t4 = timeit.default_timer()
        if verbose:
            self.print("sync: dt[%20s]=%6.2fs" % ("pos_optim", t4 - t3))

        # Loop over all overlapping areas for phase synchronisation

        # s = "MPI #%02d:" % self.mpi_rank
        # for k, v in self.mpi_neighbour_ixy.items():
        #     if k > self.mpi_rank:
        #         s += " %2d -> %2d [%2d]" % (self.mpi_rank, k, len(v[0]))
        #     else:
        #         s += " %2d -> %2d [%2d]" % (k, self.mpi_rank, len(v[0]))
        #     for i in range(min(3, len(v[0]))):
        #         s += " (%4d, %4d)" % (v[0][i], v[1][i])
        # print(s)

        # We need a minimum of overlap to synchronise two regions (?)
        min_neighbour = max([len(v[0]) for v in self.mpi_neighbour_ixy.values()]) // 4
        overlap_cs = {}
        for k, v in self.mpi_neighbour_xy.items():
            ix, iy = self.mpi_neighbour_ixy[k]
            nb = len(ix)
            # if nb < min_neighbour:
            #     # Don't synchronise for too small overlap
            #     continue
            v = np.empty((nb, 2 * nbpix, 2 * nbpix), dtype=np.complex64)
            for i in range(nb):
                v[i] = self._obj[0, iy[i] - nbpix:iy[i] + nbpix, ix[i] - nbpix:ix[i] + nbpix]
            if k > self.mpi_rank:
                # Send data to process with a higher rank, they will shift accordingly
                self.mpic.Send(v, dest=k, tag=220)
            else:
                v1 = np.empty((nb, 2 * nbpix, 2 * nbpix), dtype=np.complex64)
                # Receive data from a lower-rank process, and shift
                self.mpic.Recv(v1, source=k, tag=220)
                # Correct phase factor, weighting by object square modulus
                x0, y0, x1, y1 = v.real, v.imag, v1.real, v1.imag
                a, b = x0 * x1 + y0 * y1, x1 * y0 - x0 * y1
                alpha = np.arctan2(b, a)

                overlap_cs[k] = (np.sqrt(a ** 2 + b ** 2) * np.exp(-1j * alpha)).sum()
        overlap_cs = self.mpic.gather(overlap_cs, root=0)

        t5 = timeit.default_timer()
        if verbose:
            self.print("sync: dt[%20s]=%6.2fs" % ("phase_overlap_gather", t5 - t4))

        if self.mpi_master:
            # print(overlap_cs)

            def min_phase(phi, overlap_cs, verbose=False):
                r = 0
                for k in range(len(overlap_cs)):
                    for l, v in overlap_cs[k].items():
                        phik = phi[k - 1]
                        phil = 0 if l == 0 else phi[l - 1]
                        r -= (np.exp(1j * (phik - phil)) * v).real
                if verbose:
                    print(r)
                return r

            sol = optimize.minimize(min_phase, np.zeros(self.mpi_size - 1), method='powell',
                                    args=(overlap_cs, False))
            phi = [0]
            if sol.x.ndim:
                phi += list(sol.x)
            else:
                phi.append(float(sol.x))
        else:
            phi = None
        phi = self.mpic.scatter(phi, root=0)
        # print("MPI #%02d:sync phase" % self.mpi_rank)
        if abs(phi) > 1e-3:
            self._obj *= np.exp(-1j * phi)

        t6 = timeit.default_timer()
        if verbose:
            self.print("sync: dt[%20s]=%6.2fs" % ("phase_sync", t6 - t5))
            self.print("sync: dt[%20s]=%6.2fs" % ("total", t6 - t0))

        self._timestamp_counter += 1
        self._cpu_timestamp_counter = self._timestamp_counter

    def stitch(self, sync=True, scatter=False, verbose=False):
        """
        Gather the different object parts and stitch them
        :param sync: if True (the default, sync the positions and phase of all parts)
        :param scatter: if True, the assembled object is shared with all processes
        :param verbose: if True, print info & timings.
        :return: nothing. self.mpi_obj and self.mpi_illum_norm are updated
        """
        if np.alltrue(np.array(self.mpic.allgather(self._stitch_timestamp_counter == self._timestamp_counter))):
            return
        t0 = timeit.default_timer()
        if sync:
            self.sync(verbose=verbose)
            t1 = timeit.default_timer()
            if verbose:
                self.print("stitch: dt[%18s]=%6.2fs" % ("sync", t1 - t0))
        else:
            self.from_pu()
            t1 = timeit.default_timer()
            if verbose:
                self.print("stitch: dt[%18s]=%6.2fs" % ("from_pu", t1 - t0))

        pr_scale = self.mpic.gather((abs(self._probe) ** 2).sum(), root=0)
        if self.mpi_master:
            # Compute probe relative scales.
            # The probe should be multiplied, and the object divided by pr_scale[i]
            pr_scale = np.sqrt(pr_scale[0] / np.array(pr_scale, dtype=np.float32))

            t2 = timeit.default_timer()
            if verbose:
                self.print("stitch: dt[%18s]=%6.2fs" % ("probe_scale", t2 - t1))

            # print(pr_scale)
            # Gather object parts
            nobj = len(self._obj)
            self.mpi_obj.fill(0)
            self.mpi_illum_norm.fill(0)
            for i in range(0, self.mpi_size):
                ix, iy, nx, ny = self.mpi_obj_coords[i]
                # Could we use gather instead ? No object can be too large for pickle
                if i == 0:
                    o = self._obj
                    n = self.get_illumination_obj()
                else:
                    o = np.empty((nobj, ny, nx), dtype=np.complex64)
                    n = np.empty((ny, nx), dtype=np.float32)
                    self.mpic.Recv(o, source=i, tag=101)
                    self.mpic.Recv(n, source=i, tag=102)
                # print("stitch(): part #%d (%4d, %4d, %4d, %4d)" % (i, ix, iy, nx, ny),
                #       self.mpi_obj[:, iy:iy + ny, ix:ix + nx].shape, o.shape, n.shape)
                self.mpi_obj[:, iy:iy + ny, ix:ix + nx] += o * n * pr_scale[i]
                self.mpi_illum_norm[iy:iy + ny, ix:ix + nx] += n * pr_scale[i] ** 2
            m = self.mpi_illum_norm.max()
            self.mpi_obj /= self.mpi_illum_norm + 1e-3 * m

            t3 = timeit.default_timer()
            if verbose:
                self.print("stitch: dt[%18s]=%6.2fs" % ("obj_stitch", t3 - t2))

            if scatter:
                for i in range(0, self.mpi_size):
                    ix, iy, nx, ny = self.mpi_obj_coords[i]
                    if i == 0:
                        self._obj[:] = self.mpi_obj[:, iy:iy + ny, ix:ix + nx]
                    else:
                        self.mpic.Send(self.mpi_obj[:, iy:iy + ny, ix:ix + nx] / pr_scale[i], dest=i, tag=111)

                t4 = timeit.default_timer()
                if verbose:
                    self.print("stitch: dt[%20s]=%6.2fs" % ("obj_scatter", t4 - t3))

        else:
            o, n = self._obj, self.get_illumination_obj()
            self.mpic.Send(o, dest=0, tag=101)
            self.mpic.Send(n, dest=0, tag=102)
            if scatter:
                self.mpic.Recv(self._obj, source=0, tag=111)

        t4 = timeit.default_timer()
        # Ideally we should gather the sum of icalc from all process, and get the correct shifts
        # from that, but this should be close enough, unless there are strong inhomogeneous shifts
        phase_ramp_dx = self.mpic.gather(self.data.phase_ramp_dx, root=0)
        phase_ramp_dy = self.mpic.gather(self.data.phase_ramp_dy, root=0)
        if self.mpi_master:
            self.mpi_phase_ramp_dx = np.array(phase_ramp_dx, dtype=np.float32).mean()
            self.mpi_phase_ramp_dy = np.array(phase_ramp_dy, dtype=np.float32).mean()

        t5 = timeit.default_timer()
        if verbose:
            self.print("stitch: dt[%18s]=%6.2fs" % ("phase_ramp", t5 - t4))
            self.print("stitch: dt[%18s]=%6.2fs" % ("total", t5 - t0))

        self._stitch_timestamp_counter = self._timestamp_counter

    def set_mpi_obj(self, obj):
        """
        Set the object and distribute the different parts to the MPI processes
        :param obj: the new object
        :return: nothing
        """
        self.from_pu()
        if self.mpi_master:
            nyo, nxo = obj.shape[-2:]
            # print("set_mpi_obj(): object size=", obj.shape[-2:])
            if obj.ndim == 2:
                self.mpi_obj = np.reshape(obj.astype(np.complex64), (1, nyo, nxo))
            else:
                self.mpi_obj = obj.astype(np.complex64)
            for i in range(self.mpi_size):
                ix, iy, nx, ny = self.mpi_obj_coords[i]
                o = self.mpi_obj[:, iy:iy + ny, ix:ix + nx]
                # print("set_mpi_obj(): sending part #%d (%4d, %4d, %4d, %4d)" % (i, ix, iy, nx, ny), o.shape, o.dtype)
                if i == 0:
                    self.set_obj(o)
                else:
                    self.mpic.Send(o.copy(), dest=i, tag=110)
        else:
            # print("set_mpi_obj(): receiving part #%d " % self.mpi_rank, self._obj.shape, self._obj.dtype)
            self.mpic.Recv(self._obj, source=0, tag=110)
            self.set_obj(self._obj)  # This'll take care of the timestamps counters

    def set_mpi_probe(self, probe):
        """
        Set the probe and broadcast it to the different MPI processes
        :param probe:
        :return:
        """

    def get_llk(self, noise=None, norm=True):
        """
        Get the log-likelihood.
        :param noise: noise model, either 'poisson', 'gaussian' or 'euclidian'.
                      If None, a dictionary is returned.
        :param norm: if True (the default), the LLK values are normalised
        :return: either a single LLK value, or a dictionary
        """
        vd = self.mpic.gather(super(PtychoSplit, self).get_llk(noise=None, norm=False), root=0)
        p, g, e, n, nph = 0, 0, 0, 0, 0
        if self.mpi_master:
            for d in vd:
                p += d['poisson']
                g += d['gaussian']
                e += d['euclidian']
                n += d['nb_obs']
                nph += d['nb_photons_calc']
            if norm:
                p, g, e = p / n, g / n, e / n
            if noise is None:
                return {'poisson': p, 'gaussian': g, 'euclidian': e, 'nb_photons_calc': nph, 'nb_obs': n}
            if 'poiss' in noise.lower():
                return p
            if 'gauss' in noise.lower():
                return g
            return e
        else:
            return super(PtychoSplit, self).get_llk(noise=noise, norm=norm)
