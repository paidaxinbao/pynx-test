# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2021-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['CDIViewer']

import os
import numpy as np
import h5py as h5
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from tornado.ioloop import PeriodicCallback
from IPython.core.display import display, HTML
import ipyvolume as ipv
import ipywidgets as widgets
import ipyfilechooser
from pynx.utils.plot_utils import complex2rgbalin


class CDIViewer(widgets.Box):
    """
    Widget to display 3D objects from CDI optimisation, loaded from a result CXI file
    or a mode file.

    This is a quick & dirty implementation but should be useful.
    """

    def __init__(self, cxi=None, html_width=None):
        """

        :param cxi: the CXI filename. Can also be the path to a directory, or a directly a 3D data array
        :param html_width: html width in %. If given, the width of the notebook will be
            changed to that value (e.g. full width with 100)
        """
        super(CDIViewer, self).__init__()

        if html_width is not None:
            display(HTML("<style>.container { width:%d%% !important; }</style>" % int(html_width)))

        # focus_label = widgets.Label(value='Focal distance (cm):')
        self.threshold = widgets.FloatSlider(value=5, min=0, max=20, step=0.02, description='Contour.',
                                             disabled=False, continuous_update=True, orientation='horizontal',
                                             readout=True, readout_format='.01f')
        self.toggle_phase = widgets.ToggleButtons(options=['Abs', 'Phase'], description='',  # , 'Grad'
                                                  disabled=False, value='Phase',
                                                  button_style='')  # 'success', 'info', 'warning', 'danger' or ''

        # self.toggle_phase = widgets.ToggleButton(value=True, description='Phase', tooltips='Color surface with phase')
        self.toggle_rotate = widgets.ToggleButton(value=False, description='Rotate', tooltips='Rotate')
        self.pcb_rotate = None
        hbox1 = widgets.HBox([self.toggle_phase, self.toggle_rotate])

        self.toggle_dark = widgets.ToggleButton(value=False, description='Dark', tooltips='Dark/Light theme')
        self.toggle_box = widgets.ToggleButton(value=True, description='Box', tooltips='Box ?')
        self.toggle_axes = widgets.ToggleButton(value=True, description='Axes', tooltips='Axes ?')
        hbox_toggle = widgets.HBox([self.toggle_dark, self.toggle_box, self.toggle_axes])

        self.colormap = widgets.Dropdown(
            options=['Cool', 'Gray', 'Gray_r', 'Hot', 'Hsv', 'Inferno', 'Jet', 'Plasma', 'Rainbow', 'Viridis'],
            value='Jet', description='Colors:', disabled=True)
        self.colormap_range = widgets.FloatRangeSlider(value=[20, 80],
                                                       min=0,
                                                       max=100,
                                                       step=1,
                                                       description='Range:',
                                                       disabled=False,
                                                       continuous_update=False,
                                                       orientation='horizontal',
                                                       readout=True,
                                                       # readout_format='.1f'
                                                       )
        self.toggle_plane = widgets.ToggleButton(value=False, description='Cut planes', tooltips='Cut plane')
        self.plane_text = widgets.Text(value="", description="", tooltips='Plane equation')
        hbox_plane = widgets.HBox([self.toggle_plane, self.plane_text])

        self.clipx = widgets.FloatSlider(value=1, min=-1, max=1, step=0.1, description='Plane Ux',
                                         disabled=False, continuous_update=False, orientation='horizontal',
                                         readout=True, readout_format='.01f')
        self.clipy = widgets.FloatSlider(value=1, min=-1, max=1, step=0.1, description='Plane Uy',
                                         disabled=False, continuous_update=False, orientation='horizontal',
                                         readout=True, readout_format='.01f')
        self.clipz = widgets.FloatSlider(value=1, min=-1, max=1, step=0.1, description='Plane Uz',
                                         disabled=False, continuous_update=False, orientation='horizontal',
                                         readout=True, readout_format='.01f')
        self.clipdist = widgets.FloatRangeSlider(value=[0, 100], min=0, max=100, step=0.5, description='Planes dist',
                                                 disabled=False, continuous_update=False, orientation='horizontal',
                                                 readout=True, readout_format='.1f')

        # self.toggle_mode = widgets.ToggleButtons(options=['Volume','X','Y','Z'])
        self.progress = widgets.IntProgress(value=10, min=0, max=10,
                                            description='Processing:',
                                            bar_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                            style={'bar_color': 'green'},
                                            orientation='horizontal')

        if cxi is not None and isinstance(cxi, str):
            if os.path.isfile(cxi):
                pth = os.path.split(cxi)[0]
            else:
                pth = cxi
        else:
            pth = os.getcwd()
        self.fc = ipyfilechooser.FileChooser(pth, filter_pattern=['*.cxi', '*.h5'])
        self.fc.register_callback(self.on_select_file)

        self.vbox = widgets.VBox([self.threshold, hbox1, hbox_toggle, self.colormap, self.colormap_range, hbox_plane,
                                  self.clipx, self.clipy, self.clipz, self.clipdist, self.progress, self.fc])

        self.output_view = widgets.Output()
        with self.output_view:
            self.fig = ipv.figure(width=900, height=600, controls_light=True)
            if cxi is not None:
                if isinstance(cxi, str):
                    if os.path.isfile(cxi):
                        self.change_file(cxi)
                elif isinstance(cxi, np.ndarray):
                    self.set_data(cxi)
            display(self.fig)

        self.threshold.observe(self.on_update_plot)
        self.toggle_phase.observe(self.on_change_type)
        self.colormap.observe(self.on_update_plot)
        self.colormap_range.observe(self.on_update_plot)
        self.clipx.observe(self.on_update_plot)
        self.clipy.observe(self.on_update_plot)
        self.clipz.observe(self.on_update_plot)
        self.clipdist.observe(self.on_update_plot)
        self.toggle_plane.observe(self.on_update_plot)

        self.toggle_dark.observe(self.on_update_style)
        self.toggle_box.observe(self.on_update_style)
        self.toggle_axes.observe(self.on_update_style)

        self.toggle_rotate.observe(self.on_animate)

        self.hbox = widgets.HBox([self.output_view, self.vbox])

        self.children = [self.hbox]

    def on_update_plot(self, v=None):
        """
        Update the plot according to parameters. The points are re-computed
        :param k: ignored
        :return:
        """
        if v is not None:
            if v['name'] != 'value':
                return
        self.progress.value = 7

        # See https://github.com/maartenbreddels/ipyvolume/issues/174 to support using normals

        # Unobserve as we disable/enable buttons and that triggers events
        try:
            self.clipx.unobserve(self.on_update_plot)
            self.clipy.unobserve(self.on_update_plot)
            self.clipz.unobserve(self.on_update_plot)
            self.clipdist.unobserve(self.on_update_plot)
        except:
            pass

        if self.toggle_plane.value:
            self.clipx.disabled = False
            self.clipy.disabled = False
            self.clipz.disabled = False
            self.clipdist.disabled = False
            # Cut volume with clipping plane
            uz, uy, ux = self.clipz.value, self.clipy.value, self.clipx.value
            u = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
            if np.isclose(u, 0):
                ux = 1
                u = 1

            nz, ny, nx = self.d.shape
            z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')

            # Compute maximum range of clip planes & fix dist range
            tmpz, tmpy, tmpx = np.where(abs(self.d) >= self.threshold.value)
            tmp = (tmpx * ux + tmpy * uy + tmpz * uz) / u
            tmpmin, tmpmax = tmp.min() - 1, tmp.max() + 1
            if tmpmax > self.clipdist.min:  # will throw an exception if min>max
                self.clipdist.max = tmpmax
                self.clipdist.min = tmpmin
            else:
                self.clipdist.min = tmpmin
                self.clipdist.max = tmpmax

            # Compute clipping mask
            c = ((x * ux + y * uy + z * uz) / u > self.clipdist.value[0]) * (
                ((x * ux + y * uy + z * uz) / u < self.clipdist.value[1]))
            self.plane_text.value = "%6.1f < (%4.2f*x %+4.2f*y %+4.2f*z) < %6.1f" % (
                self.clipdist.value[0], ux / u, uy / u, uz / u, self.clipdist.value[1])
        else:
            self.clipx.disabled = True
            self.clipy.disabled = True
            self.clipz.disabled = True
            self.clipdist.disabled = True
            self.plane_text.value = ""
            c = 1
        try:
            verts, faces, normals, values = marching_cubes(abs(self.d) * c, level=self.threshold.value, step_size=1)
            vals = self.rgi(verts)
            if self.toggle_phase.value == "Phase":
                self.colormap.disabled = True
                rgba = complex2rgbalin(vals)
                color = rgba[..., :3] / 256
            elif self.toggle_phase.value in ['Abs', 'log10(Abs)']:
                self.colormap.disabled = False
                cs = cm.ScalarMappable(
                    norm=Normalize(vmin=self.colormap_range.value[0], vmax=self.colormap_range.value[1]),
                    cmap=eval('cm.%s' % (self.colormap.value.lower())))
                color = cs.to_rgba(abs(vals))[..., :3]
            else:
                # TODO: Gradient
                gx, gy, gz = self.rgi_gx(verts), self.rgi_gy(verts), self.rgi_gz(verts)
                color = np.empty((len(vals), 3), dtype=np.float32)
                color[:, 0] = abs(gx)
                color[:, 1] = abs(gy)
                color[:, 2] = abs(gz)
                color *= 100
                self.color = color
            x, y, z = verts.T
            self.mesh = ipv.plot_trisurf(x, y, z, triangles=faces, color=color)
            self.fig.meshes = [self.mesh]
        except Exception as ex:
            print(ex)

        try:
            self.clipx.observe(self.on_update_plot)
            self.clipy.observe(self.on_update_plot)
            self.clipz.observe(self.on_update_plot)
            self.clipdist.observe(self.on_update_plot)
        except:
            pass
        self.progress.value = 10

    def on_update_style(self, v):
        """
        Update the plot style - for all parameters which do not involved recomputing
        the displayed object.
        :param k: ignored
        :return:
        """
        if v['name'] == 'value':
            if self.toggle_dark.value:
                ipv.pylab.style.set_style_dark()
            else:
                ipv.pylab.style.set_style_light()
                # Fix label colours (see self.fig.style)
                ipv.pylab.style.use({'axes': {'label': {'color': 'black'}, 'ticklabel': {'color': 'black'}}})
            if self.toggle_box.value:
                ipv.pylab.style.box_on()
            else:
                ipv.pylab.style.box_off()
            if self.toggle_axes.value:
                ipv.pylab.style.axes_on()
            else:
                ipv.pylab.style.axes_off()

    def on_select_file(self, v):
        """
        Called when a file selection has been done
        :param v:
        :return:
        """
        self.change_file(self.fc.selected)

    def change_file(self, file_name):
        """
        Function used to load data from a new file
        :param file_name: the file where the object data is loaded, either a CXI or modes h5 file
        :return:
        """
        self.progress.value = 3
        print('Loading:', file_name)

        try:
            self.toggle_plane.unobserve(self.on_update_plot)
            self.toggle_plane.value = False
            self.toggle_plane.observe(self.on_update_plot)
            d = h5.File(file_name, mode='r')['entry_1/data_1/data'][()]
            if d.ndim == 4:
                d = d[0]
            d = np.swapaxes(d, 0, 2)  # Due to labelling of axes x,y,z and not z,y,x
            if 'log' in self.toggle_phase.value:
                self.d0 = d
                d = np.log10(np.maximum(0.1, abs(d)))
            self.set_data(d)
        except:
            print("Failed to load file - is this a result CXI result or a modes file from a 3D CDI analysis ?")

    def on_change_type(self, v):
        if v['name'] == 'value':
            if isinstance(v['old'], str):
                newv = v['new']
                oldv = v['old']
                if 'log' in oldv and 'log' not in newv:
                    d = self.d0
                    self.set_data(d, threshold=10 ** self.threshold.value)
                elif 'log' in newv and 'log' not in oldv:
                    self.d0 = self.d
                    d = np.log10(np.maximum(0.1, abs(self.d0)))
                    self.set_data(d, threshold=np.log10(self.threshold.value))
                    return
            self.on_update_plot()

    def set_data(self, d, threshold=None):
        self.progress.value = 5
        self.d = d
        self.toggle_phase.unobserve(self.on_change_type)
        if np.iscomplexobj(d):
            if self.toggle_phase.value == 'log10(Abs)':
                self.toggle_phase.value = 'Abs'
            self.toggle_phase.options = ('Abs', 'Phase')
        else:
            if self.toggle_phase.value == 'Phase':
                self.toggle_phase.value = 'Abs'
            self.toggle_phase.options = ('Abs', 'log10(Abs)')
        self.toggle_phase.observe(self.on_change_type)

        self.threshold.unobserve(self.on_update_plot)
        self.colormap_range.unobserve(self.on_update_plot)
        self.threshold.max = abs(self.d).max()
        if threshold is None:
            self.threshold.value = self.threshold.max / 2
        else:
            self.threshold.value = threshold
        self.colormap_range.max = abs(self.d).max()
        self.colormap_range.value = [0, abs(self.d).max()]
        self.threshold.observe(self.on_update_plot)
        self.colormap_range.observe(self.on_update_plot)

        # print(abs(self.d).max(), self.threshold.value)
        nz, ny, nx = self.d.shape
        z, y, x = np.arange(nz), np.arange(ny), np.arange(nx)
        # Interpolate probe to object grid
        self.rgi = RegularGridInterpolator((z, y, x), self.d, method='linear', bounds_error=False, fill_value=0)

        if False:
            # Also prepare the phase gradient
            gz, gy, gx = np.gradient(self.d)
            a = np.maximum(abs(self.d), 1e-6)
            ph = self.d / a
            gaz, gay, gax = np.gradient(a)
            self.rgi_gx = RegularGridInterpolator((z, y, x), ((gx - gax * ph) / (ph * a)).real, method='linear',
                                                  bounds_error=False, fill_value=0)
            self.rgi_gy = RegularGridInterpolator((z, y, x), ((gy - gay * ph) / (ph * a)).real, method='linear',
                                                  bounds_error=False, fill_value=0)
            self.rgi_gz = RegularGridInterpolator((z, y, x), ((gz - gaz * ph) / (ph * a)).real, method='linear',
                                                  bounds_error=False, fill_value=0)

        # Fix extent
        ipv.pylab.xlim(0, max(self.d.shape))
        ipv.pylab.ylim(0, max(self.d.shape))
        ipv.pylab.zlim(0, max(self.d.shape))
        ipv.squarelim()
        self.on_update_plot()

    def on_animate(self, v):
        """
        Trigger the animation (rotation around vertical axis)
        :param v:
        :return:
        """
        if self.pcb_rotate is None:
            self.pcb_rotate = PeriodicCallback(self.callback_rotate, 50.)
        if self.toggle_rotate.value:
            self.pcb_rotate.start()
        else:
            self.pcb_rotate.stop()

    def callback_rotate(self):
        """ Used for periodic rotation"""
        # ipv.view() only supports a rotation against the starting azimuth and elevation
        # ipv.view(azimuth=ipv.view()[0]+1)

        # Use a quaternion and the camera's 'up' as rotation axis
        x, y, z = self.fig.camera.up
        n = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        a = np.deg2rad(2.5) / 2  # angular step
        sa, ca = np.sin(a / 2) / n, np.cos(a / 2)
        r = Rotation.from_quat((sa * x, sa * y, sa * z, ca))
        self.fig.camera.position = tuple(r.apply(self.fig.camera.position))
