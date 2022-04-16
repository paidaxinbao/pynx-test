# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016- ESRF, The European Synchrotron (Grenoble, France)
#       2010-2015 Université Joseph Fourier and CEA/INAC/SP2M (Grenoble, France)
#       Author: Vincent Favre-Nicolin <favre@esrf.fr>

import pyopencl as cl
import threading
from numpy import *
import time


def complex2rgba(s, amin=0.5, dlogs=2):
    ph = arctan2(s.imag, s.real)
    t = pi / 3
    nx, ny = s.shape
    rgba = zeros((nx, ny, 4))
    rgba[:, :, 0] = (ph < t) * (ph > -t) + (ph > t) * (ph < 2 * t) * (2 * t - ph) / t + (ph > -2 * t) * (ph < -t) * (
    ph + 2 * t) / t
    rgba[:, :, 1] = (ph > t) + (ph < -2 * t) * (-2 * t - ph) / t + (ph > 0) * (ph < t) * ph / t
    rgba[:, :, 2] = (ph < -t) + (ph > -t) * (ph < 0) * (-ph) / t + (ph > 2 * t) * (ph - 2 * t) / t
    a = log10(abs(s))
    a -= a.max() - dlogs  # display dlogs orders of magnitude
    rgba[:, :, 3] = amin + a / dlogs * (1 - amin) * (a > 0)
    return rgba


def complex2rgbalin(s, amin=0):
    ph = arctan2(s.imag, s.real)
    t = pi / 3
    nx, ny = s.shape
    rgba = zeros((nx, ny, 4))
    rgba[:, :, 0] = (ph < t) * (ph > -t) + (ph > t) * (ph < 2 * t) * (2 * t - ph) / t + (ph > -2 * t) * (ph < -t) * (
    ph + 2 * t) / t
    rgba[:, :, 1] = (ph > t) + (ph < -2 * t) * (-2 * t - ph) / t + (ph > 0) * (ph < t) * ph / t
    rgba[:, :, 2] = (ph < -t) + (ph > -t) * (ph < 0) * (-ph) / t + (ph > 2 * t) * (ph - 2 * t) / t
    a = abs(s)
    a = a - a.min()
    rgba[:, :, 3] = a / a.max() * (1 - amin) + amin
    return rgba


def colourwheel():
    xwheel = linspace(-1, 1, 100)
    ywheel = linspace(-1, 1, 100)[:, newaxis]
    rwheel = sqrt(xwheel ** 2 + ywheel ** 2)
    phiwheel = -arctan2(ywheel, xwheel)  # Need the - sign because imshow starts at (top,left)
    rhowheel = rwheel * exp(1j * phiwheel)
    ax = axes((0.22, 0.76, 0.12, .12), facecolor='w')  # [left, bottom, width, height]
    ax.set_axis_off()
    rgba = complex2rgba(rhowheel * (rwheel < 1), 0.0)
    imshow(rgba)
    text(1.1, 0.5, '$0$', fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    text(-.1, 0.5, '$\pi$', fontsize=16, horizontalalignment='center', verticalalignment='center',
         transform=ax.transAxes)


class GPUThreads(object):
    threads = []

    def __init__(self, gpu_name="GTX 690", cl_platform="", verbose=False):
        self.verbose = verbose
        self.gpu_name = gpu_name.lower()
        self.cl_platform = cl_platform
        self.cl_devices = []
        self.gpu_name_platform_real = ""
        try:
            tmp = []
            for p in cl.get_platforms():
                if p.name.find(cl_platform) >= 0:
                    tmp += p.get_devices()
            if self.gpu_name == "gpu":  # EXPERIMENTAL => "Context failed: out of host memory" ??
                for d in tmp:
                    if d.type & cl.device_type.GPU:
                        self.cl_devices.append(d)
            else:
                for d in tmp:
                    if d.name.lower().find(self.gpu_name) >= 0:
                        self.cl_devices.append(d)
            nbthread = len(self.cl_devices)
            for i in range(nbthread):
                self.threads.append(OpenCLThread_Fresnel(self.cl_devices[i], verbose=verbose))
                self.threads[-1].setDaemon(True)
                self.threads[-1].start()
            self.gpu_name_platform_real = "OpenCL (%s):%s" % (self.cl_devices[0].platform.name, self.cl_devices[0].name)
            while self.threads[-1].context_init == False:
                time.sleep(0.01)
        except:
            print(
                "Failed importing PyOpenCL, or no platform/graphics card (platform=" + cl_platform + ", gpu_name=" + gpu_name + ") !!!")

        if self.verbose:
            print(
                "Initialized PyNX threads for: " + self.gpu_name + " (language=OpenCL," + self.cl_platform + "), REAL=" + self.gpu_name_platform_real)

    def __del__(self):
        if self.verbose: print("Deleting GPUThreads object")
        nbthread = len(self)
        for j in range(nbthread):
            self.threads[0].join_flag = True
            self.threads[0].eventStart.set()
            self.threads[0].join()
            self.threads.pop()

    def __len__(self):
        return len(self.threads)

    def __getitem__(self, i):
        return self.threads[i]


gputhreads = None

#####################################        OpenCL Kernels      ##########################################
###########################################################################################################
###########################################################################################################
CL_Fresnel_CODE = """

__kernel __attribute__((reqd_work_group_size(%(block_size)d, 1, 1))) 
void Fresnel(__global float *vreal1,__global float *vimag1,
          const float x1, const float y1, 
          const float stepx1, const float stepy1, const unsigned long nx1,const unsigned long ny1,
          __global float *vreal2,__global float *vimag2,
          __global float *vx2,__global float *vy2,__global float *vdz,
          const float wavelength)
{
  /// Each thread corresponds to one pixel in the destination plane
  
  #define BLOCKSIZE %(block_size)d
  #define pi 3.14159265358979323846f
  
  const float pilambda=pi/(wavelength);
  
  float real2=0;
  float imag2=0;
  
  // Block index : 
  //int bx = get_group_id(0);
  //int by = get_group_id(1);

  // Thread index
  int tx = get_local_id(0);
  //int ty = get_local_id(1);
  
  const unsigned long i2=tx+get_group_id(0)*BLOCKSIZE;
  
  const float x2=vx2[i2];
  const float y2=vy2[i2];
  const float dz=vdz[i2];
  
  __local float locx1[BLOCKSIZE];
  __local float locvreal1[BLOCKSIZE];
  __local float locvimag1[BLOCKSIZE];
  for (unsigned long iy1=0;iy1<ny1;iy1+=1)
  {
    const float yy1=y1+stepy1*iy1;
    for (unsigned long ix1=0;ix1<nx1;ix1+=BLOCKSIZE)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      locx1[tx]=x1+(ix1+tx)*stepx1;
      locvreal1[tx]=vreal1[ix1+iy1*nx1+tx];
      locvimag1[tx]=vimag1[ix1+iy1*nx1+tx];
      barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
        const float dx=x2-locx1[i];
        const float dy=y2-yy1;
        // Paraxial approximation
        const float tmp=pilambda*(dx*dx+dy*dy)/dz;
        const float s=native_sin(tmp);
        const float c=native_cos(tmp);
        real2+=locvreal1[i]*c-locvimag1[i]*s;
        imag2+=locvreal1[i]*s+locvimag1[i]*c;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  vreal2[i2]=real2;
  vimag2[i2]=imag2;
}"""


class OpenCLThread_Fresnel(threading.Thread):
    """(internal)
    Class to compute Fhkl in a single thread (single OpenCL platform/device)
    """

    def __init__(self, dev, verbose=False):
        threading.Thread.__init__(self)
        """ Here we assume that the number of hkl is a multiple of 32
        0-padding must already have been done by the calling program
        """
        self.dev = dev
        self.verbose = verbose
        self.dt = 0.0
        if self.verbose: print(self.dev.name)
        self.eventStart = threading.Event()
        self.eventFinished = threading.Event()
        self.join_flag = False
        self.bug_apple_cpu_workgroupsize_warning = True
        self.context_init = False
    def run(self):
        try_ctx = 5
        while try_ctx > 0:
            try:
                ctx = cl.Context([self.dev])
                try_ctx = 0
            except:
                try_ctx -= 1
                print(("Problem initializing OpenCL context... #try%d/5, wait%3.1fs" % (
                5 - try_ctx, 0.1 * (5 - try_ctx))))
                time.sleep(0.1 * (5 - try_ctx))
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags
        # Kernel program will be initialized when necessary
        CL_Fresnel = None

        self.block_size = 64
        if self.dev.max_work_group_size < self.block_size:
            self.block_size = self.dev.max_work_group_size

        if self.dev.platform.name == "Apple" and self.dev.name.find("CPU") > 0:
            if self.bug_apple_cpu_workgroupsize_warning:
                print("WARNING: workaround Apple OpenCL CPU bug: forcing group size=1")
                self.bug_apple_cpu_workgroupsize_warning = False
            self.block_size = 1
        MULTIPROCESSOR_COUNT = self.dev.max_compute_units
        if self.verbose: print(self.name, " ...beginning")

        kernel_params = {"block_size": self.block_size}

        # if "NVIDIA" in queue.device.vendor:
        options = "-cl-mad-enable -cl-fast-relaxed-math"  # -cl-unsafe-math-optimizations

        self.context_init = True
        while True:
            self.eventStart.wait()
            if self.join_flag: break
            if self.verbose: print(self.name, " ...got a job !")

            t0 = time.time()
            nxy1 = int(self.nx1 * self.ny1)

            self.vreal2 = zeros((self.n2), float32)
            self.vimag2 = zeros((self.n2), float32)

            vreal2_ = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.vreal2, size=0)
            vimag2_ = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.vimag2, size=0)
            vreal1_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vreal1, size=0)
            vimag1_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vimag1, size=0)
            vx2_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vx2, size=0)
            vy2_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vy2, size=0)
            vz2_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vdz2, size=0)
            if CL_Fresnel == None:
                if self.verbose: print("Compiling CL_Fresnel (block size=%d)" % self.block_size)
                CL_Fresnel = cl.Program(ctx, CL_Fresnel_CODE % kernel_params, ).build(options=options)

            print(self.n2, 1,self.block_size,1)
            CL_Fresnel.Fresnel(queue, (self.n2, 1), (self.block_size, 1),
                               vreal1_, vimag1_, float32(self.x1), float32(self.y1), float32(self.stepx1),
                               float32(self.stepy1), int64(self.nx1), int64(self.ny1),
                               vreal2_, vimag2_, vx2_, vy2_, vz2_, float32(self.wavelength)).wait()

            cl.enqueue_copy(queue, self.vreal2, vreal2_)
            cl.enqueue_copy(queue, self.vimag2, vimag2_)
            queue.finish()

            self.dt = time.time() - t0
            self.eventStart.clear()
            self.eventFinished.set()


def Fresnel_thread(v1, vx1, vy1, vx2, vy2, vdz2, wavelength=1e-10, verbose=False, gpu_name="GTX", cl_platform=""):
    """
    Compute the Fresnel propagation between an origin and a destination plane, using direct calculation.
    Uses OpenCL.

    NOTE: this code is experimental, mostly used for tests !
    
    :param v1: complex 2D array of the field to propagate, size=nx1*ny1
    :param vx1: 1D vectors of x and y coordinates of v1 (nx1, ny1)
    :param vy1: 1D vectors of x and y coordinates of v1 (nx1, ny1)
    :param vx2: 1D vectors of x and y coordinates of v2 (nx1, ny1)
    :param vy2: 1D vectors of x and y coordinates of v2 (nx1, ny1)
    :param vdz2: distance (m) between the origin and destination plane
    :param wavelength: wavelength
    :param verbose: if True, print calculcation messages
    :param gpu_name: name of the GPU to use (string)
    :param cl_platform: OpenCL platform name to use (string)
    :return: a complex array of the propagated wavefront with the same shape as vx2, vy2.
    """
    global gputhreads
    if gputhreads == None:
        if verbose: print(
            "Fresnel_thread: initializing gputhreads with GPU=%s, cl_platform=%s" % (gpu_name, cl_platform))
        gputhreads = GPUThreads(gpu_name, verbose=verbose, cl_platform=cl_platform)
    elif gputhreads.gpu_name != gpu_name or gputhreads.cl_platform != cl_platform:
        # GPU has changed, re-initialize
        gputhreads = None
        if verbose: print(
            "Fresnel_thread: initializing gputhreads with GPU=%s, cl_platform=%s" % (gpu_name, cl_platform))
        gputhreads = GPUThreads(gpu_name, verbose=verbose, cl_platform=cl_platform)

    # nx1 and nx2 must be multiples of BLOCKSIZE !
    BLOCKSIZE = 32
    for t in gputhreads.threads:
        if t.block_size > BLOCKSIZE:
            BLOCKSIZE = t.block_size

    # Force float32 type
    nx1 = len(vx1)
    ny1 = len(vy1)
    x1 = vx1.min()
    y1 = vy1.min()
    stepx1 = (vx1.max() - vx1.min()) / (nx1 - 1)
    stepy1 = (vy1.max() - vy1.min()) / (ny1 - 1)

    # Make sure x2,y2,dz2 have the same dimensions
    vvx2 = (vx2 + 0 * (vy2 + vdz2)).astype(float32)
    vvy2 = (vy2 + 0 * (vx2 + vdz2)).astype(float32)
    vvdz2 = (vdz2 + 0 * (vx2 + vy2)).astype(float32)
    n2 = len(vvx2.flat)

    # Create as many threads as available devices
    t0 = time.time()
    nbthread = len(gputhreads)

    # threads=[]
    for i in range(nbthread):
        a0 = i * int(n2 / nbthread)
        a1 = (i + 1) * int(n2 / nbthread)
        if verbose: print("Thread #", i, [a0, a1])
        if i == (nbthread - 1): a1 = n2
        gputhreads[i].verbose = verbose
        gputhreads[i].v1 = v1
        gputhreads[i].x1 = vx1[0]
        gputhreads[i].y1 = vy1[0]
        gputhreads[i].vx2 = vvx2.flat[a0:a1]
        gputhreads[i].vy2 = vvy2.flat[a0:a1]
        gputhreads[i].vdz2 = vvdz2.flat[a0:a1]
        gputhreads[i].nx1 = len(vx1)
        gputhreads[i].ny1 = len(vy1)
        gputhreads[i].n2 = a1 - a0
        gputhreads[i].stepx1 = stepx1
        gputhreads[i].stepy1 = stepy1
        gputhreads[i].vreal1 = v1.real.copy()
        gputhreads[i].vimag1 = v1.imag.copy()
        gputhreads[i].wavelength = wavelength
        gputhreads[i].eventFinished.clear()
        gputhreads[i].eventStart.set()
    for i in range(nbthread):
        gputhreads[i].eventFinished.wait()

    v2 = zeros((n2), dtype=complex64)
    for i in range(nbthread):
        a0 = i * int(n2 / nbthread)
        a1 = (i + 1) * int(n2 / nbthread)
        if i == (nbthread - 1): a1 = n2
        # print v2[a0:a1].shape,gputhreads[i].vreal2.shape, gputhreads[i].vimag2.shape
        v2[a0:a1] = gputhreads[i].vreal2 + 1j * gputhreads[i].vimag2
        if verbose: print("Thread #%d, dt=%7.5f" % (i, t.dt))
    # for i in xrange(nbthread): del threads[0]
    dt = time.time() - t0
    return v2.reshape(vvx2.shape), dt


if __name__ == '__main__':
    from pylab import *

    gpuname = 'Iris'

    if True:
        # Test propagation from a 200x200 square opening, incident plane wave
        n=512
        vx1=linspace(0,100e-6,n)
        vy1=linspace(0,100e-6,n)
        v1=ones((n,n),dtype=float32)
        vx2=linspace(0,100e-6,n)
        vy2=linspace(0,100e-6,n)[:,newaxis]
        dz=.2
        v2,dt=Fresnel_thread(v1,vx1,vy1,vx2,vy2,dz,wavelength=12.384e-10/8,verbose=True,gpu_name=gpuname)
        figure(1,figsize=(21,7))
        clf()
        subplot(131)
        imshow(v2.real)
        colorbar()
        subplot(132)
        imshow(v2.imag)
        colorbar()
        subplot(133)
        imshow(abs(v2))
        colorbar()

    if False:
        # test FZP
        wavelength = float32(12398.4 / 8000 * 1e-10)
        focal_length = float32(.09)
        rmax = float32(100e-6)  # radius of FZP
        r_cs = float32(32.5e-6)  # Central stop radius
        osa_z, osa_r = float32(.08), float32(25e-6)  # OSA position and radius

        sourcez = float32(-49)  # Point source position
        focal_point = 1 / (1 / focal_length - 1 / abs(sourcez))

        figure(1, figsize=(21, 7))
        clf()
        # Field @FZP
        print("Calc Field @FZP")
        N = 512
        x1 = linspace(-rmax, rmax, N).astype(float32)
        y1 = linspace(-rmax, rmax, N)[:, newaxis].astype(float32)
        r2 = x1 ** 2 + y1 ** 2
        if True:
            dxy2_slits = 100e-6
            # illumination from 200x200 micron^2 slit @ 1 meter
            x0 = linspace(-dxy2_slits, dxy2_slits, N)
            y0 = linspace(-dxy2_slits, dxy2_slits, N)[:, newaxis]
            dz = 1
            v0 = ones((N, N), dtype=float32)
            v1, dt = Fresnel_thread(v0, x0, y0, x1, y1, dz, wavelength=wavelength, verbose=False, gpu_name=gpuname)
        elif False:
            v1 = random.normal(1, 0.9, (N, N))
        else:
            v1 = 1
        # FZP form factor
        v1 = (v1 * (1 - signbit(cos(pi * r2 / (wavelength * focal_point)))) * (r2 < (rmax ** 2)) * (r2 > (r_cs ** 2))).astype(
            complex64)
        x, y = x1, y1
        subplot(131)
        pylab.imshow(complex2rgbalin(v1, amin=0.0), extent=(x.min() * 1e9, x.max() * 1e9, y.min() * 1e9, y.max() * 1e9),
                     aspect='equal', origin='lower')
        xlabel(r"$x\ (\mu m)$", fontsize=16)
        ylabel(r"$y\ (\mu m)$", fontsize=16)
        title("$Field\ @FZP$")

        # XZ plane near focus point

        # Field @OSA
        x2 = linspace(-osa_r, osa_r, N).astype(float32)
        y2 = linspace(-osa_r, osa_r, N)[:, newaxis].astype(float32)
        print("Calc Field @OSA")
        v2, dt = Fresnel_thread(v1, x1, y1, x2, y2, osa_z, wavelength=wavelength, verbose=False, gpu_name=gpuname)
        v2 = v2 * ((x2 * x2 + y2 * y2) < osa_r ** 2)
        x, y = x2, y2
        subplot(132)
        pylab.imshow(complex2rgbalin(v2, amin=0.0), extent=(x.min() * 1e6, x.max() * 1e6, y.min() * 1e6, y.max() * 1e6),
                     aspect='equal', origin='lower')
        xlabel(r"$x\ (\mu m)$", fontsize=16)
        ylabel(r"$y\ (\mu m)$", fontsize=16)
        title("$Field\ @OSA$")

        # Field @focus point
        x3 = linspace(-.2e-6, .2e-6, 128)[:, newaxis]
        y3 = 0
        z3 = focal_point + linspace(-.5e-3, .5e-3, 256)
        print("Calc Field @Focus point (2D)")
        v3, dt = Fresnel_thread(v2, x2, y2, x3, y3, z3 - osa_z, wavelength=wavelength, verbose=False, gpu_name=gpuname)
        subplot(233)
        x = x3
        z = z3 - focal_point
        pylab.imshow(complex2rgbalin(v3, amin=0.0), extent=(z.min() * 1e6, z.max() * 1e6, x.min() * 1e9, x.max() * 1e9),
                     aspect='equal', origin='lower')
        xlabel(r"$z\ (\mu m)$", fontsize=16)
        ylabel(r"$x\ (nm)$", fontsize=16)
        title("$Field\ @focus\ point$")
        # Spot size
        x4 = linspace(-.2e-6, .2e-6, 256)[:, newaxis]
        y4 = 0
        z4 = focal_point
        print("Calc Field @Focus point (line)")
        v4, dt = Fresnel_thread(v2, x2, y2, x4, y4, z4 - osa_z, wavelength=wavelength, verbose=False, gpu_name=gpuname)
        subplot(236)
        plot(x4 * 1e9, abs(v4))
        xlabel(r"$x\ (nm)$", fontsize=16)
        ylabel(r"$Amplitude$")
        title("$Field\ @focus\ point$")
