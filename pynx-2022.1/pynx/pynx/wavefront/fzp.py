# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import pyopencl as cl
import threading
import numpy as np
import time

try:
  import pycuda.driver as drv
  import pycuda.compiler as compiler
  drv.init()
  assert drv.Device.count() >= 1
  only_cpu=False
except:
  only_cpu=True

class GPUThreads(object):
    threads = []

    def __init__(self, gpu_name="GTX", cl_platform="", verbose=False):
        self.verbose = verbose
        self.gpu_name = gpu_name # Remember original gpu name
        if isinstance(gpu_name, list):
            self.gpu_names=gpu_name
        else:
            self.gpu_names=[gpu_name]
        for i in range(len(self.gpu_names)):
            self.gpu_names[i] = self.gpu_names[i].lower()
        self.cl_platform = cl_platform
        self.cl_devices = []
        self.gpu_name_platform_real = []
        try:
            tmp = []
            for p in cl.get_platforms():
                if p.name.find(cl_platform) >= 0:
                    tmp += p.get_devices()
            if self.gpu_names[0] == "gpu":  # EXPERIMENTAL => "Context failed: out of host memory" ??
                for d in tmp:
                    if d.type & cl.device_type.GPU:
                        self.cl_devices.append(d)
            else:
                for d in tmp:
                    for g in self.gpu_names:
                        if d.name.lower().find(g) >= 0:
                            self.cl_devices.append(d)
                            break
            nbthread = len(self.cl_devices)
            for i in range(nbthread):
                self.threads.append(OpenCLThread_Fresnel(self.cl_devices[i], verbose=verbose))
                self.threads[-1].setDaemon(True)
                self.threads[-1].start()
                self.gpu_name_platform_real.append("OpenCL (%s):%s" % (self.cl_devices[i].platform.name, self.cl_devices[i].name))
            while self.threads[-1].context_init == False:
                time.sleep(0.01)
        except:
            print("Failed importing PyOpenCL, or no platform/graphics card (platform=" + cl_platform + ", gpu_name=" + gpu_name + ") !!!")
            print("Available OpenCL platforms:")
            for p in cl.get_platforms():
                print(p.name+":",[d.name for d in p.get_devices()])

        if self.verbose:
            print("Initialized clFZP threads for: ", self.gpu_names, " (language=OpenCL," + self.cl_platform + "), "
                  "REAL=", self.gpu_name_platform_real)
            print("Devices used:",[d.name for d in self.cl_devices])

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

gputhreads=None

#####################################        OpenCL Kernels      ##########################################
###########################################################################################################
###########################################################################################################
CL_FZP_CODE = """
__kernel __attribute__((reqd_work_group_size(%(block_size)d, 1, 1)))
void FZP(__global float *a_real,__global float *a_imag,
         __global float *vx, __global float *vy, __global float *vz, // Why __constant for vx/y/z does not work ?
         const float sourcex,const float sourcey,const float sourcez,
         const float wavelength, const float focal_length,
         const float rmax, const float r_cs, const float osa_z, const float osa_r,
         const unsigned int nr, const unsigned int ntheta)
{
  /// Each thread corresponds to one pixel in the destination plane
  #define BLOCKSIZE %(block_size)d
  #define twopi 6.2831853071795862f
  #define pi 3.14159265358979323846f
  const float stepr=(rmax-r_cs)/nr;
  const float steptheta=twopi/ntheta;// number of theta steps must be a multiple of BLOCKSIZE
  const float osa_r2=osa_r*osa_r;

  float ar=0 , ai=0;

  // Block index :
  //int bx = get_group_id(0);
  //int by = get_group_id(1);

  // Thread index
  int tx = get_local_id(0);
  //int ty = get_local_id(1);

  const unsigned long i2=tx+get_group_id(0)*BLOCKSIZE;

  //x,y,z: coordinates of the points (probably near the focus) at which the amplitude is calculated
  const float x=vx[i2];
  const float y=vy[i2];
  const float z=vz[i2];

  __local float xfzp[BLOCKSIZE];
  __local float yfzp[BLOCKSIZE];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (float r=r_cs;r<rmax;r+=stepr)
  {
    //rho for the FZP - here assuming a plane wave illumination
    //const float rho=1+cos(pi*r*r/(wavelength*focal_length));                // Smooth(sinusoidal) FZP
    //const float rho=1-roundf(remainderf(r*r/(wavelength*focal_length),1));  // binary FZP, version 1
    const float rho=1-signbit(cos(pi*r*r/(wavelength*focal_length)));         // binary FZP, version 2
    for (int itheta=0; itheta<ntheta ; itheta+=BLOCKSIZE)
    {
      const float theta = itheta * steptheta;
      barrier(CLK_LOCAL_MEM_FENCE);
      {
        const float tmp=theta+tx*steptheta;
        const float s=native_sin(tmp);
        const float c=native_cos(tmp);
        xfzp[tx]=r*c;
        yfzp[tx]=r*s;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
        //Each point acts as a secondary source
        //Use a paraxial approximation:
        const float dx=xfzp[i]-x;
        const float dy=yfzp[i]-y;
        float dxy2=dx*dx+dy*dy;
        const float d=sqrt(dxy2 +z*z);
        const float tmp = pi*dxy2/(wavelength*z);
        const float s=native_sin(tmp);
        const float c=native_cos(tmp);

        // Take into account OSA
        const float x_osa=(xfzp[i]*(z-osa_z)+x*osa_z)/z;
        const float y_osa=(yfzp[i]*(z-osa_z)+y*osa_z)/z;
        const float t_osa=(x_osa*x_osa+y_osa*y_osa)<osa_r2;

        // Take into account source illumination (again use paraxial approximation)
        const float dxs=xfzp[i]-sourcex;
        const float dys=yfzp[i]-sourcey;
        dxy2=dxs*dxs+dys*dys;
        const float tmp1=-pi*dxy2/(wavelength*sourcez);
        const float s1=native_sin(tmp1);
        const float c1=native_cos(tmp1);

        ar +=rho*(c*c1-s*s1)/d*r*t_osa;
        ai +=rho*(s*c1+c*s1)/d*r*t_osa;
      }
    }
  }

  a_real[i2]=ar*stepr*steptheta*1000000;
  a_imag[i2]=ai*stepr*steptheta*1000000;
}"""

CL_FZP_RECT_CODE = """
__kernel __attribute__((reqd_work_group_size(%(block_size)d, 1, 1)))
void FZP(__global float *a_real,__global float *a_imag,
         __global float *vx, __global float *vy, __global float *vz,
         const float sourcex,const float sourcey,const float sourcez,
         const float wavelength, const float focal_length,
         const float fzp_xmin, const float fzp_xmax, const float fzp_ymin, const float fzp_ymax,
         const float r_cs, const float osa_z, const float osa_r,
         const unsigned int nfzp_x, const unsigned int nfzp_y)
{
  /// Each thread corresponds to one pixel in the destination plane
  #define BLOCKSIZE %(block_size)d
  #define twopi 6.2831853071795862f
  #define pi 3.14159265358979323846f
  const float stepx=(fzp_xmax-fzp_xmin)/(nfzp_x-1);
  const float stepy=(fzp_ymax-fzp_ymin)/(nfzp_y-1);// this must be a multiple of BLOCKSIZE
  const float osa_r2=osa_r*osa_r;
  const float cs_r2=r_cs*r_cs;

  float ar=0 , ai=0;

  // Block index :
  //int bx = get_group_id(0);
  //int by = get_group_id(1);

  // Thread index
  int tx = get_local_id(0);
  //int ty = get_local_id(1);

  const unsigned long i2=get_global_id(0);

  //x,y,z: coordinates of the points (probably near the focus) at which the amplitude is calculated
  const float x=vx[i2];
  const float y=vy[i2];
  const float z=vz[i2];

  __local float yfzp[BLOCKSIZE];
  __local float rhofzp[BLOCKSIZE];
  barrier(CLK_LOCAL_MEM_FENCE);
  // TODO: loop on flat xy array, and only require nx*ny to be a multiple of BLOCKSIZE
  for (float xfzp=fzp_xmin;xfzp<=fzp_xmax;xfzp+=stepx)
  {
    const float dx=xfzp-x;
    for (float yfzp0=fzp_ymin;yfzp0<=fzp_ymax;yfzp0+=stepy*BLOCKSIZE)
    {
      yfzp[tx]=yfzp0+tx*stepy;
      const float r2=xfzp*xfzp+yfzp[tx]*yfzp[tx];
      //rho for the FZP - here assuming a plane wave illumination, taking into account the central stop
      rhofzp[tx]=(1-signbit(cos(pi*r2/(wavelength*focal_length))))*(r2>cs_r2);               // binary FZP
      barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
        //Each point acts as a secondary source
        //Use a paraxial approximation:
        const float dy=yfzp[i]-y;
        float dxy2=dx*dx+dy*dy;
        const float d=sqrt(dxy2 +z*z);
        const float tmp = pi*dxy2/(wavelength*z);
        const float s=native_sin(tmp);
        const float c=native_cos(tmp);

        // Take into account OSA
        const float x_osa=(xfzp*(z-osa_z)+x*osa_z)/z;
        const float y_osa=(yfzp[i]*(z-osa_z)+y*osa_z)/z;
        const float t_osa=(x_osa*x_osa+y_osa*y_osa)<osa_r2;

        // Take into account source illumination (again use paraxial approximation)
        const float dxs=xfzp   -sourcex;
        const float dys=yfzp[i]-sourcey;
        dxy2=dxs*dxs+dys*dys;
        const float tmp1=-pi*dxy2/(wavelength*sourcez);
        const float s1=native_sin(tmp1);
        const float c1=native_cos(tmp1);

        ar +=rhofzp[i]*(c*c1-s*s1)/d*t_osa;
        ai +=rhofzp[i]*(s*c1+c*s1)/d*t_osa;
      }
    }
  }

  a_real[i2]=ar*stepx*stepy*1000000;
  a_imag[i2]=ai*stepx*stepy*1000000;
}"""


class OpenCLThread_Fresnel(threading.Thread):
    """(internal)
    Class to compute illumination from a FZP in a single thread (single OpenCL platform/device)
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
        CL_FZP = None
        CL_FZP_RECT = None

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

            self.areal = np.zeros(self.vx.shape, np.float32)
            self.aimag = np.zeros(self.vx.shape, np.float32)

            areal_ = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=self.areal, size=0)
            aimag_ = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=self.aimag, size=0)
            vx_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vx, size=0)
            vy_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vy, size=0)
            vz_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vz, size=0)
            if self.fzp_ny is None:
                # Scattering from the entire FZP up to rmax
                if CL_FZP is None:
                    if self.verbose: print("Compiling CL_FZP (block size=%d)" % self.block_size)
                    CL_FZP = cl.Program(ctx, CL_FZP_CODE % kernel_params, ).build(options=options)

                #print(self.n2,1,self.block_size,1,self.vx.size,self.vy.size,self.vz.size,self.areal.size,self.aimag.size)
                CL_FZP.FZP(queue, (self.n2, 1), (self.block_size, 1),
                                   areal_, aimag_, vx_, vy_, vz_, np.float32(self.sourcex), np.float32(self.sourcey), np.float32(self.sourcez),
                                   np.float32(self.wavelength), np.float32(self.focal_length), np.float32(self.rmax),
                                   np.float32(self.r_cs), np.float32(self.osa_z), np.float32(self.osa_r),
                                   np.uint32 (self.nr), np.uint32(self.ntheta)).wait()
                self.flop = self.nr * self.ntheta * self.vx.size * 48
            else:
                if CL_FZP_RECT is None:
                    if self.verbose: print("Compiling CL_FZP_RECT (block size=%d)" % self.block_size)
                    CL_FZP_RECT = cl.Program(ctx, CL_FZP_RECT_CODE % kernel_params, ).build(options=options)
                # print(self.n2,1,self.block_size,1,self.vx.size,self.vy.size,self.vz.size,self.areal.size,self.aimag.size)
                CL_FZP_RECT.FZP(queue, (self.n2, 1), (self.block_size, 1),
                                areal_, aimag_, vx_, vy_, vz_, np.float32(self.sourcex), np.float32(self.sourcey), np.float32(self.sourcez),
                                np.float32(self.wavelength), np.float32(self.focal_length),
                                np.float32(self.fzp_xmin), np.float32(self.fzp_xmax), np.float32(self.fzp_ymin), np.float32(self.fzp_ymax),
                                np.float32(self.r_cs), np.float32(self.osa_z), np.float32(self.osa_r),
                                np.uint32(self.fzp_nx), np.uint32(self.fzp_ny)).wait()
                self.flop = self.fzp_nx * self.fzp_ny * self.vx.size * 48
            cl.enqueue_copy(queue, self.areal, areal_)
            cl.enqueue_copy(queue, self.aimag, aimag_)
            queue.finish()

            self.dt = time.time() - t0
            self.eventStart.clear()
            self.eventFinished.set()


def FZP_thread(x, y, z, sourcex=0, sourcey=0, sourcez=-50, wavelength=1, focal_length=.129, rmax=100e-6, r_cs=0, osa_z=0, osa_r=1e6, nr=512,
               ntheta=256, fzp_xmin=None, fzp_xmax=None, fzp_ymin=None, fzp_ymax=None, fzp_nx=None, fzp_ny=None, gpu_name="GTX",
               cl_platform= "", verbose=False):
    """
    Compute illumination from a FZP, itself illuminated from a single monochromatic point source. Uses OpenCL.
    
    The integration is either made on the fill circular shaphe of the FZP, or a rectangular area.
    
    All units are SI.

    NOTE: this code is experimental, mostly used for tests !
    
    :param x: numpy array of coordinates where the illumination will be calculated
    :param y: numpy array of coordinates where the illumination will be calculated
    :param z: numpy array of coordinates where the illumination will be calculated
    :param sourcex: position of the point source illuminating the FZP (float)
    :param sourcey: position of the point source illuminating the FZP (float)
    :param sourcez: position of the point source illuminating the FZP (float)
    :param wavelength: the wavelength of the incident beam
    :param focal_length: the focal length of the FZP
    :param rmax: max radius of the FZP
    :param r_cs: radius of the central stop
    :param osa_z: z position of the Order Sorting Aperture, relative to the FZP
    :param osa_r: radius of the OSA
    :param nr: number of radial steps for the integration
    :param ntheta: number of angular (polar) steps for the integration
    :param fzp_xmin: x min coordinate for a rectangular illumination
    :param fzp_xmax: x max coordinate for a rectangular illumination
    :param fzp_ymin: y min coordinate for a rectangular illumination
    :param fzp_ymax: y max coordinate for a rectangular illumination
    :param fzp_nx: number of x steps for the integration, for a rectangular illumination
    :param fzp_ny: number of y steps for the integration, for a rectangular illumination
    :param gpu_name: name (sub-string) of the gpu to be used
    :param cl_platform: OpenCL platform to use (optional)
    :param verbose: if true, will report on the progress of threaded calculations.
    :return: a complex numpy array of the calculated illumination, with the same shape as x,y,z.
    """
    global gputhreads
    if gputhreads == None:
        if verbose:
            print("FZP_thread: initializing gputhreads with GPU=%s, cl_platform=%s" % (gpu_name, cl_platform))
        gputhreads = GPUThreads(gpu_name, verbose=verbose, cl_platform=cl_platform)
    elif gputhreads.gpu_name != gpu_name or gputhreads.cl_platform != cl_platform:
        # GPU has changed, re-initialize
        gputhreads = None
        if verbose:
            print("FZP_thread: initializing gputhreads with GPU=%s, cl_platform=%s" % (gpu_name, cl_platform))
        gputhreads = GPUThreads(gpu_name, verbose=verbose, cl_platform=cl_platform)

    # Force float32 type
    if x.shape == y.shape and x.shape == z.shape and x.dtype == np.float32 and y.dtype == np.float32 and z.dtype == np.float32:
        vx = x.ravel()
        vy = y.ravel()
        vz = z.ravel()
    else:
        vx = (x + (y + z) * 0).astype(np.float32).ravel()
        vy = (y + (x + z) * 0).astype(np.float32).ravel()
        vz = (z + (x + y) * 0).astype(np.float32).ravel()

    # nxyz must be a multiple of BLOCKSIZE
    BLOCKSIZE = 32
    for t in gputhreads.threads:
        if t.block_size > BLOCKSIZE:
            BLOCKSIZE = t.block_size  # KLUDGE?

    nxyz = vx.size

    # Use as many threads as available devices
    nbthread = len(gputhreads)
    # Split into smaller jobs for large calculations - max 250 Gflop per call
    if fzp_xmin is not None and fzp_xmax is not None and fzp_ymin is not None and fzp_ymax is not None and fzp_nx is not None and fzp_ny is not None:
        nbstep = int(np.ceil(fzp_nx * fzp_ny * 48 * nxyz / 250e9))
    else:
        nbstep = int(np.ceil(nr * ntheta * 48 * nxyz / 250e9))
    nbstep += (nbstep%nbthread)
    step = nxyz // (nbstep * BLOCKSIZE)
    steps=list(range(0,nxyz,step * BLOCKSIZE * nbthread))
    if steps[-1]!=nxyz:
        steps.append(nxyz)

    if verbose:
        print("xyz steps:",steps)

    t0 = time.time()
    flop = 0
    a = (0j * vx).astype(np.complex64)
    while len(steps)>1:
        for i in range(nbthread):
            if len(steps)==1:
                gputhreads[i].a0 = None
                gputhreads[i].a1 = None
                break
            a0 = steps[0]
            a1 = steps[1]
            steps.remove(steps[0])
            if verbose: print("Thread #", i, [a0, a1],"/",nxyz, steps)
            gputhreads[i].a0=a0
            gputhreads[i].a1=a1
            n=a1-a0
            if n % BLOCKSIZE != 0:
                if verbose:
                    print("zero-padding...")
                gputhreads[i].vx = np.zeros((n//BLOCKSIZE+1)*BLOCKSIZE,np.float32)
                gputhreads[i].vy = np.zeros((n//BLOCKSIZE+1)*BLOCKSIZE,np.float32)
                gputhreads[i].vz = np.zeros((n//BLOCKSIZE+1)*BLOCKSIZE,np.float32)
                gputhreads[i].vx[:n] = vx[a0:a1]
                gputhreads[i].vy[:n] = vy[a0:a1]
                gputhreads[i].vz[:n] = vz[a0:a1]
            else:
                gputhreads[i].vx = vx[a0:a1]
                gputhreads[i].vy = vy[a0:a1]
                gputhreads[i].vz = vz[a0:a1]
            gputhreads[i].sourcex = np.float32(sourcex)
            gputhreads[i].sourcey = np.float32(sourcey)
            gputhreads[i].sourcez = np.float32(sourcez)
            gputhreads[i].wavelength = np.float32(wavelength)
            gputhreads[i].focal_length = np.float32(focal_length)
            if rmax is not None: gputhreads[i].rmax = np.float32(rmax)
            gputhreads[i].r_cs = np.float32(r_cs)
            gputhreads[i].osa_z = np.float32(osa_z)
            gputhreads[i].osa_r = np.float32(osa_r)
            if nr is not None: gputhreads[i].nr = np.int32(nr)
            if ntheta is not None: gputhreads[i].ntheta = np.int32(ntheta)
            gputhreads[i].n2 = gputhreads[i].vx.size
            if fzp_xmin is not None: gputhreads[i].fzp_xmin = np.float32(fzp_xmin)
            if fzp_xmax is not None: gputhreads[i].fzp_xmax = np.float32(fzp_xmax)
            if fzp_ymin is not None: gputhreads[i].fzp_ymin = np.float32(fzp_ymin)
            if fzp_ymax is not None: gputhreads[i].fzp_ymax = np.float32(fzp_ymax)
            if fzp_nx is not None: gputhreads[i].fzp_nx = np.int32(fzp_nx)
            if fzp_ny is not None:
                gputhreads[i].fzp_ny = np.int32(fzp_ny)
            else:
                gputhreads[i].fzp_ny = None  # used as test, we need this variable anyway
            gputhreads[i].eventFinished.clear()
            gputhreads[i].eventStart.set()
        for i in range(nbthread):
            if gputhreads[i].a0 is not None:
                gputhreads[i].eventFinished.wait()

        # for t in threads:t.join()
        for i in range(nbthread):
            t = gputhreads[i]
            a0 = gputhreads[i].a0
            a1 = gputhreads[i].a1
            if a0 is None:
                break
            a[a0:a1] = t.areal[:a1-a0] + 1j * t.aimag[:a1-a0]
            if verbose:
                print("Thread #%d, dt=%7.5f, %6.2f Gflop" % (i, t.dt, t.flop/1e9))
            flop += t.flop
    # for i in xrange(nbthread): del threads[0]
    dt = time.time() - t0
    return a.reshape((x + y + z).shape), dt, flop

def rn(n,f=.1289,wavelength=12398.4/8000*1e-10):
  return np.sqrt(n*wavelength*f+(n*wavelength/2.)**2)

def nr(r,f=.1289,wavelength=12398.4/8000*1e-10):
  return (-2*f+wavelength*np.sqrt(r**2+f**2))/wavelength


if __name__ == '__main__':
    from pynx.ptycho import complex2rgbalog, complex2rgbalin
    from pylab import *
    wavelength = np.float32(12398.4 / 8000 * 1e-10)
    focal_length = np.float32(.09)
    rmax = np.float32(100e-6)  # radius of FZP
    nr, ntheta = np.int32(1024), np.int32(512)  # number of points for integration on FZP
    r_cs = np.float32(40e-6)  # Central stop radius
    osa_z, osa_r = np.float32(.08), np.float32(20e-6)  # OSA position and radius
    sourcex = np.float32(0e-6)  # Source position
    sourcey = np.float32(0e-6)
    sourcez = np.float32(-90)

    x = np.linspace(-.5e-6, .5e-6, 127)[:, np.newaxis]
    y = 0  # linspace(-1e-6,1e-6,256)
    z = np.linspace(-.5e-3, .5e-3, 127) + 1/(1/focal_length+1/sourcez)
    # z=linspace(.03,.150,2048)[:,newaxis]
    x = (x + (y + z) * 0).astype(np.float32)
    y = (y + (x + z) * 0).astype(np.float32)
    z = (z + (x + y) * 0).astype(np.float32)
    nxyz = len(x.flat)

    a_real = (x * 0).astype(np.float32)
    a_imag = (x * 0).astype(np.float32)

    a, dt, flop = FZP_thread(x, y, z, sourcex=0, sourcey=0, sourcez=-50, wavelength=wavelength, focal_length=.09, rmax=100e-6,
                             r_cs=0, osa_z=0, osa_r=1e6, nr=nr, ntheta=ntheta, fzp_xmin=None, fzp_xmax=None, fzp_ymin=None, fzp_ymax=None,
                             fzp_nx=None, fzp_ny=None, gpu_name="gpu", verbose=True)
    print("clFZP dt=%9.5fs, %8.2f Gflop/s" % (dt, flop / 1e9 / dt))


    clf()
    imshow(complex2rgbalin(a), extent=(z.min() * 1e3, z.max() * 1e3, x.min() * 1e6, x.max() * 1e6), aspect='auto',
                 origin='lower')
    # pylab.imshow(abs(a),vmin=0,vmax=10,extent=(x.min()*1e6,x.max()*1e6,z.min()*1e3,z.max()*1e3),aspect='auto')
    # pylab.colorbar()
    ylabel(r"$x\ (\mu m)$", fontsize=16)
    xlabel(r"$z\ (mm)$", fontsize=16)
