# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import numpy
import time
import os
import threading

# import multiprocessing

try:
    from scipy import weave

    has_weave = True
except:
    has_weave = False

try:
    from . import cpu

    has_cython = True
except:
    has_cython = False

only_cpu = True
try:
    import pyopencl as cl

    only_cpu = False
except:
    cl = None

try:
    import pycuda.driver as drv
    import pycuda.compiler as compiler

    only_cpu = False
except:
    drv = None


class GPUThreads(object):
    threads = []

    def __init__(self, gpu_name="GTX 295", language="", cl_platform="", nbCPUthread=None, verbose=False):
        self.verbose = verbose
        self.gpu_name = gpu_name.lower()
        self.language = language
        self.cl_platform = cl_platform
        self.cl_devices = []
        self.cuda_devices = []
        self.gpu_name_platform_real = ""
        if self.language.lower() == "opencl" or language == "":
            try:
                tmp = []
                for p in cl.get_platforms():
                    if cl_platform is not None and cl_platform !="":
                        if p.name.find(cl_platform) >= 0:
                            tmp += p.get_devices()
                    else:
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
                    self.threads.append(OpenCLThread_Fhkl(self.cl_devices[i], verbose=verbose))
                    self.threads[-1].setDaemon(True)
                    self.threads[-1].start()
                    if verbose:
                        print("Starting thread:%s"%(self.threads[-1].name))

                    while self.threads[-1].context_init==False:
                        time.sleep(0.01)
                        #print("Waiting for OpenCL initialization, thread#%d"%(len(self.threads)))
                self.gpu_name_platform_real = "OpenCL (%s):%s" % (
                self.cl_devices[0].platform.name, self.cl_devices[0].name)
            except Exception as ex:
                print(ex)
                if language.lower() == "opencl":
                    print("PyNX: Failed importing PyOpenCL, or no platform/graphics card (platform=" + str(cl_platform)
                          + ", gpu_name=" + str(gpu_name) + ") found => using CPU calculations only (WARNING)")
        if language.lower() == "cuda" or (language == "" and len(self.cl_devices) == 0):
            try:
                drv.init()
                cuda_devices = []
                for i in range(drv.Device.count()):
                    if drv.Device(i).name().lower().find(self.gpu_name) >= 0:
                        self.cuda_devices.append(i)
                nbthread = len(self.cuda_devices)
                for i in range(nbthread):
                    self.threads.append(CUDAThread_Fhkl(self.cuda_devices[i], verbose=verbose))
                    self.threads[-1].setDaemon(True)
                    self.threads[-1].start()
                    if verbose:
                        print("Starting thread:%s"%(self.threads[-1].name))
                    while self.threads[-1].context_init==False:
                        time.sleep(0.01)
                        #print("Waiting for OpenCL initialization, thread#%d"%(len(self.threads)))
                self.gpu_name_platform_real = "CUDA:%s" % (drv.Device(self.cuda_devices[0]).name())
            except:
                if language.lower() == "cuda":
                    print(
                    "PyNX: Failed importing PyCUDA, or no graphics card (gpu_name=" + gpu_name + ") found => using CPU calculations only (WARNING)")

        if len(self.cuda_devices) == 0 and len(self.cl_devices) == 0:
            self.gpu_name_platform_real = "CPU (C++/SSE)"
            if nbCPUthread is None:
                try:  # OSX & Linux
                    nbthread = os.sysconf("SC_NPROCESSORS_ONLN")
                except:  # For windows
                    nbthread = int(os.environ["NUMBER_OF_PROCESSORS"])
            else:
                nbthread = nbCPUthread
            for i in range(nbthread):
                self.threads.append(CPUThread_Fhkl(verbose=verbose))
                self.threads[-1].setDaemon(True)
                self.threads[-1].start()
                if i == 0 and has_weave:
                    # Must perform first calculations to ensure all weave modules are pre-compiled (compilation is not thread-safe)
                    self.threads[-1].firstCalc()
        if self.verbose:
            print(
            "Initialized PyNX threads for: " + self.gpu_name + " (language=" + self.language + "," + self.cl_platform + "), REAL=" + self.gpu_name_platform_real)

    def __del__(self):
        if self.verbose: print("Deleting GPUThreads object")
        while len(self) > 0:
            self.threads[-1].join_flag = True
            self.threads[-1].eventStart.set()
            self.threads[-1].join()
            self.threads.pop()

    def __len__(self):
        return len(self.threads)

    def __getitem__(self, i):
        return self.threads[i]


gputhreads = None

###########################################################################################################
###########################################################################################################
#####################################        CUDA KERNELS        ##########################################
###########################################################################################################
###########################################################################################################

# 32,64,128,256 - 64 gives best performance without requiring at least 64 atoms
# TODO: check calculations - R-factor with gold gets hihger with 64 and higher values ??
# TODO:make this configurable
# FHKL_BLOCKSIZE=32

mod_fhkl_str = """
__global__ void CUDA_fhkl(float *fhkl_real,float *fhkl_imag,
                        const float *vx, const float *vy, const float *vz,const long natoms,
                        const float *vh,const float *vk,const float *vl)
{
   #define BLOCKSIZE %d
   #define twopi -6.2831853071795862f
   const unsigned long ix=threadIdx.x+blockDim.x*blockIdx.x;
   const float h=twopi*vh[ix];
   const float k=twopi*vk[ix];
   const float l=twopi*vl[ix];
   float fr=0,fi=0;
   __shared__ float x[BLOCKSIZE];
   __shared__ float y[BLOCKSIZE];
   __shared__ float z[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      __syncthreads();
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
      __syncthreads();
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         float s,c;
         __sincosf(h*x[i] + k*y[i] + l*z[i] , &s,&c);
         fr +=c;
         fi +=s;
      }
      __syncthreads();
   }
   
   /* Take care of remaining atoms */
   if(threadIdx.x<(natoms-at))
   {
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
   }
   __syncthreads();
   for(long i=0;i<(natoms-at);i++)
   {
      float s,c;
      __sincosf(h*x[i] + k*y[i] + l*z[i] , &s,&c);
      fr +=c;
      fi +=s;
   }

   fhkl_real[ix]+=fr;
   fhkl_imag[ix]+=fi;
}
"""

mod_fhklo_str = """
__global__ void CUDA_fhklo(float *fhkl_real,float *fhkl_imag,
                        const float *vx, const float *vy, const float *vz, const float *vocc,
                        const long natoms,
                        const float *vh,const float *vk,const float *vl)
{
   #define BLOCKSIZE %d
   #define twopi -6.2831853071795862f
   const unsigned long ix=threadIdx.x+blockDim.x*blockIdx.x;
   const float h=twopi*vh[ix];
   const float k=twopi*vk[ix];
   const float l=twopi*vl[ix];
   float fr=0,fi=0;
   __shared__ float x[BLOCKSIZE];
   __shared__ float y[BLOCKSIZE];
   __shared__ float z[BLOCKSIZE];
   __shared__ float occ[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      __syncthreads();
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
      occ[threadIdx.x]=vocc[at+threadIdx.x];
      __syncthreads();
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         float s,c;
         __sincosf(h*x[i] + k*y[i] + l*z[i] , &s,&c);
         fr +=occ[i]*c;
         fi +=occ[i]*s;
      }
      __syncthreads();
   }
   /* Take care of remaining atoms */
   if(threadIdx.x<(natoms-at))
   {
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
      occ[threadIdx.x]=vocc[at+threadIdx.x];
   }
   __syncthreads();
   for(long i=0;i<(natoms-at);i++)
   {
      float s,c;
      __sincosf(h*x[i] + k*y[i] + l*z[i] , &s,&c);
      fr +=occ[i]*c;
      fi +=occ[i]*s;
   }
   fhkl_real[ix]+=fr;
   fhkl_imag[ix]+=fi;
}
"""

mod_fhklo_grazing_str = """
__global__ void CUDA_fhklo_grazing(float *fhkl_real,float *fhkl_imag,
                        const float *vx, const float *vy, const float *vz, const float *vocc,
                        const long natoms,
                        const float *vkx,const float *vky,const float *vkzr,const float *vkzi)
{
   #define BLOCKSIZE %d
   #define twopi -6.2831853071795862f
   const unsigned long ix=threadIdx.x+blockDim.x*blockIdx.x;
   const float kx=twopi*vkx[ix];
   const float ky=twopi*vky[ix];
   const float kzr=twopi*vkzr[ix];
   const float kzi=twopi*vkzi[ix];
   float fr=0,fi=0;
   __shared__ float x[BLOCKSIZE];
   __shared__ float y[BLOCKSIZE];
   __shared__ float z[BLOCKSIZE];
   __shared__ float occ[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      __syncthreads();
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
      occ[threadIdx.x]=vocc[at+threadIdx.x];
      __syncthreads();
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         float s,c,atten;
         __sincosf(kx*x[i] + ky*y[i] + kzr*z[i] , &s,&c);
         atten=exp(kzi*z[i]);
         fr +=occ[i]*c*atten;
         fi +=occ[i]*s*atten;
      }
      __syncthreads();
   }
   /* Take care of remaining atoms */
   if(threadIdx.x<(natoms-at))
   {
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
      occ[threadIdx.x]=vocc[at+threadIdx.x];
   }
   __syncthreads();
   for(long i=0;i<(natoms-at);i++)
   {
      float s,c,atten;
      __sincosf(kx*x[i] + ky*y[i] + kzr*z[i] , &s,&c);
      atten=exp(kzi*z[i]);
      fr +=occ[i]*c*atten;
      fi +=occ[i]*s*atten;
   }
   fhkl_real[ix]+=fr;
   fhkl_imag[ix]+=fi;
}
"""

###########################################################################################################
###########################################################################################################
#####################################          CPU CODE          ##########################################
###########################################################################################################
###########################################################################################################


code_CPU_fhkl_xyz = """
  const float PI2         = -6.28318530717958647692528676655900577f;
  for(unsigned long i=0;i<nhkl;i++)
  {
      float fr=0,fi=0;
      const float h=vh[i]*PI2;
      const float k=vk[i]*PI2;
      const float l=vl[i]*PI2;
      const float * __restrict__ px=vx;
      const float * __restrict__ py=vy;
      const float * __restrict__ pz=vz;
      __m128 vfr,vfi,vs,vc;
      float tmp[4];
      for(unsigned long at=0;at<natoms;at+=4)
      {
        float * __restrict__ ptmp=&tmp[0];
        
        // Dangerous ? Order of operation is not guaranteed - but it works...
        sincos_ps(_mm_set_ps(h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++),&vs,&vc);
        if(at==0) 
        {vfr=vc;vfi=vs;}
        else 
        {vfr=_mm_add_ps(vfr,vc);vfi=_mm_add_ps(vfi,vs);}
      }
      float tmp2[4];
      _mm_store_ps(tmp2,vfr);
      for(unsigned int j=0;j<4;++j) fr+=tmp2[j];
      _mm_store_ps(tmp2,vfi);
      for(unsigned int j=0;j<4;++j) fi+=tmp2[j];
      freal[i]=fr;
      fimag[i]=fi;
  }
"""

code_CPU_fhklo_xyz = """
  const float PI2         = -6.28318530717958647692528676655900577f;
  for(unsigned long i=0;i<nhkl;i++)
  {
      float fr=0,fi=0;
      const float h=vh[i]*PI2;
      const float k=vk[i]*PI2;
      const float l=vl[i]*PI2;
      const float * __restrict__ px=vx;
      const float * __restrict__ py=vy;
      const float * __restrict__ pz=vz;
      const float * __restrict__ pocc=vocc;
      __m128 vfr,vfi,vs,vc,vtmp;
      float tmp[4];
      for(unsigned long at=0;at<natoms;at+=4)
      {
        float * __restrict__ ptmp=&tmp[0];
        
        // Dangerous ? Order of operation is not guaranteed - but it works...
        sincos_ps(_mm_set_ps(h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++),&vs,&vc);
        vtmp=_mm_set_ps(*pocc++,*pocc++,*pocc++,*pocc++);
        if(at==0) 
        {vfr=_mm_mul_ps(vtmp,vc);vfi=_mm_mul_ps(vtmp,vs);}
        else 
        {vfr=_mm_add_ps(vfr,_mm_mul_ps(vtmp,vc));vfi=_mm_add_ps(vfi,_mm_mul_ps(vtmp,vs));}
      }
      float tmp2[4];
      _mm_store_ps(tmp2,vfr);
      for(unsigned int j=0;j<4;++j) fr+=tmp2[j];
      _mm_store_ps(tmp2,vfi);
      for(unsigned int j=0;j<4;++j) fi+=tmp2[j];
      freal[i]=fr;
      fimag[i]=fi;
  }
"""

code_CPU_fhkl_grazing_xyz = """
  const float PI2         = -6.28318530717958647692528676655900577f;
  for(unsigned long i=0;i<nhkl;i++)
  {
      float fr=0,fi=0;
      const float h=vh[i]*PI2;
      const float k=vk[i]*PI2;
      const float l=vl[i]*PI2;
      const float li=vli[i]*PI2;
      const float * __restrict__ px=vx;
      const float * __restrict__ py=vy;
      const float * __restrict__ pz=vz;
      __m128 vfr,vfi,vs,vc,vtmp;
      float tmp[4];
      for(unsigned long at=0;at<natoms;at+=4)
      {
        float * __restrict__ ptmp=&tmp[0];
        
        // Dangerous ? Order of operation is not guaranteed - but it works...
        sincos_ps(_mm_set_ps(h* *px++ +k * *py++ + l * *(pz),
                             h* *px++ +k * *py++ + l * *(pz+1),
                             h* *px++ +k * *py++ + l * *(pz+2),
                             h* *px++ +k * *py++ + l * *(pz+3)),&vs,&vc);
        vtmp=exp_ps(_mm_set_ps(*pz++*li,*pz++*li,*pz++*li,*pz++*li));
        if(at==0) 
        {vfr=_mm_mul_ps(vtmp,vc);vfi=_mm_mul_ps(vtmp,vs);}
        else 
        {vfr=_mm_add_ps(vfr,_mm_mul_ps(vtmp,vc));vfi=_mm_add_ps(vfi,_mm_mul_ps(vtmp,vs));}
      }
      float tmp2[4];
      _mm_store_ps(tmp2,vfr);
      for(unsigned int j=0;j<4;++j) fr+=tmp2[j];
      _mm_store_ps(tmp2,vfi);
      for(unsigned int j=0;j<4;++j) fi+=tmp2[j];
      freal[i]=fr;
      fimag[i]=fi;
  }
"""

code_CPU_fhklo_grazing_xyz = """
  const float PI2         = -6.28318530717958647692528676655900577f;
  for(unsigned long i=0;i<nhkl;i++)
  {
      float fr=0,fi=0;
      const float h=vh[i]*PI2;
      const float k=vk[i]*PI2;
      const float l=vl[i]*PI2;
      const float li=vli[i]*PI2;
      const float * __restrict__ px=vx;
      const float * __restrict__ py=vy;
      const float * __restrict__ pz=vz;
      const float * __restrict__ pocc=vocc;
      __m128 vfr,vfi,vs,vc,vtmp;
      float tmp[4];
      for(unsigned long at=0;at<natoms;at+=4)
      {
        float * __restrict__ ptmp=&tmp[0];
        
        // Dangerous ? Order of operation is not guaranteed - but it works...
        sincos_ps(_mm_set_ps(h* *px++ +k * *py++ + l * *(pz),
                             h* *px++ +k * *py++ + l * *(pz+1),
                             h* *px++ +k * *py++ + l * *(pz+2),
                             h* *px++ +k * *py++ + l * *(pz+3)),&vs,&vc);
        vtmp=_mm_mul_ps(_mm_set_ps(*pocc++,*pocc++,*pocc++,*pocc++),exp_ps(_mm_set_ps(*pz++*li,*pz++*li,*pz++*li,*pz++*li)));
        if(at==0) 
        {vfr=_mm_mul_ps(vtmp,vc);vfi=_mm_mul_ps(vtmp,vs);}
        else 
        {vfr=_mm_add_ps(vfr,_mm_mul_ps(vtmp,vc));vfi=_mm_add_ps(vfi,_mm_mul_ps(vtmp,vs));}
      }
      float tmp2[4];
      _mm_store_ps(tmp2,vfr);
      for(unsigned int j=0;j<4;++j) fr+=tmp2[j];
      _mm_store_ps(tmp2,vfi);
      for(unsigned int j=0;j<4;++j) fi+=tmp2[j];
      freal[i]=fr;
      fimag[i]=fi;
  }
"""

###########################################################################################################
###########################################################################################################
#####################################        OpenCL Kernels      ##########################################
###########################################################################################################
###########################################################################################################
CL_FHKL_CODE = """

__kernel __attribute__((reqd_work_group_size(%(block_size)d, 1, 1))) 
void Fhkl(__global float *fhkl_real,__global float *fhkl_imag,
                   __global float *vx,__global float *vy,__global float *vz, const long natoms,
                   __global float *vh,__global float *vk,__global float *vl)
{
   #define BLOCKSIZE %(block_size)d
   #define twopi -6.2831853071795862f
   // Block index
   int bx = get_group_id(0);
   int by = get_group_id(1);

   // Thread index
   int tx = get_local_id(0);
   //int ty = get_local_id(1);
    
   const unsigned long ix=tx+(bx+by*get_num_groups(0))*BLOCKSIZE;
   const float h=twopi*vh[ix];
   const float k=twopi*vk[ix];
   const float l=twopi*vl[ix];
   float fr=0,fi=0;
   __local float x[BLOCKSIZE];
   __local float y[BLOCKSIZE];
   __local float z[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
      barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         // it is faster to calc separately sin and cos,
         // since native_ versions of thoses exist.
         //
         // in CUDA arch, the sin and cos will actually be computed
         // together (http://forums.nvidia.com/index.php?s=ea1305aede92c332301f1d0fe6a53237&showtopic=30069&view=findpost&p=169542)
         const float tmp=h*x[i] + k*y[i] + l*z[i];
         fi +=native_sin(tmp);
         fr +=native_cos(tmp);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   /* Take care of remaining atoms */
   if(tx<(natoms-at))
   {
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   for(long i=0;i<(natoms-at);i++)
   {
      const float tmp=h*x[i] + k*y[i] + l*z[i];
      fi +=native_sin(tmp);
      fr +=native_cos(tmp);
   }

   fhkl_real[ix]+=fr;
   fhkl_imag[ix]+=fi;
}"""

CL_FHKLO_CODE = """
__kernel __attribute__((reqd_work_group_size(%(block_size)d, 1, 1))) 
__kernel void Fhkl(__global float *fhkl_real,__global float *fhkl_imag,
                   __global float *vx,__global float *vy,__global float *vz,__global float *vocc, const long natoms,
                   __global float *vh,__global float *vk,__global float *vl)
{
   #define BLOCKSIZE %(block_size)d
   #define twopi -6.2831853071795862f
   // Block index
   int bx = get_group_id(0);
   int by = get_group_id(1);

   // Thread index
   int tx = get_local_id(0);
   //int ty = get_local_id(1);
    
   const unsigned long ix=tx+(bx+by*get_num_groups(0))*BLOCKSIZE;
   const float h=twopi*vh[ix];
   const float k=twopi*vk[ix];
   const float l=twopi*vl[ix];
   float fr=0,fi=0;
   __local float x[BLOCKSIZE];
   __local float y[BLOCKSIZE];
   __local float z[BLOCKSIZE];
   __local float occ[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
      occ[tx]=vocc[at+tx];
      barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         // it is faster to calc separately sin and cos,
         // since native_ versions of thoses exist.
         //
         // in CUDA arch, the sin and cos will actually be computed
         // together (http://forums.nvidia.com/index.php?s=ea1305aede92c332301f1d0fe6a53237&showtopic=30069&view=findpost&p=169542)
         const float tmp=h*x[i] + k*y[i] + l*z[i];
         fi +=occ[i]*native_sin(tmp);
         fr +=occ[i]*native_cos(tmp);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   /* Take care of remaining atoms */
   if(tx<(natoms-at))
   {
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
      occ[tx]=vocc[at+tx*1];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   for(long i=0;i<(natoms-at);i++)
   {
      const float tmp=h*x[i] + k*y[i] + l*z[i];
      fi +=occ[i]*native_sin(tmp);
      fr +=occ[i]*native_cos(tmp);
   }

   fhkl_real[ix]+=fr;
   fhkl_imag[ix]+=fi;
}"""

CL_FHKL_grazing_CODE = """

__kernel __attribute__((reqd_work_group_size(%(block_size)d, 1, 1))) 
void Fhkl(__global float *fhkl_real,__global float *fhkl_imag,
                   __global float *vx,__global float *vy,__global float *vz, const long natoms,
                   __global float *vkx,__global float *vky,__global float *vkzr,__global float *vkzi)
{
   #define BLOCKSIZE %(block_size)d
   #define twopi -6.2831853071795862f
   // Block index
   int bx = get_group_id(0);
   int by = get_group_id(1);

   // Thread index
   int tx = get_local_id(0);
   //int ty = get_local_id(1);
    
   const unsigned long ix=tx+(bx+by*get_num_groups(0))*BLOCKSIZE;
   const float kx=twopi*vkx[ix];
   const float ky=twopi*vky[ix];
   const float kzr=twopi*vkzr[ix];
   const float kzi=twopi*vkzi[ix];
   float fr=0,fi=0;
   __local float x[BLOCKSIZE];
   __local float y[BLOCKSIZE];
   __local float z[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
      barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         // it is faster to calc separately sin and cos,
         // since native_ versions of thoses exist.
         //
         // in CUDA arch, the sin and cos will actually be computed
         // together (http://forums.nvidia.com/index.php?s=ea1305aede92c332301f1d0fe6a53237&showtopic=30069&view=findpost&p=169542)
         const float tmp=kx*x[i] + ky*y[i] + kzr*z[i];
         const float atten=exp(kzi*z[i]);
         fi +=native_sin(tmp)*atten;
         fr +=native_cos(tmp)*atten;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   /* Take care of remaining atoms */
   if(tx<(natoms-at))
   {
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   for(long i=0;i<(natoms-at);i++)
   {
      const float tmp=kx*x[i] + ky*y[i] + kzr*z[i];
      const float atten=exp(kzi*z[i]);
      fi +=native_sin(tmp)*atten;
      fr +=native_cos(tmp)*atten;
   }

   fhkl_real[ix]+=fr;
   fhkl_imag[ix]+=fi;
}"""

CL_FHKLO_grazing_CODE = """
__kernel __attribute__((reqd_work_group_size(%(block_size)d, 1, 1))) 
__kernel void Fhkl(__global float *fhkl_real,__global float *fhkl_imag,
                   __global float *vx,__global float *vy,__global float *vz,__global float *vocc, const long natoms,
                   __global float *vkx,__global float *vky,__global float *vkzr,__global float *vkzi)
{
   #define BLOCKSIZE %(block_size)d
   #define twopi -6.2831853071795862f
   // Block index
   int bx = get_group_id(0);
   int by = get_group_id(1);

   // Thread index
   int tx = get_local_id(0);
   //int ty = get_local_id(1);
    
   const unsigned long ix=tx+(bx+by*get_num_groups(0))*BLOCKSIZE;
   const float kx=twopi*vkx[ix];
   const float ky=twopi*vky[ix];
   const float kzr=twopi*vkzr[ix];
   const float kzi=twopi*vkzi[ix];
   float fr=0,fi=0;
   __local float x[BLOCKSIZE];
   __local float y[BLOCKSIZE];
   __local float z[BLOCKSIZE];
   __local float occ[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
      occ[tx]=vocc[at+tx];
      barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         // it is faster to calc separately sin and cos,
         // since native_ versions of thoses exist.
         //
         // in CUDA arch, the sin and cos will actually be computed
         // together (http://forums.nvidia.com/index.php?s=ea1305aede92c332301f1d0fe6a53237&showtopic=30069&view=findpost&p=169542)
         const float tmp=kx*x[i] + ky*y[i] + kzr*z[i];
         const float atten=exp(kzi*z[i]);
         fi +=occ[i]*native_sin(tmp)*atten;
         fr +=occ[i]*native_cos(tmp)*atten;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   /* Take care of remaining atoms */
   if(tx<(natoms-at))
   {
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
      occ[tx]=vocc[at+tx];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   for(long i=0;i<(natoms-at);i++)
   {
      const float tmp=kx*x[i] + ky*y[i] + kzr*z[i];
      const float atten=exp(kzi*z[i]);
      fi +=occ[i]*native_sin(tmp)*atten;
      fr +=occ[i]*native_cos(tmp)*atten;
   }

   fhkl_real[ix]+=fr;
   fhkl_imag[ix]+=fi;
}"""


# "Fast" (SSE-optimized) calculation using CPU -requires sse_mathfun.h
class CPUThread_Fhkl(threading.Thread):
    """ (internal)
    Fast (SSE-optimized) calculation using CPU -requires sse_mathfun.h
    """

    def __init__(self, verbose=False):
        threading.Thread.__init__(self)
        # We don't assume anymore that the number of atoms is a multiple of 4
        self.verbose = verbose
        self.dt = 0.0
        self.eventStart = threading.Event()
        self.eventFinished = threading.Event()
        self.join_flag = False
        self.block_size = 32  # would work with just 4 ?

    def run(self):
        if self.verbose: print(self.name, " ...beginning")
        while True:
            self.eventStart.wait()
            if self.join_flag: break
            if self.verbose: print(self.name, " ...got a job !")

            t0 = time.time()
            nhkl = len(self.h.flat)
            # We need a number of atoms which is a multiple of 4
            natoms = (len(self.x.flat) // 4) * 4

            self.fhkl_real = numpy.empty(self.h.shape, dtype=numpy.float32)
            self.fhkl_imag = numpy.empty(self.h.shape, dtype=numpy.float32)
            freal = self.fhkl_real
            fimag = self.fhkl_imag
            vh, vk, vl = self.h, self.k, self.l
            vx, vy, vz = self.x, self.y, self.z

            pth = os.path.dirname(os.path.abspath(__file__))
            if self.occ is None and self.vkzi is None:
                if has_weave:
                    fhkl_xyz = weave.inline("Py_BEGIN_ALLOW_THREADS\n" + code_CPU_fhkl_xyz + "Py_END_ALLOW_THREADS\n",
                                            ['freal', 'fimag', 'vh', 'vk', 'vl', 'vx', 'vy', 'vz', 'nhkl', 'natoms'],
                                            extra_compile_args=[
                                                "-O3 -I" + pth + " -w -ffast-math -msse -msse2  -msse3 -msse4.1 -march=native -mfpmath=sse -fstrict-aliasing -pipe -fomit-frame-pointer -funroll-loops -ftree-vectorize"],
                                            compiler='gcc',
                                            support_code="""#define USE_SSE2
                                                    #include "sse_mathfun.h"
                                                  """,
                                            include_dirs=['./'])
                elif has_cython:
                    cpu.fhkl_cpu(vh,vk,vl,vx,vy,vz,freal,fimag)
                # Only executed if the number of atoms is not a multiple of 4
                for i in range(natoms, len(self.x.flat)):
                    tmp = numpy.exp(2j * numpy.pi * (vx[i] * vh + vy[i] * vk + vz[i] * vl))
                    self.fhkl_real += tmp.real
                    self.fhkl_imag += tmp.imag
            elif self.occ is not None and self.vkzi is None:
                vocc = self.occ
                if has_weave:
                    fhklo_xyz = weave.inline("Py_BEGIN_ALLOW_THREADS\n" + code_CPU_fhklo_xyz + "Py_END_ALLOW_THREADS\n",
                                             ['freal', 'fimag', 'vh', 'vk', 'vl', 'vx', 'vy', 'vz', 'vocc', 'nhkl',
                                              'natoms'],
                                             extra_compile_args=[
                                                 "-O3 -I" + pth + " -w -ffast-math -msse -msse2  -msse3 -msse4.1 -march=native -mfpmath=sse -fstrict-aliasing -pipe -fomit-frame-pointer -funroll-loops -ftree-vectorize"],
                                             compiler='gcc',
                                             support_code="""#define USE_SSE2
                                                    #include "sse_mathfun.h"
                                                  """,
                                             include_dirs=['./'])
                elif has_cython:
                    cpu.fhklo_cpu(vh,vk,vl,vx,vy,vz,vocc,freal,fimag)
                # Only executed if the number of atoms is not a multiple of 4
                for i in range(natoms, len(self.x.flat)):
                    tmp = vocc[i] * numpy.exp(2j * numpy.pi * (vx[i] * vh + vy[i] * vk + vz[i] * vl))
                    self.fhkl_real += tmp.real
                    self.fhkl_imag += tmp.imag
            elif self.occ is None and self.vkzi is not None:
                vli = self.vkzi
                if has_weave:
                    fhkl_grazing_xyz = weave.inline(
                        "Py_BEGIN_ALLOW_THREADS\n" + code_CPU_fhkl_grazing_xyz + "Py_END_ALLOW_THREADS\n",
                        ['freal', 'fimag', 'vh', 'vk', 'vl', 'vli', 'vx', 'vy', 'vz', 'nhkl', 'natoms'],
                        extra_compile_args=[
                            "-O3 -I" + pth + " -w -ffast-math -msse -msse2  -msse3 -msse4.1 -march=native -mfpmath=sse -fstrict-aliasing -pipe -fomit-frame-pointer -funroll-loops -ftree-vectorize"],
                        compiler='gcc',
                        support_code="""#define USE_SSE2
                                                    #include "sse_mathfun.h"
                                                  """,
                        include_dirs=['./'])
                elif has_cython:
                    cpu.fhkl_grazing_cpu(vh,vk,vl,vli,vx,vy,vz,freal,fimag)
                # Only executed if the number of atoms is not a multiple of 4
                for i in range(natoms, len(self.x.flat)):
                    tmp = numpy.exp(2j * numpy.pi * (vx[i] * vh + vy[i] * vk + vz[i] * vl)) * numpy.exp(
                        2 * numpy.pi * vz[i] * vli)
                    self.fhkl_real += tmp.real
                    self.fhkl_imag += tmp.imag
            elif self.occ is not None and self.vkzi is not None:
                vocc = self.occ
                vli = self.vkzi
                if has_weave:
                    fhklo_grazing_xyz = weave.inline(
                        "Py_BEGIN_ALLOW_THREADS\n" + code_CPU_fhklo_grazing_xyz + "Py_END_ALLOW_THREADS\n",
                        ['freal', 'fimag', 'vh', 'vk', 'vl', 'vli', 'vx', 'vy', 'vz', 'vocc', 'nhkl', 'natoms'],
                        extra_compile_args=[
                            "-O3 -I" + pth + " -w -ffast-math -msse -msse2  -msse3 -msse4.1 -march=native -mfpmath=sse -fstrict-aliasing -pipe -fomit-frame-pointer -funroll-loops -ftree-vectorize"],
                        compiler='gcc',
                        support_code="""#define USE_SSE2
                                                    #include "sse_mathfun.h"
                                                  """,
                        include_dirs=['./'])
                elif has_cython:
                    cpu.fhklo_grazing_cpu(vh,vk,vl,vli,vx,vy,vz,vocc,freal,fimag)
                # Only executed if the number of atoms is not a multiple of 4
                for i in range(natoms, len(self.x.flat)):
                    tmp = vocc[i] * numpy.exp(2j * numpy.pi * (vx[i] * vh + vy[i] * vk + vz[i] * vl)) * numpy.exp(
                        2 * numpy.pi * vz[i] * vli)
                    self.fhkl_real += tmp.real
                    self.fhkl_imag += tmp.imag

            self.dt = time.time() - t0
            self.eventStart.clear()
            self.eventFinished.set()
            # MRAtS=nhkl*float(natoms)/self.dt/1e6

    def firstCalc(self):
        if self.verbose:
            print("CPUThread_Fhkl: performing first Weave calculations to ensure weave code is pre-compiled")
        natoms = 4
        nhkl = 4
        vh = numpy.zeros(nhkl, dtype=numpy.float32)
        vk = numpy.zeros(nhkl, dtype=numpy.float32)
        vl = numpy.zeros(nhkl, dtype=numpy.float32)
        vli = numpy.zeros(nhkl, dtype=numpy.float32)
        freal = numpy.zeros(nhkl, dtype=numpy.float32)
        fimag = numpy.zeros(nhkl, dtype=numpy.float32)
        vx = numpy.zeros(natoms, dtype=numpy.float32)
        vy = numpy.zeros(natoms, dtype=numpy.float32)
        vz = numpy.zeros(natoms, dtype=numpy.float32)
        vocc = numpy.zeros(natoms, dtype=numpy.float32)

        pth = os.path.dirname(os.path.abspath(__file__))
        fhkl_xyz = weave.inline("Py_BEGIN_ALLOW_THREADS\n" + code_CPU_fhkl_xyz + "Py_END_ALLOW_THREADS\n",
                                ['freal', 'fimag', 'vh', 'vk', 'vl', 'vx', 'vy', 'vz', 'nhkl', 'natoms'],
                                extra_compile_args=[
                                    "-O3 -I" + pth + " -w -ffast-math -msse -msse2  -msse3 -msse4.1 -march=native -mfpmath=sse -fstrict-aliasing -pipe -fomit-frame-pointer -funroll-loops -ftree-vectorize"],
                                compiler='gcc',
                                support_code="""#define USE_SSE2
                                            #include "sse_mathfun.h"
                                          """,
                                include_dirs=['./'])
        fhklo_xyz = weave.inline("Py_BEGIN_ALLOW_THREADS\n" + code_CPU_fhklo_xyz + "Py_END_ALLOW_THREADS\n",
                                 ['freal', 'fimag', 'vh', 'vk', 'vl', 'vx', 'vy', 'vz', 'vocc', 'nhkl', 'natoms'],
                                 extra_compile_args=[
                                     "-O3 -I" + pth + " -w -ffast-math -msse -msse2  -msse3 -msse4.1 -march=native -mfpmath=sse -fstrict-aliasing -pipe -fomit-frame-pointer -funroll-loops -ftree-vectorize"],
                                 compiler='gcc',
                                 support_code="""#define USE_SSE2
                                            #include "sse_mathfun.h"
                                          """,
                                 include_dirs=['./'])
        fhklo_grazing_xyz = weave.inline(
            "Py_BEGIN_ALLOW_THREADS\n" + code_CPU_fhkl_grazing_xyz + "Py_END_ALLOW_THREADS\n",
            ['freal', 'fimag', 'vh', 'vk', 'vl', 'vli', 'vx', 'vy', 'vz', 'nhkl', 'natoms'],
            extra_compile_args=[
                "-O3 -I" + pth + " -w -ffast-math -msse -msse2  -msse3 -msse4.1 -march=native -mfpmath=sse -fstrict-aliasing -pipe -fomit-frame-pointer -funroll-loops -ftree-vectorize"],
            compiler='gcc',
            support_code="""#define USE_SSE2
                                            #include "sse_mathfun.h"
                                          """,
            include_dirs=['./'])
        fhklo_grazing_xyz = weave.inline(
            "Py_BEGIN_ALLOW_THREADS\n" + code_CPU_fhklo_grazing_xyz + "Py_END_ALLOW_THREADS\n",
            ['freal', 'fimag', 'vh', 'vk', 'vl', 'vli', 'vx', 'vy', 'vz', 'vocc', 'nhkl', 'natoms'],
            extra_compile_args=[
                "-O3 -I" + pth + " -w -ffast-math -msse -msse2  -msse3 -msse4.1 -march=native -mfpmath=sse -fstrict-aliasing -pipe -fomit-frame-pointer -funroll-loops -ftree-vectorize"],
            compiler='gcc',
            support_code="""#define USE_SSE2
                                            #include "sse_mathfun.h"
                                          """,
            include_dirs=['./'])


class CUDAThread_Fhkl(threading.Thread):
    """(internal)
    Class to compute Fhkl in a single thread
    """

    def __init__(self, devID, verbose=False):
        threading.Thread.__init__(self)
        """ Here we assume that the number of hkl is a multiple of 32
        0-padding must already have been done by the calling program
        """
        assert drv.Device.count() >= devID + 1
        self.devID = devID
        self.verbose = verbose
        self.dt = 0.0
        if self.verbose: print(drv.Device(self.devID).name())
        self.eventStart = threading.Event()
        self.eventFinished = threading.Event()
        self.join_flag = False
        self.context_init=False
    def run(self):
        dev = drv.Device(self.devID)
        ctx = dev.make_context()

        # Kernel will be initialized when necessary
        CUDA_fhkl, CUDA_fhk, CUDA_fhklo, CUDA_fhko, CUDA_fhkl_grazing = None, None, None, None, None

        self.block_size = 64  # =dev.get_attribute(drv.device_attribute.WARP_SIZE) # 32
        MULTIPROCESSOR_COUNT = dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)
        self.context_init=True
        if self.verbose: print(self.name, " ...beginning")
        while True:
            self.eventStart.wait()
            if self.join_flag: break
            if self.verbose: print(self.name, " ...got a job !")

            t0 = time.time()
            nhkl0 = numpy.int32(len(self.h.flat))
            natoms = numpy.uint32(len(self.x.flat))

            nhkl = len(self.h)

            self.fhkl_real = 0 * self.h
            self.fhkl_imag = 0 * self.h

            step_atoms = 2 ** 19
            if int(natoms) * nhkl > (2 ** 31 * MULTIPROCESSOR_COUNT):
                # To avoid spending more than 5s per call
                step_atoms = int(2 ** 19 * 2 ** 31 * MULTIPROCESSOR_COUNT / self.block_size / (int(natoms) * nhkl))
                step_atoms *= self.block_size
            steps_nbatoms = list(range(0, natoms, step_atoms))

            if steps_nbatoms[-1] != natoms: steps_nbatoms.append(natoms)

            # "The maximum size of each dimension of a grid of thread blocks is 65535"
            steps_nhkl = list(range(0, nhkl, 65535 * self.block_size))
            if steps_nhkl[-1] != nhkl: steps_nhkl.append(nhkl)

            # if self.verbose: print "Atom ranges:",steps_nbatoms
            for j in range(1, len(steps_nhkl)):  # not always optimal, separate in equal sizes would be better
                for i in range(1, len(steps_nbatoms)):  # not always optimal, separate in equal sizes would be better
                    tmpx = self.x[steps_nbatoms[i - 1]:steps_nbatoms[i]]
                    tmpy = self.y[steps_nbatoms[i - 1]:steps_nbatoms[i]]
                    tmpz = self.z[steps_nbatoms[i - 1]:steps_nbatoms[i]]
                    tmpocc = None
                    if self.occ is not None:
                        tmpocc = self.occ[steps_nbatoms[i - 1]:steps_nbatoms[i]]
                    # if self.verbose: print [steps_nbatoms[i-1],steps_nbatoms[i]]
                    if type(self.occ) == type(None) and type(self.vkzi) == type(None):
                        if type(CUDA_fhkl) == type(None):
                            if self.verbose: print("Compiling CUDA_fhkl (block size=%d)" % self.block_size)
                            mod_fhkl = compiler.SourceModule(mod_fhkl_str % (self.block_size),
                                                             options=["-use_fast_math"])
                            CUDA_fhkl = mod_fhkl.get_function("CUDA_fhkl")
                        CUDA_fhkl(drv.InOut(self.fhkl_real[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                  drv.InOut(self.fhkl_imag[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                  drv.In(tmpx),
                                  drv.In(tmpy),
                                  drv.In(tmpz), numpy.int32(len(tmpx)),
                                  drv.In(self.h[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                  drv.In(self.k[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                  drv.In(self.l[steps_nhkl[j - 1]:steps_nhkl[j]]), block=(self.block_size, 1, 1),
                                  grid=((steps_nhkl[j] - steps_nhkl[j - 1]) // self.block_size, 1))
                    if type(self.occ) != type(None) and type(self.vkzi) == type(None):
                        if type(CUDA_fhklo) == type(None):
                            if self.verbose: print("Compiling CUDA_fhklo (block size=%d)" % self.block_size)
                            mod_fhklo = compiler.SourceModule(mod_fhklo_str % (self.block_size),
                                                              options=["-use_fast_math"])
                            CUDA_fhklo = mod_fhklo.get_function("CUDA_fhklo")

                        CUDA_fhklo(drv.InOut(self.fhkl_real[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                   drv.InOut(self.fhkl_imag[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                   drv.In(tmpx),
                                   drv.In(tmpy),
                                   drv.In(tmpz),
                                   drv.In(tmpocc), numpy.int32(len(tmpx)),
                                   drv.In(self.h[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                   drv.In(self.k[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                   drv.In(self.l[steps_nhkl[j - 1]:steps_nhkl[j]]), block=(self.block_size, 1, 1),
                                   grid=((steps_nhkl[j] - steps_nhkl[j - 1]) // self.block_size, 1))
                    if type(self.occ) != type(None) and type(self.vkzi) != type(None):
                        if type(CUDA_fhkl_grazing) == type(None):
                            if self.verbose: print("Compiling CUDA_fhklo_grazing (block size=%d)" % self.block_size)
                            mod_fhkl_grazing = compiler.SourceModule(mod_fhklo_grazing_str % (self.block_size),
                                                                     options=["-use_fast_math"])
                            CUDA_fhkl_grazing = mod_fhkl_grazing.get_function("CUDA_fhklo_grazing")
                        CUDA_fhkl_grazing(drv.InOut(self.fhkl_real[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          drv.InOut(self.fhkl_imag[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          drv.In(tmpx),
                                          drv.In(tmpy),
                                          drv.In(tmpz),
                                          drv.In(tmpocc),
                                          numpy.int32(len(tmpx)),
                                          drv.In(self.h[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          drv.In(self.k[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          drv.In(self.l[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          drv.In(self.vkzi[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          block=(self.block_size, 1, 1),
                                          grid=((steps_nhkl[j] - steps_nhkl[j - 1]) // self.block_size, 1))
                    if type(self.occ) == type(None) and type(self.vkzi) != type(None):
                        # Also use CUDA_fhklo_grazing, just create a temporary occ=1, performance loss is negligeable
                        if type(CUDA_fhkl_grazing) == type(None):
                            if self.verbose: print("Compiling CUDA_fhklo_grazing (block size=%d)" % self.block_size)
                            mod_fhkl_grazing = compiler.SourceModule(mod_fhklo_grazing_str % (self.block_size),
                                                                     options=["-use_fast_math"])
                            CUDA_fhkl_grazing = mod_fhkl_grazing.get_function("CUDA_fhklo_grazing")

                        CUDA_fhkl_grazing(drv.InOut(self.fhkl_real[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          drv.InOut(self.fhkl_imag[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          drv.In(tmpx),
                                          drv.In(tmpy),
                                          drv.In(tmpz),
                                          drv.In(numpy.ones(tmpz.shape).astype(numpy.float32)),
                                          numpy.int32(len(tmpx)),
                                          drv.In(self.h[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          drv.In(self.k[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          drv.In(self.l[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          drv.In(self.vkzi[steps_nhkl[j - 1]:steps_nhkl[j]]),
                                          block=(self.block_size, 1, 1),
                                          grid=((steps_nhkl[j] - steps_nhkl[j - 1]) // self.block_size, 1))
                self.dt = time.time() - t0
                self.eventStart.clear()
                self.eventFinished.set()
        # MRAtS=nhkl*float(natoms)/self.dt/1e6
        ctx.pop()


class OpenCLThread_Fhkl(threading.Thread):
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
        self.context_init=False
        if verbose:
            print("Thread %s: finished __init__"%(self.name))

    def run(self):
        try_ctx = 5
        while try_ctx > 0:
            try:
                ctx = cl.Context([self.dev])
                try_ctx = 0
            except:
                try_ctx -= 1
                print(
                ("Problem initializing OpenCL context... #try%d/5, wait%3.1fs" % (5 - try_ctx, 0.1 * (5 - try_ctx))))
                time.sleep(0.1 * (5 - try_ctx))
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags
        # Kernel program will be initialized when necessary
        CL_fhkl, CL_fhklo, CL_fhkl_grazing, CL_fhklo_grazing = None, None, None, None

        self.block_size = 64  # good default for NVidia/AMD cards ? 128 leads to random 'out of resources' error (CUDA, GTX 295)
        if self.dev.max_work_group_size < self.block_size:
            self.block_size = self.dev.max_work_group_size

        if self.dev.platform.name == "Apple" and self.dev.name.find("CPU") > 0:
            if self.bug_apple_cpu_workgroupsize_warning:
                if self.verbose:
                    print("WARNING: workaround Apple OpenCL CPU bug: forcing group size=1")
                self.bug_apple_cpu_workgroupsize_warning = False
            self.block_size = 1
        MULTIPROCESSOR_COUNT = self.dev.max_compute_units
        if self.verbose: print(self.name, " ...beginning")

        kernel_params = {"block_size": self.block_size}

        # if "NVIDIA" in queue.device.vendor:
        options = "-cl-mad-enable -cl-fast-relaxed-math"  # -cl-unsafe-math-optimizations
        self.context_init=True
        while True:
            self.eventStart.wait()
            if self.join_flag: break
            if self.verbose: print(self.name, " ...got a job !")

            t0 = time.time()
            nhkl0 = numpy.int32(len(self.h.flat))
            natoms = numpy.uint32(len(self.x.flat))

            nhkl = len(self.h)

            self.fhkl_real = 0 * self.h
            self.fhkl_imag = 0 * self.h

            step_atoms = 2 ** 19
            if int(natoms) * nhkl > (2 ** 31 * MULTIPROCESSOR_COUNT):
                # To avoid spending more than 5s per call (CUDA threshold - may need to be adapted for other platforms)
                step_atoms = int(2 ** 19 * 2 ** 31 * MULTIPROCESSOR_COUNT / self.block_size / (int(natoms) * nhkl))
                step_atoms *= self.block_size
            steps_nbatoms = list(range(0, natoms, step_atoms))

            if steps_nbatoms[-1] != natoms: steps_nbatoms.append(natoms)

            steps_nhkl = list(range(0, nhkl, 65535 * self.block_size))
            if steps_nhkl[-1] != nhkl: steps_nhkl.append(nhkl)

            if self.verbose: print("Atom ranges:", steps_nbatoms)
            for j in range(1, len(steps_nhkl)):  # not always optimal, separate in equal sizes would be better
                fhkl_real_ = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                       hostbuf=self.fhkl_real[steps_nhkl[j - 1]:steps_nhkl[j]], size=0)
                fhkl_imag_ = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                       hostbuf=self.fhkl_imag[steps_nhkl[j - 1]:steps_nhkl[j]], size=0)
                h_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h[steps_nhkl[j - 1]:steps_nhkl[j]],
                               size=0)
                k_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.k[steps_nhkl[j - 1]:steps_nhkl[j]],
                               size=0)
                l_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.l[steps_nhkl[j - 1]:steps_nhkl[j]],
                               size=0)
                for i in range(1, len(steps_nbatoms)):  # not always optimal, separate in equal sizes would be better
                    tmpx = self.x[steps_nbatoms[i - 1]:steps_nbatoms[i]]
                    tmpy = self.y[steps_nbatoms[i - 1]:steps_nbatoms[i]]
                    tmpz = self.z[steps_nbatoms[i - 1]:steps_nbatoms[i]]
                    tmpocc = None
                    if self.occ is not None:
                        tmpocc = self.occ[steps_nbatoms[i - 1]:steps_nbatoms[i]]

                    x_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmpx, size=0)
                    y_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmpy, size=0)
                    z_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmpz, size=0)
                    # if self.verbose: print [steps_nbatoms[i-1],steps_nbatoms[i]]
                    if type(self.occ) == type(None) and type(self.vkzi) == type(None):
                        if type(CL_fhkl) == type(None):
                            if self.verbose: print("Compiling CL_fhkl (block size=%d)" % self.block_size)
                            CL_fhkl = cl.Program(ctx, CL_FHKL_CODE % kernel_params, ).build(options=options)

                        # Somehow the wait() at the end allows doubling the speed (GTX 295, two gpu-in-one) ??
                        CL_fhkl.Fhkl(queue, ((steps_nhkl[j] - steps_nhkl[j - 1], 1)), (self.block_size, 1), fhkl_real_,
                                     fhkl_imag_, x_, y_, z_, numpy.int64(len(tmpx)), h_, k_, l_).wait()

                    if type(self.occ) != type(None) and type(self.vkzi) == type(None):
                        if type(CL_fhklo) == type(None):
                            if self.verbose: print("Compiling CL_fhklo (block size=%d)" % self.block_size)
                            CL_fhklo = cl.Program(ctx, CL_FHKLO_CODE % kernel_params, ).build(options=options)
                        occ_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmpocc, size=0)

                        CL_fhklo.Fhkl(queue, (steps_nhkl[j] - steps_nhkl[j - 1], 1), (self.block_size, 1), fhkl_real_,
                                      fhkl_imag_, x_, y_, z_, occ_, numpy.int64(len(tmpx)), h_, k_, l_).wait()

                    if type(self.occ) != type(None) and type(self.vkzi) != type(None):
                        if type(CL_fhklo_grazing) == type(None):
                            if self.verbose: print("Compiling CL_fhklo_grazing (block size=%d)" % self.block_size)
                            CL_fhklo_grazing = cl.Program(ctx, CL_FHKLO_grazing_CODE % kernel_params, ).build(
                                options=options)
                        vkzi_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                          hostbuf=self.vkzi[steps_nhkl[j - 1]:steps_nhkl[j]],
                                          size=self.vkzi[steps_nhkl[j - 1]:steps_nhkl[j]].nbytes)
                        occ_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmpocc, size=0)

                        CL_fhklo_grazing.Fhkl(queue, (steps_nhkl[j] - steps_nhkl[j - 1], 1), (self.block_size, 1),
                                              fhkl_real_, fhkl_imag_, x_, y_, z_, occ_, numpy.int64(len(tmpx)), h_, k_,
                                              l_, vkzi_).wait()

                    if type(self.occ) == type(None) and type(self.vkzi) != type(None):
                        if type(CL_fhkl_grazing) == type(None):
                            if self.verbose: print("Compiling CL_fhkl_grazing (block size=%d)" % self.block_size)
                            CL_fhkl_grazing = cl.Program(ctx, CL_FHKL_grazing_CODE % kernel_params, ).build(
                                options=options)
                        vkzi_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                          hostbuf=self.vkzi[steps_nhkl[j - 1]:steps_nhkl[j]],
                                          size=self.vkzi[steps_nhkl[j - 1]:steps_nhkl[j]].nbytes)

                        CL_fhkl_grazing.Fhkl(queue, (steps_nhkl[j] - steps_nhkl[j - 1], 1), (self.block_size, 1),
                                             fhkl_real_, fhkl_imag_, x_, y_, z_, numpy.int64(len(tmpx)), h_, k_, l_,
                                             vkzi_).wait()

                cl.enqueue_copy(queue, self.fhkl_real[steps_nhkl[j - 1]:steps_nhkl[j]], fhkl_real_)
                cl.enqueue_copy(queue, self.fhkl_imag[steps_nhkl[j - 1]:steps_nhkl[j]], fhkl_imag_)
                queue.finish()

            self.dt = time.time() - t0
            self.eventStart.clear()
            self.eventFinished.set()


def Fhkl_thread(h, k, l, x, y, z, occ=None, verbose=False, gpu_name="GTX 295", nbCPUthread=None, sz_imag=None,
                language="OpenCL", cl_platform=""):
    """
    Compute          F(hkl)=SUM_i exp(-2j*pi*(h*x_i  + k*y_i  + l*z_i))
    or equivalently: F(k)  =SUM_i exp(-2j*pi*(sx*x_i + sy*y_i + sz*z_i))

    nbCPUthread can be used only for CPU computing, when no GPU is available. nbCPUthread can
    be set to the number of cores available. Using None (the default) makes the program recognize
    the number of available processors/cores

    If sz_imag is not equal to None, then it means that the scattering vector s has an imaginary
    component (absorption), e.g. because of grazing incidence condition.
    
    :param h: numpy vector of h coordinates (reciprocal lattice units)
    :param k: numpy vector of k coordinates (reciprocal lattice units)
    :param l: numpy vector of l coordinates (reciprocal lattice units)
    :param x: numpy vector of x coordinates (fractional coordinates in crystal space)
    :param y: numpy vector of x coordinates (fractional coordinates in crystal space)
    :param z: numpy vector of x coordinates (fractional coordinates in crystal space)
    :param occ: numpy vector of occupancy. Can be None
    :param verbose: if True, will show some computing/threading messages
    :param gpu_name: name of the GPu to be used
    :param nbCPUthread: if using a CPU, number of threads to use
    :param sz_imag: see grazing incidence example
    :param language: either 'cuda', 'opencl' or 'cpu'
    :param cl_platform: the opencl platform to use
    :return: a tuple with (calculated structure factor as a numpy complex64 array, elapsed time in seconds)
        
    """
    global gputhreads
    if gputhreads is None:
        if verbose: print("Fhkl_thread: initilizing gputhreads with GPU=%s, language=%s,cl_platform=%s" % (
        gpu_name, language, cl_platform))
        gputhreads = GPUThreads(gpu_name, nbCPUthread=nbCPUthread, verbose=verbose, language=language,
                                cl_platform=cl_platform)
    elif gputhreads.gpu_name.lower() != gpu_name.lower() or gputhreads.language.lower() != language.lower() \
            or str(gputhreads.cl_platform).lower() != str(cl_platform).lower():
        # GPU has changed, re-initialize
        gputhreads = None
        if verbose: print("Fhkl_thread: initilizing gputhreads with GPU=%s, language=%s,cl_platform=%s" % (
        gpu_name, language, cl_platform))
        gputhreads = GPUThreads(gpu_name, nbCPUthread=nbCPUthread, verbose=verbose, language=language,
                                cl_platform=cl_platform)
    # Make sure (h,k,l) and (x,y,z) all have the same size,
    # force pseudo-1d to 3D, and change type to float32
    if h.shape == k.shape and h.shape == l.shape and h.dtype == numpy.float32 and k.dtype == numpy.float32 and l.dtype == numpy.float32:
        vh = h.ravel()
        vk = k.ravel()
        vl = l.ravel()
    else:
        vh = (h + (k + l) * 0).astype(numpy.float32)
        vk = (k + (h + l) * 0).astype(numpy.float32)
        vl = (l + (h + k) * 0).astype(numpy.float32)
        vh = vh.reshape(len(vh.flat))
        vk = vk.reshape(len(vh.flat))
        vl = vl.reshape(len(vh.flat))

    if type(sz_imag) != type(None):
        if sz_imag.shape == vh.shape and sz_imag.dtype == numpy.float32:
            vkzi = sz_imag.ravel()
        else:
            vkzi = ((h + k + l) * 0 + sz_imag).astype(numpy.float32)
            vkzi = vkzi.reshape(len(vh.flat))
    else:
        vkzi = None

    nhkl0 = numpy.int32(len(vh))
    nhkl = nhkl0

    # We need a multiple of WARPSIZE hkl
    BLOCKSIZE = 1
    for t in gputhreads.threads:
        if t.block_size > BLOCKSIZE:
            BLOCKSIZE = t.block_size
    d, m = divmod(nhkl0, BLOCKSIZE)
    nhkl = nhkl0
    if m != 0:
        if verbose: print("nhkl=%d is not a multiple of %d, using 0-padding" % (nhkl0, BLOCKSIZE))
        nhkl = numpy.int32((d + 1) * BLOCKSIZE)
        vh = numpy.resize(vh, nhkl)
        vk = numpy.resize(vk, nhkl)
        vl = numpy.resize(vl, nhkl)
        if vkzi is not None: vkzi = numpy.resize(vkzi, nhkl)

    # Force float32 type
    if x.shape == y.shape and x.shape == z.shape and x.dtype == numpy.float32 and y.dtype == numpy.float32 and z.dtype == numpy.float32:
        vx = x.ravel()
        vy = y.ravel()
        vz = z.ravel()
    else:
        vx = (x + (y + z) * 0).astype(numpy.float32)
        vy = (y + (x + z) * 0).astype(numpy.float32)
        vz = (z + (x + y) * 0).astype(numpy.float32)

    natoms = numpy.uint32(len(vx.flat))
    vx.resize(natoms)
    vy.resize(natoms)
    vz.resize(natoms)

    vocc = None
    if occ is not None:
        vocc = numpy.resize(occ + (x + y + z) * 0, natoms)
        vocc = (vocc + vx * 0).astype(numpy.float32)

    # Create as many threads as available devices
    t0 = time.time()
    nbthread = len(gputhreads)
    # threads=[]
    for i in range(nbthread):
        a0 = i * int(natoms / nbthread)
        a1 = (i + 1) * int(natoms / nbthread)
        if verbose: print("Thread #", i, [a0, a1])
        if i == (nbthread - 1): a1 = natoms
        tmpocc = vocc
        if occ is not None: tmpocc = vocc[a0:a1]
        gputhreads[i].verbose = verbose
        gputhreads[i].h = vh
        gputhreads[i].k = vk
        gputhreads[i].l = vl
        gputhreads[i].vkzi = vkzi
        gputhreads[i].x = vx[a0:a1]
        gputhreads[i].y = vy[a0:a1]
        gputhreads[i].z = vz[a0:a1]
        gputhreads[i].occ = tmpocc
        gputhreads[i].eventFinished.clear()
        gputhreads[i].eventStart.set()
    for i in range(nbthread):
        gputhreads[i].eventFinished.wait()

        # gpu_thread = CUDAThread_Fhkl(gpu_devices[i],vh,vk,vl,vx[a0:a1],vy[a0:a1],vz[a0:a1],tmpocc,verbose)
        # gpu_thread.start()
        # threads.append(gpu_thread)

    # for t in threads:t.join()
    fhkl = (0 + 0j) * vh
    for i in range(nbthread):
        t = gputhreads[i]
        fhkl += t.fhkl_real + 1j * t.fhkl_imag
        if verbose: print("Thread #%d, dt=%7.5f" % (i, t.dt))
    # for i in xrange(nbthread): del threads[0]
    dt = time.time() - t0
    MRAtS = nhkl * float(natoms) / dt / 1e6 * nbthread
    if nhkl != nhkl0:  # get back to original hkl size
        fhkl = numpy.resize(fhkl, nhkl0)
    return fhkl.reshape((h + k + l).shape), dt


def Fhkl_gold(h, k, l, x, y, z, occ=None, verbose=False, dtype=numpy.float64):
    """
    Compute reference value (using CPU) of:
       F(hkl)=SUM_i exp(2j*pi*(h*x_i + k*y_i + l*z_i))
    """
    vh = (h + (k + l) * 0).astype(dtype)
    vk = (k + (h + l) * 0).astype(dtype)
    vl = (l + (h + k) * 0).astype(dtype)
    nhkl0 = len(vh.flat)

    vx = (x + (y + z) * 0).astype(dtype)
    vy = (y + (x + z) * 0).astype(dtype)
    vz = (z + (x + y) * 0).astype(dtype)
    natoms = len(vx.flat)

    fhkl = 0j * vh
    t0 = time.time()
    if type(occ) == type(None):
        occ = numpy.float32(1.0)
    for i in range(len(vh.flat)):
        fhkl.flat[i] = (occ * numpy.exp(-2j * numpy.pi * (vh.flat[i] * vx + vk.flat[i] * vy + vl.flat[i] * vz))).sum()
    return fhkl, time.time() - t0
