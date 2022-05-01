/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Elementwise kernel to compute an update of the object and probe from Psi. This must be called in turns for
* each of the N Psi arrays in the stack, to avoid memory conflicts because of the unknown shift between the N frames.
* This should be called with a first argument array with a size of one frame size. Each parallel
* kernel execution treats one pixel, for one frame and all modes.
*/
//__kernel __attribute__((reqd_work_group_size(16, 1, 1)))
void UpdateObjQuadPhase(const int i, __global float2* psi, __global float2 *objnew, __global float2* probe,
                        __global float* objnorm, const int cx,  const int cy, const int cz,
                        const float px, const float f,
                        const int stack_size, const int nx, const int ny, const int nz,
                        const int nxo, const int nyo, const int nzo, const int nbobj, const int nbprobe)
{
  const int prx = i % nx;
  const int prz = i / (nx*ny);
  const int pry = (i - prz * nx * ny) / nx;
  const int nxyz = nx * ny * nz;
  const int nxyzo = nxo * nyo * nzo;

  // Coordinates in Psi array, fft-shifted (origin at (0,0,0)). Assume nx ny nz are multiple of 2
  const int iz = prz - nz/2 + nz * (prz<(nz/2));
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + (iy + iz * ny) * nx ;

  float2 pr;
  float prn=0;

  for(int iprobe=0;iprobe<nbprobe;iprobe++)
  {// TODO: avoid multiple access of probe value (maybe cached ?)
    pr = probe[i + iprobe*nxyz];
    prn += dot(pr,pr);
  }

  const int iobj0 = cx+prx + nxo*(cy+pry + nyo * (cz + prz));
  objnorm[iobj0] += prn ; // All the object modes have the same probe normalization.

  #if 0
  // Apply Quadratic phase factor before far field propagation
  const float y = (pry - ny/2) * pixel_size;
  const float x = (prx - nx/2) * pixel_size;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);
  #else
  const float c=1, s=0;
  #endif

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    float2 o=0;

    const int iobj  = iobj0 + iobjmode * stack_size * nxyzo;
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      pr = probe[i + iprobe*nxyz]; // TODO: avoid multiple access of probe value (maybe cached ?)
      float2 ps=psi[ipsi + stack_size * (iprobe + iobjmode * nbprobe) * nxyz];
      ps=(float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s);
      o += (float2) (pr.x*ps.x + pr.y*ps.y , pr.x*ps.y - pr.y*ps.x);
    }
    objnew[iobj] += o ;
  }
}

// Same for probe update
void UpdateProbeQuadPhase(const int i, __global float2 *obj, __global float2* probe, __global float2* psi,
                          __global float* probenorm, __global int* cx,  __global int* cy,  __global int* cz,
                          const float px, const float f, const char firstpass, const int npsi, const int stack_size,
                          const int nx, const int ny, const int nz, const int nxo, const int nyo, const int nzo,
                          const int nbobj, const int nbprobe)
{
  const int prx = i % nx;
  const int prz = i / (nx*ny);
  const int pry = (i - prz * nx * ny) / nx;
  const int nxyz = nx * ny * nz;
  const int nxyzo = nxo * nyo * nzo;

  // Coordinates in Psi array, fft-shifted (origin at (0,0,0)). Assume nx ny nz are multiple of 2
  const int iz = prz - nz/2 + nz * (prz<(nz/2));
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + (iy + iz * ny) * nx ;

  float2 o;
  float prn=0;

  for(int j=0;j<npsi;j++)
  {
    const int iobj0 = cx[j] + prx + nxo*(cy[j] + pry + nyo * (cz[j] + prz));
    for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
    {
      const int iobj  = iobj0 + iobjmode * nxyzo;
      o = obj[iobj]; // TODO 1: store object values to avoid repeated memory read
      prn += dot(o,o);
    }
  }

  // all modes have the same normalization
  if(firstpass) probenorm[i] = prn ;
  else probenorm[i] += prn ;

  #if 0
  // Apply Quadratic phase factor before far field propagation
  const float y = (pry - ny/2) * pixel_size;
  const float x = (prx - nx/2) * pixel_size;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);
  #else
  const float c=1, s=0;
  #endif

  for(int iprobemode=0;iprobemode<nbprobe;iprobemode++)
  {
    float2 p=0;
    for(int j=0;j<npsi;j++)
    {
      const int iobj0 = cx[j] + prx + nxo*(cy[j] + pry + nyo * (cz[j] + prz));
      for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
      {
        float2 ps = psi[ipsi + (j + stack_size * (iprobemode + iobjmode * nbprobe) ) * nxyz];
        ps=(float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s);

        const int iobj  = iobj0 + iobjmode * nxyzo;
        o = obj[iobj];

        p += (float2) (o.x*ps.x + o.y*ps.y , o.x*ps.y - o.y*ps.x);
      }
    }
    if(firstpass) probe[i + iprobemode * nxyz] = p ;
    else probe[i + iprobemode * nxyz] += p ;
  }
}

// The following kernel is identical to the 2D non-Bragg ptycho case.

/// Normalize object (or probe)
/// The regularization term is used as in: Marchesini et al, Inverse problems 29 (2013), 115009, eq (14)
void ObjNorm(const int i, __global float2 *obj_unnorm, __global float* objnorm, __global float2 *obj, const float reg, const int nxyzo, const int nbobj)
{
  const float norm = fmax(objnorm[i] + reg, 1e-12f);
  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
    obj[i + iobjmode*nxyzo] = (obj_unnorm[i + iobjmode*nxyzo] + reg * obj[i + iobjmode*nxyzo]) / norm ;
}

/// Normalize object, taking into account a support
/// The regularization term is used as in: Marchesini et al, Inverse problems 29 (2013), 115009, eq (14)
void ObjNormSupport(const int i, __global float2 *obj_unnorm, __global float* objnorm, __global float2 *obj, __global char *support, const float reg, const int nxyzo, const int nbobj)
{
  const float norm = support[i%nxyzo] / fmax(objnorm[i] + reg, 1e-12f) ;
  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
    obj[i + iobjmode*nxyzo] = (obj_unnorm[i + iobjmode*nxyzo] + reg * obj[i + iobjmode*nxyzo]) * norm;
}

