/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Elementwise kernel to compute an update of the object and probe from Psi. The update of the object is
* done separately for each frame to avoid memory conflicts because of the unknown shift between the frames.
* This should be called with a first argument array with a size of nx*ny, i.e. one frame size. Each parallel
* kernel execution treats one pixel, for one frames and all modes.
* NOTE: this must be optimised (coalesced memory access), as performance is terrible (for fast GPUs)
* NOTE: it is actually faster to use UpdateObjQuadPhaseN (copy to N arrays in //, then sum)
*/
void UpdateObjQuadPhase(const int i, __global float2* psi, __global float2 *objnew, __global float2* probe,
                        __global float* objnorm, const float cx,  const float cy, const float px, const float f,
                        const int stack_size, const int nx, const int ny, const int nxo, const int nyo,
                        const int nbobj, const int nbprobe, const char interp)
{
  // Coordinates in the probe array
  const int prx = i % nx;
  const int pry = i / nx;

  // Pixel coordinates in the Psi array - same as prx, pry, but, fft-shifted (origin at (0,0)). nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  //if(i<140) printf("CL prx=%2d pry=%2d cx=%3d cy=%3d stack_size=%d nx=%d ny=%d nxo=%d nyo=%d nbobj=%d nbprobe=%d, lid=(%d,%d,%d) gid=(%d,%d,%d)\\n", prx, pry, cx, cy, stack_size, nx, ny, nxo, nyo, nbobj, nbprobe,get_local_id(0),get_local_id(1),get_local_id(2),get_global_id(0),get_global_id(1),get_global_id(2));

  float prn=0;

  for(int iprobe=0;iprobe<nbprobe;iprobe++)
  {
    // We could probably ignore interpolation for the normalisation which is averaged over several scan positions.
    const float2 pr = bilinear(probe, prx, pry, iprobe, nx, ny, interp, false);
    prn += dot(pr,pr);
  }

  // Apply Quadratic phase factor after far field propagation (not interpolated ?)
  const float y = (pry - ny/2) * px;
  const float x = (prx - nx/2) * px;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    float2 o=0;
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      const float2 pr = probe[i + iprobe * nx * ny];
      float2 ps = psi[ipsi + stack_size * (iprobe + iobjmode * nbprobe) * nx * ny];
      ps=(float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s);
      o += (float2) (pr.x*ps.x + pr.y*ps.y , pr.x*ps.y - pr.y*ps.x);
    }
    bilinear_atomic_add_c(objnew, o, cx + prx, cy + pry, iobjmode, nxo, nyo, interp);
    if(iobjmode==0)
      bilinear_atomic_add_f(objnorm, prn, cx + prx, cy + pry, iobjmode, nxo, nyo, interp);
  }
}

/** Elementwise kernel to compute an update of the object and probe from Psi. The update of the object is
* cumulated using atomic operations to avoid memory access conflicts.
* This should be called with a first argument array with a size of nx*ny, i.e. one frame size. Each parallel
* kernel execution treats one pixel, for all frames and all modes.
*/

void UpdateObjAtomic(const int i, __global float2* psi, __global float2 *objnew, __global float2* probe,
                     __global float* objnorm, __global float* cx,  __global float* cy, const float px, const float f,
                     const int stack_size, const int nx, const int ny, const int nxo, const int nyo,
                     const int nbobj, const int nbprobe, const int npsi, __global float *scale, const char interp)
{
  // Coordinates in the probe array
  const int prx = i % nx;
  const int pry = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  float prn=0;

  // Object normalisation (ignores subpixel interpolation)
  for(int iprobe=0; iprobe<nbprobe; iprobe++)
  {
    const float2 pr = probe[i + iprobe*nx*ny];
    prn += dot(pr,pr);
  }

  // Apply Quadratic phase factor after far field propagation (ignores subpixel interpolation)
  const float y = (pry - ny/2) * px;
  const float x = (prx - nx/2) * px;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    for(int j=0;j<npsi;j++)
    {
      float2 o=0;
      for(int iprobe=0 ; iprobe < nbprobe ; iprobe++)
      {
        const float2 pr = probe[i + iprobe * nx * ny];
        float2 ps = psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * (nx * ny)];

        ps = (float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s);
        o += (float2)(pr.x*ps.x + pr.y*ps.y , pr.x*ps.y - pr.y*ps.x);
      }
      // Distribute the computed o on the 4 corners of the interpolated object
      bilinear_atomic_add_c  (objnew, o/ native_sqrt(scale[j]),    cx[j] + prx, cy[j] + pry, iobjmode, nxo, nyo, interp);
      if(iobjmode==0)
        bilinear_atomic_add_f(objnorm, prn, cx[j] + prx, cy[j] + pry, iobjmode, nxo, nyo, interp);
    }
  }
}

// Same for probe update
void UpdateProbeQuadPhase(const int i, __global float2 *obj, __global float2* probe, __global float2* psi,
                          __global float* probenorm, __global float* cx,  __global float* cy,
                          const float px, const float f, const char firstpass, const int npsi, const int stack_size,
                          const int nx, const int ny, const int nxo, const int nyo, const int nbobj, const int nbprobe,
                          __global float* scale, const char interp)
{
  const int prx = i % nx;
  const int pry = i / nx;

  // obj and probe are centered arrays, Psi is fft-shifted

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  float prn=0;

  for(int j=0;j<npsi;j++)
  {
    for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
    {
      const float2 o = bilinear(obj, cx[j]+prx, cy[j]+pry, iobjmode, nxo, nyo, interp, false);
      prn += dot(o,o);
    }
  }

  // all modes have the same normalization
  if(firstpass) probenorm[i] = prn ;
  else probenorm[i] += prn ;

  // Apply Quadratic phase factor after far field propagation
  const float y = (pry - ny/2) * px;
  const float x = (prx - nx/2) * px;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  for(int iprobemode=0;iprobemode<nbprobe;iprobemode++)
  {
    float2 p=0;
    for(int j=0;j<npsi;j++)
    {
      for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
      {
        float2 ps = psi[ipsi + (j + stack_size * (iprobemode + iobjmode * nbprobe) ) * nx * ny] / native_sqrt(scale[j]);
        ps=(float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s);

        const float2 o = bilinear(obj, cx[j]+prx, cy[j]+pry, iobjmode, nxo, nyo, interp, false);

        p += (float2) (o.x*ps.x + o.y*ps.y , o.x*ps.y - o.y*ps.x);
      }
    }
    if(firstpass) probe[i + iprobemode * nx * ny] = p ;
    else probe[i + iprobemode * nx * ny] += p ;
  }
}

// Normalize object.
// The regularization term is used as in: Marchesini et al, Inverse problems 29 (2013), 115009, eq (14)
void ObjNorm(const int i, __global float2 *obj_unnorm, __global float* objnorm, __global float2 *obj,
             __global float *normmax, const float inertia, const int nxyo, const int nbobj)
{
  const float reg = normmax[0] * inertia;
  const float norm = fmax(objnorm[i] + reg, 1e-12f);
  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
    obj[i + iobjmode*nxyo] = (obj_unnorm[i + iobjmode*nxyo] + reg * obj[i + iobjmode*nxyo]) / norm ;
}

// Normalize object.
// The regularization term is used as in: Marchesini et al, Inverse problems 29 (2013), 115009, eq (14)
// This version includes a scaling factor to compensate for average frame scaling
void ObjNormScale(const int i, __global float2 *obj_unnorm, __global float* objnorm, __global float2 *obj,
                  __global float *normmax, const float inertia, __global float* scale_sum, const int nb_frame,
                  const int nxyo, const int nbobj)
{
  const float reg = normmax[0] * inertia;
  const float norm = fmax((objnorm[i] + reg)/ native_sqrt(scale_sum[0] / nb_frame), 1e-12f);
  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
    obj[i + iobjmode*nxyo] = (obj_unnorm[i + iobjmode*nxyo] + reg * obj[i + iobjmode*nxyo]) / norm ;
}
