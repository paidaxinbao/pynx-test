/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

void ObjectProbeMultQuadPhase(const int i, __global float2* psi, __global float2 *obj, __global float2* probe,
                              __global int* cx, __global int* cy, __global int* cz,
                              const float pixel_size, const float f, const int npsi, const int stack_size,
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
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      const float2 p = probe[i + iprobe * nxyz];

      for(int j=0;j<npsi;j++)
      {
        // TODO: use a __local array for object values to minimize memory transfers ? Or trust the cache.
        const float2 o = obj[cx[j] + prx + nxo * (cy[j] + pry + nyo * (cz[j] + prz)) + iobjmode * nxyzo];
        float2 ps=(float2)(o.x*p.x - o.y*p.y , o.x*p.y + o.y*p.x);
        psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxyz] = (float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s);
      }
      for(int j=npsi;j<stack_size;j++)
      {
        // Need this for dummy frames at the end of the stack (to have a multiple of 16), or Chi2 would be incorrect
        // psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxyz ] = (float2)0;
        psi[i] = 0;
      }
    }
  }
}
