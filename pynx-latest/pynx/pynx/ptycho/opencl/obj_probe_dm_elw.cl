/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

// Compute 2 * P * O - Psi , with the quadratic phase factor
void ObjectProbePsiDM1(const int i, __global float2* psi, __global float2 *obj, __global float2* probe,
                       __global float* cx, __global float* cy,
                       const float pixel_size, const float f, const int npsi, const int stack_size,
                       const int nx, const int ny, const int nxo, const int nyo,
                       const int nbobj, const int nbprobe, __global float* scale, const char interp)
{
  const int prx = i % nx;
  const int pry = i / nx;
  const int nxy = nx * ny;

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Apply Quadratic phase factor before far field propagation
  const float y = (pry - ny/2) * pixel_size;
  const float x = (prx - nx/2) * pixel_size;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      const float2 p = 2 * probe[i + iprobe*nxy];

      for(int j=0;j<npsi;j++)
      {
        // Bilinear interpolation for subpixel shift
        const float2 o = bilinear(obj, cx[j]+prx, cy[j]+pry, iobjmode, nxo, nyo, interp, false) * native_sqrt(scale[j]);
        float2 ps=(float2)(o.x*p.x - o.y*p.y , o.x*p.y + o.y*p.x);
        const int ii = ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy;
        psi[ii] = (float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s) - psi[ii];
      }
      for(int j=npsi;j<stack_size;j++)
      {
        // Need this for dummy frames at the end of the stack (to have a multiple of 16), or LLK would be incorrect
        psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy] = (float2)0;
      }
    }
  }
}

/** Update Psi (with quadratic phase)
* Psi(n+1) = Psi(n) - P*O + Psi_calc ; where Psi_calc=Psi_fourier is (2*P*O - Psi(n)) after applying Fourier constraints
*/
void ObjectProbePsiDM2(const int i, __global float2* psi, __global float2* psi_fourier, __global float2 *obj, __global float2* probe,
                       __global float* cx, __global float* cy,
                       const float pixel_size, const float f, const int npsi, const int stack_size,
                       const int nx, const int ny, const int nxo, const int nyo,
                       const int nbobj, const int nbprobe, __global float* scale, const char interp)
{
  const int prx = i % nx;
  const int pry = i / nx;
  const int nxy = nx * ny;

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Apply Quadratic phase factor before far field propagation
  const float y = (pry - ny/2) * pixel_size;
  const float x = (prx - nx/2) * pixel_size;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      const float2 p = probe[i + iprobe*nxy];

      for(int j=0;j<npsi;j++)
      {
        // Bilinear interpolation for subpixel shift
        const float2 o = bilinear(obj, cx[j]+prx, cy[j]+pry, iobjmode, nxo, nyo, interp, false) * native_sqrt(scale[j]);
        float2 ps=(float2)(o.x*p.x - o.y*p.y , o.x*p.y + o.y*p.x);
        const int ii = ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy;
        psi[ii] += psi_fourier[ii] - (float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s);
      }
      for(int j=npsi;j<stack_size;j++)
      {
        // Need this for dummy frames at the end of the stack (to have a multiple of 16), or LLK would be incorrect
        psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy] = (float2)0;
      }
    }
  }
}
