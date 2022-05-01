/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

__device__ void ObjectProbeMultQuadPhase(const int i, complexf* psi, complexf *obj, complexf* probe,
                              float* cx, float* cy,
                              const float pixel_size, const float f, const int npsi, const int stack_size,
                              const int nx, const int ny, const int nxo, const int nyo,
                              const int nbobj, const int nbprobe, const bool interp)
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
  float s, c;
  __sincosf(tmp , &s, &c);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      const complexf p = probe[i + iprobe*nxy];

      for(int j=0;j<npsi;j++)
      {
        // Bilinear interpolation for subpixel shift
        const complexf o = bilinear(obj, cx[j]+prx, cy[j]+pry, iobjmode, nxo, nyo, interp, false);
        complexf ps = complexf(o.real()*p.real() - o.imag()*p.imag() , o.real()*p.imag() + o.imag()*p.real());
        psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy] = complexf(ps.real()*c - ps.imag()*s , ps.imag()*c + ps.real()*s);
      }
      for(int j=npsi;j<stack_size;j++)
      {
        // Need this for dummy frames at the end of the stack (to have a multiple of 16), or Chi2 would be incorrect
        psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy ] = complexf(0,0);
      }
    }
  }
}
