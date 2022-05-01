/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

// Compute 2 * P * O - Psi , with the quadratic phase factor
__device__ void ObjectProbePsiDM1(const int i, complexf* psi, complexf *obj, complexf* probe,
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
      const complexf p = 2.0f * probe[i + iprobe*nxy];

      for(int j=0;j<npsi;j++)
      {
        const complexf o = bilinear(obj, cx[j]+prx, cy[j]+pry, iobjmode, nxo, nyo, interp, false);
        complexf ps = complexf(o.real()*p.real() - o.imag()*p.imag() , o.real()*p.imag() + o.imag()*p.real());
        const int ii = ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy;
        psi[ii] = complexf(ps.real()*c - ps.imag()*s , ps.imag()*c + ps.real()*s) - psi[ii];
      }
      for(int j=npsi;j<stack_size;j++)
      {
        // Need this for dummy frames at the end of the stack (to have a multiple of 16), or LLK would be incorrect
        psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy] = complexf(0,0);
      }
    }
  }
}

/** Update Psi (with quadratic phase)
* Psi(n+1) = Psi(n) - P*O + Psi_calc ; where Psi_calc=Psi_fourier is (2*P*O - Psi(n)) after applying Fourier constraints
*/
__device__ void ObjectProbePsiDM2(const int i, complexf* psi, complexf* psi_fourier, complexf *obj, complexf* probe,
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
        const int ii = ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy;
        psi[ii] += psi_fourier[ii] - complexf(ps.real()*c - ps.imag()*s , ps.imag()*c + ps.real()*s);
      }
      for(int j=npsi;j<stack_size;j++)
      {
        // Need this for dummy frames at the end of the stack (to have a multiple of 16), or LLK would be incorrect
        // TODO: only loop on valid frames in llk evaluation
        psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy] = complexf(0,0);
      }
    }
  }
}
