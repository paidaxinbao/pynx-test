/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/


/** Elementwise kernel to compute an update of the object and probe from Psi. The update of the object is
* cumulated using atomic operations to avoid memory access conflicts.
* This should be called with a first argument array with a size of nx*ny, i.e. one frame size. Each parallel
* kernel execution treats one pixel, for all frames and all modes.
*/

__device__ void UpdateObjQuadPhaseAtomic(const int i, complexf* psi, complexf *objnew, complexf* probe,
                        float* objnorm, float* cx,  float* cy, const float px, const float f,
                        const int stack_size, const int nx, const int ny, const int nxo, const int nyo,
                        const int nbobj, const int nbprobe, const int npsi, const int padding, const bool interp)
{
  // Coordinates in the probe array
  const int prx = i % nx;
  const int pry = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Use a Tukey window for padded data in near field
  float cpad = 1.0;
  if(padding > 0)
  {
    // Padding factor goes from 0 on the border to 1 at 2*padding pixels from the border
    if(prx<(2*padding))     cpad *= 0.5 * (1 - __cosf(   prx   * 1.57079632679f / padding));
    if(prx>=(nx-2*padding)) cpad *= 0.5 * (1 - __cosf((nx-prx) * 1.57079632679f / padding));
    if(pry<(2*padding))     cpad *= 0.5 * (1 - __cosf(   pry   * 1.57079632679f / padding));
    if(pry>=(ny-2*padding)) cpad *= 0.5 * (1 - __cosf((ny-pry) * 1.57079632679f / padding));

/*
    if(prx<padding || prx>=(nx-padding) || pry<padding || pry>=(ny-padding)) cpad = 0.0f;
    else
    {
      // Padding factor goes from 0 from 1*padding pixels from  the border to 1 at 2*padding pixels from the border
      if(prx<(2*padding))     cpad *= 0.5 * (1 + __cosf(   prx   * 3.141592653f / padding));
      if(prx>=(nx-2*padding)) cpad *= 0.5 * (1 + __cosf((nx-prx) * 3.141592653f / padding));
      if(pry<(2*padding))     cpad *= 0.5 * (1 + __cosf(   pry   * 3.141592653f / padding));
      if(pry>=(ny-2*padding)) cpad *= 0.5 * (1 + __cosf((ny-pry) * 3.141592653f / padding));
    }
*/
  }

  // Object normalisation (ignores subpixel interpolation)
  float prn=0;

  // Apply Quadratic phase factor after far field propagation (ignores subpixel interpolation)
  const float y = (pry - ny/2) * px;
  const float x = (prx - nx/2) * px;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  float s, c;
  __sincosf(tmp , &s, &c);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    for(int j=0;j<npsi;j++)
    {
      complexf o=0;
      for(int iprobe=0 ; iprobe < nbprobe ; iprobe++)
      {
        const complexf pr = probe[i + iprobe * nx * ny];

        // Object normalisation is the same for all modes
        if((iobjmode==0) && (j==0)) prn += dot(pr,pr);

        complexf ps = psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * (nx * ny)];

        ps = complexf(ps.real()*c - ps.imag()*s , ps.imag()*c + ps.real()*s);
        o += complexf(pr.real()*ps.real() + pr.imag()*ps.imag() , pr.real()*ps.imag() - pr.imag()*ps.real());
      }
      // Distribute the computed o on the 4 corners of the interpolated object
      bilinear_atomic_add_c  (objnew, o * cpad,    cx[j] + prx, cy[j] + pry, iobjmode, nxo, nyo, interp);
      if(iobjmode==0)
        bilinear_atomic_add_f(objnorm, prn * cpad, cx[j] + prx, cy[j] + pry, iobjmode, nxo, nyo, interp);
    }
  }
}

// Same for probe update, without need for atomic operations
__device__ void UpdateProbeQuadPhase(const int i, complexf *obj, complexf* probe, complexf* psi, float* probenorm,
                                     float* cx,  float* cy, const float px, const float f, const char firstpass,
                                     const int npsi, const int stack_size, const int nx, const int ny, const int nxo,
                                     const int nyo, const int nbobj, const int nbprobe, const bool interp)
{
  const int prx = i % nx;
  const int pry = i / nx;

  // obj and probe are centered arrays, Psi is fft-shifted

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // probe normalisation
  float prn=0;

  // Apply Quadratic phase factor after far field propagation
  const float y = (pry - ny/2) * px;
  const float x = (prx - nx/2) * px;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  float s, c;
  __sincosf(tmp , &s, &c);

  for(int iprobemode=0; iprobemode<nbprobe; iprobemode++)
  {
    complexf p=0;
    for(int j=0;j<npsi;j++)
    {
      for(int iobjmode=0; iobjmode<nbobj; iobjmode++)
      {
        complexf ps = psi[ipsi + (j + stack_size * (iprobemode + iobjmode * nbprobe) ) * nx * ny];
        ps = complexf(ps.real()*c - ps.imag()*s , ps.imag()*c + ps.real()*s);

        const complexf o = bilinear(obj, cx[j]+prx, cy[j]+pry, iobjmode, nxo, nyo, interp, false);

        if(iprobemode==0) prn += dot(o,o);

        p += complexf(o.real()*ps.real() + o.imag()*ps.imag() , o.real()*ps.imag() - o.imag()*ps.real());
      }
    }
    if(firstpass) probe[i + iprobemode * nx * ny] = p ;
    else probe[i + iprobemode * nx * ny] += p ;
  }

  // all modes have the same normalization
  if(firstpass) probenorm[i] = prn ;
  else probenorm[i] += prn ;

}

// Sum the stack of N object normalisation arrays to the first array
__device__ void SumNnorm(const int iobj, float* objnormN, const int stack_size, const int nxyo)
{
  float n=0;
  for(int i=1;i<stack_size;i++)
  {
    n += objnormN[iobj + i*nxyo];
  }
  objnormN[iobj] += n;
}

// Normalize object.
// The regularization term is used as in: Marchesini et al, Inverse problems 29 (2013), 115009, eq (14)
__device__ void ObjNorm(const int i, float* objnorm, complexf* obj_unnorm, complexf *obj, float *regmax, const float inertia, const int nxyo, const int nbobj)
{
  const float reg = regmax[0] * inertia;
  const float norm = objnorm[i] + reg;
  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
    obj[i + iobjmode*nxyo] = (obj_unnorm[i + iobjmode*nxyo] + reg * obj[i + iobjmode*nxyo]) / norm ;
}

// Normalise object
// The regularization term is used as in: Marchesini et al, Inverse problems 29 (2013), 115009, eq (14)
// Additional restraint on imaginary part to steer towards a zero phase in a given area
__device__ void ObjNormZeroPhaseMask(const int i, float* obj_norm, complexf *obj_new, complexf *obj,
                                     float *zero_phase_mask, float *regmax, const float inertia,
                                     const int nxyo, const int nbobj, const int stack_size)
{
  const float reg = regmax[0] * inertia;
  const float norm_real = obj_norm[i] + reg; // The same norm applies to all object modes
  const float norm_imag = obj_norm[i] + regmax[0] * (inertia + zero_phase_mask[i]);
  for(int iobjmode=0; iobjmode<nbobj; iobjmode++)
  {
     const complexf o = reg * obj[i + iobjmode*nxyo] + obj_new[i + iobjmode * nxyo];
     obj[i + iobjmode*nxyo] = complexf(o.real() / norm_real, o.imag() / norm_imag) ;
  }
}

// Normalize object directly from the stack of N layers of object and norm, to avoid one extra memory r/w
// The regularization term is used as in: Marchesini et al, Inverse problems 29 (2013), 115009, eq (14)
__device__ void ObjNormN(const int i, float* obj_norm, complexf *obj_newN, complexf *obj, float *regmax, const float inertia, const int nxyo, const int nbobj, const int stack_size)
{
  const float reg = regmax[0] * inertia;
  const float norm = obj_norm[i] + reg; // The same norm applies to all object modes
  for(int iobjmode=0; iobjmode<nbobj; iobjmode++)
  {
     complexf o=0;
     for(int j=0;j<stack_size;j++)
     {
       const int ii = i + (j + iobjmode*stack_size) * nxyo;
       o += obj_newN[ii];
     }
    obj[i + iobjmode*nxyo] = (o + reg * obj[i + iobjmode*nxyo]) / norm ;
  }
}

// Normalise object directly from the stack of N layers of object and norm, to avoid one extra memory r/w
// The regularization term is used as in: Marchesini et al, Inverse problems 29 (2013), 115009, eq (14)
// Additional restraint on imaginary part to steer towards a zero phase in a given area
__device__ void ObjNormZeroPhaseMaskN(const int i, float* obj_norm, complexf *obj_newN, complexf *obj,
                                      float *zero_phase_mask, float *regmax, const float inertia,
                                      const int nxyo, const int nbobj, const int stack_size)
{
  const float reg = regmax[0] * inertia;
  const float norm_real = obj_norm[i] + reg; // The same norm applies to all object modes
  const float norm_imag = obj_norm[i] + regmax[0] * (inertia + zero_phase_mask[i]);
  for(int iobjmode=0; iobjmode<nbobj; iobjmode++)
  {
     complexf o = reg * obj[i + iobjmode*nxyo];
     for(int j=0;j<stack_size;j++)
     {
       const int ii = i + (j + iobjmode*stack_size) * nxyo;
       o += obj_newN[ii];
     }
     obj[i + iobjmode*nxyo] = complexf(o.real() / norm_real, o.imag() / norm_imag) ;
  }
}
