/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Estimate the shift between the back-projected Psi array and the calculated object*probe array.
* This reduction returns ((dopx[i].conj() * dpsi[i]).real(), (dopx[i].conj() * dpsi[i]).real(),
*                          abs(dopx[i])**2, abs(dopy[i])**2)
* where:
*  dopx[i] is the derivative of the object along x, for pixel i, multiplied by the probe (similarly dopy along y)
*  dpsi[i] is the difference between the back-projected Psi array (after Fourier constraints) and obj*probe, at pixel i.
*
* Only the first mode is taken into account.
* if interp=false, nearest pixel interpolation is used for the object*probe calculation.
*/
__device__ my_float4 Psi2PosShift(const int i, complexf* psi, complexf* obj, complexf* probe, float* dx, float* dy,
                                  const int nx, const int ny, const bool interp)
{
  const int ix = i % nx;
  const int iy = i / nx;
  if((ix<(dx[0]+1)) || (ix>=(nx-dx[0]-1)) || (iy<(dy[0]+1)) || (iy>=(ny-dy[0]-1)))
    return my_float4(0,0,0,0);

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  const complexf o = bilinear(obj, ix+dx[0], iy+dy[0], 0, nx, ny, interp, false);
  const complexf pr = probe[i];
  const complexf ps = psi[ipsi];
  // Compute the updated object (just from the first mode) difference
  const complexf dobj = complexf(pr.real()*ps.real() + pr.imag()*ps.imag() ,
                                 pr.real()*ps.imag() - pr.imag()*ps.real()) / dot(pr,pr) - o;

  // Assume we have some buffer before reaching the array border
  // Gradient is calculated with subpixel interpolation.
  const complexf dox =   bilinear(obj, ix+dx[0]+0.5f, iy+dy[0]     , 0, nx, ny, true, false)
                       - bilinear(obj, ix+dx[0]-0.5f, iy+dy[0]     , 0, nx, ny, true, false);
  const complexf doy =   bilinear(obj, ix+dx[0]     , iy+dy[0]+0.5f, 0, nx, ny, true, false)
                       - bilinear(obj, ix+dx[0]     , iy+dy[0]-0.5f, 0, nx, ny, true, false);
  return my_float4(dox.real() * dobj.real() + dox.imag() * dobj.imag(),
                   doy.real() * dobj.real() + doy.imag() * dobj.imag(),
                   dox.real() * dox.real() + dox.imag() * dox.imag(),
                   doy.real() * doy.real() + doy.imag() * doy.imag());
}

/** Compute the shifts
*
*/
__global__ void Psi2PosMerge(my_float4* dxy, float* vdx, float* vdy,
                             const float mult, const float max_shift)
{
  float dx = dxy[0].x / fmaxf(dxy[0].z, 1e-30f) * mult;
  float dy = dxy[0].y / fmaxf(dxy[0].w, 1e-30f) * mult;

  const float dr  = sqrt(dx * dx + dy * dy);

  if(dr > max_shift)
  {
    dx *= max_shift / dr;
    dy *= max_shift / dr;
  }

  vdx[0] += dx;
  vdy[0] += dy;
}
