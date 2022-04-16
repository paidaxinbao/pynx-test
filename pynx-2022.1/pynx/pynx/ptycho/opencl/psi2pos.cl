/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Estimate the shift between the back-project Psi array and the calculated object*probe array.
* This reduction returns ((dopx[i].conj() * dpsi[i]).real(), (dopx[i].conj() * dpsi[i]).real(),
*                          abs(dopx[i])**2, abs(dopy[i])**2)
* where:
*  dopx[i] is the derivative of the object along x, for pixel i, multiplied by the probe (simlarly dopy along y)
*  dpsi[i] is the difference between the back-projected Psi array (after Fourier constraints) and obj*probe, at pixel i.
*
* Only the first modes are taken into account.
*/
float4 Psi2PosShift(const int i, __global float2* psi, __global float2 *obj, __global float2* probe,
                    __global float* cx, __global float* cy, const float pixel_size, const float f, const int nx,
                    const int ny, const int nxo, const int nyo, __global float* scale, const int ii, const char interp)
{
  const int prx = i % nx;
  const int pry = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Apply Quadratic phase factor after far field back-propagation
  const float y = (pry - ny/2) * pixel_size;
  const float x = (prx - nx/2) * pixel_size;
  const float tmp = -f*(x*x+y*y);

  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  const float2 o = bilinear(obj, cx[ii]+prx, cy[ii]+pry, 0, nxo, nyo, interp, false) * native_sqrt(scale[ii]);
  const float2 p = probe[i];
  const float2 dpsi = (float2)(psi[ipsi].x*c - psi[ipsi].y*s - (o.x*p.x - o.y*p.y) ,
                               psi[ipsi].y*c + psi[ipsi].x*s - (o.x*p.y + o.y*p.x));

  // Assume we have some buffer before reaching the array border
  // Gradient is calculated with subpixel interpolation.
  const float2 dox =   bilinear(obj, cx[ii]+prx+0.5f, cy[ii]+pry     , 0, nxo, nyo, true, false)
                     - bilinear(obj, cx[ii]+prx-0.5f, cy[ii]+pry     , 0, nxo, nyo, true, false);
  const float2 doy =   bilinear(obj, cx[ii]+prx     , cy[ii]+pry+0.5f, 0, nxo, nyo, true, false)
                     - bilinear(obj, cx[ii]+prx     , cy[ii]+pry-0.5f, 0, nxo, nyo, true, false);
  const float2 dopx = (float2)(dox.x * p.x - dox.y * p.y, dox.x * p.y + dox.y * p.x);
  const float2 dopy = (float2)(doy.x * p.x - doy.y * p.y, doy.x * p.y + doy.y * p.x);
  return (float4)(dopx.x * dpsi.x + dopx.y * dpsi.y, dopy.x * dpsi.x + dopy.y * dpsi.y,
                  dopx.x * dopx.x + dopx.y * dopx.y, dopy.x * dopy.x + dopy.y * dopy.y);
}

/** Compute the shifts, and return the average
*
*/
float2 Psi2PosRed(const int i, __global float4* dxy, const float mult, const float max_shift,
                  const float min_shift, const float threshold, const int nb)
{
  float dx = dxy[i].x / fmax(dxy[i].z, 1e-30f) * mult;
  float dy = dxy[i].y / fmax(dxy[i].w, 1e-30f) * mult;

  if(dxy[i].z < threshold) dx = 0;
  if(dxy[i].w < threshold) dy = 0;

  const float dr  = native_sqrt(dx * dx + dy * dy);

  if(dr < min_shift)
  {
    dxy[i].x = 0;
    dxy[i].y = 0;
    return (float2)(0,0);
  }
  if(dr > max_shift)
  {
    dx *= max_shift / dr;
    dy *= max_shift / dr;
  }

  dxy[i].x = dx;
  dxy[i].y = dy;
  return (float2)(dx / nb, dy / nb);
}
