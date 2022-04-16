/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/* Compute the weighted square phase difference between two arrays, with an additional phase ramp.
*
* 3D version
*/
float DeltaPhi3(const int i , __global float2 *d1, __global float2 *d2, __global float *c,
                const int nx, const int ny, const int nz)
{
  const float x = (i % nx) / (float)nx;
  const float y = ((i % (nx * ny)) / nx) / (float)ny;
  const float z = (i / (nx * ny)) / (float)nz;

  float dphi = fabs((float)(atan2(d1[i].y, d1[i].x) - atan2(d2[i].y, d2[i].x) + c[0] + c[1] * z + c[2] * y + c[3] * x));
  dphi = min(dphi, 6.2831853071795862f - dphi);

  return length(d1[i]) * length(d2[i]) * dphi * dphi;
}

/* Compute the weighted square phase difference between two arrays, with an additional phase ramp.
*
* 2D version
*/
float DeltaPhi2(const int i , __global float2 *d1, __global float2 *d2, __global float *c,
                const int nx, const int ny)
{
  const float x = (i % nx) / (float)nx;
  const float y = (i / nx) / (float)ny;

  float dphi = fabs((float)(atan2(d1[i].y, d1[i].x) - atan2(d2[i].y, d2[i].x) + c[0] + c[1] * y + c[2] * x));
  dphi = min(dphi, 6.2831853071795862f - dphi);

  return length(d1[i]) * length(d2[i]) * dphi * dphi;
}
