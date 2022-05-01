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
__device__ float DeltaPhi3(const int i , complexf *d1, complexf *d2, float *c,
                const int nx, const int ny, const int nz)
{
  const float x = (i % nx) / (float)nx;
  const float y = ((i % (nx * ny)) / nx) / (float)ny;
  const float z = (i / (nx * ny)) / (float)nz;

  float dphi = fabs((float)(atan2(d1[i].imag(), d1[i].real())
                            - atan2(d2[i].imag(), d2[i].real())
                            + c[0] + c[1] * z + c[2] * y + c[3] * x));
  dphi = fminf(dphi, 6.2831853071795862f - dphi);

  return abs(d1[i]) * abs(d2[i]) * dphi * dphi;
}

/* Compute the weighted square phase difference between two arrays, with an additional phase ramp.
*
* 2D version
*/
__device__ float DeltaPhi2(const int i , complexf *d1, complexf *d2, float *c,
                const int nx, const int ny)
{
  const float x = (i % nx) / (float)nx;
  const float y = (i / nx) / (float)ny;

  float dphi = fabs((float)(atan2(d1[i].imag(), d1[i].real())
                            - atan2(d2[i].imag(), d2[i].real())
                            + c[0] + c[1] * y + c[2] * x));
  dphi = fminf(dphi, 6.2831853071795862f - dphi);

  return abs(d1[i]) * abs(d2[i]) * dphi * dphi;
}
