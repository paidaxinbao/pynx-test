/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/// GPS step 1
__device__ void GPS1(const int i, complexf *obj, complexf* z, const float t, const float sigma_o,
                     const int nx, const int ny, const int nz)
{
  // Gaussian window multiplication for object-space smoothing
  float g = 1.0f;
  if(sigma_o > 0.001f)
  {
    const int ix = i % nx;
    const int iy = (i % (nx * ny)) / nx;
    const int iz = (i % (nx * ny * nz)) / (nx * ny);
    const float qx = (float)(ix - nx * (ix >= (nx / 2))) / (float)nx;
    const float qy = (float)(iy - ny * (iy >= (ny / 2))) / (float)ny;
    const float qz = (float)(iz - nz * (iz >= (nz / 2))) / (float)nz;
    // 2 pi**2 = 19.7392088022
    g = expf(-19.7392088022f * sigma_o * sigma_o  * (qx * qx + qy * qy + qz * qz));
  }

  obj[i] = z[i] - (g * t) * obj[i] ;
}

/// GPS step 2
__device__ void GPS2(const int i, complexf *obj, complexf* z, const float epsilon)
{
  obj[i] = ((float)1.0f - epsilon) * obj[i] + epsilon * z[i] ;
}

/// GPS step 3
__device__ void GPS3(const int i, complexf *obj, complexf* z)
{
  const complexf zk = z[i];
  const complexf zk1 = obj[i];
  z[i] = zk1;
  obj[i] = 2.0f * zk1 - zk;
}

/// GPS step 4
__device__ void GPS4(const int i, complexf *obj, complexf* y, signed char *support, const float s, const float sigma_f,
                     signed char positivity, const int nx, const int ny, const int nz)
{
  // Gaussian window multiplication for Fourier-space smoothing
  float g = 1.0f;
  if(sigma_f > 0.001f)
  {
    const int ix = i % nx;
    const int iy = (i % (nx * ny)) / nx;
    const int iz = (i % (nx * ny * nz)) / (nx * ny);
    const float qx = (float)(ix - nx * (ix >= (nx / 2))) / (float)nx;
    const float qy = (float)(iy - ny * (iy >= (ny / 2))) / (float)ny;
    const float qz = (float)(iz - nz * (iz >= (nz / 2))) / (float)nz;
    // 2 pi**2 = 19.7392088022
    g = expf(-19.7392088022f * sigma_f * sigma_f  * (qx * qx + qy * qy + qz * qz));
  }

  complexf o = g * (y[i] + s * obj[i]);

  // Support projection
  if(positivity) obj[i] = g * ((support[i]==0) || (o.real() < 0)) * o;
  else obj[i] = g * (support[i]==0) * o;
}

