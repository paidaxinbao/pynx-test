/** Compute the center of mass of a 3d complex-valued array.
*
*
*/
__device__ my_float4 center_mass_complex(const int i, complexf *d, const int nx, const int ny,
                                         const int nz, const int power)
{
  const float v = ComplexNormN(d[i], power);
  const int ix = i % nx;
  const int iy = (i % (nx * ny)) / nx;
  const int iz = (i % (nx * ny * nz)) / (nx * ny);
  return my_float4(ix * v, iy * v, iz * v, v);
}

/** Compute the center of mass of a 3d complex-valued array.
*
*
*/
__device__ my_float4 center_mass_fftshift_complex(const int i, complexf *d, const int nx, const int ny,
                                                  const int nz, const int power)
{
  const float v = ComplexNormN(d[i], power);
  const int ix0 = i % nx;
  const int iy0 = (i % (nx * ny)) / nx;
  const int iz0 = (i % (nx * ny * nz)) / (nx * ny);

  const int ix = ix0 - nx/2 + nx * (ix0<(nx/2));
  const int iy = iy0 - ny/2 + ny * (iy0<(ny/2));
  const int iz = iz0 - nz/2 + nz * (iz0<(nz/2));

  return my_float4(ix * v, iy * v, iz * v, v);
}


/** Compute the center of mass of a 3d floating-point array.
*
*
*/
__device__ my_float4 center_mass_float(const int i, float *d, const int nx, const int ny,
                                       const int nz, const int power)
{
  const float v = pow(d[i], (float)power);
  const int ix = i % nx;
  const int iy = (i % (nx * ny)) / nx;
  const int iz = (i % (nx * ny * nz)) / (nx * ny);
  return my_float4(ix * v, iy * v, iz * v, v);
}

/** Compute the center of mass of a 3d floating-point array.
*
*
*/
__device__ my_float4 center_mass_fftshift_float(const int i, float *d, const int nx, const int ny,
                                                const int nz, const int power)
{
  const float v = pow(d[i], (float)power);
  const int ix0 = i % nx;
  const int iy0 = (i % (nx * ny)) / nx;
  const int iz0 = (i % (nx * ny * nz)) / (nx * ny);

  const int ix = ix0 - nx/2 + nx * (ix0<(nx/2));
  const int iy = iy0 - ny/2 + ny * (iy0<(ny/2));
  const int iz = iz0 - nz/2 + nz * (iz0<(nz/2));

  return my_float4(ix * v, iy * v, iz * v, v);
}
