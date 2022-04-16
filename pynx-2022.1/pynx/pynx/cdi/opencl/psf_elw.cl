inline float pow2(const float v)
{
  return v*v;
}

void Gaussian(const int i, __global float *d, const float fwhm,
              const int nx, const int ny, const int nz)
{
  int ix = i % nx;
  int iy = (i % (nx * ny)) / nx;
  int iz = (i % (nx * ny * nz)) / (nx * ny);

  ix = ix - nx + nx * (ix < (nx/2));
  iy = iy - ny + ny * (iy < (ny/2));
  iz = iz - nz + nz * (iz < (nz/2));

  const float sigma = fwhm / 2.3548f;
  const float tmp = -(ix*ix + iy*iy + iz*iz)/(2 * sigma * sigma);

  d[i] = 1 / (sigma * native_sqrt(6.2831853071795862f)) * native_exp(tmp);
}

void Lorentzian(const int i, __global float *d, const float fwhm,
                const int nx, const int ny, const int nz)
{
  int ix = i % nx;
  int iy = (i % (nx * ny)) / nx;
  int iz = (i % (nx * ny * nz)) / (nx * ny);

  ix = ix - nx + nx * (ix < (nx/2));
  iy = iy - ny + ny * (iy < (ny/2));
  iz = iz - nz + nz * (iz < (nz/2));

  d[i] = 2 / 3.141592653f * fwhm / (4*(ix*ix + iy*iy +iz*iz) + fwhm * fwhm);
}

void PseudoVoigt(const int i,__global float *d, const float fwhm, const float eta,
                 const int nx, const int ny, const int nz)
{
  int ix = i % nx;
  int iy = (i % (nx * ny)) / nx;
  int iz = (i % (nx * ny * nz)) / (nx * ny);

  ix = ix - nx + nx * (ix < (nx/2));
  iy = iy - ny + ny * (iy < (ny/2));
  iz = iz - nz + nz * (iz < (nz/2));

  const float sigma = fwhm / 2.3548f;
  const float tmp = -(ix*ix + iy*iy + iz*iz)/(2 * sigma * sigma);
  const float g = 1 / (sigma * native_sqrt(6.2831853071795862f)) * native_exp(tmp);

  const float l = 2 / 3.141592653f * fwhm / (4*(ix*ix + iy*iy +iz*iz) + fwhm * fwhm);
  d[i] = eta * l + (1-eta) * g;
}

void PSF3_Hann(const int i, __global float *psf, __global float *d,
               const int nx, const int ny, const int nz, const float scale)
{
  int ix = i % nx;
  int iy = (i % (nx * ny)) / nx;
  int iz = (i % (nx * ny * nz)) / (nx * ny);

  ix = ix - nx + nx * (ix < (nx/2));
  iy = iy - ny + ny * (iy < (ny/2));
  iz = iz - nz + nz * (iz < (nz/2));

  // Hann window
  const float qx = (float)(ix - nx * (ix >= (nx / 2))) * 3.14159265f / (float)(nx-1);
  const float qy = (float)(iy - ny * (iy >= (ny / 2))) * 3.14159265f / (float)(ny-1);
  float qz = 0.0f;
  if(nz>1) qz = (float)(iz - nz * (iz >= (nz / 2))) * 3.14159265f / (float)(nz-1);
  const float g = pow2(native_cos(qx) * native_cos(qy) * native_cos(qz));

  psf[i] = fmax(psf[i] * fmax(d[i],1e-6f) * g, 1e-30f) * scale;
}

void PSF3_Tukey(const int i, __global float *psf, __global float *d,
                const float alpha, const int nx, const int ny, const int nz, const float scale)
{
  int ix = i % nx;
  int iy = (i % (nx * ny)) / nx;
  int iz = (i % (nx * ny * nz)) / (nx * ny);

  // Coordinates relative to the center of the array (easier to compute the Tukey window

  // Tukey window
  float g = 1.0f;

  const float qx = fabs((float)(ix - nx * 0.5f) / (float)(nx-1));
  if(qx < (0.5f * alpha)) g *= 0.5 - 0.5 * native_cos(3.14159265f * qx / (0.5f * alpha));

  const float qy = fabs((float)(iy - ny * 0.5f) / (float)(ny-1));
  if(qy < (0.5f * alpha)) g *= 0.5 - 0.5 * native_cos(3.14159265f * qy / (0.5f * alpha));

  if(nz>1)
  {
    const float qz = fabs((float)(iz - nz * 0.5f) / (float)(nz-1));
    if(qz < (0.5f * alpha)) g *= 0.5 - 0.5 * native_cos(3.14159265f * qz / (0.5f * alpha));
  }

  psf[i] = fmax(psf[i] * fmax(d[i],1e-6f) * g, 1e-30f) * scale;
}
