/** Calculate intensity from a complex calculated array and a point spread function kernel
*
*/
__device__ float CalcIntensityPSF(const int i, complexf* dcalc, float* psf,
                                  const int n, const int nx, const int ny, const int nz)
{
  const int ix = i % nx;
  const int iz = i / (nx*ny);
  const int iy = (i - iz * nx * ny) / nx;

  // Convolve calculated intensities.
  // No effort is made to share data transfer, hoping transfers are naturally coalesced and cached..
  // Bottom line: this is dead slow - need either to collaborate between threads or use FFT convolution.
  float ic=0;
  if(nz>1)
  {// 3D
    for(int kix=-n;kix<=n;kix++)
    {
      for(int kiy=-n;kiy<=n;kiy++)
      {
        for(int kiz=-n;kiz<=n;kiz++)
        {
          const float vpsf = psf[kix + n + (2*n+1) * (kiy + n + (2*n+1) * (kiz + n))];
          const int i1 = (ix + kix + nx) % nx + nx *((iy + kiy + ny) % ny + ny *((iz + kiz + nz) % nz));
          ic += dot(dcalc[i1], dcalc[i1]) * vpsf;
        }
      }
    }
  }
  else
  {// 2D
    for(int kix=-n;kix<=n;kix++)
    {
      for(int kiy=-n;kiy<=n;kiy++)
      {
        const float vpsf = psf[kix + n + (2*n+1) * (kiy + n )];
        const int i1 = (ix + kix + nx) % nx + nx *((iy + kiy + ny) % ny );
        ic += dot(dcalc[i1], dcalc[i1]) * vpsf;
      }
    }
  }
  return ic;
}

/** Apply the observed amplitude (from intensities) to a complex calculated array, taking into account
* convolution from a point spread function, and a mask.
*/
__device__ void IntensityPSF(const int i, float *icalc, complexf* dcalc, float* psf,
                             const int n, const int nx, const int ny, const int nz)
{
  icalc[i] = CalcIntensityPSF(i, dcalc, psf, n, nx, ny, nz);
}
