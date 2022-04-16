__device__ int ixyz(const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
{
    int iix = (ix + nx) % nx;
    int iiy = (iy + ny) % ny;
    int iiz = (iz + nz) % nz;
    return (iiz * ny + iiy) * nx + iix;
}

/** Interpolate masked intensities using inverse distance weighting.
*
* This is not optimised (no explicit coalesced memory transfers, etc..) for testing.
* It is supposed to be only executed once so optimisation is not critical.
*
* In the input and output iobs array, the following values are used:
* - iobs >=0 are standard observed values
* - -1e19<iobs<=1 are observed values tagged to be used for free log-likelihood evaluation,
*   i.e. the >= iobs value has been replaced by -1-iobs
* - values <= -1e19 correspond to masked pixels where no valid intensity has been recorded
*   by the detector. After this function, the values will be replaced by the interpolated
*   estimation, plus 1, multiplied by -1e19 (so that values up to 1e19 can be recorded using
*   a 32-bit float)
* - values <=-1e38 (NB: min value is -1.17e-38 in float32, do not go below) remain masked
*   after this function, as it was not possible to estimate them.
* \param iobs: the observed intensity array, with <0 values used for special tagging
* \param i: index of the current pixel considered
* \param k: the kernel half size - pixels from i-k to i+k will be used along each dimension
* \param n: the inverse distance weighting will be computed as 1/d**n
* \param nx,ny,nz: size along each dimension. If nz==1, only 2D interpolation is done
*/
__device__ void mask_interp_dist(const int i, float *iobs, const int k, const int dist_n,
                                 const int nx, const int ny, const int nz)
{
  const int ix = i % nx;
  const int iz = i / (nx*ny);
  const int iy = (i - iz * nx * ny) / nx;

  const float v0 = iobs[i];
  if(v0>-1e19f) return;
  float v1 = 0.0f;
  float v1n = 0.0f;
  if(nz>1)
  {
    for(int dx=-k;dx<=k;++dx)
    {
      for(int dy=-k;dy<=k;++dy)
      {
        for(int dz=-k;dz<=k;++dz)
        {
          float v = iobs[ixyz(ix+dx,iy+dy,iz+dz, nx, ny, nz)];
          if(v>-1e19f)
          {
            if(v<0.0f) v = -v -1; // correct value, masked for free log-likelihood
            // The 1e-6 should not be useful (null distance implies value is not masked)
            const float w = 1 / pow(sqrt(fmaxf(dx*dx + dy*dy + dz*dz, 1e-6f)), (float)dist_n);
            v1 += w * v;
            v1n += w;
          }
        }
      }
    }
  }
  if(v1n > 0.0f)
  {
    v1 /= v1n;
    iobs[i] = -1e19 * (v1+1);
  }
  else iobs[i] = -1.01e38f;
}
