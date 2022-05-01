inline __device__ float pow2(const float v)
{
  return v*v;
}

__device__ float LLKPoisson(const float obs, const float calc)
{
  if(obs<=0.1) return calc; // observed intensity is zero

  return calc - obs + obs * log(obs / calc);
}

__device__ float LLKGaussian(const float obs, const float calc)
{
  return pow2(obs - calc) / (obs + 1);
}

__device__ float LLKEuclidian(const float obs, const float calc)
{
  return 4 * pow2(sqrt(obs) - sqrt(calc));
}

/** Reduction kernel function :
* compute the log-likelihood given observed data following Poisson, Gaussian and Euclidian statistics, and
* a stack of complex data corresponding to the calculated modes.
* Masked pixels (iobs<0) are ignored.
* Reference:  [New Journal of Physics 14 (2012) 063004, doi:10.1088/1367-2630/14/6/063004]
* Note that the Psi array is fft-shifted (origin at (0,0)) but iobs is not
* \param i: the point in the 4D observed intensity array for which the llk is calculated
* \param iobs: the observed intensity array, shape=(nb_proj, nbz, ny, nx)
* \param psi: the calculated complex amplitude, shape=(nb_proj, nbz, nb_obj, nb_probe, ny, nx)
* \param npsi: number of valid frames in the stack, over which the integration is performd (usually equal to
*              stack_size, except for the last stack which may be incomplete (0-padded)
* \param nbmode: number of modes = nb_probe * nb_obj
* \param nxy: ny * nx
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
__device__ my_float4 LLKAll(const int i, float *iobs, complexf *psi, const int nb_mode, const int nx, const int ny)
{
  const float obs = iobs[i];

  if(obs < 0) return my_float4(0);

  const int nxy = nx*ny;
  // Coordinates in iobs array (centered on array)
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;

  // Coordinates in Psi array (centered in (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));

  // Coordinate of first mode in Psi array
  const int i0 = ix1 + iy1 * nx + (i / nxy) * (nxy * nb_mode);

  float calc = 0.0f;
  for(int imode=0;imode<nb_mode;imode++) calc += dot(psi[i0 + imode * nxy],psi[i0 + imode * nxy]);

  return my_float4(LLKPoisson(obs, calc), LLKGaussian(obs, calc), LLKEuclidian(obs, calc), calc);
}
