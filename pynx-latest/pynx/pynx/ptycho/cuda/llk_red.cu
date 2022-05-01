inline __device__ float pow2(const float v)
{
  return v*v;
}

inline __device__ float pow3(const float v)
{
  return v*v*v;
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
  return 4 * pow2(sqrtf(obs) - sqrtf(calc));
}

/** Reduction kernel function :
* compute the log-likelihood given observed data following Poisson, Gaussian and Euclidian statistics, and
* a stack of complex data corresponding to the calculated modes.
* Masked pixels (negative intensities) are ignored.
* Reference:  [New Journal of Physics 14 (2012) 063004, doi:10.1088/1367-2630/14/6/063004]
* \param i: the point in the 3D observed intensity array for which the llk is calaculated
* \param iobs: the observed in tensity array, shape=(stack_size, ny, nx)
* \param psi: the calculated complex amplitude, shape=(nb_obj, nb_probe, stack_size, ny, nx)
* \param background: the incoherent background, of shape (ny, nx)
* \param nbmode: number of modes = nb_probe * nb_obj
* \param nxy: number of pixels in a single frame
* \param nxystack: number of frames in stack multiplied by nxy
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
__device__ my_float4 LLKAll(const int i, float *iobs, complexf *psi, float *background, const int nbmode,
                            const int nxy, const int nxystack, const float scale)
{
  const float obs = iobs[i];

  if(obs < 0) return my_float4(0);

  float calc = background[i%nxy];
  const float s2 = scale * scale;
  for(int imode=0;imode<nbmode;imode++) calc += s2 * dot(psi[i + imode * nxystack], psi[i + imode * nxystack]);
  calc = fmaxf(0.0f, calc);

  return my_float4(LLKPoisson(obs, calc), LLKGaussian(obs, calc), LLKEuclidian(obs, calc), calc);
}

/** Compute the cumulated Poisson log-likelihood
*
*/
__device__ void LLKPoissonSum(const int i, float *iobs, complexf *psi, float *background,
                                float *llk_cumul, const int nbmode, const int nxy,
                                const int nxystack, const float scale)
{
  float obs = iobs[i];

  if(obs < 0) obs = 0;  // Already masked pixel. Use obs=0 to see the result of LLK

  float calc = background[i%nxy];
  const float s2 = scale * scale;
  for(int imode=0;imode<nbmode;imode++) calc += s2 * dot(psi[i + imode * nxystack], psi[i + imode * nxystack]);
  calc = fmaxf(0.0f, calc);

  const float tmp = LLKPoisson(obs, calc);
  if(calc>obs) llk_cumul[i%nxy] += tmp;
  else llk_cumul[i%nxy] -= tmp;
}

__device__ void LLKPoissonStats(const int i, float *iobs, complexf *psi, float *background,
                                float *llk_mean, float *llk_std, float *llk_skew, float *llk_skew0,
                                const int nbmode, const int nxy, const int nxystack, const float scale)
{
  float obs = iobs[i];

  if(obs < 0) obs = 0;  // Already masked pixel. Use obs=0 to see the result of LLK

  float calc = background[i%nxy];
  const float s2 = scale * scale;
  for(int imode=0;imode<nbmode;imode++) calc += s2 * dot(psi[i + imode * nxystack], psi[i + imode * nxystack]);
  calc = fmaxf(0.0f, calc);

  float tmp = LLKPoisson(obs, calc);
  if(calc<obs) tmp = -tmp;

  llk_std[i%nxy] += pow2(tmp - llk_mean[i%nxy]);
  llk_skew[i%nxy] += pow3(tmp - llk_mean[i%nxy]);
  llk_skew0[i%nxy] += pow3(tmp);
}


/** Compute the Poisson log-likelihood histogram and cumul
*
*/
__device__ void LLKPoissonHist(const int i, float *iobs, complexf *psi, float *background,
                                float *llk_sum, short *llk_hist, const int nbin,
                                const float binsize, const int nbmode, const int nxy,
                                const int nxystack, const float scale)
{
  float obs = iobs[i];

  if(obs < 0) obs = 0;  // Already masked pixel. Use obs=0 to see the result of LLK

  float calc = background[i%nxy];
  const float s2 = scale * scale;
  for(int imode=0;imode<nbmode;imode++) calc += s2 * dot(psi[i + imode * nxystack], psi[i + imode * nxystack]);
  calc = fmaxf(0.0f, calc);

  float tmp = LLKPoisson(obs, calc);
  if(calc<obs) tmp = -tmp;

  llk_sum[i%nxy] += tmp;

  int ii = tmp / binsize + nbin/2;
  if(ii<0) ii = 0;
  if(ii>=nbin) ii = nbin-1;
  llk_hist[i%nxy + ii * nxy] += 1;
}
