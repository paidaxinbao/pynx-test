
float LLKPoisson(const float obs, const float calc)
{
  if(obs<=0.1) return calc; // observed intensity is zero

  return calc - obs + obs * log(obs / calc);
}

float LLKGaussian(const float obs, const float calc)
{
  return pown(obs - calc, 2) / (obs + 1);
}

float LLKEuclidian(const float obs, const float calc)
{
  return 4 * pown(sqrt(obs) - sqrt(calc), 2);
}

/** Reduction kernel function :
* compute the log-likelihood given observed data following Poisson, Gaussian and Euclidian statistics, and
* a stack of complex data corresponding to the calculated modes.
* Masked pixels (iobs<0) are ignored.
* Reference:  [New Journal of Physics 14 (2012) 063004, doi:10.1088/1367-2630/14/6/063004]
* \param i: the point in the 4D observed intensity array for which the llk is calculated
* \param iobs: the observed intensity array, shape=(nb_proj, nbz, ny, nx)
* \param psi: the calculated complex amplitude, shape=(nb_proj, nbz, nb_obj, nb_probe, ny, nx)
* \param npsi: number of valid frames in the stack, over which the integration is performd (usually equal to
*              stack_size, except for the last stack which may be incomplete (0-padded)
* \param nbmode: number of modes = nb_probe * nb_obj
* \param nxy: ny * nx
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
float4 LLKAll(const int i, __global float *iobs, __global float2 *psi, const int nb_mode, const int nxy)
{
  const float obs = iobs[i];

  if(obs < 0) return (float4)0;

  const int i0 = i % nxy + (i / nxy) * (nxy * nb_mode);

  float calc = 0.0f;
  for(int imode=0;imode<nb_mode;imode++) calc += dot(psi[i0 + imode * nxy],psi[i0 + imode * nxy]);

  return (float4)(LLKPoisson(obs, calc), LLKGaussian(obs, calc), LLKEuclidian(obs, calc), calc);
}
