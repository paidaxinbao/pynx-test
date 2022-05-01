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
* the calculated complex amplitude.
* Masked pixels are ignored.
* Reference:  [New Journal of Physics 14 (2012) 063004, doi:10.1088/1367-2630/14/6/063004]
* \param i: the point in the 3D observed intensity array for which the llk is calculated
* \param iobs: the observed intensity array, shape=(stack_size, ny, nx)
* \param psi: the calculated complex amplitude, shape=(nb_obj, nb_probe, stack_size, ny, nx)
* \param scale: the scale factor by which the calculated intensities will be multiplied before
*               evaluating the log-likelihood
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
float8 LLKAll(const int i, __global float *iobs, __global float2 *psi, const float scale)
{
  const float obs=iobs[i];
  const float calc = dot(psi[i],psi[i]) * scale;

  if(obs<=-1e19f) return (float8)(0,0,0,calc,0,0,0,0);

  if(obs<-.5f) return (float8)(0,0,0,calc,LLKPoisson(-obs-1, calc), LLKGaussian(-obs-1, calc), LLKEuclidian(-obs-1, calc), 0);

  return (float8)(LLKPoisson(obs, calc), LLKGaussian(obs, calc), LLKEuclidian(obs, calc), calc, 0,0,0,0);
}

/** Reduction kernel function :
* compute the log-likelihood given observed data following Poisson, Gaussian and Euclidian statistics, and
* the calculated intensity (point-spread function convoluted intensity).
* Masked pixels (negative iobs) are ignored.
* Reference:  [New Journal of Physics 14 (2012) 063004, doi:10.1088/1367-2630/14/6/063004]
* \param i: the point in the 3D observed intensity array for which the llk is calculated
* \param iobs: the observed in tensity array
* \param icalc: the calculated intensity
* \param scale: the scale factor by which the calculated intensities will be multiplied before
*               evaluating the log-likelihood
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
float8 LLKAllIcalc(const int i, __global float *iobs, __global float *icalc, const float scale)
{
  const float obs=iobs[i];
  const float calc = fmax(icalc[i], 1e-8f) * scale;

  if(obs<=-1e19f) return (float8)(0,0,0,calc,0,0,0,0);

  if(obs<-.5f) return (float8)(0,0,0,calc,LLKPoisson(-obs-1, calc), LLKGaussian(-obs-1, calc), LLKEuclidian(-obs-1, calc), 0);

  return (float8)(LLKPoisson(obs, calc), LLKGaussian(obs, calc), LLKEuclidian(obs, calc), calc, 0,0,0,0);
}
