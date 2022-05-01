
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
float4 LLKAll(const int i, __global float *iobs, __global float2 *psi, __global float *background, const int nbmode, const int nxy, const int nxystack)
{
  const float obs = iobs[i];

  if(obs < 0) return (float4)0;

  float calc = background[i%nxy];
  for(int imode=0;imode<nbmode;imode++) calc += dot(psi[i + imode * nxystack],psi[i + imode* nxystack]);
  calc = fmax(0.01f,calc);  // Minimum value avoiding INF result if calc=0 and obs>0

  return (float4)(LLKPoisson(obs, calc), LLKGaussian(obs, calc), LLKEuclidian(obs, calc), calc);
}
