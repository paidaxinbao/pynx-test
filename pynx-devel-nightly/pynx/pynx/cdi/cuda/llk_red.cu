__device__ float LLKPoisson(const float obs, const float calc)
{
  if(obs<=0.1) return calc; // observed intensity is zero

  return calc - obs + obs * log(obs / calc);
}

__device__ float LLKGaussian(const float obs, const float calc)
{
  const float tmp = obs - calc;
  return tmp * tmp / (obs + 1);
}

__device__ float LLKEuclidian(const float obs, const float calc)
{
  const float tmp = sqrtf(obs) - sqrt(calc);
  return 4 * tmp * tmp;
}

/** Reduction kernel function :
* compute the log-likelihood given observed data following Poisson, Gaussian and Euclidian statistics, and
* a stack of complex data corresponding to the calculated modes.
* Masked pixels are ignored.
* Reference:  [New Journal of Physics 14 (2012) 063004, doi:10.1088/1367-2630/14/6/063004]
* \param i: the point in the 3D observed intensity array for which the llk is calaculated
* \param iobs: the observed in tensity array, shape=(stack_size, ny, nx)
* \param psi: the calculated complex amplitude, shape=(nb_obj, nb_probe, stack_size, ny, nx)
* \param mask: the mask (0=good pixel, >0 masked pixel) of shape (ny, nx)
* \param scale: the scale factor by which the calculated intensities will be multiplied before
*               evaluating the log-likelihood
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
__device__  my_float8 LLKAll(const int i, float *iobs, pycuda::complex<float> *psi, const float scale)
{
  const float obs = iobs[i];
  const float calc = dot(psi[i],psi[i]) * scale;

  if(obs<=-1e19f) return my_float8(0,0,0,calc,0,0,0,0);

  if(obs<-.5f) return my_float8(0,0,0,calc,LLKPoisson(-obs-1, calc), LLKGaussian(-obs-1, calc), LLKEuclidian(-obs-1, calc), 0);

  return my_float8(LLKPoisson(obs, calc), LLKGaussian(obs, calc), LLKEuclidian(obs, calc), calc, 0,0,0,0);
}

/** Reduction kernel function :
* compute the log-likelihood given observed data following Poisson, Gaussian and Euclidian statistics, and
* a stack of complex data corresponding to the calculated modes.
* Masked pixels are ignored.
* Version for upsampled calc array.
*/
__device__  my_float8 LLKAllUp(const int i, float *iobs, pycuda::complex<float> *psi, const float scale,
                               const int nx, const int ny, const int nz, const int ux, const int uy, const int uz)
{
  const float obs = iobs[i];

  const int ix = i % nx;
  const int iy = (i % (nx * ny)) / nx;
  const int iz = (i % (nx * ny * nz)) / (nx * ny);

  float calc = 0.0f;
  for(int dx=0; dx<ux; dx++)
    for(int dy=0; dy<uy; dy++)
      for(int dz=0; dz<uz; dz++)
      {
        const complexf dc = psi[ix*ux + dx + nx*ux * (iy*uy + dy + ny*uy * (iz*uz + dz))];
        calc += abs(dc) * abs(dc) * scale;
      }

  if(obs<=-1e19f) return my_float8(0,0,0,calc,0,0,0,0);

  if(obs<-.5f) return my_float8(0,0,0,calc,LLKPoisson(-obs-1, calc), LLKGaussian(-obs-1, calc), LLKEuclidian(-obs-1, calc), 0);

  return my_float8(LLKPoisson(obs, calc), LLKGaussian(obs, calc), LLKEuclidian(obs, calc), calc, 0,0,0,0);
}


/** Reduction kernel function :
* compute the log-likelihood given observed data following Poisson, Gaussian and Euclidian statistics, and
* a stack of complex data corresponding to the calculated modes.
* Masked pixels are ignored.
* Reference:  [New Journal of Physics 14 (2012) 063004, doi:10.1088/1367-2630/14/6/063004]
* \param i: the point in the 3D observed intensity array for which the llk is calaculated
* \param iobs: the observed in tensity array, shape=(stack_size, ny, nx)
* \param psi: the calculated complex amplitude, shape=(nb_obj, nb_probe, stack_size, ny, nx)
* \param mask: the mask (0=good pixel, >0 masked pixel) of shape (ny, nx)
* \param scale: the scale factor by which the calculated intensities will be multiplied before
*               evaluating the log-likelihood
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
__device__  my_float8 LLKAllIcalc(const int i, float *iobs, float *icalc, const float scale)
{
  const float obs = iobs[i];
  const float calc = fmaxf(icalc[i], 1e-8f) * scale;

  if(obs<=-1e19f) return my_float8(0,0,0,calc,0,0,0,0);

  if(obs<-.5f) return my_float8(0,0,0,calc,LLKPoisson(-obs-1, calc), LLKGaussian(-obs-1, calc), LLKEuclidian(-obs-1, calc), 0);

  return my_float8(LLKPoisson(obs, calc), LLKGaussian(obs, calc), LLKEuclidian(obs, calc), calc, 0,0,0,0);
}
