/** Reduction kernel function: calculate the best scale factor between calculated and observed amplitudes.
*/
__device__ complexf ScaleAmplitude(const int i, float * iobs, pycuda::complex<float> *calc)
{
  const float obs = iobs[i];
  if(obs < 0) return complexf(0.0f, 0.0f);

  const float acalc = abs(calc[i]);
  return complexf(acalc * sqrtf(obs) , acalc * acalc);
}

/** Reduction kernel function: calculate the best scale factor between calculated and observed intensities.
*/
__device__ complexf ScaleIntensity(const int i, float * iobs, pycuda::complex<float> *calc)
{
  const float obs = iobs[i];
  if(obs < 0) return complexf(0.0f, 0.0f);

  const float icalc = dot(calc[i], calc[i]);
  return complexf(icalc * obs , icalc * icalc);
}

/** Reduction kernel function: calculate the best scale factor between calculated and observed intensities,
* for a Poisson noise model.
*/
__device__ complexf ScaleIntensityPoisson(const int i, float * iobs, pycuda::complex<float> *calc)
{
  const float obs = iobs[i];
  if(obs < 0) return complexf(0.0f, 0.0f);

  const float icalc = dot(calc[i], calc[i]);
  return complexf(obs , icalc);
}

/** Reduction kernel function: calculate the best weighted scale factor between calculated and observed intensities.
*/
__device__ complexf ScaleWeightedIntensity(const int i, float * iobs, pycuda::complex<float> *calc)
{
  const float obs = iobs[i];
  if(obs < 0) return complexf(0.0f, 0.0f);

  const float icalc = dot(calc[i], calc[i]);
  float w = obs;
  if(w<1) w = 1;
  w = 1 / w;
  return complexf(w * icalc * obs , w * icalc * icalc);
}
