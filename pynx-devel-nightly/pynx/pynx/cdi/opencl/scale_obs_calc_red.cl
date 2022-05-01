/** Reduction kernel function: calculate the best scale factor between calculated and observed amplitudes.
*/
float2 ScaleAmplitude(const int i, __global float * iobs, __global float2 *calc)
{
  const float obs = iobs[i];
  if(obs<0) return (float2)(0.0f, 0.0f);

  const float acalc = length(calc[i]);
  return (float2)(acalc * sqrt(obs), acalc * acalc);
}

/** Reduction kernel function: calculate the best scale factor between calculated and observed intensities.
*/
float2 ScaleIntensity(const int i, __global float * iobs, __global float2 *calc)
{
  const float obs = iobs[i];
  if(obs<0) return (float2)(0.0f, 0.0f);

  const float acalc = dot(calc[i], calc[i]);
  return (float2)(acalc * obs, acalc * acalc);
}

/** Reduction kernel function: calculate the best scale factor between calculated and observed intensities,
* for a Poisson noise model.
*/
float2 ScaleIntensityPoisson(const int i, __global float * iobs, __global float2 *calc)
{
  const float obs = iobs[i];
  if(obs<0) return (float2)(0.0f, 0.0f);

  const float icalc = dot(calc[i], calc[i]);
  return (float2)(obs, icalc);
}

/** Reduction kernel function: calculate the best weighted scale factor between calculated and observed intensities.
*/
float2 ScaleWeightedIntensity(const int i, __global float * iobs, __global float2 *calc)
{
  const float obs = iobs[i];
  if(obs<0) return (float2)(0.0f, 0.0f);

  const float acalc = dot(calc[i], calc[i]);
  float w;
  if(obs<1) w = 1;
  else w = 1 / obs;
  return (float2)(w * acalc * obs , w * acalc * acalc);
}
