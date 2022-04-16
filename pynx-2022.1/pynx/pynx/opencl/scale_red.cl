
/** Compute the scale factor between calculated and observed intensities. Negative observed intensities are
* treated as masked and ignored. The reduced scale factor should be multiplied by the calculated intensities to match
* observed ones.
*/

float2 scale_intensity(const int i, __global float2 *calc, __global float* obs)
{
  const float iobs = obs[i];
  if(iobs>=0) return (float2)(iobs, dot(calc[i], calc[i]));
  return (float2)(0,0);
}
