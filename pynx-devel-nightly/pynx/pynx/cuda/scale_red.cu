
/** Compute the scale factor between calculated and observed intensities. Negative observed intensities are
* treated as masked and ignored. The reduced scale factor should be multiplied by the calculated intensities to match
* observed ones.
*/

__device__ complexf scale_intensity(const int i, complexf *calc, float* obs)
{
  const float iobs = obs[i];
  if(iobs>=0) return complexf(iobs, dot(calc[i], calc[i]));
  return complexf(0,0);
}
