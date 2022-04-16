/** Compute the scale factor between calculated and observed intensities. Negative observed intensities are
* treated as masked and ignored. The reduced scale factor should be multiplied by the calculated intensities to match
* observed ones.
*/

float2 scale_intensity(const int i, __global float* obs, __global float2 *calc, __global float *background,
                       const int nxy, const int nxystack, const int nb_mode)
{
  const float iobs = obs[i];
  if(iobs < 0) return (float2)(0,0);

  float icalc = 0;
  for(int imode=0;imode<nb_mode;imode++) icalc += dot(calc[i + imode * nxystack], calc[i + imode * nxystack]);

  return (float2)(iobs - background[i % nxy], icalc);
}
