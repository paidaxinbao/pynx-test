
/** Compute the scale factor between calculated and observed intensities. Negative observed intensities are
* treated as masked and ignored. The reduced scale factor should be multiplied by the calculated intensities to match
* observed ones.
*/

__device__ complexf scale_intensity(const int i, float* obs, complexf *calc, float* background,
                                    const int nxy, const int nxystack, const int nb_mode)
{
  const float iobs = obs[i];
  if(iobs < 0) return complexf(0,0);

  float icalc = 0;
  for(int imode=0;imode<nb_mode;imode++) icalc += dot(calc[i + imode * nxystack], calc[i + imode * nxystack]);

  return complexf(iobs - background[i % nxy], icalc);
}
