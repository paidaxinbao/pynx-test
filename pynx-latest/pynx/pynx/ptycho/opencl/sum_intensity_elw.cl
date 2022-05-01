/** Compute the sum of intensities in a complex array
*/
void SumIcalc(const int i, __global float* icalc_sum, __global float2 *calc,
              const int nxy, const int nz)
{
  float icalc = 0;
  for(unsigned int j=0 ; j<nz ; j++)
  {
    const float2 dc = calc[i + j * nxy];
    icalc += dot(dc,dc);
  }
  icalc_sum[i] += icalc;
}

/** Compute the sum of observed intensities in a complex array.
* Masked values are replaced by calculated ones.
*
* Shape of the iobs array: (nz, ny, nx)
* Shape of the calc array: (nz, nb_mode, ny, nx)
*/
void SumIobs(const int i, __global float* iobs_sum, __global float* obs,
             __global float2 *calc, const int nxy, const int nz, const int nb_mode)
{
  float obs_sum = 0;
  for(unsigned int j=0 ; j<nz ; j++)
  {
    const float o = obs[i + j * nxy];
    if(o>=0) obs_sum += o;
    else
    {
      // NB: the mask may be different for every frame
      for(int mode=0; mode< nb_mode; mode++)
      {
        const float2 dc = calc[i + nxy * (mode + nb_mode * j)];
        obs_sum += dot(dc,dc);
      }
    }
  }
  iobs_sum[i] += obs_sum;
}
