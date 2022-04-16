/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/


/// Replace observed intensitites by calculated ones
/// This kernel is called for each pixel of a two-dimensional iobs array.
void Calc2Obs(const int i, __global float *iobs, __global float2 *dcalc,
              const int nb_proj, const int nb_z, const int nb_mode, const int nx, const int ny)
{
  for(int i_proj=0 ; i_proj < nb_proj ; i_proj++)
  {
    for(int iz=0; iz < nb_z; iz++)
    {
      float dc2=0;
      const int izp = iz + nb_z * i_proj;
      for(unsigned int mode=0 ; mode < nb_mode ; mode++)
      {
        const float2 dc = dcalc[i + nx * ny * (mode + nb_mode * izp)];
        dc2 += dot(dc,dc);
      }

      iobs[i + nx * ny * izp] = dc2;
    }
  }
}
