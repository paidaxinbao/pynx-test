/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/


/// Replace observed intensitites by the calculated ones
void Calc2Obs(const int i, __global float *iobs, __global float2 *dcalc, __global float *background,
              const unsigned int nbmode, const int nxy, const int nxystack)
{
  float dc2=0;
  for(unsigned int mode=0 ; mode<nbmode ; mode++)
  {
    const float2 dc = dcalc[i + mode*nxystack];
    dc2 += dot(dc,dc);
  }

  iobs[i] = dc2 + background[i % nxy];
}
