/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/


/// Replace observed intensitites by the calculated ones
__device__ void Calc2Obs(const int i, float *iobs, complexf *dcalc, float *background,
                         const int nbmode, const int nxy, const int nxystack)
{
  float dc2=0;
  for(unsigned int mode=0 ; mode<nbmode ; mode++)
  {
    const complexf dc = dcalc[i + mode*nxystack];
    dc2 += dot(dc,dc);
  }

  iobs[i] = dc2 + background[i % nxy];
}
