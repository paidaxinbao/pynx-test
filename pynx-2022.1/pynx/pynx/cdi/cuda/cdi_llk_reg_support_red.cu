/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

__device__ float LLKRegSupport(const complexf obj, const signed char support)
{
  // Real space contribution (object constraint/regularization)
  if(support==0)
  {
    return dot(obj, obj);
  }
  return 0;
}
