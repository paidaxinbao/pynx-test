/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/// Update support from smoothed amplitude
__device__ int SupportUpdate(int i, float* d, signed char * support, const float threshold, const bool force_shrink)
{
    const int tmp = d[i] > threshold;
    if(force_shrink)
    {
        support[i] *= tmp;
    }
    else
    {
        support[i] = tmp;
    }
    return tmp;
}

/// Update support from smoothed amplitude
__device__ int SupportUpdateBorder(int i, float* d, signed char * support, const float threshold, const bool force_shrink)
{
    // support & 1: original support
    // support & 2: support expanded by N pixels
    // support & 4: support shrunk by N pixels
    const signed char s = support[i];
    if(((s & (signed char)2)==0) || (s & (signed char)4))
    {
        support[i] = (s & (char)1);
        return (int)support[i];
    }

    const int tmp = d[i] > threshold;
    if(force_shrink)
    {
        support[i] = (s & (signed char)1) * tmp;
    }
    else
    {
        support[i] = tmp;
    }
    return tmp;
}

/// Init support from float2 array (for auto-correlation)
__device__ int SupportInit(int i, complexf* d, signed char * support, const float threshold)
{
    if(abs(d[i]) > threshold)
    {
       support[i] = 1;
       return 1;
    }
    support[i] = 0;
    return 0;
}

__device__ my_float4 ObjSupportStats(int i, complexf* obj, signed char * support)
{
  const float s = support[i];
  const float o2 = dot(obj[i], obj[i]);
  return my_float4(sqrtf(o2) * s, o2 * s, o2 * s, o2 * (1.0f-s));
}
