/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/// Update support from smoothed amplitude
int SupportUpdate(int i, __global float* d, __global char* support, const float threshold, const bool force_shrink)
{
    const char v = d[i] > threshold;
    if(force_shrink)
    {
        support[i] *= v;
    }
    else
    {
        support[i] = v;
    }
    return (int) v;
}

/// Update support from smoothed amplitude, only affecting pixels near border support
int SupportUpdateBorder(int i, __global float* d, __global char* support, const float threshold, const bool force_shrink)
{
    // support & 1: original support
    // support & 2: support expanded by N pixels
    // support & 4: support shrunk by N pixels
    const char s = support[i];
    if(((s & (char)2)==0) || (s & (char)4))
    {
        support[i] = (s & (char)1);
        return (int)support[i];
    }

    const char v = d[i] > threshold;
    if(force_shrink)
    {
        support[i] = (s & (char)1) * v;
    }
    else
    {
        support[i] = v;
    }
    return (int) v;
}

/// Init support from float2 array (for auto-correlation)
int SupportInit(int i, __global float2* d, __global char* support, const float threshold)
{
    if(length(d[i]) > threshold)
    {
       support[i] = 1;
       return 1;
    }
    support[i] = 0;
    return 0;
}

float4 ObjSupportStats(int i, __global float2* obj, __global char* support)
{
  const float s = support[i];
  const float o2 = dot(obj[i], obj[i]);
  return (float4)(sqrt(o2) * s, o2 * s, o2 * s, o2 * (1.0f-s));
}
