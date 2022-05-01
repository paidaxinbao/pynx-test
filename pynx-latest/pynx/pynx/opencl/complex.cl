/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/// Atomic add for OpenCL
/// Based on public code from http://suhorukov.blogspot.com/2011/12/opencl-11-atomic-operations-on-floating.html
/// and https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
inline float atomic_add_f(volatile __global float *addr, float val)
{
  // To clarify, union means that the same 32bit value can be accessed either as a fp32 or u32 value
  union
  {
    unsigned int u32;
    float        f32;
  } next, expected, current;
  current.f32    = *addr;
  do
  {
    expected.f32 = current.f32;
    next.f32     = expected.f32 + val;
    current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
  } while( current.u32 != expected.u32 );
  return current.f32;
}


/// Complex atomic add
inline float2 atomic_add_c(__global float2 *p, float2 v)
{
  return (float2)(atomic_add_f(&(((__global float*)p)[0]), v.x), atomic_add_f(&(((__global float*)p)[1]), v.y));
}
