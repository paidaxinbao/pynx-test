/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

float2 Gamma(__global float *iobs, __global float2 *psi, __global float2 *dpsi, const int i)
{
  const float obs = iobs[i];
  if(obs<0) return (float2)(0,0);
  const float SCALE = 1e-15; // Avoid overflow
  const float2 p = psi[i];
  const float2 dp = dpsi[i];

  const float psi2 = fmax(dot(p,p), 1e-12f);
  const float dpsi2 = dot(dp,dp);
  const float R_psi_dpsi = p.x * dp.x + p.y * dp.y;

  return (float2)(R_psi_dpsi * ( obs / psi2 - 1) * SCALE , dpsi2 * SCALE - obs * SCALE *( dpsi2 / psi2 - 2 * pown(R_psi_dpsi / psi2,2) ) );
}


float2 GammaSupport(__global float *iobs, __global float2 *psi, __global float2 *dpsi, __global float2 *obj, __global float2 *dobj,
                     __global char *support, const float reg_fac, const int i)
{
  const float SCALE = 1e-10; // Avoid overflow
  // Real space part
  const float2 ob = obj[i];
  const float2 dob = dobj[i];
  const int m = 1 - support[i];

  float num = -reg_fac * m * (ob.x * dob.x + ob.y * dob.y) * SCALE;
  float denom = reg_fac * m * dot(dob,dob) * SCALE;

  const float obs = iobs[i];
  if(obs>=0)
  {// Reciprocal space part, applies only to non-masked pixels
    const float2 p = psi[i];
    const float2 dp = dpsi[i];

    const float psi2 = fmax(dot(p,p), 1e-12f);
    const float dpsi2 = dot(dp,dp);
    const float R_psi_dpsi = p.x * dp.x + p.y * dp.y;

    num += R_psi_dpsi * ( obs / psi2 - 1) * SCALE;
    denom +=  dpsi2 * SCALE - obs * ( dpsi2 / psi2 - 2 * pown(R_psi_dpsi / psi2,2) ) * SCALE;
  }
  return (float2)( num , denom);
}


