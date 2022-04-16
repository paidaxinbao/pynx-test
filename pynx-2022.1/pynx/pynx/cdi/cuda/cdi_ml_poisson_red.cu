/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

inline __device__ float pow2(const float v)
{
  return v*v;
}

__device__ complexf Gamma(float *iobs, complexf *psi, complexf *dpsi, const int i)
{
  const float o = iobs[i];
  if(o<0) return complexf(0,0);
  const float SCALE = 1e-15; // Avoid overflow
  const complexf p = psi[i];
  const complexf dp = dpsi[i];

  const float psi2 = fmaxf(dot(p,p),1e-12f);
  const float dpsi2 = dot(dp,dp);
  const float R_psi_dpsi = dot(p, dp);

  return complexf(R_psi_dpsi * ( o / psi2 - 1) * SCALE , dpsi2 * SCALE - o * SCALE *( dpsi2 / psi2 - 2 * pow2(R_psi_dpsi / psi2) ) );
}


__device__ complexf GammaSupport(float *iobs, complexf *psi, complexf *dpsi, complexf *obj, complexf *dobj,
                     signed char *support, const float reg_fac, const int i)
{
  const float SCALE = 1e-15; // Avoid overflow
  // Real space part
  const complexf ob = obj[i];
  const complexf dob = dobj[i];
  const int m = 1 - support[i];

  float num = -reg_fac * m * dot(ob, dob) * SCALE;
  float denom = reg_fac * m * dot(dob,dob) * SCALE;

  const float o = iobs[i];
  if(o>=0)
  {// Reciprocal space part
    const complexf p = psi[i];
    const complexf dp = dpsi[i];

    const float psi2 = fmaxf(dot(p,p),1e-12f);
    const float dpsi2 = dot(dp,dp);
    const float R_psi_dpsi = dot(p, dp);

    num += R_psi_dpsi * ( o / psi2 - 1) * SCALE;
    denom +=  dpsi2 * SCALE - o * ( dpsi2 / psi2 - 2 * pow2(R_psi_dpsi / psi2) ) * SCALE;
  }
  return complexf( num , denom);
}


