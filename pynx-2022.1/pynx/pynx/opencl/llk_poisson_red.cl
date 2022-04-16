
// NOTE: in this file 'NXY' is the number of pixels in a 2D or 3D frame - it is the number of pixels for each frame for ptycho and 2D CDI,
// but also the total number of pixels for 3D CDI.

/** Reduction kernel function :
* compute the log-likelihood given observed data following Poisson statistics, and
* a stack of complex data corresponding to the calculated modes.
*/
float LLKPoisson(__global float *iobs, __global float2 *psi, const float background, const int i)
{
  float icalc = background;
  for(int imode=0;imode<NBMODE;imode++) icalc += dot(psi[i+imode*NXYZ],psi[i+imode*NXYZ]);
  icalc = fmax(1.e-12f,icalc);
  const float obs=iobs[i];

  if(obs<=0.1) return icalc; // observed intensity is zero

  #if IRISPROBUG // workaround Iris Pro bug, incorrect calculation of lgamma !!
  float lg;
  if(fabs(lgamma(7.5f)-7.34040f)<0.01)
    lg = lgamma(obs+1);
  else
    lg = obs*(log(obs)-1) + 0.5 * log(2*3.141592653589f*obs);

  return icalc + lg - obs*log(icalc);
  #else
  return icalc + lgamma(obs+1) - obs*log(icalc);
  #endif
}

/** Reduction kernel function :
* compute the log-likelihood given observed data following Poisson statistics, and
* a stack of complex data corresponding to the calculated modes.
* Masked pixels (when mask value is not 0) are ignored.
*/
float LLKPoissonMask(__global float *iobs, __global float2 *psi, __global char* mask, const int i)
{
  if(mask[i%NXY] == 0) return LLKPoisson(iobs, psi, 0, i);
  return 0;
}

float LLKPoissonMaskBackground(__global float *iobs, __global float2 *psi, __global char* mask, __global float* background, const int i)
{
  if(mask[i%NXY] == 0) return LLKPoisson(iobs, psi, background[i%NXY], i);
  return 0;
}

float LLKPoissonBackground(__global float *iobs, __global float2 *psi, __global float* background, const int i)
{
  return LLKPoisson(iobs, psi, background[i%NXY], i);
}
