
// NOTE: in this file 'NXY' is the number of pixels in a 2D or 3D frame - it is the number of pixels for each frame for ptycho and 2D CDI,
// but also the total number of pixels for 3D CDI.

/** kernel function :
* compute the log-likelihood given observed data following Poisson statistics, and
* a stack of complex data corresponding to the calculated modes.
*
* The LLK of each point is calculated and stored in a new array, *not* reduced.
*/
__kernel
void LLKPoisson(__global float *iobs, __global float2 *psi, __global float* llk, const float background)
{
  const unsigned long i=get_global_id(0);
  float icalc = background;
  for(int imode=0;imode<NBMODE;imode++) icalc += dot(psi[i+imode*NXYZ],psi[i+imode*NXYZ]);
  icalc = fmax(1.e-12f,icalc);
  const float obs=iobs[i];

  if(obs<=0.1)
  {
     llk[i] = icalc; // observed intensity is zero
     //if((i % NXY) == 0) printf("CL (%d, %8f, %8f): %e\\n", get_group_id(0), obs, icalc, llk[i]);
  }
  else
  {

     #if IRISPROBUG // workaround Iris Pro bug, incorrect calculation of lgamma !!
     float lg;
     if(fabs(lgamma(7.5f)-7.34040f)<0.01)
       lg = lgamma(obs+1);
     else
       lg = obs*(log(obs)-1) + 0.5 * log(2*3.141592653589f*obs);


     llk[i] = icalc + lg - obs*log(icalc);
     #else
     llk[i] = icalc + lgamma(obs+1) - obs*log(icalc);
     #endif
     //if((i % NXY) == 0) printf("CL (%d, %8f, %8f): %e\\n", get_group_id(0), obs, icalc, llk[i]);
  }
}

/** Reduction kernel function :
* compute the log-likelihood given observed data following Poisson statistics, and
* a stack of complex data corresponding to the calculated modes.
* Masked pixels (when mask value is not 0) are ignored.
*/
__kernel
void LLKPoissonMask(__global float *iobs, __global float2 *psi, __global char* mask, __global float* llk)
{
  const unsigned long i=get_global_id(0);
  if(mask[i%NXY] == 0) LLKPoisson(iobs, psi, llk, 0);
  llk[i] = 0;
}

__kernel
void LLKPoissonMaskBackground(__global float *iobs, __global float2 *psi, __global char* mask, __global float* background, __global float* llk)
{
  const unsigned long i=get_global_id(0);
  if(mask[i%NXY] == 0) LLKPoisson(iobs, psi, llk, background[i%NXY]);
  llk[i] = 0;
}

__kernel
void LLKPoissonBackground(__global float *iobs, __global float2 *psi, __global float* background, __global float* llk)
{
  const unsigned long i=get_global_id(0);
  LLKPoisson(iobs, psi, llk, background[i%NXY]);
}
