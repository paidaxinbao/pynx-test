/** Reduction kernel function :
* compute the log-likelihood given observed data following Gaussian statistics, and
* a stack of complex data corresponding to the calculated modes.
*/
float Chi2(__global float *obs, __global float2 *psi, const int i)
{
  float psi2=0;
  for(int imode=0;imode<NBMODE;imode++) psi2 += dot(psi[i+imode*NXYZ],psi[i+imode*NXYZ]);
  const float o=obs[i];
  return pown((o*o-psi2)/fmax(1.0f,o),2); // Sigma=1 for counting <=1...
}

/** Reduction kernel function :
* compute the log-likelihood given observed data following Gaussian statistics, and
* a stack of complex data corresponding to the calculated modes.
* Masked pixels (when mask value is not 0) are ignored.
*/
float Chi2Mask(__global float *obs, __global float2 *psi, __global char* mask, const int i)
{
  if(mask[i%NXY] == 0) return Chi2(obs, psi, i);
    return 0;
}
