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

// Likelihood gradient using Poisson noise (needs to be Fourier Transformed)
__device__ void PsiGradient(const int i, complexf* psi, complexf* dpsi, float* iobs,
                            const int nx, const int ny, const int nz)
{
  const float obs = iobs[i];
  if(obs<0)
  { // masked pixel
    dpsi[i] = complexf(0,0);
  }
  else
  {
    // Use a Hann window multiplication to dampen high-frequencies in the gradient
    const int ix = i % nx;
    const int iy = (i % (nx * ny)) / nx;
    const int iz = (i % (nx * ny * nz)) / (nx * ny);
    const float qx = (float)(ix - nx * (ix >= (nx / 2))) * 3.14159265f / (float)(nx-1);
    const float qy = (float)(iy - ny * (iy >= (ny / 2))) * 3.14159265f / (float)(ny-1);
    // we can have nz=1 for 2D objects
    const float qz = (float)(iz - nz * (iz >= ((float)nz / 2.0f))) * 3.14159265f / (float)(nz-0.999f);
    const float g = pow2(cosf(qx) * cosf(qy) * cosf(qz));

    const complexf ps = psi[i];
    const float ps2 = fmaxf(dot(ps,ps),1e-12f); // Calculated amplitude TODO: adjust minimum value

    const float f = (1 - obs / ps2);

    dpsi[i] = conj(f*ps);
  }
}


// Calculate gradient due to the support constraint, and adds it to the current object gradient
__device__ void RegSupportGradient(const int i, complexf* obj, complexf* objgrad, signed char* support, const float reg_fac)
{
  const complexf o = obj[i];
  const int m = 1-support[i];
  const complexf g = objgrad[i];

  objgrad[i] = g + (reg_fac * m) * o;
}
