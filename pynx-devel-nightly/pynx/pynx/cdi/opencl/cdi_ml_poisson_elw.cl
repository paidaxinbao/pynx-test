/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/


// Likelihood gradient using Poisson noise (needs to be Fourier Transformed)
void PsiGradient(const int i, __global float2* psi, __global float2* dpsi, __global float* iobs,
                 const int nx, const int ny, const int nz)
{
  const float obs = iobs[i]; // Observed amplitude
  if(obs<0)
  { // Masked pixel
    dpsi[i] = (float2)(0,0);
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
    const float g = pown(native_cos(qx) * native_cos(qy) * native_cos(qz), 2);

    const float2 ps = psi[i];
    const float ps2 = fmax(dot(ps,ps),1e-12f); // Calculated amplitude TODO: adjust minimum value

    const float f = (1 - obs / ps2);

    dpsi[i] = (float2) (f*ps.x , -f*ps.y); // (f * psi).conjugate()
  }
}


// Calculate gradient due to the support constraint, and adds it to the current object gradient
void RegSupportGradient(const int i,__global float2* obj, __global float2* objgrad, __global char* support, const float reg_fac)
{
  const float2 o = obj[i];
  const int m = 1-support[i];
  const float2 g = objgrad[i];

  objgrad[i] = (float2) ( g.x + reg_fac * m * o.x , g.y + reg_fac * m * o.y);
}

// Linear combination
void CG_linear(const int i, const float a, __global float2 *A, const float b, __global float2 *B)
{
  A[i] = (float2)(a*A[i].x + b*B[i].x, a*A[i].y + b*B[i].y);
}
