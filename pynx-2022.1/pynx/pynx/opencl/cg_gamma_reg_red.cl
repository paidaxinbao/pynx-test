// Re(z1 * conjugate(z2))
float ReZ1Z2c(float2 z1, float2 z2)
{
  return z1.x*z2.x + z1.y*z2.y;
}

/** Reduction kernel function: compute the gamma value for conjugate gradient (coordinate of line minimization),
* for the regularization term (penalty of density variation).
* Returns numerator and denominator of the coefficient in a float2 value.
*
* This is for object OR probe, so nx, ny can take different values
*/
float2 GammaReg(const int i, __global float2 *v, __global float2 *dv, const int nx, const int ny)
{
  const int x=i%nx;
  const int y=i/nx;
  const int y0=y%ny; // For multiple modes, to see if we are near a border
  float numer=0;
  float denom=0;
  const float2 v0 = v[i];
  const float2 dv0=dv[i];

  if(x>0)
  {
    const float2 v1 = v[i-1];
    const float2 dv1=dv[i-1];
    numer += ReZ1Z2c(v0,dv0) + ReZ1Z2c(v1,dv1) - ReZ1Z2c(v0,dv1) - ReZ1Z2c(v1,dv0);
    denom += pown(dv0.x,2) + pown(dv0.y,2) + pown(dv1.x,2) + pown(dv1.y,2) - 2*ReZ1Z2c(dv0,dv1);
  }

  if(x<(nx-1))
  {
    const float2 v1 = v[i+1];
    const float2 dv1=dv[i+1];
    numer += ReZ1Z2c(v0,dv0) + ReZ1Z2c(v1,dv1) - ReZ1Z2c(v0,dv1) - ReZ1Z2c(v1,dv0);
    denom += pown(dv0.x,2) + pown(dv0.y,2) + pown(dv1.x,2) + pown(dv1.y,2) - 2*ReZ1Z2c(dv0,dv1);
  }

  if(y0>0)
  {
    const float2 v1 = v[i-nx];
    const float2 dv1=dv[i-nx];
    numer += ReZ1Z2c(v0,dv0) + ReZ1Z2c(v1,dv1) - ReZ1Z2c(v0,dv1) - ReZ1Z2c(v1,dv0);
    denom += pown(dv0.x,2) + pown(dv0.y,2) + pown(dv1.x,2) + pown(dv1.y,2) - 2*ReZ1Z2c(dv0,dv1);
  }

  if(y0<(ny-1))
  {
    const float2 v1 = v[i+nx];
    const float2 dv1=dv[i+nx];
    numer += ReZ1Z2c(v0,dv0) + ReZ1Z2c(v1,dv1) - ReZ1Z2c(v0,dv1) - ReZ1Z2c(v1,dv0);
    denom += pown(dv0.x,2) + pown(dv0.y,2) + pown(dv1.x,2) + pown(dv1.y,2) - 2*ReZ1Z2c(dv0,dv1);
  }
  return (float2)(-numer, denom);
}
