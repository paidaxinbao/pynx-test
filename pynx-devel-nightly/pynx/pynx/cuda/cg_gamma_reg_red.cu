// Re(z1 * conjugate(z2))
inline __device__ float ReZ1Z2c(complexf z1, complexf z2)
{
  return z1.real() * z2.real() + z1.imag() * z2.imag();
}

/** Reduction kernel function: compute the gamma value for conjugate gradient (coordinate of line minimization),
* for the regularization term (penalty of density variation).
* Returns numerator and denominator of the coefficient in a complexf value.
*
* This is for object OR probe, so nx, ny can take different values
*/
__device__ complexf GammaReg(const int i, complexf *v, complexf *dv, const int nx, const int ny)
{
  const int x=i%nx;
  const int y=i/nx;
  const int y0=y%ny; // For multiple modes, to see if we are near a border
  float numer=0;
  float denom=0;
  const complexf v0 = v[i];
  const complexf dv0=dv[i];

  if(x>0)
  {
    const complexf v1 = v[i-1];
    const complexf dv1=dv[i-1];
    numer += ReZ1Z2c(v0,dv0) + ReZ1Z2c(v1,dv1) - ReZ1Z2c(v0,dv1) - ReZ1Z2c(v1,dv0);
    denom += dv0.real() * dv0.real() + dv0.imag()*dv0.imag() + dv1.real()*dv1.real() + dv1.imag()*dv1.imag() - 2*ReZ1Z2c(dv0,dv1);
  }

  if(x<(nx-1))
  {
    const complexf v1 = v[i+1];
    const complexf dv1=dv[i+1];
    numer += ReZ1Z2c(v0,dv0) + ReZ1Z2c(v1,dv1) - ReZ1Z2c(v0,dv1) - ReZ1Z2c(v1,dv0);
    denom += dv0.real() * dv0.real() + dv0.imag()*dv0.imag() + dv1.real()*dv1.real() + dv1.imag()*dv1.imag() - 2*ReZ1Z2c(dv0,dv1);
  }

  if(y0>0)
  {
    const complexf v1 = v[i-nx];
    const complexf dv1=dv[i-nx];
    numer += ReZ1Z2c(v0,dv0) + ReZ1Z2c(v1,dv1) - ReZ1Z2c(v0,dv1) - ReZ1Z2c(v1,dv0);
    denom += dv0.real() * dv0.real() + dv0.imag()*dv0.imag() + dv1.real()*dv1.real() + dv1.imag()*dv1.imag() - 2*ReZ1Z2c(dv0,dv1);
  }

  if(y0<(ny-1))
  {
    const complexf v1 = v[i+nx];
    const complexf dv1=dv[i+nx];
    numer += ReZ1Z2c(v0,dv0) + ReZ1Z2c(v1,dv1) - ReZ1Z2c(v0,dv1) - ReZ1Z2c(v1,dv0);
    denom += dv0.real() * dv0.real() + dv0.imag()*dv0.imag() + dv1.real()*dv1.real() + dv1.imag()*dv1.imag() - 2*ReZ1Z2c(dv0,dv1);
  }
  return complexf(-numer, denom);
}
