#include <pycuda-complex.hpp>
typedef pycuda::complex<float> complexf;

/// dot product
inline __device__ float dot(complexf a, complexf b)
{
    return a.real() * b.real() + a.imag() * b.imag();
}

/// Norm
inline __device__ float ComplexNormN(const complexf v, const int nn)
{
  const float a = sqrtf(dot(v, v));
  float an = a;
  for(int i=1;i<nn;i++) an *= a;
  return an;
}

/// Complex atomic add
inline __device__ complexf atomicAdd(complexf *p, complexf v)
{
   // TODO: avoid using private ->_M_im and ->_M_re
   return complexf(atomicAdd(&(p->_M_re), v.real()), atomicAdd(&(p->_M_im), v.imag()));
}
