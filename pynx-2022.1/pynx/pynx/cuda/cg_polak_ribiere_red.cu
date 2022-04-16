/** Reduction kernel function: compute the Polak-Ribiere coefficient for conjugate gradient (complex input).
* Returns numerator and denominator of the coefficient in a float2 value.
*/
__device__ complexf PolakRibiereComplex(complexf grad, complexf lastgrad)
{
  return complexf(dot(grad, grad-lastgrad), dot(lastgrad, lastgrad));
}

/** Reduction kernel function: compute the Polak-Ribiere coefficient for conjugate gradient (real input).
* Returns numerator and denominator of the coefficient in a float2 value.
*/
__device__ complexf PolakRibiereFloat(float grad, float lastgrad)
{
  return complexf(grad*(grad-lastgrad), lastgrad*lastgrad);
}
