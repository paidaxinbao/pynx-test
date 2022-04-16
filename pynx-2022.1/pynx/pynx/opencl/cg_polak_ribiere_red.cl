/** Reduction kernel function: compute the Polak-Ribiere coefficient for conjugate gradient (complex input).
* Returns numerator and denominator of the coefficient in a float2 value.
*/
float2 PolakRibiereComplex(float2 grad, float2 lastgrad)
{
  return (float2)(grad.x*(grad.x-lastgrad.x) + grad.y*(grad.y-lastgrad.y), lastgrad.x*lastgrad.x + lastgrad.y*lastgrad.y);
}

/** Reduction kernel function: compute the Polak-Ribiere coefficient for conjugate gradient (real input).
* Returns numerator and denominator of the coefficient in a float2 value.
*/
float2 PolakRibiereFloat(float grad, float lastgrad)
{
  return (float2)(grad*(grad-lastgrad), lastgrad*lastgrad);
}
