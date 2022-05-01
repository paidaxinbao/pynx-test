/** Reduction kernel function: compute the Fletcher-Reeves coefficient for conjugate gradient.
* Returns numerator and denominator of the coefficient in a float2 value.
*/
float2 FletcherReeves(float2 grad, float2 lastgrad)
{
  return (float2)(grad.x*grad.x + grad.y*grad.y, lastgrad.x*lastgrad.x + lastgrad.y*lastgrad.y);
}
