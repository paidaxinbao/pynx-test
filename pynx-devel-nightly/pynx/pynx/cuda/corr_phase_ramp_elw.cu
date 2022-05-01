/** Correct the phase ramp eveluated from the mis-alignment of the center of mass
* in reciprocal space
*
* \param :
*/
__device__ void CorrPhaseRamp2D(const int i, complexf *d, const float dx, const float dy,
                                const int nx, const int ny)
{
  const float x = (i % nx - nx / 2.0f) / (float)nx;
  const float y = ((i % (nx * ny)) / nx - ny / 2.0f) / (float)ny;

  float s, c;
  __sincosf(6.2831853071795862f * (dx * x + dy * y) , &s, &c);

  const complexf v = d[i];
  d[i] = complexf(v.real()*c - v.imag()*s , v.imag()*c + v.real()*s);
}
