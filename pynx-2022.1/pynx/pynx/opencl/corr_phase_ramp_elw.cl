/** Correct the phase ramp eveluated from the mis-alignment of the center of mass
* in reciprocal space
*
* \param :
*/

void CorrPhaseRamp2D(const int i, __global float2 *d, const float dx, const float dy,
                     const int nx, const int ny)
{
  const float x = (i % nx - nx / 2.0f) / (float)nx;
  const float y = ((i % (nx * ny)) / nx - ny / 2.0f) / (float)ny;

  const float tmp = 6.2831853071795862f * (dx * x + dy * y);
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  const float2 v = d[i];
  d[i] = (float2)(v.x * c - v.y * s , v.y * c + v.x * s);
}
