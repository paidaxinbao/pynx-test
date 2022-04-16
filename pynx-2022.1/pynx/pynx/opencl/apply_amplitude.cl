/** Apply the observed amplitude (from intensities) to a complex calculated array.
*
*/
__kernel
void ApplyAmplitude(__global float *iobs, __global float2* dcalc)
{
  const unsigned long i2=get_global_id(0);
  // iobs is floating point data array, dcalc is interleaved complex data array
  const float2 dc=dcalc[i2];
  const float a = native_sqrt(fmax(iobs[i2],0.0f)) / fmax(length(dc), 1e-20f);
  dcalc[i2] = (float2) (a*dc.x , a*dc.y);
}

/** Apply the observed amplitude (from intensities) to a complex calculated array.
* Version with a mask.
*/
__kernel
void ApplyAmplitudeMask(__global float *iobs, __global float2* dcalc, __global char* mask)
{
  if(mask[get_global_id(0)]>0) return;
  ApplyAmplitude(iobs, dcalc);
}
