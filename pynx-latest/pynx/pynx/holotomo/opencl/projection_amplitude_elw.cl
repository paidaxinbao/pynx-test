/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Project the complex calculated psi with the observed magnitude, unless the observed intensity is negative (masked).
*
* This should be called for the complete iobs array of shape: (nb_proj, nbz, ny, nx)
* Psi has a shape: (nb_proj, nbz, nb_obj, nb_probe, ny, nx)
*/
void ProjectionAmplitude(const int i, __global float *iobs, __global float2 *psi, const unsigned int nb_mode, const unsigned int nxy)
{
  const float obs = iobs[i];
  if(obs < 0) return;

  const int i0 = i % nxy + (i / nxy) * (nxy * nb_mode);

  float dc2=0;
  for(unsigned int mode=0 ; mode<nb_mode ; mode++)
  {
    // TODO: use local memory to avoid double-reading of dcalc
    // Would require a __local memory array with the size=number of modes
    //dc[mode] = dcalc[i + mode*nxystack];
    //dc2 += dot(dc[mode],dc[mode]);
    const float2 dc = psi[i0 + mode * nxy];
    dc2 += dot(dc,dc);
  }

  // Normalization to observed amplitude, taking into account all modes
  dc2 = fmax(dc2, 1e-12f); // TODO: KLUDGE ? 1e-12f is arbitrary
  const float d = native_sqrt(obs) * native_rsqrt(dc2);
  for(unsigned int mode=0 ; mode<nb_mode ; mode++)
  {
    //psi[i0 + mode * nxy] = (float2) (d*dc[mode].x , d*dc[mode].y);
    psi[i0 + mode * nxy] *= d;
  }
}
