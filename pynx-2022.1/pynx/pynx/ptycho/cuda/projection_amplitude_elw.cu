/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Amplitude projection: apply the observed intensity to the calculated complex amplitude.
* This should be called for the first frame of a stack (i.e. cu_obs[0]) and will apply to all valid frames
*
*/
__device__ void ProjectionAmplitude(const int i, float *iobs, complexf *dcalc, float *background,
                                    const unsigned int nbmode, const unsigned int nxy,
                                    const int nxystack, const int npsi, const float scale_in,
                                    const float scale_out)
{
  const float s2 = scale_in * scale_in;
  const float sio = scale_in * scale_out;
  const float b = background[i];
  for(int j=0;j<npsi;j++)
  {
    const float obs = iobs[i + j * nxy];
    if(obs >= 0)
    {
      float dc2=0;
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        // TODO: use local memory to avoid double-reading of dcalc !
        // Would require a __local memory array with the size=number of modes
        //dc[mode] = dcalc[i + mode*nxystack];
        //dc2 += dot(dc[mode],dc[mode]);
        const complexf dc = dcalc[i + j * nxy + mode * nxystack];
        dc2 += s2 * dot(dc,dc);
      }

      // Normalization to observed amplitude, taking into account all modes
      dc2 = fmaxf(dc2,1e-12f); // TODO: KLUDGE ? 1e-12f is arbitrary

      // Flip the sign if obs - background < 0 ?
      const float diff = obs - b;
      const float d = copysignf(sqrtf(fmaxf(diff, 0) / dc2) * sio, diff);

      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        //dcalc[i + mode*nxystack] = complexf(d*dc[mode].real() , d*dc[mode].imag());
        dcalc[i + j * nxy + mode * nxystack] *= d;
      }
    }
    else if(scale_in * scale_out != 1.0f)
    {
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        dcalc[i + j * nxy + mode * nxystack] *= sio;
      }
    }
  }
}


__device__ void ProjectionAmplitudeBackground(const int i, float *iobs, complexf *dcalc, float *background,
                                              float *vd, float *vd2, float *vz2, float *vdz2,
                                              const unsigned int nbmode, const unsigned int nxy,
                                              const int nxystack, const int npsi, const char first_pass,
                                              const float scale_in, const float scale_out)
{
  const float s2 = scale_in * scale_in;
  const float sio = scale_in * scale_out;
  const float b = background[i];
  // For the background update
  float psi2 = 0;
  float dz2 = 0;
  float d2 = 0;
  float d = 0;

  for(int j=0;j<npsi;j++)
  {
    const float obs = iobs[i + j * nxy];
    if(obs >= 0)
    {
      float dc2=0;
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        // TODO: use local memory to avoid double-reading of dcalc !
        // Would require a __local memory array with the size=number of modes
        //dc[mode] = dcalc[i + mode*nxystack];
        //dc2 += dot(dc[mode],dc[mode]);
        const complexf dc = dcalc[i + j * nxy + mode * nxystack];
        dc2 += s2 * dot(dc,dc);
      }

      // Normalization to observed amplitude, taking into account all modes
      dc2 = fmaxf(dc2,1e-12f); // TODO: KLUDGE ? 1e-12f is arbitrary

      // Background update
      const float dd = obs - b;
      psi2 += dc2;
      dz2 += dd * dc2;
      d2 += dd * dd;
      d += dd;

      const float d = sqrtf(fmaxf(obs-b, 0) / dc2) * sio;

      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        //dcalc[i + mode*nxystack] = complexf(d*dc[mode].real() , d*dc[mode].imag());
        dcalc[i + j * nxy + mode * nxystack] *= d;
      }
    }
    else if(sio != 1.0f)
    {
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        dcalc[i + j * nxy + mode * nxystack] *= sio;
      }
    }
  }
  if(first_pass)
  {
    vd  [i] = d;
    vd2 [i] = d2 ;
    vz2 [i] = psi2;
    vdz2[i] = dz2;
  }
  else
  {
    vd  [i] += d;
    vd2 [i] += d2 ;
    vz2 [i] += psi2;
    vdz2[i] += dz2
    ;
  }
}

/** Amplitude projection with background update using a mode approach.
*
*/
__device__ void ProjectionAmplitudeBackgroundMode(const int i, float *iobs, complexf *dcalc, float *background,
                                              float *background_new,
                                              const unsigned int nbmode, const unsigned int nxy,
                                              const int nxystack, const int npsi, const char first_pass,
                                              const float scale_in, const float scale_out)
{
  const float s2 = scale_in * scale_in;
  const float sio = scale_in * scale_out;
  const float b = background[i];
  float db=0;
  for(int j=0;j<npsi;j++)
  {
    const float obs = iobs[i + j * nxy];
    if(obs >= 0)
    {
      float dc2=b;
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        // TODO: use local memory to avoid double-reading of dcalc !
        // Would require a __local memory array with the size=number of modes
        //dc[mode] = dcalc[i + mode*nxystack];
        //dc2 += dot(dc[mode],dc[mode]);
        const complexf dc = dcalc[i + j * nxy + mode * nxystack];
        dc2 += s2 * dot(dc,dc);
      }

      // Normalization to observed amplitude, taking into account all modes
      dc2 = fmaxf(dc2,1e-12f); // TODO: KLUDGE ? 1e-12f is arbitrary

      const float d = sqrtf(obs / dc2) * sio;

      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        //dcalc[i + mode*nxystack] = complexf(d*dc[mode].real() , d*dc[mode].imag());
        dcalc[i + j * nxy + mode * nxystack] *= d;
      }
      // Background update as an incoherent mode
      db += b * obs / dc2;
    }
    else if(scale_in * scale_out != 1.0f)
    {
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        dcalc[i + j * nxy + mode * nxystack] *= sio;
      }
    }
  }
  if(first_pass) background_new[i] = db;
  else background_new[i] += db;
}
