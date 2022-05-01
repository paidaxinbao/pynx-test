/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Amplitude projection: apply the observed intensity to the calculated complex amplitude.
* This should be called for the first frame of a stack (i.e. cl_obs[0]) and will apply to all valid frames
*
*/
void ProjectionAmplitude(const int i, __global float *iobs, __global float2 *dcalc, __global float *background,
                         const unsigned int nbmode, const unsigned int nxy,
                         const int nxystack, const int npsi)
{
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
        const float2 dc = dcalc[i + j * nxy + mode * nxystack];
        dc2 += dot(dc,dc);
      }

      // Normalization to observed amplitude, taking into account all modes
      dc2 = fmax(dc2,1e-12f); // TODO: KLUDGE ? 1e-12f is arbitrary

      const float d = native_sqrt(fmax(obs-0.9f*b, 0) / dc2);
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        //dcalc[i + mode*nxystack] = (float2) (d*dc[mode].x , d*dc[mode].y);
        dcalc[i + j * nxy + mode * nxystack] *= d;
      }
    }
  }
}

/** Amplitude projection: apply the observed intensity to the calculated complex amplitude.
* Update the background
* This should be called for the first frame of a stack (i.e. cl_obs[0]) and will apply to all valid frames
*
*/
void ProjectionAmplitudeBackground(const int i, __global float *iobs, __global float2 *dcalc,
                                   __global float *background, __global float *vd, __global float *vd2,
                                   __global float *vz2, __global float *vdz2,
                                   const unsigned int nbmode, const unsigned int nxy,
                                   const int nxystack, const int npsi, const char first_pass)
{
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
        const float2 dc = dcalc[i + j * nxy + mode * nxystack];
        dc2 += dot(dc,dc);
      }

      // Normalization to observed amplitude, taking into account all modes
      dc2 = fmax(dc2,1e-12f); // TODO: KLUDGE ? 1e-12f is arbitrary

      // Background update
      const float dd = obs - b;
      psi2 += dc2;
      dz2 += dd * dc2;
      d2 += dd * dd;
      d += dd;

      const float d = native_sqrt(fmax(obs-b, 0) / dc2);
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        //dcalc[i + mode*nxystack] = (float2) (d*dc[mode].x , d*dc[mode].y);
        dcalc[i + j * nxy + mode * nxystack] *= d;
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
* This should be called for the first frame of a stack (i.e. cl_obs[0]) and will apply to all valid frames
*/
void ProjectionAmplitudeBackgroundMode(const int i, __global float *iobs, __global float2 *dcalc,
                                       __global float *background, __global float *background_new,
                                       const unsigned int nbmode, const unsigned int nxy,
                                       const int nxystack, const int npsi, const char first_pass)
{
  const float b = background[i];
  float db = 0;

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
        const float2 dc = dcalc[i + j * nxy + mode * nxystack];
        dc2 += dot(dc,dc);
      }

      // Normalization to observed amplitude, taking into account all modes
      dc2 = fmax(dc2,1e-12f); // TODO: KLUDGE ? 1e-12f is arbitrary

      const float d = native_sqrt(obs / dc2);
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        //dcalc[i + mode*nxystack] = (float2) (d*dc[mode].x , d*dc[mode].y);
        dcalc[i + j * nxy + mode * nxystack] *= d;
      }
      // Background update as an incoherent mode
      db += b * obs / dc2;
    }
  }
  if(first_pass) background_new[i] = db;
  else background_new[i] += db;
}


/** Amplitude projection: apply the observed intensity to the calculated complex amplitude.
* This should be called for the first frame of a stack (i.e. cl_obs[0]) and will apply to all valid frames.
*
* This version returns in dcalc the difference between the observed and the calculated amplitude, i.e.
* dcalc = (sqrt(iobs) - abs(dcalc))*exp(i*angle(dcalc))
*/
void ProjectionAmplitudeDiff(const int i, __global float *iobs, __global float2 *dcalc, __global float *background,
                             const unsigned int nbmode, const unsigned int nxy, const int nxystack, const int npsi)
{
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
        const float2 dc = dcalc[i + j * nxy + mode * nxystack];
        dc2 += dot(dc,dc);
      }

      // Normalization to observed amplitude, taking into account all modes
      dc2 = native_sqrt(dc2); // TODO: KLUDGE ? 1e-12f is arbitrary
      const float d = (native_sqrt(fmax(obs-b, 0)) -  dc2) / fmax(dc2,1e-6f);
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        dcalc[i + j * nxy + mode * nxystack] *= d;
      }
    }
  }
}
