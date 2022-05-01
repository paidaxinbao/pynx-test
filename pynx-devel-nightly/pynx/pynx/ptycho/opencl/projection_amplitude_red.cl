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
float_n ProjectionAmplitudeRed(const int i, __global float *iobs, __global float2 *dcalc, __global float *background,
                                 const unsigned int nbmode, const unsigned int nxy, const int nxystack, const int npsi)
{
  const float b = background[i];
  float_n icalc;
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
      // TODO: 1e-12f is arbitrary
      const float d = native_sqrt(fmax(obs-b, 0) / fmax(dc2,1e-12f));
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        //dcalc[i + mode*nxystack] = (float2) (d*dc[mode].x , d*dc[mode].y);
        dcalc[i + j * nxy + mode * nxystack] *= d;
      }
      icalc.v[j] = dc2;
    }
    else
    {
      icalc.v[j] = 0.0f;
    }
  }
  return icalc;
}

/** Amplitude projection: apply the observed intensity to the calculated complex amplitude.
* This should be called for each frame in a stack (i.e. cl_obs[0])
* This also returns the calculated intensity, for auto-scaling purposes.
*/
float ProjectionAmplitudeRed1(const int i, __global float *iobs, __global float2 *dcalc, __global float *background,
                              const unsigned int nbmode, const unsigned int nxy, const int nxystack)
{
  const float b = background[i];
  const float obs = iobs[i];
  if(obs >= 0)
  {
    float dc2=0;
    for(unsigned int mode=0 ; mode<nbmode ; mode++)
    {
      const float2 dc = dcalc[i + mode * nxystack];
      dc2 += dot(dc,dc);
    }

    // Normalization to observed amplitude, taking into account all modes
    // TODO: 1e-12f is arbitrary
    const float d = native_sqrt(fmax(obs-b, 0) / fmax(dc2,1e-12f));
    for(unsigned int mode=0 ; mode<nbmode ; mode++)
    {
      dcalc[i + mode * nxystack] *= d;
    }
    return dc2;
  }
  else return 0.0f;
}


/** Amplitude projection: apply the observed intensity to the calculated complex amplitude.
* In this version the arrays needed to update the incoherent background are also updated, to avoid a double access.
*
* This must be called for a single frame of observed intensities and will apply to all valid frames
*/
float_n ProjectionAmplitudeUpdateBackground(const int i, __global float *iobs, __global float2 *dcalc,
                                                 __global float *background, __global float *vd, __global float *vd2,
                                                 __global float *vz2, __global float *vdz2,
                                                 const unsigned int nbmode, const unsigned int nxy, const int nxystack,
                                                 const int npsi, const char first_pass)
{
  const float b = background[i];
  float_n icalc;

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

      const float dd = obs - b;
      psi2 += dc2;
      dz2 += dd * dc2;
      d2 += dd * dd;
      d += dd;


      // Normalization to observed amplitude, taking into account all modes
      dc2 = fmax(dc2,1e-12f); // TODO: 1e-12f is arbitrary
      const float d = native_sqrt(fmax(obs-b, 0) / dc2);
      for(unsigned int mode=0 ; mode<nbmode ; mode++)
      {
        //dcalc[i + mode*nxystack] = (float2) (d*dc[mode].x , d*dc[mode].y);
        dcalc[i + j * nxy + mode * nxystack] *= d;
      }
      icalc.v[j] = dc2;
    }
    else
    {
      icalc.v[j] = 0.0f;
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
    vdz2[i] += dz2;
  }
  return icalc;
}
