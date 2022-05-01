/** This function computes the amplitude projection scale from the observed
* and calculated intensity, taking into account different masking schemes.
*/
__device__ float CalcAmplitudeScale(float obs, const float calc, const signed char zero_mask,
                                    const float confidence_interval_factor_sigma,
                                    const float confidence_interval_factor_mask_min,
                                    const float confidence_interval_factor_mask_max)
{
  // Masked pixel
  if(obs<=-1e38f)
  {
    if(zero_mask) return 0.0f;
    return 1.0f;
  }

  // Masked pixel, but with an estimated intensity, so we use a specific
  // confidence interval
  if(obs<=-1e19f)
  {
    const float obs_min = confidence_interval_factor_mask_min * (-obs/1e19f-1);
    const float obs_max = confidence_interval_factor_mask_max * (-obs/1e19f-1);
    if(calc < obs_min) return sqrtf(obs_min / fmaxf(calc, 1e-20f));
    if(calc > obs_max) return sqrtf(obs_max / fmaxf(calc, 1e-20f));
    return 1.0f;
  }

  // Free pixel - we keep the input amplitudes
  if(obs<-0.5f)  return 1.0f;

  // Non-masked pixel
  const float sig = sqrtf(obs+1) * 0.675f * confidence_interval_factor_sigma;
  float obsconf; // Intensity in confidence interval, closest to calc
  if(fabs(obs - calc) < sig) obsconf = calc;
  else
  {
    if(calc > obs) obsconf = obs + sig;
    else           obsconf = obs - sig;  // Can't have obs < sig here
  }
  return sqrtf(obsconf / fmaxf(calc, 1e-20f));
}

/** Apply the observed amplitude to a complex calculated array.
* Masked pixels (Iobs<0) are just multiplied by a scale factor.
* \param i: index of the considered pixel
* \param iobs: array of observed values, negative values are considered masked or free (no amplitude projection)
* \param dcalc: array of calculated complex amplitudes
* \param scale_in: scale factor to be applied to calculated intensities before comparison to observed intensities
* \param scale_out: scale factor to be applied to calculated intensities on output (for FFT scaling)
* \param zero_mask: if True (non-zero), all masked pixels amplitudes are set to zero
* \param confidence_interval_factor: a relaxation factor, with the projection of calculated amplitude being done
*        towards the limit of the poisson confidence interval. A value of 1 corresponds to a 50% confidence interval,
*        a value of 0 corresponds to a strict observed amplitude projection.
* \param confidence_interval_factor_mask_{min,max}: special confidence interval factors
*        for masked & interpolated pixels, which are stored as -1e-19 * (iobs_interp+1)
*/
__device__ void ApplyAmplitude(const int i, float *iobs, complexf* dcalc, const float scale_in,
                               const float scale_out, const signed char zero_mask,
                               const float confidence_interval_factor_sigma,
                               const float confidence_interval_factor_mask_min,
                               const float confidence_interval_factor_mask_max)
{
  // iobs is floating point data array, dcalc is interleaved complex data array
  const float obs = iobs[i];
  const complexf dc = dcalc[i] * scale_in;
  const float calc = abs(dc) * abs(dc);

  dcalc[i] = (CalcAmplitudeScale(obs, calc, zero_mask, confidence_interval_factor_sigma,
                                 confidence_interval_factor_mask_min,
                                 confidence_interval_factor_mask_max) * scale_out) * dc;
}

/** Apply the observed amplitude to a complex calculated array.
* Upsampled version.
*/
__device__ void ApplyAmplitudeUp(const int i, float *iobs, complexf* dcalc, const float scale_in,
                                 const float scale_out, const signed char zero_mask,
                                 const float confidence_interval_factor_sigma,
                                 const float confidence_interval_factor_mask_min,
                                 const float confidence_interval_factor_mask_max,
                                 const int nx, const int ny, const int nz,
                                 const int ux, const int uy, const int uz)
{
  // iobs is floating point data array, dcalc is interleaved complex data array
  const float obs = iobs[i];
  const int ix = i % nx;
  const int iy = (i % (nx * ny)) / nx;
  const int iz = i / (nx * ny);

  float calc = 0.0f;
  for(int dx=0; dx<ux; dx++)
    for(int dy=0; dy<uy; dy++)
      for(int dz=0; dz<uz; dz++)
      {
        const complexf dc = dcalc[ix*ux + dx + ux*nx * (iy*uy + dy + uy*ny * (iz*uz + dz))] * scale_in;
        calc += abs(dc) * abs(dc);
      }

  const float s = CalcAmplitudeScale(obs, calc, zero_mask, confidence_interval_factor_sigma,
                                     confidence_interval_factor_mask_min,
                                     confidence_interval_factor_mask_max) * (scale_out * scale_out);
  for(int dx=0; dx<ux; dx++)
    for(int dy=0; dy<uy; dy++)
      for(int dz=0; dz<uz; dz++)
        dcalc[ix*ux + dx + ux*nx * (iy*uy + dy + uy*ny * (iz*uz + dz))] *= s;
}

/** Apply the observed amplitude to a complex calculated array, using already calculated intensities
* (taking into account PSF)
*/
__device__ void ApplyAmplitudeIcalc(const int i, float *iobs, complexf* dcalc, float *icalc, const float scale_in,
                                    const float scale_out, const signed char zero_mask,
                                    const float confidence_interval_factor_sigma,
                                    const float confidence_interval_factor_mask_min,
                                    const float confidence_interval_factor_mask_max)
{
  const float obs = iobs[i];
  // KLUDGE: The GPU FFT-based convolution can produce negative values, which can be avoided by reverting to
  // non-convoluted values...
  float calc = icalc[i] * scale_in * scale_in;
  const complexf dc = dcalc[i] * scale_in;
  if(calc < 1e-20f) calc = fmaxf(dot(dc, dc), icalc[i]);

  dcalc[i] = (CalcAmplitudeScale(obs, calc, zero_mask, confidence_interval_factor_sigma,
                                 confidence_interval_factor_mask_min,
                                 confidence_interval_factor_mask_max) * scale_out) * dc;
}

/** Compute the calculated intensity from the upsampled complex amplitude.
*/
__device__ void PsiUp2Icalc(const int i, float *icalc, complexf* psi,
                            const int nx, const int ny, const int nz,
                            const int ux, const int uy, const int uz)
{
  const int ix = i % nx;
  const int iy = (i % (nx * ny)) / nx;
  const int iz = (i % (nx * ny * nz)) / (nx * ny);

  float calc = 0.0f;
  for(int dx=0; dx<ux; dx++)
    for(int dy=0; dy<uy; dy++)
      for(int dz=0; dz<uz; dz++)
      {
        const complexf dc = psi[ix*ux + dx + ux*nx * (iy*uy + dy + uy*ny * (iz*uz + dz))];
        calc += abs(dc) * abs(dc);
      }
  icalc[i] = calc;
}

/** Apply the observed amplitude to a complex calculated array, using already calculated intensities
* (taking into account PSF). Upsampled version
*/
__device__ void ApplyAmplitudeIcalcUp(const int i, float *iobs, complexf* psi, float *icalc, const float scale_in,
                                      const float scale_out, const signed char zero_mask,
                                      const float confidence_interval_factor_sigma,
                                      const float confidence_interval_factor_mask_min,
                                      const float confidence_interval_factor_mask_max,
                                      const int nx, const int ny, const int nz,
                                      const int ux, const int uy, const int uz)
{
  const float obs = iobs[i];
  // KLUDGE: The GPU FFT-based convolution can produce negative values, which can be avoided by reverting to
  // non-convoluted values...
  float calc = icalc[i] * scale_in * scale_in;
  const complexf dc = psi[i] * scale_in;
  if(calc < 1e-20f) calc = fmaxf(dot(dc, dc), icalc[i]);

  const float s = CalcAmplitudeScale(obs, calc, zero_mask, confidence_interval_factor_sigma,
                                 confidence_interval_factor_mask_min,
                                 confidence_interval_factor_mask_max) * scale_in * scale_out;

  const int ix = i % nx;
  const int iy = (i % (nx * ny)) / nx;
  const int iz = (i % (nx * ny * nz)) / (nx * ny);

  for(int dx=0; dx<ux; dx++)
    for(int dy=0; dy<uy; dy++)
      for(int dz=0; dz<uz; dz++)
        psi[ix*ux + dx + ux*nx * (iy*uy + dy + uy*ny * (iz*uz + dz))] *= s;
}
