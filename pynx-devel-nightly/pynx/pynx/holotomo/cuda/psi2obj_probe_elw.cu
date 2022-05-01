// #include "cuda_fp16.h"
#define twopi 6.2831853071795862f
#define pi 3.1415926535897932f
/** Update object projections and probe from psi.
* Object has 4 dimensions: projections, modes, y, x.
* Probe has 4 dimensions: z (propagation distances), modes, y, x.
* Probe_norm has 3 dimensions: z, y, x (same normalisation for all modes)
* The 6-dimensional Psi stack is calculated, with dimensions:
*    projections, propagation distances, object modes, probe modes, y, x
*
* This kernel function should be called for a single projection,
* and will update all object modes.
*
* This version assumes the probe is different for each z (propagation distances)
*/
__device__ float Psi2ObjProbe(const int i, complexf* obj, complexf* obj_old, complexf* probe, complexf* psi,
                              complexf* probe_new, float* probe_norm, float* obj_phase0,
                              float *dx, float *dy, signed char *sample_flag,
                              const int nb_z, const int nb_obj, const int nb_probe,
                              const int nx, const int ny, const float obj_min, const float obj_max,
                              const float reg_obj_smooth,
                              const float beta_delta, const float weight_empty)
{
  //#define UPDATE_PROBE_FROM_OLD_OBJECT
  // TODO: handle going over boundaries & masked regions, take into account dxi, dyi, object inertia ?
  // Coordinates in object array (origin at array center)
  const int ix = i % nx;
  const int iy = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  // Sum of Psi intensity for scale factor update
  float psi2=0;

  for(int iobj=0; iobj<nb_obj; iobj++)
  {
    if(sample_flag[0]>0)
    { // Update object & probe

      //
      complexf onew(0);
      float onorm(0);

      // For probe update with old object
      const complexf o = obj[i + nx * ny * iobj];
      const float o2 = dot(o,o);

      for(int iprobe=0; iprobe<nb_probe; iprobe++)
      {
        for(int iz=0;iz<nb_z;iz++)
        {
          // We want to update the object array, so we need to read the Psi values
          // corresponding to the same object pixel. So we interpolate Psi and Probe

          const complexf ps = bilinear(psi, ix-dx[iz], iy-dy[iz], iobj + iz * nb_obj, nx, ny, false, true);

          psi2 += dot(ps, ps);

          //const complexf pr = probe[ix + nx * (iy + ny * (iprobe + nb_probe * iz))];
          const complexf pr = bilinear(probe, ix-dx[iz], iy-dy[iz], iz, nx, ny, false, false);

          // Object update
          onorm += dot(pr,pr);
          onew += complexf(pr.real()*ps.real() + pr.imag()*ps.imag() , pr.real()*ps.imag() - pr.imag()*ps.real());

          #ifdef UPDATE_PROBE_FROM_OLD_OBJECT
          // Probe update using old object
          const complexf prnew = complexf(o.real()*ps.real() + o.imag()*ps.imag(),
                                          o.real()*ps.imag() - o.imag()*ps.real());
          bilinear_atomic_add_c(probe_new, prnew, ix-dx[iz], iy-dy[iz], iz, nx, ny, false);
          bilinear_atomic_add_f(probe_norm, o2  , ix-dx[iz], iy-dy[iz], 0 , nx, ny, false);
          #endif
        }
      }
      // Object update & regularisation (avoiding borders)
      if((ix > 0) && (iy > 0)  && (ix < (nx-1)) && (iy < (ny-1))  &&(reg_obj_smooth > 0.0f))
      {
        // get neighbour pixels & old value
        const complexf ox = 0.5f * (obj_old[ix-1 + iy     * nx] + obj_old[ix+1 + iy     * nx]);
        const complexf oy = 0.5f * (obj_old[ix   + (iy-1) * nx] + obj_old[ix   + (iy+1) * nx]);

        onew = (onew + onorm * reg_obj_smooth * (ox + oy)) / (onorm * (1 + 2 * reg_obj_smooth));
      }
      else onew /= onorm;

      if(beta_delta >= 0.0f)
      {
        // Impose beta/delta ratio
        // const float ph0 = __half2float(obj_phase0[i + nx * ny * iobj]);
        const float ph0 = obj_phase0[i + nx * ny * iobj];
        if(ph0>-10) // dummy value is -100, values should be >0
        {
          // Convention: obj = abs(obj) * exp(-1j*ph)
          float ph = ph0 + fmodf(-atan2f(onew.imag(), onew.real()) - ph0, twopi);
          if((ph - ph0) >= pi) ph -= twopi;
          else if((ph - ph0) < -pi) ph += twopi;
          if(beta_delta<1e-6f)
            onew = complexf(cosf(ph), -sinf(ph)); // pure phase object
          else
            onew = complexf(expf(-beta_delta * ph) * cosf(ph), -expf(-beta_delta * ph) * sinf(ph));
        }
      }

      // Update norm for object clipping & probe update
      onorm = dot(onew, onew);

      if((obj_max > 0) && (onorm > (obj_max*obj_max)))
      {
        onew *= obj_max / sqrtf(onorm);
        onorm = obj_max * obj_max;
      }
      else if((obj_min > 0) && (onorm < (obj_min*obj_min)))
      {
        onew *= obj_min / sqrtf(onorm);
        onorm = obj_min * obj_min;
      }
      obj[i + nx * ny * iobj] = onew;

      #ifndef UPDATE_PROBE_FROM_OLD_OBJECT
      // Update probe using new object
      for(int iprobe=0; iprobe<nb_probe; iprobe++)
      {
        for(int iz=0;iz<nb_z;iz++)
        {
          const complexf ps = bilinear(psi, ix-dx[iz], iy-dy[iz], iprobe + nb_probe*(iobj + nb_obj *iz),
                                       nx, ny, false, true);
          const complexf prnew = complexf(onew.real()*ps.real() + onew.imag()*ps.imag(),
                                          onew.real()*ps.imag() - onew.imag()*ps.real());
          bilinear_atomic_add_c(probe_new , prnew, ix-dx[iz], iy-dy[iz], iprobe + nb_probe * iz, nx, ny, false);
          if(iobj==0) bilinear_atomic_add_f(probe_norm, onorm, ix-dx[iz], iy-dy[iz], iz, nx, ny, false);
        }
      }
      #endif
    }
    else
    { // sample_flag==0 => Update probe only
      // No need to interpolate Psi and probe here
      for(int iprobe=0; iprobe<nb_probe; iprobe++)
      {
        for(int iz=0;iz<nb_z;iz++)
        {
          // Coordinates in psi array
          const int i1 = ipsi + nx * ny * (iprobe + nb_probe * (iobj + nb_obj * iz));

          const complexf ps = psi[i1];
          psi2 += dot(ps, ps);
          probe_new[ix + nx * (iy + ny * (iprobe + nb_probe * iz))] += ps * weight_empty;
          probe_norm[ix + nx * (iy + ny * iz)] += weight_empty;
        }
      }
    }
  }
  // Integrated intensity for scale factor adjustment of this projection
  return psi2;
}

/** Update probe from psi, without modification of object
* Object has 4 dimensions: projections, modes, y, x.
* Probe has 4 dimensions: z (propagation distances), modes, y, x.
* Probe_norm has 3 dimensions: z, y, x (same normalisation for all modes)
* The 6-dimensional Psi stack is calculated, with dimensions:
*    projections, propagation distances, object modes, probe modes, y, x
*
* This kernel function should be called for a single projection,
* and will update all modes.
*
* This version assumes the probe is different for each z (propagation distances)
*/
__device__ void Psi2Probe(const int i, complexf* obj, complexf* probe, complexf* psi,
                              complexf* probe_new, float* probe_norm,
                              float *dx, float *dy, signed char *sample_flag,
                              const int nb_z, const int nb_obj, const int nb_probe,
                              const int nx, const int ny, const float weight_empty)
{
  // TODO: handle going over boundaries & masked regions, take into account dxi, dyi, object inertia ?
  // Coordinates in object array (origin at array center)
  // We will use bilinear to get actual object value for the shift at each iz
  const int ix = i % nx;
  const int iy = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  for(int iobj=0; iobj<nb_obj; iobj++)
  {
    complexf o = complexf(weight_empty, 0.0f);
    float onorm = weight_empty;

    for(int iprobe=0; iprobe<nb_probe; iprobe++)
    {
      for(int iz=0;iz<nb_z;iz++)
      {
        if(sample_flag[0]>0)
        {
          // o = 0;
          // onorm = 0;
          o = bilinear(obj, dx[iz]+ix, dy[iz]+iy, iobj, nx, ny, false, false);
          onorm = dot(o,o);
        }
        complexf ps = psi[ipsi + nx * ny * (iprobe + nb_probe * (iobj + nb_obj * iz))];
        ps = complexf(o.real()*ps.real() + o.imag()*ps.imag(),
                      o.real()*ps.imag() - o.imag()*ps.real());
        probe_new[ix + nx * (iy + ny * (iprobe + nb_probe * iz))] += ps;
        probe_norm[ix + nx * (iy + ny * iz)] += onorm;
      }
    }
  }
}


/** Merge & normalise probe update.
* Probe has 4 dimensions: z (propagation distances), modes, y, x.
* Probe_norm has 3 dimensions: z, y, x (same normalisation for all modes)
*
* This kernel function should be called for the first z and mode for the probe,
* and will apply to all z and modes.
*/

__device__ void Psi2ProbeMerge(const int i, complexf* probe, complexf* probe_new,
                               float* probe_norm, const float inertia,
                               const int nb_probe, const int nxy, const int nz)
{
  for(int iz=0;iz<nz;iz++)
  {
    const float n = probe_norm[i + nxy * iz];
    for(int iprobe=0; iprobe<nb_probe; iprobe++)
    {
      const int i1 = i + nxy * (iprobe + nb_probe * iz);
      probe[i1] = (probe_new[i1] + inertia * probe[i1]) / (n + inertia);
    }
  }
}

/** Merge & normalise probe update (coherent probe mode version).
* Probe has 4 dimensions: z (propagation distances), modes, y, x.
* Probe_norm has the same dimensions as the probe.
*
* This kernel function should be called for the first z and mode for the probe,
* and will apply to all z and modes.
*/

__device__ void Psi2ProbeMergeCoh(const int i, complexf* probe, complexf* probe_new,
                               float* probe_norm, const float inertia,
                               const int nb_probe, const int nxy, const int nz)
{
  for(int iz=0;iz<nz;iz++)
  {
    for(int iprobe=0; iprobe<nb_probe; iprobe++)
    {
      const int i1 = i + nxy * (iprobe + nb_probe * iz);
      probe[i1] = (probe_new[i1] + inertia * probe[i1]) / (probe_norm[i1] + inertia);
    }
  }
}



/** Update object projections and probe from psi.
* Object has 3 dimensions: projections, y, x (no object modes)
* Probe has 4 dimensions: z (propagation distances), modes, y, x.
* The 4-dimensional Psi stack is calculated, with dimensions:
*    projections, propagation distances, y, x
*
* This version assumes coherent probe modes, so there is only
* one mode calculated, but the probe is decomposed into
* probe modes independently for each projection.
*
* This kernel function should be called for a single projection.
* Each thread updates a single pixel in the object array.
* It is not useful to update multiple projections at the same time,
* as the shifts are different between projections, the probe coordinates
* will also be different and no common memory transfer is possible.
*
* This will:
* - update the object for the given projection
* - compute the new probe modes (which will need to be orthogonalised afterwards)
*
* After this is performed for all projections, the new probe modes
* must be orthonormalised, and the new projection probe mode coefficients
* must be updated.
*/
/*
__device__ void Psi2ObjProbeCohMode(const int i, complexf* obj, complexf* obj_old, complexf* probe, complexf* psi,
                                    complexf* probe_new, float* probe_coeffs, float* obj_phase0,
                                    float *dx, float *dy, signed char *sample_flag,
                                    const int nz, const int nb_obj, const int nb_probe,
                                    const int nx, const int ny, const float obj_min, const float obj_max,
                                    const float reg_obj_smooth, const float beta_delta, const float weight_empty)
{
  // Coordinates in object array (origin at array center)
  const int ix = i % nx;
  const int iy = i / nx;
  // Coordinates in Psi array (origin at (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  if(sample_flag[0]>0)
  { // Update object & probe
    //
    complexf onew(0);
    float onorm(0);

    for(int iz=0;iz<nz;iz++)
    {
      // Pixel coordinate in the first probe mode
      const int i1 = ixyz(__float2int_rn(ix-dx[iz]), __float2int_rn(iy-dy[iz]), iz, nx, ny);

      // Pixel coordinate in the Psi array
      const int i1ps = ixyz_shift(__float2int_rn(ix-dx[iz]), __float2int_rn(iy-dy[iz]), iz, nx, ny);

      // We want to update the object array, so we need to read the Psi values
      // corresponding to the same object pixel -> interpolate/shift Psi and Probe
      // const complexf ps = bilinear(psi, ix-dx[iz], iy-dy[iz], iz, nx, ny, false, true);
      const complexf ps = psi[i1ps];

      complexf pr=0;
      for(int iprobe=0; iprobe<nb_probe; iprobe++)
        pr += probe[i1 + nx * ny * (iprobe + nb_probe * iz)] * probe_coeffs[iprobe + nb_probe * iz];
        //pr += bilinear(probe, ix-dx[iz], iy-dy[iz], iprobe + nb_probe*iz, nx, ny, false, false) * probe_coeffs[iprobe + nb_probe * iz];

      // Object update
      onorm += dot(pr,pr);
      onew += complexf(pr.real()*ps.real() + pr.imag()*ps.imag() , pr.real()*ps.imag() - pr.imag()*ps.real());
    }
    // Object update & regularisation (avoiding borders)
    if((ix > 0) && (iy > 0)  && (ix < (nx-1)) && (iy < (ny-1))  &&(reg_obj_smooth > 0.0f))
    {
      // get neighbour pixels & old value
      const complexf ox = 0.5f * (obj_old[ix-1 + iy     * nx] + obj_old[ix+1 + iy     * nx]);
      const complexf oy = 0.5f * (obj_old[ix   + (iy-1) * nx] + obj_old[ix   + (iy+1) * nx]);

      onew = (onew + onorm * reg_obj_smooth * (ox + oy)) / (onorm * (1 + 2 * reg_obj_smooth));
    }
    else onew /= onorm;

    if(beta_delta >= 0.0f)
    {
      // Impose beta/delta ratio
      const float ph0 = obj_phase0[i];
      if(ph0>-10) // dummy value is -100, values should be >0
      {
        // Convention: obj = abs(obj) * exp(-1j*ph)
        float ph = ph0 + fmodf(-atan2f(onew.imag(), onew.real()) - ph0, twopi);
        if((ph - ph0) >= pi) ph -= twopi;
        else if((ph - ph0) < -pi) ph += twopi;
        if(beta_delta<1e-6f)
          onew = complexf(cosf(ph), -sinf(ph)); // pure phase object
        else
          onew = complexf(expf(-beta_delta * ph) * cosf(ph), -expf(-beta_delta * ph) * sinf(ph));
      }
    }

    // Update norm for object clipping & probe update
    onorm = dot(onew, onew);

    if((obj_max > 0) && (onorm > (obj_max*obj_max)))
    {
      onew *= obj_max / sqrtf(onorm);
      onorm = obj_max * obj_max;
    }
    else if((obj_min > 0) && (onorm < (obj_min*obj_min)))
    {
      onew *= obj_min / sqrtf(onorm);
      onorm = obj_min * obj_min;
    }
    // Keep old object and update
    const complexf objold = obj[i];
    obj[i] = onew;

    // Use old or new object to update the probe ?
    #if 1
    onew = objold;
    onorm = dot(onew,onew);
    #endif

    // Update probe modes and modes coefficients using new object
    // We don't use interpolation, so only one pixel is updated
    for(int iz=0;iz<nz;iz++)
    {
      // Pixel coordinate in the first probe mode
      const int i1 = ixyz(__float2int_rn(ix-dx[iz]), __float2int_rn(iy-dy[iz]), iz, nx, ny);
      // Pixel coordinate in the Psi array
      const int i1ps = ixyz_shift(__float2int_rn(ix-dx[iz]), __float2int_rn(iy-dy[iz]), iz, nx, ny);

      const complexf ps = psi[i1ps];

      // TODO: add inertia for the probe update - to filter out low normalisations ?
      //const complexf pr0 = bilinear(probe, ix-dx[iz], iy-dy[iz], iz, nx, ny, false, false);

      // This is the computed coherent illumination which needs to be decomposed into modes
      const complexf prnew = complexf(onew.real()*ps.real() + onew.imag()*ps.imag(),
                                      onew.real()*ps.imag() - onew.imag()*ps.real()) / onorm;

      // Write back the projection probe so it can be used later to update the projection mode coefficients,
      // once the new probe modes have been ortho-normalised.
      // This is written in the pixel unique to this thread so there should not be any conflict,
      // as long as no interpolation is used.
      psi[i1ps] = prnew;

      // Compute the updated probe modes
      for(int iprobe=0; iprobe<nb_probe; iprobe++)
      {
        complexf p =  prnew;
        for(int iprobe1=0; iprobe1<nb_probe; iprobe1++)
          p -= (iprobe1 != iprobe) * probe_coeffs[iprobe1 + nb_probe * iz] * probe[i1 + nx * ny *(iprobe1 + nb_probe * iz)];

        probe_new[i1 + nx * ny *(iprobe + nb_probe * iz)] += probe_coeffs[iprobe + nb_probe * iz] * p;
      }
    }
  }
  else
  { // sample_flag==0 => Update probe only
    // Update probe modes
    // No need to interpolate: psi=probe here
    for(int iz=0;iz<nz;iz++)
    {
      // Pixel coordinate in the first probe mode
      const int i1 = i + nx * ny * iz;
      // Pixel coordinate in the Psi array
      const int i1ps = ipsi + nx * ny * iz;

      //const complexf ps = bilinear(psi, ix-dx[iz], iy-dy[iz], iz, nx, ny, false, true);
      const complexf ps = psi[i1ps];

      // This is the updated illumination which needs to be decomposed into modes
      const complexf prnew = psi[ipsi + nx * ny * iz];

      // Same code as for sample_flag=1 above

      // Compute the updated probe modes
      for(int iprobe=0; iprobe<nb_probe; iprobe++)
      {
        complexf p =  prnew;
        for(int iprobe1=0; iprobe1<nb_probe; iprobe1++)
          p -= (iprobe1 != iprobe) * probe_coeffs[iprobe1 + nb_probe * iz] * probe[i1 + nx * ny *(iprobe1 + nb_probe * iz)];

        probe_new[i1 + nx * ny *(iprobe + nb_probe * iz)] += weight_empty * probe_coeffs[iprobe + nb_probe * iz] * p;
      }
    }
  }
}
*/
