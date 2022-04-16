// #include "cuda_fp16.h"
#define twopi 6.2831853071795862f
#define pi 3.1415926535897932f
/** Update object projections and probe from psi, and update psi for RAAR
* Object has 3 dimensions: projections, y, x.
* Probe has 3 dimensions: z (propagation distances), y, x.
* Probe_norm has 3 dimensions: z, y, x
* The 4-dimensional Psi stack is calculated, with dimensions:
*    projections, propagation distances, y, x
*
* This kernel function should be called for a single projection.
*
* This version assumes the probe is different for each z (propagation distances)
*
* \param beta: if >0, this will update Psi for the first part of the RAAR update
* to (1-2beta)*Psi + beta*Psi_old where Psi_old is Psi before the amplitude projection.
*/
__device__ float Psi2ObjProbeRAAR(const int i, complexf* obj, complexf* obj_old, complexf* probe,
                                  complexf* psi, complexf* psiold,
                                  complexf* probe_new, float* probe_norm, float* obj_phase0,
                                  float *dx, float *dy, signed char *sample_flag, const int nb_z,
                                  const int nx, const int ny, const float obj_min, const float obj_max,
                                  const float reg_obj_smooth,
                                  const float beta_delta, const float weight_empty, const float beta)
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

  if(sample_flag[0]>0)
  { // Update object & probe

    //
    complexf onew(0);
    float onorm(0);

    // For probe update with old object
    const complexf o = obj[i];
    const float o2 = dot(o,o);

    for(int iz=0;iz<nb_z;iz++)
    {
      // We want to update the object array, so we need to read the Psi values
      // corresponding to the same object pixel. So we interpolate Psi and Probe

      const complexf ps = 2.0f * bilinear(psi, ix-dx[iz], iy-dy[iz], iz, nx, ny, false, true)
                               - bilinear(psiold, ix-dx[iz], iy-dy[iz], iz, nx, ny, false, true);

      psi2 += dot(ps, ps);

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
      if(ph0<=1000) // dummy value is 1000, values should be <0
      {
        // Convention: obj = abs(obj) * exp(1j*ph)
        float ph = ph0 + fmodf(atan2f(onew.imag(), onew.real()) - ph0, twopi);
        if((ph - ph0) >= pi) ph -= twopi;
        else if((ph - ph0) < -pi) ph += twopi;
        if(beta_delta<1e-6f)
          onew = complexf(cosf(ph), sinf(ph)); // pure phase object
        else
          onew = complexf(expf(beta_delta * ph) * cosf(ph), expf(beta_delta * ph) * sinf(ph));
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
    obj[i] = onew;

    #ifndef UPDATE_PROBE_FROM_OLD_OBJECT
    // Update probe using new object
    for(int iz=0;iz<nb_z;iz++)
    {
      const complexf ps = 2.0f * bilinear(psi, ix-dx[iz], iy-dy[iz], iz, nx, ny, false, true)
                            - bilinear(psiold, ix-dx[iz], iy-dy[iz], iz, nx, ny, false, true);
      const complexf prnew = complexf(onew.real()*ps.real() + onew.imag()*ps.imag(),
                                      onew.real()*ps.imag() - onew.imag()*ps.real());
      bilinear_atomic_add_c(probe_new , prnew, ix-dx[iz], iy-dy[iz], iz, nx, ny, false);
      bilinear_atomic_add_f(probe_norm, onorm, ix-dx[iz], iy-dy[iz], iz, nx, ny, false);
    }
    #endif

    // RAAR update of Psi
    if(beta>0)
    {
      for(int iz=0;iz<nb_z;iz++)
      {
        // Updating psi - need to explicitly compute coordinates which
        // would be used by bilinear(psi,...,false, true)
        const int ix2 = (__float2int_rn(ix1-dx[iz]) + nx ) % nx;
        const int iy2 = (__float2int_rn(iy1-dy[iz]) + ny ) % ny;
        const int ipsi2 = ix2 + nx * (iy2 + ny * iz);
        const complexf ps = psi[ipsi2];
        const complexf psold = psiold[ipsi2];
        psi[ipsi2] = (1.0f-2.0f*beta) * ps + beta * psold;
      }
    }
  }
  else
  { // sample_flag==0 => Update probe only
    // No need to interpolate Psi and probe here
    for(int iz=0;iz<nb_z;iz++)
    {
      // Coordinates in psi array
      const int ipsi2 = ipsi + nx * ny * iz;
      const int ipr = ix + nx * (iy + ny * iz);

      const complexf ps = psi[ipsi2];
      const complexf psold = psiold[ipsi2];
      const complexf rps = 2.0f * ps -psold;
      psi2 += dot(rps, rps);
      probe_new[ipr] += rps * weight_empty;
      probe_norm[ipr] += weight_empty;

      // RAAR update of Psi
      if(beta>0) psi[ipsi2] = (1.0f-2.0f*beta) * ps + beta * psold;
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
__device__ void Psi2ProbeRAAR(const int i, complexf* obj, complexf* probe,
                              complexf* psi, complexf* psiold,
                              complexf* probe_new, float* probe_norm,
                              float *dx, float *dy, signed char *sample_flag,
                              const int nb_z, const int nx, const int ny,
                              const float weight_empty, const float beta)
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

  complexf o = complexf(weight_empty, 0.0f);
  float onorm = weight_empty;

  for(int iz=0;iz<nb_z;iz++)
  {
    const int ipsiz = ipsi + iz * nx * ny;
    const int iprz  = i    + iz * nx * ny;
    if(sample_flag[0]>0)
    {
      // o = 0;
      // onorm = 0;
      o = bilinear(obj, dx[iz]+ix, dy[iz]+iy, 0, nx, ny, false, false);
      onorm = dot(o,o);
    }
    const complexf ps1 = psi[ipsiz];
    const complexf psold = psiold[ipsiz];
    const complexf ps = 2.0f * ps1 - psold;
    probe_new [iprz] += complexf(o.real()*ps.real() + o.imag()*ps.imag(),
                                 o.real()*ps.imag() - o.imag()*ps.real());
    probe_norm[iprz] += onorm;
    if(beta>0) psi[ipsiz] = (1.0f-2.0f*beta) * ps1 + beta * psold;
  }
}
