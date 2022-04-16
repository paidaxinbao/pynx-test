#define twopi 6.2831853071795862f
#define pi 3.1415926535897932f
#define N %(N)d
#define NZ %(NZ)d
#define NZN2 %(NZN2)d

/** Update object projections and probe from psi.
* Object has 4 dimensions: projections, y, x (no object modes)
* Probe has 4 dimensions: z (propagation distances), modes, y, x.
* The 4-dimensional Psi stack is calculated, with dimensions:
*    projections, propagation distances, y, x
*
* This kernel function should be called for a single projection.
* Each thread updates a single pixel in the object array.
* It is not useful to update multiple projections at the same time,
* as the shifts are different between projections the probe coordinates
* will be different so no common memory transfer is possible.
*
* N must be replaced by the number of probe modes, NZ the number of distances,
* and NZN2 by NZ*N*2
* This will:
* - update the object for the given projection
* - compute the new probe modes (which will need to be orthogonalised afterwards)
* - compute the new probe mode coefficients for the projection which can be reduced
*/
__device__ float_%(NZN2)d Psi2ObjProbeRedN(const int i, complexf* obj, complexf* obj_old, complexf* probe, complexf* psi,
                                  complexf* probe_new, float* probe_new_norm, float* probe_coeffs, float* obj_phase0,
                                  float *dx, float *dy, signed char *sample_flag,
                                  const int nx, const int ny, const float obj_min, const float obj_max,
                                  const float reg_obj_smooth,
                                  const float beta_delta, const float weight_empty)
{
  // Coordinates in object array (origin at array center)
  const int ix = i %% nx;
  const int iy = i / nx;
  // Coordinates in Psi array (origin at (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  // New probe coefficients for the projection, decomposed on the old probe
  // shape: (2, nz, N=nb_probe)
  float_%(NZN2)d coeffsnew(0);

  if(sample_flag[0]>0)
  { // Update object & probe
    ////////////////////////////// Object update //////////////////////////////////////////
    complexf onew(0);
    float onorm(0);

    for(int iz=0;iz<NZ;iz++)
    {
      // Pixel coordinate in the first probe mode
      const int i1 = ixyz(__float2int_rn(ix-dx[iz]), __float2int_rn(iy-dy[iz]), 0, nx, ny);

      // We want to update the object array, so we need to read the Psi values
      // corresponding to the same object pixel -> interpolate Psi and Probe

      const complexf ps = bilinear(psi, ix-dx[iz], iy-dy[iz], iz, nx, ny, false, true);
      complexf pr=0;
      for(int iprobe=0; iprobe<N; iprobe++)
        pr += probe[i1 + nx * ny * (iprobe + N * iz)] * probe_coeffs[iprobe + N * iz];
        //pr += bilinear(probe, ix-dx[iz], iy-dy[iz], iprobe + N*iz, nx, ny, false, false) * probe_coeffs[iprobe + N * iz];

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
    // Keep old object and update
    const complexf objold = obj[i];
    obj[i] = onew;

    // Use old object for probe update ?
    #if 0
    onew = objold;
    onorm = dot(onew,onew);
    #endif

    ////////////////////////////// Probe modes & coefficients update //////////////////////////////////////////

    // We don't use interpolation, so only one pixel is updated
    for(int iz=0;iz<NZ;iz++)
    {
      // Pixel coordinate in the first probe mode
      const int i1 = ixyz(__float2int_rn(ix-dx[iz]), __float2int_rn(iy-dy[iz]), 0, nx, ny);

      // Pixel in the Psi array
      const complexf ps = psi[ixyz_shift(__float2int_rn(ix-dx[iz]), __float2int_rn(iy-dy[iz]), iz, nx, ny)];

      // Read old probe modes and coefficients
      complexf pri[N];
      float e_zjp[N];
      for(int iprobe=0; iprobe<N; iprobe++)
      {
        pri[iprobe] = probe[i1 + nx * ny *(iprobe + N * iz)];
        e_zjp[iprobe] = probe_coeffs[iprobe + N * iz];
      }

      // Compute new probe modes & coefficients
      for(int iprobe=0; iprobe<N; iprobe++)
      {
        complexf p =  0;
        for(int iprobe1=0; iprobe1<N; iprobe1++)
          p += ((float)(iprobe1 != iprobe) * e_zjp[iprobe1]) * pri[iprobe1];

        p = ps - complexf(onew.real()*p.real() - onew.imag()*p.imag(),
                          onew.real()*p.imag() + onew.imag()*p.real());

        // Probe
        probe_new[i1 + nx * ny *(iprobe + N * iz)] += e_zjp[iprobe] *
          complexf(onew.real()*p.real() + onew.imag()*p.imag(),
                   onew.real()*p.imag() - onew.imag()*p.real());
        // Probe norm
        probe_new_norm[i1 + nx * ny *(iprobe + N * iz)] += e_zjp[iprobe] * e_zjp[iprobe] * onorm;

        const complexf op = complexf(onew.real()*pri[iprobe].real()-onew.imag()*pri[iprobe].imag(),
                                     onew.real()*pri[iprobe].imag()+onew.imag()*pri[iprobe].real());
        // Coefficient
        coeffsnew[iprobe + N * iz] = op.real() * p.real() + op.imag() * p.imag();
        coeffsnew[iprobe + N * (iz+NZ)] = norm(op);
      }
    }
  }
  else
  { // sample_flag==0 => Update probe only
    // Update probe modes and modes coefficients
    // No need to interpolate: psi=probe here (with a fft-shift between)
    for(int iz=0;iz<NZ;iz++)
    {
      // Pixel in the Psi array
      const complexf ps = psi[ipsi + nx * ny * iz];

      // Read old probe modes and coefficients
      complexf pri[N];
      float e_zjp[N];
      for(int iprobe=0; iprobe<N; iprobe++)
      {
        pri[iprobe] = probe[i + nx * ny *(iprobe + N * iz)];
        e_zjp[iprobe] = probe_coeffs[iprobe + N * iz];
      }

      // Compute new probe modes & coefficients
      for(int iprobe=0; iprobe<N; iprobe++)
      {
        complexf p =  ps;
        for(int iprobe1=0; iprobe1<N; iprobe1++)
          p -= ((float)(iprobe1 != iprobe) * e_zjp[iprobe1]) * pri[iprobe1];

        // Probe
        probe_new[i + nx * ny *(iprobe + N * iz)] += e_zjp[iprobe] * p;

        // Probe norm
        probe_new_norm[i + nx * ny *(iprobe + N * iz)] += e_zjp[iprobe] * e_zjp[iprobe];

        const complexf op = pri[iprobe];

        // Coefficient
        coeffsnew[iprobe + N * iz] = op.real() * p.real() + op.imag() * p.imag();
        coeffsnew[iprobe + N * (iz+NZ)] = norm(op);
      }
    }
  }
  // Return the probe modes coefficients to be reduced
  return coeffsnew;
}

/** Compute the new projection mode coefficients from the orthonormal
* probe modes and the per-projection estimated probe.
*
* This kernel function should be called for a single projection.
* Each thread updates a single pixel in the object array.
*
* N must be replaced by the number of probe modes, NZ the number of distances,
* and NZN2 by NZ*N*2
* This will:
* - update the object for the given projection
* - compute the new probe modes (which will need to be orthogonalised afterwards)
* - compute the new probe mode coefficients for the projection which can be reduced
*/
__device__ float_%(NZN2)d Proj2ProbeMode(const int i, complexf* probe, complexf* psi,
                                         float* probe_coeffs, const int nx, const int ny)
{
  // Coordinates in probe array (origin at array center)
  const int ix = i %% nx;
  const int iy = i / nx;
  // Coordinates in Psi array (origin at (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  // New probe coefficients for the projection, to be reduced
  // shape: (2, nz, N=nb_probe)
  float_%(NZN2)d coeffsnew(0);

  for(int iz=0;iz<NZ;iz++)
  {
    // Updated illumination which needs to be decomposed into modes
    const complexf prnew = psi[ipsi + nx * ny * iz];
    // Pixel coordinate in the first probe mode
    const int i1 = i + nx * ny * iz;

    for(int iprobe=0; iprobe<N; iprobe++)
    {
      const complexf pr = probe[i1 + nx * ny *(iprobe + N * iz) ];
      coeffsnew[iprobe + N * iz] = prnew.real() * pr.real() + prnew.imag() * pr.imag();
      coeffsnew[iprobe + N * (iz + NZ)] = norm(pr);
    }
  }

  // Return the probe modes coefficients to be reduced
  return coeffsnew;
}