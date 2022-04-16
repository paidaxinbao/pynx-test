/** Multiply object and probe to get a Psi to be propagated.
* Object has 4 dimensions: projections, modes, y, x.
* Probe has 4 dimensions: z (propagation distances), modes, y, x.
* The 6-dimensional Psi stack is calculated, with dimensions:
*    projections, propagation distances, object modes, probe modes, y, x
*
* This kernel function should be called for a 2D array, but will be applied to all z, projections and modes.
*
* This version assumes the probe is different for each z (propagation distances),
* and that the probe modes are incoherent (each probe mode is propagated independently)
*/
__device__ void ObjectProbeZMultDRAP(const int i, complexf* obj, complexf* probe, complexf* psi, complexf* psiold,
                      float *dx, float *dy, signed char *sample_flag,
                      const int nb_proj, const int nb_z, const int nx, const int ny)
{
  // Coordinates in object array (origin at array center)
  const int ix = i % nx;
  const int iy = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  for(int iproj=0;iproj<nb_proj;iproj++)
  {
    complexf o;
    for(int iz=0;iz<nb_z;iz++)
    {
      // We have to read the object for each z due to shifts
      if(sample_flag[iproj]) o = bilinear(obj, dx[iz + iproj * nb_z]+ix, dy[iz + iproj * nb_z]+iy,
                                          iproj, nx, ny, false, false);

      const complexf p = probe[i + nx * ny * iz];

      const int i1 = ipsi + nx * ny * (iz + nb_z * iproj);
      if(sample_flag[iproj])
        psi[i1] += complexf(o.real() * p.real() -o.imag() * p.imag(), o.real() * p.imag() + o.imag() * p.real());
      else
        psi[i1] += p;

      psiold[i1] = psi[i1];
    }
  }
}

/** Multiply object and probe to get a Psi to be propagated.
* Object has 3 dimensions: projections, y, x. (no mode)
* Probe has 4 dimensions: z (propagation distances), modes, y, x.
* The 4-dimensional Psi stack is calculated, with dimensions:
*    projections, propagation distances, y, x
*
* This kernel function should be called for a 2D array, but will be applied to all z and projections.
*
* This version assumes the probe is different for each z (propagation distances),
* and that the probe modes are coherent - the illumination is computed for each projection
* as a linear combination of the probe modes.
*/

__device__ void ObjectProbeCohZMultDRAP(const int i, complexf* obj, complexf* probe, complexf* psi, complexf* psiold,
                      float *dx, float *dy, signed char *sample_flag, float *probe_coeffs,
                      const int nb_proj, const int nb_z, const int nb_probe,
                      const int nx, const int ny)
{
  // Coordinates in object array (origin at array center)
  const int ix = i % nx;
  const int iy = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  for(int iproj=0;iproj<nb_proj;iproj++)
  {
    complexf o;
    for(int iz=0;iz<nb_z;iz++)
    {
      // We have to read the object for each z due to shifts
      if(sample_flag[iproj]) o = bilinear(obj, dx[iz + iproj * nb_z]+ix, dy[iz + iproj * nb_z]+iy,
                                          iproj, nx, ny, false, false);

      // Assume that the probe values will be cached, so no need to store them
      complexf p=0;
      for(int iprobe=0; iprobe<nb_probe; iprobe++)
        p += probe[i + nx * ny * (iprobe + nb_probe * iz)] * probe_coeffs[iprobe + nb_probe * (iz + nb_z * iproj)];

      const int i1 = ipsi + nx * ny * (iz + nb_z * iproj);
      if(sample_flag[iproj])
        psi[i1] += complexf(o.real() * p.real() -o.imag() * p.imag(), o.real() * p.imag() + o.imag() * p.real());
      else
        psi[i1] += p;

      psiold[i1] = psi[i1];
    }
  }
}
