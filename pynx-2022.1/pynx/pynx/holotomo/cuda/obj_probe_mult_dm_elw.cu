/** Multiply object and probe to get a Psi to be propagated.
* Version for DM, computes 2*P*O-Psi
*
* Object has 4 dimensions: projections, modes, y, x.
* Probe has 4 dimensions: z (propagation distances), modes, y, x.
* The 6-dimensional Psi stack is calculated, with dimensions:
*    projections, propagation distances, object modes, probe modes, y, x
*
* This kernel function should be called for a 2D array, but will be applied to all z, projections and modes.
*
* This version assumes the probe is different for each z (propagation distances)
*/
__device__ void ObjectProbe2PsiDM1(const int i, complexf* obj, complexf* probe, complexf* psi,
                      float *dx, float *dy, signed char *sample_flag,
                      const int nb_proj, const int nb_z, const int nb_obj, const int nb_probe,
                      const int nx, const int ny)
{
  // Coordinates in object array (origin at array center)
  const int ix = i % nx;
  const int iy = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  for(int iobj=0; iobj<nb_obj; iobj++)
  {
    for(int iprobe=0; iprobe<nb_probe; iprobe++)
    {
      for(int iproj=0;iproj<nb_proj;iproj++)
      {
        if(sample_flag[iproj])
        {
          complexf o;
          for(int iz=0;iz<nb_z;iz++)
          {
            // We have to read the object for each z due to shifts
            if(sample_flag[iproj]) o = bilinear(obj, ix+dx[iz + iproj * nb_z],
                                                iy+dy[iz + iproj * nb_z], iobj + nb_obj * iproj, nx, ny, false, false);
            // if(sample_flag[iproj]) o = obj[i + nx * ny * (iobj + nb_obj * iproj)];

            const complexf p = probe[ix + nx * (iy + ny * (iprobe + nb_probe * iz))];

            const int i1 = ipsi + nx * ny * (iprobe + nb_probe * (iobj + nb_obj * (iz + nb_z * iproj)));
            if(sample_flag[iobj])
            {
              psi[i1] = 2.0f * complexf(o.real() * p.real() -o.imag() * p.imag(),
                                     o.real() * p.imag() + o.imag() * p.real()) - psi[i1];
            }
            else
            {
              psi[i1] = 2.0f * p - psi[i1];
            }
          }
        }
      }
    }
  }
}

/** 2nd stage Psi update for DM, computes Psi(n-1) - P*O + Psi,
* where Psi is the result of magnitude projection in detector space.
*
* Object has 4 dimensions: projections, modes, y, x.
* Probe has 4 dimensions: z (propagation distances), modes, y, x.
* The 6-dimensional Psi stack is calculated, with dimensions:
*    projections, propagation distances, object modes, probe modes, y, x
*
* This kernel function should be called for a 2D array, but will be applied to all z, projections and modes.
*
* This version assumes the probe is different for each z (propagation distances)
*/
__device__ void ObjectProbe2PsiDM2(const int i, complexf* obj, complexf* probe, complexf* psi, complexf* psi_old,
                      float *dx, float *dy, signed char *sample_flag,
                      const int nb_proj, const int nb_z, const int nb_obj, const int nb_probe,
                      const int nx, const int ny)
{
  // Coordinates in object array (origin at array center)
  const int ix = i % nx;
  const int iy = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  for(int iobj=0; iobj<nb_obj; iobj++)
  {
    for(int iprobe=0; iprobe<nb_probe; iprobe++)
    {
      for(int iproj=0;iproj<nb_proj;iproj++)
      {
        if(sample_flag[iproj])
        {
          for(int iz=0;iz<nb_z;iz++)
          {
            const complexf o = bilinear(obj, ix+dx[iz + iproj * nb_z],
                                        iy+dy[iz + iproj * nb_z], iobj + nb_obj * iproj, nx, ny, false, false);
            const complexf p = probe[ix + nx * (iy + ny * (iprobe + nb_probe * iz))];

            const int i1 = ipsi + nx * ny * (iprobe + nb_probe * (iobj + nb_obj * (iz + nb_z * iproj)));
            if(sample_flag[iobj])
            {
              psi[i1] = psi_old[i1] + psi[i1]
                        - complexf(o.real() * p.real() -o.imag() * p.imag(),
                                   o.real() * p.imag() + o.imag() * p.real()) ;
            }
            else
            {
              psi[i1] = psi_old[i1] - p + psi[i1];
            }
          }
        }
      }
    }
  }
}
