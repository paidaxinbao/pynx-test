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
__device__ void ProjectionAmplitude(const int i, float *iobs, complexf *psi,
                                    const int nb_mode,
                                    const int nx, const int ny)
{
  const float obs = iobs[i];
  if(obs < 0) return;

  const int nxy = nx*ny;
  // Coordinates in iobs array (centered on array)
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;

  // Coordinates in Psi array (centered in (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));

  // Coordinate of first mode in Psi array
  const int i0 = ix1 + iy1 * nx + (i / nxy) * (nxy * nb_mode);

  float dc2=0;
  for(unsigned int mode=0 ; mode<nb_mode ; mode++)
  {
    const complexf dc = psi[i0 + mode * nxy];
    dc2 += dot(dc,dc);
  }

  // Normalization to observed amplitude, taking into account all modes
  dc2 = fmaxf(dc2, 1e-12f); // TODO: KLUDGE ? 1e-12f is arbitrary
  const float d = sqrtf(obs) / sqrtf(dc2);
  for(unsigned int mode=0 ; mode<nb_mode ; mode++)
  {
    psi[i0 + mode * nxy] *= d;
  }
}
