/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/


/** Replace observed intensities by calculated ones
*
* This should be called for the complete iobs array of shape: (nb_proj, nbz, ny, nx)
* Psi has a shape: (nb_proj, nbz, nb_obj, nb_probe, ny, nx)
*/
__device__ void Calc2Obs(const int i, float *iobs, complexf *psi,
                         const int nb_mode, const int nx, const int ny)
{
  const int nxy = nx*ny;
  // Coordinates in iobs array (centered on array)
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;

  // Coordinates in Psi array (centered in (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  // Coordinate of first mode in Psi array
  const int i0 = ipsi + (i / nxy) * (nxy * nb_mode);

  float dc2=0;
  for(unsigned int mode=0 ; mode<nb_mode ; mode++)
  {
    const complexf dc = psi[i0 + mode * nxy];
    dc2 += dot(dc,dc);
  }
  iobs[i] = dc2;
}
