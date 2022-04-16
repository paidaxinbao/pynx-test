
/** Compute the scale factor between calculated and observed intensities. Negative observed intensities are
* treated as masked and ignored. The reduced scale factor should be multiplied by the calculated intensities to match
* observed ones.
*/

__device__ complexf scale_obs_calc(const int i, float* obs, complexf *psi,
                                   const int nx, const int ny, const int nb_mode)
{
  const float iobs = obs[i];
  if(iobs < 0) return complexf(0,0);

  const int nxy = nx*ny;
  // 2D coordinates in iobs array (centered on array)
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;
  const int izp = i / nxy;

  // Coordinates in Psi array (centered in (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));

  // Coordinate of first mode in Psi array
  const int i0 = ix1 + iy1 * nx + izp * (nxy * nb_mode);

  float dc2 = 0;
  for(unsigned int mode=0 ; mode<nb_mode ; mode++)
  {
    const complexf dc = psi[i0 + mode * nxy];
    dc2 += dot(dc,dc);
  }

  return complexf(iobs, dc2);
}
