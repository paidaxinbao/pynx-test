/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2020-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/


__device__ void prtf(const int i, complexf *obj, float* iobs, float* shell_calc,
          float* shell_obs, int *shell_nb,
          const int nb_shell, const int f_nyquist, const int nx, const int ny, const int nz)
{
  int ix = i % nx;
  int iy = (i % (nx * ny)) / nx;
  int iz = i / (nx * ny);
  ix = ix - nx * (ix >= (nx / 2));
  iy = iy - ny * (iy >= (ny / 2));
  if(nz>1) iz = iz - nz * (iz >= (nz / 2));
  const int ir= floor(sqrtf((float)(ix * ix + iy * iy + iz * iz)) / f_nyquist * nb_shell);
  if(ir < nb_shell)
  {
     float obs = iobs[i];
     if(obs<=-1e19f) return;
     if(obs<-0.5f) obs = -(obs+1); // Take into account free pixels

     atomicAdd(&shell_calc[ir], sqrtf(dot(obj[i], obj[i])));
     atomicAdd(&shell_obs[ir], sqrtf(fabs(obs)));
     atomicAdd(&shell_nb[ir], 1);
  }
}


__device__ void prtf_icalc(const int i, float *icalc, float* iobs, float* shell_calc,
                           float* shell_obs, int *shell_nb,
                           const int nb_shell, const int f_nyquist, const int nx, const int ny, const int nz)
{
  int ix = i % nx;
  int iy = (i % (nx * ny)) / nx;
  int iz = i / (nx * ny);
  ix = ix - nx * (ix >= (nx / 2));
  iy = iy - ny * (iy >= (ny / 2));
  if(nz>1) iz = iz - nz * (iz >= (nz / 2));
  const int ir= floor(sqrtf((float)(ix * ix + iy * iy + iz * iz)) / f_nyquist * nb_shell);
  if(ir < nb_shell)
  {
     float obs = iobs[i];
     if(obs<=-1e19f) return;
     if(obs<-0.5f) obs = -(obs+1); // Take into account free pixels

     atomicAdd(&shell_calc[ir], sqrtf(fabs(icalc[i])));
     atomicAdd(&shell_obs[ir], sqrtf(fabs(obs)));
     atomicAdd(&shell_nb[ir], 1);
  }
}
