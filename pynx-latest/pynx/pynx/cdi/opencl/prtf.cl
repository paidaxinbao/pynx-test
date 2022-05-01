/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2020-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/


void prtf(const int i, __global float2 *obj, __global float* iobs, __global float* shell_calc,
          __global float* shell_obs, __global int *shell_nb,
          const int nb_shell, const int f_nyquist, const int nx, const int ny, const int nz)
{
  int ix = i % nx;
  int iy = (i % (nx * ny)) / nx;
  int iz = i / (nx * ny);
  ix = ix - nx * (ix >= (nx / 2));
  iy = iy - ny * (iy >= (ny / 2));
  if(nz>1) iz = iz - nz * (iz >= (nz / 2));
  const int ir= floor(native_sqrt((float)(ix * ix + iy * iy + iz * iz)) / f_nyquist * nb_shell);
  if(ir < nb_shell)
  {
     float obs = iobs[i];
     // if(ir <= 5) printf("CL PRTF (%2d, %2d, %2d, %2d): %6e / %6e\\n", ix, iy, iz, ir, dot(obj[i], obj[i]), obs);
     if(obs<=-1e19f) return;
     if(obs<-0.5f) obs = -(obs+1); // Take into account free pixels

     atomic_add_f(&shell_calc[ir], native_sqrt(dot(obj[i], obj[i])));
     atomic_add_f(&shell_obs[ir], native_sqrt(fabs(obs)));
     atomic_add(&shell_nb[ir], 1);
  }
}


void prtf_icalc(const int i, __global float *icalc, __global float* iobs, __global float* shell_calc,
                __global float* shell_obs, __global int *shell_nb,
                const int nb_shell, const int f_nyquist, const int nx, const int ny, const int nz)
{
  int ix = i % nx;
  int iy = (i % (nx * ny)) / nx;
  int iz = i / (nx * ny);
  ix = ix - nx * (ix >= (nx / 2));
  iy = iy - ny * (iy >= (ny / 2));
  if(nz>1) iz = iz - nz * (iz >= (nz / 2));
  const int ir= floor(native_sqrt((float)(ix * ix + iy * iy + iz * iz)) / f_nyquist * nb_shell);
  if(ir < nb_shell)
  {
     float obs = iobs[i];
     // if(ir <= 5) printf("CL PRTF (%2d, %2d, %2d, %2d): %6e / %6e\\n", ix, iy, iz, ir, dot(obj[i], obj[i]), obs);
     if(obs<=-1e19f) return;
     if(obs<-0.5f) obs = -(obs+1); // Take into account free pixels

     atomic_add_f(&shell_calc[ir], native_sqrt(fabs(icalc[i])));
     atomic_add_f(&shell_obs[ir], native_sqrt(fabs(obs)));
     atomic_add(&shell_nb[ir], 1);
  }
}
