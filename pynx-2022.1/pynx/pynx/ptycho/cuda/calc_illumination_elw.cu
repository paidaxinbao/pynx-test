/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2020-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

__device__ void CalcIllumination(const int i, complexf* probe, float* obj_illum,
                                 float* cx, float* cy, const int npsi, const int stack_size,
                                 const int nx, const int ny, const int nxo, const int nyo,
                                 const int nbprobe, const bool interp, const int padding)
{
  const int prx = i % nx;
  const int pry = i / nx;
  const int nxy = nx * ny;
  if((prx<padding) || (prx>=(nx-padding)) || (pry<padding) || (pry>=(ny-padding))) return;

  float n = 0;
  for(int iprobe=0;iprobe<nbprobe;iprobe++)
  {
    const complexf p = probe[i + iprobe*nxy];
    n += dot(p,p);
  }
  for(int j=0;j<npsi;j++)
    bilinear_atomic_add_f(obj_illum, n, cx[j] + prx, cy[j] + pry, 0, nxo, nyo, interp);
}
