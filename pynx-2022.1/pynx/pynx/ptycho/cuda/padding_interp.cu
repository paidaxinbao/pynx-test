/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2021-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

__device__ void PaddingInterp(const int i, complexf* d, const int nx, const int ny, const int padding)
{
  const int ix = i % nx;
  const int iz = i / (nx * ny);
  const int iy = (i - (nx * ny * iz)) / nx;

//  printf("i=%5d: %3d %3d %1d %1d %1d %1d %1d   %1d\n", i, ix, iy, iz, ix >= padding, iy >= padding, ix < (nx-padding), iy < (ny-padding),
//          (ix >= padding) && (iy >= padding) && (ix < (nx-padding)) && (iy < (ny-padding)));

  if((ix >= padding) && (iy >= padding) && (ix < (nx-padding)) && (iy < (ny-padding))) return;

  // Simple shifting interpolation. Crude & not very efficient memory-wise but should not matter
  // as this is executed maybe once per analysis
  complexf v = 0;
  float n = 0;
  const int range = 2 * padding + 2;
  for(int dx=-range; dx<range; dx+=1)
    for(int dy=-range; dy<range; dy+=1)
    {
      const int ix1 = (ix + dx + nx) % nx;
      const int iy1 = (iy + dy + ny) % ny;
      if( (ix1 >= padding) && (ix1 < (nx-padding)) && (iy1 >= padding) && (iy1 < (ny-padding)) )
      {
        // Inverse distance**2 weighting
        const float w = 1 / float(dx * dx + dy * dy);
        v += d[ix1 + nx * (iy1 + ny * iz)] * w ;
        n += w;
      }
    }
  d[i] = v / n;
}
