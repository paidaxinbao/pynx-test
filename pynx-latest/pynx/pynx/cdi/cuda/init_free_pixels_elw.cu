/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Init free pixels from a list of islands coordinates.
* Each thread handles one island.
*
* This is not optimised (no coalesced memory transfers, ..) but is used just once so should be good enough,
* and still much faster than on CPU.
*/
__device__ void init_free_pixels(int i, const int* ix, const int* iy, const int* iz,
                                 float* iobs, const float* iobs0,
                                 const int nx, const int ny, const int nz,
                                 const int nb, const int radius)
{
  // Coordinates of the island handled by this thread
  const int x0 = ix[i];
  const int y0 = iy[i];
  const int z0 = iz[i];
  for(int dx=-radius; dx<=radius; dx++)
  {
    int x = x0 + dx;
    x = (x + radius * nx) % nx;
    for(int dy=-radius; dy<=radius; dy++)
    {
      int y = y0 + dy;
      y = (y + radius * ny) % ny;
      for(int dz=-radius; dz<=radius; dz++)
      {
        int z = z0 + dz;
        z = (z + radius * nz) % nz;
        if((dx*dx + dy*dy + dz*dz)<=(radius*radius))
        {
          const int ii = x + nx * (y + ny * z);
          const float v = iobs0[ii];
          if(v>=0) // otherwise the pixel is masked
          {
            // Use an atomic operation to handle collisions ?
            atomicExch(&iobs[ii], -v-1);
          }
        }
      }
    }
  }
}
