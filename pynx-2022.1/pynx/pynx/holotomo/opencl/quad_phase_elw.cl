/** Apply a quadratic phase factor to a 5-dimensional Psi stack, with dimensions:
* nb_proj, nb_z, nb_obj, nb_probe, ny, nx
* The phase factor should be only dependant on the distance, so f must be an array with nb_z values.
*
* This kernel function should be called for a 2D array, but will be applied to all z, views and modes.
*/
void QuadPhase(const int i, __global float2 *d, __global float * f, const float scale,
               const int nb_proj, const int nb_z, const int nb_mode, const int nx, const int ny)
{
  const int ix = i % nx;
  const int iy = i / nx;

  // Assume ny, nx are multiples of 2. Compute phase factor for an array with its origin at 0
  const float y = (iy - (int)ny *(int)(iy>=((int)ny/2))) / (float)ny ;
  const float x = (ix - (int)nx *(int)(ix>=((int)nx/2))) / (float)nx;

  for(int iz=0; iz < nb_z; iz++)
  {
    const float tmp = f[iz] * (x*x + y*y);
    // NOTE WARNING: if arg becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
    const float s = native_sin(tmp);
    const float c = native_cos(tmp);
    for(int i_mode=0;i_mode<nb_mode; i_mode++)
    {
      for(int i_proj=0 ; i_proj < nb_proj ; i_proj++)
      {
        const int i1 = i + nx * ny * (i_mode + nb_mode * (iz + i_proj * nb_z));
        const float2 d1 = d[i1];
        d[i1] = scale * (float2) (d1.x*c - d1.y*s , d1.y*c + d1.x*s);
      }
    }
  }
}
