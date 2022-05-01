/** Apply a quadratic phase factor to a 5-dimensional Psi stack, with dimensions:
* nb_proj, nb_z, nb_obj, nb_probe, ny, nx
* The phase factor should be only dependant on the distance, so f must be an array with nb_z values.
*
* This kernel function can be called for the whole psi array.
*/
__device__ void QuadPhase(const int i, complexf *psi, float *f, const bool forward, const float scale,
                          const int nb_z, const int nb_mode, const int nx, const int ny)
{
  const int ix = i % nx;
  const int iy = (i % (nx * ny)) / nx;
  const int iz = (i / (nx * ny * nb_mode)) % nb_z;

  // Assume ny, nx are multiples of 2. Compute phase factor for an array with its origin at 0
  const float y = (iy - (int)ny *(int)(iy>=((int)ny/2))) / (float)ny ;
  const float x = (ix - (int)nx *(int)(ix>=((int)nx/2))) / (float)nx;

  const float tmp = f[iz] * (x*x + y*y);
  float s, c;
  __sincosf(tmp , &s, &c);
  if(forward) s = -s;

  const complexf d = psi[i];
  psi[i] = scale * complexf(d.real()*c - d.imag()*s , d.imag()*c + d.real()*s);
}
