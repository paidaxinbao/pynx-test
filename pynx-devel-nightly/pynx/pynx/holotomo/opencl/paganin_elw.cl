// This function should be called for a 2D (nx*ny) iobs array as first argument, each thread will perform the
// calculation for a given propagation distances (z). Only the first mode is filled with the result of the
// transformation.
void paganin_fourier(const int i, __global float2 *psi, const int iz, const float z_delta, const float mu, const float dk, const int nx, const int ny, const int nz)
{
  // iobs shape is (nb_z, ny, nx)
  // d=psi shape is (nb_obj, nb_probe, nb_z, ny, nx)
  const int ix = i % nx;
  const int iy = (i % (nx * ny)) / nx;

  // Assume ny, nx are multiples of 2. Compute phase factor for an array with its origin at 0
  const float y = (iy - (int)ny *(int)(iy >= ((int)ny / 2))) ;
  const float x = (ix - (int)nx *(int)(ix >= ((int)nx / 2))) ;
  const float mul = mu / (z_delta * dk * dk * (x*x + y*y) + mu);

  const int ii = i + nx * ny * iz;
  psi[ii] = mul * psi[ii];
}

/** This function should be called for the whole object array
* Object has 4 dimensions: modes(=1), y, x.
* The 5-dimensional Psi stack is calculated, with dimensions:
*    object modes(1), probe modes (ignored), propagation distance (z), y, x
*
* only one propagation distance (iz) is taken into account to calculate the object thickness.
*/
void paganin_thickness(const int i, __global float2 *obj, __global float2 *psi, const int iz, const float mu, const float k_delta, const int nx, const int ny)
{
  // Coordinates in original array (origin at array center)
  const int ix = i % nx;
  const int iy = (i % (ny * nx)) / nx;

  // Coordinates in fft-shifted array. Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + nx * iy1;

  const float t = -log(length(psi[i + iz * nx * ny])) / mu;

  // Use approximations if absorption or phase shift is small
  float a = mu * t;
  if(a < 1e-4)
      a = 1 - a * 0.5 ;
  else
    a = exp(-0.5f * a);

  const float alpha = k_delta*t;
  if(alpha<1e-4)
    obj[ipsi] = (float2)(a * (1-alpha*alpha), -a * alpha);
  else
    obj[ipsi] = (float2)(a * cos(alpha), a * sin(-alpha));
}
