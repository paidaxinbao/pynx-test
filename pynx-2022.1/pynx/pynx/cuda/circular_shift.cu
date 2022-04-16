/** Circular shift of the elements of a complex 3d array into a new array
*
*/
__device__ void circular_shift(const int i, complexf *source, complexf *dest, const int dx,
                               const int dy, const int dz, const int nx, const int ny, const int nz)
{
  const int ix = (i % nx + dx + nx) % nx;
  const int iy = ((i % (nx * ny)) / nx + dy + ny) % ny;
  const int iz = ((i % (nx * ny * nz)) / (nx * ny) + dz +nz) % nz;
  dest[ix + nx *(iy + ny * iz)] = source[i];
}
