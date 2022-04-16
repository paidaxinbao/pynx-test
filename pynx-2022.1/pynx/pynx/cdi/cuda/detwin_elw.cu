__device__ void DetwinX(const int i, signed char *support, const int c,
                        const int nx)
{
  // position along x relative to the center of mass c
  const int ix = (i % nx - c + nx) % nx;
  if(ix > (nx/2)) support[i]=0;
}

__device__ void DetwinY(const int i, signed char *support, const int c,
                        const int nx, const int ny)
{
  // position along y relative to the center of mass c
  const int iy = (i % (nx * ny) / nx - c + ny) % ny;
  if(iy > (ny/2)) support[i]=0;
}

__device__ void DetwinZ(const int i, signed char *support, const int c,
                        const int nx, const int ny, const int nz)
{
  // position along z relative to the center of mass c
  const int iz = (i / (nx * ny) - c + nz) % nz;
  if(iz > (nz/2)) support[i]=0;
}
