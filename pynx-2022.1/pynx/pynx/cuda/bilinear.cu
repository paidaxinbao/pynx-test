/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/


/// get pixel coordinate with periodic boundary conditions along x and y
__device__ int ixyz(const int ix, const int iy, const int iz, const int nx, const int ny)
{
    const int iix = (ix + nx) % nx;
    const int iiy = (iy + ny) % ny;
    return (iz * ny + iiy) * nx + iix;
}


/// get pixel coordinate with periodic boundary conditions along x and y, for a fft-shifted array (along x and y)
__device__ int ixyz_shift(const int ix, const int iy, const int iz, const int nx, const int ny)
{
    const int iix = (ix + nx / 2) % nx;
    const int iiy = (iy + ny / 2) % ny;
    return (iz * ny + iiy) * nx + iix;
}


/** Returns bilinear interpolated value from a complex array, and floating points coordinates.
* Interpolation is only done in the x-y plane, the z coordinate is an integer.
* Array is wrapped around periodic boundaries, and is treated as fft-shifted in the xy plane if fft_shift is true.
* If interp=false, nearest pixel interpolation is used
*/
__device__ complexf bilinear(complexf* v, const float x, const float y, const int iz,
                             const int nx, const int ny, const bool interp, const bool fft_shift)
{
  if(!interp)
  {
   const int x0 = __float2int_rn(x);
   const int y0 = __float2int_rn(y);
   if(fft_shift)
     return v[ixyz_shift(x0  , y0  , iz, nx, ny)];
   else
     return v[ixyz(x0  , y0  , iz, nx, ny)];
  }
  const int x0 = __float2int_rd(x);
  const int y0 = __float2int_rd(y);
  const float dx = x - x0;
  const float dy = y - y0;

  if((abs(dx) < 1e-5)  && (abs(dy) < 1e-5))
  {
    if(fft_shift)
      return v[ixyz_shift(x0  , y0  , iz, nx, ny)];
    else
      return v[ixyz(x0  , y0  , iz, nx, ny)];
  }

  if(fft_shift)
  {
    const complexf v00 = v[ixyz_shift(x0  , y0  , iz, nx, ny)];
    const complexf v01 = v[ixyz_shift(x0+1, y0  , iz, nx, ny)];
    const complexf v10 = v[ixyz_shift(x0  , y0+1, iz, nx, ny)];
    const complexf v11 = v[ixyz_shift(x0+1, y0+1, iz, nx, ny)];
    return v00 * ((1-dx) * (1-dy)) + v01 * (dx * (1-dy)) + v10 * ((1-dx) * dy) + v11 * (dx * dy);
  }
  else
  {
    const complexf v00 = v[ixyz(x0  , y0  , iz, nx, ny)];
    const complexf v01 = v[ixyz(x0+1, y0  , iz, nx, ny)];
    const complexf v10 = v[ixyz(x0  , y0+1, iz, nx, ny)];
    const complexf v11 = v[ixyz(x0+1, y0+1, iz, nx, ny)];
    return v00 * ((1-dx) * (1-dy)) + v01 * (dx * (1-dy)) + v10 * ((1-dx) * dy) + v11 * (dx * dy);
  }
}

/** Distribute a bilinear complex interpolated value onto a complex array, using floating points coordinates.
* Interpolation is only done in the x-y plane, the z coordinate is an integer.
* Array is wrapped around periodic boundaries.
* If interp=false, nearest pixel interpolation is used.
*/
__device__ void bilinear_atomic_add_c(complexf *v, complexf val, const float x, const float y, const int iz,
                                      const int nx, const int ny, const bool interp)
{
  if(!interp)
  {
   const int x0 = __float2int_rn(x);
   const int y0 = __float2int_rn(y);
   atomicAdd(&v[ixyz(x0  , y0  , iz, nx, ny)], val);
   return;
  }
  const int x0 = __float2int_rd(x);
  const int y0 = __float2int_rd(y);

  const float dx = x - x0;
  const float dy = y - y0;

  if((abs(dx) < 1e-5)  && (abs(dy) < 1e-5))
    atomicAdd(&v[ixyz(x0  , y0  , iz, nx, ny)], val);

  atomicAdd(&v[ixyz(x0  , y0  , iz, nx, ny)], val * ((1-dx)*(1-dy)));
  atomicAdd(&v[ixyz(x0+1, y0  , iz, nx, ny)], val * (   dx *(1-dy)));
  atomicAdd(&v[ixyz(x0  , y0+1, iz, nx, ny)], val * ((1-dx)*  dy ));
  atomicAdd(&v[ixyz(x0+1, y0+1, iz, nx, ny)], val * (   dx *  dy ));
}

/** Distribute a bilinear floating-point interpolated value onto a complex array, using floating points coordinates.
* Interpolation is only done in the x-y plane, the z coordinate is an integer.
* Array is wrapped around periodic boundaries.
* If interp=false, nearest pixel interpolation is used.
*/
__device__ void bilinear_atomic_add_f(float *v, float val, const float x, const float y, const int iz,
                                      const int nx, const int ny, const bool interp)
{
  if(!interp)
  {
   const int x0 = __float2int_rn(x);
   const int y0 = __float2int_rn(y);
   atomicAdd(&v[ixyz(x0  , y0  , iz, nx, ny)], val);
   return;
  }

  const int x0 = __float2int_rd(x);
  const int y0 = __float2int_rd(y);

  const float dx = x - x0;
  const float dy = y - y0;

  if((abs(dx) < 1e-5)  && (abs(dy) < 1e-5))
    atomicAdd(&v[ixyz(x0  , y0  , iz, nx, ny)], val);

  atomicAdd(&v[ixyz(x0  , y0  , iz, nx, ny)], val * ((1-dx)*(1-dy)));
  atomicAdd(&v[ixyz(x0+1, y0  , iz, nx, ny)], val * (   dx *(1-dy)));
  atomicAdd(&v[ixyz(x0  , y0+1, iz, nx, ny)], val * ((1-dx)*  dy ));
  atomicAdd(&v[ixyz(x0+1, y0+1, iz, nx, ny)], val * (   dx *  dy ));
}
