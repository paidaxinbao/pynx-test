/** Zoom (inplace) an array along an axis using linear interpolation.
*
* This elementwise array must be called with a size equal to the size
* perpendicular to the zoom axis direction.
*
* \param d: the array to zoom.
* \param f: the zoom factor, f>0. If f<1 the data is shrunk, if f>1,
* \param fill: if f<1, values outside the original array are replaced by this
* \param axis: the axis along which to zoom, -1 (x), -2 (y) or -2 (z)
* \param center: the coordinate of the center from which the zoom is calculated.
* If not equal to half of the axis size, the array is assumed to be rolled (e.g. fft-shifted)
* \param nx, ny, nz: the size along each dimensions
*/
__device__ void zoom_complex(int i, complexf *d, const float f, const int axis, const int center,
                             const int nx, const int ny, const int nz, const float fill, const bool norm)
{
  int n, i0, stride;
  float s = 1.0f;
  if(norm) s = 1/f;  // Keep intensity sum constant ?
  if(axis==-1)  // x-axis
  {
    n = nx;  // size along zoomed axis
    // origin pixel for this thread, i is within [0;ny*nz[
    i0 = i * nx;
    stride = 1;  // Step to the next element along the zoomed axis
  }
  else if(axis==-2)
  {
    n = ny;
    // real origin pixel for this thread, i is within [0;nx*nz[
    i0 = (i % nx) + (i / nx) * nx * ny;
    stride = nx;  // Step to the next element along the zoomed axis
  }
  else if(axis==-3)
  {
    n = nz;
    // real origin pixel for this thread, i is within [0;nx*ny[
    i0 = i;
    stride = nx * ny;  // Step to the next element along the zoomed axis
  }

  if(f>1)
  {
    // Enlarge, so start from outside
    for(i=n/2-1;i>0;i--)
    {
      const float iz = (float)i / f;
      const int iz0 = __float2int_rd(iz);
      const float dz = iz - iz0;
      complexf d0, d1;
      if(iz0<n/2) d0 = d[i0 + ((center + iz0 + n) % n) * stride];
      else d0 = fill;
      if((iz0+1)<n/2) d1 = d[i0 + ((center + iz0 + n + 1) % n) * stride];
      else d1 = fill;
      d[i0 + ((center + i + n) % n) * stride] = (d0 * (1-dz) + d1 * dz)*s;
    }
    for(i=-n/2;i<=0;i++)
    {
      const float iz = (float)i / f;
      const int iz0 = __float2int_rd(iz);
      const float dz = iz - iz0;
      complexf d0, d1;
      if(iz0>-n/2) d0 = d[i0 + ((center + iz0 + n) % n) * stride];
      else d0 = fill;
      if((iz0+1)>-n/2) d1 = d[i0 + ((center + iz0 + n + 1) % n) * stride];
      else d1 = fill;
      d[i0 + ((center + i + n) % n) * stride] = (d0 * (1-dz) + d1 * dz) * s;
    }
  }
  else
  {
    // Shrink, so start from inside
    for(i=1;i<n/2;i++)
    {
      const float iz = (float)i / f;
      const int iz0 = __float2int_rd(iz);
      const float dz = iz - iz0;
      complexf d0, d1;
      if(iz0<n/2) d0 = d[i0 + ((center + iz0 + n) % n) * stride];
      else d0 = 0;
      if((iz0+1)<n/2) d1 = d[i0 + ((center + iz0 + n + 1) % n) * stride];
      else d1 = 0;
      d[i0 + ((center + i + n) % n) * stride] = (d0 * (1-dz) + d1 * dz) * s;
    }
    for(i=0;i>=-n/2;i--)
    {
      const float iz = (float)i / f;
      const int iz0 = __float2int_rd(iz);
      const float dz = iz - iz0;
      complexf d0, d1;
      if(iz0>-n/2) d0 = d[i0 + ((center + iz0 + n) % n) * stride];
      else d0 = 0;
      if((iz0+1)>-n/2) d1 = d[i0 + ((center + iz0 + n + 1) % n) * stride];
      else d1 = 0;
      d[i0 + ((center + i + n) % n) * stride] = (d0 * (1-dz) + d1 * dz) * s;
    }
  }
}
