/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/



/** Get bilinear-interpolated values from the projected 2D probe, given ix, iy, iz coordinates in the object array.
*
* \param probe2d: the 3D (2D+modes) complex probe array
* \param m: the matrix to transform integer (object reference frame) coordinates to the laboratory frame (in meters)
* \param mode: the probe mode to be returned
* \param px, py: probe pixel size along x and y (in meters)
* \param cx, cy: center (shift) of the probe center (in meters) in the laboratory frame
* \param ix, iy, iz: coordinates of the voxel in the 3D probe array (pixel units)
* \param nx, ny: shape of the 2D probe
*/
float2 interp_probe(__global float2* probe2d, __global float* m, const int mode, const float cx, const float cy,
                    const float px, const float py, const float ix, const float iy, const float iz,
                    const int nx, const int ny)
{
  // Get xy probe pixel coordinates from ix, iy, iz
  // and convert to pixel probe coordinates (the origin being at top, left, so a change along y is needed)
  const float x = nx / 2 - (m[0] * ix + m[1] * iy + m[2] * iz - cx) / px ;
  const float y = ny / 2 - (m[3] * ix + m[4] * iy + m[5] * iz - cy) / py ;

  // round-towards-zero
  const int x0 = convert_int_rtz(x);
  const int y0 = convert_int_rtz(y);
  const float dx = x - x0;
  const float dy = y - y0;

  if((x0<0) || (x0>=(nx-1)) || (y0<0) || (y0>=(ny-1))) return (float2)(0,0);

  const float2 v00 = probe2d[x0     + nx * (y0     + ny * mode)];
  const float2 v01 = probe2d[x0 + 1 + nx * (y0     + ny * mode)];
  const float2 v10 = probe2d[x0     + nx * (y0 + 1 + ny * mode)];
  const float2 v11 = probe2d[x0 + 1 + nx * (y0 + 1 + ny * mode)];
  return v00 * ((1-dx) * (1-dy)) + v01 * (dx * (1-dy)) + v10 * ((1-dx) * dy) + v11 * (dx * dy);
}

/** Given a probe gradient computed in the 3D object space, interpolate and distribute the gradient
* onto the projected 2D probe, given ix, iy, iz coordinates in the object array.
*
* \param probe2d_gradient: the 3D (2D+modes) complex probe gradient array
* \param val: the value which will be distributed to the different pixels
* \param m: the matrix to transform integer (object reference frame) coordinates to the laboratory frame (in meters)
* \param mode: the probe mode to be returned
* \param px, py: probe pixel size along x and y (in meters)
* \param cx, cy: center (shift) of the probe center (in meters) in the laboratory frame
* \param ix, iy, iz: coordinates of the voxel in the 3D probe array (pixel units)
* \param nx, ny: shape of the 2D probe
*/
void interp_probe_spread(__global float2* probe2d_gradient, const float2 val, __global float* m, const int mode,
                         const float cx, const float cy, const float px, const float py,
                         const float ix, const float iy, const float iz, const int nx, const int ny)
{
  // Get xy probe pixel coordinates from ix, iy, iz
  // and convert to pixel probe coordinates (the origin being at top, left, so a change along y is needed)
  const float x = nx / 2 - (m[0] * ix + m[1] * iy + m[2] * iz - cx) / px ;
  const float y = ny / 2 - (m[3] * ix + m[4] * iy + m[5] * iz - cy) / py ;

  // round-towards-zero
  const int x0 = convert_int_rtz(x);
  const int y0 = convert_int_rtz(y);
  const float dx = x - x0;
  const float dy = y - y0;

  if((x0<0) || (x0>=(nx-1)) || (y0<0) || (y0>=(ny-1))) return ;

  atomic_add_c(&probe2d_gradient[x0     + nx * (y0     + ny * mode)], ((1-dx) * (1-dy)) * val);
  atomic_add_c(&probe2d_gradient[x0 + 1 + nx * (y0     + ny * mode)], (dx * (1-dy)) * val);
  atomic_add_c(&probe2d_gradient[x0     + nx * (y0 + 1 + ny * mode)], ((1-dx) * dy) * val);
  atomic_add_c(&probe2d_gradient[x0 + 1 + nx * (y0 + 1 + ny * mode)], (dx * dy) * val);
}
