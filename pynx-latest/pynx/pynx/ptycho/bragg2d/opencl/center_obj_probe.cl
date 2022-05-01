/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Kernel to compute the center of mass of the probe*object along the XY axes, and the extent along the Z axis.
*
* \param i: voxel index in the 3D object array
* \param obj: the 3D object array (only one object mode is considered)
* \param probe2d: the 2D probe array (only the first mode is considered)
* \param m: the matrix to transform integer (object reference frame) coordinates to the laboratory frame (in meters)
* \param cx, cy: shift of the illumination position (laboratory frame units, in meters)
* \param pxp, pyp: probe pixel size along the x and y axes
* \param nxp, nyp: probe shape
* \param nxo, nyo: object shape along the X and Y axes
* \return: a vector with (ix * abs(obj*probe), iy * abs(obj*probe), abs(obj*probe),
*                         abs(obj*probe))>0 ? iz : 1e16, abs(obj*probe))>0 ? iz+1 : 1e16, 0, 0 0)
*/
float8 center_obj_probe(const int i, __global float* obj, __global float2* probe, __global float* m,
                        __global float* cx, __global float* cy, const float pxp, const float pyp,
                        const int nxp, const int nyp, const int nxo, const int nyo, const int nzo)
{
  // Coordinates in the 3D object array, relative to the object center
  const int ixo = i % nxo - nxo / 2;
  const int izo = i / (nxo * nyo) - nzo / 2;
  const int iyo = (i % (nxo * nyo)) / nxo - nyo / 2;

  const float2 p = interp_probe(probe, m, 0, cx[0], cy[0], pxp, pyp, ixo, iyo, izo, nxp, nyp);

  if((p.x == .0f) && (p.y == .0f)) return (float8)(0.0f, 0.0f, 0.0f, 1e16f, 1e16f, 0.0f, 0.0f, 0.0f);

  const float2 o = obj[i];

  if((o.x == .0f) && (o.y == .0f)) return (float8)(0.0f, 0.0f, 0.0f, 1e16f, 1e16f, 0.0f, 0.0f, 0.0f);

  const float2 op = (float2) (p.x * o.x - p.y * o.y, p.x * o.y + p.y * o.x);

  const float aop = dot(op, op);

  return (float8)(ixo * aop, iyo * aop, aop, izo, izo + 1, 0.0f, 0.0f, 0.0f);
}

float8 center_obj_probe_red(const float8 a, const float8 b)
{
  float izmin = b.s3;
  if(a.s3 < b.s3) izmin = a.s3;

  float izmax = a.s4;
  if(b.s4 < a.s4) izmax = b.s4;

  return (float8)(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, izmin, izmax, 0, 0, 0);
}
