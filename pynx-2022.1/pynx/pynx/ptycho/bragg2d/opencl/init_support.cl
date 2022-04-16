/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Init support between 0 and 100 based on an equation and a Monte-Carlo integration. The string EQUATION must be
* replaced by the expression of x, y, z (computed floating point coordinates) which must be True
* to be inside the object.
* In the resulting support array, a value of 0 indicates a voxel fully outside the support, 100 fully inside, and
* any value in between is partially inside the support.
*
* \param i: 1-dimensional index of the voxel
* \param support: support array to be initialised
* \param m: matrix to transform the integer object coordinates to xyz coordinates in the laboratory (2D probe) frame.
*           The origin is assumed to be at the center of the object.
* \param nxo, nyo, nzo: object shape
* \param ix0, iy0, iz0: shifted origin of the support array
*/
void InitSupportEq(const int i, __global char *support, __global float* m, const int nxo, const int nyo, const int nzo,
                   const int ix0, const int iy0, const int iz0)
{
  const int ixo = i % nxo - nxo / 2 + ix0;
  const int iyo = (i % (nxo * nyo)) / nxo - nyo / 2 + iy0;
  const int izo = i / (nxo * nyo) - nzo / 2 + iz0;

  // Now perform a Monte-Carlo integration to test which part of the voxel is inside the object
  #define NB_MC 1000
  unsigned int long ir = i;
  unsigned int ct_inside = 0;
  for(int ii=0 ;  ii < NB_MC ; ii++)
  {
    // Pseudo-random integer, maximum value is 2**31 = 2147483648ul
    ir= random_lcg(ir);
    const float ix = ixo + (float)ir / 2147483648.0f;
    ir= random_lcg(ir);
    const float iy = iyo + (float)ir / 2147483648.0f;
    ir= random_lcg(ir);
    const float iz = izo + (float)ir / 2147483648.0f;
    const float x = m[0] * ix + m[1] * iy + m[2] * iz ;
    const float y = m[3] * ix + m[4] * iy + m[5] * iz ;
    const float z = m[6] * ix + m[7] * iy + m[8] * iz ;
    // if((izo == 0) && (iyo ==0) && (ixo ==0) && ii < 50)
    //   printf("CL[%6d] ixyz(%3d, %3d, %3d) xyz0(%8g, %8g, %8g) xyz(%8g, %8g, %8g) : %4d\n", i, ixo, iyo, izo, x0, y0, z0, x, y, z, EQUATION_INSIDE_SUPPORT);
    if(EQUATION_INSIDE_SUPPORT) ct_inside += 1;
  }
  support[i] = 100 * ct_inside / NB_MC;
}
