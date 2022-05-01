/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/// HIO
__kernel
void HIO(__global float2 *d, __global float2* dold, __global char *support, const float beta)
{
  const unsigned long i2=get_global_id(0);
  if(support[i2]==0) d[i2] = dold[i2] - beta * d[i2] ;
}


/// HIO, biasing real part to be positive
__kernel
void HIO_real_pos(__global float2 *d, __global float2* dold, __global char *support, const float beta)
{
  const unsigned long i2=get_global_id(0);
  if((support[i2]==0)||(d[i2].x<0)) d[i2] = dold[i2] - beta * d[i2] ;
}


/// Error reduction
__kernel
void ER(__global float2 *d, __global char *support)
{
  const unsigned long i2=get_global_id(0);
  if(support[i2]==0) d[i2] = (float2)(0,0) ;
}


/// Error reduction, forcing real part to be positive
__kernel
void ER_real_pos(__global float2 *d, __global char *support)
{
  const unsigned long i2=get_global_id(0);
  if((support[i2]==0)||(d[i2].x<0)) d[i2] = (float2)(0,0) ;
}


/// Charge flipping
__kernel
void CF(__global float2 *d, __global char *support)
{
  const unsigned long i2=get_global_id(0);
  if(support[i2]==0) d[i2].y = -d[i2].y ;
}


/// RAAR
__kernel
void RAAR(__global float2 *d, __global float2* dold, __global char *support, const float beta)
{
  const unsigned long i2=get_global_id(0);
  if(support[i2]==0) d[i2] = (1 - 2 * beta) * d[i2] + beta * dold[i2];
}

/// RAAR, biasing real part to be positive
__kernel
void RAAR_real_pos(__global float2 *d, __global float2* dold, __global char *support, const float beta)
{
  const unsigned long i2=get_global_id(0);
  if((support[i2]==0)||(d[i2].x<0)) d[i2] = (1 - 2 * beta) * d[i2] + beta * dold[i2];
}
