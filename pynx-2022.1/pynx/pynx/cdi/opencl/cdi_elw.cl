/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/// HIO
void HIO(const int i, __global float2 *d, __global float2* dold, __global char *support, const float beta)
{
  float2 di = d[i];
  if(support[i]==0) di = dold[i] - beta * di ;
  dold[i] = di;
  d[i] = di;
}


/// HIO, biasing real part to be positive
void HIO_real_pos(const int i, __global float2 *d, __global float2* dold, __global char *support, const float beta)
{
  float2 di = d[i];
  if((support[i]==0)||(di.x<0)) di = dold[i] - beta * di ;
  dold[i] = di;
  d[i] = di;
}


/// Error reduction
void ER(const int i, __global float2 *d, __global char *support)
{
  if(support[i]==0) d[i] = (float2)(0,0) ;
}


/// Error reduction, forcing real part to be positive
void ER_real_pos(const int i, __global float2 *d, __global char *support)
{
  if((support[i]==0)||(d[i].x<0)) d[i] = (float2)(0,0) ;
}


/// Charge flipping
void CF(const int i, __global float2 *d, __global char *support)
{
  if(support[i]==0) d[i].y = -d[i].y ;
}

/// Charge flipping, biasing real part to be positive
void CF_real_pos(const int i, __global float2 *d, __global char *support)
{
  if((support[i]==0)||(d[i].x<0)) d[i].y = -d[i].y ;
}

/// RAAR
void RAAR(const int i, __global float2 *d, __global float2* dold, __global char *support, const float beta)
{
  float2 di = d[i];
  if(support[i]==0) di = (1 - 2 * beta) * di + beta * dold[i];
  dold[i] = di;
  d[i] = di;
}

/// RAAR, biasing real part to be positive
void RAAR_real_pos(const int i, __global float2 *d, __global float2* dold, __global char *support, const float beta)
{
  float2 di = d[i];
  if((support[i]==0)||(di.x<0)) di = (1 - 2 * beta) * di + beta * dold[i];
  dold[i] = di;
  d[i] = di;
}

/*
/// DM1
void DM1(const int i, __global float2 *d, __global float2* dold, __global char *support)
{
  const float2 v = d[i];
  dold[i] = v;
  if(support[i]==0) d[i] = -v;
  else d[i] = v;
}

/// DM1, biasing real part to be positive (need to be checked)
void DM1_real_pos(const int i, __global float2 *d, __global float2* dold, __global char *support)
{
  const float2 v = d[i];
  dold[i] = v;
  if((support[i]==0)||(v.x<0)) d[i] = -v;
  else d[i] = v;
}

/// DM2
void DM2(const int i, __global float2 *d, __global float2* dold, __global char *support)
{
  const float2 vold = dold[i];
  if(support[i]==0) d[i] = vold - d[i];
  else d[i] = 2 * vold - d[i];
}

/// DM2, biasing real part to be positive (need to be checked)
void DM2_real_pos(const int i, __global float2 *d, __global float2* dold, __global char *support)
{
  const float2 vold = dold[i];
  if((support[i]==0)||(vold.x<0)) d[i] = vold - d[i];
  else d[i] = 2 * vold - d[i];
}
*/