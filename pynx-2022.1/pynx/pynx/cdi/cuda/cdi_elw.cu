/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/// HIO
__device__ void HIO(const int i, complexf *d, complexf* dold, signed char *support, const float beta)
{
  complexf di = d[i];
  if(support[i]==0) di = dold[i] - beta * di ;
  dold[i] = di;
  d[i] = di;
}


/// HIO, biasing real part to be positive
__device__ void HIO_real_pos(const int i, complexf *d, complexf* dold, signed char *support, const float beta)
{
  complexf di = d[i];
  if((support[i]==0)||(di.real()<0)) di = dold[i] - beta * di ;
  dold[i] = di;
  d[i] = di;
}


/// Error reduction
__device__ void ER(const int i, complexf *d, signed char *support)
{
  if(support[i]==0) d[i] = complexf(0,0) ;
}


/// Error reduction, forcing real part to be positive
__device__ void ER_real_pos(const int i, complexf *d, signed char *support)
{
  if((support[i]==0)||(d[i].real()<0)) d[i] = complexf(0,0) ;
}


/// Charge flipping
__device__ void CF(const int i, complexf *d, signed char *support)
{
  if(support[i]==0) d[i].imag(-d[i].imag()) ;
}

/// Charge flipping, biasing real part to be positive
__device__ void CF_real_pos(const int i, complexf *d, signed char *support)
{
  if((support[i]==0)||(d[i].real()<0)) d[i].imag(-d[i].imag()) ;
}

/// RAAR
__device__ void RAAR(const int i, complexf *d, complexf* dold, signed char *support, const float beta)
{
  complexf di = d[i];
  if(support[i]==0) di = (1 - 2 * beta) * di + beta * dold[i];
  dold[i] = di;
  d[i] = di;
}

/// RAAR, biasing real part to be positive
__device__ void RAAR_real_pos(const int i, complexf *d, complexf* dold, signed char *support, const float beta)
{
  complexf di = d[i];
  if((support[i]==0)||(di.real()<0)) di = (1 - 2 * beta) * di + beta * dold[i];
  dold[i] = di;
  d[i] = di;
}
