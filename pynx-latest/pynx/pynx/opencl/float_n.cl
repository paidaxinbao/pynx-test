/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/* Definition of a generic float_n structure for custom reduction.
*
*
*/

typedef struct
{
  float v[FLOAT_N_SIZE];
} float_n;

float_n float_n_zero()
{
  float_n f;
  for(int i=0;i<FLOAT_N_SIZE;i++) f.v[i]=0.0f;
  return f;
}

// TODO: should this be passed as pointers to avoid a copy ?
float_n add(float_n a, float_n b)
{
  float_n f;
  for(int i=0;i<FLOAT_N_SIZE;i++) f.v[i] = a.v[i] + b.v[i];
  return f;
}
