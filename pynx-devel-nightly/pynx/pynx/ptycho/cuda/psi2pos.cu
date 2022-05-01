/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Estimate the shift between the back-projected Psi array and the calculated object*probe array.
* This reduction returns ((dopx[i].conj() * dpsi[i]).real(), (dopx[i].conj() * dpsi[i]).real(),
*                          abs(dopx[i])**2, abs(dopy[i])**2)
* where:
*  dopx[i] is the derivative of the object along x, for pixel i, multiplied by the probe (similarly dopy along y)
*  dpsi[i] is the difference between the back-projected Psi array (after Fourier constraints) and obj*probe, at pixel i.
*
* Only the first mode is taken into account.
* if interp=false, nearest pixel interpolation is used for the object*probe calculation.
*/
__device__ my_float4 Psi2PosShift(const int i, complexf* psi, complexf* obj, complexf* probe, float* cx, float* cy,
                                  const float pixel_size, const float f, const int nx, const int ny, const int nxo,
                                  const int nyo, const int ii, const bool interp)
{
  const int prx = i % nx;
  const int pry = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Apply Quadratic phase factor after far field back-propagation
  const float y = (pry - ny/2) * pixel_size;
  const float x = (prx - nx/2) * pixel_size;
  const float tmp = -f*(x*x+y*y);

  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  float s, c;
  __sincosf(tmp , &s, &c);

  const complexf o = bilinear(obj, cx[ii]+prx, cy[ii]+pry, 0, nxo, nyo, interp, false);
  const complexf p = probe[i];
  const complexf dpsi = complexf(psi[ipsi].real()*c - psi[ipsi].imag()*s - (o.real()*p.real() - o.imag()*p.imag()) ,
                                 psi[ipsi].imag()*c + psi[ipsi].real()*s - (o.real()*p.imag() + o.imag()*p.real()));

  // Assume we have some buffer before reaching the array border
  // Gradient is calculated with subpixel interpolation.
  const complexf dox =   bilinear(obj, cx[ii]+prx+0.5f, cy[ii]+pry     , 0, nxo, nyo, true, false)
                       - bilinear(obj, cx[ii]+prx-0.5f, cy[ii]+pry     , 0, nxo, nyo, true, false);
  const complexf doy =   bilinear(obj, cx[ii]+prx     , cy[ii]+pry+0.5f, 0, nxo, nyo, true, false)
                       - bilinear(obj, cx[ii]+prx     , cy[ii]+pry-0.5f, 0, nxo, nyo, true, false);
  const complexf dopx = complexf(dox.real() * p.real() - dox.imag() * p.imag(),
                                 dox.real() * p.imag() + dox.imag() * p.real());
  const complexf dopy = complexf(doy.real() * p.real() - doy.imag() * p.imag(),
                                 doy.real() * p.imag() + doy.imag() * p.real());
  return my_float4(dopx.real() * dpsi.real() + dopx.imag() * dpsi.imag(),
                   dopy.real() * dpsi.real() + dopy.imag() * dpsi.imag(),
                   dopx.real() * dopx.real() + dopx.imag() * dopx.imag(),
                   dopy.real() * dopy.real() + dopy.imag() * dopy.imag());
}

/** Compute the shifts, and return the average
*
*/
__device__ complexf Psi2PosRed(const int i, my_float4* dxy, const float mult, const float max_shift,
                               const float min_shift, const float threshold, const int nb)
{
  float dx = dxy[i].x / fmaxf(dxy[i].z, 1e-30f) * mult;
  float dy = dxy[i].y / fmaxf(dxy[i].w, 1e-30f) * mult;

  if(dxy[i].z < threshold) dx = 0;
  if(dxy[i].w < threshold) dy = 0;

  const float dr  = sqrt(dx * dx + dy * dy);

  if(dr < min_shift)
  {
    dxy[i].x = 0;
    dxy[i].y = 0;
    return complexf(0,0);
  }
  if(dr > max_shift)
  {
    dx *= max_shift / dr;
    dy *= max_shift / dr;
  }

  dxy[i].x = dx;
  dxy[i].y = dy;
  return complexf(dx / nb, dy / nb);
}


/** Estimate the shift between the back-projected Psi array and the calculated object*probe array.
* This computes the sum of ((dopx[i].conj() * dpsi[i]).real(), (dopx[i].conj() * dpsi[i]).real(),
*                            abs(dopx[i])**2, abs(dopy[i])**2)
* where:
*  dopx[i] is the derivative of the object along x, for pixel i, multiplied by the probe (similarly dopy along y)
*  dpsi[i] is the difference between the back-projected Psi array (after Fourier constraints) and obj*probe, at pixel i.
*
* Only the first mode is taken into account.
* if interp=false, nearest pixel interpolation is used for the object*probe calculation.
*
* Version using shared memory, one thread block computes the shift for a single frame.
* The input data array includes a stack of N images, for which the shifts are computed
* in different blocks.
*
* The shifts are computed and stored in dxy.
*/
__global__  void Psi2Pos(complexf* psi, complexf* obj, complexf* probe, float* cx, float* cy,
                         const float pixel_size, const float f, const int nx, const int ny, const int nxo,
                         const int nyo, const char interp, my_float4* dxy)
{
  #define BLOCKSIZE 128
  const int tid = threadIdx.x;
  const int iframe = blockIdx.x;
  __shared__ float dopxpsi[BLOCKSIZE];  dopxpsi[tid] = 0.0f;
  __shared__ float dopypsi[BLOCKSIZE];  dopypsi[tid] = 0.0f;
  __shared__ float dopxn[BLOCKSIZE]; dopxn[tid] = 0.0f;
  __shared__ float dopyn[BLOCKSIZE]; dopyn[tid] = 0.0f;

  const int nxy = nx * ny;
  float s,c;

  const float cxi = cx[iframe];
  const float cyi = cy[iframe];

  // Loop over pixels
  for(int i=tid; i<nxy; i+=BLOCKSIZE)
  {
    const int prx = i % nx;
    const int pry = i / nx;

    // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
    const int iy = pry - ny/2 + ny * (pry<(ny/2));
    const int ix = prx - nx/2 + nx * (prx<(nx/2));
    // We only consider the first mode
    const int ipsi  = ix + nx * (iy + ny * iframe) ;

    // Apply Quadratic phase factor after far field back-propagation
    const float y = (pry - ny/2) * pixel_size;
    const float x = (prx - nx/2) * pixel_size;
    const float tmp = -f*(x*x+y*y);

    __sincosf(tmp , &s, &c);

    // Load object & its gradient
    // TODO: coalesce memory transfers - this is very inefficient (4x to 8x slow-down despite cache,
    //  due to bilinear read with interpolation which could be done without, or manually)
    const complexf o = bilinear(obj, cxi+prx, cyi+pry, 0, nxo, nyo, interp, false);
    // Assume we have some buffer before reaching the array border
    // Gradient is calculated with subpixel interpolation.
    const complexf dox =   bilinear(obj, cxi+prx+0.5f, cyi+pry     , 0, nxo, nyo, true, false)
                         - bilinear(obj, cxi+prx-0.5f, cyi+pry     , 0, nxo, nyo, true, false);
    const complexf doy =   bilinear(obj, cxi+prx     , cyi+pry+0.5f, 0, nxo, nyo, true, false)
                         - bilinear(obj, cxi+prx     , cyi+pry-0.5f, 0, nxo, nyo, true, false);

    const complexf p = probe[i];
    const complexf dpsi = complexf(psi[ipsi].real()*c - psi[ipsi].imag()*s - (o.real()*p.real() - o.imag()*p.imag()) ,
                                   psi[ipsi].imag()*c + psi[ipsi].real()*s - (o.real()*p.imag() + o.imag()*p.real()));

    const complexf dopx = complexf(dox.real() * p.real() - dox.imag() * p.imag(),
                                   dox.real() * p.imag() + dox.imag() * p.real());
    const complexf dopy = complexf(doy.real() * p.real() - doy.imag() * p.imag(),
                                   doy.real() * p.imag() + doy.imag() * p.real());
    dopxpsi[tid] += dopx.real() * dpsi.real() + dopx.imag() * dpsi.imag();
    dopypsi[tid] += dopy.real() * dpsi.real() + dopy.imag() * dpsi.imag();
    dopxn[tid] += dopx.real() * dopx.real() + dopx.imag() * dopx.imag();
    dopyn[tid] += dopy.real() * dopy.real() + dopy.imag() * dopy.imag();
  }
  __syncthreads();

  // Reduce shift value
  for (unsigned int s=BLOCKSIZE/2; s>0; s>>=1)
  {
    if (tid < s)
    {
      dopxpsi[tid] += dopxpsi[tid + s];
      dopypsi[tid] += dopypsi[tid + s];
      dopxn  [tid] += dopxn  [tid + s];
      dopyn  [tid] += dopyn  [tid + s];
    }
    __syncthreads();
  }

  if(tid==0)
  {
    dxy[iframe] = my_float4(dopxpsi[0], dopypsi[0], dopxn[0], dopyn[0]);
  }
}

/** Apply the computed shifts with some limits & threshold applied
* This should be called in a single (128,1,1) block to perform an internal reduction
*/
__global__ void Psi2PosMerge(my_float4* dxy, float* cx, float* cy, const float mult,
                             const float max_shift, const float min_shift,
                             const float threshold, const int nb)
{
  #define BLOCKSIZE2 128
  const int tid = threadIdx.x;
  __shared__ float v1[BLOCKSIZE2];  v1[tid] = 0.0f;
  __shared__ float v2[BLOCKSIZE2];  v2[tid] = 0.0f;

  // Pass 1: compute max of gradient norm
  for(int i=tid; i < nb; i+= BLOCKSIZE2)
  {
    const float tmp  = sqrt(dxy[i].z * dxy[i].z + dxy[i].w * dxy[i].w);
    if(tmp > v1[tid]) v1[tid] = tmp;
  }
  __syncthreads();

  // Reduce average gradient
  for (unsigned int s=BLOCKSIZE2/2; s>0; s>>=1)
  {
    if (tid < s) v1[tid] += v1[tid + s];
    __syncthreads();
  }
  const float thres = threshold * v1[0] / nb;
  v1[tid] = 0.0f;

  // Pass 2: compute final displacements and average
  for(int i=tid; i < nb; i+= BLOCKSIZE2)
  {
    float dx = dxy[i].x / fmaxf(dxy[i].z, 1e-30f) * mult;
    float dy = dxy[i].y / fmaxf(dxy[i].w, 1e-30f) * mult;

    //printf("tid #%3d [i=%3d): (%12.5f %12.5f %12.5f %12.5f) (dx, dy)=(%12.8f, %12.8f) (%d, %d)\n", tid, i,
    //       dxy[i].x, dxy[i].y, dxy[i].z, dxy[i].w, dx, dy, dxy[i].z < thres, dxy[i].w < thres);

    // Filter based on gradient value (no shift on homogeneous areas)
    // TODO: detect small gradients *along the computed shift direction*
    if(dxy[i].z < thres) dx = 0;
    if(dxy[i].w < thres) dy = 0;

    const float dr  = sqrt(dx * dx + dy * dy);

    // Don't shift if value is too small
    if(dr < min_shift)
    {
      dx = 0;
      dy = 0;
    }
    // Limit shift amplitude
    if(dr > max_shift)
    {
      dx *= max_shift / dr;
      dy *= max_shift / dr;
    }
    v1[tid] += dx;
    v2[tid] += dy;
  }
  __syncthreads();

  // Reduce average displacement
  for (unsigned int s=BLOCKSIZE2/2; s>0; s>>=1)
  {
    if (tid < s) {v1[tid] += v1[tid + s];v2[tid] += v2[tid + s];}
    __syncthreads();
  }
  const float dx0 =  v1[0] / nb;
  const float dy0 =  v2[0] / nb;

  // Pass 3: compute and store final displacements, after average subtraction
  for(int i=tid; i < nb; i+= BLOCKSIZE2)
  {
    float dx = dxy[i].x / fmaxf(dxy[i].z, 1e-30f) * mult;
    float dy = dxy[i].y / fmaxf(dxy[i].w, 1e-30f) * mult;

    // Filter based on gradient value (no shift on homogeneous areas)
    // TODO: detect small gradients *along the computed shift direction*
    if(dxy[i].z < thres) dx = 0;
    if(dxy[i].w < thres) dy = 0;

    const float dr  = sqrt(dx * dx + dy * dy);

    // Don't shift if value is too small
    if(dr < min_shift)
    {
      dx = 0;
      dy = 0;
    }
    // Limit shift amplitude
    if(dr > max_shift)
    {
      dx *= max_shift / dr;
      dy *= max_shift / dr;
    }

    cx[i] += dx - dx0;
    cy[i] += dy - dy0;
  }
}
