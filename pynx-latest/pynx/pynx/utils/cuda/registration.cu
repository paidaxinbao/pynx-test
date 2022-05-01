#include <pycuda-complex.hpp>
typedef pycuda::complex<float> complexf;
#define TWOPI 6.2831853071795862f

/** Zoomed discrete inverse Fourier transform for registration (cross-correlation).
*
* Version using shared memory, computing a single pixel of the CC map.
*
* \param dft: the CC map array
* \param d: the complex array whose back-Fourier transform is computed
* \param x0,y0,z0: the corner coordinates of the destination DFT area to be computed
* \param dx,dy,dz: the step size of the destination DFT array
* \param nx,ny: the size of the destination DFT array along the first two dimensions
* \param nxu,nyu: the size of the d array along the first two dimensions
* \param return: idx_max(i, abs(DFT[d](i))**2)
*/
__global__ void cc_zoom(float* dft, complexf *d,
                        const float x0, const float y0, const float z0,
                        const float dx, const float dy, const float dz,
                        const int nx, const int ny, const int nxu, const int nyu, const int nzu)
{
  #define BLOCKSIZE %(blocksize)d
  const int tid = threadIdx.x;
  // Not using a shared 'complexf' array, which triggers a warning:
  //  dynamic initialization is not supported for a function-scope
  //  static __shared__ variable within a __device__/__global__ function
  __shared__ float2 vd[BLOCKSIZE];

  const float x = x0 + (float)blockIdx.x*dx;
  const float y = y0 + (float)blockIdx.y*dy;
  const float z = z0 + (float)blockIdx.z*dz;
  complexf cc=0;
  float s,c;
  const int n = nxu*nyu*nzu;
  for(int i=0; i<n; i+=BLOCKSIZE)
  {
    const complexf v1 = d[i+tid];
    const int ixu = (i+tid) %% nxu;
    const int iyu = ((i+tid) %% (nxu*nyu)) / nxu;
    const int izu = (i+tid) / (nxu*nyu);
    const float u = (float)ixu / (float)nxu - (ixu>=(nxu/2));
    const float v = (float)iyu / (float)nyu - (iyu>=(nyu/2));
    const float w = (float)izu / (float)nzu - (izu>=(nzu/2));
    __sincosf(TWOPI*(u*x + v*y + w*z) , &s,&c);
    vd[tid] = make_float2(v1.real()*c - v1.imag()*s, v1.real()*s + v1.imag()*c);
    __syncthreads();
    if(tid==0) for(int j=0;j<BLOCKSIZE;j++) cc += complexf(vd[tid].x, vd[tid].y);
    __syncthreads();
  }
  if(tid==0)
  {
    // TODO: finish this in parallel threads with check on bounds
    //  Do the reduction in parallel ? (may require power-of-two blocksize)
    for(int i=n-n %% BLOCKSIZE; i<n; i+=1)
    {
      complexf v1 = d[i];
      const int ixu = i %% nxu;
      const int iyu = (i %% (nxu*nyu)) / nxu;
      const int izu = i / (nxu*nyu);
      const float u = (float)ixu / (float)nxu - (ixu>=(nxu/2));
      const float v = (float)iyu / (float)nyu - (iyu>=(nyu/2));
      const float w = (float)izu / (float)nzu - (izu>=(nzu/2));
      __sincosf(TWOPI*(u*x + v*y + w*z) , &s,&c);
      cc += complexf(v1.real()*c - v1.imag()*s, v1.real()*s + v1.imag()*c);
    }

    dft[blockIdx.x + nx * (blockIdx.y + ny * blockIdx.z)] = abs(cc);
  }
}

/** Zoomed discrete inverse Fourier transform for registration (cross-correlation).
*
* Version using shared memory, one thread block computing the entire CC map and
* writing only the coordinates of the maximum as a result.
* The input data array includes a stack of N images, for which the CC map
* maxima and deduced shifts are computed in different blocks.
*
* \param d: the complex array whose back-Fourier transform is computed,
*           with a size nz, ny, nx, where the number of images is nz.
* \param x0,y0: the corner coordinates of the CC map to be computed.
*           Upon return these are updated with the new shift coordinates.
* \param dx,dy: the step size of the CC map (same for all images)
* \param nx,ny: the size of the CC map (same for all images)
* \param nxu,nyu: the 2D size of the d array
* \param return: nothing - x0 and y0 are updated with the new shift values
*/
__global__ void cc_zoomN(complexf *d, float* x0, float* y0, const float dx, const float dy,
                         const int nx, const int ny, const int nxu, const int nyu)
{
  #define BLOCKSIZE %(blocksize)d
  const int tid = threadIdx.x;
  const int iz = blockIdx.x;
  __shared__ float2 vd[BLOCKSIZE];
  __shared__ float vcc[BLOCKSIZE];
  float cur_max = 0;
  int cur_idx = 0;

  const int nxy = nx * ny;
  const int nxyu = nxu * nyu;
  float s,c;
  // Corner coordinates of the CC map
  const float x0z = x0[iz];
  const float y0z = y0[iz];

  // Loop over CC map pixels
  for(int i=0; i<(nxy+BLOCKSIZE-1); i+=BLOCKSIZE)
  {
    // CC map coordinates
    const float x = x0z + ((i+tid) %% nx) * dx;
    const float y = y0z + ((i+tid) / nx) * dy;
    complexf cc = 0;
    // Loop over pixels of FT(ref_img)*FT(img).conj()
    for(int j=0; j<(nxyu+BLOCKSIZE-1); j+=BLOCKSIZE)
    {
      if((j+tid) < nxyu) vd[tid] = make_float2(d[j + tid + iz * nxyu].real(), d[j + tid + iz * nxyu].imag());
      else vd[tid] = make_float2(0.0f, 0.0f);
      __syncthreads();
      for(int k=0;k<BLOCKSIZE;k++)
      {
        const int ixu = (j+k) %% nxu;
        const int iyu = (j+k) / nxu;
        const float u = (float)ixu / (float)nxu - (ixu>=(nxu/2));
        const float v = (float)iyu / (float)nyu - (iyu>=(nyu/2));
        __sincosf(TWOPI * (u*x + v*y) , &s,&c);
        cc += complexf(vd[k].x * c - vd[k].y * s, vd[k].x * s + vd[k].y * c);
      }
      __syncthreads();
    }
    if((i+tid) < nxy)
    {
      vcc[tid] = abs(cc);
      //ccd[i+tid + iz * nxy] = abs(cc);  // Store ccmap for debugging
    }
    else vcc[tid] = 0.0f;
    __syncthreads();
    //  TODO: Do the reduction in parallel ? (may require power-of-two blocksize)
    if(tid==0)
      for(int j=0; j<BLOCKSIZE; j++)
        if(vcc[j] > cur_max)
        {
          cur_max = vcc[j];
          cur_idx = i + j;
        }
  }

  if(tid==0)
  {
    x0[iz] = x0z + (cur_idx %% nx) * dx;
    y0[iz] = y0z + (cur_idx / nx) * dy;
  }
}
