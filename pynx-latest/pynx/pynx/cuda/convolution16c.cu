#include <pycuda-complex.hpp>
typedef pycuda::complex<float> complexf;

__device__ int ixyz(const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
{
    int iix = (ix + nx) % nx;
    int iiy = (iy + ny) % ny;
    int iiz = (iz + nz) % nz;
    return (iiz * ny + iiy) * nx + iix;
}

/** Perform an in-place 1D gaussian convolution on 16 points intervals per thread along the x-axis,
* The 1D kernel size is 15 (2*7+1). Convolution is done by warping across boundaries.
*/
__global__ void gauss_convolc_16x(complexf *d, const float sigma, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int tid = threadIdx.x;
    const int iy = blockIdx.y;
    const int iz = blockIdx.z;
    __shared__ complexf v[2 * BLOCKSIZE];
    __shared__ float g[BLOCKSIZE];
    g[tid] = expf(-0.5f*float(tid-7)*float(tid-7)/(sigma*sigma));
    __syncthreads();
    if(tid==15)
    {
        g[15] = g[0];
        for(unsigned int i=0; i<15;i++)
        {
            g[15] += g[i];
        }
    }
    __syncthreads();
    if(tid<15) g[tid] /= g[15];
    __syncthreads();

    for(unsigned int j=0 ; j <nx; j += 16)
    {
        const int ix = tid + j;
        if(j==0) v[tid] = d[ixyz(ix-7,iy,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix-7+16,iy,iz,nx,ny,nz)];
        __syncthreads();
        complexf v2 = v[tid]*g[0];
        for(unsigned int i=1;i<15;i++)
        {
           v2 += v[tid+i] * g[i];
        }
        d[ixyz(ix,iy,iz,nx,ny,nz)] = v2;
        __syncthreads();
    }
}

/** Perform an in-place 1D gaussian convolution on 16 points intervals per thread along the y-axis.
* The 1D kernel size is 15 (2*7+1). Convolution is done by warping across boundaries.
*/
__global__ void gauss_convolc_16y(complexf* d, const float sigma, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int ix = blockIdx.x;
    const int tid = threadIdx.y;
    const int iz = blockIdx.z;
    __shared__ complexf v[2 * BLOCKSIZE];
    __shared__ float g[BLOCKSIZE];
    g[tid] = expf(-0.5f*float(tid-7)*float(tid-7)/(sigma*sigma));
    __syncthreads();
    if(tid==15)
    {
        g[15] = g[0];
        for(unsigned int i=0; i<15;i++)
        {
            g[15] += g[i];
        }
    }
    __syncthreads();
    if(tid<15) g[tid] /= g[15];
    __syncthreads();

    for(unsigned int j=0 ; j <ny; j += 16)
    {
        const int iy = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy-7,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy-7+16,iz,nx,ny,nz)];
        __syncthreads();
        complexf v2 = v[tid]*g[0];
        for(unsigned int i=1;i<15;i++)
        {
           v2 += v[tid+i] * g[i];
        }
        d[ixyz(ix,iy,iz,nx,ny,nz)] = v2;
        __syncthreads();
    }
}

/** Perform an in-place 1D gaussian convolution on 16 points intervals per thread along the z-axis.
* The 1D kernel size is 15 (2*7+1). Convolution is done by warping across boundaries.
*/
__global__ void gauss_convolc_16z(complexf* d, const float sigma, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int ix = blockIdx.x;
    const int iy = blockIdx.y;
    const int tid = threadIdx.z;
    __shared__ complexf v[2 * BLOCKSIZE];
    __shared__ float g[BLOCKSIZE];
    g[tid] = expf(-0.5f*float(tid-7)*float(tid-7)/(sigma*sigma));
    __syncthreads();
    if(tid==15)
    {
        g[15] = g[0];
        for(unsigned int i=0; i<15;i++)
        {
            g[15] += g[i];
        }
    }
    __syncthreads();
    if(tid<15) g[tid] /= g[15];
    __syncthreads();

    for(unsigned int j=0 ; j <nz; j += 16)
    {
        const int iz = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy,iz-7,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy,iz-7+16,nx,ny,nz)];
        __syncthreads();
        complexf v2 = v[tid]*g[0];
        for(unsigned int i=1;i<15;i++)
        {
           v2 += v[tid+i] * g[i];
        }
        d[ixyz(ix,iy,iz,nx,ny,nz)] = v2;
        __syncthreads();
    }
}
