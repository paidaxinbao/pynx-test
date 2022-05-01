#include <pycuda-complex.hpp>
typedef pycuda::complex<float> complexf;

__device__ int ixyz(const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
{
    int iix = (ix + nx) % nx;
    int iiy = (iy + ny) % ny;
    int iiz = (iz + nz) % nz;
    return (iiz * ny + iiy) * nx + iix;
}

/** Perform a 1D gaussian convolution on 16 points intervals per thread along the x-axis.
* The absolute value of the input complex array is first taken, and the result is stored in a new float array.
* The 1D kernel size is 15 (2*7+1). Convolution is done by warping across boundaries.
*/
__global__ void gauss_convolc2f_16x(complexf *d, float* d_abs_conv, const float sigma, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int tid = threadIdx.x;
    const int iy = blockIdx.y;
    const int iz = blockIdx.z;
    __shared__ float v[2 * BLOCKSIZE];
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
        if(j==0) v[tid] = abs(d[ixyz(ix-7,iy,iz,nx,ny,nz)]);
        else v[tid] = v[tid+16];
        v[tid+16] = abs(d[ixyz(ix-7+16,iy,iz,nx,ny,nz)]);
        __syncthreads();
        float v2 = v[tid]*g[0];
        for(unsigned int i=1;i<15;i++)
        {
           v2 += v[tid+i] * g[i];
        }
        d_abs_conv[ixyz(ix,iy,iz,nx,ny,nz)] = v2;
        __syncthreads();
    }
}
