#include <pycuda-complex.hpp>
typedef pycuda::complex<float> complexf;

__device__ int ixyz(const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
{
    int iix = (ix + nx/2)%nx;
    int iiy = (iy + ny/2)%ny;
    int iiz = (iz + nz/2)%nz;
    return (iiz * ny + iiy) * nx + iix;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
//  In-place binary convolution kernels
///////////////////////////////////////////////////////////////////////////////////////////////////////

/** Perform a 1D binary convolution on 16 points intervals per thread along the x-axis.
* The 1D window size is (2*w+1). Convolution is done by warping across boundaries.
*/
__global__ void binary_window_convol_16x(signed char *d, const int w, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int tid = threadIdx.x;
    const int iy = blockIdx.y;
    const int iz = blockIdx.z;
    __shared__ signed char v[2 * BLOCKSIZE];

    for(unsigned int j=0 ; j <nx; j += 16)
    {
        const int ix = tid + j;
        if(j==0) v[tid] = d[ixyz(ix-7,iy,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix-7+16,iy,iz,nx,ny,nz)];
        __syncthreads();

        if(w>0)
        {
            int v2 = 0;
            for(unsigned int i=7-w;i<=7+w;i++)
            {
               v2 += v[tid+i];
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
        }
        else
        {
            int v2 = 0;
            for(unsigned int i=7+w;i<=7-w;i++)
            {
               v2 += v[tid+i]==0;
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
        }
        __syncthreads();
    }
}

/** Perform a 1D binary convolution on 16 points intervals per thread along the y-axis.
* The 1D window size is (2*w+1). Convolution is done by warping across boundaries.
*/
__global__ void binary_window_convol_16y(signed char* d, const int w, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int ix = blockIdx.x;
    const int tid = threadIdx.y;
    const int iz = blockIdx.z;
    __shared__ signed char v[2 * BLOCKSIZE];

    for(unsigned int j=0 ; j <ny; j += 16)
    {
        const int iy = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy-7,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy-7+16,iz,nx,ny,nz)];
        __syncthreads();

        if(w>0)
        {
            int v2 = 0;
            for(unsigned int i=7-w;i<=7+w;i++)
            {
               v2 += v[tid+i];
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
        }
        else
        {
            int v2 = 0;
            for(unsigned int i=7+w;i<=7-w;i++)
            {
               v2 += v[tid+i]==0;
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
        }
        __syncthreads();
    }
}

/** Perform a 1D binary convolution on 16 points intervals per thread along the z-axis.
* The 1D window size is (2*w+1). Convolution is done by warping across boundaries.
*/
__global__ void binary_window_convol_16z(signed char* d, const int w, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int ix = blockIdx.x;
    const int iy = blockIdx.y;
    const int tid = threadIdx.z;
    __shared__ signed char v[2 * BLOCKSIZE];

    for(unsigned int j=0 ; j <nz; j += 16)
    {
        const int iz = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy,iz-7,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy,iz-7+16,nx,ny,nz)];
        __syncthreads();

        if(w>0)
        {
            int v2 = 0;
            for(unsigned int i=7-w;i<=7+w;i++)
            {
               v2 += v[tid+i];
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
        }
        else
        {
            int v2 = 0;
            for(unsigned int i=7+w;i<=7-w;i++)
            {
               v2 += v[tid+i]==0;
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
        }
        __syncthreads();
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
//  In-place binary convolution kernels (with binary masking)
///////////////////////////////////////////////////////////////////////////////////////////////////////

/** Perform a 1D binary convolution on 16 points intervals per thread along the x-axis.
* The 1D window size is (2*w+1). Convolution is done by warping across boundaries.
*
* A 'normal' support value is 0 (outside support) or 1 (inside support). But binary masking
* can be used to store several supports inside a single signed 8-bit array. e.g. support==1,2,4,..64
* This is used e.g. to compute the 'border' of the support
*
* \param mask_in: the value (as a binary mask) considered to be in the support (input)
* \param mask_out: the value (as a binary mask) considered to be in the support (output)
*/
__global__ void binary_window_convol_16x_mask(signed char *d, const int w, const int nx, const int ny, const int nz,
                                              const signed char mask_in, const signed char mask_out)
{
    #define BLOCKSIZE 16
    const int tid = threadIdx.x;
    const int iy = blockIdx.y;
    const int iz = blockIdx.z;
    __shared__ signed char v[2 * BLOCKSIZE];

    const signed char mask_out_c = (signed char)127 ^ mask_out;

    for(unsigned int j=0 ; j <nx; j += 16)
    {
        const int ix = tid + j;
        if(j==0) v[tid] = d[ixyz(ix-7,iy,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix-7+16,iy,iz,nx,ny,nz)];
        __syncthreads();

        if(w>0)
        {
            int v2 = 0;
            for(unsigned int i=7-w;i<=7+w;i++)
            {
               v2 += (v[tid+i] & mask_in);
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out);
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c);
        }
        else
        {
            int v2 = 0;
            for(unsigned int i=7+w;i<=7-w;i++)
            {
               v2 += ((v[tid+i] & mask_in)==0);
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c);
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out);
        }
        __syncthreads();
    }
}

/** Perform a 1D binary convolution on 16 points intervals per thread along the y-axis.
* The 1D window size is (2*w+1). Convolution is done by warping across boundaries.
*/
__global__ void binary_window_convol_16y_mask(signed char* d, const int w, const int nx, const int ny, const int nz,
                                              const signed char mask_in, const signed char mask_out)
{
    #define BLOCKSIZE 16
    const int ix = blockIdx.x;
    const int tid = threadIdx.y;
    const int iz = blockIdx.z;
    __shared__ signed char v[2 * BLOCKSIZE];

    const signed char mask_out_c = (signed char)127 ^ mask_out;

    for(unsigned int j=0 ; j <ny; j += 16)
    {
        const int iy = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy-7,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy-7+16,iz,nx,ny,nz)];
        __syncthreads();

        if(w>0)
        {
            int v2 = 0;
            for(unsigned int i=7-w;i<=7+w;i++)
            {
               v2 += (v[tid+i] & mask_in);
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out);
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c);
        }
        else
        {
            int v2 = 0;
            for(unsigned int i=7+w;i<=7-w;i++)
            {
               v2 += ((v[tid+i] & mask_in)==0);
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c);
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out);
        }
        __syncthreads();
    }
}

/** Perform a 1D binary convolution on 16 points intervals per thread along the z-axis.
* The 1D window size is (2*w+1). Convolution is done by warping across boundaries.
*/
__global__ void binary_window_convol_16z_mask(signed char* d, const int w, const int nx, const int ny, const int nz,
                                              const signed char mask_in, const signed char mask_out)
{
    #define BLOCKSIZE 16
    const int ix = blockIdx.x;
    const int iy = blockIdx.y;
    const int tid = threadIdx.z;
    __shared__ signed char v[2 * BLOCKSIZE];

    const signed char mask_out_c = (signed char)127 ^ mask_out;

    for(unsigned int j=0 ; j <nz; j += 16)
    {
        const int iz = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy,iz-7,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy,iz-7+16,nx,ny,nz)];
        __syncthreads();

        if(w>0)
        {
            int v2 = 0;
            for(unsigned int i=7-w;i<=7+w;i++)
            {
               v2 += (v[tid+i] & mask_in);
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out);
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c);
        }
        else
        {
            int v2 = 0;
            for(unsigned int i=7+w;i<=7-w;i++)
            {
               v2 += ((v[tid+i] & mask_in)==0);
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c);
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out);
        }
        __syncthreads();
    }
}
