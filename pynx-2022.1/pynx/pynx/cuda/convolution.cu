// Convolution kernels with a variable size kernel. Artefacts will occur with too large gaussian widths compared to
// the kernel size (FWHM = 2.35*sigma, FH@10% = 4.3*sigma)
// The following parameters must be defined externally:
// #define BLOCKSIZE 16   32   64
// #define HALFBLOCK  7   15   31

/// get pixel coordinate in array
__device__ int ixyz(const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
{
  int iix = (ix + nx)%nx;
  int iiy = (iy + ny)%ny;
  int iiz = (iz + nz)%nz;
  return (iiz * ny + iiy) * nx + iix;
}


/** Perform a 1D gaussian convolution on BLOCKSIZE points intervals per thread along the x-axis.
* This applies to a float array, both real and imaginary parts are independently convolved.
* The 1D kernel size is 2*HALFBLOCK+1. Convolution is done by warping across boundaries.
*/
__global__  void gauss_convol_x(float *d, const float sigma, const int nx, const int ny, const int nz)
{
    const int tid = threadIdx.x;
    const int iy = blockIdx.y;
    const int iz = blockIdx.z;
    __shared__ float v[2 * BLOCKSIZE];  // generates a warning: __shared__ memory variable with non-empty constructor or destructor
    __shared__ float g[BLOCKSIZE];
    __syncthreads();
    g[tid] = expf(-0.5f*float(tid-HALFBLOCK)*float(tid-HALFBLOCK)/(sigma*sigma));
    __syncthreads();
    if(tid==(BLOCKSIZE-1))
    {
        g[BLOCKSIZE-1] = g[0];
        for(int i=1; i<(BLOCKSIZE-1);i++)
        {
            g[BLOCKSIZE-1] += g[i];
        }
    }
    __syncthreads();
    if(tid<(BLOCKSIZE-1)) g[tid] /= g[BLOCKSIZE-1];

    // Keep a copy of first block for wrapped-around convolution
    // (only the first half-block is used)
    __shared__ float v0[BLOCKSIZE];
    v0[tid] = d[ixyz(tid,iy,iz,nx,ny,nz)];

    __syncthreads();

    for(int j=0 ; j <nx-HALFBLOCK; j += BLOCKSIZE)
    {
        const int ix = tid + j;

        if(j==0) v[tid] = d[ixyz(ix-HALFBLOCK,iy,iz,nx,ny,nz)];
        else v[tid] = v[tid+BLOCKSIZE];

        if(ix-HALFBLOCK+BLOCKSIZE >= nx) v[tid+BLOCKSIZE] = v0[(ix-HALFBLOCK+BLOCKSIZE) % nx];
        else v[tid+BLOCKSIZE] = d[ixyz(ix-HALFBLOCK+BLOCKSIZE,iy,iz,nx,ny,nz)];

        __syncthreads();
        float v2 = v[tid]*g[0];
        for(unsigned int i=1;i<(BLOCKSIZE-1);i++)
        {
           v2 += v[tid+i] * g[i];
        }
        d[ixyz(ix,iy,iz,nx,ny,nz)] = v2;
        __syncthreads();
    }
}

/** Perform a 1D gaussian convolution on BLOCKSIZE points intervals per thread along the y-axis.
* This applies to a float array, both real and imaginary parts are independently convolved.
* The 1D kernel size is 2*HALFBLOCK+1. Convolution is done by warping across boundaries.
*/
__global__  void gauss_convol_y(float* d, const float sigma, const int nx, const int ny, const int nz)
{
    const int ix = blockIdx.x;
    const int tid = threadIdx.y;
    const int iz = blockIdx.z;
    __shared__ float v[2 * BLOCKSIZE];  // generates a warning: __shared__ memory variable with non-empty constructor or destructor
    __shared__ float g[BLOCKSIZE];
    __syncthreads();
    g[tid] = expf(-0.5f*float(tid-HALFBLOCK)*float(tid-HALFBLOCK)/(sigma*sigma));
    __syncthreads();
    if(tid==(BLOCKSIZE-1))
    {
        g[BLOCKSIZE-1] = g[0];
        for(int i=1; i<(BLOCKSIZE-1);i++)
        {
            g[BLOCKSIZE-1] += g[i];
        }
    }
    __syncthreads();
    if(tid<BLOCKSIZE-1) g[tid] /= g[BLOCKSIZE-1];

    // Keep a copy of first block for wrapped-around convolution
    // (only the first half-block is used)
    __shared__ float v0[BLOCKSIZE];
    v0[tid] = d[ixyz(ix,tid,iz,nx,ny,nz)];

    __syncthreads();

    for(int j=0 ; j <ny-HALFBLOCK; j += BLOCKSIZE)
    {
        const int iy = tid + j;

        if(j==0) v[tid] = d[ixyz(ix,iy-HALFBLOCK,iz,nx,ny,nz)];
        else v[tid] = v[tid+BLOCKSIZE];

        if(iy-HALFBLOCK+BLOCKSIZE >= ny) v[tid+BLOCKSIZE] = v0[(iy-HALFBLOCK+BLOCKSIZE) % ny];
        else v[tid+BLOCKSIZE] = d[ixyz(ix,iy-HALFBLOCK+BLOCKSIZE,iz,nx,ny,nz)];

        __syncthreads();
        float v2 = v[tid]*g[0];
        for(unsigned int i=1;i<(BLOCKSIZE-1);i++)
        {
           v2 += v[tid+i] * g[i];
        }
        d[ixyz(ix,iy,iz,nx,ny,nz)] = v2;
        __syncthreads();
    }
}

/** Perform a 1D gaussian convolution on BLOCKSIZE points intervals per thread along the z-axis.
* This applies to a float array, both real and imaginary parts are independently convolved.
* The 1D kernel size is 2*HALFBLOCK+1. Convolution is done by warping across boundaries.
*/
__global__ void gauss_convol_z(float* d, const float sigma, const int nx, const int ny, const int nz)
{
    const int ix = blockIdx.x;
    const int iy = blockIdx.y;
    const int tid = threadIdx.z;
    __shared__ float v[2 * BLOCKSIZE];  // generates a warning: __shared__ memory variable with non-empty constructor or destructor
    __shared__ float g[BLOCKSIZE];
    __syncthreads();
    g[tid] = expf(-0.5f*float(tid-HALFBLOCK)*float(tid-HALFBLOCK)/(sigma*sigma));
    __syncthreads();
    if(tid==(BLOCKSIZE-1))
    {
        g[BLOCKSIZE-1] = g[0];
        for(int i=1; i<(BLOCKSIZE-1);i++)
        {
            g[BLOCKSIZE-1] += g[i];
        }
    }
    __syncthreads();
    if(tid<(BLOCKSIZE-1)) g[tid] /= g[BLOCKSIZE-1];

    // Keep a copy of first block for wrapped-around convolution
    // (only the first half-block is used)
    __shared__ float v0[BLOCKSIZE];
    v0[tid] = d[ixyz(ix,iy,tid,nx,ny,nz)];

    __syncthreads();

    for(int j=0 ; j <nz-HALFBLOCK; j += BLOCKSIZE)
    {
        const int iz = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy,iz-HALFBLOCK,nx,ny,nz)];
        else v[tid] = v[tid+BLOCKSIZE];
        v[tid+BLOCKSIZE] = d[ixyz(ix,iy,iz-HALFBLOCK+BLOCKSIZE,nx,ny,nz)];
        __syncthreads();
        float v2 = v[tid]*g[0];
        for(int i=1;i<(BLOCKSIZE-1);i++)
        {
           v2 += v[tid+i] * g[i];
        }
        d[ixyz(ix,iy,iz,nx,ny,nz)] = v2;
        __syncthreads();
    }
}
