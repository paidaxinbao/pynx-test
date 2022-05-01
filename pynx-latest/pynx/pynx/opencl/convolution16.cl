/// get pixel coordinate in fft-shifted (origin at 0) array
int ixyz(const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
{
    int iix = (ix + nx/2)%nx;
    int iiy = (iy + ny/2)%ny;
    int iiz = (iz + nz/2)%nz;
    return (iiz * ny + iiy) * nx + iix;
}

/** Perform a 1D gaussian convolution on 16 points intervals per thread along the x-axis.
* The absolute value of the input complex array is first taken, and the result is stored in a new float array.
* The 1D kernel size is 15 (2*7+1). Convolution is done by warping across boundaries.
*/
__kernel void abs_gauss_convol_16x( __global float2 *d, __global float* d_abs_conv, const float sigma, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int tid = get_local_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    __local float v[2 * BLOCKSIZE];
    __local float g[BLOCKSIZE];
    g[tid] = exp(-0.5f*(float)(tid-7)*(float)(tid-7)/(sigma*sigma));
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid==15)
    {
        g[15] = g[0];
        for(unsigned int i=0; i<15;i++)
        {
            g[15] += g[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid<15) g[tid] /= g[15];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int j=0 ; j <nx; j += 16)
    {
        const int ix = tid + j;
        if(j==0) v[tid] = length(d[ixyz(ix-7,iy,iz,nx,ny,nz)]);
        else v[tid] = v[tid+16];
        v[tid+16] = length(d[ixyz(ix-7+16,iy,iz,nx,ny,nz)]);
        barrier(CLK_LOCAL_MEM_FENCE);
        float v2 = v[tid]*g[0];
        for(unsigned int i=1;i<15;i++)
        {
           v2 += v[tid+i] * g[i];
        }
        d_abs_conv[ixyz(ix,iy,iz,nx,ny,nz)] = v2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/** Perform a 1D gaussian convolution on 16 points intervals per thread along the x-axis.
* The 1D kernel size is 15 (2*7+1). Convolution is done by warping across boundaries.
*/
__kernel void gauss_convol_16x(__global float* d, const float sigma, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int tid = get_local_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    __local float v[2 * BLOCKSIZE];
    __local float g[BLOCKSIZE];
    g[tid] = exp(-0.5f*(float)(tid-7)*(float)(tid-7)/(sigma*sigma));
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid==15)
    {
        g[15] = g[0];
        for(unsigned int i=0; i<15;i++)
        {
            g[15] += g[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid<15) g[tid] /= g[15];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int j=0 ; j <nx; j += 16)
    {
        const int ix = tid + j;
        if(j==0) v[tid] = d[ixyz(ix-7,iy,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix-7+16,iy,iz,nx,ny,nz)];
        barrier(CLK_LOCAL_MEM_FENCE);
        float v2 = v[tid]*g[0];
        for(unsigned int i=1;i<15;i++)
        {
           v2 += v[tid+i] * g[i];
        }
        d[ixyz(ix,iy,iz,nx,ny,nz)] = v2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


/** Perform an in-place 1D gaussian convolution on 16 points intervals per thread along the y-axis.
* The 1D kernel size is 15 (2*7+1). Convolution is done by warping across boundaries.
*/
__kernel void gauss_convol_16y( __global float* d, const float sigma, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int ix = get_global_id(0);
    const int tid = get_local_id(1);
    const int iz = get_global_id(2);
    __local float v[2 * BLOCKSIZE];
    __local float g[BLOCKSIZE];
    g[tid] = exp(-0.5f*(float)(tid-7)*(float)(tid-7)/(sigma*sigma));
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid==15)
    {
        g[15] = g[0];
        for(unsigned int i=0; i<15;i++)
        {
            g[15] += g[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid<15) g[tid] /= g[15];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int j=0 ; j <ny; j += 16)
    {
        const int iy = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy-7,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy-7+16,iz,nx,ny,nz)];
        barrier(CLK_LOCAL_MEM_FENCE);
        float v2 = v[tid]*g[0];
        for(unsigned int i=1;i<15;i++)
        {
           v2 += v[tid+i] * g[i];
        }
        d[ixyz(ix,iy,iz,nx,ny,nz)] = v2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/** Perform an in-place 1D gaussian convolution on 16 points intervals per thread along the z-axis.
* The 1D kernel size is 15 (2*7+1). Convolution is done by warping across boundaries.
*/
__kernel void gauss_convol_16z( __global float* d, const float sigma, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int tid = get_local_id(2);
    __local float v[2 * BLOCKSIZE];
    __local float g[BLOCKSIZE];
    g[tid] = exp(-0.5f*(float)(tid-7)*(float)(tid-7)/(sigma*sigma));
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid==15)
    {
        g[15] = g[0];
        for(unsigned int i=0; i<15;i++)
        {
            g[15] += g[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid<15) g[tid] /= g[15];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int j=0 ; j <nz; j += 16)
    {
        const int iz = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy,iz-7,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy,iz-7+16,nx,ny,nz)];
        barrier(CLK_LOCAL_MEM_FENCE);
        float v2 = v[tid]*g[0];
        for(unsigned int i=1;i<15;i++)
        {
           v2 += v[tid+i] * g[i];
        }
        d[ixyz(ix,iy,iz,nx,ny,nz)] = v2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
