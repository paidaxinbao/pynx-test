/// get pixel coordinate in fft-shifted (origin at 0) array, wrapping around all dimensions
int ixyz(const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
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
__kernel __attribute__((reqd_work_group_size(16, 1, 1)))
void binary_window_convol_16x( __global char *d, const int w, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int tid = get_local_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    __local char v[2 * BLOCKSIZE];

    for(int j=0 ; j <nx; j += 16)
    {
        const int ix = tid + j;
        if(j==0) v[tid] = d[ixyz(ix-7,iy,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix-7+16,iy,iz,nx,ny,nz)];
        barrier(CLK_LOCAL_MEM_FENCE);

        if(w>0)
        {
            int v2 = 0;
            for(int i=7-w;i<=7+w;i++)
            {
               v2 += v[tid+i];
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
        }
        else
        {
            int v2 = 0;
            for(int i=7+w;i<=7-w;i++)
            {
               v2 += v[tid+i]==0;
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/** Perform a 1D binary convolution on 16 points intervals per thread along the y-axis.
* The 1D window size is (2*w+1). Convolution is done by warping across boundaries.
*/
__kernel __attribute__((reqd_work_group_size(1, 16, 1)))
void binary_window_convol_16y( __global char* d, const int w, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int ix = get_global_id(0);
    const int tid = get_local_id(1);
    const int iz = get_global_id(2);
    __local char v[2 * BLOCKSIZE];

    for(int j=0 ; j <ny; j += 16)
    {
        const int iy = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy-7,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy-7+16,iz,nx,ny,nz)];
        barrier(CLK_LOCAL_MEM_FENCE);
        //if(ix==256 && j ==256 && tid==0)
        //   for(int i =0;i<2*BLOCKSIZE;i++) printf("CL (%3d,%3d, %3d) %3d %5d\n", ix, iy, iz, i, v[tid]);

        if(w>0)
        {
            int v2 = 0;
            for(int i=7-w;i<=7+w;i++)
            {
               //if(ix==256 && j ==256) printf("CL (%3d,%3d, %3d) %3d %4d %5d %1d %d\n", ix, iy, iz, tid, v2, ixyz(ix,iy,iz,nx,ny,nz), d[ixyz(ix,iy,iz,nx,ny,nz)], v[tid+i]);
               v2 += v[tid+i];
            }
            //if(ix==256 && j ==256) printf("CL (%3d,%3d, %3d) %3d %4d %5d %1d\n", ix, iy, iz, tid, v2, ixyz(ix,iy,iz,nx,ny,nz), d[ixyz(ix,iy,iz,nx,ny,nz)]);
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
        }
        else
        {
            int v2 = 0;
            for(int i=7+w;i<=7-w;i++)
            {
               v2 += v[tid+i]==0;
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/** Perform a 1D binary convolution on 16 points intervals per thread along the z-axis.
* The 1D window size is (2*w+1). Convolution is done by warping across boundaries.
*/
__kernel __attribute__((reqd_work_group_size(1, 1, 16)))
void binary_window_convol_16z( __global char* d, const int w, const int nx, const int ny, const int nz)
{
    #define BLOCKSIZE 16
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int tid = get_local_id(2);
    __local char v[2 * BLOCKSIZE];

    for(int j=0 ; j <nz; j += 16)
    {
        const int iz = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy,iz-7,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy,iz-7+16,nx,ny,nz)];
        barrier(CLK_LOCAL_MEM_FENCE);

        if(w>0)
        {
            int v2 = 0;
            for(int i=7-w;i<=7+w;i++)
            {
               v2 += v[tid+i];
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
        }
        else
        {
            int v2 = 0;
            for(int i=7+w;i<=7-w;i++)
            {
               v2 += v[tid+i]==0;
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = 0;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
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
__kernel __attribute__((reqd_work_group_size(16, 1, 1)))
void binary_window_convol_16x_mask(__global char *d, const int w, const int nx, const int ny, const int nz,
                                   const char mask_in, const char mask_out)
{
    #define BLOCKSIZE 16
    const int tid = get_local_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    __local char v[2 * BLOCKSIZE];

    const char mask_out_c = (char)127 ^ mask_out;

    for(int j=0 ; j <nx; j += 16)
    {
        const int ix = tid + j;
        if(j==0) v[tid] = d[ixyz(ix-7,iy,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix-7+16,iy,iz,nx,ny,nz)];
        barrier(CLK_LOCAL_MEM_FENCE);

        if(w>0)
        {
            int v2 = 0;
            for(int i=7-w;i<=7+w;i++)
            {
               //if(ix==256 && j ==256) printf("CL (%3d,%3d, %3d) %3d %4d %5d %1d %d %d\n", ix, iy, iz, tid, v2, ixyz(ix,iy,iz,nx,ny,nz), d[ixyz(ix,iy,iz,nx,ny,nz)], v[tid+i], mask_out_c);
               v2 += (v[tid+i] & mask_in);
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out);
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c);
        }
        else
        {
            int v2 = 0;
            for(int i=7+w;i<=7-w;i++)
            {
               v2 += ((v[tid+i] & mask_in)==0);
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c);
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/** Perform a 1D binary convolution on 16 points intervals per thread along the y-axis.
* The 1D window size is (2*w+1). Convolution is done by warping across boundaries.
*/
__kernel __attribute__((reqd_work_group_size(1, 16, 1)))
void binary_window_convol_16y_mask(__global char* d, const int w, const int nx, const int ny, const int nz,
                                   const char mask_in, const char mask_out)
{
    #define BLOCKSIZE 16
    const int ix = get_global_id(0);
    const int tid = get_local_id(1);
    const int iz = get_global_id(2);
    __local char v[2 * BLOCKSIZE];

    const char mask_out_c = (char)127 ^ mask_out;

    for(int j=0 ; j <ny; j += 16)
    {
        const int iy = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy-7,iz,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy-7+16,iz,nx,ny,nz)];
        barrier(CLK_LOCAL_MEM_FENCE);

        if(w>0)
        {
            int v2 = 0;
            for(int i=7-w;i<=7+w;i++)
            {
               v2 += v[tid+i] & mask_in;
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c);
        }
        else
        {
            int v2 = 0;
            for(int i=7+w;i<=7-w;i++)
            {
               v2 += (v[tid+i] & mask_in)==0;
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/** Perform a 1D binary convolution on 16 points intervals per thread along the z-axis.
* The 1D window size is (2*w+1). Convolution is done by warping across boundaries.
*/
__kernel __attribute__((reqd_work_group_size(1, 1, 16)))
void binary_window_convol_16z_mask(__global char* d, const int w, const int nx, const int ny, const int nz,
                                   const char mask_in, const char mask_out)
{
    #define BLOCKSIZE 16
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int tid = get_local_id(2);
    __local char v[2 * BLOCKSIZE];

    const char mask_out_c = (char)127 ^ mask_out;

    for(int j=0 ; j <nz; j += 16)
    {
        const int iz = tid + j;
        if(j==0) v[tid] = d[ixyz(ix,iy,iz-7,nx,ny,nz)];
        else v[tid] = v[tid+16];
        v[tid+16] = d[ixyz(ix,iy,iz-7+16,nx,ny,nz)];
        barrier(CLK_LOCAL_MEM_FENCE);

        if(w>0)
        {
            int v2 = 0;
            for(int i=7-w;i<=7+w;i++)
            {
               v2 += v[tid+i] & mask_in;
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = (d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c);
        }
        else
        {
            int v2 = 0;
            for(int i=7+w;i<=7-w;i++)
            {
               v2 += (v[tid+i] & mask_in)==0;
            }
            if(v2>0) d[ixyz(ix,iy,iz,nx,ny,nz)] = d[ixyz(ix,iy,iz,nx,ny,nz)] & mask_out_c;
            else d[ixyz(ix,iy,iz,nx,ny,nz)] = d[ixyz(ix,iy,iz,nx,ny,nz)] | mask_out;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
