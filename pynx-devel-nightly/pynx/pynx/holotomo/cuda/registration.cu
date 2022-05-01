#include <pycuda-complex.hpp>
typedef pycuda::complex<float> complexf;
#define TWOPI 6.2831853071795862f

/** Compute object views normalised by probe for each iz, and only
* for the central part of the arrays. Only the first mode is taken into account.
*
* This should be called for the whole reg array which should have a shape (nproj*nz, dn, dn)
* where psi has a shape (nproj, nz, nobj, nprobe, ny, nx)
*/
__device__ void psi2reg1(const int i, complexf *reg, complexf *psi, complexf *probe,
                         float *dx, float *dy, signed char *sample_flag,
                         const int nb_probe, const int nb_obj,
                         const int nx, const int ny, const int nz, const int dn)
{
  // Coordinates in the reg array
  const int ix = i %% dn;
  const int iy = (i %% (dn * dn)) / dn;
  const int iz = (i %% (dn * dn * nz)) / (dn * dn);
  const int iproj = i / (dn * dn * nz);

  if(sample_flag[iproj]==0)
  {
    reg[i] = 0;
    return;
  }

  // Coordinates in the object & probe array
  const int ix0 = ix + (nx - dn) / 2;
  const int iy0 = iy + (ny - dn) / 2;
  // const int i0 = ix0 + nx * (iy0 + ny * nb_probe * nb_obj * (iz + nz * iproj));

  const complexf ps = bilinear(psi, ix0-dx[iz+nz*iproj], iy0-dy[iz+nz*iproj],
                               nb_obj * nb_probe * (iz + nz * iproj), nx, ny, false, true);

  const complexf pr = bilinear(probe, ix0-dx[iz+nz*iproj], iy0-dy[iz+nz*iproj], iz, nx, ny, false, false);

  reg[i] = complexf(pr.real()*ps.real() + pr.imag()*ps.imag() , pr.real()*ps.imag() - pr.imag()*ps.real()) / dot(pr,pr);
}

/** Compute the cross-correlation FT array reg[iz_ref]*reg[iz].conj()
*
* This should be called for the first frame of reg with size (dn, dn) and will loop
* over all the projections and distances to fill the reg array with nproj*(nz-1) frames
*/
__device__ void reg_mult_conj(const int i, complexf *reg, signed char *sample_flag, const int iz0,
                              const int nz, const int nb_proj, const int dn)
{
  for(int iproj=0; iproj<nb_proj; iproj++)
  {
    if(sample_flag[iproj])
    {
      const complexf r0 = reg[i + dn * dn * (iz0 + nz * iproj)];
      for(int iz=0; iz<(nz-1); iz++)
      {
        complexf rz;
        if(iz<iz0) rz = reg[i + dn * dn * (iz + nz * iproj)];
        else rz = reg[i + dn * dn * (iz + 1 + nz * iproj)];
        reg[i + dn * dn * (iz + (nz-1) * iproj)] = complexf(r0.real() * rz.real() + r0.imag() * rz.imag(),
                                                          r0.imag() * rz.real() - r0.real() * rz.imag());
      }
    }
  }
}

/** Find the maximum CC coordinate and update the shift coordinates
*
* Using shared memory, one thread block find the CC map maximum and
* updates the shift value.
* The input data array is a stack of nb_proj*(nz-1) images, for which the CC map
* maxima and deduced shifts are computed in different blocks.
*
* \param d: the (nb_proj*(nz-1), dn, dn) complex array where the maximum is searched
* \param dx,dy: the (nb_proj, nz) arrays which will be updated with the additional pixel shifts
*   (the values will need to be added to the previous shift values)
* \param iz0: the reference distance
* \param dn: the lateral size of each frame
* \param nz, nb_proj: number of distances and projections
*/
__global__ void cc_pixel(complexf *reg, float *dx, float *dy, const int iz0,
                         const int nz, const int nb_proj, const int dn)
{
  #define BLOCKSIZE %(blocksize)d
  const int tid = threadIdx.x;
  const int iz = blockIdx.x ;
  const int iproj = blockIdx.y;
  __shared__ int idx[BLOCKSIZE];
  __shared__ float vcc[BLOCKSIZE];
  float cur_max = 0;
  int cur_idx = 0;

  const int dn2 = dn * dn;
  for(int i=tid; i<dn2; i+=BLOCKSIZE)
  {
    const float v= dot(reg[i + dn2 * (iz + (nz-1) * iproj)], reg[i + dn2 * (iz + (nz-1) * iproj)]);
    if(v>cur_max)
    {
      cur_max = v;
      cur_idx = i;
    }
  }
  // Reduce the maximum
  idx[tid] = cur_idx;
  vcc[tid] = cur_max;
  __syncthreads();
  for (unsigned int s=BLOCKSIZE/2; s>0; s>>=1)
  {
    if (tid < s)
    {
      if(vcc[tid] < vcc[tid+s])
      {
        vcc[tid] = vcc[tid+s];
        idx[tid] = idx[tid+s];
      }
    }
    __syncthreads();
  }
  // Store pixel coordinates of maximum
  if(tid==0)
  {
    const int dx0 = idx[0] %% dn;
    const int dy0 = idx[0] / dn;
    const int iz1 = iz < iz0 ? iz : iz+1;
    dx[iz1 + nz * iproj] = dx0 - (dx0 >= dn / 2) * dn;
    dy[iz1 + nz * iproj] = dy0 - (dy0 >= dn / 2) * dn;
  }
}

/** Zoomed discrete inverse Fourier transform for registration (cross-correlation).
*
* Version using shared memory, one thread block computes the entire CC map and
* updates the shift in the found maximum.
* The input data array includes a stack of nb_proj*(nz-1) images, for which the CC map
* maxima and deduced shifts are computed in different blocks.
*
* \param d: the (nb_proj*(nz-1), dn, dn) complex array where the maximum is searched
* \param dx,dy: the (nb_proj, nz) arrays with the original shifts to be updated
* \param dx1,dy1: the (nb_proj, nz) arrays with the computed additional pixel shifts
* \param iz0: the reference distance
* \param dn: the lateral size of each frame
* \param nz, nb_proj: number of distances and projections
* \param dxyu: step in pixels in the zoomed area
* \param dnu: number of points for the upsampled area, of extent (dnu * dxyu, dnu * dxyu)
*   around x0[iproj, iz], y0[iproj, iz]. dnu must be even.
*/
__global__ void cc_zoom(complexf *d, float *dx, float *dy, float *dx1, float *dy1, const int iz0,
                        const int nz, const int nb_proj, const int dn,
                        const float dxyu, const int dnu, complexf *ccmap)
{
  #define BLOCKSIZE %(blocksize)d
  const int tid = threadIdx.x;
  const int iz = blockIdx.x;
  const int iz1 = iz < iz0 ? iz: iz+1;
  const int iproj = blockIdx.y;
  __shared__ float2 vd[BLOCKSIZE];

  const int dn2 = dn * dn;
  const int dnu2 = dnu * dnu;
  float s,c;

  // Corner coordinates of the upsampled CC map
  const float x0z = dx1[iz1 + nz * iproj] - dxyu * dnu / 2;
  const float y0z = dy1[iz1 + nz * iproj] - dxyu * dnu / 2;

  // TODO: outer loop over CC map coordinates, store accumulated cc in global array,
  //   and outer loop over CC map pixels to only read the map once

  // Loop over pixels of d = FT(ref_img)*FT(img).conj()
  for(int j=0; j<(dn2-BLOCKSIZE); j+=BLOCKSIZE)
  {
    if((j+tid) < dn2) vd[tid] = make_float2(d[j + tid + dn2 * (iz + nz * iproj)].real(),
                                            d[j + tid + dn2 * (iz + nz * iproj)].imag());
    else vd[tid] = make_float2(0.0f, 0.0f);
    __syncthreads();

    // Loop over CC map pixels
    for(int i=tid; i<dnu2; i+=BLOCKSIZE)
    {
      // CC map coordinates
      const float x = x0z + (i %% dnu) * dxyu;
      const float y = y0z + (i / dnu) * dxyu;
      complexf cc = 0;
      for(int k=0; k<BLOCKSIZE; k++)
      {
        const int ix = (j+k) %% dn;
        const int iy = (j+k) / dn;
        const float u = (float)ix / (float)dn - (ix>=(dn/2));
        const float v = (float)iy / (float)dn - (iy>=(dn/2));
        __sincosf(TWOPI * (u*x + v*y) , &s,&c);
        cc += complexf(vd[k].x * c - vd[k].y * s, vd[k].x * s + vd[k].y * c);
      }
      if(j==0) ccmap[i + dnu2 * (iz + iproj * (nz-1))] = cc;
      else ccmap[i + dnu2 * (iz + iproj * (nz-1))] += cc;
    }
    __syncthreads();
}

  // CC maximum for each thread
  float cur_max = 0;
  int cur_idx = 0;
  for(int i=tid; i<dnu2; i+=BLOCKSIZE)
  {
    const float cc2 = dot(ccmap[i + dnu2 * (iz + iproj * (nz-1))], ccmap[i + dnu2 * (iz + iproj * (nz-1))]);
    if(cc2 > cur_max)
    {
      cur_max = cc2;
      cur_idx = i;
    }
  }

  // Reduce the maximum
  __shared__ int idx[BLOCKSIZE];
  __shared__ float vcc[BLOCKSIZE];
  idx[tid] = cur_idx;
  vcc[tid] = cur_max;
  __syncthreads();
  for (unsigned int s=BLOCKSIZE/2; s>0; s>>=1)
  {
    if (tid < s)
    {
      if(vcc[tid] < vcc[tid+s])
      {
        vcc[tid] = vcc[tid+s];
        idx[tid] = idx[tid+s];
      }
    }
    __syncthreads();
  }

  if(tid==0)
  {
    const int dx0 = idx[0] %% dnu;
    const int dy0 = idx[0] / dnu;

    dx1[iz1 + nz * iproj] = x0z + dx0 * dxyu; // For development only
    dy1[iz1 + nz * iproj] = y0z + dy0 * dxyu; // For development only

    dx[iz1 + nz * iproj] += x0z + dx0 * dxyu;
    dy[iz1 + nz * iproj] += y0z + dy0 * dxyu;
  }
}
