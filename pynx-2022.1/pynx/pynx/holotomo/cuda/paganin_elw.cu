// #include "cuda_fp16.h"

/** Put observed intensities in Psi array (only in first mode), and normalise with empty_beam image
*
* This should be called for the complete iobs array of shape: (nb_proj, nbz, ny, nx)
* Psi has a shape: (nb_proj, nbz, nb_obj, nb_probe, ny, nx)
* The padded and masked pixels are interpolated
*/
__device__ void Iobs2Psi(const int i, float *iobs, float* iobs_empty, complexf *psi,
                         float* dx, float* dy, const int nb_mode,
                         const int nx, const int ny, const int nz, const int padding)
{
  const int nxy = nx*ny;
  // Coordinates in iobs array (centered on array)
  // i[iobs] = ix + nx * (iy + ny * (iz + nz * iproj))
  // So: i / nxy = iz + iproj * nz
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;
  const int iz = (i / nxy) % nz;
  const int iproj = i / (nxy * nz);

  float obs = iobs[i];

  // masked values, including padding
  if(obs<0)
  {
    // Simple shifting interpolation. Crude & not very efficient memory-wise but should not matter
    // as this is executed maybe once per analysis
    float v = 0;
    float v0 = 0;
    float n = 0;

    // If padding is used, interpolate with pixels up to padding+16 in all directions.
    // For normal masked pixels interpolate up to 16
    int range;
    if((ix < padding) || (iy < padding) || (ix >= (nx-padding)) || (iy >= (ny-padding))) range = 2 * padding + 16;
    else range = 16;
    int step = 1;
    if(range>16) step = range / 16;
    for(int dx=-range; dx<range; dx+=step)
      for(int dy=-range; dy<range; dy+=step)
      {
        const int ix0 = (ix + dx + nx) % nx;
        const int iy0 = (iy + dy + ny) % ny;
        // Inverse distance**2 weighting
        const float w = 1 / float(dx * dx + dy * dy);
        const float d0 = iobs[ix0 + nx * (iy0 + ny * (iz + nz * iproj))];
        // if((ix0 >= padding) && (iy0 >= padding) && (ix0 < (nx-padding)) && (iy0 < (ny-padding)))
        if(d0>=0)
        {
          v += d0 * w ;
          v0 += iobs_empty[ix0 + nx * (iy0 + ny * iz)] * w;
          n += w;
        }
      }
    obs = v / v0;
    // We interpolate the empty beam images and keep the values, masked
    iobs[i] = -v / n -1;
  }
  else obs = iobs[i] / iobs_empty[ix + nx * (iy + ny * iz)];

  // Coordinates in Psi array (fft-shifted). Assumes nx ny are multiple of 2
  const int iy1 = (iy - ny/2 + ny * (iy<(ny/2)) + __float2int_rn(dy[iz + nz * iproj]) + ny) % ny;
  const int ix1 = (ix - nx/2 + nx * (ix<(nx/2)) + __float2int_rn(dx[iz + nz * iproj]) + nx) % nx;
  // Coordinate of first mode in Psi array
  const int ipsi = ix1 + nx * (iy1 + ny * nb_mode * (iz + nz * iproj));

  psi[ipsi] = complexf(obs,0.0f);
}


#define twopi 6.2831853071795862f

// The Paganin filter in Fourier space is calculated for each distance independently
// This should be called for the whole iobs array
__device__ void paganin_fourier(const int i, float *iobs, complexf *psi, float* alpha, const float px,
                                const int nb_mode, const int nx, const int ny, const int nz)
{
  // iobs shape is (stack_size, nb_z, ny, nx) (not used but sets the size of the elementwise kernel)
  // psi shape is (stack_size, nb_z, nb_obj, nb_probe, ny, nx)

  // Coordinates in psi
  const int nxy = nx * ny;
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;
  const int iz = (i / (nxy)) % nz;
  const int iproj = i / (nxy * nz);
  const int ipsi = ix + nx * (iy + ny * nb_mode * (iz + nz * iproj));

  // Assumes ny, nx are multiples of 2
  const float ky = (iy - (int)ny *(int)(iy >= ((int)ny / 2))) * twopi / (px * (float)ny) ;
  const float kx = (ix - (int)nx *(int)(ix >= ((int)nx / 2))) * twopi / (px * (float)nx) ;

  // Paganin original method
  const complexf ps = psi[ipsi];
  const float a = 1.0f + 0.5f * alpha[iz] * (kx*kx + ky*ky);
  //psi[ipsi] = psi[ipsi] / float(1.0f + 0.5f * alpha[iz] * (kx*kx + ky*ky));
  psi[ipsi] = complexf(ps.real()/a, ps.imag()/a) ;
}

/** This function should be called for the whole iobs array
* Object has 4 dimensions: projections, z, modes, y, x.
* The 5-dimensional Psi stack is calculated, with dimensions:
*   nb_proj, nz, object modes, probe modes, y, x
* Iobs array has dimensions (nb_proj, nz, ny, nx) and is only used to set the elementwise kernel size.
*
* This is used at the end of the single-distance Paganin operator
*
* On return:
* - the complex object has been replaced by the one calculated
* - in the Psi array, the first mode of each z and projection has the object's mu*thickness (real)
*
*/
__device__ void paganin_thickness(const int i, float *iobs, complexf *obj, complexf *psi,
                                  float* obj_phase0, const int iz0, const float delta_beta,
                                  const int nprobe, const int nobj, const int nx,
                                  const int ny, const int nz)
{
  const int nxy = nx*ny;
  // Coordinates in iobs array (centered on array)
  // i[iobs] = ix + nx * (iy + ny * (iz + nz * iproj))
  // So: i / nxy = iz + iproj * nz
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;
  const int iz = (i / nxy) % nz;
  const int iproj = i / (nxy * nz);

  // Coordinates in first mode of object array
  // i[obj] = ix + nx * (iy + ny * (iobj + nobj * iproj))
  const int iobj0 = ix + nx * (iy + ny * nobj * iproj);

  // Coordinates in Psi array (fft-shifted). Assumes nx ny are multiple of 2
  // i[psi] = ix1 + nx * (iy1 + ny * (iprobe + nbprobe * (iobj + nbobj * (iz + nz * iproj))))
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));

  // Coordinate of first mode in Psi array
  const int ipsi0 = ix1 + nx * (iy1 + ny * nprobe * nobj * (iz + nz * iproj));

  const float mut = -logf(abs(psi[ipsi0]));
  // Store log(obj) in psi
  psi[ipsi0] = complexf(-0.5f * mut, -0.5f * mut * delta_beta);

  // Use approximations if absorption or phase shift is small
  float a = 0.5 * mut;
  if(fabs(a) < 1e-4)
      a = 1 - a ;
  else
    a = expf(-a);

  const float ph = -0.5f * mut * delta_beta;
  if(iz == iz0)
  {
    //obj_phase0[iobj0] = __float2half(ph);
    obj_phase0[iobj0] = ph;
    if(fabs(ph)<1e-4)
      obj[iobj0] = complexf(a * (1-ph*ph), a * ph);
    else
      obj[iobj0] = complexf(a * cos(ph), a * sin(ph));
  }
  // Set to 0 other object modes
  for(int iobj=1;iobj<nobj;iobj++)
    obj[iobj0 + nxy * iobj] = 0;
}


/** Copy the iobs_empty float array into the probe complex array
*
* This will also interpolate the padded & masked areas
*/
__device__ void IobsEmpty2Probe(const int i, float *iobs_empty, complexf *probe,
                         const int nprobe, const int nx, const int ny, const int nz,
                         const int padding)
{
  const int nxy = nx * ny;
  // Coordinates
  // i[iobs_empty] = ix + nx * (iy + ny * iz)
  // i[probe]      = ix + nx * (iy + ny * (iprobe + nprobe * iz))
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;
  const int iz = i / nxy;
  // Coordinate of first probe mode
  const int i0 = ix + nx * (iy + ny * nprobe * iz);

  float obs = iobs_empty[i];

  // masked values, including padding
  if(obs<0)
  {
    // Simple shifting interpolation. Crude & not very efficient memory-wise but should not matter
    // as this is executed maybe once per analysis
    float v0 = 0;
    float n = 0;

    // If padding is used, interpolate with pixels up to padding+16 in all directions.
    // For normal masked pixels interpolate up to 16
    int range;
    if((ix < padding) || (iy < padding) || (ix >= (nx-padding)) || (iy >= (ny-padding))) range = 2 * padding + 16;
    else range = 16;
    int step = 1;
    if(range>8) step = range / 8;
    for(int dx=-range; dx<range; dx+=step)
      for(int dy=-range; dy<range; dy+=step)
      {
        const int ix0 = (ix + dx + nx) % nx;
        const int iy0 = (iy + dy + ny) % ny;
        // Inverse distance**2 weighting
        const float w = 1 / float(dx * dx + dy * dy);
        const float d0 = iobs_empty[ix0 + nx * (iy0 + ny * iz)];
        if(d0>=0)
        {
          v0 += d0 * w;
          n += w;
        }
      }
    obs = v0 / n;
  }

  #if 0
  // Fill in only the first probe mode
  probe[i0] = complexf(sqrtf(obs), 0.0f);
  for(int iprobe=1;iprobe<nprobe;iprobe++)
    probe[i0 + nxy * iprobe] = complexf(0.0f, 0.0f);
  #else
  // All probe modes initialised to the same value
  for(int iprobe=0;iprobe<nprobe;iprobe++)
    probe[i0 + nxy * iprobe] = complexf(sqrtf(obs), 0.0f);
  #endif
}

// Paganin Fourier operator with multiple distances.
// This should be called for a single layer of psi (size ny*nx) and this will update
// all the projections' first mode with the FT of the object phase.
__device__ void paganin_fourier_multi(const int i, complexf *psi, float* pilambdad,
                                      const float delta_beta,
                                      const float px, const int nb_mode, const int nx, const int ny,
                                      const int nz, const int nb_proj, const float alpha)
{
  // psi shape is (stack_size, nb_z, nb_obj, nb_probe, ny, nx)

  const int nxy = nx * ny;
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;

  // Assumes ny, nx are multiples of 2
  const float ky = (iy - (int)ny *(int)(iy >= ((int)ny / 2)))  / (px * (float)ny) ;
  const float kx = (ix - (int)nx *(int)(ix >= ((int)nx / 2)))  / (px * (float)nx) ;
  const float k2 =  kx * kx + ky * ky;

  for(int iproj=0; iproj<nb_proj; iproj++)
  {

    complexf n=0;
    float d=0;

    for(int iz=0; iz<nz; iz++)
    {
      const complexf ps = psi[ix + nx * (iy + ny * nb_mode * (iz + nz * iproj))];
      n += (1 + pilambdad[iz] * delta_beta * k2) * ps;
      d += (1 + pilambdad[iz] * delta_beta * k2) * (1 + pilambdad[iz] * delta_beta * k2);
    }
    psi[ix + nx * (iy + ny * nb_mode * nz * iproj)] = n / (d + nz * alpha);
  }
}

/** Convert the result of the inverse Fourier transform of Paganin's operator to the object.
* This should be called for a single layer of psi (size ny*nx) and this will update
* all the object projections.
*
* This is used at the end of the multi-distance Paganin operator
*/
__device__ void paganin2obj(const int i, complexf *psi, complexf *obj, float* obj_phase0,
                            const float delta_beta, const int nb_probe, const int nb_obj,
                            const int nx, const int ny, const int nz, const int nb_proj)
{
  // psi shape is (stack_size, nb_z, nb_obj, nb_probe, ny, nx)
  // obj shape is (nb_proj, nb_obj, ny, nx)
  const int nxy = nx * ny;
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;

  // fft-shift coordinates in object
  const int ixo = ix - nx / 2 + nx * (ix < (nx / 2));
  const int iyo = iy - ny / 2 + ny * (iy < (ny / 2));

  for(int iproj=0; iproj<nb_proj; iproj++)
  {
    const float ph = delta_beta * 0.5f * logf(abs(psi[ix + nx * (iy + ny * nb_obj * nb_probe * nz * iproj)]));

    // Use approximations if absorption or phase shift is small
    float a = ph / delta_beta;
    if(fabs(a) < 1e-4)
        a = 1 - a ;
    else
      a = expf(a);

    // Coordinates of the first mode of object array
    // i[obj] = ix + nx * (iy + ny * (iobj + nobj * iproj))
    const int iobj0 = ixo + nx * (iyo + ny * nb_obj * iproj);

    //obj_phase0[iobj0] = __float2half(ph);
    obj_phase0[iobj0] = ph;
    if(fabs(ph)<1e-4)
      obj[iobj0] = complexf(a * (1-ph*ph), a * ph);
    else
      obj[iobj0] = complexf(a * cos(ph), a * sin(ph));

    // Set to 0 other object modes
    for(int iobj=1; iobj<nb_obj; iobj++)
      obj[iobj0 + nxy * iobj] = 0;
  }
}
