/** Calculate Psi.conj() * (1 - Iobs / Icalc), for the gradient calculation with Poisson noise.
* Masked pixels are set to zero.
* \param i: the point in the 3D observed intensity array for which the llk is calculated
* \param iobs: the observed in tensity array, shape=(stack_size, ny, nx)
* \param psi: the calculated complex amplitude, shape=(nb_obj, nb_probe, stack_size, ny, nx)
* \param mask: the mask (0=good pixel, >0 masked pixel) of shape (ny, nx)
* \param background: the incoherent background, of shape (ny, nx)
* \param npsi: number of valid frames in the stack, over which the integration is performd (usually equal to
*              stack_size, except for the last stack which may be incomplete (0-padded)
* \param nbmode: number of modes = nb_probe * nb_obj
* \param nxyz: number of pixels in a single frame
* \param nxyzstack: number of frames in stack multiplied by nxyz
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
void GradPoissonFourier(const int i, __global float *iobs, __global float2 *psi,
                        __global float *background, const int npsi, const int nbmode,
                        const int nx, const int ny, const int nz,
                        const int nxyz, const int nxyzstack)
{
  const float obs= iobs[i];

  if(obs < 0)
  {
    // Set masked values to zero
    for(int imode=0; imode < nbmode; imode++)
    {
      psi[i + imode * nxyzstack] = (float2) (0, 0);
    }
  }

  // Use a Hann window multiplication to dampen high-frequencies in the object
  const int ix = i % nx;
  const int iy = (i % (nx * ny)) / nx;
  const int iz = (i % (nx * ny * nz)) / (nx * ny);
  const float qx = (float)(ix - nx * (ix >= (nx / 2))) * 3.14159265f / (float)(nx-1);
  const float qy = (float)(iy - ny * (iy >= (ny / 2))) * 3.14159265f / (float)(ny-1);
  const float qz = (float)(iz - nz * (iz >= (nz / 2))) * 3.14159265f / (float)(nz-1);
  const float g = pown(native_cos(qx) * native_cos(qy) * native_cos(qz), 2);

  float calc = 0;
  for(int imode=0;imode<nbmode;imode++) calc += dot(psi[i + imode * nxyzstack],psi[i + imode* nxyzstack]);

  calc = fmax(1e-12f,calc);  // TODO: KLUDGE ? 1e-12f is arbitrary

  const float f = g * (1 - obs/ (calc + background[i%nxyz]));

  for(int imode=0; imode < nbmode; imode++)
  {
    // TODO: store psi to avoid double-read. Or just assume it's cached.
    const float2 ps = psi[i + imode * nxyzstack];
    psi[i + imode * nxyzstack] = (float2) (f*ps.x , f*ps.y);
  }
}


/** Elementwise kernel to compute the object gradient from psi. Almost the same as the kernel to compute the
* updated object projection, except that no normalization array is retained.
* This kernel computes the gradient contribution from a single probe position and all object modes.
*
* The returned value is actually the conjugate of the gradient.
*/
void GradObj(const int i, __global float2* psi, __global float2 *objgrad, __global float2* probe,
             const int cx,  const int cy,  const int cz, const float px, const float f,
             const int stack_size, const int nx, const int ny, const int nz,
             const int nxo, const int nyo, const int nzo, const int nbobj, const int nbprobe)
{
  const int prx = i % nx;
  const int prz = i / (nx*ny);
  const int pry = (i - prz * nx * ny) / nx;
  const int nxyz = nx * ny * nz;
  const int nxyzo = nxo * nyo * nzo;

  // Coordinates in Psi array, fft-shifted (origin at (0,0,0)). Assume nx ny nz are multiple of 2
  const int iz = prz - nz/2 + nz * (prz<(nz/2));
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + (iy + iz * ny) * nx ;

  #if 0
  // Apply Quadratic phase factor before far field propagation
  const float y = (pry - ny/2) * pixel_size;
  const float x = (prx - nx/2) * pixel_size;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);
  #else
  const float c=1, s=0;
  #endif

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    float2 grad=0;
    const int iobj0 = cx+prx + nxo*(cy+pry + nyo * (cz + prz));
    const int iobj  = iobj0 + iobjmode * nxyzo;
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      const float2 pr = probe[i + iprobe*nxyz];
      float2 ps=psi[ipsi + stack_size * (iprobe + iobjmode * nbprobe) * nxyz];
      ps=(float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s);
      grad += (float2) (pr.x*ps.x + pr.y*ps.y , pr.x*ps.y - pr.y*ps.x );
    }
    objgrad[iobj] -= grad;
  }
}

/** Elementwise kernel to compute the probe gradient from psi. Almost the same as the kernel to compute the
* updated probe projection, except that no normalization array is retained.
*/
void GradProbe(const int i, __global float2* psi, __global float2* probegrad, __global float2 *obj,
                          __global int* cx,  __global int* cy,  __global int* cz,
                          const float px, const float f, const char firstpass, const int npsi, const int stack_size,
                          const int nx, const int ny, const int nz, const int nxo, const int nyo, const int nzo,
                          const int nbobj, const int nbprobe)
{
  const int prx = i % nx;
  const int prz = i / (nx*ny);
  const int pry = (i - prz * nx * ny) / nx;
  const int nxyz = nx * ny * nz;
  const int nxyzo = nxo * nyo * nzo;

  // Coordinates in Psi array, fft-shifted (origin at (0,0,0)). Assume nx ny nz are multiple of 2
  const int iz = prz - nz/2 + nz * (prz<(nz/2));
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + (iy + iz * ny) * nx ;

  #if 0
  // Apply Quadratic phase factor before far field propagation
  const float y = (pry - ny/2) * pixel_size;
  const float x = (prx - nx/2) * pixel_size;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);
  #else
  const float c=1, s=0;
  #endif

  for(int iprobemode=0;iprobemode<nbprobe;iprobemode++)
  {
    float2 p=0;
    for(int j=0;j<npsi;j++)
    {
      const int iobj0 = cx[j] + prx + nxo*(cy[j] + pry + nyo * (cz[j] + prz));
      for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
      {
        float2 ps = psi[ipsi + (j + stack_size * (iprobemode + iobjmode * nbprobe) ) * nxyz];
        ps=(float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s);

        const int iobj  = iobj0 + iobjmode * nxyzo;
        const float2 o = obj[iobj];

        p += (float2) (o.x*ps.x + o.y*ps.y , o.x*ps.y - o.y*ps.x);
      }
    }
    if(firstpass) probegrad[i + iprobemode * nxyz] = -p ;
    else probegrad[i + iprobemode * nxyz] -= p ;
  }
}
