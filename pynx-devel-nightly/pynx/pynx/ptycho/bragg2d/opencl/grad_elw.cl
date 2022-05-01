/** Calculate Psi.conj() * (1 - Iobs / Icalc), for the gradient calculation with Poisson noise.
* Masked pixels are set to zero.
* \param i: the point in the 2D observed intensity array for which the calculation is made
* \param iobs: the observed in tensity array, shape=(stack_size, ny, nx)
* \param psi: the calculated complex amplitude, shape=(nb_obj, nb_probe, stack_size, ny, nx)
* \param background: the incoherent background, of shape (ny, nx)
* \param npsi: number of valid frames in the stack, over which the integration is performd (usually equal to
*              stack_size, except for the last stack which may be incomplete (0-padded)
* \param nbmode: number of modes = nb_probe * nb_obj
* \param nxy: number of pixels in a single frame
* \param nxystack: number of frames in stack multiplied by nxy
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
void GradPoissonFourier(const int i, __global float *iobs, __global float2 *psi,
                        __global float *background, const int nbmode,
                        const int nx, const int ny, const int nxystack)
{
  const float obs= iobs[i];

  if(obs < 0)
  {
    // Set masked values to zero
    for(int imode=0; imode < nbmode; imode++)
    {
      psi[i + imode * nxystack] = (float2) (0, 0);
    }
  }

  // Use a Hann window multiplication to dampen high-frequencies in the object
  const int ix = i % nx;
  const int iy = (i % (nx * ny)) / nx;
  const float qx = (float)(ix - nx * (ix >= (nx / 2))) * 3.14159265f / (float)(nx-1);
  const float qy = (float)(iy - ny * (iy >= (ny / 2))) * 3.14159265f / (float)(ny-1);
  const float g = pown(native_cos(qx) * native_cos(qy), 2);

  float calc = 0;
  for(int imode=0;imode<nbmode;imode++) calc += dot(psi[i + imode * nxystack], psi[i + imode* nxystack]);

  calc = fmax(1e-12f,calc);  // TODO: KLUDGE ? 1e-12f is arbitrary

  const float f = g * (1 - obs/ (calc + background[i % (nx * ny)]));

  for(int imode=0; imode < nbmode; imode++)
  {
    // TODO: store psi to avoid double-read. Or just assume it's cached.
    const float2 ps = psi[i + imode * nxystack];
    psi[i + imode * nxystack] = (float2) (f*ps.x , f*ps.y);
  }
}


/** Elementwise kernel to compute the object gradient from psi. Almost the same as the kernel to compute the
* updated object projection, except that no normalization array is retained.
* This kernel computes the gradient contribution:
* - for a single probe position (to avoid memory conflicts),
* - for all object modes
* - for a given (ix,iy) coordinate in the object, and all iz values.
* - points not inside the object support have a null gradient
*
* The stored array is the conjugate of the gradient.
*/
void GradObj(const int i, __global float2* psi, __global float2 *objgrad, __global float2* probe,
             __global char* support, __global float* m, float cx, float cy, int cixo,  int ciyo,
             float dsx, float dsy, float dsz, const float pxo, const float pyo, const float pzo,
             const float pxp, const float pyp, const float f, const int stack_size, const int nx, const int ny,
             const int nxo, const int nyo, const int nzo, const int nxp, const int nyp,
             const int nbobj, const int nbprobe)
{
  // Pixel coordinates in the projected array - X, Y only, we will loop to integrate over Z
  const int prx = i % nx;
  const int pry = i / nx;
  const int nxy = nx * ny;
  const int nxyo = nxo * nyo ;
  const int nxyzo = nxyo * nzo;

  // Pixel object coordinates
  const int ixo = (nxo - nx) / 2 + cixo + prx;
  const int iyo = (nyo - ny) / 2 + ciyo + pry;
  if((ixo<0) || (ixo>=nxo) || (iyo<0) || (iyo>=nyo)) return; // Outside object array ?

  // Pixel coordinates in the Psi array - same as prx, pry, but, fft-shifted (origin at (0,0)). nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Phase factor
  const float x = (prx - nx/2) * pxo;
  const float y = (pry - ny/2) * pyo;
  const float tmp = f*(x*x+y*y);

  // Phase factor for multi-angle
  const float tmp_dsxy = dsx * x + dsy * y;
  const float dszj = dsz * pzo;

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    const int iobjxy = ixo + nxo * iyo + iobjmode * nxyzo;
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      const float2 ps0=psi[ipsi + stack_size * (iprobe + iobjmode * nbprobe) * nxy];

      for(int prz=0; prz<nzo; prz++)
      {
        const int iobj = iobjxy + prz * nxyo;
        const float sup = (float)support[iobj] / 100.0f;
        if(sup > 0)
        {
          const float2 pr = interp_probe(probe, m, iprobe, cx, cy, pxp, pyp, ixo-nxo/2, iyo-nyo/2, prz-nzo/2, nxp, nyp) * sup;

          // TODO: check the phase factor signs
          const float tmp2 = tmp_dsxy + dszj * prz;
          const float s=native_sin(tmp2);
          const float c=native_cos(tmp2);

          const float2 ps=(float2)(ps0.x*c + ps0.y*s , ps0.y*c - ps0.x*s);
          objgrad[iobj] -= (float2) (pr.x*ps.x + pr.y*ps.y , pr.x*ps.y - pr.y*ps.x );
        }
      }
    }
  }
}

// Gaussian convolution of a 3D array (the object gradient) along its 3rd dimension. This elementwise kernel
// should be called for each of the 2D pixels in the XY base plane, and will apply to all modes and pixels along z.
void GaussConvolveZ(const int i, __global float2 *grad, const float sigma,
                    const int nxyo, const int nzo, const int nbobj)
{
   float2 g[15];  // 15 = 2*7+1 or any 2*N+1
   const float norm = 1 / (sigma *native_sqrt(2*3.141592653589f));
   for(int imode=0;imode<nbobj;imode++)
   {
     for(int iz=-7;iz<=7;iz++)
     {
       if(iz>=0) g[iz+7] = grad[i + nxyo * (iz + nzo * imode)];
       else      g[iz+7] = grad[i + nxyo * (iz + nzo * (imode+1))];
     }
     for(int iz=0; iz<nzo;iz++)
     {
        float2 v=0;
        // % could be replaced by a AND (& (2^n - 1)) if the kernel was a power of two-sized
        for(int j=-7;j<=7;j++) v += g[(iz+j+7)%15] * native_exp(-j*j/(2*sigma*sigma)) * norm ;
        grad[i + nxyo * (iz + nzo *imode)] = v;
        g[iz%15] = grad[i + nxyo * ((iz + 7 + 1) % nzo + nzo *imode)];
     }
   }
}

/** Elementwise kernel to compute the probe gradient from psi.
*
* This kernel computes the probe gradient contribution:
* - for one probe positions (NB: it could loop over probe positions, we use atomic operations)
* - for all object and probe modes
* - for a given (ix,iy) coordinate in the object, and all iz values.
* - points not inside the object support have a null gradient
*
* The probe gradient in each is the sum of the gradients computed
*
* The stored array is the conjugate of the gradient.
*/
void GradProbe(const int i, __global float2* psi, __global float2 *probe_grad, __global float2* obj,
               __global char* support, __global float* m, float cx, float cy, int cixo,  int ciyo,
               float dsx, float dsy, float dsz, const float pxo, const float pyo, const float pzo,
               const float pxp, const float pyp, const float f, const int stack_size, const int nx, const int ny,
               const int nxo, const int nyo, const int nzo, const int nxp, const int nyp,
               const int nbobj, const int nbprobe)
{
  // Pixel coordinates in the projected array - X, Y only, we will loop to integrate over Z
  const int prx = i % nx;
  const int pry = i / nx;
  const int nxy = nx * ny;
  const int nxyo = nxo * nyo ;
  const int nxyzo = nxyo * nzo;

  // Pixel object coordinates
  const int ixo = (nxo - nx) / 2 + cixo + prx;
  const int iyo = (nyo - ny) / 2 + ciyo + pry;
  if((ixo<0) || (ixo>=nxo) || (iyo<0) || (iyo>=nyo)) return; // Outside object array ?

  // Pixel coordinates in the Psi array - same as prx, pry, but, fft-shifted (origin at (0,0)). nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Phase factor
  const float x = (prx - nx/2) * pxo;
  const float y = (pry - ny/2) * pyo;
  const float tmp = f*(x*x+y*y);

  // Phase factor for multi-angle
  const float tmp_dsxy = dsx * x + dsy * y;
  const float dszj = dsz * pzo;

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    const int iobjxy = ixo + nxo * iyo + iobjmode * nxyzo;
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      const float2 ps0=psi[ipsi + stack_size * (iprobe + iobjmode * nbprobe) * nxy];

      for(int prz=0; prz<nzo; prz++)
      {
        const int iobj = iobjxy + prz * nxyo;
        const float sup = (float)support[iobj] / 100.0f;
        if(sup > 0)
        {
          const float2 o = obj[iobj] * sup;  // Really need * sup ?

          // TODO: check the phase factor signs
          const float tmp2 = tmp_dsxy + dszj * prz;
          const float s=native_sin(tmp2);
          const float c=native_cos(tmp2);

          const float2 ps = (float2)(ps0.x*c + ps0.y*s , ps0.y*c - ps0.x*s);
          interp_probe_spread(probe_grad, (float2) (o.x*ps.x + o.y*ps.y , o.x*ps.y - o.y*ps.x ),
                              m, iprobe, cx, cy, pxp, pyp, ixo-nxo/2, iyo-nyo/2, prz-nzo/2, nxp, nyp);
        }
      }
    }
  }
}
