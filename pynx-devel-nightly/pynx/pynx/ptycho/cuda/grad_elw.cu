inline __device__ float pow2(const float v)
{
  return v*v;
}

/** Calculate Psi.conj() * (1 - Iobs / Icalc), for the gradient calculation with Poisson noise.
* Masked pixels are set to zero.
* \param i: the point in the 3D observed intensity array for which the llk is calculated
* \param iobs: the observed in tensity array, shape=(stack_size, ny, nx)
* \param psi: the calculated complex amplitude, shape=(nb_obj, nb_probe, stack_size, ny, nx)
* \param background: the incoherent background, of shape (ny, nx)
* \param nbmode: number of modes = nb_probe * nb_obj
* \param nxy: number of pixels in a single frame
* \param nxystack: number of frames in stack multiplied by nxy
* \param hann_filter: if 1, will apply a Hann filter
* \param scale_in, scale_out: the fft scale the data must be multipied by to compensate for FFT.
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
__device__ void GradPoissonFourier(const int i, float *iobs, complexf *psi, float *background, float *background_grad,
                                   const int nbmode, const int nx, const int ny, const int nxystack, const int npsi,
                                   const char hann_filter, const float scale_in, const float scale_out)
{
  const float b = background[i % (nx * ny)];
  float db = 0.0f;

  float g=1;
  if(hann_filter>0)
  {
    // Use a Hann window multiplication to dampen high-frequencies in the object
    const int ix = i % nx;
    const int iy = (i % (nx * ny)) / nx;
    const float qx = (float)(ix - nx * (ix >= (nx / 2))) * 3.14159265f / (float)(nx-1);
    const float qy = (float)(iy - ny * (iy >= (ny / 2))) * 3.14159265f / (float)(ny-1);
    g = pow2(cosf(qx) * cosf(qy));
  }

  for(int j=0; j<npsi; j++)
  {
    const int ij = i + nx * ny * j;
    const float obs= iobs[ij];

    if(obs < 0)
    {
      for(int imode=0; imode < nbmode; imode++)
        psi[ij + imode * nxystack] = complexf(0.0f,0.0f);
    }
    else
    {
      float calc = b;
      for(int imode=0;imode<nbmode;imode++) calc += dot(psi[ij + imode * nxystack], psi[ij + imode* nxystack]);

      calc = 1 - obs / fmaxf(1e-12f, pow2(scale_in) * calc);  // TODO: KLUDGE ? 1e-12f is arbitrary

      db += calc;

      const float f = scale_out * g * calc;

      for(int imode=0; imode < nbmode; imode++)
      {
        // TODO: store psi to avoid double-read. Or just assume it's cached.
        const complexf ps = psi[ij + imode * nxystack];
        psi[ij + imode * nxystack] = complexf(f*ps.real() , f*ps.imag());
      }
    }
  }
  background_grad[i % (nx * ny)] = -db;  // check sign
}


/** Elementwise kernel to compute the object gradient from psi. Almost the same as the kernel to compute the
* updated object projection, except that no normalization array is retained.
* This must be called for a single frame at a time to avoid write conflicts.
*/
__device__ void GradObj(const int i, complexf* psi, complexf *objgrad, complexf* probe,
             const float cx,  const float cy, const float px, const float f,
             const int stack_size, const int nx, const int ny, const int nxo, const int nyo,
             const int nbobj, const int nbprobe, const bool interp)
{
  // Coordinate
  const int prx = i % nx;
  const int pry = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Apply Quadratic phase factor after far field propagation
  const float y = (pry - ny/2) * px;
  const float x = (prx - nx/2) * px;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  float s, c;
  __sincosf(tmp , &s, &c);
  // TODO: take into account subpixel shift for GradObj
  //const int iobj0 = __float2int_rn(cx)+prx + nxo*(__float2int_rn(cy)+pry);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    complexf grad=0;
    //const int iobj  = iobj0 + iobjmode * nxo * nyo;
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      const complexf pr = probe[i + iprobe*nx*ny];
      complexf ps=psi[ipsi + stack_size * (iprobe + iobjmode * nbprobe) * nx * ny];
      ps = complexf(ps.real()*c - ps.imag()*s , ps.imag()*c + ps.real()*s);
      grad += complexf(pr.real()*ps.real() + pr.imag()*ps.imag() , pr.real()*ps.imag() - pr.imag()*ps.real() );
    }
    //objgrad[iobj] -= grad;
    bilinear_atomic_add_c(objgrad, -grad, cx + prx, cy + pry, iobjmode, nxo, nyo, interp);
  }
}

/** Elementwise kernel to compute the object gradient from psi. Almost the same as the kernel to compute the
* updated object projection, except that no normalization array is retained.
* Atomic version, allowing to process simultaneously all frames in a stack.
* This should be called with a first argument array with a size of nx*ny, i.e. one frame size. Each parallel
* kernel execution treats one pixel, for all frames and all modes.
*/
__device__ void GradObjAtomic(const int i, complexf* psi, complexf *objgrad, complexf* probe,
                              float* cx,  float* cy, const float px, const float f,
                              const int stack_size, const int nx, const int ny, const int nxo, const int nyo,
                              const int nbobj, const int nbprobe, const int npsi, const bool interp)
{
  // Coordinate
  const int prx = i % nx;
  const int pry = i / nx;
  //const int nxyo = nxo * nyo;
  const int nxy = nx * ny;

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Apply Quadratic phase factor after far field propagation
  const float y = (pry - ny/2) * px;
  const float x = (prx - nx/2) * px;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  float s, c;
  __sincosf(tmp , &s, &c);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    for(int j=0;j<npsi;j++)
    {
      // TODO: take into account subpixel shift for GradObj
      //const int iobj0 = __float2int_rn(cx[j])+prx + nxo*(__float2int_rn(cy[j])+pry);

      complexf o=0;
      //const int iobj  = iobj0 + iobjmode * nxyo;
      for(int iprobe=0 ; iprobe < nbprobe ; iprobe++)
      {
        const complexf pr = probe[i + iprobe*nx*ny];
        complexf ps = psi[ipsi + nxy * (j + stack_size * (iprobe + iobjmode * nbprobe)) ];
        ps = complexf(ps.real()*c - ps.imag()*s , ps.imag()*c + ps.real()*s);
        o += complexf(pr.real()*ps.real() + pr.imag()*ps.imag() , pr.real()*ps.imag() - pr.imag()*ps.real());
      }
      //atomicAdd(&objgrad[iobj], o);
      bilinear_atomic_add_c(objgrad, -o, cx[j] + prx, cy[j] + pry, iobjmode, nxo, nyo, interp);
    }
  }
}

/** Elementwise kernel to compute the probe gradient from psi. Almost the same as the kernel to compute the
* updated probe projection, except that no normalization array is retained.
*/
__device__ void GradProbe(const int i, complexf* psi, complexf* probegrad, complexf *obj,float* cx,  float* cy,
                          const float px, const float f, const char firstpass, const int npsi, const int stack_size,
                          const int nx, const int ny, const int nxo, const int nyo, const int nbobj, const int nbprobe,
                          const bool interp)
{
  const int prx = i % nx;
  const int pry = i / nx;
  const int iprobe=   i;

  // obj and probe are centered arrays, Psi is fft-shifted

  // Coordinates in Psi array (origin at (0,0)). Assume nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Apply Quadratic phase factor after far field propagation
  const float y = (pry - ny/2) * px;
  const float x = (prx - nx/2) * px;
  const float tmp = f*(x*x+y*y);
  // NOTE WARNING: if the argument becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  float s, c;
  __sincosf(tmp , &s, &c);

  for(int iprobemode=0;iprobemode<nbprobe;iprobemode++)
  {
    complexf p=0;
    for(int j=0;j<npsi;j++)
    {
      for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
      {
        complexf ps = psi[ipsi + (j + stack_size * (iprobemode + iobjmode * nbprobe) ) * nx * ny];
        ps = complexf(ps.real()*c - ps.imag()*s , ps.imag()*c + ps.real()*s);

        const complexf o = bilinear(obj, cx[j]+prx, cy[j]+pry, iobjmode, nxo, nyo, interp, false);

        p += complexf(o.real()*ps.real() + o.imag()*ps.imag() , o.real()*ps.imag() - o.imag()*ps.real());
      }
    }
    if(firstpass) probegrad[iprobe + iprobemode * nx * ny] = -p ;
    else probegrad[iprobe + iprobemode * nx * ny] -= p ;
  }
}


/** Regularisation gradient, to penalise local variations in the object or probe array
*/
__device__ void GradReg(const int i, complexf *dv, complexf *v, const float alpha, const int nx, const int ny)
{
  const int x = i % nx;
  const int y = (i % (nx * ny)) / nx;

  const complexf v0=v[i];
  complexf d = complexf(0, 0);

  // The 4 cases could be put in a loop for simplicity (but not performance)
  if(x>0)
  {
    const complexf v1=v[i-1];
    d += complexf(v0.real()-v1.real(), v0.imag()-v1.imag());
  }
  if(x<(nx-1))
  {
    const complexf v1=v[i+1];
    d += complexf(v0.real()-v1.real(), v0.imag()-v1.imag());
  }
  if(y>0)
  {
    const complexf v1=v[i-nx];
    d += complexf(v0.real()-v1.real(), v0.imag()-v1.imag());
  }
  if(y<(ny-1))
  {
    const complexf v1=v[i+nx];
    d += complexf(v0.real()-v1.real(), v0.imag()-v1.imag());
  }

  dv[i] += 2 * alpha * d;
}
