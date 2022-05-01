/** Calculate Psi.conj() * (1 - Iobs / Icalc), for the gradient calculation with Poisson noise.
* Masked pixels are set to zero.
* This is called for the first frame of a stack of observed intensities, and will loop over all frames.
* \param i: the point in the 3D observed intensity array for which the llk is calculated
* \param iobs: the observed in tensity array, shape=(stack_size, ny, nx)
* \param psi: the calculated complex amplitude, shape=(nb_obj, nb_probe, stack_size, ny, nx)
* \param background: the incoherent background, of shape (ny, nx)
* \param background_grad: the incoherent background, of shape (ny, nx), always updated
* \param nbmode: number of modes = nb_probe * nb_obj
* \param nxystack: number of frames in stack multiplied by nx * ny
* \param npsi: number of frames in stack
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
void GradPoissonFourier(const int i, __global float *iobs, __global float2 *psi, __global float *background,
                        __global float *background_grad, const int nbmode, const int nx, const int ny,
                        const int nxystack, const int npsi, const char hann_filter, __global float* scale)
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
    g = pown(native_cos(qx) * native_cos(qy), 2);
  }

  for(int j=0; j<npsi; j++)
  {
    const int ij = i + nx * ny * j;
    const float obs= iobs[ij];

    if(obs < 0)
    {
      for(int imode=0; imode < nbmode; imode++)
        psi[ij + imode * nxystack] = (float2) (0.0f,0.0f);
    }
    else
    {
      float calc = b;
      for(int imode=0;imode<nbmode;imode++) calc += dot(psi[ij + imode * nxystack],psi[ij + imode* nxystack]);

      calc = 1 - obs / fmax(1e-12f, calc);  // TODO: KLUDGE ? 1e-12f is arbitrary

      db += calc * scale[j]; // is the multiplication by scale here adequate ? Effectively it's a weight

      // For the gradient multiply by scale and not sqrt(scale)
      const float f = g * calc * scale[j];

      for(int imode=0; imode < nbmode; imode++)
      {
        // TODO: store psi to avoid double-read. Or just assume it's cached.
        const float2 ps = psi[ij + imode * nxystack];
        psi[ij + imode * nxystack] = (float2) (f*ps.x , f*ps.y);
      }
    }
  }
  background_grad[i % (nx * ny)] = -db;  // check sign
}

/** Elementwise kernel to compute the object gradient from psi. Almost the same as the kernel to compute the
* updated object projection, except that no normalization array is retained.
*/
void GradObj(const int i, __global float2* psi, __global float2 *objgrad, __global float2* probe,
             const float cx,  const float cy, const float px, const float f,
             const int stack_size, const int nx, const int ny, const int nxo, const int nyo,
             const int nbobj, const int nbprobe, const char interp)
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
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    float2 grad=0;
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      const float2 pr = bilinear(probe, prx, pry, iprobe, nx, ny, interp, false);
      float2 ps = psi[ipsi + (stack_size * (iprobe + iobjmode * nbprobe) ) * nx * ny];

      ps=(float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s);
      grad += (float2) (pr.x*ps.x + pr.y*ps.y , pr.x*ps.y - pr.y*ps.x );
    }
    bilinear_atomic_add_c(objgrad, -grad, cx + prx, cy + pry, iobjmode, nxo, nyo, interp);
  }
}

/** Elementwise kernel to compute the probe gradient from psi. Almost the same as the kernel to compute the
* updated probe projection, except that no normalization array is retained.
*/
void GradProbe(const int i, __global float2* psi, __global float2* probegrad, __global float2 *obj,
                          __global float* cx,  __global float* cy, const float px, const float f, const char firstpass,
                          const int npsi, const int stack_size, const int nx, const int ny, const int nxo,
                          const int nyo, const int nbobj, const int nbprobe, const char interp)
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
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  for(int iprobemode=0;iprobemode<nbprobe;iprobemode++)
  {
    float2 p=0;
    for(int j=0;j<npsi;j++)
    {
      for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
      {
        float2 ps = psi[ipsi + (j + stack_size * (iprobemode + iobjmode * nbprobe) ) * nx * ny];
        ps=(float2)(ps.x*c - ps.y*s , ps.y*c + ps.x*s);

        const float2 o = bilinear(obj, cx[j]+prx, cy[j]+pry, iobjmode, nxo, nyo, interp, false);

        p += (float2) (o.x*ps.x + o.y*ps.y , o.x*ps.y - o.y*ps.x);
      }
    }
    if(firstpass) probegrad[iprobe + iprobemode * nx * ny] = -p ;
    else probegrad[iprobe + iprobemode * nx * ny] -= p ;
  }
}


// Sum the stack of N object gradient arrays (this must be done in this step to avoid memory access conflicts)
void SumGradN(const int i, __global float2 *objN, __global float2 *obj, const int stack_size, const int nxyo, const int nbobj)
{
  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
     float2 o=0;
     for(int j=0;j<stack_size;j++)
     {
       o += objN[i + (j + iobjmode*stack_size) * nxyo];
     }
     obj[i + iobjmode*nxyo]=o;
  }
}

/** Regularisation gradient, to penalise local variations in the object or probe array
*/
void GradReg(const int i, __global float2 *dv, __global float2 *v, const float alpha, const int nx, const int ny)
{
  const int x = i % nx;
  const int y = (i % (nx * ny)) / nx;

  const float2 v0=v[i];
  float2 d = (float2)(0, 0);

  // The 4 cases could be put in a loop for simplicity (but not performance)
  if(x>0)
  {
    const float2 v1=v[i-1];
    d += (float2)(v0.x-v1.x, v0.y-v1.y);
  }
  if(x<(nx-1))
  {
    const float2 v1=v[i+1];
    d += (float2)(v0.x-v1.x, v0.y-v1.y);
  }
  if(y>0)
  {
    const float2 v1=v[i-nx];
    d += (float2)(v0.x-v1.x, v0.y-v1.y);
  }
  if(y<(ny-1))
  {
    const float2 v1=v[i+nx];
    d += (float2)(v0.x-v1.x, v0.y-v1.y);
  }

  dv[i] += 2 * alpha * d;
}
