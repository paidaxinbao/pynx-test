/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Elementwise kernel to compute the 3D object gradient from psi. This is used to update the object value so that
* it fits the computed Psi value (during AP or DM algorithms).
* This kernel computes the object gradient:
* - for a single probe position (to avoid memory conflicts),
* - for all object modes
* - for a given (ix,iy) coordinate in the object, and all iz values.
* - points not inside the object support have a null gradient
*
* The returned value is the conjugate of the gradient.
*/
void Psi2ObjGrad(const int i, __global float2* psi, __global float2 *obj, __global float2* probe,
                 __global float2 *grad, __global char* support, __global float* m, float cx, float cy,
                 int cixo,  int ciyo, float dsx, float dsy, float dsz, const float pxo, const float pyo,
                 const float pzo, const float pxp, const float pyp, const float f, const int stack_size,
                 const int nx, const int ny, const int nxo, const int nyo, const int nzo, const int nxp, const int nyp,
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
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  // Phase factor for multi-angle
  const float tmp_dsxy = dsx * x + dsy * y;
  const float dszj = dsz * pzo;

  // printf("CL (%3d,%3d) cx=%3d cy=%3d cz=%3d ixo=%3d iyo=%3d przmin=%3d przmax=%3d (%3d %3d %3d) (%3d %3d %3d)\n", prx, pry, cx, cy, cz, ixo, iyo, przmin, przmax, nx, ny, nz, nxo, nyo, nzo);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    const int iobjxy = ixo + nxo * iyo + iobjmode * nxyzo;
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      // First calculate dpsi = Psi - SUM_z (P*O)
      float2 dpsi = psi[ipsi + stack_size * (iprobe + iobjmode * nbprobe) * nxy];
      // Correct Psi for quadratic phase factor
      dpsi = (float2)(dpsi.x * c - dpsi.y * s , dpsi.y * c + dpsi.x * s);
      for(int prz=0; prz<nzo; prz++)
      {
        float2 p = interp_probe(probe, m, iprobe, cx, cy, pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);

        if((p.x != .0f) || (p.y != .0f))
        {
          const float sup = (float)support[ixo + nxo * (iyo + nyo * prz)] / 100.0f;
          if(sup > 0)
          {
            const float2 o = obj[iobjxy + prz * nxyo] * sup;

            // Correct PO for multi-angle phase factor
            const float tmp2 = tmp_dsxy + dszj * prz;
            const float s2=native_sin(tmp2);
            const float c2=native_cos(tmp2);
            p = (float2)(p.x * c2 - p.y * s2 , p.y * c2 + p.x * s2);

            dpsi -=(float2)(o.x * p.x - o.y * p.y , o.x * p.y + o.y * p.x);
          }
        }
      }
      // Now the object gradient conjugate for each z layer
      for(int prz=0; prz<nzo; prz++)
      {
        const int iobj = iobjxy + prz * nxyo;
        if(support[iobj] > 0)
        {
          float2 p = interp_probe(probe, m, iprobe, cx, cy, pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);
          if((p.x != .0f) || (p.y != .0f))
          {
            const float sup = (float)support[ixo + nxo * (iyo + nyo * prz)] / 100.0f;
            if(sup > 0)
            {
              // Correct probe for multi-angle phase factor
              const float tmp2 = tmp_dsxy + dszj * prz;
              const float s2=native_sin(tmp2);
              const float c2=native_cos(tmp2);
              p = (float2)(p.x * c2 - p.y * s2 , p.y * c2 + p.x * s2);

              // probe.conj() * dpsi, to get
              grad[iobj] -= (float2) (p.x*dpsi.x + p.y*dpsi.y , p.x*dpsi.y - p.y*dpsi.x) * sup;
            }
          }
        }
      }
    }
  }
}

/** Elementwise kernel to compute the 2D probe gradient from psi. This is used to update the probe value so that
* it fits the computed Psi value (during AP or DM algorithms).
* This kernel computes the probe gradient:
* - for a single probe position (for simplicity, atomic_add could be used for multiple Psi)
* - for all modes
* - for a given (ix,iy) coordinate in the Psi array, back-propagated to all object layers and probe pixels
*
* The returned value is the conjugate of the gradient.
*/
void Psi2ProbeGrad(const int i, __global float2* psi, __global float2 *obj, __global float2* probe,
                   __global float2 *grad, __global char* support, __global float* m, float cx, float cy,
                   int cixo,  int ciyo, float dsx, float dsy, float dsz, const float pxo, const float pyo,
                   const float pzo, const float pxp, const float pyp, const float f, const int stack_size,
                   const int nx, const int ny, const int nxo, const int nyo, const int nzo, const int nxp, const int nyp,
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
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  // Phase factor for multi-angle
  const float tmp_dsxy = dsx * x + dsy * y;
  const float dszj = dsz * pzo;

  // printf("CL (%3d,%3d) cx=%3d cy=%3d cz=%3d ixo=%3d iyo=%3d przmin=%3d przmax=%3d (%3d %3d %3d) (%3d %3d %3d)\n", prx, pry, cx, cy, cz, ixo, iyo, przmin, przmax, nx, ny, nz, nxo, nyo, nzo);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    const int iobjxy = ixo + nxo * iyo + iobjmode * nxyzo;
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      // First calculate dpsi = Psi - SUM_z (P*O)
      float2 dpsi = psi[ipsi + stack_size * (iprobe + iobjmode * nbprobe) * nxy];
      // Correct Psi for quadratic phase factor
      dpsi = (float2)(dpsi.x * c - dpsi.y * s , dpsi.y * c + dpsi.x * s);
      for(int prz=0; prz<nzo; prz++)
      {
        float2 p = interp_probe(probe, m, iprobe, cx, cy, pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);

        if((p.x != .0f) || (p.y != .0f))
        {
          const float sup = (float)support[ixo + nxo * (iyo + nyo * prz)] / 100.0f;
          if(sup > 0)
          {
            const float2 o = obj[iobjxy + prz * nxyo] * sup;

            // Correct PO for multi-angle phase factor
            const float tmp2 = tmp_dsxy + dszj * prz;
            const float s2=native_sin(tmp2);
            const float c2=native_cos(tmp2);
            p = (float2)(p.x * c2 - p.y * s2 , p.y * c2 + p.x * s2);

            dpsi -=(float2)(o.x * p.x - o.y * p.y , o.x * p.y + o.y * p.x);
          }
        }
      }
      // Now the object gradient conjugate for each z layer
      for(int prz=0; prz<nzo; prz++)
      {
        const int iobj = iobjxy + prz * nxyo;
        if(support[iobj] > 0)
        {
          const float sup = (float)support[ixo + nxo * (iyo + nyo * prz)] / 100.0f;
          if(sup > 0)
          {
            // Get 2D xy probe pixel coordinates from ix, iy, iz
            // TODO: spread probe on several pixels for a correct interpolation
            const int ixp = round((m[0] * (ixo-nxo/2) + m[1] * (iyo-nyo/2) + m[2] * (prz-nzo/2) + cx) / pxp + nxp / 2);
            const int iyp = round((m[3] * (ixo-nxo/2) + m[4] * (iyo-nyo/2) + m[5] * (prz-nzo/2) + cy) / pyp + nyp / 2);
            if((ixp>=0) && (ixp<nxp) && (iyp>=0) && (iyp<ny))
            {
              float2 o = obj[iobj];
              // Correct object for multi-angle phase factor
              // TODO: check sign
              const float tmp2 = tmp_dsxy + dszj * prz;
              const float s2=native_sin(tmp2);
              const float c2=native_cos(tmp2);
              o = (float2)(o.x * c2 - o.y * s2 , o.y * c2 + o.x * s2);

              // obj.conj() * dpsi
              grad[ixp + nxp * (iyp + iprobe * nyp)] -= (float2) (o.x*dpsi.x + o.y*dpsi.y , o.x*dpsi.y - o.y*dpsi.x) * sup;
            }
          }
        }
      }
    }
  }
}


/** Compute the optimal gamma value to fit an object to the current Psi values (during AP or DM).
* This kernel computes the gamma contribution:
* - for a stack of probe position,
* - for all object modes
* - for a given (ix,iy) coordinate in the object, and all iz values.
*
* Returns numerator and denominator of the coefficient in a float2 value.
*/
float4 Psi2Obj_Gamma(const int i, __global float2* psi, __global float2 *obj, __global float2* probe,
                     __global char* support, __global float2* dobj, __global float* m,
                     __global float* cx, __global float* cy, __global int* cixo, __global int* ciyo,
                     __global float* dsx, __global float* dsy, __global float* dsz,
                     const float pxo, const float pyo, const float pzo, const float pxp, const float pyp,
                     const float f, const int npsi, const int stack_size, const int nx, const int ny,
                     const int nxo, const int nyo, const int nzo, const int nxp, const int nyp,
                     const int nbobj, const int nbprobe)
{
  // Pixel coordinates in the projected array - X, Y only, we will loop to integrate over Z
  const int prx = i % nx;
  const int pry = i / nx;
  const int nxy = nx * ny;
  const int nxyo = nxo * nyo ;
  const int nxyzo = nxyo * nzo;

  // Pixel coordinates in the Psi array - same as prx, pry, but, fft-shifted (origin at (0,0)). nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Avoid overflow
  //const float scale = 1e-6;

  // Phase factor
  const float x = (prx - nx/2) * pxo;
  const float y = (pry - ny/2) * pyo;
  const float tmp = f*(x*x+y*y);
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  float gamma_d = 0;
  float gamma_n = 0;
  float dpsi2 = 0;

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      for(int j=0;j<npsi;j++)
      {
        const int ixo = (nxo - nx) / 2 + cixo[j] + prx;
        const int iyo = (nyo - ny) / 2 + ciyo[j] + pry;
        // if((ixo>=0) && (ixo<nxo) && (iyo>=0) && (iyo<nyo))
        if((ixo<0) || (ixo>=nxo) || (iyo<0) || (iyo>=nyo)) continue; // Outside object array ?

        // TODO: use a __local array for psi values to minimize memory transfers ? Or trust the cache.
        float2 pdo = (float2)0;
        float2 dpsi = psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy];

        // Correct quadratic phase factor in Psi
        dpsi = (float2)(dpsi.x * c - dpsi.y * s , dpsi.y * c + dpsi.x * s);

        // Phase factor for multi-angle
        const float tmp_dsxy = dsx[j] * x + dsy[j] * y;
        const float dszj = dsz[j] * pzo;

        for(int prz=0;prz<nzo;prz++)  // Should we restrict the integration range ?
        {
          float2 p = interp_probe(probe, m, iprobe, cx[j], cy[j], pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);
          if((p.x != .0f) || (p.y != .0f))
          {
            const float sup = (float)support[ixo + nxo * (iyo + nyo * prz)] / 100.0f;
            if(sup > 0)
            {
              // Correct probe for multi-angle phase factor
              const float tmp2 = tmp_dsxy + dszj * prz;
              const float s2=native_sin(tmp2);
              const float c2=native_cos(tmp2);
              p = (float2)(p.x * c2 - p.y * s2 , p.y * c2 + p.x * s2) * sup;

              const int iobj = ixo + nxo * (iyo + nyo * prz) + iobjmode * nxyzo;
              const float2 o = obj[iobj];
              const float2 d = dobj[iobj];
              dpsi -= (float2)(o.x * p.x - o.y * p.y , o.x * p.y + o.y * p.x);
              pdo += (float2)(d.x * p.x - d.y * p.y , d.x * p.y + d.y * p.x);
            }
          }
        }
        gamma_n += dpsi.x * pdo.x + dpsi.y * pdo.y;
        gamma_d += dot(pdo, pdo);
        dpsi2 += dot(dpsi, dpsi); // For fitting statistics
      }
    }
  }
  //printf("CL: gamma %15e / %15e\\n",gamma_d, gamma_n);
  return (float4)(gamma_n, gamma_d, dpsi2, 0);
}

/** Compute the optimal gamma value to fit an object abd 2D probe to the current Psi values (during AP or DM).
* This kernel computes the gamma contribution:
* - for a stack of probe positions,
* - for all object and probe modes
* - for a given (ix,iy) coordinate in the object, and all iz values.
*
* Returns numerator and denominator of the coefficient in a float2 value.
*/
float4 Psi2ObjProbe_Gamma(const int i, __global float2* psi, __global float2 *obj, __global float2* probe,
                          __global char* support, __global float2* dobj, __global float2* dprobe, __global float* m,
                          __global float* cx, __global float* cy, __global int* cixo, __global int* ciyo,
                          __global float* dsx, __global float* dsy, __global float* dsz,
                          const float pxo, const float pyo, const float pzo, const float pxp, const float pyp,
                          const float f, const int npsi, const int stack_size, const int nx, const int ny,
                          const int nxo, const int nyo, const int nzo, const int nxp, const int nyp,
                          const int nbobj, const int nbprobe)
{
  // Pixel coordinates in the projected array - X, Y only, we will loop to integrate over Z
  const int prx = i % nx;
  const int pry = i / nx;
  const int nxy = nx * ny;
  const int nxyo = nxo * nyo ;
  const int nxyzo = nxyo * nzo;

  // Pixel coordinates in the Psi array - same as prx, pry, but, fft-shifted (origin at (0,0)). nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Avoid overflow
  //const float scale = 1e-6;

  // Phase factor
  const float x = (prx - nx/2) * pxo;
  const float y = (pry - ny/2) * pyo;
  const float tmp = f*(x*x+y*y);
  const float s=native_sin(tmp);
  const float c=native_cos(tmp);

  float gamma_d = 0;
  float gamma_n = 0;
  float dpsi2 = 0;

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      for(int j=0;j<npsi;j++)
      {
        const int ixo = (nxo - nx) / 2 + cixo[j] + prx;
        const int iyo = (nyo - ny) / 2 + ciyo[j] + pry;
        // if((ixo>=0) && (ixo<nxo) && (iyo>=0) && (iyo<nyo))
        if((ixo<0) || (ixo>=nxo) || (iyo<0) || (iyo>=nyo)) continue; // Outside object array ?

        // TODO: use a __local array for psi values to minimize memory transfers ? Or trust the cache.
        float2 odp_pdo = (float2)0;
        float2 dpdo = (float2)0;
        float2 dpsi = psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy];

        // Correct quadratic phase factor in Psi
        dpsi = (float2)(dpsi.x * c - dpsi.y * s , dpsi.y * c + dpsi.x * s);

        // Phase factor for multi-angle
        const float tmp_dsxy = dsx[j] * x + dsy[j] * y;
        const float dszj = dsz[j] * pzo;

        for(int prz=0;prz<nzo;prz++)  // Should we restrict the integration range ?
        {
          float2 p = interp_probe(probe, m, iprobe, cx[j], cy[j], pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);
          if((p.x != .0f) || (p.y != .0f))
          {
            const float sup = (float)support[ixo + nxo * (iyo + nyo * prz)] / 100.0f;
            if(sup > 0)
            {
              // Correct probe for multi-angle phase factor
              const float tmp2 = tmp_dsxy + dszj * prz;
              const float s2=native_sin(tmp2);
              const float c2=native_cos(tmp2);
              p = (float2)(p.x * c2 - p.y * s2 , p.y * c2 + p.x * s2) * sup;

              const int iobj = ixo + nxo * (iyo + nyo * prz) + iobjmode * nxyzo;
              const float2 o = obj[iobj];
              const float2 d = dobj[iobj];
              float2 dp = interp_probe(dprobe, m, iprobe, cx[j], cy[j], pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);
              // Also correct dP for phase factor
              dp = (float2)(dp.x * c2 - dp.y * s2 , dp.y * c2 + dp.x * s2) * sup;

              dpsi -= (float2)(o.x * p.x - o.y * p.y , o.x * p.y + o.y * p.x);
              odp_pdo += (float2)(d.x * p.x - d.y * p.y , d.x * p.y + d.y * p.x) + (float2)(dp.x * o.x - dp.y * o.y , dp.x * o.y + dp.y * o.x);
              dpdo += (float2)(dp.x * d.x - dp.y * d.y , dp.x * d.y + dp.y * d.x);
            }
          }
        }
        gamma_n += dpsi.x * odp_pdo.x + dpsi.y * odp_pdo.y;
        gamma_d += dot(odp_pdo, odp_pdo) + 2 * (dpsi.x * dpdo.x + dpsi.y * dpdo.y);
        dpsi2 += dot(dpsi, dpsi); // For fitting statistics
      }
    }
  }
  //printf("CL: gamma %15e / %15e\\n",gamma_d, gamma_n);
  return (float4)(gamma_n, gamma_d, dpsi2, 0);
}

/** Elementwise kernel to compute the 3D updated object and its normalisation from psi.
* This back-propagation uses a replication of Psi along all z-layers, normalised by the sum of the norm of the probe
* along the entire z stack. Psi is assumed to be the difference between Psi1-Psi0, where Psi0 = SUM_z(Obj*probe)
* and Psi1 = FourierApplyAmplitude(Psi0)
*
* This kernel computes the object:
* - for a single probe position (to avoid memory access conflicts).
* - for all object and probe modes
* - for a given (ix,iy) coordinate in the object, and all iz values.
*
* This should be called with a Psi[0,0,i], i.e. with a nx*ny sized array as first argument. The probe and object modes
* will be looped over.
*
* One difficulty for this operation is the nature of the normalisation, which can be done in different ways.
* This version uses a normalisation by the sum of the probe intensity along a z, and weights all illuminations
* by the voxel probe intensity divided by the maximum probe intensity.
*/
void Psi2ObjDiffRepZ(const int i, __global float2* dpsi, __global float2 *obj_diff, __global float2* probe,
                    __global char* support, __global float* obj_norm, __global float* m, float cx, float cy,
                    int cixo,  int ciyo, float dsx, float dsy, float dsz, const float pxo, const float pyo,
                    const float pzo, const float pxp, const float pyp, const float f, const int stack_size,
                    const int nx, const int ny, const int nxo, const int nyo, const int nzo, const int nxp,
                    const int nyp, const int nbobj, const int nbprobe, const float probe_max_norm2)
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

  // Phase factor for multi-angle
  const float tmp_dsxy = f*(x*x+y*y) - dsx * x - dsy * y;  // Psi->Obj, so - sign (and f is already <0)
  const float dszj = dsz * pzo;

  // Compute sum of probe square modulus for normalisation (?)
  float prn_zsum=0;
  for(int prz=0; prz<nzo; prz++)
  {
    if(support[ixo + nxo*(iyo + nyo * prz)] > 0)
    {
      for(int iprobe=0;iprobe<nbprobe;iprobe++)
      {
        const float2 pr = interp_probe(probe, m, iprobe, cx, cy, pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);
        prn_zsum += dot(pr,pr);
      }
    }
  }
  prn_zsum = fmax(prn_zsum, probe_max_norm2 * 1e-3f);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    const int iobjxy = ixo + nxo * iyo + iobjmode * nxyzo;
    for(int prz=0; prz<nzo; prz++)
    {
      const int iobj = iobjxy + prz * nxyo;

      // Phase factor with the quadratic and multi-angle terms
      const float tmp = tmp_dsxy - dszj * prz;  // Psi->Obj, so - sign (and f is already <0)
      const float s=native_sin(tmp);
      const float c=native_cos(tmp);

      if(support[ixo + nxo*(iyo + nyo * prz)] > 0)
      {
        float2 o=0;
        float prn = 0; // normalisation
        for(int iprobe=0;iprobe<nbprobe;iprobe++)
        {
          // Correct Psi for phase factor
          float2 ps = dpsi[ipsi + stack_size * (iprobe + iobjmode * nbprobe) * nxy];
          ps = (float2)(ps.x * c - ps.y * s , ps.y * c + ps.x * s);

          const float2 pr = interp_probe(probe, m, iprobe, cx, cy, pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);
          const float prnz = dot(pr,pr);
          if(iobjmode==0) prn += prnz;
          o += (prnz / (prn_zsum)) * (float2) (pr.x*ps.x + pr.y*ps.y , pr.x*ps.y - pr.y*ps.x);
        }
        if((o.x > 0.0f) || (o.y > 0.0f))
        {
          obj_diff[iobj] += o;
          obj_norm[iobj] += prn ;
        }
      }
    }
  }
}

/** Elementwise kernel to compute the 3D updated object and its normalisation from psi.
* This back-propagation uses a replication of Psi along all z-layers, normalised by the sum of the norm of the probe
* along the entire z stack. Psi is assumed to be the difference between Psi1-Psi0, where Psi0 = SUM_z(Obj*probe)
* and Psi1 = FourierApplyAmplitude(Psi0)
*
* This kernel computes the object:
* - for a single probe position (to avoid memory access conflicts).
* - for all object and probe modes
* - for a given (ix,iy) coordinate in the object, and all iz values.
*
* This should be called with a Psi[0,0,i], i.e. with a nx*ny sized array as first argument. The probe and object modes
* will be looped over.
*
* One difficulty for this operation is the nature of the normalisation, which can be done in different ways.
* This version uses a normalisation from the sum of the probe intensity for all illuminations, and the
* number of z-layers over which the .
*/
void Psi2ObjDiffRep1(const int i, __global float2* dpsi, __global float2 *obj_diff, __global float2* probe,
                    __global char* support, __global float* obj_norm, __global float* m, float cx, float cy,
                    int cixo,  int ciyo, float dsx, float dsy, float dsz, const float pxo, const float pyo,
                    const float pzo, const float pxp, const float pyp, const float f, const int stack_size,
                    const int nx, const int ny, const int nxo, const int nyo, const int nzo, const int nxp,
                    const int nyp, const int nbobj, const int nbprobe, const float probe_max_norm2)
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

  // Phase factor for multi-angle
  const float tmp_dsxy = f*(x*x+y*y) - dsx * x - dsy * y;  // Psi->Obj, so - sign (and f is already <0)
  const float dszj = dsz * pzo;

  // Compute sum of probe square modulus for normalisation (?)
  float prn_znorm=0;
  for(int prz=0; prz<nzo; prz++)
  {
    if(support[ixo + nxo*(iyo + nyo * prz)] > 0)
    {
      prn_znorm +=1;
    }
  }

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    const int iobjxy = ixo + nxo * iyo + iobjmode * nxyzo;
    for(int prz=0; prz<nzo; prz++)
    {
      const int iobj = iobjxy + prz * nxyo;

      // Phase factor with the quadratic and multi-angle terms
      const float tmp = tmp_dsxy - dszj * prz;  // Psi->Obj, so - sign (and f is already <0)
      const float s=native_sin(tmp);
      const float c=native_cos(tmp);

      if(support[ixo + nxo*(iyo + nyo * prz)] > 0)
      {
        float2 o=0;
        float prn = 0; // normalisation
        for(int iprobe=0;iprobe<nbprobe;iprobe++)
        {
          // Correct Psi for phase factor
          float2 ps = dpsi[ipsi + stack_size * (iprobe + iobjmode * nbprobe) * nxy];
          ps = (float2)(ps.x * c - ps.y * s , ps.y * c + ps.x * s);

          const float2 pr = interp_probe(probe, m, iprobe, cx, cy, pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);
          if(iobjmode==0) prn += dot(pr,pr);
          o += (float2) (pr.x*ps.x + pr.y*ps.y , pr.x*ps.y - pr.y*ps.x);
        }
        if((o.x > 0.0f) || (o.y > 0.0f))
        {
          obj_diff[iobj] += o;
          obj_norm[iobj] += prn * prn_znorm ;
        }
      }
    }
  }
}

// Normalize object difference computing using Psi2ObjDiffRep, and update object
// The regularization term is used as in: Marchesini et al, Inverse problems 29 (2013), 115009, eq (14)
void ObjDiffNorm(const int i, __global float2 *obj_diff, __global float* objnorm, __global float2 *obj,
                 __global float *normmax, const float reg, const int nxyo, const int nbobj, const float beta)
{
  const float norm = fmax(objnorm[i], normmax[0] * reg);
  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
    obj[i + iobjmode*nxyo] += obj_diff[i + iobjmode*nxyo] * (beta / norm) ;
}


/** Elementwise kernel to compute the 3D updated object from psi.
* This back-propagation uses a replication of Psi along all z-layers, normalised by the sum of the norm of the probe
* along the entire z stack. Psi is assumed to be the difference between Psi1-Psi0, where Psi0 = SUM_z(Obj*probe)
* and Psi1 = FourierApplyAmplitude(Psi0)
*
* This kernel computes the updated object:
* - for a single probe position.
* - for all object and probe modes
* - for a given (ix,iy) coordinate in the object, and all iz values.
*
* This should be called with a Psi[0,0,i], i.e. with a nx*ny sized array as first argument. The probe and object modes
* will be looped over.
* \param probe_max_norm2: the maximum square modulus of the probe, for regularisation
*/
void Psi2ObjIncrement1(const int i, __global float2* dpsi, __global float2 *obj, __global float2* probe,
                       __global char* support, __global float* m, float cx, float cy,
                       int cixo,  int ciyo, float dsx, float dsy, float dsz, const float pxo, const float pyo,
                       const float pzo, const float pxp, const float pyp, const float f, const int stack_size,
                       const int nx, const int ny, const int nxo, const int nyo, const int nzo, const int nxp,
                       const int nyp, const int nbobj, const int nbprobe, const float probe_max_norm2, const float beta)
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

  // Phase factor for multi-angle
  const float tmp_dsxy = f*(x*x+y*y) - dsx * x - dsy * y;  // Psi->Obj, so - sign (and f is already <0)
  const float dszj = dsz * pzo;

  // Compute sum of probe square modulus for normalisation (?)
  float prn_zsum = 0;
  for(int prz=0; prz<nzo; prz++)
  {
    if(support[ixo + nxo*(iyo + nyo * prz)] > 0)
    {
      for(int iprobe=0;iprobe<nbprobe;iprobe++)
      {
        const float2 pr = interp_probe(probe, m, iprobe, cx, cy, pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);
        prn_zsum += dot(pr,pr);
      }
    }
  }
  prn_zsum = fmax(prn_zsum, 1e-3f * probe_max_norm2);
  // printf("CL (%3d,%3d) prn_zsum=%10.5f probe_max_norm2=%10.5f\n", prx, pry, prn_zsum, probe_max_norm2);

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    const int iobjxy = ixo + nxo * iyo + iobjmode * nxyzo;
    for(int prz=0; prz<nzo; prz++)
    {
      const int iobj = iobjxy + prz * nxyo;

      // Phase factor with the quadratic and multi-angle terms
      const float tmp = tmp_dsxy - dszj * prz;  // Psi->Obj, so - sign (and f is already <0)
      const float s=native_sin(tmp);
      const float c=native_cos(tmp);

      if(support[ixo + nxo*(iyo + nyo * prz)] > 0)
      {
        float2 o=0;
        for(int iprobe=0;iprobe<nbprobe;iprobe++)
        {
          // Correct Psi for phase factor
          float2 ps = dpsi[ipsi + stack_size * (iprobe + iobjmode * nbprobe) * nxy];
          ps = (float2)(ps.x * c - ps.y * s , ps.y * c + ps.x * s);

          const float2 pr = interp_probe(probe, m, iprobe, cx, cy, pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);
          o += (float2) (pr.x*ps.x + pr.y*ps.y , pr.x*ps.y - pr.y*ps.x) * (dot(pr,pr) / prn_zsum);
        }
        if((o.x > 0.0f) || (o.y > 0.0f))
          obj[iobj] += beta * o;
      }
    }
  }
}
