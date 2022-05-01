/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

// Compute 2 * P * O - Psi , with the quadratic phase factor
void ObjectProbePsiDM1(const int i, __global float2* psi, __global float2 *obj, __global float2* probe,
                       __global char* support, __global float* m, __global float* cx, __global float* cy,
                       __global int* cixo, __global int* ciyo, __global float* dsx, __global float* dsy,
                       __global float* dsz, const float pxo, const float pyo, const float pzo, const float pxp,
                       const float pyp, const float f, const int npsi, const int stack_size, const int nx, const int ny,
                       const int nxo, const int nyo, const int nzo, const int nxp, const int nyp,
                       const int nbobj, const int nbprobe)
{
  // Pixel coordinates in the projected array - X, Y only, we will loop to integrate over Z
  const int prx = i % nx;
  const int pry = i / nx;
  const int nxy = nx * ny;
  const int nxyzo = nxo * nyo * nzo;

  // Pixel coordinates in the Psi array - same as prx, pry, but, fft-shifted (origin at (0,0)). nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Apply Quadratic phase factor before far field propagation
  const float y = (pry - ny/2) * pyo;
  const float x = (prx - nx/2) * pxo;

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      for(int j=0;j<npsi;j++)
      {
        float2 ps1 = (float2)0;
        const float tmp_dsxy = f*(x*x+y*y) + dsx[j] * x + dsy[j] * y;
        const float dszj = dsz[j] * pzo;

        // (cixo, ciyo) = (0,0) correspond to the center of the object
        const int ixo = (nxo - nx) / 2 + cixo[j] + prx;
        const int iyo = (nyo - ny) / 2 + ciyo[j] + pry;
        if((ixo>=0) && (ixo<nxo) && (iyo>=0) && (iyo<nyo))
        {
          for(int prz=0;prz<nzo;prz++)  // Should we restrict the integration range ?
          {
            const float2 p = interp_probe(probe, m, iprobe, cx[j], cy[j], pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);
            if((p.x != .0f) || (p.y != .0f))
            {
              const float sup = (float)support[ixo + nxo * (iyo + nyo * prz)] / 100.0f;
              if(sup > 0)
              {
                const float2 o = obj[ixo + nxo * (iyo + nyo * prz) + iobjmode * nxyzo] * sup;
                float2 ps=(float2)(o.x * p.x - o.y * p.y , o.x * p.y + o.y * p.x);

                // Add the phase factor with the quadratic and multi-angle terms
                const float tmp = tmp_dsxy + dszj * prz;
                const float s=native_sin(tmp);
                const float c=native_cos(tmp);

                ps1 += (float2)(ps.x * c - ps.y * s , ps.y * c + ps.x * s);
              }
            }
          }
        }
        const int ii = ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy;
        psi[ii] = 2 * ps1 - psi[ii];
      }
    }
  }
}

/** Update Psi (with quadratic phase)
* Psi(n+1) = Psi(n) - P*O + Psi_calc ; where Psi_calc=Psi_fourier is (2*P*O - Psi(n)) after applying Fourier constraints
*/
void ObjectProbePsiDM2(const int i, __global float2* psi, __global float2* psi_fourier, __global float2 *obj,
                       __global float2* probe, __global char* support, __global float* m,
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
  const int nxyzo = nxo * nyo * nzo;

  // Pixel coordinates in the Psi array - same as prx, pry, but, fft-shifted (origin at (0,0)). nx ny are multiple of 2
  const int iy = pry - ny/2 + ny * (pry<(ny/2));
  const int ix = prx - nx/2 + nx * (prx<(nx/2));
  const int ipsi  = ix + iy * nx ;

  // Apply Quadratic phase factor before far field propagation
  const float y = (pry - ny/2) * pyo;
  const float x = (prx - nx/2) * pxo;

  for(int iobjmode=0;iobjmode<nbobj;iobjmode++)
  {
    for(int iprobe=0;iprobe<nbprobe;iprobe++)
    {
      for(int j=0;j<npsi;j++)
      {
        float2 ps1 = (float2)0;
        const float tmp_dsxy = f*(x*x+y*y) + dsx[j] * x + dsy[j] * y;
        const float dszj = dsz[j] * pzo;

        // (cixo, ciyo) = (0,0) correspond to the center of the object
        const int ixo = (nxo - nx) / 2 + cixo[j] + prx;
        const int iyo = (nyo - ny) / 2 + ciyo[j] + pry;
        if((ixo>=0) && (ixo<nxo) && (iyo>=0) && (iyo<nyo))
        {
          for(int prz=0;prz<nzo;prz++)  // Should we restrict the integration range ?
          {
            const float2 p = interp_probe(probe, m, iprobe, cx[j], cy[j], pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);
            if((p.x != .0f) || (p.y != .0f))
            {
              const float sup = (float)support[ixo + nxo * (iyo + nyo * prz)] / 100.0f;
              if(sup > 0)
              {
                const float2 o = obj[ixo + nxo * (iyo + nyo * prz) + iobjmode * nxyzo] * sup;
                float2 ps=(float2)(o.x * p.x - o.y * p.y , o.x * p.y + o.y * p.x);

                // Add the phase factor with the quadratic and multi-angle terms
                const float tmp = tmp_dsxy + dszj * prz;
                const float s=native_sin(tmp);
                const float c=native_cos(tmp);

                ps1 += (float2)(ps.x * c - ps.y * s , ps.y * c + ps.x * s);
              }
            }
          }
        }
        const int ii = ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy;
        psi[ii] += psi_fourier[ii] - ps1;
      }
    }
  }
}

