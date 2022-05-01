/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Multiply object and probe in 3D, and project along z to obtain a 2D wavefront.
* This kernel must be called for a psi array with shape (nb_obj, nb_probe,stack_size, ny, nx).
* The Psi array supplied to this elementwise kernel should be the first frame, and the calculation
* will be performed for all frames in the stack
* This version uses a 2D probe, which is interpolated to compute the probe values on the 3D object grid.
*
* \param psi: stack of Psi frames (Obj*Probe).sum()
* \param obj: 3D object + modes
* \param probe: 2D probe, assumed to be constant perpendicular to the laboratory z-axis
* \param m: matrix to transform the integer object coordinates to xyz coordinates in the laboratory (2D probe) frame
* \param cx, cy: shift of the illumination position (laboratory frame units, in meters)
* \param cxo, cyo: integer coordinates of the center of the illuminated part of the object, for each frame.
* \param dsx, dsy, dsz: difference between the reference (average) scattering vector and the one for each frame.
* \param pxo, pyo, pzo: pixel sizes in the object array
* \param f: factor for quadratic phase calculation
* \param: pxp, pyp: pixel sizes in the 2D probe array
* \param npsi, stack_size: number of valid frames and total number of frames in a given stack
* \param nx, ny: probe shape
* \param nxo, nyo, nzo: object shape
* \param nbobj, nbprobe: number of object and probe modes
*/
void Object3DProbe2DMult(const int i, __global float2* psi, __global float2 *obj, __global float2* probe,
                         __global char* support, __global float* m, __global float* cx, __global float* cy,
                         __global int* cixo, __global int* ciyo, __global float* dsx, __global float* dsy,
                         __global float* dsz, const float pxo, const float pyo, const float pzo, const float pxp,
                         const float pyp, const float f, const int npsi, const int stack_size, const int nx,
                         const int ny, const int nxo, const int nyo, const int nzo, const int nxp, const int nyp,
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
            // Let's hope the probe array cache is efficient !
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
        psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy] = ps1;
      }
    }
  }
}

/** Same as Object3DProbe2DMult but store 3D arrays of object, probe and Psi before integration along Z
*/
void Object3DProbe2DMultDebug(const int i, __global float2* psi, __global float2 *obj, __global float2* probe,
                         __global char* support, __global float* m, __global float2* psi3d, __global float2 *obj3d,
                         __global float2* probe3d, __global float* cx, __global float* cy, __global int* cixo,
                         __global int* ciyo, __global float* dsx, __global float* dsy, __global float* dsz,
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
            // Let's hope the probe array cache is efficient !
            const float2 p = interp_probe(probe, m, iprobe, cx[j], cy[j], pxp, pyp, ixo-nxo/2, iyo-nyo /2, prz-nzo/2, nxp, nyp);

            const int i3d = prx + nx * (pry + ny * (prz + nzo * (j + stack_size * (iprobe + nbprobe * iobjmode))));
            probe3d[i3d] = p;

            const float sup = (float)support[ixo + nxo * (iyo + nyo * prz)] / 100.0f;

            const float2 o = obj[ixo + nxo * (iyo + nyo * prz) + iobjmode * nxyzo] * sup;
            obj3d[i3d] = o;
            float2 ps=(float2)(o.x * p.x - o.y * p.y , o.x * p.y + o.y * p.x);

            // Add the phase factor with the quadratic and multi-angle terms
            const float tmp = tmp_dsxy + dszj * prz;
            const float s=native_sin(tmp);
            const float c=native_cos(tmp);

            psi3d[i3d] = (float2)(ps.x * c - ps.y * s , ps.y * c + ps.x * s);

            ps1 += (float2)(ps.x * c - ps.y * s , ps.y * c + ps.x * s);
          }
        }
        psi[ipsi + (j + stack_size * (iprobe + iobjmode * nbprobe) ) * nxy] = ps1;
      }
    }
  }
}
