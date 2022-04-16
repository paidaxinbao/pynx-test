
/** Compute the regularization log-likelihood for a complex array:
* - for a single point in a 4D array (3D + modes).
* - points outside the support are not taken into account.
* - for a smoothing regularization either on complex values or the squared amplitude.
* Returns a float2 (llk_reg_amplitude_square, llk_reg_complex)
*/
float2 RegSmoothComplexSupportLLK(const int i, __global float2 *d, __global char* support,
                                   const int nx, const int ny, const int nz)
{
  const int nxyz = nx * ny * nz;
  const int i0 = i % nxyz;
  const int ix = i % nx;
  const int iz = i0 / (nx*ny);
  const int iy = (i0 - iz * nx * ny) / nx;

  const float2 v0=d[i];
  const float v02=dot(v0, v0);
  int nb_neighbour = 0;
  float da2 = 0;
  float dc2 = 0;

  if(support[i0 % nxyz] == 0) return (float2)(0,0);

  if(ix>0)
  {
    if(support[(i-1) % nxyz])
    {
      const float2 v1=d[i-1];
      da2 += pown(v02 - dot(v1, v1),2);
      dc2 += dot(v0-v1, v0-v1);
      nb_neighbour +=1;
    }
  }
  if(ix<(nx-1))
  {
    if(support[(i+1) % nxyz])
    {
      const float2 v1=d[i+1];
      da2 += pown(v02 - dot(v1, v1),2);
      dc2 += dot(v0-v1, v0-v1);
      nb_neighbour +=1;
    }
  }
  if(iy>0)
  {
    if(support[(i-nx) % nxyz])
    {
      const float2 v1=d[i-nx];
      da2 += pown(v02 - dot(v1, v1),2);
      dc2 += dot(v0-v1, v0-v1);
      nb_neighbour +=1;
    }
  }
  if(iy<(ny-1))
  {
    if(support[(i+nx) % nxyz])
    {
      const float2 v1=d[i+nx];
      da2 += pown(v02 - dot(v1, v1),2);
      dc2 += dot(v0-v1, v0-v1);
      nb_neighbour +=1;
    }
  }
  if(iz>0)
  {
    if(support[(i-nx*ny) % nxyz])
    {
      const float2 v1=d[i-nx*ny];
      da2 += pown(v02 - dot(v1, v1),2);
      dc2 += dot(v0-v1, v0-v1);
      nb_neighbour +=1;
    }
  }
  if(iz<(nz-1))
  {
    if(support[(i+nx*ny) % nxyz])
    {
      const float2 v1=d[i+nx*ny];
      da2 += pown(v02 - dot(v1, v1),2);
      dc2 += dot(v0-v1, v0-v1);
      nb_neighbour +=1;
    }
  }

  if(nb_neighbour)
  {
    da2 /= nb_neighbour;
    dc2 /= nb_neighbour;
  }
  return (float2)(da2, dc2);
}


/** Compute the regularization gradient for a complex array:
* - for a single point in a 4D array (3D + modes).
* - points outside the support are not taken into account.
* - for a smoothing regularization either on complex values or the squared amplitude.
* The gradient conjugate is added to the gradient array.
*/
void RegSmoothComplexSupportGrad(const int i, __global float2 *d, __global float2 *grad, __global char* support,
                                   const int nx, const int ny, const int nz,
                                   const float reg_fac_modulus2, const float reg_fac_complex)
{
  const int nxyz = nx * ny * nz;
  const int i0 = i % nxyz;
  const int ix = i % nx;
  const int iz = i0 / (nx*ny);
  const int iy = (i0 - iz * nx * ny) / nx;

  const float2 v0 = d[i];
  const float v02 = dot(v0, v0);
  int nb_neighbour = 0;
  float2 dc = 0;
  float da2 = 0;

  if(support[i0 % nxyz] == 0) return;

  if(ix>0)
  {
    if(support[(i-1) % nxyz])
    {
      const float2 v1 = d[i-1];
      const float2 dv= v0 - v1;
      dc += (float2)(dv.x, -dv.y);
      da2 += v02 - dot(v1, v1);
      nb_neighbour +=1;
    }
  }
  if(ix<(nx-1))
  {
    if(support[(i+1) % nxyz])
    {
      const float2 v1=d[i+1];
      const float2 dv= v0 - v1;
      dc += (float2)(dv.x, -dv.y);
      da2 += v02 - dot(v1, v1);
      nb_neighbour +=1;
    }
  }
  if(iy>0)
  {
    if(support[(i-nx) % nxyz])
    {
      const float2 v1=d[i-nx];
      const float2 dv= v0 - v1;
      dc += (float2)(dv.x, -dv.y);
      da2 += v02 - dot(v1, v1);
      nb_neighbour +=1;
    }
  }
  if(iy<(ny-1))
  {
    if(support[(i+nx) % nxyz])
    {
      const float2 v1=d[i+nx];
      const float2 dv= v0 - v1;
      dc += (float2)(dv.x, -dv.y);
      da2 += v02 - dot(v1, v1);
      nb_neighbour +=1;
    }
  }
  if(iz>0)
  {
    if(support[(i-nx*ny) % nxyz])
    {
      const float2 v1=d[i-nx*ny];
      const float2 dv= v0 - v1;
      dc += (float2)(dv.x, -dv.y);
      da2 += v02 - dot(v1, v1);
      nb_neighbour +=1;
    }
  }
  if(iz<(nz-1))
  {
    if(support[(i+nx*ny) % nxyz])
    {
      const float2 v1=d[i+nx*ny];
      const float2 dv= v0 - v1;
      dc += (float2)(dv.x, -dv.y);
      da2 += v02 - dot(v1, v1);
      nb_neighbour +=1;
    }
  }

  if(nb_neighbour > 0)
  {
    grad[i] += 2 * (reg_fac_complex * dc + reg_fac_modulus2 * da2 * (float2)(v0.x, -v0.y)) / nb_neighbour;
  }
}


/** Compute the linear minimization gamma factors for a complex array:
* - for a single point in a 4D array (3D + modes).
* - points outside the support are not taken into account.
* - for a smoothing regularization either on complex values or the squared amplitude.
*
* The returned value is a float4 with (numerator_amp, denominator_amp, numerator_cplx, denominator_cplx) terms,
* allowing to distinguish the gamma terms for squared amplitude and complex values regularization.
*/
float4 RegSmoothComplexSupportGamma(const int i, __global float2 *d, __global float2 *dir, __global char* support,
                                      const int nx, const int ny, const int nz)
{
  const int nxyz = nx * ny * nz;
  const int i0 = i % nxyz;
  const int ix = i % nx;
  const int iz = i0 / (nx*ny);
  const int iy = (i0 - iz * nx * ny) / nx;

  const float2 v0=d[i];
  const float v02 = dot(v0, v0);
  const float2 d0=dir[i];
  const float d02 = dot(d0, d0);
  int nb_neighbour = 0;
  // R(d + gamma*dir) = A * gamma**2 + B * gamma + Cte
  // values for complex (c) and amplitude**2 (a) regularization
  float cb = 0;
  float ca = 0;
  float ab = 0;
  float aa = 0;

  if(support[i0 % nxyz] == 0) return (float4)(0,0,0,0);

  if(ix>0)
  {
    if(support[(i-1) % nxyz])
    {
      const float2 v1 = d[i-1];
      const float2 d1 = dir[i-1];

      const float2 dv = v0 - v1;
      const float2 dd = d0 - d1;
      const float v12 = dot(v1, v1);
      const float d12 = dot(d1, d1);

      cb += 2 * (dv.x * dd.x + dv.y * dd.y);
      ca += dot(dd, dd);

      const float odo = v0.x * d0.x + v0.y * d0.y - (v1.x * d1.x + v1.y * d1.y);
      ab += 4 * (v02 - dot(v1, v1)) * odo;
      aa += 4 * odo * odo + 2 * (v02 - v12) * (d02 - d12);

      nb_neighbour +=1;
    }
  }
  if(ix<(nx-1))
  {
    if(support[(i+1) % nxyz])
    {
      const float2 v1=d[i+1];
      const float2 d1=dir[i+1];

      const float2 dv = v0 - v1;
      const float2 dd = d0 - d1;
      const float v12 = dot(v1, v1);
      const float d12 = dot(d1, d1);

      cb += 2 * (dv.x * dd.x + dv.y * dd.y);
      ca += dot(dd, dd);

      const float odo = v0.x * d0.x + v0.y * d0.y - (v1.x * d1.x + v1.y * d1.y);
      ab += 4 * (v02 - dot(v1, v1)) * odo;
      aa += 4 * odo * odo + 2 * (v02 - v12) * (d02 - d12);

      nb_neighbour +=1;
    }
  }
  if(iy>0)
  {
    if(support[(i-nx) % nxyz])
    {
      const float2 v1=d[i-nx];
      const float2 d1=dir[i-nx];

      const float2 dv = v0 - v1;
      const float2 dd = d0 - d1;
      const float v12 = dot(v1, v1);
      const float d12 = dot(d1, d1);

      cb += 2 * (dv.x * dd.x -dv.y * dd.y);
      ca += dot(dd, dd);

      const float odo = v0.x * d0.x + v0.y * d0.y - (v1.x * d1.x + v1.y * d1.y);
      ab += 4 * (v02 - dot(v1, v1)) * odo;
      aa += 4 * odo * odo + 2 * (v02 - v12) * (d02 - d12);

      nb_neighbour +=1;
    }
  }
  if(iy<(ny-1))
  {
    if(support[(i+nx) % nxyz])
    {
      const float2 v1=d[i+nx];
      const float2 d1=dir[i+nx];

      const float2 dv = v0 - v1;
      const float2 dd = d0 - d1;
      const float v12 = dot(v1, v1);
      const float d12 = dot(d1, d1);

      cb += 2 * (dv.x * dd.x + dv.y * dd.y);
      ca += dot(dd, dd);

      const float odo = v0.x * d0.x + v0.y * d0.y - (v1.x * d1.x + v1.y * d1.y);
      ab += 4 * (v02 - dot(v1, v1)) * odo;
      aa += 4 * odo * odo + 2 * (v02 - v12) * (d02 - d12);

      nb_neighbour +=1;
    }
  }
  if(iz>0)
  {
    if(support[(i-nx*ny) % nxyz])
    {
      const float2 v1=d[i-nx*ny];
      const float2 d1=dir[i-nx*ny];

      const float2 dv = v0 - v1;
      const float2 dd = d0 - d1;
      const float v12 = dot(v1, v1);
      const float d12 = dot(d1, d1);

      cb += 2 * (dv.x * dd.x + dv.y * dd.y);
      ca += dot(dd, dd);

      const float odo = v0.x * d0.x + v0.y * d0.y - (v1.x * d1.x + v1.y * d1.y);
      ab += 4 * (v02 - dot(v1, v1)) * odo;
      aa += 4 * odo * odo + 2 * (v02 - v12) * (d02 - d12);

      nb_neighbour +=1;
    }
  }
  if(iz<(nz-1))
  {
    if(support[(i+nx*ny) % nxyz])
    {
      const float2 v1=d[i+nx*ny];
      const float2 d1=dir[i+nx*ny];

      const float2 dv = v0 - v1;
      const float2 dd = d0 - d1;
      const float v12 = dot(v1, v1);
      const float d12 = dot(d1, d1);

      cb += 2 * (dv.x * dd.x + dv.y * dd.y);
      ca += dot(dd, dd);

      const float odo = v0.x * d0.x + v0.y * d0.y - (v1.x * d1.x + v1.y * d1.y);
      ab += 4 * (v02 - dot(v1, v1)) * odo;
      aa += 4 * odo * odo + 2 * (v02 - v12) * (d02 - d12);

      nb_neighbour +=1;
    }
  }

  if(nb_neighbour)
  {
    return (float4)(-ab, 2 * aa, -cb, 2 * ca) / nb_neighbour;
  }
  return (float4)(0, 0, 0, 0);
}
