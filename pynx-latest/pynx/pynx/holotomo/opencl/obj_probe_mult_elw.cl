/** Multiply object and probe to get a Psi to be propagated.
* Object has 4 dimensions: projections, modes, y, x.
* Probe has 4 dimensions: modes, z (propagation distances), y, x.
* The 6-dimensional Psi stack is calculated, with dimensions:
*    projections, propagation distances, object modes, probe modes, y, x
*
* This kernel function should be called for a 2D array, but will be applied to all z, projections and modes.
*
* This version assumes the probe is the same for all z (propagation distances), but with different shifts
*/
void ObjectProbeMult(const int i, __global float2* obj, __global float2 *probe, __global float2* psi,
                     __global int *dxi, __global int *dyi, __global char *sample_flag,
                     const int nb_proj, const int nb_z, const int nb_obj, const int nb_probe,
                     const int nx, const int ny)
{
  // Coordinates in object array (origin at array center)
  const int ix = i % nx;
  const int iy = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  for(int iobj=0; iobj<nb_obj; iobj++)
  {
    for(int iprobe=0; iprobe<nb_probe; iprobe++)
    {
      for(int iproj=0;iproj<nb_proj;iproj++)
      {
        const float2 o = obj[i + nx * ny * (iobj + nb_obj * iproj)];
        for(int iz=0;iz<nb_z;iz++)
        {
          // TODO: handle object/probe relative shifts
          // Coordinates in probe array so that array is centered if dx=dy=0
          // const int ixp = ix + dxi[iz + nb_z * iproj] + (nx_probe - nx) / 2;
          // const int iyp = iy + dyi[iz + nb_z * iproj] + (ny_probe - ny) / 2;

          const float2 p = probe[ix + nx_probe * (iy + ny_probe * iprobe)];

          const int i1 = ipsi + nx * ny * (iprobe + nb_probe * (iobj + nb_obj * (iz + nb_z * iproj)));
          if(sample_flag[iobj]>0)
          {
            psi[i1] = (float2)(o.x * p.x -o.y * p.y, o.x * p.y + o.y * p.x);
          }
          else
          {
            psi[i1] = p;
          }
        }
      }
    }
  }
}


/** Multiply object and probe to get a Psi to be propagated.
* Object has 4 dimensions: projections, modes, y, x.
* Probe has 4 dimensions: modes, z (propagation distances), y, x.
* The 6-dimensional Psi stack is calculated, with dimensions:
*    projections, propagation distances, object modes, probe modes, y, x
*
* This kernel function should be called for a 2D array, but will be applied to all z, projections and modes.
*
* This version assumes the probe is different for each z (propagation distances)
*/
void ObjectProbeZMult(const int i, __global float2* obj, __global float2 *probe, __global float2* psi,
                      __global int *dxi, __global int *dyi, __global char *sample_flag,
                      const int nb_proj, const int nb_z, const int nb_obj, const int nb_probe,
                      const int nx, const int ny)
{
  // Coordinates in object array (origin at array center)
  const int ix = i % nx;
  const int iy = i / nx;

  // Coordinates in Psi array (origin at (0,0)). Assumes nx ny are multiple of 2
  const int iy1 = iy - ny/2 + ny * (iy<(ny/2));
  const int ix1 = ix - nx/2 + nx * (ix<(nx/2));
  const int ipsi  = ix1 + iy1 * nx ;

  for(int iobj=0; iobj<nb_obj; iobj++)
  {
    for(int iprobe=0; iprobe<nb_probe; iprobe++)
    {
      for(int iproj=0;iproj<nb_proj;iproj++)
      {
        const float2 o = obj[i + nx * ny * (iobj + nb_obj * iproj)];
        for(int iz=0;iz<nb_z;iz++)
        {
          // TODO: handle object/probe relative shifts
          // Coordinates in probe array so that array is centered if dx=dy=0
          //const int ixp = ix + dxi[iz + nb_z * iproj] + (nx_probe - nx) / 2;
          //const int iyp = iy + dyi[iz + nb_z * iproj] + (ny_probe - ny) / 2;

          const float2 p = probe[   ixp + nx * (iyp + ny * (iprobe + nb_probe * iz))];

          const int i1 = ipsi + nx * ny * (iprobe + nb_probe * (iobj + nb_obj * (iz + nb_z * iproj)));
          if(sample_flag[iobj]>0)
          {
            psi[i1] = (float2)(o.x * p.x -o.y * p.y, o.x * p.y + o.y * p.x);
          }
          else
          {
            psi[i1] = p;
          }
        }
      }
    }
  }
}
