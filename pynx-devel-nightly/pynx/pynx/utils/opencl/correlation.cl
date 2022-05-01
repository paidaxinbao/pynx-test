/*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Compute shell correlation between two arrays
*
* \param v1, v2: arrays to compare
* \param shell_x: edge coordinates for the limits of each shell (nb_shell+1 elements)
* \param shell_y: in return, the correlation (between 0 and 1) between the two arrays
*                (nb_shell elements), before normalisation
* \param shell_n1: normalisation 1 for shell_y
* \param shell_n2: normalisation 2 for shell_y
* \param shell_nb: in return, the number of pair of points for the correlation, for each
*               shell (nb_shell elements)
* \param nb: the number of elements in the v1, v2 and v arrays
¥ \param shell_map: if true, shell_v_y and shell_v_nb will be computed.
* \param shell_v_y: array of shape (nb_shell, nb) where the per-pixel correlation for array 1 will be stored,
*                   unless shell_map is false.
* \param shell_v_nb: array of shape (nb_shell, nb) where the number of points used to compute shell_v_y is stored,
*                   unless shell_map is false.
*
* \note: in this kernel NB_SHELL must be replaced by the actual number of shells used.
*/
__kernel
void shell_correl(const int i, __global float *v1, __global float *v2,
                  __global float* shell_x, __global float* shell_y,
                  __global float* shell_n1, __global float* shell_n2,
                  __global int* shell_nb, const int nb, const char shell_map,
                  __global float* shell_v_y, __global int* shell_v_nb)
{
  const float v1i = v1[i];
  const float v2i = v2[i];

  float vn1[NB_SHELL];
  float vn2[NB_SHELL];
  float vy[NB_SHELL];
  float vnb[NB_SHELL];
  for(int k=0;k<NB_SHELL;k++)
  {
    vn1[k] = 0;
    vn2[k] = 0;
    vy[k] = 0;
    vnb[k] = 0;
  }

  for(int j=0;j<nb;j++)
  {
    // Accessing [(j+i)%nb] is a cheap way to coalesce memory transfers (??)
    const int ji = (j+i)%nb;
    const float d1j = fabs(v1[ji]-v1i);
    const float d2j = fabs(v2[ji]-v2i);
    for(int k=0;k<NB_SHELL;k++)
    {
      // Accessing [(k+i)%NB_SHELL] is a cheap way to coalesce memory transfers (??)
      const int ki = (k+i)%NB_SHELL;
      const float s1=shell_x[ki];
      const float s2=shell_x[ki + 1];
      if((d1j>=s1) && (d1j<s2))
      {
        vy[ki]  += d1j * d2j;
        vnb[ki] += 1;
        vn1[ki] += d1j * d1j;
        vn2[ki] += d2j * d2j;
      }
      if((d2j>=s1) && (d2j<s2))
      {
        vy[ki]  += d1j * d2j;
        vnb[ki] += 1;
        vn1[ki] += d1j * d1j;
        vn2[ki] += d2j * d2j;
      }
    }
  }
  for(int k=0;k<NB_SHELL;k++)
  {
    atomic_add_f(&shell_y[k], vy[k]);
    atomic_add_f(&shell_n1[k], vn1[k]);
    atomic_add_f(&shell_n2[k], vn2[k]);
    atomic_add(&shell_nb[k], vnb[k]);
    if(shell_map)
    {
      const float tmp = native_sqrt(vn1[k]*vn2[k]);
      shell_v_y[k * nb + i] = vy[k] /(tmp + (tmp == 0) * 1e-20);
      shell_v_nb[k * nb + i] = vnb[k];
    }
  }
}

/** Return the phase difference of two complex values
*/
inline float phase_diff(float2 a, float2 b)
{
  const float re = (a.x * b.x + a.y * b.y) / native_sqrt(b.x * b.x + b.y * b.y);
  const float im = (-a.x * b.y + a.y * b.x) / native_sqrt(b.x * b.x + b.y * b.y);
  return atan2(im, re);
}

/** Compute shell correlation between the phase of two complex arrays
*
* \param v1, v2: arrays to compare
* \param shell_x: edge coordinates for the limits of each shell (nb_shell+1 elements)
* \param shell_y: in return, the correlation (between 0 and 1) between the two arrays
*                (nb_shell elements), before normalisation
* \param shell_n1: normalisation 1 for shell_y
* \param shell_n2: normalisation 2 for shell_y
* \param shell_nb: in return, the number of pair of points for the correlation, for each
*               shell (nb_shell elements)
* \param nb: the number of elements in the v1, v2 and v arrays
¥ \param shell_map: if true, shell_v_y and shell_v_nb will be computed.
* \param shell_v_y: array of shape (nb_shell, nb) where the per-pixel correlation for array 1 will be stored,
*                   unless shell_map is false.
* \param shell_v_nb: array of shape (nb_shell, nb) where the number of points used to compute shell_v_y is stored,
*                   unless shell_map is false.
*
* \note: in this kernel NB_SHELL must be replaced by the actual number of shells used.
*/
__kernel
void shell_correl_phase(const int i, __global float2 *v1, __global float2 *v2,
                  __global float* shell_x, __global float* shell_y,
                  __global float* shell_n1, __global float* shell_n2,
                  __global int* shell_nb, const int nb, const char shell_map,
                  __global float* shell_v_y, __global int* shell_v_nb)
{
  const float2 v1i = v1[i];
  const float2 v2i = v2[i];
  float vn1[NB_SHELL];
  float vn2[NB_SHELL];
  float vy[NB_SHELL];
  float vnb[NB_SHELL];
  for(int k=0;k<NB_SHELL;k++)
  {
    vn1[k] = 0;
    vn2[k] = 0;
    vy[k] = 0;
    vnb[k] = 0;
  }

  for(int j=0;j<nb;j++)
  {
    // Accessing [(j+i)%nb] is a cheap way to coalesce memory transfers (??)
    const int ji = (j+i)%nb;
    const float d1j = fabs(phase_diff(v1[ji], v1i));
    const float d2j = fabs(phase_diff(v2[ji], v2i));
    for(int k=0;k<NB_SHELL;k++)
    {
      // Accessing [(k+i)%NB_SHELL] is a cheap way to coalesce memory transfers (??)
      const int ki = (k+i)%NB_SHELL;
      const float s1=shell_x[ki];
      const float s2=shell_x[ki + 1];
      if((d1j>=s1) && (d1j<s2))
      {
        vy[ki]  += d1j * d2j;
        vnb[ki] += 1;
        vn1[ki] += d1j * d1j;
        vn2[ki] += d2j * d2j;
      }
      if((d2j>=s1) && (d2j<s2))
      {
        vy[ki]  += d1j * d2j;
        vnb[ki] += 1;
        vn1[ki] += d1j * d1j;
        vn2[ki] += d2j * d2j;
      }
    }
  }
  for(int k=0;k<NB_SHELL;k++)
  {
    atomic_add_f(&shell_y[k], vy[k]);
    atomic_add_f(&shell_n1[k], vn1[k]);
    atomic_add_f(&shell_n2[k], vn2[k]);
    atomic_add(&shell_nb[k], vnb[k]);
    if(shell_map)
    {
      const float tmp = native_sqrt(vn1[k]*vn2[k]);
      shell_v_y[k * nb + i] = vy[k] /(tmp + (tmp == 0) * 1e-20);
      shell_v_nb[k * nb + i] = vnb[k];
    }
  }
}
