/*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/** Compute standard deviation between two arrays
*
* \param v1, v2: arrays to compare
* \param stddev: in return, array of standard deviation (same size as v1, v2)
* \param nb: the number of elements in the v1, v2 and v arrays
*
*/
__kernel
void std_dev_pair(const int i, __global float *v1, __global float *v2,
                  __global float* stddev, const int nb)
{
  const float v1i = v1[i];
  const float v2i = v2[i];
  float s = 0;
  for(int j=0;j<nb;j++)
  {
    // Accessing [(j+i)%nb] is a cheap way to coalesce memory transfers (??)
    const int ji = (j+i)%nb;
    const float dj = fabs(v1[ji]-v1i);
    const float dj_mean = (fabs(v2[ji]-v2i) + dj) * 0.5f;
    s += (dj-dj_mean) * (dj-dj_mean);
  }
  stddev[i] = sqrt(s / (nb-1));
}
