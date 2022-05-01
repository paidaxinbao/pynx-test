/** This can be used to compute the linear combination of modes, i.e. it does
* a simple linear multiplication:
* np.array([sum(d[i] * v[i, j] for i in range(len(d))) for j in range(len(d))])
*
* This should be called as an elementwise kernel with the size (nxy) of the first mode,
* after replacing N by the number of modes.
*/
__device__ void ortho_dot(const unsigned int i, complexf *d, complexf *v, const unsigned int nxy)
{
  complexf d0[%(N)d];
  complexf d1[%(N)d];
  for(unsigned int j=0 ; j < %(N)d; j++) d0[j] = d[i + nxy * j];

  for(unsigned int j=0 ; j < %(N)d; j++)
  {
    d1[j] = 0;
    for(unsigned int k=0 ; k < %(N)d; k++) d1[j] += v[k + %(N)d * j] * d0[k];
  }
  for(unsigned int j=0; j < %(N)d; j++) d[i + nxy * j] = d1[j];
}


/** This can be used to compute the linear combination of modes, i.e. it does
* a simple linear multiplication:
* np.array([sum(d[i] * v[i, j] for i in range(len(d))) for j in range(len(d))])
*
* This should be called as an reduction kernel with the size (nxy) of the first mode,
* after replacing N by the number of modes. This will return the norm of the
* computed elements for reduction.
*/
__device__ float_%(N)d ortho_dot_red(const unsigned int i, complexf *d, complexf *v, const unsigned int nxy)
{
  complexf d0[%(N)d];
  complexf d1[%(N)d];
  for(unsigned int j=0;j < %(N)d; j++) d0[j] = d[i + nxy * j];

  for(unsigned int j=0;j < %(N)d; j++)
  {
    d1[j] = 0;
    for(unsigned int k=0; k < %(N)d; k++) d1[j] += v[k + %(N)d * j] * d0[k];
  }

  float_%(N)d n;
  for(unsigned int j=0; j < %(N)d; j++)
  {
    d[i + nxy * j] = d1[j];
    n[j] = norm(d1[j]); // norm() = square modulus
  }
  return n;
}

/** Sorting of modes.
*
* \param d: the array with N modes, each of size nxy
* \param n: the norm array of size N
*/
__device__ void ortho_sort(const unsigned int i, complexf *d, const float *dnorm, const unsigned int nxy)
{
  unsigned int idx[%(N)d];
  bool sorted[%(N)d];
  for(unsigned int j=0; j < %(N)d; j++) sorted[j] = false;
  // Order by norm. *Very* dumb sort, but as N is small this should be irrelevant
  for(unsigned int j=0; j < %(N)d; j++)
  {
    float max = 0;
    for(unsigned int k =0; k < %(N)d; k++)
    {
      if( (dnorm[k]>max) && !sorted[k])
      {
        idx[j] = k;
        max = dnorm[k];
      }
    }
    sorted[idx[j]] = true;
  }
  /*
  if(i==0)
  {
    for(int j=0; j < %(N)d; j++) printf("%%12.3f ", dnorm[j]);
    printf("\n");
    for(int j=0; j < %(N)d; j++) printf("%%d ", idx[j]);
    printf("\n");
    for(int j=0; j < %(N)d; j++) printf("%%12.3f ", dnorm[idx[j]]);
    printf("\n");
  }*/

  // Order the array accordingly
  complexf v[%(N)d];
  for(unsigned int j=0; j < %(N)d; j++) v[j] = d[i + nxy * j];
  for(unsigned int j=0; j < %(N)d; j++) d[i + nxy * j] = v[idx[j]];
}

/** Normalise modes.
*
* \param d: the array with N modes, each of size nxy
* \param n: the norm (sum of square modulus for all elements), array of size N
*/
__device__ void ortho_norm(const unsigned int i, complexf *d, const float *dnorm, const unsigned int nxy)
{
  for(unsigned int j=0; j < %(N)d; j++) d[i + nxy * j] *= sqrt(nxy / dnorm[j]);
}
