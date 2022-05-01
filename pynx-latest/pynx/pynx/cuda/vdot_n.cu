// This computes [[np.vdot(p2, p1) for p1 in m] for p2 in m]
// N2 must be replaced by n**2 and N by n, where n is the
// number of
__device__ complexf_%(N2)d vdot(unsigned int i, complexf* d, const unsigned int nxy)
{
  complexf_%(N2)d v;
  for(unsigned int j=0;j<%(N)d;j++)
  {
    const complexf v1 = d[i + j * nxy];
    // Diagonal term
    v[j * (1+%(N)d)] = v1*conj(v1);

    for(unsigned int k=0;k<j;k++)
    {
      const complexf v2 = d[i + k * nxy] * conj(v1);
      // Hermitian matrix
      v[j + %(N)d * k] = v2;
      v[k + %(N)d * j] = conj(v2);
    }
  }
  return v;
}
