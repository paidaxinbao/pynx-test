/** Tukey's biweight function.
* The function has a maximum for x=c/sqrt(5) with value c*16/25
*/
__device__ float tukey_biweight(const float x, const float c)
{
  if(fabs(x) >= c) return 0;
  const float v = 1 - x * x / (c*c);
  return x * v * v;
}

/** Tukey's biweight function, version working on the log10 of x.
* The function has a maximum for log10(x)=c/sqrt(5), so in this case the relative change of
* x can be up to a factor 10**(c*16/25)
*/
__device__ float tukey_biweight_log10(const float x, const float c)
{
  return exp10f(tukey_biweight(log10f(fmaxf(x,1e-30f)), c));
}
