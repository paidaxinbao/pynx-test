__device__ void QuadPhaseMult(const int i, float2 *d, const float f, const float scale, const int nx, const int ny)
{
  const int ix = i % nx;
  const int iy = (i % (nx*ny)) / nx;

  // Assume ny, nx are multiples of 2. Compute phase factor for an array with its origin at 0
  const float y = (iy - (int)ny *(int)(iy>=((int)ny/2))) ;
  const float x = (ix - (int)nx *(int)(ix>=((int)nx/2))) ;
  const float tmp = f * (x*x + y*y);
  // NOTE WARNING: if becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  float s,c;
  __sincosf(tmp , &s,&c);
  s*= scale;
  c*=scale;

  const float2 d1 = d[i];
  d[i] = make_float2(d1.x*c - d1.y*s , d1.y*c + d1.x*s);
}
