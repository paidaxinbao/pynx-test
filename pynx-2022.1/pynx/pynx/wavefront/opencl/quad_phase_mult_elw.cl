void QuadPhaseMult(const int i, __global float2 *d, const float f, const float scale, const int nx, const int ny)
{
  const int ix = i % nx;
  const int iy = (i % (nx*ny)) / nx;

  // Assume ny, nx are multiples of 2. Compute phase factor for an array with its origin at 0
  const float y = (iy - (int)ny *(int)(iy>=((int)ny/2))) ;
  const float x = (ix - (int)nx *(int)(ix>=((int)nx/2))) ;
  const float tmp = f * (x*x + y*y);
  // NOTE WARNING: if becomes large (e.g. > 2^15, depending on implementation), native sin and cos may be wrong.
  const float s = native_sin(tmp);
  const float c = native_cos(tmp);
  //if(iy==1 && ix==0) printf("CL (%4d,%4d) pixel size = (%g, %g) f=%g, s=%f c=%f nx=%d ny=%d\\n", ix, iy, px, py, f, s, c, NX, NY);
  // d is interleaved complex data array
  const float2 d1 = d[i];
  d[i] = scale * (float2) (d1.x*c - d1.y*s , d1.y*c + d1.x*s);
}
