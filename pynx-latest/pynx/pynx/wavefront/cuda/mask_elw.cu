__device__ void CircularMask(const int i, float2 *d, const float radius, const float pixel_size, const char invert, const int nx, const int ny)
{
  const int ix = i % nx;
  const int iy = (i % (nx*ny)) / nx;

  // Assume ny, nx are multiples of 2.
  const float y = (iy - (int)ny *(int)(iy>=((int)ny/2))) * pixel_size ;
  const float x = (ix - (int)nx *(int)(ix>=((int)nx/2))) * pixel_size;
  if(invert)
  {
    if( (x*x + y*y) <= (radius*radius)) d[i] = make_float2(0.0f,0.0f);
  }
  else
  {
      if( (x*x + y*y) > (radius*radius)) d[i] = make_float2(0.0f,0.0f);
  }
}

__device__ void RectangularMask(const int i, float2 *d, const float width, const float height, const float pixel_size, const char invert, const int nx, const int ny)
{
  const int ix = i % nx;
  const int iy = (i % (nx*ny)) / nx;

  // Assume ny, nx are multiples of 2.
  const float y = (iy - (int)ny *(int)(iy>=((int)ny/2))) * pixel_size ;
  const float x = (ix - (int)nx *(int)(ix>=((int)nx/2))) * pixel_size;
  if(invert)
  {
    if( (fabs(x) <= (width / 2)) && (fabs(y) <= (height / 2))) d[i] = make_float2(0.0f,0.0f);
  }
  else
  {
      if( (fabs(x) > (width / 2)) || (fabs(y) > (height / 2))) d[i] = make_float2(0.0f,0.0f);
  }
}
