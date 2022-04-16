void CircularMask(const int i, __global float2 *d, const float radius, const float pixel_size, const char invert, const int nx, const int ny)
{
  const int ix = i % nx;
  const int iy = (i % (nx*ny)) / nx;

  // Assume ny, nx are multiples of 2.
  const float y = (iy - (int)ny *(int)(iy>=((int)ny/2))) * pixel_size ;
  const float x = (ix - (int)nx *(int)(ix>=((int)nx/2))) * pixel_size;
  if(invert)
  {
    if( (x*x + y*y) <= (radius*radius)) d[i]=(float2)(0,0);
  }
  else
  {
      if( (x*x + y*y) > (radius*radius)) d[i]=(float2)(0,0);
  }
}

void RectangularMask(const int i, __global float2 *d, const float width, const float height, const float pixel_size, const char invert, const int nx, const int ny)
{
  const int ix = i % nx;
  const int iy = (i % (nx*ny)) / nx;

  // Assume ny, nx are multiples of 2.
  const float y = (iy - (int)ny *(int)(iy>=((int)ny/2))) * pixel_size ;
  const float x = (ix - (int)nx *(int)(ix>=((int)nx/2))) * pixel_size;
  if(invert)
  {
    if( (fabs(x) <= (width / 2)) && (fabs(y) <= (height / 2))) d[i]=(float2)(0,0);
  }
  else
  {
      if( (fabs(x) > (width / 2)) || (fabs(y) > (height / 2))) d[i]=(float2)(0,0);
  }
}
