/** Reduction kernel function :
* compute the log-likelihood regularization term penalizing local density variations in a complex array (object or probe).
*/
float LLKReg(__global float2 *v, const int i, const int nx, const int ny)
{
  const int x=i%nx;
  const int y=i/nx;
  const int y0=y%ny; // For multiple modes, to see if we are near a border
  float llk=0;
  const float2 v0=v[i];

  // The 4 cases could be put in a loop for simplicity (but not performance)
  if(x>0)
  {
    const float2 v1=v[i-1];
    llk += pown(v0.x-v1.x,2) + pown(v0.y-v1.y,2);
  }

  if(x<(nx-1))
  {
    const float2 v1=v[i+1];
    llk += pown(v0.x-v1.x,2) + pown(v0.y-v1.y,2);
  }

  if(y0>0)
  {
    const float2 v1=v[i-nx];
    llk += pown(v0.x-v1.x,2) + pown(v0.y-v1.y,2);
  }

  if(y0<(ny-1))
  {
    const float2 v1=v[i+nx];
    llk += pown(v0.x-v1.x,2) + pown(v0.y-v1.y,2);
  }
  return llk;
}
