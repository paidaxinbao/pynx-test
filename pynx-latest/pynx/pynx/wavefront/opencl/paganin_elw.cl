void paganin_transfer_function(const int i, __global float2 *d, const float z_delta, const float mu, const float dk, const int nx, const int ny)
{
  const int ix = i % nx;
  const int iy = (i % (nx*ny)) / nx;

  // Assume ny, nx are multiples of 2. Compute phase factor for an array with its origin at 0
  const float y = (iy - (int)ny *(int)(iy>=((int)ny/2))) ;
  const float x = (ix - (int)nx *(int)(ix>=((int)nx/2))) ;
  const float mul = mu / (z_delta * dk * dk * (x*x + y*y) + mu);
  d[i] = mul * d[i];
}

void paganin_thickness_wavefront(const int i, __global float2 *d, const float mu, const float k_delta)
{
    const float t = -log(length(d[i]))/mu;

    // Use approximations if absorption or phase shift is small
    float a = mu * t;
    if(a < 1e-4)
        a = 1 - a * 0.5 ;
    else
        a = exp(-0.5f*a);

    const float alpha = k_delta*t;
    if(alpha<1e-4)
        d[i] = (float2)(a * (1-alpha*alpha), -a * alpha);
    else
        d[i] = (float2)(a * cos(alpha), a * sin(-alpha));
}
