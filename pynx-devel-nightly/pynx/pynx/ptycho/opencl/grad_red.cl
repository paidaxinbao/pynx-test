/** Calculate Psi.conj() * (1 - Iobs / Icalc), for the gradient calculation with Poisson noise.
* Masked pixels are set to zero.
* This version also returns the calculated intensity for reduction, to later update the scale of floating intensities.
* This must be called for a 2D frame as the first argument, and looping will be done to go through the entire
* stack of frames.
* \param i: the point in the 3D observed intensity array for which the llk is calculated
* \param iobs: the observed in tensity array, shape=(stack_size, ny, nx)
* \param psi: the calculated complex amplitude, shape=(nb_obj, nb_probe, stack_size, ny, nx)
* \param background: the incoherent background, of shape (ny, nx)
* \param nbmode: number of modes = nb_probe * nb_obj
* \param nxystack: number of frames in stack multiplied by nx * ny
* \return: a float4 vector with (poisson llk, gaussian llk, euclidian llk, icalc)
*/
float_n GradPoissonFourierRed(const int ii, __global float *iobs, __global float2 *psi, __global float *background,
                              const int nbmode, const int nx, const int ny, const int nxystack, const int npsi,
                              const char hann_filter, __global float* scale)
{
  float g=1;
  if(hann_filter > 0)
  {
    // Use a Hann window multiplication to dampen high-frequencies in the object
    const int prx = ii % nx;
    const int pry = ii / nx;
    const float qx = (float)(prx - nx/2 + nx * (prx<(nx/2))) / (float)nx;
    const float qy = (float)(pry - ny/2 + ny * (pry<(ny/2))) / (float)ny;
    g = pown(native_cos(qx) * native_cos(qy),2);
  }

  float_n icalc;
  for(int j=0;j<npsi;j++)
  {
    const int i = ii + j * nx * ny;
    const float obs= iobs[i];

    if(obs < 0) icalc.v[j] = 0.0f;
    else
    {
      float calc = 0;
      for(int imode=0;imode<nbmode;imode++) calc += dot(psi[i + imode * nxystack],psi[i + imode* nxystack]);

      // For the gradient multiply by scale and not sqrt(scale)
      const float f = g * (1 - obs/ (fmax(1e-12f,calc) + background[i % (nx * ny)])) * scale[j];

      for(int imode=0; imode < nbmode; imode++)
      {
        // TODO: store psi to avoid double-read. Or just assume it's cached.
        const float2 ps = psi[i + imode * nxystack];
        psi[i + imode * nxystack] = (float2) (f*ps.x , f*ps.y);
      }
      icalc.v[j] = calc;
    }
  }
  return icalc;
}
