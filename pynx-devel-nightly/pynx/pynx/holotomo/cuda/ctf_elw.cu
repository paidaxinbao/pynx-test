// #include "cuda_fp16.h"

#define twopi 6.2831853071795862f

// Compute the FT of the object's phase from the FT of the intensity at multiple distances
// This should be called for a single layer of psi (size ny*nx) and this will update
// all the projections' first mode with the FT of the object phase.
__device__ void ctf_fourier(const int i, complexf *psi, float* pilambdad, const float px,
                            const int nb_mode, const int nx, const int ny, const int nz,
                            const int nb_proj, const float alpha)
{
  // psi shape is (stack_size, nb_z, nb_obj, nb_probe, ny, nx)

  const int nxy = nx * ny;
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;

  // Assumes ny, nx are multiples of 2
  const float ky = (iy - (int)ny *(int)(iy >= ((int)ny / 2)))  / (px * (float)ny) ;
  const float kx = (ix - (int)nx *(int)(ix >= ((int)nx / 2)))  / (px * (float)nx) ;
  const float k2 =  kx * kx + ky * ky;

  float s,c;

  float aa=0;
  float bb=0;
  float cc=0;

  for(int iz=0; iz<nz; iz++)
  {
    __sincosf(pilambdad[iz] * k2, &s,&c);
    aa += s*c;
    bb += s*s;
    cc += c*c;
  }

  for(int iproj=0; iproj<nb_proj; iproj++)
  {
    complexf n=0;

    for(int iz=0; iz<nz; iz++)
    {
      __sincosf(pilambdad[iz] * k2, &s,&c);
      const complexf ps = psi[ix + nx * (iy + ny * nb_mode * (iz + nz * iproj))];
      n += cc * s * ps - aa * c * ps;
    }
    psi[ix + nx * (iy + ny * nb_mode * nz * iproj)] = n/(2 * (bb * cc - aa*aa) + alpha);
  }
}

/** Compute the FT of the object's phase from the FT of the intensity at multiple distances
* This should be called for a single layer of psi (size ny*nx) and this will update
* all the projections' first mode with the FT of the object phase.
* version assuming an homogeneous object, with given delta/beta
*
* \param psi: the Fourier transform of I/I0 - <I/I0> (subtracting the mean is
* equivalent to subtracting a Dirac peak in Fourier space)
* \param alpha_low, alpha_high: regularisation parameters for the low and high
* frequencies. The cutoff is set at the first zero of shortest distance denominator,
* when sin(pi*lambda*z*(kx**2+ky**2))=0 so for k = sqrt(1/lambda*z),
* using an erfc function for the transition with a sigma = 1% of Nyquist frequency
* \param sigma: parameter to change the width of the erfc transition
* Code to test the filter:
%matplotlib notebook
import numpy as np
from numpy.fft import *
from scipy.special import erfc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xraydb
nrj_ev=30000
material , density = xraydb.get_material('Fe')
delta,beta, atten = xraydb.xray_delta_beta(material, density, nrj_ev)
pixel_size = 50e-9
ny, nx = 1000, 1200
vz = np.array([0.03      , 0.03374592, 0.03795957, 0.04269935])[:1]
wavelength = 12398.4e-10/nrj_ev
kx = fftshift(fftfreq(nx)) / pixel_size
ky = fftshift(fftfreq(ny)) / pixel_size
kx,ky=np.meshgrid(kx, ky)
alpha_low, alpha_high = 1e-5, 0.2
ctf = kx*0 + alpha_low
for z in vz:
    ctf += (np.cos(np.pi*wavelength*z*(kx**2 + ky**2)) + delta/beta *np.sin(np.pi*wavelength*z*(kx**2 + ky**2)))**2

alpha_r = erfc((np.sqrt(kx**2+ky**2)-1/np.sqrt(wavelength*vz[-1]))/(0.01*kx.max()))

plt.figure(figsize=(12,5))
plt.subplot(131)
plt.imshow(abs(1/ctf), norm=LogNorm())
plt.colorbar()
plt.subplot(132)
plt.imshow(alpha_low * (alpha_r)/2 + alpha_high * (2-alpha_r)/2)
plt.colorbar()
plt.subplot(133)
plt.imshow(abs(1/(ctf + alpha_low * (alpha_r)/2 + alpha_high * (2-alpha_r)/2)), norm=LogNorm())
plt.colorbar()
*/
__device__ void ctf_fourier_homogeneous(const int i, complexf *psi, float* pilambdad,
                                        const float delta_beta,
                                        const float px, const int nb_mode, const int nx, const int ny,
                                        const int nz, const int nb_proj,
                                        const float alpha_low, const float alpha_high, const float sigma)
{
  // psi shape is (stack_size, nb_z, nb_obj, nb_probe, ny, nx)

  const int nxy = nx * ny;
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;

  // Assumes ny, nx are multiples of 2
  const float ky = (iy - (int)ny *(int)(iy >= ((int)ny / 2)))  / (px * (float)ny) ;
  const float kx = (ix - (int)nx *(int)(ix >= ((int)nx / 2)))  / (px * (float)nx) ;
  const float k2 =  kx * kx + ky * ky;

  // Get the max pilambdaz for the filter cutoff
  float lambdazmax=0;
  for(int iz=0; iz<nz; iz++)
    if(pilambdad[iz] > lambdazmax) lambdazmax = pilambdad[iz];
  lambdazmax /= 3.1415926f;

  float s,c;

  for(int iproj=0; iproj<nb_proj; iproj++)
  {

    complexf n=0;
    float d=0;

    const float alpha_r = erfcf((sqrtf(kx*kx + ky*ky)- 1 / sqrtf(lambdazmax)) / (sigma / px / 2));
    const float alpha = alpha_low * (alpha_r)/2 + alpha_high * (2-alpha_r) / 2;

    for(int iz=0; iz<nz; iz++)
    {
      __sincosf(pilambdad[iz] * k2, &s, &c);
      const complexf ps = psi[ix + nx * (iy + ny * nb_mode * (iz + nz * iproj))];
      n += (c + delta_beta * s) * ps;
      d += (c + delta_beta * s) * (c + delta_beta * s) + alpha;
    }
    psi[ix + nx * (iy + ny * nb_mode * nz * iproj)] = delta_beta / 2 * n / d;
  }
}

/** Convert the phase of the weak phase object to the complex transmission.
// This should be called for a single layer of psi (size ny*nx) and this will update
// all the object projections
*/
__device__ void ctf_phase2obj(const int i, complexf *psi, complexf *obj, float* obj_phase0,
                              const int nb_probe, const int nb_obj,
                              const int nx, const int ny, const int nz,
                              const int nb_proj, const float delta_beta)
{
  // psi shape is (stack_size, nb_z, nb_obj, nb_probe, ny, nx)
  // obj shape is (nb_proj, nb_obj, ny, nx)
  const int nxy = nx * ny;
  const int ix = i % nx;
  const int iy = (i % nxy) / nx;

  // fft-shift coordinates in object
  const int ixo = ix - nx / 2 + nx * (ix < (nx / 2));
  const int iyo = iy - ny / 2 + ny * (iy < (ny / 2));

  float s,c;

  for(int iproj=0; iproj<nb_proj; iproj++)
  {
    const float ph = -psi[ix + nx * (iy + ny * nb_obj * nb_probe * nz * iproj)].real();  // abs() or .real() ?
    // Coordinates of the first mode of object array
    // i[obj] = ix + nx * (iy + ny * (iobj + nobj * iproj))
    const int iobj0 = ixo + nx * (iyo + ny * nb_obj * iproj);

    float a = 0;
    if(delta_beta>0) a = ph / delta_beta;
    if(fabs(a) < 1e-4)
        a = 1 - a ;
    else
      a = expf(-a);

    if(fabs(ph)<1e-4)
      obj[iobj0] = complexf(a * (1-ph*ph), -a * ph);
    else
    {
      __sincosf(ph, &s, &c);
      obj[iobj0] = complexf(a * c, -a * s);
    }

    // Storing obj_phase0 should not be necessary - CTF assumes a weak phase object
    obj_phase0[ixo + nx * (iyo + ny * iproj)] = ph;

    // Set to 0 other object modes
    for(int iobj=1; iobj<nb_obj; iobj++)
      obj[iobj0 + nxy * iobj] = 0;
  }
}
