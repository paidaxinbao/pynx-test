inline __device__ float pow2(const float v)
{
  return v*v;
}

/** Reduction kernel function: compute the gamma value for conjugate gradient (coordinate of line minimization),
* for the calculated vs observed Poisson log-likelihood.
* Returns numerator and denominator of the coefficient in a complexf value.
*/
__device__ complexf CG_Poisson_Gamma(const int i, float *iobs, complexf *vPO, complexf *vPdO,
                                     complexf *vdPO, complexf *vdPdO, float* background, float* background_dir,
                                     const int nxy, const int nxystack, const int nbmode, const int npsi,
                                     const float scale)
{
  // Incoherent background
  const float backg = background[i % nxy];
  const float db = background_dir[i % nxy];

  complexf val(0,0);

  for(int j=0; j<npsi; j++)
  {
    const int ij = i + nxy * j;
    const float obs = iobs[ij]; // Observed intensity

    if(obs >=0)
    {
      float R_PO_OdP_Pdo = 0;
      float sumPO2 = 0;
      float OdP_PdO_R = 0;
      for(int imode=0;imode<nbmode;imode++)
      {
        const complexf PO   = vPO  [ij + imode*nxystack] * scale;
        const complexf PdO  = vPdO [ij + imode*nxystack] * scale;
        const complexf dPO  = vdPO [ij + imode*nxystack] * scale;
        const complexf dPdO = vdPdO[ij + imode*nxystack] * scale;
        const complexf a = dPO + PdO;
        R_PO_OdP_Pdo += PO.real() * a.real() + PO.imag() * a.imag();
        sumPO2 += dot(PO, PO);
        OdP_PdO_R += dot(dPO,dPO) + dot(PdO,PdO)
                     + 2*(PO.real() * dPdO.real() + PO.imag() * dPdO.imag() + PdO.real() * dPO.real() + PdO.real() * dPO.imag());
      }
      sumPO2 = fmaxf(sumPO2 + backg, 1e-32f);
      const float f = 1 - obs/sumPO2;

      // This is written to avoid overflows, not calculating (R_PO_OdP_Pdo*R_PO_OdP_Pdo)/(sumPO2*sumPO2)
      val+= complexf(-(2*R_PO_OdP_Pdo + db) * f,
                     2*(OdP_PdO_R * f + obs * (4*(R_PO_OdP_Pdo/sumPO2) * ((db + R_PO_OdP_Pdo) / sumPO2) + pow2(db/sumPO2)) / 2));
    }
  }
  return val;
}

/** Reduction kernel function: compute the gamma value for conjugate gradient (coordinate of line minimization),
* for the calculated vs observed Poisson log-likelihood.
* Returns the 4 coefficients for the polynomial LLK(gamma) = SUM(i=1,4)  A_i gamma**i
*/
__device__ my_float4 CG_Poisson_Gamma4(const int i, float *iobs, complexf *vPO, complexf *vPdO,
             complexf *vdPO, complexf *vdPdO, const int nxy, const int nxystack, const int nbmode,
             const float scale)
{
  // Scaling is used to avoid overflows
  const float obs = iobs[i]; // Observed intensity

  if(obs <0) return my_float4(0);  // Masked pixel

  // TODO: take into account background and background direction
  const float backg = 0; // background[i%nxy];
  // const float db = 0; //dbackg;

  float a1 = 0;
  float sumPO2 = 0;
  float a2 = 0;
  float a3 = 0;
  float a4 = 0;
  for(int imode=0;imode<nbmode;imode++)
  {
    const complexf PO   = vPO  [i + imode*nxystack] * scale;
    const complexf PdO  = vPdO [i + imode*nxystack] * scale;
    const complexf dPO  = vdPO [i + imode*nxystack] * scale;
    const complexf dPdO = vdPdO[i + imode*nxystack] * scale;
    const complexf a = dPO + PdO;
    a1 += PO.real() * a.real() + PO.imag() * a.imag();
    sumPO2 += dot(PO, PO);
    a2 += dot(dPO,dPO) + dot(PdO,PdO) + 2*(PO.real() * dPdO.real() + PO.imag() * dPdO.imag() + PdO.real() * dPO.real() + PdO.imag() * dPO.imag());
    a3 += PO.real() * a.real() + PO.imag() * a.imag();
    a4 += dot(dPdO,dPdO);
  }
  sumPO2 = fmaxf(sumPO2 + backg, 1e-20f); // Should only happen with null frames used for 16-padding

  // This is written to avoid overflows, not calculating (R_PO_OdP_Pdo*R_PO_OdP_Pdo)/(sumPO2*sumPO2)
  a1 = 2 * a1 / sumPO2;
  a2 /= sumPO2;
  a3 = 2 * a3 / sumPO2;
  a4 /= sumPO2;

  return my_float4((sumPO2 - obs) * a1,
                  sumPO2 * a2 - obs * (a2 - 0.5 * a1 * a1),
                  sumPO2 * a3 - obs * (a3 - a1 * a2 + a1*a1*a1/3),
                  sumPO2 * a4 - obs * (a4 - a1 * a3 - 0.5 * a2 * a2 + a1 * a1 * a2 -0.25 * a1*a1*a1*a1 ));
}
