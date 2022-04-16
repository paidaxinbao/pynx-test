/* -*- coding: utf-8 -*-

* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
*   (c) 2016-present : Univ. Grenoble Alpes, CEA/INAC/SP2M
*   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
*       author: Vincent Favre-Nicolin vincent.favre-nicolin@univ-grenoble-alpes.fr, favre@esrf.fr
*/

#define USE_SSE2
#include "sse_mathfun.h"
#include "Python.h"
// Note: either use Py_BEGIN_ALLOW_THREADS, or 'with nogil' in cpu.pyx

void c_fhkl_cpu(const float *vh,const float *vk, const float *vl,
                const float *vx, const float *vy, const float *vz,
                const unsigned long natoms, const unsigned long nhkl, float *freal, float *fimag)
{
  Py_BEGIN_ALLOW_THREADS
  const float PI2         = -6.28318530717958647692528676655900577f;
  unsigned long i,at,j;
  for(i=0;i<nhkl;i++)
  {
      float fr=0,fi=0;
      const float h=vh[i]*PI2;
      const float k=vk[i]*PI2;
      const float l=vl[i]*PI2;
      const float * __restrict__ px=vx;
      const float * __restrict__ py=vy;
      const float * __restrict__ pz=vz;
      __m128 vfr,vfi,vs,vc;
      float tmp[4];
      for(at=0;at<natoms;at+=4)
      {
        float * __restrict__ ptmp=&tmp[0];

        // Dangerous ? Order of operation is not guaranteed - but it works...
        sincos_ps(_mm_set_ps(h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++),&vs,&vc);
        if(at==0)
        {vfr=vc;vfi=vs;}
        else
        {vfr=_mm_add_ps(vfr,vc);vfi=_mm_add_ps(vfi,vs);}
      }
      float tmp2[4];
      _mm_store_ps(tmp2,vfr);
      for(j=0;j<4;++j) fr+=tmp2[j];
      _mm_store_ps(tmp2,vfi);
      for(j=0;j<4;++j) fi+=tmp2[j];
      freal[i]=fr;
      fimag[i]=fi;
  }
  Py_END_ALLOW_THREADS
}


void c_fhklo_cpu(const float *vh,const float *vk, const float *vl,
                 const float *vx, const float *vy, const float *vz,
                 const float *vocc,const unsigned long natoms, const unsigned long nhkl, float *freal, float *fimag)
{
  Py_BEGIN_ALLOW_THREADS
  const float PI2         = -6.28318530717958647692528676655900577f;
  unsigned long i,at,j;
  for(i=0;i<nhkl;i++)
  {
      float fr=0,fi=0;
      const float h=vh[i]*PI2;
      const float k=vk[i]*PI2;
      const float l=vl[i]*PI2;
      const float * __restrict__ px=vx;
      const float * __restrict__ py=vy;
      const float * __restrict__ pz=vz;
      const float * __restrict__ pocc=vocc;
      __m128 vfr,vfi,vs,vc,vtmp;
      float tmp[4];
      for(at=0;at<natoms;at+=4)
      {
        float * __restrict__ ptmp=&tmp[0];

        // Dangerous ? Order of operation is not guaranteed - but it works...
        sincos_ps(_mm_set_ps(h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++),&vs,&vc);
        vtmp=_mm_set_ps(*pocc++,*pocc++,*pocc++,*pocc++);
        if(at==0)
        {vfr=_mm_mul_ps(vtmp,vc);vfi=_mm_mul_ps(vtmp,vs);}
        else
        {vfr=_mm_add_ps(vfr,_mm_mul_ps(vtmp,vc));vfi=_mm_add_ps(vfi,_mm_mul_ps(vtmp,vs));}
      }
      float tmp2[4];
      _mm_store_ps(tmp2,vfr);
      for(j=0;j<4;++j) fr+=tmp2[j];
      _mm_store_ps(tmp2,vfi);
      for(j=0;j<4;++j) fi+=tmp2[j];
      freal[i]=fr;
      fimag[i]=fi;
  }
  Py_END_ALLOW_THREADS
}

void c_fhkl_grazing_cpu(const float *vh, const float *vk, const float *vl,const float *vli,
                        const float *vx, const float *vy, const float *vz,
                        const unsigned long natoms, const unsigned long nhkl, float *freal, float *fimag)
{
  Py_BEGIN_ALLOW_THREADS
  const float PI2         = -6.28318530717958647692528676655900577f;
  unsigned long i,at,j;
  for(i=0;i<nhkl;i++)
  {
      float fr=0,fi=0;
      const float h=vh[i]*PI2;
      const float k=vk[i]*PI2;
      const float l=vl[i]*PI2;
      const float li=vli[i]*PI2;
      const float * __restrict__ px=vx;
      const float * __restrict__ py=vy;
      const float * __restrict__ pz=vz;
      __m128 vfr,vfi,vs,vc,vtmp;
      float tmp[4];
      for(at=0;at<natoms;at+=4)
      {
        float * __restrict__ ptmp=&tmp[0];

        // Dangerous ? Order of operation is not guaranteed - but it works...
        sincos_ps(_mm_set_ps(h* *px++ +k * *py++ + l * *(pz),
                             h* *px++ +k * *py++ + l * *(pz+1),
                             h* *px++ +k * *py++ + l * *(pz+2),
                             h* *px++ +k * *py++ + l * *(pz+3)),&vs,&vc);
        vtmp=exp_ps(_mm_set_ps(*pz++*li,*pz++*li,*pz++*li,*pz++*li));
        if(at==0)
        {vfr=_mm_mul_ps(vtmp,vc);vfi=_mm_mul_ps(vtmp,vs);}
        else
        {vfr=_mm_add_ps(vfr,_mm_mul_ps(vtmp,vc));vfi=_mm_add_ps(vfi,_mm_mul_ps(vtmp,vs));}
      }
      float tmp2[4];
      _mm_store_ps(tmp2,vfr);
      for(j=0;j<4;++j) fr+=tmp2[j];
      _mm_store_ps(tmp2,vfi);
      for(j=0;j<4;++j) fi+=tmp2[j];
      freal[i]=fr;
      fimag[i]=fi;
  }
  Py_END_ALLOW_THREADS
}


void c_fhklo_grazing_cpu(const float *vh, const float *vk, const float *vl, const float *vli,
                         const float *vx, const float *vy, const float *vz, const float *vocc,
                         const unsigned long natoms, const unsigned long nhkl, float *freal, float *fimag)
{
  Py_BEGIN_ALLOW_THREADS
  const float PI2         = -6.28318530717958647692528676655900577f;
  unsigned long i,at,j;
  for(i=0;i<nhkl;i++)
  {
      float fr=0,fi=0;
      const float h=vh[i]*PI2;
      const float k=vk[i]*PI2;
      const float l=vl[i]*PI2;
      const float li=vli[i]*PI2;
      const float * __restrict__ px=vx;
      const float * __restrict__ py=vy;
      const float * __restrict__ pz=vz;
      const float * __restrict__ pocc=vocc;
      __m128 vfr,vfi,vs,vc,vtmp;
      float tmp[4];
      for(at=0;at<natoms;at+=4)
      {
        float * __restrict__ ptmp=&tmp[0];

        // Dangerous ? Order of operation is not guaranteed - but it works...
        sincos_ps(_mm_set_ps(h* *px++ +k * *py++ + l * *(pz),
                             h* *px++ +k * *py++ + l * *(pz+1),
                             h* *px++ +k * *py++ + l * *(pz+2),
                             h* *px++ +k * *py++ + l * *(pz+3)),&vs,&vc);
        vtmp=_mm_mul_ps(_mm_set_ps(*pocc++,*pocc++,*pocc++,*pocc++),exp_ps(_mm_set_ps(*pz++*li,*pz++*li,*pz++*li,*pz++*li)));
        if(at==0)
        {vfr=_mm_mul_ps(vtmp,vc);vfi=_mm_mul_ps(vtmp,vs);}
        else
        {vfr=_mm_add_ps(vfr,_mm_mul_ps(vtmp,vc));vfi=_mm_add_ps(vfi,_mm_mul_ps(vtmp,vs));}
      }
      float tmp2[4];
      _mm_store_ps(tmp2,vfr);
      for(j=0;j<4;++j) fr+=tmp2[j];
      _mm_store_ps(tmp2,vfi);
      for(j=0;j<4;++j) fi+=tmp2[j];
      freal[i]=fr;
      fimag[i]=fi;
  }
  Py_END_ALLOW_THREADS
}
