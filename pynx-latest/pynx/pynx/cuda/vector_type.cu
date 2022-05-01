extern "C++" {

// Useful aliases to have single-word types needed for string replacements
typedef pycuda::complex<float> complexf;
typedef unsigned int uint;
typedef unsigned long ulong;

/** this generic struct should be suitable for reduction kernels when several
* values need to be returned. TYPE can be float, complexf, int etc...
* The corresponding numpy type can be e.g. np.dtype([('x', np.float32, (5,))]),
* np.dtype([('x', np.complex64, (5,))]), etc..
*
* Volatile versions of the functions/operators are added as needed for
* pycuda's reduction kernels.
*/
struct %(TYPE)s_%(N)d
{
  typedef %(TYPE)s_%(N)d _Self;
  
  __device__ %(TYPE)s_%(N)d()                {};
  __device__ %(TYPE)s_%(N)d(const %(TYPE)s c)   { for(unsigned int i=0;i<%(N)d;i++) _v[i]=c;}
  __device__ %(TYPE)s_%(N)d(const _Self &v0) { for(unsigned int i=0;i<%(N)d;i++) _v[i]=v0[i];}
  
  __device__ const %(TYPE)s& operator[](const unsigned int i) const { return _v[i];}
  __device__ const volatile %(TYPE)s& operator[](const unsigned int i) volatile const { return _v[i];}
  __device__ %(TYPE)s& operator[](const unsigned int i) { return _v[i];}
  __device__ volatile %(TYPE)s& operator[](const unsigned int i) volatile { return _v[i];}
  
  __device__ _Self& operator=(const _Self& v0)
  { 
    for(unsigned int i=0;i<%(N)d;i++) _v[i]=v0[i];
    return *this;  
  }
  
  __device__ volatile _Self& operator=(const _Self& v0) volatile
  { 
    for(unsigned int i=0;i<%(N)d;i++) _v[i]=v0[i];
    return *this;  
  }

  __device__ _Self& operator=(const %(TYPE)s c)
  { 
    for(unsigned int i=0;i<%(N)d;i++) _v[i]=c;
    return *this;  
  }

  __device__ _Self operator+(const _Self& rhs)
  { 
    _Self v;
    for(unsigned int i=0;i<%(N)d;i++) v[i]=_v[i]+rhs[i];
    return v;  
  }

  __device__ _Self operator+(const volatile _Self& rhs) volatile
  { 
    _Self v;
    for(unsigned int i=0;i<%(N)d;i++) v[i]=_v[i]+rhs[i];
    return v;  
  }
  
  __device__ _Self operator-(const _Self& rhs)
  { 
    _Self v;
    for(unsigned int i=0;i<%(N)d;i++) v[i]=_v[i]-rhs[i];
    return v;  
  }

  __device__ _Self operator/(const _Self& rhs)
  { 
    _Self v;
    for(unsigned int i=0;i<%(N)d;i++) v[i]=_v[i]/rhs[i];
    return v;  
  }

  __device__ _Self operator*(const _Self& rhs)
  { 
    _Self v;
    for(unsigned int i=0;i<%(N)d;i++) v[i]=_v[i]*rhs[i];
    return v;  
  }

  __device__ void operator+=(const _Self& rhs) {for(unsigned int i=0;i<%(N)d;i++) _v[i] += rhs[i];}
  __device__ void operator-=(const _Self& rhs) {for(unsigned int i=0;i<%(N)d;i++) _v[i] -= rhs[i];}
  __device__ void operator*=(const _Self& rhs) {for(unsigned int i=0;i<%(N)d;i++) _v[i] *= rhs[i];}
  __device__ void operator/=(const _Self& rhs) {for(unsigned int i=0;i<%(N)d;i++) _v[i] /= rhs[i];}

  %(TYPE)s _v[%(N)d];
};

} // extern
