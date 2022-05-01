
extern "C++" {

/// We need to reimplement float4 to have the volatile version of operator=
struct my_float4
{
  // Constructors
  __device__ my_float4() : x(0), y(0), z(0), w(0) {}

  __device__ my_float4(const int& v)
    :  x(v), y(v), z(v), w(v) {}

  __device__ my_float4(const float& v)
    :  x(v), y(v), z(v), w(v) {}

  __device__ my_float4(const float& __x,const float& __y,const float& __z,const float& __w)
    :  x(__x), y(__y), z(__z), w(__w) {}

  __device__ my_float4(const my_float4& __z)
    :  x(__z.x), y(__z.y), z(__z.z), w(__z.w) {}

  // Operators
  __device__ my_float4& operator=(const my_float4& rhs)
  {
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    w = rhs.w;
    return *this;
  }

  __device__ volatile my_float4& operator=(const my_float4& rhs) volatile
  {
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    w = rhs.w;
    return *this;
  }

  __device__ my_float4& operator=(const float& rhs)
  {
    x = rhs;
    y = rhs;
    z = rhs;
    w = rhs;
    return *this;
  }

  __device__ volatile my_float4& operator=(const float& rhs) volatile
  {
    x = rhs;
    y = rhs;
    z = rhs;
    w = rhs;
    return *this;
  }

  __device__ my_float4& operator=(const int& rhs)
  {
    x = rhs;
    y = rhs;
    z = rhs;
    w = rhs;
    return *this;
  }

  __device__ volatile my_float4& operator=(const int& rhs) volatile
  {
    x = rhs;
    y = rhs;
    z = rhs;
    w = rhs;
    return *this;
  }

  __device__ my_float4& operator+= (const my_float4 &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    w += rhs.w;
    return *this;
  }

  // Data members
  float x,y,z,w;
};

inline __device__ my_float4 operator+(const my_float4 &v1, const my_float4 &v2)
{
  return my_float4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w+v2.w);
}

inline __device__ my_float4 operator+(const volatile my_float4 &v1, const volatile my_float4 &v2)
{
  return my_float4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w+v2.w);
}

struct my_float8
{
  // Constructors
  __device__ my_float8() : s0(0), s1(0), s2(0), s3(0), s4(0), s5(0), s6(0), s7(0) {}

  __device__ my_float8(const int& v)
    :  s0(v), s1(v), s2(v), s3(v), s4(v), s5(v), s6(v), s7(v) {}

  __device__ my_float8(const float& v)
    :  s0(v), s1(v), s2(v), s3(v), s4(v), s5(v), s6(v), s7(v) {}

  __device__ my_float8(const float& __s0,const float& __s1,const float& __s2,const float& __s3,
                       const float& __s4,const float& __s5,const float& __s6,const float& __s7)
    :  s0(__s0), s1(__s1), s2(__s2), s3(__s3), s4(__s4), s5(__s5), s6(__s6), s7(__s7) {}

  __device__ my_float8(const my_float8& z)
    :  s0(z.s0), s1(z.s1), s2(z.s2), s3(z.s3), s4(z.s4), s5(z.s5), s6(z.s6), s7(z.s7) {}

  // Operators
  __device__ my_float8& operator=(const my_float8& rhs)
  {
    s0 = rhs.s0;
    s1 = rhs.s1;
    s2 = rhs.s2;
    s3 = rhs.s3;
    s4 = rhs.s4;
    s5 = rhs.s5;
    s6 = rhs.s6;
    s7 = rhs.s7;
    return *this;
  }

  __device__ volatile my_float8& operator=(const my_float8& rhs) volatile
  {
    s0 = rhs.s0;
    s1 = rhs.s1;
    s2 = rhs.s2;
    s3 = rhs.s3;
    s4 = rhs.s4;
    s5 = rhs.s5;
    s6 = rhs.s6;
    s7 = rhs.s7;
    return *this;
  }

  __device__ my_float8& operator=(const float& rhs)
  {
    s0 = rhs;
    s1 = rhs;
    s2 = rhs;
    s3 = rhs;
    s4 = rhs;
    s5 = rhs;
    s6 = rhs;
    s7 = rhs;
    return *this;
  }

  __device__ volatile my_float8& operator=(const float& rhs) volatile
  {
    s0 = rhs;
    s1 = rhs;
    s2 = rhs;
    s3 = rhs;
    s4 = rhs;
    s5 = rhs;
    s6 = rhs;
    s7 = rhs;
    return *this;
  }

  __device__ my_float8& operator=(const int& rhs)
  {
    s0 = rhs;
    s1 = rhs;
    s2 = rhs;
    s3 = rhs;
    s4 = rhs;
    s5 = rhs;
    s6 = rhs;
    s7 = rhs;
    return *this;
  }

  __device__ volatile my_float8& operator=(const int& rhs) volatile
  {
    s0 = rhs;
    s1 = rhs;
    s2 = rhs;
    s3 = rhs;
    s4 = rhs;
    s5 = rhs;
    s6 = rhs;
    s7 = rhs;
    return *this;
  }

  __device__ my_float8& operator+= (const my_float8 &rhs) {
    s0 += rhs.s0;
    s1 += rhs.s1;
    s2 += rhs.s2;
    s3 += rhs.s3;
    s4 += rhs.s4;
    s5 += rhs.s5;
    s6 += rhs.s6;
    s7 += rhs.s7;
    return *this;
  }

  // Data members
  float s0, s1, s2, s3, s4, s5, s6, s7;
};

inline __device__ my_float8 operator+(const my_float8 &v1, const my_float8 &v2)
{
  return my_float8(v1.s0+v2.s0, v1.s1+v2.s1, v1.s2+v2.s2, v1.s3+v2.s3,
                   v1.s4+v2.s4, v1.s5+v2.s5, v1.s6+v2.s6, v1.s7+v2.s7);
}

inline __device__ my_float8 operator+(const volatile my_float8 &v1, const volatile my_float8 &v2)
{
  return my_float8(v1.s0+v2.s0, v1.s1+v2.s1, v1.s2+v2.s2, v1.s3+v2.s3,
                   v1.s4+v2.s4, v1.s5+v2.s5, v1.s6+v2.s6, v1.s7+v2.s7);
}

}// extern "C++"

