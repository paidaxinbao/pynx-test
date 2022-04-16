#include <pycuda-complex.hpp>
typedef pycuda::complex<float> complexf;

// This must be defined in Python using:
// argmax_dtype = np.dtype([("idx", np.int32), ("cur_max", np.float32)])
// cu_tools.get_or_register_dtype("idx_max", argmax_dtype)


struct idx_max
{
    int idx;
    float cur_max;
    __device__
    idx_max()
    { }
    __device__
    idx_max(int cidx, float cmax)
    : idx(cidx), cur_max(cmax)
    { }
    __device__ idx_max(idx_max const &src)
    : idx(src.idx), cur_max(src.cur_max)
    { }
    __device__ idx_max(idx_max const volatile &src)
    : idx(src.idx), cur_max(src.cur_max)
    { }
    __device__ idx_max volatile &operator=(
        idx_max const &src) volatile
    {
        idx = src.idx;
        cur_max = src.cur_max;
        return *this;
    }
};

 __device__ idx_max argmax_reduce(idx_max a, idx_max b)
 {
   if(a.cur_max>b.cur_max) return a;
   return b;
 }
