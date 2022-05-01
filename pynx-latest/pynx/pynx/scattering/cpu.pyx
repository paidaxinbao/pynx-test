# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2016-present : Univ. Grenoble Alpes, CEA/INAC/SP2M
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       author: Vincent Favre-Nicolin vincent.favre-nicolin@univ-grenoble-alpes.fr, favre@esrf.fr


import cython
import numpy as np
cimport numpy as np




cdef extern void c_fhkl_cpu(const float *vh,const float *vk, const float *vl,
                            const float *vx, const float *vy, const float *vz,
                            const unsigned long natoms, const unsigned long nhkl, float *freal, float *fimag) nogil

cdef extern void c_fhklo_cpu(const float *vh,const float *vk, const float *vl,
                             const float *vx, const float *vy, const float *vz, const float *vocc,
                             const unsigned long natoms, const unsigned long nhkl, float *freal, float *fimag) nogil

cdef extern void c_fhkl_grazing_cpu(const float *vh,const float *vk, const float *vl, const float *vli,
                                    const float *vx, const float *vy, const float *vz,
                                    const unsigned long natoms, const unsigned long nhkl, float *freal, float *fimag) nogil

cdef extern void c_fhklo_grazing_cpu(const float *vh,const float *vk, const float *vl, const float *vli,
                                     const float *vx, const float *vy, const float *vz, const float *vocc,
                                     const unsigned long natoms, const unsigned long nhkl, float *freal, float *fimag) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def fhkl_cpu(np.ndarray[float, ndim=1, mode="c"] vh not None,
             np.ndarray[float, ndim=1, mode="c"] vk not None,
             np.ndarray[float, ndim=1, mode="c"] vl not None,
             np.ndarray[float, ndim=1, mode="c"] vx not None,
             np.ndarray[float, ndim=1, mode="c"] vy not None,
             np.ndarray[float, ndim=1, mode="c"] vz not None,
             np.ndarray[float, ndim=1, mode="c"] freal not None,
             np.ndarray[float, ndim=1, mode="c"] fimag not None):
    cdef unsigned long natoms=len(vx), nhkl=len(vh)
    # with nogil: # either that or use Py_BEGIN_ALLOW_THREADS in C code
    c_fhkl_cpu(&vh[0],&vk[0],&vl[0],&vx[0],&vy[0],&vz[0],natoms,nhkl,&freal[0],&fimag[0])
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
def fhklo_cpu(np.ndarray[float, ndim=1, mode="c"] vh not None,
             np.ndarray[float, ndim=1, mode="c"] vk not None,
             np.ndarray[float, ndim=1, mode="c"] vl not None,
             np.ndarray[float, ndim=1, mode="c"] vx not None,
             np.ndarray[float, ndim=1, mode="c"] vy not None,
             np.ndarray[float, ndim=1, mode="c"] vz not None,
             np.ndarray[float, ndim=1, mode="c"] vocc not None,
             np.ndarray[float, ndim=1, mode="c"] freal not None,
             np.ndarray[float, ndim=1, mode="c"] fimag not None):
    cdef unsigned long natoms=len(vx), nhkl=len(vh)

    #with nogil:
    c_fhklo_cpu(&vh[0],&vk[0],&vl[0],&vx[0],&vy[0],&vz[0],&vocc[0],natoms,nhkl,&freal[0],&fimag[0])
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
def fhkl_grazing_cpu(np.ndarray[float, ndim=1, mode="c"] vh not None,
             np.ndarray[float, ndim=1, mode="c"] vk not None,
             np.ndarray[float, ndim=1, mode="c"] vl not None,
             np.ndarray[float, ndim=1, mode="c"] vli not None,
             np.ndarray[float, ndim=1, mode="c"] vx not None,
             np.ndarray[float, ndim=1, mode="c"] vy not None,
             np.ndarray[float, ndim=1, mode="c"] vz not None,
             np.ndarray[float, ndim=1, mode="c"] freal not None,
             np.ndarray[float, ndim=1, mode="c"] fimag not None):
    cdef unsigned long natoms=len(vx), nhkl=len(vh)
    #with nogil:
    c_fhkl_grazing_cpu(&vh[0],&vk[0],&vl[0],&vli[0],&vx[0],&vy[0],&vz[0],natoms,nhkl,&freal[0],&fimag[0])
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
def fhklo_grazing_cpu(np.ndarray[float, ndim=1, mode="c"] vh not None,
             np.ndarray[float, ndim=1, mode="c"] vk not None,
             np.ndarray[float, ndim=1, mode="c"] vl not None,
             np.ndarray[float, ndim=1, mode="c"] vli not None,
             np.ndarray[float, ndim=1, mode="c"] vx not None,
             np.ndarray[float, ndim=1, mode="c"] vy not None,
             np.ndarray[float, ndim=1, mode="c"] vz not None,
             np.ndarray[float, ndim=1, mode="c"] vocc not None,
             np.ndarray[float, ndim=1, mode="c"] freal not None,
             np.ndarray[float, ndim=1, mode="c"] fimag not None):
    cdef unsigned long natoms=len(vx), nhkl=len(vh)

    #with nogil:
    c_fhklo_grazing_cpu(&vh[0],&vk[0],&vl[0],&vli[0],&vx[0],&vy[0],&vz[0],&vocc[0],natoms,nhkl,&freal[0],&fimag[0])
    return None
