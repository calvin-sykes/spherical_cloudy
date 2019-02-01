#-*- mode: python -*-

import numpy as np
cimport numpy as np
DTYPE = np.float64
ctypedef np.float_t DTYPE_t
ITYPE = np.int64
ctypedef np.int_t ITYPE_t

cdef class HaloModel:
    cdef public DTYPE_t mvir
    cdef public DTYPE_t rvir
    cdef public DTYPE_t rscale
    cdef public DTYPE_t baryfrac
    cdef public DTYPE_t rhods
    cdef public object name

    cdef DTYPE_t fm(self, DTYPE_t x) nogil