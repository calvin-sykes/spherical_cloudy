#-*- mode: python -*-
import numpy as np
cimport numpy as np


cdef extern from "math.h":
    double clog "log" (double) nogil
    double catan "atan" (double) nogil


cdef class HaloModel:  
    def __init__(self, virial_mass, baryon_frac, rho_crit, conc):
        self.mvir = virial_mass
        self.rvir = (3.0 * virial_mass / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
        self.rscale = self.rvir / conc
        self.baryfrac = baryon_frac
        self.rhods = 200.0 * rho_crit * conc**3 / self.fm(conc)

    cdef DTYPE_t fm(self, DTYPE_t x) nogil:
        """Return f_M(x) s.t. M(<r) = (4 * pi * rho_s * (r_s)**3 / 3) * f_M(x)"""
        with gil:
            raise NotImplementedError("fm is not defined for base class")


cdef class NFWHalo(HaloModel):
    """NFW halo with rho(x) = rho_s / (x * (1 + x)**2) for x = r/r_s"""
    def __init__(self, virial_mass, baryon_frac, rho_crit, conc):
        HaloModel.__init__(self, virial_mass, baryon_frac, rho_crit, conc)
        self.name = 'NFW'

    cdef DTYPE_t fm(self, DTYPE_t x) nogil:
        return 3.0 * (clog(1.0 + x) - x / (1.0 + x))


cdef class BurkertHalo(HaloModel):
    """Burkert halo with rho(x) = (rho_0 * (r_0)**3) / ((r + r_0) * (r**2 + (r_0)**2)) for x = r/r_0"""
    def __init__(self, virial_mass, baryon_frac, rho_crit, conc=None, r_zero=None):
        #cnew
        if r_zero is not None:
            # Use provided r0 and determine c200
            assert conc is None
            self.rscale = r_zero
            cnew = (3.0 * virial_mass / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0) / r_zero
        else:
            # Use provided c200 and determine rscale
            assert r_zero is None
            cnew = conc
        HaloModel.__init__(self, virial_mass, baryon_frac, rho_crit, cnew)
        self.name = 'Burkert'

    cdef DTYPE_t fm(self, DTYPE_t x) nogil:
        return 1.5 * (clog(1.0 + x) + 0.5 * clog(1.0 + x**2.0) - catan(x))


# cdef class CoredHalo(HaloModel):
#     """Modified isothermal halo with rho(x) = rho_0 / (1 + (r / r_c)**2) for x = r / r_c"""
#     def __init__(self, virial_mass, baryon_frac, rho_crit, conc, acore):
#         HaloModel.__init__(self, virial_mass, baryon_frac, rho_crit, conc)
#         self.acore = acore
#         self.name = 'Cored'

#     cpdef DTYPE_t fm(self, DTYPE_t x):
#         return 3.0 * (x / ((self.acore-1.0) * (1.0 + x))
#                       + (self.acore**2.0 * np.log(1.0 + x / self.acore)
#                          + (1.0 - 2.0*self.acore) * np.log(1.0 + x)) / (1.0 - self.acore)**2.0 )


cdef class PPHalo(HaloModel):
    cdef public DTYPE_t cden
    cdef public DTYPE_t density
    def __init__(self, coldens, dens):
        self.cden = coldens
        self.density = dens
        self.name = 'PP'


cpdef get_fm(HaloModel hm, DTYPE_t x):
    cdef DTYPE_t fmx = hm.fm(x)
    return fmx
