#-*- mode: python -*-
# cython: profile=False, wraparound=False, boundscheck=False, cdivision=True

import numpy as np
import cython.parallel as para

cimport numpy as np
cimport cython
cimport openmp
DTYPE = np.float64
ctypedef np.float_t DTYPE_t
ITYPE = np.int64
ctypedef np.int_t ITYPE_t

cdef extern from "math.h":
    double csqrt "sqrt" (double) nogil
    double cexp "exp" (double) nogil
    double clog10 "log10" (double) nogil
    double cabs "fabs" (double) nogil

cimport cython_halo


def set_num_threads(int num_threads):
    if num_threads > 0:
        openmp.omp_set_num_threads(num_threads)
    else:
        openmp.omp_set_num_threads(openmp.omp_get_max_threads())


def calc_coldens(double[::1] density not None,
                 double[::1] radius not None,
                 int nummu):
    cdef int sz_r
    cdef int r, ri, mu, rb, fl
    cdef double rint, rtmp, rtmpc

    sz_r  = radius.shape[0]

    cdef double[:,::1] coldens = np.zeros((sz_r,nummu), dtype=DTYPE)
    cdef double[::1] muarr = np.linspace(-1.0,1.0,nummu)

    #print "Performing numerical integration over radius"
    with nogil:
        for mu in range(0,nummu):
            for r in range(0,sz_r):
                # Compute the r integral
                rint = 0.0
                rtmpc = radius[r]*csqrt(1.0-muarr[mu]**2)
                if muarr[mu] >= 0.0:
                    for ri in range(r,sz_r):
                        if ri == sz_r-1:
                            rtmp = radius[ri] + 0.5*(radius[ri]-radius[ri-1])
                            rint += 0.5*(density[ri-1]-density[ri]) * (csqrt( rtmp**2 - rtmpc**2 ) - csqrt( radius[ri]**2 - rtmpc**2 ))
                        else:
                            rint += 0.5*(density[ri+1]+density[ri]) * (csqrt( radius[ri+1]**2 - rtmpc**2 ) - csqrt( radius[ri]**2 - rtmpc**2 ))
                else:
                    rb = -1
                    fl = -1
                    for ri in range(1,sz_r):
                        if radius[ri] <= rtmpc:
                            continue
                        else:
                            if rb == -1: rb = ri
                            if ri == sz_r-1:
                                rtmp = radius[ri] + (radius[ri]-radius[ri-1])
                                rint += 0.5*(density[ri-1]-density[ri]) * (csqrt( rtmp**2 - rtmpc**2 ) - csqrt( radius[ri]**2 - rtmpc**2 ))
                                if fl == -1: rint += 0.5*(density[ri-1]-density[ri]) * (csqrt( rtmp**2 - rtmpc**2 ) - csqrt( radius[ri]**2 - rtmpc**2 ))
                            else:
                                rint += 0.5*(density[ri+1]+density[ri]) * (csqrt( radius[ri+1]**2 - rtmpc**2 ) - csqrt( radius[ri]**2 - rtmpc**2 ))
                                if ri < r:
                                    fl = 1
                                    rint += 0.5*(density[ri+1]+density[ri]) * (csqrt( radius[ri+1]**2 - rtmpc**2 ) - csqrt( radius[ri]**2 - rtmpc**2 ))
                    rint += 2.0 * 0.5*(density[rb]+density[rb-1]) * csqrt( radius[rb]**2 - rtmpc**2 )
                coldens[r,mu] = rint

    return np.asarray(coldens), np.asarray(muarr)


def calc_coldens_allion(double[:,::1] density not None,
                        double[::1] radius not None,
                        int nummu):
    cdef int sz_r, sz_ion
    cdef int r, ri, mu, rb, fl, ion
    cdef double rint, rtmp2, rtmpc2

    sz_r  = radius.shape[0]
    sz_ion = density.shape[0]

    cdef double[:,:,::1] coldens = np.zeros((sz_ion, sz_r, nummu), dtype=DTYPE)
    cdef double[::1] muarr = np.linspace(-1.0, 1.0, nummu)

    #print "Performing numerical integration over radius"
    with nogil:
        for ion in range(0, sz_ion):
            for r in para.prange(0, sz_r):
                for mu in range(0, nummu):
                    # Compute the r integral
                    rint = 0.0
                    rtmpc2 = radius[r]**2 * (1.0 - muarr[mu]**2)
                    if muarr[mu] >= 0.0:
                        for ri in range(r, sz_r):
                            if ri == sz_r - 1:
                                rtmp2 = (radius[ri] + 0.5 * (radius[ri] - radius[ri-1]))**2
                                rint = rint + 0.5 * (density[ion,ri-1] - density[ion,ri]) * (csqrt(rtmp2 - rtmpc2) - csqrt(radius[ri]**2 - rtmpc2))
                            else:
                                rint = rint + 0.5 * (density[ion,ri+1] + density[ion,ri]) * (csqrt(radius[ri+1]**2 - rtmpc2) - csqrt(radius[ri]**2 - rtmpc2))
                    else:
                        rb = -1
                        fl = -1
                        for ri in range(1, sz_r):
                            if radius[ri]**2 <= rtmpc2:
                                continue
                            else:
                                if rb == -1:
                                    rb = ri
                                if ri == sz_r - 1:
                                    rtmp2 = (radius[ri] + (radius[ri] - radius[ri-1]))**2
                                    rint += 0.5 * (density[ion,ri-1] - density[ion,ri]) * (csqrt(rtmp2 - rtmpc2) - csqrt(radius[ri]**2 - rtmpc2))
                                    if fl == -1:
                                        rint = rint + 0.5 * (density[ion,ri-1] - density[ion,ri]) * (csqrt(rtmp2 - rtmpc2) - csqrt(radius[ri]**2 - rtmpc2))
                                else:
                                    rint = rint + 0.5*(density[ion,ri+1] + density[ion,ri]) * (csqrt(radius[ri+1]**2 - rtmpc2) - csqrt(radius[ri]**2 - rtmpc2))
                                    if ri < r:
                                        fl = 1
                                        rint = rint + 0.5 * (density[ion,ri+1] + density[ion,ri]) * (csqrt(radius[ri+1]**2 - rtmpc2) - csqrt(radius[ri]**2 - rtmpc2))
                        rint = rint + 2.0 * 0.5 * (density[ion,rb] + density[ion,rb-1]) * csqrt(radius[rb]**2 - rtmpc2)
                    coldens[ion,r,mu] = rint
    return np.asarray(coldens), np.asarray(muarr)


def calc_coldensPP(double[::1] density not None,
                   double[::1] radius not None):
    cdef int sz_r
    cdef int r
    cdef double rint

    sz_r  = radius.shape[0]

    cdef double[::1] coldens = np.zeros((sz_r), dtype=DTYPE)

    with nogil:
        rint = 0.0
        for r in range(1,sz_r):
            rint += 0.5*(density[sz_r-r]+density[sz_r-r-1])*(radius[sz_r-r]-radius[sz_r-r-1])
            coldens[sz_r-r-1] = rint
    return np.asarray(coldens)


def calc_coldensPP_allion(double[:,::1] density not None,
                          double[::1] radius not None):
    cdef int sz_r, sz_ion
    cdef int r, ion
    cdef double rint

    sz_r  = radius.shape[0]
    sz_ion = density.shape[0]

    cdef double[:,::1] coldens = np.zeros((sz_ion,sz_r), dtype=DTYPE)

    with nogil:
        for ion in range(0, sz_ion):
            rint = 0.0
            for r in range(1,sz_r):
                rint += 0.5 * (density[ion,sz_r-r] + density[ion,sz_r-r-1]) * (radius[sz_r-r] - radius[sz_r-r-1])
            coldens[ion, sz_r-r-1] = rint
    return np.asarray(coldens)


def nint_costheta(DTYPE_t[:,:,::1] coldens not None,
                  DTYPE_t[:,::1] xsec not None,
                  DTYPE_t[::1] muarr not None,
                  DTYPE_t[::1] jzero not None):
    """
    Integrate over cos(theta)
    """
    cdef int sz_r, sz_nu, sz_mu, sz_ion
    cdef int r, nu, mu, ion
    cdef double rint, tau, taup

    sz_ion = coldens.shape[0]
    sz_r  = coldens.shape[1]
    sz_nu = jzero.shape[0]
    sz_mu = muarr.shape[0]

    cdef DTYPE_t[:,::1] retarr = np.empty((sz_nu,sz_r), dtype=DTYPE)
    
    with nogil:
        for r in para.prange(0, sz_r):
            for nu in range(0, sz_nu):
                rint = 0.0
                for mu in range(0,sz_mu-1):
                    tau = 0.0
                    taup = 0.0
                    for ion in range(sz_ion):
                        tau = tau + xsec[ion,nu] * coldens[ion,r,mu]
                        taup = taup + xsec[ion,nu] * coldens[ion,r,mu+1]
                    rint = rint + 0.5 * (cexp(-tau) + cexp(-taup)) * (muarr[mu+1] - muarr[mu])    
                retarr[nu, r] = 0.5 * jzero[nu] * rint
    return np.asarray(retarr)


def nint_pp(double[:,::1] coldens not None,
            double[:,::1] xsec not None,
            double[::1] jzero not None):
    """
    Integrate to get the flux at each radial bin
    """
    cdef int sz_r, sz_nu, sz_ion
    cdef int r, nu, ion
    cdef double tau

    sz_ion  = coldens.shape[0]
    sz_r  = coldens.shape[1]
    sz_nu = jzero.shape[0]

    cdef double[:,::1] retarr = np.zeros((sz_nu,sz_r), dtype=DTYPE)

    with nogil:
        for nu in range(sz_nu):
            for r in range(0,sz_r):
                tau  = 0.0
                for ion in range(0,sz_ion):
                    tau += xsec[ion,nu]*coldens[ion,r]
                retarr[nu,r] = jzero[nu] * cexp(-tau)
    return np.asarray(retarr)


def phionrate(double[:,::1] jnur not None,
              double[::1] xsec not None,
              double[::1] nuarr not None,
              double planck):
    """
    Calculate the local primary photoionization rate (numerically integrate Eq. 23 from Sternberg et al. 2002)
    """
    cdef int sz_r, sz_nu
    cdef int r, nu
    cdef double nint

    sz_r  = jnur.shape[1]
    sz_nu = xsec.shape[0]

    cdef double[::1] retarr = np.zeros((sz_r), dtype=DTYPE)

    #print "Performing numerical integration over frequency"
    with nogil:
        for r in range(0,sz_r):
            nint = 0.0
            for nu in range(0,sz_nu-1):
                nint += 0.5 * (xsec[nu] * jnur[nu,r] / (planck * nuarr[nu]) + xsec[nu+1] * jnur[nu+1,r] / (planck * nuarr[nu+1])) * (nuarr[nu+1] - nuarr[nu])
            retarr[r] = nint
    return np.asarray(retarr)


def phionrate_allion(double[:,::1] jnur not None,
                     double[:,::1] xsec not None,
                     double[::1] nuarr not None,
                     double planck):
    """
    Calculate the local primary photoionization rate (numerically integrate Eq. 23 from Sternberg et al. 2002)
    """
    cdef int sz_r, sz_nu, sz_ion
    cdef int r, nu
    cdef double nint

    sz_r  = jnur.shape[1]
    sz_ion = xsec.shape[0]
    sz_nu = xsec.shape[1]

    cdef double[:,::1] retarr = np.zeros((sz_ion,sz_r), dtype=DTYPE)

    #print "Performing numerical integration over frequency"
    with nogil:
        for ion in range(0, sz_ion):
            for r in para.prange(0,sz_r):
                nint = 0.0
                for nu in range(0,sz_nu-1):
                    nint = nint + (0.5 / planck) * (xsec[ion,nu] * jnur[nu,r] / nuarr[nu] + xsec[ion,nu+1] * jnur[nu+1,r] / nuarr[nu+1]) * (nuarr[nu+1] - nuarr[nu])
                retarr[ion,r] = nint
    return np.asarray(retarr)


def phheatrate_allion(double[:,::1] jnur not None,
                      double[:,::1] xsec not None,
                      double[::1] nuarr not None,
                      double[::1] nuion not None,
                      double planck):
    """
    Calculate the local primary photoheating rate (numerically integrate Eq. 23 from Sternberg et al. 2002)
    """
    cdef int sz_ion, sz_r, sz_nu
    cdef int ion, r, nu
    cdef double nint

    sz_ion = nuion.shape[0]
    sz_r = jnur.shape[1]
    sz_nu = xsec.shape[1]

    cdef double[:,::1] retarr = np.zeros((sz_ion,sz_r), dtype=DTYPE)

    #print "Performing numerical integration over frequency"
    with nogil:
        for ion in range(0,sz_ion):
            for r in range(0,sz_r):
                nint = 0.0
                for nu in range(0,sz_nu-1):
                    nint += 0.5 * (xsec[ion,  nu] * jnur[nu  ,r] * (planck * (nuarr[nu  ] - nuion[ion])) / (planck * nuarr[nu  ]) +
                                   xsec[ion,nu+1] * jnur[nu+1,r] * (planck * (nuarr[nu+1] - nuion[ion])) / (planck * nuarr[nu+1])) * (nuarr[nu+1] - nuarr[nu])
                retarr[ion,r] = nint
    return np.asarray(retarr)


def scdryrate(double[:,::1] jnur not None,
              double[::1] nuarr not None,
              double[::1] xsecHI not None,
              double[::1] xsecDI not None,
              double[::1] xsecHeI not None,
              double[::1] xsecHeII not None,
              double[::1] profHI not None,
              double[::1] profDI not None,
              double[::1] profHeI not None,
              double[::1] profHeII not None,
              double[::1] xe not None,
              double HIip, double DIip, double HeIip, double HeIIip,
              double planck, double elvolt, int flip):
    """
    Calculate the secondary photoelectron ionization rate
    """
    cdef int sz_r, sz_nu
    cdef int r, nu
    cdef double eint, divIP
    cdef double trmAa, trmAb, trmBa, trmBb, trmCa, trmCb, trmDa, trmDb, fonea, foneb, ftwoa, ftwob, yone, ytwo

    sz_r  = jnur.shape[1]
    sz_nu = xsecHI.shape[0]
    cdef double pev = planck/elvolt

    cdef double[::1] retarr = np.zeros((sz_r), dtype=DTYPE)

    #print "Performing numerical integration over frequency to get secondary photoelectron ionization"
    with nogil:
        for r in para.prange(0,sz_r):
            eint = 0.0
            if flip == 0: # H I
                yone = 0.3908 * (1.0 - xe[r]**0.4092)**1.7592
                ytwo = 0.6941 * xe[r]**0.2 * (1.0 - xe[r]**0.38)**2.0
                divIP = HIip
#	    elif flip == 1: # D I
#		yone = 0.3908 * (1.0 - xe[r]**0.4092)**1.7592
#		ytwo = 0.6941 * xe[r]**0.2 * (1.0 - xe[r]**0.38)**2.0
            elif flip == 2: # He I
                yone = 0.0554 * (1.0 - xe[r]**0.4614)**1.6660
                ytwo = 0.0984 * xe[r]**0.2 * (1.0 - xe[r]**0.38)**2.0
                divIP = HeIip
            for nu in range(0,sz_nu-1):
                # 2.41775E14 = 1.602E-19/6.626E-34
                # 4.1361E-15 = 6.626E-34/1.602E-19
                # Use the Ricotti et al. (2002) solution
                ###  H I  ###
                if pev*nuarr[nu]-HIip < 28.0:
                    trmAa = 0.0
                else:
                    trmAa = xsecHI[nu] * ((pev*nuarr[nu]-HIip)/divIP) * (yone - ytwo*(28.0/(pev*nuarr[nu]-HIip))**0.4)
                if pev*nuarr[nu+1]-HIip < 28.0:
                    trmAb = 0.0
                else:
                    trmAb = xsecHI[nu+1] * ((pev*nuarr[nu+1]-HIip)/divIP) * (yone - ytwo*(28.0/(pev*nuarr[nu+1]-HIip))**0.4)
                ###  D I  ###
                if pev*nuarr[nu]-DIip < 28.0:
                    trmDa = 0.0
                else:
                    trmDa = (profDI[r]/profHI[r]) * xsecDI[nu] * ((pev*nuarr[nu]-DIip)/divIP) * (yone - ytwo*(28.0/(pev*nuarr[nu]-DIip))**0.4)
                if pev*nuarr[nu+1]-DIip < 28.0:
                    trmDb = 0.0
                else:
                    trmDb = (profDI[r]/profHI[r]) * xsecDI[nu+1] * ((pev*nuarr[nu+1]-DIip)/divIP) * (yone - ytwo*(28.0/(pev*nuarr[nu+1]-DIip))**0.4)
                ###  He I  ###
                if pev*nuarr[nu]-HeIip < 28.0:
                    trmBa = 0.0
                else:
                    trmBa = (profHeI[r]/profHI[r]) * xsecHeI[nu] * ((pev*nuarr[nu]-HeIip)/divIP) * (yone - ytwo * (28.0/(pev*nuarr[nu]-HeIip))**0.4)
                if pev*nuarr[nu+1]-HeIip < 28.0:
                    trmBb = 0.0
                else:
                    trmBb = (profHeI[r]/profHI[r]) * xsecHeI[nu+1] * ((pev*nuarr[nu+1]-HeIip)/divIP) * (yone - ytwo * (28.0/(pev*nuarr[nu+1]-HeIip))**0.4)
                ###  He II  ###
                if pev*nuarr[nu]-HeIIip < 28.0:
                    trmCa = 0.0
                else:
                    trmCa = (profHeII[r]/profHI[r]) * xsecHeII[nu] * ((pev*nuarr[nu]-HeIIip)/divIP) * (yone - ytwo*(28.0/(pev*nuarr[nu]-HeIIip))**0.4)
                if pev*nuarr[nu+1]-HeIIip < 28.0:
                    trmCb = 0.0
                else:
                    trmCb = (profHeII[r]/profHI[r]) * xsecHeII[nu+1] * ((pev*nuarr[nu+1]-HeIIip)/divIP) * (yone* - ytwo*(28.0/(pev*nuarr[nu+1]-HeIIip))**0.4)
                ### Now calculate the integral
                eint = eint + 0.5*(((trmAa+trmDa+trmBa+trmCa)*jnur[nu,r]/(planck*nuarr[nu])) + (trmAb+trmDb+trmBb+trmCb)*jnur[nu+1,r]/(planck*nuarr[nu+1])) * (nuarr[nu+1]-nuarr[nu])
            retarr[r] = eint
    return np.asarray(retarr)


def scdryheatrate(double[:,::1] jnur not None,
                  double[::1] nuarr not None,
                  double[::1] xsec not None,
                  double[::1] xe not None,
                  double HIip, double DIip, double HeIip,
                  double planck, double elvolt, int flip):
    """
    Calculate the secondary photoelectron heating rate
    """
    cdef int sz_r, sz_nu
    cdef int r, nu
    cdef double eint
    cdef double trma, trmb, fonea, foneb, ftwoa, ftwob, yone, ytwo

    sz_r  = jnur.shape[1]
    sz_nu = xsec.shape[0]
    cdef double pev = planck/elvolt

    cdef double[::1] retarr = np.zeros((sz_r), dtype=DTYPE)

    #print "Performing numerical integration over frequency to get secondary photoelectron ionization"
    with nogil:
        for r in para.prange(0,sz_r):
            eint = 0.0
            yone = 1.0 * (1.0 - xe[r]**0.2663)**1.3163
            ytwo = 3.9811 * xe[r]**0.4 * (1.0 - xe[r]**0.34)**2.0
            for nu in range(0,sz_nu-1):
                # 2.41775E14 = 1.602E-19/6.626E-34
                # 4.1361E-15 = 6.626E-34/1.602E-19
                # Use the Ricotti et al. (2002) solution
                if flip == 0:
                    ###  H I  ###
                    if pev*nuarr[nu]-HIip < 11.0:
                        trma = 0.0
                    else:
                        trma = elvolt*(pev*nuarr[nu]-HIip) * (1.0 - yone + ytwo*(11.0/(pev*nuarr[nu]-HIip))**0.7)
                    if pev*nuarr[nu+1]-HIip < 11.0:
                        trmb = 0.0
                    else:
                        trmb = elvolt*(pev*nuarr[nu+1]-HIip) * (1.0 - yone + ytwo*(11.0/(pev*nuarr[nu+1]-HIip))**0.7)
                elif flip == 1:
                    ###  D I  ###
                    if pev*nuarr[nu]-DIip < 11.0:
                        trma = 0.0
                    else:
                        trma = elvolt*(pev*nuarr[nu]-DIip) * (1.0 - yone + ytwo*(11.0/(pev*nuarr[nu]-DIip))**0.7)
                    if pev*nuarr[nu+1]-DIip < 11.0:
                        trmb = 0.0
                    else:
                        trmb = elvolt*(pev*nuarr[nu+1]-DIip) * (1.0 - yone + ytwo*(11.0/(pev*nuarr[nu+1]-DIip))**0.7)
                elif flip == 2:
                    ###  He I  ###
                    if pev*nuarr[nu]-HeIip < 11.0:
                        trma = 0.0
                    else:
                        trma = elvolt*(pev*nuarr[nu]-HeIip) * (1.0 - yone + ytwo*(11.0/(pev*nuarr[nu]-HeIip))**0.7)
                    if pev*nuarr[nu+1]-HeIip < 11.0:
                        trmb = 0.0
                    else:
                        trmb = elvolt*(pev*nuarr[nu+1]-HeIip) * (1.0 - yone + ytwo*(11.0/(pev*nuarr[nu+1]-HeIip))**0.7)
                else:
                    with gil:
                        print "ERROR :: SECONDARY HEATING NOT IMPLEMENTED FOR THAT ION"
                        print "I will now throw a zero-division error..."
                        print 1/0
                ### Now calculate the integral
                eint = eint + 0.5*(xsec[nu]*jnur[nu,r]*trma/(planck*nuarr[nu]) + xsec[nu+1]*jnur[nu+1,r]*trmb/(planck*nuarr[nu+1])) * (nuarr[nu+1]-nuarr[nu])
            retarr[r] = eint
    return np.asarray(retarr)


def cool_rate(double[::1] total_heat not None,
              double[::1] edensity not None,
              double[::1] densitynH not None,
              double[::1] prof_YHI not None,
              double[::1] prof_YHeI not None,
              double[::1] prof_YHeII not None,
              double prim_He, double redshift):

    cdef int sz_r
    cdef int r, c
    cdef double tfunc
    cdef double cool_colexc_HI, cool_colexc_HeI, cool_colexc_HeII, cool_colexc
    cdef double cool_colion_HI, cool_colion_HeI, cool_colion_HeII, cool_colion_HeS, cool_colion
    cdef double cool_rec_HII, cool_rec_HeII, cool_rec_HeIII, cool_rec
    cdef double cool_diel, gf, cool_brem, cool_comp, total_cool

    sz_r  = total_heat.shape[0]

    cdef int sz_c = 1000
    cdef double[::1] temp = np.logspace(3.0,6.0,sz_c)
    cdef double[:,::1] coolfunc = np.zeros((sz_r, sz_c), dtype=DTYPE)

    with nogil:
        for r in para.prange(0,sz_r):
            for c in range(0,sz_c):
                tfunc = 1.0 / (1.0 + csqrt(temp[c]/1.0E5))
                # Collisional excitation cooling (Black 1981, Cen 1992)
                cool_colexc_HI =   (7.50E-19 * tfunc) * cexp(-118348.0/temp[c]) * edensity[r] * densitynH[r] * prof_YHI[r]
                cool_colexc_HeI =  (9.10E-27 * tfunc) * (temp[c]**-0.1687) * cexp(-13179.0/temp[c]) * edensity[r] * edensity[r] * prof_YHeI[r] * prim_He * densitynH[r]
                cool_colexc_HeII = (5.54E-17 * tfunc) * (temp[c]**-0.397) * cexp(-473638.0/temp[c]) * edensity[r] * prof_YHeII[r] * prim_He * densitynH[r]
                cool_colexc = cool_colexc_HI+cool_colexc_HeI+cool_colexc_HeII

                # Collisional Ionization cooling (Shapiro & Kang 1987, Cen 1992)
                cool_colion_HI =   (1.27E-21 * tfunc) * csqrt(temp[c]) * cexp(-157809.1/temp[c]) * edensity[r] * densitynH[r] * prof_YHI[r]
                cool_colion_HeI =  (9.38E-22 * tfunc) * csqrt(temp[c]) * cexp(-285335.4/temp[c]) * edensity[r] * densitynH[r] * prim_He * prof_YHeI[r]
                cool_colion_HeII = (4.95E-22 * tfunc) * csqrt(temp[c]) * cexp(-631515.0/temp[c]) * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]
                cool_colion_HeS  = (5.01E-27 * tfunc) * (temp[c]**-0.1687) * cexp(-55338.0/temp[c]) * edensity[r] * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]
                cool_colion = (cool_colion_HI+cool_colion_HeI+cool_colion_HeII+cool_colion_HeS) #* 2 # TT thinks Cen rates are wrong

                # Recombination cooling (Black 1981, Spitzer 1978)
                cool_rec_HII   = 8.70E-27 * csqrt(temp[c]) * ((temp[c]/1.0E3)**-0.2) * (1.0/(1.0+(temp[c]/1.0E6)**0.7)) * edensity[r] * (1.0-prof_YHI[r])*densitynH[r]
                cool_rec_HeII  = 1.55E-26 * (temp[c]**0.3647) * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]
                cool_rec_HeIII = 3.48E-26 * csqrt(temp[c]) * ((temp[c]/1.0E3)**-0.2) * (1.0/(1.0+(temp[c]/1.0E6)**0.7)) * edensity[r] * (1.0-prof_YHeI[r]-prof_YHeII[r]) * prim_He * densitynH[r]
                cool_rec = cool_rec_HII+cool_rec_HeII+cool_rec_HeIII

                # Dielectric recombination cooling fo He II (Cen 1992)
                cool_diel = 1.24E-13 * (temp[c]**-1.5) * (1.0 + 0.3*cexp(-94000.0/temp[c])) * cexp(-470000.0/temp[c]) * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]

                # Bremsstrahlung cooling (Black 1981, Spitzer & Hart 1979)
                gf = (1.1 + 0.34*cexp(-((5.5-clog10(temp[c]))**2.0)/3.0))
                cool_brem = 1.43E-27 * csqrt(temp[c]) * gf  * edensity[r] * ((1.0-prof_YHI[r])*densitynH[r] + prof_YHeII[r]*prim_He*densitynH[r] + (1.0-prof_YHeI[r]-prof_YHeII[r])*prim_He*densitynH[r])

                # Compton cooling or heating (Peebles 1971)
                cool_comp = 5.65E-36 * ((1.0+redshift)**4) * (temp[c] - 2.73*(1.0+redshift)) * edensity[r]

                total_cool = (cool_colexc + cool_colion + cool_rec + cool_diel + cool_brem + cool_comp)
                coolfunc[r,c] = total_cool

    return np.asarray(coolfunc)


def thermal_equilibrium_full(double[::1] total_heat not None,
                             double[:,::1] total_cool not None,
                             double[::1] old_temp not None):
    """
    Calculate the temperature corresponding to an equal heating and cooling rate in each shell
    The following rates are mostly compiled from a combination of Cen (1992) and Anninos (1997)
    """
    cdef int sz_r, sz_c, eflag
    cdef int r, c, dmin
    cdef double btmp, dtmp, dtst, pcool, gradval
    
    sz_r = total_cool.shape[0]
    sz_c = total_cool.shape[1]
    cdef double[::1] prof_temperature = np.zeros((sz_r), dtype=DTYPE)

    cdef double[::1] actual_cool = np.zeros(sz_r)

    # Generate a range of temperatures that the code is allowed to use:
    cdef double[::1] temp = np.logspace(3.0,6.0,sz_c)

    with nogil:
        eflag = 0 # No error
        for r in range(0,sz_r):
            for c in range(0,sz_c):
                if (c == sz_c-1) and (dmin == -1): # end of T range and not found intersection
                    eflag = 1
                    dmin = 0
                    break
                if c == 0: # start
                    dmin = -1
                else:
                    if (total_heat[r]>=pcool) and (total_heat[r]<total_cool[r,c]):
                        if dmin == -1:
                            dmin = c-1
                            dtmp = temp[dmin]
                            btmp = old_temp[r]-temp[dmin]
                            if btmp < 0.0: btmp *= -1.0
                        else:
                            dtst = old_temp[r]-temp[c-1]
                            if dtst < 0.0: dtst *= -1.0
                            if dtst < btmp:
                                dmin = c-1
                                dtmp = temp[dmin]
                                btmp = old_temp[r]-temp[dmin]
                                if btmp < 0.0: btmp *= -1.0
                    elif (total_heat[r]<=pcool) and (total_heat[r]>total_cool[r,c]):
                        if dmin == -1:
                            dmin = c-1
                            dtmp = temp[dmin]
                            btmp = old_temp[r]-temp[dmin]
                            if btmp < 0.0: btmp *= -1.0
                        else:
                            dtst = old_temp[r]-temp[c-1]
                            if dtst < 0.0: dtst *= -1.0
                            if dtst < btmp:
                                dmin = c-1
                                dtmp = temp[dmin]
                                btmp = old_temp[r]-temp[dmin]
                                if btmp < 0.0: btmp *= -1.0
                pcool = total_cool[r,c]
            if dmin == -1 or dmin == sz_c - 1:
                prof_temperature[r] = temp[dmin]
            else:
                gradval = (total_cool[r,dmin+1]-total_cool[r,dmin])/(temp[dmin+1]-temp[dmin]) # dcool/dT
                if gradval == 0.0:
                    prof_temperature[r] = 0.5*(temp[dmin+1]-temp[dmin])
                    #with gil:
                    #    np.savetxt("gradvalzero_dmin"+str(dmin)+".dat",np.transpose((temp,coolfunc)))
                else:
                    prof_temperature[r] = (total_heat[r]-(total_cool[r,dmin]-gradval*temp[dmin]))/gradval
                    actual_cool[r] = total_cool[r, dmin] + gradval * (prof_temperature[r] - temp[dmin])
    if eflag == 1: print "ERROR :: HEATING RATE WAS LOWER/HIGHER THAN THE COOLING RATE FUNCTION!"
    return np.asarray(prof_temperature), np.asarray(actual_cool)


def calc_xsec_energy(double[::1] xsec not None,
                     double[::1] xsecengy not None,
                     double[::1] energy not None):
    """
    For a given energy (at each radial coordinate), find the appropriate cross-section
    """
    cdef int sz_r, sz_nu
    cdef int r, nu, numin
    cdef double gradv, intc

    sz_r  = energy.shape[0]
    sz_nu = xsec.shape[0]
    cdef double[::1] retarr = np.zeros((sz_r), dtype=DTYPE)

    #print "Interpolating energy -- cross-section profile"
    with nogil:
        for r in range(0,sz_r):
            numin = -1
            for nu in range(0,sz_nu-1):
                if (energy[r]>xsecengy[nu]) and (energy[r]<xsecengy[nu+1]):
                    numin = nu
                    break
            if numin == -1:
                with gil:
                    raise ValueError("Energy out of range for yfactor (ionizations from recombinations of H+, He+, He++)")
            gradv = (xsec[nu+1]-xsec[nu])/(xsecengy[nu+1]-xsecengy[nu])
            intc  = xsec[nu] - gradv*xsecengy[nu]
            retarr[r] = gradv*energy[r] + intc
    return np.asarray(retarr)


def pressure(double[::1] temp not None,
             double[::1] radius not None,
             double[::1] mpp not None,
             cython_halo.HaloModel hmodel,
             double bturb,
             double Gcons, double kB, double mH):
    """
    Calculate the gas pressure profile (numerically integrate Eq. 21 from Sternberg et al. 2002)
    """
    cdef int sz_r
    cdef int r, x
    cdef double xint

    cdef double xvala, xvalb, cssqa, cssqb
    cdef double rhods, rscale
    cdef double pi = 3.14159265358979

    rhods = hmodel.rhods
    rscale = hmodel.rscale

    # 6.67259E-8 = Newton's gravitational constant
    # 2.795E-7 = 4.0*pi*6.67259E-8/3.0
    # 8.385E-7 = 4.0*pi*6.67259E-8
    # 8.255E7 = 1.381E-16 / 1.673E-24 = kB / mH
    # 12.56637061435 = 4.0*pi
    #cdef double vssq = 2.795E-7 * rhods * rscale * rscale
    #cdef double vgas = 8.385E-7 * rscale * rscale
    cdef double vssq = (4.0*pi*Gcons/3.0) * rhods * rscale * rscale
    cdef double vgas = (4.0*pi*Gcons) * rscale * rscale

    sz_r  = radius.shape[0]
    cdef double[::1] retarr = np.zeros((sz_r), dtype=DTYPE)

    with nogil:
        for r in range(1,sz_r):
            xint = 0.0
            for x in range(0,r):
                xvala = radius[x+1]/rscale
                xvalb = radius[x]/rscale
                cssqa = (((kB/mH)*temp[x+1])/mpp[x+1]) + (0.75*bturb*bturb)
                cssqb = (((kB/mH)*temp[x])/mpp[x]) + (0.75*bturb*bturb)
                if xvalb == 0.0:
                    xint += 0.5*(hmodel.fm(xvala)/(cssqa*xvala*xvala)) * (xvala-xvalb)
                else:
                    xint += 0.5*((hmodel.fm(xvala)/(cssqa*xvala*xvala)) +
                                 hmodel.fm(xvalb)/(cssqb*xvalb*xvalb)) * (xvala-xvalb)
            retarr[r] = cexp(-vssq*xint)
        retarr[0] = 1.0
    return np.asarray(retarr)


def fgasx(double[::1] densitym not None,
          double[::1] radius not None,
          double rscale):
    """
    Calculate the total gas mass within each radius (numerically integrate)
    """
    cdef int sz_r
    cdef int r
    cdef double rint = 0.0

    sz_r  = radius.shape[0]
    cdef double[::1] retarr = np.zeros((sz_r), dtype=DTYPE)

    with nogil:
        for r in range(1,sz_r):
            rint += 0.5*((densitym[r]*(radius[r]/rscale)**2) + (densitym[r-1]*(radius[r-1]/rscale)**2)) * (radius[r]-radius[r-1])/rscale
            retarr[r] = rint
    return np.asarray(retarr)


def mass_integral(double[::1] density not None,
                  double[::1] radius not None,
                  double virialr):
    """
    Calculate the total gas mass within the virial radius (numerically integrate)
    """
    cdef int sz_r
    cdef int r
    cdef double rint = 0.0, grad, intcp, denval

    sz_r  = radius.shape[0]

    #print "Calculating total gas mass"
    with nogil:
        for r in range(0,sz_r-1):
            if (radius[r+1] >= virialr):
                grad  = (density[r+1]-density[r])/(radius[r+1]-radius[r])
                intcp = density[r] - grad*radius[r]
                denval = grad*virialr + intcp
                rint += 0.5*((denval*virialr*virialr) + (density[r]*radius[r]*radius[r])) * (virialr-radius[r])
                break
            else:
                rint += 0.5*((density[r+1]*radius[r+1]*radius[r+1]) + (density[r]*radius[r]*radius[r])) * (radius[r+1]-radius[r])
    return np.asarray(rint)


def coldensprofile(double[::1] density not None,
                   double[::1] radius not None):
    """
    Calculate the HI column density profile (numerically integrate)
    """
    cdef int sz_r
    cdef int r, x
    cdef double rint, tmpdr

    sz_r = radius.shape[0]
    cdef double[::1] retarr = np.zeros((sz_r), dtype=DTYPE)

    # This function behaves weirdly with optimisations on
    # radius[x]**2 - radius[r]**2 valuates nonzero for x == r
    # So this gets special cased to avoid taking sqrt of negative numbers
    with nogil:
        for r in para.prange(0,sz_r):
            rint = 0.0
            for x in range(r,sz_r-1):
                if x == r:
                    tmpdr = 0.0
                else:
                    tmpdr = csqrt(radius[x]**2 - radius[r]**2)
                rint = rint + 0.5 * (csqrt(radius[x+1]**2 - radius[r]**2) + tmpdr) * (density[x+1] - density[x])
            if r == sz_r - 1:
                tmpdr = 0.0
            else:
                tmpdr = csqrt(radius[sz_r-1]**2 - radius[r]**2)
            retarr[r] = 2.0 * density[sz_r-1] * tmpdr - 2.0 * rint
    return np.asarray(retarr)
