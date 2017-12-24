# To get this running, you must do the following at the command line:
# python arb_prec_setup.py build_ext --inplace
# although I'm not really sure what the --inplace does, I think it means "only valid for this directory"
#

import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.float64
ctypedef np.float_t DTYPE_t
ITYPE = np.int64
ctypedef np.int_t ITYPE_t

cdef extern from "math.h":
    double csqrt "sqrt" (double) nogil
    double cexp "exp" (double) nogil
    double clog "log" (double) nogil
    double clog10 "log10" (double) nogil
    double catan "atan" (double) nogil

#######################
#  GENERIC FUNCTIONS  #
#######################

@cython.wraparound(False)
@cython.boundscheck(False)
def Jnur(np.ndarray[DTYPE_t, ndim=1] density not None,
            np.ndarray[DTYPE_t, ndim=1] radius not None,
            np.ndarray[DTYPE_t, ndim=1] jzero not None,
            np.ndarray[DTYPE_t, ndim=1] xsec not None,
            int nummu):
    cdef int sz_r, sz_nu
    cdef int r, ri, mu, nu, rb, fl
    cdef double rint, rtmp, rtmpc

    sz_r  = radius.shape[0]
    sz_nu = xsec.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] retarr = np.zeros((sz_nu,sz_r), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] coldens = np.zeros((sz_r,nummu), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] muarr = np.linspace(-1.0,1.0,nummu)

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

        #print "Performing numerical integration over cos(theta)"
        for nu in range(sz_nu):
            for r in range(sz_r):
                rint = 0.0
                for mu in range(0,nummu-1):
                    rint += 0.5 * (cexp(-xsec[nu]*coldens[r,mu]) + cexp(-xsec[nu]*coldens[r,mu+1])) * (muarr[mu+1]-muarr[mu])
                retarr[nu,r] = 0.5 * jzero[nu] * rint

    return retarr


@cython.wraparound(False)
@cython.boundscheck(False)
def calc_coldens(np.ndarray[DTYPE_t, ndim=1] density not None,
                np.ndarray[DTYPE_t, ndim=1] radius not None,
                int nummu):
    cdef int sz_r
    cdef int r, ri, mu, rb, fl
    cdef double rint, rtmp, rtmpc

    sz_r  = radius.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] coldens = np.zeros((sz_r,nummu), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] muarr = np.linspace(-1.0,1.0,nummu)

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

    return coldens, muarr


@cython.wraparound(False)
@cython.boundscheck(False)
def calc_coldensPP(np.ndarray[DTYPE_t, ndim=1] density not None,
                np.ndarray[DTYPE_t, ndim=1] radius not None):
    cdef int sz_r
    cdef int r
    cdef double rint

    sz_r  = radius.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] coldens = np.zeros((sz_r), dtype=DTYPE)

    with nogil:
        rint = 0.0
        for r in range(1,sz_r):
            rint += 0.5*(density[sz_r-r]+density[sz_r-r-1])*(radius[sz_r-r]-radius[sz_r-r-1])
            coldens[sz_r-r-1] = rint
    return coldens


@cython.wraparound(False)
@cython.boundscheck(False)
def nint_costheta(np.ndarray[DTYPE_t, ndim=3] coldens not None,
            np.ndarray[DTYPE_t, ndim=2] xsec not None,
            np.ndarray[DTYPE_t, ndim=1] muarr not None,
            np.ndarray[DTYPE_t, ndim=1] jzero not None):
    """
    Integrate over cos(theta)
    """
    cdef int sz_r, sz_nu, sz_mu, sz_ion
    cdef int r, mu, nu, ion
    cdef double rint, tau, taup

    sz_r  = coldens.shape[0]
    sz_ion = coldens.shape[2]
    sz_nu = jzero.shape[0]
    sz_mu = muarr.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] retarr = np.zeros((sz_nu,sz_r), dtype=DTYPE)

    with nogil:
        for nu in range(sz_nu):
            for r in range(sz_r):
                rint = 0.0
                for mu in range(0,sz_mu-1):
                    tau  = xsec[nu,0]*coldens[r,mu,0] + xsec[nu,1]*coldens[r,mu,1] + xsec[nu,2]*coldens[r,mu,2]
                    taup = xsec[nu,0]*coldens[r,mu+1,0] + xsec[nu,1]*coldens[r,mu+1,1] + xsec[nu,2]*coldens[r,mu+1,2]
                    rint += 0.5 * (cexp(-tau) + cexp(-taup)) * (muarr[mu+1]-muarr[mu])
                retarr[nu,r] = 0.5 * jzero[nu] * rint
    return retarr


@cython.wraparound(False)
@cython.boundscheck(False)
def nint_pp(np.ndarray[DTYPE_t, ndim=2] coldens not None,
            np.ndarray[DTYPE_t, ndim=2] xsec not None,
            np.ndarray[DTYPE_t, ndim=1] jzero not None):
    """
    Integrate to get the flux at each radial bin
    """
    cdef int sz_r, sz_nu, sz_ion
    cdef int r, nu, ion
    cdef double tau

    sz_r  = coldens.shape[0]
    sz_ion  = coldens.shape[1]
    sz_nu = jzero.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] retarr = np.zeros((sz_nu,sz_r), dtype=DTYPE)

    with nogil:
        for nu in range(sz_nu):
            for r in range(0,sz_r):
                tau  = 0.0
                for ion in range(0,sz_ion):
                    tau += xsec[nu,ion]*coldens[r,ion]
                retarr[nu,r] = jzero[nu] * cexp(-tau)
    return retarr


@cython.wraparound(False)
@cython.boundscheck(False)
def phionrate(np.ndarray[DTYPE_t, ndim=2] jnur not None,
            np.ndarray[DTYPE_t, ndim=1] xsec not None,
            np.ndarray[DTYPE_t, ndim=1] nuarr not None,
            double planck):
    """
    Calculate the local primary photoionization rate (numerically integrate Eq. 23 from Sternberg et al. 2002)
    """
    cdef int sz_r, sz_nu
    cdef int r, nu
    cdef double nint

    sz_r  = jnur.shape[1]
    sz_nu = xsec.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] retarr = np.zeros((sz_r), dtype=DTYPE)

    #print "Performing numerical integration over frequency"
    for r in range(0,sz_r):
        nint = 0.0
        for nu in range(0,sz_nu-1):
            nint += 0.5*(xsec[nu]*jnur[nu,r]/(planck*nuarr[nu]) + xsec[nu+1]*jnur[nu+1,r]/(planck*nuarr[nu+1])) * (nuarr[nu+1]-nuarr[nu])
        retarr[r] = nint
    return retarr


@cython.wraparound(False)
@cython.boundscheck(False)
def nhydroion(np.ndarray[DTYPE_t, ndim=1] jnu not None,
            np.ndarray[DTYPE_t, ndim=1] nuarr not None,
            double planck, double numin):
    """
    Calculate the local primary photoionization rate (numerically integrate Eq. 23 from Sternberg et al. 2002)
    """
    cdef int sz_nu
    cdef int nu
    cdef double nint = 0.0

    sz_nu = jnu.shape[0]

    for nu in range(0,sz_nu-1):
        if nuarr[nu] >= numin:
            nint += 0.5*(jnu[nu]/(planck*nuarr[nu]) + jnu[nu+1]/(planck*nuarr[nu+1])) * (nuarr[nu+1]-nuarr[nu])
    return nint


@cython.wraparound(False)
@cython.boundscheck(False)
def phheatrate(np.ndarray[DTYPE_t, ndim=2] jnur not None,
            np.ndarray[DTYPE_t, ndim=1] xsec not None,
            np.ndarray[DTYPE_t, ndim=1] nuarr not None,
            double nuion, double planck):
    """
    Calculate the local primary photoionization rate (numerically integrate Eq. 23 from Sternberg et al. 2002)
    """
    cdef int sz_r, sz_nu
    cdef int r, nu
    cdef double nint

    sz_r  = jnur.shape[1]
    sz_nu = xsec.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] retarr = np.zeros((sz_r), dtype=DTYPE)

    #print "Performing numerical integration over frequency"
    for r in range(0,sz_r):
        nint = 0.0
        for nu in range(0,sz_nu-1):
            nint += 0.5*(xsec[nu]*jnur[nu,r]*(planck*(nuarr[nu]-nuion))/(planck*nuarr[nu]) + xsec[nu+1]*jnur[nu+1,r]*(planck*(nuarr[nu+1]-nuion))/(planck*nuarr[nu+1])) * (nuarr[nu+1]-nuarr[nu])
        retarr[r] = nint

    return retarr


@cython.wraparound(False)
@cython.boundscheck(False)
def phheatrate_allion(np.ndarray[DTYPE_t, ndim=2] jnur not None,
                np.ndarray[DTYPE_t, ndim=2] xsec not None,
                np.ndarray[DTYPE_t, ndim=1] nuarr not None,
                np.ndarray[DTYPE_t, ndim=1] nuion not None,
                double planck):
    """
    Calculate the local primary photoionization rate (numerically integrate Eq. 23 from Sternberg et al. 2002)
    """
    cdef int sz_i, sz_r, sz_nu
    cdef int i, r, nu
    cdef double nint

    sz_i  = nuion.shape[0]
    sz_r  = jnur.shape[1]
    sz_nu = xsec.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] retarr = np.zeros((sz_r,sz_i), dtype=DTYPE)

    #print "Performing numerical integration over frequency"
    for i in range(0,sz_i):
        for r in range(0,sz_r):
            nint = 0.0
            for nu in range(0,sz_nu-1):
                nint += 0.5*(xsec[nu,i]*jnur[nu,r]*(planck*(nuarr[nu]-nuion[i]))/(planck*nuarr[nu]) + xsec[nu+1,i]*jnur[nu+1,r]*(planck*(nuarr[nu+1]-nuion[i]))/(planck*nuarr[nu+1])) * (nuarr[nu+1]-nuarr[nu])
            retarr[r,i] = nint
    return retarr


@cython.wraparound(False)
@cython.boundscheck(False)
def scdryrate(np.ndarray[DTYPE_t, ndim=2] jnur not None,
            np.ndarray[DTYPE_t, ndim=1] nuarr not None,
            np.ndarray[DTYPE_t, ndim=1] xsecHI not None,
            np.ndarray[DTYPE_t, ndim=1] xsecDI not None,
            np.ndarray[DTYPE_t, ndim=1] xsecHeI not None,
            np.ndarray[DTYPE_t, ndim=1] xsecHeII not None,
            np.ndarray[DTYPE_t, ndim=1] profHI not None,
            np.ndarray[DTYPE_t, ndim=1] profDI not None,
            np.ndarray[DTYPE_t, ndim=1] profHeI not None,
            np.ndarray[DTYPE_t, ndim=1] profHeII not None,
            np.ndarray[DTYPE_t, ndim=1] xe not None,
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

    cdef np.ndarray[DTYPE_t, ndim=1] retarr = np.zeros((sz_r), dtype=DTYPE)

    #print "Performing numerical integration over frequency to get secondary photoelectron ionization"
    for r in range(0,sz_r):
        eint = 0.0
        if flip == 0: # H I
            yone = 0.3908 * (1.0 - xe[r]**0.4092)**1.7592
            ytwo = 0.6941 * xe[r]**0.2 * (1.0 - xe[r]**0.38)**2.0
            divIP = HIip
#		elif flip == 1: # D I
#			yone = 0.3908 * (1.0 - xe[r]**0.4092)**1.7592
#			ytwo = 0.6941 * xe[r]**0.2 * (1.0 - xe[r]**0.38)**2.0
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
                fonea = 0.0
                ftwoa = 0.0
            else:
                fonea = 1.0
                ftwoa = (28.0/(pev*nuarr[nu]-HIip))**0.4
            if pev*nuarr[nu+1]-HIip < 28.0:
                foneb = 0.0
                ftwob = 0.0
            else:
                foneb = 1.0
                ftwob = (28.0/(pev*nuarr[nu+1]-HIip))**0.4
            trmAa = xsecHI[nu] * ((pev*nuarr[nu]-HIip)/divIP) * (yone*fonea - ytwo*ftwoa)
            trmAb = xsecHI[nu+1] * ((pev*nuarr[nu+1]-HIip)/divIP) * (yone*foneb - ytwo*ftwob)
            ###  D I  ###
            if pev*nuarr[nu]-DIip < 28.0:
                fonea = 0.0
                ftwoa = 0.0
            else:
                fonea = 1.0
                ftwoa = (28.0/(pev*nuarr[nu]-DIip))**0.4
            if pev*nuarr[nu+1]-DIip < 28.0:
                foneb = 0.0
                ftwob = 0.0
            else:
                foneb = 1.0
                ftwob = (28.0/(pev*nuarr[nu+1]-DIip))**0.4
            trmDa = (profDI[r]/profHI[r]) * xsecDI[nu] * ((pev*nuarr[nu]-DIip)/divIP) * (yone*fonea - ytwo*ftwoa)
            trmDb = (profDI[r]/profHI[r]) * xsecDI[nu+1] * ((pev*nuarr[nu+1]-DIip)/divIP) * (yone*foneb - ytwo*ftwob)
            trmDa = 0.0
            trmDb = 0.0
            ###  He I  ###
            if pev*nuarr[nu]-HeIip < 28.0:
                fonea = 0.0
                ftwoa = 0.0
            else:
                fonea = 1.0
                ftwoa = (28.0/(pev*nuarr[nu]-HeIip))**0.4
            if pev*nuarr[nu+1]-HeIip < 28.0:
                foneb = 0.0
                ftwob = 0.0
            else:
                foneb = 1.0
                ftwob = (28.0/(pev*nuarr[nu+1]-HeIip))**0.4
            trmBa = (profHeI[r]/profHI[r]) * xsecHeI[nu] * ((pev*nuarr[nu]-HeIip)/divIP) * (yone*fonea - ytwo*ftwoa)
            trmBb = (profHeI[r]/profHI[r]) * xsecHeI[nu+1] * ((pev*nuarr[nu+1]-HeIip)/divIP) * (yone*foneb - ytwo*ftwob)
            ###  He II  ###
            if pev*nuarr[nu]-HeIIip < 28.0:
                fonea = 0.0
                ftwoa = 0.0
            else:
                fonea = 1.0
                ftwoa = (28.0/(pev*nuarr[nu]-HeIIip))**0.4
            if pev*nuarr[nu+1]-HeIIip < 28.0:
                foneb = 0.0
                ftwob = 0.0
            else:
                foneb = 1.0
                ftwob = (28.0/(pev*nuarr[nu+1]-HeIIip))**0.4
            trmCa = (profHeII[r]/profHI[r]) * xsecHeII[nu] * ((pev*nuarr[nu]-HeIIip)/divIP) * (yone*fonea - ytwo*ftwoa)
            trmCb = (profHeII[r]/profHI[r]) * xsecHeII[nu+1] * ((pev*nuarr[nu+1]-HeIIip)/divIP) * (yone*foneb - ytwo*ftwob)
            ### Now calculate the integral
            eint += 0.5*(((trmAa+trmDa+trmBa+trmCa)*jnur[nu,r]/(1.0E7*planck*nuarr[nu])) + (trmAb+trmDb+trmBb+trmCb)*jnur[nu+1,r]/(1.0E7*planck*nuarr[nu+1])) * (nuarr[nu+1]-nuarr[nu])
        retarr[r] = eint
    return retarr


@cython.wraparound(False)
@cython.boundscheck(False)
def scdryheatrate(np.ndarray[DTYPE_t, ndim=2] jnur not None,
                np.ndarray[DTYPE_t, ndim=1] nuarr not None,
                np.ndarray[DTYPE_t, ndim=1] xsec not None,
                np.ndarray[DTYPE_t, ndim=1] xe not None,
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

    cdef np.ndarray[DTYPE_t, ndim=1] retarr = np.zeros((sz_r), dtype=DTYPE)

    #print "Performing numerical integration over frequency to get secondary photoelectron ionization"
    for r in range(0,sz_r):
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
                    fonea = 0.0
                    ftwoa = 0.0
                else:
                    fonea = 1.0
                    ftwoa = (11.0/(pev*nuarr[nu]-HIip))**0.7
                if pev*nuarr[nu+1]-HIip < 11.0:
                    foneb = 0.0
                    ftwob = 0.0
                else:
                    foneb = 1.0
                    ftwob = (11.0/(pev*nuarr[nu+1]-HIip))**0.7
                trma = 1.0E7*elvolt*(pev*nuarr[nu]-HIip) * (1.0 - yone*fonea + ytwo*ftwoa)
                trmb = 1.0E7*elvolt*(pev*nuarr[nu+1]-HIip) * (1.0 - yone*foneb + ytwo*ftwob)
            elif flip == 1:
                ###  D I  ###
                if pev*nuarr[nu]-DIip < 11.0:
                    fonea = 0.0
                    ftwoa = 0.0
                else:
                    fonea = 1.0
                    ftwoa = (11.0/(pev*nuarr[nu]-DIip))**0.7
                if pev*nuarr[nu+1]-DIip < 11.0:
                    foneb = 0.0
                    ftwob = 0.0
                else:
                    foneb = 1.0
                    ftwob = (11.0/(pev*nuarr[nu+1]-DIip))**0.7
                trma = 1.0E7*elvolt*(pev*nuarr[nu]-DIip) * (1.0 - yone*fonea + ytwo*ftwoa)
                trmb = 1.0E7*elvolt*(pev*nuarr[nu+1]-DIip) * (1.0 - yone*foneb + ytwo*ftwob)
            elif flip == 2:
                ###  He I  ###
                if pev*nuarr[nu]-HeIip < 11.0:
                    fonea = 0.0
                    ftwoa = 0.0
                else:
                    fonea = 1.0
                    ftwoa = (11.0/(pev*nuarr[nu]-HeIip))**0.7
                if pev*nuarr[nu+1]-HeIip < 11.0:
                    foneb = 0.0
                    ftwob = 0.0
                else:
                    foneb = 1.0
                    ftwob = (11.0/(pev*nuarr[nu+1]-HeIip))**0.7
                trma = 1.0E7*elvolt*(pev*nuarr[nu]-HeIip) * (1.0 - yone*fonea + ytwo*ftwoa)
                trmb = 1.0E7*elvolt*(pev*nuarr[nu+1]-HeIip) * (1.0 - yone*foneb + ytwo*ftwob)
            else:
                print "ERROR :: SECONDARY HEATING NOT IMPLEMENTED FOR THAT ION"
                print "I will now throw a zero-division error..."
                print 1/0
            ### Now calculate the integral
            eint += 0.5*(xsec[nu]*jnur[nu,r]*trma/(1.0E7*planck*nuarr[nu]) + xsec[nu+1]*jnur[nu+1,r]*trmb/(1.0E7*planck*nuarr[nu+1])) * (nuarr[nu+1]-nuarr[nu])
        retarr[r] = eint
    return retarr


@cython.wraparound(False)
@cython.boundscheck(False)
def thermal_equilibrium(np.ndarray[DTYPE_t, ndim=1] total_heat not None,
                        np.ndarray[DTYPE_t, ndim=1] edensity not None,
                        np.ndarray[DTYPE_t, ndim=1] densitynH not None,
                        np.ndarray[DTYPE_t, ndim=1] prof_YHI not None,
                        np.ndarray[DTYPE_t, ndim=1] prof_YHeI not None,
                        np.ndarray[DTYPE_t, ndim=1] prof_YHeII not None,
                        double prim_He, double redshift):
    """
    Calculate the temperature corresponding to an equal heating and cooling rate in each shell
    The following rates are mostly compiled from a combination of Cen (1992) and Anninos (1997)
    """
    cdef int sz_r, eflag
    cdef int r, c, cmin
    cdef double dmin, dtmp, gradval
    cdef double cool_colexc_HI, cool_colexc_HeI, cool_colexc_HeII, cool_colexc
    cdef double cool_colion_HI, cool_colion_HeI, cool_colion_HeII, cool_colion_HeS, cool_colion
    cdef double cool_rec_HII, cool_rec_HeII, cool_rec_HeIII, cool_rec
    cdef double cool_diel, cool_brem, cool_comp, total_cool

    sz_r  = total_heat.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] prof_temperature = np.zeros((sz_r), dtype=DTYPE)

    # Generate a range of temperatures that the code is allowed to use:
    cdef int sz_c = 500
    cdef np.ndarray[DTYPE_t, ndim=1] temp = 10.0**np.linspace(3.0,6.0,sz_c)
    cdef np.ndarray[DTYPE_t, ndim=1] coolfunc = np.zeros((sz_c), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] coolfuncL = np.zeros((sz_c), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] coolfuncM = np.zeros((sz_c), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] coolfuncR = np.zeros((sz_c), dtype=DTYPE)

    eflag = 0 # No error
    for r in range(0,sz_r):
        cmin=0
        for c in range(0,sz_c):
            # Collisional excitation cooling (Black 1981, Cen 1992)
            cool_colexc_HI =   (7.50E-19 / (1.0 + csqrt(temp[c]/1.0E5))) * cexp(-118348.0/temp[c]) * edensity[r] * prof_YHI[r] * densitynH[r]
            cool_colexc_HeI =  (9.10E-27 / (1.0 + csqrt(temp[c]/1.0E5))) * (temp[c]**-0.1687) * cexp(-13179.0/temp[c]) * edensity[r] * edensity[r] * prof_YHeI[r] * prim_He * densitynH[r]
            cool_colexc_HeII = (5.54E-17 / (1.0 + csqrt(temp[c]/1.0E5))) * (temp[c]**-0.397) * cexp(-473638.0/temp[c]) * edensity[r] * prof_YHeII[r] * prim_He * densitynH[r]
            cool_colexc = cool_colexc_HI+cool_colexc_HeI+cool_colexc_HeII

            # Collisional Ionization cooling (Shapiro & Kang 1987, Cen 1992)
            cool_colion_HI =   (1.27E-21 / (1.0 + csqrt(temp[c]/1.0E5))) * csqrt(temp[c]) * cexp(-157809.1/temp[c]) * edensity[r] * densitynH[r] * prof_YHI[r]
            cool_colion_HeI =  (9.38E-22 / (1.0 + csqrt(temp[c]/1.0E5))) * csqrt(temp[c]) * cexp(-285335.4/temp[c]) * edensity[r] * densitynH[r] * prim_He * prof_YHeI[r]
            cool_colion_HeII = (4.95E-22 / (1.0 + csqrt(temp[c]/1.0E5))) * csqrt(temp[c]) * cexp(-631515.0/temp[c]) * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]
            cool_colion_HeS  = (5.01E-27 / (1.0 + csqrt(temp[c]/1.0E5))) * (temp[c]**-0.1687) * cexp(-55338.0/temp[c]) * edensity[r] * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]
            cool_colion = cool_colion_HI+cool_colion_HeI+cool_colion_HeII+cool_colion_HeS

            # Recombination cooling (Black 1981, Spitzer 1978)
            cool_rec_HII   = 8.70E-27 * csqrt(temp[c]) * ((temp[c]/1.0E3)**-0.2) * (1.0/(1.0+(temp[c]/1.0E6)**0.7)) * edensity[r] * (1.0-prof_YHI[r])*densitynH[r]
            cool_rec_HeII  = 1.55E-26 * (temp[c]**0.3647) * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]
            cool_rec_HeIII = 3.48E-26 * csqrt(temp[c]) * ((temp[c]/1.0E3)**-0.2) * (1.0/(1.0+(temp[c]/1.0E6)**0.7)) * edensity[r] * (1.0-prof_YHeI[r]-prof_YHeII[r]) * prim_He * densitynH[r]
            cool_rec = cool_rec_HII+cool_rec_HeII+cool_rec_HeIII

            # Dielectric recombination cooling fo He II (Cen 1992)
            cool_diel = 1.24E-13 * (temp[c]**-1.5) * (1.0 + 0.3*cexp(-94000.0/temp[c])) * cexp(-470000.0/temp[c]) * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]

            # Bremsstrahlung cooling (Black 1981, Spitzer & Hart 1979)
            cool_brem = 1.43E-27 * csqrt(temp[c]) * (1.1 + 0.34*cexp(-((5.5-clog10(temp[c]))**2.0)/3.0)) * edensity[r] * ((1.0-prof_YHI[r])*densitynH[r] + prof_YHeII[r]*prim_He*densitynH[r] + (1.0-prof_YHeI[r]-prof_YHeII[r])*prim_He*densitynH[r])

            # Compton cooling or heating (Peebles 1971)
            cool_comp = 5.65E-36 * ((1.0+redshift)**4) * (temp[c] - 2.73*(1.0+redshift)) * edensity[r]

            total_cool = (cool_colexc + cool_colion + cool_rec + cool_diel + cool_brem + cool_comp)
            coolfunc[c] = total_cool

            if r==0:
                coolfuncL[c] = total_cool

            if (c == sz_c-1) and (total_heat[r]-total_cool > 0.0): # Total heat was always above the cooling function -- set the temperature to the maximum
                eflag = 1
                cmin = sz_c-1
                break
            if total_heat[r]-total_cool < 0.0:
                break
            if c == 0:
                cmin = 0
                dtmp = total_heat[r] - total_cool
                dmin = total_heat[r] - total_cool
                if dmin < 0.0: dmin *= -1.0
            else:
                dtmp = total_heat[r] - total_cool
                if dtmp < 0.0: dtmp *= -1.0
                if dtmp < dmin:
                    cmin = c
                    dmin = dtmp
        if cmin == 0:
            prof_temperature[r] = temp[cmin]
        elif cmin == sz_c-1:
            prof_temperature[r] = temp[cmin]
        else:
            gradval = (coolfunc[cmin+1]-coolfunc[cmin])/(temp[cmin+1]-temp[cmin])
            if gradval == 0.0:
                prof_temperature[r] = 0.5*(temp[cmin+1]-temp[cmin])
                np.savetxt("gradvalzero_cmin"+str(cmin)+".dat",np.transpose((temp,coolfunc)))
            else:
                prof_temperature[r] = (total_heat[r]-(coolfunc[cmin]-gradval*temp[cmin]))/gradval
    if eflag == 1: print "ERROR :: HEATING RATE WAS HIGHER THAN THE COOLING RATE!"
    return prof_temperature


@cython.wraparound(False)
@cython.boundscheck(False)
def thermal_equilibrium_full(np.ndarray[DTYPE_t, ndim=1] total_heat not None,
                            np.ndarray[DTYPE_t, ndim=1] old_temp not None,
                            np.ndarray[DTYPE_t, ndim=1] edensity not None,
                            np.ndarray[DTYPE_t, ndim=1] densitynH not None,
                            np.ndarray[DTYPE_t, ndim=1] prof_YHI not None,
                            np.ndarray[DTYPE_t, ndim=1] prof_YHeI not None,
                            np.ndarray[DTYPE_t, ndim=1] prof_YHeII not None,
                            double prim_He, double redshift):
    """
    Calculate the temperature corresponding to an equal heating and cooling rate in each shell
    The following rates are mostly compiled from a combination of Cen (1992) and Anninos (1997)
    """
    cdef int sz_r, eflag
    cdef int r, c, dmin
    cdef double btmp, dtmp, dtst, pcool, gradval
    cdef double cool_colexc_HI, cool_colexc_HeI, cool_colexc_HeII, cool_colexc
    cdef double cool_colion_HI, cool_colion_HeI, cool_colion_HeII, cool_colion_HeS, cool_colion
    cdef double cool_rec_HII, cool_rec_HeII, cool_rec_HeIII, cool_rec
    cdef double cool_diel, cool_brem, cool_comp, total_cool

    sz_r  = total_heat.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] prof_temperature = np.zeros((sz_r), dtype=DTYPE)

    # Generate a range of temperatures that the code is allowed to use:
    cdef int sz_c = 1000
    cdef np.ndarray[DTYPE_t, ndim=1] temp = 10.0**np.linspace(3.0,6.0,sz_c)
    cdef np.ndarray[DTYPE_t, ndim=1] coolfunc = np.zeros((sz_c), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] coolfuncL = np.zeros((sz_c), dtype=DTYPE)

    eflag = 0 # No error
    for r in range(0,sz_r):
        for c in range(0,sz_c):
            # Collisional excitation cooling (Black 1981, Cen 1992)
            cool_colexc_HI =   (7.50E-19 / (1.0 + csqrt(temp[c]/1.0E5))) * cexp(-118348.0/temp[c]) * edensity[r] * prof_YHI[r] * densitynH[r]
            cool_colexc_HeI =  (9.10E-27 / (1.0 + csqrt(temp[c]/1.0E5))) * (temp[c]**-0.1687) * cexp(-13179.0/temp[c]) * edensity[r] * edensity[r] * prof_YHeI[r] * prim_He * densitynH[r]
            cool_colexc_HeII = (5.54E-17 / (1.0 + csqrt(temp[c]/1.0E5))) * (temp[c]**-0.397) * cexp(-473638.0/temp[c]) * edensity[r] * prof_YHeII[r] * prim_He * densitynH[r]
            cool_colexc = cool_colexc_HI+cool_colexc_HeI+cool_colexc_HeII

            # Collisional Ionization cooling (Shapiro & Kang 1987, Cen 1992)
            cool_colion_HI =   (1.27E-21 / (1.0 + csqrt(temp[c]/1.0E5))) * csqrt(temp[c]) * cexp(-157809.1/temp[c]) * edensity[r] * densitynH[r] * prof_YHI[r]
            cool_colion_HeI =  (9.38E-22 / (1.0 + csqrt(temp[c]/1.0E5))) * csqrt(temp[c]) * cexp(-285335.4/temp[c]) * edensity[r] * densitynH[r] * prim_He * prof_YHeI[r]
            cool_colion_HeII = (4.95E-22 / (1.0 + csqrt(temp[c]/1.0E5))) * csqrt(temp[c]) * cexp(-631515.0/temp[c]) * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]
            cool_colion_HeS  = (5.01E-27 / (1.0 + csqrt(temp[c]/1.0E5))) * (temp[c]**-0.1687) * cexp(-55338.0/temp[c]) * edensity[r] * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]
            cool_colion = cool_colion_HI+cool_colion_HeI+cool_colion_HeII+cool_colion_HeS

            # Recombination cooling (Black 1981, Spitzer 1978)
            cool_rec_HII   = 8.70E-27 * csqrt(temp[c]) * ((temp[c]/1.0E3)**-0.2) * (1.0/(1.0+(temp[c]/1.0E6)**0.7)) * edensity[r] * (1.0-prof_YHI[r])*densitynH[r]
            cool_rec_HeII  = 1.55E-26 * (temp[c]**0.3647) * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]
            cool_rec_HeIII = 3.48E-26 * csqrt(temp[c]) * ((temp[c]/1.0E3)**-0.2) * (1.0/(1.0+(temp[c]/1.0E6)**0.7)) * edensity[r] * (1.0-prof_YHeI[r]-prof_YHeII[r]) * prim_He * densitynH[r]
            cool_rec = cool_rec_HII+cool_rec_HeII+cool_rec_HeIII

            # Dielectric recombination cooling fo He II (Cen 1992)
            cool_diel = 1.24E-13 * (temp[c]**-1.5) * (1.0 + 0.3*cexp(-94000.0/temp[c])) * cexp(-470000.0/temp[c]) * edensity[r] * densitynH[r] * prim_He * prof_YHeII[r]

            # Bremsstrahlung cooling (Black 1981, Spitzer & Hart 1979)
            cool_brem = 1.43E-27 * csqrt(temp[c]) * (1.1 + 0.34*cexp(-((5.5-clog10(temp[c]))**2.0)/3.0)) * edensity[r] * ((1.0-prof_YHI[r])*densitynH[r] + prof_YHeII[r]*prim_He*densitynH[r] + (1.0-prof_YHeI[r]-prof_YHeII[r])*prim_He*densitynH[r])

            # Compton cooling or heating (Peebles 1971)
            cool_comp = 5.65E-36 * ((1.0+redshift)**4) * (temp[c] - 2.73*(1.0+redshift)) * edensity[r]

            total_cool = (cool_colexc + cool_colion + cool_rec + cool_diel + cool_brem + cool_comp)
            coolfunc[c] = total_cool

            if r==0:
                coolfuncL[c] = total_cool

            if (c == sz_c-1) and (dmin == -1):
                eflag = 1
                dmin = 0
                break
            if c == 0:
                dmin = -1
            else:
                if (total_heat[r]>=pcool) and (total_heat[r]<total_cool):
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
                elif (total_heat[r]<=pcool) and (total_heat[r]>total_cool):
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
            pcool = total_cool
        if dmin == -1:
            prof_temperature[r] = temp[dmin]
        elif dmin == sz_c-1:
            prof_temperature[r] = temp[dmin]
        else:
            gradval = (coolfunc[dmin+1]-coolfunc[dmin])/(temp[dmin+1]-temp[dmin])
            if gradval == 0.0:
                prof_temperature[r] = 0.5*(temp[dmin+1]-temp[dmin])
                np.savetxt("gradvalzero_dmin"+str(dmin)+".dat",np.transpose((temp,coolfunc)))
            else:
                prof_temperature[r] = (total_heat[r]-(coolfunc[dmin]-gradval*temp[dmin]))/gradval
    if eflag == 1: print "ERROR :: HEATING RATE WAS LOWER/HIGHER THAN THE COOLING RATE FUNCTION!"
    return prof_temperature


@cython.wraparound(False)
@cython.boundscheck(False)
def calc_xsec_energy(np.ndarray[DTYPE_t, ndim=1] xsec not None,
                np.ndarray[DTYPE_t, ndim=1] xsecengy not None,
                np.ndarray[DTYPE_t, ndim=1] energy not None):
    """
    For a given energy (at each radial coordinate), find the appropriate cross-section
    """
    cdef int sz_r, sz_nu
    cdef int r, nu, numin
    cdef double gradv, intc

    sz_r  = energy.shape[0]
    sz_nu = xsec.shape[0]
    #cdef np.ndarray[DTYPE_t, ndim=1] retarr = np.interp(energy,xsecengy,xsec)
    cdef np.ndarray[DTYPE_t, ndim=1] retarr = np.zeros((sz_r), dtype=DTYPE)

    #print "Interpolating energy -- cross-section profile"
    for r in range(0,sz_r):
        numin = -1
        for nu in range(0,sz_nu-1):
            if (energy[r]>xsecengy[nu]) and (energy[r]<xsecengy[nu+1]):
                numin = nu
                break
        if numin == -1:
            print "ERROR :: energy out of range for yfactor (ionizations from recombinaitons of H+, He+, He++)"
        gradv = (xsec[nu+1]-xsec[nu])/(xsecengy[nu+1]-xsecengy[nu])
        intc  = xsec[nu] - gradv*xsecengy[nu]
        retarr[r] = gradv*energy[r] + intc
    return retarr


ctypedef double (*fp_massint)(double x)


@cython.boundscheck(False)
cdef double NFW_fm(double x):
    return 3.0*(clog(1.0+x) - x/(1.0+x))


@cython.boundscheck(False)
cdef double Burkert_fm(double x):
    return 1.5*(0.5*clog(1.0+(x**2.0)) + clog(1.0+x) - catan(x))


cdef double acore


@cython.boundscheck(False)
cdef double Cored_fm(double x):
    return 3.0*( x/((acore-1.0)*(1.0+x)) + ( acore**2.0 * clog(1.0+x/acore) + (1.0-2.0*acore)*clog(1.0+x) ) / (1.0-acore)**2.0 )


@cython.wraparound(False)
@cython.boundscheck(False)
def pressure(np.ndarray[DTYPE_t, ndim=1] temp not None,
             np.ndarray[DTYPE_t, ndim=1] radius not None,
             np.ndarray[DTYPE_t, ndim=1] mpp not None,
             hmodel,
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

    cdef char* name
    cdef fp_massint fm

    if hmodel.name == "NFW":
        fm = &NFW_fm
    elif hmodel.name == "Burkert":
        fm = &Burkert_fm
    elif hmodel.name == "Cored":
        fm = &Cored_fm
        acore = hmodel.acore

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
    cdef np.ndarray[DTYPE_t, ndim=1] retarr = np.zeros((sz_r), dtype=DTYPE)

    print "Calculating pressure profile"
    for r in range(1,sz_r):
        xint = 0.0
        for x in range(0,r):
            xvala = radius[x+1]/rscale
            xvalb = radius[x]/rscale
            cssqa = (((kB/mH)*temp[x+1])/mpp[x+1])# + (0.75*bturb*bturb)
            cssqb = (((kB/mH)*temp[x])/mpp[x])# + (0.75*bturb*bturb)
            if xvalb == 0.0:
                xint += 0.5*(fm(xvala)/(cssqa*xvala*xvala)) * (xvala-xvalb)
            else:
                xint += 0.5*((fm(xvala)/(cssqa*xvala*xvala)) + fm(xvalb)/(cssqb*xvalb*xvalb)) * (xvala-xvalb)
        retarr[r] = cexp(-vssq*xint)
    retarr[0] = 1.0
    return retarr


@cython.wraparound(False)
@cython.boundscheck(False)
def fgasx(np.ndarray[DTYPE_t, ndim=1] densitym not None,
        np.ndarray[DTYPE_t, ndim=1] radius not None,
        double rscale):
    """
    Calculate the total gas mass within the each radius (numerically integrate)
    """
    cdef int sz_r
    cdef int r
    cdef double rint = 0.0

    sz_r  = radius.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] retarr = np.zeros((sz_r), dtype=DTYPE)

    print "Calculating total gas mass"
    for r in range(1,sz_r):
        rint += 0.5*((densitym[r]*(radius[r]/rscale)**2) + (densitym[r-1]*(radius[r-1]/rscale)**2)) * (radius[r]-radius[r-1])/rscale
        retarr[r] = rint
    return retarr


@cython.wraparound(False)
@cython.boundscheck(False)
def mass_integral(np.ndarray[DTYPE_t, ndim=1] density not None,
                np.ndarray[DTYPE_t, ndim=1] radius not None,
                float virialr):
    """
    Calculate the total gas mass within the virial radius (numerically integrate)
    """
    cdef int sz_r
    cdef int r
    cdef double rint = 0.0, grad, intcp, denval

    sz_r  = radius.shape[0]

    #print "Calculating total gas mass"
    for r in range(0,sz_r-1):
        if (radius[r+1] >= virialr):
            grad  = (density[r+1]-density[r])/(radius[r+1]-radius[r])
            intcp = density[r] - grad*radius[r]
            denval = grad*virialr + intcp
            rint += 0.5*((denval*virialr*virialr) + (density[r]*radius[r]*radius[r])) * (virialr-radius[r])
            break
        else:
            rint += 0.5*((density[r+1]*radius[r+1]*radius[r+1]) + (density[r]*radius[r]*radius[r])) * (radius[r+1]-radius[r])
    return rint


@cython.wraparound(False)
@cython.boundscheck(False)
def coldensprofile(np.ndarray[DTYPE_t, ndim=1] density not None,
                np.ndarray[DTYPE_t, ndim=1] radius not None):
    """
    Calculate the HI column density profile (numerically integrate)
    """
    cdef int sz_r
    cdef int r, x
    cdef double rint

    sz_r = radius.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] retarr = np.zeros((sz_r), dtype=DTYPE)

    for r in range(0,sz_r):
        rint = 0.0
        for x in range(r,sz_r-1):
            rint += 0.5*(csqrt(radius[x+1]**2 - radius[r]**2) + csqrt(radius[x]**2 - radius[r]**2)) * (density[x+1]-density[x])
        retarr[r] = 2.0*density[sz_r-1]*csqrt(radius[sz_r-1]**2 - radius[r]**2) - 2.0*rint
    return retarr


@cython.wraparound(False)
@cython.boundscheck(False)
def massconc_xint(double xval, int steps):
    """
    Numerically integrate Equation 12 from Prada et al. 2012 over x
    """
    cdef int x
    cdef double xint = 0.0

    cdef np.ndarray[DTYPE_t, ndim=1] xarr = np.linspace(0.0,xval,steps)

    #print "Numerically integrating parameters in the mass-concentration relation"
    for x in range(0,steps-1):
        xint += 0.5*( (xarr[x+1]/(1.0+xarr[x+1]**3.0))**1.5 + (xarr[x]/(1.0+xarr[x]**3.0))**1.5 ) * (xarr[x+1]-xarr[x])
    return xint
