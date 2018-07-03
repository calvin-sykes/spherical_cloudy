import numpy as np
import cython_fns
import os

def hubblepar(z, cosmopar):
    Ez = np.sqrt(cosmopar[2] + cosmopar[3] * ((1.0 + z) ** 3.0))
    return Ez * 100.0 * cosmopar[0]

def massconc_Klypin11(mvir, cosmo, redshift=3):
    """
    This is not for M200 and r200 --- use the Prada 2012 implementation
    """
    sys.exit()
    rdshft = np.array([0.5,1.0,2.0,3.0,5.0])
    czero  = np.array([7.08,5.45,3.67,2.83,2.34])
    mzero  = np.array([1.5E17,2.5E15,6.8E13,6.3E12,6.6E11])
    # Redshift 3 relation:
    conc = 2.83 * (mvir/(1.0E12/cosmo[0]))**-0.075 * (1.0 + (mvir/(6.3E12/cosmo[0]))**0.26)
    return conc

def cmin_prada(xv):
    return 3.681 + (5.033-3.681)*(0.5 + np.arctan(6.948*(xv-0.424))/np.pi)

def invsigmin_prada(xv):
    return 1.047 + (1.646-1.047)*(0.5 + np.arctan(7.386*(xv-0.526))/np.pi)

def massconc_Prada12(mvir, cosmopar, redshift=3, steps=100000):
    """
    Prada et al. (2012), MNRAS, 423, 3018
    """
    xval = ((cosmopar[2]/cosmopar[3])**(1.0/3.0))/(1.0+redshift) # Eq 13
    yval = 1.0/(mvir/(1.0E12/cosmopar[0])) # Eq 23b
    xintg = cython_fns.massconc_xint(xval,steps)
    Dx = 2.5 * (cosmopar[3]/cosmopar[2])**(1.0/3.0) * np.sqrt(1.0 + xval**3) * xintg /xval**1.5 # Eq 12
    Bzero = cmin_prada(xval)/cmin_prada(1.393) # Eq18a
    Bone  = invsigmin_prada(xval)/invsigmin_prada(1.393) # Eq 18b
    sigfunc = Dx * 16.9 * yval**0.41 / ( 1.0 + 1.102*(yval**0.20) + 6.22*(yval**0.333) ) # Eq 23a
    sigdash = Bone * sigfunc  # Eq 15
    Csigdash = 2.881 * (1.0 + (sigdash/1.257)**1.022) * np.exp(0.060 / sigdash**2)  # Eq 16
    conc = Bzero * Csigdash
    return conc

def massconc_Eagle(mvir, redshift=0.0):
    if redshift != 0.0:
        raise ValueError("Eagle M-c relation is only valid for z=0.0")

    Ms, cs = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/Mvir_c_Eagle.dat"), unpack=True)

    return np.interp(mvir, Ms, cs)

def get_cosmo(use="planck"):
    if use.lower() == "planck":
        hubble = 0.673
        Omega_b = 0.02224/hubble**2
        Omega_m = 0.315
        Omega_l = 1.0-Omega_m
    else:
        # Use Planck as default
        hubble = 0.673
        Omega_b = 0.02224/hubble**2
        Omega_m = 0.315
        Omega_l = 1.0-Omega_m
    cosmopar = np.array([hubble,Omega_b,Omega_l,Omega_m])
    return cosmopar
