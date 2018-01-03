"""
See the following website for the relevant data
http://www-cfadc.phy.ornl.gov/astro/ps/data/
"""
import numpy as np
import cython_fns

# Sharing function
def yfactor(ion, xsec_HI, xsec_HeI, xsec_HeII, prof_HI, prof_HeI, prof_HeII, engyval, energy):
    """
    Sharing function - defined by Eq 22 from Jenkins (2013)
    """
    xsecrad_HI   = cython_fns.calc_xsec_energy(xsec_HI, engyval, energy)
    xsecrad_HeI  = cython_fns.calc_xsec_energy(xsec_HeI, engyval, energy)
    xsecrad_HeII = cython_fns.calc_xsec_energy(xsec_HeII, engyval, energy)
    output = np.zeros(prof_HI.size)
    divarr = (prof_HI*xsecrad_HI + prof_HeI*xsecrad_HeI + prof_HeII*xsecrad_HeII)
    w = np.where(divarr != 0.0)
    if ion == "HI":
        output[w] = (prof_HI*xsecrad_HI)[w] / divarr[w]
    elif ion == "HeI":
        output[w] = (prof_HeI*xsecrad_HeI)[w] / divarr[w]
    elif ion == "HeII":
        output[w] = (prof_HeII*xsecrad_HeII)[w] / divarr[w]
    return output

# Recombination coefficients
## H I
def alphaB_HI(temper):
    """
    Equation 14.6 from Draine book
    """
    #return 3.403E-10 * temper**-0.7827
    return 2.54E-13 * (temper/1.0E4)**(-0.8163 - 0.0208*np.log(temper/1.0E4))

def alpha1s_HI(temper):
    """
    Table 14.1 from Draine book
    """
    return 1.58E-13 * (temper/1.0E4)**(-0.540-0.017*np.log(temper/1.0E4))

## He I
def alphaB_HeI(temper):
    """
    A fit to the data in Hummer & Storey (1998)
    """
    #return 1.613E-10 * temper**-0.6872
    return 9.03E-14 * (temper/4.0E4)**(-0.830-0.0177*np.log(temper/4.0E4))

def alpha1s_HeI(temper):
    """
    Equation B12 from Jenkins (2013)
    """
    return 1.54E-13 * (temper/1.0E4)**(-0.486)

## He II
def alphaB_HeII(temper):
    #return 1.395E-09 * temper**-0.7446
    """
    Equation 14.6 from Draine book
    """
    #return 5.18E-13 * (temper/4.0E4)**(-0.833 - 0.035*np.log(temper/4.0E4))
    return 5.08E-13 * (temper/4.0E4)**(-0.8163 - 0.0208*np.log(temper/4.0E4))

def alpha1s_HeII(temper):
    """
    Equation B2 from Jenkins (2013)
    """
    return 3.16E-13 * (temper/4.0E4)**(-0.540 - 0.017*np.log(temper/4.0E4))

def alpha2p_HeII(temper):
    """
    Equation B4 from Jenkins (2013)
    """
    return 1.07E-13 * (temper/4.0E4)**(-0.681 - 0.061*np.log(temper/4.0E4))

def alphaeff2s_HeII(temper):
    """
    Equation B6 from Jenkins (2013)
    """
    return 1.68E-13 * (temper/4.0E4)**(-0.7205-0.0081*np.log(temper/4.0E4))

def alpha2s_HeII(temper):
    """
    Equation B8 from Jenkins (2013)
    """
    return 4.68E-14 * (temper/4.0E4)**(-0.537-0.019*np.log(temper/4.0E4))

def rate_function(engy, Et, Emx, Eo, so, ya, P, yw, yo, y1):
    xsec = np.zeros(engy.size)
    x = engy/Eo - yo
    y = np.sqrt(x**2 + y1**2)
    Fy = ((x-1.0)**2 + yw**2) * y**(0.5*P - 5.5) * (1.0 + np.sqrt(y/ya))**(-1.0*P)
    w = np.where((engy>=Et)&(engy<=Emx))
    xsec[w] = 1.0E-18 * so * Fy[w]
    return xsec

def other(ion,engy,profdens,densitynH,Yprof,electrondensity,phelxs,prof_temperature,elID,kB,elvolt):
    eHI   = elID["H I"].id
    eHeI  = elID["He I"].id
    eHeII = elID["He II"].id
    npts = np.size(electrondensity)
    if ion == "H I":
        prof_HI = profdens[:,eHI]
        prof_HeII = profdens[:,eHeII]
        w_HI    = np.where(prof_HI!=0.0)
        # Equation B3 from Jenkins et al. (2013)
        B3_HI   = np.zeros(npts)
        trma = alpha1s_HeII(prof_temperature) * yfactor("HI", phelxs[:,eHI], phelxs[:,eHeI], phelxs[:,eHeII], profdens[:,eHI], profdens[:,eHeI], profdens[:,eHeII], engy, elID["He II"].ip+1.0E-7*kB*prof_temperature/elvolt)
        trmb = alpha2p_HeII(prof_temperature) * (1.0 + yfactor("HI", phelxs[:,eHI], phelxs[:,eHeI], phelxs[:,eHeII], profdens[:,eHI], profdens[:,eHeI], profdens[:,eHeII], engy, 40.8*np.ones(npts)))
        trmc = 2.42*alpha2s_HeII(prof_temperature)
        trmd = (alphaB_HeII(prof_temperature) - alpha2p_HeII(prof_temperature) - alphaeff2s_HeII(prof_temperature)) * yfactor("HI", phelxs[:,eHI], phelxs[:,eHeI], phelxs[:,eHeII], profdens[:,eHI], profdens[:,eHeI], profdens[:,eHeII], engy, 50.0*np.ones(npts))
        trme = 1.42*(alphaeff2s_HeII(prof_temperature) - alpha2s_HeII(prof_temperature))
        tmpHe = 1.0-Yprof[eHeI]-Yprof[eHeII]
        B3_HI[w_HI]   = ((tmpHe*densitynH*elID["He I"].abund)*electrondensity * ( trma + trmb + trmc + trmd + trme ))[w_HI]/prof_HI[w_HI]
        # Equation B7 from Jenkins et al. (2013)
        B7_HI   = np.zeros(npts)
        trma = alpha1s_HeI(prof_temperature) * yfactor("HI", phelxs[:,eHI], phelxs[:,eHeI], phelxs[:,eHeII], profdens[:,eHI], profdens[:,eHeI], profdens[:,eHeII], engy, elID["He I"].ip+1.0E-7*kB*prof_temperature/elvolt)
        B7_HI[w_HI]   = (prof_HeII*electrondensity * ( trma + 0.96*alphaB_HeI(prof_temperature) ))[w_HI]/prof_HI[w_HI]
        tmpH   = 1.0-Yprof[eHI]
        recomb = np.zeros(npts)
        recomb[w_HI]  = ((tmpH*densitynH*elID["H I"].abund*electrondensity) * alpha1s_HI(prof_temperature) * yfactor("HI", phelxs[:,eHI], phelxs[:,eHeI], phelxs[:,eHeII], profdens[:,eHI], profdens[:,eHeI], profdens[:,eHeII], engy, elID["H I"].ip+1.0E-7*kB*prof_temperature/elvolt))[w_HI]/prof_HI[w_HI]
        gammarate = B3_HI + B7_HI + recomb
    elif ion == "D I":
        gammarate = np.zeros(npts)
    elif ion == "He I":
        tmpHe = 1.0-Yprof[eHeI]-Yprof[eHeII]
        prof_HeI = profdens[:,eHeI]
        prof_HeII = profdens[:,eHeII]
        w_HeI   = np.where(prof_HeI!=0.0)
        B2_HeI  = np.zeros(npts)
        B6_HeI  = np.zeros(npts)
        trma = alpha1s_HeII(prof_temperature) * yfactor("HeI", phelxs[:,eHI], phelxs[:,eHeI], phelxs[:,eHeII], profdens[:,eHI], profdens[:,eHeI], profdens[:,eHeII], engy, elID["He II"].ip+1.0E-7*kB*prof_temperature/elvolt)
        trmb = alpha2p_HeII(prof_temperature) * yfactor("HeI", phelxs[:,eHI], phelxs[:,eHeI], phelxs[:,eHeII], profdens[:,eHI], profdens[:,eHeI], profdens[:,eHeII], engy, 40.8*np.ones(npts))
        trmc = (alphaB_HeII(prof_temperature) - alpha2p_HeII(prof_temperature) - alphaeff2s_HeII(prof_temperature)) * yfactor("HeI", phelxs[:,eHI], phelxs[:,eHeI], phelxs[:,eHeII], profdens[:,eHI], profdens[:,eHeI], profdens[:,eHeII], engy, 50.0*np.ones(npts))
        B2_HeI[w_HeI]  = ((tmpHe*densitynH*elID["He I"].abund)*electrondensity * ( trma + trmb + trmc ))[w_HeI] / prof_HeI[w_HeI]
        B6_HeI[w_HeI]  = (prof_HeII*electrondensity * alpha1s_HeII(prof_temperature) * yfactor("HeI", phelxs[:,eHI], phelxs[:,eHeI], phelxs[:,eHeII], profdens[:,eHI], profdens[:,eHeI], profdens[:,eHeII], engy, elID["He I"].ip+1.0E-7*kB*prof_temperature/elvolt))[w_HeI]/prof_HeI[w_HeI]
        gammarate = B2_HeI + B6_HeI
    elif ion == "He II":
        tmpHe = 1.0-Yprof[eHeI]-Yprof[eHeII]
        prof_HeII = profdens[:,eHeII]
        w_HeII  = np.where(prof_HeII!=0.0)
        B1_HeII = np.zeros(npts)
        B1_HeII[w_HeII] = ((tmpHe*densitynH*elID["He I"].abund*electrondensity) * alpha1s_HeII(prof_temperature) * yfactor("HeII", phelxs[:,eHI], phelxs[:,eHeI], phelxs[:,eHeII], profdens[:,eHI], profdens[:,eHeI], profdens[:,eHeII], engy, elID["He II"].ip+1.0E-7*kB*prof_temperature/elvolt))[w_HeII]/prof_HeII[w_HeII]
        gammarate = B1_HeII
    elif ion == "N I":
        gammarate = np.zeros(npts)
    elif ion == "N II":
        gammarate = np.zeros(npts)
    elif ion == "N III":
        gammarate = np.zeros(npts)
    elif ion == "N IV":
        gammarate = np.zeros(npts)
    elif ion == "Si I":
        gammarate = np.zeros(npts)
    elif ion == "Si II":
        gammarate = np.zeros(npts)
    elif ion == "Si III":
        gammarate = np.zeros(npts)
    elif ion == "Si IV":
        gammarate = np.zeros(npts)
    else:
        gammarate = np.zeros(npts)
    return gammarate
