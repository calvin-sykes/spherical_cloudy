import numpy as np
from matplotlib import pyplot as plt

import cython_fns
import constants
import logger

import os

def HMredshifts(version='12'):
    """return list of redshifts H&M field is tabulated at"""
    if version in {'12', '15'}:
        usecols = tuple(range(60))
    elif version == '05':
        usecols = tuple(range(50))
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/radfields/HM{:s}_UVB.dat").format(version), usecols=usecols)
    return data[0,:]    

def extra_interp(elID, nu, Jnu):
    """add finer interpolation to radiation field around ionisation potentials"""
    const = constants.get()
    planck  = const["planck"]
    elvolt  = const["elvolt"]
    ediff = 1.0E-10
    ekeys = elID.keys()
    Jnurev = Jnu[::-1]
    nurev  = nu[::-1]
    Jnuadd = np.zeros(2*len(ekeys) + 2*7) # 7 additional points for secondary heat/ionizations
    nuadd  = np.zeros(2*len(ekeys) + 2*7) # 7 additional points for secondary heat/ionizations
    for i in range(len(ekeys)):
        nup = (elID[ekeys[i]].ip+ediff)*elvolt/planck
        num = (elID[ekeys[i]].ip-ediff)*elvolt/planck
        Jnuvp = np.interp(nup,nurev,Jnurev)
        Jnuvm = np.interp(num,nurev,Jnurev)
        nuadd[2*i]    = nup
        nuadd[2*i+1]  = num
        Jnuadd[2*i]   = Jnuvp
        Jnuadd[2*i+1] = Jnuvm
    # Now include the additional points for secondary heat/ionizations
    ekeysA = ["H I", "D I", "He I", "He II"]
#	ekeysA = ["H I", "He I", "He II"]
    extra = 28.0
    cntr = 2*len(ekeys)
    for i in range(len(ekeysA)):
        nup = (elID[ekeysA[i]].ip+extra+ediff)*elvolt/planck
        num = (elID[ekeysA[i]].ip+extra-ediff)*elvolt/planck
        Jnuvp = np.interp(nup,nurev,Jnurev)
        Jnuvm = np.interp(num,nurev,Jnurev)
        nuadd[2*i+cntr]    = nup
        nuadd[2*i+cntr+1]  = num
        Jnuadd[2*i+cntr]   = Jnuvp
        Jnuadd[2*i+cntr+1] = Jnuvm
    ekeysB = ["H I", "D I", "He I"]
#	ekeysB = ["H I", "He I"]
    extra = 11.0
    cntr = 2*len(ekeys) + 2*4
    for i in range(len(ekeysB)):
        nup = (elID[ekeysB[i]].ip+extra+ediff)*elvolt/planck
        num = (elID[ekeysB[i]].ip+extra-ediff)*elvolt/planck
        Jnuvp = np.interp(nup,nurev,Jnurev)
        Jnuvm = np.interp(num,nurev,Jnurev)
        nuadd[2*i+cntr]    = nup
        nuadd[2*i+cntr+1]  = num
        Jnuadd[2*i+cntr]   = Jnuvp
        Jnuadd[2*i+cntr+1] = Jnuvm
    # Append to the original arrays
    Jnut = np.append(Jnurev,Jnuadd)
    nut  = np.append(nurev,nuadd)
    argsrt = np.argsort(nut,kind='mergesort')

    Jnut = Jnut[argsrt]
    nut  =  nut[argsrt]
    egyt = nut * planck # energies

    return nut, Jnut, egyt

def HMbackground(elID,redshift=3.0, version='12', alpha_UV=0):
    Jnu, nu, discz = _HM_background_impl(elID, redshift, version, alpha_UV)
    logger.log("info", "Using HM{1:s} background at z={0:f}".format(discz, version))
    return Jnu, nu

def _HM_background_impl(elID,redshift, version, alpha_UV):
    if version in {'12', '15'}:
        usecols = tuple(range(60))
    elif version == '05':
        usecols = tuple(range(50))
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/radfields/HM{:s}_UVB.dat").format(version), usecols=usecols)
    rdshlist = data[0,:]
    amin = np.argmin(np.abs(rdshlist-redshift))
    waveAt, Jnut = data[1:,0], data[1:,amin+1]
    waveA = waveAt[1:]*1.0E-10
    #w = np.where(waveAt < 912.0)
    Jnu = Jnut[1:]
    nu = 299792458.0/waveA
    # Interpolate the Haardt & Madau background around each of the ionization potentials
    
    nut, Jnut, egyt = extra_interp(elID, nu, Jnu)

    # Shape parameter (Crighton et al 2015, https://arxiv.org/pdf/1406.4239.pdf)
    if alpha_UV != 0:
        const = constants.get()
        elvolt  = const["elvolt"]
        logJ = np.log10(Jnut)
        e0  = elID["H I"].ip * elvolt
        e1  = 10 * elID["H I"].ip * elvolt
        rge0 = np.logical_and(e0 <= egyt, egyt <= e1)
        rge1 = egyt > e1
        idx1 = np.searchsorted(egyt, e1)
        logJ[rge0] = logJ[rge0] + alpha_UV * np.log10(egyt[rge0] / e0)
        logJ[rge1] = logJ[rge1] + alpha_UV * np.log10(e1 / e0)
        Jnut = 10**logJ

    #plt.figure()
    #plt.plot(np.log10(nut), np.log10(Jnut))
    #plt.show()
    
    return Jnut, nut, rdshlist[amin]

def HM_fiducial(elID,redshift, version):
    return _HM_background_impl(elID, redshift, version, 0.0)[0]

def AGN(elID, intensity=1.0):
    """
    Use an ionising radiation field with shape given by Cloudy's AGN SED.
    The optional intensity factor is relative to the z = 0 Madau & Haardt (2015) UVB.
    """
    wl, nuJnu = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/radfields/agn_n4.radfield"), usecols=(0,1), unpack=True)
    nu = 299792458.0 / (wl * 1e-10)
    Jnu = nuJnu / (4 * np.pi * nu) # 4 pi nu J_nu is output by Cloudy

    # Now load the HM spectrum to get the same frequency scale
    HMdata = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/radfields/HM15_UVB.dat"), usecols=tuple(range(60)))
    rdshlist = HMdata[0,:]
    amin = np.argmin(np.abs(rdshlist - 0.0))
    wl_HM, Jnu_HM = HMdata[1:,0], HMdata[1:,amin+1]
    wl_HM *= 1.0E-10
    nu_HM = 299792458.0 / wl_HM
    Jnu = np.interp(nu_HM, nu, Jnu)
    
    # Now normalise the intensities
    argsrt = np.argsort(nu_HM)
    I_HM = np.trapz(4 * np.pi * Jnu_HM[argsrt], nu_HM[argsrt])
    I = np.trapz(4 * np.pi * Jnu[argsrt], nu_HM[argsrt])
    norm = (I_HM / I) * intensity
    Jnu *= norm 
    
    # Extra interpolation for IPs
    nu, Jnu, _ = extra_interp(elID, nu_HM, Jnu)
    
    return Jnu, nu

def powerlaw(elID):
    try:
        nut, Jnut = np.loadtxt("data/radfields/" + options["radfield"]+".radfield",unpack=True)
        # Now load the HM spectrum to get the same frequency scale
        data = np.loadtxt("HM12_UVB.dat")
        rdshlist = data[0,:]
        amin = np.argmin(np.abs(rdshlist-3.0))
        waveAt, Jnut = data[1:,0], data[1:,amin]
        waveA = waveAt[1:]*1.0E-10
        nu_HM = 299792458.0/waveA
        Jnut = np.interp(nu_HMrev, nut, Jnut)
    except:
        logger.log("critical", "Radiation field file: {0:s} does not exist".format(options["radfield"]+".radfield"))
        sys.exit()

    Jnut, nut, _ = extra_interp(elID, nut, Jnut)
    return Jnut, nut

def test_background(elID):
    const = constants.get()
    planck  = const["planck"]
    elvolt  = const["elvolt"]
    nurev, Jnurev = np.loadtxt("test_continuum3.dat",unpack=True)
    #Jnurev /= (2.0)
    print "Using test background radiation field (table power law -1)"
    ediff = 1.0E-10
    ekeys = elID.keys()
    Jnuadd = np.zeros(2*len(ekeys) + 2*7) # 7 additional points for secondary heat/ionizations
    nuadd  = np.zeros(2*len(ekeys) + 2*7) # 7 additional points for secondary heat/ionizations
    for i in range(len(ekeys)):
        nup = (elID[ekeys[i]].ip+ediff)*elvolt/planck
        num = (elID[ekeys[i]].ip-ediff)*elvolt/planck
        Jnuvp = np.interp(nup,nurev,Jnurev)
        Jnuvm = np.interp(nup,nurev,Jnurev)
        nuadd[2*i]    = nup
        nuadd[2*i+1]  = num
        Jnuadd[2*i]   = Jnuvp
        Jnuadd[2*i+1] = Jnuvm
    # Now include the additional points for secondary heat/ionizations
    ekeysA = ["H I", "D I", "He I", "He II"]
#	ekeysA = ["H I", "He I", "He II"]
    extra = 28.0
    cntr = 2*len(ekeys)
    for i in range(len(ekeysA)):
        nup = (elID[ekeysA[i]].ip+extra+ediff)*elvolt/planck
        num = (elID[ekeysA[i]].ip+extra-ediff)*elvolt/planck
        Jnuvp = np.interp(nup,nurev,Jnurev)
        Jnuvm = np.interp(nup,nurev,Jnurev)
        nuadd[2*i+cntr]    = nup
        nuadd[2*i+cntr+1]  = num
        Jnuadd[2*i+cntr]   = Jnuvp
        Jnuadd[2*i+cntr+1] = Jnuvm
    ekeysB = ["H I", "D I", "He I"]
#	ekeysB = ["H I", "He I"]
    extra = 11.0
    cntr = 2*len(ekeys) + 2*4
    for i in range(len(ekeysB)):
        nup = (elID[ekeysB[i]].ip+extra+ediff)*elvolt/planck
        num = (elID[ekeysB[i]].ip+extra-ediff)*elvolt/planck
        Jnuvp = np.interp(nup,nurev,Jnurev)
        Jnuvm = np.interp(nup,nurev,Jnurev)
        nuadd[2*i+cntr]    = nup
        nuadd[2*i+cntr+1]  = num
        Jnuadd[2*i+cntr]   = Jnuvp
        Jnuadd[2*i+cntr+1] = Jnuvm
    # Append to the original arrays
    Jnut = np.append(Jnurev,Jnuadd)
    nut  = np.append(nurev,nuadd)
    argsrt = np.argsort(nut,kind='mergesort')
    #plt.plot(np.log10(nu[::-1]),np.log10(Jnu[::-1]),'bo')
    #plt.plot(np.log10(nut[argsrt]),np.log10(Jnut[argsrt]),'rx')
    #plt.show()
    return Jnut[argsrt], nut[argsrt]
