import numpy as np
import cython_fns
from matplotlib import pyplot as plt
import constants
import logger
import os

def HMbackground_z0_sternberg(nu=None,maxvu=200.0,num=10000):
    Jnu0 = 2.0E-23
    nu0 = 3.287E15
    if nu is None:
        nu = np.linspace(5.0E-2,maxvu,num)
        J = Jnu0*np.ones(num)
    else:
        nu /= nu0
        J = Jnu0*np.ones(nu.size)
    w = np.where(nu>4.0)
    J[w] *= 2.512E-2 * nu[w]**-0.46
    w = np.where((nu>=1.0) & (nu<=4.0))
    J[w] *= nu[w]**-3.13
    w = np.where((nu>=0.3) & (nu<1.0))
    J[w] *= nu[w]**-5.41
    w = np.where(nu<0.3)
    J[w] *= 1.051E2 * nu[w]**-1.5
    return J, nu*nu0

def HMbackground(elID,redshift=3.0, version='12', alpha_UV=0):
    Jnu, nu, discretised_z = _HM_background_impl(elID, redshift, version, alpha_UV)
    logger.log("info", "Using HM{1:s} background at z={0:f}".format(discretised_z, version))
    return Jnu, nu

def _HM_background_impl(elID,redshift, version, alpha_UV):
    const = constants.get()
    planck  = const["planck"]
    elvolt  = const["elvolt"]
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
    #ediff = 0.00001
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

    Jnut = Jnut[argsrt]
    nut  =  nut[argsrt]
    egyt = nut * planck # energies

    # Shape parameter (Crighton et al 2015, https://arxiv.org/pdf/1406.4239.pdf)
    if alpha_UV != 0:
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
    #return np.loadtxt(os.path.join(os.path.dirname(__file__), "data/radfields/fiducial_j0_hm{:s}.dat").format(version), usecols=1)

def powerlaw(elID,):
    const = constants.get()
    planck  = const["planck"]
    elvolt  = const["elvolt"]
    try:
        nurevt, Jnurevt = np.loadtxt("data/radfields/" + options["radfield"]+".radfield",unpack=True)
        # Now load the HM spectrum to get the same frequency scale
        data = np.loadtxt("HM12_UVB.dat")
        rdshlist = data[0,:]
        amin = np.argmin(np.abs(rdshlist-3.0))
        waveAt, Jnut = data[1:,0], data[1:,amin]
        waveA = waveAt[1:]*1.0E-10
        nu = 299792458.0/waveA
        # Interpolate the background around each of the ionization potentials
        ediff = 1.0E-10
        ekeys = elID.keys()
        nurev  = nu[::-1]
        Jnurev = np.interp(nurev, nurevt, Jnurevt)
    except:
        logger.log("critical", "Radiation field file: {0:s} does not exist".format(options["radfield"]+".radfield"))
        sys.exit()
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

#def powerlaw(nuzero, index=-1.0, ioniznparam=-2.0, options=None, hionpot=13.598434005136):
#	"""
#	Important note: What is returned from this function needs to be
#	multiplied by the total H volume density before it is added to the
#	HM12 background flux
#	"""
#	if options is None: options = getoptions.default()
#	# First, define the shape of the radiation field
#	planck  = options["const"]["planck"]
#	elvolt  = options["const"]["elvolt"]
#	loRyd = 9.115E-3
#	hiRyd = 3676.0
#	nuRyd = nuzero*planck/(hionpot*elvolt)
#	lonu = loRyd * hionpot*elvolt/planck
#	hinu = hiRyd * hionpot*elvolt/planck
#	flux = (nuzero)**index
#	lojoin = np.argmin(np.abs(nuRyd-loRyd))
#	fxlo = flux[lojoin]
#	hijoin = np.argmin(np.abs(nuRyd-hiRyd))
#	fxhi = flux[hijoin]
#	wlo = np.where(nuRyd<loRyd)
#	whi = np.where(nuRyd>hiRyd)
#	flux[wlo] = fxlo * (nuzero[wlo]**2.5) / (lonu**2.5)
#	flux[whi] = fxhi * (nuzero[whi]**-2.0) / (hinu**-2.0)
#	plt.plot(np.log10(nuzero),np.log10(flux),'k-')
#	mn = np.min(np.log10(flux))
#	mx = np.max(np.log10(flux))
#	#plt.plot([np.log10(lonu),np.log10(lonu)],[mn,mx],'r-')
#	#plt.plot([np.log10(hinu),np.log10(hinu)],[mn,mx],'r-')
#	#plt.show()
#	# Now normalize to the appropriate number of H ionizing photons
#	nioniz = calc_Jnur.nhydroion(flux, nuzero, planck, hionpot*elvolt/planck)
#	fact = 2.99792458E10 * (10.0**ioniznparam) / (4.0*np.pi*nioniz)
#	return fact*flux
