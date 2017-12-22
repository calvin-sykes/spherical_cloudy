import pdb
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate
import astropy.units as u
import charge_transfer as chrgtran
import getoptions
import phionxsec
import radfields
import photoion
import colioniz
import elemids
import recomb
import cosmo
import misc
import calc_Jnur
import time
import signal
import sys
import os
from multiprocessing import cpu_count as mpCPUCount
from multiprocessing import Pool as mpPool
from multiprocessing.pool import ApplyResult

"""
METHOD:

Start with a uniform temperature of 10^4K everywhere

For a given DM halo profile, calculate the pressure profile,
with a boundary condition specifying the pressure of the IGM.

Assume the gas is initially fully ionized with Y(HI) = Y(HeI) = Y(HeII) = 0

Calculate the gas density profile, given that n(H) = P/(kT * 7/3)

Calculate the photoionization rates for H I, He I, and He II.

For the electron number density and gas temperature calculate the Ri.

Thereby calculate the new Y(HI), Y(HeI), Y(HeII)

Using the pressure profile, density profile and Y-values,
calculate the temperature profile.

Using this new temperature profile calculate the pressure profile
for the specified DM distribution.

For the pressure profile, temperature profile and Y-values,
recalculate the density profile, subject to an external pressure.

Using the density profile and the Y-values, calculate the
H I, He I, and He II volume density profiles.

Recompute the photoionization rates for H I, He I, and He II.

and so forth...
"""

def mpcoldens(j, prof, radius, nummu, geom):
    if geom == "NFW":
        coldens, muarr = calc_Jnur.calc_coldens(prof, radius, nummu)
        return [j,coldens,muarr]
    elif geom == "PP":
        coldens = calc_Jnur.calc_coldensPP(prof, radius)
        return [j,coldens]
    else:
        print "ERROR :: Geometry {0:s} is not allowed".format(geom)
        assert(False)

def mpphion(j, jnurarr, phelxs, nuzero, planck):
    phionxsec = 4.0*np.pi * calc_Jnur.phionrate(jnurarr, phelxs, nuzero, planck)
    return [j,phionxsec]


def get_radius(virialr, scale, npts, method=0):
    if method == 0:
        # Linear scaling
        radius = np.linspace(0.0, virialr*scale, npts)
    elif method == 1:
        # Linear scaling, but split up
        radius = np.linspace(0.0,0.1*virialr*scale,npts/10)
        radius = np.append(radius, np.linspace(0.1*virialr*scale,virialr*scale,7*npts/10)[1:])
        radius = np.append(radius, np.linspace(virialr*scale,10.0*virialr*scale,npts/10)[1:])
        lover = npts - radius.size + 1
        radius = np.append(radius, 10.0*np.linspace(virialr*scale,100.0*virialr*scale,lover)[1:])
    elif method == 2:
        # Log scaling
        radius = np.append(0.0, 10.0**np.linspace(np.log10(virialr*1.0E-4), np.log10(virialr*10.0), npts-1))
    else:
        print "radius method is not yet defined"
        sys.exit()
    if virialr not in radius:
        radius = np.append(virialr, radius[:-1])
        radius.sort()
    return radius

def get_halo(redshift,gastemp,bturb,metals=1.0,Hescale=1.0,cosmopar=np.array([0.673,0.04910,0.685,0.315]),ions=["H I", "He I", "He II"],prevfile=None,options=None):
    """
    bturb     : turbulent Doppler parameter
    metals    : Scale the metals by a constant
    cosmopar     : Set the cosmology of the simulation (hubble constant/100 km/s/Mpc, Omega_B, Omega_L, Omega_M)
    ions      : A list of ions to consider
    """
    # Begin the timer
    timeA = time.time()

    if options is None: options = getoptions.default()
    # Set some numerical aspects of the simulation
    miniter = options["run"]["miniter"]
    maxiter = options["run"]["maxiter"]
    npts    = options["run"]["nsample"]
    nummu   = options["run"]["nummu"]
    concrit = options["run"]["concrit"]
    ncpus   = options["run"]["ncpus"]
    kB      = options["const"]["kB"]
    cmtopc  = options["const"]["cmtopc"]
    somtog  = options["const"]["somtog"]
    Gcons   = options["const"]["Gcons"]
    planck  = options["const"]["planck"]
    elvolt  = options["const"]["elvolt"]
    protmss = options["const"]["protmss"]
    hztos   = options["const"]["hztos"]
    radmethod = 2  # Method used to define the radial coordinate

    # Set up the cosmology
    cosmoVal = FlatLambdaCDM(H0=100.0*cosmopar[0] * u.km / u.s / u.Mpc, Om0=cosmopar[3])

    # Get the element ID numbers
    elID, extions = elemids.getids(ions,metals)

    # How many ions do we need to calculate
    nions = len(ions)
    if ncpus > nions: ncpus = nions
    if ncpus > mpCPUCount(): ncpus = mpCPUCount()
    if ncpus <= 0: ncpus += mpCPUCount()
    if ncpus <= 0: ncpus = 1
    print "Using {0:d} CPUs".format(int(ncpus))

    # make multiprocessing pool if using >1 CPUs
    # the reassignment of SIGINT is needed to make Ctrl-C work while the process pool is active
    if ncpus > 1:
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        pool = mpPool(processes=ncpus)
        signal.signal(signal.SIGINT, sigint_handler)

    # Get the primordial number abundance of He relative to H
    if "He I" in ions:
        prim_He = elID["He I"].abund*Hescale
    elif "He II" in ions:
        prim_He = elID["He II"].abund*Hescale
    else:
        print "ERROR :: You must include ""He I"" and ""He II"" in your model"
        assert(False)
    if "H I" not in ions:
        print "ERROR :: You must include ""H I"" in your model"
        assert(False)

    print "Loading radiation fields"
    # Get the Haardt & Madau (2012) background at the appropriate redshift
    if options["radfield"] == "HM12":
        jzero, nuzero = radfields.HMbackground(elID,redshift=redshift,options=options)
        jzero *= options["HMscale"]
    elif options["radfield"][0:2] == "PL":
        jzero, nuzero = radfields.powerlaw(elID,options=options)
    else:
        print "The radiation field {0:s} is not implemented yet".format(options["radfield"])
        sys.exit()
    #np.savetxt("example.HMspectrum",np.transpose((nuzero,jzero)))

    #jzerodens = np.zeros(jzero.size)
    if options["powerlaw"] is not None:
        print "ERROR :: A powerlaw model has not yet been implemented. I recommend that you use a scaled HM12 background."
        assert(False)
        jzerodens = radfields.powerlaw(nuzero, index=options["powerlaw"][0], ioniznparam=options["powerlaw"][1])
        plt.plot(np.log10(nuzero),np.log10(jzerodens),'r-')
        plt.plot(np.log10(nuzero),np.log10(jzero),'k-')
        plt.show()

    # convert to an energy
    engy = planck * nuzero / elvolt

    print "Loading photoionization cross-sections"
    # Calculate the photoionization cross sections
    phelxsdata = phionxsec.load_data(elID)
    phelxs = np.zeros((engy.size,nions))
    for j in range(nions):
        #xsecv = phionxsec.xsec(ions[j],engy,elID)
        xsecv = phionxsec.rate_function_arr(engy,phelxsdata[ions[j]])
        phelxs[:,j] = xsecv.copy()
    # Calculate the photoionization heating cross sections
    #phhtxs = np.zeros((engy.size,nions))
    #for j in range(nions):
    #	xsecv = phionxsec.heat(ions[j],nuzero,planck,elvolt,elID)
    #	phhtxs[:,j] = xsecv.copy()
    # These are the same thing.
    #phelxs = phhtxs.copy()
    #plt.plot(np.log10(engy),phelxs[:,elID["H I"].id],'r-')
    #plt.plot(np.log10(engy),phhtxs[:,elID["H I"].id],'r-')
    #plt.plot(np.log10(engy),phelxs[:,elID["He I"].id],'g-')
    #plt.plot(np.log10(engy),phhtxs[:,elID["He I"].id],'g--')
    #plt.plot(np.log10(engy),phelxs[:,elID["He II"].id],'b-')
    #plt.plot(np.log10(engy),phhtxs[:,elID["He II"].id],'b--')
    #plt.show()

    print "Loading radiative recombination coefficients"
    rrecombrate = recomb.load_data_radi(elID)

    print "Loading dielectronic recombination coefficients"
    drecombrate = recomb.load_data_diel(elID)
    drecombelems = drecombrate.keys()

    print "Loading Collisional Ionization rate coefficients"
    usecolion = "Dere2007"
    if usecolion == "Dere2007":
        colionrate = colioniz.load_data(elID, rates="Dere2007")
    elif usecolion == "Voronov1997":
        colionrate = colioniz.load_data(elID, rates="Voronov1997")
    else:
        print "Error cannot load collisional ionization rates"
    #colionrateV = colioniz.load_data(elID, rates="Voronov1997")

    # Check the rates are similar
    #tempval = 10.0**np.linspace(4.0,6.0,100)
    #rateD = colioniz.rate_function_Dere2007(1.0E-7*tempval*kB/elvolt, colionrate["He I"])
    #rateV = colioniz.rate_function_arr(1.0E-7*tempval*kB/elvolt, colionrateV["He I"])
    #plt.plot(np.log10(tempval), np.log10(rateD))
    #plt.plot(np.log10(tempval), np.log10(rateV))
    #plt.show()
    #assert(False)

    print "Loading Charge Transfer rate coefficients"
    chrgtranrate = chrgtran.load_data(elID)
    chrgtran_HItargs = chrgtranrate["H I"].keys()
    chrgtran_HIItargs = chrgtranrate["H II"].keys()
    chrgtran_HeItargs = chrgtranrate["He I"].keys()
    chrgtran_HeIItargs = chrgtranrate["He II"].keys()

    print "Loading Cloudy cooling curve"
    #import coolfunc
    #coolingcurves = coolfunc.make_coolingcurve(plot=False)
    close=False

    # Calculate some properties of the dark matter halo
    hubpar = cosmo.hubblepar(redshift, cosmopar)
    rhocrit = 3.0*(hubpar*hztos)**2/(8.0*np.pi*Gcons)
    if options["geometry"]["use"] == "NFW":
        # Get the concentration of the halo
        print "Calculating halo concentration"
        virialm = options["geometry"]["NFW"][0]
        cvir = cosmo.massconc_Prada12(virialm, cosmopar, redshift=redshift)
        #cvir = massconc_Klypin11(mvir,redshift=3)
        virialr = (virialm*somtog/(4.0*np.pi*200.0*rhocrit/3.0))**(1.0/3.0)
        rscale = virialr/cvir
        rhods = 200.0 * rhocrit * cvir**3 / calc_Jnur.NFW_fm(cvir)
        print "Virial radius (pc) = {0:E}".format(virialr*cmtopc)
        print "Scale radius (pc) = {0:E}".format(rscale*cmtopc)

    # if string is passed, interpret as filename for the previous run
    # this is used as an initial solution to speed up convergence
    if prevfile is not None:
        if options["geometry"]["use"] == "NFW":
            print "Loading file {0:s}".format(prevfile)
            #print "ERROR: file not saved"  
            tdata = np.load(prevfile)
            #####################
            #//
#			ttdata = np.load(infname)
#			numarr = ttdata.shape[1]
#			# Extract radius information
#			radius  = ttdata[:,0]
#			# Calculate the new radius
#			newradius = np.linspace(0.0,np.max(radius),npts)
#			tdata = np.zeros((npts,numarr))
#			tdata[:,0] = newradius
#			for i in range(1,numarr):
#				tdata[:,i] = np.interp(newradius,radius,ttdata[:,i])
            #\\
            #####################
            strt = 4
            numarr = tdata.shape[1]
            arridx = dict({})
            arridx["voldens"] = dict({})
            arridx["coldens"] = dict({})
            for i in range((numarr-strt)/2):
                arridx["voldens"][ions[i]] = strt + i
                arridx["coldens"][ions[i]] = strt + i + (numarr-strt)/2
            old_radius = get_radius(virialr, options["geometry"]["NFW"][1], npts, method=radmethod)
            if old_radius.size != npts:
                print "Error defining radius"
                sys.exit()
            prof_coldens = np.zeros((npts,nummu,nions))
            prof_density = np.zeros((npts,nions))
            interpit = False
            if interpit:
                print "WARNING -- Interpolating by density weighting"
                wght = np.sqrt(tdata[:,arridx["voldens"]["He II"]].copy())
                wght /= (tdata[:,2].copy() * elID["He II"].abund)
                wght /= np.sum(wght)
# 				plt.subplot(311)
# 				plt.plot(old_radius, NHI, 'k-')
# 				plt.subplot(312)
# 				plt.plot(old_radius[1:], derv, 'k-')
# 				plt.subplot(313)
# 				plt.plot(old_radius[1:], derv*NHI, 'k-')
# 				plt.show()
                x_density = np.cumsum(wght)
                # rescale to match old range
                x_density -= x_density.min()
                x_density /= (x_density.max()-x_density.min())
                func = interpolate.interp1d(x_density, np.linspace(0.0, 1.0, npts), kind='linear')
                radius = func(np.linspace(0.0, 1.0, npts))*virialr*options["geometry"]["NFW"][1]
                if x_density.size != npts:
                    print "Bad radius formulation"
                    pdb.set_trace()
                    assert(False)
# 				func = interpolate.interp1d(old_radius, wght, kind='linear')
# 				newwght = func(radius)
#  	 			plt.plot(old_radius, wght, 'bo')
#  	 			plt.plot(radius, newwght, 'rx')
#  	 			plt.show()
                func = interpolate.interp1d(old_radius, tdata[:,1], kind='linear')
                prof_temperature = func(radius)
                func = interpolate.interp1d(old_radius, tdata[:,2], kind='linear')
                temp_densitynH = func(radius)
            else:
                radius = old_radius.copy()
                prof_temperature = tdata[:,1]
                temp_densitynH = tdata[:,2]
            # Extract the data from the array
            Yprofs = 1.0E-1*np.ones((npts,nions))
            for j in range(nions):
                # density of this specie = unionized fraction * H volume density * number abundance relative to H
                if interpit:
                    func = interpolate.interp1d(old_radius, tdata[:,arridx["voldens"][ions[j]]], kind='linear')
                    prof_density[:,j] = func(radius)
                else:
                    prof_density[:,j] = tdata[:,arridx["voldens"][ions[j]]]
                Yprofs[:,j] = prof_density[:,j] / (temp_densitynH * elID[ions[j]].abund)
    #		prof_YHI   = tdata[:,5]
    #		prof_YHeI  = tdata[:,6]
    #		prof_YHeII = tdata[:,7]
            prof_phionrate = np.zeros((npts,nions))
            densitym  = temp_densitynH * protmss * (1.0 + 4.0*prim_He)
        elif options["geometry"]["use"] == "PP":
            print "Never needed this"
            assert(False)
            print "Loading file {0:s}".format(prevfile)
            #print "ERROR: file not saved"  
            tdata = np.load(prevfile)
            numarr = tdata.shape[1]
            radius = np.linspace(0.0,virialr*options["geometry"]["NFW"][1],npts)
            prof_coldens = np.zeros((npts,nummu,nions))
            prof_density = np.zeros((npts,nions))
            prof_temperature = tdata[:,1]
            temp_densitynH = tdata[:,2]
            # Extract the data from the array
            strt = 3
            arridx = dict({})
            arridx["voldens"] = dict({})
            arridx["coldens"] = dict({})
            Yprofs = 1.0E-1*np.ones((npts,nions))
            for i in range((numarr-strt)/2):
                arridx["voldens"][ions[i]] = strt + i
                arridx["coldens"][ions[i]] = strt + i + (numarr-strt)/2
            for j in range(nions):
                # density of this specie = unionized fraction * H volume density * number abundance relative to H
                prof_density[:,j] = tdata[:,arridx["voldens"][ions[j]]]
                Yprofs[:,j] = prof_density[:,j] / (temp_densitynH * elID[ions[j]].abund)
    #		prof_YHI   = tdata[:,5]
    #		prof_YHeI  = tdata[:,6]
    #		prof_YHeII = tdata[:,7]
            prof_phionrate = np.zeros((npts,nions))
            densitym  = temp_densitynH * protmss * (1.0 + 4.0*prim_He)
        else:
            print "Not implemented yet"
            assert(False)
    else: # prevfile is None
        # Set the gas conditions
        if options["geometry"]["use"] == "NFW":
            radius = get_radius(virialr, options["geometry"]["NFW"][1], npts, method=radmethod)
            if radius.size != npts:
                print "Error defining radius"
                sys.exit()
            temp_densitynH = np.ones(npts)
            prof_coldens = np.zeros((npts,nummu,nions))
            prof_density = 1.0E-1*np.ones((npts,nions))
        elif options["geometry"]["use"] == "PP":
            radius  = np.linspace(0.0,1000.0*options["geometry"]["PP"][1]/cmtopc,npts)
            #radius, temper, hden, eden, Ho, Hp, H2, H2p, H3p, Hm = np.loadtxt("../cloudy/example.Hcond",unpack=True)
            #radius = np.max(radius) - radius
            #radius = np.sort(radius,kind='mergesort')
            #npts = radius.size
            densitynH = np.ones(npts)*(10.0**options["geometry"]["PP"][0])
            prof_coldens = np.zeros((npts,nions))
            prof_density = 1.0E-1*np.ones((npts,nions))
        prof_temperature = gastemp * np.ones(npts)
        prof_phionrate = np.zeros((npts,nions))
        Yprofs = 1.0E-2*np.ones((npts,nions))
        densitym = protmss * (1.0 + 4.0*prim_He) * np.ones(npts) # Default to be used for PP

    # An array used to check if convergence has been reached for each ion and in each cell.
    allionpnt = npts*np.ones(nions,dtype=np.int)

    # Calculate the mass of baryons involved:
    if options["geometry"]["use"] == "NFW":
        muion = 0.59 # for fully ionized gas
        TIGM  = 20000.0
        OmegaBxMhalo = virialm*somtog * cosmopar[1] / (cosmopar[3]-cosmopar[1])
#		cIGM = np.sqrt(1.5*kB*TIGM/(muion*protmss))
#		consfact = 3.0/(np.pi*np.sqrt(2.0))
#		rhocrit = cosmoVal.critical_density(redshift).value
#		hubbletime = cosmoVal.age(redshift).to(u.s).value
#		Mjeans = consfact * cosmopar[1] * rhocrit * ((2.0*np.pi*cIGM*hubbletime)**3)
#		print "Omega_B x M_200 (M_sun) = ", OmegaBxMhalo/somtog
#		print "Jeans Mass (M_sun) = ", Mjeans/somtog
#		if OmegaBxMhalo > Mjeans:
#			barymass = OmegaBxMhalo
#			print "Adopting the universal baryon fraction for the total baryon mass within r_200"
#		else:
#			scalefact = 1.0
#			BondiRate = (np.pi*Gcons*Gcons*virialm*somtog*virialm*somtog*rhocrit)/(cIGM**3)
#			Mbondi = scalefact*hubbletime*BondiRate
#			barymass = Mbondi
#			print "Bondi Mass (M_sun) = ", Mbondi/somtog
#			print "Adopting Bondi Mass for the total baryon mass within r_200"
        # Set the mass density profile
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print "WARNING :: REMOVE THE NEXT LINE -- THIS IS JUST DEBUGGING !!!!!!!!!!!!!!!!!!!!!!"
        #barymass = OmegaBxMhalo * options["geometry"]["NFW"][2]
        barymass = virialm*somtog * options["geometry"]["NFW"][2]
        densitym = np.ones(npts)*barymass/(4.0*np.pi*(virialr**3)/3.0)
        print np.log10(virialm), np.log10(barymass/somtog)

    elif options["geometry"]["use"] == "PP":
        pass # The mass density is set earlier for PP

    # Set the plotting colormap
    colormap = cm.bwr
    normalize = mcolors.Normalize(vmin=0, vmax=10)

    # Store old Yprofs
    nstore = 10
    istore = 0
    store_Yprofs = np.zeros(Yprofs.shape + (nstore,))

    # Set the stopping criteria flags
    tstcrit  = np.zeros(nions,dtype=np.int)  # tstHI, tstHeI, tstHeII
    iteration = 0
    answer = -1
    old_Yprofs = None
    while (not np.array_equal(tstcrit,allionpnt)) or (iteration <= miniter) or (not close):
        iteration += 1
        #print "          Iteration number: {0:d}   <--[  {1:d}/{2:d} {3:d}/{2:d} {4:d}/{2:d}  ]-->".format(iteration,tstHI,npts,tstHeI,tstHeII)
        print "   Iteration number: {0:d}".format(iteration)
        for j in range(nions): print "   <--[  {0:d}/{1:d}  ]-->  {2:s}".format(int(tstcrit[j]),npts,ions[j])

        # Store the old Yprofs
        store_Yprofs[:,:,istore] = Yprofs.copy()
        istore += 1
        if istore >= nstore:
            istore = 0

        # Calculate the pressure profile
        dof = (2.0-Yprofs[:,elID["H I"].id]) + prim_He*(3.0 - 2.0*Yprofs[:,elID["He I"].id] - 1.0*Yprofs[:,elID["He II"].id])
        masspp = (1.0 + 4.0*prim_He)/dof
        if options["geometry"]["use"] == "NFW":
            fgas = calc_Jnur.fgasx(densitym,radius,rscale)
            #prof_pressure = calc_Jnur.HEcalc_NFW(radius/rscale,prof_temperature,masspp,bturb,rhods,rscale,0.001,densitym[0],barymass)
            prof_pressure = calc_Jnur.pressure_NFW(prof_temperature,radius,masspp,bturb,rhods,rscale,Gcons,kB,protmss)
            turb_pressure = 0.75*densitym*(bturb**2.0)
            #############
            # TEST PLOT #
            #############
            #print "pressure!"
            #plt.plot(np.log10(radius*cmtopc),prof_pressure,'k-')
            #plt.show()
            #plt.clf()
            #############
            #############
            # Calculate the thermal pressure, and ensure positivity
            ther_pressure = prof_pressure-turb_pressure
            wther = np.where(ther_pressure<0.0)
            ther_pressure[wther] = 0.0
            # Calculate gas density profile
            temp_densitynH = ther_pressure / (1.5 * kB * prof_temperature * dof)
            #plt.plot(radius*cmtopc,temp_densitynH,'k-')
            if (temp_densitynH[0]==0.0):
                print "WARNING :: central density is zero"
                print "        :: Assuming no turbulent pressure for this iteration"
                temp_densitynH = prof_pressure / (1.5 * kB * prof_temperature * dof)
            temp_densitynH /= temp_densitynH[0]
            rintegral = calc_Jnur.mass_integral(temp_densitynH,radius,virialr)
            #############
            # TEST PLOT #
            #############
            #print "mass integral!"
            #plt.plot(np.log10(radius*cmtopc),prof_pressure,'k-')
            #plt.show()
            #plt.clf()
            #############
            #############
            cen_density = barymass / (4.0 * np.pi * protmss * (1.0 + 4.0*prim_He) * rintegral)
            densitynH = cen_density * temp_densitynH
            #plt.plot(np.log10(radius*cmtopc),np.log10(densitynH),'m-')
            #plt.show()
            #plt.clf()
            densitym  = densitynH * protmss * (1.0 + 4.0*prim_He)

        # Update the volume density of the unionized species
        for j in range(nions):
            # density of this specie = unionized fraction * H volume density * number abundance relative to H
            prof_density[:,j] = Yprofs[:,j] * densitynH * elID[ions[j]].abund

        # Compute the electron density
        electrondensity = densitynH * ( (1.0-Yprofs[:,elID["H I"].id]) + prim_He*Yprofs[:,elID["He II"].id] + 2.0*prim_He*(1.0-Yprofs[:,elID["He I"].id]-Yprofs[:,elID["He II"].id]) )

        #plt.plot(np.log10(radius*cmtopc),electrondensity,'k-')
        #plt.show()
        #plt.clf()
        #plt.plot(np.log10(radius*cmtopc),np.log10(prof_density[:,elID["H I"].id]),'r-')
        #plt.plot(np.log10(radius*cmtopc),np.log10(prof_density[:,elID["He I"].id]),'g-')
        #plt.plot(np.log10(radius*cmtopc),np.log10(prof_density[:,elID["He II"].id]),'b-')
        #plt.show()
        #plt.clf()

        # Calculate the column density arrays,
        if ncpus == 1:
            for j in range(nions):
                if options["geometry"]["use"] == "NFW":
                    coldens, muarr = calc_Jnur.calc_coldens(prof_density[:,j], radius, nummu)
                    prof_coldens[:,:,j] = coldens.copy()
                elif options["geometry"]["use"] == "PP":
                    coldens = calc_Jnur.calc_coldensPP(prof_density[:,j], radius)
                    prof_coldens[:,j] = coldens.copy()
                #coldensHI, muarr = calc_Jnur.calc_coldens(prof_HI, radius, nummu)
                #coldensHeI, muarr = calc_Jnur.calc_coldens(prof_HeI, radius, nummu)
                #coldensHeII, muarr = calc_Jnur.calc_coldens(prof_HeII, radius, nummu)
        else:
            async_results = []
            for j in range(nions):
                async_results.append(pool.apply_async(mpcoldens, (j, prof_density[:,j], radius, nummu, options["geometry"]["use"])))
                #if j == 0:
                #	# H I column density calculation
                #	async_results.append(pool.apply_async(mpcoldens, (j, prof_HI, radius, nummu)))
                #elif j == 1:
                #	# He I column density calculation
                #	async_results.append(pool.apply_async(mpcoldens, (j, prof_HeI, radius, nummu)))
                #elif j == 2:
                #	# He II column density calculation
                #	async_results.append(pool.apply_async(mpcoldens, (j, prof_HeII, radius, nummu)))
            map(ApplyResult.wait, async_results)
            for j in range(nions):
                getVal = async_results[j].get()
                if options["geometry"]["use"] == "NFW":
                    prof_coldens[:,:,getVal[0]] = getVal[1].copy()
                    muarr = getVal[2]
                elif options["geometry"]["use"] == "PP":
                    prof_coldens[:,getVal[0]] = getVal[1].copy()
                #if getVal[0] == 0:
                #	coldensHI, muarr = getVal[1], getVal[2]
                #elif getVal[0] == 1:
                #	coldensHeI, muarr = getVal[1], getVal[2]
                #elif getVal[0] == 2:
                #	coldensHeII, muarr = getVal[1], getVal[2]

        # integrate over all angles,
        if options["geometry"]["use"] == "NFW":
            print "Integrate over all angles"
            jnurarr = calc_Jnur.nint_costheta(prof_coldens, phelxs, muarr, jzero)
        elif options["geometry"]["use"] == "PP":
            jnurarr = calc_Jnur.nint_pp(prof_coldens, phelxs, jzero)

        #plt.plot(np.log10(radius*cmtopc),np.log10(prof_coldens[:,elID["H I"].id]),'r-')
        #plt.plot(np.log10(radius*cmtopc),np.log10(prof_coldens[:,elID["He I"].id]),'g-')
        #plt.plot(np.log10(radius*cmtopc),np.log10(prof_coldens[:,elID["He II"].id]),'b-')
        #plt.show()
        #plt.clf()

        #############
        # TEST PLOT #
        #############
        #plt.plot(np.log10(radius*cmtopc),jnurarr[100,:],'k-')
        #plt.plot(np.log10(radius*cmtopc),jnurarr[250,:],'r-')
        #plt.plot(np.log10(radius*cmtopc),jnurarr[400,:],'g-')
        #plt.plot(np.log10(radius*cmtopc),jnurarr[500,:],'b-')
        #plt.show()
        #plt.clf()
        #############
        #############
        # and calculate the photoionization rates
        print "Calculating phionization rates"
        if ncpus == 1:
            for j in range(nions):
                phionr = 4.0*np.pi * calc_Jnur.phionrate(jnurarr, phelxs[:,j], nuzero, planck*1.0E7)
                prof_phionrate[:,j] = phionr.copy()
        else:
            async_results = []
            for j in range(nions):
                async_results.append(pool.apply_async(mpphion, (j, jnurarr, phelxs[:,j], nuzero, planck*1.0E7)))
            map(ApplyResult.wait, async_results)
            for j in range(nions):
                getVal = async_results[j].get()
                prof_phionrate[:,getVal[0]] = getVal[1].copy()

        # Calculate the collisional ionization rate coefficients
        prof_colion = np.zeros((npts,nions))
        for j in range(nions):
            #ratev = colioniz.rate(ions[j],1.0E-7*prof_temperature*kB/elvolt,elID)
            if usecolion == "Dere2007":
                ratev = colioniz.rate_function_Dere2007(1.0E-7*prof_temperature*kB/elvolt, colionrate[ions[j]])
            elif usecolion == "Voronov1997":
                ratev = colioniz.rate_function_arr(1.0E-7*prof_temperature*kB/elvolt, colionrate[ions[j]])
            prof_colion[:,j] = ratev.copy()


        #############
        # TEST PLOT #
        #############
        #print "phionrate!"
        #plt.plot(np.arange(npts),prof_colion[:,elID["H I"].id],'r-')
        #plt.plot(np.arange(npts),prof_colion[:,elID["He I"].id],'g-')
        #plt.plot(np.arange(npts),prof_colion[:,elID["He II"].id],'b-')
        #plt.show()
        #plt.clf()
        #############
        #############

        # the secondary photoelectron collisional ionization rates (probably not important for metals -- Section II, Shull & van Steenberg (1985))
        print "Performing numerical integration over frequency to get secondary photoelectron ionization"
        # Make sure there are no zero H I density
        tmpcloneHI = prof_density[:,elID["H I"].id].copy()
        w = np.where(prof_density[:,elID["H I"].id] == 0.0)
        if np.size(w[0]) != 0:
            print "WARNING :: n(H I) = exactly 0.0 in some zones, setting to smallest value"
            wb = np.where(tmpcloneHI!=0.0)
            tmpcloneHI[w] = np.min(tmpcloneHI[wb])
            prof_density[:,elID["H I"].id] = tmpcloneHI.copy()
            prof_density[:,elID["D I"].id] = tmpcloneHI.copy()*elID["D I"].abund
        prof_scdryrate = np.zeros((npts,nions))

        if ncpus > 1:
            async_results = []
            # H I
            async_results.append(pool.apply_async(calc_Jnur.scdryrate, (jnurarr, nuzero,
                    phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id], # photoionisation cross-sections
                    prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id], electrondensity/(densitynH*(1.0 + 2.0*prim_He)), # densities
                    elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, # ionisation potentials
                    planck, elvolt, 0))) # constants
            # He I
            async_results.append(pool.apply_async(calc_Jnur.scdryrate, (jnurarr, nuzero,
                    phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id], # photoionisation cross-sections
                    prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id], electrondensity/(densitynH*(1.0 + 2.0*prim_He)), # densities
                    elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, # ionisation potentials
                    planck, elvolt, 2))) # constants
            map(ApplyResult.wait, async_results)

            for j in range(nions):
                if ions[j] == "H I":
                    ratev = 4.0*np.pi * async_results[0].get()
                    prof_scdryrate[:,j] = ratev.copy()
                elif ions[j] == "He I":
                    ratev = 4.0*np.pi * 10.0 * async_results[1].get()
                    prof_scdryrate[:,j] = ratev.copy()
        else:
            for j in range(nions):
                if ions[j] == "H I":
                    ratev = 4.0*np.pi * calc_Jnur.scdryrate(jnurarr, nuzero, phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id],
                       prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id],
                        electrondensity/(densitynH*(1.0 + 2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, planck, elvolt, 0)
                    prof_scdryrate[:,j] = ratev.copy()
#		    	elif ions[j] == "D I":
#		    		ratev = 4.0*np.pi * calc_Jnur.scdryrate(jnurarr, nuzero, phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id],
#		    			prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id],
#		    			electrondensity/(densitynH*(1.0 + 2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, planck, elvolt, 1)
#		    		prof_scdryrate[:,j] = ratev.copy()
                elif ions[j] == "He I":
                    ratev = 4.0*np.pi * 10.0 * calc_Jnur.scdryrate(jnurarr, nuzero, phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id],
                        prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id],
                        electrondensity/(densitynH*(1.0 + 2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, planck, elvolt, 2)
                    prof_scdryrate[:,j] = ratev.copy()

        #############
        # TEST PLOT #
        #############
        #print "scdryrate!"
        #plt.plot(np.arange(npts),prof_scdryrate[:,elID["H I"].id],'r-')
        #plt.plot(np.arange(npts),prof_scdryrate[:,elID["D I"].id],'g-')
        #plt.plot(np.arange(npts),prof_scdryrate[:,elID["He I"].id],'b-')
        #plt.show()
        #plt.clf()
        #############
        #############

        # Calculate other forms of ionization (e.g. photons from H and He recombinations)
        print "Calculate ionization rate from recombinations of H+, He+, He++"
        prof_other = np.zeros((npts,nions))
        for j in range(nions):
            ratev = photoion.other(ions[j],engy,prof_density,densitynH,Yprofs,electrondensity,phelxs,prof_temperature,elID,kB,elvolt)
            prof_other[:,j] = ratev.copy()

        #plt.plot(radius*cmtopc,phionrate_HI,'r-')
        #plt.plot(radius*cmtopc,phionrate_HeI,'g-')
        #plt.plot(radius*cmtopc,phionrate_HeII,'b-')
        #plt.plot(radius*cmtopc,scdryrate_HI,'r-')
        #plt.plot(radius*cmtopc,scdryrate_HeI,'g-')
        #plt.plot(radius*cmtopc,B1_HeII,'k--')
        #plt.plot(radius*cmtopc,B2_HeI,'g--')
        #plt.plot(radius*cmtopc,B3_HI,'r--')
        #plt.plot(radius*cmtopc,B6_HeI,'g:')
        #plt.plot(radius*cmtopc,B7_HI,'r:')
        #plt.show()
        #plt.clf()

        # Calculate the charge transfer ionization rates
        print "Calculating charge transfer rates"
        HIIdensity  = densitynH * (1.0-Yprofs[:,elID["H I"].id])
        HeIIdensity = densitynH * prim_He*Yprofs[:,elID["He II"].id]
        HIIdensity = HIIdensity.reshape((npts,1)).repeat(nions,axis=1)
        HeIIdensity = HeIIdensity.reshape((npts,1)).repeat(nions,axis=1)
        prof_chrgtraniHII = np.zeros((npts,nions))
        prof_chrgtraniHeII = np.zeros((npts,nions))
        for j in range(nions):
            #ratev = chrgtran.HII_target(ions[j],prof_temperature)
            if ions[j] in chrgtran_HIItargs:
                ratev = chrgtran.rate_function_form(chrgtranrate["H II"][ions[j]],prof_temperature)
                prof_chrgtraniHII[:,j] = ratev.copy()
            #ratev = chrgtran.HeII_target(ions[j],prof_temperature)
            if ions[j] in chrgtran_HeIItargs:
                ratev = chrgtran.rate_function_form(chrgtranrate["He II"][ions[j]],prof_temperature)
                prof_chrgtraniHeII[:,j] = ratev.copy()

        # Total all of the ionization rates
        prof_gamma = prof_phionrate + prof_scdryrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_other + prof_colion
        #prof_gamma = prof_phionrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_colion + prof_other

        #prof_gamma = prof_phionrate + prof_scdryrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_other
        #prof_gamma = prof_phionrate + prof_scdryrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_colion
        #prof_gamma = prof_phionrate + prof_scdryrate + prof_other + prof_colion
        #prof_gamma = prof_phionrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_other + prof_colion
        #prof_gamma = prof_scdryrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_other + prof_colion

        #print "THE DISCONTINUITY IS INTRODUCED IN THE SHARING FUNCTION!"

        #############
        # TEST PLOT #
        #############
        #print "B rates!"
        #plt.plot(np.arange(npts),B7_HI,'r-') # Not in this rate -- It's in rates 1, 2, 3, 6
        #plt.plot(np.arange(npts),B3_HI,'r--')
        #plt.plot(np.arange(npts),B2_HeI,'g-')
        #plt.plot(np.arange(npts),B6_HeI,'g-')
        #plt.plot(np.arange(npts),B1_HeII,'b-')
        #plt.show()
        #plt.clf()
        #############
        #############

        "Calculating recombination rates"
        prof_recomb = np.zeros((npts,nions))
        prof_recombCTHI  = np.zeros((npts,nions))
        prof_recombCTHeI = np.zeros((npts,nions))
        for j in range(nions):
            #ratev = recomb.radiative(ions[j],prof_temperature) + recomb.dielectronic(ions[j],prof_temperature)
            #ratev = recomb.radiative(ions[j],prof_temperature)
            #ratev = recomb.dielectronic(ions[j],prof_temperature)
            ratev = recomb.rate_function_radi_arr(prof_temperature, rrecombrate[ions[j]])
            if ions[j] in drecombelems: ratev += recomb.rate_function_diel_arr(prof_temperature, drecombrate[ions[j]])
            prof_recomb[:,j] = ratev.copy()
            #ratev = chrgtran.HI_target(ions[j],prof_temperature)
            if ions[j] in chrgtran_HItargs:
                ratev = chrgtran.rate_function_form(chrgtranrate["H I"][ions[j]],prof_temperature)
                prof_recombCTHI[:,j] = ratev.copy()
            #ratev = chrgtran.HeI_target(ions[j],prof_temperature)
            if ions[j] in chrgtran_HeItargs:
                ratev = chrgtran.rate_function_form(chrgtranrate["He I"][ions[j]],prof_temperature)
                prof_recombCTHeI[:,j] = ratev.copy()

        edens_allions = electrondensity.reshape((npts,1)).repeat(nions,axis=1)
        HIdensity = prof_density[:,elID["H I"].id]
        HeIdensity = prof_density[:,elID["He I"].id]
        HIdensity = HIdensity.reshape((npts,1)).repeat(nions,axis=1)
        HeIdensity = HeIdensity.reshape((npts,1)).repeat(nions,axis=1)
        prof_alpha = edens_allions*prof_recomb + HIdensity*prof_recombCTHI  + HeIdensity*prof_recombCTHeI
        #prof_alpha = edens_allions*prof_recomb
        prof_rates = prof_gamma / prof_alpha

        #############
        # TEST PLOT #
        #############
        #print "rates before!"
        #plt.plot(np.arange(npts),prof_recomb[:,elID["H I"].id],'r-')
        #plt.plot(np.arange(npts),prof_recomb[:,elID["He I"].id],'g-')
        #plt.plot(np.arange(npts),prof_recomb[:,elID["He II"].id],'b-')
        #plt.plot(np.arange(npts),prof_drecomb[:,elID["H I"].id],'r--')
        #plt.plot(np.arange(npts),prof_drecomb[:,elID["He I"].id],'g--')
        #plt.plot(np.arange(npts),prof_drecomb[:,elID["He II"].id],'b--')
        #plt.show()
        #plt.clf()
        #############
        #############
        # Store old and obtain new values for YHI, YHeI, YHeII
        #if old_Yprofs is not None: Yprofs = 0.5*(old_Yprofs+Yprofs)
        old_Yprofs = Yprofs.copy()
        tmp_Yprofs = Yprofs.copy()
        inneriter = 0
        while True:
            #if iteration > 100 and inneriter==1: break
            # Calculate the Yprofs
            Yprofs = misc.calc_yprofs(ions,prof_rates,elID)
            #prof_YHI   = 1.0/(1.0+rate_HI)
            #prof_YHeII = 1.0/(1.0+rate_HeII+(1.0/rate_HeI))
            #prof_YHeI  = prof_YHeII/rate_HeI
            # Test if the profiles have converged
            tstconv = ( (np.abs((tmp_Yprofs-Yprofs)/Yprofs)<concrit**2)|(Yprofs==0.0)).astype(np.int).sum(axis=0)
            #tstconv = (np.abs(tmp_Yprofs-Yprofs)<concrit).astype(np.int).sum(0)
            #tstHI   = (np.abs(tmp_prof_YHI-prof_YHI)<concrit).astype(np.int).sum()
            #tstHeI  = (np.abs(tmp_prof_YHeI-prof_YHeI)<concrit).astype(np.int).sum()
            #tstHeII = (np.abs(tmp_prof_YHeII-prof_YHeII)<concrit).astype(np.int).sum()
            # Reset ne and the rates
            electrondensity = densitynH * ( (1.0-Yprofs[:,elID["H I"].id]) + prim_He*Yprofs[:,elID["He II"].id] + 2.0*prim_He*(1.0-Yprofs[:,elID["He I"].id]-Yprofs[:,elID["He II"].id]) )
            edens_allions = electrondensity.reshape((npts,1)).repeat(nions,axis=1)
            # Recalculate the recombination rate profile with the new Yprofs and electrondensity
            HIIdensity  = densitynH * (1.0-Yprofs[:,elID["H I"].id])
            HeIIdensity = densitynH * prim_He*Yprofs[:,elID["He II"].id]
            HIIdensity  = HIIdensity.reshape((npts,1)).repeat(nions,axis=1)
            HeIIdensity = HeIIdensity.reshape((npts,1)).repeat(nions,axis=1)
            # Recalculate all of the ionization effects that depend on density
            if True:
                # scdryrate
                tmpcloneHI = prof_density[:,elID["H I"].id].copy()
                w = np.where(prof_density[:,elID["H I"].id] == 0.0)
                if np.size(w[0]) != 0:
                    print "WARNING :: n(H I) = exactly 0.0 in some zones, setting to smallest value"
                    wb = np.where(tmpcloneHI!=0.0)
                    tmpcloneHI[w] = np.min(tmpcloneHI[wb])
                    prof_density[:,elID["H I"].id] = tmpcloneHI.copy()
                    prof_density[:,elID["D I"].id] = tmpcloneHI.copy()*elID["D I"].abund
                prof_scdryrate = np.zeros((npts,nions))
                if ncpus > 1:
                    async_results = []
                    # H I
                    async_results.append(pool.apply_async(calc_Jnur.scdryrate, (jnurarr, nuzero,
                            phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id], # photoionisation cross-sections
                            prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id], electrondensity/(densitynH*(1.0 + 2.0*prim_He)), # densities
                            elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, # ionisation potentials
                            planck, elvolt, 0))) # constants
                    # He I
                    async_results.append(pool.apply_async(calc_Jnur.scdryrate, (jnurarr, nuzero,
                            phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id], # photoionisation cross-sections
                            prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id], electrondensity/(densitynH*(1.0 + 2.0*prim_He)), # densities
                            elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, # ionisation potentials
                            planck, elvolt, 2))) # constants
                    map(ApplyResult.wait, async_results)

                    for j in range(nions):
                        if ions[j] == "H I":
                            ratev = 4.0*np.pi * async_results[0].get()
                            prof_scdryrate[:,j] = ratev.copy()
                        elif ions[j] == "He I":
                            ratev = 4.0*np.pi * 10.0 * async_results[1].get()
                            prof_scdryrate[:,j] = ratev.copy()
                else:
                    for j in range(nions):
                        if ions[j] == "H I":
                            ratev = 4.0*np.pi * calc_Jnur.scdryrate(jnurarr, nuzero, phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id],
                               prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id],
                                electrondensity/(densitynH*(1.0 + 2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, planck, elvolt, 0)
                            prof_scdryrate[:,j] = ratev.copy()
                        elif ions[j] == "He I":
                            ratev = 4.0*np.pi * 10.0 * calc_Jnur.scdryrate(jnurarr, nuzero, phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id],
                                prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id],
                                electrondensity/(densitynH*(1.0 + 2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, planck, elvolt, 2)
                            prof_scdryrate[:,j] = ratev.copy()
                # Colion
                for j in range(nions):
                    #ratev = colioniz.rate(ions[j],1.0E-7*prof_temperature*kB/elvolt,elID)
                    #ratev = colioniz.rate_function_arr(1.0E-7*prof_temperature*kB/elvolt, colionrate[ions[j]])
                    if usecolion == "Dere2007":
                        ratev = colioniz.rate_function_Dere2007(1.0E-7*prof_temperature*kB/elvolt, colionrate[ions[j]])
                    elif usecolion == "Voronov1997":
                        ratev = colioniz.rate_function_arr(1.0E-7*prof_temperature*kB/elvolt, colionrate[ions[j]])
                    #ratev = colioniz.rate_function_Dere2007(1.0E-7*prof_temperature*kB/elvolt, colionrate[ions[j]])
                    prof_colion[:,j] = ratev.copy()
                # Other
                for j in range(nions):
                    ratev = photoion.other(ions[j],engy,prof_density,densitynH,Yprofs,electrondensity,phelxs,prof_temperature,elID,kB,elvolt)
                    prof_other[:,j] = ratev.copy()
            prof_gamma = prof_phionrate + prof_scdryrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_other + prof_colion
            #prof_gamma = prof_phionrate + prof_scdryrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_other
            #prof_gamma = prof_phionrate + prof_scdryrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_colion
            #prof_gamma = prof_phionrate + prof_scdryrate + prof_other + prof_colion
            #prof_gamma = prof_phionrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_other + prof_colion
            #prof_gamma = prof_scdryrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_other + prof_colion

            #prof_gamma = prof_phionrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_colion + prof_other
            # density of this specie = unionized fraction * H volume density * number abundance relative to H
            HIdensity  = Yprofs[:,elID["H I"].id]  * densitynH * elID["H I"].abund
            HeIdensity = Yprofs[:,elID["He I"].id] * densitynH * elID["He I"].abund
            HIdensity  = HIdensity.reshape((npts,1)).repeat(nions,axis=1)
            HeIdensity = HeIdensity.reshape((npts,1)).repeat(nions,axis=1)
            prof_alpha = edens_allions*prof_recomb + HIdensity*prof_recombCTHI  + HeIdensity*prof_recombCTHeI
            #prof_alpha = edens_allions*prof_recomb
            # Finally recalculate the rates
            prof_rates = prof_gamma / prof_alpha
            #rate_HI   = gamma_HI / (electrondensity * alphaB_HI(prof_temperature))
            #rate_HeI  = gamma_HeI / (electrondensity * alphaB_HeI(prof_temperature))
            #rate_HeII = gamma_HeII / (electrondensity * alphaB_HeII(prof_temperature))
            inneriter += 1
            #print "   Rates Iteration {0:d}   <--[  {1:d}/{2:d} {3:d}/{2:d} {4:d}/{2:d}  ]-->".format(inneriter,tstHI,npts,tstHeI,tstHeII)
            if np.array_equal(tstconv,allionpnt):
                break
            elif inneriter > 1000:
                print "Break inner loop at 1000 iterations, STATUS:"
                print "   Rates Iteration {0:d}".format(inneriter)
                for j in range(nions): print "   <--[  {0:d}/{1:d}  ]-->".format(tstconv[j],npts)
                break
            tmp_Yprofs = Yprofs.copy()
        print "Inner iteration cycled {0:d} times".format(inneriter)
        #############
        # TEST PLOT #
        #############
        #print "rates after!"
        #plt.plot(np.arange(npts),rate_HI,'r-')
        #plt.plot(np.arange(npts),rate_HeI,'g-')
        #plt.plot(np.arange(npts),rate_HeII,'b-')
        #plt.show()
        #plt.clf()
        #############
        #############

        print "Calculating heating rate"
        # Construct an array of ionization energies and the corresponding array for the indices
        ionlvl = np.zeros(nions,dtype=np.float)
        for j in range(nions):
            ionlvl[j] = elID[ions[j]].ip*elvolt/planck
        # Photoionization heating
        prof_eps  = 4.0*np.pi * calc_Jnur.phheatrate_allion(jnurarr, phelxs, nuzero, ionlvl, planck)
        #eps_HI   = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HI, nuzero, elID["H I"][3]*1.602E-19/6.626E-34)
        #eps_HeI  = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HeI, nuzero, elID["He I"][3]*1.602E-19/6.626E-34)
        #eps_HeII = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HeII, nuzero, elID["He II"][3]*1.602E-19/6.626E-34)
        prof_phionheatrate = np.zeros(npts,dtype=np.float)
        for j in range(nions):
            prof_phionheatrate += prof_eps[:,j]*densitynH*elID[ions[j]].abund*Yprofs[:,j]
        #phion_heat_rate = eps_HI*densitynH*prof_YHI + eps_HeI*densitynH*prim_He*prof_YHeI + eps_HeII*densitynH*prim_He*prof_YHeII
        # Secondary electron photoheating rate (Shull & van Steenberg 1985)
        heat_HI  = 4.0*np.pi * calc_Jnur.scdryheatrate(jnurarr,nuzero,phelxs[:,elID["H I"].id],electrondensity/(densitynH*(1.0+2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, planck, elvolt, 0)
        #heat_DI  = 4.0*np.pi * calc_Jnur.scdryheatrate(jnurarr,nuzero,phelxs[:,elID["D I"].id],electrondensity/(densitynH*(1.0+2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, planck, elvolt, 1)
        heat_HeI = 4.0*np.pi * calc_Jnur.scdryheatrate(jnurarr,nuzero,phelxs[:,elID["He I"].id],electrondensity/(densitynH*(1.0+2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, planck, elvolt, 2)
        #scdry_heat_rate = heat_HI*densitynH*Yprofs[:,elID["H I"].id] + heat_DI*densitynH*elID["D I"].abund*Yprofs[:,elID["D I"].id] + heat_HeI*densitynH*prim_He*Yprofs[:,elID["He I"].id]
        scdry_heat_rate = heat_HI*densitynH*Yprofs[:,elID["H I"].id] + heat_HeI*densitynH*prim_He*Yprofs[:,elID["He I"].id]

        # Finally, the total heating rate is:
        total_heat = prof_phionheatrate + scdry_heat_rate
        #total_heat = prof_phionheatrate
        #plt.plot(np.log10(prof_temperature),np.log10(total_heat),'k-')

        print "Deriving the temperature profile"
        #prof_temperature, temp, colexc, colion, colrec, diel, brem, comp = calc_Jnur.thermal_equilibrium(total_heat, electrondensity, densitynH, prof_YHI, prof_YHeI, prof_YHeII, prim_He, redshift)
        #prof_temperature, temp, coolfunc = calc_Jnur.thermal_equilibrium(total_heat, electrondensity, densitynH, prof_YHI, prof_YHeI, prof_YHeII, prim_He, redshift)
        #prof_temperature = calc_Jnur.thermal_equilibrium(total_heat, electrondensity, densitynH, Yprofs[:,elID["H I"].id], Yprofs[:,elID["He I"].id], Yprofs[:,elID["He II"].id], prim_He, redshift)
        old_temperature = prof_temperature.copy()
        if not close:
            prof_temperature = calc_Jnur.thermal_equilibrium_full(total_heat, old_temperature, electrondensity, densitynH, Yprofs[:,elID["H I"].id], Yprofs[:,elID["He I"].id], Yprofs[:,elID["He II"].id], prim_He, redshift)
        else:
            prof_temperature = coolfunc.thermal_equilibrium(total_heat,coolingcurves,densitynH,densitynH*Yprofs[:,elID["H I"].id]*elID["H I"].abund,old_temperature)
        if np.size(np.where(prof_temperature<=1000.0001)[0]) != 0:
            print "ERROR :: Profile temperature was estimated to be <= 1000 K"
            print "         The code is not currently designed to work in this regime"
            print "         Try a smaller radial range, or stronger radiation field"
            wtmp = np.where((prof_temperature<=1000.0001) | np.isnan(prof_temperature))
            prof_temperature[wtmp] = 1000.0001
            #assert(False)
        # Now make sure that the temperature jump is small
        if True:
            tmptemp = old_temperature-prof_temperature
            tmpsign = np.sign(tmptemp)
            tmptemp[np.where(np.abs(tmptemp)>100.0)] = 100.0
            tmptemp = np.abs(tmptemp)*tmpsign
            prof_temperature = old_temperature-tmptemp
        #print "ERROR :: TEMPERATURE SET TO A CONSTANT"
        #prof_temperature = 1.0E4*np.ones(npts)
        #np.save("tempcheck",prof_temperature)#prof_eps[:,elID["He II"].id])
        #print 1/0
        #prof_temperature = calc_Jnur.thermal_equilibrium_cf(total_heat, 10.0**coolfunc, 10.0**logT)
        #plt.plot(np.log10(temp),np.log10(colexc),'r-')
        #plt.plot(np.log10(temp),np.log10(colion),'r--')
        #plt.plot(np.log10(temp),np.log10(colrec),'g-')
        #plt.plot(np.log10(temp),np.log10(diel),'g--')
        #plt.plot(np.log10(temp),np.log10(brem),'b-')
        #plt.plot(np.log10(temp),np.log10(comp),'b--')
        #plt.plot(np.log10(cmtopc*radius),np.log10(prof_temperature),'k-')
        #plt.show()
        #plt.clf()
        #############
        # TEST PLOT #
        #############
        #print "photoheating rate!"
        #plt.plot(np.arange(radius.size),prof_YHI,'r-')
        #plt.plot(np.arange(radius.size),prof_YHeI,'g-')
        #plt.plot(np.arange(radius.size),prof_YHeII,'b-')
        #plt.show()
        #plt.clf()
        #print "heating rate!"
        #plt.plot(np.log10(radius*cmtopc),prof_temperature,'k-')
        #plt.plot(np.arange(radius.size),total_heat,'k-')
        #plt.plot(np.arange(radius.size),phion_heat_rate,'r-')
        #plt.plot(np.arange(radius.size),scdry_heat_rate,'b-')
        #plt.show()
        #plt.clf()
        #print "temperature!"
        #plt.plot(np.log10(radius*cmtopc),prof_temperature,'k-')
        #plt.plot(np.arange(radius.size),prof_temperature,'k-')
        #plt.show()
        #plt.clf()
        #print "cooling functions!"
        #plt.plot(np.log10(radius*cmtopc),prof_temperature,'k-')
        #plt.plot(10.0**np.linspace(3.0,6.0,coolfunc.size),coolfunc,'b-')
        #plt.plot(10.0**np.linspace(3.0,6.0,coolfunc.size),coolfunc,'bx')
        #plt.plot([1.0E3,1.0E6],[total_heat[0],total_heat[0]],'r-')
        #plt.plot([prof_temperature[0],prof_temperature[0]],[total_heat[0],total_heat[0]],'r-')
        #plt.plot([old_temperature[0],old_temperature[0]],[total_heat[0],total_heat[0]],'r--')
        #plt.show()
        #plt.clf()
        #############
        #############

        #np.save("tempcheck",prof_temperature)
        #print 1/0

        # Calculate a temperature profile by assuming thermal equilibrium (balancing heating and cooling)
        # Derive the total heating rate

        #if iteration > 0:
            #print "Calculating heating rate"
            ## Photoionization heating
            #eps_HI   = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HI, nuzero, 13.59844*1.602E-19/6.626E-34)
            #eps_HeI  = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HeI, nuzero, 24.58741*1.602E-19/6.626E-34)
            #eps_HeII = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HeII, nuzero, 54.41778*1.602E-19/6.626E-34)
            #phion_heat_rate = eps_HI*densitynH*prof_YHI + eps_HeI*densitynH*prim_He*prof_YHeI + eps_HeII*densitynH*prim_He*prof_YHeII
            ## Secondary electron photoheating rate (Shull & van Steenberg 1985)
            #heat_HI  = 4.0*np.pi * calc_Jnur.scdryheatrate(jnurarr,nuzero,phelxs_HI,electrondensity/(densitynH*(1.0+2.0*prim_He)),0)
            #heat_HeI = 4.0*np.pi * calc_Jnur.scdryheatrate(jnurarr,nuzero,phelxs_HeI,electrondensity/(densitynH*(1.0+2.0*prim_He)),1)
            #scdry_heat_rate = heat_HI*densitynH*prof_YHI + heat_HeI*densitynH*prim_He*prof_YHeI
            ## Finally, the total heating rate is:
            #total_heat = phion_heat_rate + scdry_heat_rate
            #plt.plot(np.log10(cmtopc*radius),np.log10(total_heat*densitynH**2),'k-')

            #print "Deriving the temperature profile"
            #prof_temperature, temp, coolfunc = calc_Jnur.thermal_equilibrium(total_heat, electrondensity, densitynH, prof_YHI, prof_YHeI, prof_YHeII, prim_He, redshift)
            #prof_temperature = calc_Jnur.thermal_equilibrium_cf(total_heat, 10.0**coolfunc, 10.0**logT)
            #plt.plot(np.log10(cmtopc*radius),np.log10(prof_temperature),'k-')
            #plt.plot(np.log10(temp),np.log10(coolfunc),'k-')
            #plt.show()
            #plt.clf()

        if iteration >= 100 and iteration%1 == 0:
            print "Averaging the stored Yprofs"
            Yprofs = np.mean(store_Yprofs, axis=2)
            #Yprofs = uniform_filter1d(Yprofs, 5, axis=0)
        #tstcrit = (np.abs(old_Yprofs-Yprofs)<concrit).astype(np.int).sum(0)
        tstcrit = ( (np.abs((old_Yprofs-Yprofs)/Yprofs)<concrit)|(Yprofs==0.0)).astype(np.int).sum(axis=0)
        # Make a random adjustment to the Yprofs to ensure the solution doesn't get locked
        #Yprofs *= 1.0 + np.random.uniform(-concrit/2.0,concrit/2.0,Yprofs.shape)
        #Yprofs[np.where(Yprofs<0.0)] = 0.0
        #Yprofs[np.where(Yprofs>1.0)] = 1.0
        if np.array_equal(tstcrit,allionpnt):
            # Once we are close to convergence, use a more reliable cooling function
            print "Getting close!! Try a more accurate cooling function"
            close = True
            break
            miniter += iteration
        #tstHI   = (np.abs(old_prof_YHI-prof_YHI)<concrit).astype(np.int).sum()
        #tstHeI  = (np.abs(old_prof_YHeI-prof_YHeI)<concrit).astype(np.int).sum()
        #tstHeII = (np.abs(old_prof_YHeII-prof_YHeII)<concrit).astype(np.int).sum()

        print "STATISTICS --"
        print "ION  INDEX   OLD VALUE    NEW VALUE   |OLD-NEW|"
        w_maxoff   = np.argmax(np.abs((old_Yprofs-Yprofs)/Yprofs),axis=0)
#		w_minoff   = np.argmin(np.abs(old_Yprofs-Yprofs),axis=0)
        for j in range(nions):
            print ions[j], w_maxoff[j], old_Yprofs[w_maxoff[j],j], Yprofs[w_maxoff[j],j], np.abs((old_Yprofs[w_maxoff[j],j] - Yprofs[w_maxoff[j],j])/Yprofs[w_maxoff[j],j])
#			print ions[j], w_minoff[j], old_Yprofs[w_minoff[j],j], Yprofs[w_minoff[j],j], np.abs(old_Yprofs[w_minoff[j],j] - Yprofs[w_minoff[j],j])
        #print w_HI, np.abs(old_prof_YHI[w_HI]-prof_YHI[w_HI]), old_prof_YHI[w_HI], prof_YHI[w_HI]
        #print w_HeI, np.abs(old_prof_YHeI[w_HeI]-prof_YHeI[w_HeI]), old_prof_YHeI[w_HeI], prof_YHeI[w_HeI]
        #print w_HeII, np.abs(old_prof_YHeII[w_HeII]-prof_YHeII[w_HeII]), old_prof_YHeII[w_HeII], prof_YHeII[w_HeII]

        # Check if the stopping criteria were met
        if iteration > maxiter:
            print "Break outer loop at maxiter={0:d} iterations, STATUS:".format(maxiter)
            #print "   Rates Iteration {0:d}   <--[  {1:d}/{2:d} {3:d}/{2:d} {4:d}/{2:d}  ]-->".format(iteration,tstHI,npts,tstHeI,tstHeII)
            break
        #############
        # TEST PLOT #
        #############
        if iteration >= maxiter-10:
            colr = colormap(normalize(maxiter-iteration))
            plt.subplot(221)
            plt.plot(np.log10(radius*cmtopc),Yprofs[:,elID["H I"].id], linestyle='-', color=colr)
            plt.subplot(222)
            plt.plot(np.log10(radius*cmtopc),Yprofs[:,elID["He I"].id], linestyle='-', color=colr)
            plt.subplot(223)
            plt.plot(np.log10(radius*cmtopc),Yprofs[:,elID["He II"].id], linestyle='-', color=colr)
            plt.subplot(224)
            yplttst = np.abs((old_Yprofs[:,elID["He II"].id]-Yprofs[:,elID["He II"].id])/Yprofs[:,elID["He II"].id])
            plt.plot(np.log10(radius*cmtopc),yplttst, linestyle='-', color=colr)
        elif iteration == 100000:
            plt.plot(np.log10(radius*cmtopc),Yprofs[:,elID["H I"].id],'r-')
            plt.plot(np.log10(radius*cmtopc),Yprofs[:,elID["He I"].id],'g-')
            plt.plot(np.log10(radius*cmtopc),Yprofs[:,elID["He II"].id],'b-')
            plt.show()
            plt.clf()
            atst = raw_input("Should I break? (y/n) ")
            if atst=="y": break
            answer = int(raw_input("When should I plot next time"))
        elif iteration == answer:
            plt.plot(np.log10(radius*cmtopc),Yprofs[:,elID["H I"].id],'r-')
            plt.plot(np.log10(radius*cmtopc),Yprofs[:,elID["He I"].id],'g-')
            plt.plot(np.log10(radius*cmtopc),Yprofs[:,elID["He II"].id],'b-')
            plt.show()
            plt.clf()
            answer = int(raw_input("When should I plot next time"))
        #############
        #############

    if iteration >= maxiter-10:
        #plt.clf()
        plt.show()
        #pdb.set_trace()
# 	mintemp = np.min(prof_temperature)
# 	print "-->", mintemp
# 	cftemp = 10.0**np.linspace(3.0,6.0,1000)
# 	plt.plot(cftemp,coolfunc,'b-')
# 	plt.plot([1.0E3,1.0E6],[total_heat[0],total_heat[0]],'r-')
# 	plt.plot([mintemp,mintemp],[np.min(coolfunc),np.max(coolfunc)],'r-')
# 	plt.show()
# 	plt.clf()

    # Calculate the density profiles
    print "Calculating volume density profiles"
    for j in range(nions):
        # density of this specie = unionized fraction * H volume density * number abundance relative to H
        prof_density[:,j] = Yprofs[:,j] * densitynH * elID[ions[j]].abund
        print ions[j], np.max(Yprofs[:,j]), np.max(prof_density[:,j])

    print "Calculating column density profiles"
    prof_coldens = np.zeros_like(prof_density)
    for j in range(nions):
        if options["geometry"]["use"] == "NFW":
            coldens = calc_Jnur.coldensprofile(prof_density[:,j], radius)
            prof_coldens[:,j] = coldens.copy()
        elif options["geometry"]["use"] == "PP":
            coldens = calc_Jnur.calc_coldensPP(prof_density[:,j], radius)
            prof_coldens[:,j] = coldens.copy()
            print ions[j], np.log10(prof_coldens[0,j]), np.max(np.log10(prof_coldens[:,j]))

    print "Calculating Ha surface brightness profile"
    Harecomb = recomb.Ha_recomb(prof_temperature)
    #plt.clf()
    #plt.plot(prof_temperature, Harecomb, 'k-')
    #plt.show()
    HIIdensity = densitynH * (1.0-Yprofs[:,elID["H I"].id])
    elecprot = Harecomb*electrondensity*HIIdensity
    HaSB = (1.0/(4.0*np.pi)) * calc_Jnur.coldensprofile(elecprot, radius)  # photons /cm^2 / s / SR
    HaSB = HaSB * (1.98645E-8/6563.0)/4.254517E10   # ergs /cm^2 / s / arcsec^2

    print "--->", np.max(np.log10(prof_coldens[:,elID["H I"].id]))
    print "inner iter = ", inneriter
    timeB = time.time()
    print "Test completed in {0:f} mins".format((timeB-timeA)/60.0)

    # Check if the output directory exists
    out_dir = 'output' + '/' + options["run"]["outdir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save the results        
    useg = options["geometry"]["use"]
    if useg == "NFW":
        mstring = ("{0:3.2f}".format(np.log10(options["geometry"][useg][0]))).replace(".","d").replace("+","p")
        rstring = ("{0:3.2f}".format(redshift)).replace(".","d").replace("+","p")
        bstring = ("{0:+3.2f}".format(np.log10(options["geometry"][useg][2]))).replace(".","d").replace("+","p").replace("-","m")
        if options["radfield"]=="HM12":
            hstring = ("HMscale{0:+3.2f}".format(np.log10(options["HMscale"]))).replace(".","d").replace("+","p").replace("-","m")
            #hstring = ("Hescale{0:+3.2f}".format(Hescale)).replace(".","d").replace("+","p").replace("-","m")
        elif options["radfield"][0:2]=="PL":
            hstring = options["radfield"]
        outfname = out_dir + "/{0:s}_mass{1:s}_redshift{2:s}_baryscl{3:s}_{4:s}_{5:d}-{6:d}".format(useg,mstring,rstring,bstring,hstring,npts,nummu)
        #outfname = "convergence/{0:s}_mass{1:s}_redshift{2:s}_baryscl{3:s}_{4:s}_{5:d}-{6:d}".format(useg,mstring,rstring,bstring,hstring,npts,nummu)
        #outfname = "output_z0UVB/{0:s}_mass{1:s}_redshift{2:s}_{3:s}_{4:d}-{5:d}".format(useg,mstring,rstring,hstring,npts,nummu)
        print "Saving file {0:s}.npy".format(outfname)
        tmpout = np.concatenate((radius.reshape((npts,1))*cmtopc,prof_temperature.reshape((npts,1)),densitynH.reshape((npts,1)),HaSB.reshape((npts,1)),prof_density,prof_coldens),axis=1)
        #np.save(outfname, np.transpose((radius,electrondensity,densitynH,prof_temperature,prof_pressure/kB,prof_YHI,prof_YHeI,prof_YHeII,prof_HIcolumndensity,prof_HeIcolumndensity,prof_HeIIcolumndensity,phionrate_HI,phionrate_HeI,phionrate_HeII,scdryrate_HI,scdryrate_HeI,B1_HeII,B2_HeI,B3_HI,B6_HeI,B7_HI)))
        np.save(outfname, tmpout)
    elif useg == "PP":
        dstring = ("{0:+3.2f}".format(options["geometry"][useg][0])).replace(".","d").replace("+","p").replace("-","m")
        rstring = ("{0:4.2f}".format(options["geometry"][useg][1])).replace(".","d").replace("+","p")
        if options["radfield"]=="HM12":
            hstring = ("HMscale{0:+3.2f}".format(np.log10(options["HMscale"]))).replace(".","d").replace("+","p").replace("-","m")
        elif options["radfield"][0:2]=="PL":
            hstring = options["radfield"]
        outfname = "output/{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}".format(useg,dstring,rstring,hstring,npts)
        #outfname = "output/ADJ-nothing_{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}".format(useg,dstring,rstring,hstring,npts)
        #outfname = "output/ADJ-chrgtran_{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}".format(useg,dstring,rstring,hstring,npts)
        #outfname = "output/ADJ-nometal_{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}".format(useg,dstring,rstring,hstring,npts)
        #outfname = "output/ADJ-colion_{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}".format(useg,dstring,rstring,hstring,npts)
        #outfname = "output/ADJ-other_{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}".format(useg,dstring,rstring,hstring,npts)
        #outfname = "output/ADJ-chrgtrani_{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}".format(useg,dstring,rstring,hstring,npts)
        #outfname = "output/ADJ-chrgtranr_{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}".format(useg,dstring,rstring,hstring,npts)
        #outfname = "output/ADJ-scdry_{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}".format(useg,dstring,rstring,hstring,npts)
        #outfname = "output/ADJ-prmry_{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}".format(useg,dstring,rstring,hstring,npts)
        #outfname = "output/CloudyTest_{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}".format(useg,dstring,rstring,hstring,npts)
        print "Saving file {0:s}.npy".format(outfname)
        tmpout = np.concatenate((radius.reshape((npts,1))*cmtopc,prof_temperature.reshape((npts,1)),densitynH.reshape((npts,1)),prof_density,prof_coldens),axis=1)
        #np.save(outfname, np.transpose((radius,electrondensity,densitynH,prof_temperature,prof_pressure/kB,prof_YHI,prof_YHeI,prof_YHeII,prof_HIcolumndensity,prof_HeIcolumndensity,prof_HeIIcolumndensity,phionrate_HI,phionrate_HeI,phionrate_HeII,scdryrate_HI,scdryrate_HeI,B1_HeII,B2_HeI,B3_HI,B6_HeI,B7_HI)))
        np.save(outfname, tmpout)
    else:
        tmpout = np.concatenate((radius.reshape((npts,1))*cmtopc,prof_temperature.reshape((npts,1)),prof_density),axis=1)
        np.savetxt("PP_HI21p08.dat",tmpout)

    # Stop the program if a large H I column density has already been reached
    if np.max(np.log10(prof_coldens[:,elID["H I"].id])) > 22.0:
        print "Terminating after maximum N(H I) has been reached"
        sys.exit()

    plotitup=False
    if plotitup:
        if options["geometry"]["use"] == "NFW":
            xval = radius/rscale
            Mds = 4.0*np.pi*rhods*(rscale**3)/3.0
            fgas = calc_Jnur.fgasx(densitym,radius,rscale)
            plt.figure(1)
            plt.plot(np.log10(radius*cmtopc),np.log10(Mds*3.0*(np.log(1.0+xval) - xval/(1.0+xval))),'k-')
            plt.plot(np.log10(radius*cmtopc),np.log10((cosmopar[1]/(cosmopar[3]-cosmopar[1]))*Mds*3.0*(np.log(1.0+xval) - xval/(1.0+xval))),'b--')
            plt.plot(np.log10(radius*cmtopc),np.log10(4.0*np.pi*rscale**3*fgas),'g-')

        plt.figure(1)
        plt.plot(prof_temperature,np.log10(prof_density[:,elID["H I"].id]),'r-')
        plt.plot(prof_temperature,np.log10(prof_density[:,elID["He I"].id]),'g-')
        plt.plot(prof_temperature,np.log10(prof_density[:,elID["He II"].id]),'b-')

        plt.figure(2)
        plt.plot(radius*cmtopc,Yprofs[:,elID["H I"].id],'r-')
        plt.plot(radius*cmtopc,Yprofs[:,elID["He I"].id],'g-')
        plt.plot(radius*cmtopc,Yprofs[:,elID["He II"].id],'b-')

        plt.figure(3)
        plt.subplot(2,2,1)
        plt.plot(densitynH,prof_temperature,'bx')
        plt.yscale('log')
        plt.subplot(2,2,2)
        #plt.plot(densitynH,prof_pressure/kB,'gx')
        plt.subplot(2,2,3)
        plt.plot(radius*cmtopc,np.log10(densitynH),'k-')
        plt.subplot(2,2,4)
        plt.plot(radius*cmtopc,prof_temperature,'k-')

        plt.figure(4)
        plt.plot(radius*cmtopc,prof_coldens[:,elID["H I"].id],'r-')
        plt.plot(radius*cmtopc,prof_coldens[:,elID["He I"].id],'g-')
        plt.plot(radius*cmtopc,prof_coldens[:,elID["He II"].id],'b-')

        plt.figure(5)
        plt.plot(radius*cmtopc,np.log10(prof_coldens[:,elID["D I"].id]/prof_coldens[:,elID["H I"].id])-np.log10(elID["D I"].abund/elID["H I"].abund),'g-')
        plt.show()
        plt.clf()
    # Return the output filename to be used as the input to the next iteration    
    return outfname + '.npy'

    # dispose of process pool
    if ncpus > 1:
        pool.close()
        pool.join()

#jnurarr = calc_Jnur.Jnur(density, radius, jzero, phelxs_HI, nummu)
#coldensHI, muarr = calc_Jnur.calc_coldens(density, radius, nummu)
#coldensHeI, muarr = calc_Jnur.calc_coldens(density, radius, nummu)
#coldensHeII, muarr = calc_Jnur.calc_coldens(density, radius, nummu)
#jnurarr_split = calc_Jnur.nint_costheta(coldensHI, phelxs_HI, coldensHeI, phelxs_HeI, coldensHeII, phelxs_HeII, muarr, jzero)

#plt.plot(engy,phelxs_HI*jnurarr[:,250],'k-')
#plt.show()
#plt.clf()

# Calculate the local primary photoionization rate as a function of radius

#plt.plot(radius,lphirate,'k-')
#plt.show()
