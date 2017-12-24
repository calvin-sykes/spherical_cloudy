import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
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
import cython_fns
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
        coldens, muarr = cython_fns.calc_coldens(prof, radius, nummu)
        return [j,coldens,muarr]
    elif geom == "PP":
        coldens = cython_fns.calc_coldensPP(prof, radius)
        return [j,coldens]
    else:
        print "ERROR :: Geometry {0:s} is not allowed".format(geom)
        assert(False)


def mpphion(j, jnurarr, phelxs, nuzero, planck):
    phionxsec = 4.0*np.pi * cython_fns.phionrate(jnurarr, phelxs, nuzero, planck)
    return [j,phionxsec]


def get_radius(virialr, scale, npts, method=0):
    if method == 0:
        # Linear scaling
        radius = np.linspace(0.0, virialr*scale, npts)
    elif method == 1:
        pass
        # Linear scaling, but split up
        radius = np.linspace(0.0,0.1*virialr*scale,npts/10)
        radius = np.append(radius, np.linspace(0.1*virialr*scale,virialr*scale,7*npts/10)[1:])
        radius = np.append(radius, np.linspace(virialr*scale,10.0*virialr*scale,npts/10)[1:])
        lover = npts - radius.size + 1
        radius = np.append(radius, 10.0*np.linspace(virialr*scale,100.0*virialr*scale,lover)[1:])
    elif method == 2:
        # Log scaling
        radius = np.append(0.0, 10.0**np.linspace(np.log10(virialr*1.0E-4), np.log10(virialr*scale), npts-1))
    else:
        print "radius method is not yet defined"
        sys.exit()
    if virialr not in radius:
        radius = np.append(virialr, radius[:-1])
        radius.sort()
    return radius


def mangle_string(str):
    return str.replace(".","d").replace("+","p").replace("-","m")


def get_halo(hmodel,redshift,gastemp,bturb,metals=1.0,Hescale=1.0,cosmopar=np.array([0.673,0.04910,0.685,0.315]),ions=["H I", "He I", "He II"],prevfile=None,options=None):
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

    geom = options["geometry"]
    geomscale = options["geomscale"]
    radmethod = 2  # Method used to define the radial coordinate

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
        prim_He = elID["He I"].abund * Hescale
    elif "He II" in ions:
        prim_He = elID["He II"].abund * Hescale
    else:
        print "ERROR :: You must include ""He I"" and ""He II"" in your model"
        assert(False)
    if "H I" not in ions:
        print "ERROR :: You must include ""H I"" in your model"
        assert(False)

    print "Loading radiation fields"
    if options["radfield"] == "HM12":
        # Get the Haardt & Madau (2012) background at the appropriate redshift
        jzero, nuzero = radfields.HMbackground(elID,redshift=redshift,options=options)
        jzero *= options["HMscale"]
    elif options["radfield"][0:2] == "PL":
        jzero, nuzero = radfields.powerlaw(elID,options=options)
    else:
        print "The radiation field {0:s} is not implemented yet".format(options["radfield"])
        sys.exit()

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
        xsecv = phionxsec.rate_function_arr(engy,phelxsdata[ions[j]])
        phelxs[:,j] = xsecv.copy()
    
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

    print "Loading Charge Transfer rate coefficients"
    chrgtranrate = chrgtran.load_data(elID)
    chrgtran_HItargs = chrgtranrate["H I"].keys()
    chrgtran_HIItargs = chrgtranrate["H II"].keys()
    chrgtran_HeItargs = chrgtranrate["He I"].keys()
    chrgtran_HeIItargs = chrgtranrate["He II"].keys()

    close=False

    # if string is passed, interpret as filename for the previous run
    # this is used as an initial solution to speed up convergence
    if prevfile is not None:
        if geom == "NFW":
            print "Loading file {0:s}".format(prevfile)
            tdata = np.load(prevfile)
            strt = 4
            numarr = tdata.shape[1]
            arridx = dict({})
            arridx["voldens"] = dict({})
            arridx["coldens"] = dict({})
            for i in range((numarr-strt)/2):
                arridx["voldens"][ions[i]] = strt + i
                arridx["coldens"][ions[i]] = strt + i + (numarr-strt)/2
            old_radius = get_radius(virialr, geomscale, npts, method=radmethod)
            if old_radius.size != npts:
                print "Error defining radius"
                sys.exit()
            prof_coldens = np.zeros((npts,nummu,nions))
            prof_density = np.zeros((npts,nions))
            radius = old_radius.copy()
            prof_temperature = tdata[:,1]
            temp_densitynH = tdata[:,2]
            # Extract the data from the array
            Yprofs = 1.0E-1*np.ones((npts,nions))
            for j in range(nions):
                # density of this specie = unionized fraction * H volume density * number abundance relative to H
                prof_density[:,j] = tdata[:,arridx["voldens"][ions[j]]]
                Yprofs[:,j] = prof_density[:,j] / (temp_densitynH * elID[ions[j]].abund)
            prof_phionrate = np.zeros((npts,nions))
            densitym  = temp_densitynH * protmss * (1.0 + 4.0*prim_He)
        elif geom == "PP":
            print "Never needed this"
            assert(False)
        else:
            print "Not implemented yet"
            assert(False)
    else: # prevfile is None
        # Set the gas conditions
        if geom == "NFW":
            radius = get_radius(hmodel.rvir, geomscale, npts, method=radmethod)
            if radius.size != npts:
                print "Error defining radius"
                sys.exit()
            temp_densitynH = np.ones(npts)
            prof_coldens = np.zeros((npts,nummu,nions))
            prof_density = 1.0E-1*np.ones((npts,nions))
        elif geom == "PP":
            radius  = np.linspace(0.0,1000.0 * geomscale/cmtopc,npts)
            densitynH = np.ones(npts) * (10.0**hmodel.mvir)
            prof_coldens = np.zeros((npts,nions))
            prof_density = 1.0E-1*np.ones((npts,nions))
        prof_temperature = gastemp * np.ones(npts)
        prof_phionrate = np.zeros((npts,nions))
        Yprofs = 1.0E-2*np.ones((npts,nions))
        densitym = protmss * (1.0 + 4.0*prim_He) * np.ones(npts) # Default to be used for PP

    # An array used to check if convergence has been reached for each ion and in each cell.
    allionpnt = npts*np.ones(nions,dtype=np.int)

    # Calculate the mass of baryons involved:
    if geom == "NFW":
        # Set the mass density profile
        barymass = hmodel.mvir * hmodel.baryfrac
        densitym = np.ones(npts) * barymass / (4.0 * np.pi * (hmodel.rvir**3) / 3.0)
    elif geom == "PP":
        pass # The mass density is set earlier for PP

    # Set the plotting colormap
    colormap = cm.bwr
    normalize = mcolors.Normalize(vmin=0, vmax=10)

    # Store old Yprofs
    nstore = 10
    istore = 0
    store_Yprofs = np.zeros(Yprofs.shape + (nstore,))

    # Set the stopping criteria flags
    tstcrit  = np.zeros(nions,dtype=np.int)
    iteration = 0
    old_Yprofs = None
    while (not np.array_equal(tstcrit,allionpnt)) or (iteration <= miniter) or (not close):
        iteration += 1
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
        if geom == "NFW":
            fgas = cython_fns.fgasx(densitym,radius,hmodel.rscale)
            prof_pressure = cython_fns.pressure(prof_temperature,radius,masspp,hmodel,bturb,Gcons,kB,protmss)
            turb_pressure = 0.75*densitym*(bturb**2.0)
             # Calculate the thermal pressure, and ensure positivity
            ther_pressure = prof_pressure-turb_pressure
            wther = np.where(ther_pressure<0.0)
            ther_pressure[wther] = 0.0
            # Calculate gas density profile
            temp_densitynH = ther_pressure / (1.5 * kB * prof_temperature * dof)
            if (temp_densitynH[0]==0.0):
                print "WARNING :: central density is zero"
                print "        :: Assuming no turbulent pressure for this iteration"
                temp_densitynH = prof_pressure / (1.5 * kB * prof_temperature * dof)
            temp_densitynH /= temp_densitynH[0]
            rintegral = cython_fns.mass_integral(temp_densitynH,radius,hmodel.rvir)
            cen_density = barymass / (4.0 * np.pi * protmss * (1.0 + 4.0*prim_He) * rintegral)
            densitynH = cen_density * temp_densitynH
            densitym  = densitynH * protmss * (1.0 + 4.0*prim_He)

        # Update the volume density of the unionized species
        for j in range(nions):
            # density of this specie = unionized fraction * H volume density * number abundance relative to H
            prof_density[:,j] = Yprofs[:,j] * densitynH * elID[ions[j]].abund

        # Compute the electron density
        electrondensity = densitynH * ( (1.0-Yprofs[:,elID["H I"].id]) + prim_He*Yprofs[:,elID["He II"].id] + 2.0*prim_He*(1.0-Yprofs[:,elID["He I"].id]-Yprofs[:,elID["He II"].id]) )

        # Calculate the column density arrays,
        if ncpus == 1:
            for j in range(nions):
                if geom["use"] == "NFW":
                    coldens, muarr = cython_fns.calc_coldens(prof_density[:,j], radius, nummu)
                    prof_coldens[:,:,j] = coldens.copy()
                elif geom["use"] == "PP":
                    coldens = cython_fns.calc_coldensPP(prof_density[:,j], radius)
                    prof_coldens[:,j] = coldens.copy()
        else:
            async_results = []
            for j in range(nions):
                async_results.append(pool.apply_async(mpcoldens, (j, prof_density[:,j], radius, nummu, geom)))
            map(ApplyResult.wait, async_results)
            for j in range(nions):
                getVal = async_results[j].get()
                if geom == "NFW":
                    prof_coldens[:,:,getVal[0]] = getVal[1].copy()
                    muarr = getVal[2]
                elif geom == "PP":
                    prof_coldens[:,getVal[0]] = getVal[1].copy()

        # integrate over all angles,
        if geom == "NFW":
            print "Integrate over all angles"
            jnurarr = cython_fns.nint_costheta(prof_coldens, phelxs, muarr, jzero)
        elif geom == "PP":
            jnurarr = cython_fns.nint_pp(prof_coldens, phelxs, jzero)

        # and calculate the photoionization rates
        print "Calculating phionization rates"
        if ncpus == 1:
            for j in range(nions):
                phionr = 4.0*np.pi * cython_fns.phionrate(jnurarr, phelxs[:,j], nuzero, planck*1.0E7)
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
            if usecolion == "Dere2007":
                ratev = colioniz.rate_function_Dere2007(1.0E-7*prof_temperature*kB/elvolt, colionrate[ions[j]])
            elif usecolion == "Voronov1997":
                ratev = colioniz.rate_function_arr(1.0E-7*prof_temperature*kB/elvolt, colionrate[ions[j]])
            prof_colion[:,j] = ratev.copy()

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
            async_results.append(pool.apply_async(cython_fns.scdryrate, (jnurarr, nuzero,
                    phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id], # photoionisation cross-sections
                    prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id], electrondensity/(densitynH*(1.0 + 2.0*prim_He)), # densities
                    elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, # ionisation potentials
                    planck, elvolt, 0))) # constants
            # He I
            async_results.append(pool.apply_async(cython_fns.scdryrate, (jnurarr, nuzero,
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
                    ratev = 4.0*np.pi * cython_fns.scdryrate(jnurarr, nuzero, phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id],
                       prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id],
                        electrondensity/(densitynH*(1.0 + 2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, planck, elvolt, 0)
                    prof_scdryrate[:,j] = ratev.copy()
                elif ions[j] == "He I":
                    ratev = 4.0*np.pi * 10.0 * cython_fns.scdryrate(jnurarr, nuzero, phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id],
                        prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id],
                        electrondensity/(densitynH*(1.0 + 2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, planck, elvolt, 2)
                    prof_scdryrate[:,j] = ratev.copy()

        # Calculate other forms of ionization (e.g. photons from H and He recombinations)
        print "Calculate ionization rate from recombinations of H+, He+, He++"
        prof_other = np.zeros((npts,nions))
        for j in range(nions):
            ratev = photoion.other(ions[j],engy,prof_density,densitynH,Yprofs,electrondensity,phelxs,prof_temperature,elID,kB,elvolt)
            prof_other[:,j] = ratev.copy()

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

        "Calculating recombination rates"
        prof_recomb = np.zeros((npts,nions))
        prof_recombCTHI  = np.zeros((npts,nions))
        prof_recombCTHeI = np.zeros((npts,nions))
        for j in range(nions):
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
        prof_rates = prof_gamma / prof_alpha

        # Store old and obtain new values for YHI, YHeI, YHeII
        old_Yprofs = Yprofs.copy()
        tmp_Yprofs = Yprofs.copy()
        inneriter = 0
        while True:
            #if iteration > 100 and inneriter==1: break
            # Calculate the Yprofs
            Yprofs = misc.calc_yprofs(ions,prof_rates,elID)
            # Test if the profiles have converged
            tstconv = ( (np.abs((tmp_Yprofs-Yprofs)/Yprofs)<concrit**2)|(Yprofs==0.0)).astype(np.int).sum(axis=0)
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
                    async_results.append(pool.apply_async(cython_fns.scdryrate, (jnurarr, nuzero,
                            phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id], # photoionisation cross-sections
                            prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id], electrondensity/(densitynH*(1.0 + 2.0*prim_He)), # densities
                            elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, # ionisation potentials
                            planck, elvolt, 0))) # constants
                    # He I
                    async_results.append(pool.apply_async(cython_fns.scdryrate, (jnurarr, nuzero,
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
                            ratev = 4.0*np.pi * cython_fns.scdryrate(jnurarr, nuzero, phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id],
                               prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id],
                                electrondensity/(densitynH*(1.0 + 2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, planck, elvolt, 0)
                            prof_scdryrate[:,j] = ratev.copy()
                        elif ions[j] == "He I":
                            ratev = 4.0*np.pi * 10.0 * cython_fns.scdryrate(jnurarr, nuzero, phelxs[:,elID["H I"].id], phelxs[:,elID["D I"].id], phelxs[:,elID["He I"].id], phelxs[:,elID["He II"].id],
                                prof_density[:,elID["H I"].id], prof_density[:,elID["D I"].id], prof_density[:,elID["He I"].id], prof_density[:,elID["He II"].id],
                                electrondensity/(densitynH*(1.0 + 2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, planck, elvolt, 2)
                            prof_scdryrate[:,j] = ratev.copy()
                # Colion
                for j in range(nions):
                    if usecolion == "Dere2007":
                        ratev = colioniz.rate_function_Dere2007(1.0E-7*prof_temperature*kB/elvolt, colionrate[ions[j]])
                    elif usecolion == "Voronov1997":
                        ratev = colioniz.rate_function_arr(1.0E-7*prof_temperature*kB/elvolt, colionrate[ions[j]])
                    prof_colion[:,j] = ratev.copy()
                # Other
                for j in range(nions):
                    ratev = photoion.other(ions[j],engy,prof_density,densitynH,Yprofs,electrondensity,phelxs,prof_temperature,elID,kB,elvolt)
                    prof_other[:,j] = ratev.copy()
            prof_gamma = prof_phionrate + prof_scdryrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_other + prof_colion

            # density of this specie = unionized fraction * H volume density * number abundance relative to H
            HIdensity  = Yprofs[:,elID["H I"].id]  * densitynH * elID["H I"].abund
            HeIdensity = Yprofs[:,elID["He I"].id] * densitynH * elID["He I"].abund
            HIdensity  = HIdensity.reshape((npts,1)).repeat(nions,axis=1)
            HeIdensity = HeIdensity.reshape((npts,1)).repeat(nions,axis=1)
            prof_alpha = edens_allions*prof_recomb + HIdensity*prof_recombCTHI  + HeIdensity*prof_recombCTHeI
            # Finally recalculate the rates
            prof_rates = prof_gamma / prof_alpha
            inneriter += 1
            if np.array_equal(tstconv,allionpnt):
                break
            elif inneriter > 1000:
                print "Break inner loop at 1000 iterations, STATUS:"
                print "   Rates Iteration {0:d}".format(inneriter)
                for j in range(nions): print "   <--[  {0:d}/{1:d}  ]-->".format(tstconv[j],npts)
                break
            tmp_Yprofs = Yprofs.copy()
        print "Inner iteration cycled {0:d} times".format(inneriter)

        print "Calculating heating rate"
        # Construct an array of ionization energies and the corresponding array for the indices
        ionlvl = np.zeros(nions,dtype=np.float)
        for j in range(nions):
            ionlvl[j] = elID[ions[j]].ip*elvolt/planck
        # Photoionization heating
        prof_eps  = 4.0*np.pi * cython_fns.phheatrate_allion(jnurarr, phelxs, nuzero, ionlvl, planck)
        prof_phionheatrate = np.zeros(npts,dtype=np.float)
        for j in range(nions):
            prof_phionheatrate += prof_eps[:,j]*densitynH*elID[ions[j]].abund*Yprofs[:,j]
        # Secondary electron photoheating rate (Shull & van Steenberg 1985)
        heat_HI  = 4.0*np.pi * cython_fns.scdryheatrate(jnurarr,nuzero,phelxs[:,elID["H I"].id],electrondensity/(densitynH*(1.0+2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, planck, elvolt, 0)
        heat_HeI = 4.0*np.pi * cython_fns.scdryheatrate(jnurarr,nuzero,phelxs[:,elID["He I"].id],electrondensity/(densitynH*(1.0+2.0*prim_He)), elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, planck, elvolt, 2)
        scdry_heat_rate = heat_HI*densitynH*Yprofs[:,elID["H I"].id] + heat_HeI*densitynH*prim_He*Yprofs[:,elID["He I"].id]

        # Finally, the total heating rate is:
        total_heat = prof_phionheatrate + scdry_heat_rate

        print "Deriving the temperature profile"
        old_temperature = prof_temperature.copy()
        if not close:
            prof_temperature = cython_fns.thermal_equilibrium_full(total_heat, old_temperature, electrondensity, densitynH, Yprofs[:,elID["H I"].id], Yprofs[:,elID["He I"].id], Yprofs[:,elID["He II"].id], prim_He, redshift)
        else:
            prof_temperature = coolfunc.thermal_equilibrium(total_heat,coolingcurves,densitynH,densitynH*Yprofs[:,elID["H I"].id]*elID["H I"].abund,old_temperature)
        if np.size(np.where(prof_temperature<=1000.0001)[0]) != 0:
            print "ERROR :: Profile temperature was estimated to be <= 1000 K"
            print "         The code is not currently designed to work in this regime"
            print "         Try a smaller radial range, or stronger radiation field"
            wtmp = np.where((prof_temperature<=1000.0001) | np.isnan(prof_temperature))
            prof_temperature[wtmp] = 1000.0001
        # Now make sure that the temperature jump is small
        if True:
            tmptemp = old_temperature-prof_temperature
            tmpsign = np.sign(tmptemp)
            tmptemp[np.where(np.abs(tmptemp)>100.0)] = 100.0
            tmptemp = np.abs(tmptemp)*tmpsign
            prof_temperature = old_temperature-tmptemp

        if iteration >= 100 and iteration%1 == 0:
            print "Averaging the stored Yprofs"
            Yprofs = np.mean(store_Yprofs, axis=2)
            #Yprofs = uniform_filter1d(Yprofs, 5, axis=0)
        tstcrit = ( (np.abs((old_Yprofs-Yprofs)/Yprofs)<concrit)|(Yprofs==0.0)).astype(np.int).sum(axis=0)
        if np.array_equal(tstcrit,allionpnt):
            # Once we are close to convergence, use a more reliable cooling function
            print "Getting close!! Try a more accurate cooling function"
            close = True
            break
            miniter += iteration

        print "STATISTICS --"
        print "ION  INDEX   OLD VALUE    NEW VALUE   |OLD-NEW|"
        w_maxoff   = np.argmax(np.abs((old_Yprofs-Yprofs)/Yprofs),axis=0)
        for j in range(nions):
            print ions[j], w_maxoff[j], old_Yprofs[w_maxoff[j],j], Yprofs[w_maxoff[j],j], np.abs((old_Yprofs[w_maxoff[j],j] - Yprofs[w_maxoff[j],j])/Yprofs[w_maxoff[j],j])

        # Check if the stopping criteria were met
        if iteration > maxiter:
            print "Break outer loop at maxiter={0:d} iterations, STATUS:".format(maxiter)
            break

    # Calculate the density profiles
    print "Calculating volume density profiles"
    for j in range(nions):
        # density of this specie = unionized fraction * H volume density * number abundance relative to H
        prof_density[:,j] = Yprofs[:,j] * densitynH * elID[ions[j]].abund
        print ions[j], np.max(Yprofs[:,j]), np.max(prof_density[:,j])

    print "Calculating column density profiles"
    prof_coldens = np.zeros_like(prof_density)
    for j in range(nions):
        if geom == "NFW":
            coldens = cython_fns.coldensprofile(prof_density[:,j], radius)
            prof_coldens[:,j] = coldens.copy()
        elif geom == "PP":
            coldens = cython_fns.calc_coldensPP(prof_density[:,j], radius)
            prof_coldens[:,j] = coldens.copy()
            print ions[j], np.log10(prof_coldens[0,j]), np.max(np.log10(prof_coldens[:,j]))

    print "Calculating Ha surface brightness profile"
    Harecomb = recomb.Ha_recomb(prof_temperature)
    HIIdensity = densitynH * (1.0-Yprofs[:,elID["H I"].id])
    elecprot = Harecomb*electrondensity*HIIdensity
    HaSB = (1.0/(4.0*np.pi)) * cython_fns.coldensprofile(elecprot, radius)  # photons /cm^2 / s / SR
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
    if geom == "NFW":
        mstring = mangle_string("{0:3.2f}".format(np.log10(hmodel.mvir / somtog)))
        rstring = mangle_string("{0:3.2f}".format(redshift))
        bstring = mangle_string("{0:+3.2f}".format(np.log10(hmodel.baryfrac)))
        if options["radfield"]=="HM12":
            hstring = mangle_string("HMscale{0:+3.2f}".format(np.log10(options["HMscale"])))
        elif options["radfield"][0:2]=="PL":
            hstring = options["radfield"]
        outfname = out_dir + ("/{0:s}_mass{1:s}_redshift{2:s}_baryscl{3:s}_{4:s}_{5:d}-{6:d}"
                              .format(geom,mstring,rstring,bstring,hstring,npts,nummu))
        print "Saving file {0:s}.npy".format(outfname)
        tmpout = np.concatenate((radius.reshape((npts,1))*cmtopc,prof_temperature.reshape((npts,1)),densitynH.reshape((npts,1)),HaSB.reshape((npts,1)),prof_density,prof_coldens),axis=1)
    elif geom == "PP":
        # needs fixing
        assert False
        dstring = mangle_string("{0:+3.2f}".format(options["geometry"][geom][0]))
        rstring = mangle_string("{0:4.2f}".format(options["geometry"][geom][1])).replace(".","d")
        if options["radfield"]=="HM12":
            hstring = mangle_string("HMscale{0:+3.2f}".format(np.log10(options["HMscale"])))
        elif options["radfield"][0:2]=="PL":
            hstring = options["radfield"]
        outfname = ("output/{0:s}_density{1:s}_radius{2:s}_{3:s}_{4:d}"
                    .format(hmodel.name,dstring,rstring,hstring,npts))
        print "Saving file {0:s}.npy".format(outfname)
        tmpout = np.concatenate((radius.reshape((npts,1))*cmtopc,prof_temperature.reshape((npts,1)),densitynH.reshape((npts,1)),prof_density,prof_coldens),axis=1)

    np.save(outfname, tmpout)

    # Stop the program if a large H I column density has already been reached
    if np.max(np.log10(prof_coldens[:,elID["H I"].id])) > 22.0:
        print "Terminating after maximum N(H I) has been reached"
        sys.exit()
    
    # dispose of process pool
    if ncpus > 1:
        pool.close()
        pool.join()

    # Return the output filename to be used as the input to the next iteration    
    return outfname + '.npy'

