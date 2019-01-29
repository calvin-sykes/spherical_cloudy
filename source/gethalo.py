from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt

from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import trapz as spTrapz
from scipy.interpolate  import InterpolatedUnivariateSpline as spIntUniSpl
from scipy.optimize import curve_fit as spCurveFit

from multiprocessing import cpu_count as mpCPUCount, Pool as mpPool
from multiprocessing.pool import ApplyResult
import signal
import sys
import time

import charge_transfer as chrgtran
import colioniz
import constants
import cosmo
import cython_fns
import elemids
import emis
import logger
import misc
import options
import phionxsec
import photoion
import radfields
import recomb
import table_coolfunc

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

def blendfxgx(f, g, xover, smooth):
    assert len(f) == len(g), (len(f), len(g))
    x = np.arange(len(f))
    if xover < smooth:
        xover += smooth
    elif xover + smooth > len(x):
        xover -= smooth
    s = lambda x, a, b: 0.5 + 0.5 * np.tanh((x - a) / b)
    smooth = s(x, xover, smooth)
    return smooth * f + (1 - smooth) * g


def mpcoldens(j, prof, radius, nummu, geom):
    if geom in {"NFW", "Burkert", "Cored"}:
        coldens, muarr = cython_fns.calc_coldens(prof, radius, nummu)
        return [j,coldens,muarr]
    elif geom == "PP":
        coldens = cython_fns.calc_coldensPP(prof, radius)
        return [j,coldens]
    else:
        logger.log("critical", "Geometry {0:s} is not allowed".format(geom))
        sys.exit(1)
    return


def mpphion(j, jnurarr, phelxs, nuzero, planck):
    phionxsec = 4.0*np.pi * cython_fns.phionrate(jnurarr, phelxs, nuzero, planck)
    return [j,phionxsec]


def get_radius(virialr, scale, npts, method=0, **kwargs):
    if method == 0:
        # Linear scaling
        radius = np.linspace(0.0, virialr*scale, npts)
    elif method == 1:
        # Scaling with finer interpolation for H II --> H I transition region
        old_radius = np.geomspace(virialr * 1.0E-4, virialr * scale, npts)
        try:
            hi_yprof = kwargs['yprof']
        except KeyError:
            logger.log("critical", "Need H I Y profile for method=1")
            sys.exit(1)
        loc_neutral = np.argmax(np.abs(np.diff(hi_yprof)))
        rad_neutral = old_radius[loc_neutral]
        num_hires = 200
        width_hires = 0.3
        num_other = npts - num_hires
        radius = np.append(0, np.geomspace(virialr * 1.0E-4, (1 - width_hires) * rad_neutral, int((float(loc_neutral) / npts) * num_other) - 1))
        len1 = len(radius)
        radius = np.append(radius, np.geomspace((1 - width_hires) * rad_neutral, (1 + width_hires) * rad_neutral, num_hires))
        len2 = len(radius)
        radius = np.append(radius, np.geomspace((1 + width_hires) * rad_neutral, virialr * scale, npts - len(radius)))
        interp_yprof = np.interp(radius, np.append(0.0, np.geomspace(virialr * 1.0E-4, virialr * scale, npts-1)),hi_yprof)
    elif method == 2:
        # Log scaling
        radius = np.append(0, np.geomspace(virialr * 1.0E-4, virialr * scale, npts - 1))
    else:
        logger.log("critical", "radius method is not yet defined")
        sys.exit()
    if virialr not in radius:
        radius = np.append(virialr, radius[:-1])
        radius.sort()
    return radius


def mangle_string(str):
    return str.replace(".","d").replace("+","p").replace("-","m")


class LivePlot:
    
    def __init__(self):
        plt.ion()
        self.figures = dict({})
    
    def draw(self, name, cmds):
        try:
            fig = self.figures[name]
        except KeyError:
            fig = plt.figure()
            self.figures[name] = fig
        fig.clear()
        cmds(fig.gca())

    def show(self):
        if len(self.figures):
            for fig in self.figures.itervalues():
                fig.canvas.draw_idle()
                fig.show()
            plt.pause(0.01)
        else:
            logger.log('warning', "No live figures to show")

    def close(self):
        for name in list(self.figures.iterkeys()):
            plt.close(self.figures[name])
            del self.figures[name]
        plt.ioff()


def get_halo(hmodel, redshift, cosmopar=cosmo.get_cosmo(),
             ions=["H I", "He I", "He II"], prevfile=None, options=None):
    """
    hmodel    : The halo model which defines the geometry
    redshift  : The redshift to evaluate the UVB and background density at
    cosmopar  : Set the cosmology of the simulation (hubble constant/100 km/s/Mpc, Omega_B, Omega_L, Omega_M)
    ions      : A list of ions to consider
    prevfile  : A filename for a previous run's output to load as an initial solution
    options   : The options dictionary which contains all the parameters/settings for the model
    """

    # disable logging if running models in parallel
    if options["run"]["pp_para"]:
        logger.disable()

    bturb   = options['phys']['bturb'  ]
    gastemp = options['phys']['gastemp']
    metals  = options['phys']['metals' ]
    Hescale = options['phys']['hescale']
    
    # Begin the timer
    timeA = time.time()

    if options is None:
        options = getoptions.default()
        logger.log('warning', "No inputs provided to get_halo, using defaults (You probably don't want this!)")

    # Set some numerical aspects of the simulation
    miniter = options["run"]["miniter"]
    maxiter = options["run"]["maxiter"]
    npts    = options["run"]["nsample"]
    nummu   = options["run"]["nummu"  ]
    concrit = options["run"]["concrit"]
    ncpus   = options["run"]["ncpus"  ]
    lv_plot = options["run"]["lv_plot"]
    out_dir = options["run"]["outdir" ]

    # What quantities should be output
    svrates = options["save"]["rates"    ] # ionisation rates
    svrcmb  = options["save"]["recomb"   ] # recombination rates
    svhtcl  = options["save"]["heat_cool"] # heating + cooling rates
    svjnu   = options["save"]["intensity"] # mean intensity

    radmethod = 2

    const = constants.get()    
    kB      = const["kB"]
    cmtopc  = const["cmtopc"]
    somtog  = const["somtog"]
    Gcons   = const["Gcons"]
    planck  = const["planck"]
    elvolt  = const["elvolt"]
    protmss = const["protmss"]
    hztos   = const["hztos"]

    # type of DM profile and outer radius in units of Rvir
    geom = options["geometry"]["profile"]
    geomscale = options["geometry"]["scale"]

    # min and max column density, and constant H volume density for plane parallel models
    if geom == 'PP':
        PP_cden  = hmodel.cden
        PP_dens  = hmodel.density

    # boundary condition to use
    use_pcon = options["phys"]["ext_press"]

    # method to derive temperature profile
    temp_method = options["phys"]["temp_method"]
    
    # Set up the cosmology
    cosmoVal = FlatLambdaCDM(H0=100.0*cosmopar[0], Om0=cosmopar[3])
    hubb_time = cosmoVal.age(redshift).value * (365.25 * 86400 * 1.0E9)
    # Get the element ID numbers
    elID = elemids.getids(ions, metals, Hescale)

    # How many ions do we need to calculate
    nions = len(ions)
    if ncpus > nions: ncpus = nions
    if ncpus > mpCPUCount(): ncpus = mpCPUCount()
    if ncpus <= 0: ncpus += mpCPUCount()
    if ncpus <= 0: ncpus = 1
    logger.log("info", "Using {0:d} CPUs".format(int(ncpus)))

    if lv_plot:
        live_plot = LivePlot()

    # make multiprocessing pool if using >1 CPUs
    # the reassignment of SIGINT is needed to make Ctrl-C work while the process pool is active
    if ncpus > 1:
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        pool = mpPool(processes=ncpus)
        signal.signal(signal.SIGINT, sigint_handler)

    # Get the primordial number abundance of He relative to H
    # (Scaling has been moved to elemids.py)
    if "He I" in ions:
        prim_He = elID["He I"].abund
    elif "He II" in ions:
        prim_He = elID["He II"].abund
    else:
        logger.log("CRITICAL", "You must include ""He I"" and ""He II"" in your model")
        sys.exit(1)
    if "H I" not in ions:
        logger.log("CRITICAL","You must include ""H I"" in your model")
        sys.exit(1)

    hubb_par = cosmo.hubblepar(redshift, cosmopar)
    rhocrit = 3.0*(hubb_par * hztos)**2 / (8.0 * np.pi * Gcons)
    bkdens = cosmopar[1] * rhocrit / (protmss * (1 + 4 * prim_He))

    logger.log("debug", "Loading radiation fields")
    if options["UVB"]["spectrum"][0:2] in {"HM", "MH"}:
        # Get the Haardt & Madau background at the appropriate redshift
        version = options["UVB"]["spectrum"][2:4] # HM12 or HM05
        slope   = options["UVB"]["slope"]         # shape parameter alpha_UV (Crighton et al 2015, https://arxiv.org/pdf/1406.4239.pdf)
        jzero, nuzero = radfields.HMbackground(elID,redshift=redshift, version=version, alpha_UV=slope)
        jzero *= options["UVB"]["scale"]
    elif options["UVB"]["spectrum"][0:2] == "PL":
        jzero, nuzero = radfields.powerlaw(elID)
    elif options["UVB"]["spectrum"] == "AGN":
        jzero, nuzero = radfields.AGN(elID)
    else:
        logger.log("CRITICAL", "The radiation field '{0:s}' is not implemented yet".format(options["UVB"]["spectrum"]))
        sys.exit()

    # convert to an energy
    engy = planck * nuzero / elvolt

    logger.log("debug", "Loading photoionization cross-sections")
    # Calculate the photoionization cross sections
    phelxsdata = phionxsec.load_data(elID)
    phelxs = np.zeros((nions,engy.size))
    for j in range(nions):
        xsecv = phionxsec.rate_function_arr(engy,phelxsdata[ions[j]])
        phelxs[j] = xsecv.copy()

    # If the slope of the UVB has been changed, renormalise it so that the hydrogen photoionisation rate
    # stays the same as for the fiducial/flat H&M UVB
    if options["UVB"]["spectrum"][0:2] in {"HM", "MH"} and slope != 0:
        flat_jzero = radfields.HM_fiducial(elID, redshift, version)
        flat_jzero *= options["UVB"]["scale"]
        flat_gamma = 4 * np.pi * spTrapz(flat_jzero / (planck * nuzero) * phelxs[elID["H I"].id], nuzero)
        temp_gamma = 4 * np.pi * spTrapz(jzero / (planck * nuzero) * phelxs[elID["H I"].id], nuzero)
        rescale_factor = temp_gamma / flat_gamma
        jzero /= rescale_factor
        #new_gamma = 4 * np.pi * spTrapz(jzero / (planck * nuzero) * phelxs[elID["H I"].id], nuzero)
        #assert np.allclose(flat_gamma, new_gamma)

    logger.log("debug", "Loading radiative recombination coefficients")
    rrecombrate = recomb.load_data_radi(elID)

    logger.log("debug", "Loading dielectronic recombination coefficients")
    drecombrate = recomb.load_data_diel(elID)
    drecombelems = drecombrate.keys()

    logger.log("debug",  "Loading collisional ionization rate coefficients")
    usecolion = "Chianti"
    if usecolion == "Dere2007":
        colioncoeff = colioniz.load_data(elID, rates="Dere2007")
    elif usecolion == "Voronov1997":
        colioncoeff = colioniz.load_data(elID, rates="Voronov1997")
    elif usecolion == "Chianti":
        colioncoeff, coliontemp = colioniz.load_data(elID, rates="Chianti")
    else:
        logger.log("error", "Cannot load collisional ionization rates")

    logger.log("debug", "Loading Charge Transfer rate coefficients")
    chrgtranrate = chrgtran.load_data(elID)
    chrgtran_HItargs = chrgtranrate["H I"].keys()
    chrgtran_HIItargs = chrgtranrate["H II"].keys()
    chrgtran_HeItargs = chrgtranrate["He I"].keys()
    chrgtran_HeIItargs = chrgtranrate["He II"].keys()

    if temp_method in {'eagle', 'blend'}:
        logger.log("info", "Using tabulated cooling function")
        eagle_cf = table_coolfunc.load_eagle_nHT(prim_He, redshift)
        # Above call returns the 2D cooling function
        # Now, construct 1D n_H-T curves for adiabatic and equilibrium cases
        
        # use finely interpolated bins to get better accuracy
        nbins = np.logspace(-7, 1, 1000)
        Tbins = np.logspace(3, 5, 1000)
        rates = np.abs(eagle_cf(Tbins, nbins))

        # equilibrium temperature is that minimising net heating/cooling
        temp_eq = Tbins[np.argmin(rates, axis=0)]

        # adiabatic temperature is that for which 1/rate is the Hubble time
        # calculated as follows
        mu = 0.6 # assume fully ionised - ok for low density where this interpolation should be used
        grddens, grdtemp = np.meshgrid(nbins, Tbins)
        grdmdens = grddens * protmss * (1 + 4 * prim_He)
        rates_ad = ((1.5 * kB * grdtemp * protmss) / (mu * grdmdens * (1 - prim_He)**2 * hubb_time))

        # The temperature solution is found by minimising | rates - rates_adiabatic |
        # This can fail, because the function has two branches for log n_H < -4
        # We always want the lower branch, so the solution is found by locating the 1st change in sign
        # of d(| rates - rates_adiabatic |)/dT
        loc_ad = np.argmax(np.gradient(np.abs(rates - rates_ad), axis=0) >= 0, axis=0)
        temp_ad = Tbins[loc_ad]

        # this is the adiabatic interpolant to use
        adiab_interp = spIntUniSpl(nbins, temp_ad)

        # this is the value for n_H for which we should switch from the interpolated to computed equilibrium temperature
        # found as the density for which the relative difference in T_eq and T_ad is less than 10%
        density_thresh_ad = nbins[np.argmax((np.abs(temp_eq - temp_ad) / temp_eq) < 0.1)]

    elif temp_method == 'relhic':
        logger.log("info", "Using RELHIC nH-T relation")
        adiab_interp = table_coolfunc.load_relhic_nHT()

    He_emissivities = emis.HeEmissivities()
    if options['save']['he_emis'] == 'all':
        He_want_wls = He_emissivities.get_wavelengths()
    elif options['save']['he_emis'] != 'none':
        He_all_wls = He_emissivities.get_wavelengths()
        try:
            He_want_wls = eval(options['save']['he_emis'])
        except:
            logger.log('critical', "Failed to eval He lines save specification {}".format(options['save']['he_emis']))
            sys.exit(1)
        if not isinstance(He_want_wls, list): # promote scalar to length-1 list
            He_want_wls = [He_want_wls]
        He_want_wls = map(float, He_want_wls)
        if not set(He_want_wls).issubset(He_all_wls):
            logger.log('warning', "One or more requested He lines are unknown")
    else:
        He_want_wls = None

    # if string is passed, interpret as filename for the previous run
    # this is used as an initial solution to speed up convergence
    if prevfile is not None:
        if geom in {"NFW", "Burkert", "Cored"}:
            logger.log("info", "Loading file {0:s}".format(prevfile))
            tdata = np.load(prevfile)
            # if the data is stored as a 'structured' array (one with fieldnames)
            # then it needs to be viewed as a plain array to select individual columns
            if tdata.dtype.names is not None:
                tdata = tdata.view(tdata.dtype[0])
            strt = 6
            numarr = tdata.shape[1]
            arridx = dict({})
            arridx["voldens"] = dict({})
            arridx["coldens"] = dict({})
            for i in range((numarr-strt)/3):
                arridx["voldens"][ions[i]] = strt + i
                arridx["coldens"][ions[i]] = strt + i + (numarr-strt)/2
            prof_coldens = np.zeros((nions,npts,nummu))
            prof_density = np.zeros((nions,npts))
            Yprofs = np.zeros((nions,npts))

            old_radius = tdata[:,0] / cmtopc
            # redefine radius to have fine interpolation across ionised -> neutral transition region
            # and resample quantities onto this new set of radii
            radius = get_radius(hmodel.rvir, geomscale, npts, method=radmethod)

            # resample quantities from old radial coordinates to new ones
            prof_temperature = np.interp(radius, old_radius, tdata[:,1])
            temp_densitynH = np.interp(radius, old_radius, tdata[:,2])
            for j in range(nions):
                prof_density[j] = np.interp(radius, old_radius, tdata[:,arridx["voldens"][ions[j]]])
                if elID[ions[j]].abund > 0.0:
                    Yprofs[j] = prof_density[j] / (temp_densitynH * elID[ions[j]].abund)
                else:
                    Yprofs[j] = prof_density[j] * 0.0

            prof_phionrate = np.zeros((nions,npts))
            densitym  = temp_densitynH * protmss * (1.0 + 4.0*prim_He)

        elif geom == "PP":
            logger.log("critical", "Previous file loading for plane parallel geometries is not implemented")
            sys.exit(1)
        else:
            logger.log("critical", "Unknown geometry")
            sys.exit(1)
    else: # prevfile is None
        # Set the gas conditions
        if geom in {"NFW", "Burkert", "Cored"}:
            radius = get_radius(hmodel.rvir, geomscale, npts, method=radmethod)
            if radius.size != npts:
                logger.log("critical", "Error defining radius")
                sys.exit(1)
            temp_densitynH = np.ones(npts)
            prof_coldens = np.zeros((nions, npts, nummu))
            prof_density = 1.0E-1*np.ones((nions,npts))
        elif geom == "PP":
            depth = PP_cden / PP_dens
            radius = np.linspace(0, depth, npts)
            densitynH = np.ones(npts) * PP_dens
            prof_coldens = np.zeros((nions,npts))
            prof_density = 1.0E-1 * np.ones((nions,npts))
        else:
            logger.log("critical", "Unknown geometry")
            sys.exit(1)
        
        prof_temperature = gastemp * np.ones(npts)
        prof_phionrate = np.zeros((nions,npts))
        Yprofs = 1.0E-2 * np.ones((nions,npts))
        densitym = protmss * (1.0 + 4.0 * prim_He) * np.ones(npts) # Default to be used for PP

    # An array used to check if convergence has been reached for each ion and in each cell.
    allionpnt = npts * np.ones(nions,dtype=np.int)
    
    # Calculate the mass of baryons involved:
    if geom in {"NFW", "Burkert", "Cored"}:
        # Set the mass density profile
        barymass = hmodel.mvir * hmodel.baryfrac
        densitym = np.ones(npts) * barymass / (4.0 * np.pi * (hmodel.rvir**3) / 3.0)
    elif geom == "PP":
        pass # The mass density is set earlier for PP

    # Store old Yprofs
    nstore = 10
    istore = 0
    store_Yprofs = np.zeros(Yprofs.shape + (nstore,))

    # Set the stopping criteria flags
    tstcrit  = np.zeros(nions, dtype=np.int)
    iteration = 0
    old_Yprofs = None

    # BEGIN MAIN LOOP
    while (not np.array_equal(tstcrit, allionpnt)) or (iteration <= miniter):
        iteration += 1
        logger.log("info", "Iteration number: {0:d}".format(iteration))
        for j in range(nions):
            logger.log("info", "<--[  {0:d}/{1:d}  ]-->  {2:s}".format(int(tstcrit[j]), npts, ions[j]))

        # Store the old Yprofs
        store_Yprofs[:,:,istore] = Yprofs.copy()

        istore += 1
        if istore >= nstore:
            istore = 0

        # Calculate the pressure profile
        dof = (2.0 - Yprofs[elID["H I"].id]) + prim_He * (3.0 - 2.0 * Yprofs[elID["He I"].id] - 1.0 * Yprofs[elID["He II"].id])
        masspp = (1.0 + 4.0 * prim_He) / dof
        if geom in {"NFW", "Burkert", "Cored"}:
            logger.log("debug", "Calculating total gas mass")
            fgas = cython_fns.fgasx(densitym,radius,hmodel.rscale)
            logger.log("debug", "Calculating pressure profile")
            prof_pressure = cython_fns.pressure(prof_temperature, radius, masspp, hmodel, bturb, Gcons, kB, protmss)
            turb_pressure = 0.75*densitym*(bturb**2.0)
             # Calculate the thermal pressure, and ensure positivity
            ther_pressure = prof_pressure-turb_pressure
            wther = np.where(ther_pressure<0.0)
            ther_pressure[wther] = 0.0
            # Calculate gas density profile
            temp_densitynH = ther_pressure / (1.5 * kB * prof_temperature * dof)
            if (temp_densitynH[0]==0.0):
                logger.log("WARNING", "Central density is zero, assuming no turbulent pressure for this iteration")
                temp_densitynH = prof_pressure / (1.5 * kB * prof_temperature * dof)
            temp_densitynH /= temp_densitynH[0]
            if use_pcon:
                # assume outermost density value is equal to mean background density
                # and use as scaling factor to get density in physical units
                dens_scale = bkdens / temp_densitynH[-1]
            else:
                # constrain central density by requiring total mass
                # to match that obtained from the baryon fraction data
                rintegral = cython_fns.mass_integral(temp_densitynH, radius, hmodel.rvir)
                dens_scale = barymass / (4.0 * np.pi * protmss * (1.0 + 4.0 * prim_He) * rintegral)
            densitynH = dens_scale * temp_densitynH
            densitym  = densitynH * protmss * (1.0 + 4.0 * prim_He)

        # Update the volume density of the unionized species
        for j in range(nions):
            # density of this specie = unionized fraction * H volume density * number abundance relative to H
            prof_density[j] = Yprofs[j] * densitynH * elID[ions[j]].abund

        # Compute the electron density
        electrondensity = densitynH * ((1.0 - Yprofs[elID["H I"].id]) +
                                       prim_He * Yprofs[elID["He II"].id] +
                                       2.0 * prim_He * (1.0 - Yprofs[elID["He I"].id] - Yprofs[elID["He II"].id]))
        
        # Calculate the column density arrays,
        if ncpus == 1:
            for j in range(nions):
                if geom in {"NFW", "Burkert", "Cored"}:
                    coldens, muarr = cython_fns.calc_coldens(prof_density[j], radius, nummu)
                    prof_coldens[j,:,:] = coldens.copy()
                elif geom == "PP":
                    coldens = cython_fns.calc_coldensPP(prof_density[j], radius)
                    prof_coldens[j,:] = coldens.copy()
        else:
            async_results = []
            for j in range(nions):
                async_results.append(pool.apply_async(mpcoldens, (j, prof_density[j], radius, nummu, geom)))
            map(ApplyResult.wait, async_results)
            for j in range(nions):
                getVal = async_results[j].get()
                if geom in {"NFW", "Burkert", "Cored"}:
                    prof_coldens[getVal[0],:,:] = getVal[1].copy()
                    muarr = getVal[2]
                elif geom == "PP":
                    prof_coldens[getVal[0],:] = getVal[1].copy()

        # integrate over all angles,
        if geom in {"NFW", "Burkert", "Cored"}:
            logger.log("debug", "Integrate over all angles")
            jnurarr = cython_fns.nint_costheta(prof_coldens, phelxs, muarr, jzero)
        elif geom == "PP":
            jnurarr = cython_fns.nint_pp(prof_coldens, phelxs, jzero)

        # and calculate the photoionization rates
        logger.log("debug", "Calculating phionization rates")
        if ncpus == 1:
            for j in range(nions):
                phionr = 4.0*np.pi * cython_fns.phionrate(jnurarr, phelxs[j], nuzero, planck)
                prof_phionrate[j] = phionr.copy()
        else:
            async_results = []
            for j in range(nions):
                async_results.append(pool.apply_async(mpphion, (j, jnurarr, phelxs[j], nuzero, planck)))
            map(ApplyResult.wait, async_results)
            for j in range(nions):
                getVal = async_results[j].get()
                prof_phionrate[getVal[0]] = getVal[1].copy()

        # Calculate the collisional ionization rate coefficients
        prof_colion = np.zeros((nions,npts))
        for j in range(nions):
            if usecolion == "Dere2007":
                ratev = colioniz.rate_function_Dere2007(prof_temperature * kB / elvolt, colioncoeff[ions[j]])
            elif usecolion == "Voronov1997":
                ratev = colioniz.rate_function_arr(prof_temperature * kB / elvolt, colioncoeff[ions[j]])
            elif usecolion == "Chianti":
                ratev = colioniz.rate_function_Chianti(prof_temperature * kB / elvolt, colioncoeff[ions[j]], coliontemp)
            prof_colion[j] = ratev.copy()

        # Calculate collisional ionisation rates
        prof_colionrate = np.zeros((nions,npts))
        for j in range(nions):
            prof_colionrate[j] = prof_colion[j] * electrondensity

        # the secondary photoelectron collisional ionization rates (probably not important for metals -- Section II, Shull & van Steenberg (1985))
        logger.log("debug", "Integrate over frequency to get secondary photoelectron ionization")
        # Make sure there are no zero H I density
        tmpcloneHI = prof_density[elID["H I"].id].copy()
        w = np.where(prof_density[elID["H I"].id] == 0.0)
        if np.size(w[0]) != 0:
            logger.log("WARNING", "(H I) = exactly 0.0 in some zones, setting to smallest value")
            wb = np.where(tmpcloneHI!=0.0)
            tmpcloneHI[w] = np.min(tmpcloneHI[wb])
            prof_density[elID["H I"].id] = tmpcloneHI.copy()
            prof_density[elID["D I"].id] = tmpcloneHI.copy() * elID["D I"].abund
        prof_scdryrate = np.zeros((nions,npts))

        scdry_args = (jnurarr, nuzero,
                      phelxs[elID["H I"].id], phelxs[elID["D I"].id], phelxs[elID["He I"].id], phelxs[elID["He II"].id], # x-sections
                      prof_density[elID["H I"].id], prof_density[elID["D I"].id], prof_density[elID["He I"].id], prof_density[elID["He II"].id], electrondensity / (densitynH*(1.0 + 2.0 * prim_He)), # densities
                      elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, elID["He II"].ip, # ionisation potentials
                      planck, elvolt) # constants
        
        if ncpus > 1:
            async_results = []
            # H I
            async_results.append(pool.apply_async(cython_fns.scdryrate, scdry_args + (0,)))
            # He I
            async_results.append(pool.apply_async(cython_fns.scdryrate, scdry_args + (2,)))
            map(ApplyResult.wait, async_results)

            for j in range(nions):
                if ions[j] == "H I":
                    ratev = 4.0 * np.pi * async_results[0].get()
                    prof_scdryrate[j] = ratev.copy()
                elif ions[j] == "He I":
                    ratev = 4.0 * np.pi * 10.0 * async_results[1].get()
                    prof_scdryrate[j] = ratev.copy()
        else:
            for j in range(nions):
                if ions[j] == "H I":
                    ratev = 4.0*np.pi * cython_fns.scdryrate(*scdry_args, flip=0)
                    prof_scdryrate[j] = ratev.copy()
                elif ions[j] == "He I":
                    ratev = 4.0*np.pi * 10.0 * cython_fns.scdryrate(*scdry_args, flip=2)
                    prof_scdryrate[j] = ratev.copy()

        # Calculate other forms of ionization (e.g. photons from H and He recombinations)
        logger.log("debug", "Calculate ionization rate from recombinations of H+, He+, He++")
        
        prof_other = np.zeros((nions,npts))
        for j in range(nions):
            ratev = photoion.other(ions[j], engy, prof_density, densitynH, Yprofs, electrondensity, phelxs, prof_temperature, elID, kB, elvolt)
            prof_other[j] = ratev.copy()

        # Calculate the charge transfer ionization rates
        logger.log("debug", "Calculating charge transfer rates")
        HIIdensity  = densitynH * (1.0 - Yprofs[elID["H I"].id])
        HeIIdensity = densitynH * prim_He * Yprofs[elID["He II"].id]
        HIIdensity = HIIdensity.reshape((1,npts)).repeat(nions, axis=0)
        HeIIdensity = HeIIdensity.reshape((1,npts)).repeat(nions, axis=0)
        prof_chrgtraniHII = np.zeros((nions,npts))
        prof_chrgtraniHeII = np.zeros((nions,npts))
        for j in range(nions):
            #ratev = chrgtran.HII_target(ions[j],prof_temperature)
            if ions[j] in chrgtran_HIItargs:
                ratev = chrgtran.rate_function_form(chrgtranrate["H II"][ions[j]], prof_temperature)
                prof_chrgtraniHII[j] = ratev.copy()
            #ratev = chrgtran.HeII_target(ions[j],prof_temperature)
            if ions[j] in chrgtran_HeIItargs:
                ratev = chrgtran.rate_function_form(chrgtranrate["He II"][ions[j]], prof_temperature)
                prof_chrgtraniHeII[j] = ratev.copy()

        # Total all of the ionization rates
        prof_gamma = prof_phionrate + prof_scdryrate + HIIdensity * prof_chrgtraniHII + HeIIdensity * prof_chrgtraniHeII + prof_other + prof_colionrate

        logger.log("debug", "Calculating recombination rates")
        prof_recomb = np.zeros((nions,npts))
        prof_recombCTHI  = np.zeros((nions,npts))
        prof_recombCTHeI = np.zeros((nions,npts))
        for j in range(nions):
            ratev = recomb.rate_function_radi_arr(prof_temperature, rrecombrate[ions[j]])
            if ions[j] in drecombelems:
                ratev += recomb.rate_function_diel_arr(prof_temperature, drecombrate[ions[j]])
            prof_recomb[j] = ratev.copy()
            if ions[j] in chrgtran_HItargs:
                ratev = chrgtran.rate_function_form(chrgtranrate["H I"][ions[j]], prof_temperature)
                prof_recombCTHI[j] = ratev.copy()
            if ions[j] in chrgtran_HeItargs:
                ratev = chrgtran.rate_function_form(chrgtranrate["He I"][ions[j]], prof_temperature)
                prof_recombCTHeI[j] = ratev.copy()

        edens_allions = electrondensity.reshape((1,npts)).repeat(nions,axis=0)
        HIdensity = prof_density[elID["H I"].id]
        HeIdensity = prof_density[elID["He I"].id]
        HIdensity = HIdensity.reshape((1,npts)).repeat(nions,axis=0)
        HeIdensity = HeIdensity.reshape((1,npts)).repeat(nions,axis=0)
        prof_alpha = edens_allions * prof_recomb + HIdensity * prof_recombCTHI  + HeIdensity * prof_recombCTHeI

        prof_rates = prof_gamma / prof_alpha

        # Store old and obtain new values for YHI, YHeI, YHeII
        old_Yprofs = Yprofs.copy()
        tmp_Yprofs = Yprofs.copy()
        inneriter = 0

        ## BEGIN INNER ITERATION
        while True:
            # Calculate the Yprofs
            Yprofs = misc.calc_yprofs(ions, prof_rates, elID)

            # Test if the profiles have converged
            tstconv_x = (np.abs((tmp_Yprofs - Yprofs) / Yprofs) < concrit**2) | (Yprofs == 0.0)
            tstconv_1mx = (np.abs(((1 - tmp_Yprofs) - (1 - Yprofs)) / (1 - Yprofs)) < concrit**2)|((1 - Yprofs) == 0.0)
            tstconv = np.logical_and(tstconv_x, tstconv_1mx).astype(np.int).sum(axis=1)
            
            # Reset ne and the rates
            electrondensity = densitynH * ((1.0 - Yprofs[elID["H I"].id]) +
                                           prim_He * Yprofs[elID["He II"].id] +
                                           2.0 * prim_He * (1.0 - Yprofs[elID["He I"].id] - Yprofs[elID["He II"].id]) )
            edens_allions = electrondensity.reshape((1,npts)).repeat(nions,axis=0)
            # Recalculate the recombination rate profile with the new Yprofs and electrondensity
            HIIdensity  = densitynH * (1.0-Yprofs[elID["H I"].id])
            HeIIdensity = densitynH * prim_He*Yprofs[elID["He II"].id]
            HIIdensity  = HIIdensity.reshape((1,npts)).repeat(nions,axis=0)
            HeIIdensity = HeIIdensity.reshape((1,npts)).repeat(nions,axis=0)
            # Recalculate all of the ionization effects that depend on density
            # scdryrate
            tmpcloneHI = prof_density[elID["H I"].id].copy()
            w = np.where(prof_density[elID["H I"].id] == 0.0)
            if np.size(w[0]) != 0:
                logger.log("WARNING", "n(H I) = exactly 0.0 in some zones, setting to smallest value")
                wb = np.where(tmpcloneHI != 0.0)
                tmpcloneHI[w] = np.min(tmpcloneHI[wb])
                prof_density[elID["H I"].id] = tmpcloneHI.copy()
                prof_density[elID["D I"].id] = tmpcloneHI.copy() * elID["D I"].abund
            prof_scdryrate = np.zeros((nions,npts))
            if ncpus > 1:
                async_results = []
                # H I
                async_results.append(pool.apply_async(cython_fns.scdryrate, scdry_args + (0,)))
                # He I
                async_results.append(pool.apply_async(cython_fns.scdryrate, scdry_args + (2,)))
                map(ApplyResult.wait, async_results)

                for j in range(nions):
                    if ions[j] == "H I":
                        ratev = 4.0*np.pi * async_results[0].get()
                        prof_scdryrate[j] = ratev.copy()
                    elif ions[j] == "He I":
                        ratev = 4.0*np.pi * 10.0 * async_results[1].get()
                        prof_scdryrate[j] = ratev.copy()
            else:
                for j in range(nions):
                    if ions[j] == "H I":
                        ratev = 4.0*np.pi * cython_fns.scdryrate(*scdry_args, flip=0)
                        prof_scdryrate[j] = ratev.copy()
                    elif ions[j] == "He I":
                        ratev = 4.0*np.pi * 10.0 * cython_fns.scdryrate(*scdry_args, flip=2)
                        prof_scdryrate[j] = ratev.copy()

            # Colion
            prof_colion = np.zeros((nions,npts))
            for j in range(nions):
                if usecolion == "Dere2007":
                    ratev = colioniz.rate_function_Dere2007(prof_temperature * kB / elvolt, colioncoeff[ions[j]])
                elif usecolion == "Voronov1997":
                    ratev = colioniz.rate_function_arr(prof_temperature * kB / elvolt, colioncoeff[ions[j]])
                elif usecolion == "Chianti":
                    ratev = colioniz.rate_function_Chianti(prof_temperature * kB / elvolt, colioncoeff[ions[j]], coliontemp)
                prof_colion[j] = ratev.copy()

            # Calculate collisional ionisation rates
            prof_colionrate = np.zeros((nions,npts))
            for j in range(nions):
                prof_colionrate[j] = prof_colion[j] * electrondensity

            # Other
            for j in range(nions):
                ratev = photoion.other(ions[j], engy, prof_density, densitynH, Yprofs, electrondensity, phelxs, prof_temperature, elID, kB, elvolt)
                prof_other[j] = ratev.copy()

            prof_gamma = prof_phionrate + prof_scdryrate + HIIdensity*prof_chrgtraniHII + HeIIdensity*prof_chrgtraniHeII + prof_other + prof_colionrate

            # density of this specie = unionized fraction * H volume density * number abundance relative to H
            HIdensity  = Yprofs[elID["H I"].id] * densitynH * elID["H I"].abund
            HeIdensity = Yprofs[elID["He I"].id] * densitynH * elID["He I"].abund
            HIdensity  = HIdensity.reshape((1,npts)).repeat(nions,axis=0)
            HeIdensity = HeIdensity.reshape((1,npts)).repeat(nions,axis=0)
            prof_alpha = edens_allions * prof_recomb + HIdensity * prof_recombCTHI  + HeIdensity * prof_recombCTHeI
            # Finally recalculate the rates
            prof_rates = prof_gamma / prof_alpha
            inneriter += 1

            if np.array_equal(tstconv,allionpnt):
                break
            elif inneriter > 100:
                logger.log("warning", "Break inner loop at 100 iterations, STATUS:")
                for j in range(nions):
                    logger.log("warning", "<--[  {0:d}/{1:d}  ]-->".format(tstconv[j],npts))
                break
            tmp_Yprofs = Yprofs.copy()
        logger.log("info", "Inner iteration cycled {0:d} times".format(inneriter))
        ## END INNER ITERATION

        # If a tabulated cooling function is used, there's no need to calculate any rates
        if temp_method not in {'eagle', 'relhic', 'isothermal'}:
            logger.log("debug", "Calculating heating rate")
            # Construct an array of ionization energies and the corresponding array for the indices
            ionlvl = np.zeros(nions, dtype=np.float)
            for j in range(nions):
                ionlvl[j] = elID[ions[j]].ip * elvolt / planck
                # Photoionization heating
                prof_eps  = 4.0 * np.pi * cython_fns.phheatrate_allion(jnurarr, phelxs, nuzero, ionlvl, planck)
                prof_phionheatrate = np.zeros(npts,dtype=np.float)
            for j in range(nions):
                prof_phionheatrate += prof_eps[j] * densitynH * elID[ions[j]].abund * Yprofs[j]
            # Secondary electron photoheating rate (Shull & van Steenberg 1985)
            heat_HI  = 4.0*np.pi * cython_fns.scdryheatrate(jnurarr, nuzero, phelxs[elID["H I"].id], electrondensity / (densitynH * (1.0 + 2.0 * prim_He)),
                                                            elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, planck, elvolt, 0)
            heat_HeI = 4.0*np.pi * cython_fns.scdryheatrate(jnurarr, nuzero, phelxs[elID["He I"].id], electrondensity / (densitynH * (1.0+2.0 * prim_He)),
                                                            elID["H I"].ip, elID["D I"].ip, elID["He I"].ip, planck, elvolt, 2)
            scdry_heat_rate = heat_HI * densitynH * Yprofs[elID["H I"].id] + heat_HeI * densitynH * prim_He * Yprofs[elID["He I"].id]

            # Finally, the total heating rate is:
            total_heat = prof_phionheatrate + scdry_heat_rate

            logger.log("debug", "Calculating cooling rate")
            # cooling rate evaluated at range of temperatures [rad_coord, temp_coord] 
            total_cool = cython_fns.cool_rate(total_heat, electrondensity, densitynH, Yprofs[elID["H I"].id], Yprofs[elID["He I"].id], Yprofs[elID["He II"].id], prim_He, redshift)

        logger.log("debug", "Deriving the temperature profile")
        old_temperature = prof_temperature.copy()
        
        if temp_method == 'original':
            prof_temperature, actual_cool = cython_fns.thermal_equilibrium_full(total_heat, total_cool, old_temperature)

        elif temp_method == 'eagle':
            prof_temperature = adiab_interp(densitynH)

        elif temp_method == 'relhic':
            prof_temperature = adiab_interp(densitynH)

        elif temp_method == 'blend':
            # below threshold 10**-4.8/cm^3, use the adiabatic result
            # above it, calculate thermal equilibrium
            # use tanh blending function to get smooth transition
            ad_temp = adiab_interp(densitynH)
            eq_temp, actual_cool = cython_fns.thermal_equilibrium_full(total_heat, total_cool, old_temperature)
            loc = None
            if np.sum((densitynH > density_thresh_ad).astype(np.int)) == 0:
                # always adiabatic
                prof_temperature = ad_temp
            else:
                loc = np.argmin(np.abs(densitynH - density_thresh_ad))
                prof_temperature = blendfxgx(ad_temp, eq_temp, loc, 50.0)

        elif temp_method == 'isothermal':
            pass # gas temperature was set at beginning, leave unchanged

        else:
            logger.log("critical", "Undefined temperature method")
            assert 0

        if np.size(np.where(prof_temperature<=1000.0001)[0]) != 0:
            logger.log("ERROR", "Profile temperature was estimated to be <= 1000 K")
            logger.log("ERROR", "The code is not currently designed to work in this regime")
            logger.log("ERROR", "Try a smaller radial range, or stronger radiation field")
            wtmp = np.where((prof_temperature<=1000.0001) | np.isnan(prof_temperature))
            prof_temperature[wtmp] = 1000.0001

        # Now make sure that the temperature jump is small
        # The maximum allowed jump is made large at the beginning and decreases as the iteration count increases
        # This helps speed up convergence (Oh no it doesn't!)
        if True:
            tmptemp = old_temperature-prof_temperature
            tmpsign = np.sign(tmptemp)
            lim = 1000 / (np.log10(iteration) + 1)
            tmptemp[np.where(np.abs(tmptemp)>lim)] = lim
            tmptemp = np.abs(tmptemp)*tmpsign
            prof_temperature = old_temperature-tmptemp

        prof_temperature = 0.7 * prof_temperature + 0.3 * old_temperature

        if iteration >= 100 #and iteration%1 == 0:
            logger.log("info", "Averaging the stored Yprofs")
            Yprofs = np.mean(store_Yprofs, axis=2)

        if lv_plot:
            live_plot.show()
            
        tstcrit = ((np.abs((old_Yprofs - Yprofs) / Yprofs) < concrit) | (Yprofs == 0.0)).astype(np.int).sum(axis=1)
        if np.array_equal(tstcrit, allionpnt):
            break

        logger.log("debug", "STATISTICS --")
        logger.log("debug", "ION  INDEX   OLD VALUE    NEW VALUE   |OLD-NEW|")
        w_maxoff   = np.argmax(np.abs((old_Yprofs - Yprofs) / Yprofs), axis=1)
        for j in range(nions):
            logger.log("debug", "{} {} {} {} {}".format(ions[j],
                                                       w_maxoff[j],
                                                       old_Yprofs[j,w_maxoff[j]],
                                                       Yprofs[j,w_maxoff[j]],
                                                       np.abs((old_Yprofs[j,w_maxoff[j]] - Yprofs[j,w_maxoff[j]]) / Yprofs[j,w_maxoff[j]])))

        # Check if the stopping criteria were met
        if iteration > maxiter:
            logger.log("warning", "Break outer loop at maxiter={0:d} iterations, STATUS:".format(maxiter))
            break
    ## END MAIN LOOP
    
    # Calculate the density profiles
    logger.log("info", "Calculating volume density profiles")
    for j in range(nions):
        # density of this specie = unionized fraction * H volume density * number abundance relative to H
        prof_density[j] = Yprofs[j] * densitynH * elID[ions[j]].abund
        logger.log("info", "{} {} {}".format(ions[j], np.max(Yprofs[j]), np.max(prof_density[j])))

    logger.log("info", "Calculating column density profiles")
    prof_coldens = np.zeros_like(prof_density)
    for j in range(nions):
        if geom in {"NFW", "Burkert", "Cored"}:
            coldens = cython_fns.coldensprofile(prof_density[j], radius)
            prof_coldens[j] = coldens.copy()
        elif geom == "PP":
            coldens = cython_fns.calc_coldensPP(prof_density[j], radius)
            prof_coldens[j] = coldens.copy()

    logger.log("info", "Calculating Ha surface brightness profile")
    Harecomb = recomb.Ha_recomb(prof_temperature)
    HIIdensity = densitynH - prof_density[elID["H I"].id]
    elecprot = Harecomb * electrondensity * HIIdensity # emissivity in photons / cm^3 / s
    if geom in {"NFW", "Burkert", "Cored"}:
        HaSB = (1.0 / (4.0 * np.pi)) * cython_fns.coldensprofile(elecprot, radius)  # photons /cm^2 / s / SR
    elif geom == "PP":
        HaSB = (1.0 / (4.0 * np.pi)) * cython_fns.calc_coldensPP(elecprot, radius)  # photons /cm^2 / s / SR
    # Multiply by Ha energy to convert from photons/s to ergs/s
    # Multiply by (4 * pi)/(4 * pi * (60^2 * 180/pi)^2) to convert from per SR to per arcsec^2
    HaSB *=  (3.0e10 * planck / 6.563e-5) / (4.254517E10)

    logger.log("info", "Calculating HeII 4686A surface brightness profile")
    HeIIrecomb = recomb.HeII4686_recomb(prof_temperature)
    HIIIdensity = (densitynH * prim_He) - prof_density[elID["He I"].id] - prof_density[elID["He II"].id]
    elecprot = HeIIrecomb * electrondensity * HIIdensity # emissivity in photons / cm^3 / s
    if geom in {"NFW", "Burkert", "Cored"}:
        HeII_SB = (1.0 / (4.0 * np.pi)) * cython_fns.coldensprofile(elecprot, radius)  # photons /cm^2 / s / SR
    elif geom == "PP":
        HeII_SB = (1.0 / (4.0 * np.pi)) * cython_fns.calc_coldensPP(elecprot, radius)  # photons /cm^2 / s / SR
    # Multiply by HeII 4686A energy to convert from photons/s to ergs/s
    # Multiply by (4 * pi)/(4 * pi * (60^2 * 180/pi)^2) to convert from per SR to per arcsec^2
    HeII_SB *=  (3.0e10 * planck / 4.686e-5) / (4.254517E10)
    
    if He_want_wls is not None:
        logger.log("info", "Calculating HeI surface brightness profile(s)")
        He_emis = He_emissivities.get_emis(He_want_wls, prof_temperature, electrondensity, prof_density[elID['He II'].id])
        He_SB = np.zeros_like(He_emis)
        for i, wl in enumerate(He_want_wls):
            # These emissivities are already in energy units so we do not multiply by the line energies here
            if geom in {"NFW", "Burkert", "Cored"}:
                He_SB[i] = (1.0 / (4.0 * np.pi)) * cython_fns.coldensprofile(He_emis[i], radius)  # erg / cm^2 / s / SR
            else:
                He_SB[i] = (1.0 / (4.0 * np.pi)) * cython_fns.calc_coldensPP(He_emis[i], radius)  # erg / cm^2 / s / SR
            # Multiply by (4 * pi)/(4 * pi * (60^2 * 180/pi)^2) to convert from per SR to per arcsec^2
            He_SB[i] *= 1 / (4.254517E10)

    timeB = time.time()
    logger.log("info", "Test completed in {0:f} mins".format((timeB - timeA) / 60.0))

    if lv_plot:
        live_plot.close()

    # Save the results
    if geom in {"NFW", "Burkert", "Cored"}:
        # Create a structured datatype with all requested fields
        save_dtype = [('rad', 'float64'),
                      ('temp', 'float64'),
                      ('hden', 'float64'),
                      ('eden', 'float64'),
                      ('pres', 'float64'),
                      ('HaSB', 'float64'),
                      ('vden', 'float64', nions),
                      ('cden', 'float64', nions),
                      ('yprf', 'float64', nions)]
        save_pressure = kB * densitynH * prof_temperature / masspp # want pressure in physical units
        mstring = mangle_string("{0:3.3f}".format(np.log10(hmodel.mvir / somtog)))
        cstring = mangle_string("{0:3.3f}".format(hmodel.rvir / hmodel.rscale))
        rstring = mangle_string("{0:3.2f}".format(redshift))
        bstring = mangle_string("{0:+3.3f}".format(np.log10(hmodel.baryfrac)))
        if options["UVB"]["spectrum"][0:2] in {"HM", "MH"}:
            hstring = mangle_string("HMscale{0:+3.2f}".format(np.log10(options["UVB"]["scale"])))
        elif options["UVB"]["spectrum"][0:2] == "PL":
            hstring = options["UVB"]["spectrum"]
        elif options["UVB"]["spectrum"] == "AGN":
            hstring = options["UVB"]["spectrum"]
        outfname = out_dir + ("{0:s}_mass{1:s}_concentration{2:s}_redshift{3:s}_baryscl{4:s}_{5:s}_{6:d}-{7:d}"
                              .format(geom,mstring,cstring,rstring,bstring,hstring,npts,nummu))
        logger.log("info", "Saving file {0:s}.npy".format(outfname))
        tmpout = np.concatenate((radius.reshape((npts,1)) * cmtopc,
                                 prof_temperature.reshape((npts,1)),
                                 densitynH.reshape((npts,1)),
                                 electrondensity.reshape((npts,1)),
                                 save_pressure.reshape((npts,1)),
                                 HaSB.reshape((npts,1)),
                                 prof_density.T,
                                 prof_coldens.T,
                                 Yprofs.T), axis=1)
    elif geom == "PP":
        save_dtype = [('rad', 'float64'),
                      ('temp', 'float64'),
                      ('hden', 'float64'),
                      ('eden', 'float64'),
                      ('HaSB', 'float64'),
                      ('vden', 'float64', nions),
                      ('cden', 'float64', nions),
                      ('yprf', 'float64', nions)]
        
        dstring = mangle_string("{0:+3.2f}".format(np.log10(PP_dens)))
        cdstring = mangle_string("{0:+3.2f}".format(np.log10(PP_cden)))
        if options["UVB"]["spectrum"][0:2] in  {"HM", "MH"}:
            hstring = mangle_string("HMscale{0:+3.2f}".format(np.log10(options["UVB"]["scale"])))
        elif options["UVB"]["spectrum"][0:2]=="PL":
            hstring = options["UVB"]["spectrum"]
        outfname = out_dir + ("{0:s}_density{1:s}_coldens{2:s}_{3:s}_{4:d}"
                    .format(geom,dstring,cdstring,hstring,npts))
        logger.log("info", "Saving file {0:s}.npy".format(outfname))
        tmpout = np.concatenate((radius.reshape((npts,1)) * cmtopc,
                                 prof_temperature.reshape((npts,1)),
                                 densitynH.reshape((npts,1)),
                                 electrondensity.reshape((npts,1)),
                                 HaSB.reshape((npts,1)),
                                 prof_density.T,
                                 prof_coldens.T,
                                 Yprofs.T), axis=1)

    if He_want_wls is not None:
        save_dtype.extend([('HeI_{:.0f}'.format(wl), 'float64') for wl in He_want_wls]) # TODO: list of HeI lines to save

        tmpout = np.concatenate((tmpout,
                                 He_SB.T), axis=1)

    if svrates:
        save_dtype.extend([('phion', 'float64', nions),  
                           ('scdry', 'float64', nions),  
                           ('CT_HII', 'float64', nions), 
                           ('CT_HeII', 'float64', nions),
                           ('other', 'float64', nions),  
                           ('colion', 'float64', nions)])
        
        tmpout = np.concatenate((tmpout,
                                 prof_phionrate.T,
                                 prof_scdryrate.T,
                                 (densitynH * (1 - Yprofs[elID["H I"].id]).reshape((1, npts)).repeat(nions, axis=0) * prof_chrgtraniHII).T,
                                 (densitynH * (prim_He * Yprofs[elID["He II"].id]).reshape((1, npts)).repeat(nions, axis=0) * prof_chrgtraniHeII).T,
                                 prof_other.T,
                                 prof_colionrate.T), axis=1)

    if svrcmb:
        save_dtype.extend([('recomb', 'float64', nions),  
                           ('rc_CTHI', 'float64', nions), 
                           ('rc_CTHeI', 'float64', nions)])

        tmpout = np.concatenate((tmpout,
                                 prof_recomb.T,
                                 prof_recombCTHI.T,
                                 prof_recombCTHeI.T), axis=1)

    if svhtcl:
        save_dtype.extend([('heat', 'float64'),
                           ('cool', 'float64')])

        tmpout = np.concatenate((tmpout,
                                 total_heat.reshape((npts,1)),
                                 actual_cool.reshape((npts,1))), axis=1)

    # TODO Decide how to handle saving Jnu(r)
    # need to record frequencies as well as intensity
    # if svjnu:
    #     save_dtype.extend([('Jnu', 'float64', jnurarr.shape[0])])
    #     tmpout = np.concatenate((tmpout,
    #                              jnurarr.T), axis=1)
    #     tmpnu = np.insert(nuzero, 0, np.nan).reshape((1, len(nuzero) + 1))
    #     tmpout = np.concatenate((tmpnu, tmpout), axis=0)
    #     #np.save(outfname + '_intensity', tmpout)

    assert np.dtype(save_dtype).itemsize == tmpout.itemsize * tmpout.shape[1]
    tmpout = np.ascontiguousarray(tmpout)
    tmpout.dtype = save_dtype
    np.save(outfname, tmpout) 

    # dispose of process pool
    if ncpus > 1:
        pool.close()
        pool.join()

    # Stop the program if a large H I column density has already been reached
    if np.max(np.log10(prof_coldens[elID["H I"].id])) > 24.0:
        print("Terminating after maximum N(H I) has been reached")
        sys.exit()

    # Everything is OK
    # Return True, and the output filename to be used as the input to the next iteration    
    return (True, outfname + '.npy')
