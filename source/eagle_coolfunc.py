import numpy as np
import h5py
import scipy.interpolate
import scipy.optimize

import logger

interp = None

def load_eagle_cf(prim_He):
    f = h5py.File('/cosma5/data/Eagle/BG_Tables/CoolingTables/z_0.000.hdf5', 'r')
    data = f['Metal_free']

    Tbins = data['Temperature_bins']
    nbins = data['Hydrogen_density_bins']
    Ybins = data['Helium_mass_fraction_bins']

    cool = data['Net_Cooling']

    # find helium mass fraction bin closest to that chosen
    bestY = np.argmin(np.abs(prim_He - Ybins.value))
    coolY = cool[bestY] #.T

    interp = scipy.interpolate.interp2d(nbins, Tbins, coolY)

    #densvals = np.logspace(-7, 0, 1000)
    #tempvals = np.logspace(3, 6, 1000)

    return interp #tempvals, densvals, np.abs(interp(tempvals, densvals))

def load_relhic_nHT():
    dens, temp = np.loadtxt('./data/relhic_nHT.dat', unpack=True)

    # Use a power-law interpolation above 1 atom per cm**3
    dens_lim = -1
    fitting_region = dens > 10**dens_lim

    plaw_fit = lambda log_n, amp, slope, offset: amp * (log_n-dens_lim)**slope + offset
    fit_params = scipy.optimize.curve_fit(plaw_fit, np.log10(dens[fitting_region]), np.log10(temp[fitting_region]))[0]
    logger.log('debug', "Fitted RELHIC nH-T extrapolation with params: ({}, {}, {})".format(*fit_params))

    extrapolation_region = np.linspace(0.01, 10, 100)
    extrap_dens = np.append(dens, 10**extrapolation_region)
    extrap_temp = np.append(temp, 10**plaw_fit(extrapolation_region, *fit_params))

    interp = scipy.interpolate.InterpolatedUnivariateSpline(extrap_dens, extrap_temp)

    return interp
