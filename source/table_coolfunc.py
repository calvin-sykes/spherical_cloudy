import numpy as np
import scipy.interpolate
import scipy.optimize

import h5py
from matplotlib import pyplot as plt
import matplotlib as mpl

import glob
import os

# suppress deprecation warning
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.resetwarnings()
#del warnings

import logger

def load_eagle_nHT(prim_He, redshift):
    def extract_z(fn):
        z = -1
        try:
            z = float(fn.rsplit('_', 1)[1].strip('.hdf5'))
        except ValueError:
            pass
        return z

    files_list = glob.glob(os.path.join(os.path.dirname(__file__), "data/CoolingTables/z_*.hdf5"))
    
    z_arr = np.fromiter(map(extract_z, files_list), float, len(files_list))
    idx_bestz = np.argmin(np.abs(redshift - z_arr))
    
    f = h5py.File(files_list[idx_bestz], 'r')
    data = f['Metal_free']

    Tbins = data['Temperature_bins']
    nbins = data['Hydrogen_density_bins']
    Ybins = data['Helium_mass_fraction_bins']

    cool = data['Net_Cooling']

    # find helium mass fraction bin closest to that chosen
    bestY = np.argmin(np.abs(prim_He - Ybins[...]))
    coolY = cool[bestY]
    coolYint = scipy.interpolate.RectBivariateSpline(Tbins, nbins, coolY)

    return coolYint


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
