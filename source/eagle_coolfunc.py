import numpy as np
import h5py
import scipy.interpolate

interp = None

def load_cf(prim_He):
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
