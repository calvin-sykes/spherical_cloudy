import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline as spBiSpl

import logger

class HeEmissivities:
    def __init__(self):
        """
        Load Ryan Porter's He line emissivities.
        Stores a dict of numpy arrays, one per line wavelength.
        Each array tabulates emissivities for different T and log n_e.
        """
        with open(os.path.join(os.path.dirname(__file__), "data/HeI_Porter2013.dat"), 'r') as f:
            currlist = []
            alllist = []
            for i, l in enumerate(f.readlines()):
                # skip header
                if i < 55:
                    continue
                # collect all line wavelengths into a dict
                elif i == 55:
                    lspl = l.split('\t')
                    wavelengths = map(float, map(lambda s: s.strip('A\n'), lspl[2:]))
                # loop over lines, collecting
                # until blank line is found signifying end of block
                # then take lists of emissivities for each line and make into numpy array
                else:
                    if l != '\n':
                        currlist.append(map(float, l.split('\t')))
                    elif currlist != []: # handle second blank line separating blocks of data
                        alllist.append(np.array(currlist))
                        currlist = []

        # Determine values of T and log n_e that emissivities are tabulated for
        self.grid_T = alllist[0][:,0]
        self.grid_ne = np.array([np.unique(arr[:,1]).squeeze() for arr in alllist])

        # Rearrange the arrays to have one 2d array per wavelength,
        # ranging over T and log n_e for each
        self.emis_dict = dict({})
        for i, wl in enumerate(wavelengths):
            icol = i + 2 # skip over T and log n_e columns
            tmparr = np.array([arr[:,icol] for arr in alllist])
            #self.emis_dict[wl] = tmparr
            self.emis_dict[wl] = spBiSpl(self.grid_ne, self.grid_T, tmparr)
        #plt.figure()
        #plt.contourf(self.grid_ne, self.grid_T, tmparr.T)
        #plt.colorbar()
        #plt.show()

    def get_wavelengths(self):
        """Return a list of the wavelengths for which data was loaded"""
        return np.array(sorted(self.emis_dict.keys()))

    def get_emis(self, lines, prof_temperature, prof_edens, prof_HeII):
        """
        Get values for line emissivities.
        lines: scalar or list giving wavelengths for which emissivities are desired
        prof_temperature: radial temperature profile
        prof_edens: radial n_e profile
        prof_HeII: radial n_HeII density profile
        """
        if not hasattr(lines, '__iter__'):
            lines = [lines] # promote to length 1 list
        ret_emis = []
        for wl in lines:
            if wl not in self.emis_dict.keys():
                logger.log('error', "HeII emissivity requested for unknown line {}A".format(wl))
                ret_emis.append(np.zeros_like(prof_temperature))
            else:
                emis = self.emis_dict[wl](np.log10(prof_edens), prof_temperature, grid=False)
                ret_emis.append(10**emis * prof_edens * prof_HeII) # values are stored as logs
        return np.array(ret_emis)

#class HydrogenicEmissivities:
#    def __init__(self):
#        self.

if __name__ == '__main__':
    HeEm = HeEmissivities()

    wls = HeEm.get_wavelengths()

    prof_temperature = np.ones(1000) * 1e4
    prof_edens = np.logspace(-6, 14, 1000)
    prof_HeII = np.ones(1000) * 1e-2

    test_em = np.array(HeEm.get_emis(wls, prof_temperature, prof_edens, prof_HeII))
    ord_intens = np.argsort(test_em.max(axis=1))[::-1] # descending order
    
    plt.figure()
    plt.xlabel(r'$\log\ n_e$')
    plt.ylabel(r'$\log\ \epsilon/(n_e\ n_{He+})$')
    for i in ord_intens[0:5]:
        lim = prof_edens >= 10
        l, = plt.plot(np.log10(prof_edens[lim]), np.log10(test_em[i][lim] / (prof_edens * prof_HeII)[lim]), label=r'$\lambda{:.0f}$'.format(wls[i]))
        lim = prof_edens < 10
        plt.plot(np.log10(prof_edens[lim]), np.log10(test_em[i][lim] / (prof_edens * prof_HeII)[lim]), ls=':', c=l.get_color())
    plt.legend()
    plt.show()
