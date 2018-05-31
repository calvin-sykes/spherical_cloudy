# ----------------------------------------------------------------- #
# If CHIANTI and ChiantiPy are not installed this script won't work #
# ----------------------------------------------------------------- #

# Fetches collisional ionisation data from the CHIANTI database using
# ChiantiPy and saves the rates to a data file 'colioniz_chianti.dat'

import numpy as np
#import matplotlib.pyplot as plt
import ChiantiPy.core as ch

if __name__ == '__main__':
    # which elements to include
    anums = [ 1 ,  2  ,  6  , 7  , 8 ]
    elems = ['h', 'he', 'c', 'n', 'o']

    #construct list of all partial ionisation stages for each element (i.e. h_1, he_1, he_2, c_1, ...
    ion_names = [elem + '_' + str(ionstage + 1) for anum, elem in enumerate(elems, start=1) for ionstage in range(anum)]

    # array of temperatures at which to evaluate the collisional ionisation rates
    temp = np.logspace(3, 5, 1000)

    # output temperatures in eV for consistency with other methods in spherical_cloudy
    kB = 1.3806488E-23
    elvolt = 1.60217657E-19
    eV_temp = temp * kB / elvolt
    
    # construct ChiantiPy objects for each ion under consideration
    ion_objs = [ch.ion(name, temperature=temp) for name in ion_names]
    
    out_data = np.zeros((len(temp), len(ion_names) + 1))
    out_data[:,0] = eV_temp

    for i, ion in enumerate(ion_objs):
        print('Getting ' + ion_names[i])
        ion.ionizRate() # populate field in ChiantiPy object with the rates
        out_data[:, i+1] = ion.IonizRate['rate'] # extract rates data

    np.savetxt('colioniz_chianti.dat', out_data, header='temp ' + ' '.join(ion_names))
    
    #plt.figure()
    #plt.plot(np.log10(out_data[:,0]), np.log10(out_data[:,1:]))
    #plt.show()
