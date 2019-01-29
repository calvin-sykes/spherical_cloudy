import numpy as np
import misc
import os
import logger

def rate_function_diel_arr(temp, arr):
    """
    Uses the Badnell rates from:
    http://amdpp.phys.strath.ac.uk/tamoc/DATA/DR/
    """
    sumval = np.zeros(temp.size)
    for i in range(arr.shape[0]):
        sumval += arr[i,0] * np.exp(-arr[i,1]/temp)
    rateval = (temp**-1.5) * sumval
    w = np.where(rateval<0.0)
    rateval[w] = 0.0
    return rateval


def rate_function_radi_arr(temp, arr):
    form = arr[0]
    if form == 0:
        a, b, T0, T1 = arr[1], arr[2], arr[3], arr[4]
        if arr.size==7: c, T2 = arr[5], arr[6]
        else: c, T2 = 0.0, 0.0
        outrate = rate_function_radi(temp, a, b, c, T0, T1, T2)
    elif form == 1:
        outrate = rate_function_pl(temp, arr[1], arr[2])
    else:
        logger.log('error', "Form not implemented for radiative recombination rates (form = {})".format(form))
    return outrate


def get_rate(temp, a, b, T0, T1):
    return a / ( np.sqrt(temp/T0) * ((1.0+np.sqrt(temp/T0))**(1.0-b)) * ((1.0+np.sqrt(temp/T1))**(1.0+b)) )


def rate_function(temp, a, b, T0, T1, tmin=3.0, tmax=1.0E10):
    rateval = get_rate(temp, a, b, T0, T1)
    wn = np.where(temp<tmin)
    wx = np.where(temp>tmax)
    if wn[0].size > 0:
        rateval[wn] = get_rate(tmin, a, b, T0, T1)
    if wx[0].size > 0:
        rateval[wx] = get_rate(tmax, a, b, T0, T1)
    return rateval


def rate_function_radi(temp, a, bt, c, T0, T1, T2):
    "Uses the Badnell rates"
    b = bt + c*np.exp(-T2/temp)
    trm0 = (1.0 + np.sqrt(temp/T0))**(1.0-b)
    trm1 = (1.0 + np.sqrt(temp/T1))**(1.0+b)
    rateval = a / (np.sqrt(temp/T0)*trm0*trm1)
    w = np.where(rateval<0.0)
    rateval[w] = 0.0
    return rateval


def rate_function_pl(temp, A, B):
    return A * ((temp/1.0E4)**-B)


def rate_function_diel_badnell(temp, ci, Ei):
    "Uses the Badnell rates"
    sumval = np.zeros(temp.size)
    for i in range(ci.size):
        sumval += ci[i] * np.exp(-Ei[i]/temp)
    rateval = (temp**-1.5) * sumval
    w = np.where(rateval<0.0)
    rateval[w] = 0.0
    return rateval

def rate_function_diel(temp, a, b, T0, T1):
    logger.log('error', "Deprecated dielectronic recombination used")
    # High temperature rate
    rateval = a * (temp**-1.5) * np.exp(-T0/temp) * (1.0 + b*np.exp(-T1/temp))
    # Low temperature rate
    #t = temp/1.0E4
    #rateval = 1.0E-12 * (a/t + b + c*t + d*t*t) * (t**-1.5) * np.exp(-f/t)
    return rateval


def load_data_diel(elID):
    datadict=dict({})
    # Load the coefficients data
    data_c = open(os.path.join(os.path.dirname(__file__), "data/recomb_diel_ci.dat")).readlines()
    data_E = open(os.path.join(os.path.dirname(__file__), "data/recomb_diel_Ei.dat")).readlines()
    ekeys = elID.keys()
    for i in range(len(data_c)):
        if data_c[i].strip()[0] == '#': continue
        datspl_c = data_c[i].split()
        if datspl_c[2] != "1": continue
        elem = misc.numtoelem(int(datspl_c[0]),subone=False)
        ion  = int(datspl_c[0])-int(datspl_c[1])-1
        ionstage = misc.numtorn(ion,subone=True)
        elion = elem+" "+ionstage
        if elion not in ekeys: continue
        datspl_E = data_E[i].split()
        ncoeff = len(datspl_c)-4
        outarr = np.zeros((ncoeff,2))
        for j in range(ncoeff):
            outarr[j,0] = float(datspl_c[4+j])
            outarr[j,1] = float(datspl_E[4+j])
        datadict[elion] = outarr.copy()
    return datadict


def load_data_radi(elID):
    datadict=dict({})
    # Load the coefficients data
    data = open(os.path.join(os.path.dirname(__file__), "data/recomb_radi.dat")).readlines()
    ekeys = elID.keys()
    for i in range(len(data)):
        if data[i].strip()[0] == '#': continue
        datspl = data[i].split()
        if datspl[2] != "1": continue
        elem = misc.numtoelem(int(datspl[0]),subone=False)
        ion  = int(datspl[0])-int(datspl[1])-1
        ionstage = misc.numtorn(ion,subone=True)
        elion = elem+" "+ionstage
        if elion not in ekeys: continue
        ncoeff = len(datspl)-4
        outarr = np.zeros(1+ncoeff)
        for j in range(ncoeff):
            outarr[j+1] = float(datspl[4+j])
        datadict[elion] = outarr.copy()
    # Manually add/correct some additional data
    datadict["D I"] = datadict["H I"].copy()
    datadict["Si I"] = np.array([1.0,5.90E-13,0.601])
    return datadict


def hydrogenic_recomb(prof_temperature, grid_temp, grid_alpha, grid_line):
    """
    Calculate hydrogen-like recombination rates from data of form given in O&F AGN3.
    """
    grid_alpha *= grid_line
    coeff = np.polyfit(np.log10(grid_temp), np.log10(grid_alpha), 2)

    return 10**np.polyval(coeff, np.log10(prof_temperature))


def Ha_recomb(prof_temperature, case='B'):
    """Table 4.2 AGN3"""
    temps = np.array([2500.0, 5000.0, 10000.0, 20000.0])
    if case.upper() == 'B':
        alpha = np.array([9.07E-14, 5.37E-14, 3.03E-14, 1.62E-14])
        scale = np.array([3.30, 3.05, 2.87, 2.76])
    elif case.upper() == 'A':
        alpha = np.array([6.61E-14, 3.78E-14, 2.04E-14, 1.03E-14])
        scale = np.array([3.42, 3.10, 2.86, 2.69])
    else:
        logger.log('error', "Ha recombination case must be 'A' or 'B'")

    return hydrogenic_recomb(prof_temperature, temps, alpha, scale)


def HeII4686_recomb(prof_temperature):
    """Table 4.3 AGN3"""
    temps = np.array([5000.0, 10000.0, 20000.0, 40000.0])
    alpha = np.array([7.40e-13, 3.72e-13, 1.77e-13, 8.2e-14])
    scale = np.ones_like(alpha)

    return hydrogenic_recomb(prof_temperature, temps, alpha, scale)
