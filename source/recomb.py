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


def radiative(ion,temp):
    form = 0
    c, T2 = 0.0, 0.0
    if ion == "H I":
        a, b, T0, T1 = 8.318E-11, 0.7472, 2.965E+00, 7.001E+05
    elif ion == "D I":
        a, b, T0, T1 = 8.318E-11, 0.7472, 2.965E+00, 7.001E+05
    elif ion == "He I":
        a, b, T0, T1, c, T2 = 5.235E-11, 0.6988, 7.301E+00, 4.475E+06, 0.0829, 1.682E+05
    elif ion == "He II":
        a, b, T0, T1 = 1.818E-10, 0.7492, 1.017E+01, 2.786E+06
    elif ion == "N I":
        a, b, T0, T1, c, T2 = 6.387E-10, 0.7308, 9.467E-02, 2.954E+06, 0.2440, 6.739E+04
    elif ion == "N II":
        a, b, T0, T1, c, T2 = 2.410E-09, 0.7948, 1.231E-01, 3.016E+06, 0.0774, 1.016E+05
    elif ion == "N III":
        a, b, T0, T1, c, T2 = 7.923E-10, 0.7768, 3.750E+00, 3.468E+06, 0.0223, 7.206E+04
    elif ion == "N IV":
        a, b, T0, T1 = 1.533E-10, 0.6682, 1.823E+02, 7.751E+06
    elif ion == "Si I":
        a, b = 5.90E-13, 0.601
        form = 1
    elif ion == "Si II":
        a, b, T0, T1, c, T2 = 1.964E-10, 0.6287, 7.712E+00, 2.951E+07, 0.1523, 4.804E+05
    elif ion == "Si III":
        a, b, T0, T1, c, T2 = 6.739E-11, 0.4931, 2.166E+02, 4.491E+07, 0.1667, 9.046E+05
    elif ion == "Si IV":
        a, b, T0, T1, c, T2 = 5.134E-11, 0.3678, 1.009E+03, 8.514E+07, 0.1646, 1.084E+06
    elif ion == "Si V":
        a, b, T0, T1, c, T2 = 2.468E-10, 0.6113, 1.649E+02, 3.231E+07, 0.0636, 9.837E+05
    else:
        return np.zeros(temp.size)
# 	if ion == "H I":
# 		a, b, T0, T1 = 7.982E-11, 0.7480, 3.148E+00, 7.036E+05
# 	elif ion == "D I":
# 		a, b, T0, T1 = 7.982E-11, 0.7480, 3.148E+00, 7.036E+05
# 	elif ion == "He I":
# 		a, b, T0, T1 = 3.294E-11, 0.6910, 1.554E+01, 3.676E+07
# 	elif ion == "He II":
# 		a, b, T0, T1 = 1.891E-10, 0.7524, 9.370E+00, 2.774E+06
#  	elif ion == "N I":
#  		a, b = 4.10E-13, 0.608
#  		form = 1
#  	elif ion == "N II":
#  		a, b = 2.20E-12, 0.639
#  		form = 1
#  	elif ion == "N III":
#  		a, b = 5.00E-12, 0.676
#  		form = 1
#  	elif ion == "N IV":
#  		a, b = 9.40E-12, 0.765
#  		form = 1
#  	elif ion == "Si I":
#  		a, b = 5.90E-13, 0.601
#  		form = 1
#  	elif ion == "Si II":
#  		a, b = 1.00E-12, 0.786
#  		form = 1
#  	elif ion == "Si III":
#  		a, b = 3.70E-12, 0.693
#  		form = 1
# 	elif ion == "Si IV":
# 		a, b, T0, T1 = 5.942E-11, 0.3930, 8.962E+02, 1.213E+07
# 	else:
# 		return np.zeros(temp.size)
# 	if ion == "H I": return 2.54E-13 * (temp/1.0E4)**(-0.8163 - 0.0208*np.log(temp/1.0E4))
# 	elif ion == "He I": return 9.03E-14 * (temp/4.0E4)**(-0.830-0.0177*np.log(temp/4.0E4))
# 	elif ion == "He II": return 5.08E-13 * (temp/4.0E4)**(-0.8163 - 0.0208*np.log(temp/4.0E4))
# 	if form==0: return rate_function(temp, a, b, T0, T1)
# 	else: return rate_function_pl(temp, a, b)
    if form == 0: return rate_function_radi(temp, a, b, c, T0, T1, T2)
    else: return rate_function_pl(temp, a, b)



def dielectronic(ion,temp):
    # Use the Badnell rates (2003,2006)
    if ion == "He I":
        ci = np.array([5.966E-04, 1.613E-04, -2.223E-05])
        Ei = np.array([4.556E+05, 5.552E+05, 8.982E+05])
    elif ion == "N I":
        ci = np.array([1.658E-08, 2.760E-08, 2.391E-09, 7.585E-07, 3.012E-04, 7.132E-04])
        Ei = np.array([1.265E+01, 8.425E+01, 2.964E+02, 5.923E+03, 1.278E+05, 2.184E+05])
    elif ion == "N II":
        ci = np.array([7.712E-08, 4.839E-08, 2.218E-06, 1.536E-03, 3.647E-03, 4.234E-05])
        Ei = np.array([7.113E+01, 2.765E+02, 1.439E+04, 1.347E+05, 2.496E+05, 2.204E+06])
    elif ion == "N III":
        ci = np.array([3.386E-06, 3.036E-05, 5.945E-05, 1.195E-03, 6.462E-03, 1.358E-03])
        Ei = np.array([1.406E+03, 6.965E+03, 2.604E+04, 1.304E+05, 1.965E+05, 4.466E+06])
    elif ion == "N IV":
        ci = np.array([2.040E-06, 6.986E-05, 3.168E-04, 4.353E-03, 7.765E-04, 5.101E-03])
        Ei = np.array([3.084E+03, 1.332E+04, 6.475E+04, 1.181E+05, 6.687E+05, 4.778E+06])
    elif ion == "Si I":
        ci = np.array([3.408E-08, 1.913E-07, 1.679E-07, 7.523E-07, 8.386E-05, 4.083E-03])
        Ei = np.array([2.431E+01, 1.293E+02, 4.272E+02, 3.729E+03, 5.514E+04, 1.295E+05])
    elif ion == "Si II":
        ci = np.array([2.930E-06, 2.803E-06, 9.023E-05, 6.909E-03, 2.582E-05])
        Ei = np.array([1.162E+02, 5.721E+03, 3.477E+04, 1.176E+05, 3.505E+06])
    elif ion == "Si III":
        ci = np.array([3.819E-06, 2.421E-05, 2.283E-04, 8.604E-03, 2.617E-03])
        Ei = np.array([3.802E+03, 1.280E+04, 5.953E+04, 1.026E+05, 1.154E+06])
    elif ion == "Si IV":
        ci = np.array([1.422E-04, 9.474E-03, 1.650E-03])
        Ei = np.array([7.685E+05, 1.208E+06, 1.839E+06])
    elif ion == "Si V":
        ci = np.array([7.163E-07, 2.656E-06, 1.119E-06, 4.796E-05, 4.052E-03, 6.101E-03, 2.366E-02])
        Ei = np.array([5.625E+02, 2.952E+03, 9.682E+03, 1.473E+05, 5.064E+05, 8.047E+05, 1.623E+06])
    else:
        return np.zeros(temp.size)
# 	if ion == "He I":
# 		a, b, T0, T1 = 1.90E-03, 3.00E-01, 4.70E+05, 9.40E+04
#  	elif ion == "N I":
#  		a, b, T0, T1 = 2.98E-03, 0.00E+00, 2.20E+05, 1.00E+05
#  	elif ion == "N II":
#  		a, b, T0, T1 = 7.41E-03, 7.64E-02, 2.01E+05, 7.37E+04
#  	elif ion == "N III":
#  		a, b, T0, T1 = 1.13E-02, 1.64E-01, 1.72E+05, 2.25E+05
#  	elif ion == "N IV":
#  		a, b, T0, T1 = 2.62E-03, 2.43E-01, 1.02E+05, 1.25E+05
#  	elif ion == "Si I":
#  		a, b, T0, T1 = 1.10E-03, 0.00E+00, 7.70E+04, 1.00E+05
#  	elif ion == "Si II":
#  		a, b, T0, T1 = 5.87E-03, 7.53E-01, 9.63E+04, 6.46E+04
#  	elif ion == "Si III":
#  		a, b, T0, T1 = 5.03E-03, 1.88E-01, 8.75E+04, 4.71E+04
# 	elif ion == "Si IV":
#  		a, b, T0, T1 = 5.43E-03, 4.50E-01, 1.05E+06, 7.98E+05
# 	else:
# 		return np.zeros(temp.size)
# 	return rate_function_diel(temp, a, b, T0, T1)
    return rate_function_diel_badnell(temp, ci, Ei)

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

def Ha_recomb(prof_temperature, case='B'):
    """
    Table 4.2, Osterbrock & Ferland (2006), pg. 73
    """
    temps = np.array([2500.0, 5000.0, 10000.0, 20000.0])
    if case.upper() == 'B':
        alpha = np.array([9.07E-14, 5.37E-14, 3.03E-14, 1.62E-14])
        scale = np.array([3.30, 3.05, 2.87, 2.76])
    elif case.upper() == 'A':
        alpha = np.array([6.61E-14, 3.78E-14, 2.04E-14, 1.03E-14])
        scale = np.array([3.42, 3.10, 2.86, 2.69])
    else:
        logger.log('error', "Ha recombination case must be 'A' or 'B'")
    
    alpha *= scale
    coeff = np.polyfit(np.log10(temps), np.log10(alpha), 2)
    #from matplotlib import pyplot as plt
    #tempm = np.linspace(3.0, 5.0, 100)
    #model = np.polyval(coeff, tempm)
    #plt.plot(np.log10(temps), np.log10(alpha), 'bx')
    #plt.plot(tempm, model, 'r-')
    #plt.show()
    model_alpha = 10.0**np.polyval(coeff, np.log10(prof_temperature))
    return model_alpha
