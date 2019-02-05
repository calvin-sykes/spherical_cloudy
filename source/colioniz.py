"""
See the following website for the relevant data
http://www.pa.uky.edu/~verner/col.html
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import scipy.special as specialfunc
import misc

def get_rate(Te, dE, P, A, X, K):
    U = dE/Te
    num = 1 + P*np.sqrt(U)
    den = X + U
    rateval = A * (num/den) * (U**K) * np.exp(-U)
    w = np.where(U>80.0)
    rateval[w] = 0.0
    return rateval


def rate_function_arr(Te, arr):
    dE = arr[0]
    P  = arr[1]
    A  = arr[2]
    X  = arr[3]
    K  = arr[4]
    tmin = arr[5]
    tmax = arr[6]
#	tmax *= 1000.0 # Convert keV to eV
    rateval = get_rate(Te, dE, P, A, X, K)
    wn = np.where(rateval<0.0)
#	wn = np.where((rateval<0.0)|(Te<dE))
    rateval[wn] = 0.0
#	wx = np.where(Te>tmax)
#	if wn[0].size > 0:
#		rateval[wn] = 0.0#get_rate(tmin, dE, P, A, X, K)
#	if wx[0].size > 0:
#		rateval[wx] = get_rate(tmax, dE, P, A, X, K)
    return rateval


def rate_function_Dere2007(Te, arr):
    """
    temperature is in eV
    """
    tck = interpolate.splrep(arr[2], arr[3], s=1) # spline representation of ionisation rate
    # arr[2] is x values, arr[3] is scaled rates
    redt = Te/arr[0] # ratio of temperature in eV to IP
    xval = 1.0 - np.log(2.0)/np.log(2.0 + redt) # Eq. (4) of Dere+ 2007
    #print(xval)
    rho = interpolate.splev(xval, tck, der=0, ext=0)
    rateval = (redt**-0.5) * (arr[0]**-1.5) * (1.0E-6*rho) * specialfunc.exp1(1.0/redt)
    return rateval


def rate_function_Chianti(Te, arr, arrTe):
    """
    temperature is in eV
    """
    return np.interp(Te, arrTe, arr)


def load_data(elidx, rates="Voronov1997"):
    """
    Load Dima Verner's data (which is pulled from the Voronov 1997 compilation).
    To call, use the following call
    info = load_data(dict({}))
    print info["C IV"]
    """
    datadict = dict({})
    ekeys = elidx.keys()
    if rates=="Voronov1997":
        data = np.loadtxt("data/colioniz.dat",usecols=(2,3,4,5,6,7,8,9,10))
        for i in range(data.shape[0]):
            elem = misc.numtoelem(int(data[i,0]))
            ionstage = misc.numtorn(int(data[i,0])-int(data[i,1]),subone=True)
            elion = elem+" "+ionstage
            if elion not in ekeys: continue # Don't include unnecessary data
            data[i,2] = elidx[elion][2]
            datadict[elion] = data[i,2:]
        # Manually add some additional species
        # D I
        datadict["D I"] = np.array([elidx["D I"][2], 0.0, 0.291E-07, 0.2320, 0.39, 1, 20])
    elif rates=="Dere2007":
        lines = open("data/colioniz_dere2007.dat").readlines()
        for i in range(len(lines)):
            if len(lines[i].strip())==0: continue # Nothing on a line
            if lines[i].strip()[0]=='#': continue # A comment line
            linspl = lines[i].split("|")
            elem = misc.numtoelem(int(linspl[0]))
            ionstage = misc.numtorn(int(linspl[1]),subone=False)
            elion = elem+" "+ionstage
            if elion not in ekeys: continue # Don't include unnecessary data
            nspl = int(linspl[2])
            xspl, yspl = np.zeros(nspl), np.zeros(nspl)
            ip = elidx[elion][2]
            mintemp = float(linspl[4])
            for j in range(0,nspl):
                xspl[j] = float(linspl[j+5])
                yspl[j] = float(linspl[j+25])
            datadict[elion] = [ip, mintemp, xspl.copy(), yspl.copy()]
            if elion == "H I":
                # Manually add in D I
                datadict["D I"] = [elidx["D I"].ip, mintemp, xspl.copy(), yspl.copy()]
    elif rates=="Chianti":
        with open("data/colioniz_chianti.dat") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # header has ion names
                if i == 0:
                    ion_labels = line.lstrip('#').split()
                    labels = []
                    for label in ion_labels:
                        if label == 'temp': continue # first column is temperature, ignore
                        # convert format of ion labels
                        # saved as e.g. 'he_2'; want 'He II'
                        lsplit = label.split('_')
                        elem = lsplit[0].capitalize()
                        ionstage = misc.numtorn(int(lsplit[1]))
                        elion = elem + ' ' + ionstage
                        if elion not in ekeys: continue # Don't include unnecessary data
                        # keep track of order ions are in
                        labels.append(elion)
                        datadict[elion] = np.zeros(len(lines) - 1)
                    temp = np.zeros(len(lines) - 1)
                else:
                    lsplit = line.split()
                    isubone = i - 1
                    temp[isubone] = float(lsplit[0])
                    for colidx, elion in enumerate(labels):
                        datadict[elion][isubone] = float(lsplit[colidx + 1]) # + 1 to skip temperature
        # manually add D I
        datadict["D I"] = datadict["H I"].copy()
        # for this method also need to return temperature grid
        return datadict, temp
    else:
        print "Error: Collisional Ionization rates not found"
        sys.exit()
    # Return the result
    return datadict
