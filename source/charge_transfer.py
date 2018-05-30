"""
See the following website for the relevant data
https://web.archive.org/web/20160526002340/http://www-cfadc.phy.ornl.gov/astro/ps/data/
"""
import numpy as np
import misc

def rate_function_form(param, temp):
    # First determine the form of the profile
    if len(param.shape) == 2: form = param[0,0]
    else: form = param[0]
    if form==0:
        rate = rate_function_arr(temp,param)
    elif form == 1:
        rate = rate_function_arr_ion(temp,param)
    elif form == 2:
        rate = rate_function_arr_break(temp,param)
    elif form == 3:
        rate = rate_function_arr_DIHI(temp,param)
    else:
        print "ERROR :: Charge transfer functional form not implemented yet", form
        assert(False)
    return rate

def rate_function_arr(temp, arr):
    a, b, c, d, tmin, tmax = arr[1], arr[2], arr[3], arr[4], arr[5], arr[6]
    rate = 1.0E-9 * a * ((temp/1.0E4)**b) * (1.0 + c*np.exp(d*temp/1.0E4))
    wz = np.where(rate<0.0)
    rate[wz] = 0.0
# 	wn = np.where(temp < tmin)
# 	wx = np.where(temp > tmax)
# 	if wn[0].size != 0:
# 		rate[wn] = 1.0E-9 * a * ((tmin/1.0E4)**b) * (1.0 + c*np.exp(d*tmin/1.0E4))
# 	if wx[0].size != 0:
# 		rate[wx] = 1.0E-9 * a * ((tmax/1.0E4)**b) * (1.0 + c*np.exp(d*tmax/1.0E4))
    return rate

def rate_function_arr_ion(temp, arr):
    a, b, c, d, tmin, tmax, Ek = arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]
    rate = 1.0E-9 * a * ((temp/1.0E4)**b) * (1.0 + c*np.exp(d*temp/1.0E4)) * np.exp(-Ek*1.0E4/temp)
    wz = np.where(rate<0.0)
    rate[wz] = 0.0
# 	wn = np.where(temp < tmin)
# 	wx = np.where(temp > tmax)
# 	if wn[0].size != 0:
# 		rate[wn] = 1.0E-9 * a * ((tmin/1.0E4)**b) * (1.0 + c*np.exp(d*tmin/1.0E4)) * np.exp(-Ek*1.0E4/temp)
# 	if wx[0].size != 0:
# 		rate[wx] = 1.0E-9 * a * ((tmax/1.0E4)**b) * (1.0 + c*np.exp(d*tmax/1.0E4)) * np.exp(-Ek*1.0E4/temp)
    return rate

def rate_function_arr_DIHI(temp, arr):
    """
    Taken from Daniel Wolf Savin (2002), ApJ, 566, 599

    fb = 1 corresponds to D + H+ -> D+ + H
    fb = 2 corresponds to D+ + H -> D + H+
    """
    a, b, c, d, e, fb, tmin, tmax = arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8]
    rate = a * ((temp)**b) * np.exp(-c/temp) +  d * (temp**e)
    wz = np.where(rate<0.0)
    rate[wz] = 0.0
    if fb==1: return rate
    elif fb==2: return rate*np.exp(42.5/temp)
    else:
        print "ERROR defining D/H charge transfer rates!"
        sys.exit()
# 	wn = np.where(temp < tmin)
# 	wx = np.where(temp > tmax)
# 	if wn[0].size != 0:
# 		rate[wn] = a * ((tmin)**b) * np.exp(-c/tmin) +  d * (tmin**e)
# 	if wx[0].size != 0:
# 		rate[wx] = a * ((tmax)**b) * np.exp(-c*tmax) +  d * (tmax**e)
#	return rate

def rate_function_arr_break(temp, arr):
    rate = np.zeros(temp.size)
    i=0
    a, b, c, d, tmin, tmax = arr[1,i], arr[2,i], arr[3,i], arr[4,i], arr[5,i], arr[6,i]
    rate = 1.0E-9 * a * ((temp/1.0E4)**b) * (1.0 + c*np.exp(d*temp/1.0E4))
    for i in range(1,arr.shape[1]):
        w = np.where((temp>tmin)&(temp<=tmax))
        if np.size(w[0]) != 0:
            rate[w] = 1.0E-9 * a * ((temp[w]/1.0E4)**b) * (1.0 + c*np.exp(d*temp[w]/1.0E4))
    i=arr.shape[1]-1
    a, b, c, d, tmin, tmax = arr[1,i], arr[2,i], arr[3,i], arr[4,i], arr[5,i], arr[6,i]
    w = np.where(temp>tmin)
    if np.size(w[0]) != 0:
        rate[w] = 1.0E-9 * a * ((temp[w]/1.0E4)**b) * (1.0 + c*np.exp(d*temp[w]/1.0E4))
    # Make sure there's no zero rates
    wz = np.where(rate<0.0)
    rate[wz] = 0.0
    return rate

def rate_function(temp, a, b, c, d, tmin, tmax):
    rate = 1.0E-9 * a * ((temp/1.0E4)**b) * (1.0 + c*np.exp(d*temp/1.0E4))
    wz = np.where(rate<0.0)
    rate[wz] = 0.0
# 	wn = np.where(temp < tmin)
# 	wx = np.where(temp > tmax)
# 	if wn[0].size != 0:
# 		rate[wn] = 1.0E-9 * a * ((tmin/1.0E4)**b) * (1.0 + c*np.exp(d*tmin/1.0E4))
# 	if wx[0].size != 0:
# 		rate[wx] = 1.0E-9 * a * ((tmax/1.0E4)**b) * (1.0 + c*np.exp(d*tmax/1.0E4))
    return rate

def rate_function_ion(temp, a, b, c, d, tmin, tmax, Ek):
    rate = 1.0E-9 * a * ((temp/1.0E4)**b) * (1.0 + c*np.exp(d*temp/1.0E4)) * np.exp(-Ek*1.0E4/temp)
    wz = np.where(rate<0.0)
    rate[wz] = 0.0
# 	wn = np.where(temp < tmin)
# 	wx = np.where(temp > tmax)
# 	if wn[0].size != 0:
# 		rate[wn] = 1.0E-9 * a * ((tmin/1.0E4)**b) * (1.0 + c*np.exp(d*tmin/1.0E4)) * np.exp(-Ek*1.0E4/temp)
# 	if wx[0].size != 0:
# 		rate[wx] = 1.0E-9 * a * ((tmax/1.0E4)**b) * (1.0 + c*np.exp(d*tmax/1.0E4)) * np.exp(-Ek*1.0E4/temp)
    return rate

def rate_function_DIHI(temp, a, b, c, d, e, tmin, tmax):
    """
    Taken from Danial Wolf Savin (2002), ApJ, 566, 599
    """
    rate = a * ((temp)**b) * np.exp(-c/temp) +  d * (temp**e)
    wz = np.where(rate<0.0)
    rate[wz] = 0.0
# 	wn = np.where(temp < tmin)
# 	wx = np.where(temp > tmax)
# 	if wn[0].size != 0:
# 		rate[wn] = a * ((tmin)**b) * np.exp(-c/tmin) +  d * (tmin**e)
# 	if wx[0].size != 0:
# 		rate[wx] = a * ((tmax)**b) * np.exp(-c*tmax) +  d * (tmax**e)
    return rate

def rate_function_break(temp, al, bl, cl, dl, ah, bh, ch, dh, tmin, tmax, tbrk):
    rate = 1.0E-9 * al * ((temp/1.0E4)**bl) * (1.0 + cl*np.exp(dl*temp/1.0E4))
#	wn = np.where(temp < tmin)
    wb = np.where(temp >= tbrk)
#	wx = np.where(temp > tmax)
# 	if wn[0].size != 0:
# 		rate[wn] = 1.0E-9 * al * ((tmin/1.0E4)**bl) * (1.0 + cl*np.exp(dl*tmin/1.0E4))
    if wb[0].size != 0:
        rate[wb] = 1.0E-9 * ah * ((temp[wb]/1.0E4)**bh) * (1.0 + ch*np.exp(dh*temp[wb]/1.0E4))
# 	if wx[0].size != 0:
# 		rate[wx] = 1.0E-9 * ah * ((tmax/1.0E4)**bh) * (1.0 + ch*np.exp(dh*tmax/1.0E4))
    wz = np.where(rate<0.0)
    rate[wz] = 0.0
    return rate

def HI_target(lowion, temp):
    # Raise the ionization level by one (this is recombination to lowion, for example D II + H I --> D I + H II)
    ispl = lowion.split()
    tmp = misc.rntonum(ispl[1])
    ion = ispl[0]+" "+misc.numtorn(tmp+1,subone=True)
    # Now calculate the rate
    if ion == "D II":
        a, b, c, d, e, tmin, tmax = 2.06E-10, 0.396, 33.0, 2.03E-9, -0.332, 2.7, 2.0E5
        return rate_function_DIHI(temp, a, b, c, d, e, tmin, tmax)
    elif ion == "He II":
        a, b, c, d, tlo, thi = 7.47E-6, 2.06, 9.93, -3.89, 6E+3, 1E+5
    elif ion == "He III":
        a, b, c, d, tlo, thi = 1.00E-5, 0.00, 0.00, 0.00, 1.0E+3, 1.0E+7
    elif ion == "N II":
        a, b, c, d, tlo, thi = 1.01E-3, -0.29, -0.92, -8.38, 1.0E+2, 5.0E+4
    elif ion == "N III":
        a, b, c, d, tlo, thi = 3.05E-1, 0.60, 2.65, -0.93, 1.0E+3, 1.0E+5
    elif ion == "N IV":
        a, b, c, d, tlo, thi = 4.54, 0.57, -0.65, -0.89, 1.0E+1, 1.0E+5
    elif ion == "N V":
        a, b, c, d, tlo, thi = 2.95, 0.55, -0.39, -1.07, 1.0E+3, 1E+6
    elif ion == "Si III":
        a, b, c, d, tlo, thi = 6.77, 7.36E-2, -0.43, -0.11, 5.0E+2, 1.0E+5
    elif ion == "Si IV":
        a, b, c, d, tlo, thi = 4.90E-1, -8.74E-2, -0.36, -0.79, 1.0E+3, 3.0E+4
    elif ion == "Si V":
        a, b, c, d, tlo, thi = 7.58, 0.37, 1.06, -4.09, 1.0E+3, 5E+4
    else:
        return np.zeros(temp.size)
    return rate_function(temp, a, b, c, d, tlo, thi)

def HII_target(ion, temp):
    if ion == "D I":
        a, b, c, d, e, tmin, tmax = 2.00E-10, 0.402, 37.1, -3.31E-17, 1.48, 2.7, 2.0E5
        return rate_function_DIHI(temp, a, b, c, d, e, tmin, tmax)
    elif ion == "N I":
        a, b, c, d, tlo, thi, Ek = 4.55E-3, -0.29, -0.92, -8.38, 1.0E+2, 5.0E+4, 1.086
    elif ion == "Si I":
        a, b, c, d, tlo, thi, Ek = 0.92, 1.15, 0.80, -0.24, 1.0E+3, 2.0E+5, 0.000
    elif ion == "Si II":
        a, b, c, d, tlo, thi, Ek = 2.26, 7.36E-2, -0.43, -0.11, 2.0E+3, 1.0E+5, 3.031
    else:
        return np.zeros(temp.size)
    return rate_function_ion(temp, a, b, c, d, tlo, thi, Ek)

def HeI_target(lowion, temp):
    # Raise the ionization level by one (this is recombination to lowion, for example D II + H I --> D I + H II)
    ispl = lowion.split()
    tmp = misc.rntonum(ispl[1])
    ion = ispl[0]+" "+misc.numtorn(tmp+1,subone=True)
    if ion == "N III":
        al, bl, cl, dl = 4.84E-1, 0.92, 2.37, -1.02E1
        ah, bh, ch, dh = 3.17, 0.20, -0.72, -4.81E-2
        tlo, tbr, thi = 1.0E3, 4.0E4, 1.0E7
        return rate_function_break(temp, al, bl, cl, dl, ah, bh, ch, dh, tlo, thi, tbr)
    elif ion == "N IV":
        a, b, c, d, tlo, thi = 2.05, 0.23, -0.72, -0.19, 1.0E+3, 1.0E+7
        return rate_function(temp, a, b, c, d, tlo, thi)
    elif ion == "N V":
        al, bl, cl, dl = 1.26E-2, 1.55, 1.12E1, -7.82
        ah, bh, ch, dh = 3.75E-1, 0.54, -0.82, -2.07E-2
        tlo, tbr, thi = 1.0E3, 9.0E4, 1.0E7
        return rate_function_break(temp, al, bl, cl, dl, ah, bh, ch, dh, tlo, thi, tbr)
    elif ion == "Si IV":
        a, b, c, d, tlo, thi = 1.03, 0.60, -0.61, -1.42, 1.0E+2, 1.0E+6
        return rate_function(temp, a, b, c, d, tlo, thi)
    elif ion == "Si V":
        a, b, c, d, tlo, thi = 5.75E-1, 0.93, 1.33, -0.29, 1E+3, 5E+5
        return rate_function(temp, a, b, c, d, tlo, thi)
    else:
        return np.zeros(temp.size)
    return None

def HeII_target(ion, temp):
    if ion == "Si I":
        a, b, c, d, tlo, thi = 1.30, 0.00, 0.00, 0.00, 1.0E+1, 1.0E+4
    else:
        return np.zeros(temp.size)
    return rate_function(temp, a, b, c, d, tlo, thi)

def load_data(elID):
    datadict=dict({})
    datadict["H I"] = dict({})
    datadict["H II"] = dict({})
    datadict["He I"] = dict({})
    datadict["He II"] = dict({})
    ekeys = elID.keys()
    # Load the data for the H I target
    data = open("data/chrgtran_HItarget.dat").readlines()
    for i in range(len(data)):
        if data[i].strip()[0] == '#': continue
        datspl = data[i].split()
        elem = datspl[0].split("^")[0]
        ion  = int(datspl[0].split("^")[1].replace("+",""))
        ionstage = misc.numtorn(ion,subone=False) # Note -- this is subone=False because we need to lower all ionization dtages by one (in the same way that D II is lowered to D I below))
        elion = elem+" "+ionstage
        if elion not in ekeys: continue
        flag=0
        if elion in datadict["H I"].keys():
            prevarr = datadict["H I"][elion]
            if len(temparr.shape)==1:
                prevarr[0] = 2 # Change the form to be a broken function
                prevarr = prevarr.reshape(prevarr.shape[0],1)
            temparr = np.array([2])
            flag = 1
        else:
            temparr = np.array([0])
        for j in range(1,len(datspl)-1): temparr = np.append(temparr,float(datspl[j]))
        if flag==1:
            prevarr = np.append(prevarr,temparr.reshape(temparr.shape[0],1),axis=1)
            datadict["H I"][elion] = prevarr.copy()
        else: datadict["H I"][elion] = temparr.copy()
    # Load the data for the H II target
    data = open("data/chrgtran_HIItarget.dat").readlines()
    for i in range(len(data)):
        if data[i].strip()[0] == '#': continue
        datspl = data[i].split()
        elem = datspl[0].split("^")[0]
        ion  = int(datspl[0].split("^")[1].replace("+",""))
        ionstage = misc.numtorn(ion,subone=True)
        elion = elem+" "+ionstage
        if elion not in ekeys: continue
        flag=0
        if elion in datadict["H II"].keys():
            prevarr = datadict["H II"][elion]
            if len(temparr.shape)==1:
                prevarr[0] = 2 # Change the form to be a broken function
                prevarr = prevarr.reshape(prevarr.shape[0],1)
            temparr = np.array([2])
            flag = 1
        else:
            temparr = np.array([1])
        for j in range(1,len(datspl)-1): temparr = np.append(temparr,float(datspl[j]))
        if flag==1:
            prevarr = np.append(prevarr,temparr.reshape(temparr.shape[0],1),axis=1)
            datadict["H II"][elion] = prevarr.copy()
        else: datadict["H II"][elion] = temparr.copy()
    # Load the data for the He I target
    data = open("data/chrgtran_HeItarget.dat").readlines()
    for i in range(len(data)):
        if data[i].strip()[0] == '#': continue
        datspl = data[i].split()
        elem = datspl[0].split("^")[0]
        ion  = int(datspl[0].split("^")[1].replace("+",""))
        ionstage = misc.numtorn(ion,subone=False) # Note -- this is subone=False because we need to lower all ionization dtages by one (in the same way that D II is lowered to D I below))
        elion = elem+" "+ionstage
        if elion not in ekeys: continue
        flag=0
        if elion in datadict["He I"].keys():
            prevarr = datadict["He I"][elion]
            if len(prevarr.shape)==1:
                prevarr[0] = 2 # Change the form to be a broken function
                prevarr = prevarr.reshape(prevarr.shape[0],1)
            temparr = np.array([2])
            flag = 1
        else:
            temparr = np.array([0])
        for j in range(1,len(datspl)-1): temparr = np.append(temparr,float(datspl[j]))
        if flag==1:
            prevarr = np.append(prevarr,temparr.reshape(temparr.shape[0],1),axis=1)
            datadict["He I"][elion] = prevarr.copy()
        else: datadict["He I"][elion] = temparr.copy()
    # Load the data for the He II target
    if "Si I" in ekeys:
        datadict["He II"]["Si I"] = np.array([0, 1.30, 0.00, 0.00, 0.00, 1e+1, 1e+4])
    # Manually add some additional species
    # D I
    # Note for H I + D I: D II has been lowered to D I
    #datadict["H I"]["D I"] = np.array([3, 2.06E-10, 0.396, 33.0, 2.03E-9, -0.332, 2, 2.7, 2.0E5])
    datadict["H I"]["D I"] = np.array([3, 2.00E-10, 0.402, 37.1, -3.31E-17, 1.48, 2, 2.7, 2.0E5])
    datadict["H II"]["D I"] = np.array([3, 2.00E-10, 0.402, 37.1, -3.31E-17, 1.48, 1, 2.7, 2.0E5])
    # Return the result
    return datadict
