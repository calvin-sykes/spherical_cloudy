"""
See the following website for the relevant data
http://www-cfadc.phy.ornl.gov/astro/ps/data/
"""
import numpy as np
import misc

def rate_function_arr(engy, arr):
	Et, Emx, Eo, so, ya, P, yw, yo, y1 = arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8]
	xsec = np.zeros(engy.size)
	x = engy/Eo - yo
	y = np.sqrt(x**2 + y1**2)
	Fy = ((x-1.0)**2 + yw**2) * y**(0.5*P - 5.5) * (1.0 + np.sqrt(y/ya))**(-1.0*P)
	w = np.where((engy>=Et)&(engy<=Emx))
	xsec[w] = 1.0E-18 * so * Fy[w]
	return xsec

def rate_function(engy, Et, Emx, Eo, so, ya, P, yw, yo, y1):
	xsec = np.zeros(engy.size)
	x = engy/Eo - yo
	y = np.sqrt(x**2 + y1**2)
	Fy = ((x-1.0)**2 + yw**2) * y**(0.5*P - 5.5) * (1.0 + np.sqrt(y/ya))**(-1.0*P)
	w = np.where((engy>=Et)&(engy<=Emx))
	xsec[w] = 1.0E-18 * so * Fy[w]
	return xsec

def xsec(ion,engy,elidx):
	if ion == "H I":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 13.59843, 5.000E+04, 4.298E-01, 5.475E+04, 3.288E+01, 2.963E+00, 0.000E+00, 0.000E+00, 0.000E+00
	elif ion == "D I":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 13.60213, 5.000E+04, 4.298E-01, 5.475E+04, 3.288E+01, 2.963E+00, 0.000E+00, 0.000E+00, 0.000E+00
	elif ion == "He I":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 24.58741, 5.000E+04, 1.361E+01, 9.492E+02, 1.469E+00, 3.188E+00, 2.039E+00, 4.434E-01, 2.136E+00
	elif ion == "He II":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 54.41778, 5.000E+04, 1.720E+00, 1.369E+04, 3.288E+01, 2.963E+00, 0.000E+00, 0.000E+00, 0.000E+00
	elif ion == "N I":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 14.53414, 4.048E+02, 4.034E+00, 8.235E+02, 8.033E+01, 3.928E+00, 9.097E-02, 8.598E-01, 2.325E+00
	elif ion == "N II":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 29.60130, 4.236E+02, 6.128E-02, 1.944E+00, 8.163E+02, 8.773E+00, 1.043E+01, 4.280E+02, 2.030E+01
	elif ion == "N III":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 47.44924, 4.473E+02, 2.420E-01, 9.375E-01, 2.788E+02, 9.156E+00, 1.850E+00, 1.877E+02, 3.999E+00
	elif ion == "N IV":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 77.47350, 4.753E+02, 5.494E+00, 1.690E+04, 1.714E+00, 1.706E+01, 7.904E+00, 6.415E-03, 1.937E-02
	elif ion == "Si I":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 8.151690, 1.060E+02, 2.317E+01, 2.506E+01, 2.057E+01, 3.546E+00, 2.837E-01, 1.672E-05, 4.207E-01
	elif ion == "Si II":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 16.34585, 1.186E+02, 2.556E+00, 4.140E+00, 1.337E+01, 1.191E+01, 1.570E+00, 6.634E+00, 1.272E-01
	elif ion == "Si III":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 33.49302, 1.311E+02, 1.659E-01, 5.790E-04, 1.474E+02, 1.336E+01, 8.626E-01, 9.613E+01, 6.442E-01
	elif ion == "Si IV":
		Et, Emx, Eo, so, ya, P, yw, yo, y1 = 45.14181, 1.466E+02, 1.288E+01, 6.083E+00, 1.356E+06, 3.353E+00, 0.000E+00, 0.000E+00, 0.000E+00
	else:
		return np.zeros(engy.size)
	# Use the ionization energy from NIST:
	Et = elidx[ion][2]
	return rate_function(engy, Et, Emx, Eo, so, ya, P, yw, yo, y1)

def load_data(elidx):
	"""
	Load Dima Verner's data.
	To call, use the following call
	info = load_data(dict({}))
	print info["C IV"]
	"""
	datadict = dict({})
	data = np.loadtxt("data/phionxsec.dat")
	ekeys = elidx.keys()
	for i in range(data.shape[0]):
		elem = misc.numtoelem(int(data[i,0]))
 		ionstage = misc.numtorn(int(data[i,0])-int(data[i,1]),subone=True)
 		elion = elem+" "+ionstage
 		if elion not in ekeys: continue # Don't include the unnecessary data
 		data[i,2] = elidx[elion][2]
		datadict[elion] = data[i,2:]
	# Manually add some additional species
	# D I
	datadict["D I"] = np.array([elidx["D I"][2], 5.000E+04, 4.298E-01, 5.475E+04, 3.288E+01, 2.963E+00, 0.000E+00, 0.000E+00, 0.000E+00])
	# Return the result
	return datadict
