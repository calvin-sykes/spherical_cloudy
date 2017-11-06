from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import elemids
import phionxsec

def func(x, a, b, c, d):
	return a * x**(b + c*np.log10(x))
	#return a * x**b +  c*np.sqrt(x)

#ions=["H I", "D I", "He I", "He II", "Li I", "Li II", "Li III"]
#colr=["r-", None, "g-", "g--", "b-", "b--", "b:"]

phelxsdata = np.loadtxt("data/phionxsec.dat")

# Get the element ID numbers
print phelxsdata.shape
nions = phelxsdata.shape[0]
engy = 10.0**np.linspace(1.0, 3.0, 1000)
for j in range(nions):
	if phelxsdata[j,0] == 1:
		abund = 1.0
		colr = 'r'
	elif phelxsdata[j,0] == 2:
		abund = 1.0/12.0
		colr = 'g'
	else:
		continue
		abund = 1.0E-4
		colr = 'b-'
	xsecv = phionxsec.rate_function_arr(engy, phelxsdata[j,2:])
	# Fit the x-sections with a polynomial
	ww = np.where((xsecv!=0.0) & (engy<200.0))
	# Note, use ionization energy of H I here
	HIengy = 13.6
	popt, pcov = curve_fit(func, engy[ww]/HIengy, xsecv[ww], p0=[1.09310810e-17,-2.0,-0.2, 1.0E-17])
	model = func(engy[ww]/HIengy, *popt)
	print popt
	# Plot it
	plt.subplot(211)
	plt.plot(np.log10(engy), np.log10(xsecv), colr+':')
	plt.plot(np.log10(engy[ww]), np.log10(model), colr+'-')
	#plt.plot(np.log10(engy), np.log10(abund*xsecv), colr)
	plt.subplot(212)
	plt.plot(np.log10(engy), np.exp(-abund*xsecv*1.0E18), colr)
	plt.plot(np.log10(engy), np.exp(-abund*xsecv*1.0E19), colr+'-')
	plt.plot(np.log10(engy), np.exp(-abund*xsecv*1.0E20), colr+'-')
xval = np.log10(6.626E-34*299792458.0/(584.0E-10*1.602E-19))
plt.plot([xval, xval], [0.0,1.0], 'b-')
plt.show()

HeIengy = 2.459E+01
print 6.626E-34*299792458.0/(HeIengy*1.0E-10*1.602E-19)