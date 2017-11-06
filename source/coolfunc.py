import numpy as np
from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import calc_Jnur

"""
Example of use:
coolcurve = make_coolingcurve(plot=True)
tstnH   = -1.0
tstnHI  = -1.5
temp, cool = eval_coolingcurve(coolcurve,tstnH,tstnHI)
plt.plot(np.log10(temp),np.log10(cool),'k-')
plt.show()
"""

def make_coolingcurve(plot=False):
	# Setup the grid of values
	nH,nHI,temp = np.mgrid[-2:0:3j,-6:3:10j,1000:21000:21j]

	arr_temp = np.linspace(1000.0,21000.0,21)
	arr_nH   = np.linspace(-2,0,3)
	arr_nHI  = np.linspace(-6,3,10)
	ntemp = arr_temp.size
	nnH = arr_nH.size

	if plot:
		values = range(ntemp)
		jet = cm = plt.get_cmap('jet') 
		cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
		scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

	# Read in the data for this grid and store in the data array
	data = 1.0E-26*np.ones_like(nH)
	for n in range(nnH):
		ntxt = "{0:+d}".format(int(arr_nH[n])).replace("+","p").replace("-","m")
		for t in range(ntemp):
			fname = "../cooling/cf_T{0:d}E3_D{1:s}.".format(int(arr_temp[t]/1000.0),ntxt)
			try:
				lrad_H,  lnHI,  lnHII  = np.loadtxt(fname+"dH",unpack=True,usecols=(0,1,2))
				wmin, wmax = np.argmin(lnHI), np.argmax(lnHI)
# 				diff = lnHI[1:]-lnHI[:-1]
# 				w = np.argwhere(diff>0.0)[0]
# 				wmin, wmax = 0, nHI.size
# 				if w.size != 0:
# 					if lnHI[1]-lnHI[0] < 0.0:
# 						wmin = w[0]
# 						if w.size > 1:
# 							wmax = w[1]
# 					else:
# 						wmax = w[0]
				#rad_He, nHeI, nHeII = np.loadtxt(fname+"dHe",unpack=True,usecols=(0,1,2))
				lrad_c, lheat, lcool = np.loadtxt(fname+"cf",unpack=True,usecols=(0,2,3))
			except:
				print "File not found: "+ fname
				continue
			asrt = np.argsort(lnHI[wmin:wmax],kind='mergesort')
			usecool = lcool[wmin:wmax]
			usenHI  = lnHI[wmin:wmax]
			coolint = np.interp(arr_nHI,usenHI[asrt],usecool[asrt])
			data[n,:,t] = coolint.copy()
			if plot and n==1:
				colorVal = scalarMap.to_rgba(values[t])
				plt.plot(np.log10(lnHI),np.log10(lcool),'k-')
				plt.plot(np.log10(usenHI[asrt]),np.log10(usecool[asrt]),color=colorVal)
	if plot: plt.show()

	# Reshapes the points into the required LinearNDInterpolator input
	coords = np.array((nH.ravel(),nHI.ravel(),temp.ravel())).T
	datar = data.ravel()

	coolfunc = LinearNDInterpolator(coords,datar)
	return coolfunc

def eval_coolingcurve(coolfunc,nH,nHI):
	temparr = np.linspace(1000.0,20000.0,1000)
	tstarray = np.zeros((temparr.size,3))
	tstarray[:,0] = nH
	tstarray[:,1] = nHI
	tstarray[:,2] = temparr
	return temparr, coolfunc(tstarray)

def thermal_equilibrium(heatfunc,coolcurves,nH,nHI,old_temp):
	lognH = np.log10(nH)
	lognHI = np.log10(nHI)
	npts = heatfunc.size
	prof_temperature = np.zeros(npts)
	for i in range(npts):
		temp, coolfunc = eval_coolingcurve(coolcurves,lognH[i],lognHI[i])
 		if i%100 == 0:
 			plt.clf()
 			plt.plot(temp,coolfunc,'k-')
 			plt.axhline(heatfunc[i],c='red')
 			plt.show()
		tval = calc_Jnur.thermal_equilibrium_coolfunc(temp, coolfunc, heatfunc[i], old_temp[i])
		prof_temperature[i] = tval
	return prof_temperature

#coolcurve = make_coolingcurve(plot=True)
