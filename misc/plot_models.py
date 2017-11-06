import numpy as np
from matplotlib import pyplot as plt
import calc_Jnur

hubble = 0.673
Omega_b = 0.02202/hubble**2
Omega_m = 0.315
Omega_l = 1.0-Omega_m

#  0: radius
#  1: electrondensity
#  2: density
#  3: prof_temperature
#  4: prof_pressure
#  5: prof_YHI
#  6: prof_YHeI
#  7: prof_YHeII
#  8: prof_HIcolumndensity
#  9: prof_HeIcolumndensity
# 10: prof_HeIIcolumndensity
# 11: phionrate_HI
# 12: phionrate_HeI
# 13: phionrate_HeII
# 14: scdryrate_HI
# 15: scdryrate_HeI
# 16: B1_HeII
# 17: B2_HeI
# 18: B3_HI
# 19: B6_HeI
# 20: B7_HI

def hubblepar(z):
	Ez = np.sqrt(Omega_l + Omega_m * ((1.0 + z) ** 3.0))
	return Ez * 100.0 * hubble

def cmin_prada(xv):
	return 3.681 + (5.033-3.681)*(0.5 + np.arctan(6.948*(xv-0.424))/np.pi)

def invsigmin_prada(xv):
	return 1.047 + (1.646-1.047)*(0.5 + np.arctan(7.386*(xv-0.526))/np.pi)

def massconc_Prada12(mvir,redshift=3,steps=100000):
	"""
	Prada et al. (2012), MNRAS, 423, 3018
	"""
	xval = ((Omega_l/Omega_m)**(1.0/3.0))/(1.0+redshift) # Eq 13
	yval = 1.0/(mvir/(1.0E12/hubble)) # Eq 23b
	xintg = calc_Jnur.massconc_xint(xval,steps)
	Dx = 2.5 * (Omega_m/Omega_l)**(1.0/3.0) * np.sqrt(1.0 + xval**3) * xintg /xval**1.5 # Eq 12
	Bzero = cmin_prada(xval)/cmin_prada(1.393) # Eq18a
	Bone  = invsigmin_prada(xval)/invsigmin_prada(1.393) # Eq 18b
	sigfunc = Dx * 16.9 * yval**0.41 / ( 1.0 + 1.102*(yval**0.20) + 6.22*(yval**0.333) ) # Eq 23a
	sigdash = Bone * sigfunc  # Eq 15
	Csigdash = 2.881 * (1.0 + (sigdash/1.257)**1.022) * np.exp(0.060 / sigdash**2)  # Eq 16
	conc = Bzero * Csigdash
	return conc

mn_mvir = 7.0
mx_mvir = 8.5
nummvir = 16

gastemp = 8000.0
redshift = 3.0
Gcons   = 6.67259E-8 # Gravitational constant in cgs
hztos   = 3.241E-20  # Conversion between km/s/Mpc to s
cmtopc  = 3.24E-19   # Conversion between cm and parsec
somtog  = 1.989E33   # Conversion between solar masses and grams
hubpar = hubblepar(redshift)
rhocrit = 3.0*(hubpar*hztos)**2/(8.0*np.pi*Gcons)




virialm = 10.0**np.linspace(mn_mvir,mx_mvir,nummvir)

for i in range(nummvir):
	mstring = ("{0:3.2f}".format(np.log10(virialm[i]))).replace(".","p").replace("+","")
	tstring = ("{0:3.2E}".format(gastemp)).replace(".","p").replace("+","")
	rstring = ("{0:3.2f}".format(redshift)).replace(".","p").replace("+","")
	infname = "data/Burkert_model_mass{0:s}_temp{1:s}_redshift{2:s}.npy".format(mstring,tstring,rstring)
	data = np.load(infname)
	radius, prof_HIcd = data[:,0], data[:,8]
	cvir = massconc_Prada12(virialm[i],redshift=redshift)
	virialr = (virialm[i]*somtog/(4.0*np.pi*200.0*rhocrit/3.0))**(1.0/3.0)
	rscale = virialr/cvir
	rhods = 200.0 * rhocrit * cvir**3 / calc_Jnur.Burkert_fm(cvir)
	Mds = 4.0*np.pi*rhods*(rscale**3.0)/(3.0*somtog)
	Mth = Mds * calc_Jnur.Burkert_fm(300.0/(cmtopc*rscale))
	xval = 0.0
	print "Virial mass = {0:E}, M300 = {1:E}".format(virialm[i],Mth)
	if i == 10:
		print np.log10(virialm[i])
		plt.plot(radius,prof_HIcd,'k-')
	else:
		plt.plot(radius,prof_HIcd,'r-')
plt.show()