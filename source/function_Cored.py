import numpy as np
from matplotlib import pyplot as plt
import calc_Jnur
import time
import sys
from multiprocessing import Pool as mpPool
from multiprocessing.pool import ApplyResult

hubble = 0.673
Omega_b = 0.02202/hubble**2
Omega_m = 0.315
Omega_l = 1.0-Omega_m

"""
METHOD:

Start with a uniform temperature of 10^4K everywhere

For a given DM halo profile, calculate the pressure profile,
with a boundary condition specifying the pressure of the IGM.

Assume the gas is initially fully ionized with Y(HI) = Y(HeI) = Y(HeII) = 0

Calculate the gas density profile, given that n(H) = P/(kT * 7/3)

Calculate the photoionization rates for H I, He I, and He II.

For the electron number density and gas temperature calculate the Ri.

Thereby calculate the new Y(HI), Y(HeI), Y(HeII)

Using the pressure profile, density profile and Y-values,
calculate the temperature profile.

Using this new temperature profile calculate the pressure profile
for the specified DM distribution.

For the pressure profile, temperature profile and Y-values,
recalculate the density profile, subject to an external pressure.

Using the density profile and the Y-values, calculate the
H I, He I, and He II volume density profiles.

Recompute the photoionization rates for H I, He I, and He II.

and so forth...
"""

def hubblepar(z):
	Ez = np.sqrt(Omega_l + Omega_m * ((1.0 + z) ** 3.0))
	return Ez * 100.0 * hubble

def HMbackground_z0_sternberg(nu=None,maxvu=200.0,num=10000):
	Jnu0 = 2.0E-23
	nu0 = 3.287E15
	if nu is None:
		nu = np.linspace(5.0E-2,maxvu,num)
		J = Jnu0*np.ones(num)
	else:
		nu /= nu0
		J = Jnu0*np.ones(nu.size)
	w = np.where(nu>4.0)
	J[w] *= 2.512E-2 * nu[w]**-0.46
	w = np.where((nu>=1.0) & (nu<=4.0))
	J[w] *= nu[w]**-3.13
	w = np.where((nu>=0.3) & (nu<1.0))
	J[w] *= nu[w]**-5.41
	w = np.where(nu<0.3)
	J[w] *= 1.051E2 * nu[w]**-1.5
	return J, nu*nu0

def HMbackground_z0():
	waveAt, Jnut = np.loadtxt("HM12_UVB.dat",usecols=(0,1),unpack=True)
	waveA = waveAt[1:]*1.0E-10
	Jnu = Jnut[1:]
	nu = 299792458.0/waveA
	return Jnu[::-1], nu[::-1]

def HMbackground_z3():
	waveAt, Jnut = np.loadtxt("HM12_UVB.dat",usecols=(0,29),unpack=True)
	waveA = waveAt[1:]*1.0E-10
	Jnu = Jnut[1:]
	nu = 299792458.0/waveA
	return Jnu[::-1], nu[::-1]

def HMbackground(redshift=3.0):
	data = np.loadtxt("HM12_UVB.dat")
	rdshlist = data[0,:]
	amin = np.argmin(np.abs(rdshlist-redshift))
	waveAt, Jnut = data[1:,0], data[1:,amin]
	waveA = waveAt[1:]*1.0E-10
	#w = np.where(waveAt < 912.0)
	Jnu = Jnut[1:]
	nu = 299792458.0/waveA
	#plt.plot(nu[::-1][w], nu[::-1][w]*Jnu[::-1][w], 'k-')
	#waveAt, Jnut = np.loadtxt("HM12_UVB.dat",usecols=(0,29),unpack=True)
	#waveAtmp = waveAt[1:]*1.0E-10
	#Jnutmp = Jnut[1:]
	#nutmp = 299792458.0/waveAtmp
	#plt.plot(nutmp[::-1][w], nutmp[::-1][w]*Jnutmp[::-1][w], 'r--')
	#plt.show()
	#plt.clf()
	return Jnu[::-1], nu[::-1]

def massconc_Klypin11(mvir,redshift=3):
	"""
	This is not for M200 and r200 --- use the Prada 2012 implementation
	"""
	sys.exit()
	rdshft = np.array([0.5,1.0,2.0,3.0,5.0])
	czero  = np.array([7.08,5.45,3.67,2.83,2.34])
	mzero  = np.array([1.5E17,2.5E15,6.8E13,6.3E12,6.6E11])
	# Redshift 3 relation:
	conc = 2.83 * (mvir/(1.0E12/hubble))**-0.075 * (1.0 + (mvir/(6.3E12/hubble))**0.26)
	return conc

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

def phionxsec_HI(engy):
	"""
	Photoionization cross-sections by Verner et al. (1996) ApJ, 465, 487
	"""
	xsec = np.zeros(engy.size)
	Et = 1.360E+1
	Eo = 4.298E-1
	so = 5.475E+4
	ya = 3.288E+1
	P  = 2.963E+0
	yw = 0.0
	yo = 0.0
	y1 = 0.0
	x = engy/Eo - yo
	y = np.sqrt(x**2 + y1**2)
	Fy = ((x-1.0)**2 + yw**2) * y**(0.5*P - 5.5) * (1.0 + np.sqrt(y/ya))**(-1.0*P)
	w = np.where(engy>=Et)
	xsec[w] = 1.0E-18 * so * Fy[w]
	return xsec

def phionxsec_HeII(engy):
	"""
	Photoionization cross-sections by Verner et al. (1996) ApJ, 465, 487
	"""
	xsec = np.zeros(engy.size)
	Et = 5.442E+1
	Eo = 1.720E+0
	so = 1.369E+4
	ya = 3.288E+1
	P  = 2.963E+0
	yw = 0.0
	yo = 0.0
	y1 = 0.0
	x = engy/Eo - yo
	y = np.sqrt(x**2 + y1**2)
	Fy = ((x-1.0)**2 + yw**2) * y**(0.5*P - 5.5) * (1.0 + np.sqrt(y/ya))**(-1.0*P)
	w = np.where(engy>=Et)
	xsec[w] = 1.0E-18 * so * Fy[w]
	return xsec

def phionxsec_HeI(engy):
	"""
	Photoionization cross-sections by Verner et al. (1996) ApJ, 465, 487
	"""
	xsec = np.zeros(engy.size)
	Et = 2.459E+1
	Eo = 1.361E+1
	so = 9.492E+2
	ya = 1.469E+0
	P  = 3.188E+0
	yw = 2.039E+0
	yo = 4.434E-1
	y1 = 2.136E+0
	x = engy/Eo - yo
	y = np.sqrt(x**2 + y1**2)
	Fy = ((x-1.0)**2 + yw**2) * y**(0.5*P - 5.5) * (1.0 + np.sqrt(y/ya))**(-1.0*P)
	w = np.where(engy>=Et)
	xsec[w] = 1.0E-18 * so * Fy[w]
	return xsec


# Recombination coefficients
## H I
def alphaB_HI(temper):
	"""
	Equation 14.6 from Draine book
	"""
	#return 3.403E-10 * temper**-0.7827
	return 2.54E-13 * (temper/1.0E4)**(-0.8163 - 0.0208*np.log(temper/1.0E4))

## He I
def alphaB_HeI(temper):
	"""
	A fit to the data in Hummer & Storey (1998)
	"""
	#return 1.613E-10 * temper**-0.6872
	return 9.03E-14 * (temper/4.0E4)**(-0.830-0.0177*np.log(temper/4.0E4))

def alpha1s_HeI(temper):
	"""
	Equation B2 from Jenkins (2013)
	"""
	return 1.54E-13 * (temper/1.0E4)**(-0.486)

## He II
def alphaB_HeII(temper):
	#return 1.395E-09 * temper**-0.7446
	"""
	Equation 14.6 from Draine book
	"""
	#return 5.18E-13 * (temper/4.0E4)**(-0.833 - 0.035*np.log(temper/4.0E4))
	return 5.08E-13 * (temper/4.0E4)**(-0.8163 - 0.0208*np.log(temper/4.0E4))

def alpha1s_HeII(temper):
	"""
	Equation B2 from Jenkins (2013)
	"""
	return 3.16E-13 * (temper/4.0E4)**(-0.540 - 0.017*np.log(temper/4.0E4))

def alpha2p_HeII(temper):
	"""
	Equation B4 from Jenkins (2013)
	"""
	return 1.07E-13 * (temper/4.0E4)**(-0.681 - 0.061*np.log(temper/4.0E4))

def alphaeff2s_HeII(temper):
	"""
	Equation B6 from Jenkins (2013)
	"""
	return 1.68E-13 * (temper/4.0E4)**(-0.7205-0.0081*np.log(temper/4.0E4))

def alpha2s_HeII(temper):
	"""
	Equation B8 from Jenkins (2013)
	"""
	return 4.68E-14 * (temper/4.0E4)**(-0.537-0.019*np.log(temper/4.0E4))


# Sharing function
def yfactor(ion, xsec_HI, xsec_HeI, xsec_HeII, prof_HI, prof_HeI, prof_HeII, engyval, energy):
	"""
	Sharing function - defined by Eq 22 from Jenkins (2013)
	"""
	xsecrad_HI   = calc_Jnur.calc_xsec_energy(xsec_HI, engyval, energy)
	xsecrad_HeI  = calc_Jnur.calc_xsec_energy(xsec_HeI, engyval, energy)
	xsecrad_HeII = calc_Jnur.calc_xsec_energy(xsec_HeII, engyval, energy)
	#wHeII = np.where(energy<54.41778)
	#xsecrad_HeII[wHeII] = 0.0
	#print "Ion = ", ion
	#plt.plot(np.log10(energy),xsecrad_HeII,'r-')
	#plt.plot(np.log10(engyval),xsec_HeII,'g-')
	#plt.plot(np.arange(xsecrad_HeI.size),xsecrad_HeI,'g-')
	#plt.plot(np.arange(xsecrad_HeII.size),xsecrad_HeII,'b-')
	#plt.show()
	#plt.clf()
	output = np.zeros(prof_HI.size)
	divarr = (prof_HI*xsecrad_HI + prof_HeI*xsecrad_HeI + prof_HeII*xsecrad_HeII)
	w = np.where(divarr != 0.0)
	if ion == "HI":
		output = (prof_HI*xsecrad_HI)[w] / divarr[w]
	elif ion == "HeI":
		output = (prof_HeI*xsecrad_HeI)[w] / divarr[w]
	elif ion == "HeII":
		output = (prof_HeII*xsecrad_HeII)[w] / divarr[w]
	return output


def mpcoldens(j, prof, radius, nummu):
	coldens, muarr = calc_Jnur.calc_coldens(prof, radius, nummu)
	return [j,coldens,muarr]

def mpphion(j, jnurarr, phelxs, nuzero):
	phionxsec = 4.0*np.pi * calc_Jnur.phionrate(jnurarr, phelxs, nuzero)
	return [j,phionxsec]



def get_halo(virialm,redshift,gastemp,acore=0.5):
	"""
	acore is the ratio of the core radius to the scale radius
	"""
	# Begin the timer
	timeA = time.time()

	# Get the background radiation field
	#nu0 = 3.287E15
	#J, nu = HMbackground_z0()
	#w = np.where(nu>=nu0)
	#temp = J[w]/(6.626E-27*nu[w])
	#Jstar = 0.5*(temp[1:]+temp[:-1]) * (nu[w][1:]-nu[w][:-1])
	#print "J* = {0:f}".format(4.0*np.pi*np.sum(Jstar))
	#JHM, nuHM = HMbackground_z3()

	#jzero, nuzero = HMbackground_z0(nu=engy*1.602E-19/6.626E-34)
	#jzerot, nuzerot = HMbackground_z3()

	####
	# Sternberg et al. (2002) Figure 3.
	#
	#plt.plot(nuzero,nuzero*jzero,'k-')
	#plt.plot(nuzerot,nuzerot*jzerot,'r--') # Also compare this with the HM background
	#plt.yscale('log')
	#plt.xscale('log')
	#plt.show()
	#plt.clf()
	#sys.exit()

	# Get the photoelectric cross-sections
	#engy = np.linspace(13.6,5.0E4,100000)
	#engy = 10.0**(np.linspace(np.log10(13.6),np.log10(5.0E4),1000))
	#engy = 6.626E-34 * nuzero / 1.602E-19
	#engy = np.append(np.linspace(13.6,200.0,500),np.linspace(200.0,5.0E4,500)[1:])
	#phelxs_HI = phionxsec_HI(engy)
	#phelxs_HeI = phionxsec_HeI(engy)
	#phelxs_HeII = phionxsec_HeII(engy)
	####
	# Osterbrock (1989) Fig 2.2
	#
	#plt.plot(engy*1.602E-19/6.626E-34,phelxs_HI*1.0E18,'k-')
	#plt.plot(engy*1.602E-19/6.626E-34,phelxs_HeI*1.0E18,'k-')
	#plt.plot(engy*1.602E-19/6.626E-34,phelxs_HeII*1.0E18,'k-')
	#plt.show()
	#plt.clf()

	# Set some numerical aspects of the simulation
	miniter = 10         # Minimum number of iterations to perform
	maxiter = 100000     # Maximum number of iterations to perform
	npts = 500           # Number of radial points to consider
	nummu = 360          # Number of numerical integration points when integrating over cos(theta)
	concrit = 1.0E-5     # Convergence criteria for deriving Y-values
	kB      = 1.38E-16   # Boltmann constant in units of erg/K
	cmtopc  = 3.24E-19   # Conversion between cm and parsec
	somtog  = 1.989E33   # Conversion between solar masses and grams
	Gcons   = 6.67259E-8 # Gravitational constant in cgs
	hztos   = 3.241E-20  # Conversion between km/s/Mpc to s
	prim_He = 1.0/12.0   # Primordial number abundance of He relative to H
	ncpus   = 3          # Number of CPUs to use
	#gastemp = 8.0E3      # Fix the gas temperature

	# Set the mass of the halo
	#virialm = 1.0E8 # Virial mass of the halo (in solar masses) -- this is M200
	#redshift = 3.0
	# Get the concentration of the halo
	print "Calculating halo concentration"
	cvir = massconc_Prada12(virialm, redshift=redshift)#massconc_Klypin11(mvir,redshift=3)

	# Get the Haardt & Madau (2012) background at the appropriate redshift
	jzero, nuzero = HMbackground(redshift=redshift)

	# Load the cooling function
	#logT, coolfunc = np.loadtxt("mzero.cie",unpack=True,usecols=(0,4))
	#plt.plot(logT,coolfunc,'k-')
	#plt.show()
	#sys.exit()

	# Get the photoelectric cross-sections
	#engy = np.linspace(13.6,5.0E4,100000)
	#engy = 10.0**(np.linspace(np.log10(13.6),np.log10(5.0E4),1000))
	engy = 6.626E-34 * nuzero / 1.602E-19
	#engy = np.append(np.linspace(13.6,200.0,500),np.linspace(200.0,5.0E4,500)[1:])
	phelxs_HI = phionxsec_HI(engy)
	phelxs_HeI = phionxsec_HeI(engy)
	phelxs_HeII = phionxsec_HeII(engy)

	hubpar = hubblepar(redshift)
	rhocrit = 3.0*(hubpar*hztos)**2/(8.0*np.pi*Gcons)
	virialr = (virialm*somtog/(4.0*np.pi*200.0*rhocrit/3.0))**(1.0/3.0)
	rscale = virialr/cvir
	rhods = 200.0 * rhocrit * cvir**3 / calc_Jnur.Cored_fm(cvir, acore)
	print "Virial radius (pc) = {0:E}".format(virialr*cmtopc)
	print "Scale radius (pc) = {0:E}".format(rscale*cmtopc)

	loadprev=True
	if loadprev and virialm != 1.0E7:
		mstring = ("{0:3.2f}".format(np.log10(virialm)-0.1)).replace(".","p").replace("+","")
		rstring = ("{0:3.2f}".format(redshift)).replace(".","p").replace("+","")
		infname = "data/Cored_model_mass{0:s}_redshift{1:s}.npy".format(mstring,rstring)
		print "Loading file {0:s}".format(infname)
		#print "ERROR: file not saved"
		tdata = np.load(infname)
		radius = tdata[:,0]
		temp_densitynH = tdata[:,2]
		prof_temperature = tdata[:,3]
		prof_YHI   = tdata[:,5]
		prof_YHeI  = tdata[:,6]
		prof_YHeII = tdata[:,7]
		densitym  = temp_densitynH * 1.673E-24 * (1.0 + 4.0*prim_He)
	else:
		# Set the gas conditions
		radius  = np.linspace(0.0,virialr,npts)
		#radius  = np.linspace(0.0,rscale,npts)
		temp_densitynH = np.ones(npts)
		prof_temperature = gastemp * np.ones(npts)
		prof_YHI   = 0.50*np.ones(npts)
		prof_YHeI  = 0.35*np.ones(npts)
		prof_YHeII = 0.35*np.ones(npts)
		densitym = 1.673E-24 * (1.0 + 4.0*prim_He) * np.ones(npts)

	## Set the gas conditions
	#radius  = np.linspace(0.0,virialr,npts)
	##radius  = np.linspace(0.0,rscale,npts)
	#prof_temperature = gastemp * np.ones(npts)
	#prof_YHI   = 0.50*np.ones(npts)
	#prof_YHeI  = 0.35*np.ones(npts)
	#prof_YHeII = 0.35*np.ones(npts)
	#temp_densitynH = np.ones(npts)
	"""
	rintegral = calc_Jnur.mass_integral(temp_densitynH,radius)
	cen_density = virialm*somtog * Omega_b / ( (Omega_m-Omega_b) * 4.0 * np.pi * 1.673E-24 * (1.0 + 4.0*prim_He) * rintegral)
	densitynH = cen_density * temp_densitynH
	densitym = densitynH * 1.673E-24 * (1.0 + 4.0*prim_He)
	density = 0.1*(radius/radius[npts/2])**2
	"""
	barymass = virialm*somtog * Omega_b / (Omega_m-Omega_b)
	#densitym = 1.673E-24 * (1.0 + 4.0*prim_He) * np.ones(npts)

	# Set the stopping criteria flags
	critHI, critHeI, critHeII = True, True, True
	iteration = 0
	tstHI, tstHeI, tstHeII = 0, 0, 0
	answer = -1
	while (critHI or critHeI or critHeII) or (iteration <= miniter):
		critHI, critHeI, critHeII = True, True, True
		iteration += 1
		print "          Iteration number: {0:d}   <--[  {1:d}/{2:d} {3:d}/{2:d} {4:d}/{2:d}  ]-->".format(iteration,tstHI,npts,tstHeI,tstHeII)
	
		# Calculate the pressure profile
		dof = (2.0-prof_YHI) + prim_He*(3.0 - 2.0*prof_YHeI - 1.0*prof_YHeII)
		masspp = (1.0 + 4.0*prim_He)/dof
		fgas = calc_Jnur.fgasx(densitym,radius,rscale)
		"""
		densitym = calc_Jnur.HEcalc_Cored(radius/rscale,prof_temperature,masspp,rhods,rscale,0.01,densitym[0],barymass)
		xarray = radius/rscale
		lhs = 2.0 * (densitym[1:]-densitym[:-1])/((densitym[1:]+densitym[:-1])*(xarray[1:]-xarray[:-1]))
		cssq = kB*(prof_temperature[1:]+prof_temperature[:-1])/(1.673E-24*(masspp[1:]+masspp[:-1]))
		vssq = 2.795E-7 * rhods * rscale * rscale
		vgas = 8.385E-7*rscale*rscale
		fmxt = 3.0*(np.log(1.0+xarray) - xarray/(1.0+xarray))/xarray**2
		fmx = 0.5*(fmxt[1:]+fmxt[:-1])
		rfunc = densitym*xarray**2
		irho = 4.0*np.cumsum(0.5*(rfunc[1:]+rfunc[:-1])*(xarray[1:]-xarray[:-1]))/(xarray[1:]+xarray[:-1])**2
		rhs = (-1.0/cssq) * (vssq*fmx + vgas*irho)
		plt.plot(0.5*(xarray[1:]+xarray[:-1]),lhs,'k-')
		plt.plot(0.5*(xarray[1:]+xarray[:-1]),rhs,'r--')
		plt.show()
		sys.exit()
		#plt.plot(np.log10(radius),np.log10(densitym),'k-')
		#plt.show()
		#plt.clf()
		densitynH = densitym / (1.673E-24 * (1.0 + 4.0*prim_He))
		prof_pressure = dof * densitynH * kB * prof_temperature
		#sys.exit()
		"""
#		prof_pressure = calc_Jnur.pressure_Coredgas(prof_temperature, fgas, radius, masspp, rhods, rscale, acore)
		prof_pressure = calc_Jnur.pressure_Cored(prof_temperature, radius, masspp, rhods, rscale, acore)
		#############
		# TEST PLOT #
		#############
		#print "pressure!"
		#plt.plot(np.log10(radius*cmtopc),prof_pressure,'k-')
		#plt.show()
		#plt.clf()
		#############
		#############
#		prof_pressure = ext_pressure * kB * temp_pressure/temp_pressure[-1]
#		prof_pressure = cen_pressure * kB * temp_pressure
		# Calculate gas density profile
		temp_densitynH = prof_pressure / (kB * prof_temperature * dof)
		temp_densitynH /= temp_densitynH[0]
		rintegral = calc_Jnur.mass_integral(temp_densitynH,radius)
		#############
		# TEST PLOT #
		#############
		#print "mass integral!"
		#plt.plot(np.log10(radius*cmtopc),prof_pressure,'k-')
		#plt.show()
		#plt.clf()
		#############
		#############
		cen_density = barymass/ (4.0 * np.pi * 1.673E-24 * (1.0 + 4.0*prim_He) * rintegral)
		densitynH = cen_density * temp_densitynH
		densitym  = densitynH * 1.673E-24 * (1.0 + 4.0*prim_He)
		# Update pressure now that the central pressure is defined
		prof_pressure = dof * densitynH * kB * prof_temperature

		# and the volume density of the unionized species
		prof_HI   = prof_YHI * densitynH
		prof_HeI  = prof_YHeI * densitynH * prim_He
		prof_HeII = prof_YHeII * densitynH * prim_He

		# Compute the electron density
		electrondensity = densitynH * ( (1.0-prof_YHI) + prim_He*prof_YHeII + 2.0*prim_He*(1.0-prof_YHeI-prof_YHeII) )

		#plt.plot(radius*cmtopc,electrondensity,'k-')
		#plt.show()
		#plt.clf()

		# Calculate the column density arrays,
		if ncpus == 1:
			coldensHI, muarr = calc_Jnur.calc_coldens(prof_HI, radius, nummu)
			coldensHeI, muarr = calc_Jnur.calc_coldens(prof_HeI, radius, nummu)
			coldensHeII, muarr = calc_Jnur.calc_coldens(prof_HeII, radius, nummu)
		else:
			pool = mpPool(processes=ncpus)
			async_results = []
			for j in range(ncpus):
				if j == 0:
					# H I column density calculation
					async_results.append(pool.apply_async(mpcoldens, (j,prof_HI, radius, nummu)))
				elif j == 1:
					# He I column density calculation
					async_results.append(pool.apply_async(mpcoldens, (j,prof_HeI, radius, nummu)))
				elif j == 2:
					# He II column density calculation
					async_results.append(pool.apply_async(mpcoldens, (j,prof_HeII, radius, nummu)))
			pool.close()
			pool.join()
			map(ApplyResult.wait, async_results)
			for j in range(ncpus):
				getVal = async_results[j].get()
				if getVal[0] == 0:
					coldensHI, muarr = getVal[1], getVal[2]
				elif getVal[0] == 1:
					coldensHeI, muarr = getVal[1], getVal[2]
				elif getVal[0] == 2:
					coldensHeII, muarr = getVal[1], getVal[2]

		# integrate over all angles,
		jnurarr = calc_Jnur.nint_costheta(coldensHI, phelxs_HI, coldensHeI, phelxs_HeI, coldensHeII, phelxs_HeII, muarr, jzero)
		#############
		# TEST PLOT #
		#############
		#print "integrate over cos theta!"
		#plt.plot(np.log10(radius*cmtopc),jnurarr[100,:],'k-')
		#plt.plot(np.log10(radius*cmtopc),jnurarr[250,:],'r-')
		#plt.plot(np.log10(radius*cmtopc),jnurarr[400,:],'g-')
		#plt.plot(np.log10(radius*cmtopc),jnurarr[500,:],'b-')
		#plt.show()
		#plt.clf()
		#############
		#############
		# and calculate the photoionization rates
		print "Calculating phionization rates"
		if ncpus == 1:
			phionrate_HI = 4.0*np.pi * calc_Jnur.phionrate(jnurarr, phelxs_HI, nuzero)
			phionrate_HeI = 4.0*np.pi * calc_Jnur.phionrate(jnurarr, phelxs_HeI, nuzero)
			phionrate_HeII = 4.0*np.pi * calc_Jnur.phionrate(jnurarr, phelxs_HeII, nuzero)
		else:
			pool = mpPool(processes=ncpus)
			async_results = []
			for j in range(ncpus):
				if j == 0:
					# H I photoionization rate
					async_results.append(pool.apply_async(mpphion, (j, jnurarr, phelxs_HI, nuzero)))
				elif j == 1:
					# He I photoionization rate
					async_results.append(pool.apply_async(mpphion, (j, jnurarr, phelxs_HeI, nuzero)))
				elif j == 2:
					# He II photoionization rate
					async_results.append(pool.apply_async(mpphion, (j, jnurarr, phelxs_HeII, nuzero)))
			pool.close()
			pool.join()
			map(ApplyResult.wait, async_results)
			for j in range(ncpus):
				getVal = async_results[j].get()
				if getVal[0] == 0:
					phionrate_HI = getVal[1]
				elif getVal[0] == 1:
					phionrate_HeI = getVal[1]
				elif getVal[0] == 2:
					phionrate_HeII = getVal[1]

		#############
		# TEST PLOT #
		#############
		#print "phionrate!"
		#plt.plot(np.arange(npts),phionrate_HI,'r-')
		#plt.plot(np.arange(npts),phionrate_HeI,'g-')
		#plt.plot(np.arange(npts),phionrate_HeII,'b-')
		#plt.show()
		#plt.clf()
		#############
		#############
		# the secondary photoelectron collisional ionization rates
		scdryrate_HI = 4.0*np.pi * calc_Jnur.scdryrate(jnurarr, nuzero, phelxs_HI, phelxs_HeI, phelxs_HeII, prof_HI, prof_HeI, prof_HeII, electrondensity/(densitynH*(1.0 + 2.0*prim_He)), 0)
		scdryrate_HeI = 4.0*np.pi * 10.0 * calc_Jnur.scdryrate(jnurarr, nuzero, phelxs_HI, phelxs_HeI, phelxs_HeII, prof_HI, prof_HeI, prof_HeII, electrondensity/(densitynH*(1.0 + 2.0*prim_He)), 1)
		#############
		# TEST PLOT #
		#############
		#print "scdryrate!"
		#plt.plot(np.arange(npts),scdryrate_HI,'r-')
		#plt.plot(np.arange(npts),scdryrate_HeI,'g-')
		#plt.show()
		#plt.clf()
		#############
		#############

		# and finally some photoionization from He and H recombinations, listed in Appendix B of Jenkins (2013)
		B1_HeII = np.zeros(npts)
		B2_HeI  = np.zeros(npts)
		B3_HI   = np.zeros(npts)
		B6_HeI  = np.zeros(npts)
		B7_HI   = np.zeros(npts)
		w_HI    = np.where(prof_HI!=0.0)
		w_HeI   = np.where(prof_HeI!=0.0)
		w_HeII  = np.where(prof_HeII!=0.0)
		B1_HeII[w_HeII] = (((1.0-prof_YHeI-prof_YHeII)*densitynH*prim_He*electrondensity) * alpha1s_HeII(prof_temperature) * yfactor("HeII", phelxs_HI, phelxs_HeI, phelxs_HeII, prof_HI, prof_HeI, prof_HeII, engy, 54.41778+kB*prof_temperature/1.602E-12))[w_HeII]/prof_HeII[w_HeII]
		trma = alpha1s_HeII(prof_temperature) * yfactor("HeI", phelxs_HI, phelxs_HeI, phelxs_HeII, prof_HI, prof_HeI, prof_HeII, engy, 54.41778+kB*prof_temperature/1.602E-12)
		trmb = alpha2p_HeII(prof_temperature) * yfactor("HeI", phelxs_HI, phelxs_HeI, phelxs_HeII, prof_HI, prof_HeI, prof_HeII, engy, 40.8*np.ones(npts))
		trmc = (alphaB_HeII(prof_temperature) - alpha2p_HeII(prof_temperature) - alphaeff2s_HeII(prof_temperature)) * yfactor("HeI", phelxs_HI, phelxs_HeI, phelxs_HeII, prof_HI, prof_HeI, prof_HeII, engy, 50.0*np.ones(npts))
		B2_HeI[w_HeI]  = (((1.0-prof_YHeI-prof_YHeII)*densitynH*prim_He)*electrondensity * ( trma + trmb + trmc ))[w_HeI] / prof_HeI[w_HeI]
		trma = alpha1s_HeII(prof_temperature) * yfactor("HI", phelxs_HI, phelxs_HeI, phelxs_HeII, prof_HI, prof_HeI, prof_HeII, engy, 54.41778+kB*prof_temperature/1.602E-12)
		trmb = alpha2p_HeII(prof_temperature) * (1.0 + yfactor("HI", phelxs_HI, phelxs_HeI, phelxs_HeII, prof_HI, prof_HeI, prof_HeII, engy, 40.8*np.ones(npts)))
		trmc = 2.42*alpha2s_HeII(prof_temperature)
		trmd = (alphaB_HeII(prof_temperature) - alpha2p_HeII(prof_temperature) - alphaeff2s_HeII(prof_temperature)) * yfactor("HI", phelxs_HI, phelxs_HeI, phelxs_HeII, prof_HI, prof_HeI, prof_HeII, engy, 50.0*np.ones(npts))
		trme = 1.42*(alphaeff2s_HeII(prof_temperature) - alpha2s_HeII(prof_temperature))
		B3_HI[w_HI]   = (((1.0-prof_YHeI-prof_YHeII)*densitynH*prim_He)*electrondensity * ( trma + trmb + trmc + trmd + trme ))[w_HI]/prof_HI[w_HI]
		B6_HeI[w_HeI]  = (prof_HeII*electrondensity * alpha1s_HeII(prof_temperature) * yfactor("HeI", phelxs_HI, phelxs_HeI, phelxs_HeII, prof_HI, prof_HeI, prof_HeII, engy, 24.58741+kB*prof_temperature/1.602E-12))[w_HeI]/prof_HeI[w_HeI]
		trma = alpha1s_HeI(prof_temperature) * yfactor("HI", phelxs_HI, phelxs_HeI, phelxs_HeII, prof_HI, prof_HeI, prof_HeII, engy, 24.58741+kB*prof_temperature/1.602E-12)
		B7_HI[w_HI]   = (prof_HeII*electrondensity * ( trma + 0.96*alphaB_HeI(prof_temperature) ))[w_HI]/prof_HI[w_HI]

		#plt.plot(radius*cmtopc,phionrate_HI,'r-')
		#plt.plot(radius*cmtopc,phionrate_HeI,'g-')
		#plt.plot(radius*cmtopc,phionrate_HeII,'b-')
		#plt.plot(radius*cmtopc,scdryrate_HI,'r-')
		#plt.plot(radius*cmtopc,scdryrate_HeI,'g-')
		#plt.plot(radius*cmtopc,B1_HeII,'k--')
		#plt.plot(radius*cmtopc,B2_HeI,'g--')
		#plt.plot(radius*cmtopc,B3_HI,'r--')
		#plt.plot(radius*cmtopc,B6_HeI,'g:')
		#plt.plot(radius*cmtopc,B7_HI,'r:')
		#plt.show()
		#plt.clf()

		#print "Compare the fits for He I from Benjamin et al. 1999 and Hummer & Storey 1998"
		#sys.exit()

		gamma_HI   = phionrate_HI   + scdryrate_HI  + B3_HI  + B7_HI
		gamma_HeI  = phionrate_HeI  + scdryrate_HeI + B2_HeI + B6_HeI
		gamma_HeII = phionrate_HeII + B1_HeII

		print "THE DISCONTINUITY IS INTRODUCED IN THE SHARING FUNCTION!"

		#############
		# TEST PLOT #
		#############
		#print "B rates!"
		#plt.plot(np.arange(npts),B7_HI,'r-') # Not in this rate -- It's in rates 1, 2, 3, 6
		#plt.plot(np.arange(npts),B3_HI,'r--')
		#plt.plot(np.arange(npts),B2_HeI,'g-')
		#plt.plot(np.arange(npts),B6_HeI,'g-')
		#plt.plot(np.arange(npts),B1_HeII,'b-')
		#plt.show()
		#plt.clf()
		#############
		#############

		rate_HI   = gamma_HI / (electrondensity * alphaB_HI(prof_temperature))
		rate_HeI  = gamma_HeI / (electrondensity * alphaB_HeI(prof_temperature))
		rate_HeII = gamma_HeII / (electrondensity * alphaB_HeII(prof_temperature))
		#############
		# TEST PLOT #
		#############
		#print "rates before!"
		#plt.plot(np.arange(npts),rate_HI,'r-')
		#plt.plot(np.arange(npts),rate_HeI,'g-')
		#plt.plot(np.arange(npts),rate_HeII,'b-')
		#plt.show()
		#plt.clf()
		#############
		#############
		# Store old and obtain new values for YHI, YHeI, YHeII
		old_prof_YHI   = prof_YHI.copy()
		old_prof_YHeI  = prof_YHeI.copy()
		old_prof_YHeII = prof_YHeII.copy()
		tmp_prof_YHI   = prof_YHI.copy()
		tmp_prof_YHeI  = prof_YHeI.copy()
		tmp_prof_YHeII = prof_YHeII.copy()
		inneriter = 0
		while True:
			prof_YHI   = 1.0/(1.0+rate_HI)
			prof_YHeII = 1.0/(1.0+rate_HeII+(1.0/rate_HeI))
			prof_YHeI  = prof_YHeII/rate_HeI
			tstHI   = (np.abs(tmp_prof_YHI-prof_YHI)<concrit).astype(np.int).sum()
			tstHeI  = (np.abs(tmp_prof_YHeI-prof_YHeI)<concrit).astype(np.int).sum()
			tstHeII = (np.abs(tmp_prof_YHeII-prof_YHeII)<concrit).astype(np.int).sum()
			# Reset ne and the rates
			electrondensity = densitynH * ( (1.0-prof_YHI) + prim_He*prof_YHeII + 2.0*prim_He*(1.0-prof_YHeI-prof_YHeII) )
			rate_HI   = gamma_HI / (electrondensity * alphaB_HI(prof_temperature))
			rate_HeI  = gamma_HeI / (electrondensity * alphaB_HeI(prof_temperature))
			rate_HeII = gamma_HeII / (electrondensity * alphaB_HeII(prof_temperature))
			inneriter += 1
			#print "   Rates Iteration {0:d}   <--[  {1:d}/{2:d} {3:d}/{2:d} {4:d}/{2:d}  ]-->".format(inneriter,tstHI,npts,tstHeI,tstHeII)
			if tstHI == npts and tstHeI == npts and tstHeII == npts:
				break
			elif inneriter > 1000:
				print "Break inner loop at 1000 iterations, STATUS:"
				print "   Rates Iteration {0:d}   <--[  {1:d}/{2:d} {3:d}/{2:d} {4:d}/{2:d}  ]-->".format(inneriter,tstHI,npts,tstHeI,tstHeII)
				break
			tmp_prof_YHI   = prof_YHI.copy()
			tmp_prof_YHeI  = prof_YHeI.copy()
			tmp_prof_YHeII = prof_YHeII.copy()

		#############
		# TEST PLOT #
		#############
		#print "rates after!"
		#plt.plot(np.arange(npts),rate_HI,'r-')
		#plt.plot(np.arange(npts),rate_HeI,'g-')
		#plt.plot(np.arange(npts),rate_HeII,'b-')
		#plt.show()
		#plt.clf()
		#############
		#############


		print "Calculating heating rate"
		# Photoionization heating
		eps_HI   = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HI, nuzero, 13.59844*1.602E-19/6.626E-34)
		eps_HeI  = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HeI, nuzero, 24.58741*1.602E-19/6.626E-34)
		eps_HeII = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HeII, nuzero, 54.41778*1.602E-19/6.626E-34)
		phion_heat_rate = eps_HI*densitynH*prof_YHI + eps_HeI*densitynH*prim_He*prof_YHeI + eps_HeII*densitynH*prim_He*prof_YHeII
		# Secondary electron photoheating rate (Shull & van Steenberg 1985)
		heat_HI  = 4.0*np.pi * calc_Jnur.scdryheatrate(jnurarr,nuzero,phelxs_HI,electrondensity/(densitynH*(1.0+2.0*prim_He)),0)
		heat_HeI = 4.0*np.pi * calc_Jnur.scdryheatrate(jnurarr,nuzero,phelxs_HeI,electrondensity/(densitynH*(1.0+2.0*prim_He)),1)
		scdry_heat_rate = heat_HI*densitynH*prof_YHI + heat_HeI*densitynH*prim_He*prof_YHeI
		# Finally, the total heating rate is:
		total_heat = phion_heat_rate + scdry_heat_rate
		#plt.plot(np.log10(prof_temperature),np.log10(total_heat),'k-')

		print "Deriving the temperature profile"
		#prof_temperature, temp, colexc, colion, colrec, diel, brem, comp = calc_Jnur.thermal_equilibrium(total_heat, electrondensity, densitynH, prof_YHI, prof_YHeI, prof_YHeII, prim_He, redshift)
		#prof_temperature, temp, coolfunc = calc_Jnur.thermal_equilibrium(total_heat, electrondensity, densitynH, prof_YHI, prof_YHeI, prof_YHeII, prim_He, redshift)
		prof_temperature = calc_Jnur.thermal_equilibrium(total_heat, electrondensity, densitynH, prof_YHI, prof_YHeI, prof_YHeII, prim_He, redshift)
		#prof_temperature = calc_Jnur.thermal_equilibrium_cf(total_heat, 10.0**coolfunc, 10.0**logT)
		#plt.plot(np.log10(temp),np.log10(colexc),'r-')
		#plt.plot(np.log10(temp),np.log10(colion),'r--')
		#plt.plot(np.log10(temp),np.log10(colrec),'g-')
		#plt.plot(np.log10(temp),np.log10(diel),'g--')
		#plt.plot(np.log10(temp),np.log10(brem),'b-')
		#plt.plot(np.log10(temp),np.log10(comp),'b--')
		#plt.plot(np.log10(cmtopc*radius),np.log10(prof_temperature),'k-')
		#plt.show()
		#plt.clf()
		#############
		# TEST PLOT #
		#############
		#print "photoheating rate!"
		#plt.plot(np.arange(radius.size),prof_YHI,'r-')
		#plt.plot(np.arange(radius.size),prof_YHeI,'g-')
		#plt.plot(np.arange(radius.size),prof_YHeII,'b-')
		#plt.show()
		#plt.clf()
		#print "heating rate!"
		#plt.plot(np.log10(radius*cmtopc),prof_temperature,'k-')
		#plt.plot(np.arange(radius.size),total_heat,'k-')
		#plt.plot(np.arange(radius.size),phion_heat_rate,'r-')
		#plt.plot(np.arange(radius.size),scdry_heat_rate,'b-')
		#plt.show()
		#plt.clf()
		#print "temperature!"
		#plt.plot(np.log10(radius*cmtopc),prof_temperature,'k-')
		#plt.plot(np.arange(radius.size),prof_temperature,'k-')
		#plt.show()
		#plt.clf()
		#print "cooling functions!"
		#plt.plot(np.log10(radius*cmtopc),prof_temperature,'k-')
		#plt.plot(10.0**np.linspace(3.0,6.0,coolfuncL.size),coolfuncL,'b-')
		#plt.plot(10.0**np.linspace(3.0,6.0,coolfuncL.size),coolfuncL,'bx')
		#plt.plot(10.0**np.linspace(3.0,6.0,coolfuncM.size),coolfuncM,'g-')
		#plt.plot(10.0**np.linspace(3.0,6.0,coolfuncM.size),coolfuncM,'gx')
		#plt.plot(10.0**np.linspace(3.0,6.0,coolfuncR.size),coolfuncR,'r-')
		#plt.plot(10.0**np.linspace(3.0,6.0,coolfuncR.size),coolfuncR,'rx')
		#plt.plot([1.0E3,1.0E6],[total_heat[414],total_heat[414]],'b--')
		#plt.plot([1.0E3,1.0E6],[total_heat[415],total_heat[415]],'g--')
		#plt.plot([1.0E3,1.0E6],[total_heat[416],total_heat[416]],'r--')
		#plt.show()
		#plt.clf()
		#############
		#############

		# Calculate a temperature profile by assuming thermal equilibrium (balancing heating and cooling)
		# Derive the total heating rate

		#if iteration > 0:
			#print "Calculating heating rate"
			## Photoionization heating
			#eps_HI   = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HI, nuzero, 13.59844*1.602E-19/6.626E-34)
			#eps_HeI  = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HeI, nuzero, 24.58741*1.602E-19/6.626E-34)
			#eps_HeII = 4.0*np.pi * calc_Jnur.phheatrate(jnurarr, phelxs_HeII, nuzero, 54.41778*1.602E-19/6.626E-34)
			#phion_heat_rate = eps_HI*densitynH*prof_YHI + eps_HeI*densitynH*prim_He*prof_YHeI + eps_HeII*densitynH*prim_He*prof_YHeII
			## Secondary electron photoheating rate (Shull & van Steenberg 1985)
			#heat_HI  = 4.0*np.pi * calc_Jnur.scdryheatrate(jnurarr,nuzero,phelxs_HI,electrondensity/(densitynH*(1.0+2.0*prim_He)),0)
			#heat_HeI = 4.0*np.pi * calc_Jnur.scdryheatrate(jnurarr,nuzero,phelxs_HeI,electrondensity/(densitynH*(1.0+2.0*prim_He)),1)
			#scdry_heat_rate = heat_HI*densitynH*prof_YHI + heat_HeI*densitynH*prim_He*prof_YHeI
			## Finally, the total heating rate is:
			#total_heat = phion_heat_rate + scdry_heat_rate
			#plt.plot(np.log10(cmtopc*radius),np.log10(total_heat*densitynH**2),'k-')

			#print "Deriving the temperature profile"
			#prof_temperature, temp, coolfunc = calc_Jnur.thermal_equilibrium(total_heat, electrondensity, densitynH, prof_YHI, prof_YHeI, prof_YHeII, prim_He, redshift)
			#prof_temperature = calc_Jnur.thermal_equilibrium_cf(total_heat, 10.0**coolfunc, 10.0**logT)
			#plt.plot(np.log10(cmtopc*radius),np.log10(prof_temperature),'k-')
			#plt.plot(np.log10(temp),np.log10(coolfunc),'k-')
			#plt.show()
			#plt.clf()

		tstHI   = (np.abs(old_prof_YHI-prof_YHI)<concrit).astype(np.int).sum()
		tstHeI  = (np.abs(old_prof_YHeI-prof_YHeI)<concrit).astype(np.int).sum()
		tstHeII = (np.abs(old_prof_YHeII-prof_YHeII)<concrit).astype(np.int).sum()

		print "STATISTICS -- {0:3.2f}".format(np.log10(virialm))
		w_HI   = np.argmax(np.abs(old_prof_YHI-prof_YHI))
		w_HeI  = np.argmax(np.abs(old_prof_YHeI-prof_YHeI))
		w_HeII = np.argmax(np.abs(old_prof_YHeII-prof_YHeII))
		print w_HI, np.abs(old_prof_YHI[w_HI]-prof_YHI[w_HI]), old_prof_YHI[w_HI], prof_YHI[w_HI]
		print w_HeI, np.abs(old_prof_YHeI[w_HeI]-prof_YHeI[w_HeI]), old_prof_YHeI[w_HeI], prof_YHeI[w_HeI]
		print w_HeII, np.abs(old_prof_YHeII[w_HeII]-prof_YHeII[w_HeII]), old_prof_YHeII[w_HeII], prof_YHeII[w_HeII]

		# Check if the stopping criteria were met
		if tstHI   == npts: critHI = False
		if tstHeI  == npts: critHeI = False
		if tstHeII == npts: critHeII = False
		if iteration > maxiter:
			print "Break outer loop at maxiter={0:d} iterations, STATUS:".format(maxiter)
			print "   Rates Iteration {0:d}   <--[  {1:d}/{2:d} {3:d}/{2:d} {4:d}/{2:d}  ]-->".format(iteration,tstHI,npts,tstHeI,tstHeII)
			break
		#############
		# TEST PLOT #
		#############
		if iteration == 1000:
			plt.plot(np.log10(radius*cmtopc),prof_YHI,'r-')
			plt.plot(np.log10(radius*cmtopc),prof_YHeI,'g-')
			plt.plot(np.log10(radius*cmtopc),prof_YHeII,'b-')
			plt.show()
			plt.clf()
			answer = int(raw_input("When should I plot next time {0:3.2f}".format(np.log10(virialm))))
		elif iteration == answer:
			plt.plot(np.log10(radius*cmtopc),prof_YHI,'r-')
			plt.plot(np.log10(radius*cmtopc),prof_YHeI,'g-')
			plt.plot(np.log10(radius*cmtopc),prof_YHeII,'b-')
			plt.show()
			plt.clf()
			answer = int(raw_input("When should I plot next time {0:3.2f}".format(np.log10(virialm))))
		#############
		#############

	print "Calculating H I, He I, and He II column density profiles"
	prof_HIcolumndensity = calc_Jnur.coldens(prof_YHI*densitynH,radius)
	prof_HeIcolumndensity = calc_Jnur.coldens(prof_YHeI*densitynH*prim_He,radius)
	prof_HeIIcolumndensity = calc_Jnur.coldens(prof_YHeII*densitynH*prim_He,radius)

	timeB = time.time()
	print "Test completed in {0:f} mins".format((timeB-timeA)/60.0)

	# Save the results
	mstring = ("{0:3.2f}".format(np.log10(virialm))).replace(".","p").replace("+","")
	rstring = ("{0:3.2f}".format(redshift)).replace(".","p").replace("+","")
	outfname = "data/Cored_model_mass{0:s}_redshift{1:s}".format(mstring,rstring)
	print "Saving file {0:s}.npy".format(outfname)
	#print "ERROR: file not saved"
	np.save(outfname, np.transpose((radius,electrondensity,densitynH,prof_temperature,prof_pressure/kB,prof_YHI,prof_YHeI,prof_YHeII,prof_HIcolumndensity,prof_HeIcolumndensity,prof_HeIIcolumndensity,phionrate_HI,phionrate_HeI,phionrate_HeII,scdryrate_HI,scdryrate_HeI,B1_HeII,B2_HeI,B3_HI,B6_HeI,B7_HI)))

	plotitup=False
	if plotitup:
		xval = radius/rscale
		Mds = 4.0*np.pi*rhods*(rscale**3)/3.0
		plt.plot(np.log10(radius*cmtopc),np.log10(Mds*3.0*(np.log(1.0+xval) - xval/(1.0+xval))),'k-')
		plt.plot(np.log10(radius*cmtopc),np.log10((Omega_b/(Omega_m-Omega_b))*Mds*3.0*(np.log(1.0+xval) - xval/(1.0+xval))),'b--')
		plt.plot(np.log10(radius*cmtopc),np.log10(4.0*np.pi*rscale**3*fgas),'g-')
		plt.show()
		plt.clf()

		plt.plot(radius*cmtopc,prof_YHI,'r-')
		plt.plot(radius*cmtopc,prof_YHeI,'g-')
		plt.plot(radius*cmtopc,prof_YHeII,'b-')
		plt.show()
		plt.clf()

		plt.subplot(2,2,1)
		plt.plot(densitynH,prof_temperature,'bx')
		plt.yscale('log')
		plt.subplot(2,2,2)
		plt.plot(densitynH,prof_pressure/kB,'gx')
		plt.subplot(2,2,3)
		plt.plot(radius*cmtopc,densitynH,'k-')
		plt.subplot(2,2,4)
		plt.plot(radius*cmtopc,prof_temperature,'k-')
		plt.show()
		plt.clf()

		plt.plot(radius*cmtopc,prof_HIcolumndensity,'r-')
		plt.plot(radius*cmtopc,prof_HeIcolumndensity,'g-')
		plt.plot(radius*cmtopc,prof_HeIIcolumndensity,'b-')
		plt.show()
		plt.clf()

#jnurarr = calc_Jnur.Jnur(density, radius, jzero, phelxs_HI, nummu)
#coldensHI, muarr = calc_Jnur.calc_coldens(density, radius, nummu)
#coldensHeI, muarr = calc_Jnur.calc_coldens(density, radius, nummu)
#coldensHeII, muarr = calc_Jnur.calc_coldens(density, radius, nummu)
#jnurarr_split = calc_Jnur.nint_costheta(coldensHI, phelxs_HI, coldensHeI, phelxs_HeI, coldensHeII, phelxs_HeII, muarr, jzero)

#plt.plot(engy,phelxs_HI*jnurarr[:,250],'k-')
#plt.show()
#plt.clf()

# Calculate the local primary photoionization rate as a function of radius

#plt.plot(radius,lphirate,'k-')
#plt.show()
