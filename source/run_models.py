import numpy as np
import halomodel
import gethalo
import getoptions
import cosmo

bturb = 3.0
metals = 1.0E-3
ions = ["H I", "D I", "He I", "He II"]

gastemp = 20000.0

mn_mvir = 8.0
mx_mvir = 8.0 #9.65
nummvir = 1   #21

mn_reds = 0.0
mx_reds = 0.0
numreds = 1

mn_bary = 1.0
mx_bary = 1.0
numbary = 1

mn_HMscl = 1.0
mx_HMscl = 1.0
numHMscl = 1

virialm = np.linspace(mn_mvir,mx_mvir,nummvir)
redshift = np.linspace(mn_reds,mx_reds,numreds)
baryscale = np.linspace(mn_bary,mx_bary,numbary)
HMscale = np.linspace(mn_HMscl,mx_HMscl,numHMscl)[::-1]

# Load baryon fraction as a function of halo mass
halomass, barymass = np.loadtxt("data/baryfrac.dat", unpack=True)
baryfracvals = 10.0**barymass / 10.0**halomass
baryfrac = np.interp(virialm, halomass, baryfracvals)

# Set the options dictionary
options = getoptions.default()
# Overwrite the defaults
options["run"]["nsample"]  = 1000
options["run"]["ncpus"]    = -1
options["run"]["nummu"]    = 30
options["run"]["concrit"] = 1.0E-3
options["run"]["maxiter"] = 500
options["run"]["outdir"] = "test" # PUT RUN NAME HERE
options["geometry"] = "NFW"
options["geomscale"] = 10
#options["radfield"] = "PLm1d5_IPm6"
options["radfield"] = "HM12"
options["HMscale"] = 1.0

# Get the working cosmology
cosmopar = cosmo.get_cosmo()

hztos = options["const"]["hztos"]
Gcons = options["const"]["Gcons"]
somtog = options["const"]["somtog"]
hubpar = cosmo.hubblepar(redshift, cosmopar)
rhocrit = 3.0*(hubpar*hztos)**2/(8.0*np.pi*Gcons)

# get_halo returns the name of the file it writes the output to
# so that it can be passed back on the next loop iteration to use as an intitial solution
prev_fname = None

for i in range(nummvir):
    for j in range(numreds):
        print "#########################"
        print "#########################"
        print "  virialm  {0:d}/{1:d}".format(i+1,nummvir)
        print "  redshift {0:d}/{1:d}".format(j+1,numreds)
        for k in range(numbary):
            print "  baryon scale {0:d}/{1:d}".format(k+1,numbary)
            concentration = cosmo.massconc_Prada12(10**virialm[i], cosmopar, redshift[j])
            model = halomodel.NFWHalo(10**virialm[i] * somtog, baryfrac[i] * baryscale[k], rhocrit, concentration)
            for l in range(numHMscl):
                print "  UVB scale {0:d}/{1:d}".format(l+1,numHMscl)
                print "#########################"
                print "#########################"

                # Let's go!
                prev_fname = gethalo.get_halo(model,redshift[j],gastemp,bturb,Hescale=1.0,metals=metals,cosmopar=cosmopar,ions=ions,prevfile=prev_fname,options=options)
