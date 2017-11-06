import numpy as np
import function_NFW
import getoptions
import cosmo

metals = 1.0E-3
gastemp = 10000.0
bturb = 3.0

#ions = ["H I", "D I", "He I", "He II", "N I", "N II", "N III", "Si I", "Si II", "Si III"]
#ions = ["H I", "D I", "He I", "He II"]
#ions = ["H I", "He I", "He II"]
ions = ["H I", "D I", "He I", "He II", "C I", "C II", "C III", "C IV", "N I", "N II", "N III", "N IV", "O I", "O II", "O III", "O IV", "Mg I", "Mg II", "Mg III", "Mg IV", "Si I", "Si II", "Si III", "Si IV"]

# Set the options dictionary
options = getoptions.default()
# Overwrite the defaults
options["run"]["nsample"]  = 1000
options["run"]["ncpus"]    = -1
options["geometry"]["use"] = "PP"
options["HMscale"] = 1.0
options["radfield"] = "HM12"
#options["radfield"] = "PLm1_IPm3"

mn_dens = -2.0
mx_dens = 2.0
numdens = 9

density = np.linspace(mn_dens,mx_dens,numdens)

# Get the working cosmology
cosmopar = cosmo.get_cosmo()

for d in range(numdens):
    #radius = 3.24/(10.0**density[d])  # 3.24 = 3.24E-22*1.0E21 = cm_to_kpc * total H column density
    radius = 0.6/(10.0**density[d])
    options["geometry"]["PP"] = [density[d],radius]
    print "#########################"
    print "#########################"
    print "  density  {0:d}/{1:d}".format(d+1,numdens)
    print "  radius  {0:.4f}".format(radius)
    print "#########################"
    print "#########################"
    function_NFW.get_halo(3.0,gastemp,bturb,metals=metals,cosmopar=cosmopar,ions=ions,options=options)




