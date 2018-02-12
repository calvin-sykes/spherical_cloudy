import numpy as np
import halomodel
import gethalo
import options
import constants
import cosmo
import os

bturb = 3.0
metals = 1.0E-3
ions = ["H I", "D I", "He I", "He II"]

gastemp = 20000.0

#mn_mvir = 8.0
#mx_mvir = 9.65
#nummvir = 21

mn_reds = 0.0
mx_reds = 0.0
numreds = 1

mn_bary = 1.0
mx_bary = 1.0
numbary = 1

mn_HMscl = 1.0
mx_HMscl = 1.0
numHMscl = 1

virialm = np.concatenate([np.arange(8.0, 9.0, 0.1), np.arange(9.0, 9.5, 0.05), np.arange(9.5, 9.65, 0.01)])
nummvir = len(virialm)

redshift = np.linspace(mn_reds,mx_reds,numreds)
baryscale = np.linspace(mn_bary,mx_bary,numbary)
HMscale = np.linspace(mn_HMscl,mx_HMscl,numHMscl)[::-1]

# Load baryon fraction as a function of halo mass
halomass, barymass = np.loadtxt("data/baryfrac.dat", unpack=True)
baryfracvals = 10.0**barymass / 10.0**halomass
baryfrac = np.interp(virialm, halomass, baryfracvals)

# Whether to resume from a previously written output file
# None: don't resume
# "last": pick up from the most recent file
# number: pick up from this index in all the models (negative to go from the end backward)
resume = None

# Set the options dictionary
options = options.default()
# Overwrite the defaults
options["run"]["nsample"] = 1000
options["run"]["ncpus"]   = -1
options["run"]["nummu"]   = 30
options["run"]["concrit"] = 1.0E-3
options["run"]["maxiter"] = 500
options["run"]["outdir"]  = "test_new_eagle_temps" # PUT RUN NAME HERE
options["geometry"] = "NFW"
options["geomscale"] = 100
#options["radfield"] = "PLm1d5_IPm6"
options["radfield"] = "HM12"
options["HMscale"] = 1.0

# Whether to impose external pressure condition
options["force_P"] = True

# Method used to derive temperature
# equilibrium - always use thermal equilibrium
# adiabatic - always use 1/rate = Hubble time
# eagle - use cooling rate table from Eagle
# original - use Ryan's original thermal equilibrium function
options["temp_method"] = "eagle"

# Get the working cosmology
cosmopar = cosmo.get_cosmo()

# Get some constants needed to define the halo model
constants = constants.get()
hztos = constants["hztos"]
Gcons = constants["Gcons"]
somtog = constants["somtog"]
hubpar = cosmo.hubblepar(redshift, cosmopar)
rhocrit = 3.0*(hubpar*hztos)**2/(8.0*np.pi*Gcons)

# Find the name of a previously written file, and initialise loop counters
# such that the run starts from reading back this file
# params: 'where' is an index back from the last file written (-2=last but one)
def init_resume(where):
    wd = os.getcwd()
    out_path = 'output/' + options["run"]["outdir"] + '/'
    os.chdir(out_path)
    files = sorted(os.listdir('.'), key=os.path.getmtime)
    # return to working directory
    os.chdir(wd)
    if where == 'last':
        file_idx = len(files) - 1
    elif type(where) is int:
        file_idx = where
    else:
        print("Couldn't understand resume command!")
        exit(1)
    print(file_idx)
    fname = out_path + files[file_idx - 1]
    smvir = file_idx % nummvir
    sHMscl = (file_idx // nummvir) % numHMscl
    sbary = (file_idx // (nummvir * numHMscl)) % numbary
    sreds = (file_idx // (nummvir * numHMscl * numbary))
    print(fname, smvir, sHMscl, sbary, sreds)
    return (fname, smvir, sHMscl, sbary, sreds)

if resume is not None:
    prev_fname, smvir, sHMscl, sbary, sreds = init_resume(resume)
else:
    sreds = 0
    sbary = 0
    sHMscl = 0
    smvir = 0
    prev_fname = None

for i in range(sreds, numreds):     
    for j in range(sbary, numbary):
        for k in range(sHMscl, numHMscl):
            for l in range(smvir, nummvir):
                print "#########################"
                print "#########################"
                print "  virialm  {0:d}/{1:d}".format(l+1,nummvir)
                print "  redshift {0:d}/{1:d}".format(k+1,numreds)
                print "  baryon scale {0:d}/{1:d}".format(j+1,numbary)
                print "  UVB scale {0:d}/{1:d}".format(i+1,numHMscl)
                print "#########################"
                print "#########################"
                concentration = cosmo.massconc_Prada12(10**virialm[l], cosmopar, redshift[k])
                model = halomodel.NFWHalo(10**virialm[l] * somtog, baryfrac[l] * baryscale[j], rhocrit, concentration)
                # Let's go!
                ok, res = gethalo.get_halo(model,redshift[k],gastemp,bturb,Hescale=1.0,metals=metals,cosmopar=cosmopar,ions=ions,prevfile=prev_fname,options=options)
                
                if ok:
                    prev_fname = res
                else:
                    # something went wrong with the model
                    print "ERROR :: {:s}".format(res)
                    # Don't terminate entirely; just move on to the next bit of the grid
                    break
            prev_fname = None
