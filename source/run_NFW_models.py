import numpy as np
import function_NFW
import getoptions
import cosmo

bturb = 3.0
metals = 1.0E-3
#pl_ionizparam = -3.0
#pl_index = -0.92

#ions = ["H I", "D I", "He I", "He II", "N I", "N II", "N III", "Si I", "Si II", "Si III"]
ions = ["H I", "D I", "He I", "He II"]
#ions = ["H I", "He I", "He II"]
#ions = ["H I", "D I", "He I", "He II", "C I", "C II", "C III", "C IV", "C V", "C VI", "N I", "N II", "N III", "N IV", "O I", "O II", "O III", "O IV", "Mg I", "Mg II", "Mg III", "Mg IV", "Si I", "Si II", "Si III", "Si IV"]

gastemp = 20000.0


mn_mvir = 8.0
mx_mvir = 9.65
nummvir = 21

mn_reds = 0.0
mx_reds = 0.0
numreds = 1

mn_bary = 0.95
mx_bary = 0.99
numbary = 5
#mx_bary = 0.0
#numbary = 5

#mn_mvir = 7.0
#mx_mvir = 7.0
#nummvir = 1

virialm = np.linspace(mn_mvir,mx_mvir,nummvir)
redshift = np.linspace(mn_reds,mx_reds,numreds)
#baryfrac = np.linspace(mn_bary,mx_bary,numbary)
HMscale = np.linspace(mn_bary,mx_bary,numbary)[::-1]

# Load baryon fraction as a function of halo mass
halomass, barymass = np.loadtxt("data/baryfrac.dat", unpack=True)
baryfracvals = 10.0**barymass / 10.0**halomass
baryfrac = np.interp(virialm, halomass, baryfracvals)

convergence=False

if convergence:
    nmu = [30, 90, 180]
    nrad = [500, 1000, 2000]
    for r in range(len(nrad)):
        for mu in range(len(nmu)):
            # Set the options dictionary
            options = getoptions.default()
            # Overwrite the defaults
            options["run"]["nsample"]  = nrad[r]
            options["run"]["ncpus"]    = -1
            options["run"]["nummu"]    = nmu[mu]
            options["geometry"]["use"] = "NFW"
            options["HMscale"] = 1.0
            options["radfield"] = "HM12"
            #options["radfield"] = "PLm1d5_IPm6"

            # Get the working cosmology
            cosmopar = cosmo.get_cosmo()

            for i in range(nummvir):
                for j in range(numreds):
                    print "#########################"
                    print "#########################"
                    print "  virialm  {0:d}/{1:d}".format(i+1,nummvir)
                    print "  redshift {0:d}/{1:d}".format(j+1,numreds)
                    print "#########################"
                    print "#########################"
                    for k in range(numbary):
                        options["geometry"]["NFW"] = [10.0**virialm[i],10.0,10.0**baryfrac[k]]
                        function_NFW.get_halo(redshift[j],gastemp,bturb,metals=metals,cosmopar=cosmopar,ions=ions,options=options)
else:
    # Set the options dictionary
    options = getoptions.default()
    # Overwrite the defaults
    options["run"]["nsample"]  = 1000
    options["run"]["ncpus"]    = -1
    options["run"]["nummu"]    = 30
    options["geometry"]["use"] = "NFW"
    options["HMscale"] = 1.0
    options["radfield"] = "HM12"
    #options["radfield"] = "PLm1d5_IPm6"

    # Get the working cosmology
    cosmopar = cosmo.get_cosmo()

    for i in range(nummvir):
        for j in range(numreds):
            print "#########################"
            print "#########################"
            print "  virialm  {0:d}/{1:d}".format(i+1,nummvir)
            print "  redshift {0:d}/{1:d}".format(j+1,numreds)
            print "#########################"
            print "#########################"
            for k in range(numbary):
                #options["geometry"]["NFW"] = [10.0**virialm[i],1.0,10.0**baryfrac[k]]
                options["geometry"]["NFW"] = [10.0**virialm[i],1.0,baryfrac[i]]
                options["HMscale"] = 1.0
                #options["HMscale"] = HMscale[k]
                options["run"]["concrit"] = 1.0E-3
                options["run"]["maxiter"] = 500
                function_NFW.get_halo(redshift[j],gastemp,bturb,Hescale=HMscale[k],metals=metals,cosmopar=cosmopar,ions=ions,options=options)
                #function_NFW.get_halo(redshift[j],gastemp,bturb,Hescale=1.0,metals=metals,cosmopar=cosmopar,ions=ions,options=options)

