
def default():
    options = dict({})
    # Set the default options for running
    # Set some of the basic operation modes
    runpar = dict({})
    runpar["miniter"] = 10         # Minimum number of iterations to perform
    runpar["maxiter"] = 100000     # Maximum number of iterations to perform
    runpar["nsample"] = 500        # Number of radial points to consider
    runpar["nummu"]   = 360        # Number of numerical integration points when integrating over cos(theta)
    runpar["concrit"] = 1.0E-5     # Convergence criteria for deriving Y-values
    runpar["ncpus"]   = 4          # Number of CPUs to use
    options["run"] = runpar
    # Set the constants to be used throughout the code
    consdict = dict({})
    consdict["kB"]      = 1.3806488E-16   # Boltmann constant in units of erg/K
    consdict["cmtopc"]  = 3.24E-19        # Conversion between cm and parsec
    consdict["somtog"]  = 1.989E33        # Conversion between solar masses and grams
    consdict["Gcons"]   = 6.67259E-8      # Gravitational constant in cgs
    consdict["planck"]  = 6.62606957E-34  # Planck constant in mks
    consdict["elvolt"]  = 1.60217657E-19  # 1 eV in J
    consdict["protmss"] = 1.67262178E-24  # Mass of a proton in g
    consdict["hztos"]   = 3.241E-20       # Conversion between km/s/Mpc to s
    options["const"] = consdict
    # Set the geometry
    geomdict = dict({})
    geomdict["use"] = "NFW"         # Which geometry should be used
    geomdict["NFW"] = [1.0E8,3,1.0] # NFW dark matter halo. Arg1 : virial mass, Arg2 : radial extent (units=r_vir), Arg 3 : scale the baryon fraction of the halo (1=universal baryon frtaction)
    geomdict["PP"]  = [-1.0,4.0]    # plane parallel. Arg1 : log(n_H), Arg2 : radial extent (units=kpc)
    options["geometry"] = geomdict
    # Set the radiation field
    options["radfield"] = "PLm1_IPm3"   # Set the radiation field. Options include ("HM12", "PLm1_IPm6"..."PLm1_IPm1")
    options["HMscale"]  = 1.0           # Scale the HM12 background radiation field
    options["powerlaw"] = None          # a two element array containing [index, ionization parameter] or the radiation field (None will ignore this radiation field)
    return options