def get():
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

    return consdict