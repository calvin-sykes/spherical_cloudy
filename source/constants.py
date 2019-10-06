LOG_COLDENS_TARGET = 21.0
LOG_COLDENS_MAX = 24.0

def get():
    # Set the constants to be used throughout the code
    consdict = dict({})
    consdict["kB"]      = 1.3806488E-16   # Boltmann constant in units of erg/K
    consdict["cmtopc"]  = 3.24E-19        # Conversion between cm and parsec
    consdict["somtog"]  = 1.989E33        # Conversion between solar masses and grams
    consdict["Gcons"]   = 6.67259E-8      # Gravitational constant in cgs
    consdict["planck"]  = 6.62606957E-27  # Planck constant in cgs
    consdict["elvolt"]  = 1.60217657E-12  # 1 eV in erg
    consdict["protmss"] = 1.67262178E-24  # Mass of a proton in g
    consdict["hztos"]   = 3.241E-20       # Conversion between km/s/Mpc to s

    return consdict

def get_nt():

    constdict = get()

    from collections import namedtuple
    constnt = namedtuple('Constants', constdict.iterkeys())
    return constnt(constdict.itervalues())
