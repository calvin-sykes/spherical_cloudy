import collections
import configparser

import logger

## NOTE: If options are added to the program b changing the dictionaries defined below,
##       this file needs to be executed as a standalone script to update the default config file

# What types each setting should be
option_types = collections.defaultdict(lambda: int) # default to integer type
for str_opt in ['geometry:profile', 'UVB:spectrum', 'phys:temp_method', 'run:outdir', 'run:resume', 'log:level', 'log:file',
                'grid:virialm', 'grid:redshift', 'grid:baryscale', 'grid:radscale']:
    option_types[str_opt] = str
for flt_opt in ['geometry:scale', 'geometry:acore', 'UVB:scale', 'UVB:slope', 'run:concrit', 'phys:gastemp', 'phys:metals', 'phys:bturb',
                'phys:hescale']:
    option_types[flt_opt] = float
for bool_opt in ['phys:ext_press', 'run:do_ref', 'run:do_smth']:
    option_types[bool_opt] = lambda s: s.capitalize() == 'TRUE'

# Default settings for options
def default(save=False):
    options = dict({})
    # Set the default options for running
    # Set some of the basic operation modes
    runpar = dict({})
    runpar['miniter'] = 10         # Minimum number of iterations to perform
    runpar['maxiter'] = 100000     # Maximum number of iterations to perform
    runpar['nsample'] = 500        # Number of radial points to consider
    runpar['nummu'  ] = 360        # Number of numerical integration points when integrating over cos(theta)
    runpar['concrit'] = 1.0E-5     # Convergence criteria for deriving Y-values
    runpar['ncpus'  ] = 4          # Number of CPUs to use
    runpar['outdir' ] = 'not_set'  # Directory under '/output' to save results to
    runpar['resume' ] = 'none'     # Whether to resume from a previously written output file
                                   #   'last': pick up from the most recent file
                                   #   'refine_last' : take the most recent file and try to refine it to get rid of the discontinuity
                                   #   string: find file with matching name and start from there
    runpar['do_ref' ] = False      # Whether to attempt refining
    runpar['do_smth'] = True       # Whether to smooth out discontinuities in the H I Y profile
    options['run'   ] = runpar

    # Set logging settings
    logpar = dict({})
    logpar['level']  = 'debug' # Minimum level of warning to log ('debug', 'info', 'warning', 'error', 'critical')
    logpar['file' ]  = 'none'  # File to save the log to (console if 'none')
    options['log' ]  = logpar

    # Set the geometry
    geompar = dict({})
    geompar['profile' ] = 'NFW'         # Which geometry should be used
    geompar['scale'   ] = 100           # Outer radius in units of R_vir
    geompar['acore'   ] = 0.5           # Ratio of core radius to virial radius (Cored density profile only)
    geompar['PP_depth'] = 1             # Depth of slab in kpc (Plane parallel geometry only)
    geompar['PP_dens' ] = -1            # log of H number density in slab in cm^-3 (Plane parallel geometry only)
    options['geometry'] = geompar

    # Set the radiation field
    radpar = dict({})
    radpar['spectrum'] = 'HM12'   # Set the radiation field. Options include ('HM12', 'PLm1_IPm6'...'PLm1_IPm1')
    radpar['scale'   ] = 1.0      # Constant to scale the background radiation field by
    radpar['slope'   ] = 0.0      # HM radiation field shape parameter (Crighton et al 2015, https://arxiv.org/pdf/1406.4239.pdf)
    options['UVB'    ] = radpar

    # Set physical conditions
    physpar = dict({})
    physpar['ext_press'  ] = False      # Whether to impose condition that density should approach cosmic mean
    physpar['temp_method'] = 'original' # Method to use to calculate temperature:
                                        #   equilibrium - always use thermal equilibrium
                                        #   adiabatic - always use 1/rate = Hubble time
                                        #   eagle - use cooling rate table from Eagle
                                        #   original - use Ryan's original thermal equilibrium function
                                        #   relhic - use tabulated nH-T relation from ABL paper
    physpar['bturb'  ] = 0.0        # Value of turbulent Doppler parameter in km/s
    physpar['metals' ] = 1.0E-3     # metallicity relative to solar
    physpar['gastemp'] = 20000      # initial gas temperature in Kelvin
    physpar['hescale'] = 1.0        # Constant to scale the helium abundance by
    options['phys'   ] = physpar

    # Set grid of parameters
    # Each option should be a string containing a statement that gets eval'd to produce an interable of parameter values
    gridpar = dict({})
    gridpar['virialm'  ] = "np.linspace(8.0, 10.0, 21)" # log of halo virial masses
    gridpar['redshift' ] = "np.zeros(1)" # redshifts
    gridpar['baryscale'] = "np.ones(1)" # scaling of universal baryon fraction
    gridpar['radscale' ] = "np.ones(1)" # scaling of UVB intensity
    options['grid'     ] = gridpar

    if save:
        parser = configparser.ConfigParser()
        parser.read_dict(options)

        with open('defaults.ini', 'w') as f:
            parser.write(f)        

    return options

# Read a plain text file containing options
# Each line should contain three fields
def read_options(filename):
    # Start with default settings
    options = default()

    logger.init(level='warning', name='options')
    
    parser = configparser.ConfigParser()
    config = parser.read(filename)

    for section in parser.sections():
        if section in options.keys():
            for key in parser[section]:
                if key in options[section].keys():
                    try:
                        options[section][key] = option_types['{}:{}'.format(section, key)](parser[section][key])
                    except ValueError:
                        logger.log('error', "Bad value {} for option {}:{}, using default {}"
                                   .format(parser[section][key], section, key, options[section][key]), 'options')
                    logger.log('debug', "New setting for option {}:{} is {}"
                               .format(section, key, parser[section][key]), 'options')
                else:
                    logger.log('warning', "Input file option {:s} is invalid".format(key), 'options')
        else:
            logger.log('warning', "Input file section {:s} is invalid".format(section), 'options')

    return options

if __name__ == '__main__':
    default(True)
