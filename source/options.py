import collections
import configparser
import sys

import logger

## NOTE: If options are added to the program by changing the dictionaries defined below,
##       this file needs to be executed as a standalone script to update the default config file

# What types each setting should be
option_types = collections.defaultdict(lambda: int) # default to integer type
for str_opt in ['geometry:profile', 'geometry:concrel', 'UVB:spectrum', 'phys:temp_method', 'run:outdir', 'run:resume', 'log:level', 'log:file',
                'grid:virialm', 'grid:redshift', 'grid:baryscale', 'grid:radscale', 'grid:pp_cdens', 'grid:pp_dens', 'save:he_emis']:
    option_types[str_opt] = str
for flt_opt in ['geometry:scale', 'geometry:acore', 'UVB:scale', 'UVB:slope', 'run:concrit', 'phys:gastemp', 'phys:metals', 'phys:bturb',
                'phys:hescale']:
    option_types[flt_opt] = float
for bool_opt in ['phys:ext_press', 'run:pp_para', 'run:lv_plot',
                 'save:rates', 'save:heat_cool', 'save:recomb', 'save:intensity']:
    option_types[bool_opt] = lambda s: s.upper() == 'TRUE'

# Default settings for options
def default(save=False):
    options = dict({})
    # Set the default options for running
    # Set some of the basic operation modes
    runpar = dict({})
    runpar['pp_para'] = False      # Whether to run all plane parallel models in parallel
    runpar['miniter'] = 10         # Minimum number of iterations to perform
    runpar['maxiter'] = 100000     # Maximum number of iterations to perform
    runpar['nsample'] = 500        # Number of radial points to consider
    runpar['nummu'  ] = 360        # Number of numerical integration points when integrating over cos(theta)
    runpar['concrit'] = 1.0E-5     # Convergence criteria for deriving Y-values
    runpar['ncpus'  ] = 4          # Number of CPUs to use
    runpar['outdir' ] = 'not_set'  # Directory under '/output' to save results to
    runpar['resume' ] = 'none'     # Whether to resume from a previously written output file
                                   #   'last': pick up from the most recent file
                                   #   string: find file with matching name and start from there
    runpar['lv_plot'] = False      # Whether to plot each iteration without interrupting calculation
                                   # (edit the source code to change what is plotted)
    options['run'   ] = runpar

    savepar = dict({})
    savepar['rates'    ] = False   # Whether to save the ionisation rates
    savepar['recomb'   ] = False   # Whether to save the recombination rates
    savepar['heat_cool'] = False   # Whether to save the heating and cooling rates
    savepar['intensity'] = False   # Whether to save the mean intensity J_nu
    savepar['he_emis'  ] = 'none'  # Whether to save HeI line surface brightnesses
                                   # if 'none', no SBs will be saved
                                   # if 'all' all lines will be saved
                                   # if a single value, that line will be saved
                                   # if a list of values, each will be saved as a separate column
    options['save'     ] = savepar

    # Set logging settings
    logpar = dict({})
    logpar['level']  = 'debug' # Minimum level of warning to log ('debug', 'info', 'warning', 'error', 'critical')
    logpar['file' ]  = 'none'  # File to save the log to (console if 'none')
    options['log' ]  = logpar

    # Set the geometry
    geompar = dict({})
    geompar['profile' ] = 'NFW'         # Which geometry should be used
    geompar['concrel' ] = 'Ludlow'      # Which mass-concentration relation should be used
                                        # (chg  27/9/18 from Prada as that isn't for Planck cosmo)
    geompar['scale'   ] = 100           # Outer radius in units of R_vir
    geompar['acore'   ] = 0.5           # Ratio of core radius to virial radius (Cored density profile only)
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
                                        #   eagle - use cooling rate table from Eagle
                                        #   original - use Ryan's original thermal equilibrium function
                                        #   relhic - use tabulated nH-T relation from ABL paper
                                        #   blend - use eqbm above nH=10**-4.8 and relhic below, interpolating smoothly between them
                                        #   isothermal - use constant temperature given by phys:gastemp
    physpar['bturb'  ] = 0.0        # Value of turbulent Doppler parameter in km/s
    physpar['metals' ] = 1.0E-3     # metallicity relative to solar
    physpar['gastemp'] = 20000      # initial gas temperature in Kelvin
    physpar['hescale'] = 1.0        # Constant to scale the helium mass fraction Y by
                                    # Internally, a new value of the number abundance is calculated: y = (Y_P * Hescale) / (4 - 4 * Y_P * Hescale)
                                    # Default value of 1 gives primordial values y_P = 0.083, Y_P = 0.25
    options['phys'   ] = physpar

    # Set grid of parameters
    # Each option should be a string containing a statement that gets eval'd to produce an iterable of parameter values
    gridpar = dict({})
    gridpar['virialm'  ] = "np.linspace(8.0, 10.0, 21)" # log of halo virial masses
    gridpar['redshift' ] = "np.zeros(1)" # redshifts
    gridpar['baryscale'] = "np.ones(1)" # scaling of universal baryon fraction
    gridpar['radscale' ] = "np.ones(1)" # scaling of UVB intensity
    gridpar['pp_cdens' ] = "np.full(1, 18)" # log of column density depth in slab in cm^-2 (Plane parallel geometry only)
    gridpar['pp_dens'  ] = "np.full(1, -1.0)" # log of H number density in slab in cm^-3 (Plane parallel geometry only)
    options['grid'     ] = gridpar

    if save:
        parser = configparser.ConfigParser()
        parser.read_dict(options)

        with open('./input/defaults.ini', 'w') as f:
            parser.write(f)        

    return options

# Read a plain text file containing options
# Each line should contain three fields
def read_options(filename):
    # Start with default settings
    options = default()

    logger.init(level='warning', name='options')
    
    parser = configparser.ConfigParser()
    try:
        config = parser.read(filename)
    except Exception as e:
        logger.log('critical', "Error parsing config file {} (details follow)".format(filename), 'options')
        logger.log('critical', e.message, 'options')
        sys.exit(1)        

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
                    logger.log('warning', "Input file option {}:{} is invalid".format(section, key), 'options')
        else:
            logger.log('warning', "Input file section {:s} is invalid".format(section), 'options')

    return options

if __name__ == '__main__':
    default(True)
