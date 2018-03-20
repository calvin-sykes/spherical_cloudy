import collections
import configparser

import logger

# What types each setting should be
option_types = collections.defaultdict(lambda: int) # default to floating-point type
for str_opt in ['geometry:profile', 'UVB:spectrum', 'phys:temp_method', 'run:outdir', 'run:resume', 'log:level', 'log:file']:
    option_types[str_opt] = str
for flt_opt in ['geometry:scale', 'geometry:acore', 'UVB:scale', 'run:concrit']:
    option_types[flt_opt] = float
option_types['phys:ext_press'] = bool

# Default settings for options
def default(save=False):
    options = dict({})
    # Set the default options for running
    # Set some of the basic operation modes
    runpar = dict({})
    runpar['miniter'] = 10         # Minimum number of iterations to perform
    runpar['maxiter'] = 100000     # Maximum number of iterations to perform
    runpar['nsample'] = 500        # Number of radial points to consider
    runpar['nummu']   = 360        # Number of numerical integration points when integrating over cos(theta)
    runpar['concrit'] = 1.0E-5     # Convergence criteria for deriving Y-values
    runpar['ncpus']   = 4          # Number of CPUs to use
    runpar['outdir']  = 'not_set'  # Directory under '/output' to save results to
    runpar['resume']  = 'none'     # Whether to resume from a previously written output file
                                   #   'last': pick up from the most recent file
                                   #   'refine_last' : take the most recent file and try to refine it to get rid of the discontinuity
                                   #   number: pick up from this index in the models *already run* (negative is from the end backward)
    options['run']    = runpar

    # Set logging settings
    logdict = dict({})
    logdict['level']  = 'debug' # Minimum level of warning to log ('debug', 'info', 'warning', 'error', 'critical')
    logdict['file']   = 'none'  # File to save the log to (console if 'none')
    options['log']    = logdict

    # Set the geometry
    geompar = dict({})
    geompar['profile']  = 'NFW'         # Which geometry should be used
    geompar['scale']    = 100           # Outer radius in units of R_vir
    geompar['acore']    = 0.5           # Ratio of core radius to virial radius (Cored density profile only)
    options['geometry'] = geompar

    # Set the radiation field
    radpar = dict({})
    radpar['spectrum'] = 'HM12'   # Set the radiation field. Options include ('HM12', 'PLm1_IPm6'...'PLm1_IPm1')
    radpar['scale']    = 1.0      # Constant to scale the background radiation field by
    options['UVB']     = radpar

    # Set physical conditions
    physpar = dict({})
    physpar['ext_press']   = False      # Whether to impose condition that density should approach cosmic mean
    physpar['temp_method'] = 'original' # Method to use to calculate temperature:
                                        #   equilibrium - always use thermal equilibrium
                                        #   adiabatic - always use 1/rate = Hubble time
                                        #   eagle - use cooling rate table from Eagle
                                        #   original - use Ryan's original thermal equilibrium function
                                        #   relhic - use tabulated nH-T relation from ABL paper
    options['phys'] = physpar

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

    logger.init(level='debug', name='warning')
    
    parser = configparser.ConfigParser()
    config = parser.read(filename)

    for section in parser.sections():
        if section in options.keys():
            for key in parser[section]:
                if key in options[section].keys():
                    options[section][key] = option_types['{}:{}'.format(section, key)](parser[section][key])
                    logger.log('debug', "New setting for option {}:{} is {}".format(section, key, parser[section][key]), 'options')
                else:
                    logger.log('warning', "Input file option {:s} is invalid".format(key), 'options')
        else:
            logger.log('warning', "Input file section {:s} is invalid".format(section), 'options')

    return options

if __name__ == '__main__':
    default(True)
