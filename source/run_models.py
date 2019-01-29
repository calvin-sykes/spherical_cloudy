import numpy as np
import os
import shutil
import sys
import traceback

import halomodel
import gethalo
import options
import constants
import cosmo
import logger
from jug import Task, is_jug_running

# Check the output directory exists, create it if not
def init_outdir(options):
    out_dir = '/cosma/home/dp004/dc-syke1/data/spherical_cloudy/output/' + options['run']['outdir'] + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # rewrite outdir to include path
    options['run']['outdir'] = out_dir


# Set up the main logging sink
def init_log(options):
    if options['log']['file'] == 'none':
        logger.init(level=options['log']['level'])
    else:
        logpath = options['run']['outdir'] + options['log']['file']
        logger.init(level=options['log']['level'], filename=logpath) 
    return


def init_resume(options, dims):
    where = options['run']['resume']
    if where == 'none':
        fname = None
        start_idxs = [0] * len(dims)
    else:
        wd = os.getcwd()
        out_path = options['run']['outdir']
        os.chdir(out_path)
        files = sorted(os.listdir('.'))
        files = list(filter(lambda fn: fn.endswith('.npy'), files))
        files = list(filter(lambda fn: '_rates' not in fn and '_heat_cool' not in fn, files))
        # return to working directory
        os.chdir(wd)
        where = options['run']['resume']
        if where.lower() == 'last':
            file_idx = len(files) - 1
        else:
            try:
                file_idx = files.index(where)
            except ValueError:
                logger.log('critical', "Couldn't understand resume command {}".format(where))
                sys.exit(1)
        fname = out_path + files[file_idx]
        model_idx = file_idx + 2 # files are 0-indexed, models are 1-indexed
                                 # plus an extra 1 to get the *next* model to do
        # Now, work out the indexes through each of the arrays
        # that results in resuming at the right place
        start_idxs = []
        for i in range(len(dims)):
            divisor = 1
            for j in range(0, i):
                divisor *= dims[j]
            start_idxs.append((model_idx // divisor) % dims[i])
    return fname, start_idxs


def init_grid(options):
    gridparams = dict({})
    try:
        for param, stmnt in options['grid'].iteritems():
            gridparams[param] = eval(stmnt)
    except:
        logger.log('critical', "Failed to eval grid specification {}".format(stmnt))
        logger.log('critical', traceback.format_exc())
        sys.exit(1)
    return gridparams


def run_grid(opt, cosmopar, ions, dryrun=False):
    # Get arrays defining the grid of models to run
    gridparams = init_grid(opt)
    virialm   = gridparams['virialm'  ]
    redshift  = gridparams['redshift' ]
    baryscale = gridparams['baryscale']
    HMscale   = gridparams['radscale' ]
    
    nummvir, numreds, numbary, numHMscl = map(len, [virialm, redshift, baryscale, HMscale])

    prev_fname, (smvir, sHMscl, sbary, sreds) = init_resume(opt, [nummvir, numreds, numbary, numHMscl])

    # build list of parameters for each model to run
    models = []
    for i in range(sreds, numreds):     
        for j in range(sbary, numbary):
            for k in range(sHMscl, numHMscl):
                for l in range(smvir, nummvir):
                    models.append((i, j, k, l))
    models.reverse()

    # Load baryon fraction as a function of halo mass
    halomass, barymass = np.loadtxt('data/baryfrac.dat', unpack=True)
    baryfracvals = 10.0**barymass / 10.0**halomass
    baryfrac = np.interp(virialm, halomass, baryfracvals)

    # Get some constants needed to define the halo model
    const  = constants.get()
    hztos  = const['hztos' ]
    Gcons  = const['Gcons' ]
    somtog = const['somtog']
    
    while models:
        i, j, k, l = models.pop()
        logger.log('info', "###########################")
        logger.log('info', "###########################")
        logger.log('info', " virialm 10**{2:.3f}  ({0:d}/{1:d})".format(l+1,nummvir, virialm[l]))
        logger.log('info', " redshift {2:.2f}     ({0:d}/{1:d})".format(k+1,numreds, redshift[k]))
        logger.log('info', " baryon scale {2:.2f} ({0:d}/{1:d})".format(j+1,numbary, baryscale[j]))
        logger.log('info', " UVB scale {2:.2f}    ({0:d}/{1:d})".format(i+1,numHMscl, HMscale[i]))
        logger.log('info', "###########################")
        logger.log('info', "###########################")
        if opt['geometry']['concrel'] == "Prada":
            concentration = cosmo.massconc_Prada12(10**virialm[l], cosmopar, redshift[k])
        elif opt['geometry']['concrel'] == "Ludlow":
            concentration = cosmo.massconc_Ludlow16(10**virialm[l], cosmopar, redshift[k])
        elif opt['geometry']['concrel'] == "Bose":
            concentration = cosmo.massconc_Bose16(10**virialm[l], cosmopar, redshift[k])
        else:
            raise ValueError("Unknown concentration relation")

        hubpar = cosmo.hubblepar(redshift[k], cosmopar)
        rhocrit = 3.0*(hubpar*hztos)**2/(8.0*np.pi*Gcons)
        model = halomodel.make_halo(opt['geometry']['profile'],
                                    10**virialm[l] * somtog,
                                    baryfrac[l] * baryscale[j],
                                    rhocrit,
                                    concentration,
                                    acore=opt['geometry']['acore'])
        # Let's go!
        if not dryrun:
            ok, res = gethalo.get_halo(model, redshift[k], cosmopar, ions, prevfile=prev_fname, options=opt)
            if ok == True:
                # model complete
                prev_fname = res
            else:
                # something went wrong with the model
                logger.log('error', res)
                # move onto next grid
                # (keep popping elements till mass counter wraps back to 0)
                while models and (models[-1][3] > l):
                    models.pop()
        # once a run over increasing halo masses is complete, clear the previous filename
        # if doing subsequent runs varying other parameters, don't want to load this run's output!
        if l == nummvir - 1:
            prev_fname = None
    return

# PP models don't load each other
# so they can all be run in parallel
def run_grid_PP(opt, cosmopar, ions, dryrun=False):
    # Get arrays defining the grid of models to run
    gridparams = init_grid(opt)
    density  = gridparams['pp_dens'  ]
    coldens  = gridparams['pp_cdens' ]
    redshift = gridparams['redshift' ]

    numdens, numcdens, numreds = map(len, [density, coldens, redshift])

    prev_fname, (sdens, scdens, sreds) = init_resume(opt, [numdens, numcdens, numreds])

    # build list of parameters for each model to run
    models = []
    for i in range(sdens, numdens):
        for j in range(scdens, numcdens):
            for k in range(sreds, numreds):
                models.append((i, j, k))
    models.reverse()

    useMP = opt['run']['pp_para']
    if useMP:
        tasks = []

    while models:
        i, j, k = models.pop()
        model = halomodel.make_halo(opt['geometry']['profile'],
                                    10.0**coldens[j],
                                    10.0**density[i])
        # Let's go!
        if not dryrun:
            if useMP:
                tasks.append(Task(gethalo.get_halo, hmodel=model, redshift=redshift[k], cosmopar=cosmopar,
                                  ions=ions, prevfile=None, options=opt))
            else:
                logger.log('info', "###########################")
                logger.log('info', "###########################")
                logger.log('info', " H density 10**{2:.1f} ({0:d}/{1:d})".format(i+1,numdens, density[i]))
                logger.log('info', " H column  10**{2:.1f} ({0:d}/{1:d})".format(j+1,numcdens, coldens[j]))
                logger.log('info', " redshift {2:.2f}      ({0:d}/{1:d})".format(k+1,numreds, redshift[k]))
                logger.log('info', "###########################")
                logger.log('info', "###########################")
                ok, res = gethalo.get_halo(model, redshift[k], cosmopar, ions, prevfile=None, options=opt)
                if ok != True:
                    # something went wrong with the model
                    logger.log('error', res)
    return


################
## START HERE ##
################

if __name__ == '__main__' or is_jug_running():
    # Load input file using first argument as filename
    try:
        input_file = sys.argv[1]
    except IndexError:
        print("Input file must be provided")
        sys.exit(1)

    # Check input file exists
    if not os.path.isfile(input_file):
        print("Input file does not exist")
        sys.exit(1)

    opt = options.read_options(input_file)

    # second argument disables logging to file if given
    try:
        if sys.argv[2] == 'redirect':
            opt['log']['file'] = 'none'
        opt['log']['level'] = sys.argv[3]
    except IndexError:
        pass
    
    # if running from interactive shell disable Jug parallel processing
    if not is_jug_running():
        opt['run']['pp_para'] = False

    # Check output directory exists
    init_outdir(opt)

    # copy input file to output directory for reference
    shutil.copy(input_file, opt['run']['outdir'])

    # Set up logging
    init_log(opt)

    logger.log('info', "Using input file {}".format(input_file))

    # Set the ions used in the models
    ions = ['H I', 'D I', 'He I', 'He II']

    # Get the working cosmology
    cosmopar = cosmo.get_cosmo("Planck")

    # Do the thing
    # Plane parallel is a special case, because the grid parameters are different
    try:
        if opt['geometry']['profile'] in {'NFW', 'Burkert', 'Cored'}:
            run_grid(opt, cosmopar, ions)
        elif opt['geometry']['profile'] == 'PP':
            run_grid_PP(opt, cosmopar, ions)
    except Exception:
        logger.log('critical', traceback.format_exc())
        sys.exit(1)
    
