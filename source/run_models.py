import numpy as np
import glob
import os
import shutil
import sys
import traceback

try:
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
except ImportError:
    print("Warning: importing MPI4py failed.\nMPI-parallelised plane parallel runs will fail.")

import halomodel
import gethalo
import options
import constants
import cosmo
import logger

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
        model_idx = file_idx + 1 # plus an extra 1 to get the *next* model to do
        # Now, work out the indexes through each of the arrays
        # that results in resuming at the right place
        start_idxs = []
        for i in range(len(dims)):
            divisor = 1
            for j in range(0, i):
                divisor *= dims[j]
                #print(j, divisor)
            start_idxs.append((model_idx // divisor) % dims[i])
    return fname, start_idxs


def init_grid(options):
    gridparams = dict({})
    try:
        for param, stmnt in options['grid'].iteritems():
            if param == 'cd_target': continue
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
    
    nummvir, numreds, numbary = map(len, [virialm, redshift, baryscale])

    prev_fname, (smvir, sbary, sreds) = init_resume(opt, [nummvir, numreds, numbary])

    # build list of parameters for each model to run
    models = []
    for i in range(sreds, numreds):     
        for j in range(sbary, numbary):
            for k in range(smvir, nummvir):
                models.append((i, j, k))
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
        j, k, l = models.pop()
        logger.log('info', "###########################")
        logger.log('info', "###########################")
        logger.log('info', " virialm 10**{2:.3f}  ({0:d}/{1:d})".format(l+1,nummvir, virialm[l]))
        logger.log('info', " redshift {2:.2f}     ({0:d}/{1:d})".format(k+1,numreds, redshift[k]))
        logger.log('info', " baryon scale {2:.2f} ({0:d}/{1:d})".format(j+1,numbary, baryscale[j]))
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
        hmodel = halomodel.make_halo(opt['geometry']['profile'],
                                    10**virialm[l] * somtog,
                                    baryfrac[l] * baryscale[j],
                                    rhocrit,
                                    concentration)
                                    #acore=opt['geometry']['acore'])
        # Let's go!
        if not dryrun:
            ok, res = gethalo.get_halo(hmodel, redshift[k], cosmopar, ions, prevfile=prev_fname, options=opt)
            if ok == True:
                # model complete, keep going
                prev_fname = res
            else:
                # column density target reached
                iterate_cden_target(opt, res, (virialm[l], virialm[l-1], redshift[k], baryscale[j]), prev_fname)

        # once a run over increasing halo masses is complete, clear the previous filename
        # if doing subsequent runs varying other parameters, don't want to load this run's output!
        if l == nummvir - 1:
            prev_fname = None
    return

def find_mass(fn):
    i1 = fn.find('mass')
    i2 = fn.find('_', i1)
    return fn[i1+4:i2].replace('d', '.')

def find_cden(fn, idx):
    return np.log10(np.load(fn)['cden'][...,idx].max())

#import matplotlib.pyplot as plt

from scipy.interpolate import interp1d as int1d
def iterate_cden_target(opt, res, params, fname_last):
    # linearly interpolate on achieved column densities to get mass we should run next time
    fname_this = res
    mvir_this, mvir_last, redshift, baryscale = params
    ionidx = ions.index('H I')
    
    iter_attempts = sorted(glob.glob(opt['run']['outdir'] + 'iter*'))
    if len(iter_attempts) < 2:
        max_cden_last = np.log10(np.load(fname_last)['cden'][...,ionidx].max())
        max_cden_this = np.log10(np.load(fname_this)['cden'][...,ionidx].max())
        grad = (mvir_this - mvir_last) / (max_cden_this - max_cden_last)
        c = mvir_this - grad * max_cden_this
        mvir_next = grad * opt['grid']['cd_target'] + c
        if np.abs(max_cden_last - opt['grid']['cd_target']) < np.abs(max_cden_this - opt['grid']['cd_target']):
            #print("swapping")
            fname_this = fname_last
    else:
        masses = np.fromiter(map(find_mass, iter_attempts), dtype=np.float, count=len(iter_attempts))
        coldens = np.fromiter(map(lambda fn: find_cden(fn, ionidx), iter_attempts), dtype=np.float, count=len(iter_attempts))
        #mvir_next = np.interp(opt['grid']['cd_target'], coldens, masses)
        mvir_next = int1d(coldens, masses, fill_value='extrapolate')(opt['grid']['cd_target'])
        fname_this = iter_attempts[np.argmin(np.abs(coldens - opt['grid']['cd_target']))]
        #plt.figure()
        #plt.plot(masses, coldens, 'bo-')
        #plt.plot(mvir_next, opt['grid']['cd_target'], 'rx')
        #plt.show()

    #print(mvir_last, mvir_this, mvir_next)
    #print(max_cden_last, max_cden_this)

    logger.log('info', "###########################")
    logger.log('info', "###########################")
    logger.log('info', "Iterating to reach target N_HI")
    logger.log('info', " virialm 10**{:.6f}".format(mvir_next))
    logger.log('info', "###########################")
    logger.log('info', "###########################")

    # Load baryon fraction as a function of halo mass
    halomass, barymass = np.loadtxt('data/baryfrac.dat', unpack=True)
    baryfracvals = 10.0**barymass / 10.0**halomass
    baryfrac = np.interp(mvir_next, halomass, baryfracvals)

    # Get some constants needed to define the halo model
    const  = constants.get()
    hztos  = const['hztos' ]
    Gcons  = const['Gcons' ]
    somtog = const['somtog']

    if opt['geometry']['concrel'] == "Prada":
        concentration = cosmo.massconc_Prada12(10**mvir_next, cosmopar, redshift)
    elif opt['geometry']['concrel'] == "Ludlow":
        concentration = cosmo.massconc_Ludlow16(10**mvir_next, cosmopar, redshift)
    elif opt['geometry']['concrel'] == "Bose":
        concentration = cosmo.massconc_Bose16(10**mvir_next, cosmopar, redshift)
    else:
        raise ValueError("Unknown concentration relation")

    hubpar = cosmo.hubblepar(redshift, cosmopar)
    rhocrit = 3.0*(hubpar*hztos)**2/(8.0*np.pi*Gcons)
    hmodel = halomodel.make_halo(opt['geometry']['profile'],
                                 10**mvir_next * somtog,
                                 baryfrac * baryscale,
                                 rhocrit,
                                 concentration)

    ok, res = gethalo.get_halo(hmodel, redshift, cosmopar, ions, prevfile=fname_this, options=opt, prefix='iter_')
    max_cden_next = np.log10(np.load(res)['cden'][...,ionidx].max())
    if np.abs(max_cden_next - opt['grid']['cd_target']) < 0.001:
        logger.log('info', "GOOD ENOUGH")
        iter_files = glob.glob(opt['run']['outdir'] + 'iter*.npy')
        iter_files.remove(res)
        logger.log('info', "Cleaning up")
        logger.log('info', "Removing intermediate files:")
        for f in iter_files:
            logger.log('info', f)
            os.remove(f)
        os.rename(res, res.replace('iter_', ''))
        sys.exit()
    else:
        iterate_cden_target(opt, res, (mvir_next, mvir_this, redshift, baryscale), fname_this)

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
        hmodel = halomodel.make_halo(opt['geometry']['profile'],
                                    10.0**coldens[j],
                                    10.0**density[i])
        # Let's go!
        if not dryrun:
            if useMP:
                tasks.append((hmodel, redshift[k], cosmopar, ions, None, opt))
            else:
                logger.log('info', "###########################")
                logger.log('info', "###########################")
                logger.log('info', " H density 10**{2:.1f} ({0:d}/{1:d})".format(i+1,numdens, density[i]))
                logger.log('info', " H column  10**{2:.1f} ({0:d}/{1:d})".format(j+1,numcdens, coldens[j]))
                logger.log('info', " redshift {2:.2f}      ({0:d}/{1:d})".format(k+1,numreds, redshift[k]))
                logger.log('info', "###########################")
                logger.log('info', "###########################")
                ok, res = gethalo.get_halo(hmodel, redshift[k], cosmopar, ions, prevfile=None, options=opt)
                if ok != True:
                    # something went wrong with the model
                    logger.log('error', res)
    # If using MPI parallelism, dispatch the tasks to worker processes
    if useMP:
        with MPIPoolExecutor() as executor:
            executor.starmap(gethalo.get_halo, tasks, chunksize=10)
    return


################
## START HERE ##
################

if __name__ == '__main__':
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
    
