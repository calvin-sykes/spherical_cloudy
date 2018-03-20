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

bturb = 3.0
metals = 1.0E-3
ions = ['H I', 'D I', 'He I', 'He II']

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
HMscale = np.linspace(mn_HMscl,mx_HMscl,numHMscl)

# Load baryon fraction as a function of halo mass
halomass, barymass = np.loadtxt('data/baryfrac.dat', unpack=True)
baryfracvals = 10.0**barymass / 10.0**halomass
baryfrac = np.interp(virialm, halomass, baryfracvals)

def init_outdir(options):
    out_dir = 'output/' + options['run']['outdir'] + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # rewrite outdir to include path
    options['run']['outdir'] = out_dir

def init_log(options):
    if options['log']['file'] == 'none':
        logger.init(level=options['log']['level'])
    else:
        logpath = options['run']['outdir'] + options['log']['file']
        logger.init(level=options['log']['level'], filename=logpath) 

# Find the name of a previously written file, and initialise loop counters
# such that the run starts from reading back this file
def init_resume(options):
    wd = os.getcwd()
    out_path = options['run']['outdir']
    os.chdir(out_path)
    files = sorted(os.listdir('.'))
    files = list(filter(lambda fn: fn.endswith('.npy'), files))
    # return to working directory
    os.chdir(wd)
    where = options['run']['resume']
    if where == 'last':
        file_idx = len(files) - 1
    elif where == 'refine_last':
        file_idx = len(files) - 1
        options['refine'] = True
    elif where.isdigit():
        file_idx = int(where)
    else:
        logger.log('critical', "Couldn't understand resume command {}".format(where))
        sys.exit(1)
    if file_idx < 0:
        file_idx = len(files) + file_idx
    fname = out_path + files[file_idx]
    # if we are refining, want to repeat the loaded model not use it to run the next one
    if where == 'refine_last':
        model_idx = file_idx
    else:
        model_idx = file_idx + 1 # files are 0-indexed, models are 1-indexed
    smvir = model_idx % nummvir
    sHMscl = (model_idx // nummvir) % numHMscl
    sbary = (model_idx // (nummvir * numHMscl)) % numbary
    sreds = (model_idx // (nummvir * numHMscl * numbary))
    return (fname, smvir, sHMscl, sbary, sreds)

################
## START HERE ##
################

if __name__ == '__main__':
    # Load input file using first argument as filename
    try:
        input_file = sys.argv[1]
    except IndexError:
        print("Input file must be provided.")
        sys.exit(1)

    opt = options.read_options(input_file)

    # Whether to rerun the model to get better convergence
    # This will be set to True by the code whenever a model
    #   in which the neutral fraction exceeds 1/2 is run
    # The model will be recomputed using a finer radius
    #   interpolation about the ionised -> neutral transition region
    # Then this option is set back to False and the code continues with the next model
    opt['run']['refine'] = False

    # Check output directory exists
    init_outdir(opt)

    # copy input file to output directory for reference
    shutil.copy(input_file, opt['run']['outdir'])

    # Set up logging
    init_log(opt)

    logger.log('info', "Using input file {}".format(input_file))

    if opt['run']['resume'] != 'none':
        prev_fname, smvir, sHMscl, sbary, sreds = init_resume(opt)
    else:
        sreds = 0
        sbary = 0
        sHMscl = 0
        smvir = 0
        prev_fname = None

    # Get the working cosmology
    cosmopar = cosmo.get_cosmo()

    # Get some constants needed to define the halo model
    constants = constants.get()
    hztos = constants['hztos']
    Gcons = constants['Gcons']
    somtog = constants['somtog']
    hubpar = cosmo.hubblepar(redshift, cosmopar)
    rhocrit = 3.0*(hubpar*hztos)**2/(8.0*np.pi*Gcons)

    models = []

    # build list of parameters for each model to run
    for i in range(sreds, numreds):     
        for j in range(sbary, numbary):
            for k in range(sHMscl, numHMscl):
                for l in range(smvir, nummvir):
                    models.append((i, j, k, l))
    models.reverse()

    dryrun = False # set True to see what models will be run without actually running them

    try:
        while models:
            i, j, k, l = models.pop()
            logger.log('info', "###########################")
            logger.log('info', "###########################")
            logger.log('info', " virialm 10**{2:.2f} ({0:d}/{1:d})".format(
                l+1,nummvir, virialm[l]))
            logger.log('info', " redshift {2:.2f}     ({0:d}/{1:d})".format(
                k+1,numreds, redshift[k]))
            logger.log('info', " baryon scale {2:.2f} ({0:d}/{1:d})".format(
                j+1,numbary, baryscale[j]))
            logger.log('info', " UVB scale {2:.2f}    ({0:d}/{1:d})".format(
                i+1,numHMscl, HMscale[i]))
            logger.log('info', "###########################")
            logger.log('info', "###########################")
            concentration = cosmo.massconc_Prada12(10**virialm[l], cosmopar, redshift[k])
            model = halomodel.make_halo(opt['geometry']['profile'],
                                        10**virialm[l] * somtog,
                                        baryfrac[l] * baryscale[j],
                                        rhocrit,
                                        concentration,
                                        acore=opt['geometry']['acore'])

            # Let's go!
            if not dryrun:
                ok, res = gethalo.get_halo(model,redshift[k],gastemp,bturb,Hescale=1.0,metals=metals,cosmopar=cosmopar,ions=ions,prevfile=prev_fname,options=opt)
                if ok == True:
                    prev_fname = res
                    opt['run']['refine'] = False
                elif ok == 'needs_refine':
                    prev_fname = res
                    opt['run']['refine'] = True
                    logger.log('info', "High neutral hydrogen fraction detected. Running refinement model next.")
                    models.append((i, j, k, l))
                else:
                    # something went wrong with the model
                    logger.log('error', res)
                    # move onto next grid
                    # (keep popping elements till mass counter wraps back to 0)
                    while models and (models[-1][3] > l):
                        models.pop()

            # once a run over increasing halo masses is complete, clear the previous filename
            # if doing subsequent runs varying other parameters, don't want to load this run's output!
            if l == nummvir - 1 and opt['run']['refine'] == False:
                prev_fname = None
    except Exception:
        logger.log('critical', traceback.format_exc())
        sys.exit(1)
