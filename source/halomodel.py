import cython_halo as chalo

import sys
import logger
import numpy as np

init_map = {'NFW'     : chalo.NFWHalo,
            'Burkert' : chalo.BurkertHalo,
            #'Cored'   : chalo.CoredHalo,
            'PP'      : chalo.PPHalo }


def make_halo(name, *args, **kwargs):
    """
    Return a halo model object of type 'name' (i.e. NFW or Burkert).
    Args are (virial_mass, baryon_frac, rho_crit, conc)
    Attributes are accessible from Python but mass function isn't.
    Use the helper function halo_model.fm() to get this.
    """
    try:
        return init_map[name](*args, **kwargs)
    except KeyError:
        logger.log('critical', "Halo model {} is invalid.".format(name))
        sys.exit(1)


def fm(hm, x):
    """
    Evaluates the (C-implemented) mass function and returns the result to Python code.
    Args are a HaloModel object and the value x = r/r_s (or arraylike of values) to evaluate f_M at.
    """
    if hasattr(x, '__iter__'):
        ret = np.zeros_like(x)
        for i in range(ret.shape[0]):
            ret[i] = chalo.get_fm(hm, x[i])
        return ret
    else:
        return chalo.get_fm(hm, x)
