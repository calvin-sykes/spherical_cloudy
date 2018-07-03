import numpy as np
import sys

import logger

class HaloModel:
    def __init__(self, virial_mass, baryon_frac, rho_crit, conc):
        self.mvir = virial_mass
        self.rvir = (3.0 * virial_mass / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
        self.rscale = self.rvir / conc
        self.baryfrac = baryon_frac
        self.rhods = 200.0 * rho_crit * conc**3 / self.fm(conc)

    def fm(self, x):
        raise NotImplementedError("fm is not defined for base class")


class NFWHalo(HaloModel):
    def __init__(self, virial_mass, baryon_frac, rho_crit, conc, **kwargs):
        HaloModel.__init__(self, virial_mass, baryon_frac, rho_crit, conc)
        self.name = 'NFW'

    def fm(self, x):
        return 3.0 * (np.log(1.0 + x) - x / (1.0 + x))


class BurkertHalo(HaloModel):
    def __init__(self, virial_mass, baryon_frac, rho_crit, conc, **kwargs):
        HaloModel.__init__(self, virial_mass, baryon_frac, rho_crit, conc)
        self.name = 'Burkert'

    def fm(self, x):
        return 1.5 * (0.5 * np.log(1.0 + x**2.0) + np.log(1.0+x) - np.arctan(x))


class CoredHalo(HaloModel):
    def __init__(self, virial_mass, baryon_frac, rho_crit, conc, acore):
        HaloModel.__init__(self, virial_mass, baryon_frac, rho_crit, conc)
        self.acore = acore
        self.name = 'Cored'

    def fm(self, x):
        return 3.0 * (x / ((self.acore-1.0) * (1.0 + x))
                      + (self.acore**2.0 * np.log(1.0 + x / self.acore)
                         + (1.0 - 2.0*self.acore) * np.log(1.0 + x)) / (1.0 - self.acore)**2.0 )


class PPHalo(HaloModel):
    def __init__(self, coldens, dens):
        self.cden = coldens
        self.density = dens
        self.name = 'PP'


func_map = {'NFW'     : NFWHalo,
            'Burkert' : BurkertHalo,
            'Cored'   : CoredHalo,
            'PP'      : PPHalo }


def make_halo(name, *args, **kwargs):
    try:
        return func_map[name](*args, **kwargs)
    except KeyError:
        logger.log('critical', "Halo model {} is invalid.".format(name))
        sys.exit(1)
