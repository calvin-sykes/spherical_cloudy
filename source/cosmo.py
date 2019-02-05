import numpy as np

def hubblepar(z, cosmopar):
    Ez = np.sqrt(cosmopar[2] + cosmopar[3] * ((1.0 + z) ** 3.0))
    return Ez * 100.0 * cosmopar[0]


def massconc_Ludlow16(mvir, cosmopar, redshift=0.0):
    """Ludlow+ 2016 appendix C"""
    hubble, Omega_b, Omega_l, Omega_m = cosmopar # These are the subscript 0 values!!!
    ## Step 1: calculate nu
    # Eq. (C12)
    fun_OmLz = lambda z: Omega_l / (Omega_l + Omega_m * (1. + z)**3.)
    # Eq. (C11)
    fun_Psi = lambda z: (1. - fun_OmLz(z))**(4. / 7.) - fun_OmLz(z) + (1. + (1. - fun_OmLz(z)) / 2.) * (1 + fun_OmLz(z) / 70.)
    OmLz = fun_OmLz(redshift)
    Ommz = 1 - OmLz
    # Eq. (C10)
    Dz = (Ommz / Omega_m) * (fun_Psi(0.) / fun_Psi(redshift)) / (1. + redshift)
    # Eq. (C9)
    xi = (1e10 / hubble) / mvir
    # Eq. (C8)
    sigma = Dz * 22.26 * xi**0.292 / (1. + 1.53 * xi**0.275 + 3.36 * xi**0.198)
    # text above Eq. (C1)
    nu = 1.686 / sigma
    ## Step 2: Evaluate c(nu)
    # Eqs. (C2-C6)
    c0 = 3.395 * (1. + redshift)**-0.215
    beta = 0.307 * (1. + redshift)**0.540
    gam1 = 0.628 * (1. + redshift)**-0.047
    gam2 = 0.317 * (1. + redshift)**-0.893
    a =  1. / (1. + redshift)
    nu0 = (4.135 - (0.564 / a) - (0.210 / a**2) + (0.0557 / a**3) -  0.00348 / a**4) / Dz
    # Step 3: Evaluate c(nu)
    # Eq. (C1)
    conc = c0 * (nu / nu0)**-gam1 * (1. + (nu / nu0)**(1./beta))**(-beta * (gam2 - gam1))
    return conc


def massconc_Bose16(mvir, cosmopar, redshift=0.0):
    """From Eq. 17 of Bose et. al. 2016"""
    mass_hm = 2e8 / cosmopar[0]
    gam1 = 60
    gam2 = 0.17
    fun_beta = lambda z: 0.026 * z - 0.04

    conc_CDM = massconc_Ludlow16(mvir, cosmopar, redshift)
    conc_WDM = conc_CDM * (1. + gam1 * mass_hm / mvir)**(-gam2) * (1. + redshift)**fun_beta(redshift)
    return conc_WDM


def get_cosmo(use="planck"):
    if use.lower() == "planck":
        hubble = 0.673
        Omega_b = 0.0491
        Omega_m = 0.315
        Omega_l = 1.0 - Omega_m
    elif use.lower() == "wmap7":
        hubble = 0.70
        Omega_b = 0.0469
        Omega_m = 0.27
        Omega_l = 1.0 - Omega_m
    elif use.lower() == "wmap9":
        hubble = 0.697
        Omega_b = 0.0461
        Omega_m = 0.282
        Omega_l = 1.0 - Omega_m    
    else:
        # Use Planck as default
        hubble = 0.673
        Omega_b = 0.0491
        Omega_m = 0.315
        Omega_l = 1.0-Omega_m
    return np.array([hubble, Omega_b, Omega_l, Omega_m])


if __name__ == "__main__":    
    import matplotlib.pyplot as plt

    Ms = np.logspace(7, 12, 100)
    zs = np.linspace(0, 3, 100)

    Mgrid, zgrid = np.meshgrid(Ms, zs)
    cgrid = np.zeros_like(Mgrid)
    
    for i in range(len(zs)):
        cgrid[i] = massconc_Ludlow16(Ms, get_cosmo(), zs[i])

    plt.figure()
    plt.xscale('log')
    plt.pcolormesh(Mgrid, zgrid, cgrid)
    plt.colorbar()
    plt.show()   
    
    cs_CDM = massconc_Ludlow16(Ms, get_cosmo(), 0)
    cs_WDM = massconc_Bose16(Ms, get_cosmo(), 0)
    
    plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    plt.plot(Ms, cs_CDM)
    plt.plot(Ms, cs_WDM)
    plt.show()

    plt.figure()
    plt.xscale('log')
    plt.plot(Ms, cs_WDM / cs_CDM)
    plt.show()
    
    
