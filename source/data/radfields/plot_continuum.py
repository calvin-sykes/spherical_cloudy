import numpy as np
from matplotlib import pyplot as plt

nu, fnu = np.loadtxt("test_continuum.dat", unpack=True)
nu2, fnu2 = np.loadtxt("test_continuum2.dat", unpack=True)
nu3, fnu3 = np.loadtxt("test_continuum3.dat", unpack=True)

plt.plot(np.log10(nu),np.log10(fnu),'r-')
plt.plot(np.log10(nu2),np.log10(fnu2),'g-')
plt.plot(np.log10(nu3),np.log10(fnu3),'b-')
plt.show()