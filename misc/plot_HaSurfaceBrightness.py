import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import cumtrapz
import plotting_routines as pr

# radius (pc), temperature (K), prof_density (cm^-3), prof_coldens (cm^-2)
ions = ["H I", "D I", "He I", "He II"]
prim_He = 1.0/12.0

# Load the data
fname = "NFW_mass8d84_redshift0d00_HMscalep0d00_1000-30.npy"
data = np.load("../version4/output_z0UVB/"+fname)

numarr = data.shape[1]

strt = 4
idx = dict({})
idx["voldens"] = dict({})
idx["coldens"] = dict({})
for i in range((numarr-strt)/2):
    idx["voldens"][ions[i]] = strt + i
    idx["coldens"][ions[i]] = strt + i + (numarr-strt)/2

# Extract radius information
radius = data[:,0]
prof_temperature = data[:,1]
prof_densitynH = data[:,2]
prof_HaSB = data[:,3]  # photons /cm^2 / s

#DIHI_fiducial = np.log10(fiducial[:,idx["coldens"]["D I"]]/fiducial[:,idx["coldens"]["H I"]])

fig = plt.figure(figsize=(8, 6.8))
ax1 = fig.add_subplot(111)
ax1.locator_params(nbins=7)
ax1.minorticks_on()
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.yaxis.set_major_formatter(FormatStrFormatter(r'$%.1f$'))
ax1.set_xlabel(r"Impact Parameter (pc)")
ax1.set_ylabel(r"H$\alpha$ SB")
ax1.plot(radius, prof_HaSB,'k-')
ax1.set_xlim([0.0,600.0])
#ax1.set_ylim([17.0,21.1])

# Finally, draw the plot
fig.tight_layout()
plt.savefig('HaSB.pdf')
plt.show()