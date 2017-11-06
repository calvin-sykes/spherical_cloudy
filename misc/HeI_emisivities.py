import pdb
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

data = np.loadtxt("data/HeI_Porter2013.dat")
datarsh = data.reshape((14,data.shape[0]/14,data.shape[1]))
#pdb.set_trace()

temp = datarsh[0,:,0]
dens = datarsh[:,0,1]
labels = ["2945A", "3188A", "3614A", "3889A", "3965A", "4026A", "4121A", "4388A", "4438A", "4471A", "4713A", "4922A", "5016A", "5048A", "5876A", "6678A", "7065A", "7281A", "9464A", "10830A", "11013A", "11969A", "12527A", "12756A", "12785A", "12790A", "12846A", "12968A", "12985A", "13412A", "15084A", "17003A", "18556A", "18685A", "18697A", "19089A", "19543A", "20427A", "20581A", "20602A", "21118A", "21130A", "21608A", "21617A"]

colormap = cm.Spectral_r
normalize = mcolors.Normalize(vmin=dens.min(), vmax=dens.max())

ndens = 4
#ndens = dens.size
for i in range(len(labels)):
	plt.subplot(7,7,i+1)
	plt.title(labels[i])
	for d in range(ndens):
		plt.plot(temp, datarsh[d,:,2+i], color=colormap(normalize(dens[d])))
#plt.colorbar()
#plt.show()

plt.clf()
normalize = mcolors.Normalize(vmin=0, vmax=len(labels)-1)
mtemp = np.linspace(1000, 40000.0, 1000)
for i in range(len(labels)):
	coeff = np.polyfit(np.log10(temp), datarsh[0,:,2+i], 2)
	model = np.polyval(coeff, np.log10(mtemp))
	plt.plot(temp, datarsh[0,:,2+i], color=colormap(normalize(i)), label=labels[i])
	plt.plot(mtemp, model, 'r--')
plt.legend()
plt.show()

#3889, 5876, 10830