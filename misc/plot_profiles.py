import numpy as np
from matplotlib import pyplot as plt

cmtopc = 3.24E-19

tdata = np.load('data/NFW_model_mass8p00_redshift3p10.npy')
print tdata.shape

lt= "-"
radius8p0 = tdata[:,0]*cmtopc
densitynH8p0 = tdata[:,2]
prof_temperature8p0 = tdata[:,3]
prof_pressure8p0 = tdata[:,4]
prof_YHI8p0 = tdata[:,5]
prof_YHeI8p0 = tdata[:,6]
prof_YHeII8p0 = tdata[:,7]
prof_HIcd8p0 = tdata[:,8]
prof_HeIcd8p0 = tdata[:,9]
prof_HeIIcd8p0 = tdata[:,10]
#plt.plot(radius8p0,np.log10(densitynH8p0),"k"+lt)
plt.plot(radius8p0,prof_YHI8p0,"r"+lt)
plt.plot(radius8p0,prof_YHeI8p0,"g"+lt)
plt.plot(radius8p0,prof_YHeII8p0,"b"+lt)

lt= "--"
tdata = np.load("data/NFW_model_mass8p10_redshift3p10.npy")
radius8p1 = tdata[:,0]*cmtopc
densitynH8p1 = tdata[:,2]
prof_temperature8p1 = tdata[:,3]
prof_pressure8p1 = tdata[:,4]
prof_YHI8p1 = tdata[:,5]
prof_YHeI8p1 = tdata[:,6]
prof_YHeII8p1 = tdata[:,7]
prof_HIcd8p1 = tdata[:,8]
prof_HeIcd8p1 = tdata[:,9]
prof_HeIIcd8p1 = tdata[:,10]
#plt.plot(radius8p1,np.log10(densitynH8p1),"k"+lt)
plt.plot(radius8p1,prof_YHI8p1,"r"+lt)
plt.plot(radius8p1,prof_YHeI8p1,"g"+lt)
plt.plot(radius8p1,prof_YHeII8p1,"b"+lt)

lt= ":"
tdata = np.load("data/NFW_model_mass8p20_redshift3p10.npy")
radius8p2 = tdata[:,0]*cmtopc
densitynH8p2 = tdata[:,2]
prof_temperature8p2 = tdata[:,3]
prof_pressure8p2 = tdata[:,4]
prof_YHI8p2 = tdata[:,5]
prof_YHeI8p2 = tdata[:,6]
prof_YHeII8p2 = tdata[:,7]
prof_HIcd8p2 = tdata[:,8]
prof_HeIcd8p2 = tdata[:,9]
prof_HeIIcd8p2 = tdata[:,10]
#plt.plot(radius8p2,np.log10(densitynH8p2),"k"+lt)
plt.plot(radius8p2,prof_YHI8p2,"r"+lt)
plt.plot(radius8p2,prof_YHeI8p2,"g"+lt)
plt.plot(radius8p2,prof_YHeII8p2,"b"+lt)

lt= "-."
tdata = np.load("data/NFW_model_mass8p30_redshift3p10.npy")
radius8p3 = tdata[:,0]*cmtopc
densitynH8p3 = tdata[:,2]
prof_temperature8p3 = tdata[:,3]
prof_pressure8p3 = tdata[:,4]
prof_YHI8p3 = tdata[:,5]
prof_YHeI8p3 = tdata[:,6]
prof_YHeII8p3 = tdata[:,7]
prof_HIcd8p3 = tdata[:,8]
prof_HeIcd8p3 = tdata[:,9]
prof_HeIIcd8p3 = tdata[:,10]
#plt.plot(radius8p3,np.log10(densitynH8p3),"k"+lt)
plt.plot(radius8p3,prof_YHI8p3,"r"+lt)
plt.plot(radius8p3,prof_YHeI8p3,"g"+lt)
plt.plot(radius8p3,prof_YHeII8p3,"b"+lt)

plt.show()

