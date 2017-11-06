import numpy as np
from matplotlib import pyplot as plt
data = np.loadtxt("HM12_UVB.dat")
reds12 = data[0,:]
wave12, jnu12 = data[1:,0], data[1:,1:]

data05 = open("haardt_madau_galaxy05.dat",'r').readlines()

# Read in the 2005 HM background
cntr = 0
prelines=[]
reds05 = np.array([])
wave05 = np.array([])
jnu05  = []
for i in range(len(data05)):
	if data05[i].strip()[0] == '#':
		prelines += data05[i]
		continue
	if cntr == 0:
		prelines += data05[i]
		cntr += 1
		continue
	elif cntr == 1:
		wavetxt = data05[i]
		redspl = data05[i].split()
		for j in range(1,len(redspl)):
			rspl = redspl[j].split("=")
			reds05 = np.append(reds05,float(rspl[1]))
			jnu05.append(np.array([]))
		cntr += 1
	else:
		datspl = data05[i].split()
		wave05 = np.append(wave05,float(datspl[0]))
		for j in range(1,len(datspl)):
			jnu05[j-1] = np.append(jnu05[j-1],float(datspl[j]))
		cntr += 1

#idx = 10
#plt.plot(np.log10(wave05),np.log10(jnu05[idx]),'r-')
#plt.plot(np.log10(wave12),np.log10(jnu12[:,idx]),'g-')
#plt.show()

# Now interpolate the 2012 background to the 2005 format
# Match the first 50 redshifts from the 2012 data
fin_reds = reds05.copy()
for r in range(50):
	jnu_new = np.interp(wave05,wave12,jnu12[:,r])
	wh = np.where(wave05>np.max(wave12))
	if np.size(wh[0]) != 0:
		print "adjusting redshift", r
		jnu_new[wh] = jnu05[r][wh]

# Now write the data:
outfile = open("haardt_madau_galaxy12.dat",'w')
for i in prelines: outfile.write(i)
outfile.write(wavetxt)
for w in range(wave05.size):
	outstr = "{0:.7f}".format(wave05[w])
	for i in range(len(jnu05)):
		outstr += "\t{0:4.3E}".format(jnu05[i][w])
	outfile.write(outstr+"\n")