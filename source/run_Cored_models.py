import numpy as np
import function_Cored

acore = 0.5  # Set the ratio of the core-to-scale radius

mn_mvir = 7.0
mx_mvir = 9.5
nummvir = 26

mn_reds = 3.1
mx_reds = 3.1
numreds = 1

mn_gast = 8000.0
mx_gast = 8000.0
numgast = 1

"""
mn_mvir = 8.5
mx_mvir = 8.5
nummvir = 1

mn_reds = 3.0
mx_reds = 3.0
numreds = 1

mn_gast = 8000.0
mx_gast = 8000.0
numgast = 1
"""

virialm = np.linspace(mn_mvir,mx_mvir,nummvir)
redshift = np.linspace(mn_reds,mx_reds,numreds)
gastemp = np.linspace(mn_gast,mx_gast,numgast)

#print virialm
#print redshift
#print gastemp

for i in range(nummvir):
	for j in range(numreds):
		print "#########################"
		print "#########################"
		print "  virialm  {0:d}/{1:d}".format(i+1,nummvir)
		print "  redshift {0:d}/{1:d}".format(j+1,numreds)
		print "#########################"
		print "#########################"
		for k in range(numgast):
			function_Cored.get_halo(10.0**virialm[i],redshift[j],gastemp[k],acore=acore)
