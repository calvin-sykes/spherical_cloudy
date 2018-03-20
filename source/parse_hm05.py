import numpy as np
from matplotlib import pyplot as plt

numrows_pre = 10
cntrows_pre = 0

numrows_z = 5
cntrows_z = 0

numrows_lambda = 54
cntrows_lambda = 0

numrows_data = 44
cntrows_data = 0

cntfill = 0

redshift = []
wlen = []

tmpdata = []

with open('hm05_galaxy.ascii', 'r') as hm05:
    for line in hm05:
        if line.strip()[0] is '#':
            continue
        if cntrows_pre < numrows_pre:
            if cntrows_pre == 4:
                num_redshift = int(line)
            if cntrows_pre == 5:
                num_lambda = int(line)
                data = np.zeros((num_redshift, num_lambda))
            cntrows_pre += 1
            continue
        elif cntrows_z < numrows_z:
            redshift.extend(map(float, line.split()))
            cntrows_z += 1
            continue
        elif cntrows_lambda < numrows_lambda:
            wlen.extend(map(float, line.split()))
            cntrows_lambda += 1
            continue
        else:
            tmpdata.extend(map(float, line.split()))
            cntrows_data += 1
            if cntrows_data == numrows_data:
                data[cntfill] = tmpdata
                cntfill += 1
                cntrows_data = 0
                tmpdata = []

with open('HM05_UVB.dat', 'w') as svfile:
    for z in redshift:
        svfile.write(str(z))
        svfile.write('\t')
    svfile.write('\n')
    for wl_idx, wl in enumerate(wlen):
        svfile.write(str(wl))
        svfile.write('\t')
        for val in data[:, wl_idx]:
            svfile.write(str(val))
            svfile.write('\t')
        svfile.write('\n')

    
#[plt.plot(np.log10(wlen), np.log10(data[i]), label=redshift[i]) for i in range(0, num_redshift, 5)]
#plt.legend()
#plt.show()
