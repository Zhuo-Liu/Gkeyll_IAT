import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import postgkyl as pg
import numpy as np
import adios as ad
import sys

simName = 'IAT_E1_wider'

fileName = {'field':'fieldEnergy','ionTemp':'ion_intM2Thermal'}

filePath = '../' + simName +'_'+fileName['ionTemp']+'.bp'
f = ad.file(filePath)
times = []
datas = []
for i in np.arange(0,1600):
    timeName = 'TimeMesh' + str(i)
    dataName = 'Data' + str(i)
    t = f[timeName][...]
    d = f[dataName][...]
    if np.isscalar(t):
        times.append(t)
        if np.isscalar(d):
            datas.append(d)
        else:
            datas.append(d[0])

    elif t.shape[0] != 1:
        continue
    else:
        times.append(t[0])
        datas.append(d[0][0])

savePath = './saved_data/' + fileName['ionTemp'] +'.txt'
savePatht = './saved_data/' + fileName['ionTemp'] +'_time.txt'
np.savetxt(savePatht,times)
np.savetxt(savePath,datas)


#plt.plot(times[:],datas[:])
#plt.yscale('log')
#plt.ylim(-10e-3,10e-7)
#plt.savefig('./plots/fieldEnergy.png')

