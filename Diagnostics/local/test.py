import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import postgkyl as pg
import numpy as np
import adios as ad
import sys
sys.path.insert(0, '/home/zhuol/bin/gkyl-python/pgkylLiu/2x2v/')
#sys.path.insert(0, '/global/u2/z/zliu1997/bin/gkeyl_plot/2x2v/')
import pgkylUtil as pgu

dataDir = '../'
outDir = './saved_data/'
simName = 'IAT_E2'

nFrames = 1+pgu.findLastFrame(dataDir+simName+'_field_','bp')
pgu.checkMkdir(outDir)


fileName = {'field':'fieldEnergy','ionTemp':'ion_intM2Thermal','elcTemp':'elc_intM2Thermal','elcJ':'elc_intM1i','ionJ':'ion_intM1i'}

#fileName = {'field':'fieldEnergy','ionTemp':'ion_intM2Thermal','elcTemp':'elc_intM2Thermal'}

for name in fileName:
    print(fileName[name])
    filePath = '../' + simName +'_'+fileName[name]+'.bp'
    f = ad.file(filePath)
    times = []
    datas = []
    if name == 'field':
        for i in np.arange(0,nFrames):
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
            elif t.shape[0] == 1:
                times.append(t[0])
                datas.append(d[0][0]+d[0][1])
            elif t.shape[0] == 2:
                times.append(t[0])
                times.append(t[1])
                datas.append(d[0][0]+d[0][1])
                datas.append(d[1][0]+d[1][1])
            else:
                continue
    else:
        for i in np.arange(0,nFrames):
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
            elif t.shape[0] == 1:
                times.append(t[0])
                datas.append(d[0][0])
                if d.ndim:
                    datas.append(d[0][0])
                else:
                    datas.append(d[0])
            elif t.shape[0] == 2:
                times.append(t[0])
                times.append(t[1])
                if d.ndim == 2:
                    datas.append(d[0][0])
                    datas.append(d[1][0])
                else:
                    datas.append(d[0])
                    datas.append(d[1])
            else:
                continue        
    savePath = outDir + fileName[name] +'.txt'
    savePatht = outDir + fileName[name] +'_time.txt'
    np.savetxt(savePatht,times)
    np.savetxt(savePath,datas)


#plt.plot(times[:],datas[:])
#plt.yscale('log')
#plt.ylim(-10e-3,10e-7)
#plt.savefig('./plots/fieldEnergy.png')

