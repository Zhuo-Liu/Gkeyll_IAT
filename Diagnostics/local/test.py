import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import postgkyl as pg
import numpy as np
import adios as ad
import sys

f = ad.file('../IATwExt_2x2v_pert_fieldEnergy.bp')
times = []
datas = []
for i in np.arange(0,1400):
    timeName = 'TimeMesh' + str(i)
    dataName = 'Data' + str(i)
    t = f[timeName][...]
    d = f[dataName][...]
    if np.isscalar(t):
        times.append(t)
        datas.append(d[0])
    elif t.shape[0] != 1:
        continue
    else:
        times.append(t[0])
        datas.append(d[0][0])

plt.plot(times,datas)
plt.yscale('log')
plt.savefig('./fieldEnergy.png')

