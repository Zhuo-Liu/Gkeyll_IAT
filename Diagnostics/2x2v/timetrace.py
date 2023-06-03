import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import postgkyl as pg
import numpy as np
import adios as ad
import sys
#sys.path.insert(0, '/home/zhuol/bin/gkyl-python/pgkylLiu/2x2v/')
sys.path.insert(0, '/global/u2/z/zliu1997/bin/gkeyl_plot/2x2v/')
import pgkylUtil as pgu

dataDir = '../'
outDir = './saved_data/'
simName = 'IAT_E2'

nFrames = 1+pgu.findLastFrame(dataDir+simName+'_field_','bp')
pgu.checkMkdir(outDir)

fileName = {'field':'fieldEnergy','ionTemp':'ion_intM2Thermal','elcTemp':'elc_intM2Thermal','elcJ':'elc_intM1i','ionJ':'ion_intM1i'}

#fileName = {'field':'fieldEnergy','ionTemp':'ion_intM2Thermal','elcTemp':'elc_intM2Thermal'}

fileName_2 = {'ionTemp':'ion_M2Thermal', 'ionFlow':'ion_M2Flow','elcTemp':'elc_M2Thermal', 'elcFlow':'elc_M2Flow'}

def new():
    for name in fileName_2:
        print(fileName_2[name])
        datas = []
        
        for i in range(0,450):
            filePath = '../' + simName +'_'+fileName_2[name]+'_'+str(i)+'.bp'
            data = np.array(pgu.getInterpData(filePath,2,'ms'))
            datas.append(data)

        datas = np.array(data)
        savePath = outDir + fileName_2[name] +'.npz'
        np.save(savePath,datas)


def output(fieldEnergyonly=False):
    if fieldEnergyonly:
        filePath = '../' + simName +'_'+'fieldEnergy'+'.bp'
        f = ad.file(filePath)
        times = []
        datas = []
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
        savePath = outDir + 'fieldEnergy' +'.txt'
        savePatht = outDir + 'fieldEnergy' +'_time.txt'
        np.savetxt(savePatht,times)
        np.savetxt(savePath,datas)
    else:
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

def make_plots(cutoff=0,fieldEnergyonly=False):
    if fieldEnergyonly:
        fieldEnergy = np.loadtxt('./saved_data/fieldEnergy.txt')
        fieldEnergy_time = np.loadtxt('./saved_data/fieldEnergy_time.txt')

        cut_field = int((1-cutoff)*(fieldEnergy_time.shape[0]))
        plt.plot(fieldEnergy_time[0:cut_field],fieldEnergy[0:cut_field])
        plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=30)
        plt.yscale('log')
        plt.tick_params(labelsize = 26)

        plt.savefig('fieldEnergy.png')
        plt.close()

    else:
        elcTemp = np.loadtxt('./saved_data/elc_intM2Thermal.txt')
        elcTemp_time = np.loadtxt('./saved_data/elc_intM2Thermal_time.txt')
        if elcTemp.shape != elcTemp_time.shape:
            print('warning')

        ionTemp = np.loadtxt('./saved_data/ion_intM2Thermal.txt')
        ionTemp_time = np.loadtxt('./saved_data/ion_intM2Thermal_time.txt')
        if ionTemp.shape != ionTemp_time.shape:
            print('warning')

        fieldEnergy = np.loadtxt('./saved_data/fieldEnergy.txt')
        fieldEnergy_time = np.loadtxt('./saved_data/fieldEnergy_time.txt')

        elcCurrent = np.loadtxt('./saved_data/elc_intM1i.txt')
        ionCurrent = -np.loadtxt('./saved_data/elc_intM1i.txt')
        Current_time = np.loadtxt('./saved_data/elc_intM1i_time.txt')

        fig, axs = plt.subplots(2,2,figsize=(30, 25), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace =.1)

        cut = int((1-cutoff)*(ionTemp_time.shape[0]))
        axs[0,0].plot(ionTemp_time[0:cut],ionTemp[0:cut]/ionTemp[0])
        axs[0,0].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=30)
        axs[0,0].set_ylabel(r'$T_i/T_{i0}$',fontsize=30)
        axs[0,0].tick_params(labelsize = 26)

        axs[0,1].plot(elcTemp_time[0:cut],elcTemp[0:cut])
        axs[0,1].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=30)
        axs[0,1].set_ylabel(r'$T_e/T_{e0}$',fontsize=30)
        axs[0,1].tick_params(labelsize = 26)

        cut_field = int((1-cutoff)*(fieldEnergy_time.shape[0]))
        axs[1,0].plot(fieldEnergy_time[0:cut_field],fieldEnergy[0:cut_field])
        axs[1,0].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=30)
        axs[1,0].set_yscale('log')
        axs[1,0].tick_params(labelsize = 26)

        axs[1,1].plot(Current_time[0:cut],elcCurrent[0:cut])
        axs[1,1].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=30)
        axs[1,1].set_ylabel(r'$J_z$',fontsize=30)
        axs[1,1].tick_params(labelsize = 26)

        fig.tight_layout()
        plt.savefig('./trace.png', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    output()
    make_plots(0.2)

