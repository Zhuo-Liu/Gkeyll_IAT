from attr import field
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from scipy.special import gamma

def make_time_traces_plot(name_list,timecut, cutoff=0):
    elcTemp_list = []
    elcTemp_time_list = []
    ionTemp_list = []
    ionTemp_time_list = []
    fieldEnergy_list = []
    fieldEnergy_time_list = []
    elcCurrent_list = []
    Current_time_list = []

    label_list = []
    for name in name_list:
        label = name.rsplit('/',2)[1]
        label_list.append(label)

    for name in name_list:

        filename_root = name + 'saved_data/'
        elcTemp_name = filename_root + 'elc_intM2Thermal.txt'
        elcTemp_time_name = filename_root + 'elc_intM2Thermal_time.txt'
        ionTemp_name = filename_root + 'ion_intM2Thermal.txt'
        ionTemp_time_name = filename_root + 'ion_intM2Thermal_time.txt'
        fieldEnergy_name = filename_root + 'fieldEnergy.txt'
        fieldEnergy_time_name = filename_root + 'fieldEnergy_time.txt'
        elcCurrent_name = filename_root + 'elc_intM1i.txt'
        Current_time_name = filename_root + 'elc_intM1i_time.txt'

        elcTemp = np.loadtxt(elcTemp_name)
        elcTemp_time = np.loadtxt(elcTemp_time_name)

        ionTemp = np.loadtxt(ionTemp_name)
        ionTemp_time = np.loadtxt(ionTemp_time_name)

        fieldEnergy = np.loadtxt(fieldEnergy_name)
        fieldEnergy_time = np.loadtxt(fieldEnergy_time_name)

        elcCurrent = np.loadtxt(elcCurrent_name)
        #ionCurrent = -np.loadtxt('./saved_data/elc_intM1i.txt')
        Current_time = np.loadtxt(Current_time_name)

        elcTemp_list.append(elcTemp)
        elcTemp_time_list.append(elcTemp_time)
        ionTemp_list.append(ionTemp)
        ionTemp_time_list.append(ionTemp_time)
        fieldEnergy_list.append(fieldEnergy)
        fieldEnergy_time_list.append(fieldEnergy_time)
        elcCurrent_list.append(elcCurrent)
        Current_time_list.append(Current_time)

    elcCurrent_list[0] = 2.0*elcCurrent_list[0]

    fig, axs = plt.subplots(2,2,figsize=(30, 25), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace =.2)

    #cut = int((1-cutoff)*(ionTemp_time.shape[0]))
    #cut_field = int((1-cutoff)*(fieldEnergy_time.shape[0]))

    ####
    #determine the range for each run
    ####
    timecut_energy_list = []
    timecut_list = []
    for i in range(len(name_list)):
        #for field energy
        time_list = fieldEnergy_time_list[i]
        for j in range(len(time_list)):
            if time_list[j] > timecut:
                timecut_energy_list.append(j)
                break
        
        #for other quantity
        time_list = elcTemp_time_list[i]
        for j in range(len(time_list)):
            if time_list[j] > timecut:
                timecut_list.append(j)
                break

    for i in range(len(name_list)):
        axs[0,0].plot(ionTemp_time_list[i][0:timecut_list[i]],ionTemp_list[i][0:timecut_list[i]]/ionTemp_list[i][0],label=label_list[i])

        axs[0,1].plot(elcTemp_time_list[i][0:timecut_list[i]],elcTemp_list[i][0:timecut_list[i]],label=label_list[i])

        axs[1,0].plot(fieldEnergy_time_list[i][0:timecut_energy_list[i]],fieldEnergy_list[i][0:timecut_energy_list[i]],label=label_list[i])
        
        if i > 0:
            axs[1,1].plot(Current_time_list[i][0:timecut_list[i]],elcCurrent_list[i][0:timecut_list[i]]*2.0,label=label_list[i])
        else:
            axs[1,1].plot(Current_time_list[i][0:timecut_list[i]],elcCurrent_list[i][0:timecut_list[i]],label=label_list[i])


    axs[0,0].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=30)
    axs[0,0].set_ylabel(r'$T_i/T_{i0}$',fontsize=30)
    axs[0,0].tick_params(labelsize = 26)
    axs[0,0].legend(fontsize=30)

    axs[0,1].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=30)
    axs[0,1].set_ylabel(r'$T_e/T_{e0}$',fontsize=30)
    axs[0,1].tick_params(labelsize = 26)
    axs[0,1].legend(fontsize=30)

    axs[1,0].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=30)
    axs[1,0].set_yscale('log')
    axs[1,0].tick_params(labelsize = 26)
    axs[1,0].legend(fontsize=30)

    axs[1,1].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=30)
    axs[1,1].set_ylabel(r'$J_z$',fontsize=30)
    axs[1,1].tick_params(labelsize = 26)
    #axs[1,1].set_xlim(1,1200)
    axs[1,1].legend(fontsize=30)

    #fig.tight_layout()
    fig.savefig('./output.jpg')


if __name__ == '__main__':
    #name_list = ['./mass100/Erec/highres/','./mass100/Erec/lowres/']
    #name_list = ['./Cori/mass400/highres/','./Cori/mass400/lowres/']
    #name_list = ['./Cori/mass25/High/','./Cori/mass25/Low/','./Cori/mass25/lowcol/']
    #name_list = ['./Diagnostics/local/E_external/E1/0/','./Diagnostics/local/Cori/mass25/rescheck/4/', './Diagnostics/local/Cori/mass25/rescheck/3/']
    #name_list = ['./Diagnostics/local/Cori/mass25/rescheck/4/', './Diagnostics/local/E_external/E2/1D/']
    name_list = ['./Diagnostics/local/massRatio/mass100/E1/','./Diagnostics/local/massRatio/mass100/E1-low1/','./Diagnostics/local/massRatio/mass100/E1-low2/','./Diagnostics/local/massRatio/mass100/E1-low3/']
    make_time_traces_plot(name_list,2000)