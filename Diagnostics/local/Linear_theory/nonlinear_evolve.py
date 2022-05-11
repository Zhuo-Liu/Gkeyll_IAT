import numpy as np
from matplotlib import pyplot as plt

def mass25():
    fig, axs = plt.subplots(2, 2, figsize=(16,12))

    ######### Field Energy #######
    fieldEnergy1 = np.loadtxt('./Diagnostics/local/E_external/Cori/Low/saved_data/fieldEnergy.txt')
    time_fieldEnergy1 = np.loadtxt('./Diagnostics/local/E_external/Cori/Low/saved_data/fieldEnergy_time.txt')
    fieldEnergy2 = np.loadtxt('./Diagnostics/local/E_external/Cori/High/saved_data/fieldEnergy.txt')
    time_fieldEnergy2 = np.loadtxt('./Diagnostics/local/E_external/Cori/High/saved_data/fieldEnergy_time.txt')

    Iontemp1 = np.loadtxt('./Diagnostics/local/E_external/Cori/Low/saved_data/ion_intM2Thermal.txt')
    time_Iontemp1 = np.loadtxt('./Diagnostics/local/E_external/Cori/Low/saved_data/ion_intM2Thermal_time.txt')
    Iontemp2 = np.loadtxt('./Diagnostics/local/E_external/Cori/High/saved_data/ion_intM2Thermal.txt')
    time_Iontemp2 = np.loadtxt('./Diagnostics/local/E_external/Cori/High/saved_data/ion_intM2Thermal_time.txt')

    current1 = np.loadtxt('./Diagnostics/local/E_external/Cori/Low/saved_data/elc_intM1i.txt')*2
    time_current1 = np.loadtxt('./Diagnostics/local/E_external/Cori/Low/saved_data/elc_intM1i_time.txt')
    current2 = np.loadtxt('./Diagnostics/local/E_external/Cori/High/saved_data/elc_intM1i.txt')*2
    time_current2 = np.loadtxt('./Diagnostics/local/E_external/Cori/High/saved_data/elc_intM1i_time.txt')

    dJdt1 = np.zeros(np.size(current1)-1)
    nu_eff1 = np.zeros(np.size(current1)-1)
    for i in range(np.size(current1)-1):
        dJdt1[i] = (current1[i+1] - current1[i]) / (time_current1[i+1] - time_current1[i])
    for i in range(np.size(current1)-2):
        nu_eff1[i] = (0.00005 - dJdt1[i]) / current1[i]

    dJdt2 = np.zeros(np.size(current2)-1)
    nu_eff2 = np.zeros(np.size(current2)-1)
    for i in range(np.size(current2)-1):
        dJdt2[i] = (current2[i+1] - current2[i]) / (time_current2[i+1] - time_current2[i])
    for i in range(np.size(current2)-1):
        nu_eff2[i] = (0.00005 - dJdt2[i]) / current2[i]


    axs[0,0].set_title("perturbed electric field energy",fontsize=28)
    axs[0,0].plot(time_fieldEnergy1[:80],fieldEnergy1[:80],label='#0',linewidth=5)
    axs[0,0].plot(time_fieldEnergy2[:80],fieldEnergy2[:80],label='#2',linewidth=5)
    axs[0,0].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=32)
    axs[0,0].set_ylabel(r'$\int dydz |\delta E_z|^2 + |\delta E_y|^2$',fontsize=28)
    axs[0,0].set_yscale('log')
    axs[0,0].set_xlim(0,2000)
    axs[0,0].tick_params(labelsize = 28)
    axs[0,0].legend(fontsize=24)


    axs[0,1].set_title("ion temperature",fontsize=28)
    axs[0,1].plot(time_Iontemp1[:400],Iontemp1[:400]/Iontemp1[0],label='#0',linewidth=5)
    axs[0,1].plot(time_Iontemp2[:400],Iontemp2[:400]/Iontemp2[0],label='#2',linewidth=5)
    axs[0,1].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=32)
    axs[0,1].set_ylabel(r'$T_i/T_{i0}$',fontsize=32)
    axs[0,1].set_xlim(0,2000)
    axs[0,1].tick_params(labelsize = 28)
    axs[0,1].legend(fontsize=24)

    axs[1,0].set_title("perturbed electric field energy",fontsize=28)
    axs[1,0].plot(time_current1[:400],current1[:400]/0.02,label='#0',linewidth=5)
    axs[1,0].plot(time_current2[:400],current2[:400]/0.02,label='#2',linewidth=5)
    axs[1,0].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=32)
    axs[1,0].set_ylabel(r'$<J_z> [en_0 v_{the}]$',fontsize=32)
    axs[1,0].set_xlim(0,2000)
    axs[1,0].tick_params(labelsize = 28)
    axs[1,0].legend(fontsize=24)

    axs[1,1].set_title("effective collision frequency",fontsize=28)
    axs[1,1].plot(time_current1[9:401],nu_eff1[8:400],label='#0',linewidth=5)
    axs[1,1].plot(time_current2[9:401],nu_eff2[8:400],label='#2',linewidth=5)
    axs[1,1].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=32)
    axs[1,1].set_ylabel(r'$\nu_{eff} [\omega_{pe}^{-1}]$',fontsize=32)
    axs[1,1].set_xlim(0,2000)
    axs[1,1].tick_params(labelsize = 28)
    axs[1,1].legend(fontsize=24)

    plt.tight_layout()
    plt.savefig('./field.pdf')

if __name__ == '__main__':
    mass25()