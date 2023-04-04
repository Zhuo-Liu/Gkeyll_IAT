import numpy as np
from matplotlib import pyplot as plt

def mass25():
    fig, axs = plt.subplots(2, 2, figsize=(16,12))

    ######### Field Energy #######
    fieldEnergy1 = np.loadtxt('./Cori/mass25/rescheck/1/saved_data/fieldEnergy.txt')
    time_fieldEnergy1 = np.loadtxt('./Cori/mass25/rescheck/1/saved_data/fieldEnergy_time.txt')
    fieldEnergy2 = np.loadtxt('./Cori/mass25/rescheck/2/saved_data/fieldEnergy.txt')
    time_fieldEnergy2 = np.loadtxt('./Cori/mass25/rescheck/2/saved_data/fieldEnergy_time.txt')
    fieldEnergy3 = np.loadtxt('./Cori/mass25/rescheck/3/saved_data/fieldEnergy.txt')
    time_fieldEnergy3 = np.loadtxt('./Cori/mass25/rescheck/3/saved_data/fieldEnergy_time.txt')
    fieldEnergy3_1 = np.loadtxt('./Cori/mass25/rescheck/3-1/saved_data/fieldEnergy.txt')/2
    time_fieldEnergy3_1 = np.loadtxt('./Cori/mass25/rescheck/3-1/saved_data/fieldEnergy_time.txt')
    fieldEnergy3_2 = np.loadtxt('./Cori/mass25/rescheck/3-2/saved_data/fieldEnergy.txt')
    time_fieldEnergy3_2 = np.loadtxt('./Cori/mass25/rescheck/3-2/saved_data/fieldEnergy_time.txt')
    fieldEnergy4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/fieldEnergy.txt')
    time_fieldEnergy4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/fieldEnergy_time.txt')

    Iontemp1 = np.loadtxt('./Cori/mass25/rescheck/1/saved_data/ion_intM2Thermal.txt')
    time_Iontemp1 = np.loadtxt('./Cori/mass25/rescheck/1/saved_data/ion_intM2Thermal_time.txt')
    Iontemp2 = np.loadtxt('./Cori/mass25/rescheck/2/saved_data/ion_intM2Thermal.txt')
    time_Iontemp2 = np.loadtxt('./Cori/mass25/rescheck/2/saved_data/ion_intM2Thermal_time.txt')
    Iontemp3 = np.loadtxt('./Cori/mass25/rescheck/3/saved_data/ion_intM2Thermal.txt')
    time_Iontemp3 = np.loadtxt('./Cori/mass25/rescheck/3/saved_data/ion_intM2Thermal_time.txt')
    Iontemp3_1 = np.loadtxt('./Cori/mass25/rescheck/3-1/saved_data/ion_intM2Thermal.txt')
    time_Iontemp3_1 = np.loadtxt('./Cori/mass25/rescheck/3-1/saved_data/ion_intM2Thermal_time.txt')
    Iontemp3_2 = np.loadtxt('./Cori/mass25/rescheck/3-2/saved_data/ion_intM2Thermal.txt')
    time_Iontemp3_2 = np.loadtxt('./Cori/mass25/rescheck/3-2/saved_data/ion_intM2Thermal_time.txt')
    Iontemp4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/ion_intM2Thermal.txt')
    time_Iontemp4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/ion_intM2Thermal_time.txt')

    current1 = np.loadtxt('./Cori/mass25/rescheck/1/saved_data/elc_intM1i.txt')*2
    time_current1 = np.loadtxt('./Cori/mass25/rescheck/1/saved_data/elc_intM1i_time.txt')
    current2 = np.loadtxt('./Cori/mass25/rescheck/2/saved_data/elc_intM1i.txt')*2
    time_current2 = np.loadtxt('./Cori/mass25/rescheck/2/saved_data/elc_intM1i_time.txt')
    current3 = np.loadtxt('./Cori/mass25/rescheck/3/saved_data/elc_intM1i.txt')*2
    time_current3 = np.loadtxt('./Cori/mass25/rescheck/3/saved_data/elc_intM1i_time.txt')
    current3_1 = np.loadtxt('./Cori/mass25/rescheck/3-1/saved_data/elc_intM1i.txt')
    time_current3_1 = np.loadtxt('./Cori/mass25/rescheck/3-1/saved_data/elc_intM1i_time.txt')
    current3_2 = np.loadtxt('./Cori/mass25/rescheck/3-2/saved_data/elc_intM1i.txt')*2
    time_current3_2 = np.loadtxt('./Cori/mass25/rescheck/3-2/saved_data/elc_intM1i_time.txt')
    current4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/elc_intM1i.txt')*2
    time_current4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/elc_intM1i_time.txt')

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
    for i in range(np.size(current2)-2):
        nu_eff2[i] = (0.00005 - dJdt2[i]) / current2[i]

    dJdt3 = np.zeros(np.size(current3)-1)
    nu_eff3 = np.zeros(np.size(current3)-1)
    for i in range(np.size(current3)-1):
        dJdt3[i] = (current3[i+1] - current3[i]) / (time_current3[i+1] - time_current3[i])
    for i in range(np.size(current3)-1):
        nu_eff3[i] = (0.00005 - dJdt3[i]) / current3[i]

    dJdt3_1 = np.zeros(np.size(current3_1)-1)
    nu_eff3_1 = np.zeros(np.size(current3_1)-1)
    for i in range(np.size(current3_1)-1):
        dJdt3_1[i] = (current3_1[i+1] - current3_1[i]) / (time_current3_1[i+1] - time_current3_1[i])
    for i in range(np.size(current3_1)-1):
        nu_eff3_1[i] = (0.00005 - dJdt3_1[i]) / current3_1[i]
    
    dJdt3_2 = np.zeros(np.size(current3_2)-1)
    nu_eff3_2 = np.zeros(np.size(current3_2)-1)
    for i in range(np.size(current3_2)-1):
        dJdt3_2[i] = (current3_2[i+1] - current3_2[i]) / (time_current3_2[i+1] - time_current3_2[i])
    for i in range(np.size(current3_2)-1):
        nu_eff3_2[i] = (0.00005 - dJdt3_2[i]) / current3_2[i]

    dJdt4 = np.zeros(np.size(current4)-1)
    nu_eff4 = np.zeros(np.size(current4)-1)
    for i in range(np.size(current4)-1):
        dJdt4[i] = (current4[i+1] - current4[i]) / (time_current4[i+1] - time_current4[i])
    for i in range(np.size(current4)-1):
        nu_eff4[i] = (0.00005 - dJdt4[i]) / current4[i]

    axs[0,0].set_title("perturbed electric field energy",fontsize=28)
    axs[0,0].plot(time_fieldEnergy1[:],fieldEnergy1[:],label='#1',linewidth=5)
    axs[0,0].plot(time_fieldEnergy2[:],fieldEnergy2[:],label='#2',linewidth=5)
    axs[0,0].plot(time_fieldEnergy3[:],fieldEnergy3[:],label='#3',linewidth=5)
    #axs[0,0].plot(time_fieldEnergy3_1[:],fieldEnergy3_1[:],label='#3-1',linewidth=5)
    axs[0,0].plot(time_fieldEnergy3_2[:],fieldEnergy3_2[:],label='#3-1',linewidth=5)
    axs[0,0].plot(time_fieldEnergy4[:],fieldEnergy4[:],label='#4',linewidth=5)
    axs[0,0].set_xlabel(r'$t \quad (\omega_{pe}^-1]$',fontsize=32)
    axs[0,0].set_ylabel(r'$\int dydz |\delta E_z|^2 + |\delta E_y|^2$',fontsize=28)
    axs[0,0].set_yscale('log')
    axs[0,0].set_xlim(0,1800)
    axs[0,0].tick_params(labelsize = 28)
    axs[0,0].legend(fontsize=24)


    axs[0,1].set_title("ion temperature",fontsize=28)
    axs[0,1].plot(time_Iontemp1[:400],Iontemp1[:400]/Iontemp1[0],label='#1',linewidth=5)
    axs[0,1].plot(time_Iontemp2[:400],Iontemp2[:400]/Iontemp1[0],label='#2',linewidth=5)
    axs[0,1].plot(time_Iontemp3[:400],Iontemp3[:400]/Iontemp3[0],label='#3',linewidth=5)
    #axs[0,1].plot(time_Iontemp3_1[:400],Iontemp3_1[:400]/Iontemp3_1[0],label='#3-1',linewidth=5)
    axs[0,1].plot(time_Iontemp3_2[:400],Iontemp3_2[:400]/Iontemp3_2[0],label='#3-1',linewidth=5)
    axs[0,1].plot(time_Iontemp4[:400],Iontemp4[:400]/Iontemp4[0],label='#4',linewidth=5)
    axs[0,1].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=32)
    axs[0,1].set_ylabel(r'$T_i/T_{i0}$',fontsize=32)
    axs[0,1].set_xlim(0,1800)
    axs[0,1].tick_params(labelsize = 28)
    axs[0,1].legend(fontsize=24)

    axs[1,0].set_title("current",fontsize=28)
    axs[1,0].plot(time_current1[:400],current1[:400]/0.02,label='#1',linewidth=5)
    axs[1,0].plot(time_current2[:400],current2[:400]/0.02,label='#2',linewidth=5)
    axs[1,0].plot(time_current3[:400],current3[:400]/0.02,label='#3',linewidth=5)
    #axs[1,0].plot(time_current3_1[:400],current3_1[:400]/0.02,label='#3-1',linewidth=5)
    axs[1,0].plot(time_current3_2[:400],current3_2[:400]/0.02,label='#3-1',linewidth=5)
    axs[1,0].plot(time_current4[:400],current4[:400]/0.02,label='#4',linewidth=5)
    axs[1,0].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=32)
    axs[1,0].set_ylabel(r'$<J_z> [en_0 v_{te}]$',fontsize=32)
    axs[1,0].set_xlim(0,1800)
    axs[1,0].tick_params(labelsize = 28)
    axs[1,0].legend(fontsize=24)

    axs[1,1].set_title("effective collision frequency",fontsize=28)
    axs[1,1].plot(time_current1[9:401],nu_eff1[8:400],label='#1',linewidth=5)
    axs[1,1].plot(time_current2[9:401],nu_eff2[8:400],label='#2',linewidth=5)
    axs[1,1].plot(time_current3[9:401],nu_eff3[8:400],label='#3',linewidth=5)
    #axs[1,1].plot(time_current3_1[9:401],nu_eff3_1[8:400],label='#3-1',linewidth=5)
    axs[1,1].plot(time_current3_2[9:401],nu_eff3_2[8:400],label='#3-1',linewidth=5)
    axs[1,1].plot(time_current4[9:401],nu_eff4[8:400],label='#4',linewidth=5)
    axs[1,1].set_xlabel(r'$t [\omega_{pe}^-1]$',fontsize=32)
    axs[1,1].set_ylabel(r'$\nu_{eff} [\omega_{pe}^{-1}]$',fontsize=32)
    # axs[1,1].hlines(0.0006,0,2000,linestyle='--')
    # axs[1,1].text(1200,0.0007,'quasi-linear \n estimation',fontsize=24)
    axs[1,1].set_xlim(0,1800)
    axs[1,1].tick_params(labelsize = 28)
    axs[1,1].legend(fontsize=24)

    plt.tight_layout()
    #plt.show()
    plt.savefig('./mass25_nonlinear_test.pdf')

if __name__ == '__main__':
    mass25()