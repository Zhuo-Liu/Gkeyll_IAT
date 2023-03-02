import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

####### Load Data ########

fieldEnergy = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/fieldEnergy.txt')
time_fieldEnergy = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/fieldEnergy_time.txt')

time_current = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM1i_time.txt')
current = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM1i.txt')*2

Iontemp = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/ion_intM2Thermal.txt')*100
time_Iontemp = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/ion_intM2Thermal_time.txt')
Elctemp = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM2Thermal.txt')
time_Elctemp = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM2Thermal_time.txt')

dJdt = np.zeros(np.size(current)-1)
nu_eff = np.zeros(np.size(current)-1)
for i in range(np.size(current)-1):
    dJdt[i] = (current[i+1] - current[i]) / (time_current[i+1] - time_current[i])
for i in range(np.size(current)-1):
    nu_eff[i] = (0.00005 - dJdt[i]) / current[i]


def fieldenergy_current():
    fig      = plt.figure(figsize=(14.5,8.5))
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
    ax2 = ax.twinx()

    t1 = np.arange(1000,2000)
    c1 = 0.019 + 0.000017*(t1-1000) 
    t2 = np.arange(2000,3800)
    c2 = 0.040 + 0.000035*(t2-2000)
    ax.plot(time_current,current/0.02,label='2D',linewidth=5,color='blue')
    #ax.plot(time_current_1d,current_1d/0.02,label='1D',linewidth=4)
    ax.plot(t1,c1/0.02,linewidth=6,linestyle='dashed',color='green')
    ax.plot(t2,c2/0.02,linewidth=6,linestyle='dashed',color='orange')

    ax2.plot(time_fieldEnergy,fieldEnergy/1e-4,linewidth=5,color='red')
    ax2.set_yscale('log')
    #ax2.set_ylim(1e-11,1e-4)
    ax2.tick_params(labelsize = 26,colors='red')

    ax.vlines(500,0,7.0,linestyle='--',linewidth=2,color='black')
    ax.vlines(1000,0,7.0,linestyle='--',linewidth=2,color='black')
    ax.vlines(2000,0,7.0,linestyle='--',linewidth=2,color='black')
    ax.vlines(3800,0,7.0,linestyle='--',linewidth=2,color='black')
    ax.set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    ax.set_ylabel(r'$<J_z> [en_0 v_{Te0}]$',fontsize=32,color='blue')
    ax.set_xlim(0,4500)
    ax.set_ylim(0,7.0)
    ax.tick_params(labelsize = 26)
    ax.tick_params(axis='y',colors = 'blue')
    #plt.show()
    plt.savefig('./Figures/figures_temp/current_and_fieldenergy.jpg')
    plt.clf()

def temperature_ratio():
    fig      = plt.figure(figsize=(11.0,9.0))
    ax      = fig.add_axes([0.12, 0.16, 0.75, 0.80])
    ax2 = ax.twinx()

    ax.plot(time_Elctemp[:],Elctemp[:]/Elctemp[0],label='electron',linewidth=5,color='black',linestyle='-.')
    ax.plot(time_Iontemp[:],Iontemp[:]/Iontemp[0],label='ion',linewidth=5,color='black',linestyle='--')
    #ax2.plot(time_Iontemp4[70:95],Elctemp4[70:95]/Iontemp4[70:95],linewidth=5,color='blue')
    ax2.plot(time_Iontemp[:],Elctemp[:]/Iontemp[:],linewidth=5,color='blue')
    ax.set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    ax.set_ylabel(r'$T/T_{0}$',fontsize=36,color='black')
    ax2.set_ylabel(r'$T_e/T_i$',fontsize=36,color='blue')
    ax.set_xlim(0,4500)
    ax.set_ylim(0,30)
    ax2.set_ylim(10,70)
    # ax.vlines(500,0,30,linestyles='--',linewidth=3)
    # ax.vlines(1000,0,30,linestyles='--',linewidth=3)
    # ax.vlines(2000,0,30,linestyles='--',linewidth=3)
    # ax.vlines(3800,0,30,linestyles='--',linewidth=3)
    # ax.text(675,27.5,"II",fontsize=36)
    # ax.text(1350,27.5,"III",fontsize=36)
    # ax.text(2700,27.5,"IV",fontsize=36)
    # ax.text(4100,27.5,"V",fontsize=36)

    #ax.set_xlim(750,1800)
    #ax.set_ylim(0,18)
    ax.tick_params(labelsize = 28)
    ax.tick_params(axis='y',colors='black')
    ax2.tick_params(labelsize = 24,colors='blue')
    ax.legend(fontsize=30,loc='center right',bbox_to_anchor=(1.0, 0.5))
    #ax.grid()
    #ax.legend(fontsize=30,loc='lower left')
    #plt.savefig('./Cori/figure_temp/temp.jpg')
    #plt.show()
    plt.savefig('./Figures/figures_temp/tempratio.jpg')
    plt.clf()

if __name__ == "__main__":
    #fieldenergy_current()
    temperature_ratio()