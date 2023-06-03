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

# Iontemp = np.loadtxt('./massRatio/mass100/E1/saved_data/ion_intM2Thermal.txt')*100
# time_Iontemp = np.loadtxt('./massRatio/mass100/E1/saved_data/ion_intM2Thermal_time.txt')
# Elctemp = np.loadtxt('./massRatio/mass100/E1/saved_data/elc_intM2Thermal.txt')
# time_Elctemp = np.loadtxt('./massRatio/mass100/E1/saved_data/elc_intM2Thermal_time.txt')

dJdt = np.zeros(np.size(current)-1)
nu_eff = np.zeros(np.size(current)-1)
for i in range(np.size(current)-1):
    dJdt[i] = (current[i+1] - current[i]) / (time_current[i+1] - time_current[i])
for i in range(np.size(current)-1):
    nu_eff[i] = (0.00005 - dJdt[i]) / current[i]


def fieldenergy_current():
    fig      = plt.figure(figsize=(14.5,8.5))
    ax      = fig.add_axes([0.12, 0.16, 0.75, 0.75])
    ax2 = ax.twinx()

    t1 = np.arange(1000,1900)
    c1 = 0.018 + 0.000016*(t1-1000) 
    t2 = np.arange(1900,3800)
    c2 = 0.035 + 0.000035*(t2-1900)
    ax.plot(time_current,current/0.02,label='2D',linewidth=5,color='blue')
    #ax.plot(time_current_1d,current_1d/0.02,label='1D',linewidth=4)
    ax.plot(t1,c1/0.02,linewidth=6,linestyle='dashed',color='green')
    ax.plot(t2,c2/0.02,linewidth=6,linestyle='dashed',color='orange')

    ax2.plot(time_fieldEnergy,fieldEnergy/1e-4,linewidth=5,color='red')
    ax2.set_yscale('log')
    #ax2.set_ylim(1e-11,1e-4)
    ax2.tick_params(labelsize = 26,colors='red')

    ax.vlines(240,0,7.0,linestyle=':',linewidth=4,color='red')
    ax.vlines(500,0,7.0,linestyle='--',linewidth=2,color='black')
    ax.vlines(500,0,7.0,linestyle='--',linewidth=2,color='black')
    ax.vlines(1000,0,7.0,linestyle='--',linewidth=2,color='black')
    ax.vlines(1900,0,7.0,linestyle='--',linewidth=2,color='black')
    ax.vlines(3800,0,7.0,linestyle='--',linewidth=2,color='black')
    #ax.hlines(0.2,0,4500,linestyle=':',linewidth=5,color='black')
    ax.set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    ax.set_ylabel(r'$<J_z> [en_0 v_{Te0}]$',fontsize=32,color='blue')
    ax2.set_ylabel(r'$\int |E_z|^2+|E_y|^2 dydz/4\pi n_0T_{e0}$',fontsize=32,color='red')
    ax.set_xlim(0,4500)
    ax.set_ylim(0,7.0)
    ax.tick_params(labelsize = 26)
    ax.tick_params(axis='y',colors = 'blue')
    plt.show()
    #plt.savefig('./Figures/figures_temp/current_and_fieldenergy.jpg')
    plt.clf()

def temperature_ratio():
    fig      = plt.figure(figsize=(11.0,9.0))
    ax      = fig.add_axes([0.12, 0.16, 0.75, 0.80])
    
    #ax2 = ax.twinx()

    ax.plot(time_Elctemp[:],Elctemp[:]/Elctemp[0],label='electron',linewidth=12,color='black',linestyle=':')
    ax.plot(time_Iontemp[:],Iontemp[:]/Iontemp[0],label='ion',linewidth=12,color='black',linestyle='--')
    #ax2.plot(time_Iontemp4[70:95],Elctemp4[70:95]/Iontemp4[70:95],linewidth=5,color='blue')
    #ax2.plot(time_Iontemp[:],Elctemp[:]/Iontemp[:],linewidth=5,color='blue')
    ax.set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    ax.set_ylabel(r'$T/T_{0}$',fontsize=36,color='black')
    #ax2.set_ylabel(r'$T_e/T_i$',fontsize=36,color='blue')
    ax.set_xlim(0,4500)
    ax.set_ylim(0,30)
    #ax2.set_ylim(10,70)

    ax.tick_params(labelsize = 58)
    ax.tick_params(axis='y',colors='black')
    #ax2.tick_params(labelsize = 28,colors='blue')
    
    #ax.legend(fontsize=30,loc='center right',bbox_to_anchor=(1.0, 0.12))
    ax.grid()
    #ax.legend(fontsize=30,loc='lower left')
    #plt.savefig('./Cori/figure_temp/temp.jpg')
    plt.show()
    
    #plt.savefig('./Figures/figures_temp/tempratio.jpg')
    plt.clf()


#######################
# Not used by the paper
#######################

def all_current():
    current25_E1 = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM1i.txt')*2
    time_current25_E1 = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM1i_time.txt') + 300
    current100_E1 = np.loadtxt('./massRatio/mass100/E1/saved_data/elc_intM1i.txt')*2
    time_curren100_E1 = np.loadtxt('./massRatio/mass100/E1/saved_data/elc_intM1i_time.txt') + 160
    current400_E1 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/elc_intM1i.txt')*2
    time_current400_E1 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/elc_intM1i_time.txt') + 80

    current25_E2 = np.loadtxt('./massRatio/mass25/E2/saved_data/elc_intM1i.txt')*2
    time_current25_E2 = np.loadtxt('./massRatio/mass25/E2/saved_data/elc_intM1i_time.txt') + 150
    current25_E5 = np.loadtxt('./massRatio/mass25/E5/saved_data/elc_intM1i.txt')*2
    time_current25_E5 = np.loadtxt('./massRatio/mass25/E5/saved_data/elc_intM1i_time.txt')

    current100_E5 = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM1i.txt')*2
    time_curren100_E1 = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM1i_time.txt') 
    current400_E5 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/elc_intM1i.txt')*2
    time_current400_E5 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/elc_intM1i_time.txt') + 80

    fig      = plt.figure(figsize=(10.5,7.5))
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    ax.plot(time_current1[:],current1,label='E1',linewidth=5)
    # ax.plot(time_current2[:],current2,label='E2',linewidth=5)
    # ax.plot(time_current3[:],current3,label='E3',linewidth=5)
    # ax.plot(time_current4[:],current4,label='E4',linewidth=5)
    # ax.plot(time_current5[:],current5,label='E5',linewidth=5)

    # ax.plot(time_current1[1:],nu_eff1[:],label='E1',linewidth=5)
    # ax.plot(time_current2[1:241],nu_eff2[:240],label='E2',linewidth=5)
    # ax.plot(time_current3[1:201],nu_eff3[:200],label='E3',linewidth=5)
    # ax.plot(time_current4[1:181],nu_eff4[:180],label='E4',linewidth=5)
    # ax.plot(time_current5[2:141],nu_eff5[1:140],label='E5',linewidth=5)


    ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)

    #ax.set_ylabel(r'$<J_z> [en_0 v_{Te0}]$',fontsize=32,color='blue')
    # ax.set_xlim(0,2700)
    # ax.set_ylim(0,5.0)
    ax.tick_params(labelsize = 26)
    ax.tick_params(axis='y',colors = 'blue')
    ax.legend()
    ax.grid()
    #ax.set_xlim(0,3500)
    plt.show()

def ion_temp():
    fig      = plt.figure(figsize=(11.0,9.0))
    ax      = fig.add_axes([0.12, 0.16, 0.75, 0.80])
    ax2 = ax.twinx()

    ax.plot(time_Iontemp[:],Elctemp[0]/Iontemp[:],label='ion',linewidth=5,color='black',linestyle='--')
    #ax.set_xlim(0,10000)
    #ax.set_ylim(0,55)
    ax.grid()
    plt.show()
    plt.clf()

def energy():

    lz = 1.0
    ly = 0.5
    nz = 96
    ny = 48
    dz = lz/nz
    dy = ly/ny

    W_list = []
    for i in range(75):
        fignum = str(i).zfill(4)
        #phi = np.loadtxt('./massRatio/mass100/E5_H2/field/M100_E5_field_' + fignum + '.txt')
        phi = np.loadtxt('./massRatio/mass25/E5_switchoff/field/M25_E5_so_field_' + fignum + '.txt')

        E_z, E_y = np.gradient(phi)
        E_z = E_z/dz
        E_y = E_y/dy

        W = 0
        for i in range(96):
            for j in range(48):
                W += 0.5*(E_z[i,j]**2 + E_y[i,j]**2)
        
        W = W / nz / ny
        W_list.append(W)

    W_list = np.array(W_list)
    time = np.arange(75)*10   

    # plt.plot(time, W_list,label='W')
    # plt.plot(time_fieldEnergy, fieldEnergy,label='Gkeyll')
    # plt.legend()
    # plt.show()

    # Ion_thermal = np.load('./massRatio/mass100/E5_H2/saved_data/ion_M2Thermal.npy')*100
    # Elc_thermal = np.load('./massRatio/mass100/E5_H2/saved_data/elc_M2Thermal.npy')
    # Ion_kinetic = np.load('./massRatio/mass100/E5_H2/saved_data/ion_M2Flow.npy')*100
    # Elc_kinetic = np.load('./massRatio/mass100/E5_H2/saved_data/elc_M2Flow.npy')

    Ion_thermal = np.load('./massRatio/mass25/E5_switchoff/saved_data/ion_M2Thermal.npy')*25
    Elc_thermal = np.load('./massRatio/mass25/E5_switchoff/saved_data/elc_M2Thermal.npy')
    Ion_kinetic = np.load('./massRatio/mass25/E5_switchoff/saved_data/ion_M2Flow.npy')*25*2
    Elc_kinetic = np.load('./massRatio/mass25/E5_switchoff/saved_data/elc_M2Flow.npy')*2


    def integrate(input):
        return np.average(input,axis=(1,2,3))*0.5
    
    Ion_thermal = integrate(Ion_thermal)
    Elc_thermal = integrate(Elc_thermal)
    Ion_kinetic = integrate(Ion_kinetic)
    Elc_kinetic = integrate(Elc_kinetic)

    plt.plot(time,Ion_thermal,label='Ion_T',color='red')
    plt.plot(time,Elc_thermal,label='Elc_T',color='blue')
    plt.plot(time,Ion_kinetic,label='Ion_U',color='red',linestyle='--')
    plt.plot(time,Elc_kinetic,label='elc_U',color='blue',linestyle='--')
    plt.plot(time,2*W_list,label='W')

    plt.plot(time, 2*W_list+ Ion_kinetic+Elc_kinetic+Elc_thermal+Ion_thermal,label='total')

    #plt.plot(time_fieldEnergy,fieldEnergy,label='W')


    plt.legend()
    plt.yscale('log')
    plt.ylim(1e-7,5e-3)

    plt.show()




if __name__ == "__main__":
    #ion_temp()
    
    # fieldenergy_current()
    
    # temperature_ratio()

    energy()