#####
# For discussing mass-ratio and electric field dependence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# if no add-on, it means that the resolution is H1

###############
#data reading
###############


fieldEnergy25 = np.loadtxt('./massRatio/mass25/E1/saved_data/fieldEnergy.txt')
time_fieldEnergy25 = np.loadtxt('./massRatio/mass25/E1/saved_data/fieldEnergy_time.txt')
fieldEnergy50 = np.loadtxt('./massRatio/mass100/E1-low1/saved_data/fieldEnergy.txt')
time_fieldEnergy50 = np.loadtxt('./massRatio/mass100/E1-low1/saved_data/fieldEnergy_time.txt')
fieldEnergy100 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/fieldEnergy.txt')
time_fieldEnergy100 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/fieldEnergy_time.txt')


Iontemp25 = np.loadtxt('./massRatio/mass25/E5/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp25 = np.loadtxt('./massRatio/mass25/E5/saved_data/ion_intM2Thermal_time.txt')
Elctemp25= np.loadtxt('./massRatio/mass25/E5/saved_data/elc_intM2Thermal.txt')
time_Elctemp25 = np.loadtxt('./massRatio/mass25/E5/saved_data/elc_intM2Thermal_time.txt')
Iontemp50 = np.loadtxt('./massRatio/mass50/E5_L1/saved_data/ion_intM2Thermal.txt')*50
time_Iontemp50 = np.loadtxt('./massRatio/mass50/E5_L1/saved_data/ion_intM2Thermal_time.txt')
Elctemp50 = np.loadtxt('./massRatio/mass50/E5_L1/saved_data/elc_intM2Thermal.txt')
time_Elctemp50 = np.loadtxt('./massRatio/mass50/E5_L1/saved_data/elc_intM2Thermal_time.txt')
Iontemp100 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/ion_intM2Thermal.txt')*100
time_Iontemp100 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/ion_intM2Thermal_time.txt')
Elctemp100 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/elc_intM2Thermal.txt')
time_Elctemp100 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/elc_intM2Thermal_time.txt')
# Iontemp200 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/ion_intM2Thermal.txt')*100
# time_Iontemp200 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/ion_intM2Thermal_time.txt')
# Elctemp200 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/elc_intM2Thermal.txt')
# time_Elctemp200 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/elc_intM2Thermal_time.txt')
Iontemp400 = np.loadtxt('./Cori/nouse/mass400/highres/saved_data/ion_intM2Thermal.txt')*400
time_Iontemp400 = np.loadtxt('./Cori/nouse/mass400/highres/saved_data/ion_intM2Thermal_time.txt')
Elctemp400 = np.loadtxt('./Cori/nouse/mass400/highres/saved_data/elc_intM2Thermal.txt')
time_Elctemp400 = np.loadtxt('./Cori/nouse/mass400/highres/saved_data/elc_intM2Thermal_time.txt')


# current25 = np.loadtxt('./massRatio/mass25/E5/saved_data/elc_intM1i.txt')*2
# time_current25 = np.loadtxt('./massRatio/mass25/E5/saved_data/elc_intM1i_time.txt')
current25 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/elc_intM1i.txt')*2
time_current25 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/elc_intM1i_time.txt')
current50 = np.loadtxt('./massRatio/mass50/E5_L1/saved_data/elc_intM1i.txt')*2
time_current50 = np.loadtxt('./massRatio/mass50/E5_L1/saved_data/elc_intM1i_time.txt')
current100 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/elc_intM1i.txt')*2
time_current100 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/elc_intM1i_time.txt')
# current200 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/elc_intM1i.txt')*2
# time_current200 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/elc_intM1i_time.txt')
# current400 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/elc_intM1i.txt')*2
# time_current400 = np.loadtxt('./massRatio/mass100/E5_old_L0/saved_data/elc_intM1i_time.txt')

dJdt25 = np.zeros(np.size(current25)-1)
nu_eff25 = np.zeros(np.size(current25)-1)
for i in range(np.size(current25)-1):
    dJdt25[i] = (current25[i+1] - current25[i]) / (time_current25[i+1] - time_current25[i])
for i in range(np.size(current25)-1):
    nu_eff25[i] = (0.00005 - dJdt25[i]) / current25[i]

dJdt50 = np.zeros(np.size(current50)-1)
nu_eff50 = np.zeros(np.size(current50)-1)
for i in range(np.size(current50)-1):
    dJdt50[i] = (current50[i+1] - current50[i]) / (time_current50[i+1] - time_current50[i])
for i in range(np.size(current50)-1):
    nu_eff50[i] = (0.00005 - dJdt50[i]) / current50[i]

dJdt100 = np.zeros(np.size(current100)-1)
nu_eff100 = np.zeros(np.size(current100)-1)
for i in range(np.size(current100)-1):
    dJdt100[i] = (current100[i+1] - current100[i]) / (time_current100[i+1] - time_current100[i])
for i in range(np.size(current100)-1):
    nu_eff100[i] = (0.00005 - dJdt100[i]) / current100[i]

# dJdt200 = np.zeros(np.size(current200)-1)
# nu_eff200 = np.zeros(np.size(current200)-1)
# for i in range(np.size(current200)-1):
#     dJdt200[i] = (current200[i+1] - current200[i]) / (time_current200[i+1] - time_current200[i])
# for i in range(np.size(current200)-1):
#     nu_eff200[i] = (0.00005 - dJdt200[i]) / current200[i]

# dJdt400 = np.zeros(np.size(current400)-1)
# nu_eff400 = np.zeros(np.size(current400)-1)
# for i in range(np.size(current400)-1):
#     dJdt400[i] = (current400[i+1] - current400[i]) / (time_current400[i+1] - time_current400[i])
# for i in range(np.size(current400)-1):
#     nu_eff400[i] = (0.00005 - dJdt400[i]) / current400[i]



########################################################################################################

def current_massratio():
    fig      = plt.figure(figsize=(10.5,7.5))
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    ax.plot(time_current25[:],current25,label='25',linewidth=5)
    ax.plot(time_current50[:],current50,label='50',linewidth=5)
    ax.plot(time_current100[:],current100,label='100',linewidth=5)

    # #ax.plot(time_current1[1:],nu_eff1,label='25',linewidth=5,color='red',linestyle='-')
    # #ax.plot(time_fieldEnergy1,fieldEnergy1/0.0004,linewidth=5,color='red',linestyle='--')
    # #ax.plot(time_current2[1:],nu_eff2,label='100',linewidth=5,color='green',linestyle='-')
    # #ax.plot(time_fieldEnergy2,fieldEnergy2/0.0004,linewidth=5,color='green',linestyle='--')
    # ax.plot(time_current3[1:],nu_eff3,label='400',linewidth=5,color='blue',linestyle='-')
    # ax.plot(time_fieldEnergy3,fieldEnergy3/0.0006,label='W/nTe',linewidth=5,color='blue',linestyle='--')


    ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)

    #ax.set_ylabel(r'$<J_z> [en_0 v_{Te0}]$',fontsize=32,color='blue')
    # ax.set_xlim(0,2700)
    # ax.set_ylim(0,5.0)
    ax.tick_params(labelsize = 26)
    ax.tick_params(axis='y',colors = 'blue')
    ax.legend(fontsize=25)
    ax.grid()
    ax.set_xlim(0,2000)
    plt.show()
    plt.clf()

def nueff_massratio():
    fig      = plt.figure(figsize=(10.5,7.5))
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    ax.plot(time_current25[2:],nu_eff25[1:],label='25',linewidth=5,color='red',linestyle='-')
    ax.plot(time_current50[2:],nu_eff50[1:],label='50',linewidth=5,color='green',linestyle='-')
    ax.plot(time_current100[2:],nu_eff100[1:],label='100',linewidth=5,color='blue',linestyle='-')
    #ax.plot(time_fieldEnergy3,fieldEnergy3/0.0006,label='W/nTe',linewidth=5,color='blue',linestyle='--')

    i = 0
    new_Elctemp25 = []
    new_Elctemp100 = []
    new_Elctemp400 = []
    for t in Elctemp25:
        if i%4 == 0:
            new_Elctemp25.append(t)
        i += 1
    new_Elctemp25 = np.array(new_Elctemp25)
    i = 0
    for t in Elctemp100:
        if i%5 == 0:
            new_Elctemp100.append(t)
        i += 1
    i = 0
    for t in Elctemp400:
        if i%3 == 0:
            new_Elctemp400.append(t)
        i += 1
    new_Elctemp400 = np.array(new_Elctemp400)
    ax.plot(time_fieldEnergy25,fieldEnergy25/new_Elctemp25,label=r'$(W/nTe)_{25}$',linewidth=3,linestyle='--')
    ax.plot(time_fieldEnergy100,fieldEnergy100/new_Elctemp100,label=r'$(W/nTe)_{100}$',linewidth=3,linestyle='--')
    #ax.plot(time_fieldEnergy400,fieldEnergy400/new_Elctemp400,label=r'$(W/nTe)_{400}$',linewidth=3,linestyle='--')

    ax.hlines(0.0059,0,2000, linestyles='--',color='black',linewidth=3)
    ax.text(650,0.0066-0.001,"quasi-linear",fontsize = 18)
    ax.text(700,0.0063-0.001,"estimate",fontsize = 18)
    ax.text(650,0.0060-0.001,"for $m_i/m_e=25$",fontsize = 18)

    ax.set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)

    ax.set_ylim(-0.0005,0.009)
    ax.tick_params(labelsize = 23)
    ax.tick_params(axis='y')
    ax.legend(fontsize=25)
    ax.grid()
    ax.set_xlim(0,2000)
    plt.savefig('nueff.jpg')
    #plt.show()
    #plt.clf()

def heating_massratio():
    fig     = plt.figure(figsize=(11.0,10.0))
    ax      = fig.add_axes([0.15, 0.15, 0.75, 0.82])

    ax.plot(time_Iontemp25[:],Iontemp25[:]/Iontemp25[0],linewidth=5,color='blue',label=r'$m_i/m_e=25$')
    ax.plot(time_Iontemp100[:400],Iontemp100[:400]/Iontemp100[0],linewidth=5,color='red',label=r'$m_i/m_e=100$')
    ax.plot(time_Iontemp400[:],Iontemp400[:]/Iontemp400[0],linewidth=5,color='green',label=r'$m_i/m_e=400$')

    ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
    ax.set_ylabel(r'$T_e/T_{i}$',fontsize=36,color='black')
    #ax.set_xlim(0,8000)
    ax.grid()
    ax.tick_params(labelsize = 24)
    ax.legend(fontsize=30)
    plt.show()

def heating_electricfield():
    pass

def temp_massratio():
    # coming this week!!!!
    pass

def temp_electricfield():
    # coming this week!!!!
    pass

if __name__ == '__main__':

    nueff_massratio()
    # nueff_electricfield()

    # tempratio_ratio()

    # ionheatingtrace()

    # ionheating()

    # elcheatingtrace_mass()
    # elcheatingtrace_E()

    # elcheating_rate()
