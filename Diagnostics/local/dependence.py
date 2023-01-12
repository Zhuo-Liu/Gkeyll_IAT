#####
# For discussing mass-ratio and electric field dependence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

#The default E is E5
#The default resolution is H1

#path_25 = './Cori/mass25/rescheck/1/saved_data/'
path_25 = './massRatio/mass25/E5/saved_data/'
path_50 = './massRatio/mass50/E5_H1/saved_data/'
path_100 = './massRatio/mass100/E5_H1/saved_data/'
path_200 = './massRatio/mass200/E5_L1/saved_data/'
path_400 = './massRatio/mass400/E5_L1/saved_data/'
#path_400 = './massRatio/nouse/mass400_Cori/saved_data/'
path_25_E1 = './massRatio/mass25/E1/saved_data/'
path_25_E2 = './massRatio/mass25/E2/saved_data/'
path_25_E3 = './massRatio/mass25/E3/saved_data/'
path_25_E4 = './massRatio/mass25/E4/saved_data/'

###############
#data reading
###############

######### Field Energy ############
fieldEnergy25 = np.loadtxt(path_25 + 'fieldEnergy.txt')
time_fieldEnergy25 = np.loadtxt(path_25 + 'fieldEnergy_time.txt')

fieldEnergy50 = np.loadtxt(path_50 + 'fieldEnergy.txt')
time_fieldEnergy50 = np.loadtxt(path_50 + 'fieldEnergy_time.txt')

fieldEnergy100 = np.loadtxt(path_100 + 'fieldEnergy.txt')
time_fieldEnergy100 = np.loadtxt(path_100 + 'fieldEnergy_time.txt')

fieldEnergy200 = np.loadtxt(path_200 + 'fieldEnergy.txt')
time_fieldEnergy200 = np.loadtxt(path_200 + 'fieldEnergy_time.txt')

fieldEnergy400 = np.loadtxt(path_400 + 'fieldEnergy.txt')
time_fieldEnergy400 = np.loadtxt(path_400 + 'fieldEnergy_time.txt')

fieldEnergy25_E1 = np.loadtxt(path_25_E1 + 'fieldEnergy.txt')
time_fieldEnergy25_E1 = np.loadtxt(path_25_E1 + 'fieldEnergy_time.txt')

fieldEnergy25_E2 = np.loadtxt(path_25_E2 + 'fieldEnergy.txt')
time_fieldEnergy25_E2 = np.loadtxt(path_25_E2 + 'fieldEnergy_time.txt')

fieldEnergy25_E3 = np.loadtxt(path_25_E3 + 'fieldEnergy.txt')
time_fieldEnergy25_E3 = np.loadtxt(path_25_E3 + 'fieldEnergy_time.txt')

fieldEnergy25_E4 = np.loadtxt(path_25_E4 + 'fieldEnergy.txt')
time_fieldEnergy25_E4 = np.loadtxt(path_25_E4 + 'fieldEnergy_time.txt')

######### Temperature ############
Iontemp25 = np.loadtxt(path_25 + 'ion_intM2Thermal.txt')*25
time_Iontemp25 = np.loadtxt(path_25 + 'ion_intM2Thermal_time.txt')
Elctemp25= np.loadtxt(path_25 + 'elc_intM2Thermal.txt')
time_Elctemp25 = np.loadtxt(path_25 + 'elc_intM2Thermal_time.txt')

Iontemp50 = np.loadtxt(path_50 + 'ion_intM2Thermal.txt')*50
time_Iontemp50 = np.loadtxt(path_50 + 'ion_intM2Thermal_time.txt')
Elctemp50 = np.loadtxt(path_50 + 'elc_intM2Thermal.txt')
time_Elctemp50 = np.loadtxt(path_50 + 'elc_intM2Thermal_time.txt')

Iontemp100 = np.loadtxt(path_100 + 'ion_intM2Thermal.txt')*100
time_Iontemp100 = np.loadtxt(path_100 + 'ion_intM2Thermal_time.txt')
Elctemp100 = np.loadtxt(path_100 + 'elc_intM2Thermal.txt')
time_Elctemp100 = np.loadtxt(path_100 + 'elc_intM2Thermal_time.txt')

Iontemp200 = np.loadtxt(path_200 + 'ion_intM2Thermal.txt')*200
time_Iontemp200 = np.loadtxt(path_200 + 'ion_intM2Thermal_time.txt')
Elctemp200 = np.loadtxt(path_200 + 'elc_intM2Thermal.txt')
time_Elctemp200 = np.loadtxt(path_200 + 'elc_intM2Thermal_time.txt')

Iontemp400 = np.loadtxt(path_400 + 'ion_intM2Thermal.txt')*400
time_Iontemp400 = np.loadtxt(path_400 + 'ion_intM2Thermal_time.txt')
Elctemp400 = np.loadtxt(path_400 + 'elc_intM2Thermal.txt')
time_Elctemp400 = np.loadtxt(path_400 + 'elc_intM2Thermal_time.txt')

Iontemp25_E1 = np.loadtxt(path_25_E1 + 'ion_intM2Thermal.txt')*25
time_Iontemp25_E1 = np.loadtxt(path_25_E1 + 'ion_intM2Thermal_time.txt')
Elctemp25_E1= np.loadtxt(path_25_E1 + 'elc_intM2Thermal.txt')
time_Elctemp25_E1 = np.loadtxt(path_25_E1 + 'elc_intM2Thermal_time.txt')

Iontemp25_E2 = np.loadtxt(path_25_E2 + 'ion_intM2Thermal.txt')*25
time_Iontemp25_E2 = np.loadtxt(path_25_E2 + 'ion_intM2Thermal_time.txt')
Elctemp25_E2 = np.loadtxt(path_25_E2 + 'elc_intM2Thermal.txt')
time_Elctemp25_E2 = np.loadtxt(path_25_E2 + 'elc_intM2Thermal_time.txt')

Iontemp25_E3 = np.loadtxt(path_25_E3 + 'ion_intM2Thermal.txt')*25
time_Iontemp25_E3 = np.loadtxt(path_25_E3 + 'ion_intM2Thermal_time.txt')
Elctemp25_E3 = np.loadtxt(path_25_E3 + 'elc_intM2Thermal.txt')
time_Elctemp25_E3 = np.loadtxt(path_25_E3 + 'elc_intM2Thermal_time.txt')

Iontemp25_E4 = np.loadtxt(path_25_E4 + 'ion_intM2Thermal.txt')*25
time_Iontemp25_E4 = np.loadtxt(path_25_E4 + 'ion_intM2Thermal_time.txt')
Elctemp25_E4 = np.loadtxt(path_25_E4 + 'elc_intM2Thermal.txt')
time_Elctemp25_E4 = np.loadtxt(path_25_E4 + 'elc_intM2Thermal_time.txt')


######### Current ############
current25 = np.loadtxt(path_25 + 'elc_intM1i.txt')*2
time_current25 = np.loadtxt(path_25 + 'elc_intM1i_time.txt')

current50 = np.loadtxt(path_50 + 'elc_intM1i.txt')*2
time_current50 = np.loadtxt(path_50 + 'elc_intM1i_time.txt')

current100 = np.loadtxt(path_100 + 'elc_intM1i.txt')*2
time_current100 = np.loadtxt(path_100 + 'elc_intM1i_time.txt')

current200 = np.loadtxt(path_200 + 'elc_intM1i.txt')*2
time_current200 = np.loadtxt(path_200 + 'elc_intM1i_time.txt')

current400 = np.loadtxt(path_400 + 'elc_intM1i.txt')*2
time_current400 = np.loadtxt(path_400 + 'elc_intM1i_time.txt')

current25_E1 = np.loadtxt(path_25_E1 + 'elc_intM1i.txt')*2
time_current25_E1 = np.loadtxt(path_25_E1 + 'elc_intM1i_time.txt')

current25_E2 = np.loadtxt(path_25_E2 + 'elc_intM1i.txt')*2
time_current25_E2 = np.loadtxt(path_25_E2 + 'elc_intM1i_time.txt')

current25_E3 = np.loadtxt(path_25_E3 + 'elc_intM1i.txt')*2
time_current25_E3 = np.loadtxt(path_25_E3 + 'elc_intM1i_time.txt')

current25_E4 = np.loadtxt(path_25_E4 + 'elc_intM1i.txt')*2
time_current25_E4 = np.loadtxt(path_25_E4 + 'elc_intM1i_time.txt')

######### Anomalous resistivity ############

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

dJdt200 = np.zeros(np.size(current200)-1)
nu_eff200 = np.zeros(np.size(current200)-1)
for i in range(np.size(current200)-1):
    dJdt200[i] = (current200[i+1] - current200[i]) / (time_current200[i+1] - time_current200[i])
for i in range(np.size(current200)-1):
    nu_eff200[i] = (0.00005 - dJdt200[i]) / current200[i]

dJdt400 = np.zeros(np.size(current400)-1)
nu_eff400 = np.zeros(np.size(current400)-1)
for i in range(np.size(current400)-1):
    dJdt400[i] = (current400[i+1] - current400[i]) / (time_current400[i+1] - time_current400[i])
for i in range(np.size(current400)-1):
    nu_eff400[i] = (0.00005 - dJdt400[i]) / current400[i]

dJdt25_E1 = np.zeros(np.size(current25_E1)-1)
nu_eff25_E1 = np.zeros(np.size(current25_E1)-1)
for i in range(np.size(current25_E1)-1):
    dJdt25_E1[i] = (current25_E1[i+1] - current25_E1[i]) / (time_current25_E1[i+1] - time_current25_E1[i])
for i in range(np.size(current25_E1)-1):
    nu_eff25_E1[i] = (0.00001 - dJdt25_E1[i]) / current25_E1[i]

dJdt25_E2 = np.zeros(np.size(current25_E2)-1)
nu_eff25_E2 = np.zeros(np.size(current25_E2)-1)
for i in range(np.size(current25_E2)-1):
    dJdt25_E2[i] = (current25_E2[i+1] - current25_E2[i]) / (time_current25_E2[i+1] - time_current25_E2[i])
for i in range(np.size(current25_E2)-1):
    nu_eff25_E2[i] = (0.00002 - dJdt25_E2[i]) / current25_E2[i]

dJdt25_E3 = np.zeros(np.size(current25_E3)-1)
nu_eff25_E3 = np.zeros(np.size(current25_E3)-1)
for i in range(np.size(current25_E3)-1):
    dJdt25_E3[i] = (current25_E3[i+1] - current25_E3[i]) / (time_current25_E3[i+1] - time_current25_E3[i])
for i in range(np.size(current25_E3)-1):
    nu_eff25_E3[i] = (0.00003 - dJdt25_E3[i]) / current25_E3[i]

dJdt25_E4 = np.zeros(np.size(current25_E4)-1)
nu_eff25_E4 = np.zeros(np.size(current25_E4)-1)
for i in range(np.size(current25_E4)-1):
    dJdt25_E4[i] = (current25_E4[i+1] - current25_E4[i]) / (time_current25_E4[i+1] - time_current25_E4[i])
for i in range(np.size(current25_E4)-1):
    nu_eff25_E4[i] = (0.00004 - dJdt25_E4[i]) / current25_E4[i]


nu_eff25_pd = pd.Series(nu_eff25)
windows_25 = nu_eff25_pd.rolling(1)
nu_eff25_ = np.array(windows_25.mean().tolist())

nu_eff25_E4_pd = pd.Series(nu_eff25_E4)
windows_25_E4 = nu_eff25_E4_pd.rolling(1)
nu_eff25_E4_ = np.array(windows_25_E4.mean().tolist())

nu_eff25_E3_pd = pd.Series(nu_eff25_E3)
windows_25_E3 = nu_eff25_E3_pd.rolling(1)
nu_eff25_E3_ = np.array(windows_25_E3.mean().tolist())

nu_eff25_E2_pd = pd.Series(nu_eff25_E2)
windows_25_E2 = nu_eff25_E2_pd.rolling(1)
nu_eff25_E2_ = np.array(windows_25_E2.mean().tolist())

nu_eff25_E1_pd = pd.Series(nu_eff25_E1)
windows_25_E1 = nu_eff25_E1_pd.rolling(1)
nu_eff25_E1_ = np.array(windows_25_E1.mean().tolist())

#####################
#end of data reading
####################
########################################################################################################

def current_massratio():
    fig      = plt.figure(figsize=(10.5,7.5))
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    ax.plot(time_current25[:],current25,label='25',linewidth=5)
    #ax.plot(time_current50[:],current50,label='50',linewidth=5)
    ax.plot(time_current100[:],current100,label='100',linewidth=5)
    #ax.plot(time_current200[:],current200,label='200',linewidth=5)
    ax.plot(time_current400[:],current400,label='400',linewidth=5)

    # #ax.plot(time_current1[1:],nu_eff1,label='25',linewidth=5,color='red',linestyle='-')
    # #ax.plot(time_fieldEnergy1,fieldEnergy1/0.0004,linewidth=5,color='red',linestyle='--')
    # #ax.plot(time_current2[1:],nu_eff2,label='100',linewidth=5,color='green',linestyle='-')
    # #ax.plot(time_fieldEnergy2,fieldEnergy2/0.0004,linewidth=5,color='green',linestyle='--')
    # ax.plot(time_current3[1:],nu_eff3,label='400',linewidth=5,color='blue',linestyle='-')
    # ax.plot(time_fieldEnergy3,fieldEnergy3/0.0006,label='W/nTe',linewidth=5,color='blue',linestyle='--')


    ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)

    #ax.set_ylabel(r'$<J_z> [en_0 v_{Te0}]$',fontsize=32,color='blue')
    ax.set_xlim(0,2800)
    ax.set_ylim(0,0.035)
    ax.tick_params(labelsize = 26)
    ax.tick_params(axis='y',colors = 'blue')
    ax.legend(fontsize=25)
    ax.grid()
    #ax.set_xlim(0,2000)
    plt.show()
    plt.clf()

def nueff_massratio():
    fig      = plt.figure(figsize=(9.5,7.5))
    ax      = fig.add_axes([0.18, 0.16, 0.75, 0.75])

    ax.plot(time_current25[4:],nu_eff25[3:],label='$m_i/m_e = 25$',linewidth=4,linestyle='-')
    #ax.plot(time_current50[4:],nu_eff50[3:],label='50',linewidth=3,linestyle='-')
    ax.plot(time_current100[4:],nu_eff100[3:],label='$m_i/m_e = 100$',linewidth=4,linestyle='-')
    #ax.plot(time_current200[4:],nu_eff200[3:],label='200',linewidth=3,linestyle='-')
    ax.plot(time_current400[4:],nu_eff400[3:],label='$m_i/m_e = 400$',linewidth=4,linestyle='-')

    i = 0
    new_Elctemp25 = []
    new_Elctemp400 = []
    for t in Elctemp25:
        if i%4 == 0:
            new_Elctemp25.append(t)
        i += 1
    new_Elctemp25 = np.array(new_Elctemp25)
    i = 0
    for t in Elctemp400:
        if i%3 == 0:
            new_Elctemp400.append(t)
        i += 1
    new_Elctemp400 = np.array(new_Elctemp400)
    ax.plot(time_fieldEnergy25,fieldEnergy25/new_Elctemp25,label=r'$(W/nTe)_{25}$',linewidth=3,linestyle='--')
    #ax.plot(time_fieldEnergy400,fieldEnergy400/new_Elctemp400,label='W/nTe400',linewidth=5,linestyle='--')

    ax.hlines(0.0059,0,2000, linestyles='--',color='black',linewidth=3)
    ax.text(650,0.0066-0.001,"quasi-linear",fontsize = 18)
    ax.text(700,0.0063-0.001,"estimate",fontsize = 18)
    ax.text(650,0.0060-0.001,"for $m_i/m_e=25$",fontsize = 18)

    ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
    ax.set_ylabel(r'$\nu_{eff} [\omega_{pe}]$',fontsize=32)

    ax.set_ylim(-0.0005,0.007)
    ax.tick_params(labelsize = 23)
    ax.tick_params(axis='y')
    ax.legend(fontsize=25)
    ax.grid()
    ax.set_xlim(300,2000)
    plt.show()
    plt.cla()

def nueff_electricfield():
    fig      = plt.figure(figsize=(9.5,7.5))
    ax      = fig.add_axes([0.18, 0.16, 0.75, 0.75])


    ax.plot(time_current25[5:],nu_eff25_[4:],label=r'$E_{ext} = 5\times 10^{-5} en_0 d_e/\epsilon_0$',linewidth=4,linestyle='-')
    # ax.plot(time_current25_E4[4:]+75,nu_eff25_E4_[3:],label='$E4$',linewidth=4,linestyle='-')
    # ax.plot(time_current25_E3[4:]+100,nu_eff25_E3_[3:],label='$E3$',linewidth=4,linestyle='-')
    ax.plot(time_current25_E2[4:]+150,nu_eff25_E2_[3:],label=r'$E_{ext} = 2\times 10^{-5} en_0 d_e/\epsilon_0$',linewidth=4,linestyle='-')
    ax.plot(time_current25_E1[4:]+300,nu_eff25_E1_[3:],label=r'$E_{ext} = 1\times 10^{-5} en_0 d_e/\epsilon_0$',linewidth=4,linestyle='-')



    # ax.hlines(0.0059,0,2500, linestyles='--',color='black',linewidth=3)
    # ax.text(20,0.0055,"quasi-linear",fontsize = 20)
    # ax.text(50,0.0051,"estimate",fontsize = 20)

    ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
    ax.set_ylabel(r'$\nu_{eff} [\omega_{pe}]$',fontsize=32)
    ax.set_ylim(-0.0005,0.005)
    ax.tick_params(labelsize = 23)
    ax.tick_params(axis='y')
    ax.legend(fontsize=25)
    ax.grid()
    ax.set_xlim(300,2000)
    plt.show()
    plt.cla()
    
def ionheatingtrace():
    fig     = plt.figure(figsize=(11.0,10.0))
    ax      = fig.add_axes([0.15, 0.15, 0.75, 0.82])

    ax.plot(time_Iontemp25[:],Iontemp25[:]/Iontemp25[0],linewidth=5,label=r'$m_i/m_e=25$')
    ax.plot(time_Iontemp50[:],Iontemp50[:]/Iontemp50[0],linewidth=5,label=r'$m_i/m_e=50$')
    ax.plot(time_Iontemp100[:],Iontemp100[:]/Iontemp100[0],linewidth=5,label=r'$m_i/m_e=100$')
    ax.plot(time_Iontemp200[:],Iontemp200[:]/Iontemp200[0],linewidth=5,label=r'$m_i/m_e=200$')
    ax.plot(time_Iontemp400[:],Iontemp400[:]/Iontemp400[0],linewidth=5,label=r'$m_i/m_e=400$')

    ax.plot(time_Iontemp25_E1[:],Iontemp25_E1[:]/Iontemp25_E1[0],linewidth=5,label=r'$E1$')
    ax.plot(time_Iontemp25_E2[:],Iontemp25_E2[:]/Iontemp25_E2[0],linewidth=5,label=r'$E2$')
    ax.plot(time_Iontemp25_E3[:],Iontemp25_E3[:]/Iontemp25_E3[0],linewidth=5,label=r'$E3$')
    ax.plot(time_Iontemp25_E4[:],Iontemp25_E4[:]/Iontemp25_E4[0],linewidth=5,label=r'$E4$')

    ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
    ax.set_ylabel(r'$T_e/T_{i}$',fontsize=36,color='black')
    #ax.set_xlim(0,8000)
    ax.grid()
    ax.tick_params(labelsize = 24)
    ax.legend(fontsize=30)
    plt.show()

def tempratio_ratio():
    fig      = plt.figure(figsize=(9.5,7.5))
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    ax.plot(time_Iontemp25[:],Elctemp25[:]/Iontemp25[:],linewidth=5,label=r'$E_{5}, m_i/m_e=25$')
    ax.plot(time_Iontemp50[:],Elctemp50[:]/Iontemp50[:],linewidth=5,label=r'$E_{5}, m_i/m_e=50$')
    ax.plot(time_Iontemp100[:250],Elctemp100[:250]/Iontemp100[:250],linewidth=5,label=r'$E_{5}, m_i/m_e=100$')

    ax.plot(time_Iontemp25_E1[:],Elctemp25_E1[:]/Iontemp25_E1[:],linewidth=5,label=r'$E_{1}, m_i/m_e=25$')
    ax.plot(time_Iontemp25_E2[:],Elctemp25_E2[:]/Iontemp25_E2[:],linewidth=5,label=r'$E_{2}, m_i/m_e=25$')

    ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)

    ax.set_ylabel(r'$T_e/T_i$',fontsize=32)
    ax.set_xlim(0,2800)
    #ax.set_ylim(0,0.035)
    ax.tick_params(labelsize = 26)
    ax.tick_params(axis='y')
    ax.legend(fontsize=24)
    ax.grid()
    #ax.set_xlim(0,2000)
    plt.show()
    plt.clf()    

def ionheating():
    fig      = plt.figure(figsize=(9.5,7.5))
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.70])
    ax2 = ax.twiny()
    
    mass_ratio = np.sqrt(np.array([25,50,100,200,400]))
    Ti_mass = np.array([16,21,27,41,56])

    E_ext = np.array([1,2,3,4,5]) * 1e-5
    Ti_E = np.array([4.2,8.1,11.3,13.8,15.3])

    def linear_model(x,a,b):
        return a*x + b

    mass = np.sqrt(np.arange(24,400,1))
    popt1, pcov1 = curve_fit(linear_model, mass_ratio, Ti_mass)
    constructed_1 = linear_model(mass,*popt1)

    Ee = np.arange(1,6,1) * 1e-5
    popt2, pcov2 = curve_fit(linear_model, E_ext, Ti_E)
    constructed_2 = linear_model(Ee,*popt2)

    ax2.scatter(mass_ratio, Ti_mass, marker='x',s = 500, color='red')
    ax.scatter(E_ext, Ti_E, marker='x',s = 500, color='blue')
    ax2.plot(mass, constructed_1, color='red', linestyle='--',linewidth=3)
    ax.plot(Ee, constructed_2, color='blue', linestyle='--',linewidth=3)

    ax.tick_params(axis='y', labelsize=26)
    ax.tick_params(axis='x',labelsize=26,colors='blue')
    ax2.tick_params(axis='x', labelsize=26,colors='red')
    
    ax.xaxis.offsetText.set_fontsize(24)
    ax.xaxis.offsetText.set_color('blue')

    ax2.set_xlabel(r'$\sqrt{m_i/m_e}$',fontsize=32,color='red')
    ax.set_ylabel(r'$T_{if}/T_{i0}$',fontsize=32)
    ax.set_xlabel(r'$E_{ext}/(en_0d_e/\epsilon_0)$',fontsize=32, color='blue')

    ax.grid(axis='x',color='blue',linestyle='--')
    ax.grid(axis='y',linestyle='--')
    ax2.grid(color='red',linestyle='--')

    plt.show()


def elcheatingtrace_mass():
    fig     = plt.figure(figsize=(11.0,10.0))
    ax      = fig.add_axes([0.15, 0.15, 0.75, 0.82])

    dTedt25 = np.zeros(np.size(Elctemp25)-1)
    for i in range(np.size(Elctemp25)-1):
        dTedt25[i] = (Elctemp25[i+1] - Elctemp25[i]) / (time_Elctemp25[i+1] - time_Elctemp25[i])

    dTedt50 = np.zeros(np.size(Elctemp50)-1)
    for i in range(np.size(Elctemp50)-1):
        dTedt50[i] = (Elctemp50[i+1] - Elctemp50[i]) / (time_Elctemp50[i+1] - time_Elctemp50[i])

    dTedt100 = np.zeros(np.size(Elctemp100)-1)
    for i in range(np.size(Elctemp100)-1):
        dTedt100[i] = (Elctemp100[i+1] - Elctemp100[i]) / (time_Elctemp100[i+1] - time_Elctemp100[i])

    dTedt200 = np.zeros(np.size(Elctemp200)-1)
    for i in range(np.size(Elctemp200)-1):
        dTedt200[i] = (Elctemp200[i+1] - Elctemp200[i]) / (time_Elctemp200[i+1] - time_Elctemp200[i])

    dTedt400 = np.zeros(np.size(Elctemp400)-1)
    for i in range(np.size(Elctemp400)-1):
        dTedt400[i] = (Elctemp400[i+1] - Elctemp400[i]) / (time_Elctemp400[i+1] - time_Elctemp400[i])

    dTedtE1_pd = pd.Series(dTedt25)
    windows_1 = dTedtE1_pd.rolling(100)
    dTedtE1_ = np.array(windows_1.mean().tolist())

    dTedtE2_pd = pd.Series(dTedt50)
    windows_2 = dTedtE2_pd.rolling(100)
    dTedtE2_ = np.array(windows_2.mean().tolist())

    dTedtE3_pd = pd.Series(dTedt100)
    windows_3 = dTedtE3_pd.rolling(100)
    dTedtE3_ = np.array(windows_3.mean().tolist())

    dTedtE4_pd = pd.Series(dTedt200)
    windows_4 = dTedtE4_pd.rolling(100)
    dTedtE4_ = np.array(windows_4.mean().tolist())

    dTedtE5_pd = pd.Series(dTedt400)
    windows_5 = dTedtE5_pd.rolling(100)
    dTedtE5_ = np.array(windows_5.mean().tolist())

    # ax.plot(time_Elctemp25[:],Elctemp25[:]/Elctemp25[0],linewidth=5,label=r'$m_i/m_e=25$')
    # ax.plot(time_Elctemp50[:],Elctemp50[:]/Elctemp50[0],linewidth=5,label=r'$m_i/m_e=50$')
    # ax.plot(time_Elctemp100[:],Elctemp100[:]/Elctemp100[0],linewidth=5,label=r'$m_i/m_e=100$')
    # ax.plot(time_Elctemp200[:],Elctemp200[:]/Elctemp200[0],linewidth=5,label=r'$m_i/m_e=200$')
    # ax.plot(time_Elctemp400[:],Elctemp400[:]/Elctemp400[0],linewidth=5,label=r'$m_i/m_e=400$')

    ax.plot(time_Elctemp25[1:],dTedtE1_,linewidth=5,label=r'$m_i/m_e=25$')
    ax.plot(time_Elctemp50[1:],dTedtE2_,linewidth=5,label=r'$m_i/m_e=50$')
    ax.plot(time_Elctemp100[1:],dTedtE3_,linewidth=5,label=r'$m_i/m_e=100$')
    ax.plot(time_Elctemp200[1:],dTedtE4_,linewidth=5,label=r'$m_i/m_e=200$')
    ax.plot(time_Elctemp400[1:],dTedtE5_,linewidth=5,label=r'$m_i/m_e=400$')

    ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
    ax.set_ylabel(r'$T_e/T_{e0}$',fontsize=36,color='black')
    #ax.set_xlim(0,2000)
    #ax.set_ylim(0,7)
    ax.grid()
    ax.tick_params(labelsize = 24)
    ax.legend(fontsize=30)
    plt.show()

def elcheatingtrace_E():
    fig     = plt.figure(figsize=(11.0,10.0))
    ax      = fig.add_axes([0.15, 0.15, 0.75, 0.82])

    # ax.plot(time_Elctemp25_E1[:280],Elctemp25_E1[:280]/Elctemp25_E1[0],linewidth=5,label=r'$E1$') #110
    # ax.plot(time_Elctemp25_E2[:220],Elctemp25_E2[:220]/Elctemp25_E2[0],linewidth=5,label=r'$E2$') #130
    # ax.plot(time_Elctemp25_E3[:170],Elctemp25_E3[:170]/Elctemp25_E3[0],linewidth=5,label=r'$E3$') #180
    # ax.plot(time_Elctemp25_E4[:130],Elctemp25_E4[:130]/Elctemp25_E4[0],linewidth=5,label=r'$E4$') #220
    # ax.plot(time_Elctemp25[:220],Elctemp25[:220]/Elctemp25[0],linewidth=5,label=r'$E5$')

    dTedtE1 = np.zeros(np.size(Elctemp25_E1)-1)
    for i in range(np.size(Elctemp25_E1)-1):
        dTedtE1[i] = (Elctemp25_E1[i+1] - Elctemp25_E1[i]) / (time_Elctemp25_E1[i+1] - time_Elctemp25_E1[i])

    dTedtE2 = np.zeros(np.size(Elctemp25_E2)-1)
    for i in range(np.size(Elctemp25_E2)-1):
        dTedtE2[i] = (Elctemp25_E2[i+1] - Elctemp25_E2[i]) / (time_Elctemp25_E2[i+1] - time_Elctemp25_E2[i])

    dTedtE3 = np.zeros(np.size(Elctemp25_E3)-1)
    for i in range(np.size(Elctemp25_E3)-1):
        dTedtE3[i] = (Elctemp25_E3[i+1] - Elctemp25_E3[i]) / (time_Elctemp25_E3[i+1] - time_Elctemp25_E3[i])

    dTedtE4 = np.zeros(np.size(Elctemp25_E4)-1)
    for i in range(np.size(Elctemp25_E4)-1):
        dTedtE4[i] = (Elctemp25_E4[i+1] - Elctemp25_E4[i]) / (time_Elctemp25_E4[i+1] - time_Elctemp25_E4[i])

    dTedtE5 = np.zeros(np.size(Elctemp25)-1)
    for i in range(np.size(Elctemp25)-1):
        dTedtE5[i] = (Elctemp25[i+1] - Elctemp25[i]) / (time_Elctemp25[i+1] - time_Elctemp25[i])

    dTedtE1_pd = pd.Series(dTedtE1)
    windows_1 = dTedtE1_pd.rolling(50)
    dTedtE1_ = np.array(windows_1.mean().tolist())

    dTedtE2_pd = pd.Series(dTedtE2)
    windows_2 = dTedtE2_pd.rolling(50)
    dTedtE2_ = np.array(windows_2.mean().tolist())

    dTedtE3_pd = pd.Series(dTedtE3)
    windows_3 = dTedtE3_pd.rolling(50)
    dTedtE3_ = np.array(windows_3.mean().tolist())

    dTedtE4_pd = pd.Series(dTedtE4)
    windows_4 = dTedtE4_pd.rolling(50)
    dTedtE4_ = np.array(windows_4.mean().tolist())

    dTedtE5_pd = pd.Series(dTedtE5)
    windows_5 = dTedtE5_pd.rolling(50)
    dTedtE5_ = np.array(windows_5.mean().tolist())

    ax.plot(time_Elctemp25_E1[1:280],dTedtE1_[1:280],linewidth=5,label=r'$E1$')
    ax.plot(time_Elctemp25_E2[1:220],dTedtE2_[1:220],linewidth=5,label=r'$E2$')
    #ax.plot(time_Elctemp25_E3[1:170],dTedtE3_[1:170],linewidth=5,label=r'$E3$')
    ax.plot(time_Elctemp25_E4[1:130],dTedtE4_[1:130],linewidth=5,label=r'$E4$')
    ax.plot(time_Elctemp25[1:220],dTedtE5_[1:220],linewidth=5,label=r'$E5$')

    ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
    ax.set_ylabel(r'$T_e/T_{e0}$',fontsize=36,color='black')
    #ax.set_xlim(0,8000)
    ax.grid()
    ax.tick_params(labelsize = 24)
    ax.legend(fontsize=30)
    plt.show()

def elcheating_rate():
    dTedt25 = np.zeros(np.size(Elctemp25)-1)
    for i in range(np.size(Elctemp25)-1):
        dTedt25[i] = (Elctemp25[i+1] - Elctemp25[i]) / (time_Elctemp25[i+1] - time_Elctemp25[i])

    dTedt50 = np.zeros(np.size(Elctemp50)-1)
    for i in range(np.size(Elctemp50)-1):
        dTedt50[i] = (Elctemp50[i+1] - Elctemp50[i]) / (time_Elctemp50[i+1] - time_Elctemp50[i])

    dTedt100 = np.zeros(np.size(Elctemp100)-1)
    for i in range(np.size(Elctemp100)-1):
        dTedt100[i] = (Elctemp100[i+1] - Elctemp100[i]) / (time_Elctemp100[i+1] - time_Elctemp100[i])

    dTedt200 = np.zeros(np.size(Elctemp200)-1)
    for i in range(np.size(Elctemp200)-1):
        dTedt200[i] = (Elctemp200[i+1] - Elctemp200[i]) / (time_Elctemp200[i+1] - time_Elctemp200[i])

    dTedt400 = np.zeros(np.size(Elctemp400)-1)
    for i in range(np.size(Elctemp400)-1):
        dTedt400[i] = (Elctemp400[i+1] - Elctemp400[i]) / (time_Elctemp400[i+1] - time_Elctemp400[i])

    dTedtE1 = np.zeros(np.size(Elctemp25_E1)-1)
    for i in range(np.size(Elctemp25_E1)-1):
        dTedtE1[i] = (Elctemp25_E1[i+1] - Elctemp25_E1[i]) / (time_Elctemp25_E1[i+1] - time_Elctemp25_E1[i])

    dTedtE2 = np.zeros(np.size(Elctemp25_E2)-1)
    for i in range(np.size(Elctemp25_E2)-1):
        dTedtE2[i] = (Elctemp25_E2[i+1] - Elctemp25_E2[i]) / (time_Elctemp25_E2[i+1] - time_Elctemp25_E2[i])

    dTedtE3 = np.zeros(np.size(Elctemp25_E3)-1)
    for i in range(np.size(Elctemp25_E3)-1):
        dTedtE3[i] = (Elctemp25_E3[i+1] - Elctemp25_E3[i]) / (time_Elctemp25_E3[i+1] - time_Elctemp25_E3[i])

    dTedtE4 = np.zeros(np.size(Elctemp25_E4)-1)
    for i in range(np.size(Elctemp25_E4)-1):
        dTedtE4[i] = (Elctemp25_E4[i+1] - Elctemp25_E4[i]) / (time_Elctemp25_E4[i+1] - time_Elctemp25_E4[i])


    dTedt25_pd = pd.Series(dTedt25)
    windows_25 = dTedt25_pd.rolling(100)
    dTedt25_ = np.array(windows_25.mean().tolist())

    dTedt50_pd = pd.Series(dTedt50)
    windows_50 = dTedt50_pd.rolling(50)
    dTedt50_ = np.array(windows_50.mean().tolist())

    dTedt100_pd = pd.Series(dTedt100)
    windows_100 = dTedt100_pd.rolling(50)
    dTedt100_ = np.array(windows_100.mean().tolist())

    dTedt200_pd = pd.Series(dTedt200)
    windows_200 = dTedt200_pd.rolling(50)
    dTedt200_ = np.array(windows_200.mean().tolist())

    dTedt400_pd = pd.Series(dTedt400)
    windows_400 = dTedt400_pd.rolling(50)
    dTedt400_ = np.array(windows_400.mean().tolist())

    dTedtE1_pd = pd.Series(dTedtE1)
    windows_1 = dTedtE1_pd.rolling(50)
    dTedtE1_ = np.array(windows_1.mean().tolist())

    dTedtE2_pd = pd.Series(dTedtE2)
    windows_2 = dTedtE2_pd.rolling(50)
    dTedtE2_ = np.array(windows_2.mean().tolist())

    dTedtE3_pd = pd.Series(dTedtE3)
    windows_3 = dTedtE3_pd.rolling(50)
    dTedtE3_ = np.array(windows_3.mean().tolist())

    dTedtE4_pd = pd.Series(dTedtE4)
    windows_4 = dTedtE4_pd.rolling(50)
    dTedtE4_ = np.array(windows_4.mean().tolist())

    fig     = plt.figure(figsize=(13.0,10.0))
    ax      = fig.add_axes([0.15, 0.15, 0.75, 0.82])

    ax.plot(time_Elctemp400[1:],dTedt400_/0.0004,linewidth=5,label=r'$400,E5$')
    ax.plot(time_Elctemp200[1:],dTedt200_/0.0004,linewidth=5,label=r'$200,E5$')
    ax.plot(time_Elctemp100[1:],dTedt100_/0.0004,linewidth=5,label=r'$100,E5$')   
    ax.plot(time_Elctemp50[1:],dTedt50_/0.0004,linewidth=5,label=r'$50,E5$')

    ax.plot(time_Elctemp25[1:280],dTedt25_[1:280]/0.0004,linewidth=5,label=r'$25,E5$')
    ax.plot(time_Elctemp25_E4[1:150],dTedtE4_[1:150]/0.0004,linewidth=5,label=r'$25,E4$')
    ax.plot(time_Elctemp25_E3[1:200],dTedtE3_[1:200]/0.0004,linewidth=5,label=r'$25,E3$')
    ax.plot(time_Elctemp25_E2[1:240],dTedtE2_[1:240]/0.0004,linewidth=5,label=r'$25,E2$')
    ax.plot(time_Elctemp25_E1[1:280],dTedtE1_[1:280]/0.0004,linewidth=5,label=r'$25,E1$')

    ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
    ax.set_ylabel(r'$ (1/T_{e0}) d T_e / d t [\omega_{pe}^-1]$',fontsize=32,color='black')
    ax.set_xlim(0,3000)
    ax.grid()
    ax.tick_params(labelsize = 24)
    ax.legend(fontsize=30)
    plt.show()

if __name__ == '__main__':

    # nueff_massratio()
    # nueff_electricfield()

    # tempratio_ratio()

    # ionheatingtrace()

    # ionheating()

    # elcheatingtrace_mass()
    # elcheatingtrace_E()

    elcheating_rate()