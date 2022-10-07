import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np


fieldEnergy1 = np.loadtxt('./massRatio/mass25/E1/saved_data/fieldEnergy.txt')
time_fieldEnergy1 = np.loadtxt('./massRatio/mass25/E1/saved_data/fieldEnergy_time.txt') + 300
fieldEnergy2 = np.loadtxt('./massRatio/mass25/E2/saved_data/fieldEnergy.txt')
time_fieldEnergy2 = np.loadtxt('./massRatio/mass25/E2/saved_data/fieldEnergy_time.txt') + 150
fieldEnergy3 = np.loadtxt('./massRatio/mass25/E3/saved_data/fieldEnergy.txt')
time_fieldEnergy3 = np.loadtxt('./massRatio/mass25/E3/saved_data/fieldEnergy_time.txt') + 100
fieldEnergy4 = np.loadtxt('./massRatio/mass25/E4/saved_data/fieldEnergy.txt')
time_fieldEnergy4 = np.loadtxt('./massRatio/mass25/E4/saved_data/fieldEnergy_time.txt') + 75
fieldEnergy5 = np.loadtxt('./massRatio/mass25/E5/saved_data/fieldEnergy.txt')
time_fieldEnergy5 = np.loadtxt('./massRatio/mass25/E5/saved_data/fieldEnergy_time.txt')

Iontemp1 = np.loadtxt('./massRatio/mass25/E1/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp1 = np.loadtxt('./massRatio/mass25/E1/saved_data/ion_intM2Thermal_time.txt') + 300
Elctemp1= np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM2Thermal.txt')
time_Elctemp1 = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM2Thermal_time.txt') + 300
Iontemp2 = np.loadtxt('./massRatio/mass25/E2/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp2 = np.loadtxt('./massRatio/mass25/E2/saved_data/ion_intM2Thermal_time.txt') + 150
Elctemp2 = np.loadtxt('./massRatio/mass25/E2/saved_data/elc_intM2Thermal.txt')
time_Elctemp2 = np.loadtxt('./massRatio/mass25/E2/saved_data/elc_intM2Thermal_time.txt') + 150
Iontemp3 = np.loadtxt('./massRatio/mass25/E3/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp3 = np.loadtxt('./massRatio/mass25/E3/saved_data/ion_intM2Thermal_time.txt') + 100
Elctemp3 = np.loadtxt('./massRatio/mass25/E3/saved_data/elc_intM2Thermal.txt')
time_Elctemp3 = np.loadtxt('./massRatio/mass25/E3/saved_data/elc_intM2Thermal_time.txt') + 100
Iontemp4 = np.loadtxt('./massRatio/mass25/E4/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp4 = np.loadtxt('./massRatio/mass25/E4/saved_data/ion_intM2Thermal_time.txt') + 75
Elctemp4 = np.loadtxt('./massRatio/mass25/E4/saved_data/elc_intM2Thermal.txt')
time_Elctemp4 = np.loadtxt('./massRatio/mass25/E4/saved_data/elc_intM2Thermal_time.txt') + 75
Iontemp5 = np.loadtxt('./massRatio/mass25/E5/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp5 = np.loadtxt('./massRatio/mass25/E5/saved_data/ion_intM2Thermal_time.txt') 
Elctemp5 = np.loadtxt('./massRatio/mass25/E5/saved_data/elc_intM2Thermal.txt')
time_Elctemp5 = np.loadtxt('./massRatio/mass25/E5/saved_data/elc_intM2Thermal_time.txt') 

current1 = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM1i.txt')*2
time_current1 = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM1i_time.txt') + 300
current2 = np.loadtxt('./massRatio/mass25/E2/saved_data/elc_intM1i.txt')*2
time_current2 = np.loadtxt('./massRatio/mass25/E2/saved_data/elc_intM1i_time.txt') + 150
current3 = np.loadtxt('./massRatio/mass25/E3/saved_data/elc_intM1i.txt')*2
time_current3 = np.loadtxt('./massRatio/mass25/E3/saved_data/elc_intM1i_time.txt') + 100
current4 = np.loadtxt('./massRatio/mass25/E4/saved_data/elc_intM1i.txt')*2
time_current4 = np.loadtxt('./massRatio/mass25/E4/saved_data/elc_intM1i_time.txt') + 75
current5 = np.loadtxt('./massRatio/mass25/E5/saved_data/elc_intM1i.txt')*2
time_current5 = np.loadtxt('./massRatio/mass25/E5/saved_data/elc_intM1i_time.txt')

dJdt1 = np.zeros(np.size(current1)-1)
nu_eff1 = np.zeros(np.size(current1)-1)
for i in range(np.size(current1)-1):
    dJdt1[i] = (current1[i+1] - current1[i]) / (time_current1[i+1] - time_current1[i])
for i in range(np.size(current1)-1):
    nu_eff1[i] = (0.00001 - dJdt1[i]) / current1[i]

dJdt2 = np.zeros(np.size(current2)-1)
nu_eff2 = np.zeros(np.size(current2)-1)
for i in range(np.size(current2)-1):
    dJdt2[i] = (current2[i+1] - current2[i]) / (time_current2[i+1] - time_current2[i])
for i in range(np.size(current2)-1):
    nu_eff2[i] = (0.00002 - dJdt2[i]) / current2[i]

dJdt3 = np.zeros(np.size(current3)-1)
nu_eff3 = np.zeros(np.size(current3)-1)
for i in range(np.size(current3)-1):
    dJdt3[i] = (current3[i+1] - current3[i]) / (time_current3[i+1] - time_current3[i])
for i in range(np.size(current3)-1):
    nu_eff3[i] = (0.00003 - dJdt3[i]) / current3[i]

dJdt4 = np.zeros(np.size(current4)-1)
nu_eff4 = np.zeros(np.size(current4)-1)
for i in range(np.size(current4)-1):
    dJdt4[i] = (current4[i+1] - current4[i]) / (time_current4[i+1] - time_current4[i])
for i in range(np.size(current4)-1):
    nu_eff4[i] = (0.00004 - dJdt4[i]) / current4[i]

dJdt5 = np.zeros(np.size(current5)-1)
nu_eff5 = np.zeros(np.size(current5)-1)
for i in range(np.size(current5)-1):
    dJdt5[i] = (current5[i+1] - current5[i]) / (time_current5[i+1] - time_current5[i])
for i in range(np.size(current5)-1):
    nu_eff5[i] = (0.00005 - dJdt5[i]) / current5[i]
#####################################
##### Current ####
###################################

# fig      = plt.figure(figsize=(10.5,7.5))
# ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

# ax.plot(time_current1[:],current1,label='E1',linewidth=5)
# ax.plot(time_current2[:],current2,label='E2',linewidth=5)
# ax.plot(time_current3[:],current3,label='E3',linewidth=5)
# ax.plot(time_current4[:],current4,label='E4',linewidth=5)
# ax.plot(time_current5[:],current5,label='E5',linewidth=5)

# # ax.plot(time_current1[1:],nu_eff1[:],label='E1',linewidth=5)
# # ax.plot(time_current2[1:241],nu_eff2[:240],label='E2',linewidth=5)
# # ax.plot(time_current3[1:201],nu_eff3[:200],label='E3',linewidth=5)
# # ax.plot(time_current4[1:181],nu_eff4[:180],label='E4',linewidth=5)
# # ax.plot(time_current5[2:141],nu_eff5[1:140],label='E5',linewidth=5)


# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)

# #ax.set_ylabel(r'$<J_z> [en_0 v_{Te0}]$',fontsize=32,color='blue')
# # ax.set_xlim(0,2700)
# # ax.set_ylim(0,5.0)
# ax.tick_params(labelsize = 26)
# ax.tick_params(axis='y',colors = 'blue')
# ax.legend()
# ax.grid()
# #ax.set_xlim(0,3500)
# plt.show()
# plt.clf()


#####################################
##### Temp ####
###################################
# fig     = plt.figure(figsize=(11.0,10.0))
# ax      = fig.add_axes([0.15, 0.15, 0.75, 0.82])
# ax2 = ax.twinx()

# ax.plot(time_Iontemp1[:300],Iontemp1[:300]/Iontemp1[0],linewidth=5,color='blue',linestyle='-',label='E1')
# #ax2.plot(time_Iontemp1[:],Iontemp1[:]/Iontemp1[0],linewidth=5,color='blue',linestyle='--')
# #ax2.plot(time_Iontemp1[:],Elctemp1[:]/Iontemp1[:],linewidth=5,color='blue')
# ax.plot(time_Iontemp2[:240],Iontemp2[:240]/Iontemp2[0],linewidth=5,color='red',linestyle='-',label='E2')
# #ax2.plot(time_Iontemp2[:240],Iontemp2[:240]/Iontemp2[0],linewidth=5,color='red',linestyle='--')
# #ax2.plot(time_Iontemp2[:],Elctemp2[:]/Iontemp2[:],linewidth=5,color='red')
# ax.plot(time_Iontemp3[:200],Iontemp3[:200]/Iontemp3[0],linewidth=5,color='green',linestyle='-',label='E3')
# #ax2.plot(time_Iontemp3[:200],Iontemp3[:200]/Iontemp3[0],linewidth=5,color='green',linestyle='--')
# #ax2.plot(time_Iontemp3[:],Elctemp3[:]/Iontemp3[:],linewidth=5,color='green')
# ax.plot(time_Iontemp4[:180],Iontemp4[:180]/Iontemp4[0],linewidth=5,color='purple',linestyle='-',label='E4')
# #ax2.plot(time_Iontemp4[:180],Iontemp4[:180]/Iontemp4[0],linewidth=5,color='purple',linestyle='--')
# #ax2.plot(time_Iontemp3[:],Elctemp3[:]/Iontemp3[:],linewidth=5,color='green')
# ax.plot(time_Iontemp5[:250],Iontemp5[:250]/Iontemp5[0],linewidth=5,color='cyan',linestyle='-',label='E5')
# #ax2.plot(time_Iontemp5[:400],Iontemp5[:400]/Iontemp5[0],linewidth=5,color='cyan',linestyle='--')
# #ax2.plot(time_Iontemp3[:],Elctemp3[:]/Iontemp3[:],linewidth=5,color='green')
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
# ax.set_ylabel(r'$T_i/T_{i0}$',fontsize=36,color='black')
# #ax2.set_ylabel(r'$T_i/T_{i0}$',fontsize=36,color='blue')
# # ax.set_xlim(0,2600)
# # ax.set_ylim(0,30)
# #ax.set_xlim(750,1800)
# #ax2.set_ylim(9,)
# ax.tick_params(labelsize = 24)
# #ax2.tick_params(labelsize = 24)
# #ax2.ticklabel_format(axis='y', style='sci', scilimits=(1,1))
# #ax.legend(fontsize=30,loc='center right',bbox_to_anchor=(1.0, 0.3))
# #ax.legend(fontsize=30,loc='lower left')
# ax.legend(fontsize=30)
# ax.grid()
# plt.show()

# plt.clf()

fig     = plt.figure(figsize=(11.0,10.0))
ax      = fig.add_axes([0.15, 0.15, 0.75, 0.82])

ax.plot(time_Iontemp1[:],Elctemp1[:]/Iontemp1[:],linewidth=5,color='blue',label=r'$E_{ext} = 10^{-5} $')
ax.plot(time_Iontemp2[:],Elctemp2[:]/Iontemp2[:],linewidth=5,color='red',label=r'$E_{ext} = 2 \times 10^{-5} $')
ax.plot(time_Iontemp3[:],Elctemp3[:]/Iontemp3[:],linewidth=5,color='green',label=r'$E_{ext} = 3 \times10^{-5} $')
ax.plot(time_Iontemp4[:],Elctemp4[:]/Iontemp4[:],linewidth=5,color='purple',label=r'$E_{ext} = 4 \times10^{-5} $')
ax.plot(time_Iontemp5[:],Elctemp5[:]/Iontemp5[:],linewidth=5,color='cyan',label=r'$E_{ext} = 5 \times10^{-5} $')

ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
ax.set_ylabel(r'$T_e/T_{i}$',fontsize=36,color='black')
#ax.set_xlim(0,6000)
ax.tick_params(labelsize = 24)
ax.legend(fontsize=30)
ax.grid()
plt.show()
#plt.savefig('./massRatio/mass25/tempratio.jpg')


#####################################
##### FIELD ENERGY ####
###################################
# fig      = plt.figure(figsize=(10.5,7.5))
# ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

# ax.plot(time_fieldEnergy1[:],fieldEnergy1[:],label='E1',linewidth=5)
# ax.plot(time_fieldEnergy2[:],fieldEnergy2[:],label='E2',linewidth=5)
# #ax.plot(time_fieldEnergy3[:],fieldEnergy3[:],label='E3',linewidth=5)
# #ax.plot(time_fieldEnergy4[:],fieldEnergy4[:],label='E4',linewidth=5)
# #ax.plot(time_fieldEnergy5[:],fieldEnergy5[:],label='E5',linewidth=5)


# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)

# #ax.set_ylabel(r'$<J_z> [en_0 v_{Te0}]$',fontsize=32,color='blue')
# # ax.set_xlim(0,2700)
# # ax.set_ylim(0,5.0)
# ax.tick_params(labelsize = 26)
# ax.tick_params(axis='y',colors = 'blue')
# ax.legend()
# ax.set_yscale('log')
# #ax.set_xlim(300,1500)
# #ax.set_ylim(1e-7,5e-5)
# plt.show()
# plt.clf()