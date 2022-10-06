import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np

fieldEnergy1 = np.loadtxt('./massRatio/mass25/E1/saved_data/fieldEnergy.txt')
time_fieldEnergy1 = np.loadtxt('./massRatio/mass25/E1/saved_data/fieldEnergy_time.txt')
fieldEnergy2 = np.loadtxt('./massRatio/mass100/E1-low1/saved_data/fieldEnergy.txt')
time_fieldEnergy2 = np.loadtxt('./massRatio/mass100/E1-low1/saved_data/fieldEnergy_time.txt')
fieldEnergy3 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/fieldEnergy.txt')
time_fieldEnergy3 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/fieldEnergy_time.txt')

Iontemp1 = np.loadtxt('./massRatio/mass25/E1/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp1 = np.loadtxt('./massRatio/mass25/E1/saved_data/ion_intM2Thermal_time.txt')
Elctemp1= np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM2Thermal.txt')
time_Elctemp1 = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM2Thermal_time.txt')
Iontemp2 = np.loadtxt('./massRatio/mass100/E1-low1/saved_data/ion_intM2Thermal.txt')*100
time_Iontemp2 = np.loadtxt('./massRatio/mass100/E1-low1/saved_data/ion_intM2Thermal_time.txt')
Elctemp2 = np.loadtxt('./massRatio/mass100/E1-low1/saved_data/elc_intM2Thermal.txt')
time_Elctemp2 = np.loadtxt('./massRatio/mass100/E1-low1/saved_data/elc_intM2Thermal_time.txt')
Iontemp3 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/ion_intM2Thermal.txt')*400
time_Iontemp3 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/ion_intM2Thermal_time.txt')
Elctemp3 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/elc_intM2Thermal.txt')
time_Elctemp3 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/elc_intM2Thermal_time.txt')

current1 = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM1i.txt')*2
time_current1 = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM1i_time.txt') + 300
current2 = np.loadtxt('./massRatio/mass100/E1-low1/saved_data/elc_intM1i.txt')*2
time_current2 = np.loadtxt('./massRatio/mass100/E1-low1/saved_data/elc_intM1i_time.txt') + 160
current3 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/elc_intM1i.txt')*2
time_current3 = np.loadtxt('./massRatio/mass400/E1-low1/saved_data/elc_intM1i_time.txt') + 80

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
    nu_eff2[i] = (0.00001 - dJdt2[i]) / current2[i]

dJdt3 = np.zeros(np.size(current3)-1)
nu_eff3 = np.zeros(np.size(current3)-1)
for i in range(np.size(current3)-1):
    dJdt3[i] = (current3[i+1] - current3[i]) / (time_current3[i+1] - time_current3[i])
for i in range(np.size(current3)-1):
    nu_eff3[i] = (0.00001 - dJdt3[i]) / current3[i]

# fig      = plt.figure(figsize=(12.5,7.5))
# ax      = fig.add_axes([0.16, 0.16, 0.82, 0.75])
# #ax.set_title("perturbed electric field energy",fontsize=16)
# ax.plot(time_fieldEnergy4,fieldEnergy4,label='#4',linewidth=5)
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=36)
# ax.set_ylabel(r'$\epsilon_0 \int (|\delta E_z|^2 + |\delta E_y|^2) dydz / n_0 T_{e0}$',fontsize=24,color='red')
# ax.vlines(350,0,1e-5,linestyle='--',linewidth=2,color='black')
# ax.vlines(750,0,1e-5,linestyle='--',linewidth=2,color='black')
# ax.vlines(1100,0,1e-5,linestyle='--',linewidth=2,color='black')
# ax.vlines(1800,0,1e-5,linestyle='--',linewidth=2,color='black')
# ax.set_yscale('log')
# ax.set_xlim(0,650)
# ax.set_ylim(1e-11,1e-5)
# ax.tick_params(labelsize = 28)
# #plt.savefig('./Cori/figure_temp/field_energy.jpg')
# plt.show()
# # ax.legend(fontsize=14)
# # plt.savefig('./Cori/figure_temp/field_energy.jpg')
# # plt.clf()

# fig      = plt.figure(figsize=(6.5,5.5))
# ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
# #ax.set_title("perturbed electric field energy",fontsize=16)
# ax.plot(time_Iontemp4,Iontemp4,label='#4',linewidth=5)
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=16)
# ax.set_ylabel(r'$T_i/T_{i0}$',fontsize=16)
# ax.set_xlim(0,2000)
# ax.tick_params(labelsize = 14)
# plt.savefig('./Cori/figure_temp/ion_temp.jpg')
# plt.clf()

# fig      = plt.figure(figsize=(6.5,5.5))
# ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
# #ax.set_title("perturbed electric field energy",fontsize=16)
# ax.plot(time_Elctemp4,Elctemp4/Elctemp4[0],label='#4',linewidth=5)
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=16)
# ax.set_ylabel(r'$T_e/T_{e0}$',fontsize=16)
# ax.set_xlim(0,2000)
# ax.tick_params(labelsize = 14)
# plt.savefig('./Cori/figure_temp/elc_temp.jpg')
# plt.clf()

# fig      = plt.figure(figsize=(6.5,5.5))
# ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
# #ax.set_title("perturbed electric field energy",fontsize=16)
# ax.plot(time_Iontemp4,Iontemp4/Iontemp4[0],label='#4',linewidth=5)
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=16)
# ax.set_ylabel(r'$T_i/T_{i0}$',fontsize=16)
# ax.set_xlim(0,2000)
# ax.tick_params(labelsize = 16)
# plt.savefig('./Cori/figure_temp/ion_temp.jpg')
# plt.clf()


#####################################
##### Current ####
###################################
 
fig      = plt.figure(figsize=(10.5,7.5))
ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

# ax.plot(time_current1[:],current1,label='25',linewidth=5)
# ax.plot(time_current2[:],current2,label='100',linewidth=5)
# ax.plot(time_current3[:],current3,label='400',linewidth=5)

#ax.plot(time_current1[1:],nu_eff1,label='25',linewidth=5,color='red',linestyle='-')
#ax.plot(time_fieldEnergy1,fieldEnergy1/0.0004,linewidth=5,color='red',linestyle='--')
#ax.plot(time_current2[1:],nu_eff2,label='100',linewidth=5,color='green',linestyle='-')
#ax.plot(time_fieldEnergy2,fieldEnergy2/0.0004,linewidth=5,color='green',linestyle='--')
ax.plot(time_current3[1:],nu_eff3,label='400',linewidth=5,color='blue',linestyle='-')
ax.plot(time_fieldEnergy3,fieldEnergy3/0.0006,label='W/nTe',linewidth=5,color='blue',linestyle='--')


ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)

#ax.set_ylabel(r'$<J_z> [en_0 v_{Te0}]$',fontsize=32,color='blue')
# ax.set_xlim(0,2700)
# ax.set_ylim(0,5.0)
ax.tick_params(labelsize = 26)
ax.tick_params(axis='y',colors = 'blue')
ax.legend(fontsize=25)
ax.grid()
#ax.set_xlim(0,3500)
plt.show()
#plt.clf()


#####################################
##### Temp ####
###################################
# fig     = plt.figure(figsize=(11.0,10.0))
# ax      = fig.add_axes([0.15, 0.15, 0.75, 0.82])
# ax2 = ax.twinx()

# ax.plot(time_Elctemp1[:],Elctemp1[:]/Elctemp1[0],linewidth=5,color='blue',linestyle='-',label='25')
# ax2.plot(time_Iontemp1[:],Iontemp1[:]/Iontemp1[0],linewidth=5,color='blue',linestyle='--')
# #ax2.plot(time_Iontemp1[:],Elctemp1[:]/Iontemp1[:],linewidth=5,color='blue')
# ax.plot(time_Elctemp2[:],Elctemp2[:]/Elctemp2[0],linewidth=5,color='red',linestyle='-',label='100')
# ax2.plot(time_Iontemp2[:],Iontemp2[:]/Iontemp2[0],linewidth=5,color='red',linestyle='--')
# #ax2.plot(time_Iontemp2[:],Elctemp2[:]/Iontemp2[:],linewidth=5,color='red')
# ax.plot(time_Elctemp3[:],Elctemp3[:]/Elctemp3[0],linewidth=5,color='green',linestyle='-',label='400')
# ax2.plot(time_Iontemp3[:],Iontemp3[:]/Iontemp3[0],linewidth=5,color='green',linestyle='--')
# #ax2.plot(time_Iontemp3[:],Elctemp3[:]/Iontemp3[:],linewidth=5,color='green')
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
# ax.set_ylabel(r'$T_e/T_{e0}$',fontsize=36,color='black')
# ax2.set_ylabel(r'$T_i/T_{ei0}$',fontsize=36,color='black')
# #ax.set_xlim(0,6000)
# #ax.set_ylim(0.9,3)
# #ax.set_xlim(750,1800)
# #ax2.set_ylim(0.9,13)
# ax.tick_params(labelsize = 24)
# ax2.tick_params(labelsize = 24)
# #ax2.ticklabel_format(axis='y', style='sci', scilimits=(1,1))
# #ax.legend(fontsize=30,loc='center right',bbox_to_anchor=(1.0, 0.3))
# #ax.legend(fontsize=30,loc='lower left')
# ax.legend(fontsize=30)
# plt.show()

# fig     = plt.figure(figsize=(11.0,10.0))
# ax      = fig.add_axes([0.15, 0.15, 0.75, 0.82])

# ax.plot(time_Iontemp1[:],Elctemp1[:]/Iontemp1[:],linewidth=5,color='blue',label='25')
# ax.plot(time_Iontemp2[:],Elctemp2[:]/Iontemp2[:],linewidth=5,color='red',label='100')
# ax.plot(time_Iontemp3[:],Elctemp3[:]/Iontemp3[:],linewidth=5,color='green',label='400')

# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
# ax.set_ylabel(r'$T_e/T_{i}$',fontsize=36,color='black')
# #ax.set_xlim(0,6000)
# ax.tick_params(labelsize = 24)
# ax.legend(fontsize=30)
# plt.show()

dTidt1 = np.zeros(np.size(Iontemp1)-1)
dWdt1 = np.zeros(np.size(fieldEnergy1)-1)
for i in range(np.size(Iontemp1)-1):
    dTidt1[i] = (Iontemp1[i+1] - Iontemp1[i]) / (time_Iontemp1[i+1] - time_Iontemp1[i])
for i in range(np.size(fieldEnergy1)-1):
    dWdt1[i] = (fieldEnergy1[i+1] - fieldEnergy1[i]) / (time_fieldEnergy1[i+1] - time_fieldEnergy1[i])
dTidt2 = np.zeros(np.size(Iontemp2)-1)
dWdt2 = np.zeros(np.size(fieldEnergy2)-1)
for i in range(np.size(Iontemp2)-1):
    dTidt2[i] = (Iontemp2[i+1] - Iontemp2[i]) / (time_Iontemp2[i+1] - time_Iontemp2[i])
for i in range(np.size(fieldEnergy2)-1):
    dWdt2[i] = (fieldEnergy2[i+1] - fieldEnergy2[i]) / (time_fieldEnergy2[i+1] - time_fieldEnergy2[i])
dTidt3 = np.zeros(np.size(Iontemp3)-1)
dWdt3 = np.zeros(np.size(fieldEnergy3)-1)
for i in range(np.size(Iontemp3)-1):
    dTidt3[i] = (Iontemp3[i+1] - Iontemp3[i]) / (time_Iontemp3[i+1] - time_Iontemp3[i])
for i in range(np.size(fieldEnergy3)-1):
    dWdt3[i] = (fieldEnergy3[i+1] - fieldEnergy3[i]) / (time_fieldEnergy3[i+1] - time_fieldEnergy3[i])

fig     = plt.figure(figsize=(11.0,10.0))
ax      = fig.add_axes([0.15, 0.15, 0.75, 0.82])

ax.plot(time_Iontemp1[1:],dTidt1,linewidth=3,label='20*dTdt1',color='red',linestyle='-')
ax.plot(time_fieldEnergy1,fieldEnergy1/20,linewidth=3,label='W1',color='red',linestyle='--')
ax.plot(time_Iontemp2[1:],dTidt2,linewidth=3,label='40*dTdt2',color='green',linestyle='-')
ax.plot(time_fieldEnergy2,fieldEnergy2/40,linewidth=3,label='W2',color='green',linestyle='--')
ax.plot(time_Iontemp3[1:],dTidt3,linewidth=3,label='80*dTdt3',color='blue',linestyle='-')
ax.plot(time_fieldEnergy3,fieldEnergy3/80,linewidth=3,label='W3',color='blue',linestyle='--')
#ax.plot(time_fieldEnergy2[1:],dWdt2,linewidth=5,label='dWdt')

ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
#ax.set_xlim(0,6000)
ax.set_ylim(1e-9,5e-6)
ax.tick_params(labelsize = 24)
ax.legend(fontsize=30)
ax.set_yscale('log')
plt.show()

#####################################
##### FIELD ENERGY ####
###################################
fig      = plt.figure(figsize=(10.5,7.5))
ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

ax.plot(time_fieldEnergy1[:],fieldEnergy1[:],label='25',linewidth=5)
ax.plot(time_fieldEnergy2[:],fieldEnergy2[:],label='100',linewidth=5)
ax.plot(time_fieldEnergy3[:],fieldEnergy3[:],label='400',linewidth=5)


ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
ax.tick_params(labelsize = 26)
ax.tick_params(axis='y',colors = 'blue')
ax.legend()
ax.set_yscale('log')
ax.set_xlim(300,)
#ax.set_ylim(1e-8,)
plt.show()
plt.clf()