import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


fieldEnergy4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/fieldEnergy.txt')
time_fieldEnergy4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/fieldEnergy_time.txt')
fieldEnergy = np.loadtxt('./Cori/mass25/temp200/saved_data/fieldEnergy.txt')
time_fieldEnergy= np.loadtxt('./Cori/mass25/temp200/saved_data/fieldEnergy_time.txt')
Iontemp4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/ion_intM2Thermal_time.txt')
Elctemp4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/elc_intM2Thermal.txt')
time_Elctemp4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/elc_intM2Thermal_time.txt')
current4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/elc_intM1i.txt')*2
time_current4 = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/elc_intM1i_time.txt')

time_current_1d = np.loadtxt('./Cori/mass25/rescheck/4/1d/elc_intM1i_time.txt')
current_1d = np.loadtxt('./Cori/mass25/rescheck/4/1d/elc_intM1i.txt')

dJdt4 = np.zeros(np.size(current4)-1)
nu_eff4 = np.zeros(np.size(current4)-1)
for i in range(np.size(current4)-1):
    dJdt4[i] = (current4[i+1] - current4[i]) / (time_current4[i+1] - time_current4[i])
for i in range(np.size(current4)-1):
    nu_eff4[i] = (0.00005 - dJdt4[i]) / current4[i]

# fig      = plt.figure(figsize=(12.5,7.5))
# ax      = fig.add_axes([0.16, 0.16, 0.82, 0.75])
# #ax.set_title("perturbed electric field energy",fontsize=16)
# ax.plot(time_fieldEnergy4,fieldEnergy4,label='temp-50',linewidth=5)
# ax.plot(time_fieldEnergy+10,fieldEnergy,label='temp-200',linewidth=5)
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=36)
# ax.set_ylabel(r'$\epsilon_0 \int (|\delta E_z|^2 + |\delta E_y|^2) dydz / n_0 T_{e0}$',fontsize=24,color='red')
# # ax.vlines(350,0,1e-5,linestyle='--',linewidth=2,color='black')
# # ax.vlines(750,0,1e-5,linestyle='--',linewidth=2,color='black')
# # ax.vlines(1100,0,1e-5,linestyle='--',linewidth=2,color='black')
# # ax.vlines(1800,0,1e-5,linestyle='--',linewidth=2,color='black')
# ax.set_yscale('log')
# ax.set_xlim(0,700)
# ax.set_ylim(1e-11,1e-5)
# ax.tick_params(labelsize = 28)
# ax.legend(fontsize=28)
# #plt.savefig('./Cori/figure_temp/field_energy.jpg')
# plt.show()
# ax.legend(fontsize=14)
# plt.savefig('./Cori/figure_temp/field_energy.jpg')
# plt.clf()

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

# fig      = plt.figure(figsize=(14.5,7.5))
# ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
# ax2 = ax.twinx()

# t1 = np.arange(750,1200)
# c1 = 0.0195 + 0.000025*(t1-750) - 0.001
# t2 = np.arange(1200,1900)
# c2 = 0.032 + 0.00004*(t2-1200) - 0.002
# ax.plot(time_current4,current4/0.02,label='2D',linewidth=5,color='blue')
# #ax.plot(time_current_1d,current_1d/0.02,label='1D',linewidth=4)
# #ax.plot(t1,c1/0.02,linewidth=6,linestyle='dashed',color='green')
# #ax.plot(t2,c2/0.02,linewidth=6,linestyle='dashed',color='orange')

# ax2.plot(time_fieldEnergy4,fieldEnergy4/1e-4,linewidth=5,color='red')
# ax2.set_yscale('log')
# #ax2.set_ylim(1e-11,1e-4)
# ax2.tick_params(labelsize = 26,colors='red')

# ax.vlines(350,0,6.0,linestyle='--',linewidth=2,color='black')
# ax.vlines(750,0,6.0,linestyle='--',linewidth=2,color='black')
# ax.vlines(1200,0,6.0,linestyle='--',linewidth=2,color='black')
# ax.vlines(1800,0,6.0,linestyle='--',linewidth=2,color='black')
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
# ax.set_ylabel(r'$<J_z> [en_0 v_{Te0}]$',fontsize=32,color='blue')
# ax.set_xlim(750,2000)
# ax.set_ylim(0,5.0)
# ax.tick_params(labelsize = 26)
# ax.tick_params(axis='y',colors = 'blue')
# #plt.show()
# plt.savefig('./Cori/figure_temp/current.jpg')
# plt.clf()

# fig      = plt.figure(figsize=(11.0,9.0))
# ax      = fig.add_axes([0.20, 0.16, 0.75, 0.75])
# #ax.set_title("perturbed electric field energy",fontsize=16)
# ax.plot(time_current4[4:],nu_eff4[3:],label=r'$\nu_{eff}$',linewidth=5)
# ax.plot(time_fieldEnergy4[2:],fieldEnergy4[2:]/0.0008,label=r'$\omega_{pe} W/nT_e$',linewidth=5)
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=36)
# ax.set_ylabel(r'$\omega_{pe}$',fontsize=36)
# ax.tick_params(labelsize = 28)
# ax.legend(fontsize=28,loc='center right')
# ax.set_xlim(0,1800)
# ax.set_ylim(-0.0002,0.006)
# ax.grid()
# ax.vlines(700,0,0.006,linestyles='--',linewidth=3)
# ax.vlines(1200,0,0.006,linestyles='--',linewidth=3)
# ax.vlines(350,0,0.006,linestyles='--',linewidth=3)
# ax.text(490,0.0055,"II",fontsize=36)
# ax.text(900,0.0055,"III",fontsize=36)
# ax.text(1450,0.0055,"IV",fontsize=36)
# plt.savefig('./Cori/figure_temp/nu_eff.jpg')

# #plt.savefig('./Cori/figure_temp/field_energy.jpg')
# plt.show()


###########################################################

fig      = plt.figure(figsize=(11.0,9.0))
ax      = fig.add_axes([0.12, 0.16, 0.75, 0.80])
ax2 = ax.twinx()

ax.plot(time_Elctemp4[:],Elctemp4[:]/Elctemp4[0],label='electron',linewidth=5,color='red',linestyle='-')
ax.plot(time_Iontemp4[:],Iontemp4[:]/Iontemp4[0],label='ion',linewidth=5,color='red',linestyle='--')
#ax2.plot(time_Iontemp4[70:95],Elctemp4[70:95]/Iontemp4[70:95],linewidth=5,color='blue')
ax2.plot(time_Iontemp4[:],Elctemp4[:]/Iontemp4[:],linewidth=5,color='blue')
ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
ax.set_ylabel(r'$T/T_{0}$',fontsize=36,color='red')
ax2.set_ylabel(r'$T_e/T_i$',fontsize=36,color='blue')
ax.set_xlim(0,2600)
ax.set_ylim(0,30)
ax.vlines(700,0,30,linestyles='--',linewidth=3)
ax.vlines(1200,0,30,linestyles='--',linewidth=3)
ax.vlines(350,0,30,linestyles='--',linewidth=3)
ax.vlines(1850,0,30,linestyles='--',linewidth=3)
ax.text(490,27.5,"II",fontsize=36)
ax.text(900,27.5,"III",fontsize=36)
ax.text(1450,27.5,"IV",fontsize=36)
ax.text(2150,27.5,"V",fontsize=36)

#ax.set_xlim(750,1800)
#ax.set_ylim(0,18)
ax.tick_params(labelsize = 28)
ax.tick_params(axis='y',colors='red')
ax2.tick_params(labelsize = 24,colors='blue')
ax.legend(fontsize=30,loc='center right',bbox_to_anchor=(1.0, 0.3))
ax.grid()
#ax.legend(fontsize=30,loc='lower left')
plt.savefig('./Cori/figure_temp/temp.jpg')
#plt.show()
plt.clf()