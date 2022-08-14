import matplotlib.pyplot as plt
import numpy as np


fieldEnergy4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/fieldEnergy.txt')
time_fieldEnergy4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/fieldEnergy_time.txt')
Iontemp4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/ion_intM2Thermal.txt')
time_Iontemp4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/ion_intM2Thermal_time.txt')
Elctemp4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/elc_intM2Thermal.txt')
time_Elctemp4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/elc_intM2Thermal_time.txt')
current4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/elc_intM1i.txt')*2
time_current4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/elc_intM1i_time.txt')

time_current_1d = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/1d/elc_intM1i_time.txt')
current_1d = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/1d/elc_intM1i.txt')

dJdt4 = np.zeros(np.size(current4)-1)
nu_eff4 = np.zeros(np.size(current4)-1)
for i in range(np.size(current4)-1):
    dJdt4[i] = (current4[i+1] - current4[i]) / (time_current4[i+1] - time_current4[i])
for i in range(np.size(current4)-1):
    nu_eff4[i] = (0.00005 - dJdt4[i]) / current4[i]

fig      = plt.figure(figsize=(6.5,5.5))
ax      = fig.add_axes([0.16, 0.16, 0.8, 0.75])
#ax.set_title("perturbed electric field energy",fontsize=16)
ax.plot(time_fieldEnergy4,fieldEnergy4,label='#4',linewidth=5)
ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=18)
ax.set_ylabel(r'$\int |\delta E_z|^2 + |\delta E_y|^2  dydz$',fontsize=16)
ax.set_yscale('log')
ax.set_xlim(0,2000)
ax.tick_params(labelsize = 14)
plt.show()
#ax.legend(fontsize=14)
# plt.savefig('./Diagnostics/local/Cori/figure_temp/field_energy.jpg')
# plt.clf()

# fig      = plt.figure(figsize=(6.5,5.5))
# ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
# #ax.set_title("perturbed electric field energy",fontsize=16)
# ax.plot(time_Iontemp4,Iontemp4,label='#4',linewidth=5)
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=16)
# ax.set_ylabel(r'$T_i/T_{i0}$',fontsize=16)
# ax.set_xlim(0,2000)
# ax.tick_params(labelsize = 14)
# plt.savefig('./Diagnostics/local/Cori/figure_temp/ion_temp.jpg')
# plt.clf()

# fig      = plt.figure(figsize=(6.5,5.5))
# ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
# #ax.set_title("perturbed electric field energy",fontsize=16)
# ax.plot(time_Elctemp4,Elctemp4/Elctemp4[0],label='#4',linewidth=5)
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=16)
# ax.set_ylabel(r'$T_e/T_{e0}$',fontsize=16)
# ax.set_xlim(0,2000)
# ax.tick_params(labelsize = 14)
# plt.savefig('./Diagnostics/local/Cori/figure_temp/elc_temp.jpg')
# plt.clf()

# fig      = plt.figure(figsize=(6.5,5.5))
# ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
# #ax.set_title("perturbed electric field energy",fontsize=16)
# ax.plot(time_Iontemp4,Iontemp4/Iontemp4[0],label='#4',linewidth=5)
# ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=16)
# ax.set_ylabel(r'$T_i/T_{i0}$',fontsize=16)
# ax.set_xlim(0,2000)
# ax.tick_params(labelsize = 16)
# plt.savefig('./Diagnostics/local/Cori/figure_temp/ion_temp.jpg')
# plt.clf()


#####################################
##### Current ####
###################################

fig      = plt.figure(figsize=(12.5,8.5))
ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
t1 = np.arange(750,1150)
c1 = 0.0195 + 0.000025*(t1-750) - 0.001
t2 = np.arange(1150,1900)
c2 = 0.032 + 0.00004*(t2-1200) - 0.002
ax.plot(time_current4,current4/0.02,label='2D',linewidth=4)
#ax.plot(time_current_1d,current_1d/0.02,label='1D',linewidth=4)
ax.plot(t1,c1/0.02,linewidth=4,linestyle='dotted',color='green')
ax.plot(t2,c2/0.02,linewidth=4,linestyle='dotted',color='orange')
ax.text(30,0.8,r'$\frac{d J_z}{d t} = \frac{e^2nE_{ext}}{m_e}$',fontsize=18)
ax.text(760,0.7,r'$\frac{d J_z}{d t} = 0.5 \frac{e^2nE_{ext}}{m_e}$',fontsize=18)
ax.text(1450,1.8,r'$\frac{d J_z}{d t} = 0.8 \frac{e^2nE_{ext}}{m_e}$',fontsize=18)

ax.vlines(350,0,3.5,linestyle='--',linewidth=2,color='black')
ax.vlines(750,0,3.5,linestyle='--',linewidth=2,color='black')
ax.vlines(1100,0,3.5,linestyle='--',linewidth=2,color='black')
ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=24)
ax.set_ylabel(r'$<J_z> [en_0 v_{Te}]$',fontsize=24)
ax.set_xlim(0,2000)
ax.set_ylim(0,3.25)
ax.tick_params(labelsize = 16)
plt.show()
# plt.savefig('./Diagnostics/local/Cori/figure_temp/current.jpg')
# plt.clf()