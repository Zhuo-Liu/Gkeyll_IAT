from dataclasses import field
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np


fieldEnergy = np.loadtxt('./massRatio/mass25/E1/saved_data/fieldEnergy.txt')
time_fieldEnergy = np.loadtxt('./massRatio/mass25/E1/saved_data/fieldEnergy_time.txt')
Iontemp = np.loadtxt('./massRatio/mass25/E1/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp = np.loadtxt('./massRatio/mass25/E1/saved_data/ion_intM2Thermal_time.txt')
Elctemp = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM2Thermal.txt')
time_Elctemp = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM2Thermal_time.txt')
current = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM1i.txt')*2
time_current = np.loadtxt('./massRatio/mass25/E1/saved_data/elc_intM1i_time.txt')
dJdt = np.zeros(np.size(current)-1)
nu_eff = np.zeros(np.size(current)-1)
for i in range(np.size(current)-1):
    dJdt[i] = (current[i+1] - current[i]) / (time_current[i+1] - time_current[i])
for i in range(np.size(current)-1):
    nu_eff[i] = (0.00005 - dJdt[i]) / current[i]

#####
# field energy
####
fig      = plt.figure(figsize=(14.5,7.5))
ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])
ax2 = ax.twinx()

ax.plot(time_current,current/0.02,linewidth=5,color='blue')
ax2.plot(time_fieldEnergy,fieldEnergy/1e-4,linewidth=5,color='red')
ax.vlines(2800,0,6.0,linestyle='--',linewidth=2,color='black')

ax2.set_yscale('log')
#ax2.set_ylim(1e-11,1e-4)
ax2.tick_params(labelsize = 26,colors='red')
ax2.set_yscale('log')
#ax2.set_ylim(1e-11,1e-4)
ax2.tick_params(labelsize = 26,colors='red')
ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
ax.set_ylabel(r'$<J_z> [en_0 v_{Te0}]$',fontsize=32,color='blue')
ax.set_xlim(0,4000)
ax.set_ylim(0,1.5)
ax.tick_params(labelsize = 26)
ax.tick_params(axis='y',colors = 'blue')
plt.show()
#plt.savefig('./Cori/figure_temp/current.jpg')
plt.clf()

########
# Temp #
########
fig     = plt.figure(figsize=(11.0,10.0))
ax      = fig.add_axes([0.15, 0.15, 0.75, 0.82])
ax2 = ax.twinx()

ax.plot(time_Elctemp[:250],Elctemp[:250]/Elctemp[0],label='electron',linewidth=5,color='black',linestyle='-.')
ax.plot(time_Iontemp[:250],Iontemp[:250]/Iontemp[0],label='ion',linewidth=5,color='black',linestyle='--')
ax2.plot(time_fieldEnergy[:90],fieldEnergy[:90],linewidth=5,color='blue')
#ax2.plot(time_Iontemp[:220],Elctemp[:220]/Iontemp[:220],linewidth=5,color='blue')
ax.set_xlabel(r'$t \quad [\omega_{pe}^-1]$',fontsize=32)
ax.set_ylabel(r'$T/T_{0}$',fontsize=36,color='black')
#ax2.set_ylabel(r'$T_e/T_i$',fontsize=36,color='blue')
# ax.set_xlim(0,2600)
# ax.set_ylim(0,30)
#ax.set_xlim(750,1800)
#ax2.set_ylim(9,)
ax.tick_params(labelsize = 28)
ax2.tick_params(labelsize = 24,colors='blue')
ax2.ticklabel_format(axis='y', style='sci', scilimits=(1,1))
ax2.set_yscale('log')
#ax.legend(fontsize=30,loc='center right',bbox_to_anchor=(1.0, 0.3))
ax.legend(fontsize=30,loc='lower left')
#plt.savefig('./Diagnostics/local/Cori/figure_temp/temp.jpg')
plt.show()
#plt.clf()

########
# current and resistivity #
########
