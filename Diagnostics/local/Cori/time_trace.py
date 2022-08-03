import matplotlib.pyplot as plt
import numpy as np


fieldEnergy4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/fieldEnergy.txt')
time_fieldEnergy4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/fieldEnergy_time.txt')
Iontemp4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/ion_intM2Thermal.txt')
time_Iontemp4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/ion_intM2Thermal_time.txt')
current4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/elc_intM1i.txt')*2
time_current4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/saved_data/elc_intM1i_time.txt')
dJdt4 = np.zeros(np.size(current4)-1)
nu_eff4 = np.zeros(np.size(current4)-1)
for i in range(np.size(current4)-1):
    dJdt4[i] = (current4[i+1] - current4[i]) / (time_current4[i+1] - time_current4[i])
for i in range(np.size(current4)-1):
    nu_eff4[i] = (0.00005 - dJdt4[i]) / current4[i]


plt.title("perturbed electric field energy",fontsize=28)
plt.plot(time_fieldEnergy4,fieldEnergy4,label='#4',linewidth=5)
plt.xlabel(r'$t \quad (\omega_{pe}^-1]$',fontsize=32)
plt.ylabel(r'$\int dydz |\delta E_z|^2 + |\delta E_y|^2$',fontsize=28)
plt.yscale('log')
plt.xlim(0,2000)
plt.tick_params(labelsize = 28)
plt.legend(fontsize=24)
plt.show()