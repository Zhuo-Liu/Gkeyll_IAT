#.Make plots from Gkyl data.
# Z.Liu 5/2021
# Auxiliary for crash data files

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from scipy.special import gamma


# fileName = {'field':'fieldEnergy','ionTemp':'ion_intM2Thermal','elcTemp':'elc_intM2Thermal','elcJ':'elc_intM1i','ionJ':'ion_intM1i'}
# for name in fileName:
#     print(fileName[name])
# gamma_list = -1.0*np.array([0.0475188, 0.0472384, 0.0469466, 0.0466454, 0.0463367, 0.0460225, \
# 0.0457049, 0.045386, 0.0450678, 0.0447526, 0.0441392, 0.0435616, \
# 0.0430343, 0.0425696, 0.0421773, 0.0414183, 0.0421853, 0.0440303])/0.3
# u_list = np.array([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3,3.5,4,4.5,5,7.5,10,15])

# f = interp1d(u_list, gamma_list,kind='cubic')
# us = np.arange(0.25,15,0.1)

# # plt.plot(us,f(us))
# # plt.title(r'$k\lambda_{De} = 4$')
# # plt.xlabel('$u/c_s$',fontsize=16)
# # plt.ylabel('$\gamma/\omega$',fontsize=16)
# # plt.ylim(-0.3,0)
# # plt.show()

# k_list = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 115, 120, 130, 140, \
# 150, 160, 170, 180, 190, 200])*0.02
# gamma_list_2 = np.array([0.00681695, 0.011875, 0.0144837, 0.0150025, 0.0141825, 0.0126662, \
# 0.0108203, 0.00875846, 0.0064431, 0.00379251, 0.000742666, \
# -0.000942886, -0.00273767, -0.00665352, -0.0109939, -0.0157396, \
# -0.0208678, -0.026355, -0.0321793, -0.0383183, -0.0447526]) / k_list


#########
#Buneman#
#########

# # #### t=1000
# gamma_list = np.array([0.0177,0.02,0.022,0.023,0.024,0.0243,0.0241,0.0234,0.0224,0.021,0.0192,0.017,0.015,0.0123,0.0096,0.00664])
# k_list=np.arange(10,42,2)
# f = interp1d(k_list, gamma_list,kind='cubic')
# plt.plot(k_list,f(k_list))
# plt.title(r'$t=1000$',fontsize=20)
# plt.xlabel(r'$k$',fontsize=16)
# plt.ylabel('$\gamma$',fontsize=16)
# #plt.xlim(0,2.0)
# plt.ylim(0,0.06)
# #plt.hlines(0,0.2,4,linestyles='--')
# plt.show()

# #### t=1200
# gamma_list = np.array([0.021,0.0237,0.0256,0.0267,0.0272,0.0270,0.0261,0.0247,0.0227,0.0203,0.0175,0.0143,0.0109,0.0071,0.0034,0.000])
# k_list=np.arange(10,42,2)
# f = interp1d(k_list, gamma_list,kind='cubic')
# plt.plot(k_list,f(k_list))
# plt.title(r'$t=1200$',fontsize=20)
# plt.xlabel(r'$k$',fontsize=16)
# plt.ylabel('$\gamma$',fontsize=16)
# plt.ylim(0,0.06)
# plt.show()

# #### t=1400
# gamma_list = np.array([0.03,0.034,0.036,0.0371,0.0370,0.0360,0.034,0.0313,0.0280,0.0241,0.0198,0.0152,0.0103,0.005,0.,-0.006])
# k_list=np.arange(10,42,2)
# f = interp1d(k_list, gamma_list,kind='cubic')
# plt.plot(k_list,f(k_list))
# plt.title(r'$t=1400$',fontsize=20)
# plt.xlabel(r'$k$',fontsize=16)
# plt.ylabel('$\gamma$',fontsize=16)
# plt.ylim(0,0.06)
# plt.show()

# #### t=1700
# gamma_list = np.array([0.0446,0.0499,0.0534,0.0552,0.0554,0.054,0.0515,0.048,0.044,0.0389,0.0337,0.0283,0.0227,0.017,0.0112,0.005])
# k_list=np.arange(10,42,2)
# f = interp1d(k_list, gamma_list,kind='cubic')
# plt.plot(k_list,f(k_list))
# plt.title(r'$t=1700$',fontsize=20)
# plt.xlabel(r'$k$',fontsize=16)
# plt.ylabel('$\gamma$',fontsize=16)
# plt.ylim(0,0.06)
# plt.show()

# f = interp1d(k_list, gamma_list_2,kind='cubic')
# us = np.arange(0.2,4,0.01)
# #us = np.arange(10,100,1)
# fs = f(us)

# plt.plot(us,fs)
# plt.title(r'$u_e = 0.01$')
# plt.xlabel(r'$k\lambda_{De}$',fontsize=16)
# plt.ylabel('$\gamma/k$',fontsize=16)
# plt.xlim(0,2.0)
# plt.ylim(0,0.04)
# #plt.hlines(0,0.2,4,linestyles='--')
# plt.show()

# Ez = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/high/data/Ez.txt')
# time_Ez = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/high/data/Ez_time.txt')
# plt.figure(figsize=(16, 12), dpi=80)
# plt.plot(time_Ez[:430],Ez[:430],label='absolute value of perturbed electric field, Ez')
# #plt.hlines(5e-5,0,2300,label='external elctric field, E_ext',color='red',linestyles='--')

# plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
# #plt.yscale('log')
# plt.tick_params(labelsize = 28)
# plt.legend(fontsize=24)
# plt.show()


# ####### Field Energy Plot #######
# fieldEnergy1 = np.loadtxt('./Diagnostics/local/Cori/Low/saved_data/fieldEnergy.txt')
# time_fieldEnergy1 = np.loadtxt('./Diagnostics/local/Cori/Low/saved_data/fieldEnergy_time.txt')
# fieldEnergy2 = np.loadtxt('./Diagnostics/local/Ratio/mass100/saved_data/fieldEnergy.txt')
# time_fieldEnergy2 = np.loadtxt('./Diagnostics/local/Ratio/mass100/saved_data/fieldEnergy_time.txt')
# fieldEnergy3 = np.loadtxt('./Diagnostics/local/Ratio/mass400/saved_data/fieldEnergy.txt')
# time_fieldEnergy3 = np.loadtxt('./Diagnostics/local/Ratio/mass400/saved_data/fieldEnergy_time.txt')
# fieldEnergy4 = np.loadtxt('./Diagnostics/local/Ratio/temp200/saved_data/fieldEnergy.txt')
# time_fieldEnergy4 = np.loadtxt('./Diagnostics/local/Ratio/temp200/saved_data/fieldEnergy_time.txt')
# fieldEnergy5 = np.loadtxt('./Diagnostics/local/Ratio/temp1/saved_data/fieldEnergy.txt')
# time_fieldEnergy5 = np.loadtxt('./Diagnostics/local/Ratio/temp1/saved_data/fieldEnergy_time.txt')


# # fieldEnergy2 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_check/E2_0.0001_2/saved_data/fieldEnergy.txt')
# # time_fieldEnergy2 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_check/E2_0.0001_2/saved_data/fieldEnergy_time.txt')
# # fieldEnergy3 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_check/E2_0.0001_3/saved_data/fieldEnergy.txt')
# # time_fieldEnergy3 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_check/E2_0.0001_3/saved_data/fieldEnergy_time.txt')
# # fieldEnergy4 = np.loadtxt('./Diagnostics/local/E2_0/low/saved_data/fieldEnergy.txt')
# # time_fieldEnergy4 = np.loadtxt('./Diagnostics/local/E2_0/low/saved_data/fieldEnergy_time.txt')
# # fieldEnergy5 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_check/E2_0.0001_recheck/saved_data/fieldEnergy.txt')
# # time_fieldEnergy5 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_check/E2_0.0001_recheck/saved_data/fieldEnergy_time.txt')

# plt.figure(figsize=(16, 12), dpi=80)
# #plt.plot(time_fieldEnergy2[:],fieldEnergy_z2[:],linewidth=5,label='high resolution, old')
# plt.plot(time_fieldEnergy1[:],fieldEnergy1[:],label='mass=25,temp=50')
# plt.plot(time_fieldEnergy2[:],fieldEnergy2[:],label='mass=100,temp=50')
# plt.plot(time_fieldEnergy3[:],fieldEnergy3[:],label='mass=400,temp=50')
# # plt.plot(time_fieldEnergy4[:],fieldEnergy4[:],label='mass=25,temp=200')
# # plt.plot(time_fieldEnergy5[:],fieldEnergy5[:],label='mass=25,temp=1')

# plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
# plt.ylabel(r'$\int dydz |\delta E_z|^2 + |\delta E_y|^2$',fontsize=36)
# plt.yscale('log')
# #plt.xlim(0,3200)
# plt.tick_params(labelsize = 28)
# plt.legend(fontsize=24)
# plt.show()

# # ####### Ion Temp Plot #######
# # Iontemp1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/ion_intM2Thermal.txt')
# # time_Iontemp1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/ion_intM2Thermal_time.txt')
# # Iontemp2 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/high/data/ion_intM2Thermal.txt')
# # time_Iontemp2 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/high/data/ion_intM2Thermal_time.txt')
# # Iontemp3 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/low/data/ion_intM2Thermal.txt')
# # time_Iontemp3 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/low/data/ion_intM2Thermal_time.txt')
# # Iontemp4 = np.loadtxt('./Diagnostics/local/reduced_cost/spaceyzionelc/E2_nu0.0001/data/ion_intM2Thermal.txt')
# # time_Iontemp4 = np.loadtxt('./Diagnostics/local/reduced_cost/spaceyzionelc/E2_nu0.0001/data/ion_intM2Thermal_time.txt')
# Iontemp1 = np.loadtxt('./Diagnostics/local/Cori/Low/saved_data/ion_intM2Thermal.txt')
# time_Iontemp1 = np.loadtxt('./Diagnostics/local/Cori/Low/saved_data/ion_intM2Thermal_time.txt')
# Iontemp2 = np.loadtxt('./Diagnostics/local/Ratio/mass100/saved_data/ion_intM2Thermal.txt')
# time_Iontemp2 = np.loadtxt('./Diagnostics/local/Ratio/mass100/saved_data/ion_intM2Thermal_time.txt')
Iontemp3 = np.loadtxt('./Diagnostics/local/Ratio/mass400/saved_data/ion_intM2Thermal.txt')
time_Iontemp3 = np.loadtxt('./Diagnostics/local/Ratio/mass400/saved_data/ion_intM2Thermal_time.txt')
# Iontemp4 = np.loadtxt('./Diagnostics/local/Ratio/temp200/saved_data/ion_intM2Thermal.txt')
# time_Iontemp4 = np.loadtxt('./Diagnostics/local/Ratio/temp200/saved_data/ion_intM2Thermal_time.txt')
# Iontemp5 = np.loadtxt('./Diagnostics/local/Ratio/temp1/saved_data/ion_intM2Thermal.txt')
# time_Iontemp5 = np.loadtxt('./Diagnostics/local/Ratio/temp1/saved_data/ion_intM2Thermal_time.txt')

# plt.figure(figsize=(16, 12), dpi=80)
# #plt.plot(time_Iontemp[:],Iontemp[1:]/Iontemp[0],linewidth=5,label='high resolution, old')
# plt.plot(time_Iontemp1[:650],Iontemp1[:650]/Iontemp1[0],label='mass=25,temp=50')
# plt.plot(time_Iontemp2[:],Iontemp2[:]/Iontemp2[0],label='mass=100,temp=50')
# plt.plot(time_Iontemp3[:],Iontemp3[:]/Iontemp3[0],label='mass=400,temp=50')
# #plt.plot(time_Iontemp4[:],Iontemp4[:]/Iontemp4[0],label='mass=25,temp=200')
# #plt.plot(time_Iontemp5[:],Iontemp5[:]/Iontemp5[0],label='mass=25,temp=1')
# plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
# plt.ylabel(r'$T_i/T_{i0}$',fontsize=36)
# #plt.yscale('log')
# #plt.ylim(0,10)
# plt.tick_params(labelsize = 28)
# plt.legend(fontsize=24)
# plt.show()

# # ####### Elc Temp Plot #######
# # Elctemp1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/elc_intM2Thermal.txt')
# # time_Elctemp1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/elc_intM2Thermal_time.txt')
# # Elctemp2 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/high/data/elc_intM2Thermal.txt')
# # time_Elctemp2 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/high/data/elc_intM2Thermal_time.txt')
# # Elctemp3 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/low/data/elc_intM2Thermal.txt')
# # time_Elctemp3 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/low/data/elc_intM2Thermal_time.txt')
# # Elctemp3 = np.loadtxt('./Diagnostics/local/reduced_cost/spaceyzionelc/E2_nu0.0001/data/elc_intM2Thermal.txt')
# # time_Elctemp3 = np.loadtxt('./Diagnostics/local/reduced_cost/spaceyzionelc/E2_nu0.0001/data/elc_intM2Thermal_time.txt')
# Elctemp1 = np.loadtxt('./Diagnostics/local/Cori/Low/saved_data/elc_intM2Thermal.txt')
# time_Elctemp1 = np.loadtxt('./Diagnostics/local/Cori/Low/saved_data/elc_intM2Thermal_time.txt')
# Elctemp2 = np.loadtxt('./Diagnostics/local/Ratio/mass100/saved_data/elc_intM2Thermal.txt')
# time_Elctemp2 = np.loadtxt('./Diagnostics/local/Ratio/mass100/saved_data/elc_intM2Thermal_time.txt')
Elctemp3 = np.loadtxt('./Diagnostics/local/Ratio/mass400/saved_data/elc_intM2Thermal.txt')
time_Elctemp3 = np.loadtxt('./Diagnostics/local/Ratio/mass400/saved_data/elc_intM2Thermal_time.txt')
# Elctemp4 = np.loadtxt('./Diagnostics/local/Ratio/temp200/saved_data/elc_intM2Thermal.txt')
# time_Elctemp4 = np.loadtxt('./Diagnostics/local/Ratio/temp200/saved_data/elc_intM2Thermal_time.txt')
# Elctemp5 = np.loadtxt('./Diagnostics/local/Ratio/temp1/saved_data/elc_intM2Thermal.txt')
# time_Elctemp5 = np.loadtxt('./Diagnostics/local/Ratio/temp1/saved_data/elc_intM2Thermal_time.txt')

plt.figure(figsize=(16, 12), dpi=80)
plt.title('mass ratio=400, temperature ratio=50',fontsize=28)
#plt.plot(time_Elctemp1[:650],Elctemp1[:650]/Elctemp1[0],label='mass=25,temp=50')
#plt.plot(time_Elctemp2[:],Elctemp2[:]/Elctemp2[0],label='mass=100,temp=50')
plt.plot(time_Elctemp3[:300],Elctemp3[:300]/Elctemp3[0],label='ELectron Temperature')
plt.plot(time_Iontemp3[:300],Iontemp3[:300]/Iontemp3[0],label='Ion Temperature')
# plt.plot(time_Elctemp4[:],Elctemp4[:]/Elctemp4[0],label='mass=25,temp=200')
# plt.plot(time_Elctemp5[:],Elctemp5[:]/Elctemp5[0],label='mass=25,temp=1')
plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
plt.ylabel(r'$T/T_{0}$',fontsize=36)
#plt.yscale('log')
#plt.ylim(0,10)
plt.tick_params(labelsize = 28)
plt.legend(fontsize=24)
plt.show()

# tempratio1 = Elctemp1/Iontemp1 /25
# tempratio2 = Elctemp2/Iontemp2 /100
# tempratio3 = Elctemp3/Iontemp3 /400
# plt.figure(figsize=(16, 12), dpi=80)
# plt.plot(time_Elctemp1[:650],tempratio1[:650],label='mass=25,temp=50')
# plt.plot(time_Elctemp2[:],tempratio2[:],label='mass=100,temp=50')
# plt.plot(time_Elctemp3[:],tempratio3[:],label='mass=400,temp=50')
# plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
# plt.ylabel(r'$T_e/T_i$',fontsize=36)
# plt.tick_params(labelsize = 28)
# plt.legend(fontsize=24)
# plt.show()

# # vTe = 0.02
# # vte_2 = np.sqrt(Elctemp2/Elctemp2[0])*vTe
# # vte_3 = np.sqrt(Elctemp3/Elctemp3[0])*vTe
####### Current Plot #######
# current1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/elc_intM1i.txt')*2
# time_current1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/elc_intM1i_time.txt')
# current2 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/high/data/elc_intM1i.txt')*2
# time_current2 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/high/data/elc_intM1i_time.txt')
# current3 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/low/data/elc_intM1i.txt')*2
# time_current3 = np.loadtxt('./Diagnostics/local/E2_nu0.0001_Cori/low/data/elc_intM1i_time.txt')
# current3 = np.loadtxt('./Diagnostics/local/reduced_cost/spaceyzionelc/E2_nu0.0001/data/elc_intM1i.txt')*2
# time_current3 = np.loadtxt('./Diagnostics/local/reduced_cost/spaceyzionelc/E2_nu0.0001/data/elc_intM1i_time.txt')
current1 = np.loadtxt('./Diagnostics/local/Cori/Low/saved_data/elc_intM1i.txt')*2
time_current1 = np.loadtxt('./Diagnostics/local/Cori/Low/saved_data/elc_intM1i_time.txt')
current2 = np.loadtxt('./Diagnostics/local/Ratio/mass100/saved_data/elc_intM1i.txt')*2
time_current2 = np.loadtxt('./Diagnostics/local/Ratio/mass100/saved_data/elc_intM1i_time.txt')
current3 = np.loadtxt('./Diagnostics/local/Ratio/mass400/saved_data/elc_intM1i.txt')*2
time_current3 = np.loadtxt('./Diagnostics/local/Ratio/mass400/saved_data/elc_intM1i_time.txt')
current4 = np.loadtxt('./Diagnostics/local/Ratio/temp200/saved_data/elc_intM1i.txt')*2
time_current4 = np.loadtxt('./Diagnostics/local/Ratio/temp200/saved_data/elc_intM1i_time.txt')
current5 = np.loadtxt('./Diagnostics/local/Ratio/temp1/saved_data/elc_intM1i.txt')*2
time_current5 = np.loadtxt('./Diagnostics/local/Ratio/temp1/saved_data/elc_intM1i_time.txt')

print(current1.shape)
print(current2.shape)
print(current3.shape)

def lineFunc(x,a,b):
  #.Compute the function y = a*x + b.
  return np.add(np.multiply(x,a),b)

poptMaxima1, _ = curve_fit(lineFunc, time_current1[120:160], current1[120:160])
poptMaxima2, _ = curve_fit(lineFunc, time_current2[90:120], current2[90:120])
poptMaxima3, _ = curve_fit(lineFunc, time_current3[130:180], current3[130:180])

# print(poptMaxima1)
# print(poptMaxima2)
# print(poptMaxima3)
plt.figure(figsize=(16, 12), dpi=80)
# plt.plot(time_current3[:],vte_2,linewidth=2,label='vthe, high resolution',color='red',linestyle='--')
# plt.plot(time_current4[:],vte_3,linewidth=2,label='vthe, low resolution',color='blue',linestyle='--')
plt.plot(time_current1[:550],current1[:550],label='mass=25,temp=50')
plt.plot(time_current2[:],current2[:],label='mass=100,temp=50')
plt.plot(time_current3[:300],current3[:300],label='mass=400,temp=50')
# plt.hlines(0.019,0,2000,linestyles='--')
#plt.plot(time_current4[:],current4[:],label='mass=25,temp=200')
#plt.plot(time_current5[:],current5[:],label='mass=25,temp=1')

# plt.plot(time_current1[120:160],lineFunc(time_current1[120:160],*poptMaxima1),label='fit1',linestyle='None',marker='o',markersize=4,markevery=2)
# plt.plot(time_current2[90:120],lineFunc(time_current2[90:120],*poptMaxima2),label='fit2',linestyle='None',marker='o',markersize=4,markevery=2)
# plt.plot(time_current3[130:180],lineFunc(time_current3[130:180],*poptMaxima3),label='fit3',linestyle='None',marker='o',markersize=4,markevery=2)
# plt.text(1500,0.05,str(poptMaxima1[0]),fontsize=24)
# plt.text(1800,0.035,str(poptMaxima2[0]),fontsize=24)
# plt.text(1750,0.02,str(poptMaxima3[0]),fontsize=24)
plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
plt.ylabel(r'$|<J_z>|$',fontsize=36)
#plt.xlim(0,2000)
plt.tick_params(labelsize = 28)
plt.legend(fontsize=24)
plt.show()



# ####### nu_eff Plot #######
# dJdt1 = np.zeros(np.size(current1)-2)
# nu_eff1 = np.zeros(np.size(current1)-2)
# for i in range(np.size(current1)-2):
#     dJdt1[i] = (current1[i+1] - current1[i]) / (time_current1[i+1] - time_current1[i])

# for i in range(np.size(current1)-2):
#     nu_eff1[i] = (0.00005 - dJdt1[i]) / current1[i]

# nu_eff_smooth1 = np.zeros(np.size(current1)-2)
# for i in range(np.size(current1)-50):
#     nu_eff_smooth1[i] = np.average(nu_eff1[i:i+48])

# dJdt2 = np.zeros(np.size(current2)-1)
# nu_eff2 = np.zeros(np.size(current2)-1)
# for i in range(np.size(current2)-1):
#     dJdt2[i] = (current2[i+1] - current2[i]) / (time_current2[i+1] - time_current2[i])

# for i in range(np.size(current2)-1):
#     nu_eff2[i] = (0.00005 - dJdt2[i]) / current2[i]

# nu_eff_smooth2 = np.zeros(np.size(current2)-1-50)
# for i in range(np.size(current2)-51):
#     nu_eff_smooth2[i] = np.average(nu_eff2[i:i+50])

# dJdt3 = np.zeros(np.size(current3)-1)
# nu_eff3 = np.zeros(np.size(current3)-1)
# for i in range(np.size(current3)-1):
#     dJdt3[i] = (current3[i+1] - current3[i]) / (time_current3[i+1] - time_current3[i])

# for i in range(np.size(current3)-2):
#     nu_eff3[i] = (0.00005 - dJdt3[i]) / current3[i]

# nu_eff_smooth3 = np.zeros(np.size(current3)-51)
# for i in range(np.size(current3)-51):
#     nu_eff_smooth3[i] = np.average(nu_eff3[i:i+50])

# dJdt4 = np.zeros(np.size(current4)-1)
# nu_eff4 = np.zeros(np.size(current4)-1)
# for i in range(np.size(current4)-1):
#     dJdt4[i] = (current4[i+1] - current4[i]) / (time_current4[i+1] - time_current4[i])

# for i in range(np.size(current4)-2):
#     nu_eff4[i] = (0.00005 - dJdt4[i]) / current4[i]

# nu_eff_smooth4 = np.zeros(np.size(current4)-51)
# for i in range(np.size(current4)-51):
#     nu_eff_smooth4[i] = np.average(nu_eff4[i:i+50])

# plt.figure(figsize=(16, 12), dpi=80)
# #plt.plot(time_current1[10:],nu_eff1[9:],label='high resolution, nu=0.0001')
# #plt.plot(time_current[10:1020],nu_eff_smooth[10:1020])
# plt.plot(time_current1[9:],nu_eff1[7:],label='mass=25,temp=50')
# plt.plot(time_current2[9:],nu_eff2[8:],label='mass=100,temp=50')
# plt.plot(time_current3[9:],nu_eff3[8:],label='mass=400,temp=50')
# plt.plot(time_current4[9:],nu_eff4[8:],label='mass=25,temp=200')
# plt.hlines(0.0002,0,3500,linestyles='--')
# plt.text(100,0.00025,'0.0002',fontsize=28)
# plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
# plt.ylabel(r'$<nu_{eff}>$',fontsize=36)
# plt.tick_params(labelsize = 28)
# plt.legend(fontsize=28)
# plt.show()


################# 1D distribution function ######################
# f_e = np.loadtxt('./Diagnostics/local/Cori/Low/saved_data/elc_dist_120.txt')
# v_z = np.loadtxt('./Diagnostics/local/Cori/Low/saved_data/elc_dist_vz_120.txt')
# f_e_0 = np.loadtxt('./Diagnostics/local/Cori/Low/saved_data/elc_dist_0.txt')

# def maxwellian_0(v):
#     A = 83.095
#     B = 2*0.02**2
#     return A*np.exp(-(v-0.00)**2/B)

# def maxwellian_1(v):
#     A = 46.3404
#     B = 2*0.019**2
#     C = 0.03

#     return A*np.exp(-(v-C)**2/B)

# def maxwellian_2(v):
#     A = 50.31563834802252
#     B = 2*0.015**2
#     C = 0.00

#     return A*np.exp(-(v-C)**2/B)

# def maxw(x,A,B,C):
#     return A*np.exp(-(x-C)**2/B)

# f_1 = np.array([maxwellian_1(v) for v in v_z])
# f_2 = np.array([maxwellian_2(v) for v in v_z])
# deltaf = f_e-f_1

f_e = np.loadtxt('./Diagnostics/local/Ratio/mass400/saved_data/elc_dist_130.txt')
v_z = np.loadtxt('./Diagnostics/local/Ratio/mass400/saved_data/elc_dist_vz_130.txt')
f_e_0 = np.loadtxt('./Diagnostics/local/Ratio/mass400/saved_data/elc_dist_0.txt')

def maxwellian(v):
    t = 1.7
    A = 83.095*np.sqrt(t)
    B = 2*0.02**2/t
    return A*np.exp(-(v-0.004)**2/B)

def maxwellian_0(v):
    A = 83.095
    B = 2*0.02**2
    return A*np.exp(-(v-0.00)**2/B)

def maxwellian_1(v):
    A = 12.0
    B = 2*0.022**2
    C = 0.07

    return A*np.exp(-(v-C)**2/B)

def maxwellian_2(v):
    A = 28.31563834802252
    B = 2*0.03**2
    C = 0.002

    return A*np.exp(-(v-C)**2/B)

def maxw(x,A,B,C):
    return A*np.exp(-(x-C)**2/B)

f_max_0 = np.array([maxwellian_0(v) for v in v_z])
f_max = np.array([maxwellian(v) for v in v_z])
f_1 = np.array([maxwellian_1(v) for v in v_z])
f_2 = np.array([maxwellian_2(v) for v in v_z])
deltaf = f_e-f_1

# popt, pcov = curve_fit(maxw, v_z, deltaf)

# plt.plot(v_z,f_max_0)    
#plt.plot(v_z,f_e_0)
#plt.plot(v_z,f_2,label="marginal unstable maxwellian F_e")    
plt.plot(v_z,f_e,label="F_e (t=1300)",linewidth=3)
plt.xlabel(r'$V_z$',fontsize=28)
plt.ylabel('$F_e(V_z)$',fontsize=28)
# plt.vlines(0.004,-10,80,linestyles='--',linewidth=3)
# plt.text(0.006,75,r'$U_{th}$',fontsize=32)
# plt.vlines(0.0451935 - 0.008,-10,40,linestyles='--',linewidth=3)
# plt.text(0.047,35,r'$u $',fontsize=32)
plt.plot(v_z,f_2,label="bulk",linewidth=2)
plt.plot(v_z,f_1,label="tail",linewidth=2)
plt.plot(v_z,f_2+f_1,label="bulk+tail",linewidth=2)
plt.legend()
plt.grid()
plt.show()

# # interp = 0
# # f_int1 = 0
# # f_int2 = 0
# # for i in range(359):
# #     deltat = v_z[i+1]-v_z[i]
# #     f = 0.5*(f_e[i]-f_max[i]+f_e[i+1]-f_max[i+1])
# #     interp += deltat*f
# #     f_int1 += deltat*(f_e_0[i]+f_e_0[i+1])*0.5
# #     f_int2 += deltat*(f_max_0[i]+f_max_0[i+1])*0.5
# # print(interp)
# # print(f_int1)
# # print(f_int2)
# #plt.xlim(-0.07,0.15)
# plt.legend(fontsize=36)
# plt.tick_params(labelsize = 28)
# plt.grid()
# plt.show()

##########
# def maxwellian_2d(vthe, udrift):
#     return 1/(2*np.pi*vthe**2)*np.exp(-((vthe[0]-udrift[0])**2+(vthe[1]-udrift[1])**2)/(2*vthe**2))

# def maxwellian_1d(v, vthe, udrift):
#     return 1/(2*np.pi*vthe**2)*np.exp(-((v-udrift)**2)/(2*vthe**2))

# C = np.sqrt(4*np.pi)

# tempamp = 1.0
# vte = 0.02*tempamp
# v0 = 0.02
# massratio = 25
# tempratio = 50
# cs = 0.02/np.sqrt(massratio)
# nuee = 0.0001
# betap = 0.18
# beta0 = 0.3
# E = 0.00005

# nuE = np.sqrt(9*np.pi/8)*E/cs
# ENL = 0.004*50/6/np.pi * C
# KN = E/ENL
# K = KN/(np.sqrt(KN+1)-1)
# vM = vte * (9*np.pi*nuee*vte**2/(betap*K*nuE*cs**2))**(1/6)
# tNL = tempratio/E/100

# nuee_crit = nuE/tempratio * (beta0/K + betap*K/np.pi/9)

# print(nuee_crit)
# print(vM)

# def fe_1(v):
#     exp1 = np.exp(-vM**2/2/vte**2 * (1/6*np.log((v**2+vM**2)**2/(v**4-v**2*vM**2+vM**4)+ 1/np.sqrt(3)*np.arctan(np.sqrt(3)*v**2/(2*vM**2-v**2))) ))
#     exp2 = np.exp(-np.pi*vM**2 / (3*np.sqrt(3)*vte**2))
#     if v<=0.041:
#         return maxwellian_1d(v,vte,0)*(exp1-exp2) + 84.2
#     else:
#         return maxwellian_1d(v,vte,0)*(exp1-exp2)
#     #return 10*(exp1-exp2)

# def tau1(t):
#     Be = beta0*nuE*cs**2*v0**3/K
#     return Be*t

# def fe_2(v,t):
#     return 1/4/np.pi/gamma(3/5)*(5*tau1(t)**3)**(-0.2)*np.exp(-(v-0.1)**5/25/tau1(t)) / 40 /10

# v = np.arange(0.06,0.26,0.001)
# f1 = np.array([fe_1(it) for it in v])
# f2 = np.array([fe_2(it,1000) for it in v])
# fM = np.array([maxwellian_1d(it,vte,0) for it in v])


# plt.plot(v,f2,label='relaxation')
# # plt.plot(v,f2,label='1')
# # plt.plot(v,f3,label='2')
# plt.plot(v/cs,fM,label='Maxwellian')
# plt.xlabel('$v/c_s$',fontsize=28)
# plt.legend(fontsize=28)
# plt.tick_params(labelsize = 28)
# plt.show()

# f_e = np.loadtxt('./Diagnostics/local/elc_dist_360.txt')[:]
# v_z = np.loadtxt('./Diagnostics/local/elc_dist_vz_360.txt')[:]

# def fefit(x,A,B,C):
#     return A*np.exp(-(x-B)**5/C)

# ff = np.array([fefit(it,32,0.036,0.000006) for it in v])

# #popt, pcov = curve_fit(fefit, v_z, f_e)
# #print(popt)

# plt.plot(v_z,f_e,label = 't=1800')
# plt.plot(v,ff,label='$\sim exp(-v^5)$')
# plt.legend(fontsize=28)
# plt.tick_params(labelsize = 28)
# plt.show()

