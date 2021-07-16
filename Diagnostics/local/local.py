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

####### Field Energy Plot #######
fieldEnergy_z = np.loadtxt('./Diagnostics/local/E2_nu0.0001_reduced/data/fieldEnergy.txt')
#fieldEnergy_y = np.loadtxt('./Diagnostics/local/E2/data/fieldEnergy_y.txt')
#fieldEnergy = fieldEnergy_z + fieldEnergy_z
time_fieldEnergy = np.loadtxt('./Diagnostics/local/E2_nu0.0001_reduced/data/fieldEnergy_time.txt')
fieldEnergy_z2 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/fieldEnergy.txt')
#fieldEnergy_y = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/fieldEnergy_y.txt')
#fieldEnergy = fieldEnergy_z + fieldEnergy_z
time_fieldEnergy2 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/fieldEnergy_time.txt')
fieldEnergy_z3 = np.loadtxt('./Diagnostics/local/E2_nu0.00001/data/fieldEnergy.txt')
time_fieldEnergy3 = np.loadtxt('./Diagnostics/local/E2_nu0.00001/data/fieldEnergy_time.txt')

plt.figure(figsize=(16, 12), dpi=80)
plt.plot(time_fieldEnergy[:1020],fieldEnergy_z[:1020],label='low resolution, nu=0.0001')
plt.plot(time_fieldEnergy2,fieldEnergy_z2,label='high resolution, nu=0.0001')
plt.plot(time_fieldEnergy3,fieldEnergy_z3,label='high resolution, nu=0.0005')
plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
plt.ylabel(r'$\int dydz |\delta E_z|^2 + |\delta E_y|^2$',fontsize=36)
plt.yscale('log')
#plt.xlim(0,3200)
plt.tick_params(labelsize = 28)
plt.legend(fontsize=28)
plt.show()

# ####### Ion Temp Plot #######
Iontemp = np.loadtxt('./Diagnostics/local/E2_nu0.0001_reduced/data/ion_intM2Thermal.txt')
time_Iontemp = np.loadtxt('./Diagnostics/local/E2_nu0.0001_reduced/data/ion_intM2Thermal_time.txt')
Iontemp1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/ion_intM2Thermal.txt')
time_Iontemp1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/ion_intM2Thermal_time.txt')
Iontemp2 = np.loadtxt('./Diagnostics/local/E2_nu0.0005/data/ion_intM2Thermal.txt')
time_Iontemp2 = np.loadtxt('./Diagnostics/local/E2_nu0.0005/data/ion_intM2Thermal_time.txt')

plt.figure(figsize=(16, 12), dpi=80)
plt.plot(time_Iontemp[:1010],Iontemp[:1010]/Iontemp[0],label='low resolution, nu=0.0001')
plt.plot(time_Iontemp1[:],Iontemp1[1:]/Iontemp1[0],label='high resolution, nu=0.0001')
plt.plot(time_Iontemp2[:380],Iontemp2[:380]/Iontemp2[0],label='high resolution, nu=0.0005')
plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
plt.ylabel(r'$T_i/T_{i0}$',fontsize=36)
#plt.yscale('log')
#plt.ylim(0,10)
plt.tick_params(labelsize = 28)
plt.legend(fontsize=28)
plt.show()

# ####### Elc Temp Plot #######
Elctemp = np.loadtxt('./Diagnostics/local/E2_nu0.0001_reduced/data/elc_intM2Thermal.txt')
time_Elctemp = np.loadtxt('./Diagnostics/local/E2_nu0.0001_reduced/data/elc_intM2Thermal_time.txt')
Elctemp1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/elc_intM2Thermal.txt')
time_Elctemp1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/elc_intM2Thermal_time.txt')
Elctemp2 = np.loadtxt('./Diagnostics/local/E2_nu0.0005/data/elc_intM2Thermal.txt')
time_Elctemp2 = np.loadtxt('./Diagnostics/local/E2_nu0.0005/data/elc_intM2Thermal_time.txt')

plt.figure(figsize=(16, 12), dpi=80)
plt.plot(time_Elctemp[:1010],Elctemp[:1010]/Elctemp[0],label='low resolution, nu=0.0001')
plt.plot(time_Elctemp1[:],Elctemp1[1:]/Elctemp1[0],label='high resolution, nu=0.0001')
plt.plot(time_Elctemp2[:380],Elctemp2[:380]/Elctemp2[0],label='high resolution, nu=0.0005')
plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
plt.ylabel(r'$T_e/T_{e0}$',fontsize=36)
#plt.yscale('log')
#plt.ylim(0,10)
plt.tick_params(labelsize = 28)
plt.legend(fontsize=28)
plt.show()

####### Current Plot #######
current = np.loadtxt('./Diagnostics/local/E2_nu0.0001_reduced/data/elc_intM1i.txt')*2
time_current = np.loadtxt('./Diagnostics/local/E2_nu0.0001_reduced/data/elc_intM1i_time.txt')
current1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/elc_intM1i.txt')*2
time_current1 = np.loadtxt('./Diagnostics/local/E2_nu0.0001/data/elc_intM1i_time.txt')
current2 = -np.loadtxt('./Diagnostics/local/E2_nu0.0005/data/elc_z.txt')
time_current2 = np.loadtxt('./Diagnostics/local/E2_nu0.0005/data/elc_z_time.txt')

plt.figure(figsize=(16, 12), dpi=80)
plt.plot(time_current[:1020],current[:1020],label='low resolution, nu=0.0001')
plt.plot(time_current1[:],current1[1:],label='high resolution, nu=0.0001')
plt.plot(time_current2[:380],current2[:380],label='high resolution, nu=0.0005')
plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
plt.ylabel(r'$|<J_z>|$',fontsize=36)
plt.tick_params(labelsize = 28)
plt.legend(fontsize=28)
plt.show()

####### nu_eff Plot #######
dJdt = np.zeros(np.size(current)-2)
nu_eff = np.zeros(np.size(current)-2)
for i in range(np.size(current)-2):
    dJdt[i] = (current[i+1] - current[i]) / (time_current[i+1] - time_current[i])

for i in range(np.size(current)-2):
    nu_eff[i] = (0.00005 - dJdt[i]) / current[i]

nu_eff_smooth = np.zeros(np.size(current)-2)
for i in range(np.size(current)-50):
    nu_eff_smooth[i] = np.average(nu_eff[i:i+48])
dJdt1 = np.zeros(np.size(current1)-2)
nu_eff1 = np.zeros(np.size(current1)-2)
for i in range(np.size(current1)-2):
    dJdt1[i] = (current1[i+1] - current1[i]) / (time_current1[i+1] - time_current1[i])

for i in range(np.size(current1)-2):
    nu_eff1[i] = (0.00005 - dJdt1[i]) / current1[i]

plt.figure(figsize=(16, 12), dpi=80)
plt.plot(time_current[10:1020],nu_eff[10:1020],label='low resolution')
#plt.plot(time_current[10:1020],nu_eff_smooth[10:1020])
plt.plot(time_current1[10:],nu_eff1[9:],label='high resolution')
plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
plt.ylabel(r'$<nu_{eff}>$',fontsize=36)
plt.tick_params(labelsize = 28)
#plt.legend(fontsize=28)
plt.show()

# res = np.loadtxt('./Diagnostics/local/resistivity.txt')
# #res = res[300:]

# a = np.zeros(1228)
# b = 2*np.arange(0,1228) + 200
# # for i in range(500):
# #     a[i] = np.average(res[i:i+100])

# plt.plot(b,res)
# plt.show()

# res = np.loadtxt('./Diagnostics/local/data/nu_eff.txt')[:286]
# times = np.loadtxt('./Diagnostics/local/data/time_nu.txt')[:286]
# a = np.zeros(86)
# for i in range(0,86):
#     a[i] = np.average(res[i+100:i+200])

# # plt.plot(times,res)
# # plt.show()

# print(times[180])
# print(np.average(res[180:]))



################# 1D distribution function ######################
# f_e = np.loadtxt('./Diagnostics/local/data/elc_dist_1615.txt')
# v_z = np.loadtxt('./Diagnostics/local/data/elc_dist_vz_1615.txt')
# f_e_0 = np.loadtxt('./Diagnostics/local/data/elc_dist_0.txt')

# def maxwellian(v):
#     t = 1.7
#     A = 83.095*np.sqrt(t)
#     B = 2*0.02**2/t
#     return A*np.exp(-(v-0.004)**2/B)

# def maxwellian_0(v):
#     A = 83.095
#     B = 2*0.02**2
#     return A*np.exp(-(v-0.00)**2/B)

# def maxwellian_1(v):
#     A = 22.3404
#     B = 0.000657877
#     C = 0.0451935 - 0.008

#     return A*np.exp(-(v-C)**2/B)

# def maxwellian_2(v):
#     A = 67.31563834802252
#     B = 0.0006571545907741459
#     C = 0.004277912978811665

#     return A*np.exp(-(v-C)**2/B)

# def maxw(x,A,B,C):
#     return A*np.exp(-(x-C)**2/B)

# f_max_0 = np.array([maxwellian_0(v) for v in v_z])
# f_max = np.array([maxwellian(v) for v in v_z])
# f_1 = np.array([maxwellian_1(v) for v in v_z])
# f_2 = np.array([maxwellian_2(v) for v in v_z])
# deltaf = f_e-f_1

# # popt, pcov = curve_fit(maxw, v_z, deltaf)

# # plt.plot(v_z,f_max_0)    
# #plt.plot(v_z,f_e_0)
# #plt.plot(v_z,f_2,label="marginal unstable maxwellian F_e")    
# plt.plot(v_z,f_e_0,label="F_e (t=3200)",linewidth=1)
# plt.xlabel(r'$V_z$',fontsize=28)
# plt.ylabel('$F_e(V_z)$',fontsize=28)
# plt.vlines(0.004,-10,80,linestyles='--',linewidth=3)
# plt.text(0.006,75,r'$U_{th}$',fontsize=32)
# plt.vlines(0.0451935 - 0.008,-10,40,linestyles='--',linewidth=3)
# plt.text(0.047,35,r'$u $',fontsize=32)
# plt.plot(v_z,f_2,label="bulk",linewidth=3)
# plt.plot(v_z,f_1,label="tail",linewidth=3)
# plt.plot(v_z,f_2+f_1,label="bulk+tail",linewidth=5)


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

def maxwellian_1d(v, vthe, udrift):
    return 1/(2*np.pi*vthe**2)*np.exp(-((v-udrift)**2)/(2*vthe**2))

C = np.sqrt(4*np.pi)

tempamp = 1.0
vte = 0.02*tempamp
v0 = 0.02
massratio = 25
tempratio = 50
cs = 0.02/np.sqrt(massratio)
nuee = 0.0001
betap = 0.18
beta0 = 0.3
E = 0.00005

nuE = np.sqrt(9*np.pi/8)*E/cs
ENL = 0.004*50/6/np.pi * C
KN = E/ENL
K = KN/(np.sqrt(KN+1)-1)
vM = vte * (9*np.pi*nuee*vte**2/(betap*K*nuE*cs**2))**(1/6)
tNL = tempratio/E/100

nuee_crit = nuE/tempratio * (beta0/K + betap*K/np.pi/9)

print(nuee_crit)
print(vM)

def fe_1(v):
    exp1 = np.exp(-vM**2/2/vte**2 * (1/6*np.log((v**2+vM**2)**2/(v**4-v**2*vM**2+vM**4)+ 1/np.sqrt(3)*np.arctan(np.sqrt(3)*v**2/(2*vM**2-v**2))) ))
    exp2 = np.exp(-np.pi*vM**2 / (3*np.sqrt(3)*vte**2))
    if v<=0.041:
        return maxwellian_1d(v,vte,0)*(exp1-exp2) + 84.2
    else:
        return maxwellian_1d(v,vte,0)*(exp1-exp2)
    #return 10*(exp1-exp2)

def tau1(t):
    Be = beta0*nuE*cs**2*v0**3/K
    return Be*t

def fe_2(v,t):
    return 1/4/np.pi/gamma(3/5)*(5*tau1(t)**3)**(-0.2)*np.exp(-(v-0.1)**5/25/tau1(t)) / 40 /10

v = np.arange(0.06,0.26,0.001)
f1 = np.array([fe_1(it) for it in v])
f2 = np.array([fe_2(it,1000) for it in v])
fM = np.array([maxwellian_1d(it,vte,0) for it in v])


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