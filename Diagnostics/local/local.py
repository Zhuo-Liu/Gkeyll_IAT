#.Make plots from Gkyl data.
# Z.Liu 5/2021
# Auxiliary for crash data files

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

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
fieldEnergy_z = np.loadtxt('./Diagnostics/local/data/fieldEnergy.txt')
fieldEnergy_y = np.loadtxt('./Diagnostics/local/data/fieldEnergy_y.txt')
fieldEnergy = fieldEnergy_z + fieldEnergy_z
time_fieldEnergy = np.loadtxt('./Diagnostics/local/data/fieldEnergy_y_time.txt')

plt.figure(figsize=(16, 12), dpi=80)
plt.plot(time_fieldEnergy,fieldEnergy)
plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
plt.ylabel(r'$\int dydz |\delta E_z|^2 + |\delta E_y|^2$',fontsize=36)
plt.yscale('log')
plt.xlim(0,3200)
plt.tick_params(labelsize = 28)
plt.show()

####### Ion Temp Plot #######
Iontemp = np.loadtxt('./Diagnostics/local/data/Iontemp.txt')
time_Iontemp = np.loadtxt('./Diagnostics/local/data/Iontemp_times.txt')

plt.figure(figsize=(16, 12), dpi=80)
plt.plot(time_Iontemp,Iontemp/Iontemp[0])
plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
plt.ylabel(r'$T_i/T_{i0}$',fontsize=36)
plt.yscale('log')
plt.ylim(0,10)
plt.tick_params(labelsize = 28)
plt.show()

current = np.loadtxt('./Diagnostics/local/data/current.txt')
time_current = np.loadtxt('./Diagnostics/local/data/time_nu.txt')

plt.figure(figsize=(16, 12), dpi=80)
plt.plot(time_current[time_current!=0],-current[current!=0])
plt.xlabel(r'$t [\omega_{pe}^-1]$',fontsize=36)
plt.ylabel(r'$|<J_z>|$',fontsize=36)
plt.tick_params(labelsize = 28)
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

f_e = np.loadtxt('./Diagnostics/local/data/elc_dist_1615.txt')
v_z = np.loadtxt('./Diagnostics/local/data/elc_dist_vz_1615.txt')
f_e_0 = np.loadtxt('./Diagnostics/local/data/elc_dist_0.txt')

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
    A = 22.3404
    B = 0.000657877
    C = 0.0451935 - 0.008

    return A*np.exp(-(v-C)**2/B)

def maxwellian_2(v):
    A = 67.31563834802252
    B = 0.0006571545907741459
    C = 0.004277912978811665

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
plt.plot(v_z,f_e_0,label="F_e (t=3200)",linewidth=1)
plt.xlabel(r'$V_z$',fontsize=28)
plt.ylabel('$F_e(V_z)$',fontsize=28)
plt.vlines(0.004,-10,80,linestyles='--',linewidth=3)
plt.text(0.006,75,r'$U_{th}$',fontsize=32)
plt.vlines(0.0451935 - 0.008,-10,40,linestyles='--',linewidth=3)
plt.text(0.047,35,r'$u $',fontsize=32)
plt.plot(v_z,f_2,label="bulk",linewidth=3)
plt.plot(v_z,f_1,label="tail",linewidth=3)
plt.plot(v_z,f_2+f_1,label="bulk+tail",linewidth=5)


# interp = 0
# f_int1 = 0
# f_int2 = 0
# for i in range(359):
#     deltat = v_z[i+1]-v_z[i]
#     f = 0.5*(f_e[i]-f_max[i]+f_e[i+1]-f_max[i+1])
#     interp += deltat*f
#     f_int1 += deltat*(f_e_0[i]+f_e_0[i+1])*0.5
#     f_int2 += deltat*(f_max_0[i]+f_max_0[i+1])*0.5
# print(interp)
# print(f_int1)
# print(f_int2)
#plt.xlim(-0.07,0.15)
plt.legend(fontsize=36)
plt.tick_params(labelsize = 28)
plt.grid()
plt.show()