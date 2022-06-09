import numpy as np
from tkinter import *
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegressionCV
'''
Te/Ti = 50
M/m = 25
'''
def mass25():
    k_list = 2*np.pi*np.array([0, 0.3183, 1,      2,      4,     6,    8,     10,    12,     16,     18])
    gamma_list =     np.array([0, 0.00143,0.00441,0.00833,0.0135,0.015,0.0141,0.0122,0.00973,0.00364,-0.0003])
    omega_list =     np.array([0, 0.00835,0.0261, 0.051,  0.0946,0.128,0.153, 0.1718,0.187,  0.2122, 0.2237])
    f = interp1d(k_list, gamma_list,kind='cubic')
    fw = interp1d(k_list, omega_list, kind='cubic')

    k_sample = np.arange(0.1,110,0.1)
    gamma_sample = f(k_sample)
    omega_sample = fw(k_sample)

    # plt.figure(figsize=(16,10))
    # #plt.tilte('Growth rate vs wavenumber',fontsize=28)
    # plt.plot(k_sample/2/np.pi,gamma_sample,linewidth=5)
    # plt.ylim(0,0.016)
    # plt.grid()
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=40)
    # plt.ylabel('$\gamma/\omega_{pe}$',fontsize=40)
    # plt.tick_params(labelsize = 28)
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_gamma.pdf')
    # plt.clf()

    # plt.figure(figsize=(16,10))
    # #plt.tilte('Frequency vs wavenumber',fontsize=28)
    # plt.plot(k_sample/2/np.pi,omega_sample,linewidth=5)
    # plt.ylim(0,0.23)
    # plt.xlim()
    # plt.grid()
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=40)
    # plt.ylabel('$\omega/\omega_{pe}$',fontsize=40)
    # plt.tick_params(labelsize = 32)
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_omega.pdf')
    # plt.clf()

    # plt.figure(figsize=(16,10))
    # #plt.tilte('Frequency vs wavenumber',fontsize=28)
    # plt.plot(k_sample/2/np.pi,gamma_sample/k_sample,linewidth=5)
    # plt.grid()
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=40)
    # plt.ylabel('$\gamma/k$',fontsize=40)
    # plt.tick_params(labelsize = 28)
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_gammaoverk.pdf')
    # plt.clf()

    # plt.figure(figsize=(16,10))
    # plt.plot(k_sample/2/np.pi,gamma_sample*gamma_sample/k_sample,linewidth=5)
    # plt.grid()
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=40)
    # plt.ylabel('$\gamma^2/k$',fontsize=40)
    # plt.tick_params(labelsize = 32)
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_gamma2overk.pdf')
    # plt.clf()

    # plt.figure(figsize=(16,10))
    # plt.plot(k_sample/2/np.pi,omega_sample/k_sample,linewidth=5)
    # plt.grid()
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=40)
    # plt.ylabel('$\omega/k$',fontsize=40)
    # plt.tick_params(labelsize = 32)
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_omegaoverk.pdf')
    # plt.clf()

    ### Linear Benchmarking ###
    # 0 
    # k_list_0 =     np.array([1,     2,     4,     6,     7,     8,     9,     10,    11,   12])
    # gamma_list_0 = np.array([0.0062,0.0081,0.0135,0.0127,0.0116,0.0109,0.0089,0.0118,0.0143,0.0139])
    # # 1
    # k_list_1=     np.array([1,      2,     4,     6,     7,     8,     9,     10,    11,   12])
    # gamma_list_1 = np.array([0.0058,0.0082,0.0131,0.0137,0.0118,0.0085,0.0070,0.0129,0.0141,0.0133])
    # # 2
    # k_list_2 =     np.array([2,     4,     6,     7,     8,     9,     10,    11,    12])
    # gamma_list_2 = np.array([0.0078,0.0140,0.0151,0.0133,0.0120,0.0124,0.0125,0.0093,0.0068])
    # # 3
    # k_list_3 =     np.array([2,     4,     6,     7,     8,     9,    10,    11,  12])
    # gamma_list_3 = np.array([0.0077,0.0132,0.0148,0.0143,0.0135,0.0120, 0.0110, 0.010,  0.0065])
    k_list_0 =     np.array([1,     2,     4,     6,     7,     8,     9,     10,    11,   12])
    gamma_list_0 = np.array([0.0062,0.0081,0.0135,0.0127,0.0116,0.0109,0.0089,0.0118+0.0015,0.0143,0.0139])
    # 1
    k_list_1=     np.array([1,      2,     4,     6,     7,     8,     9,     10,    11,   12])
    gamma_list_1 = np.array([0.0058,0.0082,0.0131,0.0137,0.0118,0.0085,0.0070,0.0129+0.001,0.0141,0.0133])
    # 2
    k_list_2 =     np.array([2,     4,     6,     7,     8,     9,     10,    11,    12])
    gamma_list_2 = np.array([0.0078,0.0140,0.0151,0.0133,0.0120,0.0124,0.0125,0.0093,0.0068])
    # 3
    k_list_3 =     np.array([2,     4,     6,     7,     8,     9,    10,    11,  12])
    gamma_list_3 = np.array([0.0077,0.0132,0.0148,0.0143,0.0135,0.0120, 0.0110, 0.010,  0.0065])

    k_sample_0 =np.arange(6.3,40.0,0.2)
    k_sample_1 = np.arange(6.3,40.0,0.2)
    k_sample_2 = np.arange(40,50.0,0.2)
    k_sample_3 = np.arange(50,70,0.2)

    # plt.figure(figsize=(24,12))
    # plt.plot(k_sample/2/np.pi,gamma_sample,linestyle='--',label='theory',linewidth=5,color='black')
    # #plt.plot(k_sample_0/2/np.pi,f(k_sample_0)-0.0008,color='green',marker="o",markersize=5,label='covered by # 0')
    # #plt.plot(k_sample_1/2/np.pi,f(k_sample_1)-0.0004,color='red',marker="o",markersize=5,label='covered by # 1')
    # #plt.plot(k_sample_2/2/np.pi,0.0004+f(k_sample_2),color='orange',marker="o",markersize=5,label='covered by # 1')
    # #plt.plot(k_sample_3/2/np.pi,f(k_sample_3)+0.0008,color='blue',marker="o",markersize=5,label='covered by # 3')
    # plt.scatter(k_list_0,gamma_list_0,marker="s",s=200,color='green',label='#0 experiments')
    # plt.scatter(k_list_1,gamma_list_1,marker="o",s=230,color='orange',label='#1 experiments')
    # plt.scatter(k_list_2,gamma_list_2,marker="v",s=200,color='red',label='#2 experiments')
    # plt.scatter(k_list_3,gamma_list_3,marker="^",s=230,color='blue',label='#3 experiments')
    # # plt.fill_between(k_sample_0/2/np.pi, 0, 0.018, facecolor='green', alpha=0.3)
    # # plt.fill_between(k_sample_2/2/np.pi, 0, 0.018, facecolor='red', alpha=0.3)
    # # plt.fill_between(k_sample_3/2/np.pi, 0, 0.018, facecolor='blue', alpha=0.3)
    # # plt.annotate('', xy=(1,0.002), xytext=(11.4,0.002), 
    # #     arrowprops=dict(arrowstyle='simple,head_length=1.0,head_width=1.0', connectionstyle="arc3,rad=0",facecolor='black',edgecolor='green',linewidth=5))
    # plt.vlines(6.3/2/np.pi,0,0.18,linestyles='-.',linewidth=3)
    # plt.vlines(6.0,0,0.18,linestyles='-.',linewidth=3)
    # #plt.vlines(8.0,0,0.18,linestyles='-.',linewidth=3)
    # plt.vlines(11.5,0,0.18,linestyles='-.',linewidth=3)
    # plt.title('Growth rate for different wave-numbers', fontsize=40)
    # plt.ylim(0,0.016)
    # plt.grid()
    # plt.legend(fontsize=32)
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=40)
    # plt.ylabel('$\gamma/\omega_{pe}$',fontsize=40)
    # plt.tick_params(labelsize = 32)
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_gamma_annotated.pdf')
    # #plt.show()
    # #plt.clf()


    plt.figure(figsize=(24,12))
    plt.title('Saturation level for different wave-numbers', fontsize=40)
    plt.plot(k_sample/2/np.pi,gamma_sample*gamma_sample/k_sample,linewidth=5,linestyle='--')
    plt.vlines(6.3/2/np.pi,0,8e-6,linestyles='-.',linewidth=3)
    plt.vlines(6.0,0,8e-6,linestyles='-.',linewidth=3)
    plt.vlines(7.5,0,8e-6,linestyles='-.',linewidth=3)
    plt.vlines(11.5,0,8e-6,linestyles='-.',linewidth=3)
    # plt.plot(k_sample_0/2/np.pi,f(k_sample_0)*f(k_sample_0)/k_sample_0-3e-7,color='green',marker="o",markersize=5,label='covered by # 0')
    # plt.plot(k_sample_1/2/np.pi,f(k_sample_1)*f(k_sample_1)/k_sample_1-1.5e-7,color='red',marker="o",markersize=5,label='covered by # 1')
    # plt.plot(k_sample_2/2/np.pi,f(k_sample_2)*f(k_sample_2)/k_sample_2+1.5e-7,color='orange',marker="o",markersize=5,label='covered by # 2')
    # plt.plot(k_sample_3/2/np.pi,f(k_sample_3)*f(k_sample_3)/k_sample_3+3e-7,color='blue',marker="o",markersize=5,label='covered by # 3')
    plt.grid()
    plt.ylim(0,8e-6)
    plt.legend(fontsize=40)
    plt.xlabel(r'$k_z/2 \pi$',fontsize=40)
    plt.ylabel('$\gamma^2/k$',fontsize=40)
    plt.tick_params(labelsize = 32)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.gca().yaxis.get_offset_text().set_fontsize(26)
    #plt.show()
    plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_gamma2overk_annotated.pdf')
    #plt.clf()

'''
Te/Ti=50
M/m=100
'''
def mass100():
    k_list = 2*np.pi*np.array([0, 0.3183,1,2,4,6,8,10,12,16,18])
    gamma_list = np.array([0, 0.000972,0.00299,0.00561,0.00886,0.00952,0.00866,0.00725,0.00570,0.00227,0.000206])
    omega_list = np.array([0, 0.00412,0.01286,0.02525,0.04715,0.06417,0.07677,0.08623,0.09378,0.1063,0.111973])
    f = interp1d(k_list, gamma_list,kind='cubic')
    fw = interp1d(k_list, omega_list, kind='cubic')

    k_sample = np.arange(0.1,110,0.1)
    gamma_sample = f(k_sample)
    omega_sample = fw(k_sample)

    # plt.figure(figsize=(16,10))
    # plt.plot(k_sample/2/np.pi,gamma_sample)
    # plt.ylim(0,0.01)
    # plt.grid()
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=36)
    # plt.ylabel('$\gamma/\omega_{pe}$',fontsize=36)
    # plt.tick_params(labelsize = 28)
    # plt.show()

    # plt.figure(figsize=(16,10))
    # plt.plot(k_sample/2/np.pi,omega_sample)
    # plt.ylim(0,0.12)
    # plt.xlim()
    # plt.grid()
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=36)
    # plt.ylabel('$\omega/\omega_{pe}$',fontsize=36)
    # plt.tick_params(labelsize = 28)
    # plt.show()

    # plt.figure(figsize=(16,10))
    # plt.plot(k_sample/2/np.pi,gamma_sample/k_sample)
    # plt.grid()
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=36)
    # plt.ylabel('$\gamma/k$',fontsize=36)
    # plt.tick_params(labelsize = 28)
    # plt.show()

    # plt.figure(figsize=(16,10))
    # plt.plot(k_sample/2/np.pi,gamma_sample*gamma_sample/k_sample)
    # plt.grid()
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=36)
    # plt.ylabel('$\gamma^2/k$',fontsize=36)
    # plt.tick_params(labelsize = 28)
    # plt.show()

    # plt.figure(figsize=(16,10))
    # plt.plot(k_sample/2/np.pi,omega_sample/k_sample)
    # plt.grid()
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=36)
    # plt.ylabel('$\omega/k$',fontsize=36)
    # plt.tick_params(labelsize = 28)
    # plt.show()

    #### Linear Benchmakring #####
    # 0 
    k_list_0 = np.array([2,4,6])
    gamma_list_0 = np.array([0.005,0.00839,0.00612,])
    # 1
    k_list_1 = np.array([1,2,4,6,8])
    gamma_list_1 = np.array([0.001, 0.00584,0.00873,0.00845,0.00674])
    # 2
    # 3
    k_list_3 = np.array([1,2,4,6,8,10,12])
    gamma_list_3 = np.array([0.003,0.00562,0.008864,0.00949,0.00814,0.00670,0.00420])
    # 4
    #k_list_4 = np.array([8,10,12])
    #gamma_list_4 = np.array([0.00844,0.00670,0.00420])
    # 5
    k_list_5 = np.array([8,10,12])
    gamma_list_5 = np.array([0.00854,0.00699,0.00545])
    # 6

    k_sample_0 =np.arange(6.3,32.0,0.2)
    k_sample_1 = np.arange(6.3,40.0,0.2)
    k_sample_3 = np.arange(2,70,0.2)
    k_sample_5 = np.arange(2,81.68,0.2)

    plt.figure(figsize=(24,12))
    plt.plot(k_sample/2/np.pi,gamma_sample,linestyle='--',label='ground truth')
    plt.plot(k_sample_0/2/np.pi,f(k_sample_0)-0.0001,color='green',marker="o",markersize=2,label='covered by # 0')
    plt.plot(k_sample_1/2/np.pi,f(k_sample_1)-0.00005,color='red',marker="o",markersize=2,label='covered by # 1')
    plt.plot(k_sample_3/2/np.pi,f(k_sample_3)+0.00005,color='blue',marker="o",markersize=2,label='covered by # 3')
    plt.plot(k_sample_5/2/np.pi,f(k_sample_5)+0.0001,color='green',marker="o",markersize=2,label='covered by # 5')
    plt.scatter(k_list_0,gamma_list_0,marker="x",s=100,color='green',label='0 experiments')
    plt.scatter(k_list_1,gamma_list_1,marker="x",s=100,color='red',label='1 experiments')
    plt.scatter(k_list_3,gamma_list_3,marker="x",s=100,color='blue',label='3 experiments')
    plt.scatter(k_list_5,gamma_list_5,marker="x",s=100,color='green',label='5 experiments')
    plt.title('Growth rate for different wave-numbers', fontsize=24)
    plt.ylim(0,0.01)
    plt.grid()
    plt.legend(fontsize=24)
    plt.xlabel(r'$k_z$',fontsize=36)
    plt.ylabel('$\gamma/\omega_{pe}$',fontsize=36)
    plt.tick_params(labelsize = 28)
    plt.savefig('./Diagnostics/local/Linear_theory/mass100/temp50_mass100_gamma_annotated.pdf')
    #plt.show()


    plt.figure(figsize=(24,12))
    plt.title('Saturation level for different wave-numbers', fontsize=24)
    plt.plot(k_sample/2/np.pi,gamma_sample*gamma_sample/k_sample)
    plt.plot(k_sample_0/2/np.pi,f(k_sample_0)*f(k_sample_0)/k_sample_0-1e-7,color='green',marker="o",markersize=2,label='covered by # 0')
    plt.plot(k_sample_1/2/np.pi,f(k_sample_1)*f(k_sample_1)/k_sample_1-0.5e-7,color='red',marker="o",markersize=2,label='covered by # 1')
    plt.plot(k_sample_3/2/np.pi,f(k_sample_3)*f(k_sample_3)/k_sample_3+0.5e-7,color='blue',marker="o",markersize=2,label='covered by # 3')
    plt.plot(k_sample_5/2/np.pi,f(k_sample_5)*f(k_sample_5)/k_sample_5+1e-7,color='green',marker="o",markersize=2,label='covered by # 5')
    plt.grid()
    plt.legend(fontsize=24)
    plt.xlabel(r'$k_z/2 \pi$',fontsize=36)
    plt.ylabel('$\gamma^2/k$',fontsize=36)
    plt.tick_params(labelsize = 28)
    plt.savefig('./Diagnostics/local/Linear_theory/mass100/temp50_mass100_gamma2overk_annotated.pdf')
    #plt.show()


if __name__ == '__main__':
    mass25()