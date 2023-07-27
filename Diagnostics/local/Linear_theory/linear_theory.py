from matplotlib.backend_tools import AxisScaleBase
from matplotlib.lines import lineStyles
import numpy as np
from tkinter import *
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegressionCV

def linear_theory():
    k_list_25 = 2*np.pi*np.array([0, 0.3183, 1,      2,      4,     6,    8,     10,    12,     16,     18])
    gamma_list_25 =     np.array([0, 0.00143,0.00441,0.00833,0.0135,0.015,0.0141,0.0122,0.00973,0.00364,-0.0003])
    omega_list_25 =     np.array([0, 0.00835,0.0261, 0.051,  0.0946,0.128,0.153, 0.1718,0.187,  0.2122, 0.2237])
    f_25 = interp1d(k_list_25, gamma_list_25, kind='cubic')
    fw_25 = interp1d(k_list_25, omega_list_25, kind='cubic')
    k_sample_25 = np.arange(0.1,113,0.1)
    gamma_sample_25 = f_25(k_sample_25)
    omega_sample_25 = fw_25(k_sample_25)

    k_list_100 = 2*np.pi*np.array([0, 0.3183,1,2,4,6,8,10,12,16,18,19])
    gamma_list_100 = np.array([0, 0.000972,0.00299,0.00561,0.00886,0.00952,0.00866,0.00725,0.00570,0.00227,0.000206,-0.00094])
    omega_list_100 = np.array([0, 0.00412,0.01286,0.02525,0.04715,0.06417,0.07677,0.08623,0.09378,0.1063,0.111973,0.114734])
    f_100 = interp1d(k_list_100, gamma_list_100,kind='cubic')
    fw_100 = interp1d(k_list_100, omega_list_100, kind='cubic')
    k_sample_100 = np.arange(0.1,114,0.1)
    gamma_sample_100 = f_100(k_sample_100)
    omega_sample_100 = fw_100(k_sample_100)

    k_list_400 = 2*np.pi*np.array([0, 0.3183,1,2,4,6,8,10,12,16,18,19])
    gamma_list_400 = np.array([0, 0.000549,0.00169,0.00316,0.00496,0.00527,0.00474,0.00392,0.00306,0.00125,0.00019,-0.00039])
    omega_list_400 = np.array([0, 0.002037,0.00637,0.01252,0.02350,0.03210,0.03846,0.04320,0.04697,0.05320,0.05602,0.05740])
    f_400 = interp1d(k_list_400, gamma_list_400,kind='cubic')
    fw_400 = interp1d(k_list_400, omega_list_400, kind='cubic')
    k_sample_400 = np.arange(0.1,114,0.1)
    gamma_sample_400 = f_400(k_sample_400)
    omega_sample_400 = fw_400(k_sample_400)

    k_sample_25 = k_sample_25 / 50
    k_sample_100 = k_sample_100 / 50
    k_sample_400 = k_sample_400 / 50

    axpos   = [0.13, 0.16, 0.8, 0.8]

    #plt.tilte('Growth rate vs wavenumber',fontsize=28)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes(axpos)
    ax.plot(k_sample_25/2/np.pi,gamma_sample_25,linewidth=3,label='$m_i/m_e = 25$')
    ax.plot(k_sample_100/2/np.pi,gamma_sample_100,linewidth=3,label='$m_i/m_e = 100$')
    ax.plot(k_sample_400/2/np.pi,gamma_sample_400,linewidth=3,label='$m_i/m_e = 400$')
    ax.set_xlim(0.0,)
    ax.set_ylim(0,0.016)
    ax.grid()
    ax.set_xlabel(r'$k_z/2 \pi \quad [\lambda_{De}^{-1}]$',fontsize=26)
    ax.set_ylabel('$\gamma \quad  [\omega_{pe}]$',fontsize=26)
    ax.tick_params(labelsize = 18)
    ax.yaxis.offsetText.set_fontsize(16)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(2,0))
    ax.legend(fontsize=24)
    plt.savefig('./Linear_theory/temp50_gamma.jpg')
    plt.clf()

    #plt.tilte('Frequency vs wavenumber',fontsize=28)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes(axpos)
    ax.plot(k_sample_25/2/np.pi,omega_sample_25,linewidth=3,label='$m_i/m_e = 25$')
    ax.plot(k_sample_100/2/np.pi,omega_sample_100,linewidth=3,label='$m_i/m_e = 100$')
    ax.plot(k_sample_400/2/np.pi,omega_sample_400,linewidth=3,label='$m_i/m_e = 400$')
    ax.set_ylim(0,0.23)
    ax.set_xlim()
    ax.grid()
    ax.set_xlabel(r'$k_z/2 \pi \quad [\lambda_{De}^{-1}]$',fontsize=26)
    ax.set_ylabel('$\omega \quad  [\omega_{pe}^{-1}]$',fontsize=26)
    ax.tick_params(labelsize = 18)
    ax.legend(fontsize=24)
    plt.savefig('./Linear_theory/temp50_omega.jpg')
    plt.clf()

    #plt.tilte('Frequency vs wavenumber',fontsize=28)
    #axpos = [0.13, 0.16, 0.8, 0.8]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes(axpos)
    ax.plot(k_sample_25/2/np.pi,gamma_sample_25/k_sample_25,linewidth=3,label='$m_i/m_e = 25$')
    ax.plot(k_sample_100/2/np.pi,gamma_sample_100/k_sample_100,linewidth=3,label='$m_i/m_e = 100$')
    ax.plot(k_sample_400/2/np.pi,gamma_sample_400/k_sample_400,linewidth=3,label='$m_i/m_e = 400$')
    ax.grid()
    ax.set_xlim(0,)
    ax.set_xlabel(r'$k_z/2 \pi \quad  [\lambda_{De}^{-1}]$',fontsize=26)
    ax.set_ylabel('$\gamma/k \quad [v_{Te0}]$',fontsize=26)
    ax.tick_params(labelsize = 18)
    ax.yaxis.offsetText.set_fontsize(16)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(2,0))
    ax.legend(fontsize=24)
    plt.savefig('./Linear_theory/temp50_gammaoverk.jpg')
    plt.clf()

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.14, 0.16, 0.8, 0.8])    
    ax.plot(k_sample_25/2/np.pi,omega_sample_25/k_sample_25,linewidth=3,label='$m_i/m_e = 25$')
    ax.plot(k_sample_100/2/np.pi,omega_sample_100/k_sample_100,linewidth=3,label='$m_i/m_e = 100$')
    ax.plot(k_sample_400/2/np.pi,omega_sample_400/k_sample_400,linewidth=3,label='$m_i/m_e = 400$')
    ax.grid()
    ax.set_xlim(0,)
    ax.set_xlabel(r'$k_z/2 \pi \quad [\lambda_{De}^{-1}]$',fontsize=26)
    ax.set_ylabel('$\omega/k \quad [v_{Te0}]$',fontsize=26)
    ax.tick_params(labelsize = 18)
    ax.yaxis.offsetText.set_fontsize(16)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(2,0))
    ax.legend(fontsize=24)
    plt.savefig('./Linear_theory/temp50_omegaoverk.jpg')
    plt.clf()

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

    k_sample = np.arange(0.1,113,0.1)
    gamma_sample = f(k_sample)
    omega_sample = fw(k_sample)

    axpos   = [0.13, 0.16, 0.8, 0.8]

    #plt.tilte('Growth rate vs wavenumber',fontsize=28)
    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_axes(axpos)
    # ax.plot(k_sample/2/np.pi,gamma_sample,linewidth=3)
    # ax.set_ylim(0,0.016)
    # ax.grid()
    # ax.set_xlabel(r'$k_z/2 \pi \quad (d_e^{-1})$',fontsize=26)
    # ax.set_ylabel('$\gamma \quad  (\omega_{pe})$',fontsize=26)
    # ax.tick_params(labelsize = 18)
    # ax.yaxis.offsetText.set_fontsize(16)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(2,0))
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_gamma.pdf')
    # plt.clf()

    # #plt.tilte('Frequency vs wavenumber',fontsize=28)
    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_axes(axpos)
    # ax.plot(k_sample/2/np.pi,omega_sample,linewidth=3)
    # ax.set_ylim(0,0.23)
    # ax.set_xlim()
    # ax.grid()
    # ax.set_xlabel(r'$k_z/2 \pi \quad (d_e^{-1})$',fontsize=26)
    # ax.set_ylabel('$\omega \quad   (\omega_{pe}^{-1})$',fontsize=26)
    # ax.tick_params(labelsize = 18)
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_omega.pdf')
    # plt.clf()

    # #plt.tilte('Frequency vs wavenumber',fontsize=28)
    # #axpos = [0.13, 0.16, 0.8, 0.8]
    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_axes(axpos)
    # ax.plot(k_sample/2/np.pi,gamma_sample/k_sample,linewidth=3)
    # ax.grid()
    # ax.set_xlim(0,)
    # ax.set_xlabel(r'$k_z/2 \pi \quad  (d_e^{-1})$',fontsize=26)
    # ax.set_ylabel('$\gamma/k \quad (d_e \omega_{pe})$',fontsize=26)
    # ax.tick_params(labelsize = 18)
    # ax.yaxis.offsetText.set_fontsize(16)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(2,0))
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_gammaoverk.pdf')
    # plt.clf()

    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_axes(axpos)    
    # ax.plot(k_sample/2/np.pi,omega_sample/k_sample,linewidth=3)
    # ax.grid()
    # ax.set_xlim(0,)
    # ax.set_xlabel(r'$k_z/2 \pi (d_e^{-1})$',fontsize=26)
    # ax.set_ylabel('$\omega/k \quad (\omega_{pe} d_e)$',fontsize=26)
    # ax.tick_params(labelsize = 18)
    # ax.yaxis.offsetText.set_fontsize(16)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(2,0))
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_omegaoverk.pdf')
    # plt.clf()

    # plt.figure(figsize=(16,12))
    # plt.plot(k_sample/2/np.pi,gamma_sample*gamma_sample/k_sample,linewidth=3)
    # plt.grid()
    # plt.xlabel(r'$k_z/2 \pi (d_e^{-1})$',fontsize=40)
    # plt.ylabel('$\gamma^2/k$',fontsize=40)
    # plt.tick_params(labelsize = 32)
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_gamma2overk.pdf')
    # plt.clf()

    ###########################
    ### Linear Benchmarking ###
    ###########################
    # 0 
    k_list_0 =     np.array([1,     2,     3,      4,     6,     7,     8,     9])
    gamma_list_0 = np.array([0.0062,0.0081,0.0114, 0.0135,0.0127,0.0116,0.0109,0.0089])
    # 1
    k_list_1=     np.array([1,      2,     3,      4,     6,     7,     8,     9])
    gamma_list_1 = np.array([0.0058,0.0082,0.0113,0.0131,0.0137,0.0118,0.0085,0.0070])
    # 2
    k_list_2 =     np.array([2,     3,      4,     6,     7,     8,     9,     10,    11,    12])
    gamma_list_2 = np.array([0.0078,0.0114,0.0140,0.0151,0.0133,0.0120,0.0124,0.0125,0.0093,0.0068])
    # 3
    k_list_3 =     np.array([2,     3,      4,     6,     7,     8,     9,    10,    11,     12])
    gamma_list_3 = np.array([0.0077,0.0114,0.0132,0.0148,0.0143,0.0135,0.0120, 0.0110, 0.010,  0.0065])

    plt.figure(figsize=(24,12))
    plt.plot(k_sample/2/np.pi/50,gamma_sample,linestyle='--',label='theory',linewidth=5,color='black')
    plt.scatter(k_list_0/50,gamma_list_0,marker="s",s=200,color='green',label='#1')
    plt.scatter(k_list_1/50,gamma_list_1,marker="o",s=230,color='orange',label='#2')
    plt.scatter(k_list_2/50,gamma_list_2,marker="v",s=200,color='red',label='#3')
    plt.scatter(k_list_3/50,gamma_list_3,marker="^",s=230,color='blue',label='#4')
    # plt.fill_between(k_sample_0/2/np.pi, 0, 0.018, facecolor='green', alpha=0.3)
    # plt.fill_between(k_sample_2/2/np.pi, 0, 0.018, facecolor='red', alpha=0.3)
    # plt.fill_between(k_sample_3/2/np.pi, 0, 0.018, facecolor='blue', alpha=0.3)
    # plt.annotate('', xy=(1,0.002), xytext=(11.4,0.002), 
    #     arrowprops=dict(arrowstyle='simple,head_length=1.0,head_width=1.0', connectionstyle="arc3,rad=0",facecolor='black',edgecolor='green',linewidth=5))
    plt.vlines(6.3/2/np.pi/50,0,0.18,linestyles='-.',linewidth=3)
    plt.vlines(7.0/50,0,0.18,linestyles='-.',linewidth=3)
    #plt.vlines(8.0,0,0.18,linestyles='-.',linewidth=3)
    plt.vlines(11.5/50,0,0.18,linestyles='-.',linewidth=3)
    #plt.title('Measured growth rates for different wave modes', fontsize=40)
    plt.ylim(0,0.016)
    plt.grid()
    plt.legend(fontsize=40)
    plt.xlabel(r'$k_z/2 \pi \quad [\lambda_{De}^{-1}]$',fontsize=40)
    plt.ylabel(r'$\gamma \quad [\omega_{pe}]$',fontsize=46)
    plt.tick_params(labelsize = 32)
    plt.savefig('./Linear_theory/mass25/temp50_mass25_gamma_annotated.jpg')
    #plt.show()
    #plt.clf()


    # plt.figure(figsize=(24,12))
    # plt.title('Saturation level for different wave-numbers', fontsize=40)
    # plt.plot(k_sample/2/np.pi,gamma_sample*gamma_sample/k_sample,linewidth=5,linestyle='--')
    # plt.vlines(6.3/2/np.pi,0,8e-6,linestyles='-.',linewidth=3)
    # plt.vlines(6.0,0,8e-6,linestyles='-.',linewidth=3)
    # plt.vlines(7.5,0,8e-6,linestyles='-.',linewidth=3)
    # plt.vlines(11.5,0,8e-6,linestyles='-.',linewidth=3)
    # # plt.plot(k_sample_0/2/np.pi,f(k_sample_0)*f(k_sample_0)/k_sample_0-3e-7,color='green',marker="o",markersize=5,label='covered by # 0')
    # # plt.plot(k_sample_1/2/np.pi,f(k_sample_1)*f(k_sample_1)/k_sample_1-1.5e-7,color='red',marker="o",markersize=5,label='covered by # 1')
    # # plt.plot(k_sample_2/2/np.pi,f(k_sample_2)*f(k_sample_2)/k_sample_2+1.5e-7,color='orange',marker="o",markersize=5,label='covered by # 2')
    # # plt.plot(k_sample_3/2/np.pi,f(k_sample_3)*f(k_sample_3)/k_sample_3+3e-7,color='blue',marker="o",markersize=5,label='covered by # 3')
    # plt.grid()
    # plt.ylim(0,8e-6)
    # plt.legend(fontsize=40)
    # plt.xlabel(r'$k_z/2 \pi$',fontsize=40)
    # plt.ylabel('$\gamma^2/k$',fontsize=40)
    # plt.tick_params(labelsize = 32)
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # plt.gca().yaxis.get_offset_text().set_fontsize(26)
    # #plt.show()
    # plt.savefig('./Diagnostics/local/Linear_theory/mass25/temp50_mass25_gamma2overk_annotated.pdf')
    # #plt.clf()

'''
Te/Ti=50
M/m=100
'''
def mass100():
    k_list = 2*np.pi*np.array([0, 0.3183,1,2,4,6,8,10,12,16,18,19])
    gamma_list = np.array([0, 0.000972,0.00299,0.00561,0.00886,0.00952,0.00866,0.00725,0.00570,0.00227,0.000206,-0.00094])
    omega_list = np.array([0, 0.00412,0.01286,0.02525,0.04715,0.06417,0.07677,0.08623,0.09378,0.1063,0.111973,0.114734])
    f = interp1d(k_list, gamma_list,kind='cubic')
    fw = interp1d(k_list, omega_list, kind='cubic')

    k_sample = np.arange(0.1,114,0.1)
    gamma_sample = f(k_sample)
    omega_sample = fw(k_sample)

    axpos   = [0.13, 0.16, 0.8, 0.8]

    # #plt.tilte('Growth rate vs wavenumber',fontsize=28)
    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_axes(axpos)
    # ax.plot(k_sample/2/np.pi,gamma_sample,linewidth=3)
    # ax.set_ylim(0,0.01)
    # ax.grid()
    # ax.set_xlabel(r'$k_z/2 \pi \quad (d_e^{-1})$',fontsize=26)
    # ax.set_ylabel('$\gamma \quad  (\omega_{pe})$',fontsize=26)
    # ax.tick_params(labelsize = 18)
    # ax.yaxis.offsetText.set_fontsize(16)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(2,0))
    # plt.savefig('./Diagnostics/local/Linear_theory/mass100/temp50_mass100_gamma.pdf')
    # plt.clf()

    # #plt.tilte('Frequency vs wavenumber',fontsize=28)
    # axpos1 = [0.15, 0.16, 0.82, 0.8]
    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_axes(axpos1)
    # ax.plot(k_sample/2/np.pi,omega_sample,linewidth=3)
    # ax.set_ylim(0,0.12)
    # ax.set_xlim()
    # ax.grid()
    # ax.set_xlabel(r'$k_z/2 \pi \quad (d_e^{-1})$',fontsize=26)
    # ax.set_ylabel('$\omega \quad   (\omega_{pe}^{-1})$',fontsize=26)
    # ax.tick_params(labelsize = 18)
    # plt.savefig('./Diagnostics/local/Linear_theory/mass100/temp50_mass100_omega.pdf')
    # plt.clf()

    # #plt.tilte('Frequency vs wavenumber',fontsize=28)
    # #axpos = [0.13, 0.16, 0.8, 0.8]
    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_axes(axpos)
    # ax.plot(k_sample/2/np.pi,gamma_sample/k_sample,linewidth=3)
    # ax.grid()
    # ax.set_xlim(0,)
    # ax.set_xlabel(r'$k_z/2 \pi \quad  (d_e^{-1})$',fontsize=26)
    # ax.set_ylabel('$\gamma/k \quad (d_e \omega_{pe})$',fontsize=26)
    # ax.tick_params(labelsize = 18)
    # ax.yaxis.offsetText.set_fontsize(16)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(2,0))
    # plt.savefig('./Diagnostics/local/Linear_theory/mass100/temp50_mass100_gammaoverk.pdf')
    # plt.clf()

    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_axes(axpos)    
    # ax.plot(k_sample/2/np.pi,omega_sample/k_sample,linewidth=3)
    # ax.grid()
    # ax.set_xlim(0,)
    # ax.set_xlabel(r'$k_z/2 \pi (d_e^{-1})$',fontsize=26)
    # ax.set_ylabel('$\omega/k \quad (\omega_{pe} d_e)$',fontsize=26)
    # ax.tick_params(labelsize = 18)
    # ax.yaxis.offsetText.set_fontsize(16)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(2,0))
    # plt.savefig('./Diagnostics/local/Linear_theory/mass100/temp50_mass100_omegaoverk.pdf')
    # plt.clf()

    ###########################
    ### Linear Benchmarking ###
    ###########################
    # 0 
    k_list_0 =     np.array([2,     3,     4,     5,     6,     7])
    gamma_list_0 = np.array([0.0049,0.0079,0.0084,0.0072,0.0058,0.0056])
    # 1
    k_list_1=     np.array([2,     3,      4,     5,     6,     7,     8   ])
    gamma_list_1 = np.array([0.0058,0.0080,0.0087,0.0090,0.0086,0.0077,0.0069])
    # 2
    k_list_2 =     np.array([2,     3,     4,     5,     6,     7,])
    gamma_list_2 = np.array([0.0051,0.0079,0.0077,0.0067,0.0041,0.0055])
    # 3
    k_list_3 =     np.array([2,     3,     4,     5,     6,     7,     8,     10,  12])
    gamma_list_3 = np.array([0.0056,0.0080,0.0089,0.0094,0.0095,0.0085,0.0084, 0.0067,  0.0042])

    # k_sample_0 =np.arange(6.3,40.0,0.2)
    # k_sample_1 = np.arange(6.3,40.0,0.2)
    # k_sample_2 = np.arange(40,50.0,0.2)
    # k_sample_3 = np.arange(50,70,0.2)

    plt.figure(figsize=(24,12))
    plt.plot(k_sample/2/np.pi,gamma_sample,linestyle='--',label='theory',linewidth=5,color='black')
    #plt.plot(k_sample_0/2/np.pi,f(k_sample_0)-0.0008,color='green',marker="o",markersize=5,label='covered by # 0')
    #plt.plot(k_sample_1/2/np.pi,f(k_sample_1)-0.0004,color='red',marker="o",markersize=5,label='covered by # 1')
    #plt.plot(k_sample_2/2/np.pi,0.0004+f(k_sample_2),color='orange',marker="o",markersize=5,label='covered by # 1')
    #plt.plot(k_sample_3/2/np.pi,f(k_sample_3)+0.0008,color='blue',marker="o",markersize=5,label='covered by # 3')
    plt.scatter(k_list_0,gamma_list_0,marker="s",s=200,color='green',label='#1')
    plt.scatter(k_list_1,gamma_list_1,marker="o",s=230,color='orange',label='#2')
    plt.scatter(k_list_2,gamma_list_2,marker="v",s=200,color='red',label='#3')
    plt.scatter(k_list_3,gamma_list_3,marker="^",s=230,color='blue',label='#4')
    # plt.fill_between(k_sample_0/2/np.pi, 0, 0.018, facecolor='green', alpha=0.3)
    # plt.fill_between(k_sample_2/2/np.pi, 0, 0.018, facecolor='red', alpha=0.3)
    # plt.fill_between(k_sample_3/2/np.pi, 0, 0.018, facecolor='blue', alpha=0.3)
    # plt.annotate('', xy=(1,0.002), xytext=(11.4,0.002), 
    #     arrowprops=dict(arrowstyle='simple,head_length=1.0,head_width=1.0', connectionstyle="arc3,rad=0",facecolor='black',edgecolor='green',linewidth=5))
    plt.vlines(6.3/2/np.pi,0,0.18,linestyles='-.',linewidth=3)
    plt.vlines(4.5,0,0.18,linestyles='-.',linewidth=3)
    plt.vlines(7.0,0,0.18,linestyles='-.',linewidth=3)
    plt.vlines(11.5,0,0.18,linestyles='-.',linewidth=3)
    plt.title('Growth rate for different wave-numbers, $m_i/m_e=100$', fontsize=40)
    plt.ylim(0,0.01)
    plt.grid()
    plt.legend(fontsize=32)
    plt.xlabel(r'$k_z/2 \pi \quad (d_e^{-1})$',fontsize=40)
    plt.ylabel('$\gamma \quad (\omega_{pe})$',fontsize=40)
    plt.tick_params(labelsize = 40)
    plt.savefig('./Diagnostics/local/Linear_theory/mass100/temp50_mass100_gamma_annotated.pdf')
    #plt.show()
    #plt.clf()


def eaw():
    # k_list = 2*np.pi*np.array([0.6369, 0.9554, 1.2739, 1.5924,1.9108,2.2293,2.5478, 2.86624, 3.1847,3.5032,3.82167,4.1403])
    # gamma_list =     np.array([0.047, 0.0668,0.0817,0.09110,0.09455,0.092105,0.0841,0.0709,0.0531,0.03125,0.006,-0.023])
    # omega_list =     np.array([0.08, 0.123,0.168, 0.215,  0.2637,0.31288,0.362, 0.4105,0.4581,0.50445,0.5495, 0.5931])
    # f = interp1d(k_list, gamma_list, kind='cubic')
    # fw = interp1d(k_list, omega_list, kind='cubic')
    # k_sample = np.arange(0.65,4,0.1)*2*np.pi
    # gamma_sample = f(k_sample)
    # omega_sample = fw(k_sample)

    k_list = 2*np.pi*np.arange(9,29,2)*0.1
    omega_list = np.array([0.0876485, 0.108983, 0.131577, 0.155846, 0.182402, 0.212296, 0.247897, 0.302664, 0.341438, 0.368753])
    fw = interp1d(k_list, omega_list, kind='cubic')
    k_sample = np.arange(9,27,0.1)*2*np.pi*0.1
    omega_sample = fw(k_sample)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.16, 0.16, 0.72, 0.8])
    ax2 = ax.twinx()
    ax.plot(k_sample/2/np.pi/50,omega_sample,linewidth=5,label=r'$\omega$',color='red',linestyle='-')
    ax2.plot(k_sample/2/np.pi/50,omega_sample/k_sample/0.02,color='blue',linewidth=5,label=r'$\omega/k$')
    #ax.set_ylim(0,0.6)
    ax.grid()
    ax.set_xlabel(r'$k_z\lambda_{De}/2 \pi$',fontsize=26)
    ax.set_ylabel(r'$\omega \quad [\omega_{pe}]$',fontsize=26,color='red')
    ax.yaxis.offsetText.set_fontsize(16)
    ax.legend(fontsize=26,loc='center left')
    ax2.legend(fontsize=26,loc='center right')
    ax2.set_ylim(0,1.2)
    ax2.set_ylabel(r'$\omega/k \quad [v_{Te0}]$',fontsize=26,color='blue')
    ax.tick_params(labelsize = 18)
    ax.tick_params(axis='y',colors='red')
    ax2.tick_params(labelsize = 18,colors='blue')
    plt.savefig('./Linear_theory/mass25/eaw.jpg')
    #plt.show()
    plt.clf()

if __name__ == '__main__':
    #mass100()
    #linear_theory()
    #mass25()
    #eaw()