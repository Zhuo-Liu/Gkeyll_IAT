import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gamma
import matplotlib.ticker as ticker

ElcGridPath = './massRatio/mass100/E5_H2/dist_function_save/elc_velocities.npz'
IonGridPath = './massRatio/mass100/E5_H2/dist_function_save/ion_velocities.npz'
SpacePath = './massRatio/mass100/E5_H2/dist_function_save/sapce.npz'

def maxw(x,A,B,C):
    return A/np.sqrt(np.pi)/B*np.exp(-(x-C)**2/B**2)

def elc_main():
    df_500 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/500.0_elc_1d.txt')
    df_700 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/650.0_elc_1d.txt')
    df_1000 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1000.0_elc_1d.txt')
    df_1500 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1500.0_elc_1d.txt')
    df_3500 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/3500.0_elc_1d.txt')
    df_4000= np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/4000.0_elc_1d.txt')
    #df7 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/2400.0_elc_1d.txt')

    grid = np.load(ElcGridPath)
    velocities_z = grid['arr_0']
    
    maxw1 = np.array([maxw(x,1.4,0.0246,0.00) for x in velocities_z])
    maxw2 = np.array([maxw(x,1.15,0.02716,0.0444) for x in velocities_z])
    maxw3 = np.array([maxw(x,1.15,0.02716,0.0444+0.0075) for x in velocities_z])

    #maxw2 = np.array([maxw(x,1.15,0.031,0.042) for x in velocities_z])
    #maxw3 = np.array([maxw(x,1.15,0.033,0.047) for x in velocities_z])

    fig = plt.figure(figsize=(16,10),facecolor='w', edgecolor='k')

    plt.plot(velocities_z/0.02, df_500,label=r'$\omega_{pe}t=500$',linewidth=6)
    plt.plot(velocities_z/0.02, df_700,label=r'$\omega_{pe}t=700$',linewidth=6)
    plt.plot(velocities_z/0.02, df_1000,label=r'$\omega_{pe}t=1000$',linewidth=6)
    plt.plot(velocities_z/0.02, df_1500,label=r'$\omega_{pe}t=1500$',linewidth=6)
    plt.plot(velocities_z/0.02, df_3500,label=r'$\omega_{pe}t=3500$',linewidth=6)
    plt.plot(velocities_z/0.02, df_4000,label=r'$\omega_{pe}t=4000$',linewidth=6)
    plt.vlines(1.0,0,25,linewidth=3,linestyles='--',color='black')
    #plt.plot(velocities_z/0.02, maxw1,linewidth=5,linestyle='--',label=r'bulk electron after saturation')
    #plt.plot(velocities_z/0.02, maxw2,linewidth=5,linestyle='--',label=r'run-away tail at $\omega_{pe}t=750$',color='orange')
    #plt.plot(velocities_z/0.02, maxw3,linewidth=5,linestyle='--',label=r'run-away tail at $\omega_{pe}t=900$',color='green')

    # plt.plot(velocities_z/0.02, maxw1,linewidth=5,linestyle='--',label=r'bulk',color='black')
    # plt.plot(velocities_z/0.02, maxw2,linewidth=5,linestyle='--',label=r'tail at $\omega_{pe}t=750$',color='green')
    # plt.plot(velocities_z/0.02, maxw3,linewidth=5,linestyle='--',label=r'tail at $\omega_{pe}t=900$',color='red')
    # plt.plot(velocities_z/0.02, df6,label=r'$\omega_{pe}t=1600$',linewidth=6)
    # plt.plot(velocities_z/0.02, df7,label=r'$\omega_{pe}t=2400$',linewidth=6)

    # plt.plot(velocities_z, df2_constructed,label='',linewidth=2)
    # plt.plot(velocities_z, df3_constructed,label='',linewidth=2)

    plt.xlabel(r'$v_z/v_{Te0}$', fontsize=36)
    plt.ylabel(r'$F_e (v_z)$', fontsize=36)
    plt.grid()
    plt.legend(fontsize=26)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 28)
    plt.xlim(-0.06/0.02,0.30/0.02)
    #plt.savefig('./Cori/figure_temp/elc_1d.jpg')
    plt.show()

if __name__ == '__main__':
    elc_main()