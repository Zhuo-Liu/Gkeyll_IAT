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
    df_1600 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1600.0_elc_1d.txt')
    df_3500 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/3500.0_elc_1d.txt')
    df_4000= np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/4000.0_elc_1d.txt')
    #df7 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/2400.0_elc_1d.txt')

    grid = np.load(ElcGridPath)
    velocities_z = grid['arr_0']
    
    # maxw1 = np.array([maxw(x,1.75,0.032,0.002) for x in velocities_z])
    # maxw2 = np.array([maxw(x,0.84,0.033,0.06) for x in velocities_z])
    # maxw3 = np.array([maxw(x,0.76,0.033,0.08) for x in velocities_z])

    maxw1 = np.array([maxw(x,1.66,0.0313,0.000) for x in velocities_z])
    maxw2 = np.array([maxw(x,1.14,0.035,0.0487) for x in velocities_z])
    maxw3 = np.array([maxw(x,0.82,0.035,0.078) for x in velocities_z])

    fig = plt.figure(figsize=(16,10),facecolor='w', edgecolor='k')

    plt.plot(velocities_z/0.02, df_500,label=r'$\omega_{pe}t=500$',linewidth=6)
    plt.plot(velocities_z/0.02, df_700,label=r'$\omega_{pe}t=700$',linewidth=6)
    plt.plot(velocities_z/0.02, df_1000,label=r'$\omega_{pe}t=1000$',linewidth=6)
    plt.plot(velocities_z/0.02, df_1600,label=r'$\omega_{pe}t=1600$',linewidth=6)
    plt.plot(velocities_z/0.02, maxw1,linewidth=5,linestyle='--',label=r'bulk',color='black')
    plt.plot(velocities_z/0.02, maxw2,linewidth=5,linestyle='--',label=r'tail at $\omega_{pe}t=1000$',color='green')
    plt.plot(velocities_z/0.02, maxw3,linewidth=5,linestyle='--',label=r'tail at $\omega_{pe}t=1600$',color='red')
    plt.plot(velocities_z/0.02, df_3500,label=r'$\omega_{pe}t=3500$',linewidth=6)
    plt.plot(velocities_z/0.02, df_4000,label=r'$\omega_{pe}t=4000$',linewidth=6)

    resonance = np.arange(-1.2,1.4,0.1)
    plt.fill_between(resonance, 0, 56, facecolor='grey', alpha=0.5)
    #plt.vlines(1.0,0,25,linewidth=3,linestyles='--',color='black')


    plt.xlabel(r'$v_z/v_{Te0}$', fontsize=36)
    plt.ylabel(r'$F_e (v_z)$', fontsize=36)
    plt.grid()
    plt.legend(fontsize=26)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 28)
    plt.xlim(-0.08/0.02,0.30/0.02)
    plt.ylim(-3,56)
    #plt.savefig('./Cori/figure_temp/elc_1d.jpg')
    plt.show()


def fit_1d_numerical(fName, GridFile):
    def double_maxwellian(x,A1,B1,C1,A2,B2,C2):
        return maxw(x,A1,B1,C1) + maxw(x,A2,B2,C2)
    

    f_e = np.loadtxt(fName)
    grid = np.load(GridFile)
    v_z = grid['arr_0']

    popt, pcov = curve_fit(double_maxwellian, v_z, f_e)
    constructed_f_e = double_maxwellian(v_z,*popt)

    maxw1 = np.array([maxw(x,popt[0],popt[1],popt[2]) for x in v_z])
    maxw2 = np.array([maxw(x,popt[3],popt[4],popt[5]) for x in v_z])


    plt.plot(v_z,f_e,label='fe2')
    plt.plot(v_z,constructed_f_e,label='fit')
    plt.plot(v_z,maxw1,label='1',linestyle='--')
    plt.plot(v_z,maxw2,label='2',linestyle='--')
    plt.legend()
    plt.show()
    if popt[2] < popt[5]:
        print('==========Maxwell 1=========')
        print("Norm:",popt[0])
        print("Vte",popt[1])
        print("Drift",popt[2])
        print('==========Maxwell 2=========')
        print("Norm:",popt[3])
        print("Vte",popt[4])
        print("Drift",popt[5])
        print("")
        print("ratio:",popt[3]/popt[0])
    else:
        print('==========Maxwell 1=========')
        print("Norm:",popt[3])
        print("Vte",popt[4])
        print("Drift",popt[5])
        print('==========Maxwell 2=========')
        print("Norm:",popt[0])
        print("Vte",popt[1])
        print("Drift",popt[2])
        print("")
        print("ratio:",popt[0]/popt[3])

def fit_1d_1600(fName, GridFile):
    f_e = np.loadtxt(fName)
    grid = np.load(GridFile)
    v_z = grid['arr_0']


    # maxw1 = np.array([maxw(x,1.76,0.033,0.002) for x in v_z])
    # maxw2 = np.array([maxw(x,0.75,0.033,0.08) for x in v_z])
    maxw1 = np.array([maxw(x,1.66,0.031,0.000) for x in v_z])
    maxw2 = np.array([maxw(x,0.82,0.035,0.078) for x in v_z])

    fig      = plt.figure(figsize=(12.5,7.5))
    fig.add_axes([0.1, 0.16, 0.8, 0.8])
    plt.plot(v_z/0.02,f_e, linewidth=5,label=r'$F_e(v_z)$')

    plt.plot(v_z/0.02,maxw1,label='Max1',linestyle='--',linewidth=5)
    plt.plot(v_z/0.02,maxw2,label='Max2',linestyle='--',linewidth=5)
    plt.plot(v_z/0.02,maxw2+maxw1,label='1+2',linestyle='--',linewidth=5)
    plt.grid()
    plt.xlim(-6,12)
    plt.legend(fontsize=28,loc='upper left')
    plt.xlabel(r'$v_z/v_{Te0}$',fontsize=28)
    plt.tick_params(labelsize = 24)
    plt.show()

    return

def fit_1d_1000(fName, GridFile):
    f_e = np.loadtxt(fName)
    grid = np.load(GridFile)
    v_z = grid['arr_0']


    # maxw1 = np.array([maxw(x,1.85,0.032,0.002) for x in v_z])
    # maxw2 = np.array([maxw(x,0.86,0.033,0.05) for x in v_z])
    maxw1 = np.array([maxw(x,1.66,0.0313,0.000) for x in v_z])
    maxw2 = np.array([maxw(x,1.14,0.035,0.0487) for x in v_z])

    fig      = plt.figure(figsize=(12.5,7.5))
    fig.add_axes([0.1, 0.16, 0.8, 0.8])
    plt.plot(v_z/0.02,f_e, linewidth=5,label=r'$F_e(v_z)$')

    plt.plot(v_z/0.02,maxw1,label='Max1',linestyle='--',linewidth=5)
    plt.plot(v_z/0.02,maxw2,label='Max2',linestyle='--',linewidth=5)
    plt.plot(v_z/0.02,maxw2+maxw1,label='1+2',linestyle='--',linewidth=5)
    plt.grid()
    plt.xlim(-6,12)
    plt.legend(fontsize=28,loc='upper left')
    plt.xlabel(r'$v_z/v_{Te0}$',fontsize=28)
    plt.tick_params(labelsize = 24)
    plt.show()

    return

def fit_1d(fname,GridFile):
    f_e = np.loadtxt(fname)
    grid = np.load(GridFile)
    v_z = grid['arr_0']


    #maxw1 = np.array([maxw(x,1.68,0.0315,0.0014) for x in v_z])
    #maxw2 = np.array([maxw(x,0.76,0.0338,0.079) for x in v_z])
    maxw1 = np.array([maxw(x,2.8,0.02828,0.01) for x in v_z])

    fig      = plt.figure(figsize=(12.5,7.5))
    fig.add_axes([0.1, 0.16, 0.8, 0.8])
    plt.plot(v_z/0.02,f_e, linewidth=5,label=r'$F_e(v_z)$')

    plt.plot(v_z/0.02,maxw1,label='Max1',linestyle='--',linewidth=5)
    #plt.plot(v_z/0.02,maxw2,label='Max2',linestyle='--',linewidth=5)
    #plt.plot(v_z/0.02,maxw2+maxw1,label='1+2',linestyle='--',linewidth=5)
    plt.grid()
    plt.xlim(-6,12)
    plt.legend(fontsize=28,loc='upper left')
    plt.xlabel(r'$v_z/v_{Te0}$',fontsize=28)
    plt.tick_params(labelsize = 24)
    plt.show()

    return

def ion_main():
    df_500 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/500.0_ion_1d.txt')
    df_700 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/650.0_ion_1d.txt')
    df_1000 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1000.0_ion_1d.txt')
    df_1600 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1600.0_ion_1d.txt')
    df_3500 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/3500.0_ion_1d.txt')
    df_4000= np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/4000.0_ion_1d.txt')

    grid = np.load(IonGridPath)
    velocities_z = grid['arr_0']
    


    plt.figure(figsize=(13,12))

    plt.plot(velocities_z/0.002, df_500,label=r'$\omega_{pe}t=500$',linewidth=6)
    plt.plot(velocities_z/0.002, df_700,label=r'$\omega_{pe}t=700$',linewidth=6)
    plt.plot(velocities_z/0.002, df_1000,label=r'$\omega_{pe}t=1000$',linewidth=6)
    plt.plot(velocities_z/0.002, df_1600,label=r'$\omega_{pe}t=1600$',linewidth=6)
    plt.plot(velocities_z/0.002, df_3500,label=r'$\omega_{pe}t=3500$',linewidth=6)
    #plt.plot(velocities_z/0.002, df_4000,label=r'$\omega_{pe}t=4000$',linewidth=6)
    #plt.vlines(1.0,0,25,linewidth=3,linestyles='--',color='black')


    plt.xlabel(r'$v_z/c_{s0}$', fontsize=36)
    plt.ylabel(r'$F_i (v_z)$', fontsize=36)
    plt.grid()
    plt.legend(fontsize=36)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 36)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.gca().yaxis.offsetText.set_fontsize(28)
    plt.xlim(-3,3)
    plt.ylim(0,100000)
    plt.show()

if __name__ == '__main__':
    # ion_main()
    #elc_main()

    #### Fitting 
    #fit_1d_numerical('./massRatio/mass100/E5_H2/dist_function_save/1600.0_elc_1d.txt', ElcGridPath)
    #fit_1d_1600('./massRatio/mass100/E5_H2/dist_function_save/1600.0_elc_1d.txt', ElcGridPath)
    ##fit_1d_1000('./massRatio/mass100/E5_H2/dist_function_save/1000.0_elc_1d.txt', ElcGridPath)
    #fit_1d_numerical('./massRatio/mass100/E5_H2/dist_function_save/1400.0_elc_1d.txt', ElcGridPath)
    
    fit_1d_numerical('./massRatio/mass100/E5_H2/dist_function_save/500.0_elc_1d.txt', ElcGridPath)
    #fit_1d('./massRatio/mass100/E5_H2/dist_function_save/200.0_elc_1d.txt', ElcGridPath)