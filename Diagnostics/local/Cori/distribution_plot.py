import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gamma

ElcGridPath = './Diagnostics/local/Cori/mass25/rescheck/4/dist_function/elc_velocities.npz'
IonGridPath = './Diagnostics/local/Cori/mass25/rescheck/4/dist_function/ion_velocities.npz'
def maxw(x,A,B,C):
    return A/np.sqrt(np.pi)/B*np.exp(-(x-C)**2/B**2)

def kappa(x,kap,A,B,C):
    return A/(np.sqrt(np.pi))/B*gamma(kap+1)/(kap**(1.5))/gamma(kap-0.5)*(1+(x-C)**2/kap/B**2)**(-kap-1)

def double_maxwellian(x,A1,B1,C1,A2,B2,C2):
    return maxw(x,A1,B1,C1) + maxw(x,A2,B2,C2)

def plot_2d_distribution(fName, GridFile):
    df = np.loadtxt(fName)
    grid = np.load(GridFile)
    velocities_z = grid['arr_0']
    velocities_y = grid['arr_1']

    vz_plot, vy_plot = np.meshgrid(velocities_z,velocities_y,indexing='ij')
    
    fig = plt.figure(figsize=(8,6),facecolor='w', edgecolor='k')
    
    plt.pcolormesh(vz_plot, vy_plot, df,cmap='inferno')
    #plt.scatter(np.squeeze(vz_plot[np.where(df==np.max(df))]),np.squeeze(vy_plot[np.where(df==np.max(df))]),s = 40, marker = 'x', alpha = 1)
    plt.xlabel(r'$v_z$', fontsize=30)
    plt.ylabel(r'$v_y$', fontsize=30, labelpad=-1)
    #plt.set_title(r'$<F_e(v_z,v_y)>_{z,y},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.xlim(-0.04,0.14)
    plt.ylim(-0.1,0.1)
    plt.tick_params(labelsize = 26)
    plt.grid()
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22)

    plt.savefig(r'./Diagnostics/local/Cori/mass25/rescheck/4/'+r'1800_f2D_.jpg', bbox_inches='tight')
    plt.close()

def plot_1d_distribution(fName, GridFile):
    df = np.loadtxt(fName)
    grid = np.load(GridFile)
    velocities_z = grid['arr_0']
    
    fig = plt.figure(figsize=(8,6),facecolor='w', edgecolor='k')

    plt.plot(velocities_z, df)
    plt.xlabel(r'$v_z$', fontsize=30)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 26)

    plt.show()
    #plt.savefig(r'./Diagnostics/local/Cori/mass25/rescheck/3/'+rf'_f1D_.jpg', bbox_inches='tight')
    #plt.close()

def fit_1d(fName,GridFile):
    # def double_maxwellian(x,A,B1,C1,B2,C2):
    #     return maxw(x,A,B1,C1) + maxw(x,3*A,B2,C2)
    
    def kappa_maxwellian(x,A1,B1,C1,A2,B2,C2):
        return kappa(x,10,A1,B1,C1) + maxw(x,A2,B2,C2)

    f_e = np.loadtxt(fName)
    grid = np.load(GridFile)
    v_z = grid['arr_0']

    popt, pcov = curve_fit(double_maxwellian, v_z, f_e)
    constructed_f_e = double_maxwellian(v_z,*popt)

    # popt, pcov = curve_fit(kappa_maxwellian, v_z, f_e)
    # constructed_f_e = kappa_maxwellian(v_z,*popt)

    #maxw1 = np.array([maxw(x,popt[0],popt[1],popt[2]) for x in v_z])
    #maxw2 = np.array([maxw(x,popt[3],popt[4],popt[5]) for x in v_z])

    maxw1 = np.array([maxw(x,1.47,0.0246,0.00) for x in v_z])
    maxw2 = np.array([maxw(x,1.25,0.02716,0.0444) for x in v_z])

    f_e2 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/850.0_elc_1d.txt')
    maxw3 = np.array([maxw(x,1.25,0.02716,0.0444+0.005) for x in v_z])
    plt.plot(v_z,f_e2)
    #plt.plot(v_z,constructed_f_e,label='fit')
    plt.plot(v_z,maxw1,label='1',linestyle='--')
    plt.plot(v_z,maxw3,label='2',linestyle='--')
    plt.plot(v_z,maxw3+maxw1,label='fit',linestyle='--')
    plt.legend()
    plt.show()

    print(popt)

    #maxw3 = np.array([maxw(x,popt[3],popt[4],popt[5]+0.005) for x in v_z])

    # f_e2 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/750.0_elc_1d.txt')
    # plt.plot(v_z,f_e2,label='fe2')
    # #plt.plot(v_z,constructed_f_e,label='fit')
    # plt.plot(v_z,maxw1,label='1',linestyle='--')
    # #plt.plot(v_z,maxw2,label='2',linestyle='--')
    # plt.plot(v_z,maxw3,label='3',linestyle='--')
    # plt.plot(v_z,maxw3+maxw1,label='fit2',linestyle='--')
    # plt.legend()
    # plt.show()
    return constructed_f_e, maxw1, maxw2
    # if popt[2] < popt[5]:
    #     print('==========Maxwell 1=========')
    #     print("Norm:",popt[0])
    #     print("Vte",popt[1])
    #     print("Drift",popt[2])
    #     print('==========Maxwell 2=========')
    #     print("Norm:",popt[3])
    #     print("Vte",popt[4])
    #     print("Drift",popt[5])
    #     print("")
    #     print("ratio:",popt[3]/popt[0])
    # else:
    #     print('==========Maxwell 1=========')
    #     print("Norm:",popt[3])
    #     print("Vte",popt[4])
    #     print("Drift",popt[5])
    #     print('==========Maxwell 2=========')
    #     print("Norm:",popt[0])
    #     print("Vte",popt[1])
    #     print("Drift",popt[2])
    #     print("")
    #     print("ratio:",popt[0]/popt[3])

def elc_main():
    df1 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/400.0_elc_1d.txt')
    df2 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/750.0_elc_1d.txt')
    df3 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/900.0_elc_1d.txt')
    df4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1600.0_elc_1d.txt')
    df5 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/2400.0_elc_1d.txt')

    grid = np.load(ElcGridPath)
    velocities_z = grid['arr_0']
    
    maxw1 = np.array([maxw(x,1.4,0.0246,0.00) for x in velocities_z])
    maxw2 = np.array([maxw(x,1.15,0.02716,0.0444) for x in velocities_z])
    maxw3 = np.array([maxw(x,1.15,0.02716,0.0444+0.0075) for x in velocities_z])

    fig = plt.figure(figsize=(8,6),facecolor='w', edgecolor='k')

    plt.plot(velocities_z/0.02, df1,label=r'$\omega_{pe}t=400$ (before saturation)',linewidth=4)
    plt.plot(velocities_z/0.02, df2,label=r'$\omega_{pe}t=750$  (after saturation)',linewidth=4)
    plt.plot(velocities_z/0.02, df3,label=r'$\omega_{pe}t=900$  (after saturation)',linewidth=4)
    plt.plot(velocities_z/0.02, df4,label=r'$\omega_{pe}t=1600$  (nonlinear phase affected by heated ions)',linewidth=4)
    plt.plot(velocities_z/0.02, df5,label=r'$\omega_{pe}t=2400$  (second instatbility phase)',linewidth=4)
    # plt.plot(velocities_z, df2_constructed,label='',linewidth=2)
    # plt.plot(velocities_z, df3_constructed,label='',linewidth=2)
    plt.plot(velocities_z/0.02, maxw1,linewidth=3,linestyle='--',label=r'bulk electron after saturation')
    plt.plot(velocities_z/0.02, maxw2,linewidth=3,linestyle='--',label=r'run-away tail after saturation at $\omega_{pe}t=750$',color='orange')
    plt.plot(velocities_z/0.02, maxw3,linewidth=3,linestyle='--',label=r'run-away tail after saturation at $\omega_{pe}t=900$',color='green')

    plt.xlabel(r'$v_z/v_{Te}$', fontsize=30)
    plt.ylabel(r'$F_e (v_z)$', fontsize=30)
    plt.grid()
    plt.legend(fontsize=20)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 26)
    plt.xlim(-0.06/0.02,0.24/0.02)

    plt.show()

def ion_main():
    df1 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/400.0_ion_1d.txt')
    df2 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/750.0_ion_1d.txt')
    df3 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/900.0_ion_1d.txt')
    df4 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1600.0_ion_1d.txt')
    df5 = np.loadtxt('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/2400.0_ion_1d.txt')

    grid = np.load(IonGridPath)
    velocities_z = grid['arr_0']
    
    # fig = plt.figure(figsize=(8,6),facecolor='w', edgecolor='k')

    plt.plot(velocities_z/(0.02/np.sqrt(50)/5), df1,label=r'$\omega_{pe}t=400$',linewidth=2)
    plt.plot(velocities_z/(0.02/np.sqrt(50)/5), df2,label=r'$\omega_{pe}t=750$',linewidth=2)
    plt.plot(velocities_z/(0.02/np.sqrt(50)/5), df4,label=r'$\omega_{pe}t=1600$',linewidth=2)
    plt.plot(velocities_z/(0.02/np.sqrt(50)/5), df5,label=r'$\omega_{pe}t=2400$',linewidth=2)

    plt.vlines(0.004/(0.02/np.sqrt(50)/5),0,26000,color='black',linestyles='dashed',linewidth=3)
    plt.text(4,10000,r'$c_s$',fontsize=30)

    plt.xlabel(r'$v_z/v_{Ti0}$', fontsize=30)
    plt.ylabel(r'$F_i (v_z)$', fontsize=30)
    plt.grid()
    plt.legend(fontsize=20)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 26)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.gca().yaxis.offsetText.set_fontsize(24)
    plt.xlim(-20,20)
    plt.ylim(0,25000)
    plt.show()

if __name__ == '__main__':
    ion_main()
    #fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/750.0_elc_1d.txt',ElcGridPath)
    #popt = fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/650.0_elc_1d.txt',ElcGridPath)

    # fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/875.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1000.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1130.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1200.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1380.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1500.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1630.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1750.0_elc_1d.txt',ElcGridPath)
    #plot_2d_distribution('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/625.0_elc_2d.txt',ElcGridPath)
    #plot_2d_distribution('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1800.0_elc_2d.txt',ElcGridPath)

   
    #plot_1d_distribution('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/625.0_ion_1d.txt', IonGridPath)
    #plot_1d_distribution('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1800.0_ion_1d.txt', IonGridPath)