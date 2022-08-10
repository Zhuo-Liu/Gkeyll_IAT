import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

ElcGridPath = './Diagnostics/local/Cori/mass25/rescheck/4/dist_function/elc_velocities.npz'
IonGridPath = './Diagnostics/local/Cori/mass25/rescheck/4/dist_function/ion_velocities.npz'

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
    def maxw(x,A,B,C):
        return A*np.exp(-(x-C)**2/B)
    
    def double_maxwellian(x,A1,B1,C1,A2,B2,C2):
        return maxw(x,A1,B1,C1) + maxw(x,A2,B2,C2)
    
    f_e = np.loadtxt(fName)
    grid = np.load(GridFile)
    v_z = grid['arr_0']

    popt, pcov = curve_fit(double_maxwellian, v_z, f_e)
    constructed_f_e = double_maxwellian(v_z,*popt)

    maxw1 = np.array([maxw(x,popt[0],popt[1],popt[2]) for x in v_z])
    maxw2 = np.array([maxw(x,popt[3],popt[4],popt[5]) for x in v_z])

    plt.plot(v_z,f_e,label='fe')
    plt.plot(v_z,constructed_f_e,label='fit')
    plt.plot(v_z,maxw1,label='1')
    plt.plot(v_z,maxw2,label='2')
    plt.legend()
    plt.show()

    
    print(popt[2])
    print(popt[5])

if __name__ == '__main__':
    #fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/750.0_elc_1d.txt',ElcGridPath)

    #plot_2d_distribution('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/625.0_elc_2d.txt',ElcGridPath)
    #plot_2d_distribution('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1800.0_elc_2d.txt',ElcGridPath)

   
    plot_1d_distribution('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/625.0_ion_1d.txt', IonGridPath)
    plot_1d_distribution('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/1800.0_ion_1d.txt', IonGridPath)