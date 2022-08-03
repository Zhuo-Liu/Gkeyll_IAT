import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

ElcGridPath = './Diagnostics/local/Cori/mass25/rescheck/4/dist_function/'

def plot_2d_distribution(fName, GridFile):
    df = np.loadtxt(fName)
    grid = np.loadtxt(GridFile)
    velocities_z = grid[0]
    velocities_y = grid[1]

    vz_plot, vy_plot = np.meshgrid(velocities_z,velocities_y,indexing='ij')
    
    fig = plt.figure(figsize=(8,6),facecolor='w', edgecolor='k')
    
    plt.pcolormesh(vz_plot, vy_plot, df)
    plt.scatter(np.squeeze(vz_plot[np.where(df==np.max(df))]),np.squeeze(vy_plot[np.where(df==np.max(df))]),s = 40, marker = 'x', alpha = 1)
    plt.set_xlabel(r'$v_z$', fontsize=30)
    plt.set_ylabel(r'$v_y$', fontsize=30, labelpad=-1)
    #plt.set_title(r'$<F_e(v_z,v_y)>_{z,y},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 26)
    cbar = fig.colorbar()
    cbar.ax.tick_params(labelsize=22)

    plt.savefig(r'./Diagnostics/local/Cori/mass25/rescheck/3/'+rf'_f2D_.jpg', bbox_inches='tight')
    plt.close()

def plot_1d_distribution(fName, GridFile):
    df = np.loadtxt(fName)
    grid = np.loadtxt(GridFile)
    velocities_z = grid[0]
    
    fig = plt.figure(figsize=(8,6),facecolor='w', edgecolor='k')

    plt.plot(velocities_z, df)
    plt.set_xlabel(r'$v_z$', fontsize=30)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 26)
    cbar = fig.colorbar()
    cbar.ax.tick_params(labelsize=22)

    plt.savefig(r'./Diagnostics/local/Cori/mass25/rescheck/3/'+rf'_f1D_.jpg', bbox_inches='tight')
    plt.close()

def fit_1d(fName,GridFile):
    def maxw(x,A,B,C):
        return A*np.exp(-(x-C)**2/B)
    
    def double_maxwellian(x,a,A1,B1,C1,A2,B2,C2):
        return a*maxw(x,A1,B1,C1) + (1-a)*maxw(A2,B2,C2)
    
    f_e = np.loadtxt(fName)
    grid = np.loadtxt(GridFile)
    v_z = grid[0]

    popt, pcov = curve_fit(double_maxwellian, v_z, f_e)
    constructed_f_e = double_maxwellian(v_z,*popt)

    maxw1 = np.array([maxw(x) for x in v_z])
    maxw2 = np.array([maxw(x) for x in v_z])

    plt.plot(v_z,f_e,label='fe')
    plt.plot(v_z,constructed_f_e,label='fit')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    fit_1d('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/')