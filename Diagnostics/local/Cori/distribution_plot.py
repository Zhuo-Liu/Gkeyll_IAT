import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import numpy as np

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

    plt.pcolormesh(velocities_z, df)
    plt.set_xlabel(r'$v_z$', fontsize=30)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 26)
    cbar = fig.colorbar()
    cbar.ax.tick_params(labelsize=22)

    plt.savefig(r'./Diagnostics/local/Cori/mass25/rescheck/3/'+rf'_f1D_.jpg', bbox_inches='tight')
    plt.close()