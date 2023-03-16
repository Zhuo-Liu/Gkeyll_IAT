#####
# For Fig. 3
######

import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gamma
import matplotlib.ticker as ticker
matplotlib.use('TkAgg')

ElcGridPath = './massRatio/mass100/E5_H2/dist_function_save/elc_velocities.npz'
IonGridPath = './massRatio/mass100/E5_H2/dist_function_save/ion_velocities.npz'
SpacePath = './massRatio/mass100/E5_H2/dist_function_save/sapce.npz'

lz = 1.0
ly = 0.5

nz = 96
ny = 48

dz = lz/nz
dy = ly/ny

z_plot = np.linspace(0,lz,nz)
y_plot = np.linspace(0,ly,ny)
ZZ, YY = np.meshgrid(z_plot, y_plot, indexing= 'xy')
ZZ = np.transpose(ZZ)
YY = np.transpose(YY)

kz_plot  = 2.0*np.pi*np.linspace(-int(nz/2), int(nz/2-1), nz)/lz
ky_plot  = 2.0*np.pi*np.linspace(-int(ny/2), int(ny/2-1), ny)/ly
K_z, K_y = np.meshgrid(kz_plot, ky_plot, indexing = 'xy')
K_z = np.transpose(K_z)
K_y = np.transpose(K_y)

def plot_2d_elc_distribution(fName, GridFile):
    df = np.loadtxt(fName)
    grid = np.load(GridFile)
    velocities_z = grid['arr_0']
    velocities_y = grid['arr_1']

    vz_plot, vy_plot = np.meshgrid(velocities_z,velocities_y,indexing='ij')
    
    fig = plt.figure(figsize=(8.5,7.5),facecolor='w', edgecolor='k')
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    df[df<1] = 1.0

    ax.pcolormesh(vz_plot/0.02, vy_plot/0.02, df,cmap='inferno')
    ax.set_xlabel(r'$v_z/v_{Te0}$', fontsize=36)
    ax.set_ylabel(r'$v_y/v_{Te0}$', fontsize=36, labelpad=-1)
    ax.set_xlim(-4,13)
    ax.set_ylim(-6,6)
    ax.tick_params(labelsize = 30)
    ax.grid(which='major',color='grey', linestyle='-', linewidth=1)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=26)

    #plt.savefig(r'./paper_figures/2D_elc.jpg', bbox_inches='tight')
    plt.show()
    plt.cla()

def plot_2d_ion_distribution(fName, GridFile):
    df = np.loadtxt(fName)
    grid = np.load(GridFile)
    velocities_z = grid['arr_0']
    velocities_y = grid['arr_1']

    vz_plot, vy_plot = np.meshgrid(velocities_z,velocities_y,indexing='ij')
    
    fig = plt.figure(figsize=(8.5,7.5),facecolor='w', edgecolor='k')
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    #df[df<100] = 1.0

    ax.pcolormesh(vz_plot/(0.02/np.sqrt(50*100)), vy_plot/(0.02/np.sqrt(50*100)), df,cmap='inferno')
    ax.set_xlabel(r'$v_z/v_{Ti0}$', fontsize=36)
    ax.set_ylabel(r'$v_y/v_{Ti0}$', fontsize=36, labelpad=-1)
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    ax.tick_params(labelsize = 30)
    ax.grid(which='major',color='grey', linestyle='-', linewidth=1)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=26)

    #plt.savefig(r'./paper_figures/2D_ion.jpg', bbox_inches='tight')
    plt.show()
    plt.cla()

def plot_phase_space(fName,GridFile):
    df = loadtxt(fName)
    grid_v = np.load(GridFile)
    grid_z = np.load(SpacePath)
    velocities_z = grid_v['arr_0']
    z = grid_z['arr_0']

    zz, vv = np.meshgrid(z,velocities_z,indexing='ij')
    fig = plt.figure(figsize=(8.5,7.5),facecolor='w', edgecolor='k')
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    df[df<5] = 0
    ax.pcolormesh(zz, vv/0.02, df,cmap='inferno')
    #ax.pcolormesh(zz, vv/0.02, df)
    ax.set_xlabel(r'$z /d_e$', fontsize=32)
    ax.set_ylabel(r'$v_z /v_{Te0}$', fontsize=32, labelpad=-1)

    ax.set_ylim(-4,13)
    ax.tick_params(labelsize = 32)

    #ax.grid(which='major')

    plt.show()
    #plt.savefig(r'./paper_figures/phase.jpg', bbox_inches='tight')
    plt.cla()

def plot_phase_space_ion(fName,GridFile):
    df = loadtxt(fName)
    grid_v = np.load(GridFile)
    grid_z = np.load(SpacePath)
    velocities_z = grid_v['arr_0']
    z = grid_z['arr_0']

    zz, vv = np.meshgrid(z,velocities_z,indexing='ij')
    fig = plt.figure(figsize=(8.5,7.5),facecolor='w', edgecolor='k')
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    df[df<5] = 0
    ax.pcolormesh(zz, vv/(0.02/np.sqrt(50*100)), df,cmap='inferno')
    ax.set_xlabel(r'$z /d_e$', fontsize=32)
    ax.set_ylabel(r'$v_z /v_{Ti0}$', fontsize=32, labelpad=-1)

    ax.set_ylim(-24,36)
    ax.tick_params(labelsize = 32)

    #ax.grid(which='major')

    plt.show()
    #plt.savefig(r'./paper_figures/phase.jpg', bbox_inches='tight')
    plt.cla()

def plot_phi(fName):
    phi = np.loadtxt(fName)

    fig      = plt.figure(figsize=(10.5,7.5),facecolor='w', edgecolor='k')
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    ax.set_xlabel(r'$z/d_e$',fontsize=36)
    ax.set_ylabel(r'$y/d_e$',fontsize=36)
    ax.tick_params(labelsize = 32)
    #ax.grid()
    #ax.contourf(ZZ, YY, phi, 30,  zdir='z', cmap=matplotlib.cm.coolwarm)
    pos = ax.contourf(ZZ, YY, phi/0.0004, 30,  zdir='z', cmap='inferno')
    cbar = fig.colorbar(pos,ax=ax)
    cbar.set_label(r"$e\phi/T_{e0}$",fontsize=32)
    cbar.ax.tick_params(labelsize=22)
    plt.show()
    #plt.savefig(r'./paper_figures/phi.jpg', bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    #plot_phase_space('./massRatio/mass100/E5_H2/dist_function/1000.0_elc_phase.txt', ElcGridPath)

    #plot_phi('./massRatio/mass100/E5_H2/field/M100_E5_field_0100.txt')
    #plot_phase_space('./massRatio/mass100/E5_H2/dist_function/1800.0_elc_phase.txt', ElcGridPath)
    #plot_phase_space_ion('./massRatio/mass100/E5_H2/dist_function/1800.0_ion_phase.txt', IonGridPath)
    #plot_2d_elc_distribution('./massRatio/mass100/E5_H2/dist_function/1800.0_elc_2d.txt',ElcGridPath)
    #plot_2d_ion_distribution('./massRatio/mass100/E5_H2/dist_function/1800.0_ion_2d.txt',IonGridPath)

    #plot_phi('./massRatio/mass100/E5_H2/field/M100_E5_field_0376.txt')
    plot_phase_space('./massRatio/mass100/E5_H2/dist_function/1800.0_elc_phase.txt', ElcGridPath)
    #plot_phase_space_ion('./massRatio/mass100/E5_H2/dist_function/3500.0_ion_phase.txt', IonGridPath)
    #plot_2d_elc_distribution('./massRatio/mass100/E5_H2/dist_function/3200.0_elc_2d.txt',ElcGridPath)
    #plot_2d_ion_distribution('./massRatio/mass100/E5_H2/dist_function/3700.0_ion_2d.txt',IonGridPath)

    #plot_phi('./Cori/mass25/rescheck/4/field/M25_E2_3_field_0430.txt')
    #plot_phase_space('./massRatio/mass100/E5_H2/dist_function/4300.0_elc_phase.txt', ElcGridPath)
    #plot_phase_space_ion('./massRatio/mass100/E5_H2/dist_function/4300.0_ion_phase.txt', IonGridPath)
    #plot_2d_elc_distribution('./massRatio/mass100/E5_H2/dist_function/4300.0_elc_2d.txt',ElcGridPath)
    #plot_2d_ion_distribution('./massRatio/mass100/E5_H2/dist_function/4200.0_ion_2d.txt',IonGridPath)