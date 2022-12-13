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

ElcGridPath = './Cori/mass25/rescheck/4/dist_function_save/elc_velocities.npz'
IonGridPath = './Cori/mass25/rescheck/4/dist_function_save/ion_velocities.npz'
SpacePath = './Cori/mass25/rescheck/4/dist_function_save/sapce.npz'

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

def plot_2d_distribution(fName, GridFile):
    df = np.loadtxt(fName)
    grid = np.load(GridFile)
    velocities_z = grid['arr_0']
    velocities_y = grid['arr_1']

    vz_plot, vy_plot = np.meshgrid(velocities_z,velocities_y,indexing='ij')
    
    fig = plt.figure(figsize=(8.5,7.5),facecolor='w', edgecolor='k')
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    df[df<10] = 10

    ax.pcolormesh(vz_plot/0.02, vy_plot/0.02, df,cmap='inferno')
    ax.xlabel(r'$v_z/v_{Te0}$', fontsize=36)
    ax.ylabel(r'$v_y/v_{Te0}$', fontsize=36, labelpad=-1)
    ax.xlim(-4,8)
    ax.ylim(-6,6)
    ax.tick_params(labelsize = 26)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22)

    #plt.savefig(r'./Cori/mass25/rescheck/4/'+r'1800_f2D_.jpg', bbox_inches='tight')
    plt.show()

def plot_phase_space(fName,GridFile):
    df = loadtxt(fName)
    grid_v = np.load(GridFile)
    grid_z = np.load(SpacePath)
    velocities_z = grid_v['arr_0']
    z = grid_z['arr_0']

    zz, vv = np.meshgrid(z,velocities_z,indexing='ij')
    fig = plt.figure(figsize=(8.5,7.5),facecolor='w', edgecolor='k')
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    ax.pcolormesh(zz, vv/0.02, df,cmap='inferno')
    ax.set_xlabel(r'$z /d_e$', fontsize=36)
    ax.set_ylabel(r'$v_z /v_{Te0}$', fontsize=36, labelpad=-1)

    ax.ylim(-4,8)
    ax.tick_params(labelsize = 26)
    ax.grid()

    plt.show()
    plt.clf()

def plot_phi(fName):
    phi = np.loadtxt(fName)

    fig      = plt.figure(figsize=(8.5,7.5))
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    ax.set_xlabel('z',fontsize=32)
    ax.set_ylabel('y',fontsize=32)
    ax.tick_params(labelsize = 26)
    #ax.grid()
    ax.contourf(ZZ, YY, phi, 30,  zdir='z', cmap=matplotlib.cm.coolwarm)
    plt.show()
    plt.clf()

if __name__ == '__main__':
    plot_phi('./Cori/mass25/rescheck/4/field/M25_E2_3_field_0150.txt')
    plot_phase_space('./Cori/mass25/rescheck/4/dist_function_save/750.0_elc_phase.txt', ElcGridPath)
    plot_2d_distribution('./Cori/mass25/rescheck/4/dist_function_save/750.0_elc_2d.txt',ElcGridPath)