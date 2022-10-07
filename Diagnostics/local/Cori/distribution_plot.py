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
    
    fig = plt.figure(figsize=(15,12),facecolor='w', edgecolor='k')

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    #ax.contour3D(vz_plot, vy_plot, df, 100)
    # p = ax.plot_surface(vz_plot, vy_plot, df, rstride=4, cstride=4, linewidth=0,cmap=cm.coolwarm)
    # ax.set_xlabel('v_z')
    # ax.set_ylabel('v_Y')

    # plt.show()
    
    df[df<10] = 10

    plt.pcolormesh(vz_plot/0.02, vy_plot/0.02, df,cmap='inferno')
    #plt.scatter(np.squeeze(vz_plot[np.where(df==np.max(df))]),np.squeeze(vy_plot[np.where(df==np.max(df))]),s = 40, marker = 'x', alpha = 1)
    plt.xlabel(r'$v_z/v_{Te0}$', fontsize=36)
    plt.ylabel(r'$v_y/v_{Te0}$', fontsize=36, labelpad=-1)
    #plt.set_title(r'$<F_e(v_z,v_y)>_{z,y},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.xlim(-4,8)
    plt.ylim(-6,6)
    plt.tick_params(labelsize = 26)
    plt.grid()
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22)

    plt.savefig(r'./Cori/mass25/rescheck/4/'+r'1800_f2D_.jpg', bbox_inches='tight')
    plt.show()
    
    #plt.close()

def plot_phase_space(fName,GridFile):
    df = loadtxt(fName)
    grid_v = np.load(GridFile)
    grid_z = np.load(SpacePath)
    velocities_z = grid_v['arr_0']

    z = grid_z['arr_0']

    zz, vv = np.meshgrid(z,velocities_z,indexing='ij')
    fig = plt.figure(figsize=(15,12),facecolor='w', edgecolor='k')



    plt.pcolormesh(zz, vv/0.02, df,cmap='inferno')
    plt.xlabel(r'$z /d_e$', fontsize=36)
    plt.ylabel(r'$v_z /v_{Te0}$', fontsize=36, labelpad=-1)
    #plt.ylabel(r'$v_z /v_{Te0}$', fontsize=36, labelpad=-1)
    # plt.xlim(-0.04,0.14)
    plt.ylim(-4,10)
    plt.tick_params(labelsize = 26)
    plt.grid()
    cbar = plt.colorbar()
    #cbar.formatter.set_powerlimits((0, 0))
    #cbar.ax.yaxis.get_offset_text().set_fontsize(20)
    cbar.ax.tick_params(labelsize=22)
    #cbar.update_ticks()

    plt.show()

def plot_real_space(fName):
    df = loadtxt(fName)
    grid = np.load(SpacePath)
    z = grid['arr_0']
    y = grid['arr_1']

    zz, yy = np.meshgrid(z,y,indexing='ij')
    fig = plt.figure(figsize=(15,12),facecolor='w', edgecolor='k')

    plt.pcolormesh(zz, yy, df,cmap='inferno')
    plt.xlabel(r'$z/d_e$', fontsize=36)
    plt.ylabel(r'$y/d_e$', fontsize=36, labelpad=-1)
    # plt.xlim(-0.04,0.14)
    # plt.ylim(-0.1,0.1)
    plt.tick_params(labelsize = 26)
    plt.grid()
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22)
    cbar.update_ticks()

    plt.show()

def plot_1d_distribution(fName, GridFile):
    df = np.loadtxt(fName)
    grid = np.load(GridFile)
    velocities_z = grid['arr_0']
    
    fig = plt.figure(figsize=(8,6),facecolor='w', edgecolor='k')

    plt.plot(velocities_z, df)
    plt.xlabel(r'$v_z$', fontsize=30)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 26)
    plt.grid()

    plt.show()
    #plt.savefig(r'./Cori/mass25/rescheck/3/'+rf'_f1D_.jpg', bbox_inches='tight')
    #plt.close()

def new_1d(fName, GridFile):
    df = np.loadtxt(fName)
    grid = np.load(GridFile)
    velocities_z = grid['arr_0']
    velocities_y = grid['arr_1']

    # deltavz = velocities_z[1] - velocities_z[0]
    # deltavy = velocities_y[1] - velocities_y[0]

    newdf = np.zeros(240)
    newgrid = np.zeros(240)
    for i in range(240):
        newdf[i] = df[4*i,144+i]
        if i < 72:
            newgrid[i] = - np.sqrt(velocities_y[144+i]**2 + velocities_z[4*i]**2)
        else:
            newgrid[i] = np.sqrt(velocities_y[144+i]**2 + velocities_z[4*i]**2)
    
    plt.plot(newgrid,newdf)
    plt.show()

def fit_1d(fName,GridFile):
    def double_maxwellian(x,A1,B1,C1,A2,B2,C2):
        return maxw(x,A1,B1,C1) + maxw(x,A2,B2,C2)
    
    # def kappa_maxwellian(x,A1,B1,C1,A2,B2,C2):
    #     return kappa(x,10,A1,B1,C1) + maxw(x,A2,B2,C2)

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

    f_e2 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/1700.0_elc_1d.txt')
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

    # f_e2 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function/750.0_elc_1d.txt')
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

def fit_1d_t(fName,GridFile):
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

def elc_main():
    df1 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/750.0_elc_1d.txt')
    df2 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/1000.0_elc_1d.txt')
    df3 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/1200.0_elc_1d.txt')
    df4 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/1400.0_elc_1d.txt')
    df5 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/1600.0_elc_1d.txt')
    df6 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/1800.0_elc_1d.txt')
    df7 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/2400.0_elc_1d.txt')

    grid = np.load(ElcGridPath)
    velocities_z = grid['arr_0']
    
    maxw1 = np.array([maxw(x,1.4,0.0246,0.00) for x in velocities_z])
    maxw2 = np.array([maxw(x,1.15,0.02716,0.0444) for x in velocities_z])
    maxw3 = np.array([maxw(x,1.15,0.02716,0.0444+0.0075) for x in velocities_z])

    #maxw2 = np.array([maxw(x,1.15,0.031,0.042) for x in velocities_z])
    #maxw3 = np.array([maxw(x,1.15,0.033,0.047) for x in velocities_z])

    fig = plt.figure(figsize=(16,10),facecolor='w', edgecolor='k')

    # plt.plot(velocities_z/0.02, df1,label=r'$\omega_{pe}t=750$',linewidth=6)
    # plt.plot(velocities_z/0.02, df2,label=r'$\omega_{pe}t=1000$',linewidth=6)
    # plt.plot(velocities_z/0.02, df3,label=r'$\omega_{pe}t=1200$',linewidth=6)
    # plt.plot(velocities_z/0.02, df4,label=r'$\omega_{pe}t=1400$',linewidth=6)
    # plt.plot(velocities_z/0.02, df5,label=r'$\omega_{pe}t=1600$',linewidth=6)
    plt.plot(velocities_z/0.02, df6,label=r'$\omega_{pe}t=1800$',linewidth=6)
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
    plt.xlim(-0.06/0.02,0.18/0.02)
    #plt.savefig('./Cori/figure_temp/elc_1d.jpg')
    plt.show()

def ion_main():
    df1 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function/350.0_ion_1d.txt')
    df2 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function/400.0_ion_1d.txt')
    df3 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function/450.0_ion_1d.txt')
    df4 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function/500.0_ion_1d.txt')
    df5 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function/550.0_ion_1d.txt')
    df6 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function/1600.0_ion_1d.txt')
    df7 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function/2400.0_ion_1d.txt')

    grid = np.load(IonGridPath)
    velocities_z = grid['arr_0']
    
    # fig = plt.figure(figsize=(8,6),facecolor='w', edgecolor='k')

    plt.figure(figsize=(13,12))
    plt.plot(velocities_z/(0.004), df1,label=r'$\omega_{pe}t=350$',linewidth=6)
    plt.plot(velocities_z/(0.004), df2,label=r'$\omega_{pe}t=400$',linewidth=6)
    plt.plot(velocities_z/(0.004), df3,label=r'$\omega_{pe}t=450$',linewidth=6)
    plt.plot(velocities_z/(0.004), df4,label=r'$\omega_{pe}t=500$',linewidth=6)
    plt.plot(velocities_z/(0.004), df5,label=r'$\omega_{pe}t=550$',linewidth=6)
    #plt.plot(velocities_z/(0.004), df6,label=r'$\omega_{pe}t=1600$',linewidth=6)
    #plt.plot(velocities_z/(0.004), df7,label=r'$\omega_{pe}t=2400$',linewidth=6)

    #plt.vlines(0.004/(0.004),0,26000,color='black',linestyles='dashed',linewidth=6)
    #plt.text(1.5,10000,r'$c_{s0}$',fontsize=42)

    plt.xlabel(r'$v_z/c_{s0}$', fontsize=40)
    plt.ylabel(r'$F_i (v_z)$', fontsize=36)
    plt.grid()
    plt.legend(fontsize=36)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 36)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.gca().yaxis.offsetText.set_fontsize(28)
    plt.xlim(-2.5,2.5)
    plt.ylim(0,28000)
    plt.savefig('./Cori/figure_temp/ion_1d.jpg')
    #plt.show()

if __name__ == '__main__':
    #new_1d('./Cori/mass25/rescheck/4/dist_function/1800.0_elc_2d.txt',ElcGridPath)
    #plot_1d_distribution('./Cori/mass25/rescheck/4/dist_function_save/1700.0_elc_1d.txt', ElcGridPath)
    #ion_main()
    elc_main()

    #popt = fit_1d('./Cori/mass25/rescheck/4/dist_function/650.0_elc_1d.txt',ElcGridPath)
    #fit_1d('./Cori/mass25/rescheck/4/dist_function_save/1700.0_elc_1d.txt',ElcGridPath)
    # fit_1d_t('./Cori/mass25/rescheck/4/dist_function_save/1800.0_elc_1d.txt',ElcGridPath)
    # fit_1d_t('./Cori/mass25/rescheck/4/dist_function/700.0_elc_1d.txt',ElcGridPath)
    # fit_1d_t('./Cori/mass25/rescheck/4/dist_function/750.0_elc_1d.txt',ElcGridPath)
    # fit_1d_t('./Cori/mass25/rescheck/4/dist_function/900.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Cori/mass25/rescheck/4/dist_function/1000.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Cori/mass25/rescheck/4/dist_function/1130.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Cori/mass25/rescheck/4/dist_function/1200.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Cori/mass25/rescheck/4/dist_function/1380.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Cori/mass25/rescheck/4/dist_function/1500.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Cori/mass25/rescheck/4/dist_function/1630.0_elc_1d.txt',ElcGridPath)
    # fit_1d('./Cori/mass25/rescheck/4/dist_function/1750.0_elc_1d.txt',ElcGridPath)
    #plot_2d_distribution('./Cori/mass25/rescheck/4/dist_function/625.0_elc_2d.txt',ElcGridPath)
    # plot_2d_distribution('./Cori/mass25/rescheck/4/dist_function/2400.0_ion_2d.txt',IonGridPath)

   
    #plot_1d_distribution('./Cori/mass25/rescheck/4/dist_function/625.0_ion_1d.txt', IonGridPath)
    #plot_1d_distribution('./Cori/mass25/rescheck/4/dist_function/1800.0_ion_1d.txt', IonGridPath)
    #plot_2d_distribution('./Cori/mass25/rescheck/4/dist_function_save/1000.0_elc_2d.txt',ElcGridPath)
    
    #plot_phase_space('./Cori/mass25/rescheck/4/dist_function_save/600.0_elc_phase.txt', ElcGridPath)
    #plot_real_space('./Cori/mass25/rescheck/4/dist_function/1800.0_elc_space.txt')