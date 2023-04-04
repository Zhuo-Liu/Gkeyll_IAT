import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gamma
import matplotlib.ticker as ticker
matplotlib.use('TkAgg')

####### Load Data ########

fieldEnergy = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/fieldEnergy.txt')
time_fieldEnergy = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/fieldEnergy_time.txt')

fieldEnergy_20 = np.loadtxt('./tempRatio/20/saved_data/fieldEnergy.txt')
time_fieldEnergy_20 = np.loadtxt('./tempRatio/20/saved_data/fieldEnergy_time.txt')

fieldEnergy_100 = np.loadtxt('./tempRatio/100/saved_data/fieldEnergy.txt')
time_fieldEnergy_100 = np.loadtxt('./tempRatio/100/saved_data/fieldEnergy_time.txt')


time_current = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/elc_intM1i_time.txt')
current = np.loadtxt('./Cori/mass25/rescheck/4/saved_data/elc_intM1i.txt')*2

time_current_20 = np.loadtxt('./tempRatio/20/saved_data/elc_intM1i_time.txt')
current_20 = np.loadtxt('./tempRatio/20/saved_data/elc_intM1i.txt')*2

time_current_100 = np.loadtxt('./tempRatio/100/saved_data/elc_intM1i_time.txt')
current_100 = np.loadtxt('./tempRatio/100/saved_data/elc_intM1i.txt')*2


Iontemp_20 = np.loadtxt('./tempRatio/20/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp_20 = np.loadtxt('./tempRatio/20/saved_data/ion_intM2Thermal_time.txt')
Elctemp_20 = np.loadtxt('./tempRatio/20/saved_data/elc_intM2Thermal.txt')
time_Elctemp_20 = np.loadtxt('./tempRatio/20/saved_data/elc_intM2Thermal_time.txt')

Iontemp_50 = np.loadtxt('./Cori/mass25/rescheck/3/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp_50 = np.loadtxt('./Cori/mass25/rescheck/3/saved_data/ion_intM2Thermal_time.txt')
Elctemp_50 = np.loadtxt('./Cori/mass25/rescheck/3/saved_data/elc_intM2Thermal.txt')
time_Elctemp_50 = np.loadtxt('./Cori/mass25/rescheck/3/saved_data/elc_intM2Thermal_time.txt')

Iontemp_100 = np.loadtxt('./tempRatio/100/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp_100 = np.loadtxt('./tempRatio/100/saved_data/ion_intM2Thermal_time.txt')
Elctemp_100 = np.loadtxt('./tempRatio/100/saved_data/elc_intM2Thermal.txt')
time_Elctemp_100 = np.loadtxt('./tempRatio/100/saved_data/elc_intM2Thermal_time.txt')

Iontemp_200 = np.loadtxt('./tempRatio/200/saved_data/ion_intM2Thermal.txt')*25
time_Iontemp_200 = np.loadtxt('./tempRatio/200/saved_data/ion_intM2Thermal_time.txt')
Elctemp_200 = np.loadtxt('./tempRatio/200/saved_data/elc_intM2Thermal.txt')
time_Elctemp_200 = np.loadtxt('./tempRatio/200/saved_data/elc_intM2Thermal_time.txt')

dJdt = np.zeros(np.size(current)-1)
nu_eff = np.zeros(np.size(current)-1)
for i in range(np.size(current)-1):
    dJdt[i] = (current[i+1] - current[i]) / (time_current[i+1] - time_current[i])
for i in range(np.size(current)-1):
    nu_eff[i] = (0.00005 - dJdt[i]) / current[i]

dJdt_20 = np.zeros(np.size(current_20)-1)
nu_eff_20 = np.zeros(np.size(current_20)-1)
for i in range(np.size(current_20)-1):
    dJdt_20[i] = (current_20[i+1] - current_20[i]) / (time_current_20[i+1] - time_current_20[i])
for i in range(np.size(current_20)-1):
    nu_eff_20[i] = (0.00005 - dJdt_20[i]) / current_20[i]

dJdt_100 = np.zeros(np.size(current_100)-1)
nu_eff_100 = np.zeros(np.size(current_100)-1)
for i in range(np.size(current_100)-1):
    dJdt_100[i] = (current_100[i+1] - current_100[i]) / (time_current_100[i+1] - time_current_100[i])
for i in range(np.size(current_100)-1):
    nu_eff_100[i] = (0.00005 - dJdt_100[i]) / current_100[i]

def maxw(x,A,B,C):
    return A/np.sqrt(np.pi)/B*np.exp(-(x-C)**2/B**2)

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

def temp_plot():
    fig      = plt.figure(figsize=(11.0,9.0))
    ax      = fig.add_axes([0.16, 0.16, 0.78, 0.78])

    # ax.plot(time_Iontemp_20[:],Elctemp_20[:]/Iontemp_20[:],label='20',linewidth=5)
    # ax.plot(time_Iontemp_50[:],Elctemp_50[:]/Iontemp_50[:],label='50',linewidth=5)
    # ax.plot(time_Iontemp_100[:],Elctemp_100[:]/Iontemp_100[:],label='100',linewidth=5)
    # ax.plot(time_Iontemp_200[:],Elctemp_200[:]/Iontemp_200[:],label='200',linewidth=5)

    ax.plot(time_Iontemp_20[:],Iontemp_20[:]/Iontemp_20[0],label='20',linewidth=5)
    ax.plot(time_Iontemp_50[:],Iontemp_50[:]/Iontemp_50[0],label='50',linewidth=5)
    ax.plot(time_Iontemp_100[:],Iontemp_100[:]/Iontemp_100[0],label='100',linewidth=5)
    ax.plot(time_Iontemp_200[:],Iontemp_200[:]/Iontemp_200[0],label='200',linewidth=5)

    ax.set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    ax.set_ylabel(r'$T_e/T_{i} $',fontsize=32,color='black')

    ax.set_xlim(0,1500)
    ax.set_ylim(-3,60)
    ax.tick_params(labelsize = 28)
    ax.tick_params(axis='y',colors='black')

    ax.legend(fontsize=36)

    #plt.savefig(r'./Figures/figures_temp/1D/ion_temp.jpg', bbox_inches='tight')
    plt.show()
    plt.clf()

def compare():
    fig, axs = plt.subplots(2, 2, figsize=(16,12))

    axs[0,0].set_title("perturbed electric field energy",fontsize=28)
    axs[0,0].plot(time_fieldEnergy_20[:],fieldEnergy_20[:],label='20',linewidth=5)
    axs[0,0].plot(time_fieldEnergy[:],fieldEnergy[:],label='50',linewidth=5)
    axs[0,0].plot(time_fieldEnergy_100[:],fieldEnergy_100[:],label='100',linewidth=5)
    axs[0,0].set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    axs[0,0].set_ylabel(r'$\int dydz |\delta E_z|^2 + |\delta E_y|^2$',fontsize=28)
    axs[0,0].set_yscale('log')
    axs[0,0].set_xlim(0,2000)
    axs[0,0].tick_params(labelsize = 28)
    axs[0,0].legend(fontsize=24)


    axs[0,1].set_title("ion temperature",fontsize=28)
    axs[0,1].plot(time_Iontemp_20[:],Iontemp_20[:]/Elctemp_20[0],label='20',linewidth=5)
    axs[0,1].plot(time_Iontemp_50[:],Iontemp_50[:]/Elctemp_50[0],label='50',linewidth=5)
    axs[0,1].plot(time_Iontemp_100[:],Iontemp_100[:]/Elctemp_100[0],label='100',linewidth=5)
    axs[0,1].set_xlabel(r'$t [omega_{pe}^{-1}]$',fontsize=32)
    axs[0,1].set_ylabel(r'$T_i/T_{i0}$',fontsize=32)
    axs[0,1].set_xlim(0,2000)
    #axs[0,1].set_ylim(-3,40)
    axs[0,1].tick_params(labelsize = 28)
    axs[0,1].legend(fontsize=24)

    axs[1,0].set_title("current",fontsize=28)
    axs[1,0].plot(time_current_20[:],current_20/0.02,label='20',linewidth=5)
    axs[1,0].plot(time_current[:],current/0.02,label='50',linewidth=5)
    axs[1,0].plot(time_current_100[:],current_100/0.02,label='100',linewidth=5)
    axs[1,0].set_xlabel(r'$t [\omega_{pe}^{-1}]$',fontsize=32)
    axs[1,0].set_ylabel(r'$<J_z> [en_0 v_{te}]$',fontsize=32)
    axs[1,0].set_xlim(0,2000)
    axs[1,0].tick_params(labelsize = 28)
    axs[1,0].legend(fontsize=24)

    axs[1,1].set_title("effective collision frequency",fontsize=28)
    axs[1,1].plot(time_current_20[11:],nu_eff_20[10:],label='20',linewidth=5)
    axs[1,1].plot(time_current[11:],nu_eff[10:],label='50',linewidth=5)
    axs[1,1].plot(time_current_100[11:],nu_eff_100[10:],label='100',linewidth=5)
    axs[1,1].set_xlabel(r'$t [\omega_{pe}^{-1}]$',fontsize=32)
    axs[1,1].set_ylabel(r'$\nu_{eff} [\omega_{pe}^{-1}]$',fontsize=32)
    axs[1,1].set_xlim(0,2000)
    axs[1,1].tick_params(labelsize = 28)
    axs[1,1].legend(fontsize=24)

    plt.tight_layout()
    plt.show()
    #plt.savefig(r'./Figures/figures_temp/1D/field.pdf', bbox_inches='tight')    

def distribution():
    df_20_1 = np.loadtxt('./tempRatio/20/dist_function/600.0_elc_1d.txt')
    df_50_1 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/550.0_elc_1d.txt')

    # df_20_2 = np.loadtxt('./tempRatio/20/dist_function/650.0_elc_1d.txt')
    # df_50_2 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/600.0_elc_1d.txt')

    df_20_750 = np.loadtxt('./tempRatio/20/dist_function/750.0_elc_1d.txt')
    df_50_750 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/700.0_elc_1d.txt')

    df_20_1000 = np.loadtxt('./tempRatio/20/dist_function/1000.0_elc_1d.txt')
    df_50_1000 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/950.0_elc_1d.txt')

    df_20_1250 = np.loadtxt('./tempRatio/20/dist_function/1250.0_elc_1d.txt')
    df_50_1250 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/1200.0_elc_1d.txt')

    #df_50_1800 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/1650.0_elc_1d.txt')

    ElcGridPath_20 = './tempRatio/20/dist_function/elc_velocities.npz'
    ElcGridPath_50 = './Cori/mass25/rescheck/4/dist_function_save/elc_velocities.npz'
    grid_20 = np.load(ElcGridPath_20)
    grid_50 = np.load(ElcGridPath_50)
    velocities_z_20 = grid_20['arr_0']
    velocities_z_50 = grid_50['arr_0']
    

    fig = plt.figure(figsize=(16,10),facecolor='w', edgecolor='k')

    plt.plot(velocities_z_20/0.02, df_20_1,label=r'$20,\omega_{pe}t=600$',linewidth=6,linestyle='--',color='red')
    plt.plot(velocities_z_50/0.02, df_50_1,label=r'$50,\omega_{pe}t=550$',linewidth=6,linestyle='-',color='red')

    # plt.plot(velocities_z_20/0.02, df_20_750,label=r'$20,\omega_{pe}t=750$',linewidth=6,linestyle='--',color='green')
    # plt.plot(velocities_z_50/0.02, df_50_750,label=r'$50,\omega_{pe}t=700$',linewidth=6,linestyle='-',color='green')

    plt.plot(velocities_z_20/0.02, df_20_1000,label=r'$20,\omega_{pe}t=1000$',linewidth=6,linestyle='--',color='blue')
    plt.plot(velocities_z_50/0.02, df_50_1000,label=r'$50,\omega_{pe}t=950$',linewidth=6,linestyle='-',color='blue')

    plt.plot(velocities_z_20/0.02, df_20_1250,label=r'$20,\omega_{pe}t=1250$',linewidth=6,linestyle='--',color='orange')
    plt.plot(velocities_z_50/0.02, df_50_1250,label=r'$50,\omega_{pe}t=1200$',linewidth=6,linestyle='-',color='orange')


    resonance = np.arange(-1.2,1.4,0.1)

    plt.xlabel(r'$v_z/v_{Te0}$', fontsize=36)
    plt.ylabel(r'$F_e (v_z)$', fontsize=36)
    plt.grid()
    plt.legend(fontsize=26)
    plt.tick_params(labelsize = 28)
    plt.xlim(-0.08/0.02,0.16/0.02)
    plt.ylim(-3,56)
    plt.show()

if __name__ == '__main__':
    temp_plot()
    #compare()
    #distribution()

    # Finding the fitting parameter for 50, 25 
    # ElcGridPath_50 = './Cori/mass25/rescheck/4/dist_function_save/elc_velocities.npz'
    # IonGridPath_50 = './Cori/mass25/rescheck/4/dist_function_save/ion_velocities.npz'

    # f_700 = './Cori/mass25/rescheck/4/dist_function_save/1100.0_elc_1d.txt'
    # fit_1d_numerical(f_700, ElcGridPath_50)
