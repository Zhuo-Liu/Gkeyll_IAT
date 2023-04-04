import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

####### Load Data ########

fieldEnergy = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/fieldEnergy.txt')
time_fieldEnergy = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/fieldEnergy_time.txt')

fieldEnergy_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/fieldEnergy.txt')
time_fieldEnergy_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/fieldEnergy_time.txt')

time_current = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM1i_time.txt')
current = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM1i.txt')*2

time_current_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/elc_intM1i_time.txt')
current_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/elc_intM1i.txt')

Iontemp = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/ion_intM2Thermal.txt')*100
time_Iontemp = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/ion_intM2Thermal_time.txt')
Elctemp = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM2Thermal.txt')
time_Elctemp = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM2Thermal_time.txt')

Iontemp_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/ion_intM2Thermal.txt')*100
time_Iontemp_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/ion_intM2Thermal_time.txt')
Elctemp_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/elc_intM2Thermal.txt')
time_Elctemp_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/elc_intM2Thermal_time.txt')

dJdt = np.zeros(np.size(current)-1)
nu_eff = np.zeros(np.size(current)-1)
for i in range(np.size(current)-1):
    dJdt[i] = (current[i+1] - current[i]) / (time_current[i+1] - time_current[i])
for i in range(np.size(current)-1):
    nu_eff[i] = (0.00005 - dJdt[i]) / current[i]

dJdt_1D = np.zeros(np.size(current_1D)-1)
nu_eff_1D = np.zeros(np.size(current_1D)-1)
for i in range(np.size(current_1D)-1):
    dJdt_1D[i] = (current_1D[i+1] - current_1D[i]) / (time_current_1D[i+1] - time_current_1D[i])
for i in range(np.size(current_1D)-1):
    nu_eff_1D[i] = (0.00005 - dJdt_1D[i]) / current_1D[i]

def current_plot():
    fig      = plt.figure(figsize=(11.0,9.0))
    ax      = fig.add_axes([0.16, 0.16, 0.78, 0.78])

    ax.plot(time_current[:],current/0.02,label='2D',linewidth=5)
    ax.plot(time_current_1D[:],current_1D/0.02,label='1D',linewidth=5)

    ax.set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    ax.set_ylabel(r'$<J_z> [en_0 v_{Te0}]$',fontsize=32,color='black')

    ax.set_xlim(0,4500)
    #ax.set_ylim(0,30)

    ax.tick_params(labelsize = 28)
    ax.tick_params(axis='y',colors='black')

    ax.legend(fontsize=36)

    plt.savefig(r'./Figures/figures_temp/1D/current.jpg', bbox_inches='tight')
    #plt.show()
    plt.clf()

def resistivity_plot():
    fig      = plt.figure(figsize=(11.0,9.0))
    ax      = fig.add_axes([0.16, 0.16, 0.78, 0.78])

    ax.plot(time_current[11:],nu_eff[10:],label='2D',linewidth=5)
    ax.plot(time_current_1D[11:],nu_eff_1D[10:],label='1D',linewidth=5)

    ax.set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    ax.set_ylabel(r'$\nu_{eff} [\omega_{pe}^{-1}]$',fontsize=32,color='black')

    ax.set_xlim(0,4000)

    ax.tick_params(labelsize = 28)
    ax.tick_params(axis='y',colors='black')

    ax.legend(fontsize=36)

    plt.savefig(r'./Figures/figures_temp/1D/nueff.jpg', bbox_inches='tight')
    #plt.show()
    plt.clf()

def fieldenergy_plot():
    fig      = plt.figure(figsize=(11.0,9.0))
    ax      = fig.add_axes([0.16, 0.16, 0.78, 0.78])

    ax.plot(time_fieldEnergy[:],fieldEnergy[:],label='2D',linewidth=5)
    ax.plot(time_fieldEnergy_1D[:],fieldEnergy_1D[:],label='1D',linewidth=5)

    ax.set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    ax.set_ylabel(r'$\int |\delta E_z|^2 dz $',fontsize=26,color='black')

    ax.set_xlim(0,4500)
    ax.set_yscale('log')

    ax.tick_params(labelsize = 28)
    ax.tick_params(axis='y',colors='black')

    ax.legend(fontsize=36)

    plt.savefig(r'./Figures/figures_temp/1D/field_energy.jpg', bbox_inches='tight')
    #plt.show()
    plt.clf()

def temp_plot():
    fig      = plt.figure(figsize=(11.0,9.0))
    ax      = fig.add_axes([0.16, 0.16, 0.78, 0.78])

    ax.plot(time_Iontemp[:],Iontemp[:]/Iontemp[0],label='2D',linewidth=5)
    ax.plot(time_Iontemp_1D[:],Iontemp_1D[:]/Iontemp_1D[0],label='1D',linewidth=5)

    ax.set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    ax.set_ylabel(r'$T_i/T_{i0} $',fontsize=32,color='black')

    ax.set_xlim(0,4000)
    ax.set_ylim(-3,40)

    ax.tick_params(labelsize = 28)
    ax.tick_params(axis='y',colors='black')

    ax.legend(fontsize=36)

    plt.savefig(r'./Figures/figures_temp/1D/ion_temp.jpg', bbox_inches='tight')
    #plt.show()
    plt.clf()

def plot_all():
    fig, axs = plt.subplots(2, 2, figsize=(16,12))

    axs[0,0].set_title("perturbed electric field energy",fontsize=28)
    axs[0,0].plot(time_fieldEnergy[:],fieldEnergy[:],label='2D',linewidth=5)
    axs[0,0].plot(time_fieldEnergy_1D[:],fieldEnergy_1D[:],label='1D',linewidth=5)
    axs[0,0].set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    axs[0,0].set_ylabel(r'$\int dydz |\delta E_z|^2 + |\delta E_y|^2$',fontsize=28)
    axs[0,0].set_yscale('log')
    axs[0,0].set_xlim(0,4000)
    axs[0,0].tick_params(labelsize = 28)
    axs[0,0].legend(fontsize=24)


    axs[0,1].set_title("ion temperature",fontsize=28)
    axs[0,1].plot(time_Iontemp[:],Iontemp[:]/Iontemp[0],label='2D',linewidth=5)
    axs[0,1].plot(time_Iontemp_1D[:],Iontemp_1D[:]/Iontemp_1D[0],label='1D',linewidth=5)
    axs[0,1].set_xlabel(r'$t [\omega_{pe}^{-1}]$',fontsize=32)
    axs[0,1].set_ylabel(r'$T_i/T_{i0}$',fontsize=32)
    axs[0,1].set_xlim(0,4000)
    axs[0,1].set_ylim(-3,40)
    axs[0,1].tick_params(labelsize = 28)
    axs[0,1].legend(fontsize=24)

    axs[1,0].set_title("current",fontsize=28)
    axs[1,0].plot(time_current[:],current/0.02,label='2D',linewidth=5)
    axs[1,0].plot(time_current_1D[:],current_1D/0.02,label='1D',linewidth=5)
    axs[1,0].set_xlabel(r'$t [\omega_{pe}^{-1}]$',fontsize=32)
    axs[1,0].set_ylabel(r'$<J_z> [en_0 v_{te}]$',fontsize=32)
    axs[1,0].set_xlim(0,4000)
    axs[1,0].tick_params(labelsize = 28)
    axs[1,0].legend(fontsize=24)

    axs[1,1].set_title("effective collision frequency",fontsize=28)
    axs[1,1].plot(time_current[11:],nu_eff[10:],label='2D',linewidth=5)
    axs[1,1].plot(time_current_1D[11:],nu_eff_1D[10:],label='1D',linewidth=5)
    axs[1,1].set_xlabel(r'$t [\omega_{pe}^{-1}]$',fontsize=32)
    axs[1,1].set_ylabel(r'$\nu_{eff} [\omega_{pe}^{-1}]$',fontsize=32)
    axs[1,1].set_xlim(0,4000)
    axs[1,1].tick_params(labelsize = 28)
    axs[1,1].legend(fontsize=24)

    plt.tight_layout()
    plt.savefig(r'./Figures/figures_temp/1D/field.pdf', bbox_inches='tight')

def distribution():
    ElcGridPath = './massRatio/mass100/E5_H2/dist_function/elc_velocities.npz'
    ElcGridPath_1D = './massRatio/mass100/1D/dist_function/elc_velocities.npz'

    df_1000_1d = np.loadtxt('./massRatio/mass100/1D/dist_function/1000.0_elc_1d.txt')
    df_1900_1d = np.loadtxt('./massRatio/mass100/1D/dist_function/1900.0_elc_1d.txt')
    df_1000 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1000.0_elc_1d.txt')
    df_1900 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1900.0_elc_1d.txt')

    grid = np.load(ElcGridPath)
    velocities_z = grid['arr_0']

    grid_1D = np.load(ElcGridPath_1D)
    velocities_z_1D = grid_1D['arr_0']


    fig = plt.figure(figsize=(16,10),facecolor='w', edgecolor='k')

    plt.plot(velocities_z_1D[1:]/0.02, df_1000_1d,label=r'$\omega_{pe}t=1000$ 1D',linewidth=6,color = u'#1f77b4',linestyle='--')
    plt.plot(velocities_z[:]/0.02, df_1000/2.8,label=r'$\omega_{pe}t=1000$',linewidth=6,color = u'#1f77b4')
    plt.plot(velocities_z_1D[1:]/0.02, df_1900_1d,label=r'$\omega_{pe}t=1900$ 1D',linewidth=6,color = u'#ff7f0e',linestyle='--')
    plt.plot(velocities_z[:]/0.02, df_1900/2.8,label=r'$\omega_{pe}t=1900$',linewidth=6, color = u'#ff7f0e')
    #plt.plot(velocities_z/0.02, df_700,label=r'$\omega_{pe}t=700$',linewidth=6)


    plt.xlabel(r'$v_z/v_{Te0}$', fontsize=36)
    plt.ylabel(r'$F_e (v_z)$', fontsize=36)
    plt.grid()
    plt.legend(fontsize=26)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 28)
    plt.xlim(-0.10/0.02,0.26/0.02)
    plt.ylim(-0.5,14)
    plt.savefig('elc_1d.jpg')
    plt.show()

if __name__ == '__main__':
    #current_plot()
    # resistivity_plot()
    # fieldenergy_plot()
    # temp_plot()
    # plot_all()

    distribution()