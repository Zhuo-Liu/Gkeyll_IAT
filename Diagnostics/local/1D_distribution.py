import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gamma
import matplotlib.ticker as ticker
#matplotlib.use('TkAgg')

ElcGridPath = './massRatio/mass100/E5_H2/dist_function_save/elc_velocities.npz'
IonGridPath = './massRatio/mass100/E5_H2/dist_function_save/ion_velocities.npz'
SpacePath = './massRatio/mass100/E5_H2/dist_function_save/sapce.npz'

def maxw(x,A,B,C):
    return A/np.sqrt(np.pi)/B*np.exp(-(x-C)**2/B**2)

def elc_main():
    df_500 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/500.0_elc_1d.txt')
    df_750 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/750.0_elc_1d.txt')
    df_1000 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1000.0_elc_1d.txt')
    df_1600 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1600.0_elc_1d.txt')
    #df_2200 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/2200.0_elc_1d.txt')
    df_3500 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/3500.0_elc_1d.txt')
    df_4000= np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/4000.0_elc_1d.txt')
    #df7 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/2400.0_elc_1d.txt')

    grid = np.load(ElcGridPath)
    velocities_z = grid['arr_0']
    
    # maxw1 = np.array([maxw(x,1.75,0.032,0.002) for x in velocities_z])
    # maxw2 = np.array([maxw(x,0.84,0.033,0.06) for x in velocities_z])
    # maxw3 = np.array([maxw(x,0.76,0.033,0.08) for x in velocities_z])

    maxw1 = np.array([maxw(x,1.66,0.031,0.000) for x in velocities_z])
    maxw2 = np.array([maxw(x,1.14,0.0256,0.0485) for x in velocities_z])
    #maxw2 = np.array([maxw(x,1.14,0.035,0.0487) for x in velocities_z])
    maxw3 = np.array([maxw(x,1.15,0.035,0.0485) for x in velocities_z])
    maxw4 = np.array([maxw(x,1.3,0.0478,0.0614) for x in velocities_z])

    fig = plt.figure(figsize=(16,10),facecolor='w', edgecolor='k')

    plt.plot(velocities_z/0.02, df_500,label=r'$\omega_{pe}t=500$',linewidth=6)
    plt.plot(velocities_z/0.02, df_750,label=r'$\omega_{pe}t=750$',linewidth=6)
    plt.plot(velocities_z/0.02, df_1000,label=r'$\omega_{pe}t=1000$',linewidth=6)
    plt.plot(velocities_z/0.02, df_1600,label=r'$\omega_{pe}t=1600$',linewidth=6)
    plt.plot(velocities_z/0.02, maxw1,linewidth=5,linestyle='--',label=r'bulk',color='black')
    plt.plot(velocities_z/0.02, maxw2,linewidth=5,linestyle='--',label=r'tail at $\omega_{pe}t=750$',color=u'#ff7f0e')
    #plt.plot(velocities_z/0.02, maxw3,linewidth=5,linestyle='--',label=r'tail at $\omega_{pe}t=1000$',color=u'#2ca02c')
    plt.plot(velocities_z/0.02, maxw4,linewidth=5,linestyle='--',label=r'tail at $\omega_{pe}t=1600$',color=u'#d62728')
    #plt.plot(velocities_z/0.02, df_2200,label=r'$\omega_{pe}t=2200$',linewidth=6)
    plt.plot(velocities_z/0.02, df_3500,label=r'$\omega_{pe}t=3500$',linewidth=6)
    #plt.plot(velocities_z/0.02, df_4000,label=r'$\omega_{pe}t=4000$',linewidth=6)

    resonance = np.arange(-1.4,1.5,0.1)
    plt.fill_between(resonance, 0, 56, facecolor='grey', alpha=0.5)
    #plt.vlines(1.0,0,25,linewidth=3,linestyles='--',color='black')


    plt.xlabel(r'$v_z/v_{Te0}$', fontsize=36)
    plt.ylabel(r'$F_e (v_z)$', fontsize=36)
    #plt.grid()
    plt.legend(fontsize=26)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 36)
    plt.xlim(-0.08/0.02,0.30/0.02)
    plt.ylim(-3,56)
    plt.savefig('./Figures/figures_temp/elc_1d.jpg')
    #plt.show()


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

def ion_main():
    df_500 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/500.0_ion_1d.txt')
    df_750 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/750.0_ion_1d.txt')
    df_1000 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1000.0_ion_1d.txt')
    df_1600 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1600.0_ion_1d.txt')
    df_2500 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/2500.0_ion_1d.txt')
    df_3500 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/3500.0_ion_1d.txt')
    df_4000= np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/4000.0_ion_1d.txt')

    grid = np.load(IonGridPath)
    velocities_z = grid['arr_0']


    plt.figure(figsize=(10.5,10))

    plt.plot(velocities_z/0.002, df_500,label=r'$\omega_{pe}t=500$',linewidth=6)
    plt.plot(velocities_z/0.002, df_750,label=r'$\omega_{pe}t=750$',linewidth=6)
    plt.plot(velocities_z/0.002, df_1000,label=r'$\omega_{pe}t=1000$',linewidth=6)
    plt.plot(velocities_z/0.002, df_1600,label=r'$\omega_{pe}t=1600$',linewidth=6)
    #plt.plot(velocities_z/0.002, df_2500,label=r'$\omega_{pe}t=2500$',linewidth=6)
    plt.plot(velocities_z/0.002, df_3500,label=r'$\omega_{pe}t=3500$',linewidth=6)
    #plt.plot(velocities_z/0.002, df_4000,label=r'$\omega_{pe}t=4000$',linewidth=6)
    #plt.vlines(1.0,0,25,linewidth=3,linestyles='--',color='black')


    plt.xlabel(r'$v_z/c_{s0}$', fontsize=36)
    plt.ylabel(r'$F_i (v_z)$', fontsize=36)
    #plt.grid()
    plt.legend(fontsize=26)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 36)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.gca().yaxis.offsetText.set_fontsize(32)
    plt.xlim(-3,3)
    plt.ylim(0,100000)
    plt.savefig('./Figures/figures_temp/ion_1d.jpeg')
    #plt.show()

def fit_1d(fname,GridFile,A1,A2,B1,B2,C1,C2):
    f_e = np.loadtxt(fname)
    grid = np.load(GridFile)
    v_z = grid['arr_0']


    maxw1 = np.array([maxw(x,A1,B1,C1) for x in v_z])
    maxw2 = np.array([maxw(x,A2,B2,C2) for x in v_z])

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

def justification():

    k_list = np.arange(2,17,1)/50
    
    # gamma_list_1900 = np.array([-25,-98,-205,-352,-541,-775,-1058,-1390,-1773])*1e-6
    # gamma_list_1600 = np.array([134,139,133,112,75,19,-56,-153,-274])*1e-6
    # gamma_list_1000 = np.array([119,143,166,188,209,227,241,252,257])*1e-6
    # gamma_list_750 = np.array([547,647,742,833,917,995,1066,1131,1188])*1e-6

    gamma_list_1900 = np.array([-0.0011,-0.0033,-0.007,-0.012,-0.019,-0.027,-0.037,-0.047,-0.058,-0.07,-0.082,-0.095,-0.108,-0.122,-0.137])
    gamma_list_1600 = np.array([-0.0003,-0.0013,-0.0033,-0.0064,-0.011,-0.016,-0.022,-0.029,-0.037,-0.045,-0.054,-0.063,-0.073,-0.083,-0.094])
    gamma_list_1000 = np.array([0.00096,0.0011,0.00087,0.0,-0.0013,-0.0033,-0.0059,-0.0091,-0.0127,-0.0169,-0.021,-0.026,-0.032,-0.037,-0.043])
    gamma_list_750 = np.array([0.0006,0.00096,0.0012,0.0013,0.00126,0.0011,0.0007,0.0001,-0.00076,-0.002,-0.0034,-0.0051,-0.007,-0.009,-0.0116])

    gamma_list_1000_artifical  = np.array([-0.003,-0.006,-0.01,-0.0154,-0.022,-0.03,-0.039,-0.049,-0.06,-0.07,-0.084,-0.097,-0.11,-0.124,-0.138])

    #gamma_list_1000_artifical  = np.array([-270,-330,-410,-490,-590,-698,-823,-964,-1124])*1e-6

    fig      = plt.figure(figsize=(11.5,9.5))
    ax      = fig.add_axes([0.16, 0.16, 0.75, 0.75])

    ax.plot(k_list,gamma_list_750,label='$\omega_{pe} t = 750$',linewidth=5)
    ax.plot(k_list,gamma_list_1000,label='$\omega_{pe} t = 1000$',linewidth=5)
    ax.plot(k_list,gamma_list_1600,label='$\omega_{pe} t = 1600$',linewidth=5)
    ax.plot(k_list,gamma_list_1900,label='$\omega_{pe} t = 1900$',linewidth=5)
    ax.plot(k_list,gamma_list_1000_artifical,label='$\omega_{pe} t = 1000$ artificial',linewidth=5,color=u'#ff7f0e',linestyle='--')

    ax.hlines(0.0,0.9/50,16/50,linestyles='--',linewidth=7,colors='black')
    ax.hlines(-0.00333,0.9/50,16/50,linestyles=':',linewidth=7,colors='black')
    
    
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

    ax.legend(fontsize=22)
    #ax.legend(fontsize=22,loc='center right',bbox_to_anchor=(1.0, 0.3))

    ax.set_xlim(0.9/50,12/50)
    ax.set_ylim(-0.03,)

    ax.set_xlabel(r'$k_z \lambda_{De}/2\pi$',fontsize=32)
    ax.set_ylabel(r'$\gamma \quad [\omega_{pe}]$',fontsize=32)
    ax.tick_params(labelsize = 26)
    ax.yaxis.get_offset_text().set_fontsize(22)

    #plt.show()
    plt.savefig('./Figures/figures_temp/omega_k_time.jpg')

def compare_dist():
    ElcGridPath_25 = './Cori/mass25/rescheck/4/dist_function_save/elc_velocities.npz'
    ElcGridPath_E1 = './massRatio/mass25/E1/dist_function/elc_velocities.npz'

    df_1200_25 = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/1200.0_elc_1d.txt')
    df_1900 = np.loadtxt('./massRatio/mass100/E5_H2/dist_function_save/1900.0_elc_1d.txt')
    df_3000_25 = np.loadtxt('./massRatio/mass25/E1/dist_function/3200.0_elc_1d.txt')

    grid = np.load(ElcGridPath)
    velocities_z = grid['arr_0']

    grid_25 = np.load(ElcGridPath_25)
    velocities_z_25 = grid_25['arr_0']

    grid_E1 = np.load(ElcGridPath_E1)
    velocities_z_E1 = grid_E1['arr_0']


    fig = plt.figure(figsize=(16,10),facecolor='w', edgecolor='k')

    plt.plot(velocities_z_25[:]/0.02, df_1200_25/2.8,label=r'25, $\omega_{pe}t=1200$',linewidth=6)
    plt.plot(velocities_z[:]/0.02, df_1900/2.8,label=r'100, $\omega_{pe}t=1900$',linewidth=6)
    plt.plot(velocities_z_E1[:]/0.02, df_3000_25/2.8,label=r'E1, $\omega_{pe}t=3000$',linewidth=6)


    plt.xlabel(r'$v_z/v_{Te0}$', fontsize=36)
    plt.ylabel(r'$F_e (v_z)$', fontsize=36)
    plt.grid()
    plt.legend(fontsize=26)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    plt.tick_params(labelsize = 28)
    plt.xlim(-0.10/0.02,0.26/0.02)
    plt.ylim(-0.5,20)
    #plt.savefig('elc_1d.jpg')
    plt.show()

def eaw():
    df = np.loadtxt('./Cori/mass25/rescheck/4/dist_function_save/2000.0_elc_1d.txt')

    grid = np.load('./Cori/mass25/rescheck/4/dist_function_save/elc_velocities.npz')
    velocities_z = grid['arr_0']
    maxw1 = np.array([maxw(x,0.502,0.0251,-0.0027) for x in velocities_z])
    maxw2 = np.array([maxw(x,2.315,0.0511,0.0778) for x in velocities_z])

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.12, 0.16, 0.8, 0.8])

    ax.plot(velocities_z/0.02, df,label=r'$F_e(v_z)$',linewidth=6)
    ax.plot(velocities_z/0.02, maxw1,linewidth=5,linestyle='--',label=r'$F_{e1}(v_z)$')
    ax.plot(velocities_z/0.02, maxw2,linewidth=5,linestyle='--',label=r'$F_{e2}(v_z)$')
    ax.plot(velocities_z/0.02, maxw1+maxw2,linewidth=5,linestyle='--',label=r'$F_{e1} + F_{e2}$')


    ax.set_xlabel(r'$v_z/v_{Te0}$', fontsize=26)
    ax.set_ylabel(r'$F_e (v_z)$', fontsize=26)
    #ax.grid()
    ax.legend(fontsize=26)
    #plt.set_title(r'$<F_e(v_z)>_{z},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
    ax.tick_params(labelsize = 18)
    ax.set_xlim(-0.08/0.02,0.24/0.02)
    #plt.show()
    plt.savefig('./Figures/figures_temp/elc_1d_eaw.jpg')    


if __name__ == '__main__':
    #ion_main()
    #elc_main()

    ######### Fitting 
    #fit_1d_numerical('./massRatio/mass100/E5_H2/dist_function_save/1600.0_elc_1d.txt', ElcGridPath)
    #fit_1d_1600('./massRatio/mass100/E5_H2/dist_function_save/1600.0_elc_1d.txt', ElcGridPath)
    ##fit_1d_1000('./massRatio/mass100/E5_H2/dist_function_save/1000.0_elc_1d.txt', ElcGridPath)
    #fit_1d_numerical('./massRatio/mass100/E5_H2/dist_function_save/1400.0_elc_1d.txt', ElcGridPath)
    
    #fit_1d_numerical('./massRatio/mass100/E5_H2/dist_function_save/1800.0_elc_1d.txt', ElcGridPath)
    # fit_1d('./massRatio/mass100/E5_H2/dist_function_save/1900.0_elc_1d.txt', ElcGridPath, 1.23,1.58,0.0327,0.0565,0.001,0.0646)
    # fit_1d('./massRatio/mass100/E5_H2/dist_function_save/1600.0_elc_1d.txt', ElcGridPath, 1.48,1.32,0.0326,0.0478,0.0,0.0613)
    # fit_1d('./massRatio/mass100/E5_H2/dist_function_save/1000.0_elc_1d.txt', ElcGridPath, 1.66,1.14,0.031,0.035,0.0,0.0487)
    # fit_1d('./massRatio/mass100/E5_H2/dist_function_save/750.0_elc_1d.txt', ElcGridPath, 1.69,1.1,0.0296,0.0255,0.00136,0.0479)

    # compare_dist()

    # ElcGridPath_E1 = './Cori/mass25/rescheck/4/dist_function_save/elc_velocities.npz'
    # fit_1d_numerical('./Cori/mass25/rescheck/4/dist_function_save/2000.0_elc_1d.txt', ElcGridPath_E1)
    #eaw()
    #print(1/5.6)
    justification()
