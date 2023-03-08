import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

####### Load Data ########

# fieldEnergy = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/fieldEnergy.txt')
# time_fieldEnergy = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/fieldEnergy_time.txt')

# fieldEnergy_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/fieldEnergy.txt')
# time_fieldEnergy_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/fieldEnergy_time.txt')

# time_current = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM1i_time.txt')
# current = np.loadtxt('./massRatio/mass100/E5_H2/saved_data/elc_intM1i.txt')*2

# time_current_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/elc_intM1i_time.txt')
# current_1D = np.loadtxt('./massRatio/mass100/1D/saved_data/elc_intM1i.txt')

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

def temp_plot():
    fig      = plt.figure(figsize=(11.0,9.0))
    ax      = fig.add_axes([0.16, 0.16, 0.78, 0.78])

    ax.plot(time_Iontemp_20[:],Elctemp_20[:]/Iontemp_20[:],label='20',linewidth=5)
    ax.plot(time_Iontemp_50[:],Elctemp_50[:]/Iontemp_50[:],label='50',linewidth=5)
    ax.plot(time_Iontemp_100[:],Elctemp_100[:]/Iontemp_100[:],label='100',linewidth=5)
    ax.plot(time_Iontemp_200[:],Elctemp_200[:]/Iontemp_200[:],label='200',linewidth=5)

    ax.set_xlabel(r'$t \quad [\omega_{pe}^{-1}]$',fontsize=32)
    ax.set_ylabel(r'$T_i/T_{i0} $',fontsize=32,color='black')

    ax.set_xlim(0,1500)
    ax.set_ylim(-3,40)

    ax.tick_params(labelsize = 28)
    ax.tick_params(axis='y',colors='black')

    ax.legend(fontsize=36)

    #plt.savefig(r'./Figures/figures_temp/1D/ion_temp.jpg', bbox_inches='tight')
    plt.show()
    plt.clf()

if __name__ == '__main__':
    temp_plot()