#.Make plots from Gkyl data.
#.Manaure Francisquez (base) and Lucio Milanese (updates and extensions).
#.Spring 2019.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import postgkyl as pg
import numpy as np
import adios as ad
import sys
from scipy.optimize import curve_fit
sys.path.insert(0, '/home/zhuol/bin/gkyl-python/pgkylLiu/2x2v/')
#sys.path.insert(0, '/global/u2/z/zliu1997/bin/gkeyl_plot/2x2v/')
from shutil import copyfile
import pgkylUtil as pgu
import os

fileName   = 'IAT_E2'    #.Root name of files to process.
dataDir = '../'
outDir  = './'
outfigDir = './dist_function/'

fourier_transform = True
auto_loading      = False
#creating the directory for plots if it does not exist yet
pgu.checkMkdir(outDir)
pgu.checkMkdir(outfigDir)

polyOrder  = 2
basisType  = 'ms'
m_ion = 25
vTe0 = 0.02
alpha = 0.00
cSound0 = vTe0/np.sqrt(m_ion)

# time window for growth rate calculation
timeWindow = [[900/1600, 1200/1600]]

#=====================================================================#
#=============================Setup===================================#
#=====================================================================#
#.Some RGB colors. These are MATLAB-like.
defaultBlue    = [0, 0.4470, 0.7410]
defaultOrange  = [0.8500, 0.3250, 0.0980]
defaultGreen   = [0.4660, 0.6740, 0.1880]
defaultPurple  = [0.4940, 0.1840, 0.5560]
defaultRed     = [0.6350, 0.0780, 0.1840]
defaultSkyBlue = [0.3010, 0.7450, 0.9330]
grey           = [0.5, 0.5, 0.5]
#.Colors in a single array.
defaultColors = [defaultBlue,defaultOrange,defaultGreen,defaultPurple,defaultRed,defaultSkyBlue,grey,'black']

#.LineStyles in a single array.
lineStyles = ['-','--',':','-.','None','None','None','None']
markers    = ['None','None','None','None','o','d','s','+']

#.Some fontsizes used in plots.
xyLabelFontSize       = 17
titleFontSize         = 17
colorBarLabelFontSize = 17
tickFontSize          = 14
legendFontSize        = 14

figureFileFormat = '.png'
#.Component of the quantity we wish to extract from data file.
#.For field files this specifies the field component (e.g. Ex,
#.Ey, Ez, Bx, By, or Bz) while for Mi1 it specifies the vector
#.component of the momentum density.
compZero = 0

    #..................... NO MORE USER INPUTS BELOW (maybe) ....................#

nFrames = 1+pgu.findLastFrame(dataDir+fileName+'_field_','bp')
fileRoot = dataDir+fileName+'_'

#.Extract grid details from one of the data files for each species
fName_elc = dataDir+fileName+'_elc_0.bp'
fName_ion = dataDir+fileName+'_ion_0.bp'

# getGrid data
x_elc, _, nx, lx, _ = pgu.getGrid(fName_elc,polyOrder,basisType,location='center')
x_ion, _, _, _, _ = pgu.getGrid(fName_ion,polyOrder,basisType,location='center')


#Store needed data from getGrid
nxIntD2 = nx // 2

lz = lx[0]  #get box length along z, needed for Fourier transform
ly = lx[1]  #get box length along y

nz = nx[0]
ny = nx[1]

points_z = np.array(x_elc[0])
points_y = np.array(x_elc[1])

def lineFunc(x,a,b):
  #.Compute the function y = a*x + b.
  return np.add(np.multiply(x,a),b)

def expFunc(x,b,lna):
  #.Compute the function y = a*(e^(b*x)) = e^(b*x + ln(a))
  return np.exp(np.add(np.multiply(b,x),lna))


#=====================================================================#
#=====================Frequency Measurement===========================#
#=====================================================================#

def measureFrequency(frameWindow,makeplot=True):
    #.Compute the mode frequency based on an FFT of the electric field at one point
    #.in the time frame given by frameWindow[0] to frameWindow[1].
    pFramesN = frameWindow[1]-(frameWindow[0]-1)

    EzMid    = np.zeros(pFramesN)
    time     = np.zeros(pFramesN)

    cF = 0
    for nFr in np.arange(frameWindow[0],frameWindow[1]+1):
        #.Extract the time from file.
        time[cF]  = pgu.getTime(fileRoot+'field_'+str(nFr)+'.bp')
    
        #.Electric field in x direction at simulation center.
        fName     = fileRoot+'field_'+str(nFr)+'.bp'    #.Complete file name.
        Ez        = pgu.getInterpData(fName,polyOrder,basisType,comp=0)
        EzMid[cF] = Ez[nxIntD2[0]//2,nxIntD2[1]//2] #Why we are using the electron grid in field???
        EzMid[cF] = (1.0-np.cos(2.0*np.pi*cF/(pFramesN-1)))*EzMid[cF]

        cF = cF+1

    #.Compute the FFT of mid-point electric field in time.
    EzMidw      = np.fft.rfft(EzMid)
    absEzMidwSq = np.power(np.absolute(EzMidw),2)

    omegas = 2.0*np.pi*np.fft.rfftfreq(pFramesN,d=(time[-1]-time[0])/(pFramesN-1))    #.Frequencies.

    modeOmega = omegas[np.argmax(absEzMidwSq)]

    #.Frequency analysis of the electric field at the middle of the domain.
    #.The second entry in plotFFTofE indicates whether to apply a Hann window.
    if makeplot == True:
        print("-> plotFFTofE")

        #.Prepare figure.
        figProp2a = (6,4)
        ax2aPos   = [0.16, 0.16, 0.83, 0.83]
        fig2      = plt.figure(figsize=figProp2a)
        ax2a      = fig2.add_axes(ax2aPos)
        
        hpl2a = ax2a.semilogy(omegas,absEzMidwSq,color=defaultBlue,linestyle='-')
        #ax2a.axis( (time[0],omceOompe*tEnd,np.amin(bzMid),np.amax(bzMid)) )
        ax2a.text( 0.6, 0.7, r'max @ $\omega= $'+'{:10.4e}'.format(modeOmega), transform=ax2a.transAxes)
        ax2a.set_xlabel(r'$\omega/\omega_{pe}$', fontsize=16)
        ax2a.set_ylabel(r'$\left|\mathcal{F}\left[E_{(x=L_x/2)}\right]\right|^2$', fontsize=16)
        
        #if plotFFTofE[1]:
        plt.savefig(outDir+'FourierAmplitudeExMid-Hann_frames'+figureFileFormat)
        
        plt.close()
        
    return modeOmega, omegas, absEzMidwSq


#=====================================================================#
#=====================Growth Rate Measurement=========================#
#=====================================================================#
def calcOmegabNgammaL():
    fileRootIn   = dataDir+fileName+'_'
    #.Field energy.
    fName        = fileRootIn+'fieldEnergy.bp'    #.Complete file name.
    pgData       = pg.GData(fName)    #.Read data with pgkyl.
    fieldEnergy  = pgData.getValues()[:,0]
    intTime      = pgData.getGrid()[0] #.Time of the simulation.
    nFrames      = len(intTime)

    #.Prepare figure showing fit from which we measured w_r and w_I.
    figProp3a = (6,4)
    ax3aPos   = [0.16, 0.16, 0.83, 0.83]
    fig3      = plt.figure(figsize=figProp3a)
    ax3a      = fig3.add_axes(ax3aPos)
    mSize     = 4     #.Size of markers in plot.
    mStride   = 20    #.For plotting every other mStride markers.

    hpl3a = ax3a.semilogy(intTime,fieldEnergy,color='black',linestyle='--')
    ax3a.set_xlabel(r'Time $\omega_{pe} t$', fontsize=16)
    ax3a.set_ylabel(r'$\int dx\thinspace\left|E(x,t)\right|^2$', fontsize=16)

    iFit = 0
    omegaIm = np.zeros(np.size(timeWindow))
    for timeW in timeWindow:
        #.Time index where fit starts and ends.
        iTfit = [int((nFrames-1)*timeW[0]), int((nFrames-1)*timeW[1])]

        #.Locate the local maxima.
        fEmaximaTs = intTime[iTfit[0]:iTfit[1]]
        fEmaxima   = fieldEnergy[iTfit[0]:iTfit[1]]

        #.Space between maxima is twice the period. Compute the angular frequency:
        # omegaRe = np.mean(2.0*pi/(2.0*(fEmaximaTs[1:]-fEmaximaTs[0:-1])))

        #.Fit a line to the natural log of the local maxima.
        poptMaxima, _ = curve_fit(lineFunc, fEmaximaTs, np.log(fEmaxima))
        #.Compute the growth rate:
        omegaIm[iFit] = poptMaxima[0]*0.5
        
        print(" Imaginary frequency, omega_i: ",omegaIm[iFit])

        #.Plot exponential fit to linear-fit of local maxima.
        hpl3b = ax3a.semilogy(intTime[iTfit[0]:iTfit[1]],
                            expFunc(intTime[iTfit[0]:iTfit[1]],*poptMaxima),
                            color=defaultOrange,linestyle='None',marker='o',markersize=4,markevery=20)
        ax3a.text( 0.15+iFit*0.5, 0.75-iFit*0.5, r'$\omega_I= $'+'{:10.4e}'.format(omegaIm[iFit]), transform=ax3a.transAxes)
        iFit = iFit+1

    plt.savefig(outDir+'GrowthRateMeasuredFromFieldEnergy'+figureFileFormat)
    plt.close()

    return intTime, omegaIm[0]


#=====================================================================#
#=====================Current and Resistivity=========================#
#=====================================================================#
def current_vs_electric(frameWindow,E):
    #.in the time frame given by frameWindow[0] to frameWindow[1].
    pFramesN = frameWindow[1]-(frameWindow[0]-1)
    time     = np.zeros(pFramesN)

    eField_boxavg_z = np.zeros(pFramesN)
    J_boxavg_z = np.zeros(pFramesN)
    dJdt = np.zeros(pFramesN)
    E_over_J_rolling = np.zeros(pFramesN)
    nu_eff = np.zeros(pFramesN)

    cF = 0
    for nFr in np.arange(frameWindow[0],frameWindow[1]+1):
        #.Extract the time from file.
        time[cF]  = pgu.getTime(fileRoot+'field_'+str(nFr)+'.bp')
    
        #.Electric field in x direction at simulation center.
        fNameM0_ion = fileRoot+'ion_M0_'+str(nFr)+'.bp'
        fNameM0_elc = fileRoot+'elc_M0_'+str(nFr)+'.bp'
        fNameM1_ion = fileRoot+'ion_M1i_'+str(nFr)+'.bp'
        fNameM1_elc = fileRoot+'elc_M1i_'+str(nFr)+'.bp'

        fName_field = fileRoot+'field_'+str(nFr)+'.bp'    #.Complete file name.

        elcM1_z = np.squeeze(pgu.getInterpData(fNameM1_elc,polyOrder,basisType,comp=0))
        #elcM1_y = np.squeeze(pgu.getInterpData(fNameM1_elc,polyOrder,basisType,comp=1))
        ionM1_z = np.squeeze(pgu.getInterpData(fNameM1_ion,polyOrder,basisType,comp=0))
        #ionM1_y = np.squeeze(pgu.getInterpData(fNameM1_ion,polyOrder,basisType,comp=1))
        Ez      = np.squeeze(pgu.getInterpData(fName_field,polyOrder,basisType,comp=0))
        #Ey      = np.squeeze(pgu.getInterpData(fName_field,polyOrder,basisType,comp=1))

        # elcM0 = np.squeeze(pgu.getInterpData(fNameM0_elc,polyOrder,basisType,comp=0)) # don't have to specify the component here
        # ionM0 = np.squeeze(pgu.getInterpData(fNameM0_ion,polyOrder,basisType,comp=0))

        boxavg_Ez = np.average(Ez)
        #boxavg_Ey = np.average(Ey)

        eField_boxavg_z[cF] = boxavg_Ez
    
        Jz = ionM1_z - elcM1_z
        #Jy = ionM1_y - ionM1_y

        J_boxavg_z[cF] = np.sum(Jz)/(nz*ny)
        dJdt = (J_boxavg_z[cF] - J_boxavg_z[cF-1])/(time[cF]-time[cF-1])

        cF = cF+1

    Navg = 3
    # for n in range(Navg,pFramesN):
    #     E_over_J_rolling[n] = np.sum(eField_boxavg_z[n-Navg:n])/np.sum(J_boxavg_z[n-Navg:n])
    # for n in range(Navg):
    #     E_over_J_rolling[n] =  E_over_J_rolling[Navg]  #bfill the first Navg values

    for n in range(Navg,pFramesN-Navg-1):
        for i in range(0,Navg): 
            nu_eff[n] += 1/Navg * ((E+dJdt[n+i])/J_boxavg_z[n+i])
    for n in range(Navg):
            nu_eff[n] += 1/Navg * ((E+dJdt[n])/J_boxavg_z[n])  #bfill the first Navg values

    fig, axs = plt.subplots(1,3,figsize=(30, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace =.1)
    axs = axs.ravel()

    axs[0].plot(time,eField_boxavg_z)
    axs[0].set_xlabel(r'$t \ [\omega_{pe}^{-1}]$', fontsize=30)
    axs[0].set_ylabel(r'$\langle E_z \rangle$', fontsize=30)
    axs[0].tick_params(labelsize = 26)

    axs[1].plot(time,J_boxavg_z)
    axs[1].set_xlabel(r'$t \ [\omega_{pe}^{-1}]$', fontsize=30)
    axs[1].set_ylabel(r'$\langle J_z \rangle$', fontsize=30)
    axs[1].tick_params(labelsize = 26)

    axs[2].plot(time,nu_eff)
    axs[2].set_xlabel(r'$t \ [\omega_{pe}^{-1}]$', fontsize=30)
    #axs[2].set_ylabel(r'$\langle E_z\rangle /\langle\, J_z\rangle \ [\nu_{\mathrm{eff}}/ \omega_{pe}]$', fontsize=30)
    axs[2].set_ylabel(r'$\nu_{\mathrm{eff}}/ \omega_{pe}$', fontsize=30)
    axs[2].tick_params(labelsize = 26)

    fig.tight_layout()
    plt.savefig(outDir+'ElectrcField_Current'+figureFileFormat)
    plt.close()

def distribution_function_plot(frameWindow):

    pFramesN = frameWindow[1]-(frameWindow[0]-1)

    velocitiesz_elc = np.array(x_elc[2])  #attempt!!
    velocitiesy_elc = np.array(x_elc[3])  #attempt!!
    velocitiesz_ion = np.array(x_ion[2])  #attempt!!
    velocitiesy_ion = np.array(x_ion[3])  #attempt!!

    Vz_elc, Vy_elc = np.meshgrid(velocitiesz_elc,velocitiesy_elc,indexing='ij')
    Vz_ion, Vy_ion = np.meshgrid(velocitiesz_ion,velocitiesy_ion,indexing='ij')
    
    times = np.zeros(pFramesN)

    fName_elc0 = dataDir + fileName+'_elc_0.bp'
    elcd0 = np.squeeze(pgu.getInterpData(fName_elc0,polyOrder,basisType))
    elcd_box_avg_z0 = np.average(elcd0,axis= (0,1,3))

    fName_ion0 = dataDir + fileName+'_ion_0.bp'
    iond0 = np.squeeze(pgu.getInterpData(fName_ion0,polyOrder,basisType))
    iond_box_avg_z0 = np.average(iond0,axis= (0,1,3))

    for nFr in np.arange(frameWindow[0],frameWindow[1]+1):
        fignum = str(nFr).zfill(4)

        fName_elc = dataDir + fileName+'_elc_'+str(nFr)+'.bp'
        fName_ion = dataDir + fileName+'_ion_'+str(nFr)+'.bp'

        hF         = ad.file(fName_elc)
        times[nFr] = hF['time'].read()
        hF.close()
        time = float('%.3g' % times[nFr])

        elcd = np.squeeze(pgu.getInterpData(fName_elc,polyOrder,basisType))
        iond = np.squeeze(pgu.getInterpData(fName_ion,polyOrder,basisType))

        # elcd_box_avg = np.average(elcd, axis = (0,1))
        # iond_box_avg = np.average(iond, axis = (0,1))

        elcd_box_avg_z = np.average(elcd,axis= (0,1,3)) - elcd_box_avg_z0
        iond_box_avg_z = np.average(iond,axis= (0,1,3)) - iond_box_avg_z0

        # fig, axs = plt.subplots(1,2,figsize=(25, 10), facecolor='w', edgecolor='k')
        # fig.subplots_adjust(hspace = .5, wspace =.1)
        # axs = axs.ravel()

        # pos0 = axs[0].pcolormesh(Vz_elc/cSound0, Vy_elc/cSound0, elcd_box_avg)
        # #xs[0].scatter(boxavg_uElc_z, boxavg_uElc_y, s = 60)
        # axs[0].scatter(np.squeeze(Vz_elc[np.where(elcd_box_avg==np.max(elcd_box_avg))]),np.squeeze(Vy_elc[np.where(elcd_box_avg==np.max(elcd_box_avg))]),s = 40, marker = 'x', alpha = 1)
        # axs[0].set_xlabel(r'$v_z/c_s$', fontsize=30)
        # axs[0].set_ylabel(r'$v_y/c_s$', fontsize=30, labelpad=-1)
        # axs[0].set_title(r'$<F_e(v_z,v_y)>_{z,y},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
        # axs[0].tick_params(labelsize = 26)
        # cbar = fig.colorbar(pos0, ax=axs[0])
        # cbar.ax.tick_params(labelsize=22)

        # pos1 = axs[1].pcolormesh(Vz_ion/cSound0, Vy_ion/cSound0, iond_box_avg)
        # axs[1].set_xlabel(r'$v_z/c_s$', fontsize=30)
        # axs[1].set_ylabel(r'$v_y/c_s$', fontsize=30, labelpad=-1)
        # axs[1].set_title(r'$<F_i(v_z,v_y)>_{z,y},$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
        # axs[1].tick_params(labelsize = 26)
        # cbar = fig.colorbar(pos1, ax=axs[1])
        # cbar.ax.tick_params(labelsize=22)

        # fig.tight_layout()
        # plt.savefig(outfigDir+fileName+rf'_f2D_{fignum}.png', bbox_inches='tight')
        # plt.close()

        fig, axs = plt.subplots(1,2,figsize=(25, 10), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace =.1)
        axs = axs.ravel()

        pos3 = axs[0].plot(velocitiesz_elc/cSound0, elcd_box_avg_z)
        axs[0].set_xlabel(r'$v_z/c_s$', fontsize=30)
        axs[0].set_title(r'$<F_e(v_z)>,$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
        axs[0].tick_params(labelsize = 26)

        pos4 = axs[1].plot(velocitiesz_ion/cSound0, iond_box_avg_z)
        axs[1].set_xlabel(r'$v_z/c_s$', fontsize=30)
        axs[1].set_title(r'$<F_i(v_z)>,$' + rf't = {time}'+ r' [$\omega_{pe}^{-1}$]', fontsize=26)
        axs[1].tick_params(labelsize = 26)

        fig.tight_layout()
        plt.savefig(outfigDir+fileName+rf'_f1D_{fignum}.png', bbox_inches='tight')
        plt.close()



# frame window for frequency calculation
iTw_frequency = [int((nFrames-1)*0.1), int((nFrames-1)*0.3)]
iTw_nu = [int((nFrames-1)*0.1), int((nFrames-1)*0.99)]
iTw_dis = [int((nFrames-1)*0.05), int((nFrames-1)*0.99)]

#modeOmega, _, _ = measureFrequency(iTw_frequency)
#times, gamL = calcOmegabNgammaL()
current_vs_electric(iTw_nu,0.00005)

distribution_function_plot(iTw_dis) 