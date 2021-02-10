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
sys.path.insert(0, '/home/zhuol/bin/gkyl-python/pgkylLiu/')
from shutil import copyfile
import pgkylUtil as pgu
import os
directories = ['.']

fileName   = 'IAT_2x2v'    #.Root name of files to process.
dataDir = '/home/zhuol/work1/zhuol/IAT/linear/2x2v/1d/'
outDir  = '/home/zhuol/work1/zhuol/IAT/linear/2x2v/1d/post/'

outDir     = './post/'               #.Output directory where figures are saved.

fourier_transform = True
auto_loading      = False
#creating the directory for plots if it does not exist yet
pgu.checkMkdir(outDir)

polyOrder  = 2
basisType  = 'ms'
m_ion = 25
vTe0 = 0.02
alpha = 0.00
cSound0 = vTe0/np.sqrt(m_ion)

# time window for growth rate calculation
timeWindow = [[2.9/66.63, 66.4/66.63]]

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

print(nxIntD2.shape)
print(nxIntD2[0]//2)
print(nxIntD2[1]//2)

lz = lx[0]  #get box length along z, needed for Fourier transform
ly = lx[1]  #get box length along y

points_z = np.array(x_elc[0])
points_y = np.array(x_elc[1])

def lineFunc(x,a,b):
  #.Compute the function y = a*x + b.
  return np.add(np.mu0ltiply(x,a),b)

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
        print(Ez.shape) 
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
    print(fieldEnergy.shape)
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

    plt.savefig(outDir+'omegaMeasuredFromFieldEnergy'+figureFileFormat)
    plt.close()

    return intTime, omegaIm[0]

# frame window for frequency calculation
iTw             = [int((nFrames-1)*0.1), int((nFrames-1)*0.3)]
modeOmega, _, _ = measureFrequency(iTw)
times, gamL = calcOmegabNgammaL()
