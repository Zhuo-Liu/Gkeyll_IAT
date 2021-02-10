from pylab import *
import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
sys.path.insert(0, '/home/zhuol/bin/gkyl-python/pgkylLiu/')
import pgkylUtil as pgu
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import matplotlib.transforms as transforms
import h5py

dataDir = '/home/zhuol/work1/zhuol/IAT/linear/1x1v/lucio1/'
outDir  = '/home/zhuol/work1/zhuol/IAT/linear/1x1v/lucio1/post/'

fileName = 'IAT_1x1v'    #.Root name of files to process.

polyOrder = 2
basisType = 'ms'

#species = 'bump'    #.Species quantities to plot, 'elc' or 'bump'.

qElc = 1.0       #.Normalized absolute electron charge.
mElc = 1.0       #.Normalized electron mass.

# kNumber = 0.84*(4.8/4.4)*0.3    #.Normalized wavenumber of the perturbation.
# omegape = 1.12
# vPhase  = omegape/kNumber  #.Phase speed of the mode.

outFigureFile    = True    #.Output a figure file?.
figureFileFormat = '.png'    #.Can be .png, .pdf, .ps, .eps, .svg.

epsilon0 = 1.0
elcMass  = 1.0
#ionMass  = 1836.2
ionMass = 25.0

#.Time window in which to perform some analysis (given as a percentage of the whole time).
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

#.If the output directory doesn't exist, create it.
pgu.checkMkdir(outDir)

if pgu.checkFile(dataDir+fileName+'_elc_0.bp'):
    xIntElc, dimIntElc, nxIntElc, lxIntElc, dxIntElc = pgu.getGrid(dataDir+fileName+'_elc_0.bp',polyOrder,basisType,location='center')
    nxIntD2 = nxIntElc//2

if pgu.checkFile(dataDir+fileName+'_ion_0.bp'):
    xIntIon, dimIntIon, nxIntIon, lxIntIon, dxIntIon = pgu.getGrid(dataDir+fileName+'_ion_0.bp',polyOrder,basisType,location='center')


def lineFunc(x,a,b):
  #.Compute the function y = a*x + b.
  return np.add(np.multiply(x,a),b)
def expFunc(x,b,lna):
  #.Compute the function y = a*(e^(b*x)) = e^(b*x + ln(a))
  return np.exp(np.add(np.multiply(b,x),lna))

nFrames = 1+pgu.findLastFrame(dataDir+fileName+'_field_','bp')
fileRoot = dataDir+fileName+'_'

#=====================================================================#
#=====================Frequency Measurement===========================#
#=====================================================================#

def measureFrequency(frameWindow,makeplot=True):
    #.Compute the mode frequency based on an FFT of the electric field at one point
    #.in the time frame given by frameWindow[0] to frameWindow[1].
    pFramesN = frameWindow[1]-(frameWindow[0]-1)

    ExMid    = np.zeros(pFramesN)
    time     = np.zeros(pFramesN)

    cF = 0
    for nFr in np.arange(frameWindow[0],frameWindow[1]+1):
        #.Extract the time from file.
        time[cF]  = pgu.getTime(fileRoot+'field_'+str(nFr)+'.bp')
    
        #.Electric field in x direction at simulation center.
        fName     = fileRoot+'field_'+str(nFr)+'.bp'    #.Complete file name.
        Ex        = pgu.getInterpData(fName,polyOrder,basisType,comp=0) 
        ExMid[cF] = Ex[nxIntD2[0]//2] #Why we are using the electron grid in field???

        #if plotFFTofE[1]:
            #.Apply a Hann window to mimic periodicity.
        ExMid[cF] = (1.0-np.cos(2.0*np.pi*cF/(pFramesN-1)))*ExMid[cF]

        cF = cF+1

    #.Compute the FFT of mid-point electric field in time.
    ExMidw      = np.fft.rfft(ExMid)
    absExMidwSq = np.power(np.absolute(ExMidw),2)

    omegas = 2.0*np.pi*np.fft.rfftfreq(pFramesN,d=(time[-1]-time[0])/(pFramesN-1))    #.Frequencies.

    modeOmega = omegas[np.argmax(absExMidwSq)]

    #.Frequency analysis of the electric field at the middle of the domain.
    #.The second entry in plotFFTofE indicates whether to apply a Hann window.
    if makeplot == True:
        print("-> plotFFTofE")

        iTw = [int((nFrames-1)*timeWindow[0][0]), int((nFrames-1)*timeWindow[0][1])]

        #modeOmega, omegas, absExMidwSq = measureFrequency(iTw)  

        #.Prepare figure.
        figProp2a = (6,4)
        ax2aPos   = [0.16, 0.16, 0.83, 0.83]
        fig2      = plt.figure(figsize=figProp2a)
        ax2a      = fig2.add_axes(ax2aPos)
        
        hpl2a = ax2a.semilogy(omegas,absExMidwSq,color=defaultBlue,linestyle='-')
        #ax2a.axis( (time[0],omceOompe*tEnd,np.amin(bzMid),np.amax(bzMid)) )
        ax2a.text( 0.6, 0.7, r'max @ $\omega= $'+'{:10.4e}'.format(modeOmega), transform=ax2a.transAxes)
        ax2a.set_xlabel(r'$\omega/\omega_{pe}$', fontsize=16)
        ax2a.set_ylabel(r'$\left|\mathcal{F}\left[E_{(x=L_x/2)}\right]\right|^2$', fontsize=16)
        
        #if plotFFTofE[1]:
        plt.savefig(outDir+'FourierAmplitudeExMid-Hann_frames'+str(iTw[0])+'-'+str(iTw[1])+figureFileFormat)
        #else:
        #plt.savefig(outDir+'FourierAmplitudeExMid_frames'+str(iTw[0])+'-'+str(iTw[1])+figureFileFormat)
        plt.close()
        
    return modeOmega, omegas, absExMidwSq
#=====================================================================#
#=====================Growth Rate Measurement=========================#
#=====================================================================#
def calcOmegabNgammaL(dataDirIn, outDirIn):
    fileRootIn   = dataDirIn+fileName+'_'
    #.Field energy.
    fName        = fileRootIn+'fieldEnergy.bp'    #.Complete file name.
    pgData       = pg.GData(fName)    #.Read data with pgkyl.
    fieldEnergy  = pgData.getValues()[:,0]
    intTime      = pgData.getGrid()[0] #.Time of the simulation.
    nFrames      = len(intTime)

    # pgu.checkMkdir(outDirIn)    #.If the output directory doesn't exist, create it.
    # # Save the data in HDF5 file.
    # postDir  = outDirIn+'data/'
    # pgu.checkMkdir(postDir)
    # avH5file = h5py.File(postDir+'omegabDgamma0_vs_t.h5', "w")
    # avH5file.create_dataset('time', (np.size(intTime),), dtype='f8', data=intTime)

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
        #fEmaximaTs, fEmaxima = pgu.locatePeaks(intTime,fieldEnergy,[intTime[iTfit[0]],intTime[iTfit[1]]],0.0)
        #if np.size(fEmaximaTs) < 2:
        fEmaximaTs = intTime[iTfit[0]:iTfit[1]]
        fEmaxima   = fieldEnergy[iTfit[0]:iTfit[1]]

        #.Space between maxima is twice the period. Compute the angular frequency:
        # omegaRe = np.mean(2.0*pi/(2.0*(fEmaximaTs[1:]-fEmaximaTs[0:-1])))

        #.Fit a line to the natural log of the local maxima.
        poptMaxima, _ = curve_fit(lineFunc, fEmaximaTs, np.log(fEmaxima))
        #.Compute the growth rate:
        omegaIm[iFit] = poptMaxima[0]*0.5
        
        #print(" Real frequency, omega_r:      ",omegaRe)
        print(" Imaginary frequency, omega_i: ",omegaIm[iFit])

        #.Save this measurement to HDF5 file.
        #avH5file.create_dataset('gamma'+str(iFit), (1,), dtype='f8', data=omegaIm[iFit])

        #.Plot exponential fit to linear-fit of local maxima.
        hpl3b = ax3a.semilogy(intTime[iTfit[0]:iTfit[1]],
                            expFunc(intTime[iTfit[0]:iTfit[1]],*poptMaxima),
                            color=defaultOrange,linestyle='None',marker='o',markersize=4,markevery=20)
    #    ax3a.axis( (intTime[0],intTime[iTfit[1]]*1.1,fieldEnergy[0]*0.01,np.amax(fieldEnergy)) )
    #    ax3a.axis( (intTime[0],intTime[iTfit[1]]*1.1,1e-9,3e-9) )
    #    ax3a.text( 0.6, 0.4, r'$\omega_r= $'+'{:10.4e}'.format(omegaRe), transform=ax3a.transAxes)
    #    ax3a.text( 0.6, 0.35, r'$\omega_I= $'+'{:10.4e}'.format(omegaIm), transform=ax3a.transAxes)
        ax3a.text( 0.15+iFit*0.5, 0.8-iFit*0.5, r'$\omega_r= $'+'{:10.4e}'.format(omegaRe), transform=ax3a.transAxes)
        ax3a.text( 0.15+iFit*0.5, 0.75-iFit*0.5, r'$\omega_I= $'+'{:10.4e}'.format(omegaIm[iFit]), transform=ax3a.transAxes)
        iFit = iFit+1

    plt.savefig(outDirIn+'omegaMeasuredFromFieldEnergy'+figureFileFormat)
    plt.close()


    # wb = np.sqrt(kNumber*np.sqrt((2.0/lxInt[0])*fieldEnergy))  #.Normalized bounce frequency.

    # #.Save the bounce frequency to file and close the file.
    # avH5file.create_dataset('omegab', (np.size(intTime),), dtype='f8', data=wb)
    # avH5file.close()

    return intTime, omegaIm[0]

#=====================================================================#
#===================Distribution Function Plot========================#
#=====================================================================#
def plotDistFunc():
      #.Create a new directory for these frames.
    outDirdist = outDir + '/distFslice/'
    pgu.checkMkdir(outDirdist)  #.If the output directory doesn't exist, create it.

    distFIon  = np.zeros(nxIntIon[1])
    distFElc  = np.zeros(nxIntElc[1])

    nFrames    = 1+pgu.findLastFrame(dataDir+fileName+'_bump_','bp')
    iFrame     = 0
    fFrame     = nFrames
    frameCount = 0

    fName     = fileRoot+'ion_'+str(0)+'.bp'    #.Complete file name.
    fDataIon = pgu.getInterpData(fName,polyOrder,basisType,comp=0)
    distFIon = fDataIon[0,:]
    #.Time of this frame.
    cTime = pgu.getTime(fName)

    fName     = fileRoot+'elc_'+str(0)+'.bp'    #.Complete file name.
    fDataElc = pgu.getInterpData(fName,polyOrder,basisType,comp=0)
    distFElc = fDataElc[0,:]

    #.Prepare figure.
    figProp8a = (7.6,4.2)
    ax8aPos   = [0.13, 0.14, 0.86, 0.76]
    fig8      = plt.figure(figsize=figProp8a)
    ax8a      = fig8.add_axes(ax8aPos)
    
    hpl8a = ax8a.plot(xIntIon[1],distFIon,color=defaultBlue,linestyle='-')
    hpl8b = ax8a.plot(xIntElc[1],distFElc,color=defaultOrange,linestyle='-')
    ax8a.axis( (1.85,7.85,0.0,np.amax(distFIon)*1.02) )
    ax8a.set_xlabel(r'Velocity, $v/c$', fontsize=16)
    ax8a.set_ylabel(r'$\left\langle f(x,v,t)\right\rangle_{x,t}$', fontsize=16)
    ax8a.set_title(r'$\omega_{pe}t = $'+('{:06.4f}'.format(cTime)), fontsize=titleFontSize)
    
    for nFr in np.arange(iFrame,fFrame):
    
        #.Average the distribution function along x and add it to distF (to average in time later).
        fName     = fileRoot+'bump_'+str(nFr)+'.bp'    #.Complete file name.
        fDataBump = pgu.getInterpData(fName,polyOrder,basisType,comp=0)
        distFBump = fDataBump[0,:]
        #.Time of this frame.
        cTime = pgu.getTime(fName)

        fName     = fileRoot+'elc_'+str(nFr)+'.bp'    #.Complete file name.
        fDataBulk = pgu.getInterpData(fName,polyOrder,basisType,comp=0)
        distFBulk = fDataBulk[0,:]

        frameCount = frameCount+1

        #.Substitute new data into figure.
        hpl8a[0].set_ydata(distFBump)
        hpl8b[0].set_ydata(distFBulk)

        ax8a.set_title(r'$\omega_{pe}t = $'+('{:06.4f}'.format(cTime)), fontsize=titleFontSize)

        plt.savefig(outDirdist+'distF_'+str(nFr).zfill(4)+figureFileFormat)

iTw             = [int((nFrames-1)*0.1), int((nFrames-1)*0.3)]
modeOmega, _, _ = measureFrequency(iTw)
times, gamL = calcOmegabNgammaL(dataDir, outDir)