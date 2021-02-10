import postgkyl as pg
import numpy as np
import adios as ad
#.These are used for creating directories.
import os
from os import path
import errno
import shutil

sqrt2    = np.sqrt(2.0)
rsqrt2   = np.sqrt(2.0)/2.0
rsqrt2Cu = 1.0/np.sqrt(2.0**3)
sqrt3    = np.sqrt(3.0)
sqrt3d2  = np.sqrt(3.0/2.0)
sqrt5    = np.sqrt(5.0)
sqrt7    = np.sqrt(7.0)

#.Function to check existence of file.......#
def checkFile(fileIn):
    if os.path.exists(fileIn):
        return True
    else:
        return False

#.Function to check existence of directory.......#
def checkDir(dirIn):
    if os.path.exists(os.path.dirname(dirIn)):
        return True
    else:
        return False

#.Function to check existence and/or make directory.......#
def checkMkdir(dirIn):
    if not os.path.exists(os.path.dirname(dirIn)):
        try:
            os.makedirs(os.path.dirname(dirIn))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

#.Read the time variable in file..........................#
def getTime(dataFile):
    hF      = ad.file(dataFile)
    timeOut = hF['time'].read()
    hF.close()
    return timeOut

#.Establish last frame outputted (useful for unfinished runs):
def findLastFrame(absFileName,fileExt='bp'):
    #.Input is the file name with its absolute address attached to it.
    #.Indicate file type: fileExt='bp' for ADIOS files, ='h5' for HDF5.
    cF         = 0
    moreFrames = os.path.isfile(absFileName+str(cF+1)+'.'+fileExt)
    while moreFrames:
        cF = cF+1
        moreFrames = os.path.isfile(absFileName+str(cF+1)+'.'+fileExt)
    return cF

#.Establish the grid......................................#
def getGrid(dataFile,p,basisType,**opKey):
    pgData         = pg.GData(dataFile)                       #.Read data with pgkyl.
    if basisType=='ms':
        pgInterp       = pg.GInterpModal(pgData, p, basisType)    #.Interpolate data.
    elif basisType == 'ns':
        pgInterp       = pg.GInterpNodal(pgData, p, basisType)    #.Interpolate data.
    xNodal, dataInterp = pgInterp.interpolate()
    dimOut         = np.shape(xNodal)[0]			    #.Number of dimensions in data.

    #.If desired, output cell center values of grid coordinates instead of nodal coordinates.
    if 'location' in opKey:
        if opKey['location']=='center':
            xOut = [[] for i in range(dimOut)]
        for i in range(dimOut): 
            nNodes  = np.shape(xNodal[i])[0]
            xOut[i] = np.zeros(nNodes-1)
            xOut[i] = np.multiply(0.5,xNodal[i][0:nNodes-1]+xNodal[i][1:nNodes])
        else:
            xOut = xNodal
    else:
        xOut = xNodal

    nxOut = np.zeros(dimOut,dtype='int')
    lxOut = np.zeros(dimOut,dtype='double')
    dxOut = np.zeros(dimOut,dtype='double')
    for i in range(dimOut):
        nxOut[i] = np.size(xOut[i])         # grids of each dimension
        lxOut[i] = xOut[i][-1]-xOut[i][0]   # real length of each dimension
        dxOut[i] = xOut[i][ 1]-xOut[i][0]   # delta of each dimension
    return xOut, dimOut, nxOut, lxOut, dxOut

#.Interpolate DG data.....................................#
def getInterpData(dataFile,p,basisType,**opKey):
    pgData        = pg.GData(dataFile)                     #.Read data with pgkyl.
    if basisType=='ms':
        pgInterp       = pg.GInterpModal(pgData, p, basisType)    #.Interpolate data.
    elif basisType == 'ns':
        pgInterp       = pg.GInterpNodal(pgData, p, basisType)    #.Interpolate data.
    if 'comp' in opKey:
        xOut, dataOut = pgInterp.interpolate(opKey['comp'])
    else:
        xOut, dataOut = pgInterp.interpolate()
    return dataOut

#.This function finds the index of the grid point nearest to a given fix value.
def findNearestIndex(array,value):
  return (np.abs(array-value)).argmin()

#.Function to find the local maxima of the energy oscillations
#.within the time interval given by (interval[0],interval[1]).
def locatePeaks(timeIn,energyIn,interval,floor):
    nT  = np.shape(timeIn)[0]
    tLo = findNearestIndex(timeIn,interval[0])
    tUp = findNearestIndex(timeIn,interval[1])

    energyMaxima      = np.empty(1)
    energyMaximaTimes = np.empty(1)
    for it in range(tLo,tUp):
        if (energyIn[it]>energyIn[it-1]) and (energyIn[it]>energyIn[it+1]) and (energyIn[it]>floor):
            energyMaxima      = np.append(energyMaxima,energyIn[it])
            energyMaximaTimes = np.append(energyMaximaTimes,timeIn[it])

  #.Don't return random first value introduced by np.empty.
    return energyMaximaTimes[1:-1], energyMaxima[1:-1]