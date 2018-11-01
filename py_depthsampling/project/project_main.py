# -*- coding: utf-8 -*-
"""Project parameter estimates into a visual space representation."""

# Part of py_depthsampling library
# Copyright (C) 2018  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import numpy as np
import multiprocessing as mp
from py_depthsampling.project.load_par import load_par
from py_depthsampling.project.project_par import project_par
from py_depthsampling.project.plot import plot


# -----------------------------------------------------------------------------
def project(strRoi, strCon, strDpth, strDpthLbl, strPthNpy, varNumSub,
            lstSubIds, strPthData, strPthMneEpi, strPthR2, strPthX, strPthY,
            strPthSd, strCsvRoi, varNumDpth, varThrR2, varNumX, varNumY,
            varExtXmin, varExtXmax, varExtYmin, varExtYmax, strPthPltOt,
            strFlTp, varMin=-3.0, varMax=3.0, varTr=None):
    """Project parameter estimates into a visual space representation."""
    # File name of npy file for current condition:
    strPthNpyTmp = strPthNpy.format(strRoi,
                                    strCon,
                                    strDpthLbl)

    if os.path.isfile(strPthNpyTmp):

        print('--Load existing visual field projection')

        # Load existing projection:
        aryVslSpc = np.load(strPthNpyTmp)

    else:

        # ---------------------------------------------------------------------
        # *** Load data

        print('--Load data from vtk meshes')

        # Number of processes to run in parallel:
        varPar = varNumSub

        # Create a queue to put the results in:
        queOut = mp.Queue()

        # Empty list for processes:
        lstPrcs = [None] * varPar

        # Empty list for results of parallel processes:
        lstRes = [None] * varPar

        # Create processes:
        for idxPrc in range(varPar):
            lstPrcs[idxPrc] = mp.Process(target=load_par,
                                         args=(lstSubIds[idxPrc],
                                               strCon,
                                               strRoi,
                                               strPthData,
                                               strPthMneEpi,
                                               strPthR2,
                                               strPthX,
                                               strPthY,
                                               strPthSd,
                                               strCsvRoi,
                                               varNumDpth,
                                               strDpth,
                                               varTr,
                                               idxPrc,
                                               queOut)
                                         )

            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

        # Don't create more processes than number of subjects:
        # varParTmp = int(np.min([varPar, len(lstSubIds)]))

        # Start processes:
        for idxPrc in range(varPar):
            lstPrcs[idxPrc].start()

        # Collect results from queue:
        for idxPrc in range(varPar):
            lstRes[idxPrc] = queOut.get(True)

        # Join processes:
        for idxPrc in range(varPar):
            lstPrcs[idxPrc].join()

        # List for single-subject data vectors:
        lstData = [None] * varPar

        # List for single-subject mean EPI vectors:
        lstMneEpi = [None] * varPar

        # List of single subject R2 vectors:
        lstR2 = [None] * varPar

        # List for single subject SD vectors (pRF sizes):
        lstSd = [None] * varPar

        # List for single subject x-position vectors:
        lstX = [None] * varPar

        # List for single subject y-position vectors:
        lstY = [None] * varPar

        # Put output into correct order (unnecessary in this context
        # but kept for consistency):
        for idxRes in range(varPar):

            # Index of results (first item in output list):
            varTmpIdx = lstRes[idxRes][0]

            # Put fitting results into list, in correct order:
            lstData[varTmpIdx] = lstRes[idxRes][1]
            lstMneEpi[varTmpIdx] = lstRes[idxRes][2]
            lstR2[varTmpIdx] = lstRes[idxRes][3]
            lstSd[varTmpIdx] = lstRes[idxRes][4]
            lstX[varTmpIdx] = lstRes[idxRes][5]
            lstY[varTmpIdx] = lstRes[idxRes][6]

        # Concatenate arrays from all subjects:
        vecData = np.concatenate(lstData[:])
        vecMneEpi = np.concatenate(lstMneEpi[:])
        vecR2 = np.concatenate(lstR2[:])
        vecSd = np.concatenate(lstSd[:])
        vecX = np.concatenate(lstX[:])
        vecY = np.concatenate(lstY[:])

        # Delete original lists:
        del(lstData)
        del(lstMneEpi)
        del(lstR2)
        del(lstSd)
        del(lstX)
        del(lstY)

        # ---------------------------------------------------------------------
        # *** Convert cope to percent signal change

        # According to the FSL documentation
        # (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT/UserGuide), the PEs can
        # be scaled to signal change with respect to the mean (over time within
        # voxel): "This is achieved by scaling the PE or COPE values by (100*)
        # the peak-peak height of the regressor (or effective regressor in the
        # case of COPEs) and then by dividing by mean_func (the mean over time
        # of filtered_func_data)."

        # Only perform scaling if the data is from an FSL cope file:
        if (('cope' in strCon) or ('_pe' in strCon)):
            print('--Convert cope to percent signal change.')

            # The peak-peak height depends on the predictor (i.e.
            # condition).
            if 'sst' in strCon:
                varPpheight = 1.268049
            elif 'trn' in strCon:
                varPpheight = 0.2269044

            # In order to avoid division by zero, avoid zero-voxels:
            lgcTmp = np.not_equal(vecData, 0.0)

            # Apply PSC scaling, as described above:
            vecData[lgcTmp] = \
                np.multiply(
                            np.divide(
                                      np.multiply(
                                                  vecData[lgcTmp],
                                                  (100.0 * varPpheight)
                                                  ),
                                      vecMneEpi[lgcTmp]),
                            1.0  # 1.4
                            )

        # ---------------------------------------------------------------------
        # *** Project data into visual space

        print('--Project data into visual space')

        # Number of processes to run in parallel:
        varPar = 11

        # Split data into chunks:
        lstData = np.array_split(vecData, varPar)
        lstR2 = np.array_split(vecR2, varPar)
        lstSd = np.array_split(vecSd, varPar)
        lstX = np.array_split(vecX, varPar)
        lstY = np.array_split(vecY, varPar)

        # Create a queue to put the results in:
        queOut = mp.Queue()

        # Empty list for processes:
        lstPrcs = [None] * varPar

        # Empty list for results of parallel processes:
        lstRes = [None] * varPar

        # Create processes:
        for idxPrc in range(varPar):
            lstPrcs[idxPrc] = mp.Process(target=project_par,
                                         args=(idxPrc,
                                               lstData[idxPrc],
                                               lstX[idxPrc],
                                               lstY[idxPrc],
                                               lstSd[idxPrc],
                                               lstR2[idxPrc],
                                               varThrR2,
                                               varNumX,
                                               varNumY,
                                               varExtXmin,
                                               varExtXmax,
                                               varExtYmin,
                                               varExtYmax,
                                               queOut)
                                         )

            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

        # Start processes:
        for idxPrc in range(varPar):
            lstPrcs[idxPrc].start()

        # Collect results from queue:
        for idxPrc in range(varPar):
            lstRes[idxPrc] = queOut.get(True)

        # Join processes:
        for idxPrc in range(varPar):
            lstPrcs[idxPrc].join()

        # List for results after re-ordering (visual space arrays and
        # normalisation arrays):
        lstVslSpc = [None] * varPar
        lstNorm = [None] * varPar

        # Put output into correct order (unnecessary in this context but kept
        # for consistency):
        for idxRes in range(varPar):

            # Index of results (first item in output list):
            varTmpIdx = lstRes[idxRes][0]

            # Put fitting results into list, in correct order:
            lstVslSpc[varTmpIdx] = lstRes[idxRes][1]
            lstNorm[varTmpIdx] = lstRes[idxRes][2]

        # Visual space array (2D array with bins of locations in visual space):
        aryVslSpc = np.zeros((varNumX, varNumY))

        # Array for normalisation (parameter estimates are summed up over the
        # visual field; the normalisation array is needed to normalise the
        # sum):
        aryNorm = np.zeros((varNumX, varNumY))

        # Add up results from separate processes:
        for idxPrc in range(varPar):
            aryVslSpc = np.add(lstVslSpc[idxPrc], aryVslSpc)
            aryNorm = np.add(lstNorm[idxPrc], aryNorm)

        # Normalise:
        aryVslSpc = np.divide(aryVslSpc, aryNorm)

        # Save results to disk:
        np.save(strPthNpyTmp, aryVslSpc)

    # -------------------------------------------------------------------------
    # *** Plot results

    print('--Plot results')

    if varTr is None:

        # Output path for plot:
        strPthPltOtTmp = (strPthPltOt.format(strRoi,
                                             strCon,
                                             strDpthLbl)
                          + strFlTp)

    else:

        # In case of time series data, the volume index is part of the file
        # name.
        strPthPltOtTmp = (strPthPltOt.format(strRoi,
                                             strCon,
                                             strDpthLbl)
                          + '_volume_'
                          + str(varTr)
                          + strFlTp)

    # Plot title:
    strTmpTtl = (strRoi
                 + ' '
                 + strCon
                 + ' '
                 + strDpthLbl)

    # Create plot:
    plot(aryVslSpc,
         strTmpTtl,
         'x-position',
         'y-position',
         strPthPltOtTmp,
         tpleLimX=(varExtXmin, varExtXmax, 3.0),
         tpleLimY=(varExtYmin, varExtYmax, 3.0),
         varMin=varMin,
         varMax=varMax)
# -----------------------------------------------------------------------------
