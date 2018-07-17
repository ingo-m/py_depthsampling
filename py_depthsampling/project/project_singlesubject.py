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
# *** Define parameters

# Load/save existing projection from/to (ROI, condition, depth level label, and
# subject ID left open):
strPthNpy = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/project_single_subject/{}_{}_{}_{}.npy'  #noqa

# List of subject identifiers:
lstSubIds = ['20171023',  # '20171109',
             '20171204_01',
             '20171204_02',
             '20171211',
             '20171213',
             '20180111',
             '20180118']

# Nested list with depth levels to average over. For instance, if `lstDpth =
# [[0, 1, 2], [3, 4, 5]]`, on a first iteration, the average over the first
# three depth levels is calculated, and on a second iteration the average over
# the subsequent three depth levels is calculated. If 1lstDpth= [[None]]1,
# average over all depth levels.
lstDpth = [None, [0, 1, 2], [4, 5, 6], [8, 9, 10]]
# lstDpth = [[x] for x in range(11)]
# Depth level condition labels (output file will contain this label):
lstDpthLbl = ['allGM', 'deepGM', 'midGM', 'superficialGM']
# lstDpthLbl = [str(x) for x in range(11)]

# ROI ('v1' or 'v2'):
lstRoi = ['v1', 'v2', 'v3']

# Output path & prefix for plots (ROI, condition, depth level label, and
# subject ID left open):
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/project/pe_single_subject/{}_{}_{}_{}'  #noqa

# File type suffix for plot:
# strFlTp = '.svg'
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Condition levels (used to complete file names):
# lstCon = ['polar_angle', 'x_pos', 'y_pos', 'SD', 'R2']
lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst']

# Path of vtk mesh with data to project into visual space (e.g. parameter
# estimates; subject ID, hemisphere, and contion level left open).
strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/feat_level_2_{}_cope.vtk'  #noqa
# strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/feat_level_2_{}_zstat.vtk'  #noqa
# strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_{}.vtk'  #noqa

# Path of mean EPI (for scaling to percent signal change; subject ID and
# hemisphere left open):
strPthMneEpi = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/combined_mean.vtk'  #noqa

# Path of vtk mesh with R2 values from pRF mapping (at multiple depth levels;
# subject ID and hemisphere left open).
strPthR2 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_R2.vtk'  #noqa

# Path of vtk mesh with pRF sizes (at multiple depth levels; subject ID and
# hemisphere left open).
strPthSd = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_SD.vtk'  #noqa

# Path of vtk mesh with pRF x positions at multiple depth levels; subject ID
# and hemisphere left open).
strPthX = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_x_pos.vtk'  #noqa

# Path of vtk mesh with pRF y positions at multiple depth levels; subject ID
# and hemisphere left open).
strPthY = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_y_pos.vtk'  #noqa

# Path of csv file with ROI definition (subject ID, hemisphere, and ROI left
# open).
strCsvRoi = '/home/john/PhD/GitLab/PacMan/analysis/{}/08_depthsampling/{}/{}_mod.csv'  #noqa
# strCsvRoi = '/Users/john/1_PhD/GitLab/PacMan/analysis/{}/08_depthsampling/{}/{}_mod.csv'  #noqa

# Number of cortical depths.
varNumDpth = 11

# Extent of visual space from centre of the screen in negative x-direction
# (i.e. from the fixation point to the left end of the screen) in degrees of
# visual angle.
varExtXmin = -5.19
# Extent of visual space from centre of the screen in positive x-direction
# (i.e. from the fixation point to the right end of the screen) in degrees of
# visual angle.
varExtXmax = 5.19
# Extent of visual space from centre of the screen in negative y-direction
# (i.e. from the fixation point to the lower end of the screen) in degrees of
# visual angle.
varExtYmin = -5.19
# Extent of visual space from centre of the screen in positive y-direction
# (i.e. from the fixation point to the upper end of the screen) in degrees of
# visual angle.
varExtYmax = 5.19

# R2 threshold for vertex inclusion (vertices with R2 value below threshold are
# not considered for plot):
varThrR2 = 0.15

# Number of bins for visual space representation in x- and y-direction (ratio
# of number of x and y bins should correspond to ratio of size of visual space
# in x- and y-directions).
varNumX = 200
varNumY = 200
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Parent loop

print('-Project parametric map into visual space')

# Loop through subjects, depth levels, ROIs, and conditions:
for idxSub in range(len(lstSubIds)):  #noqa
    for idxDpth in range(len(lstDpth)):
        for idxRoi in range(len(lstRoi)):
            for idxCon in range(len(lstCon)):

                # File name of npy file for current condition:
                strPthNpyTmp = strPthNpy.format(lstRoi[idxRoi],
                                                lstCon[idxCon],
                                                lstDpthLbl[idxDpth],
                                                lstSubIds[idxSub])

                if os.path.isfile(strPthNpyTmp):

                    print('--Load existing visual field projection')

                    # Load existing projection:
                    aryVslSpc = np.load(strPthNpyTmp)

                else:

                    # ---------------------------------------------------------
                    # *** Load data

                    print('--Load data from vtk meshes')

                    # Number of processes to run in parallel (one for single
                    # subject plots):
                    varPar = 1

                    # Create a queue to put the results in:
                    queOut = mp.Queue()

                    # Empty list for processes:
                    lstPrcs = [None] * varPar

                    # Empty list for results of parallel processes:
                    lstRes = [None] * varPar

                    # Create processes:
                    for idxPrc in range(varPar):
                        lstPrcs[idxPrc] = mp.Process(target=load_par,
                                                     args=(lstSubIds[idxSub],
                                                           lstCon[idxCon],
                                                           lstRoi[idxRoi],
                                                           strPthData,
                                                           strPthMneEpi,
                                                           strPthR2,
                                                           strPthX,
                                                           strPthY,
                                                           strPthSd,
                                                           strCsvRoi,
                                                           varNumDpth,
                                                           lstDpth[idxDpth],
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

                    # Put output into correct order (unnecessary in this
                    # context but kept for consistency):
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

                    # ---------------------------------------------------------
                    # *** Convert cope to percent signal change

                    # According to the FSL documentation
                    # (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT/UserGuide),
                    # the PEs can be scaled to signal change with respect to
                    # the mean (over time within voxel): "This is achieved by
                    # scaling the PE or COPE values by (100*) the peak-peak
                    # height of the regressor (or effective regressor in the
                    # case of COPEs) and then by dividing by mean_func (the
                    # mean over time of filtered_func_data)."

                    # Only perform scaling if the data is from an FSL cope
                    # file:
                    if 'cope' in strPthData:
                        print('--Convert cope to percent signal change.')

                        # The peak-peak height depends on the predictor (i.e.
                        # condition).
                        if 'sst' in lstCon[idxCon]:
                            varPpheight = 1.268049
                        elif 'trn' in lstCon[idxCon]:
                            varPpheight = 0.2269044

                        # In order to avoid division by zero, avoid
                        # zero-voxels:
                        lgcTmp = np.not_equal(vecData, 0.0)

                        # Apply PSC scaling, as described above:
                        vecData[lgcTmp] = \
                            np.multiply(
                                        np.divide(
                                                  np.multiply(
                                                              vecData[lgcTmp],
                                                              (100.0
                                                               * varPpheight)
                                                              ),
                                                  vecMneEpi[lgcTmp]),
                                        1.0  # 1.4
                                        )

                    # ---------------------------------------------------------
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

                    # List for results after re-ordering (visual space arrays
                    # and normalisation arrays):
                    lstVslSpc = [None] * varPar
                    lstNorm = [None] * varPar

                    # Put output into correct order (unnecessary in this
                    # context but kept for consistency):
                    for idxRes in range(varPar):

                        # Index of results (first item in output list):
                        varTmpIdx = lstRes[idxRes][0]

                        # Put fitting results into list, in correct order:
                        lstVslSpc[varTmpIdx] = lstRes[idxRes][1]
                        lstNorm[varTmpIdx] = lstRes[idxRes][2]

                    # Visual space array (2D array with bins of locations in
                    # visual space):
                    aryVslSpc = np.zeros((varNumX, varNumY))

                    # Array for normalisation (parameter estimates are summed
                    # up over the visual field; the normalisation array is
                    # needed to normalise the sum):
                    aryNorm = np.zeros((varNumX, varNumY))

                    # Add up results from separate processes:
                    for idxPrc in range(varPar):
                        aryVslSpc = np.add(lstVslSpc[idxPrc], aryVslSpc)
                        aryNorm = np.add(lstNorm[idxPrc], aryNorm)

                    # Normalise:
                    aryVslSpc = np.divide(aryVslSpc, aryNorm)

                    # Save results to disk:
                    np.save(strPthNpyTmp, aryVslSpc)

                # -------------------------------------------------------------
                # *** Plot results

                print('--Plot results')

                # Output path for plot:
                strPthPltOtTmp = (strPthPltOt.format(lstRoi[idxRoi],
                                                     lstCon[idxCon],
                                                     lstDpthLbl[idxDpth],
                                                     lstSubIds[idxSub])
                                  + strFlTp)

                # Plot title:
                strTmpTtl = (lstRoi[idxRoi]
                             + ' '
                             + lstCon[idxCon]
                             + ' '
                             + lstDpthLbl[idxDpth])

                # Create plot:
                plot(aryVslSpc,
                     strTmpTtl,
                     'x-position',
                     'y-position',
                     strPthPltOtTmp,
                     tpleLimX=(varExtXmin, varExtXmax, 3.0),
                     tpleLimY=(varExtYmin, varExtYmax, 3.0),
                     varMin=-2.5,
                     varMax=2.5)
# -----------------------------------------------------------------------------