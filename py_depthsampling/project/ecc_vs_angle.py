# -*- coding: utf-8 -*-
"""Plot parameter estimates by polar angle and by eccentricity."""

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
from py_depthsampling.project.plot import plot


# -----------------------------------------------------------------------------
# *** Define parameters

# Load/save existing data from/to (ROI and condition left open):
strPthNpz = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/ecc_vs_angle/{}_{}.npy'  #noqa

# List of subject identifiers:
lstSubIds = ['20171023',
             '20171204_01',
             '20171204_02',
             '20171211',
             '20171213',
             '20180111',
             '20180118']

# Draining model suffix ('' for non-corrected profiles):
lstMdl = ['']

# ROI ('v1' or 'v2'):
lstRoi = ['v1', 'v2', 'v3']

# Output path & prefix for plots (ROI and condition left open):
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/ecc_vs_angle/{}_{}'  #noqa

# File type suffix for plot:
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Condition levels (used to complete file names):
lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst',
          'Pd_trn', 'Cd_trn', 'Ps_trn',
          'Pd_min_Ps_sst', 'Pd_min_Cd_sst', 'Cd_min_Ps_sst', 'Linear_sst',
          'Pd_min_Ps_trn', 'Pd_min_Cd_trn', 'Cd_min_Ps_trn', 'Linear_trn']
# lstCon = ['polar_angle', 'x_pos', 'y_pos', 'SD', 'R2']

# Path of vtk mesh with data to project into visual space (e.g. parameter
# estimates; subject ID, hemisphere, and contion level left open).
strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/feat_level_2_{}_cope.vtk'  #noqa
# strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_{}.vtk'  #noqa

# Path of vtk mesh with R2 values from pRF mapping (at multiple depth levels;
# subject ID and hemisphere left open).
strPthR2 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_R2.vtk'  #noqa

# Path of vtk mesh with pRF sizes (at multiple depth levels; subject ID and
# hemisphere left open). (Not actually needed but kept for modularity.)
strPthSd = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_SD.vtk'  #noqa

# Path of vtk mesh with pRF polar angle at multiple depth levels; subject ID
# and hemisphere left open).
strPthAngl = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_polar_angle.vtk'  #noqa

# Path of vtk mesh with pRF eccentricity at multiple depth levels; subject ID
# and hemisphere left open).
strPthEcc = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_eccentricity.vtk'  #noqa

# Path of csv file with ROI definition (subject ID, hemisphere, and ROI left
# open).
strCsvRoi = '/home/john/PhD/GitHub/PacMan/analysis/{}/08_depthsampling/{}/{}_mod.csv'  #noqa

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
varNumX = 100
varNumY = 100
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Parent loop

print('-Plot parameter estimates by polar angle and by eccentricity')

# Number of subjects:
varNumSub = len(lstSubIds)

# Loop through models, ROIs, and conditions:
for idxRoi in range(len(lstRoi)):  #noqa
    for idxCon in range(len(lstCon)):

        # File name of npy file for current condition:
        strPthNpzTmp = strPthNpz.format(lstRoi[idxRoi],
                                        lstCon[idxCon])

        if os.path.isfile(strPthNpzTmp):

            # Load data from disk:

            # Load data from npz file:
            objNpz = np.load(strPthNpzTmp)

            # Retrieve arrays from npz object (dictionary):
            vecData = objNpz['vecData']
            vecR2 = objNpz['vecR2']
            # vecSd = objNpz['vecSd']
            vecAngl = objNpz['vecAngl']
            vecEcc = objNpz['vecEcc']

        else:

            # -------------------------------------------------------------
            # *** Load data

            print('--Load data')

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
                                                   lstCon[idxCon],
                                                   lstRoi[idxRoi],
                                                   strPthData,
                                                   strPthR2,
                                                   strPthAngl,
                                                   strPthEcc,
                                                   strPthSd,
                                                   strCsvRoi,
                                                   varNumDpth,
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

            # List of single subject R2 vectors:
            lstR2 = [None] * varPar

            # List for single subject SD vectors (pRF sizes):
            lstSd = [None] * varPar

            # List for single subject x-position vectors:
            lstAngl = [None] * varPar

            # List for single subject y-position vectors:
            lstEcc = [None] * varPar

            # Put output into correct order (unnecessary in this context
            # but kept for consistency):
            for idxRes in range(varPar):

                # Index of results (first item in output list):
                varTmpIdx = lstRes[idxRes][0]

                # Put fitting results into list, in correct order:
                lstData[varTmpIdx] = lstRes[idxRes][1]
                lstR2[varTmpIdx] = lstRes[idxRes][2]
                lstSd[varTmpIdx] = lstRes[idxRes][3]
                lstAngl[varTmpIdx] = lstRes[idxRes][4]
                lstEcc[varTmpIdx] = lstRes[idxRes][5]

            # Concatenate arrays from all subjects:
            vecData = np.concatenate(lstData[:])
            vecR2 = np.concatenate(lstR2[:])
            vecSd = np.concatenate(lstSd[:])
            vecAngl = np.concatenate(lstAngl[:])
            vecEcc = np.concatenate(lstEcc[:])

            # Delete original lists:
            del(lstData)
            del(lstR2)
            del(lstSd)
            del(lstAngl)
            del(lstEcc)

            # Save data to disk:
            np.savez(strPthNpzTmp,
                     vecData=vecData,
                     vecR2=vecR2,
                     vecSd=vecSd,
                     vecAngl=vecAngl,
                     vecEcc=vecEcc)

            # -------------------------------------------------------------
            # ***



#        # -----------------------------------------------------------------
#        # *** Plot group results
#
#        # Output path for plot:
#        strPthPltOtTmp = (strPthPltOt.format(lstRoi[idxRoi],
#                                             lstCon[idxCon])
#                          + strFlTp)
#
#        # Create plot:
#        plot(aryVslSpc,
#             'Visual field projection',
#             'x-position',
#             'y-position',
#             strPthPltOtTmp)
# -----------------------------------------------------------------------------
