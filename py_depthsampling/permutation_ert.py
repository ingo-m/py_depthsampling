# -*- coding: utf-8 -*-
"""
Permutation test for conditions differences in event-related timecourses.

Compares event-related timecourses from two different conditions. Prints
results of permutation test.

Inputs are pickle files containing event-related timecourses for each subject.
"""

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


import numpy as np
from py_depthsampling.permutation.perm_max import permute_max
import pandas as pd
import pickle


# -----------------------------------------------------------------------------
# *** Define parameters

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['centre', 'edge', 'background']

# ROI ('v1', 'v2', or 'v3'):
lstRoi = ['v1', 'v2']

# Path of timecourses to load (meta-condition & ROI left open):
strPthErt = '/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/{}/era_{}.pickle'  #noqa

# Condition levels. The order of conditions has to be the same as in that in
# `py_depthsampling.ert.py` (i.e. in the script used to create the
# timecourses).
lstCon = ['bright_square', 'kanizsa', 'kanizsa_rotated']

# Which conditions to compare (list of tuples with condition indices):
lstDiff = [(0, 1), (0, 2), (2, 1)]

# Number of resampling iterations (set to `None` in case of small enough sample
# size for exact test, otherwise Monte Carlo resampling is performed):
varNumIt = None

# Time window within which to compare timecourses (volume indices):
tplCmp = (7, 11)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Preparations

# Number of comparisons:
varNumDiff = len(lstDiff)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Permutation test

# Loop through models, ROIs, hemispheres, and conditions to create plots:
for idxMtaCn in range(len(lstMetaCon)):  #noqa

    # Array for p-values:
    aryData = np.zeros((len(lstRoi), len(lstCon)))

    # List for comparison labels:
    lstLbls = [None] * len(lstDiff)

    for idxRoi in range(len(lstRoi)):
        for idxDiff in range(len(lstDiff)):

            # Indices of conditions to compare on this iteration:
            varIdxCon01 = lstDiff[idxDiff][0]
            varIdxCon02 = lstDiff[idxDiff][1]

            # Condition names:
            strTmpCon01 = lstCon[varIdxCon01]
            strTmpCon02 = lstCon[varIdxCon02]

            # Path of timecourse:
            strTmpPth = strPthErt.format(
                lstMetaCon[idxMtaCn], lstRoi[idxRoi])

            # Load previously prepared event-related timecourses from pickle:
            dicAllSubsRoiErt = pickle.load(open(strTmpPth, 'rb'))

            # Get subject IDs:
            lstSubIds = list(dicAllSubsRoiErt.keys())

            # Get number of volumes:
            varNumVol = dicAllSubsRoiErt[lstSubIds[0]][0].shape[2]

            # Get number of depth levels:
            varNumDpth = dicAllSubsRoiErt[lstSubIds[0]][0].shape[1]

            # Number of subjects:
            varNumSub = len(lstSubIds)

            # Arrays for single-subject timecourses for both conditions:
            # aryErt01 = np.zeros((varNumSub, varNumVol))
            # aryErt02 = np.zeros((varNumSub, varNumVol))
            aryErt01 = np.zeros((varNumSub, varNumDpth))
            aryErt02 = np.zeros((varNumSub, varNumDpth))

            # Vector for number of vertices per subject (for weighter
            # averaging across subjects):
            vecNumInc = np.zeros(varNumSub)

            # Access timecourses - condition 01:
            for idxSub in range(varNumSub):

                # Get ERT array for current subject, shape:
                # aryErtTmp[condition, depth, time].
                aryErtTmp = dicAllSubsRoiErt[lstSubIds[idxSub]][0]

                # Access condition to compare (new shape
                # aryErtTmp[depth, time]).
                aryErtTmp = aryErtTmp[varIdxCon01]

                # Mean over depth levels:
                # aryErt01[idxSub] = np.mean(aryErtTmp, axis=0)
                # Mean over time window:
                aryErt01[idxSub] = np.mean(aryErtTmp[:, tplCmp[0]:tplCmp[1]],
                                           axis=1)

                # Number of vertices for current subject:
                vecNumInc[idxSub] = dicAllSubsRoiErt[lstSubIds[idxSub]][1]

            # Access timecourses - condition 02:
            for idxSub in range(varNumSub):

                # Get ERT array for current subject, shape:
                # aryErtTmp[condition, depth, time].
                aryErtTmp = dicAllSubsRoiErt[lstSubIds[idxSub]][0]

                # Access condition to compare (new shape
                # aryErtTmp[depth, time]).
                aryErtTmp = aryErtTmp[varIdxCon02]

                # Mean over depth levels:
                # aryErt02[idxSub] = np.mean(aryErtTmp, axis=0)
                # Mean over time window:
                aryErt02[idxSub] = np.mean(aryErtTmp[:, tplCmp[0]:tplCmp[1]],
                                           axis=1)

            # Access time window for comparison:
            # aryErt01 = aryErt01[:, tplCmp[0]:tplCmp[1]]
            # aryErt02 = aryErt02[:, tplCmp[0]:tplCmp[1]]

            # Run permutation test:
            varP = permute_max(aryErt01,
                               aryErt02,
                               vecNumInc=vecNumInc,
                               varNumIt=varNumIt)

            # Put p-value of current ROI & comparison into array:
            aryData[idxRoi, idxDiff] = varP

            strMsg = ('---Permutation p-value \n'
                      + '   Meta-condition: '
                      + lstMetaCon[idxMtaCn]
                      + '\n'
                      + '   ROI: '
                      + lstRoi[idxRoi]
                      + '\n'
                      + '   Condition: '
                      + strTmpCon01
                      + ' minus '
                      + strTmpCon02
                      + '\n'
                      + '   p = '
                      + str(varP))

            print(strMsg)

            # Comparison label:
            strTmp = (strTmpCon01 + '-' + strTmpCon02)

            # Column label for dataframe:
            lstLbls[idxDiff] = strTmp

    # p-values into dataframe:
    dfData = pd.DataFrame(data=aryData,
                          index=lstRoi,
                          columns=lstLbls)
    print('')
    print(dfData)
    print('')
# -----------------------------------------------------------------------------
