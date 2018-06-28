# -*- coding: utf-8 -*-
"""
Permutation test for difference between conditions in depth profiles.

Compares cortical depth profiles from two different conditions (e.g. PacMan
dynamic vs. PacMan statis). Tests whether the difference between the two
conditions is significant at any cortical depth.

Inputs are *.npz files containing depth profiles for each subject.
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


# -----------------------------------------------------------------------------
# *** Define parameters

# Draining model suffix ('' for non-corrected profiles):
lstMdl = ['', '_deconv_model_1']

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['stimulus']

# ROI ('v1', 'v2', or 'v3'):
lstRoi = ['v1', 'v2', 'v3']

# Hemisphere ('rh' or 'lh'):
lstHmsph = ['rh']

# Path of depth-profile to load (meta-condition, ROI, hemisphere, condition,
# and deconvolution suffix left open):
strPthPrf = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}{}.npz'  #noqa

# Label for axes:
strXlabel = 'Cortical depth level'
strYlabel = 'fMRI signal change [a.u.]'

# Condition levels (used to complete file names):
# lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst',
#           'Pd_trn', 'Cd_trn', 'Ps_trn']
lstCon = ['Pd_sst', 'Ps_sst', 'Cd_sst', 'Ps_sst_plus_Cd_sst']
# lstCon = ['Pd_trn', 'Ps_trn', 'Cd_trn', 'Ps_trn_plus_Cd_trn']

# Which conditions to compare (list of tuples with condition indices):
lstDiff = [(0, 1), (0, 2), (1, 2), (0, 3)]

# Number of resampling iterations:
varNumIt = 100000
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
    for idxMdl in range(len(lstMdl)):  #noqa
        for idxHmsph in range(len(lstHmsph)):

            # Array for p-values:
            aryData = np.zeros((len(lstRoi), len(lstCon)))

            # List for comparison labels:
            lstLbls = [None] * len(lstDiff)

            for idxRoi in range(len(lstRoi)):
                for idxDiff in range(len(lstDiff)):

                    # Condition names:
                    strTmpCon01 = lstCon[lstDiff[idxDiff][0]]
                    strTmpCon02 = lstCon[lstDiff[idxDiff][1]]

                    # Only conduct test for stimulus-sustained and
                    # perihpery-transient:
                    lgcPass = (
                               (
                                ('stimulus' in lstMetaCon[idxMtaCn])
                                and
                                ('sst' in strTmpCon01)
                               )
                               or
                               (
                                ('periphery' in lstMetaCon[idxMtaCn])
                                and
                                ('trn' in strTmpCon01)
                                )
                               )
                    if lgcPass:

                        # Path of first depth profile:
                        strTmpPth01 = strPthPrf.format(
                            lstMetaCon[idxMtaCn], lstRoi[idxRoi],
                            lstHmsph[idxHmsph], strTmpCon01, lstMdl[idxMdl])

                        # Path of second depth profile:
                        strTmpPth02 = strPthPrf.format(
                            lstMetaCon[idxMtaCn], lstRoi[idxRoi],
                            lstHmsph[idxHmsph], strTmpCon02, lstMdl[idxMdl])

                        # Load single subject depth profiles (shape
                        # aryDpth[subject, depth]):
                        objNpz01 = np.load(strTmpPth01)
                        aryDpth01 = objNpz01['arySubDpthMns']
                        objNpz02 = np.load(strTmpPth02)
                        aryDpth02 = objNpz02['arySubDpthMns']

                        # Array with number of vertices (for weighted averaging
                        # across subjects), shape: vecNumInc[subjects].
                        vecNumInc01 = objNpz01['vecNumInc']
                        vecNumInc02 = objNpz02['vecNumInc']

                        # Number of vertices are assumed to be the same for the
                        # two conditions (since the data is sampled from the
                        # same ROI). If not, raise an error.
                        if np.all(np.equal(vecNumInc01, vecNumInc02)):
                            vecNumInc = vecNumInc01
                        else:
                            strErrMsg = ('ERROR. Number of vertices within ROI'
                                         + ' is not consistent across'
                                         + ' conditions.')
                            raise ValueError(strErrMsg)

                        # Run permutation test:
                        varP = permute_max(aryDpth01,
                                           aryDpth02,
                                           vecNumInc=vecNumInc,
                                           varNumIt=varNumIt)

                        # Put p-value of current ROI & comparison into array:
                        aryData[idxRoi, idxDiff] = varP

                        strMsg = ('---Permutation p-value \n'
                                  + '   Model: '
                                  + lstMdl[idxMdl]
                                  + '\n'
                                  + '   Meta-condition: '
                                  + lstMetaCon[idxMtaCn]
                                  + '\n'
                                  + '   ROI: '
                                  + lstRoi[idxRoi]
                                  + '\n'
                                  + '   Hemisphere: '
                                  + lstHmsph[idxHmsph]
                                  + '   Condition: '
                                  + strTmpCon01
                                  + ' minus '
                                  + strTmpCon02
                                  + '\n'
                                  + '   p = '
                                  + str(varP))

                        print(strMsg)

                        # Column label for dataframe:
                        lstLbls[idxDiff] = (strTmpCon01 + '-' + strTmpCon02)

            # p-values into dataframe:
            dfData = pd.DataFrame(data=aryData,
                                  index=lstRoi,
                                  columns=lstLbls)
            print('')
            print(dfData)
            print('')
# -----------------------------------------------------------------------------
