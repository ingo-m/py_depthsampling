# -*- coding: utf-8 -*-
"""
Permutation test for difference between conditions in depth profiles.

Compares cortical depth profiles from two different conditions (e.g. PacMan
dynamic vs. PacMan statis). Tests whether the difference between the two
conditions is significant at *any* cortical depth.

Prints results of permutation test.

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
# lstMdl = ['', '_deconv_model_1']
lstMdl = ['_deconv_model_1']

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['centre']

# ROI ('v1', 'v2', or 'v3'):
lstRoi = ['v1', 'v2']

# Path of depth-profile to load (meta-condition, ROI, condition, and
# deconvolution suffix left open):
strPthPrf = '/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/{}/{}_{}{}.npz'  #noqa

# Condition levels (used to complete file names):
lstCon = ['bright_square_sst_pe',
          'kanizsa_rotated_sst_pe',
          'kanizsa_sst_pe',
          'kanizsa_sst_pe_plus_kanizsa_rotated_sst_pe']

# Which conditions to compare (list of tuples with condition indices):
lstDiff = [(0, 1), (0, 2), (2, 1), (0, 3)]

# Number of resampling iterations (set to `None` in case of small enough sample
# size for exact test, otherwise Monte Carlo resampling is performed):
varNumIt = None
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

        # Array for p-values:
        aryData = np.zeros((len(lstRoi), len(lstCon)))

        # List for comparison labels:
        lstLbls = [None] * len(lstDiff)

        for idxRoi in range(len(lstRoi)):
            for idxDiff in range(len(lstDiff)):

                # Condition names:
                strTmpCon01 = lstCon[lstDiff[idxDiff][0]]
                strTmpCon02 = lstCon[lstDiff[idxDiff][1]]

                # Path of first depth profile:
                strTmpPth01 = strPthPrf.format(
                    lstMetaCon[idxMtaCn], lstRoi[idxRoi], strTmpCon01,
                    lstMdl[idxMdl])

                # Path of second depth profile:
                strTmpPth02 = strPthPrf.format(
                    lstMetaCon[idxMtaCn], lstRoi[idxRoi], strTmpCon02,
                    lstMdl[idxMdl])

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

                # Because of asymetric ROIs in the surface experiment (between
                # 'Kanizsa square' and 'Kanizsa rotated' conditions), there
                # may be a slight discrepancy in number of vertices across
                # ROIs. We take the mean.
                vecNumInc = np.around(
                                      np.multiply(
                                                  np.add(vecNumInc01,
                                                         vecNumInc02
                                                         ).astype(np.float32),
                                                  0.5)
                                      ).astype(np.int32)

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

                # Abbreviate condition labels:
                strTmp = strTmp.replace('bright_square_sst_pe', 'BS')
                strTmp = strTmp.replace('kanizsa_sst_pe', 'KS')
                strTmp = strTmp.replace('kanizsa_rotated_sst_pe', 'KR')

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
