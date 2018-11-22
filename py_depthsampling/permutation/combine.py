# -*- coding: utf-8 -*-
"""
Combine conditions for permutation.

Prepare permutation test of experimental condition (PacMan Dynamic) against
combination (mean) of two control conditions (PacMan Static and Control
Dynamic).

After having found no siginificant difference between the two control
conditions, we can compare the experimental condition against the combined
control conditions. Here, the respective `*.npz` files are prepared.
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

# Label for axes:
strXlabel = 'Cortical depth level'
strYlabel = 'fMRI signal change [a.u.]'

# Condition levels (used to complete file names):
lstCon = ['kanizsa_rotated_sst_pe',
          'kanizsa_sst_pe']

# Which conditions to combine (list of tuples with condition indices):
lstDiff = [(0, 1)]
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

                # Take the mean of the conditions (separately for each
                # subject and depth level):
                aryMne = np.multiply(
                                     np.add(
                                            aryDpth01,
                                            aryDpth02
                                            ),
                                     0.5
                                     )

                # Combine condition labels:
                strLblsComb = (strTmpCon01 + '_plus_' + strTmpCon02)

                # Complete output paths:
                strPthComb = strPthPrf.format(
                    lstMetaCon[idxMtaCn], lstRoi[idxRoi], strLblsComb,
                    lstMdl[idxMdl])

                # Save mean depth profiles, and number of vertices per
                # subject:
                np.savez(strPthComb,
                         arySubDpthMns=aryMne,
                         vecNumInc=vecNumInc)
# -----------------------------------------------------------------------------
