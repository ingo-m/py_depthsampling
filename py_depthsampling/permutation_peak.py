# -*- coding: utf-8 -*-
"""
Permutation test peak position in depth profiles of condition difference.

This version: Do not perform permutation test directly on single-condition
              depth profiles, but on depth profiles of condition differences
              (e.g., is the peak position in the contrast condition A vs. B the
              same as in the contrast A vs. C).

Performe a permutation hypothesis test for a difference in the peak position in
cortical depth profiles between two contrasts. More specifically, the equality
of distributions of the peak positions is tested (i.e. a possible difference
could be due to a difference in mean, variance, or the shape of the
distribution).

Because contrast labels are permuted within subjects, single subject depth
profiles need to be provided.

The procedure is as follows:
- Condition contrasts are computed within subjects (i.e. depth profiles of both
  comparisons are subtracted, e.g. A-B and A-C).
- Contrast labels (i.e. first comparison, e.g. A-B, and second comparison, e.g.
  A-C) are permuted within subjects for each permutation data set (i.e. on each
  resampling iteration).
- For each resamling iteration, the mean difference depth profile is computed
  (across across subjects within permutation samples).
- For each resampling iteration, the peak position in both (randomly resampled)
  contrasts is located.
- The difference in peak positions between permutation samples is computed
  (separately for each permutation sample). The distribution of differences in
  peak position is the null distribution.
- The peak difference on the empirical contrasts is calculated, and the
  permutation p-value with respect to the null distribution is produced.

Function of the depth sampling pipeline.
"""

# Part of py_depthsampling library
# Copyright (C) 2017  Ingo Marquardt
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
import pandas as pd
from py_depthsampling.permutation.peak_pos_perm_diff import peak_diff


# -----------------------------------------------------------------------------
# *** Define parameters

# Which draining model to load ('' for none):
lstMdl = ['', '_deconv_model_1']

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['stimulus']

# ROI ('v1', 'v2', or 'v3'):
lstRoi = ['v1', 'v2', 'v3']

# Hemisphere ('rh' or 'lh'):
lstHmsph = ['rh']

# Path of depth-profiles (meta-condition, ROI, hemisphere, condition, and model
# index left open):
strPthData = '/Users/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}{}.npz'  #noqa

# Condition levels (used to complete file names):
lstCon = ['Pd_sst', 'Ps_sst', 'Cd_sst']
# lstCon = ['Pd_sst', 'Ps_sst', 'Cd_sst', 'Ps_sst_plus_Cd_sst']

# Which conditions to compare (nested list of tuples with condition indices):
lstDiff = [[(0, 1), (0, 2)],
           [(0, 1), (1, 2)],
           [(0, 2), (1, 2)]]

# Number of resampling iterations:
varNumIt = 1000
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Preparations

# List of features for dataframe:
lstFtr = ['ROI', 'Deconvolution', 'pRF-position', 'Hemisphere', 'Contrast',
          'Emp.-peak-pos.-diff.', 'p-value']

# Number of samples:
varNumSmpl = (len(lstMetaCon) * len(lstMdl) * len(lstRoi) * len(lstHmsph)
              * len(lstDiff))

# Create dataframe:
objDf = pd.DataFrame(None, index=np.arange(varNumSmpl), columns=lstFtr)

# Dictionary for dataframe column datatypes:
dicType = {'ROI': str,
           'Deconvolution': str,
           'pRF-position': str,
           'Hemisphere': str,
           'Contrast': str,
           'Emp.-peak-pos.-diff.': np.float64,
           'p-value': np.float64}

objDf.astype(dicType)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Loop through ROIs / conditions

print('-Peak position permutation test')

# Counter for samples:
idxSmpl = 0

# Loop through metaconditions, models, ROIs, hemispheres, and comparisons:
for idxMtaCn in range(len(lstMetaCon)):
    for idxMdl in range(len(lstMdl)):
        for idxRoi in range(len(lstRoi)):
            for idxHmsph in range(len(lstHmsph)):
                for idxDiff in range(len(lstDiff)):

                    # Permutation test:
                    varTmpP, varTmpDiff= \
                        peak_diff(strPthData.format(lstMetaCon[idxMtaCn],
                                                    lstRoi[idxRoi],
                                                    lstHmsph[idxHmsph],
                                                    '{}',
                                                    lstMdl[idxMdl]),
                                  lstDiff[idxDiff],
                                  lstCon,
                                  varNumIt=1000)

                    # Current comparison:
                    strTmp = (lstCon[lstDiff[idxDiff][0][0]]
                              + '-'
                              + lstCon[lstDiff[idxDiff][0][1]]
                              + '_vs_'
                              + lstCon[lstDiff[idxDiff][1][0]]
                              + '-'
                              + lstCon[lstDiff[idxDiff][1][1]])

                    # Result to data frame:
                    objDf.at[idxSmpl, 'ROI'] = lstRoi[idxRoi].upper()
                    if lstMdl[idxMdl] == '':
                        objDf.at[idxSmpl, 'Deconvolution'] = 'No'
                    else:
                        objDf.at[idxSmpl, 'Deconvolution'] = 'Yes'
                    objDf.at[idxSmpl, 'pRF-position'] = lstMetaCon[idxMtaCn]
                    objDf.at[idxSmpl, 'Hemisphere'] = lstHmsph[idxHmsph]
                    objDf.at[idxSmpl, 'Contrast'] = strTmp
                    objDf.at[idxSmpl, 'Emp.-peak-pos.-diff.'] = varTmpDiff
                    objDf.at[idxSmpl, 'p-value'] = varTmpP

                    # Increment counter:
                    idxSmpl += 1

print(' ')
print(objDf)
print(' ')

# ----------------------------------------------------------------------------

print('-Done.')
