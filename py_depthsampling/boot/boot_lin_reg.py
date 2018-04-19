# -*- coding: utf-8 -*-
"""
Prepare bootstrapping linear regression in R.

Can be used to test for differences in depth profiles between ROIs and for
differences across cortical depth, e.g. on depth profiles of semisaturation
constant bwetween V1 and V2. (Or, for instance, on residual variance depth
profiles & response at 50% contrast profiles.)

The bootstrap linear regression is performed in R. Here, we only prepare an npy
file (containing an np array) that can be read by R.

Function of the depth sampling pipeline.
"""

# Part of py_depthsampling library
# Copyright (C) 2017  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from ds_crfFit import crf_fit


# ----------------------------------------------------------------------------
# *** Define parameters

# Corrected or  uncorrected depth profiles?
strCrct = 'uncorrected'

# Which CRF to use ('power' for power function or 'hyper' for hyperbolic ratio
# function).
strFunc = 'power'

# File to load resampling from:
strPthIn = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/crf_permutation_{}_{}.npz'  #noqa

strPthIn = strPthIn.format(strCrct, strFunc)

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using percent (i.e. from zero to 100), the search for the luminance at
# half maximum response would need to be adjusted.
vecEmpX = np.array([0.025, 0.061, 0.163, 0.72])

# Output path for array to be analysed in R (bootstrap regerssion):
strPthOt = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/R_aryEmp{}_{}_{}.npy'


# ----------------------------------------------------------------------------
# *** Load resampling

print('-Resampling test on depth profiles')

print('---Loading resampling results')

# Load data from npz file:
objNpz = np.load(strPthIn)

# Retrieve arrays from npz object (dictionary):
aryDpth01 = objNpz['aryDpth01']
aryDpth02 = objNpz['aryDpth02']
aryMdlY = objNpz['aryMdlY']
aryHlfMax = objNpz['aryHlfMax']
arySemi = objNpz['arySemi']
aryRes = objNpz['aryRes']


# ----------------------------------------------------------------------------
# *** Fit CRF on empirical profiles

print('---Fit CRF on empirical depth profiles')

# The mean difference on the empirical profiles needs to be calculated for
# comparison with the permutation distribution.

# Number of subject:
varNumSubs = aryDpth01.shape[0]

# Number of conditions:
varNumCon = aryDpth01.shape[1]

# Number of depth levels:
varNumDpt = aryDpth01.shape[2]

# Array for single-subject estimates of semisaturation constant for both ROIs:
arySemiRoi01 = np.zeros((varNumSubs, varNumDpt))
arySemiRoi02 = np.zeros((varNumSubs, varNumDpt))

# Array for single-subject estimates of 50% contrast response:
aryHlfMaxRoi01 = np.zeros((varNumSubs, varNumDpt))
aryHlfMaxRoi02 = np.zeros((varNumSubs, varNumDpt))

# Array for residual variance at empirical contrast levels:
aryResRoi01 = np.zeros((varNumSubs, varNumDpt, varNumCon))
aryResRoi02 = np.zeros((varNumSubs, varNumDpt, varNumCon))

# Fit CRF for each subject & depth level:
for idxSub in range(varNumSubs):

    print(('---Subject: ' + str(idxSub)))

    for idxDpt in range(varNumDpt):

        print(('------Depth: ' + str(idxDpt)))

        # Temporary array to fit required input dimensions:
        aryTmp = aryDpth01[idxSub, :, idxDpt].reshape(1, varNumCon)

        # CRF fitting for current subject & depth level, ROI 1:
        _, aryHlfMaxRoi01[idxSub, idxDpt], arySemiRoi01[idxSub, idxDpt], \
            aryResRoi01[idxSub, idxDpt, :] = crf_fit(vecEmpX,
                                                     aryTmp,
                                                     strFunc=strFunc,
                                                     varNumX=aryMdlY.shape[3])

        # Temporary array to fit required input dimensions:
        aryTmp = aryDpth02[idxSub, :, idxDpt].reshape(1, varNumCon)

        # CRF fitting for current subject & depth level, ROI 2:
        _, aryHlfMaxRoi02[idxSub, idxDpt], arySemiRoi02[idxSub, idxDpt], \
            aryResRoi02[idxSub, idxDpt, :] = crf_fit(vecEmpX,
                                                     aryTmp,
                                                     strFunc=strFunc,
                                                     varNumX=aryMdlY.shape[3])


# ----------------------------------------------------------------------------
# *** PREPARE BOOTSTRAPPING LINEAR REGRESSION

# The bootstrap linear regression is performed in R. Here, we only prepare an
# npy file (containing a np array) that can be read by R and used for the
# analysis.

# Array to be used in R for bootstrap linear regression, of the form
# aryR[(idxSub * idxDpt * idxRoi), 4], where the first dimension corresponds to
# the number of subjects * cortical depth levels * ROIs, and the second
# dimension corresponds to four columns for the linear model, representing: the
# signal (e.g. the semisaturation constant or response at 50% contrast, which
# is the dependent variable), the depth level (independent variable), the ROI
# membership (independent variable), and the subject number (independent
# variable).
aryR = np.zeros(((varNumSubs * varNumDpt * 2), 4))

# The second column is to contain the depth level:
aryR[:, 1] = np.tile(np.arange(0, varNumDpt), (varNumSubs * 2))

# Effect coding; subtraction of mean through column so that parameter estimates
# reflect effect at mean depth level.
aryR[:, 1] = np.subtract(aryR[:, 1],
                                np.mean(aryR[:, 1]))

# The third column is to contain the ROI labels (-1 for V1, 1 for V2):
vecTmp = np.array(([-1.0] * (varNumDpt * varNumSubs),
                   [1.0] * (varNumDpt * varNumSubs))).flatten()
aryR[:, 2] = np.copy(vecTmp)

# The fourth column is to contain the subject ID:
vecTmp = np.arange(0, varNumSubs).repeat(varNumDpt)
# Tile twice (for two ROIs):
vecTmp = np.tile(vecTmp, 2)
aryR[:, 3] = np.copy(vecTmp)

# Ouput array for R - response at 50% contrast:
aryHlfMaxRoiR = np.copy(aryR)
# First column - dependent data:
aryHlfMaxRoiR[:, 0] = np.array((aryHlfMaxRoi01, aryHlfMaxRoi02)).flatten()

# Ouput array for R - semisaturation contrast:
arySemiRoiR = np.copy(aryR)
# First column - dependent data:
arySemiRoiR[:, 0] = np.array((arySemiRoi01, arySemiRoi02)).flatten()

# Model residuals - take mean across conditions:
aryResRoi01 = np.mean(aryResRoi01, axis=2)
aryResRoi02 = np.mean(aryResRoi02, axis=2)

# Ouput array for R - semisaturation contrast:
aryResRoiR = np.copy(aryR)
# First column - dependent data:
aryResRoiR[:, 0] = np.array((aryResRoi01, aryResRoi02)).flatten()

# Save arrays to disk:
np.save(strPthOt.format('HlfMax', strCrct, strFunc),
        aryHlfMaxRoiR)
np.save(strPthOt.format('Semi', strCrct, strFunc),
        arySemiRoiR)
np.save(strPthOt.format('Resd', strCrct, strFunc),
        aryResRoiR)


# ----------------------------------------------------------------------------

print('-Done.')
