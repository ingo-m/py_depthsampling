# -*- coding: utf-8 -*-
"""
Permutation & bootstrapping tests for semisaturation constant depth profiles.

PART 1 - PERMUTATION TEST FOR DIFFERENCES BETWEEN ROIs

Takes the mean intensity across cortical depth, and compares the difference
between this mean between two depth profiles (e.g. the semisaturation constant
depth profile from V1 and V2). The null hypothesis is that the two
distributions are equal (any difference could be due to either the mean or the
variance or both not being equal).

The permutations need to be provided first (they can be created using
ds_permMain).

PART 2 - PREPARE BOOTSTRAPPING LINEAR REGRESSION

The bootstrap linear regression is performed in R. Here, we only prepare an npy
file (containing a np array) that can be read by R and used for the analysis.

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
import multiprocessing as mp
from ds_crfParBoot02 import crf_par_02


# ----------------------------------------------------------------------------
# *** Define parameters

# Corrected or  uncorrected depth profiles?
strCrct = 'corrected'

# Which CRF to use ('power' for power function or 'hyper' for hyperbolic ratio
# function).
strFunc = 'hyper'

# File to load resampling from:
strPthOut = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/crf_permutation_{}_{}.npz'  #noqa

strPthOut = strPthOut.format(strCrct, strFunc)

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using percent (i.e. from zero to 100), the search for the luminance at
# half maximum response would need to be adjusted.
vecEmpX = np.array([0.025, 0.061, 0.163, 0.72])

# Output path for array to be analysed in R (bootstrap regerssion):
strPthSemi = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/aryEmpSemi_{}_{}.npy'.format(strCrct, strFunc)  #noqa


# ----------------------------------------------------------------------------
# *** Load / create resampling

print('-Resampling test on depth profiles')

print('---Loading resampling results')

# Load data from npz file:
objNpz = np.load(strPthOut)

# Retrieve arrays from npz object (dictionary):
aryDpth01 = objNpz['aryDpth01']
aryDpth02 = objNpz['aryDpth02']
aryMdlY = objNpz['aryMdlY']
aryHlfMax = objNpz['aryHlfMax']
arySemi = objNpz['arySemi']
aryRes = objNpz['aryRes']


# ----------------------------------------------------------------------------
# *** Fit CRF on empirical profiles

print('---Calculating mean difference on empirical depth profiles')

# The mean difference on the empirical profiles needs to be calculated for
# comparison with the permutation distribution.

# Number of subject:
varNumSubs = aryDpth01.shape[0]

# Number of conditions:
varNumCon = aryDpth01.shape[1]

# Number of depth levels:
varNumDpt = aryDpth01.shape[2]

# Put the two ROI depth profile arrays into one array of the form
# aryDpth[idxRoi, idxSub, idxCondition, idxDpt]
aryDpth = np.array([aryDpth01, aryDpth02])

# Create a queue to put the results in:
queOut = mp.Queue()

# Pseudo-randomisation array to use the bootstrapping function to get empirical
# CRF fit:
aryRnd = np.arange(0, varNumSubs, 1)
aryRnd = np.array(aryRnd, ndmin=2)

# Number of x-values for which to solve the function when calculating model
# fit:
varNumX = aryMdlY.shape[3]

# Fit contrast response function on empirical depth profiles:
crf_par_02(0,
           aryDpth,
           vecEmpX,
           strFunc,
           aryRnd,
           varNumX,
           queOut)

# Retrieve results from queue:
lstCrf = queOut.get(True)
_, aryEmpMdlY, aryEmpHlfMax, aryEmpSemi, aryEmpRes = lstCrf


# ----------------------------------------------------------------------------
# *** PART 1 - PERMUTATION TEST FOR DIFFERENCES BETWEEN ROIs

print('---PART 1 - PERMUTATION TEST FOR DIFFERENCES BETWEEN ROIs')

# We test for differences in the mean semisaturation constant (mean across
# cortical depth levels) between the two ROIs (e.g. V1 and V2).

# Difference in mean semisaturaion constant across depth levels:
varDiffSemiEmp = np.subtract(np.mean(aryEmpSemi[0, 0, :]),
                             np.mean(aryEmpSemi[1, 0, :]))

# Mean difference in permutation samples (null distribution):
vecNull = np.subtract(np.mean(arySemi[0, :, :], axis=1),
                      np.mean(arySemi[1, :, :], axis=1))

# Absolute of the mean difference between the two randomised groups:
vecNullAbs = np.absolute(vecNull)

# Number of resampled cases with absolute mean difference that is at least as
# large as the empirical mean difference (permutation p-value):
varNumGe = np.sum(np.greater_equal(vecNullAbs, varDiffSemiEmp))

print('------Number of resampled cases with absolute mean difference that is')
print(('      at least as large as the empirical mean difference: '
       + str(varNumGe)))

# Total number of resampling interations:
varNumIt = arySemi.shape[1]

print('------Total number of resampling iterations: ' + str(varNumIt))

# Ratio of resampled cases with absolute mean difference that is at least as
# large as the empirical mean difference (permutation p-value):
varP = np.divide(float(varNumGe), float(varNumIt))

print(('------Permutation p-value for equality of distributions: '
       + str(np.around(varP, decimals=5))))


# ----------------------------------------------------------------------------
# *** PART 2 - PREPARE BOOTSTRAPPING LINEAR REGRESSION

# The bootstrap linear regression is performed in R. Here, we only prepare an
# npy file (containing a np array) that can be read by R and used for the
# analysis.

# Array to be used in R for bootstrap linear regression, of the form
# aryEmpSemiR[idxDpt, 3], where the first dimension corresponds to the number
# of cortical depth levels * number of ROIs, and the second dimension
# corresponds to three columns for the linear model, representing: the signal
# (i.e. the semisaturation constant, which is the idenpendent variable), the
# depth level (dependent variable) and the ROI membership (dependent variable).
aryEmpSemiR = np.zeros(((varNumDpt * 2), 3))

# The first column is to contain the semisaturation constants:
aryEmpSemiR[:, 0] = aryEmpSemi.flatten()

# The second column is to contain the depth level:
aryEmpSemiR[0:varNumDpt, 1] = np.arange(0, varNumDpt)
aryEmpSemiR[varNumDpt:, 1] = np.arange(0, varNumDpt)

# The third column is to contain the ROI labels (1 for V1, 2 for V2):
aryEmpSemiR[0:varNumDpt, 2] = 0.0
aryEmpSemiR[varNumDpt:, 2] = 1.0

# Save array to disk:
np.save(strPthSemi, aryEmpSemiR)


# ----------------------------------------------------------------------------

print('-Done.')
