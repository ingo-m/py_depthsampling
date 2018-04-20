# -*- coding: utf-8 -*-
"""
Permutation test on depth profiles for differences between ROIs.

Can be used to test for differences in depth profiles between ROIs, e.g. on
depth profiles of semisaturation constant bwetween V1 and V2. (Differences
across cortical depths are ignored, use ds_bootLinReg.py & ds_bootLinReg.r
instead).

Takes the mean intensity across cortical depth, and compares the difference
between this mean between two depth profiles (e.g. the semisaturation constant
depth profile from V1 and V2). The null hypothesis is that the two
distributions are equal (any difference could be due to either the mean or the
variance or both not being equal).

The permutations need to be provided first (they can be created using
ds_permMain).

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
strFunc = 'power'

# File to load resampling from:
strPthIn = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/crf_permutation_{}_{}.npz'  #noqa

strPthIn = strPthIn.format(strCrct, strFunc)

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using percent (i.e. from zero to 100), the search for the luminance at
# half maximum response would need to be adjusted.
vecEmpX = np.array([0.025, 0.061, 0.163, 0.72])

# Test semisaturation constant depth profiles, or residual variance depth
# profiles ('semi' or 'residuals')?
strSwitch = 'semi'


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
# *** Permutation test for differences between ROIs

print('---Permutation test for differences between ROIs')

# We test for differences in the mean semisaturation constant (mean across
# cortical depth levels) between the two ROIs (e.g. V1 and V2).

if strSwitch == 'semi':

    # Absolute difference in mean semisaturaion constant across depth levels:
    varDiffEmp = np.absolute(np.subtract(np.mean(aryEmpSemi[0, 0, :]),
                                         np.mean(aryEmpSemi[1, 0, :])))

    # Mean difference in permutation samples (null distribution):
    vecNull = np.subtract(np.mean(arySemi[0, :, :]),  # axis=1),
                          np.mean(arySemi[1, :, :]))  # axis=1))


elif strSwitch == 'residuals':

    # Absolute difference in mean semisaturaion constant across depth levels:
    varDiffEmp = np.absolute(np.subtract(np.mean(aryEmpRes[0, 0, :]),
                                         np.mean(aryEmpRes[1, 0, :])))

    # Mean difference in permutation samples (null distribution):
    vecNull = np.subtract(np.mean(aryRes[0, :, :]),  # axis=1),
                          np.mean(aryRes[1, :, :]))  # axis=1))

# Absolute of the mean difference between the two randomised groups:
vecNullAbs = np.absolute(vecNull)

# Number of resampled cases with absolute mean difference that is at least as
# large as the empirical mean difference (permutation p-value):
varNumGe = np.sum(np.greater_equal(vecNullAbs, varDiffEmp))

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

print('-Done.')
