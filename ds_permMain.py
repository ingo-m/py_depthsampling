# -*- coding: utf-8 -*-
"""
Permutation test for difference in peak position for half-maximum response.

The purpose of this script is to performe a permutation hypothesis test for a
difference in the peak position in the cortical depth profiles of the response
at half-maximum contrast between V1 and V2. More specifically, the equality of
distributions of the peak positions is tested (i.e. the a possible difference
could be due to a difference in means, variance, or the shape of the
distribution).

The procedure is as follow:
- Condition labels (i.e. V1 & V2) are permuted within subjects for each
  permutation data set (i.e. on each iteration).
- For each permutation dataset, the mean depth profile of the two groups (i.e.
  randomly created 'V1' and 'V2' assignments) are calculated.
- The contrast response function (CRF) is fitted for the two mean depth
  profiles.
- The response at half-maximum contrast is calculated from the fitted CRF.
- The peak of the half-maximum contrast profile is identified for both
  randomised groups.
- The mean difference in peak position between the two randomised groups is the  
  null distribution.
- The peak difference on the full profile is calculated, and the permutation
  p-value with respect to the null distribution is produced.

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
# import json
from ds_permCrf import perm_hlf_max_peak
from ds_findPeak import find_peak


# ----------------------------------------------------------------------------
# *** Define parameters

# Use existing resampling results or create new one ('load' or 'create')?
strSwitch = 'load'

# Corrected or  uncorrected depth profiles?
strCrct = 'uncorrected'

# Which CRF to use ('power' for power function or 'hyper' for hyperbolic ratio
# function).
strFunc = 'power'

# File to load resampling from / save resampling to (corrected/uncorrected and
# power/hyper left open):
strPthOut = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/crf_permutation_{}_{}.npz'  #noqa

strPthOut = strPthOut.format(strCrct, strFunc)

# Path of depth-profiles:
if strCrct == 'uncorrected':
    objDpth01 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy'  #noqa
    objDpth02 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2.npy'  #noqa
if strCrct == 'corrected':
    objDpth01 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1_corrected.npy'  #noqa
    objDpth02 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2_corrected.npy'  #noqa

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using percent (i.e. from zero to 100), the search for the luminance at
# half maximum response would need to be adjusted.
vecEmpX = np.array([0.025, 0.061, 0.163, 0.72])

# Number of resampling iterations:
varNumIt=3000

# Number of processes to run in parallel:
varPar=11


# ----------------------------------------------------------------------------
# *** Load / create resampling

print('-Permutation CRF fitting')

if strSwitch == 'load':

    print('---Loading bootstrapping results')

    # Load previously prepared file:
    # with open(strPthJson, 'r') as objJson:
    #      lstJson = json.load(objJson)

    # Retrieve numpy arrays from nested list:
    # aryDpth01 = np.array(lstJson[0])
    # aryDpth02 = np.array(lstJson[1])
    # aryMdlY = np.array(lstJson[2])
    # aryHlfMax = np.array(lstJson[3])
    # arySemi = np.array(lstJson[4])
    # aryRes = np.array(lstJson[5])

    # Load data from npz file:
    objNpz = np.load(strPthOut)

    # Retrieve arrays from npz object (dictionary):
    aryDpth01 = objNpz['aryDpth01']
    aryDpth02 = objNpz['aryDpth02']
    aryMdlY = objNpz['aryMdlY']
    aryHlfMax = objNpz['aryHlfMax']
    arySemi = objNpz['arySemi']
    aryRes = objNpz['aryRes']

elif strSwitch == 'create':

    # ------------------------------------------------------------------------
    # *** Parallelised permutation & CRF fitting

    print('---Parallelised permutation & CRF fitting')

    aryDpth01, aryDpth02, aryMdlY, aryHlfMax, arySemi, aryRes = \
        perm_hlf_max_peak(objDpth01, objDpth02, vecEmpX, strFunc=strFunc,
                          varNumIt=varNumIt, varPar=varPar)

    # ------------------------------------------------------------------------
    # *** Save results

    print('---Saving bootstrapping results as npz object')

    # Put results into nested list:
    # lstJson = [aryDpth01.tolist(),  # Original depth profiles V1
    #            aryDpth02.tolist(),  # Original depth profiles V2
    #            aryMdlY.tolist(),    # Fitted y-values
    #            aryHlfMax.tolist(),  # Predicted response at 50% contrast
    #            arySemi.tolist(),    # Semisaturation contrast
    #            aryRes.tolist()]     # Residual variance

    # Save results to disk:
    # with open(strPthJson, 'w') as objJson:
    #      json.dump(lstJson, objJson)

    # Save result as npz object:
    np.savez(strPthOut,
             aryDpth01=aryDpth01,
             aryDpth02=aryDpth02,
             aryMdlY=aryMdlY,
             aryHlfMax=aryHlfMax,
             arySemi=arySemi,
             aryRes=aryRes)


# ----------------------------------------------------------------------------
# *** Find peaks in contrast at half maximum profiles


# Find peaks in first permutation group:
vecPeaks01, vecPos01 = find_peak(aryHlfMax[0, :, :], lgcPos=True)

# Find peaks in second permutation group:
vecPeaks02, vecPos02 = find_peak(aryHlfMax[1, :, :], lgcPos=True)

# Put peak locations together, array of the form aryPeak[idxIteration, idxRoi]:
aryPeaks = np.zeros((varNumIt, 2))
aryPeaks[vecPos01, 0] = vecPeaks01
aryPeaks[vecPos02, 1] = vecPeaks02

# Identify cases for which a peak has been identified for both groups:
vecCon = np.greater(np.multiply(aryPeaks[:, 0], aryPeaks[:, 1]),
                    0.0)

# Select cases with peaks for both groups:
vecPeaks01 = aryPeaks[vecCon, 0]
vecPeaks02 = aryPeaks[vecCon, 1]


# ----------------------------------------------------------------------------
# *** Create null distribution

# The mean difference in peak position between the two randomised groups is the  
# null distribution.
vecNull = np.subtract(vecPeaks01, vecPeaks02)


# ----------------------------------------------------------------------------
# *** Empirical mean difference

# The peak difference on the full profile needs to be calculated for comparison
# with the null distribution.

























print('-Done.')