# -*- coding: utf-8 -*-
"""
Permutation test for difference in peak position for half-maximum response.

The purpose of this script is to performe a permutation hypothesis test for a
difference in the peak position in the cortical depth profiles of the response
at half-maximum contrast between V1 and V2. More specifically, the equality of
distributions of the peak positions is tested (i.e. a possible difference could
be due to a difference in means, variance, or the shape of the distribution).

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
import multiprocessing as mp
from ds_permCrf import perm_hlf_max_peak
from ds_findPeak import find_peak
from ds_crfParBoot02 import crf_par_02


# ----------------------------------------------------------------------------
# *** Define parameters

# Use existing resampling results or create new one ('load' or 'create')?
strSwitch = 'create'

# Corrected or  uncorrected depth profiles?
strCrct = 'corrected'

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
    objDpth01 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1_corrected_model_1.npy'  #noqa
    objDpth02 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2_corrected_model_1.npy'  #noqa

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using percent (i.e. from zero to 100), the search for the luminance at
# half maximum response would need to be adjusted.
vecEmpX = np.array([0.025, 0.061, 0.163, 0.72])

# Number of x-values for which to solve the function when calculating model
# fit:
varNumX = 1000

# Number of resampling iterations:
varNumIt = 10000

# Number of processes to run in parallel:
varPar = 11


# ----------------------------------------------------------------------------
# *** Load / create resampling

print('-Permutation CRF fitting')
print(('--Profiles: ' + strCrct.upper()))
print(('--Function: ' + strFunc))

if strSwitch == 'load':

    print('---Loading resampling results')

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

    print('---Parallelised resampling & CRF fitting')

    aryDpth01, aryDpth02, aryMdlY, aryHlfMax, arySemi, aryRes = \
        perm_hlf_max_peak(objDpth01, objDpth02, vecEmpX, strFunc=strFunc,
                          varNumIt=varNumIt, varPar=varPar)

    # ------------------------------------------------------------------------
    # *** Save results

    print('---Saving resampling results as npz object')

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

print('---Find peaks in resampled contrast-at-half-maximum profiles, ROI 1')

# Find peaks in first permutation group:
vecPeaks01 = find_peak(aryHlfMax[0, :, :])

print('---Find peaks in resampled contrast-at-half-maximum profiles, ROI 2')

# Find peaks in second permutation group:
vecPeaks02 = find_peak(aryHlfMax[1, :, :])


# ----------------------------------------------------------------------------
# *** Create null distribution

print('---Create null distribution')

# The mean difference in peak position between the two randomised groups is the
# null distribution.
vecNull = np.subtract(vecPeaks01, vecPeaks02)


# ----------------------------------------------------------------------------
# *** Empirical mean difference

print('---Find peaks in empirical contrast-at-half-maximum profiles')

# The peak difference on the full profile needs to be calculated for comparison
# with the null distribution.

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

# Find peaks in empirical CRF fit:
vecEmpPeaks01 = find_peak(aryEmpHlfMax[0, :, :])
print(('------Peak in empirical contrast-at-half-maximum profile, ROI 1: '
       + str(np.around(np.multiply(vecEmpPeaks01[0], 100.0), 2))
       + '%'))

# Find peaks in second permutation group:
vecEmpPeaks02 = find_peak(aryEmpHlfMax[1, :, :])
print(('------Peak in empirical contrast-at-half-maximum profile, ROI 2: '
       + str(np.around(np.multiply(vecEmpPeaks02[0], 100.0), 2))
       + '%'))

# Absolute peak difference in empirical profiles:
varPeakDiff = np.absolute(np.subtract(vecEmpPeaks01, vecEmpPeaks02))


# ----------------------------------------------------------------------------
# *** Calculate p-value

print('---Calculate p-value')

# Absolute of the mean difference in peak position between the two randomised
# groups (null distribution).
vecNullAbs = np.absolute(vecNull)

# Number of resampled cases with absolute peak position difference that is at
# least as large as the empirical peak difference (permutation p-value):
varNumGe = np.sum(np.greater_equal(vecNullAbs, varPeakDiff))

print('------Number of resampled cases with absolute peak position')
print('      difference that is at least as large as the empirical peak')
print(('      difference: '
      + str(varNumGe)))

# Total number of resampled cases with identified peak position:
varNumPk = np.shape(vecNull)[0]

print('------Total number of resampled cases with identified peak position:')
print(('      ' + str(varNumPk)))

# Ratio of resampled cases with absolute peak position difference that is at
# least as large as the empirical peak difference (permutation p-value):
varP = np.divide(float(varNumGe), float(varNumPk))

print('------Permutation p-value for equality of distributions of peak')
print('      position of contrast-at-half-maximum response depth profiles')
print(('      between the two ROIs: '
       + str(np.around(varP, decimals=5))))
# ----------------------------------------------------------------------------

print('-Done.')
