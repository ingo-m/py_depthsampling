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
import json
from ds_permCrf import perm_hlf_max_peak


# ----------------------------------------------------------------------------
# *** Define parameters

# Use existing resampling results or create new one ('load' or 'create')?
strSwitch = 'create'

# Corrected or  uncorrected depth profiles?
strCrct = 'uncorrected'

# Which CRF to use ('power' for power function or 'hyper' for hyperbolic ratio
# function).
strFunc = 'power'

# JSON to load resampling from / save resampling to (corrected/uncorrected and
# power/hyper left open):
strPthJson = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/crf_permutation_{}_{}.json'  #noqa

strPthJson = strPthJson.format(strCrct, strFunc)

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
varNumIt=2000

# Number of processes to run in parallel:
varPar=10


# ----------------------------------------------------------------------------
# *** Load / create resampling

print('-Permutation CRF fitting')

if strSwitch == 'load':

    print('---Loading bootstrapping results')

    # Load previously prepared file:
    with open(strPthJson, 'r') as objJson:
         lstJson = json.load(objJson)

    # Retrieve numpy arrays from nested list:
    aryDpth01 = np.array(lstJson[0])
    aryDpth02 = np.array(lstJson[1])
    aryMdlY = np.array(lstJson[2])
    aryHlfMax = np.array(lstJson[3])
    arySemi = np.array(lstJson[4])
    aryRes = np.array(lstJson[5])

elif strSwitch == 'create':

    # ------------------------------------------------------------------------
    # *** Parallelised permutation & CRF fitting

    print('---Parallelised permutation & CRF fitting')

    aryDpth01, aryDpth02, aryMdlY, aryHlfMax, arySemi, aryRes = \
        perm_hlf_max_peak(objDpth01, objDpth02, vecEmpX, strFunc='power',
                          varNumIt=1000, varPar=10)

    # ------------------------------------------------------------------------
    # *** Save results

    print('---Saving bootstrapping results as json object')

    # Put results into nested list:
    lstJson = [aryDpth01.tolist(),  # Original depth profiles V1
               aryDpth02.tolist(),  # Original depth profiles V2
               aryMdlY.tolist(),    # Fitted y-values
               aryHlfMax.tolist(),  # Predicted response at 50% contrast
               arySemi.tolist(),    # Semisaturation contrast
               aryRes.tolist()]     # Residual variance

    # Save results to disk:
    with open(strPthJson, 'w') as objJson:
         json.dump(lstJson, objJson)


print('-Done.')