# -*- coding: utf-8 -*-
"""
Permutation test for difference in shape of cortical depth profiles.

Performe a permutation hypothesis test for a difference in the shape of depth
profiles of condition differences between ROIs. In other words, it is tested
whether the peak position of the contrast condition A vs. condition B is the
same between two ROIs (e.g. V1 and V2).

A second-degree polynomial function (i.e. quadratic function) is fitted to the
cortical depth profiles.

More specifically, the equality of distributions of the peak positions is
tested (i.e. a possible difference could be due to a difference in means,
variance, or the shape of the distribution).

Because ROI/condition labels are permuted within subjects, single subject depth
profiles need to be provided. The input are npz files with two dimensions
(subject and depth levels).

The procedure is as follow:
- Condition labels are permuted within subjects for each permutation data set
  (i.e. on each iteration).
- For each permutation dataset, the mean depth profile of the two randomised
  groups are calculated.
- The peak of the depth profiles is identified for both randomised groups.
- The mean difference in peak position between the two randomised groups is the
  null distribution.
- The peak difference on the empirical profile is calculated, and the
  permutation p-value with respect to the null distribution is produced.

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


import itertools
import numpy as np
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------------
# *** Define parameters

## Path of depth-profiles - first ROI, first condition:
#strRoi01Con01 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v1_rh_Pd_sst_deconv_model_1.npz'
## Path of depth-profiles - first ROI, second condition:
#strRoi01Con02 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v1_rh_Cd_sst_deconv_model_1.npz'
## Path of depth-profiles - second ROI, first condition:
#strRoi02Con01 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v2_rh_Pd_sst_deconv_model_1.npz'
## Path of depth-profiles - second ROI, second condition:
#strRoi02Con02 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v2_rh_Cd_sst_deconv_model_1.npz'

# Path of depth-profiles - first ROI, first condition:
strRoi01Con01 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v1_rh_Pd_sst_deconv_model_1.npz'
# Path of depth-profiles - first ROI, second condition:
strRoi01Con02 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v1_rh_Ps_sst_plus_Cd_sst_deconv_model_1.npz'
# Path of depth-profiles - second ROI, first condition:
strRoi02Con01 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v2_rh_Pd_sst_deconv_model_1.npz'
# Path of depth-profiles - second ROI, second condition:
strRoi02Con02 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v2_rh_Ps_sst_plus_Cd_sst_deconv_model_1.npz'

# Number of resampling iterations (set to `None` in case of small enough sample
# size for exact test, otherwise Monte Carlo resampling is performed):
varNumIt = None


# ----------------------------------------------------------------------------
# *** 2nd degree polynomial function

def funcPoly2(varX, varA, varB, varC):
    """2nd degree polynomial function to be fitted to the data."""
    varOut = (varA * np.power(varX, 2) +
              varB * np.power(varX, 1) +
              varC)
    return varOut


# ----------------------------------------------------------------------------
# *** Load depth profiles

print('-Peak position permutation test')

if not(varNumIt is None):
    print(('--Resampling iterations: ' + str(varNumIt)))

# Load single-condition depth profiles from npz files:
objNpzTmp = np.load(strRoi01Con01)
aryRoi01Con01 = objNpzTmp['arySubDpthMns']

# Array with number of vertices (for weighted averaging across subjects),
# shape: vecNumInc[subjects].
vecNumIncRoi01 = objNpzTmp['vecNumInc']

# Load single-condition depth profiles from npz files:
objNpzTmp = np.load(strRoi01Con02)
aryRoi01Con02 = objNpzTmp['arySubDpthMns']

# Load single-condition depth profiles from npz files:
objNpzTmp = np.load(strRoi02Con01)
aryRoi02Con01 = objNpzTmp['arySubDpthMns']

# Array with number of vertices (for weighted averaging across subjects),
# shape: vecNumInc[subjects].
vecNumIncRoi02 = objNpzTmp['vecNumInc']

# Load single-condition depth profiles from npz files:
objNpzTmp = np.load(strRoi02Con02)
aryRoi02Con02 = objNpzTmp['arySubDpthMns']


# ----------------------------------------------------------------------------
# *** Create condition contrasts

# Within-subject condition contrast, first ROI:
aryCtrRoi01 = np.subtract(aryRoi01Con01, aryRoi01Con02)

# Within-subject condition contrast, second ROI:
aryCtrRoi02 = np.subtract(aryRoi02Con01, aryRoi02Con02)

# Number of subject:
varNumSubs = aryCtrRoi01.shape[0]

# Number of depth levels:
varNumDpt = aryCtrRoi01.shape[1]


# ----------------------------------------------------------------------------
# *** Empirical mean difference

print('---Fit quadratic function to empirical depth profiles')

# The vertex (maximum quadratic function) on the empirical profiles needs to be
# calculated for comparison with the permutation null distribution.

# Weighted mean across subjects:
vecDpthMne01 = np.average(aryCtrRoi01, axis=0, weights=vecNumIncRoi01)
vecDpthMne02 = np.average(aryCtrRoi02, axis=0, weights=vecNumIncRoi02)

# Independent variable data:
vecInd = np.linspace(0.0, 1.0, num=varNumDpt)

# Fit 2nd degree polynomial function:
vecPoly2ModelPar01, vecPoly2ModelCov01 = curve_fit(funcPoly2,
                                                   vecInd,
                                                   vecDpthMne01)
vecPoly2ModelPar02, vecPoly2ModelCov02 = curve_fit(funcPoly2,
                                                   vecInd,
                                                   vecDpthMne02)

# Calculate fitted values:
vecFittedPoly01 = funcPoly2(vecInd,
                            vecPoly2ModelPar01[0],
                            vecPoly2ModelPar01[1],
                            vecPoly2ModelPar01[2])
vecFittedPoly02 = funcPoly2(vecInd,
                            vecPoly2ModelPar02[0],
                            vecPoly2ModelPar02[1],
                            vecPoly2ModelPar02[2])

# Combine arrays for visualisation in ipython:
# ary01 = np.array([vecDpthMne01, vecFittedPoly01]).T
# ary02 = np.array([vecDpthMne02, vecFittedPoly02]).T

# Vertex (maximum or minimum) of polynomial, analytical solution:
varEmpVertx01 = -vecPoly2ModelPar01[1] / (2.0 * vecPoly2ModelPar01[0])
varEmpVertx02 = -vecPoly2ModelPar02[1] / (2.0 * vecPoly2ModelPar02[0])

# Absolute vertex difference in empirical profiles of condition contrast:
varEmpVertxDiff = np.absolute(np.subtract(varEmpVertx01, varEmpVertx02))

print(('------Vertex positions in mean empirical profiles, ROI 1:  '
       + str(np.around(varEmpVertx01, decimals=2))))
print(('------Vertex positions in mean empirical profiles, ROI 2:  '
       + str(np.around(varEmpVertx02, decimals=2))))
print(('------Absolute difference in vertex positions (empirical): '
       + str(np.around(varEmpVertxDiff, decimals=2))))


# ----------------------------------------------------------------------------
# *** Create permutation samples

print('---Create permutation samples')

# Random array that is used to permute V1 and V2 labels within subjects, of the
# form aryRnd[idxIteration, idxSub]. For each iteration and subject, there is
# either a zero or a one. 'Zero' means that the actual V1 value gets assigned
# to the permuted 'V1' group and the actual V2 value gets assigned to the
# permuted 'V2' group. 'One' means that the labels are switched, i.e. the
# actual V1 label get assignet to the 'V2' group and vice versa.
if not(varNumIt is None):
    # Monte Carlo resampling:
    aryRnd = np.random.randint(0, high=2, size=(varNumIt, varNumSubs))
else:
    # In case of tractable number of permutations, create a list of all
    # possible permutations (Bernoulli sequence).
    lstBnl = list(itertools.product([0, 1], repeat=varNumSubs))
    aryRnd = np.array(lstBnl)
    # Number of resampling cases:
    varNumIt = len(lstBnl)
    print('------Testing complete set of possible resampling combinations.')
print(('------Number of combinations: ' + str(varNumIt)))

# We need two versions of the randomisation array, one for sampling from the
# first input array (e.g. V1), and a second version to sample from the second
# input array (e.g. V2). (I.e. the second version is the opposite of the first
# version.)
aryRnd01 = np.equal(aryRnd, 1)
aryRnd02 = np.equal(aryRnd, 0)
del(aryRnd)

# Arrays with permuted depth profiles for the two randomised groups:
aryDpthRnd01 = np.zeros((varNumIt, varNumSubs, varNumDpt))
aryDpthRnd02 = np.zeros((varNumIt, varNumSubs, varNumDpt))

# Arrays for number of vertices for randomised groups:
aryNumIncRnd01 = np.zeros((varNumIt, varNumSubs, varNumDpt))
aryNumIncRnd02 = np.zeros((varNumIt, varNumSubs, varNumDpt))

# Loop through iterations:
for idxIt in range(varNumIt):

    # Assign values from original group 1 to permutation group 1:
    aryDpthRnd01[idxIt, aryRnd01[idxIt, :], :] = \
        aryCtrRoi01[aryRnd01[idxIt, :], :]

    # Assign values from original group 2 to permutation group 1:
    aryDpthRnd01[idxIt, aryRnd02[idxIt, :], :] = \
        aryCtrRoi02[aryRnd02[idxIt, :], :]

    # Assign values from original group 1 to permutation group 2:
    aryDpthRnd02[idxIt, aryRnd02[idxIt, :], :] = \
        aryCtrRoi01[aryRnd02[idxIt, :], :]

    # Assign values from original group 2 to permutation group 2:
    aryDpthRnd02[idxIt, aryRnd01[idxIt, :], :] = \
        aryCtrRoi02[aryRnd01[idxIt, :], :]

    # Assign number of vertices from original group 1 to permutation group 1:
    aryNumIncRnd01[idxIt, aryRnd01[idxIt, :], :] = \
        vecNumIncRoi01[aryRnd01[idxIt, :], None]

    # Assign number of vertices from original group 2 to permutation group 1:
    aryNumIncRnd01[idxIt, aryRnd02[idxIt, :], :] = \
        vecNumIncRoi02[aryRnd02[idxIt, :], None]

    # Assign number of vertices from original group 1 to permutation group 2:
    aryNumIncRnd02[idxIt, aryRnd02[idxIt, :], :] = \
        vecNumIncRoi01[aryRnd02[idxIt, :], None]

    # Assign number of vertices from original group 2 to permutation group 2:
    aryNumIncRnd02[idxIt, aryRnd01[idxIt, :], :] = \
        vecNumIncRoi02[aryRnd01[idxIt, :], None]

# Take mean across subjects in permutation samples:
aryDpthRnd01 = np.average(aryDpthRnd01, axis=1, weights=aryNumIncRnd01)
aryDpthRnd02 = np.average(aryDpthRnd02, axis=1, weights=aryNumIncRnd02)


# ----------------------------------------------------------------------------
# *** Find vertices in permutation samples

print('---Find vertices in permutation samples & create null distribution.')

# Array for vertex differences in permutation samples - null distribution.
vecNull = np.zeros(varNumIt)

# Loop through iterations:
for idxIt in range(varNumIt):

    # Fit 2nd degree polynomial function:
    vecPoly2ModelPar01, vecPoly2ModelCov01 = curve_fit(funcPoly2,
                                                       vecInd,
                                                       aryDpthRnd01[idxIt, :])
    vecPoly2ModelPar02, vecPoly2ModelCov02 = curve_fit(funcPoly2,
                                                       vecInd,
                                                       aryDpthRnd02[idxIt, :])

    # Vertex (maximum or minimum) of polynomial, analytical solution:
    varEmpVertx01 = -vecPoly2ModelPar01[1] / (2.0 * vecPoly2ModelPar01[0])
    varEmpVertx02 = -vecPoly2ModelPar02[1] / (2.0 * vecPoly2ModelPar02[0])

    # Difference in vertex position (along x-axis):
    vecNull[idxIt] = np.subtract(varEmpVertx01, varEmpVertx02)


# ----------------------------------------------------------------------------
# *** Calculate p-value

print('---Calculate p-value')

# Absolute of the mean difference in peak position between the two randomised
# groups (null distribution).
vecNullAbs = np.absolute(vecNull)

# Number of resampled cases with absolute peak position difference that is at
# least as large as the empirical peak difference:
varNumGe = np.sum(np.greater_equal(vecNullAbs,
                                   varEmpVertxDiff),
                  axis=0)

print('------Number of resampled cases with absolute peak position')
print('      difference that is at least as large as the empirical peak')
print(('      difference: '
      + str(varNumGe)))

# Ratio of resampled cases with absolute peak position difference that is at
# least as large as the empirical peak difference (permutation p-value):
vecP = np.divide(float(varNumGe), float(varNumIt))

print('------Permutation p-value for equality of distributions of peak')
print(('      position between the two ROIs: '
       + str(np.around(vecP, decimals=4))))
# ----------------------------------------------------------------------------

print('-Done.')
