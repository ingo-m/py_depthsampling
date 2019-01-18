# -*- coding: utf-8 -*-
"""
Permutation test for difference in peak position for cortical depth profiles.

Performe a permutation hypothesis test for a difference in the peak position in
cortical depth profiles between ROIs or experimental conditions. More
specifically, the equality of distributions of the peak positions is tested
(i.e. a possible difference could be due to a difference in means, variance, or
the shape of the distribution).

Because ROI/condition labels are permuted within subjects, single subject depth
profiles need to be provided (i.e. the input depth profiles have three
dimensions, corresponding to subjects, conditions, depth levels).

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
from py_depthsampling.main.find_peak import find_peak


# ----------------------------------------------------------------------------
# *** Define parameters

# Corrected or  uncorrected depth profiles?
strCrct = 'corrected'

# Path of depth-profiles:
if strCrct == 'uncorrected':
    objDpth01 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy'  #noqa
    objDpth02 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2.npy'  #noqa
if strCrct == 'corrected':
    objDpth01 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1_corrected_model_1.npy'  #noqa
    objDpth02 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2_corrected_model_1.npy'  #noqa

# Stimulus luminance contrast levels (only needed for visualisation):
# vecEmpX = np.array([0.025, 0.061, 0.163, 0.72])

# Number of resampling iterations (set to `None` in case of small enough sample
# size for exact test, otherwise Monte Carlo resampling is performed):
varNumIt = None


# ----------------------------------------------------------------------------
# *** Load depth profiles

print('-Peak position permutation test')

print(('--') + strCrct.upper() + ' depth profiles.')

if not(varNumIt is None):
    print(('--Resampling iterations: ' + str(varNumIt)))

# Load depth profiles from npy files:
aryDpth01 = np.load(objDpth01)
aryDpth02 = np.load(objDpth02)

# Number of subject:
varNumSubs = aryDpth01.shape[0]

# Number of conditions:
varNumCon = aryDpth01.shape[1]

# Number of depth levels:
varNumDpt = aryDpth01.shape[2]


# ----------------------------------------------------------------------------
# *** Empirical mean difference

print('---Find peaks in empirical depth profiles')

# The peak difference on the full profile needs to be calculated for comparison
# with the null distribution.

# Mean depth profiles (mean across subjects):
aryDpthMne01 = np.mean(aryDpth01, axis=0)
aryDpthMne02 = np.mean(aryDpth02, axis=0)

# Peak positions in empirical depth profiles:
vecEmpPeaks01 = find_peak(aryDpthMne01, lgcStat=False)
vecEmpPeaks02 = find_peak(aryDpthMne02, lgcStat=False)

# Absolute peak difference in empirical profiles:
vecEmpPeakDiff = np.absolute(np.subtract(vecEmpPeaks01, vecEmpPeaks02))

print(('------Peak positions in mean empirical profiles, ROI 1:  '
       + str(vecEmpPeaks01)))
print(('------Peak positions in mean empirical profiles, ROI 2:  '
       + str(vecEmpPeaks02)))
print(('------Absolute difference in peak positions (empirical): '
       + str(vecEmpPeakDiff)))


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
aryDpthRnd01 = np.zeros((varNumIt, varNumSubs, varNumCon, varNumDpt))
aryDpthRnd02 = np.zeros((varNumIt, varNumSubs, varNumCon, varNumDpt))

# Loop through iterations:
for idxIt in range(0, varNumIt):

    # Assign values from original group 1 to permutation group 1:
    aryDpthRnd01[idxIt, aryRnd01[idxIt, :], :, :] = \
        aryDpth01[aryRnd01[idxIt, :], :, :]

    # Assign values from original group 2 to permutation group 1:
    aryDpthRnd01[idxIt, aryRnd02[idxIt, :], :, :] = \
        aryDpth02[aryRnd02[idxIt, :], :, :]

    # Assign values from original group 1 to permutation group 2:
    aryDpthRnd02[idxIt, aryRnd02[idxIt, :], :, :] = \
        aryDpth01[aryRnd02[idxIt, :], :, :]

    # Assign values from original group 2 to permutation group 2:
    aryDpthRnd02[idxIt, aryRnd01[idxIt, :], :, :] = \
        aryDpth02[aryRnd01[idxIt, :], :, :]

# Take mean across subjects in permutation samples:
aryDpthRnd01 = np.mean(aryDpthRnd01, axis=1)
aryDpthRnd02 = np.mean(aryDpthRnd02, axis=1)


# ----------------------------------------------------------------------------
# *** Find peaks in permutation samples

print('---Find peaks in permutation samples')

# Array for peak positions in permutation samples, of the form
# aryPermPeaks01[idxCondition, idxIteration]
aryPermPeaks01 = np.zeros((varNumCon, varNumIt))
aryPermPeaks02 = np.zeros((varNumCon, varNumIt))

# Loop through conditions and find peaks:
for idxCon in range(0, varNumCon):
    aryPermPeaks01[idxCon, :] = find_peak(aryDpthRnd01[:, idxCon, :],
                                          lgcStat=False)
    aryPermPeaks02[idxCon, :] = find_peak(aryDpthRnd02[:, idxCon, :],
                                          lgcStat=False)


# ----------------------------------------------------------------------------
# *** Create null distribution

print('---Create null distribution')

# The mean difference in peak position between the two randomised groups is the
# null distribution (aryNull[idxCondition, idxIteration]).
aryNull = np.subtract(aryPermPeaks01, aryPermPeaks02)


# ----------------------------------------------------------------------------
# *** Calculate p-value

print('---Calculate p-value')

# Absolute of the mean difference in peak position between the two randomised
# groups (null distribution).
aryNullAbs = np.absolute(aryNull)

# Number of resampled cases with absolute peak position difference that is at
# least as large as the empirical peak difference:
vecNumGe = np.sum(np.greater_equal(aryNullAbs,
                                   vecEmpPeakDiff[:, None]),
                  axis=1)

print('------Number of resampled cases with absolute peak position')
print('      difference that is at least as large as the empirical peak')
print(('      difference: '
      + str(vecNumGe)))

# Ratio of resampled cases with absolute peak position difference that is at
# least as large as the empirical peak difference (permutation p-value):
vecP = np.divide(vecNumGe.astype(np.float64),
                 float(varNumIt))

print('------Permutation p-value for equality of distributions of peak')
print('      position of contrast-at-half-maximum response depth profiles')
print(('      between the two ROIs: '
       + str(np.around(vecP, decimals=4))))
# ----------------------------------------------------------------------------

print('-Done.')
