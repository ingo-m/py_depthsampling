# -*- coding: utf-8 -*-
"""
Permutation test for difference in peak position for cortical depth profiles.

Performe a permutation hypothesis test for a difference in the peak position in
depth profiles of condition differences between ROIs. In other words, it is
tested whether the peak position of the contrast condition A vs. condition B is
the same between two ROIs (e.g. V1 and V2).

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
from py_depthsampling.main.find_peak import find_peak


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

# Standard deviation of the Gaussian kernel used for smoothing, relative to
# cortical thickness (i.e. a value of 0.05 would result in a Gaussian with SD
# of 5 percent of the cortical thickness).
varSd = 0.05

# Amplitude threshold for peak identification. For example, if varThr = 0.05,
# peaks with an absolute amplitude that is greater than the mean amplitude
# (over cortical depth) plus 0.05 are identified. The rationale for this is
# that even for very flat profiles a peak is identified. The threshold does not
# influence the peak search; instead, if a threshold is provided, an additional
# output vector is returned, containing boolean values (true if peak amplitude
# exceeds threshold).
varThr = 0.05


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

print('---Find peaks in empirical depth profiles')

# The peak difference on the full profile needs to be calculated for comparison
# with the null distribution.

# Weighted mean across subjects:
aryDpthMne01 = np.average(aryCtrRoi01, axis=0, weights=vecNumIncRoi01)
aryDpthMne02 = np.average(aryCtrRoi02, axis=0, weights=vecNumIncRoi02)

# Peak positions in empirical depth profiles:
vecEmpPeaks01, vecEmpLgc01 = find_peak(aryDpthMne01.reshape(1, varNumDpt),
                                       varSd=varSd,
                                       varThr=varThr,
                                       lgcStat=False)
vecEmpPeaks02, vecEmpLgc02 = find_peak(aryDpthMne02.reshape(1, varNumDpt),
                                       varSd=varSd,
                                       varThr=varThr,
                                       lgcStat=False)

# The peak finding function returns a vector, even in case of a single
# depth profile.
varEmpPeaks01 = vecEmpPeaks01[0]
varEmpPeaks02 = vecEmpPeaks02[0]
lgcEmpPeaks01 = vecEmpLgc01[0]
lgcEmpPeaks02 = vecEmpLgc02[0]

# Absolute peak difference in empirical profiles of condition contrast:
if np.multiply(lgcEmpPeaks01, lgcEmpPeaks02):
    # If there is a peak in both profiles, calculate distance between
    # peaks:
    varEmpPeakDiff = np.absolute(np.subtract(varEmpPeaks01, varEmpPeaks02))
elif np.logical_or(lgcEmpPeaks01, lgcEmpPeaks02):
    # If only one profile has a peak, difference is maximal.
    varEmpPeakDiff = 1.0
else:
    # If both profiles don't have a peak, the difference is zero.
    varEmpPeakDiff = 0.0

print(('------Peak positions in mean empirical profiles, ROI 1:  '
       + str(varEmpPeaks01)))
print(('------Peak positions in mean empirical profiles, ROI 2:  '
       + str(varEmpPeaks02)))
print(('------Absolute difference in peak positions (empirical): '
       + str(np.around(varEmpPeakDiff, decimals=2))))


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
# *** Find peaks in permutation samples

print('---Find peaks in permutation samples')

# Find peaks, permutation group 1:
vecPermPeaks01, vecLgc01 = find_peak(aryDpthRnd01,
                                     varSd=varSd,
                                     varThr=varThr,
                                     lgcStat=False)

# Find peaks, permutation group 2:
vecPermPeaks02, vecLgc02 = find_peak(aryDpthRnd02,
                                     varSd=varSd,
                                     varThr=varThr,
                                     lgcStat=False)

# Ratio of iterations with peak:
varRatioPeak = (float(np.sum(vecLgc01) + np.sum(vecLgc02))
                / float(2.0 * varNumIt))

print(('------Percentage of permutation samples with peak: '
      + str(np.around(varRatioPeak, decimals=3))))


# -------------------------------------------------------------------------
# *** Create null distribution

print('---Create null distribution')

# The mean difference in peak position between the two randomised groups is
# the null distribution (vecNull[idxIteration]).
vecNull = np.zeros((varNumIt))

# If there is a peak in both profiles, calculate distance between peaks:
lgcTmp = np.multiply(vecLgc01, vecLgc02)
vecNull[lgcTmp] = np.subtract(vecPermPeaks01[lgcTmp], vecPermPeaks02[lgcTmp])

# If only one profile has a peak, difference is maximal.
lgcTmp = np.logical_xor(vecLgc01, vecLgc02)
vecNull[lgcTmp] = 1.0

# If both profiles don't have a peak, the difference is zero.
lgcTmp = np.invert(np.multiply(vecLgc01, vecLgc02))
vecNull[lgcTmp] = 0.0


# ----------------------------------------------------------------------------
# *** Calculate p-value

print('---Calculate p-value')

# Absolute of the mean difference in peak position between the two randomised
# groups (null distribution).
vecNullAbs = np.absolute(vecNull)

# Number of resampled cases with absolute peak position difference that is at
# least as large as the empirical peak difference:
varNumGe = np.sum(np.greater_equal(vecNullAbs,
                                   varEmpPeakDiff),
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
