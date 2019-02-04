# -*- coding: utf-8 -*-
"""
Compare granular-agranular score between ROIs.

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

# Which depth levels to include in granular compartment:
lstGrn = [4, 5, 6]

# Which depth levels to include in agranular compartment:
lstAgr = [8, 9, 10]


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
# *** Empirical granularity score

print('---Compute granularity score on empirical depth profiles')

# The agranularity score on the empirical depth profiles will be compared with
# the permutation null distribution.

# Weighted mean across subjects:
vecDpthMne01 = np.average(aryCtrRoi01, axis=0, weights=vecNumIncRoi01)
vecDpthMne02 = np.average(aryCtrRoi02, axis=0, weights=vecNumIncRoi02)

# Mean signal in granular compartment, first ROI:
varGrn01 = np.mean(vecDpthMne01[lstGrn])

# Mean signal in granular compartment, second ROI:
varGrn02 = np.mean(vecDpthMne02[lstGrn])

# Mean signal in agranular compartment, first ROI:
varAgr01 = np.mean(vecDpthMne01[lstAgr])

# Mean signal in agranular compartment, second ROI:
varAgr02 = np.mean(vecDpthMne02[lstAgr])

# Granularity score - first ROI:
varScr01 = np.subtract(varGrn01, varAgr01)

# Granularity score - second ROI:
varScr02 = np.subtract(varGrn02, varAgr02)

# Difference in empirical granularity score:
varDiff = np.absolute(np.subtract(varScr01, varScr02))


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

a1 = aryDpthRnd01.T
a2 = aryDpthRnd02.T


# ----------------------------------------------------------------------------
# *** Permutation granularity score

print('---Compute granularity score on resampled depth profiles')

# The agranularity score on the empirical depth profiles will be compared with
# the permutation null distribution.

# Mean signal in granular compartment, first randomised ROI:
vecGrn01 = np.mean(aryDpthRnd01[:, lstGrn], axis=1)

# Mean signal in granular compartment, second randomised ROI:
vecGrn02 = np.mean(aryDpthRnd02[:, lstGrn], axis=1)

# Mean signal in agranular compartment, first randomised ROI:
vecAgr01 = np.mean(aryDpthRnd01[:, lstAgr], axis=1)

# Mean signal in agranular compartment, second randomised ROI:
vecAgr02 = np.mean(aryDpthRnd02[:, lstAgr], axis=1)

# Granularity score - first ROI:
vecScr01 = np.subtract(vecGrn01, vecAgr01)

# Granularity score - second ROI:
vecScr02 = np.subtract(vecGrn02, vecAgr02)


# -------------------------------------------------------------------------
# *** Create null distribution

print('---Create null distribution')

# The absolute difference in granularity scores from the two resampled groups
# is the null distribution:
vecNull = np.absolute(np.subtract(vecScr01, vecScr02))


# ----------------------------------------------------------------------------
# *** Calculate p-value

print('---Calculate p-value')

# Number of resampled cases with absolute difference in granulairty score that
# is at least as large as the empirical peak difference:
varNumGe = np.sum(np.greater_equal(vecNull,
                                   varDiff),
                  axis=0)

print('------Number of resampled cases with absolute difference in ')
print('      granulairty score that is at least as large as the empirical ')
print(('      difference: '
      + str(varNumGe)))

# Ratio of resampled cases with absolute peak position difference that is at
# least as large as the empirical peak difference (permutation p-value):
varP = np.divide(float(varNumGe), float(varNumIt))

print('------Permutation p-value for equality of distributions between the')
print(('      two ROIs: '
       + str(np.around(varP, decimals=4))))
# ----------------------------------------------------------------------------

print('-Done.')
