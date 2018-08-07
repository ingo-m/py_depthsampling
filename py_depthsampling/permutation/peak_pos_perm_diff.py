# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

# Part of py_depthsampling library
# Copyright (C) 2018  Ingo Marquardt
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
from py_depthsampling.main.find_peak import find_peak


def peak_diff(strPthData, lstDiff):
    """
    Plot across-subject cortical depth profiles with SEM.

    Parameters
    ----------


    Returns
    -------


    Notes
    -----


    Function of the depth sampling pipeline.
    """
    # -------------------------------------------------------------------------
    # *** Load data from disk

    # [(0, 1), (0, 2)]

    # We compare the the peak positions between profiles of condition
    # differences. Thus, four profiles need to be loaded (two for each condition
    # comparison.

    # Get npz files:
    objNpzA01 = np.load(strPthData.format(lstDiff[0][0]))
    objNpzA02 = np.load(strPthData.format(lstDiff[0][1]))
    objNpzB01 = np.load(strPthData.format(lstDiff[1][0]))
    objNpzB02 = np.load(strPthData.format(lstDiff[1][1]))

    # Get arrays form npz files, shape aryDpth[subject, depth]:
    aryDpthA01 = objNpzA01['arySubDpthMns']
    aryDpthA02 = objNpzA02['arySubDpthMns']
    aryDpthB01 = objNpzB01['arySubDpthMns']
    aryDpthB02 = objNpzB02['arySubDpthMns']

    # Array with number of vertices (for weighted averaging across subjects),
    # shape: vecNumInc[subjects].
    vecNumIncA01 = objNpzA01['vecNumInc']
    vecNumIncA02 = objNpzA02['vecNumInc']
    vecNumIncB01 = objNpzB01['vecNumInc']
    vecNumIncB02 = objNpzB02['vecNumInc']

    # Number of subjects:
    varNumSub = aryDpthA01.shape[0]

    # Get number of depth levels:
    varNumDpth = aryDpthA01.shape[1]

    # --------------------------------------------------------------------------
    # *** Empirical mean difference

    print('---Find peaks in empirical condition difference')

    # The peak difference on the empirical profile needs to be calculated for
    # comparison with the null distribution.

    # Condition difference comparison A:
    aryDpthDiffA = np.subtract(aryDpthA01, aryDpthA02)
    # Condition difference comparison B:
    aryDpthDiffB = np.subtract(aryDpthB01, aryDpthB02)

    # Weighted average (across subjects):
    vecDpthDiffA = np.average(aryDpthDiffA, axis=0, weights=vecNumIncA01)
    vecDpthDiffB = np.average(aryDpthDiffB, axis=0, weights=vecNumIncB01)
    # New array shape: vecDpthDiffA[depth]

    # Peak positions in empirical depth profiles:
    vecEmpPeaksA = find_peak(vecDpthDiffA.reshape(1, varNumDpt), lgcStat=False)
    vecEmpPeaksB = find_peak(vecDpthDiffB.reshape(1, varNumDpt), lgcStat=False)

    # Absolute peak difference in contrast of empirical profiles:
    vecEmpPeakDiff = np.absolute(np.subtract(vecEmpPeaks01, vecEmpPeaks02))

    print(('------Peak positions in mean empirical profiles, contrast A:  '
           + str(vecEmpPeaksA)))
    print(('------Peak positions in mean empirical profiles, contrast B:  '
           + str(vecDpthDiffB)))
    print(('------Absolute difference in peak positions (empirical): '
           + str(vecEmpPeakDiff)))

    # --------------------------------------------------------------------------









    # Create condition labels for differences:
    lstDiffLbl = [None] * varNumCon
    for idxDiff in range(varNumCon):
        lstDiffLbl[idxDiff] = ((lstConLbl[lstDiff[idxDiff][0]])
                               + ' minus '
                               + (lstConLbl[lstDiff[idxDiff][1]]))
    lstConLbl = lstDiffLbl

***



# ----------------------------------------------------------------------------
# *** Create permutation samples

print('---Create permutation samples')

# Random array that is used to permute V1 and V2 labels within subjects, of the
# form aryRnd[idxIteration, idxSub]. For each iteration and subject, there is
# either a zero or a one. 'Zero' means that the actual V1 value gets assigned
# to the permuted 'V1' group and the actual V2 value gets assigned to the
# permuted 'V2' group. 'One' means that the labels are switched, i.e. the
# actual V1 label get assignet to the 'V2' group and vice versa.
aryRnd = np.random.randint(0, high=2, size=(varNumIt, varNumSubs))

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
