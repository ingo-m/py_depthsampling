
# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

# Part of py_depthsampling library
# Copyright (C) 2018  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
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


def peak_diff(strPthData, lstDiff, lstCon, varNumIt=1000, varThr=0.05):
    """
    Permutation test for condition differences on depth profiles.

    Parameters
    ----------
    strPthData : str
        Path of single subject depth-profiles (npz files, condition left open).
    lstDiff : list
        Which conditions to compare (list of tuples with condition indices,
        with respect to `lstCon`).
    lstCon : list
        Condition levels (list of strings, used to complete file names).
    varNumIt : int or None
        Number of resampling iterations. Set to `None` in case of small enough
        sample size for exact test (i.e. all possible resamples), otherwise
        Monte Carlo resampling is performed.
    varThr : float
        Amplitude threshold for peak identification. For example, if `varThr =
        0.05`, peaks with an absolute amplitude that is greater than the mean
        amplitude (over cortical depth) plus 0.05 are identified. The rationale
        for this is that even for very flat profiles a peak is identified. The
        threshold does not influence the peak search directly; instead, in case
        of a peak below thershold, the difference in peak position (between the
        two conditions that are compared) is set to zero for the respective
        comparison.

    Returns
    -------
    varP : float
        Permutation p-value for difference in peak position. (The ratio of
        resampled cases with an absolute peak position difference that is at
        least as large as the empirical peak difference.
    varEmpPeakDiff : float
        Absolute peak difference in empirical profiles of condition contrast
        (in relative cortical depth, i.e. between zero and one).
    lgcEmpPeaksA : bool
        Was a peak found in empirical contrast A?
    lgcEmpPeaksB : bool
        Was a peak found in empirical contrast B?
    varEmpPeaksA : float
        Peak position (relative cortical depth) in empricial profile A.
    varEmpPeaksB : float
        Peak position (relative cortical depth) in empricial profile B.
    varRatioPeak : float
        Ratio of depth profiles (iterations) where a peak was found, according
        to amplitude threshold.

    Notes
    -----
    The permutation test is performed on depth profiles of condition
    differences (e.g., is the peak position in the contrast condition A vs. B
    the same as in the contrast A vs. C). The equality of distributions of the
    peak positions is tested (i.e. a possible difference could be due to a
    difference in means, variance, or the shape of the distribution).

    Function of the depth sampling pipeline.

    """
    # -------------------------------------------------------------------------
    # *** Load data from disk

    # We compare the the peak positions between profiles of condition
    # differences. Thus, four profiles need to be loaded (two for each
    # condition comparison).

    # For example:
    # A01 = PacMan Dynamic
    # A02 = PacMan Static
    # B01 = PacMan Dynamic
    # B02 = Control dynamic

    # Get npz files:
    objNpzA01 = np.load(strPthData.format(lstCon[lstDiff[0][0]]))
    objNpzA02 = np.load(strPthData.format(lstCon[lstDiff[0][1]]))
    objNpzB01 = np.load(strPthData.format(lstCon[lstDiff[1][0]]))
    objNpzB02 = np.load(strPthData.format(lstCon[lstDiff[1][1]]))

    # Get arrays form npz files, shape aryDpth[subject, depth]:
    aryDpthA01 = objNpzA01['arySubDpthMns']
    aryDpthA02 = objNpzA02['arySubDpthMns']
    aryDpthB01 = objNpzB01['arySubDpthMns']
    aryDpthB02 = objNpzB02['arySubDpthMns']

    # Array with number of vertices (for weighted averaging across subjects),
    # shape: vecNumInc[subjects]. Only one instance is needed per comparison,
    # because the number of vertices is assumed to be constant across
    # conditions (at least within the comparison). The two comparisons could
    # have an unequal number of vertices (e.g. if comparing the same contrast
    # between ROI).
    vecNumIncA01 = objNpzA01['vecNumInc']
    # vecNumIncA02 = objNpzA02['vecNumInc']
    vecNumIncB01 = objNpzB01['vecNumInc']
    # vecNumIncB02 = objNpzB02['vecNumInc']

    # Number of subjects:
    varNumSub = aryDpthA01.shape[0]

    # Number of depth levels:
    varNumDpth = aryDpthA01.shape[1]

    # -------------------------------------------------------------------------
    # *** Empirical peak position difference

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

    # Peak finding on absolute difference (so as to also count negative peaks):
    vecDpthDiffA = np.absolute(vecDpthDiffA)
    vecDpthDiffB = np.absolute(vecDpthDiffB)

    # Peak positions in empirical depth profiles:
    vecEmpPeaksA, vecEmpLgcA = find_peak(vecDpthDiffA.reshape(1, varNumDpth),
                                         varThr=varThr)
    vecEmpPeaksB, vecEmpLgcB = find_peak(vecDpthDiffB.reshape(1, varNumDpth),
                                         varThr=varThr)

    # The peak finding function returns a vector, even in case of a single
    # depth profile.
    varEmpPeaksA = vecEmpPeaksA[0]
    varEmpPeaksB = vecEmpPeaksB[0]
    lgcEmpPeaksA = vecEmpLgcA[0]
    lgcEmpPeaksB = vecEmpLgcB[0]

    # Absolute peak difference in empirical profiles of condition contrast:
    if np.multiply(lgcEmpPeaksA, lgcEmpPeaksB):
        # If there is a peak in both profiles, calculate distance between
        # peaks:
        varEmpPeakDiff = np.absolute(np.subtract(varEmpPeaksA, varEmpPeaksB))
    elif np.logical_or(lgcEmpPeaksA, lgcEmpPeaksB):
        # If only one profile has a peak, difference is maximal.
        varEmpPeakDiff = 1.0
    else:
        # If both profiles don't have a peak, the difference is zero.
        varEmpPeakDiff = 0.0

    print(('------Peak positions in mean empirical profiles, contrast A:  '
           + str(np.around(varEmpPeaksA, decimals=3))))
    print(('------Peak positions in mean empirical profiles, contrast B:  '
           + str(np.around(varEmpPeaksB, decimals=3))))
    print(('------Absolute difference in peak positions (empirical): '
           + str(np.around(varEmpPeakDiff, decimals=3))))

    # -------------------------------------------------------------------------
    # *** Create permutation samples

    print('---Create permutation samples')

    # Random array that is used to permute condition labels within subjects, of
    # the form aryRnd[idxIteration, idxSub]. For each iteration and subject,
    # there is either a zero or a one. 'Zero' means that the actual label gets
    # assigned to the permuted group. 'One' means that the labels are switched.
    if not(varNumIt is None):
        # Monte Carlo resampling:
        aryRnd = np.random.randint(0, high=2, size=(varNumIt, varNumSub))
    else:
        # In case of tractable number of permutations, create a list of all
        # possible permutations (Bernoulli sequence).
        lstBnl = list(itertools.product([0, 1], repeat=varNumSub))
        aryRnd = np.array(lstBnl)
        # Number of resampling cases:
        varNumIt = len(lstBnl)

    # We need two versions of the randomisation array, one for sampling from
    # the first input array, and a second version to sample from the second
    # input array. (I.e. the second version is the inverse of the first
    # version.)
    aryRnd01 = np.equal(aryRnd, 1)
    aryRnd02 = np.equal(aryRnd, 0)
    del(aryRnd)

    # Arrays for permuted depth profiles for the randomised groups:
    aryDpthRndA = np.zeros((varNumIt, varNumSub, varNumDpth))
    aryDpthRndB = np.zeros((varNumIt, varNumSub, varNumDpth))

    # Loop through iterations:
    for idxIt in range(0, varNumIt):

        # Assign values from original group A to permutation group A:
        aryDpthRndA[idxIt, aryRnd01[idxIt, :], :] = \
            aryDpthDiffA[aryRnd01[idxIt, :], :]

        # Assign values from original group B to permutation group A:
        aryDpthRndA[idxIt, aryRnd02[idxIt, :], :] = \
            aryDpthDiffB[aryRnd02[idxIt, :], :]

        # Assign values from original group A to permutation group B:
        aryDpthRndB[idxIt, aryRnd02[idxIt, :], :] = \
            aryDpthDiffA[aryRnd02[idxIt, :], :]

        # Assign values from original group B to permutation group B:
        aryDpthRndB[idxIt, aryRnd01[idxIt, :], :] = \
            aryDpthDiffB[aryRnd01[idxIt, :], :]

    # -------------------------------------------------------------------------
    # *** Average within permutation samples

    # Weighted average (across subjects within permutation samples):
    aryDpthRndA = np.average(aryDpthRndA, axis=1, weights=vecNumIncA01)
    aryDpthRndB = np.average(aryDpthRndB, axis=1, weights=vecNumIncB01)

    # -------------------------------------------------------------------------
    # *** Find peaks in permutation samples

    print('---Find peaks in permutation samples')

    # Absolute difference (so as to also count negative peaks):
    aryDpthRndA = np.absolute(aryDpthRndA)
    aryDpthRndB = np.absolute(aryDpthRndB)

    # Find peaks:
    vecPermPeaksA, vecLgcA = find_peak(aryDpthRndA, varThr=varThr)
    vecPermPeaksB, vecLgcB = find_peak(aryDpthRndB, varThr=varThr)

    # Ratio of iterations with peak:
    varRatioPeak = (float(np.sum(vecLgcA) + np.sum(vecLgcB))
                    / float(2.0 * varNumIt))
    print(('------Percentage of permutation samples with peak: '
          + str(np.around(varRatioPeak, decimals=3))))

    # Cases for which no peak was identified in both permutation samples:
    # vecLgc = np.invert(np.multiply(vecLgcA, vecLgcB))

    # -------------------------------------------------------------------------
    # *** Create null distribution

    print('---Create null distribution')

    # The mean difference in peak position between the two randomised groups is
    # the null distribution (vecNull[idxIteration]).
    vecNull = np.zeros((varNumIt))

    # If there is a peak in both profiles, calculate distance between peaks:
    lgcTmp = np.multiply(vecLgcA, vecLgcB)
    vecNull[lgcTmp] = np.subtract(vecPermPeaksA[lgcTmp], vecPermPeaksB[lgcTmp])

    # If only one profile has a peak, difference is maximal.
    lgcTmp = np.logical_xor(vecLgcA, vecLgcB)
    vecNull[lgcTmp] = 1.0

    # If both profiles don't have a peak, the difference is zero.
    lgcTmp = np.invert(np.multiply(vecLgcA, vecLgcB))
    vecNull[lgcTmp] = 0.0

    # -------------------------------------------------------------------------
    # *** Calculate p-value

    print('---Calculate p-value')

    #  # Absolute of the mean difference in peak position between the two
    # randomised # groups (null distribution).
    vecNullAbs = np.absolute(vecNull)

    # Number of resampled cases with absolute peak position difference that is
    # at least as large as the empirical peak difference:
    varNumGe = np.sum(np.greater_equal(vecNullAbs, varEmpPeakDiff))

    print('------Number of resampled cases with absolute peak position')
    print('      difference that is at least as large as the empirical peak')
    print(('      difference: '
          + str(varNumGe)))

    # Ratio of resampled cases with absolute peak position difference that is
    # at least as large as the empirical peak difference (permutation p-value):
    varP = np.divide(float(varNumGe), float(varNumIt))

    print('------Permutation p-value for equality of distributions of peak')
    print('      position of contrast-at-half-maximum response depth profiles')
    print(('      between the two ROIs: '
           + str(np.around(varP, decimals=4))))

    return varP, varEmpPeakDiff, lgcEmpPeaksA, lgcEmpPeaksB, varEmpPeaksA, \
        varEmpPeaksB, varRatioPeak
# -----------------------------------------------------------------------------
