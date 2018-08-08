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


# Which draining model to load ('' for none):
lstMdl = ['', '_deconv_model_1']

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['stimulus']

# ROI ('v1', 'v2', or 'v3'):
lstRoi = ['v1', 'v2', 'v3']

# Hemisphere ('rh' or 'lh'):
lstHmsph = ['rh']

# Path of depth-profiles (meta-condition, ROI, hemisphere, condition, and model
# index left open):
strPthData = '/Users/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}{}.npz'  #noqa

# Condition levels (used to complete file names):
lstCon = ['Pd_sst', 'Ps_sst', 'Cd_sst']
# lstCon = ['Pd_sst', 'Ps_sst', 'Cd_sst', 'Ps_sst_plus_Cd_sst']

# Which conditions to compare (nested list of tuples with condition indices):
lstDiff = [[(0, 1), (0, 2)],
           [(0, 1), (1, 2)],
           [(0, 2), (1, 2)]]

# Number of resampling iterations:
varNumIt = 500

lstDiff = lstDiff[0]
strPthData = strPthData.format(lstMetaCon[0],
                               lstRoi[0],
                               lstHmsph[0],
                               '{}',
                               lstMdl[1])



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
    # shape: vecNumInc[subjects].
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

    # Peak positions in empirical depth profiles:
    vecEmpPeaksA = find_peak(vecDpthDiffA.reshape(1, varNumDpth))
    vecEmpPeaksB = find_peak(vecDpthDiffB.reshape(1, varNumDpth))

    # Absolute peak difference in contrast of empirical profiles:
    vecEmpPeakDiff = np.absolute(np.subtract(vecEmpPeaksA, vecEmpPeaksB))

    print(('------Peak positions in mean empirical profiles, contrast A:  '
           + str(np.around(vecEmpPeaksA, decimals=3))))
    print(('------Peak positions in mean empirical profiles, contrast B:  '
           + str(np.around(vecEmpPeaksB, decimals=3))))
    print(('------Absolute difference in peak positions (empirical): '
           + str(np.around(vecEmpPeakDiff, decimals=3))))

    # -------------------------------------------------------------------------
    # *** Create permutation samples

    print('---Create permutation samples')

    # Random array that is used to permute condition labels within subjects, of
    # the form aryRnd[idxIteration, idxSub]. For each iteration and subject,
    # there is either a zero or a one. 'Zero' means that the actual label
    # gets assigned to the permuted group. 'One' means that the labels are
    # switched.
    aryRnd = np.random.randint(0, high=2, size=(varNumIt, varNumSub))

    # We need two versions of the randomisation array, one for sampling from the
    # first input array, and a second version to sample from the second
    # input array. (I.e. the second version is the inverse of the first
    # version.)
    aryRnd01 = np.equal(aryRnd, 1)
    aryRnd02 = np.equal(aryRnd, 0)
    del(aryRnd)

    # Arrays for permuted depth profiles for the randomised groups:
    aryDpthRndA01 = np.zeros((varNumIt, varNumSub, varNumDpth))
    aryDpthRndA02 = np.zeros((varNumIt, varNumSub, varNumDpth))
    aryDpthRndB01 = np.zeros((varNumIt, varNumSub, varNumDpth))
    aryDpthRndB02 = np.zeros((varNumIt, varNumSub, varNumDpth))

    # Arrays for number of vertices per subject in permuation samples:
    vecNumIncRndA = np.zeros((varNumIt, varNumSub))
    vecNumIncRndB = np.zeros((varNumIt, varNumSub))

    # Loop through iterations:
    for idxIt in range(0, varNumIt):

        # Assign permuted depth profiles for comparison A:
        
        # Assign values from original group 1 to permutation group 1:
        aryDpthRndA01[idxIt, aryRnd01[idxIt, :], :] = \
            aryDpthA01[aryRnd01[idxIt, :], :]

        # Assign values from original group 2 to permutation group 1:
        aryDpthRndA01[idxIt, aryRnd02[idxIt, :], :] = \
            aryDpthA01[aryRnd02[idxIt, :], :]

        # Assign values from original group 1 to permutation group 2:
        aryDpthRndA02[idxIt, aryRnd02[idxIt, :], :] = \
            aryDpthA02[aryRnd02[idxIt, :], :]

        # Assign values from original group 2 to permutation group 2:
        aryDpthRndA02[idxIt, aryRnd01[idxIt, :], :] = \
            aryDpthA02[aryRnd01[idxIt, :], :]

        # Number of vertices included in permutation samples for comparison A:
        vecNumIncRndA[idxIt, aryRnd01[idxIt, :]] = \
            vecNumIncA01[aryRnd01[idxIt, :]]
        vecNumIncRndA[idxIt, aryRnd02[idxIt, :]] = \
            vecNumIncA01[aryRnd02[idxIt, :]]

        # Assign permuted depth profiles for comparison B:
        
        # Assign values from original group 1 to permutation group 1:
        aryDpthRndB01[idxIt, aryRnd01[idxIt, :], :] = \
            aryDpthB01[aryRnd01[idxIt, :], :]

        # Assign values from original group 2 to permutation group 1:
        aryDpthRndB01[idxIt, aryRnd02[idxIt, :], :] = \
            aryDpthB01[aryRnd02[idxIt, :], :]

        # Assign values from original group 1 to permutation group 2:
        aryDpthRndB02[idxIt, aryRnd02[idxIt, :], :] = \
            aryDpthB02[aryRnd02[idxIt, :], :]

        # Assign values from original group 2 to permutation group 2:
        aryDpthRndB02[idxIt, aryRnd01[idxIt, :], :] = \
            aryDpthB02[aryRnd01[idxIt, :], :]

        # Number of vertices included in permutation samples for comparison B:
        vecNumIncRndB[idxIt, aryRnd01[idxIt, :]] = \
            vecNumIncB01[aryRnd01[idxIt, :]]
        vecNumIncRndB[idxIt, aryRnd02[idxIt, :]] = \
            vecNumIncB01[aryRnd02[idxIt, :]]

    # -------------------------------------------------------------------------
    # *** 

    # Condition difference (within subjects) in permuted samples:

    # Condition difference comparison A:
    aryDpthRndDiffA = np.subtract(aryDpthRndA01, aryDpthRndA02)
    # Condition difference comparison B:
    aryDpthRndDiffB = np.subtract(aryDpthRndB01, aryDpthRndB02)

    del(aryDpthRndA01)
    del(aryDpthRndA02)
    del(aryDpthRndB01)
    del(aryDpthRndB02)





    # Weighted average (across subjects):
    vecDpthDiffA = np.average(aryDpthDiffA, axis=0, weights=vecNumIncA01)
    vecDpthDiffB = np.average(aryDpthDiffB, axis=0, weights=vecNumIncB01)




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
