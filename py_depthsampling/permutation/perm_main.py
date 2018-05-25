# -*- coding: utf-8 -*-
"""
Permutation test for difference between conditions in depth profiles.

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


def permute(aryDpth01, aryDpth02, varNumIt=10000, varLow=2.5, varUp=97.5):
    """
    Permutation test for difference between conditions in depth profiles.

    Parameters
    ----------
    aryDpth01 : np.array
        Array with depth profiles from first experimental condition (e.g.
        'PacMan Dynamic'), shape: aryDpth01[subject, depth].
    aryDpth02 : np.array
        Array with depth profiles from second experimental condition (e.g.
        'PacMan Static'), shape: aryDpth01[subject, depth].
    varNumIt : int
        Number of resampling iterations.
    varLow : float
        Lower bound of null distribution.
    varUp : float
        Upper bound of null distribution.

    Returns
    -------
    aryNull : np.array
        Array with parameters of permutation null distribution, shape
        aryNull[3, varNumDpth]. First dimension corresponds to lower bound,
        mean, and upper bound of the permutation null distribution. For
        instance, if `varLow = 2.5` and `varUp = 97.5`, the bounds of the 95%
        confidence interval of the null distribution are returned.
    vecP : np.array
        Array with one p-value for each depth level (shape vecP[depth]),
        pertaining to the probability of obtaining a difference between
        conditions as equal to or greater than the empirically observed
        condition difference.
    aryEmpDiffMdn : np.array
        Empirical difference between conditions (mean across subjects).

    Notes
    -----
    Compares cortical depth profiles from two different conditions (e.g. PacMan
    dynamic vs. PacMan statis). Tests whether the difference between the two
    conditions is significant at any cortical depth.

    The procedure is as follow:
    - Condition labels (i.e. 'PacMan Dynamic' and 'PacMan Static') are permuted
      within subjects for each permutation data set (i.e. on each iteration).
    - For each permutation dataset, the difference between the two conditions
      (i.e. the randomly created 'PacMan Dynamic' and 'PacMan Static'
      assignments) are calculated, separately for each subject. The result is
      one depth profile per subject (with the difference between randomly
      assigned conditions).
    - The average difference across subejcts is calculated.
    - The resulte is a distribution of condition differences for each depth
      level (dimensions are `number of resampling iterations` * `number of
      depth level`. This distribution is the null distribution.
    - The empirical difference between conditions can be compared agains this
      null distribution.
    """
    # -------------------------------------------------------------------------
    # *** Preparations
    print('-Permutation test')

    # Number of subject:
    varNumSubs = aryDpth01.shape[0]

    # Number of depth levels:
    varNumDpt = aryDpth01.shape[1]

    # -------------------------------------------------------------------------
    # *** Create null distribution

    print('---Create null distribution')

    # Random array that is used to permute condition labels within subjects, of
    # the form aryRnd[idxIteration, idxSub]. For each iteration and subject,
    # there is either a zero or a one. For instance, assume that conditions are
    # 'PacMan Dynamic' and 'PacMan Static'. 'Zero' means that the actual
    # condition value 'PacMan Dynamic' gets assigned to the permuted 'PacMan
    # Dynamic' group, and the actual 'PacMan Static' value gets assigned to the
    # permuted 'PacMan Static' group. 'One' means that the labels are switched,
    # i.e. the actual 'PacMan Dynamic' value get assignet to the 'PacMan
    # Static' group, and vice versa.
    aryRnd = np.random.randint(0, high=2, size=(varNumIt, varNumSubs))

    # We need two versions of the randomisation array, one for sampling from
    # the first input array (e.g. 'PacMan Dynamic'), and a second version to
    # sample from the second input array (e.g. 'PacMan Static'). (I.e. the
    # second version is the opposite of the first version.)
    aryRnd01 = np.equal(aryRnd, 1)
    aryRnd02 = np.equal(aryRnd, 0)
    del(aryRnd)

    # Arrays for permuted depth profiles for the two randomised groups:
    aryDpthRnd01 = np.zeros((varNumIt, varNumSubs, varNumDpt))
    aryDpthRnd02 = np.zeros((varNumIt, varNumSubs, varNumDpt))

    # Loop through iterations:
    for idxIt in range(0, varNumIt):

        # Assign values from original group 1 to permutation group 1:
        aryDpthRnd01[idxIt, aryRnd01[idxIt, :], :] = \
            aryDpth01[aryRnd01[idxIt, :], :]

        # Assign values from original group 2 to permutation group 1:
        aryDpthRnd01[idxIt, aryRnd02[idxIt, :], :] = \
            aryDpth02[aryRnd02[idxIt, :], :]

        # Assign values from original group 1 to permutation group 2:
        aryDpthRnd02[idxIt, aryRnd02[idxIt, :], :] = \
            aryDpth01[aryRnd02[idxIt, :], :]

        # Assign values from original group 2 to permutation group 2:
        aryDpthRnd02[idxIt, aryRnd01[idxIt, :], :] = \
            aryDpth02[aryRnd01[idxIt, :], :]

    # Within-subject difference between conditions (separately for each
    # iteration, subject, and depth level):
    aryPermDiff = np.subtract(aryDpthRnd01, aryDpthRnd02)

    # Mean condition difference across subjects (separately for each
    # iteration and depth level):
    aryPermDiff = np.mean(aryPermDiff, axis=1)

    # Mean of permutation distribution - i.e. the mean difference between
    # randomly permuted conditions - the mean difference expected by chance.
    aryPermDiffMne = np.mean(aryPermDiff, axis=0)

    # Lower and upper bound of the permutation null distribution. For instance,
    # if `varLow = 2.5` and `varUp = 97.5`, this corresponds to the bounds of
    # the 95% confidence interval of the null distribution.
    aryPermDiffPrcnt = np.percentile(aryPermDiff, (varLow, varUp), axis=0).T

    # Create output array of shape aryNull[3, varNumDpth]. First dimension
    # corresponds to lower bound, mean, and upper bound of the permutation
    # null distribution.
    aryNull = np.array([aryPermDiffPrcnt[:, 0],
                        aryPermDiffMne,
                        aryPermDiffPrcnt[:, 1]])

    # -------------------------------------------------------------------------
    # *** Calculate empirical difference

    print('---Calculate empirical difference')

    # Empirical difference between conditions. First, calculate within-subject
    # difference:
    aryEmpDiff = np.subtract(aryDpth01, aryDpth02)

    # Mean difference across subjects:
    aryEmpDiffMdn = np.mean(aryEmpDiff, axis=0)

    # -------------------------------------------------------------------------
    # *** Calculate p-value

    print('---Calculate p-value')

    # Array for p-value at each depth level:
    vecP = np.zeros((varNumDpt))

    # Take absolute difference?
    # aryPermDiff = np.absolute(aryPermDiff)
    # aryEmpDiffMne = np.absolute(aryEmpDiffMne)

    for idxDpt in range(varNumDpt):

        # Number of resampling cases with a condition difference greater or
        # equal to the 'actual', empricial difference between conditions:
        vecP[idxDpt] = np.sum(
                              np.greater_equal(
                                               aryPermDiff[:, idxDpt],
                                               aryEmpDiffMdn[idxDpt]
                                               )
                              ).astype(np.float64)

    # Convert count of cases into p-value:
    vecP = np.divide(vecP, float(varNumIt))

    return aryNull, vecP, aryEmpDiffMdn
    # -------------------------------------------------------------------------
