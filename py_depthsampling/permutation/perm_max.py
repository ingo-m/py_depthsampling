# -*- coding: utf-8 -*-
"""
Permutation test for difference between conditions in depth profiles.

Function of the depth sampling pipeline.
"""

# Part of py_depthsampling library
# Copyright (C) 2018 Ingo Marquardt
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


def permute_max(aryDpth01, aryDpth02, vecNumInc=None, varNumIt=10000):
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
    vecNumInc : np.array
        1D array with number of vertices per subject, used for weighted
        averaging across subjects. If `None`, number of vertices is assumed to
        be equal across subjects.
    varNumIt : int
        Number of resampling iterations.

    Returns
    -------
    varP : float
        Permutation p-value.

    Notes
    -----
    Compares cortical depth profiles from two different conditions (e.g. PacMan
    dynamic vs. PacMan statis). Tests whether the difference between the two
    conditions is significant at *any* cortical depth. This is in contrast to
    `perm_main.py`, where the permutation test is conducted separately for each
    depth level. The problem with a separate test at each depth level is that
    this leads to a massive multiple comparisons problem. Here, we test whether
    the maximal difference in amplitude (across coritcal depth) is different
    between conditions, irrespective of where along the cortical depth the
    maximum difference in amplitude occurs.

    The procedure is as follow:
    - Condition labels (i.e. 'PacMan Dynamic' and 'PacMan Static') are permuted
      within subjects for each permutation data set (i.e. on each iteration).
    - For each permutation dataset, the difference between the two conditions
      (i.e. the randomly created 'PacMan Dynamic' and 'PacMan Static'
      assignments) are calculated, separately for each subject. The result is
      one depth profile per subject (with the difference between randomly
      assigned conditions).
    - The average difference across subjects is calculated.
    - We take the maximum of the difference across cortical depth. The
      distribution of maxima is the null distribution.
    - The empirical maximum of the difference between conditions can be
      compared agains this null distribution.
    """
    # -------------------------------------------------------------------------
    # *** Preparations
    print('-Permutation test')

    # Number of subject:
    varNumSubs = aryDpth01.shape[0]

    # Number of depth levels:
    varNumDpt = aryDpth01.shape[1]

    # If number of vertices per subject is not provided, assume it to be the
    # same across subjects (for weighted averaging):
    if vecNumInc is None:
        vecNumInc = np.ones((varNumSubs))

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
    # aryPermDiff = np.mean(aryPermDiff, axis=1)
    aryPermDiff = np.average(aryPermDiff, weights=vecNumInc, axis=1)

    # Maximum difference across cortical depth:
    vecPermDiffMax = np.max(np.absolute(aryPermDiff), axis=1)

    # -------------------------------------------------------------------------
    # *** Calculate empirical difference

    print('---Calculate empirical parameter value')

    # Empirical difference between conditions. First, calculate within-subject
    # difference:
    aryEmpDiff = np.subtract(aryDpth01, aryDpth02)

    # Mean difference across subjects:
    aryEmpDiffMne = np.average(aryEmpDiff, weights=vecNumInc, axis=0)

    # Maximum difference across cortical depth:
    # varEmpDiffMneMax = np.max(aryEmpDiffMne)
    varEmpDiffMneMax = np.max(np.absolute(aryEmpDiffMne))

    # -------------------------------------------------------------------------
    # *** Calculate p-value

    print('---Calculate p-value')

    # Number of resampling cases with a condition difference greater or
    # equal to the 'actual', empricial difference between conditions:
    varP = np.sum(
                  np.greater_equal(
                                   vecPermDiffMax,
                                   varEmpDiffMneMax
                                   )
                  ).astype(np.float64)

    # Convert count of cases into p-value:
    varP = np.divide(varP, float(varNumIt))

    return varP
    # -------------------------------------------------------------------------
