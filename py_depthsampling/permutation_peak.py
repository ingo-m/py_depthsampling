# -*- coding: utf-8 -*-
"""
Permutation test peak position for depth profiles of condition difference.

This version: Do not perform permutation test directly on single-condition
              depth profiles, but on depth profiles of condition differences
              (e.g. is the peak position in contrast of condition A vs. B the
              same as in the contrast A vs. C).

Performe a permutation hypothesis test for a difference in the peak position in
cortical depth profiles between experimental conditions. More specifically, the
equality of distributions of the peak positions is tested (i.e. a possible
difference could be due to a difference in means, variance, or the shape of the
distribution).

Because condition labels are permuted within subjects, single subject depth
profiles need to be provided (i.e. the input depth profiles have three
dimensions, corresponding to subjects, conditions, depth levels).

The procedure is as follows:
- Condition labels are permuted within subjects for each permutation data set
  (i.e. on each iteration).

- The condition contrast

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


import numpy as np


# -----------------------------------------------------------------------------
# *** Define parameters

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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Loop through ROIs / conditions

print('-Peak position permutation test')

# Loop through metaconditions, models, ROIs, hemispheres, and comparisons to create plots:
for idxMtaCn in range(len(lstMetaCon)):  #noqa
    for idxMdl in range(len(lstMdl)):  #noqa
        for idxRoi in range(len(lstRoi)):
            for idxHmsph in range(len(lstHmsph)):
                for idxDiff in range(len(lstDiff)):



                diff_sem(strPthData.format(lstMetaCon[idxMtaCn],
                                           lstRoi[idxRoi],
                                           lstHmsph[idxHmsph],
                                           '{}',
                                           lstMdl[idxMdl]),
                         (strPthPltOt.format(lstMetaCon[idxMtaCn],
                                             lstRoi[idxRoi],
                                             lstHmsph[idxHmsph],
                                             lstMdl[idxMdl])
                          + strFlTp),
                         lstCon,

                         lstDiff=lstDiff)
# ----------------------------------------------------------------------------









print('-Done.')
