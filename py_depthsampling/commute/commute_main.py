# -*- coding: utf-8 -*-
"""
Test for commutative property of drain model (deconvolution).

Test for commutative property of drain model (deconvolution) with respect to
subtraction. In other words, does it make a difference whether we:
    - First apply the deconvolution (separately for each condition for each
      subject), subsequently calculate differences between conditions (within
      subjects), and finally take the mean across subjects
or
    - First calculate the difference between conditions (within subject),
      apply the deconvolution on the difference score, and finally take the
      mean across subject?

Approach A:
(1) Apply deconvolution (separately for each subject and condition).
(2) Calculate difference between conditions (within subjects).
(3) Take mean across subjects.

Approach B:
(1) Calculate difference between conditions (within subject).
(2) Apply deconvolution (on differences, separately for each subject).
(3) Take mean across subjects.

Note: Median would be better than mean, but this would nee to be implemented in
conjunction with (bootrapped?) confidence intervals.
"""

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


import numpy as np
from py_depthsampling.commute.commute_deconv import deconv
from py_depthsampling.plot.plt_dpth_prfl import plt_dpth_prfl


# -----------------------------------------------------------------------------
# *** Define parameters

# Which draining model to use (1, 2, 3, 4, 5, or 6 - see above for details):
lstMdl = [1]

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['stimulus']

# ROI ('v1' or 'v2'):
lstRoi = ['v1', 'v2']

# Hemisphere ('rh' or 'lh'):
lstHmsph = ['rh']

# Path of depth-profile to correct (meta-condition, ROI, hemisphere, and
# condition left open):
strPthPrf = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}.npy'  #noqa

# Output path & prefix for plots (meta-condition, ROI, hemisphere, condition,
# and model index left open):
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/commutative/{}_{}_{}_deconv_model_{}_'  #noqa

# File type suffix for plot:
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [arbitrary units]'

# Condition levels (used to complete file names):
lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst']

# Condition labels:
lstConLbl = ['PacMan Dynamic Sustained',
             'Control Dynamic Sustained',
             'PacMan Static Sustained']

# Which conditions to compare (list of tuples with condition indices):
lstDiff = [(0, 1), (0, 2)]

# Number of resampling iterations for peak finding (for models 1, 2, and 3) or
# random noise samples (models 4 and 5):
# varNumIt = 10000

# Lower & upper bound of percentile bootstrap (in percent), for bootstrap
# confidence interval (models 1, 2, and 3) - this value is only printed, not
# plotted - or plotted confidence intervals in case of model 5:
# varCnfLw = 0.5
# varCnfUp = 99.5

# Limits of y-axis:
varYmin = -50.0
varYmax = 50.0

# Figure scaling factor:
varDpi = 80.0
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Preparations

# Number of comparisons:
varNumDiff = len(lstDiff)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Approach A

# Approach A:
# (1) Apply deconvolution (separately for each subject and condition).
# (2) Calculate difference between conditions (within subjects).
# (3) Take mean across subjects.

# Loop through models, ROIs, hemispheres, and conditions to create plots:
for idxMtaCn in range(len(lstMetaCon)):  #noqa
    for idxMdl in range(len(lstMdl)):  #noqa
        for idxRoi in range(len(lstRoi)):
            for idxHmsph in range(len(lstHmsph)):

                # List for condition labels:
                lstConLbl = []

                for idxDiff in range(len(lstDiff)):

                    # Condition names:
                    strTmpCon01 = lstCon[lstDiff[idxDiff][0]]
                    strTmpCon02 = lstCon[lstDiff[idxDiff][1]]

                    # Append condition labels (for plot legend):
                    lstConLbl.append((strTmpCon01 + ' minus ' + strTmpCon02))

                    # Path of first depth profile:
                    strTmpPth01 = strPthPrf.format(
                        lstMetaCon[idxMtaCn], lstRoi[idxRoi],
                        lstHmsph[idxHmsph], strTmpCon01)

                    # Path of second depth profile:
                    strTmpPth02 = strPthPrf.format(
                        lstMetaCon[idxMtaCn], lstRoi[idxRoi],
                        lstHmsph[idxHmsph], strTmpCon02)

                    # Load original (i.e. non-convolved) single subject depth
                    # profiles (shape aryDpth[subject, depth]):
                    aryDpth01 = np.load(strTmpPth01)
                    aryDpth02 = np.load(strTmpPth02)

                    # Dimensions:
                    varNumSub = aryDpth01.shape[0]
                    varNumDpth = aryDpth01.shape[1]

                    # On first iteration, pre-allocate array for results (can
                    # not be done earlier because number of depth levels needs
                    # to be known):
                    if idxDiff == 0:  # not('aryMneA' in locals()):
                        # Array for results of approach A, mean & SD:
                        aryMneA = np.zeros((varNumDiff, varNumDpth))
                        aryStdA = np.zeros((varNumDiff, varNumDpth))

                    # Reshape to aryDpth[subject, 1, depth] (dummy condition
                    # dimension for compatibility):
                    aryDpth01 = aryDpth01.reshape((varNumSub, 1, varNumDpth))
                    aryDpth02 = aryDpth02.reshape((varNumSub, 1, varNumDpth))

                    # (1) Apply deconvolution (separately for each subject and
                    #     condition).
                    aryDpth01 = deconv(aryDpth01, lstRoi[idxRoi], varMdl=1)
                    aryDpth02 = deconv(aryDpth02, lstRoi[idxRoi], varMdl=1)

                    # (2) Calculate difference between conditions (within
                    #     subjects).
                    aryDiff = np.subtract(aryDpth01, aryDpth02)

                    # (3) Mean across subjects.
                    aryMneA[idxDiff, :] = np.mean(aryDiff, axis=0)

                    # Standard deviation:
                    aryStdA[idxDiff, :] = np.std(aryDiff, axis=0)

                # Plot results:
                plt_dpth_prfl(aryMneA,
                              aryStdA,
                              varNumDpth,
                              varNumDiff,
                              varDpi,
                              varYmin,
                              varYmax,
                              False,
                              lstConLbl,
                              'Cortical depth level (equivolume)',
                              'fMRI signal change (arbitraty units)',
                              (lstRoi[idxRoi].upper()
                               + ' '
                               + lstHmsph[idxHmsph].upper()),
                              True,
                              (strPthPltOt.format(lstMetaCon[idxMtaCn],
                                                  lstRoi[idxRoi],
                                                  lstHmsph[idxHmsph],
                                                  lstMdl[idxMdl])
                               + 'approach_A' + strFlTp)
                              )
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Approach B

# Approach B:
# (1) Calculate difference between conditions (within subject).
# (2) Apply deconvolution (on differences, separately for each subject).
# (3) Take mean across subjects.

# Loop through models, ROIs, hemispheres, and conditions to create plots:
for idxMtaCn in range(len(lstMetaCon)):  #noqa
    for idxMdl in range(len(lstMdl)):  #noqa
        for idxRoi in range(len(lstRoi)):
            for idxHmsph in range(len(lstHmsph)):

                # List for condition labels:
                lstConLbl = []

                for idxDiff in range(len(lstDiff)):

                    # Condition names:
                    strTmpCon01 = lstCon[lstDiff[idxDiff][0]]
                    strTmpCon02 = lstCon[lstDiff[idxDiff][1]]

                    # Append condition labels (for plot legend):
                    lstConLbl.append((strTmpCon01 + ' minus ' + strTmpCon02))

                    # Path of first depth profile:
                    strTmpPth01 = strPthPrf.format(
                        lstMetaCon[idxMtaCn], lstRoi[idxRoi],
                        lstHmsph[idxHmsph], strTmpCon01)

                    # Path of second depth profile:
                    strTmpPth02 = strPthPrf.format(
                        lstMetaCon[idxMtaCn], lstRoi[idxRoi],
                        lstHmsph[idxHmsph], strTmpCon02)

                    # Load original (i.e. non-convolved) single subject depth
                    # profiles (shape aryDpth[subject, depth]):
                    aryDpth01 = np.load(strTmpPth01)
                    aryDpth02 = np.load(strTmpPth02)

                    # Dimensions:
                    varNumSub = aryDpth01.shape[0]
                    varNumDpth = aryDpth01.shape[1]

                    # On first iteration, pre-allocate array for results (can
                    # not be done earlier because number of depth levels needs
                    # to be known):
                    if idxDiff == 0:  # not('aryMneB' in locals()):
                        # Array for results of approach A, mean & SD:
                        aryMneB = np.zeros((varNumDiff, varNumDpth))
                        aryStdB = np.zeros((varNumDiff, varNumDpth))

                    # Reshape to aryDpth[subject, 1, depth] (dummy condition
                    # dimension for compatibility):
                    aryDpth01 = aryDpth01.reshape((varNumSub, 1, varNumDpth))
                    aryDpth02 = aryDpth02.reshape((varNumSub, 1, varNumDpth))

                    # (1) Calculate difference between conditions (within
                    #     subject).
                    aryDiff = np.subtract(aryDpth01, aryDpth02)

                    # (2) Apply deconvolution (on differences, separately for
                    #     each subject).
                    aryDiff = deconv(aryDiff, lstRoi[idxRoi], varMdl=1)

                    # (3) Take mean across subjects.
                    aryMneB[idxDiff, :] = np.mean(aryDiff, axis=0)

                    # Standard deviation:
                    aryStdB[idxDiff, :] = np.std(aryDiff, axis=0)

                # Plot results:
                plt_dpth_prfl(aryMneB,
                              aryStdB,
                              varNumDpth,
                              varNumDiff,
                              varDpi,
                              varYmin,
                              varYmax,
                              False,
                              lstConLbl,
                              'Cortical depth level (equivolume)',
                              'fMRI signal change (arbitraty units)',
                              (lstRoi[idxRoi].upper()
                               + ' '
                               + lstHmsph[idxHmsph].upper()),
                              True,
                              (strPthPltOt.format(lstMetaCon[idxMtaCn],
                                                  lstRoi[idxRoi],
                                                  lstHmsph[idxHmsph],
                                                  lstMdl[idxMdl])
                               + 'approach_B' + strFlTp)
                              )

# -----------------------------------------------------------------------------
