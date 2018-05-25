# -*- coding: utf-8 -*-
"""
Permutation test for difference between conditions in depth profiles.

Compares cortical depth profiles from two different conditions (e.g. PacMan
dynamic vs. PacMan statis). Tests whether the difference between the two
conditions is significant at any cortical depth.

Inputs are *.npy files containing depth profiles for each subject.
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
from py_depthsampling.permutation.perm_main import permute
from py_depthsampling.plot.plt_dpth_prfl import plt_dpth_prfl


# -----------------------------------------------------------------------------
# *** Define parameters

# Draining model suffix ('' for non-corrected profiles):
lstMdl = ['', '_deconv_model_1']

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['stimulus']

# ROI ('v1' or 'v2'):
lstRoi = ['v1', 'v2']

# Hemisphere ('rh' or 'lh'):
lstHmsph = ['rh']

# Path of depth-profile to load (meta-condition, ROI, hemisphere, condition,
# and deconvolution suffix left open):
strPthPrf = '/home/john/PhD/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}{}.npy'  #noqa

# Output path & prefix for plots (meta-condition, ROI, hemisphere, condition,
# and deconvolution suffix left open):
strPthPltOt = '/home/john/PhD/PacMan_Plots/permutation/{}_{}_{}{}_'  #noqa

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

# Number of resampling iterations:
varNumIt = 10000

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
# ***



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

                    # Load single subject depth profiles (shape
                    # aryDpth[subject, depth]):
                    aryDpth01 = np.load(strTmpPth01)
                    aryDpth02 = np.load(strTmpPth02)

                    permute(aryDpth01, aryDpth02, varNumIt=varNumIt)



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
