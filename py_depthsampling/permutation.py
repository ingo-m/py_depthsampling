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
lstMdl = [''] # , '_deconv_model_1']

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['stimulus']

# ROI ('v1', 'v2', or 'v3'):
lstRoi = ['v1', 'v2', 'v3']

# Hemisphere ('rh' or 'lh'):
lstHmsph = ['rh']

# Path of depth-profile to load (meta-condition, ROI, hemisphere, condition,
# and deconvolution suffix left open):
strPthPrf = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}{}.npy'  #noqa

# Output path & prefix for plots (meta-condition, ROI, hemisphere, condition,
# and deconvolution suffix left open):
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/permutation/{}_{}_{}_{}{}_'  #noqa

# File type suffix for plot:
strFlTp = '.svg'

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [a.u.]'

# Condition levels (used to complete file names):
# lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst',
#           'Pd_trn', 'Cd_trn', 'Ps_trn']
lstCon = ['Pd_sst', 'Ps_sst', 'Cd_sst']


# Which conditions to compare (list of tuples with condition indices):
lstDiff = [(0, 1), (0, 2), (2, 1)]

# Number of resampling iterations:
varNumIt = 1000000

# Upper and lower bound of confidence interval of permutation null
# distribution (for plot):
varLow = 2.5
varUp = 97.5

# Limits of y-axis:
varYmin = -50.0
varYmax = 50.0

# Figure scaling factor:
varDpi = 100.0
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Preparations

# Number of comparisons:
varNumDiff = len(lstDiff)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Permutation test

# Loop through models, ROIs, hemispheres, and conditions to create plots:
for idxMtaCn in range(len(lstMetaCon)):  #noqa
    for idxMdl in range(len(lstMdl)):  #noqa
        for idxRoi in range(len(lstRoi)):
            for idxHmsph in range(len(lstHmsph)):
                for idxDiff in range(len(lstDiff)):

                    # Condition names:
                    strTmpCon01 = lstCon[lstDiff[idxDiff][0]]
                    strTmpCon02 = lstCon[lstDiff[idxDiff][1]]

                    # Only create plots for stimulus-sustained and
                    # perihpery-transient:
                    lgcPass = (
                               (
                                ('stimulus' in lstMetaCon[idxMtaCn])
                                and
                                ('sst' in strTmpCon01)
                               )
                               or
                               (
                                ('periphery' in lstMetaCon[idxMtaCn])
                                and
                                ('trn' in strTmpCon01)
                                )
                               )
                    if lgcPass:

                        # Plot title:
                        strTtle = ((strTmpCon01 + ' minus ' + strTmpCon02))

                        # Condition name for output file path:
                        strPthCon = ((strTmpCon01 + '_min_' + strTmpCon02))

                        # Path of first depth profile:
                        strTmpPth01 = strPthPrf.format(
                            lstMetaCon[idxMtaCn], lstRoi[idxRoi],
                            lstHmsph[idxHmsph], strTmpCon01, lstMdl[idxMdl])

                        # Path of second depth profile:
                        strTmpPth02 = strPthPrf.format(
                            lstMetaCon[idxMtaCn], lstRoi[idxRoi],
                            lstHmsph[idxHmsph], strTmpCon02, lstMdl[idxMdl])

                        # Load single subject depth profiles (shape
                        # aryDpth[subject, depth]):
                        aryDpth01 = np.load(strTmpPth01)
                        aryDpth02 = np.load(strTmpPth02)

                        # Number of depth levels:
                        varNumDpt = aryDpth01.shape[1]

                        # Run permutation test:
                        aryNull, vecP, aryEmpDiffMdn = permute(aryDpth01,
                                                               aryDpth02,
                                                               varNumIt=10000,
                                                               varLow=varLow,
                                                               varUp=varUp)

                        # Data array to be passed into plotting function,
                        # containing the empirical condition difference and the
                        # permutation difference:
                        aryPlot01 = np.zeros((2, varNumDpt))
                        aryPlot01[0, :] = aryEmpDiffMdn.flatten()
                        aryPlot01[1, :] = aryNull[1, :].flatten()

                        # Data array to be passed into plotting function,
                        # containing the error shading (dummy array for
                        # empirical data, because we do not plot the empirical
                        # variance for better visibility, and the lower and
                        # upper bounds of the permutation distribution:
                        aryPlotErrLw = np.zeros((2, varNumDpt))
                        aryPlotErrLw[1, :] = aryNull[0, :].flatten()
                        aryPlotErrUp = np.zeros((2, varNumDpt))
                        aryPlotErrUp[1, :] = aryNull[2, :].flatten()

                        # Plot empirical condition difference and permutation
                        # null distribution:
                        plt_dpth_prfl(aryPlot01,
                                      None,
                                      varNumDpt,
                                      2,
                                      varDpi,
                                      varYmin,
                                      varYmax,
                                      False,
                                      ['Empirical condition difference',
                                       'Permutation null distribution'],
                                      'Cortical depth level (equivolume)',
                                      'fMRI signal change [a.u.]',
                                      (lstRoi[idxRoi].upper()
                                       + ' '
                                       + lstHmsph[idxHmsph].upper()
                                       + ' '
                                       + strTtle),
                                      True,
                                      (strPthPltOt.format(lstMetaCon[idxMtaCn],
                                                          lstRoi[idxRoi],
                                                          lstHmsph[idxHmsph],
                                                          strPthCon,
                                                          lstMdl[idxMdl])
                                       + strFlTp),
                                      aryCnfLw=aryPlotErrLw,
                                      aryCnfUp=aryPlotErrUp)

                        # Reshape p-values for plot:
                        vecP = vecP.reshape((1, varNumDpt))

                        # Plot p-value:
                        plt_dpth_prfl(vecP,
                                      np.zeros(vecP.shape),
                                      varNumDpt,
                                      1,
                                      varDpi,
                                      0.0,
                                      0.5,
                                      False,
                                      ['p-value'],
                                      'Cortical depth level (equivolume)',
                                      'p-value',
                                      (lstRoi[idxRoi].upper()
                                       + ' '
                                       + lstHmsph[idxHmsph].upper()
                                       + ' '
                                       + strTtle),
                                      False,
                                      (strPthPltOt.format(lstMetaCon[idxMtaCn],
                                                          lstRoi[idxRoi],
                                                          lstHmsph[idxHmsph],
                                                          strPthCon,
                                                          lstMdl[idxMdl])
                                       + 'pval'
                                       + strFlTp),
                                      varNumLblY=6)
# -----------------------------------------------------------------------------
