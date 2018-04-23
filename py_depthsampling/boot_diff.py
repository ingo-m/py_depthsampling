# -*- coding: utf-8 -*-
"""
Plot difference between conditions.

Plot  median between stimulus conditions difference, bootstrapped across
subjects, with confidence intervals.
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


from py_depthsampling.boot.boot_plot_diff import boot_plot

# -----------------------------------------------------------------------------
# *** Define parameters

# Which draining model to plot ('' for none):
lstMdl = ['', '_deconv_model_1']

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['stimulus', 'periphery']

# ROI ('v1' or 'v2'):
lstRoi = ['v1', 'v2']

# Hemisphere ('rh' or 'lh'):
lstHmsph = ['lh', 'rh']

# Output path for corrected depth-profiles (meta-condition, ROI, hemisphere,
# condition, and model index left open):
strPthPrfOt = '/home/john/PhD/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}{}.npy'  #noqa

# Output path & prefix for plots (meta-condition, ROI, ROI, hemisphere,
# condition, and model index left open):
strPthPltOt = '/home/john/PhD/PacMan_Plots/boot_diff/{}/{}/{}_{}_{}_{}_'  #noqa

# File type suffix for plot:
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [arbitrary units]'

# Condition levels (used to complete file names) - nested list:
lstCon = ['Pd', 'Cd', 'Ps']

# Condition labels:
lstConLbl = ['PacMan Dynamic', 'Control Dynamic', 'PacMan Static']

# Which conditions to compare (list of tuples with condition indices):
lstDiff = [(0, 1), (0, 2)]

# Number of resampling iterations for peak finding (for models 1, 2, and 3) or
# random noise samples (models 4 and 5):
# varNumIt = 10000

# Lower & upper bound of percentile bootstrap (in percent), for bootstrap
# confidence interval (models 1, 2, and 3) - this value is only printed, not
# plotted - or plotted confidence intervals in case of model 5:
varCnfLw = 0.5
varCnfUp = 99.5
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Loop through ROIs / conditions

# Loop through models, ROIs, hemispheres, and conditions to create plots:
for idxMtaCn in range(len(lstMetaCon)):  #noqa
    for idxMdl in range(len(lstMdl)):  #noqa
        for idxRoi in range(len(lstRoi)):
            for idxHmsph in range(len(lstHmsph)):
                for idxDiff in range(len(lstDiff)):

                    # Limits of axes need to be adjusted based on ROI,
                    # condition, hemisphere.

                    if idxRoi == 0:  # v1

                        if idxDiff == 0:  # v1 Pd_min_Cd
                            if lstMetaCon[idxMtaCn] == 'stimulus':
                                if lstMdl[idxMdl] == '':
                                    varYmin = -100.0
                                    varYmax = 100.0
                                if lstMdl[idxMdl] == 'deconv_model_1':
                                    varYmin = -100.0
                                    varYmax = 100.0
                            if lstMetaCon[idxMtaCn] == 'periphery':
                                if lstMdl[idxMdl] == '':
                                    varYmin = -100.0
                                    varYmax = 100.0
                                if lstMdl[idxMdl] == 'deconv_model_1':
                                    varYmin = -100.0
                                    varYmax = 100.0

                        elif idxDiff == 1:  # v1 Pd_min_Ps
                            if lstMetaCon[idxMtaCn] == 'stimulus':
                                if lstMdl[idxMdl] == '':
                                    varYmin = -100.0
                                    varYmax = 100.0
                                if lstMdl[idxMdl] == 'deconv_model_1':
                                    varYmin = -100.0
                                    varYmax = 100.0
                            if lstMetaCon[idxMtaCn] == 'periphery':
                                if lstMdl[idxMdl] == '':
                                    varYmin = -100.0
                                    varYmax = 100.0
                                if lstMdl[idxMdl] == 'deconv_model_1':
                                    varYmin = -100.0
                                    varYmax = 100.0

                    elif idxRoi == 1:  # v2

                        if idxDiff == 0:  # v1 Pd_min_Cd
                            if lstMetaCon[idxMtaCn] == 'stimulus':
                                if lstMdl[idxMdl] == '':
                                    varYmin = -100.0
                                    varYmax = 100.0
                                if lstMdl[idxMdl] == 'deconv_model_1':
                                    varYmin = -100.0
                                    varYmax = 100.0
                            if lstMetaCon[idxMtaCn] == 'periphery':
                                if lstMdl[idxMdl] == '':
                                    varYmin = -100.0
                                    varYmax = 100.0
                                if lstMdl[idxMdl] == 'deconv_model_1':
                                    varYmin = -100.0
                                    varYmax = 100.0

                        elif idxDiff == 1:  # v1 Pd_min_Ps
                            if lstMetaCon[idxMtaCn] == 'stimulus':
                                if lstMdl[idxMdl] == '':
                                    varYmin = -100.0
                                    varYmax = 100.0
                                if lstMdl[idxMdl] == 'deconv_model_1':
                                    varYmin = -100.0
                                    varYmax = 100.0
                            if lstMetaCon[idxMtaCn] == 'periphery':
                                if lstMdl[idxMdl] == '':
                                    varYmin = -100.0
                                    varYmax = 100.0
                                if lstMdl[idxMdl] == 'deconv_model_1':
                                    varYmin = -100.0
                                    varYmax = 100.0

                    # Call drain model function:
                    boot_plot(strPthPrfOt.format(lstMetaCon[idxMtaCn],
                                                 lstRoi[idxRoi],
                                                 lstHmsph[idxHmsph],
                                                 'Pd',
                                                 lstMdl[idxMdl]),
                              strPthPltOt.format(lstMetaCon[idxMtaCn],
                                                 lstRoi[idxRoi],
                                                 lstRoi[idxRoi],
                                                 lstHmsph[idxHmsph],
                                                 str(idxDiff),
                                                 lstMdl[idxMdl]),
                              lstConLbl,
                              varNumIt=10000,
                              varConLw=2.5,
                              varConUp=97.5,
                              strTtl='',
                              varYmin=varYmin,
                              varYmax=varYmax,
                              strXlabel='Cortical depth level (equivolume)',
                              strYlabel='fMRI signal change [arbitrary units]',
                              lgcLgnd=True,
                              tplDiff=lstDiff[idxDiff])
# -----------------------------------------------------------------------------
