# -*- coding: utf-8 -*-
"""
Plot difference between conditions.

Plot mean of between stimulus conditions difference, with SEM.
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


from py_depthsampling.diff.diff_sem import diff_sem


# -----------------------------------------------------------------------------
# *** Define parameters

# Which parameter to plot - 'mean' or 'median'.
strParam = 'mean'

# Which draining model to plot ('' for none):
lstMdl = ['', '_deconv_model_1']

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['centre', 'edge', 'background']

# ROI ('v1', 'v2', or 'v3'):
lstRoi = ['v1', 'v2']

# Path of corrected depth-profiles (meta-condition, ROI, condition, and model
# index left open):
strPthData = '/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/{}/{}_{}{}.npz'  #noqa

# Output path & prefix for plots (meta-condition, ROI, and model index left
# open):
strPthPltOt = '/home/john/PhD/Surface_Plots/diff/{}_{}{}_SEM'  #noqa

# Output path for single subject plot, (ROI, metacondition, hemisphere, drain
# model, and condition left open):
# strPthPtlSnglOt = '/Users/john/Dropbox/PacMan_Plots/diff_sngle/{}_{}_{}{}_{}'

# File type suffix for plot:
strFlTp = '.svg'

# Figure scaling factor:
varDpi = 120.0

# Label for axes:
strXlabel = 'Cortical depth level'
strYlabel = 'Signal change [%]'

# Condition levels (used to complete file names):
lstCon = ['bright_square_sst_pe',
          'kanizsa_rotated_sst_pe',
          'kanizsa_sst_pe']

# Condition labels:
lstConLbl = lstCon

# Which conditions to compare (list of tuples with condition indices):
lstDiff = [(0, 1), (0, 2), (2, 1)]
# lstDiff = [(0, 3)]

# Padding around labelled values on y:
tplPadY = (0.0, 0.05)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Loop through ROIs / conditions

# Loop through models, ROIs, hemispheres, and conditions to create plots:
for idxMtaCn in range(len(lstMetaCon)):  #noqa
    for idxMdl in range(len(lstMdl)):  #noqa
        for idxRoi in range(len(lstRoi)):

            # Limits of axes can be adjusted based on ROI, condition,
            # hemisphere, deconvolution model.

            if lstMdl[idxMdl] == '':
                varYmin = -0.5
                varYmax = 0.5
                varNumLblY = 3
            if lstMdl[idxMdl] == '_deconv_model_1':
                varYmin = -0.25
                varYmax = 0.5
                varNumLblY = 4

            # Create average plots:
            diff_sem(strPthData.format(lstMetaCon[idxMtaCn],
                                       lstRoi[idxRoi],
                                       '{}',
                                       lstMdl[idxMdl]),
                     (strPthPltOt.format(lstMetaCon[idxMtaCn],
                                         lstRoi[idxRoi],
                                         lstMdl[idxMdl])
                      + strFlTp),
                     lstCon,
                     lstConLbl,
                     varYmin=varYmin,
                     varYmax=varYmax,
                     tplPadY=tplPadY,
                     varNumLblY=varNumLblY,
                     varDpi=varDpi,
                     strXlabel=strXlabel,
                     strYlabel=strYlabel,
                     lgcLgnd=True,
                     lstDiff=lstDiff,
                     strParam=strParam)

        # Create single subject plot(s):
        # boot_plot_sngl(strPthData.format(lstMetaCon[idxMtaCn],
        #                                  lstRoi[idxRoi],
        #                                  lstHmsph[idxHmsph],
        #                                  '{}',
        #                                  lstMdl[idxMdl]),
        #                (strPthPtlSnglOt.format(lstRoi[idxRoi],
        #                                        lstMetaCon[idxMtaCn],
        #                                        lstHmsph[idxHmsph],
        #                                        lstMdl[idxMdl],
        #                                        '{}')
        #                 + strFlTp),
        #                lstCon,
        #                lstConLbl,
        #                strXlabel='Cortical depth level (equivolume)',
        #                strYlabel='Subject',
        #                lstDiff=lstDiff)
# -----------------------------------------------------------------------------
