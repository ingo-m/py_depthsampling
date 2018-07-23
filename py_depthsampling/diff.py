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
lstMetaCon = ['stimulus']
# lstMetaCon = ['periphery']

# ROI ('v1', 'v2', or 'v3'):
lstRoi = ['v1', 'v2', 'v3']

# Hemisphere ('rh' or 'lh'):
lstHmsph = ['rh']

# Path of corrected depth-profiles (meta-condition, ROI, hemisphere,
# condition, and model index left open):
strPthData = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}{}.npz'  #noqa

# Output path & prefix for plots (meta-condition, ROI, hemisphere, and
# model index left open):
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/diff/{}_{}_{}{}_SEM'  #noqa

# Output path for single subject plot, (ROI, metacondition, hemisphere, drain
# model, and condition left open):
# strPthPtlSnglOt = '/home/john/Dropbox/PacMan_Plots/diff_sngle/{}_{}_{}{}_{}'

# File type suffix for plot:
strFlTp = '.svg'

# Figure scaling factor:
varDpi = 100.0

# Label for axes:
strXlabel = 'Cortical depth level'
strYlabel = 'Signal change [%]'

# Condition levels (used to complete file names):
lstCon = ['Pd_sst', 'Ps_sst', 'Cd_sst']
# lstCon = ['Pd_sst', 'Ps_sst', 'Cd_sst', 'Ps_sst_plus_Cd_sst']
# lstCon = ['Pd_sst', 'Ps_sst', 'Cd_sst']
# lstCon = ['Pd_trn', 'Ps_trn', 'Cd_trn', 'Ps_trn_plus_Cd_trn']

# Condition labels:
lstConLbl = lstCon

# Which conditions to compare (list of tuples with condition indices):
lstDiff = [(0, 1), (0, 2), (1, 2)]
# lstDiff = [(0, 3)]

# Padding around labelled values on y:
tplPadY = (0.01, 0.01)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Loop through ROIs / conditions

# Loop through models, ROIs, hemispheres, and conditions to create plots:
for idxMtaCn in range(len(lstMetaCon)):  #noqa
    for idxMdl in range(len(lstMdl)):  #noqa
        for idxRoi in range(len(lstRoi)):
            for idxHmsph in range(len(lstHmsph)):

                # Limits of axes need to be adjusted based on ROI,
                # condition, hemisphere.

                if idxRoi == 0:  # v1

                    if lstMetaCon[idxMtaCn] == 'stimulus':
                        if lstMdl[idxMdl] == '':
                            varYmin = -0.25
                            varYmax = 0.5
                        if lstMdl[idxMdl] == '_deconv_model_1':
                            varYmin = -0.25
                            varYmax = 0.5
                    if lstMetaCon[idxMtaCn] == 'periphery':
                        if lstMdl[idxMdl] == '':
                            varYmin = -0.25
                            varYmax = 0.5
                        if lstMdl[idxMdl] == '_deconv_model_1':
                            varYmin = -0.25
                            varYmax = 0.5

                elif (idxRoi == 1) or (idxRoi == 3):  # v2 & v3

                    if lstMetaCon[idxMtaCn] == 'stimulus':
                        if lstMdl[idxMdl] == '':
                            varYmin = -0.25
                            varYmax = 0.5
                        if lstMdl[idxMdl] == '_deconv_model_1':
                            varYmin = -0.25
                            varYmax = 0.5
                    if lstMetaCon[idxMtaCn] == 'periphery':
                        if lstMdl[idxMdl] == '':
                            varYmin = -0.25
                            varYmax = 0.5
                        if lstMdl[idxMdl] == '_deconv_model_1':
                            varYmin = -0.25
                            varYmax = 0.5

                # Create average plots:
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
                         lstConLbl,
                         varYmin=varYmin,
                         varYmax=varYmax,
                         tplPadY=tplPadY,
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
