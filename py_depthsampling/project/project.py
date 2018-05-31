# -*- coding: utf-8 -*-
"""
...
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
import pandas as pd
import seaborn as sns
from py_depthsampling.project.utilities import get_data


# -----------------------------------------------------------------------------
# *** Define parameters

# List of subject identifiers:
lstSubIds = ['20171023',  # '20171109',
             '20171204_01',
             '20171204_02',
             '20171211',
             '20171213',
             '20180111',
             '20180118']

# Draining model suffix ('' for non-corrected profiles):
lstMdl = ['']  # , '_deconv_model_1']

# ROI ('v1' or 'v2'):
lstRoi = ['v1', 'v2']

# Output path & prefix for plots (ROI, condition, and deconvolution suffix left
# open):
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/project/{}_{}_{}'  #noqa

# File type suffix for plot:
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Label for axes:
strXlabel = 'x-position'
strYlabel = 'y-position'

# Condition levels (used to complete file names):
lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst']

# Condition labels:
# lstConLbl = ['PacMan Dynamic Sustained',
#              'Control Dynamic Sustained',
#              'PacMan Static Sustained']

# Path of vtk mesh with data to project into visual space (e.g. parameter
# estimates; subject ID, hemisphere, and contion level left open).
strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/feat_level_2_{}_cope.vtk'  #noqa

# Path of vtk mesh with R2 values from pRF mapping (at multiple depth levels;
# subject ID and hemisphere left open).
strPthR2 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_R2.vtk'  #noqa

# Path of vtk mesh with pRF sizes (at multiple depth levels; subject ID and
# hemisphere left open).
strPthSd = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_SD.vtk'  #noqa

# Path of vtk mesh with pRF x positions at multiple depth levels; subject ID
# and hemisphere left open).
strPthX = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_x_pos.vtk'  #noqa

# Path of vtk mesh with pRF y positions at multiple depth levels; subject ID
# and hemisphere left open).
strPthY = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_y_pos.vtk'  #noqa

# Path of csv file with ROI definition (subject ID, hemisphere, and ROI left
# open).
strCsvRoi = '/home/john/PhD/GitHub/PacMan/analysis/{}/08_depthsampling/{}/{}_mod.csv'  #noqa

# Number of cortical depths.
varNumDpth = 11

# Extent of visual space from centre of the screen in negative x-direction
# (i.e. from the fixation point to the left end of the screen) in degrees of
# visual angle.
varExtXmin = -5.19
# Extent of visual space from centre of the screen in positive x-direction
# (i.e. from the fixation point to the right end of the screen) in degrees of
# visual angle.
varExtXmax = 5.19
# Extent of visual space from centre of the screen in negative y-direction
# (i.e. from the fixation point to the lower end of the screen) in degrees of
# visual angle.
varExtYmin = -5.19
# Extent of visual space from centre of the screen in positive y-direction
# (i.e. from the fixation point to the upper end of the screen) in degrees of
# visual angle.
varExtYmax = 5.19
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Parent loop

# Number of subjects:
varNumSub = len(lstSubIds)

# Loop through models, ROIs, and conditions:
for idxMdl in range(len(lstMdl)):
    for idxRoi in range(len(lstRoi)):
        for idxCon in range(len(lstCon)):

            # -----------------------------------------------------------------
            # *** Load data

            # Number of hemispheres:
            varNumHmsp = varNumSub * 2

            # List for single-subject data vectors:
            lstData = [None] * varNumHmsp

            # List of single subject R2 vectors:
            lstR2 = [None] * varNumHmsp

            # List for single subject SD vectors (pRF sizes):
            lstSd = [None] * varNumHmsp

            # List for single subject x-position vectors:
            lstX = [None] * varNumHmsp

            # List for single subject y-position vectors:
            lstY = [None] * varNumHmsp

            # Loop through subjects:
            for idxSub in range(len(varNumSub)):

                # Temporary input paths:
                strTmpPthData = strPthData.format(lstSubIds[idxSub],
                                                  'lh', lstCon[idxCon])
                strTmpPthR2 = strPthR2.format(lstSubIds[idxSub], 'lh')
                strTmpPthSd = strPthSd.format(lstSubIds[idxSub], 'lh')
                strTmpPthX = strPthX.format(lstSubIds[idxSub], 'lh')
                strTmpPthY = strPthY.format(lstSubIds[idxSub], 'lh')
                strTmpCsvRoi = strCsvRoi(lstSubIds[idxSub], 'lh',
                                         lstRoi[idxRoi])

                # Load single subject data for left hemisphere:
                lstData[idxSub], lstR2[idxSub], lstSd[idxSub], lstX[idxSub],
                lstY[idxSub] = get_data(
                    strTmpPthData, strTmpPthR2, strTmpPthSd, strTmpPthX,
                    strTmpPthY, strTmpCsvRoi, varNumDpth=varNumDpth)

            # -----------------------------------------------------------------
            # *** Combine single subject data

# -----------------------------------------------------------------------------
