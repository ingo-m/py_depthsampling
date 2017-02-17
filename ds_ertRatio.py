# -*- coding: utf-8 -*-
"""
Function of the depth sampling library.

Function of the event-related timecourses depth sampling library.

The purpose of this script is to calculate the ratio between the positive BOLD
response after stimulus onset and the post-stimulus undershoot, across cortical
depth levels. This analysis is performed at the group level (i.e. the ratio is
calculated on the across-subjects event-related averages).

The input to this script are event-related timecourses (for several subjects)
that have been created with 'ds_ertMain' before (saved in an npy file).
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


# *****************************************************************************
# *** Import modules
import numpy as np
from ds_pltAcrDpth import funcPltAcrDpth
# *****************************************************************************


# *****************************************************************************
# *** Define parameters

# Name of npy file from which to load time course data or save time course data
# to:
strPthNpy = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Higher_Level_Analysis/event_related_timecourses/era_v1.npy'  #noqa

# Number of subjects:
varNumSub = 2

# Number of conditions:
varNumCon = 4

# Number of cortical depths:
varNumDpth = 11

# Number of timepoints:
# varNumVol = 14

# Time points at which maximum response is expected (for calculation of ratio
# positive response / undershoot):
tplIdxMax = (5, 6, 7)

# Time points at which post-stimulus undershoot is expected (for calculation
# of ratio positive response / undershoot):
tplIdxMin = (10, 11, 12)

# Condition labels:
# lstConLbl = ['2.5%', '6.1%', '16.3%', '72.0%']
lstConLbl = ['72.0%', '16.3%', '6.1%', '2.5%']

# Title for mean positive responst plot:
strTtl01 = 'Positive response V1'
# Limits of y-axis for ratio-across-depth plot:
varYmin01 = 0.00
varYmax01 = 0.10
# Label for axes:
strXlabel01 = 'Cortical depth level'
strYlabel01 = 'Percent signal change'
# Output path for ratio-across-depth plots:
strPathOut01 = '/home/john/Desktop/tex_era/plots_v1/main_resp_acr_dpths.png'
# Plot legend?
lgcLgnd01 = True

# Title for mean undershoot responst plot:
strTtl02 = 'Undershoot V1'
# Limits of y-axis for ratio-across-depth plot:
varYmin02 = -0.02
varYmax02 = 0.02
# Label for axes:
strXlabel02 = 'Cortical depth level'
strYlabel02 = 'Percent signal change'
# Output path for ratio-across-depth plots:
strPathOut02 = '/home/john/Desktop/tex_era/plots_v1/undrsht_acr_dpths.png'
# Plot legend?
lgcLgnd02 = False

# Title for ratio-across-depth plot:
strTtl03 = 'Ratio positive response / undershoot V1'
# Limits of y-axis for ratio-across-depth plot:
varYmin03 = 1.0
varYmax03 = 1.12
# Label for axes:
strXlabel03 = 'Cortical depth level'
strYlabel03 = 'Ratio'
# Output path for ratio-across-depth plots:
strPathOut03 = '/home/john/Desktop/tex_era/plots_v1/ratio_acr_dpths_not_demeaned.png'  #noqa
# Plot legend?
lgcLgnd03 = False

# Figure scaling factor:
varDpi = 96.0
# *****************************************************************************


# *****************************************************************************
# *** Load data

print('-Depth-dependent BOLD positive-undershoot ratio')

# Number of conditions:
# varNumCon = len(lstCsvPath[0])

print('---Loading data npy file')

# Load previously prepared event-related timecourses from npy file:
aryAllSubsRoiErt = np.load(strPthNpy)

# The array has the form:
# aryAllSubsRoiErt[Subject, Condition, Depth, Volume]
# *****************************************************************************


# *****************************************************************************
# *** Calculate ratio

print('---Calculate ratio')

# Calculate the ratio between the positive BOLD response and the post-stimulus
# undershoot, separately for each subject / condition / depth level. First,
# extract the two respective time segments:
aryMax = aryAllSubsRoiErt[:, :, :, (tplIdxMax)]
aryMin = aryAllSubsRoiErt[:, :, :, (tplIdxMin)]

# Take the mean over time within the two segments:
aryMax = np.mean(aryMax, axis=3)
aryMin = np.mean(aryMin, axis=3)

# Calculate the ratio, element-wise:
aryRatio = np.divide(aryMax, aryMin)
# *****************************************************************************


# *****************************************************************************
# *** Plot ratio across depth levels

print('---Plotting ratio')

# Calculate mean ratio (mean across subjects):
aryRatioMean = np.mean(aryRatio, axis=0)

# Calculate standard error of the mean (for error bar):
aryRatioSem = np.divide(np.std(aryRatio, axis=0),
                        np.sqrt(varNumSub))

# Plot ratio across cortrtical depth:
funcPltAcrDpth(aryRatioMean,
               aryRatioSem,
               varNumDpth,
               varNumCon,
               varDpi,
               varYmin03,
               varYmax03,
               False,
               lstConLbl,
               strXlabel03,
               strYlabel03,
               strTtl03,
               lgcLgnd03,
               strPathOut03)
# *****************************************************************************


# *****************************************************************************
# *** Subtract baseline mean

# The input to this function are timecourses that have been normalised to the
# pre-stimulus baseline individually for each trial. The datapoints are signal
# intensity relative to the pre-stimulus baseline, and the pre-stimulus
# baseline has a mean of one. We subtract one, so that the datapoints are
# percent signal change relative to baseline.
aryAllSubsRoiErt = np.subtract(aryAllSubsRoiErt, 1.0)

# Extract the time segments for the positive response and the post-stimulus
# undershoot again (because they have changed when subtracting the baseline
# mean):
aryMax = aryAllSubsRoiErt[:, :, :, (tplIdxMax)]
aryMin = aryAllSubsRoiErt[:, :, :, (tplIdxMin)]

# Take the mean over time within the two segments:
aryMax = np.mean(aryMax, axis=3)
aryMin = np.mean(aryMin, axis=3)
# *****************************************************************************


# *****************************************************************************
# *** Plot mean positive response across depth levels

print('---Plotting mean positive response')

# Calculate mean positive response across subjects:
aryPosMean = np.mean(aryMax, axis=0)

# Calculate standard error of the mean (for error bar):
aryPosSem = np.divide(np.std(aryMax, axis=0),
                      np.sqrt(varNumSub))

funcPltAcrDpth(aryPosMean,
               aryPosSem,
               varNumDpth,
               varNumCon,
               varDpi,
               varYmin01,
               varYmax01,
               True,
               lstConLbl,
               strXlabel01,
               strYlabel01,
               strTtl01,
               lgcLgnd01,
               strPathOut01)
# *****************************************************************************


# *****************************************************************************
# *** Plot mean undershoot across depth levels

print('---Plotting mean undershoot')

# Calculate mean positive response across subjects:
aryNegMean = np.mean(aryMin, axis=0)

# Calculate standard error of the mean (for error bar):
aryNegSem = np.divide(np.std(aryMin, axis=0),
                      np.sqrt(varNumSub))

funcPltAcrDpth(aryNegMean,
               aryNegSem,
               varNumDpth,
               varNumCon,
               varDpi,
               varYmin02,
               varYmax02,
               True,
               lstConLbl,
               strXlabel02,
               strYlabel02,
               strTtl02,
               lgcLgnd02,
               strPathOut02)
# *****************************************************************************
