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
import pickle
import numpy as np
from ds_pltAcrDpth import funcPltAcrDpth
# *****************************************************************************


# *****************************************************************************
# *** Define parameters

# Name of npy file from which to load time course data or save time course data
# to:
strPthPic = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/era_v1.pickle'  #noqa

# Number of subjects:
varNumSub = 11

# Number of conditions:
varNumCon = 4

# Number of time points in the event-related average segments:
varNumVol = 14

# Number of cortical depths:
varNumDpth = 11

# Number of timepoints:
# varNumVol = 14

# Time points at which maximum response is expected (for calculation of ratio
# positive response / undershoot):
tplIdxMax = (5, 5)

# Time points at which post-stimulus undershoot is expected (for calculation
# of ratio positive response / undershoot):
tplIdxMin = (10, 11, 12)

# Condition labels:
# lstConLbl = ['2.5%', '6.1%', '16.3%', '72.0%']
lstConLbl = ['72.0%', '16.3%', '6.1%', '2.5%']

# Title for single-subject positive response plot:
strTtl04 = 'Positive response V1'
# Limits of y-axis for ratio-across-depth plot:
varYmin04 = 0.0
varYmax04 = 0.1
# Label for axes:
strXlabel04 = 'Cortical depth level'
strYlabel04 = 'Percent signal change'
# Output path for ratio-across-depth plots (subject ID left open):
strPathOut04 = '/home/john/Desktop/tex_era/plots_v1/{}_main_resp_acr_dpths.png'
# Plot legend?
lgcLgnd04 = True

# Title for mean positive response plot:
strTtl01 = 'Positive response V1'
# Limits of y-axis for ratio-across-depth plot:
varYmin01 = 0.0
varYmax01 = 2.0
# Label for axes:
strXlabel01 = 'Cortical depth level'
strYlabel01 = 'Percent signal change'
# Output path for ratio-across-depth plots:
strPathOut01 = '/home/john/Desktop/tex_era/plots_v1/main_resp_acr_dpths.png'
# Plot legend?
lgcLgnd01 = True

# Title for mean undershoot response plot:
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

print('-Depth-dependent BOLD ERT plots')

# Number of conditions:
# varNumCon = len(lstCsvPath[0])

print('---Loading data pickle file')

# Load previously prepared event-related timecourses from pickle:
dicAllSubsRoiErt = pickle.load(open(strPthPic, "rb" ))

# The dictionary contains one array per subject, of the form:
# aryRoiErt[Condition, Depth, Volume]
# *****************************************************************************


# *****************************************************************************
# *** Create across-subjects data array

# Create across-subjects data array of the form:
# aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]
aryAllSubsRoiErt = np.zeros((varNumSub, varNumCon, varNumDpth, varNumVol))
idxSub = 0
for aryRoiErt in dicAllSubsRoiErt.values():
    aryAllSubsRoiErt[idxSub, :, :, :] = aryRoiErt
    idxSub += 1
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
               True,
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
# pre-stimulus baseline. The datapoints are signal intensity relative to the
# pre-stimulus baseline, and the pre-stimulus baseline has a mean of one. We
# subtract one, so that the datapoints are percent signal change relative to
# baseline.
for strSubID, aryRoiErt in dicAllSubsRoiErt.items():
    aryRoiErt = np.subtract(aryRoiErt, 1.0)
    # Is this line necessary (hard copy)?
    dicAllSubsRoiErt[strSubID] = aryRoiErt
# *****************************************************************************


# *****************************************************************************
# *** Create across-subjects data array (again)

# This has to be done again because the data has changed when subtracting the
# baseline mean.

# Create across-subjects data array of the form:
# aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]
aryAllSubsRoiErt = np.zeros((varNumSub, varNumCon, varNumDpth, varNumVol))
idxSub = 0
for aryRoiErt in dicAllSubsRoiErt.values():
    aryAllSubsRoiErt[idxSub, :, :, :] = aryRoiErt
    idxSub += 1

# Likewise, extract the time segments for the positive response and the
# post-stimulus undershoot again :
aryMax = aryAllSubsRoiErt[:, :, :, (tplIdxMax)]
aryMin = aryAllSubsRoiErt[:, :, :, (tplIdxMin)]

# Take the mean over time within the two segments:
aryMax = np.mean(aryMax, axis=3)
aryMin = np.mean(aryMin, axis=3)
# *****************************************************************************


# *****************************************************************************
# *** Plot positive response for each subject

for strSubID, aryRoiErt in dicAllSubsRoiErt.items():

    # Extract the two time segment of the expected maximal response:
    aryPos = aryRoiErt[:, :, (tplIdxMax)]



    # Calculate 'grand mean', i.e. the mean PE across depth levels and
    # conditions, for the current subject:
    # varGrndMean = np.mean(aryPos[:, :])

    # Divide all values by the grand mean:
    # aryPos[:, :] = np.divide(aryPos[:, :], varGrndMean)


    
    # Take the mean over time within the segment:
    aryPosMean = np.mean(aryPos, axis=2)

    # We don't have the variances across trials (within subjects),
    # therefore we create an empty array as a placeholder. NOTE: This
    # should be replaced by between-trial variance once the depth sampling
    # is fully scriptable.
    aryDummy = np.zeros(aryPosMean.shape)

    # Create single-subject depth plots for positive response:
    funcPltAcrDpth(aryPosMean,   # Data to be plotted: aryData[Condition, Depth]
                   aryDummy,   # Error shading: aryError[Condition, Depth]
                   varNumDpth,   # Number of depth levels (on the x-axis)
                   varNumCon,    # Number of conditions (separate lines)
                   varDpi,       # Resolution of the output figure
                   varYmin04,    # Minimum of Y axis
                   varYmax04,    # Maximum of Y axis
                   False,        # Boolean: whether to convert y axis to %
                   lstConLbl,    # Labels for conditions (separate lines)
                   strXlabel04,  # Label on x axis
                   strYlabel04,  # Label on y axis
                   strTtl04,     # Figure title
                   lgcLgnd04,    # Boolean: whether to plot a legend
                   strPathOut04.format(strSubID))  # Output path figure
# *****************************************************************************


# *****************************************************************************
# *** Normalisation by division

for idxSub in range(0, varNumSub):

    # Calculate 'grand mean', i.e. the mean PE across depth levels and
    # conditions, for the current subject:
    varGrndMean = np.mean(aryMax[idxSub, :, :])

    # Divide all values by the grand mean:
    aryMax[idxSub, :, :] = np.divide(aryMax[idxSub, :, :], varGrndMean)
    # aryDpthConf = np.divide(aryDpthConf, varGrndMean)
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
