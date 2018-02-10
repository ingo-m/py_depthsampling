# -*- coding: utf-8 -*-
"""
Function of the depth sampling library.

Function of the event-related timecourses depth sampling library.

The purpose of this script is to plot event-related timecourses sampled across
cortical depth levels.

The input to this script are custom-made 'mesh time courses'. Timecourses have
to be cut into event-related segments and averaged across trials (using the
'ds_cutSgmnts.py' script of the depth-sampling library). Depth-sampling has to
be performed with CBS tools, resulting in a 3D mesh for each time point. Here,
3D meshes (with values for all depth-levels at one point in time, for one
condition) are combined across time and conditions to be plotted and analysed.
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
from ds_ertGetSubData import funcGetSubData
from ds_ertPlt import funcPltErt
# *****************************************************************************


# *****************************************************************************
# *** Define parameters

# Load data from previously prepared pickle? If 'False', data is loaded from
# vtk meshes (or npy files) and saved as pickle.
lgcPic = False

# ROI ('v1' or 'v2'):
strRoi = 'v1'

# Hemisphere ('rh' or 'lh'):
strHmsph = 'rh'

# Name of pickle file from which to load time course data or save time course
# data to (ROI name and hemisphere left open):
# strPthPic = '/home/john/PhD/PacMan_Depth_Data/Higher_Level_Analysis/era_long_{}_{}.pickle'  #noqa
strPthPic = '/media/PacMan_Depth_Data/Higher_Level_Analysis/era_long_{}_{}.pickle'  #noqa

# List of subject IDs:
lstSubId = ['20171109',
            '20171211',
            '20171213']

# Condition levels (used to complete file names):
lstCon = ['pacman_dynamic_long']

# Base name of vertex inclusion masks (subject ID, hemisphere, subject ID,
# & ROI left open):
strVtkMsk = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs_distcor/{}/{}_vertex_inclusion_mask_{}.vtk'  #noqa

# Base name of single-volume vtk meshes that together make up the timecourse
# (subject ID, hemisphere, stimulus level, and volume index left open):
strVtkPth = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs_distcor/{}_era/{}/vol_{}.vtk'  #noqa

# Number of cortical depths:
varNumDpth = 11

# Number of timepoints:
varNumVol = 400

# Beginning of string which precedes vertex data in data vtk files (i.e. in the
# statistical maps):
strPrcdData = 'SCALARS'

# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Limits of y-axis:
varAcrSubsYmin = -0.06
varAcrSubsYmax = 0.04

# Convert y-axis values to percent (i.e. divide label values by 100)?
lgcCnvPrct = True

# Label for axes:
strXlabel = 'Time [s]'
strYlabel = 'Percent signal change'

# Volume TR (in seconds, for the plot):
varTr = 2.079

# Time scaling factor (factor by which timecourse was temporally upsampled; if
# it was not upsampled, varTmeScl = 1.0):
varTmeScl = 10.0

# Stimulus onset in seconds (for the plot):
varStimStrt = 5.0 * varTr  # 5 volumes prestimulus interval in ERT

# Stimulus offset in seconds (for the plot):
varStimEnd = varStimStrt + 25.0  # 25 s stimulus plus prestimulus interval

# Condition labels:
# lstConLbl = ['72.0%', '16.3%', '6.1%', '2.5%']
lstConLbl = ['Control dynamic', 'Pacman dynamic', 'Pacman static']

# Plot legend - single subject plots:
lgcLgnd01 = True
# Plot legend - across subject plots:
lgcLgnd02 = True

# Output path for plots - prefix (ROI and hemisphere left open):
# strPltOtPre = '/home/john/PhD/PacMan_Plots/era_long/{}_{}/'
strPltOtPre = '/media/PacMan_Plots/era_long/{}_{}/'
# Output path for plots - suffix:
strPltOtSuf = '_ert_long.png'

# Figure scaling factor:
varDpi = 70.0
# *****************************************************************************


# *****************************************************************************
# *** Load data

print('-Event-related timecourses depth sampling')

# Complete strings:
strPthPic = strPthPic.format(strRoi, strHmsph)
strPltOtPre = strPltOtPre.format(strRoi, strHmsph)

# Number of subjects:
varNumSub = len(lstSubId)

# Number of conditions:
varNumCon = len(lstCon)

if lgcPic:

    print('---Loading data pickle')

    # Load previously prepared event-related timecourses from pickle:
    dicAllSubsRoiErt = pickle.load(open(strPthPic, "rb"))

else:

    print('---Loading data from vtk meshes')

    # Dictionary for ROI event-related averages. NOTE: Once the Depth-sampling
    # can be scripted, this array should be extended to contain one timecourse
    # per trial (per subject & depth level).

    # The keys for the dictionary will be the subject IDs, and for each
    # subject there is an array of the form:
    # aryRoiErt[varNumCon, varNumDpth, varNumVol]
    dicAllSubsRoiErt = {}

    # Loop through subjects and load data:
    for strSubID in lstSubId:

        print(('------Subject: ' + strSubID))

        # Complete file path of vertex inclusion mask for current subject:
        strVtkMskTmp = strVtkMsk.format(strSubID, strHmsph, strSubID, strRoi)

        # Load data for current subject (returns array of the form:
        # aryRoiErt[varNumCon, varNumDpth, varNumVol]):
        dicAllSubsRoiErt[strSubID] = funcGetSubData(strSubID,
                                                    strHmsph,
                                                    strVtkMskTmp,
                                                    strVtkPth,
                                                    lstCon,
                                                    varNumVol,
                                                    varNumDpth,
                                                    strPrcdData,
                                                    varNumLne)

    # Save event-related timecourses to disk as pickle:
    pickle.dump(dicAllSubsRoiErt, open(strPthPic, "wb"))
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
# *** Plot single subjet results (at individual depth levels)

if True:

    print('---Ploting single-subjects event-related averages')

    # Loop through subjects:
    for strSubID, aryRoiErt in dicAllSubsRoiErt.items():

        # Loop through depth levels (we only create plots for three depth
        # levels):
        for idxDpth in [0, 5, 10]:

            # Title for plot:
            strTmpTtl = (strSubID + ' ERA, depth level ' + str(idxDpth))

            # Output filename:
            strTmpPth = (strPltOtPre + strSubID + '_dpth_' + str(idxDpth)
                         + strPltOtSuf)

            # We don't have the variances across trials (within subjects),
            # therefore we create an empty array as a placeholder. NOTE: This
            # should be replaced by between-trial variance once the depth
            # sampling is fully scriptable.
            aryDummy = np.zeros(aryRoiErt[:, idxDpth, :].shape)

            # We create one plot per depth-level.
            funcPltErt(aryRoiErt[:, idxDpth, :],
                       aryDummy,
                       varNumDpth,
                       varNumCon,
                       varNumVol,
                       varDpi,
                       varAcrSubsYmin,
                       varAcrSubsYmax,
                       varStimStrt,
                       varStimEnd,
                       varTr,
                       lstConLbl,
                       lgcLgnd01,
                       strXlabel,
                       strYlabel,
                       lgcCnvPrct,
                       strTmpTtl,
                       strTmpPth,
                       varXlbl=10,
                       varTmeScl=varTmeScl)
# *****************************************************************************


# *****************************************************************************
# *** Plot single subjet results (mean across depth levels)

if True:

    print('---Ploting single-subjects event-related averages - mean across '
          + 'depth levels')

    # Loop through subjects:
    for strSubID, aryRoiErt in dicAllSubsRoiErt.items():

        # Title for plot:
        strTmpTtl = (strSubID + ' ERA, mean across depth levels ')

        # Output filename:
        strTmpPth = (strPltOtPre + strSubID + '_mean_' + strPltOtSuf)

        # Mean ERA across depth levels:
        aryMneTmp = np.mean(aryRoiErt, axis=1)

        # SD across depth levels:
        arySdTmp = np.std(aryRoiErt, axis=1)

        # We create one plot per depth-level.
        funcPltErt(aryMneTmp,
                   arySdTmp,
                   varNumDpth,
                   varNumCon,
                   varNumVol,
                   varDpi,
                   varAcrSubsYmin,
                   varAcrSubsYmax,
                   varStimStrt,
                   varStimEnd,
                   varTr,
                   lstConLbl,
                   lgcLgnd01,
                   strXlabel,
                   strYlabel,
                   lgcCnvPrct,
                   strTmpTtl,
                   strTmpPth,
                   varXlbl=10,
                   varTmeScl=varTmeScl)
# *****************************************************************************


# *****************************************************************************
# *** Plot across-subjects average

print('---Ploting across-subjects average')

# Create across-subjects data array of the form:
# aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]
aryAllSubsRoiErt = np.zeros((varNumSub, varNumCon, varNumDpth, varNumVol))
idxSub = 0
for aryRoiErt in dicAllSubsRoiErt.values():
    aryAllSubsRoiErt[idxSub, :, :, :] = aryRoiErt
    idxSub += 1

# Calculate mean event-related time courses (mean across subjects):
aryRoiErtMean = np.mean(aryAllSubsRoiErt, axis=0)

# Calculate standard error of the mean (for error bar):
aryRoiErtSem = np.divide(np.std(aryAllSubsRoiErt, axis=0),
                         np.sqrt(varNumSub))

# Loop through depth levels:
# for idxDpth in range(0, varNumDpth):
for idxDpth in [0, 5, 10]:

    # Title for plot:
    # strTmpTtl = ('Event-related average, depth level ' + str(idxDpth))
    strTmpTtl = ''

    # Output filename:
    strTmpPth = (strPltOtPre + 'acr_subs_dpth_' + str(idxDpth) + strPltOtSuf)

    # The mean array now has the form:
    # aryRoiErtMean[varNumCon, varNumDpth, varNumVol]

    # We create one plot per depth-level.
    funcPltErt(aryRoiErtMean[:, idxDpth, :],
               aryRoiErtSem[:, idxDpth, :],
               varNumDpth,
               varNumCon,
               varNumVol,
               varDpi,
               varAcrSubsYmin,
               varAcrSubsYmax,
               varStimStrt,
               varStimEnd,
               varTr,
               lstConLbl,
               lgcLgnd02,
               strXlabel,
               strYlabel,
               lgcCnvPrct,
               strTmpTtl,
               strTmpPth,
               varXlbl=10,
               varTmeScl=varTmeScl)
# *****************************************************************************


# *****************************************************************************
# *** Plot across-subjects average (mean across depth levels)

print('---Ploting across-subjects average - mean across depth levels')

# Event-related time courses have the form:
# aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]

# Calculate mean across depth:
aryMneDpth = np.mean(aryAllSubsRoiErt, axis=2)

# Now of the form:
# aryMneTmp[varNumSub, varNumCon, varNumVol]

# Calculate mean across subjects:
aryMneDpthSub = np.mean(aryMneDpth, axis=0)

# Calculate SD across subjects:
arySdDpthSub = np.std(aryMneDpth, axis=0)

# Now of the form:
# aryMneDpthSub[varNumCon, varNumVol]
# arySdDpthSub[varNumCon, varNumVol]

# Title for plot:
strTmpTtl = ''

# Output filename:
strTmpPth = (strPltOtPre + 'acr_dpth_acr_subs' + strPltOtSuf)

# We create one plot per depth-level.
funcPltErt(aryMneDpthSub,
           arySdDpthSub,
           varNumDpth,
           varNumCon,
           varNumVol,
           varDpi,
           varAcrSubsYmin,
           varAcrSubsYmax,
           varStimStrt,
           varStimEnd,
           varTr,
           lstConLbl,
           lgcLgnd02,
           strXlabel,
           strYlabel,
           lgcCnvPrct,
           strTmpTtl,
           strTmpPth,
           varXlbl=10,
           varTmeScl=varTmeScl)
# *****************************************************************************
