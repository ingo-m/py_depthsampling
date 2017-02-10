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
import numpy as np
from ds_ertGetSubData import funcGetSubData
from ds_ertPlt import funcPltErt
# *****************************************************************************


# *****************************************************************************
# *** Define parameters

# Load data from previously prepared npy file? If 'False', data is loaded from
# vtk meshes and saved as npy.
lgcNpy = True

# Name of npy file from which to load time course data or save time course data
# to:
strPthNpy = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Higher_Level_Analysis/event_related_timecourses/era_v1.npy'  #noqa

# List of subject IDs:
lstSubId = ['20150930',
            '20151118',
            '20151127_01',
            '20151130_01',
            '20151130_02']

# Vertex inclusion masks:
lstVtkMsk = ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_vertec_inclusion_mask_v1.vtk',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_vertec_inclusion_mask_v1.vtk',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh/20151127_01_vertec_inclusion_mask_patch_01.vtk',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_vertec_inclusion_mask_v1.vtk',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_vertec_inclusion_mask_v1.vtk']  #noqa

# The paths of single-volume vtk meshes that together make up the timecourse
# are defined in CSV files (one file per condition per subject). Provide the
# paths of those CSV files here (list of lists):
lstCsvPath = [['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh_era/filelist_stim_lvl_01.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh_era/filelist_stim_lvl_02.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh_era/filelist_stim_lvl_03.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh_era/filelist_stim_lvl_04.csv'],  #noqa
              ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh_era/filelist_stim_lvl_01.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh_era/filelist_stim_lvl_02.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh_era/filelist_stim_lvl_03.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh_era/filelist_stim_lvl_04.csv'],  #noqa
              ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh_era/filelist_stim_lvl_01.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh_era/filelist_stim_lvl_02.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh_era/filelist_stim_lvl_03.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh_era/filelist_stim_lvl_04.csv'],  #noqa
              ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh_era/filelist_stim_lvl_01.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh_era/filelist_stim_lvl_02.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh_era/filelist_stim_lvl_03.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh_era/filelist_stim_lvl_04.csv'],  #noqa
              ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh_era/filelist_stim_lvl_01.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh_era/filelist_stim_lvl_02.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh_era/filelist_stim_lvl_03.csv',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh_era/filelist_stim_lvl_04.csv']]  #noqa

# Number of cortical depths:
varNumDpth = 11

# Number of timepoints:
varNumVol = 14

# Beginning of string which precedes vertex data in data vtk files (i.e. in the
# statistical maps):
strPrcdData = 'SCALARS'

# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Limits of axis for single subject plots (list of tuples, [(Ymin, Ymax)]):
lstLimY = [(0.0, 2.5),
           (0.0, 2.5),
           (0.0, 2.5),
           (0.0, 2.5),
           (0.0, 2.5)]

# Limits of axis for across subject plot:
varAcrSubsYmin = -0.02
varAcrSubsYmax = 0.10

# Convert y-axis values to percent (i.e. divide label values by 100)?
lgcCnvPrct = True

# Label for axes:
strXlabel = 'Time [s]'
strYlabel = 'Percent signal change'

# Volume index of start of stimulus period (i.e. index of first volume during
# which stimulus was on - for the plot):
varStimStrt = 3
# Volume index of end of stimulus period (i.e. index of last volume during
# which stimulus was on - for the plot):
varStimEnd = 6
# Volume TR (in seconds, for the plot):
varTr = 2.94

# Condition labels:
# lstConLbl = ['2.5%', '6.1%', '16.3%', '72.0%']
lstConLbl = ['72.0%', '16.3%', '6.1%', '2.5%']
# Plot legend - single subject plots:
lgcLgnd01 = False
# Plot legend - across subject plots:
lgcLgnd02 = False

# Output path for plots - prfix:
strPltOtPre = '/home/john/Desktop/tex_era/plots_v1/'
# Output path for plots - suffix:
strPltOtSuf = '_ert.svg'

# Figure scaling factor:
varDpi = 96.0
# *****************************************************************************


# *****************************************************************************
# *** Load data

print('-Event-related timecourses depth sampling')

# Number of subjects:
varNumSub = len(lstSubId)

# Number of conditions:
varNumCon = len(lstCsvPath[0])

if lgcNpy:

    print('---Loading data npy file')

    # Load previously prepared event-related timecourses from npy file:
    aryAllSubsRoiErt = np.load(strPthNpy)

else:

    print('---Loading data from vtk meshes')

    # Array for ROI event-related averages. NOTE: Once the Depth-sampling can
    # be scripted, this array should be extended to contain one timecourse per
    # trial (per subject & depth level).
    aryAllSubsRoiErt = np.zeros((varNumSub, varNumCon, varNumDpth, varNumVol))

    # Loop through subjects and load data:
    for idxSub in range(0, varNumSub):

        print(('------Subject: ' + lstSubId[idxSub]))

        # Load data for current subject:
        aryAllSubsRoiErt[idxSub, :, :, :] = funcGetSubData(lstSubId[idxSub],
                                                           lstVtkMsk[idxSub],
                                                           lstCsvPath[idxSub],
                                                           varNumDpth,
                                                           varNumVol,
                                                           strPrcdData,
                                                           varNumLne)

        # print(lstCsvPath[idxSub])

    # Save event-related timecourses to disk as npy file:
    np.save(strPthNpy, aryAllSubsRoiErt)
# *****************************************************************************


# *****************************************************************************
# *** Subtract baseline mean

# The input to this function are timecourses that have been normalised to the
# pre-stimulus baseline. The datapoints are signal intensity relative to the
# pre-stimulus baseline, and the pre-stimulus baseline has a mean of one. We
# subtract one, so that the datapoints are percent signal change relative to
# baseline.
aryAllSubsRoiErt = np.subtract(aryAllSubsRoiErt, 1.0)
# *****************************************************************************


# *****************************************************************************
# *** Plot single subjet results

print('---Ploting single-subjects event-related averages')

# Structure of the data array:
# aryAllSubsRoiErt[Subject, Condition, Depth, Volume]

# Loop through subjects:
for idxSub in range(0, varNumSub):

    # Loop through depth levels (we only create plots for three depth levels):
    for idxDpth in [0, 5, 10]:

        # Title for plot:
        # strTmpTtl = ('Event-related average, depth level ' + str(idxDpth))
        strTmpTtl = ''

        # Output filename:
        strTmpPth = (strPltOtPre + lstSubId[idxSub] + '_dpth_' +
                     str(idxDpth) + strPltOtSuf)

        # We don't have the variances across trials (within subjects),
        # therefore we create an empty array as a placeholder. NOTE: This
        # should be replaced by between-trial variance once the depth sampling
        # is fully scriptable.
        aryDummy = np.zeros(aryAllSubsRoiErt[idxSub, :, idxDpth, :].shape)

        # We create one plot per depth-level.
        funcPltErt(aryAllSubsRoiErt[idxSub, :, idxDpth, :],
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
                   strTmpPth)
# *****************************************************************************


# *****************************************************************************
# *** Plot across-subjects average

print('---Ploting across-subjects average')

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
               strTmpPth)
# *****************************************************************************
