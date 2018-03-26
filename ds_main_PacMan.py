# -*- coding: utf-8 -*-

"""
VTK depth samling across subjects.

The purpose of this script is to visualise cortical depth sampling results from
vtk files. The script can visualise statistical maps from vtk files. Vertices
are selected according to several criteria:

    (1) The vertex has to be contained within the ROI (as defined by by a csv
        file containing the indices of the ROI vertices, can be created with
        paraview based on a retinotopic map).
    (2) Selection criterion 2 -  vertices that are BELOW threshold at any depth
        levels are excluded. (For example, a venogram, or a T2* weighted EPI
        image with low intensities around veins that is defined at all depth
        level can be used.)
    (3) Selection criterion 3 - same as (2). Vertices that are BELOW threshold
        at any depth level are excluded.
    (4) Selection criterion 4 - same as (2). Vertices that are BELOW threshold
        at any depth level are excluded.
"""

# Part of py_depthsampling library
# Copyright (C) 2018  Ingo Marquardt
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
import multiprocessing as mp
from ds_acrSubsGetData import funcAcrSubGetSubsData
from ds_pltAcrSubsMean import funcPltAcrSubsMean
# *****************************************************************************

print('-Visualisation of depth sampling results')

# *****************************************************************************
# *** Define parameters

# Region of interest ('v1' or 'v2'):
strRoi = 'v2'

# Hemisphere ('lh' or 'rh'):
strHmsph = 'rh'

# List of subject identifiers:
lstSubIds = ['20171109',
             '20171211',
             '20171213',
             '20180111',
             '20180118']

# Condition levels (used to complete file names):
lstCon = ['Pd',
          'Cd',
          'Ps']
# lstCon = ['Pd_min_Ps']
# lstCon = ['Pd_min_Cd']


# Condition labels:
lstConLbl = ['PacMan Dynamic',
             'Control Dynamic',
             'PacMan Static']
# lstConLbl = ['PacMan D - PacMan S']
# lstConLbl = ['PacMan D - Control D']

# Base path of vtk files with depth-sampled data, e.g. parameter estimates
# (with subject ID, hemisphere, and stimulus level left open):
strVtkDpth01 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs_distcor/{}/{}_pe1.vtk'  # noqa

# (1)
# Restrict vertex selection to region of interest (ROI)?
lgcSlct01 = True
# Base path of csv files with ROI definition (i.e. patch of cortex selected on
# the surface, e.g. V1 or V2) - i.e. the first vertex selection criterion (with
# subject ID, hemisphere, and ROI left open):
strCsvRoi = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs_distcor/{}/{}.csv'  #noqa
# Number of header lines in ROI CSV file:
varNumHdrRoi = 1

# (2)
# Use vertex selection criterion 2 (vertices that are BELOW threshold at any
# depth level are excluded)?:
lgcSlct02 = True
# Path of vtk files with for vertex selection criterion. This vtk file is
# supposed to contain one set of data values for each depth level. (With
# subject ID and hemisphere left open.)
strVtkSlct02 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs_distcor/{}/R2_multi.vtk'  #noqa
# Threshold for vertex selection:
varThrSlct02 = 0.1

# (3)
# Use vertex selection criterion 2 (vertices that are BELOW threshold at any
# depth level are excluded)?:
lgcSlct03 = True
# Path of vtk files with for vertex selection criterion. This vtk file is
# supposed to contain one set of data values for each depth level. (With
# subject ID and hemisphere left open.)
strVtkSlct03 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs_distcor/{}/combined_mean.vtk'  #noqa
# Threshold for vertex selection:
varThrSlct03 = 7000.0

# (4)
# Use vertex selection criterion 2 (vertices that are BELOW threshold at any
# depth level are excluded)?:
lgcSlct04 = True
# Path of vtk files with for vertex selection criterion. This vtk file is
# supposed to contain one set of data values for each depth level. (With
# subject ID and hemisphere left open.)
if strHmsph == 'lh':
    strVtkSlct04 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs_distcor/{}/pRF_ovrlp_ratio_right_visual_field.vtk'  #noqa
elif strHmsph == 'rh':
    strVtkSlct04 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs_distcor/{}/pRF_ovrlp_ratio_left_visual_field.vtk'  #noqa
# Threshold for vertex selection:
varThrSlct04 = 90.0

# Number of cortical depths:
varNumDpth = 11

# Beginning of string which precedes vertex data in data vtk files (i.e. in the
# statistical maps):
strPrcdData = 'SCALARS'

# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Title for mean plot:
strTitle = strRoi.upper()

# Limits of y-axis for single subject plots (list of tuples, [(Ymin, Ymax)]):
if strRoi == 'v1':
    lstLimY = [(-750.0, 25.0)] * len(lstSubIds)  # v1 simple contrasts
    # lstLimY = [(-30.0, 0.0)] * len(lstSubIds)  # v1 Pd_min_Ps
    # lstLimY = [(0.0, 75.0)] * len(lstSubIds)  # v1 Pd_min_Cd
elif strRoi == 'v2':
    lstLimY = [(-500.0, 20.0)] * len(lstSubIds)  # v2 simple contrasts
    # lstLimY = [(-50.0, 10.0)] * len(lstSubIds)  # v2 Pd_min_Ps
    # lstLimY = [(0.0, 100.0)] * len(lstSubIds)  # v2 Pd_min_Cd

# Limits of y-axis for across subject plot:
varAcrSubsYmin = -750.0
varAcrSubsYmax = 250.0

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal [a.u.]'

# Output path for plots - prefix:
strPltOtPre = '/home/john/PhD/PacMan_Plots/pe/plots_{}/'.format(strRoi)

# Output path for plots - suffix:
strPltOtSuf = '_{}_{}_{}.png'.format(strHmsph, strRoi, lstCon[0])

# Figure scaling factor:
varDpi = 80.0

# If normalisation - data from which input file to divide by?
# (Indexing starts at zero.) Note: This functionality is not used at the
# moment. Instead of dividing by a reference condition, all profiles are
# divided by the grand mean within subjects before averaging across subjects
# (if lgcNormDiv is true).
varNormIdx = 0

# Normalise by division?
lgcNormDiv = False

# Output path for depth samling results (within subject means):
strDpthMeans = '/home/john/PhD/PacMan_Depth_Data/Higher_Level_Analysis/{}_{}.npy'.format(strRoi, strHmsph)  #noqa

# Maximum number of processes to run in parallel: *** NOT IMPLEMENTED
# varPar = 10
# *****************************************************************************


# *****************************************************************************
# *** Plot and retrieve single subject data

print('---Plotting & retrieving single subject data')

# Number of subjects:
varNumSubs = len(lstSubIds)

# Number of conditions (i.e. number of data vtk files per subject):
varNumCon = len(lstCon)

# List for single subject data (mean PE over depth levels):
lstSubData01 = [None] * varNumSubs

# Empty list to collect results from parallelised function:
lstParResult = [None] * varNumSubs

# Empty list for processes:
lstPrcs = [None] * varNumSubs

# Create a queue to put the results in:
queOut = mp.Queue()

# Loop through subjects:
for idxSub in range(0, varNumSubs):

    # Create list with complete file names for the data to be depth-sampled:
    lstVtkDpth01 = [strVtkDpth01.format(lstSubIds[idxSub],
                                        strHmsph,
                                        strTmp) for strTmp in lstCon]

    # Complete file paths:
    strCsvRoiTmp = strCsvRoi.format(lstSubIds[idxSub], strHmsph, strRoi)
    strVtkSlct02Tmp = strVtkSlct02.format(lstSubIds[idxSub], strHmsph)
    strVtkSlct03Tmp = strVtkSlct03.format(lstSubIds[idxSub], strHmsph)
    strVtkSlct04Tmp = strVtkSlct04.format(lstSubIds[idxSub], strHmsph)

    # Prepare processes that plot & return single subject data:
    lstPrcs[idxSub] = \
        mp.Process(target=funcAcrSubGetSubsData,
                   args=(idxSub,              # Process ID
                         lstSubIds[idxSub],   # Data struc - Subject ID
                         lstVtkDpth01,        # Data struc - Pth vtk I
                         varNumDpth,          # Data struc - Num. depth levels
                         strPrcdData,         # Data struc - Str prcd VTK data
                         varNumLne,           # Data struc - Lns prcd data VTK
                         lgcSlct01,           # Criterion 1 - Yes or no?
                         strCsvRoiTmp,        # Criterion 1 - CSV path
                         varNumHdrRoi,        # Criterion 1 - Header lines
                         lgcSlct02,           # Criterion 2 - Yes or no?
                         strVtkSlct02Tmp,     # Criterion 2 - VTK path
                         varThrSlct02,        # Criterion 2 - Threshold
                         lgcSlct03,           # Criterion 3 - Yes or no?
                         strVtkSlct03Tmp,     # Criterion 3 - VTK path
                         varThrSlct03,        # Criterion 3 - Threshold
                         lgcSlct04,           # Criterion 4 - Yes or no?
                         strVtkSlct04Tmp,     # Criterion 4 - VTK path
                         varThrSlct04,        # Criterion 4 - Threshold
                         lgcNormDiv,          # Normalisation - Yes or no?
                         varNormIdx,          # Normalisation - Which reference
                         varDpi,              # Plot - dots per inch
                         lstLimY[idxSub][0],  # Plot - Minimum of Y axis
                         lstLimY[idxSub][1],  # Plot - Maximum of Y axis
                         lstConLbl,           # Plot - Condition labels
                         strXlabel,           # Plot - X axis label
                         strYlabel,           # Plot - Y axis label
                         strTitle,            # Plot - Title
                         strPltOtPre,   # Plot - Output file path prefix
                         strPltOtSuf,   # Plot - Output file path suffix
                         queOut))       # Queue for output list

    # Daemon (kills processes when exiting):
    lstPrcs[idxSub].Daemon = True

# Start processes:
for idxSub in range(0, varNumSubs):
    lstPrcs[idxSub].start()

# Collect results from queue:
for idxSub in range(0, varNumSubs):
    lstParResult[idxSub] = queOut.get(True)

# Join processes:
for idxSub in range(0, varNumSubs):
    lstPrcs[idxSub].join()

# Create list  to put the function output into the correct order:
lstPrcId = [None] * varNumSubs
lstSubData01 = [None] * varNumSubs

# Put output into correct order:
for idxRes in range(0, varNumSubs):

    # Index of results (first item in output list):
    varTmpIdx = lstParResult[idxRes][0]

    # Put fitting results into list, in correct order:
    lstSubData01[varTmpIdx] = lstParResult[idxRes][1]

# Array with single-subject depth sampling results, of the form
# aryDpthMeans[idxSub, idxCondition, idxDpth].
arySubDpthMns = np.zeros((varNumSubs, varNumCon, varNumDpth))

# Retrieve single-subject data from list:
for idxSub in range(0, varNumSubs):
    arySubDpthMns[idxSub, :, :] = lstSubData01[idxSub]
# *****************************************************************************


# *****************************************************************************
# *** Save results

# We save the mean parameter estimates of each subject to disk. This file can
# be used to plot results from different ROIs in one plot.

np.save(strDpthMeans, arySubDpthMns)
# *****************************************************************************


# *****************************************************************************
# *** Plot mean over subjects

print('---Plot results - mean over subjects.')

funcPltAcrSubsMean(arySubDpthMns,
                   varNumSubs,
                   varNumDpth,
                   varNumCon,
                   varDpi,
                   varAcrSubsYmin,
                   varAcrSubsYmax,
                   lstConLbl,
                   strXlabel,
                   strYlabel,
                   strTitle,
                   strPltOtPre,
                   strPltOtSuf)
# *****************************************************************************
