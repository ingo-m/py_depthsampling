# -*- coding: utf-8 -*-

"""
VTK depth samling across subjects.

THIS VERSION:
Different output directories (for reproduction of results with different
settings for review).

The purpose of this script is to visualise cortical depth sampling results from
vtk files. The script can visualise statistical maps from vtk files. Vertices
are selected according to several criteria. Two vertex selection criteria are
always applied:

    (1) The vertex has to be contained within the ROI (as defined by by a csv
        file).
    (2) The vertex has to surpass some intensity criterion defined at one depth
        level (legacy option).

Other optional vertex selection criteria are:

    (3) Multi-depth level criterion I -  vertices that are BELOW a certain
        threshold at any depth levels are excluded. For example, a venogram,
        (or a T2* weighted EPI image with low intensities around veins) that is
        defined at all depth level can be used.
    (4) Multi-depth level criterion II - same as (3). Vertices that are BELOW a
        certain threshold at any depth level are excluded.

    (5) Multi-level data distribution criterion I
        Selection based on combination of z-conjunction-mask mask and
        distribution of z-values.
    (6) Multi-level data distribution criterion II
        Calculates maximum data value across depth levels, and excludes
        vertices whose across-depth-maximum-value is at the lower and/or upper
        end of the distribution across vertices (as specified by the user).
        (The distribution across those vertices that have survived all previous
        exclusion criteria is used here.)
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
import multiprocessing as mp
from ds_acrSubsGetData import funcAcrSubGetSubsData
from ds_pltAcrSubsMean import funcPltAcrSubsMean
from ds_linRegAcrSubs import funcLinRegAcrSubs
# *****************************************************************************

print('-Visualisation of depth sampling results')

# *****************************************************************************
# *** Define parameters

# Region of interest ('v1' or 'v2'):
strRoi = 'v1'

# List of subject identifiers:
lstSubIds = ['20150930']

# Condition levels (used to complete file names):
lstCon = ['01', '02', '03', '04']
# Condition labels:
lstConLbl = ['2.5%', '6.1%', '16.3%', '72.0%']

# Base path of first set of vtk files with depth-sampled data, e.g. parameter
# estimates (with subject ID and stimulus level left open):
strVtkDpth01 = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Depth_Data_Review/{}/cbs_distcor_triln/lh/pe_stim_lvl_{}.vtk'  # noqa

# (1)
# Base path of csv files with ROI definition (i.e. patch of cortex selected on
# the surface, e.g. V1 or V2) - i.e. the first vertex selection criterion (with
# subject ID & ROI left open):
strCsvRoi = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Depth_Data_Review/{}/cbs_distcor_triln/lh/{}.csv'  #noqa

# (2)
# Use second selection criterion defined at one depth level (legacy function):
lgcSlct02 = False
# Base path of vtk files with 2nd vertex selection criterion. This vtk file is
# supposed to contain one set of data values (e.g. at mid-grey-matter).
strVtkSlct02 = ''  #noqa
# Threshold for vertex selection for 2nd selection criterion (vertex excluded
# if data value is below threshold):
varThrSlct02 = 100.0

# (3)
# Use third selection criterion (vertices that are BELOW threshold at any depth
# level are excluded):
lgcSlct03 = True
# Path of vtk files with 3rd vertex selection criterion. This vtk file is
# supposed to contain one set of data values for each depth level.
strVtkSlct03 = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Depth_Data_Review/{}/cbs_distcor_triln/lh/R2_multi.vtk'  #noqa
# Threshold for vertex selection:
varThrSlct03 = 0.12

# (4)
# Use fourth selection criterion (vertices that are BELOW threshold at any
# depth level are excluded):
lgcSlct04 = True
# Path of exclusion mask:
strVtkSlct04 = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Depth_Data_Review/{}/cbs_distcor_triln/lh/combined_mean.vtk'  #noqa
# Threshold for vertex selection:
varThrSlct04 = 7000.0

# (5)
# Load second set of vtk data files (z-values) and use them for vertex
# selection?
lgcVtk02 = True
# How many vertices to select for each subject?
lstNumVrtx = [1000] * len(lstSubIds)

# Base name of second set of vtk files with depth-sampled data, e.g. z-values
# (with subject ID and stimulus level left open):
strVtkDpth02 = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Depth_Data_Review/{}/cbs_distcor_triln/lh/zstat_lvl_{}.vtk'  #noqa

# (6)
# Use PE range?
lgcPeRng = False
# Lower bound of PE range (vertices with a maximum PE across depths that
# is below this percentile in the distribution of those values across
# vertices will be excluded):
varPeRngLw = 0.0
# Upper bound of PE range (vertices with a maximum PE across depths that
# is above this percentile in the distribution of those values across
# vertices will be excluded):
varPeRngUp = 70.0

# Number of header lines in ROI CSV file:
varNumHdrRoi = 1

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
lstLimY = [(0.0, 2.5)] * len(lstSubIds)
# lstLimY = [(0.0, 1200.0)] * len(lstSubIds)

# Limits of y-axis for across subject plot:
varAcrSubsYmin = -0.05
varAcrSubsYmax = 2.0
# varAcrSubsYmax = 1200.0

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [arbitrary units]'

# Output path for plots - prfix:
strPltOtPre = '/home/john/PhD/Tex/tex_pe_review/plots_{}_triln/'.format(strRoi)

# Output path for plots - suffix:
strPltOtSuf = '_R2multi_0p12_combinedmean_7000_z_1000_review_z.svg'

# Figure scaling factor:
varDpi = 80.0

# If normalisation - data from which input file to divide by?
# (Indexing starts at zero.) Note: This functionality is not used at the
# moment. Instead of dividing by a reference condition, all profiles are
# divided by the grand mean within subjects before averaging across subjects
# (if lgcNormDiv is true).
varNormIdx = 0

# Normalise by division?
lgcNormDiv = True

# Linear regression?
lgcLinReg = False

# Constrast vector for simple linear regression model (array with one value per
# condition, e.g. per stimulus contrast level):
vecLinRegMdl = np.array([-3.0, -1.0, 1.0, 3.0])

# P-threshold (uncorrected) for regression model (will be corrected for
# multiple comparison automatically): *** NOT IMPLEMENTED
varLinRegP = 0.001

# Range of y-axis for regression plots:
varLinRegYmin = 0.0
varLinRegYmax = 0.18

# Linear regression plot - label for axes:
strLinRegYlabel = 'Regression coefficient'

# Output path for depth samling results (within subject means):
strDpthMeans = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Depth_Data_Review/Higher_Level_Analysis/{}_review_triln.npy'.format(strRoi)  #noqa

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
                                        strTmp) for strTmp in lstCon]

    # Complete file paths:
    strCsvRoiTmp = strCsvRoi.format(lstSubIds[idxSub], strRoi)
    strVtkSlct02Tmp = strVtkSlct02.format(lstSubIds[idxSub])
    strVtkSlct03Tmp = strVtkSlct03.format(lstSubIds[idxSub])
    strVtkSlct04Tmp = strVtkSlct04.format(lstSubIds[idxSub])

    # Create list with complete file names for multi-level data distribution
    # criterion II:
    lstVtkDpth02 = [strVtkDpth02.format(lstSubIds[idxSub],
                                        strTmp) for strTmp in lstCon]
    # lstVtkDpth02 = [strVtkDpth02.format(lstSubIds[idxSub])] * len(lstCon)

    # Prepare processes that plot & return single subject data:
    lstPrcs[idxSub] = \
        mp.Process(target=funcAcrSubGetSubsData,
                   args=(idxSub,                # Process ID
                         lstSubIds[idxSub],     # Data struc - Subject ID
                         lstVtkDpth01,          # Data struc - Pth vtk I
                         strCsvRoiTmp,          # Data struc - ROI CSV fle
                         varNumDpth,    # Data struc - Num. depth levels
                         varNumHdrRoi,  # Data struc - Header lines CSV
                         strPrcdData,   # Data struc - Str. prcd. VTK data
                         varNumLne,     # Data struc - Lns. prcd. data VTK
                         lgcSlct02,             # Criterion 2 - Yes or no?
                         strVtkSlct02Tmp,       # Criterion 2 - VTK path
                         varThrSlct02,          # Criterion 2 - Threshold
                         lgcSlct03,             # Criterion 3 - Yes or no?
                         strVtkSlct03Tmp,       # Criterion 3 - VTK path
                         varThrSlct03,          # Criterion 3 - Threshold
                         lgcSlct04,             # Criterion 4 - Yes or no?
                         strVtkSlct04Tmp,       # Criterion 4 - VTK path
                         varThrSlct04,          # Criterion 4 - Threshold
                         lgcVtk02,              # Criterion 5 - Yes or no?
                         lstVtkDpth02,          # Criterion 5 - VTK path
                         lstNumVrtx[idxSub],    # Criterion 5 - Num vrtx
                         lgcPeRng,              # Criterion 6 - Yes or no?
                         varPeRngLw,            # Criterion 6 - Lower lim
                         varPeRngUp,            # Criterion 6 - Upper lim
                         lgcNormDiv,    # Normalisation - Yes or no?
                         varNormIdx,    # Normalisation - Which reference
                         varDpi,              # Plot - dots per inch
                         lstLimY[idxSub][0],  # Plot - Minimum of Y axis
                         lstLimY[idxSub][1],  # Plot - Maximum of Y axis
                         lstConLbl,           # Plot - Condition labels
                         strXlabel,           # Plot - X axis label
                         strYlabel,           # Plot - Y axis label
                         strTitle,            # Plot - Title
                         strPltOtPre,   # Plot - Output file path prefix
                         strPltOtSuf,   # Plot - Output file path suffix
                         queOut)        # Queue for output list
                   )

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


# *****************************************************************************
# *** Plot mean simple regression (across-subjects)

if lgcLinReg:

    print('---Plot results - simple linear regression (across-subjects).')

    funcLinRegAcrSubs(arySubDpthMns,
                      vecLinRegMdl,
                      varNumSubs,
                      varNumDpth,
                      strTitle,
                      strXlabel,
                      strLinRegYlabel,
                      varLinRegYmin,
                      varLinRegYmax,
                      varLinRegP,
                      varDpi,
                      strPltOtPre,
                      strPltOtSuf)
# *****************************************************************************
