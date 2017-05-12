# -*- coding: utf-8 -*-

"""
VTK depth samling across subjects.

The purpose of this script is to extract and plot depth profiles from vtk files
using existing ROIs. The ROIs can be created using ds_main.py.
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


# ----------------------------------------------------------------------------
# *** Import modules

import numpy as np
import multiprocessing as mp
from ds_sampleUtils import load_single_vtk_par
from ds_sampleUtils import load_multi_vtk_par
from ds_pltAcrDpth import funcPltAcrDpth


print('-Visualisation of depth sampling results')


# ----------------------------------------------------------------------------
# *** Define parameters

# Region of interest ('v1' or 'v2'):
strRoi = 'v1'

# Path of vtk file with vertex inclusion mask (with subject ID, subject ID, &
# ROI left open; needs to be generated with ds_main.py):
strMsk = '/home/john/PhD/ParCon_Depth_Data/{}/cbs_distcor/lh/{}_vertec_inclusion_mask_{}.vtk'  #noqa

# List of subject identifiers:
lstSubIds = ['20150930',
             '20151118',
             '20151127_01',
             '20151130_02',
             '20161205',
             '20161207',
             '20161212_02',
             '20161214',
             '20161219_01',
             '20161219_02']

# Base path of vtk files with depth-sampled data (with subject ID left open):
strVtkDpth = '/home/john/PhD/ParCon_Depth_Data/{}/cbs_distcor/lh/SD.vtk'

# Number of cortical depths:
varNumDpth = 11

# Beginning of string which precedes vertex data in data vtk files (i.e. in the
# statistical maps):
strPrcdData = 'SCALARS'

# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Title for mean plot:
strTitle = strRoi.upper()

# Limits of y-axis for across subject plot:
varYmin = 0.3
varYmax = 0.9

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'SD'

# Output path for plot:
strPltOt = '/home/john/PhD/Tex/tex_pe/plots_{}/SD.svg'.format(strRoi)

# Figure scaling factor:
varDpi = 80.0

# Normalise by division?
lgcNormDiv = True

# Output path for depth samling results (subject means):
strDpthMeans = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/{}_SD.npy'.format(strRoi)  #noqa

# Maximum number of processes to run in parallel: *** NOT IMPLEMENTED
# varPar = 10

# Number of bootstrap samples for confidence intervals:
varNumIt = 10000

# Lower & upper limits of confidence interval for plot:
varConLw=2.5
varConUp=97.5


# ----------------------------------------------------------------------------
# *** Retrieve single subject depth profiles

print('---Retrieving single subject depth profiles')

# Number of subjects:
varNumSubs = len(lstSubIds)

# Empty list to collect results from parallelised function:
lstParResult = [None] * varNumSubs

# Empty list for processes:
lstPrcs = [None] * varNumSubs

# Create a queue to put the results in:
queOut = mp.Queue()

# Loop through subjects & load depth profiles:
for idxSub in range(0, varNumSubs):

    # Complete file paths:
    strVtkDpthTmp = strVtkDpth.format(lstSubIds[idxSub])

    # Prepare processes that returns subject data:
    lstPrcs[idxSub] = \
        mp.Process(target=load_multi_vtk_par,
                   args=(idxSub,
                         strVtkDpthTmp,
                         varNumDpth,
                         strPrcdData,
                         varNumLne,
                         queOut)
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
lstDpth = [None] * varNumSubs

# Put output into correct order:
for idxRes in range(0, varNumSubs):

    # Index of results (first item in output list):
    varTmpIdx = lstParResult[idxRes][0]

    # Put results into list, in correct order:
    lstDpth[varTmpIdx] = lstParResult[idxRes][1]


# ----------------------------------------------------------------------------
# *** Retrieve single subject ROI masks

print('---Retrieving single subject ROI masks')

# Empty list to collect results from parallelised function:
lstParResult = [None] * varNumSubs

# Empty list for processes:
lstPrcs = [None] * varNumSubs

# Create a queue to put the results in:
queOut = mp.Queue()

# Loop through subjects & load depth profiles:
for idxSub in range(0, varNumSubs):

    # Complete file paths:
    strMskTmp = strMsk.format(lstSubIds[idxSub], lstSubIds[idxSub], strRoi)

    # Prepare processes that returns subject data:
    lstPrcs[idxSub] = \
        mp.Process(target=load_single_vtk_par,
                   args=(idxSub,
                         strMskTmp,
                         strPrcdData,
                         varNumLne,
                         queOut)
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
lstMsks = [None] * varNumSubs

# Put output into correct order:
for idxRes in range(0, varNumSubs):

    # Index of results (first item in output list):
    varTmpIdx = lstParResult[idxRes][0]

    # Put results into list, in correct order:
    lstMsks[varTmpIdx] = lstParResult[idxRes][1]


# ----------------------------------------------------------------------------
# *** Apply mask & take mean across vertices

# Array for subject mean depth profiles, of the form aryDpth[idxSub, idxDpth]:
aryDpth = np.zeros((varNumSubs, varNumDpth))

# Loop through subjects:
for idxSub in range(0, varNumSubs):

    # Depth values for current subject:
    aryDpthTmp = lstDpth[idxSub]

    # Mask for current subject:
    vecMskTmp = lstMsks[idxSub]
    vecMskTmp = np.greater(vecMskTmp, 0.5)

    # Apply mask to depth values (i.e. select depth values from ROI):
    aryDpthTmp = aryDpthTmp[vecMskTmp, :]

    # Take mean across vertices:
    aryDpth[idxSub, :] = np.mean(aryDpthTmp, axis=0)


# ----------------------------------------------------------------------------
# *** Bootstrap confidence intervals

# Random array with subject indicies for bootstrapping of the form
# aryRnd[varNumIt, varNumSmp]. Each row includes the indicies of the subjects
# to the sampled on that iteration.
aryRnd = np.random.randint(0,
                           high=varNumSubs,
                           size=(varNumIt, varNumSubs))

# Array for bootstrap samples, of the form
# aryBoo[idxIteration, idxSubject, idxCondition, idxDpth]):
aryBoo = np.zeros((varNumIt, varNumSubs, varNumDpth))

# Loop through bootstrap iterations:
for idxIt in range(varNumIt):
    # Indices of current bootstrap sample:
    vecRnd = aryRnd[idxIt, :]
    # Put current bootstrap sample into array:
    aryBoo[idxIt, :, :] = aryDpth[vecRnd, :]

# Median for each bootstrap sample (across subjects within the bootstrap
# sample):
aryBooMed = np.median(aryBoo, axis=1)

# Delete large bootstrap array:
del(aryBoo)

# Percentile bootstrap for median:
aryPrct = np.percentile(aryBooMed, (varConLw, varConUp), axis=0)
aryPrctLw = np.array(aryPrct[0, :], ndmin=2)
aryPrctUp = np.array(aryPrct[1, :], ndmin=2)


# ----------------------------------------------------------------------------
# *** Plot result

# Median across subjects:
aryDpthMne = np.median(aryDpth, axis=0, keepdims=True)

# Create plot:
funcPltAcrDpth(aryDpthMne, None, varNumDpth, 1, 80.0, varYmin, varYmax, False,
               [''], strXlabel, strYlabel, strRoi.upper(), False, strPltOt,
               varSizeX=1800.0, varSizeY=1600.0, varNumLblY=6,
               varPadY=(0.1, 0.1), aryCnfLw=aryPrctLw, aryCnfUp=aryPrctUp)


# ----------------------------------------------------------------------------
