"""
Fit contrast response function to fMRI data.

Function of the depth sampling pipeline.

THIS VERSION:
Different input/output for tests performed for responses to reviewer comments.
Segmentations are available for the right hemispheres of three subjects, and
the CRF is fitted on these three right hemispheres of three subjects. Because
bootstrapping does not make sense with such a small sample size, no
bootstrapping is performed.

The purpose of this function is to fit a contrast response function to fMRI
depth profiles, separately for each cortical depth level. In order to obtain
an estimate of the across-subjects variability, the fitting is performed
repeatedly on a random subset of subjects (bootstrapping, with replacement).
This ensure stable results (which would not be the case when fitting CRF for
each subject individually).
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


import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from ds_crfParBoot02 import crf_par_02
from ds_pltAcrDpth import funcPltAcrDpth
from ds_crfPlot import plt_crf
from ds_findPeak import find_peak


# ----------------------------------------------------------------------------
# *** Define parameters

# Corrected or uncorrected depth profiles?
strCrct = 'corrected'

# Which CRF to use ('power' for power function or 'hyper' for hyperbolic ratio
# function).
strFunc = 'power'

# File to load bootstrap from / save bootstrap to (corrected/uncorrected and
# power/hyper left open):
strPthNpz = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Depth_Data_Review/Higher_Level_Analysis/bootstrap_{}_{}.npz'  #noqa
strPthNpz = strPthNpz.format(strCrct, strFunc)

# Path of depth-profiles:
if strCrct == 'uncorrected':
    dicPthDpth = {'V1': '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Depth_Data_Review/Higher_Level_Analysis/v1_RH.npy'}  #noqa
if strCrct == 'corrected':
    dicPthDpth = {'V1': '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Depth_Data_Review/Higher_Level_Analysis/v1_RH_corrected_model_1.npy'}  #noqa

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using percent (i.e. from zero to 100), the search for the luminance at
# half maximum response would need to be adjusted.
vecEmpX = np.array([0.025, 0.061, 0.163, 0.72])

# Output path for plot:
if strCrct == 'uncorrected':
    strPthOt = '/home/john/PhD/Tex/contrast_response_boot_review/combined_uncorrected/crf'  #noqa
if strCrct == 'corrected':
    strPthOt = '/home/john/PhD/Tex/contrast_response_boot_review/combined_corrected/crf'  #noqa

# Limits of x-axis for contrast response plots
varXmin = 0.0
varXmax = 1.0

# Limits of y-axis for contrast response plots
varYmin = 0.0
varYmax = 2.5

# Axis labels
strLblX = 'Luminance contrast'
strLblY = 'fMRI signal change [a.u.]'

# File type for CRF plots:
strFleTyp = '.svg'

# Title for contrast response plots
strTtle = ''

# Figure scaling factor:
varDpi = 80.0

# Number of x-values for which to solve the function when calculating model
# fit:
varNumX = 1000

# Lower limits for parameters (factor, exponent) - for power function:
vecLimPowLw = np.array([0.0, 0.0])
# Upper limits for parameters (factor, exponent) - for power function:
vecLimPowUp = np.array([10.0, 1.0])

# Lower limits for parameters (maximum response, semisaturation contrast, and
# exponent) - for hyperbolic function:
vecLimHypLw = np.array([0.0, 0.0, 0.0])
# Upper limits for parameters (maximum response, semisaturation contrast, and
# exponent) - for hyperbolic function:
vecLimHypUp = np.array([np.inf, np.inf, np.inf])
# vecLimHypUp = np.array([10.0, np.inf, np.inf])

# Lower & upper bound of percentile bootstrap (in percent). Note:
# Semisaturation constant depth plots use (2.5, 97.5 percentile) interval.
varCnfLw = 0.5
varCnfUp = 99.5

# Number of process to run in parallel:
varPar = 11

# How many iterations (i.e. how often to sample):
varNumIt = 100


# ----------------------------------------------------------------------------
# *** Load depth profiles

print('-CRF fitting')

print(('--' + strCrct.upper() + ' profiles, ' + strFunc + ' CRF.'))

# Number of inputs:
varNumIn = len(dicPthDpth.values())

print('---Loading depth profiles')

# List for arrays with depth data for ROIs (i.e. for V1 and V2):
lstDpth = [None] * varNumIn

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    # Load array with single-subject corrected depth profiles, of the form
    # aryDpth[idxSub, idxCondition, idxDpt]. Array is first placed into
    # list, because we don't know the dimensions of the array yet, so we
    # can't preallocate the final np array yet.
    lstDpth[idxIn] = np.load(dicPthDpth.values()[idxIn])

# Number of subjects:
varNumSubs = lstDpth[0].shape[0]

# Number of conditions:
varNumCon = lstDpth[0].shape[1]  # same as vecEmpX.shape[0]

# Number of depth levels:
varNumDpt = lstDpth[0].shape[2]

# Now that we know the data dimensions, we can relocate the data from a
# list of arrays to an array (so that all relevant data can be saved to
# disk in npz format later): aryDpth[idxRoi, idxSub, idxCondition, idxDpt]
aryDpth = np.zeros((varNumIn, varNumSubs, varNumCon, varNumDpt))
for idxIn in range(0, varNumIn):
    aryDpth[idxIn, :, :, :] = lstDpth[idxIn]


# ----------------------------------------------------------------------------
# *** CRF fitting on full dataset

print('---CRF fitting on full dataset')

# We fit the CRF model on the mean of the full dataset:
aryDpthEmpMed = np.mean(aryDpth, axis=1, keepdims=True)

# Create a queue to put the results in:
queOut = mp.Queue()

# Pseudo-randomisation array to use the bootstrapping function to get empirical
# CRF fit (here, we fit the CRF separately for each subject in order to be
# able to plot error bars without bootstrapping):
aryRnd = np.array([0, 1, 2], ndmin=(2)).T

# Fit contrast response function on empirical depth profiles:
crf_par_02(0,
           aryDpth,
           vecEmpX,
           strFunc,
           aryRnd,
           varNumX,
           queOut)

# Retrieve results from queue:
lstCrf = queOut.get(True)
_, aryEmpMdlY, aryEmpHlfMax, aryEmpSemi, aryEmpRes = lstCrf


# ----------------------------------------------------------------------------
# *** Find peak of response at half maximum contrast

print('---Searching peak of response at half maximum contrast')

# Peak is determined on the mean across subjects:
aryHlfMax = np.mean(aryEmpHlfMax, axis=1, keepdims=True)

# List for arrays for relative position of peaks:
lstPeakHlfMax = [None] * varNumIn

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):
    # Find peaks for response at half maximum:
    lstPeakHlfMax[idxIn] = find_peak(aryHlfMax[idxIn, :, :],
                                     varNumIntp=1000,
                                     varSd=0.05)

# Array for relative peak positions:
vecPeakHlfMaxMed = np.array(lstPeakHlfMax)


# ----------------------------------------------------------------------------
# *** Plot response at half maximum contrast across depth

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [a.u.]'


# Median and percentile:
# aryHlfMaxMne = np.median(aryEmpHlfMax, axis=1)
# aryHlfMaxCnfLw = np.percentile(aryEmpHlfMax, 0.5, axis=1)
# aryHlfMaxCnfUp = np.percentile(aryEmpHlfMax, 99.5, axis=1)

# Mean and standar error:
aryHlfMaxMne = np.mean(aryEmpHlfMax, axis=1)
aryError = np.std(aryEmpHlfMax, axis=1)

funcPltAcrDpth(aryHlfMaxMne,       # aryData[Condition, Depth]
               aryError,           # aryError[Con., Depth]
               varNumDpt,          # Number of depth levels (on the x-axis)
               varNumIn,           # Number of conditions (separate lines)
               varDpi,             # Resolution of the output figure
               0.0,                # Minimum of Y axis
               2.0,                # Maximum of Y axis
               False,              # Boolean: whether to convert y axis to %
               dicPthDpth.keys(),  # Labels for conditions (separate lines)
               strXlabel,          # Label on x axis
               strYlabel,          # Label on y axis
               'Response at 50% contrast',  # Figure title
               True,               # Boolean: whether to plot a legend
               (strPthOt + '_' + strFunc + '_half_max_response' + strFleTyp),
               varSizeX=2000.0,
               varSizeY=1400.0,
               # aryCnfLw=aryHlfMaxCnfLw,
               # aryCnfUp=aryHlfMaxCnfUp,
               lstVrt=list(vecPeakHlfMaxMed))
# ----------------------------------------------------------------------------

print('-Done.')
