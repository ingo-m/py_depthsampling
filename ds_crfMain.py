"""
Fit contrast response function to fMRI data.

Function of the depth sampling pipeline.

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
from ds_crfParBoot01 import crf_par_01
from ds_crfParBoot02 import crf_par_02
from ds_pltAcrDpth import funcPltAcrDpth
from ds_crfPlot import plt_crf
from ds_findPeak import find_peak


# ----------------------------------------------------------------------------
# *** Define parameters

# Load existing bootstrap or create new one ('load' or 'create')?
strSwitch = 'create'

# Corrected or  uncorrected depth profiles?
strCrct = 'corrected'

# Which CRF to use ('power' for power function or 'hyper' for hyperbolic ratio
# function).
strFunc = 'power'

# File to load bootstrap from / save bootstrap to (corrected/uncorrected and
# power/hyper left open):



# strPthNpz = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/bootstrap_100k_{}_{}.npz'  #noqa
strPthNpz = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Higher_Level_Analysis/bootstrap_100k_{}_{}.npz'  #noqa



strPthNpz = strPthNpz.format(strCrct, strFunc)

# Path of depth-profiles:
if strCrct == 'uncorrected':
    dicPthDpth = {'V1': '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy',  #noqa
                  'V2': '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2.npy'}  #noqa
if strCrct == 'corrected':
    dicPthDpth = {'V1': '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1_corrected.npy',  #noqa
                  'V2': '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2_corrected.npy'}  #noqa

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using percent (i.e. from zero to 100), the search for the luminance at
# half maximum response would need to be adjusted.
vecEmpX = np.array([0.025, 0.061, 0.163, 0.72])

# Output path for plot:
if strCrct == 'uncorrected':
    strPthOt = '/home/john/PhD/Tex/contrast_response_boot/combined_uncorrected/crf'  #noqa
if strCrct == 'corrected':
    strPthOt = '/home/john/PhD/Tex/contrast_response_boot/combined_corrected/crf'  #noqa

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

# Lower & upper bound of percentile bootstrap (in percent):
varCnfLw = 2.5
varCnfUp = 97.5

# Number of process to run in parallel:
varPar = 11

# How many iterations (i.e. how often to sample):
varNumIt = 100000


# ----------------------------------------------------------------------------
# *** Load / create bootstrap

print('-CRF fitting')

print(('--' + strCrct.upper() + ' profiles, ' + strFunc + ' CRF.'))

# Number of inputs:
varNumIn = len(dicPthDpth.values())

if strSwitch == 'load':

    print('---Loading bootstrapping results from file')

    # Load data from npz file:
    objNpz = np.load(strPthNpz)

    # Retrieve arrays from npz object (dictionary):
    aryDpth = objNpz['aryDpth']
    aryMdlY = objNpz['aryMdlY']
    aryHlfMax = objNpz['aryHlfMax']
    arySemi = objNpz['arySemi']
    aryRes = objNpz['aryRes']

elif strSwitch == 'create':

    # ------------------------------------------------------------------------
    # *** Load depth profiles

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

    # ------------------------------------------------------------------------
    # *** Parallelised CRF bootstrapping

    aryMdlY, aryHlfMax, arySemi, aryRes = crf_par_01(aryDpth,
                                                     vecEmpX,
                                                     strFunc=strFunc,
                                                     varNumIt=varNumIt,
                                                     varPar=varPar,
                                                     varNumX=varNumX)

    # ------------------------------------------------------------------------
    # *** Save results

    print('---Saving bootstrapping results to file')

    # Save result as npz object:
    np.savez(strPthNpz,
             aryDpth=aryDpth,
             aryMdlY=aryMdlY,
             aryHlfMax=aryHlfMax,
             arySemi=arySemi,
             aryRes=aryRes)


# ----------------------------------------------------------------------------
# *** Average across iterations

print('---Averaing across iterations')

# Number of subjects:
varNumSubs = aryDpth.shape[1]

# Number of conditions:
varNumCon = aryDpth.shape[2]

# Number of depth levels:
varNumDpt = aryDpth.shape[3]

# Initialise arrays for across-iteration averages & confidence intervals:

aryMdlYMne = np.zeros((varNumIn, varNumDpt, varNumX))
aryMdlYCnfLw = np.zeros((varNumIn, varNumDpt, varNumX))
aryMdlYCnfUp = np.zeros((varNumIn, varNumDpt, varNumX))

aryHlfMaxMne = np.zeros((varNumIn, varNumDpt))
aryHlfMaxCnfLw = np.zeros((varNumIn, varNumDpt))
aryHlfMaxCnfUp = np.zeros((varNumIn, varNumDpt))

arySemiMne = np.zeros((varNumIn, varNumDpt))
arySemiSCnfLw = np.zeros((varNumIn, varNumDpt))
arySemiSCnfUp = np.zeros((varNumIn, varNumDpt))

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    # Median modelled y-values:
    aryMdlYMne[idxIn, :, :] = np.median(aryMdlY[idxIn, :, :, :], axis=0)
    # Confidence interval:
    aryMdlYCnfLw[idxIn, :, :] = np.percentile(aryMdlY[idxIn, :, :, :],
                                              varCnfLw,
                                              axis=0)
    aryMdlYCnfUp[idxIn, :, :] = np.percentile(aryMdlY[idxIn, :, :, :],
                                              varCnfUp,
                                              axis=0)

    # Median response at half-maximum contrast:
    aryHlfMaxMne[idxIn, :] = np.median(aryHlfMax[idxIn, :, :], axis=0)
    # Confidence interval:
    aryHlfMaxCnfLw[idxIn, :] = np.percentile(aryHlfMax[idxIn, :, :],
                                             varCnfLw,
                                             axis=0)
    aryHlfMaxCnfUp[idxIn, :] = np.percentile(aryHlfMax[idxIn, :, :],
                                             varCnfUp,
                                             axis=0)

    # Median semi-saturation contrast:
    arySemiMne[idxIn, :] = np.median(arySemi[idxIn, :, :], axis=0)
    # Confidence interval:
    arySemiSCnfLw[idxIn, :] = np.percentile(arySemi[idxIn, :, :],
                                            varCnfLw,
                                            axis=0)
    arySemiSCnfUp[idxIn, :] = np.percentile(arySemi[idxIn, :, :],
                                            varCnfUp,
                                            axis=0)


# ----------------------------------------------------------------------------
# *** CRF fitting on full dataset

print('---CRF fitting on full dataset')

# We fit the CRF model on the median of the full dataset:
aryDpthEmpMed = np.median(aryDpth, axis=1, keepdims=True)

# Create a queue to put the results in:
queOut = mp.Queue()

# Pseudo-randomisation array to use the bootstrapping function to get empirical
# CRF fit:
# aryRnd = np.arange(0, varNumSubs, 1)
aryRnd = np.array((0), ndmin=2)

# Fit contrast response function on empirical depth profiles:
crf_par_02(0,
           aryDpthEmpMed,
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

# List for arrays for relative position of peaks:
lstPeakHlfMax = [None] * varNumIn

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):
    # Find peaks for response at half maximum for all bootstrapping
    # iterations:
    lstPeakHlfMax[idxIn] = find_peak(aryHlfMax[idxIn, :, :],
                                     varNumIntp=1000,
                                     varSd=0.05)

# Array for relative peak positions:
vecPeakHlfMaxMed = np.zeros((varNumIn))

# Array for confidence itnerval of relative peak position:
aryPeakHlfMaxCnf = np.zeros((2, varNumIn))

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    # Median relative peak position for response at half maximum:
    vecPeakHlfMaxMed[idxIn] = np.median(lstPeakHlfMax[idxIn], axis=0)

    # Lower bound of confidence interval of relative peak position:
    aryPeakHlfMaxCnf[0, idxIn] = np.percentile(lstPeakHlfMax[idxIn],
                                               varCnfLw,
                                               axis=0)
    # Upper bound of confidence interval of relative peak position:
    aryPeakHlfMaxCnf[1, idxIn] = np.percentile(lstPeakHlfMax[idxIn],
                                               varCnfUp,
                                               axis=0)

# print('---Median relative peak position for response at half maximum,')
# print(('------' + strCrct.upper() + ' profiles, ' + strFunc + ' CRF:'))
# for idxIn in range(0, varNumIn):
#     strTmp = ('---------ROI ' + str(idxIn + 1) + ': '
#               + str(np.around(vecPeakHlfMaxMed[idxIn], decimals=3)))
#     print(strTmp)


# ----------------------------------------------------------------------------
# *** Find peak of semisaturation contrast

print('---Searching peak of semisaturation contrast')

# List for arrays for relative position of peaks:
lstPeakSemi = [None] * varNumIn

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):
    # Find peaks for response at half maximum for all bootstrapping
    # iterations:
    lstPeakSemi[idxIn] = find_peak(arySemi[idxIn, :, :],
                                   varNumIntp=1000)

# Array for relative peak positions:
vecPeakSemiMed = np.zeros((varNumIn))

# Array for confidence itnerval of relative peak position:
aryPeakSemiCnf = np.zeros((2, varNumIn))

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    # Median relative peak position for semisaturation contrast:
    vecPeakSemiMed[idxIn] = np.median(lstPeakSemi[idxIn], axis=0)

    # Lower bound of confidence interval of relative peak position:
    aryPeakSemiCnf[0, idxIn] = np.percentile(lstPeakSemi[idxIn],
                                             varCnfLw,
                                             axis=0)
    # Upper bound of confidence interval of relative peak position:
    aryPeakSemiCnf[1, idxIn] = np.percentile(lstPeakSemi[idxIn],
                                             varCnfUp,
                                             axis=0)


# ----------------------------------------------------------------------------
# *** Plot peak of response at half maximum contrast (violin plot)

# Figure dimensions:
varSizeX = 1000.0
varSizeY = 400.0

# Create plot:
fig01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                            (varSizeY * 0.5) / varDpi),
                   dpi=varDpi)
axs01 = fig01.add_subplot(111)


plt01 = axs01.violinplot((lstPeakHlfMax[0], lstPeakHlfMax[1]),
                         vert=False,
                         showmedians=True,
                         showmeans=False,
                         showextrema=False)
#plt01 = axs01.boxplot(lstPeakHlfMax,
#                      vert=False,
#                      sym='',
#                      widths=0.5)

# Set x-axis range:
axs01.set_xlim([-0.05, 1.05])
# Set y-axis range:
# axs01.set_ylim([0.0, (varNumIn * (varBarH + varBarS) + varBarS)])

# Which x values to label with ticks (WM & CSF boundary):
axs01.set_xticks([0.0, 1.0])

# Set tick labels for x ticks:
axs01.set_xticklabels(['WM', 'CSF'])

# Title:
axs01.set_title('Peak of response at 50% contrast', fontsize=14)

# Make plot & axis labels fit into figure:
plt.tight_layout(pad=0.5)

# Save figure:
fig01.savefig((strPthOt + '_' + strFunc + '_half_max_response_violin'
               + strFleTyp),
              dpi=(varDpi * 2.0),
              facecolor='w',
              edgecolor='w',
              transparent=False,
              frameon=None)

# Close figure:
plt.close(fig01)


# ----------------------------------------------------------------------------
# *** Plot peak of response at half maximum contrast (bar plot)

print('---Plotting results')

# Error bars are plotted as deviation from mean, so we have to take the
# difference between mean and confidence interval, separately for the ROIs:
aryPeakHlfMaxCnfScld = np.zeros(aryPeakHlfMaxCnf.shape)
for idxIn in range(0, varNumIn):
    aryPeakHlfMaxCnfScld[:, idxIn] = np.subtract(aryPeakHlfMaxCnf[:, idxIn],
                                                 vecPeakHlfMaxMed[idxIn])
# The error bars need the absolute deviation as input:
aryPeakHlfMaxCnfScld = np.absolute(aryPeakHlfMaxCnfScld)

# Height of bars:
varBarH = 0.35
# Spacing between bars:
varBarS = 0.2

# Vector with y coordinates of the bars:
vecBarY = np.linspace((varBarH * 0.5 + varBarS),
                      (varBarS * varNumIn
                       + (varBarH * (varNumIn * 0.5) + varBarH * 0.5)),
                      num=varNumIn,
                      endpoint=True)

# Figure dimensions:
varSizeX = 700.0
varSizeY = 300.0

# Create plot:
fig01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                            (varSizeY * 0.5) / varDpi),
                   dpi=varDpi)
axs01 = fig01.add_subplot(111)

plt01 = axs01.barh(vecBarY,
                   vecPeakHlfMaxMed,
                   height=varBarH,
                   color=(0.1, 0.5, 0.75),
                   align='center',
                   tick_label=dicPthDpth.keys(),
                   xerr=aryPeakHlfMaxCnfScld,
                   error_kw=dict(elinewidth=2.5,
                                 capsize=0.0,
                                 capthick=0.0,
                                 ecolor=(0.1, 0.1, 0.1))
                   )

# Reduce framing box:
# axs01.spines['top'].set_visible(False)
# axs01.spines['right'].set_visible(False)
# axs01.spines['bottom'].set_visible(True)
# axs01.spines['left'].set_visible(True)

# Set x-axis range:
axs01.set_xlim([0.0, 1.05])
# Set y-axis range:
axs01.set_ylim([0.0, (varNumIn * (varBarH + varBarS) + varBarS)])

# Which x values to label with ticks (WM & CSF boundary):
axs01.set_xticks([0.0, 1.0])

# Set tick labels for x ticks:
axs01.set_xticklabels(['WM', 'CSF'])

# Title:
axs01.set_title('Peak of response at 50% contrast', fontsize=14)

# Make plot & axis labels fit into figure:
plt.tight_layout(pad=0.5)

# Save figure:
fig01.savefig((strPthOt + '_' + strFunc + '_half_max_response_box'
               + strFleTyp),
              dpi=(varDpi * 2.0),
              facecolor='w',
              edgecolor='w',
              transparent=False,
              frameon=None)

# Close figure:
plt.close(fig01)


# ----------------------------------------------------------------------------
# *** Plot peak of semisaturation contrast

# Error bars are plotted as deviation from mean, so we have to take the
# difference between mean and confidence interval, separately for the ROIs:
aryPeakSemiCnfScld = np.zeros(aryPeakSemiCnf.shape)
for idxIn in range(0, varNumIn):
    aryPeakSemiCnfScld[:, idxIn] = np.subtract(aryPeakSemiCnf[:, idxIn],
                                               vecPeakSemiMed[idxIn])
# The error bars need the absolute deviation as input:
aryPeakSemiCnfScld = np.absolute(aryPeakSemiCnfScld)

# Height of bars:
varBarH = 0.35
# Spacing between bars:
varBarS = 0.2

# Vector with y coordinates of the bars:
vecBarY = np.linspace((varBarH * 0.5 + varBarS),
                      (varBarS * varNumIn
                       + (varBarH * (varNumIn * 0.5) + varBarH * 0.5)),
                      num=varNumIn,
                      endpoint=True)

# Figure dimensions:
varSizeX = 700.0
varSizeY = 300.0

# Create plot:
fig01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                            (varSizeY * 0.5) / varDpi),
                   dpi=varDpi)
axs01 = fig01.add_subplot(111)

plt01 = axs01.barh(vecBarY,
                   vecPeakSemiMed,
                   height=varBarH,
                   color=(0.1, 0.5, 0.75),
                   align='center',
                   tick_label=dicPthDpth.keys(),
                   xerr=aryPeakSemiCnfScld,
                   error_kw=dict(elinewidth=2.5,
                                 capsize=0.0,
                                 capthick=0.0,
                                 ecolor=(0.1, 0.1, 0.1))
                   )

# Reduce framing box:
# axs01.spines['top'].set_visible(False)
# axs01.spines['right'].set_visible(False)
# axs01.spines['bottom'].set_visible(True)
# axs01.spines['left'].set_visible(True)

# Set x-axis range:
axs01.set_xlim([0.0, 1.0])
# Set y-axis range:
axs01.set_ylim([0.0, (varNumIn * (varBarH + varBarS) + varBarS)])

# Which x values to label with ticks (WM & CSF boundary):
axs01.set_xticks([0.05, 0.95])

# Set tick labels for x ticks:
axs01.set_xticklabels(['WM', 'CSF'])

# Title:
axs01.set_title('Peak of semisaturation constant', fontsize=14)

# Make plot & axis labels fit into figure:
plt.tight_layout(pad=0.5)

# Save figure:
fig01.savefig((strPthOt + '_' + strFunc + '_semisaturationcontrast_box'
               + strFleTyp),
              dpi=(varDpi * 2.0),
              facecolor='w',
              edgecolor='w',
              transparent=False,
              frameon=None)

# Close figure:
plt.close(fig01)


# ------------------------------------------------------------------------
# *** Plot contrast response functions

# Vector for which the function has been fitted:
vecMdlX = np.linspace(varXmin, varXmax, num=varNumX, endpoint=True)

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    # Across-subjects mean of empirical contrast responses:
    vecEmpYMne = np.mean(aryDpth[idxIn, :, :, :], axis=0)
    # SEM:
    vecEmpYSem = np.divide(np.std(aryDpth[idxIn, :, :, :], axis=0),
                           np.sqrt(varNumSubs)
                           )

    # Loop through depth levels:
    for idxDpt in range(0, varNumDpt):

        #        # Create string for model parameters of exponential function:
        #        if strFunc == 'power':
        #            varParamA = np.around(vecMdlPar[0], decimals=2)
        #            varParamB = np.around(vecMdlPar[1], decimals=2)
        #            strMdlTmp = ('Model: R(C) = '
        #                         + str(varParamA)
        #                         + ' * C ^ '
        #                         + str(varParamB)
        #                         )
        #        elif strFunc == 'hyper':
        #            varRmax = np.around(vecMdlPar[0], decimals=2)
        #            varC50 = np.around(vecMdlPar[1], decimals=2)
        #            varN = np.around(vecMdlPar[2], decimals=2)
        #            strMdlTmp = ('R(C) = '
        #                         + str(varRmax)
        #                         + ' * (C^'
        #                         + str(varN)
        #                         + ' / (C^'
        #                         + str(varN)
        #                         + ' + '
        #                         + str(varC50)
        #                         + '^'
        #                         + str(varN)
        #                         + '))'
        #                         )

        # Title for current CRF plot:
        strTtleTmp = (strTtle
                      + dicPthDpth.keys()[idxIn]
                      + ', depth level: '
                      + str(np.around(np.divide(np.float64(idxDpt),
                                                np.float64(varNumDpt - 1)),
                                      decimals=1)
                            )
                      )

        # Output path for current plot:
        strPthOtTmp = (strPthOt
                       + '_'
                       + strFunc
                       + '_'
                       + dicPthDpth.keys()[idxIn]
                       + '_dpth_'
                       + str(idxDpt)
                       + strFleTyp)

        # Plot CRF for current depth level:
        plt_crf(vecMdlX,
                aryMdlYMne[idxIn, idxDpt, :],
                strPthOtTmp,
                vecMdlYCnfLw=aryMdlYCnfLw[idxIn, idxDpt, :],
                vecMdlYCnfUp=aryMdlYCnfUp[idxIn, idxDpt, :],
                vecEmpX=vecEmpX,
                vecEmpYMne=vecEmpYMne[:, idxDpt],
                vecEmpYSem=vecEmpYSem[:, idxDpt],
                varXmin=varXmin,
                varXmax=varXmax,
                varYmin=varYmin,
                varYmax=varYmax,
                strLblX=strLblX,
                strLblY=strLblY,
                strTtle=strTtleTmp,
                varDpi=80.0,
                lgcLgnd=True)

# ----------------------------------------------------------------------------
# *** Plot response at half maximum contrast across depth

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [a.u.]'

#funcPltAcrDpth(aryEmpHlfMax[:, 0, :],  # aryData[Condition, Depth]
funcPltAcrDpth(aryHlfMaxMne,       # aryData[Condition, Depth]
               0,                  # aryError[Con., Depth]
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
               aryCnfLw=aryHlfMaxCnfLw,
               aryCnfUp=aryHlfMaxCnfUp,
               lstVrt=list(vecPeakHlfMaxMed))


# ----------------------------------------------------------------------------
# *** Plot semisaturation contrast

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Percent luminance contrast'

# Convert contrast values to percent (otherwise rounding will be a problem for
# y-axis values):
# arySemiMne = np.multiply(arySemiMne, 100.0)
# arySemiSCnfLw = np.multiply(arySemiSCnfLw, 100.0)
# arySemiSCnfUp = np.multiply(arySemiSCnfUp, 100.0)

# Line colours:
aryClr = np.array([[15.0, 255.0, 135.0],
                   [15.0, 15.0, 255.0]])
aryClr = np.divide(aryClr, 255.0)

# funcPltAcrDpth(aryEmpSemi[:, 0, :],  # aryData[Condition, Depth]
funcPltAcrDpth(arySemiMne,         # aryData[Condition, Depth]
               0,                  # aryError[Con., Depth]
               varNumDpt,          # Number of depth levels (on the x-axis)
               varNumIn,           # Number of conditions (separate lines)
               varDpi,             # Resolution of the output figure
               0.0,                # Minimum of Y axis
               0.25,               # Maximum of Y axis
               True,              # Boolean: whether to convert y axis to %
               dicPthDpth.keys(),  # Labels for conditions (separate lines)
               strXlabel,          # Label on x axis
               strYlabel,          # Label on y axis
               'Semisaturation constant',  # Figure title
               True,               # Boolean: whether to plot a legend
               (strPthOt + '_' + strFunc + '_semisaturationcontrast'
                + strFleTyp),
               aryClr=aryClr,
               varSizeX=2000.0,
               varSizeY=1400.0,
               varNumLblY=6,
               aryCnfLw=arySemiSCnfLw,
               aryCnfUp=arySemiSCnfUp)


# ----------------------------------------------------------------------------
# *** Plot residual variance across depth

# aryRes[idxRoi, idxIteration, idxCondition, idxDpt]

# Mean residual variance across conditions:
aryResMne01 = np.mean(aryRes, axis=2)
# Confidence interval - we are interested in the variability across
# iterations, not across conditions, therefore we calculate the confidence
# interval based on the mean across conditions:
aryResCnfLw02 = np.percentile(aryResMne01,
                              varCnfLw,
                              axis=1)
aryResCnfUp02 = np.percentile(aryResMne01,
                              varCnfUp,
                              axis=1)
# Mean residual variance across iterations:
aryResMne02 = np.mean(aryResMne01, axis=1)

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Residual variance'

funcPltAcrDpth(aryResMne02,        # aryData[Condition, Depth]
               0,                  # aryError[Condition, Depth]
               varNumDpt,          # Number of depth levels (on the x-axis)
               varNumIn,           # Number of conditions (separate lines)
               varDpi,             # Resolution of the output figure
               0.0,                # Minimum of Y axis
               0.09,               # Maximum of Y axis
               False,              # Boolean: whether to convert y axis to %
               dicPthDpth.keys(),  # Labels for conditions (separate lines)
               strXlabel,          # Label on x axis
               strYlabel,          # Label on y axis
               'Model fit across cortical depth',  # Figure title
               True,               # Boolean: whether to plot a legend
               (strPthOt + '_' + strFunc + '_modelfit' + strFleTyp),
               aryCnfLw=aryResCnfLw02,
               aryCnfUp=aryResCnfUp02)


# ----------------------------------------------------------------------------
# *** Plot mean residual variance

# aryRes[idxRoi, idxIteration, idxCondition, idxDpt]

# Plot of mean residuals for V1 and V2 (average across depth levels and
# conditions).

# Mean residuals across conditions and depth levels (needed to calculate
# confidence intervals):
aryResMne03 = np.mean(aryRes, axis=(2, 3))

# Confidence interval - we are interested in the variability across
# iterations, not across conditions and/or depth levels, therefore we
# calculate the confidence interval based on the mean across conditions and
# depth levels:
aryResCnfLw04 = np.percentile(aryResMne03,
                              varCnfLw,
                              axis=1)
aryResCnfUp04 = np.percentile(aryResMne03,
                              varCnfUp,
                              axis=1)
# Mean residuals across iterations (y data for bars):
aryResMne04 = np.mean(aryResMne03, axis=1)

# Upper limit of y-axis (needs to be calculated before scaling of confidence
# interval):
varYmaxBar = np.around(np.max(aryResCnfUp04), decimals=2)

varYmaxBar = 0.06

# Error bars are plotted as deviation from mean, so we have take difference
# between mean and confidence interval:
aryResCnfLw04 = np.absolute(np.subtract(aryResMne04, aryResCnfLw04))
aryResCnfUp04 = np.absolute(np.subtract(aryResMne04, aryResCnfUp04))

# Stack confidence intervals:
aryResCnf04 = np.array((aryResCnfLw04, aryResCnfUp04))

# fig01 = plt.figure()
# axs01 = fig01.add_subplot(111, aspect='100.0')

# Vector with x coordinates of the left sides of the bars:
vecBarX = np.arange(1.0, (varNumIn + 1.0))

# Figure dimensions:
varSizeX = 400.0
varSizeY = 700.0

# Create plot:
fig01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                            (varSizeY * 0.5) / varDpi),
                   dpi=varDpi)
axs01 = fig01.add_subplot(111)
plt01 = axs01.bar(vecBarX,
                  aryResMne04,
                  width=0.8,
                  color=(0.1, 0.5, 0.75),
                  tick_label=dicPthDpth.keys())  # yerr=aryResCnf04)

# Limits of axes:
varYminBar = 0.0
axs01.set_ylim([varYminBar, varYmaxBar + 0.005])

# Which y values to label with ticks:
vecYlbl = np.linspace(varYminBar, varYmaxBar, num=8, endpoint=True)
vecYlbl = np.around(vecYlbl, decimals=2)
# Set ticks:
axs01.set_yticks(vecYlbl)

# Adjust labels:
axs01.tick_params(labelsize=14)
axs01.set_ylabel('Mean residual variance (SEM)', fontsize=16)

# Title:
axs01.set_title('Model fit', fontsize=14)

# Make plot & axis labels fit into figure:
plt.tight_layout(pad=0.5)

# Save figure:
fig01.savefig((strPthOt + '_' + strFunc + '_modelfit_bars' + strFleTyp),
              dpi=(varDpi * 2.0),
              facecolor='w',
              edgecolor='w',
              transparent=False,
              frameon=None)

# Close figure:
plt.close(fig01)
# ----------------------------------------------------------------------------

print('-Done.')
