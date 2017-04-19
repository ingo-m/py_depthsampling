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


import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from ds_crfParBoot01 import crf_par_01
from ds_pltAcrDpth import funcPltAcrDpth
from ds_crfPlot import plt_crf


# ----------------------------------------------------------------------------
# *** Define parameters

# Use existing bootstrap or create new one('load' or 'create')?
strSwitch = 'create'

# Pickle to load bootstrap from / save bootstrap to:
strPthPkl = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/bootstrap_uncorrected.pickle'  #noqa

# Pickle file to load bootstrapping results from

# Which CRF to use ('power' for power function or 'hyper' for hyperbolic ratio
# function).
strFunc = 'power'

# Path of draining-corrected depth-profiles:
dicPthDpth = {'V1': '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy',  #noqa
              'V2': '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2.npy'}  #noqa

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using percent (i.e. from zero to 100), the search for the luminance at
# half maximum response would need to be adjusted.
vecEmpX = np.array([0.025, 0.061, 0.163, 0.72])

# Output path for plot:
strPthOt = '/home/john/PhD/Tex/contrast_response_boot/combined_uncorrected/crf'  #noqa

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
strFleTyp = '.png'

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
varNumIt = 10000


# ----------------------------------------------------------------------------
# *** Load / create bootstrap

print('-CRF fitting')

# Number of inputs:
varNumIn = len(dicPthDpth.values())

if strSwitch == 'load':
    
    print('---Loading bootstrapping results from pickle')

    # Load previously prepared pickle:
    lstPkl = pickle.load(open(strPthPkl, "rb"))
    lstDpth, aryMdlY, aryHlfMax, arySemi, aryRes = lstPkl[:]

elif strSwitch == 'create':

    # ------------------------------------------------------------------------
    # *** Load depth profiles

    print('---Loading depth profiles')

    # List for arrays with depth data for ROIs (i.e. for V1 and V2):
    lstDpth = [None] * varNumIn

    # Loop through ROIs (i.e. V1 and V2):
    for idxIn in range(0, varNumIn):
        # Load array with single-subject corrected depth profiles, of the form
        # aryDpth[idxSub, idxCondition, idxDpt].
        lstDpth[idxIn] = np.load(dicPthDpth.values()[idxIn])

    # ------------------------------------------------------------------------
    # *** Parallelised CRF bootstrapping

    aryMdlY, aryHlfMax, arySemi, aryRes = crf_par_01(lstDpth,
                                                     vecEmpX,
                                                     varNumIt=varNumIt,
                                                     varPar=varPar,
                                                     varNumX=varNumX)

    # ------------------------------------------------------------------------
    # *** Save results

    print('---Saving bootstrapping results as pickle')

    # Put results into list and save as pickle:
    lstPkl = [lstDpth, aryMdlY, aryHlfMax, arySemi, aryRes]
    pickle.dump(lstPkl, open(strPthPkl, "wb"))


# ----------------------------------------------------------------------------
# *** Average across iterations

print('---Averaing across iterations')

# Number of subjects:
varNumSubs = lstDpth[0].shape[0]

# Number of conditions:
varNumCon = lstDpth[0].shape[1]  # same as vecEmpX.shape[0]

# Number of depth levels:
varNumDpt = lstDpth[0].shape[2]


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

#aryResMne01 = np.zeros((varNumIn, varNumCon, varNumDpt))
#aryResCnfLw01 = np.zeros((varNumIn, varNumCon, varNumDpt))
#aryResCnfUp01 = np.zeros((varNumIn, varNumCon, varNumDpt))

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    # Mean modelled y-values:
    aryMdlYMne[idxIn, :, :] = np.mean(aryMdlY[idxIn, :, :, :], axis=0)
    # Confidence interval:
    aryMdlYCnfLw[idxIn, :, :] = np.percentile(aryMdlY[idxIn, :, :, :],
                                              varCnfLw,
                                              axis=0)
    aryMdlYCnfUp[idxIn, :, :] = np.percentile(aryMdlY[idxIn, :, :, :],
                                              varCnfUp,
                                              axis=0)

    # Mean response at half-maximum contrast:
    aryHlfMaxMne[idxIn, :] = np.mean(aryHlfMax[idxIn, :, :], axis=0)
    # Confidence interval:
    aryHlfMaxCnfLw[idxIn, :] = np.percentile(aryHlfMax[idxIn, :, :],
                                             varCnfLw,
                                             axis=0)
    aryHlfMaxCnfUp[idxIn, :] = np.percentile(aryHlfMax[idxIn, :, :],
                                             varCnfUp,
                                             axis=0)

    # Mean semi-saturation contrast:
    arySemiMne[idxIn, :] = np.mean(arySemi[idxIn, :, :], axis=0)
    # Confidence interval:
    arySemiSCnfLw[idxIn, :] = np.percentile(arySemi[idxIn, :, :],
                                            varCnfLw,
                                            axis=0)
    arySemiSCnfUp[idxIn, :] = np.percentile(arySemi[idxIn, :, :],
                                            varCnfUp,
                                            axis=0)

#    # Mean residuals:
#    aryResMne01[idxIn, :, :] = np.mean(aryRes[idxIn, :, :, :], axis=0)
#    # Confidence interval:
#    aryResCnfLw01[idxIn, :, :] = np.percentile(aryRes[idxIn, :, :, :],
#                                               varCnfLw,
#                                               axis=0)
#    aryResCnfUp01[idxIn, :, :] = np.percentile(aryRes[idxIn, :, :, :],
#                                               varCnfUp,
#                                               axis=0)

# del(aryMdlY)
# del(aryHlfMax)
# del(arySemi)
# del(aryRes)


# ------------------------------------------------------------------------
# *** Plot contrast response functions

print('---Plotting results')

# Vector for which the function has been fitted:
vecMdlX = np.linspace(varXmin, varXmax, num=varNumX, endpoint=True)

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    # Across-subjects mean of empirical contrast responses:
    vecEmpYMne = np.mean(lstDpth[idxIn], axis=0)
    # SEM:
    vecEmpYSem = np.divide(np.std(lstDpth[idxIn], axis=0),
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
                      + ' , depth level: '
                      + str(idxDpt))

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
               'Response at half maximum contrast',  # Figure title
               True,               # Boolean: whether to plot a legend
               (strPthOt + '_' + strFunc + '_half_max_response.png'),
               varSizeX=2000.0,
               varSizeY=1400.0,
               aryCnfLw=aryHlfMaxCnfLw,
               aryCnfUp=aryHlfMaxCnfUp)


# ----------------------------------------------------------------------------
# *** Plot semisaturation contrast

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Percent luminance contrast'

# Convert contrast values to percent (otherwise rounding will be a problem for
# y-axis values):
arySemiMne = np.multiply(arySemiMne, 100.0)
arySemiSCnfLw = np.multiply(arySemiSCnfLw, 100.0)
arySemiSCnfUp = np.multiply(arySemiSCnfUp, 100.0)

# Line colours:
aryClr = np.array([[0.2, 0.2, 0.9],
                   [0.9, 0.2, 0.2]])

funcPltAcrDpth(arySemiMne,         # aryData[Condition, Depth]
               0,                  # aryError[Con., Depth]
               varNumDpt,          # Number of depth levels (on the x-axis)
               varNumIn,           # Number of conditions (separate lines)
               varDpi,             # Resolution of the output figure
               0.0,                # Minimum of Y axis
               15.0,               # Maximum of Y axis
               False,              # Boolean: whether to convert y axis to %
               dicPthDpth.keys(),  # Labels for conditions (separate lines)
               strXlabel,          # Label on x axis
               strYlabel,          # Label on y axis
               'Semisaturation contrast',  # Figure title
               True,               # Boolean: whether to plot a legend
               (strPthOt + '_' + strFunc + '_semisaturationcontrast.png'),
               aryClr=aryClr,
               varSizeX=2000.0,
               varSizeY=1400.0,
               varNumLblY=4,
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
               (strPthOt + '_' + strFunc + '_modelfit.png'),
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
                  color=(0.3, 0.3, 0.8),
                  tick_label=dicPthDpth.keys(),
                  yerr=aryResCnf04)

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
fig01.savefig((strPthOt + '_' + strFunc + '_modelfit_bars.png'),
              dpi=(varDpi * 2.0),
              facecolor='w',
              edgecolor='w',
              transparent=False,
              frameon=None)

# Close figure:
plt.close(fig01)
# ----------------------------------------------------------------------------

print('-Done.')
