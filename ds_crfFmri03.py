"""
Fit contrast response function to fMRI data.

*** BOOTSTAPPING VERSION ***
This version bootstraps the across-subjects CRF fitting.

Function of the depth sampling pipeline.

The purpose of this function is to fit a contrast response function to fMRI
depth profiles, separately for each cortical depth level.
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
import matplotlib.pyplot as plt
from ds_pltAcrDpth import funcPltAcrDpth
from ds_crfPlot import plt_crf
from ds_crfFit import crf_fit

# ----------------------------------------------------------------------------
# *** Define parameters

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
strPthOt = '/home/john/PhD/Tex/contrast_response_boot/crf'

# Limits of x-axis for contrast response plots
varXmin = 0.0
varXmax = 1.0

# Limits of y-axis for contrast response plots
varYmin = 0.0
varYmax = 1.5

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

# Scaling factor for confidence interval (if varCfd = 1.0, the SEM is plotted,
# if varCfd = 1.96, the 95% confidence interval is plotted)
varCfd = 1.96


# ----------------------------------------------------------------------------
# *** Load depth profiles

# Number of inputs:
varNumIn = len(dicPthDpth.values())

# List for arrays with depth data for ROIs (i.e. for V1 and V2):
lstDpth = [None] * varNumIn

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):
    # Load array with single-subject corrected depth profiles, of the form
    # aryDpth[idxSub, idxCondition, idxDpt].
    lstDpth[idxIn] = np.load(dicPthDpth.values()[idxIn])


# ----------------------------------------------------------------------------
# *** Prepare bootstrapping

# We will sample subjects with replacement. How many subjects to sample on
# each iteration:
varNumSmp = 9

# How many iterations (i.e. how often to sample):
varNumIt = 10

# Number of subjects:
varNumSubs = lstDpth[0].shape[0]

# Random array with subject indicies for bootstrapping of the form
# aryRnd[varNumIt, varNumSmp]. Each row includes the indicies of the subjects
# to the sampled on that iteration.
aryRnd = np.random.randint(0,
                           high=varNumSubs,
                           size=(varNumIt, varNumSmp))

# Number of conditions:
varNumCon = lstDpth[idxIn].shape[1] # same as vecEmpX.shape[0]

# Number of depth levels:
varNumDpt = lstDpth[idxIn].shape[2]

# Arrays for y-values of fitted function (for each iteration & depth level):
aryMdlY = np.zeros((varNumIn, varNumIt, varNumDpt, varNumX))

# Array for responses at half maximum contrast:
aryHlfMax = np.zeros((varNumIn, varNumIt, varNumDpt))

# List of vectors for semisaturation contrast:
arySemi = np.zeros((varNumIn, varNumIt, varNumDpt))

# List of arrays for residual variance:
aryRes = np.zeros((varNumIn, varNumIt, varNumCon, varNumDpt))


# ----------------------------------------------------------------------------
# *** Fit contrast response function

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    print('------ROI: ' + dicPthDpth.keys()[idxIn])

    # Loop through bootstrapping iterations:
    for idxIt in range(0, varNumIt):

        print('---------Iteration: ' + str(idxIt))

        # Indicies of subjects to sample on current iteration:
        vecSmpl = aryRnd[idxIt, :]

        # Loop through depth levels:
        for idxDpt in range(0, varNumDpt):

            # Access contrast response profiles of current subset of subjects
            # and current depth level:
            aryEmpY = lstDpth[idxIn][vecSmpl, :, idxDpt]

            # Fit CRF:
            aryMdlY[idxIn, idxIt, idxDpt, :], \
            aryHlfMax[idxIn, idxIt, idxDpt], \
            arySemi[idxIn, idxIt, idxDpt], \
            aryRes[idxIn, idxIt, :, idxDpt] = crf_fit(vecEmpX,
                                                      aryEmpY,
                                                      strFunc='power',
                                                      varNumX=1000,
                                                      varXmin=0.0,
                                                      varXmax=1.0)


# ----------------------------------------------------------------------------
# *** Average across iterations

# Initialise arrays for across-iteration averages & error:
aryMdlYMne = np.zeros((varNumIn, varNumDpt, varNumX))
aryMdlYSem = np.zeros((varNumIn, varNumDpt, varNumX))
aryHlfMaxMne = np.zeros((varNumIn, varNumDpt))
aryHlfMaxSem = np.zeros((varNumIn, varNumDpt))    
arySemiMne = np.zeros((varNumIn, varNumDpt))
arySemiSem = np.zeros((varNumIn, varNumDpt))
aryResMne01 = np.zeros((varNumIn, varNumCon, varNumDpt))
aryResSem01 = np.zeros((varNumIn, varNumCon, varNumDpt))
    
# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    # Mean modelled y-values:
    aryMdlYMne[idxIn, :, :] = np.mean(aryMdlY[idxIn], axis=0)
    # SEM / confidence interval:
    aryMdlYSem[idxIn, :, :] = (
                               (np.std(aryMdlY[idxIn], axis=0)
                               / np.sqrt(varNumIt))
                               * varCfd
                               )

    # Mean response at half-maximum contrast:
    aryHlfMaxMne[idxIn, :] = np.mean(aryHlfMax[idxIn], axis=0)
    # SEM / confidence interval:
    aryHlfMaxSem[idxIn, :] = (
                              (np.std(aryHlfMax[idxIn], axis=0)
                              / np.sqrt(varNumIt))
                              * varCfd
                              )

    # Mean semi-saturation contrast:
    arySemiMne[idxIn, :] = np.mean(arySemi[idxIn], axis=0)
    # SEM / confidence interval:
    arySemiSem[idxIn, :] = (
                            (np.std(arySemi[idxIn], axis=0)
                            / np.sqrt(varNumIt))
                            * varCfd
                            )


    # Mean residuals:
    aryResMne01[idxIn, :, :] = np.mean(aryRes[idxIn], axis=0)
    # SEM / confidence interval:
    aryResSem01[idxIn, :, :] = (
                                (np.std(aryRes[idxIn], axis=0)
                                / np.sqrt(varNumIt))
                                * varCfd
                                )

del(aryMdlY)
del(aryHlfMax)
del(arySemi)
# del(aryRes)


# ------------------------------------------------------------------------
# *** Plot contrast response functions

# Vector for which the function has been fitted:
vecMdlX = np.linspace(varXmin, varXmax, num=varNumX, endpoint=True)

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    # Across-subjects mean of empirical contrast responses:
    vecEmpYMne = np.mean(lstDpth[idxIn], axis=0)
    # SEM / confidence interval:
    vecEmpYSem = (
                  (np.std(lstDpth[idxIn], axis=0)
                  / np.sqrt(varNumSubs))
                  * varCfd
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
                vecMdlYerr=aryMdlYSem[idxIn, idxDpt, :],
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
               aryHlfMaxSem,       # aryError[Con., Depth]
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
               varSizeY=1400.0)


# ----------------------------------------------------------------------------
# *** Plot semisaturation contrast

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Percent luminance contrast'

# Convert contrast values to percent (otherwise rounding will be a problem for
# y-axis values):
arySemiMne = np.multiply(arySemiMne, 100.0)
arySemiSem = np.multiply(arySemiSem, 100.0)

# Line colours:
aryClr = np.array([[0.2, 0.2, 0.9],
                   [0.9, 0.2, 0.2]])

funcPltAcrDpth(arySemiMne,         # aryData[Condition, Depth]
               arySemiSem,         # aryError[Con., Depth]
               varNumDpt,          # Number of depth levels (on the x-axis)
               varNumIn,           # Number of conditions (separate lines)
               varDpi,             # Resolution of the output figure
               0.0,                # Minimum of Y axis
               10.0,               # Maximum of Y axis
               False,              # Boolean: whether to convert y axis to %
               dicPthDpth.keys(),  # Labels for conditions (separate lines)
               strXlabel,          # Label on x axis
               strYlabel,          # Label on y axis
               'Semisaturation contrast',  # Figure title
               True,               # Boolean: whether to plot a legend
               (strPthOt + '_' + strFunc + '_semisaturationcontrast.png'),
               aryClr=aryClr,
               varSizeX=2000.0,
               varSizeY=1400.0)


# ----------------------------------------------------------------------------
# *** Plot residual variance across depth

# Mean residual variance across conditions:
aryResMne02 = np.mean(aryResMne01, axis=1)
aryResSem02 = np.mean(aryResSem01, axis=1)

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Residual variance'

funcPltAcrDpth(aryResMne02,        # aryData[Condition, Depth]
               aryResSem02,        # aryError[Condition, Depth]
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
               (strPthOt + '_' + strFunc + '_modelfit.png'))


# ----------------------------------------------------------------------------
# *** Plot mean residual variance

# Plot of mean residuals for V1 and V2 (average across depth levels and
# conditions).

# Vector with x coordinates of the left sides of the bars:
vecBarX = np.arange(1.0, (varNumIn + 1.0))

# Y data for bars - mean residuals across depth levels (data already averaged
# across iterations & conditions):
aryResMne03 = np.mean(aryResMne01, axis=(1, 2))
# Y data for bars - SEM:
aryResSem03 = (
               (np.std(aryResMne01, axis=(1, 2))
               / np.sqrt(varNumIt))
               * varCfd
               )

# fig01 = plt.figure()
# axs01 = fig01.add_subplot(111, aspect='100.0')

# Figure dimensions:
varSizeX = 400.0
varSizeY = 700.0

# Create plot:
fig01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                            (varSizeY * 0.5) / varDpi),
                   dpi=varDpi)
axs01 = fig01.add_subplot(111)
plt01 = axs01.bar(vecBarX,
                  aryResMne03,
                  width=0.8,
                  color=(0.3, 0.3, 0.8),
                  tick_label=dicPthDpth.keys(),
                  yerr=aryResSem03)

# Limits of axes:
varYminBar = 0.0
varYmaxBar = np.around(np.max(aryResMne03), decimals=2)
axs01.set_ylim([varYminBar, varYmaxBar + 0.005])

# Which y values to label with ticks:
vecYlbl = np.linspace(varYminBar, varYmaxBar, num=4, endpoint=True)
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
