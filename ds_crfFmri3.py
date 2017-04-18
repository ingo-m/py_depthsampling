"""
Fit contrast response function for fMRI data.

*** WORK IN PROGRESS ***
*** BOOTSTAPPING VERSION ***

Function of the depth sampling pipeline.

The purpose of this function is to apply fit a contrast response function to
fMRI depth profiles, separately for each depth level.

The contrast response function used here is that proposed by Boynton et al.
(1999). This contrast response function is specifically design for fMRI data.
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

from ds_pltAcrDpth import funcPltAcrDpth
from ds_crfPlot import plt_crf
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# *** Define parameters

# Which CRF to use ('power' for power function or 'hyper' for hyperbolic ratio
# function).
strFunc = 'hyper'

# Path of draining-corrected depth-profiles:
# dicPthDpth = {'V1': '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy',  #noqa
#               'V2': '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2.npy'}  #noqa
dicPthDpth = {'V1': '/Users/john/Desktop/Higher_Level_Analysis/v1_corrected.npy',  #noqa
              'V2': '/Users/john/Desktop/Higher_Level_Analysis/v2_corrected.npy'}  #noqa

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using precent (i.e. from zero to 100), the search for the luminance at
# half maximum response would need to be adjusted.
vecCon = np.array([0.025, 0.061, 0.163, 0.72])

# Output path for plot:
strPthOt = '/Users/john/Desktop/crf'

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


# ----------------------------------------------------------------------------
# *** Define contrast reponse function


# Contrast-fMRI-response function as defined in Boynton et al. (1999).
#   - varR is response
#   - varC is stimulus contrast
#   - varP - determines shape of contrast-response function, typical value: 0.3
#   - varQ - determines shape of contrast-response function, typical value: 2.0
#   - varS - ?
#   - varA - Scaling factor
# def crf_fmri(varC, varS, varA):
#    """Contrast-fMRI-response function as defined in Boynton et al. (1999)"""
#    # varR = varS * np.log(varC) + varA
#    varP = 0.3
#    varQ = 2.0
#    varR = varA * np.divide(
#                            np.power(varC, (varP + varQ)),
#                            (np.power(varC, varQ) + np.power(varS, varQ))
#                            )
#    return varR


# Power function:
def crf_power(varC, varA, varB):
    """
    Power contrast response function.

    Parameters
    ----------
    varC : float
        Stimulus contrast (input parameter).
    varA : float
        Factor. Specifies overall response amplitude.
    varB : float
        Exponent. Specifies the rate of change, or slope, of the function.
        (Free parameter to be fitted.)

    Returns
    -------
    varR : float
        Neuronal response.

    Notes
    -----
    Simple power function. Can be used to model the contrast response of
    visual neurons.
    """
    varR = varA * np.power(varC, varB)
    return varR


def crf_hyper(varC, varRmax, varC50, varN):
    """
    Hyperbolic ratio contrast response function.

    Parameters
    ----------
    varC : float
        Stimulus contrast (input parameter).
    varRmax : float
        The maximum neural response (saturation point). (Free parameter to be
        fitted.)
    varC50 : float
        The contrast that gives a half-maximal response, know as
        semisaturation contrast. The semisaturation constant moves the curve
        horizontally and provides a good index of the contrast sensitivity at
        half the maximum response. (Free parameter to be fitted.)
    varN : float
        Exponent. Specifies the rate of change, or slope, of the function.
        (Free parameter to be fitted.)

    Returns
    -------
    varR : float
        Neuronal response.

    Notes
    -----
    Hyperbolic ratio function, a function used to model the contrast response
    of visual neurons. Also known as Naka-Rushton equation.

    References
    ----------
    - Albrecht, D. G., & Hamilton, D. B. (1982). Striate cortex of monkey and
      cat: contrast response function. Journal of neurophysiology, 48(1),
      217-237.
    - Niemeyer, J. E., & Paradiso, M. A. (2017). Contrast sensitivity, V1
      neural activity, and natural vision. Journal of neurophysiology, 117(2),
      492-508.
    """
    varR = (varRmax
            * np.power(varC, varN)
            / (np.power(varC, varN) + np.power(varC50, varN)))
    return varR


# ----------------------------------------------------------------------------
# *** Load depth profiles

# Number of inputs:
varNumIn = len(dicPthDpth.values())

# List for arrays with depth data for ROIs (i.e. for V1 and V2):
lstDpth = [None] * varNumIn

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):
    # Load array with single-subject corrected depth profiles, of the form
    # aryDpth[idxSub, idxCondition, idxDpth].
    lstDpth[idxIn] = np.load(dicPthDpth.values()[idxIn])



# ----------------------------------------------------------------------------
# *** Prepare bootstrapping

# We will sample subjects with replacement. How many subjects to sample on
# each iteration:
varNumSmp = 9

# How many iterations (i.e. how often to sample):
varNumIt = 1000

# Number of subjects:
varNumSubs = lstDpth[0].shape[0]

# Random array with subject indicies for bootstrapping of the form
# aryRnd[varNumIt, varNumSmp]. Each row includes the indicies of the subjects
# to the sampled on that iteration.
aryRnd = np.random.randint(0,
                           high=varNumSubs,
                           size=(varNumIt, varNumSmp))





# List for arrays with mean depth data for ROIs (i.e. for V1 and V2):
lstDpthMne = [None] * varNumIn

# List for arrays with SEM depth data for ROIs (i.e. for V1 and V2):
lstDpthSem = [None] * varNumIn

# Number of depth levels:
varNumDpth = lstDpth[idxIn].shape[2]



# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):


        # ------------------------------------------------------------------------
        # *** Plot contrast response functions

        # Create string for model parameters of exponential function:
        if strFunc == 'power':
            varParamA = np.around(vecMdlPar[0], decimals=2)
            varParamB = np.around(vecMdlPar[1], decimals=2)
            strMdlTmp = ('Model: R(C) = '
                         + str(varParamA)
                         + ' * C ^ '
                         + str(varParamB)
                         )
        elif strFunc == 'hyper':
            varRmax = np.around(vecMdlPar[0], decimals=2)
            varC50 = np.around(vecMdlPar[1], decimals=2)
            varN = np.around(vecMdlPar[2], decimals=2)
            strMdlTmp = ('R(C) = '
                         + str(varRmax)
                         + ' * (C^'
                         + str(varN)
                         + ' / (C^'
                         + str(varN)
                         + ' + '
                         + str(varC50)
                         + '^'
                         + str(varN)
                         + '))'
                         )

        # strModel = ('R(C) = '
        #             + str(varParamA)
        #             + ' * C^(' + str(varP)
        #             + '+'
        #             + str(varQ)
        #             + ') '
        #             + '/ '
        #             + '(C^'
        #             + str(varQ)
        #             + ' + '
        #             + str(varParamB)
        #             + '^'
        #             + str(varQ)
        #             + ')'
        #             )

        # Title for current CRF plot:
        strTtleTmp = (strTtle
                      + dicPthDpth.keys()[idxIn]
                      + ' , depth level: '
                      + str(idxDpth))

        # Output path for current plot:
        strPthOtTmp = (strPthOt
                       + '_'
                       + dicPthDpth.keys()[idxIn]
                       + '_dpth_'
                       + str(idxDpth)
                       + strFleTyp)

        # Plot CRF for current depth level:
        plt_crf(vecX,
                lstFit[idxIn][idxDpth, :],
                strPthOtTmp,
                vecEmpX=vecCon,
                vecEmpYMne=lstDpthMne[idxIn][:, idxDpth],
                vecEmpYSem=lstDpthSem[idxIn][:, idxDpth],
                varXmin=varXmin,
                varXmax=varXmax,
                varYmin=varYmin,
                varYmax=varYmax,
                strLblX=strLblX,
                strLblY=strLblY,
                strTtle=strTtleTmp,
                varDpi=80.0,
                lgcLgnd=True,
                strMdl=strMdlTmp)

# ----------------------------------------------------------------------------
# *** Plot response at half maximum contrast across depth

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [a.u.]'

# Stack the vectors for the two ROIs (V1 & V2):
aryHlfMaxResp = np.vstack(lstHlfMaxResp[:])

funcPltAcrDpth(aryHlfMaxResp,      # aryData[Condition, Depth]
               np.zeros(np.shape(aryHlfMaxResp)),  # aryError[Con., Depth]
               varNumDpth,         # Number of depth levels (on the x-axis)
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
               (strPthOt + '_half_max_response.png'),
               varSizeX=2000.0,
               varSizeY=1400.0)


# ----------------------------------------------------------------------------
# *** Plot semisaturation contrast

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Percent luminance contrast'

# Stack the vectors for the two ROIs (V1 & V2):
aryHlfMaxCont = np.vstack(lstHlfMaxCont[:])

# Convert contrast values to percent (otherwise rounding will be a problem for
# y-axis values):
aryHlfMaxCont = np.multiply(aryHlfMaxCont, 100.0)

# Line colours:
aryClr = np.array([[0.2, 0.2, 0.9],
                   [0.9, 0.2, 0.2]])

funcPltAcrDpth(aryHlfMaxCont,      # aryData[Condition, Depth]
               np.zeros(np.shape(aryHlfMaxCont)),  # aryError[Con., Depth]
               varNumDpth,         # Number of depth levels (on the x-axis)
               varNumIn,           # Number of conditions (separate lines)
               varDpi,             # Resolution of the output figure
               0.0,                # Minimum of Y axis
               10.0,                # Maximum of Y axis
               False,              # Boolean: whether to convert y axis to %
               dicPthDpth.keys(),  # Labels for conditions (separate lines)
               strXlabel,          # Label on x axis
               strYlabel,          # Label on y axis
               'Semisaturation contrast',  # Figure title
               True,               # Boolean: whether to plot a legend
               (strPthOt + '_semisaturationcontrast.png'),
               aryClr=aryClr,
               varSizeX=2000.0,
               varSizeY=1400.0)


# ----------------------------------------------------------------------------
# *** Plot residual variance across depth

# Lists for mean & SEM of residuals (across conditions):
lstResMne = [None] * varNumIn
lstResSem = [None] * varNumIn

# Mean residual variance across conditions:
for idxIn in range(0, varNumIn):
    # Mean residuals (across conditions):
    lstResMne[idxIn] = np.mean(lstRes[idxIn], axis=0, keepdims=True)
    # Standard error of the mean:
    lstResSem[idxIn] = np.divide(np.std(lstRes[idxIn], axis=0),
                                 np.sqrt(varNumCon))
    lstResSem[idxIn] = np.array(lstResSem[idxIn], ndmin=2)

# Stack the vectors for the two ROIs (V1 & V2):
aryResMne = np.vstack(lstResMne[:])
aryResSem = np.vstack(lstResSem[:])

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Residual variance'

funcPltAcrDpth(aryResMne,          # aryData[Condition, Depth]
               aryResSem,          # aryError[Condition, Depth]
               varNumDpth,         # Number of depth levels (on the x-axis)
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
               (strPthOt + '_modelfit.png'))

# ----------------------------------------------------------------------------
# *** Plot mean residual variance

# Plot of mean residuals for V1 and V2 (average across depth levels and
# conditions).

# Vector with x coordinates of the left sides of the bars:
vecBarX = np.arange(1.0, (varNumIn + 1.0))

# Y data for bars - mean residuals across depth levels (data are averaged
# across conditions so far):
aryResMneMne = np.mean(aryResMne, axis=1)
# Y data for bars - SEM:
aryResMneSem = np.divide(np.std(aryResMne, axis=1),
                         np.sqrt(varNumDpth))

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
                  aryResMneMne,
                  width=0.8,
                  color=(0.3, 0.3, 0.8),
                  tick_label=dicPthDpth.keys(),
                  yerr=aryResMneSem)

# Limits of axes:
varYminBar = 0.0
varYmaxBar = np.around(np.max(aryResMneMne), decimals=2)
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
fig01.savefig((strPthOt + '_modelfit_bars.png'),
              dpi=(varDpi * 2.0),
              facecolor='w',
              edgecolor='w',
              transparent=False,
              frameon=None)

# Close figure:
plt.close(fig01)
# ----------------------------------------------------------------------------
