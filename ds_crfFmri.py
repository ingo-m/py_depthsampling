"""
Fit contrast response function for fMRI data.

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
from scipy.optimize import curve_fit
from ds_pltAcrDpth import funcPltAcrDpth
from ds_crfPlot import plt_crf
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# *** Define parameters

# Path of draining-corrected depth-profiles:
dicPthDpth = {'V1': '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1_corrected.npy',  #noqa
              'V2': '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2_corrected.npy'}  #noqa

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using precent (i.e. from zero to 100), the search for the luminance at
# half maximum response below would need to be adjusted.
vecCon = np.array([0.025, 0.061, 0.163, 0.72])

# Output path for plot:
strPthOt = '/home/john/PhD/Tex/contrast_response/combined_corrected/crf'

# Limits of x-axis for contrast response plots
varXmin = 0.0
varXmax = 1.0

# Limits of y-axis for contrast response plots
varYmin = 0.0
varYmax = 1.5

# Axis labels
strLblX = 'Luminance contrast'
strLblY = 'fMRI signal change [a.u.]'

# Title for contrast response plots
strTtle = ''

# Figure scaling factor:
varDpi = 80.0

# Lower limits for parameters (factor, exponent):
vecLimA = np.array([0.0, 0.0])

# Upper limits for parameters (factor, exponent):
vecLimB = np.array([10.0, 1.0])


# ----------------------------------------------------------------------------
# *** Define contrast reponse function

# Contrast-fMRI-response function as defined in Boynton et al. (1999).
#   - varR is response
#   - varC is stimulus contrast
#   - varP - determines shape of contrast-response function, typical value: 0.3
#   - varQ - determines shape of contrast-response function, typical value: 2.0
#   - varS - ?
#   - varA - Scaling factor
# def funcCrf(varC, varS, varA):
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
def funcCrf(varC, varA, varB):
    """Contrast-fMRI-response function."""
    varR = varA * np.power(varC, varB)
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
# *** Average across subjects

# List for arrays with mean depth data for ROIs (i.e. for V1 and V2):
lstDpthMne = [None] * varNumIn

# List for arrays with SEM depth data for ROIs (i.e. for V1 and V2):
lstDpthSem = [None] * varNumIn

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    # Number of subjects:
    varNumSubs = lstDpth[idxIn].shape[0]

    # Number of conditions:
    varNumCon = lstDpth[idxIn].shape[1]

    # Number of depth levels:
    varNumDpth = lstDpth[idxIn].shape[2]

    # Across-subjects mean for measured response:
    lstDpthMne[idxIn] = np.mean(lstDpth[idxIn], axis=0)

    # Standard error of the mean (across subjects):
    lstDpthSem[idxIn] = np.divide(np.std(lstDpth[idxIn], axis=0),
                                  np.sqrt(varNumSubs))


# ----------------------------------------------------------------------------
# *** Fit CRF across depth levels

# Number of x-values for which to solve the function:
varNumX = 1000

# Vector for which the function will be fitted:
vecX = np.linspace(varXmin, varXmax, num=varNumX, endpoint=True)

# List of vectors for y-values of fitted function (for each depth level):
lstFit = [np.zeros((varNumDpth, varNumX)) for i in range(varNumIn)]

# List of vectors for response at half maximum contrast:
lstHlfMaxResp = [np.zeros((1, varNumDpth)) for i in range(varNumIn)]

# List of vectors for contrast at half maximum response:
lstHlfMaxCont = [np.zeros((1, varNumDpth)) for i in range(varNumIn)]

# List of arrays for residual variance:
lstRes = [np.zeros((varNumCon, varNumDpth)) for i in range(varNumIn)]

# Loop through ROIs (i.e. V1 and V2):
for idxIn in range(0, varNumIn):

    # We fit the contrast response function separately for all depth levels.

    # Loop through depth levels:
    for idxDpth in range(0, varNumDpth):

        # --------------------------------------------------------------------
        # *** Fit contrast reponse function
        vecModelPar, vecModelCov = curve_fit(funcCrf,
                                             vecCon,
                                             lstDpthMne[idxIn][:, idxDpth],
                                             maxfev=100000,
                                             bounds=(vecLimA, vecLimB),
                                             p0=(0.5, 0.5))

        # --------------------------------------------------------------------
        # *** Apply reponse function

        # Calculate fitted y-values:
        lstFit[idxIn][idxDpth, :] = funcCrf(vecX,
                                            vecModelPar[0],
                                            vecModelPar[1])

        # --------------------------------------------------------------------
        # *** Calculate response at half maximum contrast

        # The response at half maximum contrast (i.e. at a luminance contrast
        # of 50%):
        lstHlfMaxResp[idxIn][0, idxDpth] = funcCrf(0.5,
                                                   vecModelPar[0],
                                                   vecModelPar[1])

        # --------------------------------------------------------------------
        # *** Calculate contrast at half maximum response

        # The maximum response (defined as the response at 100% luminance
        # contrast):
        varResp50 = funcCrf(1.0,
                            vecModelPar[0],
                            vecModelPar[1])

        # Half maximum response:
        varResp50 = np.multiply(varResp50, 0.5)

        # Search for the luminance contrast level at half maximum response. A
        # while loop is more practical than an analytic solution - it is easy
        # to implement and reliable because of the contraint nature of the
        # problem. The problem is contraint because the luminance contrast has
        # to be between zero and one.

        # Initial value for the contrast level (will be incremented until the
        # half maximum response is reached).
        varHlfMaxCont = 0.0

        # Initial value for the resposne.
        varRespTmp = 0.0

        # Increment the contrast level until the half maximum response is
        # reached:
        while np.less(varRespTmp, varResp50):
            varHlfMaxCont += 0.000001
            varRespTmp = funcCrf(varHlfMaxCont,
                                 vecModelPar[0],
                                 vecModelPar[1])
        lstHlfMaxCont[idxIn][0, idxDpth] = varHlfMaxCont

        # --------------------------------------------------------------------
        # *** Calculate residual variance

        # In order to assess the fit of the model, we calculate the deviation
        # of the measured response from the fitted model (average across
        # conditions). First we have to calculate the deviation for each
        # condition.
        for idxCon in range(0, varNumCon):

            # Model prediction for current contrast level:
            varTmp = funcCrf(vecCon[idxCon], vecModelPar[0], vecModelPar[1])

            # Residual = absolute of difference between prediction and
            #            measurement
            lstRes[idxIn][idxCon, idxDpth] = \
                np.absolute(np.subtract(lstDpthMne[idxIn][idxCon, idxDpth],
                                        varTmp))

        # ------------------------------------------------------------------------
        # *** Plot contrast response functions

        # Create string for model parameters of exponential function:
        varParamA = np.around(vecModelPar[0], decimals=2)
        varParamB = np.around(vecModelPar[1], decimals=2)
        strMdlTmp = ('Model: R(C) = '
                     + str(varParamA)
                     + ' * C ^ '
                     + str(varParamB)
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
                       + '.svg')

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
strYlabel = 'fMRI signal change [arbitrary units]'

# Stack the vectors for the two ROIs (V1 & V2):
aryHlfMaxResp = np.vstack(lstHlfMaxResp[:])

funcPltAcrDpth(aryHlfMaxResp,      # aryData[Condition, Depth]
               np.zeros(np.shape(aryHlfMaxResp)),  # aryError[Con., Depth]
               varNumDpth,         # Number of depth levels (on the x-axis)
               varNumIn,           # Number of conditions (separate lines)
               varDpi,             # Resolution of the output figure
               0.0,                # Minimum of Y axis
               1.4,                # Maximum of Y axis
               False,              # Boolean: whether to convert y axis to %
               dicPthDpth.keys(),  # Labels for conditions (separate lines)
               strXlabel,          # Label on x axis
               strYlabel,          # Label on y axis
               'Response at half maximum contrast',  # Figure title
               True,               # Boolean: whether to plot a legend
               (strPthOt + '_half_max_response.png'))


# ----------------------------------------------------------------------------
# *** Plot contrast at half maximum response across depth

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
               'Contrast at half maximum response',  # Figure title
               True,               # Boolean: whether to plot a legend
               (strPthOt + '_half_max_contrast.png'),
               aryClr=aryClr)


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
               0.08,                # Maximum of Y axis
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

# Create plot:
fig01 = plt.figure()
axs01 = fig01.add_subplot(111, aspect='100.0')
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

# Save figure:
fig01.savefig((strPthOt + '_modelfit_bars.png'),
              dpi=(varDpi * 2.0),
              facecolor='w',
              edgecolor='w',
              transparent=False,
              frameon=None)


# ----------------------------------------------------------------------------
