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
from ds_crfPlot import funcCrfPlt


# ----------------------------------------------------------------------------
# *** Define parameters

# Path of draining-corrected depth-profiles:
# strPthPrfOt = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2_not_norm_corrected.npy'  #noqa
strPthPrfOt = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2_corrected.npy'  #noqa

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using precent (i.e. from zero to 100), the search for the luminance at
# half maximum response below would need to be adjusted.
vecCon = np.array([0.025, 0.061, 0.163, 0.72])

# Output path for plot:
# strPthOt = '/home/john/PhD/Tex/contrast_response_not_norm/v2_corrected/crf'
strPthOt = '/home/john/PhD/Tex/contrast_response/v2_corrected/crf'

# Limits of x-axis for contrast response plots
varXmin = 0.0
varXmax = 1.0

# Limits of y-axis for contrast response plots
varYmin = 0.0
# varYmax = 5.0
varYmax = 1.5

# Axis labels
strLblX = 'Luminance contrast'
strLblY = 'fMRI signal change [arbitrary units]'

# Title for contrast response plots
strTtle = 'fMRI contrast response function'

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

# Load array with single-subject corrected depth profiles, of the form
# aryDpth[idxSub, idxCondition, idxDpth].
aryDpth = np.load(strPthPrfOt)


# ----------------------------------------------------------------------------
# *** Average across subjects

# Number of subjects:
varNumSubs = aryDpth.shape[0]

# Number of conditions:
varNumCon = aryDpth.shape[1]

# Number of depth levels:
varNumDpth = aryDpth.shape[2]

# Scaling:
# aryDpth = np.multiply(aryDpth, 0.01)

# Across-subjects mean for measured response:
aryDpthMne = np.mean(aryDpth, axis=0)

# Standard error of the mean (across subjects):
aryDpthSem = np.divide(np.std(aryDpth, axis=0),
                       np.sqrt(varNumSubs))


# ----------------------------------------------------------------------------
# *** Fit CRF across depth levels

# We fit the contrast response function separately for all depth levels.

# Number of x-values for which to solve the function:
varNumX = 1001

# Vector for which the function will be fitted:
vecX = np.linspace(varXmin, varXmax, num=varNumX, endpoint=True)

# Vector for y-values of fitted function (for each depth level):
aryFit = np.zeros((varNumDpth, varNumX))

# Vector for response at half maximum contrast:
vecHlfMaxResp = np.zeros((1, varNumDpth))

# Vector for contrast at half maximum response:
vecHlfMaxCont = np.zeros((1, varNumDpth))

# Array for residual variance:
aryRes = np.zeros((varNumCon, varNumDpth))

# Loop through depth levels:
for idxDpth in range(0, varNumDpth):

    # ------------------------------------------------------------------------
    # *** Fit contrast reponse function
    vecModelPar, vecModelCov = curve_fit(funcCrf,
                                         vecCon,
                                         aryDpthMne[:, idxDpth],
                                         maxfev=100000,
                                         bounds=(vecLimA, vecLimB),
                                         p0=(0.5, 0.5))

    # ------------------------------------------------------------------------
    # *** Apply reponse function

    # Calculate fitted y-values:
    aryFit[idxDpth, :] = funcCrf(vecX,
                                 vecModelPar[0],
                                 vecModelPar[1])

    # ------------------------------------------------------------------------
    # *** Calculate response at half maximum contrast

    # The response at half maximum contrast (i.e. at a luminance contrast of
    # of 50%):
    vecHlfMaxResp[0, idxDpth] = funcCrf(0.5,
                                        vecModelPar[0],
                                        vecModelPar[1])

    # ------------------------------------------------------------------------
    # *** Calculate contrast at half maximum response

    # The maximum response (defined as the response at 100% luminance
    # contrast):
    varResp50 = funcCrf(1.0,
                        vecModelPar[0],
                        vecModelPar[1])

    # Half maximum response:
    varResp50 = np.multiply(varResp50, 0.5)

    # Search for the luminance contrast level at half maximum response. A
    # while loop is more practical than an analytic solution - it is easy to
    # implement and reliable because of the contraint nature of the problem.
    # The problem is contraint because the luminance contrast has to be
    # between zero and one.

    # Initial value for the contrast level (will be incremented until the half
    # maximum response is reached).
    varHlfMaxCont = 0.0

    # Initial value for the resposne.
    varRespTmp = 0.0

    # Increment the contrast level until the half maximum response is reached:
    while np.less(varRespTmp, varResp50):
        varHlfMaxCont += 0.000001
        varRespTmp = funcCrf(varHlfMaxCont, vecModelPar[0], vecModelPar[1])
    vecHlfMaxCont[0, idxDpth] = varHlfMaxCont

    # ------------------------------------------------------------------------
    # *** Calculate residual variance

    # In order to assess the fit of the model, we calculate the deviation of
    # the measured response from the fitted model (average across conditions).
    # First we have to calculate the deviation for each condition.
    for idxCon in range(0, varNumCon):

        # Model prediction for current contrast level:
        varTmp = funcCrf(vecCon[idxCon], vecModelPar[0], vecModelPar[1])

        # Residual = absolute of difference between prediction and measurement
        aryRes[idxCon, idxDpth] = np.absolute(np.subtract(aryDpthMne[idxCon,
                                                                     idxDpth],
                                                          varTmp))

    # ----------------------------------------------------------------------------
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
    #            + str(varParamA)
    #            + ' * C^(' + str(varP) + '+' + str(varQ) + ') '
    #            + '/ '
    #            + '(C^' + str(varQ) + ' + ' + str(varParamB) + '^' + str(varQ)
    #            + ')'
    #            )

    # Title for current CRF plot:
    strTtleTmp = (strTtle + ', depth level: ' + str(idxDpth))

    # Plot CRF for current depth level:
    funcCrfPlt(vecX,
               aryFit[idxDpth, :],
               vecCon,
               aryDpthMne[:, idxDpth],
               aryDpthSem[:, idxDpth],
               strPthOt,
               varXmin=varXmin,
               varXmax=varXmax,
               varYmin=varYmin,
               varYmax=varYmax,
               strLblX=strLblX,
               strLblY=strLblY,
               strTtle=strTtleTmp,
               varDpi=80.0,
               strMdl=strMdlTmp,
               idxDpth=idxDpth
               )


# ----------------------------------------------------------------------------
# *** Plot response at half maximum contrast across depth

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [arbitrary units]'

funcPltAcrDpth(vecHlfMaxResp,     # aryData[Condition, Depth]
               np.zeros(np.shape(vecHlfMaxResp)),  # aryError[Con., Depth]
               varNumDpth,  # Number of depth levels (on the x-axis)
               1,           # Number of conditions (separate lines)
               varDpi,      # Resolution of the output figure
               varYmin,     # Minimum of Y axis
               varYmax,     # Maximum of Y axis
               False,       # Boolean: whether to convert y axis to %
               [' '],       # Labels for conditions (separate lines)
               strXlabel,   # Label on x axis
               strYlabel,   # Label on y axis
               'Response at half maximum contrast',  # Figure title
               False,       # Boolean: whether to plot a legend
               (strPthOt + '_half_max_response.png'))


# ----------------------------------------------------------------------------
# *** Plot contrast at half maximum response across depth

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Percent luminance contrast'

# Convert contrast values to percent (otherwise rounding will be a problem for
# y-axis values):
vecHlfMaxCont = np.multiply(vecHlfMaxCont, 100.0)

funcPltAcrDpth(vecHlfMaxCont,  # aryData[Condition, Depth]
               np.zeros(np.shape(vecHlfMaxCont)),  # aryError[Con., Depth]
               varNumDpth,  # Number of depth levels (on the x-axis)
               1,           # Number of conditions (separate lines)
               varDpi,      # Resolution of the output figure
               0.0,     # Minimum of Y axis
               9.0,     # Maximum of Y axis
               False,       # Boolean: whether to convert y axis to %
               [' '],       # Labels for conditions (separate lines)
               strXlabel,   # Label on x axis
               strYlabel,   # Label on y axis
               'Contrast at half maximum response',  # Figure title
               False,       # Boolean: whether to plot a legend
               (strPthOt + '_half_max_contrast.png'))


# ----------------------------------------------------------------------------
# *** Plot residual variance across depth

# Mean residual variance across subjects:
aryResMne = np.mean(aryRes, axis=0, keepdims=True)

# Standard error of the mean:
aryResSem = np.divide(np.std(aryRes, axis=0),
                      np.sqrt(varNumSubs))
aryResSem = np.array(aryResSem, ndmin=2)

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Residual variance'

funcPltAcrDpth(aryResMne,   # aryData[Condition, Depth]
               aryResSem,   # aryError[Condition, Depth]
               varNumDpth,  # Number of depth levels (on the x-axis)
               1,           # Number of conditions (separate lines)
               varDpi,      # Resolution of the output figure
               0.0,         # Minimum of Y axis
               0.2,        # Maximum of Y axis
               False,       # Boolean: whether to convert y axis to %
               [' '],       # Labels for conditions (separate lines)
               strXlabel,   # Label on x axis
               strYlabel,   # Label on y axis
               'Model fit across cortical depth',  # Figure title
               False,       # Boolean: whether to plot a legend
               (strPthOt + '_modelfit.png'))
