"""
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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ds_pltAcrDpth import funcPltAcrDpth


# ----------------------------------------------------------------------------
# *** Define parameters

# Path of draining-corrected depth-profiles:
strPthPrfOt = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2.npy'

# Stimulus luminance contrast levels. NOTE: Should be between zero and one.
# When using precent (i.e. from zero to 100), the search for the luminance at
# half maximum response below would need to be adjusted.
vecCont = np.array([0.025, 0.061, 0.163, 0.72])

# Output path for plot:
strPthOt = '/home/john/PhD/Tex/contrast_response/v2/contrast_response'

# Limits of x-axis for contrast response plots
varXmin = 0.0
varXmax = 1.0

# Limits of y-axis for contrast response plots
varYmin = 0.0
varYmax = 2.0

# Axis labels
strLblX = 'Luminance contrast'
strLblY = 'fMRI response'

# Title for contrast response plots
strTtle = 'fMRI contrast response function'

# Figure scaling factor:
varDpi = 80.0

# Limits of y-axis for response-at-half-maximum plot
# varYmin = 0.0
# varYmax = 2.0

# ----------------------------------------------------------------------------
# *** Define contrast reponse function

# Contrast-fMRI-response function as defined in Boynton et al. (1999).
#   - varR is response
#   - varC is stimulus contrast
#   - varP - determines shape of contrast-response function, typical value: 0.3
#   - varQ - determines shape of contrast-response function, typical value: 2.0
#   - varS - ?
#   - varA - Scaling factor
def funcCrf(varC, varS, varA):
    varP = 0.25
    varQ = 2.0
    varR = varA * np.divide(
                            np.power(varC, (varP + varQ)),
                            (np.power(varC, varQ) + np.power(varS, varQ))
                            )
    return varR

varP = 0.25
varQ = 2.0

# ----------------------------------------------------------------------------
# *** Load depth profiles

# Load array with single-subject corrected depth profiles, of the form
# aryDpthSnSb[idxSub, idxCondition, idxDpth].
aryDpthSnSb = np.load(strPthPrfOt)


# ----------------------------------------------------------------------------
# *** Average over subjects

# Across-subjects mean:
aryDpthMne = np.mean(aryDpthSnSb, axis=0)

# Number of subjects:
varNumSubs = aryDpthSnSb.shape[0]

# Standard error of the mean (across subjects):
aryDpthSem = np.divide(np.std(aryDpthSnSb, axis=0),
                              np.sqrt(varNumSubs))

# Mean across depth-levels:
# aryMne = np.mean(aryDpth, axis=1)
# arySd = np.std(aryDpth, axis=1)


# ----------------------------------------------------------------------------
# *** Fit CRF separately for all depth levels

# Number of conditions:
varNumCon = aryDpthMne.shape[0]

# Number of depth levels:
varNumDpth = aryDpthMne.shape[1]

# Counter for plot file names:
varCntPlt = 0

# Vector for response at half maximum contrast at all depth levels:
vecHlfMaxResp = np.zeros((1, varNumDpth))

# Vector for contrast at half maximum response at all depth levels:
vecHlfMaxCont = np.zeros((1, varNumDpth))

# Loop through depth levels
for idxDpth in range(0, varNumDpth):
# for idxDpth in range(5, 6):

    # ------------------------------------------------------------------------
    # *** Fit contrast reponse function
    
    vecModelPar, vecModelCov = curve_fit(funcCrf,
                                         vecCont,
                                         aryDpthMne[:, idxDpth])
    
    
    # ------------------------------------------------------------------------
    # *** Apply reponse function
    
    # In order to plot the fitted function, we have to apply it to a range of
    # values:
    vecX = np.linspace(varXmin, varXmax, num=1000, endpoint=True)
    
    # Calculate fitted values:
    vecFit = funcCrf(vecX,
                     vecModelPar[0],
                     vecModelPar[1])

    # print(vecModelPar[0])

    # Create string for model parameters of exponential function:
    varParamA = np.around(vecModelPar[0], decimals=4)
    varParamS = np.around(vecModelPar[1], decimals=2)

    strModel = ('R(C) = '
                + str(varParamA)
                + ' * C^(' + str(varP) + '+' + str(varQ) + ') '
                + '/ '
                + '(C^' + str(varQ) + ' + ' + str(varParamS) + '^' + str(varQ)
                + ')'
                )


    # ------------------------------------------------------------------------
    # *** Calculate response at half maximum contrast

    # The response at half maximum contrast (i.e. luminance contrast of 50%)
    vecHlfMaxResp[0, idxDpth] = funcCrf(0.5,
                                        vecModelPar[0],
                                        vecModelPar[1])


    # ------------------------------------------------------------------------
    # *** Calculate contrast at half maximum response

    # The maximum response defined as the response at 100% luminance contrast
    varRes50 = funcCrf(1.0,
                       vecModelPar[0],
                       vecModelPar[1])

    # Half maximum response:
    varRes50 = np.multiply(varRes50, 0.5)

    # Search for the luminance contrast level at half maximum response. A
    # while loop is more practical than an analytic solution - it is easy to
    # implement and reliable because of the contraint nature of the problem.
    # The problem is contraint because the luminance contrast has to be
    # between zero and one.

    # Initial value for the contrast level (will be incremented until the
    # half maximum response is reached).
    varHlfMaxCont = 0.0

    # Initial value for the resposne.
    varResTmp = 0.0

    # Increment the contrast level until the half maximum response is reached:
    while np.less(varResTmp, varRes50):
        varHlfMaxCont += 0.001
        varResTmp = funcCrf(varHlfMaxCont, vecModelPar[0], vecModelPar[1])

    vecHlfMaxCont[0, idxDpth] = varResTmp


    # ------------------------------------------------------------------------
    # *** Create plots

    fig01 = plt.figure()

    axs01 = fig01.add_subplot(111)

    # Plot the average dependent data with error bars:
    plt01 = axs01.errorbar(vecCont,
                           aryDpthMne[:, idxDpth],
                           yerr=aryDpthSem[:, idxDpth],
                           color='blue',
                           label='Mean (SD)',
                           linewidth=0.9,
                           antialiased=True)

    # Plot model prediction:
    plt02 = axs01.plot(vecX,
                       vecFit,
                       color='red',
                       label=strModel,
                       linewidth=1.0,
                       antialiased=True)

    # Limits of the x-axis:
    # axs01.set_xlim([np.min(vecInd), np.max(vecInd)])
    axs01.set_xlim([varXmin, varXmax])

    # Limits of the y-axis:
    axs01.set_ylim([varYmin, varYmax])

    # Adjust labels for axis 1:
    axs01.tick_params(labelsize=10)
    axs01.set_xlabel(strLblX, fontsize=9)
    axs01.set_ylabel(strLblY, fontsize=9)

    # Title:
    strTtleTmp = (strTtle + ', depth level: ' + str(idxDpth))
    axs01.set_title(strTtleTmp, fontsize=9)

    # Add legend:
    axs01.legend(loc=0, prop={'size': 9})

    # Add vertical grid lines:
    axs01.xaxis.grid(which=u'major',
                     color=([0.5, 0.5, 0.5]),
                     linestyle='-',
                     linewidth=0.3)

    # Add horizontal grid lines:
    axs01.yaxis.grid(which=u'major',
                     color=([0.5, 0.5, 0.5]),
                     linestyle='-',
                     linewidth=0.3)

    # Save figure:
    fig01.savefig((strPthOt + '_plot_' + str(varCntPlt) + '.png'),
                  dpi=200,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  papertype='a6',
                  transparent=False,
                  frameon=None)

    # Increment file name counter:
    varCntPlt += 1


# ----------------------------------------------------------------------------
# *** Plot response at half maximum contrast

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [arbitrary units]'

funcPltAcrDpth(vecHlfMaxResp,   # Data to be plotted: aryData[Condition, Depth]
               np.zeros((1, varNumDpth)),  # aryError[Condition, Depth]
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
               (strPthOt + 'half_max_response.png'))


# ----------------------------------------------------------------------------
# *** Plot contrast at half maximum response

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Luminance contrast'

funcPltAcrDpth(vecHlfMaxCont,   # Data to be plotted: aryData[Condition, Depth]
               np.zeros((1, varNumDpth)),  # aryError[Condition, Depth]
               varNumDpth,  # Number of depth levels (on the x-axis)
               1,           # Number of conditions (separate lines)
               varDpi,      # Resolution of the output figure
               0.0,     # Minimum of Y axis
               1.0,     # Maximum of Y axis
               False,       # Boolean: whether to convert y axis to %
               [' '],       # Labels for conditions (separate lines)
               strXlabel,   # Label on x axis
               strYlabel,   # Label on y axis
               'Contrast at half maximum response',  # Figure title
               False,       # Boolean: whether to plot a legend
               (strPthOt + 'half_max_contrast.png'))
# ----------------------------------------------------------------------------

