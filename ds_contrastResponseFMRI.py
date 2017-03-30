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
varXmin = 0.001
varXmax = 1.0

# Limits of y-axis for contrast response plots
varYmin = 0.0
varYmax = 2.0

# Axis labels
strLblX = 'Luminance contrast'
strLblY = 'fMRI signal change [arbitrary units]'

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
    # varR = varS * np.log(varC) + varA
    varP = 0.3
    varQ = 2.0
    varR = varA * np.divide(
                            np.power(varC, (varP + varQ)),
                            (np.power(varC, varQ) + np.power(varS, varQ))
                            )
    return varR

#varP = 0.25
#varQ = 2.0


# ----------------------------------------------------------------------------
# *** Load depth profiles

# Load array with single-subject corrected depth profiles, of the form
# aryDpth[idxSub, idxCondition, idxDpth].
aryDpth = np.load(strPthPrfOt)


# ----------------------------------------------------------------------------
# *** Fit CRF

# We fit the contrast response function separately for all subjects & depth
# levels.

# Number of subjects:
varNumSubs = aryDpth.shape[0]

# Number of conditions:
varNumCon = aryDpth.shape[1]

# Number of depth levels:
varNumDpth = aryDpth.shape[2]

# Number of x-values for which to solve the function:
varNumX = 1000

# Vector for which the function will be fitted:
vecX = np.linspace(varXmin, varXmax, num=varNumX, endpoint=True)

# Vector for y-values of fitted function (for each subject & depth level):
aryFit = np.zeros((varNumSubs, varNumDpth, varNumX))

# Vector for response at half maximum contrast:
aryHlfMaxResp = np.zeros((varNumSubs, varNumDpth))

# Vector for contrast at half maximum response:
aryHlfMaxCont = np.zeros((varNumSubs, varNumDpth))

# Array for residual variance:
aryRes = np.zeros((varNumSubs, varNumDpth))

# Loop through subjects:
for idxSub in range(0, varNumSubs):

    # Loop through depth levels:
    for idxDpth in range(0, varNumDpth):

        # --------------------------------------------------------------------
        # *** Fit contrast reponse function
        
        vecModelPar, vecModelCov = curve_fit(funcCrf,
                                             vecCont,
                                             aryDpth[idxSub, :, idxDpth],
                                             maxfev=100000)
        
        
        # --------------------------------------------------------------------
        # *** Apply reponse function

        # Calculate fitted y-values:
        aryFit[idxSub, idxDpth, :] = funcCrf(vecX,
                                             vecModelPar[0],
                                             vecModelPar[1])


        # --------------------------------------------------------------------
        # *** Calculate response at half maximum contrast
    
        # The response at half maximum contrast (i.e. at a luminance contrast
        # of 50%):
        aryHlfMaxResp[idxSub, idxDpth] = funcCrf(0.5,
                                                vecModelPar[0],
                                                vecModelPar[1])
    
    
        # --------------------------------------------------------------------
        # *** Calculate contrast at half maximum response
    
        # The maximum response (defined as the response at 100% luminance
        # contrast):
        varRes50 = funcCrf(1.0,
                           vecModelPar[0],
                           vecModelPar[1])
    
        # Half maximum response:
        varRes50 = np.multiply(varRes50, 0.5)
    
        # Search for the luminance contrast level at half maximum response. A
        # while loop is more practical than an analytic solution - it is easy
        # to implement and reliable because of the contraint nature of the
        # problem. The problem is contraint because the luminance contrast has
        # to be between zero and one.
    
        # Initial value for the contrast level (will be incremented until the
        # half maximum response is reached).
        varHlfMaxCont = 0.0
    
        # Initial value for the resposne.
        varResTmp = 0.0
    
        # Increment the contrast level until the half maximum response is
        # reached:
        while np.less(varResTmp, varRes50):
            varHlfMaxCont += 0.01
            varResTmp = funcCrf(varHlfMaxCont, vecModelPar[0], vecModelPar[1])
        aryHlfMaxCont[idxSub, idxDpth] = varResTmp


        # --------------------------------------------------------------------
        # *** Calculate residual variance

        # In order to assess the fit of the model, we calculate the deviation
        # of the measured response from the fitted model (average across
        # conditions). First we have to calculate the deviation for each
        # condition.
        vecRes = np.zeros(varNumCon)
        for idxCon in range(0, varNumCon):

            # Model prediction for current contrast level:
            varTmp = funcCrf(vecCont[idxCon], vecModelPar[0], vecModelPar[1])

            # Residual = absolute of difference between prediction and
            # measurement
            vecRes[idxCon] = np.absolute(np.subtract(aryDpth[idxSub,
                                                             idxCon,
                                                             idxDpth],
                                                     varTmp))

        # Mean residual across conditions:
        aryRes[idxSub, idxDpth] = np.mean(vecRes)


# ----------------------------------------------------------------------------
# *** Average across subjects

# Across-subjects mean for measured response:
aryDpthMne = np.mean(aryDpth, axis=0)
# Standard error of the mean (across subjects):
aryDpthSem = np.divide(np.std(aryDpth, axis=0),
                       np.sqrt(varNumSubs))

# Across-subjects mean for fitted CRF:
aryFitMne = np.mean(aryFit, axis=0)
# Standard error of the mean (across subjects):
aryFitSme = np.divide(np.std(aryFit, axis=0),
                      np.sqrt(varNumSubs))

# Average response at half maximum contrast:
aryHlfMaxRespMne = np.mean(aryHlfMaxResp, axis=0, keepdims=True)
# Standard error of the mean (across subjects):
aryHlfMaxRespSme = np.divide(np.std(aryHlfMaxResp, axis=0),
                             np.sqrt(varNumSubs))
# Restore original number of dimensions:
aryHlfMaxRespSme = np.array(aryHlfMaxRespSme, ndmin=2)

# Average response at half maximum contrast:
aryHlfMaxContMne = np.mean(aryHlfMaxCont, axis=0, keepdims=True)
# Standard error of the mean (across subjects):
aryHlfMaxContSme = np.divide(np.std(aryHlfMaxCont, axis=0),
                             np.sqrt(varNumSubs))
# Restore original number of dimensions:
aryHlfMaxContSme = np.array(aryHlfMaxContSme, ndmin=2)

# ----------------------------------------------------------------------------
# *** Plot CRF

# Counter for plot file names:
varCntPlt = 0

# We plot the average CRF (averaged across subjects) for all depth levels.
for idxDpth in range(0, varNumDpth):


    # ------------------------------------------------------------------------
    # *** Create plots

    fig01 = plt.figure()

    axs01 = fig01.add_subplot(111)

    # Plot  model prediction
    plt01 = axs01.plot(vecX,  #noqa
                       aryFitMne[idxDpth, :],
                       color='red',
                       alpha=0.9,
                       label='Modelled mean (SEM)',
                       linewidth=2.0,
                       antialiased=True,
                       zorder=2)

    # Plot error shading
    plot02 = axs01.fill_between(vecX,  #noqa
                                np.subtract(aryFitMne[idxDpth, :],
                                            aryFitSme[idxDpth, :]),
                                np.add(aryFitMne[idxDpth, :],
                                       aryFitSme[idxDpth, :]),
                                alpha=0.4,
                                edgecolor='red',
                                facecolor='red',
                                linewidth=0.0,
                                # linestyle='dashdot',
                                antialiased=True,
                                zorder=1)

    # Plot the average dependent data with error bars:
    plt03 = axs01.errorbar(vecCont,
                           aryDpthMne[:, idxDpth],
                           yerr=aryDpthSem[:, idxDpth],
                           color='blue',
                           label='Empirical mean (SEM)',
                           linewidth=2.0,
                           antialiased=True,
                           zorder=3)

    # Limits of the x-axis:
    # axs01.set_xlim([np.min(vecInd), np.max(vecInd)])
    axs01.set_xlim([varXmin, varXmax])

    # Limits of the y-axis:
    axs01.set_ylim([varYmin, varYmax])

    # Which y values to label with ticks:
    vecYlbl = np.linspace(varYmin, varYmax, num=5, endpoint=True)
    # Round:
    # vecYlbl = np.around(vecYlbl, decimals=2)
    # Set ticks:
    axs01.set_yticks(vecYlbl)

    # Adjust labels for axis 1:
    axs01.tick_params(labelsize=14)
    axs01.set_xlabel(strLblX, fontsize=16)
    axs01.set_ylabel(strLblY, fontsize=16)

    # Title:
    strTtleTmp = (strTtle + ', depth level: ' + str(idxDpth))
    axs01.set_title(strTtleTmp, fontsize=14)

    # Add legend:
    axs01.legend(loc=0, prop={'size': 10})

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
                  dpi=(varDpi * 2.0),
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)

    # Increment file name counter:
    varCntPlt += 1


## Create string for model parameters of exponential function:
#varParamA = np.around(vecModelPar[0], decimals=4)
#varParamS = np.around(vecModelPar[1], decimals=2)
#
#strModel = ('R(C) = '
#            + str(varParamA)
#            + ' * C^(' + str(varP) + '+' + str(varQ) + ') '
#            + '/ '
#            + '(C^' + str(varQ) + ' + ' + str(varParamS) + '^' + str(varQ)
#            + ')'
#            )


# ----------------------------------------------------------------------------
# *** Plot response at half maximum contrast

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [arbitrary units]'

funcPltAcrDpth(aryHlfMaxRespMne,  # aryData[Condition, Depth]
               aryHlfMaxRespSme,  # aryError[Condition, Depth]
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
# *** Plot contrast at half maximum response

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Luminance contrast'

funcPltAcrDpth(aryHlfMaxContMne,  # aryData[Condition, Depth]
               aryHlfMaxContSme,  # aryError[Condition, Depth]
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
               (strPthOt + '_half_max_contrast.png'))


# ----------------------------------------------------------------------------
# *** Plot residual variance

# Mean residual variance across subjects:
vecResMne = np.mean(aryRes, axis=0, keepdims=True)

# Standard error of the mean:
vecResSem = np.divide(np.std(aryRes, axis=0),
                      np.sqrt(varNumSubs))
vecResSem = np.array(vecResSem, ndmin=2)

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'Residual variance'

funcPltAcrDpth(vecResMne,   # aryData[Condition, Depth]
               vecResSem,   # aryError[Condition, Depth]
               varNumDpth,  # Number of depth levels (on the x-axis)
               1,           # Number of conditions (separate lines)
               varDpi,      # Resolution of the output figure
               0.0,     # Minimum of Y axis
               0.15,     # Maximum of Y axis
               False,       # Boolean: whether to convert y axis to %
               [' '],       # Labels for conditions (separate lines)
               strXlabel,   # Label on x axis
               strYlabel,   # Label on y axis
               'Model fit across cortical depth',  # Figure title
               False,       # Boolean: whether to plot a legend
               (strPthOt + '_modelfit.png'))

