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


# ----------------------------------------------------------------------------
# *** Define parameters

# Path of draining-corrected depth-profiles:
strPthPrfOt = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy'

# Stimulus luminance contrast levels:
vecCont = np.array([0.025, 0.061, 0.163, 0.72])

# Output path for plot:
strPthOt = '/home/john/Desktop/tmp/contrast_response'

# Limits of x-axis:
varXmin = 0.0
varXmax = 1.0

# Limits of x-axis:
varYmin = 0.0
varYmax = 2.0

# Axis labels
strLblX = 'Luminance contrast'
strLblY = 'fMRI response'

# Title for plots
lstTtle = ['fMRI contrast response function']

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
# *** Define contrast reponse function

# Contrast-fMRI-response function as defined in Boynton et al. (1999).
#   - varR is response
#   - varC is stimulus contrast
#   - varP - determines shape of contrast-response function, typical value: 0.3
#   - varQ - determines shape of contrast-response function, typical value: 2.0
#   - varS - ?
#   - varA - Scaling factor
def funcCrf(varC, varS, varA):
    varP = 0.3
    varQ = 2.0
    varR = varA * np.divide(
                            np.power(varC, (varP + varQ)),
                            (np.power(varC, varQ) + np.power(varS, varQ))
                            )
    return varR


# ----------------------------------------------------------------------------
# *** Fit CRF separately for all depth levels

# Number of conditions:
varNumCon = aryDpthMne.shape[0]

# Number of depth levels:
varNumDpth = aryDpthMne.shape[1]

# Counter for plot file names:
varCntPlt = 0

# Loop through depth levels
for idxDpth in range(0, varNumDpth):


    # ----------------------------------------------------------------------------
    # *** Fit contrast reponse function
    
    vecModelPar, vecModelCov = curve_fit(funcCrf,
                                         vecCont,
                                         aryDpthMne[:, idxDpth])
    
    
    # ----------------------------------------------------------------------------
    # *** Apply reponse function
    
    # In order to plot the fitted function, we have to apply it to a range of
    # values:
    vecX = np.linspace(varXmin, varXmax, num=1000, endpoint=True)
    
    # Calculate fitted values:
    vecFit = funcCrf(vecX,
                     vecModelPar[0],
                     vecModelPar[1])
    
    varP = 0.3
    varQ = 2.0
    
    # Create string for model parameters of exponential function:
    varParamA = np.around(vecModelPar[0], 2)
    varParamS = np.around(vecModelPar[1], 2)
    strModel = ('R(C) = '
                + str(varParamA)
                + ' * C^(' + str(varP) + '+' + str(varQ) + ') '
                + '/ '
                + '(C^' + str(varQ) + ' + ' + str(varParamA) + '^' + str(varQ)
                + ')'
                )
    
    
    # ------------------------------------------------------------------------
    # *** Create plots
    
    # List with model predictions:
    lstModPre = [vecFit]
    
    # List with model parameters:
    lstModPar = [strModel]
    
    # We create one plot per function:
    for idxPlt in range(0, len(lstModPre)):
    
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
                           lstModPre[idxPlt],
                           color='red',
                           label=lstModPar[idxPlt],
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
        strTtleTmp = (lstTtle[idxPlt] + ', depth level: ' + str(idxDpth))
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
    # ------------------------------------------------------------------------
