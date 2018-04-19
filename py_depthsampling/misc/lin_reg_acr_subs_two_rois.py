# -*- coding: utf-8 -*-
"""
Supplement to the depth sampling pipeline.

The purpose of this (makeshift) script is to plot simple linear regression
results for two ROIs (V1 and V2). The same regression model that is used in the
main analysis pipeline is employed, with the only difference that the results
for two ROIs are combined into one plot.

@author: Ingo Marquardt, 07.11.2016
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
from scipy import stats
import matplotlib.colors as colors

# *****************************************************************************
# *** Settings

# Path of .npy files with depth sampling results from the main analysis
# pipeline, separately for the two ROIs:
strDpthMeansV1 = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Higher_Level_Analysis/depthsampling_mean_pe/v1.npy'  #noqa
strDpthMeansV2 = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Higher_Level_Analysis/depthsampling_mean_pe/v2.npy'  #noqa

# Number of subjects:
varNumSubs = 5

# Number of cortical depths:
varNumDpth = 11

# Constrast vector for simple linear regression model (array with one value per
# condition, e.g. per stimulus contrast level):
vecLinRegMdl = np.array([-3.0, -1.0, 1.0, 3.0])

# Range of y-axis for regression plots:
varLinRegYmin = 0.025  # 0.085  # 0.025  # 0.09
varLinRegYmax = 0.175  # 0.165  # 0.15  # 0.17

# Linear regression plot - label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strLinRegYlabel = 'Regression coefficient'

# Output path for plot:
strPltOt = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Higher_Level_Analysis/depthsampling_mean_pe/linregplot.png'  #noqa

# Figure scaling factor:
varDpi = 96.0
# *****************************************************************************


# *****************************************************************************
# *** Load data

# Load mean depth-dependent parameter estimates for both ROIs:
arySubDpthMnsV1 = np.load(strDpthMeansV1)
arySubDpthMnsV2 = np.load(strDpthMeansV2)
# *****************************************************************************


# *****************************************************************************
# *** Linear regression V1

# Repeat linear regression model for each subject:
vecLinRegX = np.tile(vecLinRegMdl, varNumSubs)

# Vectors for regression results (one regression per depth level):

# Slope of the regression line:
vecV1Slpe = np.zeros(varNumDpth)
# Intercept of the regression line:
vecV1Itrcpt = np.zeros(varNumDpth)
# Correlation coefficient:
vecV1Cor = np.zeros(varNumDpth)
# p-value:
vecV1P = np.zeros(varNumDpth)
# Standard error of the estimate:
vecV1StdErr = np.zeros(varNumDpth)

# Loop through depth levels to calculate linear regression independently
# at each depth level:
for idxDpth in range(0, varNumDpth):

    # Get a flat array with the parameter estiamtes for each condition for
    # each subject:
    vecV1LinRegY = arySubDpthMnsV1[:, :, idxDpth].flatten()

    # Fit the regression model:
    vecV1Slpe[idxDpth], vecV1Itrcpt[idxDpth], vecV1Cor[idxDpth], \
        vecV1P[idxDpth], vecV1StdErr[idxDpth] = stats.linregress(vecLinRegX,
                                                                 vecV1LinRegY)
# *****************************************************************************


# *****************************************************************************
# *** Linear regression V2

# Repeat linear regression model for each subject:
vecLinRegX = np.tile(vecLinRegMdl, varNumSubs)

# Vectors for regression results (one regression per depth level):

# Slope of the regression line:
vecV2Slpe = np.zeros(varNumDpth)
# Intercept of the regression line:
vecV2Itrcpt = np.zeros(varNumDpth)
# Correlation coefficient:
vecV2Cor = np.zeros(varNumDpth)
# p-value:
vecV2P = np.zeros(varNumDpth)
# Standard error of the estimate:
vecV2StdErr = np.zeros(varNumDpth)

# Loop through depth levels to calculate linear regression independently
# at each depth level:
for idxDpth in range(0, varNumDpth):

    # Get a flat array with the parameter estiamtes for each condition for
    # each subject:
    vecV2LinRegY = arySubDpthMnsV2[:, :, idxDpth].flatten()

    # Fit the regression model:
    vecV2Slpe[idxDpth], vecV2Itrcpt[idxDpth], vecV2Cor[idxDpth], \
        vecV2P[idxDpth], vecV2StdErr[idxDpth] = stats.linregress(vecLinRegX,
                                                                 vecV2LinRegY)
# *****************************************************************************


# *****************************************************************************
# *** Plot results

# Create figure:
fgr01 = plt.figure(figsize=(800.0/varDpi, 500.0/varDpi),
                   dpi=varDpi)
# Create axis:
axs01 = fgr01.add_subplot(111)

# Vector for x-data:
vecX = range(0, varNumDpth)

# Prepare colour map:
objClrNorm = colors.Normalize(vmin=0, vmax=1)
objCmap = plt.cm.winter
vecClr01 = objCmap(objClrNorm(0.1))
vecClr02 = objCmap(objClrNorm(0.9))

# Plot depth profile for V1:
plt01 = axs01.errorbar(vecX,  #noqa
                       vecV1Slpe,
                       yerr=vecV1StdErr,
                       elinewidth=2.5,
                       color=vecClr01,
                       alpha=0.8,
                       label=('V1'),
                       linewidth=5.0,
                       antialiased=True)

# Plot depth profile for V2:
plt02 = axs01.errorbar(vecX,  #noqa
                       vecV2Slpe,
                       yerr=vecV2StdErr,
                       elinewidth=2.5,
                       color=vecClr02,
                       alpha=0.8,
                       label=('V2'),
                       linewidth=5.0,
                       antialiased=True)

# Set x-axis range:
axs01.set_xlim([-1, varNumDpth])
# Set y-axis range:
axs01.set_ylim([varLinRegYmin, varLinRegYmax])

# Which x values to label with ticks (WM & CSF boundary):
axs01.set_xticks([-0.5, (varNumDpth - 0.5)])
# Labels for x ticks:
axs01.set_xticklabels(['WM', 'CSF'])

# Set x & y tick font size:
axs01.tick_params(labelsize=13)

# Adjust labels:
axs01.set_xlabel(strXlabel,
                 fontsize=13)
axs01.set_ylabel(strLinRegYlabel,
                 fontsize=13)

# Adjust title:
# axs01.set_title(('Linear regression, n=' + str(varNumSubs)),
#                 fontsize=13)

# Legend for axis 1:
axs01.legend(loc=0,
             prop={'size': 13})

# Save figure:
fgr01.savefig(strPltOt,
              facecolor='w',
              edgecolor='w',
              orientation='landscape',
              transparent=False,
              frameon=None)

# Close figure:
plt.close(fgr01)
