# -*- coding: utf-8 -*-
"""
Supplement to the depth sampling pipeline.

The purpose of this script is to load results from the main depth sampling
pipeline and plot the across-subjects depth dependent signal change, normalsied
by division to a reference condition.

For example, the signal change for all conditions (stimulus level) can be
divided by the signal at the lowest stimulus level (separately for all depth
levels).

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
import matplotlib.colors as colors


# *****************************************************************************
# *** Settings

# Path of .npy files with depth sampling results from the main analysis
# pipeline:
strDpthMeans = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Higher_Level_Analysis/depthsampling_mean_pe/v1.npy'  #noqa

# Reference condition for normalisation (all other conditions are divided by
# this condition, separately for each depth level):
varNormIdx = 0

# Number of subjects:
varNumSubs = 5

# Number of conditions:
varNumCon = 4

# Number of cortical depths:
varNumDpth = 11

# Range of y-axis for regression plots:
varYmin = 0.0
varYmax = 1.2

# Linear regression plot - label for axes:
strXlabel = 'Stimulus luminance contrast'
strYlabel = 'fMRI signal change [arbitrary units]'

# X axis tick labels:
lstConLbl = ['2.5%', '6.1%', '16.3%', '72.0%']

# Output path for plot:
strPltOt = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Higher_Level_Analysis/depthsampling_mean_pe/norm_subtraction_v1.png'  #noqa

# Figure scaling factor:
varDpi = 96.0
# *****************************************************************************


# *****************************************************************************
# *** Load data

# Load mean depth-dependent parameter estimates for both ROIs:
arySubDpthMns = np.load(strDpthMeans)
# *****************************************************************************


# *****************************************************************************
# *** Plot results

# Across-subjects mean:
aryAcrSubDpthMean = np.mean(arySubDpthMns, axis=0)

# Calculate 95% confidence interval for the mean, obtained by multiplying
# the standard error of the mean (SEM) by 1.96. We obtain  the SEM by
# dividing the standard deviation by the squareroot of the sample size n.
# aryArcSubDpthConf = np.multiply(np.divide(np.std(arySubDpthMns, axis=0),
#                                           np.sqrt(varNumSubs)),
#                                 1.96)

# Calculate standard error of the mean.
aryArcSubDpthConf = np.divide(np.std(arySubDpthMns, axis=0),
                              np.sqrt(varNumSubs))

# Vector for normalisation:
vecNorm = np.array(aryAcrSubDpthMean[varNormIdx, :], ndmin=2)
# Divide all rows by reference row:
aryAcrSubDpthMean = np.divide(aryAcrSubDpthMean, vecNorm)


# Create figure:
fgr01 = plt.figure(figsize=(800.0/varDpi, 500.0/varDpi),
                   dpi=varDpi)
# Create axis:
axs01 = fgr01.add_subplot(111)

# Vector for x-data:
vecX = range(0, varNumDpth)

# Prepare colour map:
objClrNorm = colors.Normalize(vmin=0, vmax=(varNumCon - 1))
objCmap = plt.cm.winter

# Loop through input files:
for idxIn in range(0, varNumCon):

    # Adjust the colour of current line:
    vecClrTmp = objCmap(objClrNorm(varNumCon - 1 - idxIn))

    # Plot depth profile for current input file:
    plt01 = axs01.errorbar(vecX,  #noqa
                           aryAcrSubDpthMean[idxIn, :],
                           yerr=aryArcSubDpthConf[idxIn, :],
                           elinewidth=2.5,
                           color=vecClrTmp,
                           alpha=0.8,
                           label=('Luminance contrast '
                                  + lstConLbl[idxIn]),
                           # label=('Linear contrast [-3 -1 +1 +3]'),
                           linewidth=5.0,
                           antialiased=True)

# Set x-axis range:
axs01.set_xlim([-1, varNumDpth])
# Set y-axis range:
axs01.set_ylim([varYmin, varYmax])

# Which x values to label with ticks (WM & CSF boundary):
axs01.set_xticks([-0.5, (varNumDpth - 0.5)])
# Labels for x ticks:
axs01.set_xticklabels(['WM', 'CSF'])

# Set x & y tick font size:
axs01.tick_params(labelsize=13)

# Adjust labels:
axs01.set_xlabel(strXlabel,
                 fontsize=13)
axs01.set_ylabel(strYlabel,
                 fontsize=13)

# Adjust title:
axs01.set_title(('V1 normalised to stimulus level 1, n=' + str(varNumSubs)),
                fontsize=10)

# Legend for axis 1:
# axs01.legend(loc=0,
#             prop={'size': 13})

# # Add vertical grid lines:
#    axs01.xaxis.grid(which=u'major',
#                     color=([0.5,0.5,0.5]),
#                     linestyle='-',
#                     linewidth=0.2)

# Save figure:
fgr01.savefig(strPltOt,
              facecolor='w',
              edgecolor='w',
              orientation='landscape',
              transparent=False,
              frameon=None)

# Close figure:
plt.close(fgr01)
