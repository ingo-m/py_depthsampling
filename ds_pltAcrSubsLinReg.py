# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

@author: Ingo Marquardt, 06.11.2016
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

import numpy as np  # noqa
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.colors as colors


def funcPltAcrSubsLinReg(arySubDpthMns,
                         vecLinRegMdl,
                         varNumSubs,
                         varNumDpth,
                         strTitle,
                         strXlabel,
                         strLinRegYlabel,
                         varLinRegYmin,
                         varLinRegYmax,
                         varLinRegP,
                         varDpi,
                         strPltOtPre,
                         strPltOtSuf):
    """
    Calculate & plot across-subjects simple linear regression.

    A simple linear regression is model is calculated to test the dependence of
    the parameter estimates (i.e. signal change) to stimulus condition (i.e.
    luminance contrast), separately for each depth level.
    """
    # Repeat linear regression model for each subject:
    vecLinRegX = np.tile(vecLinRegMdl, varNumSubs)

    # Vectors for regression results (one regression per depth level):

    # Slope of the regression line:
    vecSlpe = np.zeros(varNumDpth)
    # Intercept of the regression line:
    vecItrcpt = np.zeros(varNumDpth)
    # Correlation coefficient:
    vecCor = np.zeros(varNumDpth)
    # p-value:
    vecP = np.zeros(varNumDpth)
    # Standard error of the estimate:
    vecStdErr = np.zeros(varNumDpth)

    # Loop through depth levels to calculate linear regression independently
    # at each depth level:
    for idxDpth in range(0, varNumDpth):

        # Get a flat array with the parameter estiamtes for each condition for
        # each subject:
        vecLinRegY = arySubDpthMns[:, :, idxDpth].flatten()

        # Fit the regression model:
        vecSlpe[idxDpth], vecItrcpt[idxDpth], vecCor[idxDpth], vecP[idxDpth], \
            vecStdErr[idxDpth] = stats.linregress(vecLinRegX, vecLinRegY)

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
    vecClr = objCmap(objClrNorm(0))

    # Plot depth profile for current input file:
    plt01 = axs01.errorbar(vecX,  #noqa
                           vecSlpe,
                           yerr=vecStdErr,
                           elinewidth=2.5,
                           color=vecClr,
                           alpha=0.8,
                           # label=(('Stimulus level ' + str(idxIn + 1))),
                           # label=('Linear contrast [-3 -1 +1 +3]'),
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
    axs01.set_title((strTitle + ' linear regression, n=' + str(varNumSubs)),
                    fontsize=13)

    # # Add vertical grid lines:
    #    axs01.xaxis.grid(which=u'major',
    #                     color=([0.5,0.5,0.5]),
    #                     linestyle='-',
    #                     linewidth=0.2)

    # File name for figure:
    strPltOt = strPltOtPre + 'acrsSubsLinReg' + strPltOtSuf

    # Save figure:
    fgr01.savefig(strPltOt,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)

    # Close figure:
    plt.close(fgr01)
