# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

@author: Ingo Marquardt, 07.04.2017
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


def funcCrfPlt(vecX,        # X-values for fitted data
               aryFitMne,   # Fitted y-values aryFitMne[idxDpth, idxContr]
               aryFitSme,   # Error of fitted y; aryFitSme[idxDpth, idxContr]
               vecCont,     # Empirical stimulus luminance contrast levels
               aryDpthMne,  # Empirical y-data; aryDpthMne[idxCon, idxDpth]
               aryDpthSem,  # Error of emp. y; aryDpthSem[idxCon, idxDpth]
               strPthOt,    # Output path
               varXmin=0.0,  # Lower limit of x axis
               varXmax=1.0,  # Upper limit of x axis
               varYmin=0.0,  # Lower limit of y axis
               varYmax=2.0,  # Upper limit of y axis
               strLblX='',   # Label for x axis
               strLblY='',   # Label for y axis
               strTtle='',   # Plot title
               varDpi=80.0   # Figure scaling factor
               ):
    """
    Plot contrast response function.
    
    Function of the depth sampling pipeline.
    """
    # Number of depth levels
    varNumDpth = aryFitMne.shape[0]

    # Counter for plot file names:
    varCntPlt = 0

    # We plot the average CRF (averaged across subjects) for all depth levels.
    for idxDpth in range(0, varNumDpth):

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
        plt03 = axs01.errorbar(vecCont,  #noqa
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
