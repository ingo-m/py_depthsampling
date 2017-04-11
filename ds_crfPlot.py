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


def funcCrfPlt(vecX,         # x-values for fitted data
               vecFit,       # Fitted y-values as a function of contrast
               vecCon,       # Empirical stimulus luminance contrast levels
               vecDpthMne,   # Empirical y-data; vecDpthMne[idxCon]
               vecDpthSem,   # Error of emp. y; vecDpthSem[idxCon]
               strPthOt,     # Output path
               varXmin=0.0,  # Lower limit of x axis
               varXmax=1.0,  # Upper limit of x axis
               varYmin=0.0,  # Lower limit of y axis
               varYmax=2.0,  # Upper limit of y axis
               strLblX='',   # Label for x axis
               strLblY='',   # Label for y axis
               strTtle='',   # Plot title
               varDpi=80.0,  # Figure scaling factor
               strMdl='Modelled mean (SEM)',  # Figure legend
               idxDpth=0
               ):
    """
    Plot contrast response function.

    Function of the depth sampling pipeline.
    """
    fig01 = plt.figure()

    axs01 = fig01.add_subplot(111)

    # Plot  model prediction
    plt01 = axs01.plot(vecX,  #noqa
                       vecFit,
                       color='red',
                       alpha=0.9,
                       label=strMdl,
                       linewidth=2.0,
                       antialiased=True,
                       zorder=2)

    # Plot the average dependent data with error bars:
    plt03 = axs01.errorbar(vecCon,  #noqa
                           vecDpthMne,
                           yerr=vecDpthSem,
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
    vecYlbl = np.linspace(varYmin, varYmax, num=4, endpoint=True)
    # Round:
    # vecYlbl = np.around(vecYlbl, decimals=2)
    # Set ticks:
    axs01.set_yticks(vecYlbl)

    # Adjust labels for axis 1:
    axs01.tick_params(labelsize=14)
    axs01.set_xlabel(strLblX, fontsize=16)
    axs01.set_ylabel(strLblY, fontsize=16)

    # Title:
    axs01.set_title(strTtle, fontsize=14)

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
    fig01.savefig(strPthOt,
                  dpi=(varDpi * 2.0),
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)
