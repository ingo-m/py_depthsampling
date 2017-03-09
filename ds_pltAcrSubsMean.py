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
import matplotlib.colors as colors


def funcPltAcrSubsMean(arySubDpthMns,
                       varNumSubs,
                       varNumDpth,
                       varNumCon,
                       varDpi,
                       varAcrSubsYmin,
                       varAcrSubsYmax,
                       lstConLbl,
                       strXlabel,
                       strYlabel,
                       strTitle,
                       strPltOtPre,
                       strPltOtSuf):
    """Calculate & plot across-subjects mean."""
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

    # Create figure:
    fgr01 = plt.figure(figsize=(1200.0/varDpi, 800.0/varDpi),
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
        plt01 = axs01.plot(vecX,  #noqa
                           aryAcrSubDpthMean[idxIn, :],
                           color=vecClrTmp,
                           alpha=0.9,
                           label=('Luminance contrast '
                                  + lstConLbl[idxIn]),
                           linewidth=8.0,
                           antialiased=True)

        # Plot error shading:
        plot02 = axs01.fill_between(vecX,  #noqa
                                    np.subtract(aryAcrSubDpthMean[idxIn, :],
                                                aryArcSubDpthConf[idxIn, :]),
                                    np.add(aryAcrSubDpthMean[idxIn, :],
                                           aryArcSubDpthConf[idxIn, :]),
                                    alpha=0.4,
                                    edgecolor=vecClrTmp,
                                    facecolor=vecClrTmp,
                                    linewidth=0,
                                    # linestyle='dashdot',
                                    antialiased=True)

    # Reduce framing box:
    axs01.spines['top'].set_visible(False)
    axs01.spines['right'].set_visible(False)
    axs01.spines['bottom'].set_visible(True)
    axs01.spines['left'].set_visible(True)

    # Set x-axis range:
    axs01.set_xlim([-1, varNumDpth])
    # Set y-axis range:
    axs01.set_ylim([varAcrSubsYmin, varAcrSubsYmax])

    # Which x values to label with ticks (WM & CSF boundary):
    axs01.set_xticks([-0.5, (varNumDpth - 0.5)])
    # Labels for x ticks:
    axs01.set_xticklabels(['WM', 'CSF'])

    # Which y values to label with ticks:
    # vecYlbl = np.linspace(varAcrSubsYmin,
    #                       varAcrSubsYmax,
    #                       num=5,
    #                       endpoint=True)
    vecYlbl = np.linspace(0, varAcrSubsYmax, num=5, endpoint=True)
    # Round:
    # vecYlbl = np.around(vecYlbl, decimals=2)
    # Set ticks:
    axs01.set_yticks(vecYlbl)

    # Set x & y tick font size:
    axs01.tick_params(labelsize=36,
                      top='off',
                      right='off')

    # Adjust labels:
    axs01.set_xlabel(strXlabel,
                     fontsize=36)
    axs01.set_ylabel(strYlabel,
                     fontsize=36)

    # Adjust title:
    axs01.set_title(strTitle, fontsize=36, fontweight="bold")

    # Legend for axis 1:
    #axs01.legend(loc=0,
    #             frameon=False,
    #             prop={'size': 22})

    # # Add vertical grid lines:
    #    axs01.xaxis.grid(which=u'major',
    #                     color=([0.5,0.5,0.5]),
    #                     linestyle='-',
    #                     linewidth=0.2)

    # File name for figure:
    strPltOt = strPltOtPre + 'acrsSubsMeanShading' + strPltOtSuf

    # Save figure:
    fgr01.savefig(strPltOt,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)

    # Close figure:
    plt.close(fgr01)
