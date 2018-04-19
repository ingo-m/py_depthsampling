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
                       strPltOtSuf,
                       varSizeX=1800.0,
                       varSizeY=1600.0,
                       strErr='conf95',
                       vecX=None):
    """
    Calculate & plot across-subjects mean.

    Parameters
    ----------
    arySubDpthMns : np.array
        Array with depth data to be plotted, of the form:
        aryDpthMeans[Subject, Condition, Depth]
    varNumSubs : int
        Number of subect
    varNumDpth : int
        Number of cortical depth levels
    varNumCon : int
        Number of conditions
    varDpi : int
        DPI value for the plot
    varAcrSubsYmin : int
        Minimum of the y-axis
    varAcrSubsYmax : int
        Maximum of the y-axis
    lstConLbl : list
        List of labels for the conditions
    strXlabel : str
        Label for the x-axis
    strYlabel : str
        Label for the y-axis
    strTitle : str
        Title for the plot
    strPltOtPre : str
        Path for saving plots (prefix)
    strPltOtSuf : str
        File type for saing plots (suffix)
    varSizeX : float
        Width of output figure.
    varSizeY : float
        Height of figure.
    strErr : str
        Which parameter to use for the error bar. Can be one of the following:
        'conf95': Plot 95% confidence interval for the mean across subjects,
                  obtained by multiplying the standard error of the mean (SEM)
                  by 1.96 (default).
        'sd':     Plot standard deviation across subjects, or, depending on
                  the input, across iterations.
        'sem':    Plot standard error of the mean.
        'prct95': Plot limits of 2.5th and 97.5th percentile across subjects,
                  or, depending on the input, across iterations. Because the
                  percentile does not depend on the sample size n, this option
                  is useful if the input to this function are not individual
                  subject's depth profiles, but, for instance, depth profiles
                  created by iteratively changing model assumtions to test
                  the robustness of the results ('model 4' in ds_drainModel).
    vecX : np.array
        1D array with x-position of data points. If not provided, data points
        are equally spaced in the range ```range(0, varNumDpth)```.
    """
    # Across-subjects mean:
    aryAcrSubDpthMean = np.mean(arySubDpthMns, axis=0)

    if strErr == 'conf95':
        # Calculate 95% confidence interval for the mean, obtained by
        # multiplying the standard error of the mean (SEM) by 1.96. We obtain
        # the SEM by dividing the standard deviation by the squareroot of the
        # sample size n.
        aryArcSubDpthConf = np.multiply(np.divide(np.std(arySubDpthMns,
                                                         axis=0),
                                                  np.sqrt(varNumSubs)),
                                        1.96)
        # Lower bound (as deviation from the mean):
        aryArcSubDpthConfLw = np.subtract(aryAcrSubDpthMean, aryArcSubDpthConf)
        # Upper bound (as deviation from the mean):
        aryArcSubDpthConfUp = np.add(aryAcrSubDpthMean, aryArcSubDpthConf)

    elif strErr == 'sd':
        # Calculate standard deviation across subjects:
        aryArcSubDpthConf = np.std(arySubDpthMns, axis=0)
        # Lower bound (as deviation from the mean):
        aryArcSubDpthConfLw = np.subtract(aryAcrSubDpthMean, aryArcSubDpthConf)
        # Upper bound (as deviation from the mean):
        aryArcSubDpthConfUp = np.add(aryAcrSubDpthMean, aryArcSubDpthConf)

    elif strErr == 'sem':
        # Calculate standard error of the mean.
        aryArcSubDpthConf = np.divide(np.std(arySubDpthMns, axis=0),
                                      np.sqrt(varNumSubs))
        # Lower bound (as deviation from the mean):
        aryArcSubDpthConfLw = np.subtract(aryAcrSubDpthMean, aryArcSubDpthConf)
        # Upper bound (as deviation from the mean):
        aryArcSubDpthConfUp = np.add(aryAcrSubDpthMean, aryArcSubDpthConf)

    elif strErr == 'prct95':
        # Calculate 2.5th percentile:
        aryArcSubDpthConfLw = np.percentile(arySubDpthMns, 2.5, axis=0)
        # Calculate 97.5th percentile:
        aryArcSubDpthConfUp = np.percentile(arySubDpthMns, 97.5, axis=0)

    # Figure dimensions:
    varSizeX = 1800.0
    varSizeY = 1600.0

    fgr01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                                (varSizeY * 0.5) / varDpi),
                       dpi=varDpi)

    # Create axis:
    axs01 = fgr01.add_subplot(111)

    # Vector for x-data:
    if vecX is None:
        vecX = np.linspace(0.0, 1.0, num=varNumDpth, endpoint=True)

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
                           label=(lstConLbl[idxIn]),
                           linewidth=8.0,
                           antialiased=True)

        # Plot error shading.
        plot02 = axs01.fill_between(vecX,  #noqa
                                    aryArcSubDpthConfLw[idxIn, :],
                                    aryArcSubDpthConfUp[idxIn, :],
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
    axs01.set_xlim([(np.min(vecX) - 0.07),
                    (np.max(vecX) + 0.07)])
    # axs01.set_xlim([-0.07, 1.07])

    # Set y-axis range:
    axs01.set_ylim([varAcrSubsYmin, varAcrSubsYmax])

    # Which x values to label with ticks (WM & CSF boundary):
    axs01.set_xticks([(np.min(vecX) - 0.04),
                      (np.max(vecX) + 0.04)])
    # axs01.set_xticks([-0.04, 1.04])

    # Labels for x ticks:
    axs01.set_xticklabels(['WM', 'CSF'])

    # Which y values to label with ticks:
    vecYlbl = np.linspace(np.ceil(varAcrSubsYmin),
                          np.floor(varAcrSubsYmax),
                          num=5,
                          endpoint=True)

    # Round:
    # vecYlbl = np.around(vecYlbl, decimals=2)

    # Set ticks:
    axs01.set_yticks(vecYlbl)

    # Set x & y tick font size:
    axs01.tick_params(labelsize=36,
                      top=False,
                      right=False)

    # Adjust labels:
    axs01.set_xlabel(strXlabel,
                     fontsize=36)
    axs01.set_ylabel(strYlabel,
                     fontsize=36)

    # Adjust title:
    axs01.set_title(strTitle, fontsize=36, fontweight="bold")

    # Legend for axis 1:
    axs01.legend(loc=0,
                 frameon=False,
                 prop={'size': 22})

    # # Add vertical grid lines:
    #    axs01.xaxis.grid(which=u'major',
    #                     color=([0.5,0.5,0.5]),
    #                     linestyle='-',
    #                     linewidth=0.2)

    # File name for figure:
    strPltOt = strPltOtPre + 'acrsSubsMean' + strPltOtSuf

    # Make plot & axis labels fit into figure:
    plt.tight_layout(pad=0.5)

    # Save figure:
    fgr01.savefig(strPltOt,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)

    # Close figure:
    plt.close(fgr01)
