# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  #noqa
import matplotlib.colors as colors  #noqa


def plt_psf(aryData,  #noqa
            strPath,
            vecX=None,
            aryError=None,
            aryCnfLw=None,
            aryCnfUp=None,
            varDpi=80.0,
            varYmin=None,
            varYmax=None,
            lstConLbl=None,
            strXlabel='',
            strYlabel='',
            strTitle='',
            lgcLgnd=True,
            varXmin=None,
            varXmax=None,
            varSizeX=1800.0,
            varSizeY=1600.0,
            varNumLblY=5,
            varPadY=(0.0, 0.0),
            lstVrt=None,
            aryClr=None):
    """
    Plot data across depth level for variable number of conditions.

    Parameters
    ----------
    aryData : np.array
        Data to be plotted. Numpy array of the form aryData[Condition, x-axis].
    strPath : str
        Output path for the figure
    vecX : np.array
        1D array with x-position of data points. If not provided, data points
        are equally spaced in the range ```range(0, aryData.shape[1])```.
    aryError : np.array or None
        Error in the data to be plotted (e.g. SEM). Numpy array of the form
        aryError[Condition, x-axis]. The values will be added & subtracted from
        the values in aryData, and the region in between will be shaded.
        Alternatively, (possibly assymetric) confidence intervals can be
        provided (see below); in that case aryError is ignored.
    aryCnfLw : np.array
        Lower bound of confidence interval. Numpy array of form
        aryCnfLw[idxCondition, x-axis]. If both aryCnfLw and aryCnfUp are
        provided, confidence intervals are plotted and aryError is ignored.
    aryCnfUp : np.array
        Upper bound of confidence interval. Numpy array of form
        aryCnfUp[idxCondition, x-axis]. If both aryCnfLw and aryCnfUp are
        provided, confidence intervals are plotted and aryError is ignored.
    varDpi : float
        Resolution of the output figure.
    varYmin : float
        Minimum of Y axis.
    varYmax : float
        Maximum of Y axis.
    lstConLbl : list
        List of strings with labels for conditions.
    strXlabel : str
        Label on x axis.
    strYlabel : str
        Label on y axis
    strTitle : str
        Figure title
    lgcLgnd : bool
        Whether to plot a legend.
    varXmin : float
        Minimum of X axis. If None (default), maximum is determined from data.
    varXmax : float
        Maximum of X axis. If None (default), maximum is determined from data.
    varSizeX : float
        Width of output figure.
    varSizeY : float
        Height of figure.
    varNumLblY : int
        Number of labels on y axis.
    varPadY : tuple
        Padding around labelled values on y.
    lstVrt : list
        If a list of values is provided, vertical lines are plotted at the
        respective position on the x-axis. If lstVrt and aryClr are provided,
        they need to contain the same number of conditions (i.e. lstVrt needs
        to contain the same number of values as the size of the first dimension
        of aryClr).
    aryClr : np.array
        Line colours. Numpy array of form aryClr[idxCon, 3]; where the first
        dimension corresponds to the number of conditions, and the second
        dimension corresponds to the three RGB values for each condition.

    Returns
    -------
    None : None
        This function has no return value.

    The purpose of this function is to plot data & error intervals across
    cortical depth levels (represented on the x-axis), separately for
    conditions (separate lines).
    """
    # Number of conditions in input data:
    varNumCon = aryData.shape[0]

    # Number of x-values:
    varNumX = aryData.shape[1]

    # Create figure:
    fgr01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                                (varSizeY * 0.5) / varDpi),
                       dpi=varDpi)

    # Create axis:
    axs01 = fgr01.add_subplot(111)

    # Vector for x-data:
    if vecX is None:
        vecX = np.linspace(0.0, 1.0, num=varNumX, endpoint=True)

    if aryClr is None:
        # Prepare colour map:
        # objClrNorm = colors.Normalize(vmin=0, vmax=(varNumCon - 1))
        # objCmap = plt.cm.winter
        objClrNorm = colors.Normalize(vmin=0, vmax=9)
        objCmap = plt.cm.tab10

    # Create empty labels if none are provided:
    if lstConLbl is None:
        lstConLbl = [' '] * varNumCon

    # Loop through conditions:
    for idxCon in range(0, varNumCon):

        if aryClr is None:
            # Adjust the colour of current line:
            vecClrTmp = objCmap(objClrNorm(varNumCon - 1 - idxCon))
        else:
            vecClrTmp = aryClr[idxCon, :]

        # Plot depth profile for current input file:
        plt01 = axs01.plot(vecX,  #noqa
                           aryData[idxCon, :],
                           color=vecClrTmp,
                           alpha=0.9,
                           label=(lstConLbl[idxCon]),
                           linewidth=9.0,
                           antialiased=True)

        # If no confidence intervals have been supplied, plot SEM:
        if (aryCnfLw is None) or (aryCnfUp is None):
            # Plot error shading:
            plot02 = axs01.fill_between(vecX,  #noqa
                                        np.subtract(aryData[idxCon, :],
                                                    aryError[idxCon, :]),
                                        np.add(aryData[idxCon, :],
                                               aryError[idxCon, :]),
                                        alpha=0.1,
                                        edgecolor=vecClrTmp,
                                        facecolor=vecClrTmp,
                                        linewidth=0,
                                        antialiased=True)
        else:
            # Plot error shading - confidence intervals:
            plot02 = axs01.fill_between(vecX,  #noqa
                                        aryCnfLw[idxCon, :],
                                        aryCnfUp[idxCon, :],
                                        alpha=0.1,
                                        edgecolor=vecClrTmp,
                                        facecolor=vecClrTmp,
                                        linewidth=0,
                                        antialiased=True)

    # Determine minima and maxima of axes:
    if varXmin is None:
        varXmin = np.min(vecX)
    if varXmax is None:
        varXmax = np.max(vecX)
    if varYmin is None:
        # varYmin = np.min(aryData)
        # varYmin = (np.floor(varYmin * 10.0) / 10.0)
        varYmin = np.floor(np.min(aryData))
    if varYmax is None:
        # varYmax = np.max(aryData)
        # varYmax = (np.ceil(varYmax * 10.0) / 10.0)
        varYmax = np.ceil(np.max(aryData))

    # Plot vertical lines (e.g. representing peak position):
    if lstVrt is not None:

        # Loop through list with line positions:
        varNumVrt = len(lstVrt)
        for idxVrt in range(0, varNumVrt):

            # Apsolute Position of vertical line (input values refer to
            # relative position):
            # varVrtTmp = lstVrt[idxVrt] / (float(varXmax)) + float(varXmin)
            varVrtTmp = lstVrt[idxVrt]

            # Plot vertical line:
            axs01.axvline(varVrtTmp,
                          color=[0.3, 0.3, 0.3],
                          linewidth=3.0,
                          alpha=0.5,
                          linestyle=':',
                          antialiased=True)

    # Set x-axis range:
    # axs01.set_xlim([-0.2, (varNumDpth - 0.8)])
    axs01.set_xlim([(varXmin - 0.07),
                    (varXmax + 0.07)])

    # Set y-axis range:
    axs01.set_ylim([(varYmin - varPadY[0]),
                    (varYmax + varPadY[1])])

    # Which x values to label with ticks (WM & CSF boundary):
    # axs01.set_xticks([-0.1, (varNumDpth - 0.9)])
    # axs01.set_xticks([(varXmin - 0.04),
    #                   (varXmax + 0.04)])

    # Set tick labels for x ticks:
    # axs01.set_xticklabels(['WM', 'CSF'])

    # Which y values to label with ticks:
    vecYlbl = np.linspace(varYmin, varYmax, num=varNumLblY, endpoint=True)
    # Round:
    # vecYlbl = np.around(vecYlbl, decimals=2)

    # Set ticks:
    axs01.set_yticks(vecYlbl)

    # Set tick labels for y ticks:
    # axs01.set_yticklabels(lstYlbl)

    # Set x & y tick font size:
    axs01.tick_params(labelsize=36,
                      top=False,
                      right=False)

    # Adjust labels:
    axs01.set_xlabel(strXlabel,
                     fontsize=36)
    axs01.set_ylabel(strYlabel,
                     fontsize=36)

    # Reduce framing box:
    axs01.spines['top'].set_visible(False)
    axs01.spines['right'].set_visible(False)
    axs01.spines['bottom'].set_visible(True)
    axs01.spines['left'].set_visible(True)

    # Adjust title:
    axs01.set_title(strTitle, fontsize=36, fontweight="bold")

    # Legend for axis 1:
    if lgcLgnd:
        axs01.legend(loc=0,
                     frameon=False,
                     prop={'size': 26})

    # Make plot & axis labels fit into figure (this may not always work,
    # depending on the layout of the plot, matplotlib sometimes throws a
    # ValueError ("left cannot be >= right").
    try:
        plt.tight_layout(pad=0.5)
    except ValueError:
        pass

    # Save figure:
    fgr01.savefig(strPath,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)

    # Close figure:
    plt.close(fgr01)
