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
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def plt_dpth_prfl(aryData, aryError, varNumDpth, varNumCon, varDpi, varYmin,
                  varYmax, lgcCnvPrct, lstConLbl, strXlabel, strYlabel,
                  strTitle, lgcLgnd, strPath, vecX=None, varXmin=None,
                  varXmax=None, varSizeX=1800.0, varSizeY=1600.0,
                  varNumLblY=5, varPadY=(0.0, 0.0), aryClr=None,
                  aryCnfLw=None, aryCnfUp=None, lstVrt=None, varRound=2):
    """
    Plot data across depth level for variable number of conditions.

    Parameters
    ----------
    aryData : np.array
        Data to be plotted. Numpy array of the form aryData[Condition, Depth].
    aryError : np.array or None
        Error in the data to be plotted (e.g. SEM). Numpy array of the form
        aryError[Condition, Depth]. The values will be added & subtracted from
        the values in aryData, and the region in between will be shaded.
        Alternatively, (possibly assymetric) confidence intervals can be
        provided (see below); in that case aryError is ignored.
    varNumDpth : int
        Number of depth levels (depth levels will be represented on the
        x-axis).
    varNumCon : int
        Number of conditions (conditions will be represented as separate lines
        in the plot).
    varDpi : float
        Resolution of the output figure.
    varYmin : float
        Minimum of Y axis.
    varYmax : float
        Maximum of Y axis.
    lgcCnvPrct : bool
        Whether to convert y axis to percent.
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
    strPath : str
        Output path for the figure
    vecX : np.array
        1D array with x-position of data points. If not provided, data points
        are equally spaced in the range ```range(0, varNumDpth)```.
    varXmin : float
        Minimum of X axis. If None (default), maximum is determined from vecX
        or varNumDpth.
    varXmax : float
        Maximum of X axis. If None (default), maximum is determined from vecX
        or varNumDpth.
    varSizeX : float
        Width of output figure.
    varSizeY : float
        Height of figure.
    varNumLblY : int
        Number of labels on y axis.
    varPadY : tuple
        Padding around labelled values on y.
    aryClr : np.array
        Line colours. Numpy array of form aryClr[idxCon, 3]; where the first
        dimension corresponds to the number of conditions, and the second
        dimension corresponds to the three RGB values for each condition.
    aryCnfLw : np.array
        Lower bound of confidence interval. Numpy array of form
        aryCnfLw[idxCondition, idxDepth]. If both aryCnfLw and aryCnfUp are
        provided, confidence intervals are plotted and aryError is ignored.
    aryCnfUp : np.array
        Upper bound of confidence interval. Numpy array of form
        aryCnfUp[idxCondition, idxDepth]. If both aryCnfLw and aryCnfUp are
        provided, confidence intervals are plotted and aryError is ignored.
    lstVrt : None or list
        If a list of values is provided, vertical lines are plotted at the
        respective relative positions along the cortical depth. For instance,
        if the value 0.3 is provided, a vertical line is plotted at 30% of the
        cortical depth. If lstVrt and aryClr are provided, they need to contain
        the same number of conditions (i.e. lstVrt needs to contain the same
        number of values as the size of the first dimension of aryClr).
    varRound : int
        Number of digits after decimal point for labels on y-axis (e.g. if
        `varRound=1`, labels may be '0.0, 1.0, ...').

    Returns
    -------
    None : None
        This function has no return value.


    The purpose of this function is to plot data & error intervals across
    cortical depth levels (represented on the x-axis), separately for
    conditions (separate lines).
    """
    # Create figure:
    fgr01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                                (varSizeY * 0.5) / varDpi),
                       dpi=varDpi)

    # Create axis:
    axs01 = fgr01.add_subplot(111)

    # Vector for x-data:
    if vecX is None:
        vecX = np.linspace(0.0, 1.0, num=varNumDpth, endpoint=True)

    if aryClr is None:
        # Prepare colour map:
        objClrNorm = colors.Normalize(vmin=0, vmax=(varNumCon - 1))
        objCmap = plt.cm.winter

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
                                        alpha=0.4,
                                        edgecolor=vecClrTmp,
                                        facecolor=vecClrTmp,
                                        linewidth=0,
                                        antialiased=True)
        else:
            # Plot error shading - confidence intervals:
            plot02 = axs01.fill_between(vecX,  #noqa
                                        aryCnfLw[idxCon, :],
                                        aryCnfUp[idxCon, :],
                                        alpha=0.4,
                                        edgecolor=vecClrTmp,
                                        facecolor=vecClrTmp,
                                        linewidth=0,
                                        antialiased=True)

    # Determine minimum and maximum of x-axis:
    if varXmin is None:
        varXmin = np.min(vecX)
    if varXmax is None:
        varXmax = np.max(vecX)

    # Plot vertical lines (e.g. representing peak position):
    if lstVrt is not None:

        # Use same colours as for lines:
        if aryClr is None:
            # Prepare colour map:
            objClrNorm = colors.Normalize(vmin=0, vmax=(varNumCon - 1))
            objCmap = plt.cm.winter

        # Loop through list with line positions:
        varNumVrt = len(lstVrt)
        for idxVrt in range(0, varNumVrt):

            # Adjust colour of current vertical line:
            if aryClr is None:
                # Adjust the colour of current line:
                vecClrTmp = objCmap(objClrNorm(varNumVrt - 1 - idxVrt))
            else:
                vecClrTmp = aryClr[idxVrt, :]

            # Apsolute Position of vertical line (input values refer to
            # relative position):
            varVrtTmp = lstVrt[idxVrt] / (float(varXmax)) + float(varXmin)

            # Plot vertical line:
            axs01.axvline(varVrtTmp,
                          color=vecClrTmp,
                          linewidth=8.0,
                          linestyle='--',
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
    axs01.set_xticks([(varXmin - 0.04),
                      (varXmax + 0.04)])

    # Set tick labels for x ticks:
    axs01.set_xticklabels(['WM', 'CSF'])

    # Which y values to label with ticks:
    vecYlbl = np.linspace(varYmin, varYmax, num=varNumLblY, endpoint=True)
    # Round:
    # vecYlbl = np.around(vecYlbl, decimals=2)

    # Set ticks:
    axs01.set_yticks(vecYlbl)

    # Convert labels to percent?
    if lgcCnvPrct:
        # Multiply by 100 to convert to percent:
        vecYlbl = np.multiply(vecYlbl, 100.0)
        # Convert labels from float to a list of strings, with well-defined
        # number of decimals (including trailing zeros):
        lstYlbl = [None] * vecYlbl.shape[0]
        for idxLbl in range(vecYlbl.shape[0]):
            lstYlbl[idxLbl] = ('{:0.'
                               + str(varRound)
                               + 'f}').format(vecYlbl[idxLbl])
    else:
        # Convert labels from float to a list of strings, with well-defined
        # number of decimals (including trailing zeros):
        lstYlbl = [None] * vecYlbl.shape[0]
        for idxLbl in range(vecYlbl.shape[0]):
            lstYlbl[idxLbl] = ('{:0.'
                               + str(varRound)
                               + 'f}').format(vecYlbl[idxLbl])

    # Set tick labels for y ticks:
    axs01.set_yticklabels(lstYlbl)

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

    # Make plot & axis labels fit into figure (this may not always work):
    # try:
    plt.tight_layout(pad=0.5)
    # except ...:
    #     pass

    # Save figure:
    fgr01.savefig(strPath,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)

    # Close figure:
    plt.close(fgr01)
