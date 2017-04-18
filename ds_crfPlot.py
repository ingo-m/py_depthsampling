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


def plt_crf(vecMdlX,
            vecMdlY,
            strPthOt,
            vecMdlYCnfLw=None,
            vecMdlYCnfUp=None,
            vecEmpX=None,
            vecEmpYMne=None,
            vecEmpYSem=None,
            varXmin=0.0,
            varXmax=1.0,
            varYmin=0.0,
            varYmax=2.0,
            strLblX='',
            strLblY='',
            strTtle='',
            varDpi=80.0,
            lgcLgnd=True,
            strMdl='Modelled mean (SEM)',
            lgcGrid=False):
    """
    Plot contrast response function.

    Parameters
    ----------
    vecMdlX : np.array
        x-values at which model has been fitted (e.g. contrast against which
        to plot modelled resposne).
    vecMdlY : np.array
        Modelled response (y-values as a function of contrast).
    strPthOt : str
        Output path for plot.
    vecMdlYCnfLw : None or np.array
        Lower bound of the confidence interval of the model fit. Optional.
    vecMdlYCnfUp : None or np.array
        Upper bound of the confidence interval of the model fit. Optional.
    vecEmpX : np.array
        Empirical stimulus luminance contrast levels (i.e. x-values at which
        a response has been measured).
    vecEmpYMne : np.array
        Mean empirical (i.e. measured) contrast response (of the form
        vecEmpYMne[idxCon]).
    vecEmpYSem : np.array
        Standard error of the mean empirical (i.e. measured) contrast response
        (same shape as vecEmpYMne, i.e. vecEmpYSem[idxCon]).
    varXmin : float
        Lower limit of x axis.
    varXmax : float
        Upper limit of x axis.
    varYmin : float
        Lower limit of y axis.
    varYmax : float
        Upper limit of y axis.
    strLblX : str
        Label for x axis.
    strLblY : str
        Label for y axis.
    strTtle : str
        Plot title.
    varDpi : float
        Figure scaling factor.
    lgcLgnd : bool
        Whether to plot a legend.
    strMdl : str
        Legend text.
    lgcGrid : bool
        Whether to plot gird lines.

    Returns
    -------
    None : None
        This function has no return value.

    Notes
    -----
    Plot fitted contrast response function, and - optionally - measured
    contrast responses.

    Function of the depth sampling pipeline.
    """
    # Figure dimensions:
    varSizeX = 800.0
    varSizeY = 600.0

    fig01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                                (varSizeY * 0.5) / varDpi),
                       dpi=varDpi)

    axs01 = fig01.add_subplot(111)

    # Plot model prediction
    plt01 = axs01.plot(vecMdlX,  #noqa
                       vecMdlY,
                       color=[0.9, 0.1, 0.1],
                       alpha=0.8,
                       label=strMdl,
                       linewidth=4.0,
                       antialiased=True,
                       zorder=2)

    if (vecMdlYCnfLw is not None) and (vecMdlYCnfUp is not None):
        plot02 = axs01.fill_between(vecMdlX,  #noqa
                                    vecMdlYCnfLw,
                                    vecMdlYCnfUp,
                                    alpha=0.4,
                                    edgecolor=[0.9, 0.1, 0.1],
                                    facecolor=[0.9, 0.1, 0.1],
                                    linewidth=0,
                                    antialiased=True)

    if vecEmpX is not None:
        # Plot the average dependent data with error bars:
        plt03 = axs01.errorbar(vecEmpX,  #noqa
                               vecEmpYMne,
                               yerr=vecEmpYSem,
                               color=[0.1, 0.1, 0.9],
                               label='Empirical mean (SEM)',
                               linewidth=4.0,
                               antialiased=True,
                               zorder=3)

    # Set axis limits:
    axs01.set_xlim([varXmin, varXmax])
    axs01.set_ylim([varYmin, varYmax])

    # Which y values to label with ticks:
    vecYlbl = np.linspace(varYmin, varYmax, num=4, endpoint=True)
    # Round:
    # vecYlbl = np.around(vecYlbl, decimals=2)
    # Set y-axis ticks:
    axs01.set_yticks(vecYlbl)

    # Adjust axis label size:
    axs01.tick_params(labelsize=16,
                      top='off',
                      right='off')
    axs01.set_xlabel(strLblX, fontsize=16)
    axs01.set_ylabel(strLblY, fontsize=16)

    # Reduce framing box:
    axs01.spines['top'].set_visible(False)
    axs01.spines['right'].set_visible(False)
    axs01.spines['bottom'].set_visible(True)
    axs01.spines['left'].set_visible(True)

    # Title:
    axs01.set_title(strTtle, fontsize=16)

    # Add legend:
    axs01.legend(loc=0,
                 frameon=False,
                 prop={'size': 8})  # 14})

    if lgcGrid:
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

    # Make plot & axis labels fit into figure:
    plt.tight_layout(pad=0.5)

    # Save figure:
    fig01.savefig(strPthOt,
                  dpi=(varDpi * 2.0),
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)

    # Close figure:
    plt.close(fig01)
