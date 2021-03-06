# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

# Part of py_depthsampling library
# Copyright (C) 2018  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
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
# import matplotlib.colors as colors
# from matplotlib.colors import BoundaryNorm


def ecc_plot_simple(aryMean, vecEccBin, strPathOut):
    """
    Plot results for eccentricity & cortical depth analysis.

    This version plots the values using only a single linear colourmap.

    Plots statistical parameters (e.g. parameter estimates) by cortical depth
    (x-axis) and pRF eccentricity (y-axis). This function is part of a tool for
    analysis of cortical-depth-dependent fMRI responses at different
    retinotopic eccentricities.
    """
    # Number of eccentricity bins:
    varEccNum = vecEccBin.shape[0]

    # Font type:
    strFont = 'Liberation Sans'

    # Font colour:
    vecFontClr = np.array([17.0/255.0, 85.0/255.0, 124.0/255.0])

    # Find minimum and maximum correlation values:
    # varMin = np.percentile(aryMean, 0.0)
    # varMax = np.percentile(aryMean, 100.0)
    varMin = np.min(aryMean)
    varMax = np.max(aryMean)

    # Round:
    # varMin = (np.floor(varMin * 10.0) / 10.0)
    # varMax = (np.ceil(varMax * 10.0) / 10.0)
    # varMin = (np.floor(varMin * 0.1) / 0.1)
    # varMax = (np.ceil(varMax * 0.1) / 0.1)
    varMin = np.floor(varMin)
    varMax = np.ceil(varMax)

    # Create main figure:
    fig01 = plt.figure(figsize=(4.0, 3.0),
                       dpi=200.0,
                       facecolor=([1.0, 1.0, 1.0]),
                       edgecolor=([1.0, 1.0, 1.0]))

    # Big subplot in the background for common axes labels:
    axsCmn = fig01.add_subplot(111)

    # Turn off axis lines and ticks of the big subplot:
    axsCmn.spines['top'].set_color('none')
    axsCmn.spines['bottom'].set_color('none')
    axsCmn.spines['left'].set_color('none')
    axsCmn.spines['right'].set_color('none')
    axsCmn.tick_params(labelcolor='w',
                       top=False,
                       bottom=False,
                       left=False,
                       right=False)

    # Set and adjust common axes labels:
    axsCmn.set_xlabel('V1 relative cortical depth',
                      alpha=1.0,
                      fontname=strFont,
                      fontweight='normal',
                      fontsize=7.0,
                      color=vecFontClr,
                      position=(0.5, 0.0))
    axsCmn.set_ylabel('pRF eccentricity',
                      alpha=1.0,
                      fontname=strFont,
                      fontweight='normal',
                      fontsize=7.0,
                      color=vecFontClr,
                      position=(0.0, 0.5))
    axsCmn.set_title('fMRI signal change',
                     alpha=1.0,
                     fontname=strFont,
                     fontweight='bold',
                     fontsize=10.0,
                     color=vecFontClr,
                     position=(0.5, 1.1))

    # Create colour-bar axis:
    axsTmp = fig01.add_subplot(111)

    # Plot correlation coefficients of current depth level:
    pltTmpCorr = plt.imshow(aryMean,
                            interpolation='none',  # 'bicubic',
                            origin='lower',
                            vmin=varMin,
                            vmax=varMax,
                            cmap='viridis')

    # Position of labels for the x-axis:
    vecXlblsPos = np.array([0, (aryMean.shape[1] - 1)])
    # Set position of labels for the x-axis:
    axsTmp.set_xticks(vecXlblsPos)
    # Create list of strings for labels:
    lstXlblsStr = ['WM', 'CSF']
    # Set the content of the labels (i.e. strings):
    axsTmp.set_xticklabels(lstXlblsStr,
                           alpha=0.9,
                           fontname=strFont,
                           fontweight='bold',
                           fontsize=8.0,
                           color=vecFontClr)

    # Position of labels for the y-axis:
    vecYlblsPos = np.arange(-0.5, (varEccNum - 0.5), 1.0)
    # Set position of labels for the y-axis:
    axsTmp.set_yticks(vecYlblsPos)
    # Create list of strings for labels:
    # lstYlblsStr = map(str,
    #                   np.around(vecEccBin, decimals=1)
    #                   )
    lstYlblsStr = [str(x) for x in np.around(vecEccBin, decimals=1)]
    # Set the content of the labels (i.e. strings):
    axsTmp.set_yticklabels(lstYlblsStr,
                           alpha=0.9,
                           fontname=strFont,
                           fontweight='bold',
                           fontsize=8.0,
                           color=vecFontClr)

    # Turn of ticks:
    axsTmp.tick_params(labelcolor=([0.0, 0.0, 0.0]),
                       top=False,
                       bottom=False,
                       left=False,
                       right=False)

    # We create invisible axes for the colour bar slightly to the right of the
    # position of the last data-axes. First, retrieve position of last
    # data-axes:
    objBbox = axsTmp.get_position()
    # We slightly adjust the x-position of the colour-bar axis, by shifting
    # them to the right:
    vecClrAxsPos = np.array([(objBbox.x0 * 7.5),
                             objBbox.y0,
                             objBbox.width,
                             objBbox.height])
    # Create colour-bar axis:
    axsClr = fig01.add_axes(vecClrAxsPos,
                            frameon=False)

    # Add colour bar:
    pltClrbr = fig01.colorbar(pltTmpCorr,
                              ax=axsClr,
                              fraction=1.0,
                              shrink=1.0)

    # The values to be labeled on the colour bar:
    vecClrLblsPos = np.linspace(varMin, varMax, num=3)

    # The labels (strings):
    # vecClrLblsStr = map(str, vecClrLblsPos)
    vecClrLblsStr = [str(x) for x in vecClrLblsPos]

    # Set labels on coloubar:
    pltClrbr.set_ticks(vecClrLblsPos)
    pltClrbr.set_ticklabels(vecClrLblsStr)
    # Set font size of colour bar ticks, and remove the 'spines' on the right
    # side:
    pltClrbr.ax.tick_params(labelsize=8.0,
                            tick2On=False)

    # Make colour-bar axis invisible:
    axsClr.axis('off')

    # Save figure:
    fig01.savefig(strPathOut,
                  dpi=160.0,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  bbox_inches='tight',
                  pad_inches=0.2,
                  transparent=False,
                  frameon=None)
