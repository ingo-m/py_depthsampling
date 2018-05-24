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
import matplotlib.patches as patches


def ecc_histogram(vecEcc, vecEccBin, strPathOut):
    """
    Plot histogram for eccentricity & cortical depth analysis.

    Plots a histogram of the distribution of eccentricity values within an ROI.
    This function is part of a tool for analysis of cortical-depth-dependent
    fMRI responses at different retinotopic eccentricities.
    """
    # Lower and upper eccentricity bound of the visual stimulus:
    varStimEccLw = 0.0
    varStimEccUp = 3.75

    # Round the minimum & maximum for histogram endpoints:
    varHistXmin = 0.0  # np.floor(np.min(vecEcc))
    varHistXmax = 7.0  # np.ceil(np.max(vecEcc))

    # Bins for the histogram:
    vecHistBins = np.linspace(int(varHistXmin),
                              int(varHistXmax),
                              num=50,
                              endpoint=True)

    # Create figure:
    fgr01 = plt.figure()
    # Create axis:
    axs01 = fgr01.add_subplot(111)

    # Plot eccentricity bins (vertical lines):
    for idxBin in range(0, vecEccBin.shape[0]):
        axs01.axvline(vecEccBin[idxBin],
                      color=(0.0, 0.0, 0.0),
                      alpha=0.8,
                      linewidth=1.0,
                      antialiased=True)

    # Plot the histogram:
    plt01 = plt.hist(vecEcc,  #noqa
                     vecHistBins,
                     alpha=1.0,
                     align='mid',
                     rwidth=1.0,
                     color=(70.0/255.0, 188.0/255.0, 117.0/255.0),
                     linewidth=1.0,
                     edgecolor=(0.5, 0.5, 0.5),
                     antialiased=True,
                     zorder=2)

    # Get the maximum of the y-axis of the histogram:
    varHistYmax = axs01.get_ylim()[1]

    # Plot eccentricity bound of stimulus (patch):
    axs01.add_patch(patches.Rectangle((varStimEccLw, 0),  # xy  #noqa
                                      (varStimEccUp - varStimEccLw),  # width
                                      varHistYmax,  # height
                                      angle=0.0,
                                      alpha=0.5,
                                      edgecolor=(0.5, 0.5, 0.5),
                                      facecolor=(0.5, 0.5, 0.5),
                                      fill=True,
                                      zorder=1))

    # Font type & colour:
    strFont = 'Liberation Sans'
    tplFontClr = (0.0, 0.0, 0.0)

    # Set and adjust common axes labels:
    axs01.set_xlabel('pRF eccentricity',
                     alpha=1.0,
                     fontname=strFont,
                     fontweight='normal',
                     fontsize=14.0,
                     color=tplFontClr)  # position=(0.5, 0.0))
    axs01.set_ylabel('Number of vertices',
                     alpha=1.0,
                     fontname=strFont,
                     fontweight='normal',
                     fontsize=14.0,
                     color=tplFontClr)  # position=(0.0, 0.5))
    axs01.set_title('pRF eccentricity distribution',
                    alpha=1.0,
                    fontname=strFont,
                    fontweight='bold',
                    fontsize=16.0,
                    color=tplFontClr)  # position=(0.5, 1.1))

    # Adjust axis ticks:
    axs01.tick_params(labelsize=12.0)

    # Save figure:
    fgr01.savefig(strPathOut,
                  dpi=100.0,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  bbox_inches='tight',
                  pad_inches=0.5,
                  transparent=False,
                  frameon=None)
