# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

Function of the event-related timecourses depth sampling sub-pipeline.
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


def funcPltErt(aryRoiErtMeanDpth,  #noqa
               aryError,
               varNumDpth,
               varNumCon,
               varNumVol,
               varDpi,
               varYmin,
               varYmax,
               varStimStrt,
               varStimEnd,
               varTr,
               lstConLbl,
               lgcLgnd,
               strXlabel,
               strYlabel,
               lgcCnvPrct,
               strTitle,
               strPthOut,
               varXlbl=2,
               varTmeScl=1.0):
    """Plot event-related timecourses."""
    # Create figure:
    fgr01 = plt.figure(figsize=(1200.0/varDpi, 800.0/varDpi),
                       dpi=varDpi)
    # Create axis:
    axs01 = fgr01.add_subplot(111)

    # Vector for x-data:
    vecX = range(0, varNumVol)

    # Convert the volume indicies ('0, 1, 2 ...') into seconds. First divide by
    # the temporal scaling factor (in case ERTs were temporally upsampled for
    # the averaging), then multiply with the TR in order to convert from
    # from indicies to seconds:
    vecX = np.multiply(
                       np.divide(np.array((vecX), dtype=np.float64),
                                 varTmeScl),
                       varTr
                       )

    # Subtract pre-stimulus interval:
    vecX = np.subtract(vecX, varStimStrt)

    # Time points to label in the negative range:
    vecXlblNeg = np.arange(
                           np.multiply(-1.0,
                                       np.floor(varStimStrt)
                                       ),
                           0.0,
                           int(varXlbl)
                           )

    # Time points to label in the positive range:
    vecXlblPos = np.arange(
                           0.0,
                           np.floor(np.max(vecX)),
                           int(varXlbl)
                           )

    # Put together negative and positive range:
    vecXlbl = np.concatenate((vecXlblNeg, vecXlblPos), axis=0)

    # Get index of time point zero (i.e. of stimulus onset):
    # varIdxZero = np.where((np.around(vecX, decimals=5) == 0.0))[0]

    # Prepare colour map:
    objClrNorm = colors.Normalize(vmin=0, vmax=(varNumCon - 1))
    objCmap = plt.cm.winter

    # Loop through conditions:
    # for idxCon in [3, 2, 1, 0]:
    for idxCon in range(0, varNumCon):

        # Adjust the colour of current line:
        # vecClrTmp = objCmap(objClrNorm(varNumCon - 1 - idxCon))
        vecClrTmp = objCmap(objClrNorm(idxCon))

        # Plot timecourse for current condition:
        plt01 = axs01.plot(vecX,  #noqa
                           aryRoiErtMeanDpth[idxCon, :],
                           color=vecClrTmp,
                           alpha=0.9,
                           label=(lstConLbl[idxCon]),
                           linewidth=8.0,
                           antialiased=True)

        # Plot error shading:
        plot02 = axs01.fill_between(vecX,  #noqa
                                    np.subtract(aryRoiErtMeanDpth[idxCon, :],
                                                aryError[idxCon, :]),
                                    np.add(aryRoiErtMeanDpth[idxCon, :],
                                           aryError[idxCon, :]),
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
    axs01.set_xlim([varStimStrt, varStimEnd])
    # Set y-axis range:
    axs01.set_ylim([varYmin, varYmax])

    # Which x values to label with ticks:
    axs01.set_xticks(vecXlbl)

    # Convert labels from float to list of strings:
    # lstXlbl = map(str, vecXlbl)

    # Labels for x ticks:
    # axs01.set_xticklabels(lstXlbl)

    # Which y values to label with ticks:
    vecYlbl = np.linspace(varYmin, varYmax, num=5, endpoint=True)
    # vecYlbl = np.arange(varYmin, varYmax, 0.02)
    # vecYlbl = np.linspace(0.0, varYmax, num=5, endpoint=True)
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
            lstYlbl[idxLbl] = '{:0.1f}'.format(vecYlbl[idxLbl])
    else:
        # Convert labels from float to a list of strings, with well-defined
        # number of decimals (including trailing zeros):
        lstYlbl = [None] * vecYlbl.shape[0]
        for idxLbl in range(vecYlbl.shape[0]):
            lstYlbl[idxLbl] = '{:0.2f}'.format(vecYlbl[idxLbl])
    # Set tick labels for y ticks:
    axs01.set_yticklabels(lstYlbl)

    # Set x & y tick parameters:
    axs01.tick_params(labelsize=36,  # Fontsize
                      length=8,     # Height of the ticks
                      width=2,       # Width of the ticks
                      top=False,
                      right=False)

    # Adjust labels:
    axs01.set_xlabel(strXlabel,
                     fontsize=36)
    axs01.set_ylabel(strYlabel,
                     fontsize=36)

    # Adjust title:
    axs01.set_title(strTitle, fontsize=36)

    # Legend for axis 1:
    if lgcLgnd:
        axs01.legend(loc=0,
                     frameon=False,
                     prop={'size': 32})

    # Plot horizontal bar for stimulus interval:
    plot03 = plt.plot(((0.0), (varStimEnd - varStimStrt)),  #noqa
                      ((varYmin + 0.002), (varYmin + 0.002)),
                      color=(0.3, 0.3, 0.3),
                      linewidth=8.0,
                      label='_nolegend_')

    # Save figure:
    fgr01.savefig(strPthOut,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)

    # Close figure:
    plt.close(fgr01)
