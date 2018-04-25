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
import matplotlib.colors as colors
# from matplotlib.colors import BoundaryNorm


def boot_plot_sngl(objDpth, strPath, lstCon, lstConLbl, varMin=None,
                   varMax=None, strXlabel='Cortical depth level (equivolume)',
                   strYlabel='Subject', lstDiff=None):
    """
    Plot single subject cortical depth profiles or difference scores.

    Parameters
    ----------
    objDpth : np.array or str
        Array with single-subject cortical depth profiles, of the form:
        aryDpth[idxSub, idxCondition, idxDpth]. Either a numpy array or a
        string with the path to an npy file containing the array.
    strPath : str
        Output path for plot, condition name left open.
    lstCon : list
        Abbreviated condition levels used to complete file names (e.g. 'Pd').
        Number of abbreviated condition labels has to be the same as number of
        conditions in `objDpth`.
    lstConLbl : list
        List containing condition labels (strings), e.g. 'PacMan Dynamic'.
        Number of condition labels has to be the same as number of conditions
        in `objDpth`.
    varMin : float
        Minimum of Y axis.
    varMax : float
        Maximum of Y axis.
    strXlabel : str
        Label for x axis.
    strYlabel : str
        Label for y axis.
    lstDiff : list or None
        If None, the depth profiles are plotted separately for each condition.
        If a list of tuples of condition indices is provided, on each
        bootstrapping iteration the difference between the two conditions is
        calculated, and is plotted. The the second condition from the tuple is
        subtracted from the first (e.g. if lstDiff = [(0, 1)], then condition 1
        is subtracted from condition 0).

    Returns
    -------
    None : None
        This function has no return value.

    Notes
    -----
    Plot a normalised difference score of the cortical depth profiles between
    two conditions (i.e. (A - B) / (A + B)) across cortical depth (x axis)
    separately for each subjcet (y axis).

    This version plots the values using two separate colourmaps for negative
    and positive values.
    """
    # -------------------------------------------------------------------------
    # *** Data preparations

    # Test whether the input is a numpy array or a string (with the path to a
    # numpy array):
    lgcAry = (type(objDpth) == np.ndarray)
    lgcStr = (type(objDpth) == str)

    # If input is a string, load array from npy file:
    if lgcAry:
        aryDpth = objDpth
    elif lgcStr:
        aryDpth = np.load(objDpth)
    else:
        print(('---Error in bootPlot: input needs to be numpy array or path '
               + 'to numpy array.'))

    # Get number of subjects from input array:
    varNumSub = aryDpth.shape[0]

    # Get number of conditions from input array:
    if lstDiff is None:
        varNumCon = aryDpth.shape[1]
    else:
        # Will plot differences between condition pairs:
        varNumCon = len(lstDiff)

    # Get number of depth levels from input array:
    varNumDpth = aryDpth.shape[2]

    # Loop through conditions:
    for idxCon in range(varNumCon):

        # ---------------------------------------------------------------------
        # *** Calculate difference scores

        if lstDiff is None:

            # Will plot simple cortical depth profiles:
            aryPlot = aryDpth[:, idxCon, :]

        else:

            # Calculate difference scores:
            aryPlot = \
                np.divide(
                    np.subtract(
                        aryDpth[:, lstDiff[idxCon][0], :],
                        aryDpth[:, lstDiff[idxCon][1], :]
                        ),
                    np.add(
                        aryDpth[:, lstDiff[idxCon][0], :],
                        aryDpth[:, lstDiff[idxCon][1], :]
                        )
                    )

        # ---------------------------------------------------------------------
        # *** Prepare figure attributes

        # Font type:
        strFont = 'Liberation Sans'

        # Font colour:
        vecFontClr = np.array([17.0/255.0, 85.0/255.0, 124.0/255.0])

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
        axsCmn.set_xlabel(strXlabel,
                          alpha=1.0,
                          fontname=strFont,
                          fontweight='normal',
                          fontsize=7.0,
                          color=vecFontClr,
                          position=(0.5, 0.0))
        axsCmn.set_ylabel(strYlabel,
                          alpha=1.0,
                          fontname=strFont,
                          fontweight='normal',
                          fontsize=7.0,
                          color=vecFontClr,
                          position=(0.0, 0.5))

        if lstDiff is None:

            # Title for simple signal change plot:
            strTtle = 'fMRI signal change'
    
        else:
    
            # Title for difference score plot:
            strTtle = (lstConLbl[lstDiff[idxCon][0]]
                       + ' minus '
                       + lstConLbl[lstDiff[idxCon][1]])

        axsCmn.set_title(strTtle,
                         alpha=1.0,
                         fontname=strFont,
                         fontweight='bold',
                         fontsize=10.0,
                         color=vecFontClr,
                         position=(0.5, 1.1))

        # ---------------------------------------------------------------------
        # *** Colour mapping

        # Create colour-bar axis:
        axsTmp = fig01.add_subplot(111)

        # Number of colour increments:
        varNumClr = 20

        # Colour values for the first colormap (used for negative values):
        aryClr01 = plt.cm.PuBu(np.linspace(0.1, 1.0, varNumClr))

        # Invert the first colour map:
        aryClr01 = np.flipud(np.array(aryClr01, ndmin=2))

        # Colour values for the second colormap (used for positive values):
        aryClr02 = plt.cm.OrRd(np.linspace(0.1, 1.0, varNumClr))

        # Combine negative and positive colour arrays:
        aryClr03 = np.vstack((aryClr01, aryClr02))

        # Create new custom colormap, combining two default colormaps:
        objCustClrMp = colors.LinearSegmentedColormap.from_list('custClrMp',
                                                                aryClr03)

        # Find minimum and maximum values:
        if varMin is None:
            varMin = -0.25
            #varMin = np.percentile(aryPlot[:, :], 5.0)
            # Round:
            #varMin = (np.floor(varMin * 0.1) / 0.1)
        if varMax is None:
            varMax = 0.25
            #varMax = np.percentile(aryPlot[:, :], 95.0)
            # Round:
            #varMax = (np.ceil(varMax * 0.1) / 0.1)

        # Lookup vector for negative colour range:
        vecClrRngNeg = np.linspace(varMin, 0.0, num=varNumClr)

        # Lookup vector for positive colour range:
        vecClrRngPos = np.linspace(0.0, varMax, num=varNumClr)

        # Stack lookup vectors:
        vecClrRng = np.hstack((vecClrRngNeg, vecClrRngPos))

        # 'Normalize' object, needed to use custom colour maps and lookup table
        # with matplotlib:
        objClrNorm = colors.BoundaryNorm(vecClrRng, objCustClrMp.N)

        # ---------------------------------------------------------------------
        # *** Create plot

        # Plot correlation coefficients of current depth level:
        pltTmpCorr = plt.imshow(aryPlot,
                                interpolation='none',  # 'bicubic',
                                origin='lower',
                                norm=objClrNorm,
                                cmap=objCustClrMp)

        # Position of labels for the x-axis:
        vecXlblsPos = np.array([0, (varNumDpth - 1)])
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
        vecYlblsPos = np.arange(-0.5, (varNumSub - 0.5), 1.0)
        # Set position of labels for the y-axis:
        axsTmp.set_yticks(vecYlblsPos)
        # Create list of strings for labels:
        lstYlblsStr = map(str,
                          range(1, (varNumSub + 1))
                          )
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

        # We create invisible axes for the colour bar slightly to the right of
        # the position of the last data-axes. First, retrieve position of last
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
        # vecClrLblsPos01 = np.arange(varMin, 0.0, 10)
        # vecClrLblsPos02 = np.arange(0.0, varMax, 100)
        vecClrLblsPos01 = np.linspace(varMin, 0.0, num=3)
        vecClrLblsPos02 = np.linspace(0.0, varMax, num=3)
        vecClrLblsPos = np.hstack((vecClrLblsPos01, vecClrLblsPos02))

        # The labels (strings):
        vecClrLblsStr = [str(x) for x in vecClrLblsPos]

        # Set labels on coloubar:
        pltClrbr.set_ticks(vecClrLblsPos)
        pltClrbr.set_ticklabels(vecClrLblsStr)
        # Set font size of colour bar ticks, and remove the 'spines' on the
        # right side:
        pltClrbr.ax.tick_params(labelsize=8.0,
                                tick2On=False)

        # Make colour-bar axis invisible:
        axsClr.axis('off')

        if lstDiff is None:
            # Output path:
            strPathTmp = strPath.format(lstCon[idxCon])

        else:
            # Output path difference scores:
            strPathTmp = strPath.format((lstCon[lstDiff[idxCon][0]]
                                         + '_minus_'
                                         + lstCon[lstDiff[idxCon][1]]))

        # Save figure:
        fig01.savefig(strPathTmp,
                      dpi=160.0,
                      facecolor='w',
                      edgecolor='w',
                      orientation='landscape',
                      bbox_inches='tight',
                      pad_inches=0.2,
                      transparent=False,
                      frameon=None)

        # Close figure:
        plt.close(fig01)
        # ---------------------------------------------------------------------
