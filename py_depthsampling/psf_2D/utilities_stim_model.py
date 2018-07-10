# -*- coding: utf-8 -*-
"""Functions of the depth sampling library."""

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
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns


def psf_stim_mdl(aryPacMan, aryEdge, aryPeri, varSd, varFctCntr, varFctEdge,
                 varFctPeri):
    """
    Stimulus-model based point spread function.

    Parameters
    ----------
    aryPacMan : np.array
        2D numpy array containing binary mask of stimulus centre. Created from
        `py_depthsampling.psf_2D.psf_from_stim_model.py`.
    aryEdge : np.array
        2D numpy array containing binary mask of stimulus edge. Created from
        `py_depthsampling.psf_2D.psf_from_stim_model.py`.
    aryPeri : np.array
        2D numpy array containing binary mask of periphery (i.e. part of visual
        space that is not contained in the stimulus & edge masks). Created from
        `py_depthsampling.psf_2D.psf_from_stim_model.py`.
    varSd : float
        Width (standard deviation) of Gaussian function used to for point
        spread function.
    varFctCntr: float
        Factor by which stimulus centre mask (`aryPacMan`) is multiplied.
    varFctEdge: float
        Factor by which stimulus centre mask (`aryEdge`) is multiplied.
    varFctPeri: float
        Factor by which stimulus centre mask (`aryPeri`) is multiplied.

    Returns
    -------
    aryOut : np.array
        2D numpy array with same shape as input arrays, containing visual field
        projection after application of point spread function.

    Notes
    -----
    This function may be used to fit an explicit model of a the spatial extent
    of stimulus-evoked activation (in visual space) represented by `aryPacMan`,
    `aryEdge`, and `aryPeri`) to empirically observed activation patterns.

    """
    # Apply scaling:
    aryPacMan = np.multiply(aryPacMan, varFctCntr)
    aryEdge = np.multiply(aryEdge, varFctEdge)
    aryPeri = np.multiply(aryPeri, varFctPeri)

    aryOut = (aryPacMan + aryEdge + aryPeri)

    # Apply Gaussian filter:
    aryOut = gaussian_filter(aryOut,
                             varSd,
                             order=0,
                             mode='nearest',
                             truncate=4.0)

    return aryOut


def psf_diff_stim_mdl(vecParams, aryPacMan, aryEdge, aryPeri, aryTrgt,
                      vecVslX):
    """
    Calculate difference btwn. visual field projections, given PSF parameters.

    Parameters
    ----------
    vecParams : np.array
        1D numpy array containing parameters of point spread function
        (parameters need to be passed in this form in order to comply with the
        optimization function scipy.optimize.minimize). Expected parameters
        are:
            varSd : float
                Width (standard deviation) of Gaussian function used to for
                point spread function.
            varFctCntr: float
                Factor by which stimulus centre mask (`aryPacMan`) is
                multiplied.
            varFctEdge: float
                Factor by which stimulus centre mask (`aryEdge`) is multiplied.
            varFctPeri: float
                Factor by which stimulus centre mask (`aryPeri`) is multiplied.
    aryPacMan : np.array
        2D numpy array containing binary mask of stimulus centre. Created from
        `py_depthsampling.psf_2D.psf_from_stim_model.py`.
    aryEdge : np.array
        2D numpy array containing binary mask of stimulus edge. Created from
        `py_depthsampling.psf_2D.psf_from_stim_model.py`.
    aryPeri : np.array
        2D numpy array containing binary mask of periphery (i.e. part of visual
        space that is not contained in the stimulus & edge masks). Created from
        `py_depthsampling.psf_2D.psf_from_stim_model.py`.
    aryTrgt : np.array
        2D numpy array containing (empirical) target visual field projection.
        The point spread function is applied to the visual field projection
        models (`aryPacMan`, `aryEdge`, `aryPeri`) so that they become more
        similar to the target visual field projection.
    vecVslX : np.array
        Vector with visual space x-coordinates.

    Returns
    -------
    varDiff : float
        Mean absolute difference between visual field projection models and
        target visual field projection.

    Notes
    -----
    This function may be used to fit an explicit model of a the spatial extent
    of stimulus-evoked activation (in visual space) represented by `aryPacMan`,
    `aryEdge`, and `aryPeri`) to empirically observed activation patterns.

    """
    # Get width of Gaussian and multiplication factors from input vector:
    varSd = vecParams[0]
    varFctCntr = vecParams[1]
    varFctEdge = vecParams[2]
    varFctPeri = vecParams[3]

    # Apply point spread function to reference visual field projection:
    aryFit = psf_stim_mdl(aryPacMan, aryEdge, aryPeri, varSd, varFctCntr,
                          varFctEdge, varFctPeri)

    # Calculate difference between filtered reference and target visual field
    # projections:
    varDiff = np.mean(np.absolute(np.subtract(aryTrgt, aryFit)))

    return varDiff


def plot_psf_params(strPathOut, strX, strY, strHue, objData, lstRoi,
                    varNumClr, varCi=90.0):
    """Plot parameters of point spread function."""
    # Create seaborn colour palette:
    # objClr = sns.light_palette((210, 90, 60), input="husl",
    #                            n_colors=varNumClr)
    lstClr = ["amber", "greyish", "faded green"]
    objClr = sns.xkcd_palette(lstClr)

    # Draw nested barplot:
    fgr01 = sns.factorplot(x=strX, y=strY, hue=strHue, data=objData, size=6,
                           kind="bar", palette=objClr, ci=varCi)

    # Set x-axis labels to upper case ROI labels:
    lstRoiUp = [x.upper() for x in lstRoi]
    fgr01.set_xticklabels(lstRoiUp)

    # Save figure:
    fgr01.savefig(strPathOut)
