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


def psf(aryIn, varSd, varFct):
    """
    Cortical depth point spread function.

    Parameters
    ----------
    aryIn : np.array
        2D numpy array containing visual field projection of activation map
        (e.g. percent signal change). Visual field projection can be created
        with `py_depthsampling.project.project_main.py`.
    varSd : float
        Width (standard deviation) of Gaussian function used to for point
        spread function.
    varFct: float
        Factor by which visual field projection is multiplied (scaling is
        necessary to account for different percent signal change levels between
        cortical depth levels).

    Returns
    -------
    aryOut : np.array
        2D numpy array with same shape as `aryIn`, containing visual field
        projection after application of point spread function.

    Notes
    -----
    Gaussian filtering and multiplication are commutative.

    """
    # Apply scaling:
    aryOut = np.multiply(aryIn, varFct)

    # Apply Gaussian filter:
    aryOut = gaussian_filter(aryOut,
                             varSd,
                             order=0,
                             mode='nearest',
                             truncate=4.0)

    return aryOut


def psf_diff(vecParams, aryDeep, aryTrgt):
    """
    Calculate difference btwn. visual field projections, given PSF parameters.

    Parameters
    ----------
    vecParams : np.array
        1D numpy array containing two parameters of point spread function
        (parameters need to be passed in this form in order to comply with the
        optimization function scipy.optimize.minimize). The two parameters are:
            varSd : float
                Width (standard deviation) of Gaussian function used to for
                point spread function.
            varFct: float
                Factor by which visual field projection is multiplied (scaling
                is necessary to account for different percent signal change
                levels between cortical depth levels).
    aryDeep : np.array
        2D numpy array containing visual field projection of deepest depth
        level (reference visual field projection). The point spread function
        is applied to this array.
    aryTrgt : np.array
        2D numpy array containing target visual field projection (at more
        superficial cortical depth). The point spread function is applied to
        the deepest (reference) visual field projection in order to become more
        similar to the target visual field projection.

    Returns
    -------
    varDiff : float
        Sum of absolute difference between reference (`aryDeep`) and taregt
        (`aryTrgt`) visual field projections.

    """
    # Get width of Gaussian and multiplication factor from input vector:
    varSd = vecParams[0]
    varFct = vecParams[1]

    # Apply point spread function to reference visual field projection:
    aryDeep_fltr = psf(aryDeep, varSd, varFct)

    # Calculate difference between filtered reference and target visual field
    # projections:
    varDiff = np.mean(np.absolute(np.subtract(aryTrgt, aryDeep_fltr)))

    return varDiff
