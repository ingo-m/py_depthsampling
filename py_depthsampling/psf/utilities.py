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


def crt_gauss_1D(varSize, varPosEcc, varSd):
    """
    Create 2D Gaussian kernel.

    Parameters
    ----------
    varSize : int, positive
        Size of the vector in which the Gaussian is created (number of points).
    varPosEcc : int, positive
        Position of centre of 1D Gaussian.
    varSd : float, positive
        Standard deviation of 1D Gaussian.

    Returns
    -------
    vecGauss : 1d numpy array, shape [varSize]
        1D Gaussian.
    """
    varSize = int(varSize)

    vecEcc = np.arange(0.0, varSize)

    # The actual creation of the Gaussian array:
    vecGauss = (
        (np.square((vecEcc - varPosEcc))
         ) /
        (2.0 * np.square(varSd))
        )

    vecGauss = np.exp(-vecGauss) / (2.0 * np.pi * np.square(varSd))

    return vecGauss
