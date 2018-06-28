# -*- coding: utf-8 -*-
"""Fit Gaussian function to data 1D vector."""

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
from scipy.optimize import curve_fit


def funcGauss(varX, varMu, varSd, varInt):
    """
    Gaussian function to be fitted to the data.

    Parameters
    ----------
    varX : float
        Input value for function.
    varMu : float
        Position of the center of the peak of the Gaussian.
    varSd : float
        Width of the Gaussian (standard deviation).
    varInt : float
        Intercept. Because of the range of the input data (containing negative
        values), we need to include an intercept.

    Returns
    -------
    varOut : float
        Output value of Gaussian function (f(x)).

    """
    # The exponent of the Gaussian function:
    varExp = np.multiply(
                         -0.5,
                         np.square(
                                   ((varX - varMu) / varSd)
                                   )
                         )

    # The factor of the Gaussian function:
    varFac = np.divide(
                       1.0,
                       np.multiply(
                                   varSd,
                                   np.sqrt((2.0 * np.pi))
                                   )
                       )

    # Bringing things together (textbook Gaussian function):
    varOut = np.multiply(
                         varFac,
                         np.exp(varExp)
                         )

    # Include intercept:
    varOut = varOut + varInt

    return varOut


def fitGauss(vecX, vecY):
    """
    Fit Gaussian function.

    Parameters
    ----------
    vecX : np.array
        1D numpy array with dependent data.
    vecY : np.array
        1D numpy array with independet data.

    Returns
    -------
    varMu : float
        Position of the center of the peak of the Gaussian.
    varSd : float
        Width of the Gaussian (standard deviation).
    varInt : float
        Intercept. Because of the range of the input data (containing negative
        values), we need to include an intercept.

    """
    # Fit function to data. (p0 are initial values for parameters, and bounds
    # are chosen to achieve solutions within plausible range.)
    vecGaussMdlPar, vecExpMdlCov = curve_fit(funcGauss, vecX, vecY,
                                             p0=(3.75, 1.0, -3.0),
                                             bounds=([2.0, 0.0, -20.0],
                                                     [5.5, 100.0, 10.0])
                                             )

    varMu = vecGaussMdlPar[0]
    varSd = vecGaussMdlPar[1]
    varInt = vecGaussMdlPar[2]

    return varMu, varSd, varInt

# def funcExp(varX, varA, varB, varC):
#     """Exponential function to be fitted to the data."""
#     varOut = varA * np.exp(-varB * varX) + varC
#     return varOut
#
#
# def funcLn(varX, varA, varB):
#     """Logarithmic function to be fitted to the data."""
#     varOut = varA * np.log(varX) + varB
#     return varOut
#
#
# def funcPoly2(varX, varA, varB, varC):
#     """2nd degree polynomial function to be fitted to the data."""
#     varOut = (varA * np.power(varX, 2) +
#               varB * np.power(varX, 1) +
#               varC)
#     return varOut
#
# def funcPoly3(varX, varA, varB, varC, varD):
#     """3rd degree polynomial function to be fitted to the data."""
#     varOut = (varA * np.power(varX, 3) +
#               varB * np.power(varX, 2) +
#               varC * np.power(varX, 1) +
#               varD)
#     return varOut
#
#
# def funcPow(varX, varA, varB, varC, varD):
#     """Power function to be fitted to the data."""
#     varOut = (varA * np.power((varX + varB), varC) + varD)
#     return varOut
