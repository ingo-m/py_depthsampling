# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.
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


def crf_power(varC, varA, varB):
    """
    Power contrast response function.

    Parameters
    ----------
    varC : float
        Stimulus contrast (input parameter).
    varA : float
        Factor. Specifies overall response amplitude.
    varB : float
        Exponent. Specifies the rate of change, or slope, of the function.
        (Free parameter to be fitted.)

    Returns
    -------
    varR : float
        Neuronal response.

    Notes
    -----
    Simple power function. Can be used to model the contrast response of
    visual neurons.
    """
    varR = varA * np.power(varC, varB)
    return varR


def crf_hyper(varC, varRmax, varC50, varN):
    """
    Hyperbolic ratio contrast response function.

    Parameters
    ----------
    varC : float
        Stimulus contrast (input parameter).
    varRmax : float
        The maximum neural response (saturation point). (Free parameter to be
        fitted.)
    varC50 : float
        The contrast that gives a half-maximal response, know as
        semisaturation contrast. The semisaturation constant moves the curve
        horizontally and provides a good index of the contrast sensitivity at
        half the maximum response. (Free parameter to be fitted.)
    varN : float
        Exponent. Specifies the rate of change, or slope, of the function.
        (Free parameter to be fitted.)

    Returns
    -------
    varR : float
        Neuronal response.

    Notes
    -----
    Hyperbolic ratio function, a function used to model the contrast response
    of visual neurons. Also known as Naka-Rushton equation.

    References
    ----------
    - Albrecht, D. G., & Hamilton, D. B. (1982). Striate cortex of monkey and
      cat: contrast response function. Journal of neurophysiology, 48(1),
      217-237.
    - Niemeyer, J. E., & Paradiso, M. A. (2017). Contrast sensitivity, V1
      neural activity, and natural vision. Journal of neurophysiology, 117(2),
      492-508.
    """
    varR = (varRmax
            * np.power(varC, varN)
            / (np.power(varC, varN) + np.power(varC50, varN)))
    return varR

# Contrast-fMRI-response function as defined in Boynton et al. (1999).
#   - varR is response
#   - varC is stimulus contrast
#   - varP - determines shape of contrast-response function, typical value: 0.3
#   - varQ - determines shape of contrast-response function, typical value: 2.0
#   - varS - ?
#   - varA - Scaling factor
# def crf_fmri(varC, varS, varA):
#    """Contrast-fMRI-response function as defined in Boynton et al. (1999)"""
#    # varR = varS * np.log(varC) + varA
#    varP = 0.3
#    varQ = 2.0
#    varR = varA * np.divide(
#                            np.power(varC, (varP + varQ)),
#                            (np.power(varC, varQ) + np.power(varS, varQ))
#                            )
#    return varR
