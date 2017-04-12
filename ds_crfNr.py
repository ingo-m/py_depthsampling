# -*- coding: utf-8 -*-
"""
Contrast response function - hyperbolic ratio.

Notes
-----
The purpose of this script is to plot the hyperbolic ratio function, a
function used to model the contrast response of visual neurons, also known as
Naka-Rushton equation.

Reference
---------
- Albrecht, D. G., & Hamilton, D. B. (1982). Striate cortex of monkey and cat:
  contrast response function. Journal of neurophysiology, 48(1), 217-237.
- Niemeyer, J. E., & Paradiso, M. A. (2017). Contrast sensitivity, V1 neural
  activity, and natural vision. Journal of neurophysiology, 117(2), 492-508.
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
from ds_crfPlot import plt_crf


def crf_hyper(varC, varRmax, varC50, varN):
    """
    Hyperbolic ratio function.

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
        Exponent. It specifies the rate of change, or slope, of the function.
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


# *** Define parameters

strPthOt = '/home/john/Desktop/hyperbolic_ratio.png'

# *** Response parameters

# Maximum response:
varRmax = 5.0
# Contrast that gives a half-maximal response:
varC50 = 0.5
# Average baseline firing rate:
# varM = 0.1
# Exponent:
varN = 0.6

strMdl = ('R(C) = '
          + str(varRmax)
          + ' * (C^'
          + str(varN)
          + ' / (C^'
          + str(varN)
          + ' + '
          + str(varC50)
          + '^'
          + str(varN)
          + '))'
          )

# *** Apply function

# x-values for which the model will be fitted:
vecMdlX = np.linspace(0.0, 1.0, num=1000.0, endpoint=True)

# Modelled response (y-values as a function of contrast):
vecMdlY = crf_hyper(vecMdlX, varRmax, varC50, varN)

# *** Plot data

plt_crf(vecMdlX,
        vecMdlY,
        strPthOt,
        varYmin=0.0,
        varYmax=3.0,
        strLblX='Luminance contrast',
        strLblY='Response',
        strTtle='Hyperbolic ratio CRF',
        varDpi=80.0,
        lgcLgnd=False,
        strMdl=strMdl)
