# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

@author: Ingo Marquardt, 06.11.2016
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
from scipy import stats
from ds_pltAcrDpth import funcPltAcrDpth


def funcLinRegAcrSubs(arySubDpthMns,
                      vecLinRegMdl,
                      varNumSubs,
                      varNumDpth,
                      strTitle,
                      strXlabel,
                      strLinRegYlabel,
                      varLinRegYmin,
                      varLinRegYmax,
                      varLinRegP,
                      varDpi,
                      strPltOtPre,
                      strPltOtSuf):
    """
    Calculate & plot across-subjects simple linear regression.

    A simple linear regression is model is calculated to test the dependence of
    the parameter estimates (i.e. signal change) to stimulus condition (i.e.
    luminance contrast), separately for each depth level.
    """
    # Repeat linear regression model for each subject:
    vecLinRegX = np.tile(vecLinRegMdl, varNumSubs)

    # Vectors for regression results (one regression per depth level):

    # Slope of the regression line:
    vecSlpe = np.zeros(varNumDpth)
    # Intercept of the regression line:
    vecItrcpt = np.zeros(varNumDpth)
    # Correlation coefficient:
    vecCor = np.zeros(varNumDpth)
    # p-value:
    vecP = np.zeros(varNumDpth)
    # Standard error of the estimate:
    vecStdErr = np.zeros(varNumDpth)

    # Loop through depth levels to calculate linear regression independently
    # at each depth level:
    for idxDpth in range(0, varNumDpth):

        # Get a flat array with the parameter estiamtes for each condition for
        # each subject:
        vecLinRegY = arySubDpthMns[:, :, idxDpth].flatten()

        # Fit the regression model:
        vecSlpe[idxDpth], vecItrcpt[idxDpth], vecCor[idxDpth], vecP[idxDpth], \
            vecStdErr[idxDpth] = stats.linregress(vecLinRegX, vecLinRegY)

    # File name for figure:
    strPltOt = strPltOtPre + 'acrsSubsLinReg' + strPltOtSuf

    # Create plot:
    funcPltAcrDpth(vecSlpe[None, :],    # Data to be plotted
                   vecStdErr[None, :],  # Error shading
                   varNumDpth,          # Number of depth levels
                   1,                   # Number of conditions (separate lines)
                   varDpi,              # Resolution of the output figure
                   varLinRegYmin,       # Minimum of Y axis
                   varLinRegYmax,       # Maximum of Y axis
                   False,               # Whether to convert y axis to %
                   [''],                # Labels for conditions
                   strXlabel,           # Label on x axis
                   strLinRegYlabel,     # Label on y axis
                   strTitle,            # Figure title
                   False,               # Boolean: whether to plot a legend
                   strPltOt)            # Output path for the figure
