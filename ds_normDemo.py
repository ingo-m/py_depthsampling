# -*- coding: utf-8 -*-
"""
Function of the depth sampling library.

The purpose of this script is to demonstrate normalisation of depth profiles on
simulated data. This is useful to understand which effect normalisation (e.g.
by division or subtraction) has on the depth profiles.
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
from ds_pltAcrDpth import funcPltAcrDpth


# -----------------------------------------------------------------------------
# *** Define parameters:

# Number of depths:
varNumDpth = 100

# Scaling factor for amplitude of the mid-GM 'bump' (higher value --> higher
# amplitude):
varBumpAmp = 1.0

# Scaling factor for the width of the mid-GM 'bump' (higher value --> sharper
# bump):
varBumpWidth = 8.0

# Figure scaling factor:
varDpi = 80.0

# Output folder:
strPthOt = '/home/john/PhD/Tex/dpth_norm_demo/plots/'
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create profile templates

# Linear datapoints:
aryLin = np.linspace(0.0, 1.0, varNumDpth, endpoint=True)

# Sinusoidal datapoints:
arySin = np.sin(np.linspace(0.0, np.pi, varNumDpth, endpoint=True))

# Make the sinusoid more sharp:
arySin = np.power(arySin, varBumpWidth)

# Scale the sinusoid:
arySin = np.multiply(arySin, varBumpAmp)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create condition profiles

# Create linear & sinusoidal components

# Linear & sinusiodal term additive:
aryDemoAddLin = np.array((np.add(aryLin, 0.0),
                          np.add(aryLin, 1.0),
                          np.add(aryLin, 2.0),
                          np.add(aryLin, 3.0)))
aryDemoAddSin = np.array((np.add(arySin, 0.0),
                          np.add(arySin, 1.0),
                          np.add(arySin, 2.0),
                          np.add(arySin, 3.0)))

# Linear & sinusiodal term multiplicative:
aryDemoMulLin = np.array((np.multiply(aryLin, 1.0),
                          np.multiply(aryLin, 2.0),
                          np.multiply(aryLin, 3.0),
                          np.multiply(aryLin, 4.0)))
aryDemoMulSin = np.array((np.multiply(arySin, 1.0),
                          np.multiply(arySin, 2.0),
                          np.multiply(arySin, 3.0),
                          np.multiply(arySin, 4.0)))

# Linear is additive, sinusoidal is multiplicative:
aryDemoMix01Lin = np.array((np.add(aryLin, 0.0),
                            np.add(aryLin, 1.0),
                            np.add(aryLin, 2.0),
                            np.add(aryLin, 3.0)))
aryDemoMix01Sin = np.array((np.multiply(arySin, 1.0),
                            np.multiply(arySin, 2.0),
                            np.multiply(arySin, 3.0),
                            np.multiply(arySin, 4.0)))


# Linear is multiplicative, sinusoidal is additive:
aryDemoMix02Lin = np.array((np.multiply(aryLin, 1.0),
                            np.multiply(aryLin, 2.0),
                            np.multiply(aryLin, 3.0),
                            np.multiply(aryLin, 4.0)))
aryDemoMix02Sin = np.array((np.add(arySin, 0.0),
                            np.add(arySin, 1.0),
                            np.add(arySin, 2.0),
                            np.add(arySin, 3.0)))

# Put profiles into list for plotting:
lstLin = [aryDemoAddLin, aryDemoMulLin, aryDemoMix01Lin, aryDemoMix02Lin]
lstSin = [aryDemoAddSin, aryDemoMulSin, aryDemoMix01Sin, aryDemoMix02Sin]


# Scale linear & sinusoidal components (so that final plots have comparabale
# range):
for idxPlt in range(len(lstLin)):
    # Maximum of linear profiles:
    varTmpMax = np.max(lstLin[idxPlt])
    # Scale maximum:
    lstLin[idxPlt] = np.multiply(np.divide(lstLin[idxPlt],
                                           varTmpMax),
                                 2.0)
    # Scale minimum:
    lstLin[idxPlt] = np.add(lstLin[idxPlt], 0.5)

    # Maximum of sinusoidal profiles:
    varTmpMax = np.max(lstSin[idxPlt])
    # Scale maximum:
    lstSin[idxPlt] = np.multiply(np.divide(lstSin[idxPlt],
                                           varTmpMax),
                                 2.0)
    # Scale minimum:
    lstSin[idxPlt] = np.add(lstSin[idxPlt], 0.5)

# Create combined profiles from linear & sinusoidal terms:
lstComb = [None] * len(lstLin)
for idxPlt in range(len(lstLin)):
    lstComb[idxPlt] = np.add(lstLin[idxPlt],
                             lstSin[idxPlt])

# Plot titles:
lstTtl = ['Linear & sinusiodal term additive',
          'Linear & sinusiodal term multiplicative',
          'Linear additive, sinusoidal multiplicative',
          'Linear multiplicative, sinusoidal additive']
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Plot profile templates

# Plot counter:
varCnt = 1

# Array with values for error bars:
aryError = np.zeros(lstComb[0].shape)
aryError = np.add(aryError, 0.01)

# Label on x axis
strXlabel = 'Cortical depth'

# Label on y axis
strYlabel = 'Signal change'

# Condition labels:
lstConLbl = ['72.0%', '16.3%', '6.1%', '2.5%']

# Scale linear term and add 2nd dimension:
aryTmp = np.array([np.multiply(aryLin, 1.0)], ndmin=2)

# Plot linear term:
funcPltAcrDpth(aryTmp,
               aryError,
               varNumDpth,
               1,
               (varDpi * 2.0),
               0.0,
               1.0,
               False,
               lstConLbl,
               strXlabel,
               strYlabel,
               'Linear term',
               False,
               (strPthOt + 'plt_' + str(varCnt).zfill(2) + '.svg'),
               varPadY=(0.1, 0.1))

varCnt += 1

# Scale sinusoidal term and add 2nd dimension:
aryTmp = np.array([np.divide(arySin, varBumpAmp)], ndmin=2)

# Plot sinusoidal term:
funcPltAcrDpth(aryTmp,
               aryError,
               varNumDpth,
               1,
               (varDpi * 2.0),
               0.0,
               1.0,
               False,
               lstConLbl,
               strXlabel,
               strYlabel,
               'Sinusoidal term',
               False,
               (strPthOt + 'plt_' + str(varCnt).zfill(2) + '.svg'),
               varPadY=(0.1, 0.1))

varCnt += 1
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create plots before normalisation

for idxPlt in range(len(lstComb)):

    # Plot linear component:
    funcPltAcrDpth(lstLin[idxPlt],
                   aryError,
                   varNumDpth,
                   4,
                   (varDpi * 2.0),
                   0.5,
                   2.5,
                   False,
                   lstConLbl,
                   strXlabel,
                   strYlabel,
                   '',
                   False,
                   (strPthOt + 'plt_' + str(varCnt).zfill(2) + '.svg'),
                   varPadY=(0.2, 0.2),
                   varNumLblY=3)
    varCnt += 1

    # Plot sinusoidal component:
    funcPltAcrDpth(lstSin[idxPlt],
                   aryError,
                   varNumDpth,
                   4,
                   (varDpi * 2.0),
                   0.5,
                   2.5,
                   False,
                   lstConLbl,
                   strXlabel,
                   strYlabel,
                   '',
                   False,
                   (strPthOt + 'plt_' + str(varCnt).zfill(2) + '.svg'),
                   varPadY=(0.2, 0.2),
                   varNumLblY=3)
    varCnt += 1

    # Plot combined profile:
    funcPltAcrDpth(lstComb[idxPlt],
                   aryError,
                   varNumDpth,
                   4,
                   varDpi,
                   1.0,
                   5.0,
                   False,
                   lstConLbl,
                   strXlabel,
                   strYlabel,
                   lstTtl[idxPlt],
                   False,
                   (strPthOt + 'plt_' + str(varCnt).zfill(2) + '.svg'),
                   varSizeX=1600.0,
                   varSizeY=1200.0,
                   varPadY=(0.2, 0.2),
                   varNumLblY=5)
    varCnt += 1
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Normalise by subtraction

lstCombNormSub = [None] * len(lstComb)

for idxPlt in range(len(lstComb)):

    # Profile of first stimulus condition:
    vecNorm = np.array(lstComb[idxPlt][0, :], ndmin=2)

    # Divide all conditions by first condition:
    lstCombNormSub[idxPlt] = np.subtract(lstComb[idxPlt],
                                         vecNorm)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create plots after normalisation by subtraction

for idxPlt in range(len(lstComb)):

    # Plot combined profile:
    funcPltAcrDpth(lstCombNormSub[idxPlt],
                   aryError,
                   varNumDpth,
                   4,
                   varDpi,
                   0.0,
                   3.0,
                   False,
                   lstConLbl,
                   strXlabel,
                   strYlabel,
                   lstTtl[idxPlt],
                   False,
                   (strPthOt + 'plt_' + str(varCnt).zfill(2) + '.svg'),
                   varSizeX=1600.0,
                   varSizeY=1200.0,
                   varPadY=(0.2, 0.2),
                   varNumLblY=4)
    varCnt += 1
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Normalise by division

lstCombNormDiv = [None] * len(lstComb)

for idxPlt in range(len(lstComb)):

    # Profile of first stimulus condition:
    vecNorm = np.array(lstComb[idxPlt][0, :], ndmin=2)

    # Divide all conditions by first condition:
    lstCombNormDiv[idxPlt] = np.divide(lstComb[idxPlt],
                                       vecNorm)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create plots after normalisation by division

for idxPlt in range(len(lstComb)):

    # Plot combined profile:
    funcPltAcrDpth(lstCombNormDiv[idxPlt],
                   aryError,
                   varNumDpth,
                   4,
                   varDpi,
                   1.0,
                   4.0,
                   False,
                   lstConLbl,
                   strXlabel,
                   strYlabel,
                   lstTtl[idxPlt],
                   False,
                   (strPthOt + 'plt_' + str(varCnt).zfill(2) + '.svg'),
                   varSizeX=1600.0,
                   varSizeY=1200.0,
                   varPadY=(0.2, 0.2),
                   varNumLblY=4)
    varCnt += 1
# -----------------------------------------------------------------------------
