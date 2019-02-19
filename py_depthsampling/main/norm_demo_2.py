# -*- coding: utf-8 -*-
"""
Function of the depth sampling library.

Demonstrate normalisation of depth profiles on simulated data. The effect of
subtractive and divisive normalisation is demonstrated for three scenarios:

(1) Two positive activation peaks at mid-grey matter.
(2) One positive activation peak at mid-grey matter, no activation in control
    condition.
(3) Positive and negative activation peaks.
"""

# Part of py_depthsampling library
# Copyright (C) 2019  Ingo Marquardt
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
from py_depthsampling.plot.plt_dpth_prfl import plt_dpth_prfl
from scipy.signal import fftconvolve


# -----------------------------------------------------------------------------
# *** Define parameters:

# Number of depths:
varNumDpth = 100

# Scaling factor for amplitude of the mid-GM 'bump' (higher value --> higher
# amplitude):
varBumpAmp = 0.25

# Scaling factor for the width of the mid-GM 'bump' (higher value --> sharper
# bump):
varBumpWidth = 20.0

# Figure scaling factor:
varDpi = 80.0

# Draining vein factor (higher value --> more unidirectional signal spread):
varFctr = 0.75

# Output folder:
strPthOt = '/home/john/Dropbox/Thesis/Chapters/General_Discussion/Figures/Figure_X_Normalisation/elements/'  #noqa

# Figure output file type:
strFleTyp = '.svg'
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create profile templates

# Linear datapoints:
vecLin = np.linspace(1.0, 0.0, varNumDpth, endpoint=True)

# Cubid decay:
vecDcy = np.power(vecLin, 2.0)

# Sinusoidal datapoints:
arySin = np.sin(np.linspace((0.15 * np.pi),
                            np.pi,
                            varNumDpth,
                            endpoint=True))

# Make the sinusoid more sharp:
arySin = np.power(arySin, varBumpWidth)

# Scale the sinusoid:
arySin = np.multiply(arySin, varBumpAmp)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create condition profiles

# Create components

# (1)
# Scenario 1: Two positive activation peaks at mid-grey matter.
vecSin01a = np.add(np.multiply(arySin, 0.5), 0.01)
vecSin01b = np.add(np.multiply(arySin, 1.0), 0.01)
arySin01 = np.array((vecSin01a, vecSin01b))

# (2)
# Scenario 2: One positive activation peak at mid-grey matter, no activation in
# control condition.
vecSin02a = np.add(np.multiply(arySin, 0.0), 0.01)
vecSin02b = np.add(np.multiply(arySin, 1.0), 0.01)
arySin02 = np.array((vecSin02a, vecSin02b))

# (3)
# Scenario 3: Positive and negative activation peaks.
vecSin03a = np.add(np.multiply(arySin, -1.0), -0.01)
vecSin03b = np.add(np.multiply(arySin, 1.0), 0.01)
arySin03 = np.array((vecSin03a, vecSin03b))

# Number of conditions:
varNumCon = arySin01.shape[0]

# Model draining vein effect:
for idxCon in range(varNumCon):
    arySin01[idxCon, :] = fftconvolve(arySin01[idxCon, :],
                                      vecDcy,
                                      mode='full')[0:varNumDpth]
    arySin02[idxCon, :] = fftconvolve(arySin02[idxCon, :],
                                      vecDcy,
                                      mode='full')[0:varNumDpth]
    arySin03[idxCon, :] = fftconvolve(arySin03[idxCon, :],
                                      vecDcy,
                                      mode='full')[0:varNumDpth]
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Normalisation by subtraction

vecNormSub01 = np.subtract(arySin01[1, :], arySin01[0, :])
vecNormSub02 = np.subtract(arySin02[1, :], arySin02[0, :])
vecNormSub03 = np.subtract(arySin03[1, :], arySin03[0, :])
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Normalisation by division

vecNormDiv01 = np.divide(arySin01[1, :], arySin01[0, :])
vecNormDiv02 = np.divide(arySin02[1, :], arySin02[0, :])
vecNormDiv03 = np.divide(arySin03[1, :], arySin03[0, :])
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Preparations

# Plot counter:
varCnt = 1

# Array with values for error bars:
aryError = np.zeros((varNumCon, varNumDpth))
aryError = np.add(aryError, 0.001)

# Label on x axis
strXlabel = 'Cortical depth'

# Label on y axis
strYlabel = 'Signal change [%]'

# Condition labels:
# lstConLbl = ['72.0%', '16.3%', '6.1%', '2.5%']
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create plots - Scenario 1

# Plot components without draining vein effect:
plt_dpth_prfl(np.array((vecSin01a, vecSin01b)),
              aryError,
              varNumDpth,
              varNumCon,
              (varDpi * 1.8),
              0.0,
              0.3,
              False,
              ['1', '2'],
              strXlabel,
              strYlabel,
              '',
              False,
              (strPthOt + 'scenario_01_components' + strFleTyp),
              tplPadY=(0.01, 0.01),
              varNumLblY=4,
              varRound=0)

# Plot hypothetical fMRI signal profile (with draining vein effect):
plt_dpth_prfl(arySin01,
              aryError,
              varNumDpth,
              varNumCon,
              (varDpi * 1.8),
              0.0,
              4.0,
              False,
              ['1', '2'],
              strXlabel,
              strYlabel,
              '',
              False,
              (strPthOt + 'scenario_01_fMRI' + strFleTyp),
              tplPadY=(0.1, 0.1),
              varNumLblY=5,
              varRound=0)

# Plot subtractive normalisation:
plt_dpth_prfl(vecNormSub01.reshape(1, varNumDpth),
              aryError,
              varNumDpth,
              1,
              (varDpi * 1.8),
              0.0,
              2.0,
              False,
              ['1', '2'],
              strXlabel,
              'Difference',
              '',
              False,
              (strPthOt + 'scenario_01_sub' + strFleTyp),
              tplPadY=(0.1, 0.1),
              varNumLblY=3,
              varRound=0)

# Plot divisive normalisation:
plt_dpth_prfl(vecNormDiv01.reshape(1, varNumDpth),
              aryError,
              varNumDpth,
              1,
              (varDpi * 1.8),
              0.0,
              2.0,
              False,
              ['1', '2'],
              strXlabel,
              'Ratio',
              '',
              False,
              (strPthOt + 'scenario_01_div' + strFleTyp),
              tplPadY=(0.1, 0.1),
              varNumLblY=3,
              varRound=0)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create plots - Scenario 2

# Plot components without draining vein effect:
plt_dpth_prfl(np.array((vecSin02a, vecSin02b)),
              aryError,
              varNumDpth,
              varNumCon,
              (varDpi * 1.8),
              0.0,
              0.3,
              False,
              ['1', '2'],
              strXlabel,
              strYlabel,
              '',
              False,
              (strPthOt + 'scenario_02_components' + strFleTyp),
              tplPadY=(0.01, 0.01),
              varNumLblY=4,
              varRound=0)

# Plot hypothetical fMRI signal profile (with draining vein effect):
plt_dpth_prfl(arySin02,
              aryError,
              varNumDpth,
              varNumCon,
              (varDpi * 1.8),
              0.0,
              4.0,
              False,
              ['1', '2'],
              strXlabel,
              strYlabel,
              '',
              False,
              (strPthOt + 'scenario_02_fMRI' + strFleTyp),
              tplPadY=(0.1, 0.1),
              varNumLblY=5,
              varRound=0)

# Plot subtractive normalisation:
plt_dpth_prfl(vecNormSub02.reshape(1, varNumDpth),
              aryError,
              varNumDpth,
              1,
              (varDpi * 1.8),
              0.0,
              4.0,
              False,
              ['1', '2'],
              strXlabel,
              'Difference',
              '',
              False,
              (strPthOt + 'scenario_02_sub' + strFleTyp),
              tplPadY=(0.1, 0.1),
              varNumLblY=5,
              varRound=0)

# Plot divisive normalisation:
plt_dpth_prfl(vecNormDiv02.reshape(1, varNumDpth),
              aryError,
              varNumDpth,
              1,
              (varDpi * 1.8),
              0.0,
              15.0,
              False,
              ['1', '2'],
              strXlabel,
              'Ratio',
              '',
              False,
              (strPthOt + 'scenario_02_div' + strFleTyp),
              tplPadY=(0.1, 0.1),
              varNumLblY=4,
              varRound=0)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create plots - Scenario 3

# Plot components without draining vein effect:
plt_dpth_prfl(np.array((vecSin03a, vecSin03b)),
              aryError,
              varNumDpth,
              varNumCon,
              (varDpi * 1.8),
              -0.3,
              0.3,
              False,
              ['1', '2'],
              strXlabel,
              strYlabel,
              '',
              False,
              (strPthOt + 'scenario_03_components' + strFleTyp),
              tplPadY=(0.01, 0.01),
              varNumLblY=3,
              varRound=0)

# Plot hypothetical fMRI signal profile (with draining vein effect):
plt_dpth_prfl(arySin03,
              aryError,
              varNumDpth,
              varNumCon,
              (varDpi * 1.8),
              -4.0,
              4.0,
              False,
              ['1', '2'],
              strXlabel,
              strYlabel,
              '',
              False,
              (strPthOt + 'scenario_03_fMRI' + strFleTyp),
              tplPadY=(0.15, 0.15),
              varNumLblY=5,
              varRound=0)

# Plot subtractive normalisation:
plt_dpth_prfl(vecNormSub03.reshape(1, varNumDpth),
              aryError,
              varNumDpth,
              1,
              (varDpi * 1.8),
              0.0,
              8.0,
              False,
              ['1', '2'],
              strXlabel,
              'Difference',
              '',
              False,
              (strPthOt + 'scenario_03_sub' + strFleTyp),
              tplPadY=(0.15, 0.15),
              varNumLblY=5,
              varRound=0)

# Plot divisive normalisation:
plt_dpth_prfl(vecNormDiv03.reshape(1, varNumDpth),
              aryError,
              varNumDpth,
              1,
              (varDpi * 1.8),
              -1.0,
              1.0,
              False,
              ['1', '2'],
              strXlabel,
              'Ratio',
              '',
              False,
              (strPthOt + 'scenario_03_div' + strFleTyp),
              tplPadY=(0.1, 0.1),
              varNumLblY=3,
              varRound=0)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Plot convolution term

plt_dpth_prfl(vecDcy.reshape(1, varNumDpth),
              aryError,
              varNumDpth,
              1,
              (varDpi * 1.8),
              0.0,
              1.0,
              False,
              ['1'],
              strXlabel,
              'Signal spread',
              '',
              False,
              (strPthOt + 'convolution_term' + strFleTyp),
              tplPadY=(0.1, 0.1),
              varNumLblY=3,
              varRound=0)
# -----------------------------------------------------------------------------
