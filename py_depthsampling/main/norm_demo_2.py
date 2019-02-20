
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
from scipy.signal import fftconvolve
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter1d
from py_depthsampling.plot.plt_dpth_prfl import plt_dpth_prfl
from py_depthsampling.drain_model.drain_model_decon_01 import deconv_01


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

# Output folder:
strPthOt = '/home/john/Dropbox/Thesis/Chapters/General_Discussion/Figures/Figure_5_1_Normalisation/Version_02_exp/elements/'  #noqa

# Figure output file type:
strFleTyp = '.svg'
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create profile templates

# Linear datapoints (not all the way to zero to avoid division by zero):
vecLin = np.linspace(1.0, 0.0, varNumDpth, endpoint=False)

# Decay term:
# vecDcy = np.power(vecLin, 1.0)
vecDcy = np.exp(vecLin)
# vecDcy = np.subtract(vecDcy, 1.0)
# vecDcy = np.divide(vecDcy, (np.exp(1.0) - 1.0))

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
aryScn01 = np.array((vecSin01a, vecSin01b))
# Keep copy original array:
aryScnCpy01 = np.copy(aryScn01)

# (2)
# Scenario 2: One positive activation peak at mid-grey matter, no activation in
# control condition.
vecSin02a = np.add(np.multiply(arySin, 0.0), 0.01)
vecSin02b = np.add(np.multiply(arySin, 1.0), 0.01)
aryScn02 = np.array((vecSin02a, vecSin02b))
# Keep copy original array:
aryScnCpy02 = np.copy(aryScn02)

# (3)
# Scenario 3: Positive and negative activation peaks.
vecSin03a = np.add(np.multiply(arySin, -1.0), -0.01)
vecSin03b = np.add(np.multiply(arySin, 1.0), 0.01)
aryScn03 = np.array((vecSin03a, vecSin03b))
# Keep copy original array:
aryScnCpy03 = np.copy(aryScn03)

# List of original arrays (needed for plots):
lstScnCpy = [aryScnCpy01, aryScnCpy02, aryScnCpy03]

# Number of conditions:
varNumCon = 2

# List for results from scenarios:
lstScn = [aryScn01, aryScn02, aryScn03]

# Number of scenarios:
varNumScn = len(lstScn)

# Subtraction of original model activation ("ground truth"):
aryTrth = np.zeros((varNumScn, varNumDpth))
for idxSnc in range(varNumScn):
    aryTrth[idxSnc, :] = np.subtract(lstScn[idxSnc][1, :],
                                     lstScn[idxSnc][0, :])

# Scale "truth":
aryTrth = np.multiply(aryTrth, 15.0)

# Model draining vein effect:
for idxSnc in range(varNumScn):
    for idxCon in range(varNumCon):
        lstScn[idxSnc][idxCon, :] = fftconvolve(lstScn[idxSnc][idxCon, :],
                                                vecDcy,
                                                mode='full')[0:varNumDpth]
        lstScn[idxSnc][idxCon, :] = np.multiply(lstScn[idxSnc][idxCon, :],
                                                0.3)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Normalisation by subtraction

# List for results:
lstSub = [None] * varNumScn

for idxSnc in range(varNumScn):
    lstSub[idxSnc] = np.subtract(lstScn[idxSnc][1, :], lstScn[idxSnc][0, :])
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Normalisation by division

# List for results:
lstDiv = [None] * varNumScn

for idxSnc in range(varNumScn):
    lstDiv[idxSnc] = np.divide(lstScn[idxSnc][1, :], lstScn[idxSnc][0, :])
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Normalised difference
#
# vecNormDiv01 = np.divide(
#                          np.subtract(arySin01[1, :], arySin01[0, :]),
#                          np.add(arySin01[1, :], arySin01[0, :])
#                          )
#
# vecNormDiv02 = np.divide(
#                          np.subtract(arySin02[1, :], arySin02[0, :]),
#                          np.add(arySin02[1, :], arySin02[0, :])
#                          )
#
# vecNormDiv03 = np.divide(
#                          np.subtract(arySin03[1, :], arySin03[0, :]),
#                          np.add(arySin03[1, :], arySin03[0, :])
#                          )
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Spatial deconvolution

# Vector for downsampled empirical depth profiles:
aryEmp5 = np.zeros((varNumScn, varNumCon, 5))

# Position of sample points (for spatial interpolation, because deconvolution
# is defined at a limited number of depth levels):
vecPosMdl = np.linspace(0.0, 1.0, num=5)
vecPosEmp = np.linspace(0.0, 1.0, num=varNumDpth)

# Loop through conditions and downsample the depth profiles:
for idxSnc in range(varNumScn):
    for idxCon in range(varNumCon):
        # Interpolation:
        aryEmp5[idxSnc, idxCon, :] = griddata(vecPosEmp,
                                              lstScn[idxSnc][idxCon, :],
                                              vecPosMdl,
                                              method='linear')

# Array for deconvolution result in model space (5 depth levels):
aryDecon5 = np.zeros((varNumScn, varNumCon, 5))

# Deconvolution based on Markuerkiaga et al. (2016):
for idxSnc in range(varNumScn):
    aryDecon5[idxSnc, :, :] = deconv_01(varNumCon, aryEmp5[idxSnc, :, :])

# Array for deconvolution results:
aryDecon = np.zeros((varNumScn, varNumCon, varNumDpth))

# Interpolation back into equi-volume space:
for idxSnc in range(varNumScn):
    for idxCon in range(varNumCon):
        # Interpolation:
        aryDecon[idxSnc, idxCon, :] = griddata(vecPosMdl,
                                               aryDecon5[idxSnc, idxCon, :],
                                               vecPosEmp,
                                               method='linear')

aryDecon = gaussian_filter1d(aryDecon,
                             (0.1 * float(varNumDpth)),
                             axis=2,
                             mode='nearest')
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Subtraction on deconvolved profiles

aryDeconSub = np.subtract(aryDecon[:, 1, :], aryDecon[:, 0, :])
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
# *** Create plots

for idxScn in range(varNumScn):

    # Layout, depending on scenario:
    if (idxScn == 0) or (idxScn == 1):
        varMinY = 0.0
        varMaxY = 0.3
        varNumLblY = 4
    elif idxScn == 2:
        varMinY = -0.3
        varMaxY = 0.3
        varNumLblY = 3

    # Plot components without draining vein effect:
    plt_dpth_prfl(lstScnCpy[idxScn],
                  aryError,
                  varNumDpth,
                  varNumCon,
                  (varDpi * 1.8),
                  varMinY,
                  varMaxY,
                  False,
                  ['1', '2'],
                  strXlabel,
                  strYlabel,
                  '',
                  False,
                  (strPthOt
                   + 'scenario_'
                   + str(idxScn).zfill(2)
                   + '_components'
                   + strFleTyp),
                  tplPadY=(0.01, 0.01),
                  varNumLblY=varNumLblY,
                  varRound=1)

    # Layout, depending on scenario:
    if (idxScn == 0) or (idxScn == 1):
        varMinY = 0.0
        varMaxY = 4.0
        varNumLblY = 5
    elif idxScn == 2:
        varMinY = -4.0
        varMaxY = 4.0
        varNumLblY = 5

    # Plot hypothetical fMRI signal profile (with draining vein effect):
    plt_dpth_prfl(lstScn[idxScn],
                  aryError,
                  varNumDpth,
                  varNumCon,
                  (varDpi * 1.8),
                  varMinY,
                  varMaxY,
                  False,
                  ['1', '2'],
                  strXlabel,
                  strYlabel,
                  '',
                  False,
                  (strPthOt
                   + 'scenario_'
                   + str(idxScn).zfill(2)
                   + '_fMRI'
                   + strFleTyp),
                  tplPadY=(0.1, 0.1),
                  varNumLblY=varNumLblY,
                  varRound=0)

    # Layout, depending on scenario:
    if idxScn == 0:
        varMinY = 0.0
        varMaxY = 2.0
        varNumLblY = 3
        tplPadY = (0.1, 0.1)
    elif idxScn == 1:
        varMinY = 0.0
        varMaxY = 4.0
        varNumLblY = 5
        tplPadY = (0.2, 0.1)
    elif idxScn == 2:
        varMinY = 0.0
        varMaxY = 8.0
        varNumLblY = 5
        tplPadY = (0.15, 0.15)

    # Plot subtractive normalisation:
    plt_dpth_prfl(lstSub[idxScn].reshape(1, varNumDpth),
                  aryError,
                  varNumDpth,
                  1,
                  (varDpi * 1.8),
                  varMinY,
                  varMaxY,
                  False,
                  ['1'],
                  strXlabel,
                  'Difference',
                  '',
                  False,
                  (strPthOt
                   + 'scenario_'
                   + str(idxScn).zfill(2)
                   + '_sub'
                   + strFleTyp),
                  tplPadY=tplPadY,
                  varNumLblY=varNumLblY,
                  varRound=0)

    # Layout, depending on scenario:
    if idxScn == 0:
        varMinY = 0.0
        varMaxY = 2.0
        varNumLblY = 3
        tplPadY = (0.1, 0.1)
    elif idxScn == 1:
        varMinY = 0.0
        varMaxY = 10.0
        varNumLblY = 3
        tplPadY = (0.1, 2.0)
    elif idxScn == 2:
        varMinY = -1.0
        varMaxY = 1.0
        varNumLblY = 3
        tplPadY = (0.1, 0.1)

    # Plot divisive normalisation:
    plt_dpth_prfl(lstDiv[idxScn].reshape(1, varNumDpth),
                  aryError,
                  varNumDpth,
                  1,
                  (varDpi * 1.8),
                  varMinY,
                  varMaxY,
                  False,
                  ['1'],
                  strXlabel,
                  'Ratio',
                  '',
                  False,
                  (strPthOt
                   + 'scenario_'
                   + str(idxScn).zfill(2)
                   + '_div'
                   + strFleTyp),
                  tplPadY=tplPadY,
                  varNumLblY=varNumLblY,
                  varRound=0)

    # Layout, depending on scenario:
    if idxScn == 0:
        varMinY = 0.0
        varMaxY = 1.0
        varNumLblY = 2
        tplPadY = (0.1, 0.3)
    elif idxScn == 1:
        varMinY = 0.0
        varMaxY = 2.0
        varNumLblY = 3
        tplPadY = (0.1, 0.5)
    elif idxScn == 2:
        varMinY = 0.0
        varMaxY = 5.0
        varNumLblY = 2
        tplPadY = (0.1, 0.6)

    # Plot subtraction of deconvolved profiles:
    plt_dpth_prfl(aryDeconSub[idxScn, :].reshape(1, varNumDpth),
                  aryError,
                  varNumDpth,
                  1,
                  (varDpi * 1.8),
                  varMinY,
                  varMaxY,
                  False,
                  ['1'],
                  strXlabel,
                  'Difference',
                  '',
                  False,
                  (strPthOt
                   + 'scenario_'
                   + str(idxScn).zfill(2)
                   + '_deconv_sub'
                   + strFleTyp),
                  tplPadY=tplPadY,
                  varNumLblY=varNumLblY,
                  varRound=0)

    # Layout, depending on scenario:
    if idxScn == 0:
        varMinY = 0.0
        varMaxY = 2.0
        varNumLblY = 3
        tplPadY = (0.1, 0.1)
    elif idxScn == 1:
        varMinY = 0.0
        varMaxY = 4.0
        varNumLblY = 5
        tplPadY = (0.2, 0.1)
    elif idxScn == 2:
        varMinY = 0.0
        varMaxY = 8.0
        varNumLblY = 5
        tplPadY = (0.15, 0.15)

    # Plot "ground truth":
    plt_dpth_prfl(aryTrth[idxScn, :].reshape(1, varNumDpth),
                  aryError,
                  varNumDpth,
                  1,
                  (varDpi * 1.8),
                  varMinY,
                  varMaxY,
                  False,
                  ['1'],
                  strXlabel,
                  'Difference',
                  '',
                  False,
                  (strPthOt
                   + 'scenario_'
                   + str(idxScn).zfill(2)
                   + '_truth'
                   + strFleTyp),
                  tplPadY=tplPadY,
                  varNumLblY=varNumLblY,
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
              3.0,
              False,
              ['1'],
              strXlabel,
              'Signal spread',
              '',
              False,
              (strPthOt + 'convolution_term' + strFleTyp),
              tplPadY=(0.1, 0.1),
              varNumLblY=4,
              varRound=0)
# -----------------------------------------------------------------------------
