# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

Notes
-----

The purpose of this script is to apply the CRF fitting function to the data
from Tootell et al. (1988), p. 1602, Figure 6. In brief, tracer uptake in
response to prolonged exposure to visual stimuli at four different different
luminance contrast levels (0.08, 0.18, 0.4, 1.0) was measured in layers 3, 4A,
4B, 4Ca, 4Cb, 5, and 6 of Macaque V1. Normalised uptake was (scaled by a factor
of 1000 with respect to original data):

Layer 3:   [70.0, 210.0, 560.0, 570.0]
Layer 4A:  [47.0, 285.0, 550.0, 680.0]
Layer 4B:  [288.0, 341.0, 633.0, 582.0]
Layer 4Ca: [435.0, 461.0, 622.0, 768.0]
Layer 4Cb: [8.0, 117.0, 492.0, 819.0]
Layer 5:   [81.0, 239.0, 432.0, 475.0]
Layer 6:   [455.0, 636.0, 859.0, 996.0]


The profile of semisaturation contrast is plotted with respect to the relative
thickness of layers derived from Hawken et al. 1988, p. 3544, Figure 4. The
relative position of centre of each layer, from 0 at WM/GM interface to 1000
at GM/CSF interface, are:

Layer 6:   55.0
Layer 5:   170.0
Layer 4Cb: 285.0
Layer 4Ca: 362.0
Layer 4B:  450.0
Layer 4A:  530.0
Layer 3:   670.0
Layer 2:   890.0


References
----------
Hawken, M. J., Parker, A. J., & Lund, J. S. (1988). Laminar organization and
    contrast sensitivity of direction-selective cells in the striate cortex of
    the Old World monkey. The Journal of Neuroscience, 8(10), 3541–3548.
Tootell, R. B., Hamilton, S. L., & Switkes, E. (1988). Functional anatomy of
    macaque striate cortex. IV. Contrast and magno-parvo streams. The Journal
    of Neuroscience, 8(5), 1594–1609.
"""

import numpy as np
from ds_crfFit import crf_fit
from ds_pltAcrDpth import funcPltAcrDpth

print('-CRF fitting on data from Tootell et al. (1988)')

# Stimulus contrast levels (independent variable):
vecEmpX = np.array([0.08, 0.18, 0.4, 1.0])

# Number of empirical contrast levels:
varNumConEmp = vecEmpX.shape[0]

# Tracer uptake (response, dependent variable):
lstEmpY = [np.array([70.0, 210.0, 560.0, 570.0]),  # Layer 3
           np.array([47.0, 285.0, 550.0, 680.0]),  # Layer 4A
           np.array([288.0, 341.0, 633.0, 582.0]),  # Layer 4B
           np.array([435.0, 461.0, 622.0, 768.0]),  # Layer 4Ca
           np.array([8.0, 117.0, 492.0, 819.0]),  # Layer 4Cb
           np.array([81.0, 239.0, 432.0, 475.0]),  # Layer 5
           np.array([455.0, 636.0, 859.0, 996.0])]  # Layer 6

# Relative position of layers:
lstLayPos = [670.0,  # Layer 3
             530.0,  # Layer 4A
             450.0,  # Layer 4B
             362.0,  # Layer 4Ca
             285.0,  # Layer 4Cb
             170.0,  # Layer 5
             55.0]  # Layer 6

# Get number of layers:
#varNumLayers = len(dictEmpY.keys())
varNumLayers = len(lstLayPos)

# Number of x-values for which to solve the function when calculating model
# fit:
varNumX=1000

# Array for modelled y values, of the form aryMdlY[idxLayer, idxContrast]:
aryMdlY = np.zeros((varNumLayers, varNumX))

# Vector for response at 50% contrast:
vecHlfMax = np.zeros((varNumLayers))

# Vector for semisaturation contrast:
vecSemi = np.zeros((varNumLayers))

# Vector for model residuals, of the form aryRes[idxLayer, idxContrast]:
aryRes = np.zeros((varNumLayers, varNumConEmp))

# Vector for relative laminar depth of results (for plot):
vecLayPos = np.zeros((varNumLayers))

# Loop through layers & fit function:
for idxLay in range(varNumLayers):

    # Reshape to fit expected function input:
    vecEmpY = lstEmpY[idxLay].reshape((1, varNumConEmp))

    # Rescale values:
    vecEmpY = np.multiply(vecEmpY, 0.001)

    aryMdlY[idxLay, :], vecHlfMax[idxLay], vecSemi[idxLay], \
        aryRes[idxLay, :] = crf_fit(vecEmpX, vecEmpY,
        strFunc='power', varNumX=varNumX)

    # Put relative laminar position of current layer into vector (for
    # plotting):
    vecLayPos[idxLay] = lstLayPos[idxLay]

# Rescale laminar position (to fit with plot settings):
vecLayPos = np.multiply(vecLayPos, 6.0)
vecLayPos = np.divide(vecLayPos, 1000.0)

# Reshape semisaturation contrast array for plot:
vecSemi = vecSemi.reshape((1, varNumLayers))

# Dummy array for error bars:
aryError = np.zeros(vecSemi.shape)

# Path for figure:
strPath = '/home/john/Desktop/semiTootell.png'

# Plot semisaturation contrast:
funcPltAcrDpth(vecSemi, aryError, varNumLayers, 1, 80.0, 0.0, 0.5, True,
               [''], 'Cortical depth', 'Percent luminance contrast',
               'Semisaturation contrast', False, strPath, vecX=vecLayPos,
               varNumLblY=6)
