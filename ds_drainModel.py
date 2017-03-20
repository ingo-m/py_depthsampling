# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

The purpose of this script is to remove the contribution of lower cortical
depth levels to the signal at each consecutive depth level. In other words,
at a given depth level, the contribution from lower depth levels is removed
based on the model proposed by Markuerkiaga et al. (2016).

The following data from Markuerkiaga et al. (2016) is used in this script:

    'The cortical layer boundaries of human V1 in the model were fixed
    following de Sousa et al. (2010) and Burkhalter and Bernardo (1989):
    layer VI, 20%;
    layer V, 10%;
    layer IV, 40%;
    layer II/III, 20%;
    layer I, 10%
    (values rounded to the closest multiple of 10).' (p. 492)

Moreover, the absolute (?) signal contribution in each layer for a GE sequence
as depicted in Figure 3F (p. 495):

    Layer VI:
        1.9 * var6
    Layer V:
        (1.5 * var 5)
        + ((2.1 - 1.5) * var6)
    Layer IV:
        (2.2 * var4)
        + ((2.5 - 2.2) * var5)
        + ((3.1 - 2.5) * var6)
    Layer II/III:
        (1.7 * var23)
        + ((3.0 - 1.7) * var4)
        + ((3.3 - 3.0) * var5)
        + ((3.8 - 3.3) * var6)
    Layer I:
        (1.6* var1)
        + ((2.3 - 1.6) * var23)
        + ((3.6 - 2.3) * var4)
        + ((3.9 - 3.6) * var5)
        + ((4.4 - 3.9) * var6)

These values are translated into the a transfer function of the effect of
neuronal activity at each depth layer on the haemodynamic response at all
layers. There is one vector for each depth level, each entry gives the effect
of one unit of neural activity at that depth level on the haemodynamic signal
at all depth levels.

    Layer VI:
    vecTrsf6 = np.array([1.9, 0.6, 0.6, 0.5, 0.5])

    Layer V:
    vecTrsf5 = np.array([0.0, 1.5, 0.3, 0.3, 0.3])

    Layer IV:
    vecTrsf4 = np.array([0.0, 0.0, 2.2, 1.3, 1.3])

    Layer II/III:
    vecTrsf23 = np.array([0.0, 0.0, 0.0, 1.7, 0.7])

    Layer I:
    vecTrsf23 = np.array([0.0, 0.0, 0.0, 0.0, 1.6])

Multiplying these vectors with the local neuronal activity in each layer, and
adding up the result gives the predicted haemodynamic signal (forward model).

Reference:
Markuerkiaga, I., Barth, M., & Norris, D. G. (2016). A cortical vascular model
    for examining the specificity of the laminar BOLD signal. Neuroimage, 132,
    491-498.

@author: Ingo Marquardt, 16.03.2017
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
from scipy.interpolate import griddata


# *** Define parameters

# Number of equi-volume depth levels in the input data:
varNumDpth = 11

# Path of depth-profile to correct:
strPthPrf = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy'


# *** Draining model (Markuerkiaga et al. 2016)

# Forward model (neuronal activity --> fMRI signal):

# The following vectors contain the predicted fMRI signal at each depth level
# if the neuronal activity at each depth level is equal (one).

## Layer VI:
#vecTrsfVI = np.array([1.9, 0.6, 0.6, 0.5, 0.5])
## Layer V:
#vecTrsfV = np.array([0.0, 1.5, 0.3, 0.3, 0.3])
## Layer IV:
#vecTrsfIV = np.array([0.0, 0.0, 2.2, 1.3, 1.3])
## Layer II/III:
#vecTrsfII_III = np.array([0.0, 0.0, 0.0, 1.7, 0.7])
## Layer I:
#vecTrsfI = np.array([0.0, 0.0, 0.0, 0.0, 1.6])
## Stacking the transfer functions:
#aryTrsf = np.vstack((vecTrsfVI, vecTrsfV, vecTrsfIV, vecTrsfII_III, vecTrsfI))

# Layer VI:
vecTrsfVI = np.array([0.0, 0.6, 0.6, 0.5, 0.5])
varTrsfVI = 1.9
vecTrsfVI = np.divide(vecTrsfVI, varTrsfVI)

# Layer V:
vecTrsfV = np.array([0.0, 0.0, 0.3, 0.3, 0.3])
varTrsfV = 1.5
vecTrsfV = np.divide(vecTrsfV, varTrsfV)

# Layer IV:
vecTrsfIV = np.array([0.0, 0.0, 0.0, 1.3, 1.3])
varTrsfIV = 2.2
vecTrsfIV = np.divide(vecTrsfIV, varTrsfIV)

# Layer II/III:
vecTrsfII_III = np.array([0.0, 0.0, 0.0, 0.0, 0.7])
varTrsfII_III = 1.7
vecTrsfII_III = np.divide(vecTrsfII_III, varTrsfII_III)

# Layer I:
vecTrsfI = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
varTrsfI = 1.6
# vecTrsfI = np.divide(vecTrsfI, varTrsfI)

# Stacking the transfer functions:
aryTrsf = np.vstack((vecTrsfVI, vecTrsfV, vecTrsfIV, vecTrsfII_III, vecTrsfI))
lstTrsf = [varTrsfVI, varTrsfV, varTrsfIV, varTrsfII_III, varTrsfI]


# *** Load depth profile from disk

# Array with single-subject depth sampling results, of the form
# arySubDpthMns[idxSub, idxCondition, idxDpth].
aryDpthSnSb = np.load(strPthPrf)

# Across-subjects mean:
aryDpth = np.mean(aryDpthSnSb, axis=0)

# Number of conditions:
varNumCon = aryDpth.shape[0]


# *** Interpolation (downsampling)

# The empirical depth profiles are defined at more depth levels than the
# draining model. We downsample the empirical depth profiles to the number of
# depth levels of the model.

# Relative thickness of the layers (layer VI, 20%; layer V, 10%; layer IV,
# 40%; layer II/III, 20%; layer I, 10%; Markuerkiaga et al. 2016).
# lstThck = [0.2, 0.1, 0.4, 0.2, 0.1]
# From the relative thickness, we derive the relative position of the layers
# (we set the position of each layer to the sum of all lower layers plus half
# its own thickness):
vecPosMdl = np.array([0.1, 0.25, 0.5, 0.8, 0.95])

# Position of empirical datapoints:
vecPosEmp = np.linspace(0.0, 1.0, num=varNumDpth, endpoint=True)

aryDpth5 = np.zeros((varNumCon, 5))

for idxCon in range(0, varNumCon):

    # The actual itnerpolation:
    aryDpth5[idxCon] = griddata(vecPosEmp,
                                aryDpth[idxCon, :],
                                vecPosMdl,
                                method='linear')

# Array for corrected depth profiles:
aryCrct = np.copy(aryDpth5)

for idxCon in range(0, varNumCon):

    aryDrain = np.zeros(aryTrsf.shape)
    
    for idxDpth in range(0, 5):
        
        aryDrain = np.multiply(aryTrsf[idxDpth, :],
                               aryDpth5[idxCon, idxDpth])
        
        print(aryDrain)

        aryCrct[idxCon, :] = np.subtract(aryCrct[idxCon, :],
                                         aryDrain)

    for idxDpth in range(0, 5):

        aryCrct[idxCon, idxDpth] = np.divide(aryCrct[idxCon, idxDpth],
                                             lstTrsf[idxDpth])

aa1 = aryDpth5.T
aa2 = aryCrct.T

print('lala')