# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

This version of the script removes only the draining effect. It does not
account for different neuronal-to-fMRI-signal coupling in different layers.

In other words, if the neuronal signal at each layer is the same, this would
result in different fMRI signal strength at each layer even without the
draining effect, according to the model proposed by Markuerkiaga et al.
(2016). This version of the script account only for the draining effect.

The purpose of this script is to remove the contribution of lower cortical
depth levels to the signal at each consecutive depth level. In other words,
at a given depth level, the contribution from lower depth levels is removed
based on the model proposed by Markuerkiaga et al. (2016).

The following data from Markuerkiaga et al. (2016) is used in this script:

    "The cortical layer boundaries of human V1 in the model were fixed
    following de Sousa et al. (2010) and Burkhalter and Bernardo (1989):
    layer VI, 20%;
    layer V, 10%;
    layer IV, 40%;
    layer II/III, 20%;
    layer I, 10%
    (values rounded to the closest multiple of 10)." (p. 492)

Following the model by Markuerkiaga et al. (2016), the fMRI signal for each
cortical layer for a GE sequence  is corrected for a draining effect as
follows:

    Layer VI:

        varCrctVI = varEmpVI

    Layer V:

        varCrctV = varEmpV
                   - (0.6 / 1.9) * varCrctVI

    Layer IV:

        varCrctIV = varEmpIV
                    - (0.3 / 1.5) * varCrctV
                    - (0.6 / 1.9) * varCrctVI

    Layer II/III:

        varCrctII_III = varEmpII_III
                        - (1.3 / 2.2) * varCrctIV
                        - (0.3 / 1.5) * varCrctV
                        - (0.5 / 1.9) * varCrctVI) 

    Layer I:

        varCrctI = varEmpI
                   - (0.7 / 1.7) * varCrctII_III
                   - (1.3 / 2.2) * varCrctIV
                   - (0.3 / 1.5) * varCrctV
                   - (0.5 / 1.9) * varCrctVI

Reference:
Markuerkiaga, I., Barth, M., & Norris, D. G. (2016). A cortical vascular model
    for examining the specificity of the laminar BOLD signal. Neuroimage, 132,
    491-498.

@author: Ingo Marquardt, 21.03.2017
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
from ds_pltAcrSubsMean import funcPltAcrSubsMean


# ----------------------------------------------------------------------------
# *** Define parameters

# Path of depth-profile to correct:
strPthPrf = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy'

# Output path for corrected depth-profiles:
strPthPrfOt = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1_corrected.npy'

# Output path & prefix for plots:
strPthPltOt = '/home/john/PhD/Tex/deconv/tex_deconv_v1_reduced/deconvolution_'

# File type suffix for plot:
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Limits of y-axis for across subject plot:
varAcrSubsYmin01 = -0.05
varAcrSubsYmax01 = 2.0
varAcrSubsYmin02 = -0.05
varAcrSubsYmax02 = 2.0

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [arbitrary units]'

# Condition labels:
lstConLbl = ['2.5%', '6.1%', '16.3%', '72.0%']


# ----------------------------------------------------------------------------
# *** Load depth profile from disk

# Array with single-subject depth sampling results, of the form
# aryEmpSnSb[idxSub, idxCondition, idxDpth].
aryEmpSnSb = np.load(strPthPrf)

#aryEmpSnSb = np.zeros((1, 3, 5))
#aryEmpSnSb[0, 0, :] = np.add(np.array([1.0, 1.316, 1.516, 2.054, 2.466]),
#                             -0.5)
#aryEmpSnSb[0, 1, :] = np.array([1.0, 1.316, 1.516, 2.054, 2.466])
#aryEmpSnSb[0, 2, :] = np.add(np.array([1.0, 1.316, 1.516, 2.054, 2.466]),
#                             0.5)

# Number of subjects:
varNumSub = aryEmpSnSb.shape[0]

# Number of conditions:
varNumCon = aryEmpSnSb.shape[1]

# Number of equi-volume depth levels in the input data:
varNumDpth = aryEmpSnSb.shape[2]


# ----------------------------------------------------------------------------
# *** Loop through subjects

# Array for single-subject interpolation result (before deconvolution):
aryEmp5SnSb = np.zeros((varNumSub, varNumCon, 5))

# Array for single-subject deconvolution result:
aryNrnSnSb = np.zeros((varNumSub, varNumCon, 5))

for idxSub in range(0, varNumSub):


    # ----------------------------------------------------------------------------
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
    # vecPosEmp = np.array([0.1, 0.25, 0.5, 0.8, 0.95])
    
    # Vector for downsampled empirical depth profiles:
    aryEmp5 = np.zeros((varNumCon, 5))
    
    # Loop through conditions and downsample the depth profiles:
    for idxCon in range(0, varNumCon):
        # Interpolation:
        aryEmp5[idxCon] = griddata(vecPosEmp,
                                   aryEmpSnSb[idxSub, idxCon, :],
                                   vecPosMdl,
                                   method='linear')

    # Put interpolation result for this subject into the array:
    aryEmp5SnSb[idxSub, :, :] = np.copy(aryEmp5)


    # ----------------------------------------------------------------------------
    # *** Subtraction of draining effect
    
    for idxCon in range(0, varNumCon):

        # Array for corrected depth profiles:
        aryNrn = np.zeros(aryEmp5SnSb[idxSub, :, :].shape)

        # Layer VI:
        aryNrn[idxCon, 0] = aryEmp5SnSb[idxSub, idxCon, 0]

        # Layer V:
        aryNrn[idxCon, 1] = (aryEmp5SnSb[idxSub, idxCon, 1]
                             - (0.6 / 1.9) * aryNrn[idxCon, 0])

        # Layer IV:
        aryNrn[idxCon, 2] = (aryEmp5SnSb[idxSub, idxCon, 2]
                             - (0.3 / 1.5) * aryNrn[idxCon, 1]
                             - (0.6 / 1.9) * aryNrn[idxCon, 0])

        # Layer II/III:
        aryNrn[idxCon, 3] = (aryEmp5SnSb[idxSub, idxCon, 3]
                             - (1.3 / 2.2) * aryNrn[idxCon, 2]
                             - (0.3 / 1.5) * aryNrn[idxCon, 1]
                             - (0.5 / 1.9) * aryNrn[idxCon, 0])

        # Layer I:
        aryNrn[idxCon, 4] = (aryEmp5SnSb[idxSub, idxCon, 4]
                             - (0.7 / 1.7) * aryNrn[idxCon, 3]
                             - (1.3 / 2.2) * aryNrn[idxCon, 2]
                             - (0.3 / 1.5) * aryNrn[idxCon, 1]
                             - (0.5 / 1.9) * aryNrn[idxCon, 0])

        # Put deconvolution result for this subject into the array:
        aryNrnSnSb[idxSub, idxCon, :] = np.copy(aryNrn[idxCon, :])


    # ----------------------------------------------------------------------------
    # *** Normalisation

    # Calculate 'grand mean', i.e. the mean PE across depth levels and
    # conditions:
    # varGrndMean = np.mean(aryNrnSnSb[idxSub, :, :])

    # Divide all values by the grand mean:
    # aryNrnSnSb[idxSub, :, :] = np.divide(aryNrnSnSb[idxSub, :, :],
    #                                      varGrndMean)


# ----------------------------------------------------------------------------
# *** Save corrected depth profiles

# Save array with single-subject corrected depth profiles, of the form
# aryNrnSnSb[idxSub, idxCondition, idxDpth].
np.save(strPthPrfOt,
        aryNrnSnSb)


# ----------------------------------------------------------------------------
# *** Plot results

# Plot across-subjects mean before deconvolution:
strTmpTtl = 'Before deconvolution'
strTmpPth = (strPthPltOt + 'before')
funcPltAcrSubsMean(aryEmp5SnSb,
                   varNumSub,
                   5,
                   varNumCon,
                   varDpi,
                   varAcrSubsYmin01,
                   varAcrSubsYmax01,
                   lstConLbl,
                   strXlabel,
                   strYlabel,
                   strTmpTtl,
                   strTmpPth,
                   strFlTp)


# Across-subjects mean after deconvolution:
strTmpTtl = 'After deconvolution'
strTmpPth = (strPthPltOt + 'after')
funcPltAcrSubsMean(aryNrnSnSb,
                   varNumSub,
                   5,
                   varNumCon,
                   varDpi,
                   varAcrSubsYmin02,
                   varAcrSubsYmax02,
                   lstConLbl,
                   strXlabel,
                   strYlabel,
                   strTmpTtl,
                   strTmpPth,
                   strFlTp)

#aryEmp5 = np.mean(aryEmp5SnSb, axis=0).T
#aryNrn = np.mean(aryNrnSnSb, axis=0).T

