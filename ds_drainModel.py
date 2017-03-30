# -*- coding: utf-8 -*-
"""
**Model-based correction of draining effect.**

Function of the depth sampling pipeline.

Notes
-----

The purpose of this script is to remove the contribution of lower cortical
depth levels to the signal at each consecutive depth level. In other words,
at a given depth level, the contribution from lower depth levels is removed
based on the model proposed by Markuerkiaga et al. (2016).

The correction for the draining effect is done in a function called by this
script. There are three different option for correction (see respective
functions for details):

(1) Only correct draining effect (based on model by Markuerkiaga et al., 2016).

(2) Correct draining effect (based on model by Markuerkiaga et al., 2016) &
    perform scaling to account for different vascular density and/or
    haemodynamic coupling between depth levels based on model by Markuerkiaga
    et al. (2016).

(3) Correct draining effect (based on model by Markuerkiaga et al., 2016) &
    perform scaling to account for different vascular density and/or
    haemodynamic coupling between depth levels based on data by Weber et al.
    (2008). This option allows for different correction for V1 & extrastriate
    cortex.

The following data from Markuerkiaga et al. (2016) is used in this script,
irrespective of which draining effect model is choosen:

    "The cortical layer boundaries of human V1 in the model were fixed
    following de Sousa et al. (2010) and Burkhalter and Bernardo (1989):
    layer VI, 20%;
    layer V, 10%;
    layer IV, 40%;
    layer II/III, 20%;
    layer I, 10%
    (values rounded to the closest multiple of 10)." (p. 492)

References
----------
Markuerkiaga, I., Barth, M., & Norris, D. G. (2016). A cortical vascular model
for examining the specificity of the laminar BOLD signal. Neuroimage, 132,
491-498.

Weber, B., Keller, A. L., Reichold, J., & Logothetis, N. K. (2008). The
microvascular system of the striate and extrastriate visual cortex of the
macaque. Cerebral Cortex, 18(10), 2318-2330.
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
from ds_drainModelDecon01 import depth_deconv_01
from ds_drainModelDecon02 import depth_deconv_02
from ds_drainModelDecon03 import depth_deconv_03

# ----------------------------------------------------------------------------
# *** Define parameters

# Which draining model to use (1, 2, or 3 - see above for details):
varMdl = 3

# ROI (V1 or V2):
strRoi = 'v2'

# Path of depth-profile to correct:
strPthPrf = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/{}.npy'.format(strRoi)

# Output path for corrected depth-profiles:
strPthPrfOt = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/{}_corrected.npy'.format(strRoi)

# Output path & prefix for plots:
strPthPltOt = '/home/john/PhD/Tex/deconv/tex_deconv_{}_model_{}/deconv_{}_m{}_'.format(strRoi, str(varMdl), strRoi, str(varMdl))

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

print('-Model-based correction of draining effect')

print('---Loading data')

# Array with single-subject depth sampling results, of the form
# aryEmpSnSb[idxSub, idxCondition, idxDpth].
aryEmpSnSb = np.load(strPthPrf)

# Number of subjects:
varNumSub = aryEmpSnSb.shape[0]

# Number of conditions:
varNumCon = aryEmpSnSb.shape[1]

# Number of equi-volume depth levels in the input data:
varNumDpth = aryEmpSnSb.shape[2]


# ----------------------------------------------------------------------------
# *** Loop through subjects

print('---Subject-by-subject deconvolution')

# Array for single-subject interpolation result (before deconvolution):
aryEmp5SnSb = np.zeros((varNumSub, varNumCon, 5))

# Array for single-subject deconvolution result:
aryNrnSnSb = np.zeros((varNumSub, varNumCon, 5))

for idxSub in range(0, varNumSub):


    # ----------------------------------------------------------------------------
    # *** Interpolation (downsampling)
    
    # The empirical depth profiles are defined at more depth levels than the
    # draining model. We downsample the empirical depth profiles to the number
    # of depth levels of the model.

    # The relative thickness of the layers differs between V1 & V2.
    if strRoi == 'v1':
        print('------Interpolation - V1')
        # Relative thickness of the layers (layer VI, 20%; layer V, 10%; layer IV,
        # 40%; layer II/III, 20%; layer I, 10%; Markuerkiaga et al. 2016).
        # lstThck = [0.2, 0.1, 0.4, 0.2, 0.1]
        # From the relative thickness, we derive the relative position of the layers
        # (we set the position of each layer to the sum of all lower layers plus half
        # its own thickness):
        vecPosMdl = np.array([0.1, 0.25, 0.5, 0.8, 0.95])

    elif strRoi == 'v2':
        print('------Interpolation - V2')
        # Relative position of the layers (accordign to Weber et al., 2008,
        # Figure 5C, p. 2322). We start with the absolute depth:
        vecPosMdl = np.array([160.0, 590.0, 1110.0, 1400.0, 1620.0])
        # Divide by overall thickness (1.7 mm):
        vecPosMdl = np.divide(vecPosMdl, 1700.0)

    # Position of empirical datapoints:
    vecPosEmp = np.linspace(0.0, 1.0, num=varNumDpth, endpoint=True)

    
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

    # (1) Deconvolution based on Markuerkiaga et al. (2016).
    if varMdl == 1:
        aryNrnSnSb[idxSub, :, :] = depth_deconv_01(varNumCon,
                                                   aryEmp5SnSb[idxSub, :, :])

    # (2) Deconvolution based on Markuerkiaga et al. (2016) & scaling based on
    #     Markuerkiaga et al. (2016).
    elif varMdl == 2:
        aryNrnSnSb[idxSub, :, :] = depth_deconv_02(varNumCon,
                                                   aryEmp5SnSb[idxSub, :, :])

    # (2) Deconvolution based on Markuerkiaga et al. (2016) & scaling based on
    #     Weber et al. (2008).
    elif varMdl == 3:
        aryNrnSnSb[idxSub, :, :] = depth_deconv_03(varNumCon,
                                                   aryEmp5SnSb[idxSub, :, :],
                                                   strRoi=strRoi)


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

print('---Plot results')

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

print('-Done.')
