# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

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

Let varEmpVI, varEmpV, varEmpIV, varEmpII_III, and varEmpI be the observed
(empirical) signal at the different depth levels, and varNrnVI, varNrnV,
varNrnIV, varNrnII_III, and varNrnI the underlying neuronal activity.
Following to the model by Markuerkiaga et al. (2016), the absolute fMRI signal
for each layer for a GE sequence can be predicted as follows (forward model,
as depicted in Figure 3F, p. 495):

    Layer VI:

        varEmpVI = 1.9 * varNrnVI

    Layer V:

        varEmpV = 1.5 * varNrnV
                  + 0.6 * varNrnVI

    Layer IV:

        varEmpIV = 2.2 * varNrnIV
                   + 0.3 * varNrnV
                   + 0.6 * varNrnVI

    Layer II/III:

        varEmpII_III = 1.7 * varNrnII_III
                       + 1.3 * varNrnIV
                       + 0.3 * varNrnV
                       + 0.5 * varNrnVI

    Layer I:

        varEmpI = 1.6 * varNrnI
                  + 0.7 * varNrnII_III
                  + 1.3 * varNrnIV
                  + 0.3 * varNrnV
                  + 0.5 * varNrnVI

These values are translated into the a transfer function to estimate the local
neural activity at each layer given an empirically observed fMRI signal depth
profile:

    Layer VI:

        varNrnVI = varEmpVI / 1.9

    Layer V:

        varNrnV = (varEmpV
                   - 0.6 * varNrnVI) / 1.5

    Layer IV:

        varNrnIV = (varEmpIV
                    - 0.3 * varNrnV
                    - 0.6 * varNrnVI) / 2.2

    Layer II/III:

        varNrnII_III = (varEmpII_III
                        - 1.3 * varNrnIV
                        - 0.3 * varNrnV
                        - 0.5 * varNrnVI) / 1.7

    Layer I:

        varNrnI = (varEmpI
                   - 0.7 * varNrnII_III
                   - 1.3 * varNrnIV
                   - 0.3 * varNrnV
                   - 0.5 * varNrnVI) / 1.6

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

# Output path & prefix:
strPthOt = '/home/john/Desktop/deconvolution_'

# File type suffix for plot:
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Limits of y-axis for across subject plot:
varAcrSubsYmin = -0.05
varAcrSubsYmax = 2.0  # 1.90

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
        aryNrn = np.zeros(aryEmp5.shape)

        # Layer VI:
        aryNrn[idxCon, 0] = aryEmp5[idxCon, 0] / 1.9

        #Layer V:
        aryNrn[idxCon, 1] = (aryEmp5[idxCon, 1]
                             - 0.6 * aryNrn[idxCon, 0]) / 1.5

        # Layer IV:
        aryNrn[idxCon, 2] = (aryEmp5[idxCon, 2]
                             - 0.3 * aryNrn[idxCon, 1]
                             - 0.6 * aryNrn[idxCon, 0]) / 2.2

        # Layer II/III:
        aryNrn[idxCon, 3] = (aryEmp5[idxCon, 3]
                             - 1.3 * aryNrn[idxCon, 2]
                             - 0.3 * aryNrn[idxCon, 1]
                             - 0.5 * aryNrn[idxCon, 0]) / 1.7

        # Layer I:
        aryNrn[idxCon, 4] = (aryEmp5[idxCon, 4]
                             - 0.7 * aryNrn[idxCon, 3]
                             - 1.3 * aryNrn[idxCon, 2]
                             - 0.3 * aryNrn[idxCon, 1]
                             - 0.5 * aryNrn[idxCon, 0]) / 1.6

        # Put deconvolution result for this subject into the array:
        aryNrnSnSb[idxSub, idxCon, :] = np.copy(aryNrn[idxCon, :])


# ----------------------------------------------------------------------------
# *** Plot results

# Plot across-subjects mean before deconvolution:
strTmpTtl = 'Before deconvolution'
strTmpPth = (strPthOt + 'before')
funcPltAcrSubsMean(aryEmp5SnSb,
                   varNumSub,
                   5,
                   varNumCon,
                   varDpi,
                   varAcrSubsYmin,
                   varAcrSubsYmax,
                   lstConLbl,
                   strXlabel,
                   strYlabel,
                   strTmpTtl,
                   strTmpPth,
                   strFlTp)


# Across-subjects mean after deconvolution:
strTmpTtl = 'After deconvolution'
strTmpPth = (strPthOt + 'after')
funcPltAcrSubsMean(aryNrnSnSb,
                   varNumSub,
                   5,
                   varNumCon,
                   varDpi,
                   varAcrSubsYmin,
                   varAcrSubsYmax,
                   lstConLbl,
                   strXlabel,
                   strYlabel,
                   strTmpTtl,
                   strTmpPth,
                   strFlTp)

aryEmp5 = np.mean(aryEmp5SnSb, axis=0).T
aryNrn = np.mean(aryNrnSnSb, axis=0).T

