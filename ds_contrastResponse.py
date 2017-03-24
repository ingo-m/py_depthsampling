"""
Function of the depth sampling pipeline.

The purpose of this function is to apply fit a contrast response function to
fMRI depth profiles, separately for each depth level.

The contrast response function used here is the Naka-Rushton equation
(Albrecht & Hamilton, 1982; cited in Niemeyer & Paradiso, 2016). This
neuronal contrast response function is applied to the depth profiles after
the application of the draining model.
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
# import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------------
# *** Define parameters

# Path of draining-corrected depth-profiles:
strPthPrfOt = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1_corrected.npy'


# ----------------------------------------------------------------------------
# *** Load depth profiles

# Load array with single-subject corrected depth profiles, of the form
# aryNrnSnSb[idxSub, idxCondition, idxDpth].
aryNrnSnSb = np.load(strPthPrfOt)


# ----------------------------------------------------------------------------
# *** Average over subjects

# Across-subjects mean:
aryNrn = np.mean(aryNrnSnSb, axis=0)


# ----------------------------------------------------------------------------
# *** Fit contrast reponse function

# Neuronal contrast response function, following the 'Naka-Rushton equation'
# (see Niemeyer & Paradiso, 2016).
#    - varR is the neuronal response (in spikes/s)
#    - varC is contrast
#    - varRmax is the maximum neural response
#    - varC50 is the contrast that gives a half-maximal response
#    - varM is the average baseline firing rate
#    - varN in the exponent, i.e. the parameter to be fitted
def funcCrf(varC, varRmax, varC50, varM, varN):
    varR = (varRmax
            * np.power(varC, varN)
            / (np.power(varC, varN) + np.power(varC50, varN))
            + varM)
    return varR






vecModelPar, vecModelCov = curve_fit(lambda varC, varN: funcCrf(varC,
                                                                varRmax,
                                                                varC50,
                                                                varM,
                                                                varN),
                                     x1,
                                     x2)




