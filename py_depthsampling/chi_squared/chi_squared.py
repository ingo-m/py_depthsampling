# -*- coding: utf-8 -*-
"""
Chi-squared test on superficial vs. non-superficial peak in depth profiles.

Test the H0 that the number of superficial peaks in single subject cortical
depth profiles does not differ between ROIs.
"""

# Part of py_depthsampling library
# Copyright (C) 2020  Ingo Marquardt
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
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from statsmodels.stats.proportion import proportions_chisquare


# ----------------------------------------------------------------------------
# *** Define parameters

# Directory with npz files containing cortical depth profiles:
strPath = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/'

# List of file names of depth profiles, first condition, several ROIs (i.e. two
# or more) ROIs. First item from this list will be compared with first item of
# second list (i.e. `lstCon02[0] - lstCon01[0]`, etc. for all items).
lstCon01 = ['v1_rh_Pd_sst_deconv_model_1.npz',
            'v2_rh_Pd_sst_deconv_model_1.npz']

# List of file names of depth profiles, second condition, several ROIs (i.e. two
# or more) ROIs.
lstCon02 = ['v1_rh_Ps_sst_plus_Cd_sst_deconv_model_1.npz',
            'v2_rh_Ps_sst_plus_Cd_sst_deconv_model_1.npz']


# ----------------------------------------------------------------------------
# *** Load depth profiles

# Number of ROIs:
varNumRoi = len(lstCon01)

# List for single-condition depth profiles:
lstDpthCon01 = [None] * varNumRoi
lstDpthCon02 = [None] * varNumRoi

# List for number of vertices per subject (only one value per ROI, same for
# both conditions):
lstNumInc = [None] * varNumRoi

# Loop through ROIs and load profiles:
for idxRoi in range(varNumRoi):

    # First condition.

    # Load single-condition depth profiles from npz files:
    objNpzTmp = np.load((strPath + lstCon01[idxRoi]))
    lstDpthCon01[idxRoi] = objNpzTmp['arySubDpthMns']

    # Second condition.

    # Load single-condition depth profiles from npz files:
    objNpzTmp = np.load((strPath + lstCon02[idxRoi]))
    lstDpthCon02[idxRoi] = objNpzTmp['arySubDpthMns']

    # Array with number of vertices, shape: vecNumInc[subjects].
    lstNumInc[idxRoi] = objNpzTmp['vecNumInc']


# ----------------------------------------------------------------------------
# *** Create condition contrasts

# List for within-subject condition contrast:
lstCtr = [None] * varNumRoi

# Loop through ROIs and calculate condition contrast:
for idxRoi in range(varNumRoi):

    lstCtr[idxRoi] = np.subtract(lstDpthCon01[idxRoi],
                                 lstDpthCon02[idxRoi])

# Number of subject:
varNumSubs = lstCtr[0].shape[0]

# Number of depth levels:
varNumDpt = lstCtr[0].shape[1]


# ----------------------------------------------------------------------------
# *** Upsample depth profiles

# Upsample and slightly smooth depth profiles to get a more fine grained
# estimate of the peak position.

# Number of depth levels to upsample to:
varNumIntp = 100

# Amount of smoothing (relative to cortical depth):
varSd = 0.05

# Scale the standard deviation of the Gaussian kernel:
varSdSc = np.float64(varNumIntp) * varSd

# Position of original datapoints (before interpolation):
vecPosOrig = np.linspace(0, 1.0, num=varNumDpt, endpoint=True)

# Positions at which to sample (interpolate) depth profiles:
vecPosIntp = np.linspace(0, 1.0, num=varNumIntp, endpoint=True)

# List for upsampled depth profiles (condition contrast):
lstCtrUp = [None] * varNumRoi

# Loop through ROIs and upsample profiles:
for idxRoi in range(varNumRoi):

    # Create function for interpolation:
    func_interp = interp1d(vecPosOrig,
                           lstCtr[idxRoi],
                           kind='linear',
                           axis=1,
                           fill_value='extrapolate')


    # Apply interpolation function:
    aryTmp = func_interp(vecPosIntp)

    # Smooth interpolated depth profiles:
    lstCtrUp[idxRoi] = gaussian_filter1d(aryTmp,
                                         varSdSc,
                                         axis=1,
                                         order=0,
                                         mode='nearest')


# ----------------------------------------------------------------------------
# *** Find peaks

# List for arrays with single-subject peak positions:
lstPeaks = [None] * varNumRoi

# Loop through ROIs and find peaks in cortical depth profiles:
for idxRoi in range(varNumRoi):

    # Vector with peak positions (one per subeject):
    vecTmp = np.argmax(lstCtrUp[idxRoi], axis=1).astype(np.float32)

    # Scale to range (0, 1):
    lstPeaks[idxRoi] = np.divide(vecTmp, varNumIntp)

# Are the peaks at superficial cortical depth? We define 'superficial' as
# greater than 0.66 cortical depth.
lstLgcSuper = [np.greater(x, 0.66) for x in lstPeaks]

# Number of superficial peaks in each area:
lstNumSuper = [np.sum(x) for x in lstLgcSuper]

# Chi-squared test:
chi2stat, pval, _ = proportions_chisquare(lstNumSuper,
                                          [varNumSubs] * varNumRoi)

print('Chi-squared test for differences in depth profiles between ROIs')
print('   Test the H0 that the number of superficial peaks in single subject')
print('   cortical depth profiles does not differ between ROIs.')
print('   chi-squared = ' + str(np.around(chi2stat, decimals = 2)))
print('   p = ' + str(np.around(pval, decimals=2)))
