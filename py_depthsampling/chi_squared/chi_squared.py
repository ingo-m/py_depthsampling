# -*- coding: utf-8 -*-
"""
Chi-squared test on superficial vs. non-superficial peak in depth profiles.
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

## Path of depth-profiles - first ROI, first condition:
#strRoi01Con01 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v1_rh_Pd_sst_deconv_model_1.npz'
## Path of depth-profiles - first ROI, second condition:
#strRoi01Con02 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v1_rh_Cd_sst_deconv_model_1.npz'
## Path of depth-profiles - second ROI, first condition:
#strRoi02Con01 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v2_rh_Pd_sst_deconv_model_1.npz'
## Path of depth-profiles - second ROI, second condition:
#strRoi02Con02 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v2_rh_Cd_sst_deconv_model_1.npz'

# Path of depth-profiles - first ROI, first condition:
strRoi01Con01 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v1_rh_Pd_sst_deconv_model_1.npz'
# Path of depth-profiles - first ROI, second condition:
strRoi01Con02 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v1_rh_Ps_sst_plus_Cd_sst_deconv_model_1.npz'
# Path of depth-profiles - second ROI, first condition:
strRoi02Con01 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v2_rh_Pd_sst_deconv_model_1.npz'
# Path of depth-profiles - second ROI, second condition:
strRoi02Con02 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v2_rh_Ps_sst_plus_Cd_sst_deconv_model_1.npz'


# ----------------------------------------------------------------------------
# *** Load depth profiles

# Load single-condition depth profiles from npz files:
objNpzTmp = np.load(strRoi01Con01)
aryRoi01Con01 = objNpzTmp['arySubDpthMns']

# Array with number of vertices (for weighted averaging across subjects),
# shape: vecNumInc[subjects].
vecNumIncRoi01 = objNpzTmp['vecNumInc']

# Load single-condition depth profiles from npz files:
objNpzTmp = np.load(strRoi01Con02)
aryRoi01Con02 = objNpzTmp['arySubDpthMns']

# Load single-condition depth profiles from npz files:
objNpzTmp = np.load(strRoi02Con01)
aryRoi02Con01 = objNpzTmp['arySubDpthMns']

# Array with number of vertices (for weighted averaging across subjects),
# shape: vecNumInc[subjects].
vecNumIncRoi02 = objNpzTmp['vecNumInc']

# Load single-condition depth profiles from npz files:
objNpzTmp = np.load(strRoi02Con02)
aryRoi02Con02 = objNpzTmp['arySubDpthMns']


# ----------------------------------------------------------------------------
# *** Create condition contrasts

# Within-subject condition contrast, first ROI:
aryCtrRoi01 = np.subtract(aryRoi01Con01, aryRoi01Con02)

# Within-subject condition contrast, second ROI:
aryCtrRoi02 = np.subtract(aryRoi02Con01, aryRoi02Con02)

# Number of subject:
varNumSubs = aryCtrRoi01.shape[0]

# Number of depth levels:
varNumDpt = aryCtrRoi01.shape[1]


# ----------------------------------------------------------------------------
# *** Upsample depth profiles

# Upsample and slightly smooth depth profiles to get a more fine grained
# estimate of the peak position.

# Number of depth levels to upsample to:
varNumIntp = 100

# Amount of smoothing (relative to cortical depth):
varSd = 0.05

# Position of original datapoints (before interpolation):
vecPosOrig = np.linspace(0, 1.0, num=varNumDpt, endpoint=True)

# Positions at which to sample (interpolate) depth profiles:
vecPosIntp = np.linspace(0, 1.0, num=varNumIntp, endpoint=True)

# Create function for interpolation:
func_interp_01 = interp1d(vecPosOrig,
                          aryCtrRoi01,
                          kind='linear',
                          axis=1,
                          fill_value='extrapolate')
func_interp_02 = interp1d(vecPosOrig,
                          aryCtrRoi02,
                          kind='linear',
                          axis=1,
                          fill_value='extrapolate')

# Apply interpolation function:
aryCtrRoiInt01 = func_interp_01(vecPosIntp)
aryCtrRoiInt02 = func_interp_02(vecPosIntp)

# Scale the standard deviation of the Gaussian kernel:
varSdSc = np.float64(varNumIntp) * varSd

# Smooth interpolated depth profiles:
aryCtrRoiSmth01 = gaussian_filter1d(aryCtrRoiInt01,
                                    varSdSc,
                                    axis=1,
                                    order=0,
                                    mode='nearest')
aryCtrRoiSmth02 = gaussian_filter1d(aryCtrRoiInt02,
                                    varSdSc,
                                    axis=1,
                                    order=0,
                                    mode='nearest')


# ----------------------------------------------------------------------------
# *** Find peaks

# Find peaks in cortical depth profiles of first & second ROI:
vecPeaks01 = np.argmax(aryCtrRoiSmth01, axis=1).astype(np.float32)
vecPeaks02 = np.argmax(aryCtrRoiSmth02, axis=1).astype(np.float32)

# Scale to range (0, 1):
vecPeaks01 = np.divide(vecPeaks01, varNumIntp)
vecPeaks02 = np.divide(vecPeaks02, varNumIntp)

# Are the peaks at superficial cortical depth? We define 'superficial' as
# greater than 0.66 cortical depth.
vecLgcSuper01 = np.greater(vecPeaks01, 0.66)
vecLgcSuper02 = np.greater(vecPeaks02, 0.66)

# Number of superficial peaks in each area:
varSuper01 = np.sum(vecLgcSuper01)
varSuper02 = np.sum(vecLgcSuper02)

# Chi-squared test:
chi2stat, pval, _ = proportions_chisquare([varSuper01, varSuper02],
                                       [varNumSubs, varNumSubs])

print('Chi-squared test for differences in depth profiles between ROIs')
print('   Test the H0 that the number of superficial peaks in single subject')
print('   cortical depth profiles does not differ between ROIs.')
print('   chi-squared = ' + str(np.around(chi2stat, decimals = 2)))
print('   p = ' + str(np.around(pval, decimals=2)))






