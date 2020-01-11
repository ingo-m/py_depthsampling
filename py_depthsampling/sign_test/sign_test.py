# -*- coding: utf-8 -*-
"""
Sign test on superficial vs. non-superficial peak in condition contrasts.

Test whether 
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


import itertools
import numpy as np
from py_depthsampling.main.find_peak import find_peak
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
# ***

# Find peaks in first & second ROI:
# vecPeaks01 = find_peak(aryCtrRoi01, varNumIntp=100, varSd=0.1)
# vecPeaks02 = find_peak(aryCtrRoi02, varNumIntp=100, varSd=0.1)

vecPeaks01 = np.argmax(aryCtrRoi01, axis=1).astype(np.float32)
vecPeaks02 = np.argmax(aryCtrRoi02, axis=1).astype(np.float32)

# Scale to range (0, 1):
vecPeaks01 = np.divide(vecPeaks01, varNumDpt)
vecPeaks02 = np.divide(vecPeaks02, varNumDpt)

# Are the peaks at superficial cortical depth? We define 'superficial' as
# greater than 0.66 cortical depth.
vecLgcSuper01 = np.greater(vecPeaks01, 0.66)
vecLgcSuper02 = np.greater(vecPeaks02, 0.66)

# Number of superficial peaks in each area:
varSuper01 = np.sum(vecLgcSuper01)
varSuper02 = np.sum(vecLgcSuper02)


roi01 = aryCtrRoi01.T
roi02 = aryCtrRoi02.T




peak = argrelextrema(profile,
                     np.greater,
                     axis=0,
                     order=3,
                     mode='clip')


chi2stat, p, _ = proportions_chisquare([varSuper01, varSuper02],
                                       [varNumSubs, varNumSubs])













