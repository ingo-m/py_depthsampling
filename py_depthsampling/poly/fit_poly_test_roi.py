# -*- coding: utf-8 -*-
"""
Fit 2nd degree polynomial function to depth profiles.

Cortical depth profiles of condition differences are computed from single
condition cortical depth profiles (condition A minus condition B, separately
for each subject and ROI). A 2nd degree polynomial function is fitted to the
single-subject depth profiles, separately for each of two ROIs. A t-test is
performed to test for equality of peak position between the two ROIs.

Function of the depth sampling pipeline.
"""

# Part of py_depthsampling library
# Copyright (C) 2019 Ingo Marquardt
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
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel


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

# Names of ROIs to be compared:
strRoi01 = 'v1'
strRoi02 = 'v2'

# Name of conditions for contrast (01 minus 02):
strCon01 = 'Pd_sst'
strCon02 = 'Ps_sst_plus_Cd_sst'

# Input path of depth profiles, ROI and condition left open:
strPthIn = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/{}_rh_{}_sst_deconv_model_1.npz'

# Output path for plots
strPthPlt = '/home/john/Dropbox/PacMan_Plots/poly/{}_and_{}_condition_{}_minus_{}_subject_{}.png'


# ----------------------------------------------------------------------------
# *** 2nd degree polynomial function

def funcPoly2(varX, varA, varB, varC):
    """2nd degree polynomial function to be fitted to the data."""
    varOut = (varA * np.power(varX, 2) +
              varB * np.power(varX, 1) +
              varC)
    return varOut


# ----------------------------------------------------------------------------
# *** Load depth profiles

print('-Peak position t-test')

# Path of depth-profiles - first ROI, first condition:
strRoi01Con01 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/{}_rh_{}_deconv_model_1.npz'.format(strRoi01, strCon01)
# Path of depth-profiles - first ROI, second condition:
strRoi01Con02 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/{}_rh_{}_deconv_model_1.npz'.format(strRoi01, strCon02)
# Path of depth-profiles - second ROI, first condition:
strRoi02Con01 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/{}_rh_{}_deconv_model_1.npz'.format(strRoi02, strCon01)
# Path of depth-profiles - second ROI, second condition:
strRoi02Con02 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/{}_rh_{}_deconv_model_1.npz'.format(strRoi02, strCon02)

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
# *** Fit & plot quadratic function

print('---Fit quadratic function to depth profiles')

sns.set(style="darkgrid")

# Independent variable data:
vecInd = np.linspace(0.0, 1.0, num=varNumDpt)

# Independent variable data with more sampling points (for identification of
# peak position):
varNumDptHd = 1000
vecIndHd = np.linspace(0.0, 1.0, num=varNumDptHd)

# Indices of empirical datapoints in higher-resolution model space (points in
# between in the dataframes are filled with NaNs):
vecIdx = np.linspace(0, (varNumDptHd - 1), num=varNumDpt, dtype=np.int64)

# Array for predicted values (based on function fit), for the two ROIs:
aryPred01 = np.zeros((varNumSubs, varNumDpt))
aryPred02 = np.zeros((varNumSubs, varNumDpt))

# Array for peak positions, for the two ROIs:
vecPeak01 = np.zeros(varNumSubs)
vecPeak02 = np.zeros(varNumSubs)

# Line colours:
lstClr = [(49/255,163/255,84/255),
          (161/255,217/255,155/255),
          (230/255,85/255,13/255),
          (253/255,174/255,107/255)]

# ary01 = aryCtrRoi01.T
# ary02 = aryCtrRoi02.T

# Loop through subjects:
for idxSub in range(varNumSubs):

    # Fit 2nd degree polynomial function:
    vecPoly2ModelPar01, vecPoly2ModelCov01 = curve_fit(funcPoly2,
                                                       vecInd,
                                                       aryCtrRoi01[idxSub, :])
    vecPoly2ModelPar02, vecPoly2ModelCov02 = curve_fit(funcPoly2,
                                                       vecInd,
                                                       aryCtrRoi02[idxSub, :])

    # Calculate fitted values:
    vecFitted01 = funcPoly2(vecIndHd,
                            vecPoly2ModelPar01[0],
                            vecPoly2ModelPar01[1],
                            vecPoly2ModelPar01[2])
    vecFitted02 = funcPoly2(vecIndHd,
                            vecPoly2ModelPar02[0],
                            vecPoly2ModelPar02[1],
                            vecPoly2ModelPar02[2])

    # Get peak positions (on x-axis, in relative cortical depth, i.e. between 0
    # and 1) of model prediction:
    vecPeak01[idxSub] = float(np.argmax(vecFitted01)) / float(varNumDptHd)
    vecPeak02[idxSub] = float(np.argmax(vecFitted02)) / float(varNumDptHd)

    # Populate dataframe:
    df = pd.DataFrame(np.nan,
                      index=list(range(varNumDptHd)),
                      columns=['Cortical depth',
                               ('Empirical ' + strRoi01),
                               ('Prediction ' + strRoi01),
                               ('Empirical ' + strRoi02),
                               ('Prediction ' + strRoi02)])
    df['Cortical depth'] = vecIndHd
    df[('Prediction ' + strRoi01)] = vecFitted01
    df[('Prediction ' + strRoi02)] = vecFitted02
    df[('Empirical ' + strRoi01)][vecIdx] = aryCtrRoi01[idxSub, :]
    df[('Empirical ' + strRoi02)][vecIdx] = aryCtrRoi02[idxSub, :]
    df = df.melt('Cortical depth', var_name='ROI',  value_name='PSC')

    # Plot the responses for different events and regions
    objPlt = sns.lineplot(x='Cortical depth',
                          y='PSC',
                          hue='ROI',
                          data=df,
                          palette=lstClr)
    objFig = objPlt.get_figure()
    objFig.savefig(strPthPlt.format(strRoi01,
                                    strRoi02,
                                    strCon01,
                                    strCon02,
                                    idxSub))
    objFig.clf()


# ----------------------------------------------------------------------------
# ***  T-test

# Test for difference in peak position between ROIs:
varT, varP = ttest_rel(vecPeak01, vecPeak02)

print('-Done.')
# ----------------------------------------------------------------------------

