# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

Plot statistical parameter by eccentricity & cortical depth.

The purpose of this script is to plot a statistical parameter, such as
parameter estimates, by pRF eccentricity and cortical depth from information
contained in vtk files. To this end, information on the pRF eccentricity of
each vertex is loaded from a vtk file. This vtk file is defined at a single
cortical depth (e.g. mid-GM). Second, a vtk file with statistical information
at different depth levels in needed.

@author: Ingo Marquardt, 05.12.2016
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

# *****************************************************************************
# *** Import modules
import numpy as np
from ds_paramEccDpthGet import funcParamEccDpthGet
from ds_paramEccDpthPlt import funcParamEccDpthPlt
from ds_paramEccDpthHist import funcParamEccDpthHist
# *****************************************************************************


# *****************************************************************************
# *** Define parameters

# List of subject IDs:
lstSubId = ['20150930',
            '20151118',
            '20151127_01',
            '20151130_02',
            '20161205',
            '20161207',
            '20161212_02',
            '20161214',
            '20161219_01',
            '20161219_02']

# Path of vtk files with eccentricity information (subject ID left open):
strVtkEcc = '/home/john/PhD/ParCon_Depth_Data/{}/cbs_distcor/lh/eccentricity.vtk'  #noqa

# Minimum & maximum eccentricity:
# varEccMin = 0.5
# varEccMax = 6.0
# Number of eccentricity bins:
# varEccNum = 8
# Create eccentricity bins:
# vecEccBin = np.linspace(varEccMin, varEccMax, num=varEccNum, endpoint=True)
vecEccBin = np.array([0.2,
                      1.5,
                      2.0,
                      2.5,
                      3.0,
                      3.5,
                      4.0,
                      4.5,
                      7.0])

# Path of vtk file with statistical parameters (at several depth levels;
# subject ID left open):
strVtkParam = '/home/john/PhD/ParCon_Depth_Data/{}/cbs_distcor/lh/pe_stim_lvl_04.vtk'  #noqa
# Number of depth levels in the parameter vtk files:
varNumDpth = 11

# Beginning of string which precedes vertex data in data vtk files:
strPrcdData = 'SCALARS'
# Number of header lines in vtk files:
# varNumHdrRoi = 1
# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Paths of csv files with ROI information (created with paraview on a vtk mesh
# in the same space as the above vtk files; subject ID left open):
strCsvRoi = '/home/john/PhD/ParCon_Depth_Data/{}/cbs_distcor/lh/v1.csv'  #noqa

# Number of header lines in ROI CSV file:
varNumHdrRoi = 1

# Paths of vtk files with intensity information for thresholding (at all depth
# levels, e.g. R2 from pRF analysis; subject ID left open):
strVtkThr = '/home/john/PhD/ParCon_Depth_Data/{}/cbs_distcor/lh/R2_multi.vtk'  #noqa
# Threshold (e.g. minimum R2 value - if vertex is below this value at any
# depth level, vertex is excluded):
varThr = 0.16

# Output basename:
strPathOut = '/home/john/Desktop/paramEccV1/plot_stim_lvl_04'
# *****************************************************************************


# *****************************************************************************
# *** Load & sample data

print('-Plot parameters by eccentricity and depth')

print('---Loading data')

# Number of datasets:
varNumSub = len(lstSubId)

# List for single subject data - mean statistical parameters (for each
# eccentricity & cortical depth):
lstSubMean = [None] * varNumSub

# List for single subject data - eccentricity values for ROI:
lstSubEcc = [None] * varNumSub

# Loop through subjects and load eccentricity-by-depth data within ROI:
for idxSub in range(0, varNumSub):

    print(('------Dataset: ' + lstSubId[idxSub]))

    lstSubMean[idxSub], lstSubEcc[idxSub] = \
        funcParamEccDpthGet(strVtkEcc.format(lstSubId[idxSub]),
                            strPrcdData,
                            varNumLne,
                            strVtkParam.format(lstSubId[idxSub]),
                            varNumDpth,
                            strCsvRoi.format(lstSubId[idxSub]),
                            varNumHdrRoi,
                            vecEccBin,
                            strVtkThr.format(lstSubId[idxSub]),
                            varThr)
# *****************************************************************************


# *****************************************************************************
# *** Plot single subject eccentricity histograms

print('---Ploting single subject eccentricity histograms')

# Loop through subjects and plot single subject eccentricity histograms:
for idxSub in range(0, varNumSub):

    print(('------Dataset: ' + lstSubId[idxSub]))

    strTmp = (strPathOut + '_sngl_sub_' + lstSubId[idxSub] + '_ecc.png')

    funcParamEccDpthHist(lstSubEcc[idxSub],
                         vecEccBin,
                         strTmp)
# *****************************************************************************


# *****************************************************************************
# *** Plot across-subjects histograms

print('---Ploting across-subjects eccentricity histograms')

# Concatenate all single-subject eccentricity vectors:
vecEccAcrSubs = np.empty((0), dtype=np.float64)
for idxSub in lstSubEcc:
    vecEccAcrSubs = np.append(vecEccAcrSubs, idxSub)

strTmp = (strPathOut + '_acrsSubsEcc.png')

funcParamEccDpthHist(vecEccAcrSubs,
                     vecEccBin,
                     strTmp)
# *****************************************************************************


## *****************************************************************************
## *** Plot single subject results
#
#print('---Ploting single subject results')
#
## Loop through subjects and plot single subject results:
#for idxSub in range(0, varNumSub):
#
#    print(('------Dataset: ' + lstSubId[idxSub]))
#
#    strTmp = (strPathOut + '_sngl_sub_' + lstSubId[idxSub] + '.png')
#
#    funcParamEccDpthPlt(lstSubMean[idxSub],
#                        vecEccBin,
#                        strTmp)
## *****************************************************************************


# *****************************************************************************
# *** Grand mean scaling

# Before averaging across subjects, we apply grand mean scaling; i.e. we
# divide all PE values for a subject (i.e. all depth levels, all
# eccentricities) by the grand mean (i.e. the mean across depth levels &
# eccentricities).
for idxSub in range(0, varNumSub):

    # Calculate 'grand mean', i.e. the mean PE across depth levels and
    # conditions:
    #varGrndMean = np.mean(lstSubMean[idxSub])
    varGrndMean = np.median(lstSubMean[idxSub])

    # Divide all values by the grand mean:
    lstSubMean[idxSub] = np.divide(lstSubMean[idxSub], varGrndMean)

    # Rescale data (multiplication by 100):
    lstSubMean[idxSub] = np.multiply(lstSubMean[idxSub], 100.0)
# *****************************************************************************


# *****************************************************************************
# *** Plot across subject results

print('---Ploting across subjects results')

# Number of eccentricity bins:
varEccNum = vecEccBin.shape[0]

# Array data from all subjects:
arySubData = np.zeros(((varEccNum - 1), varNumDpth, varNumSub))

# Fill array:
for idxSub in range(0, varNumSub):
    arySubData[:, :, idxSub] = lstSubMean[idxSub]

# Take mean across subjects:
arySubData = np.mean(arySubData, axis=2)

# Plot across subjects mean:
strTmp = (strPathOut + '_acrsSubsMean.png')
funcParamEccDpthPlt(arySubData,
                    vecEccBin,
                    strTmp)
# *****************************************************************************
