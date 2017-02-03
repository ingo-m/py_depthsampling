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
            # '20151118',
            '20151130_01',
            '20151130_02']

# Path of vtk files with eccentricity information:
lstVtkEcc = ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_eccentricity.vtk',  #noqa
             #'/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_eccentricity.vtk',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_eccentricity.vtk',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_eccentricity.vtk']  #noqa

# Minimum & maximum eccentricity:
# varEccMin = 0.5
# varEccMax = 6.0
# Number of eccentricity bins:
# varEccNum = 8
# Create eccentricity bins:
# vecEccBin = np.linspace(varEccMin, varEccMax, num=varEccNum, endpoint=True)
vecEccBin = np.array([0.2,
                      1.0,
                      1.5,
                      2.0,
                      2.5,
                      3.0,
                      3.5,
                      4.0,
                      4.5,
                      7.0])

# Path of vtk file with statistical parameters (at several depth levels):
lstVtkParam = ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_pe_stim_lvl_04.vtk',  #noqa
               #'/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_pe_stim_lvl_04.vtk',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_pe_stim_lvl_04.vtk',  #noqa
               '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_pe_stim_lvl_04.vtk']  #noqa
# Number of depth levels in the parameter vtk files:
varNumDpth = 11

# Beginning of string which precedes vertex data in data vtk files:
strPrcdData = 'SCALARS'
# Number of header lines in vtk files:
# varNumHdrRoi = 1
# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Paths of csv files with ROI information (created with paraview on a vtk mesh
# in the same space as the above vtk files):
# lstCsvRoi = ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/v1_liberal.csv',  #noqa
#             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/v1_liberal.csv',  #noqa
#             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/v1_liberal.csv',  #noqa
#             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/v1_liberal.csv']  #noqa
lstCsvRoi = ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/v2.csv',  #noqa
             #'/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/v2.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/v2.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/v2.csv']  #noqa

# Number of header lines in ROI CSV file:
varNumHdrRoi = 1

# Paths of vtk files with intensity information for thresholding (one value per
# vertex, e.g. R2 from pRF analysis):
lstVtkThr = ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_R2.vtk',  #noqa
             #'/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_R2.vtk',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_R2.vtk',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_R2.vtk']  #noqa

# Threshold (e.g. minimum R2 value):
# varThr = 0.15
varThr = 0.125

# Output basename:
strPathOut = '/home/john/Desktop/paramEccV2/plot'
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
        funcParamEccDpthGet(lstVtkEcc[idxSub],
                            strPrcdData,
                            varNumLne,
                            lstVtkParam[idxSub],
                            varNumDpth,
                            lstCsvRoi[idxSub],
                            varNumHdrRoi,
                            vecEccBin,
                            lstVtkThr[idxSub],
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


# *****************************************************************************
# *** Plot single subject results

print('---Ploting single subject results')

# Loop through subjects and plot single subject results:
for idxSub in range(0, varNumSub):

    print(('------Dataset: ' + lstSubId[idxSub]))

    strTmp = (strPathOut + '_sngl_sub_' + lstSubId[idxSub] + '.png')

    funcParamEccDpthPlt(lstSubMean[idxSub],
                        vecEccBin,
                        strTmp)
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
