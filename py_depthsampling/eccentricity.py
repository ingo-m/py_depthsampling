# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

Plot statistical parameter by eccentricity & cortical depth.

Plot a statistical parameter, such as parameter estimates, by pRF eccentricity
and cortical depth from information contained in vtk files. To this end,
information on the pRF eccentricity of each vertex is loaded from a vtk file
(the eccentricity at each vertex is defined as the median across depth levels,
so as not to bias the selection to any depth level). Second, a vtk file with
statistical information at different depth levels in needed.
"""

# Part of py_depthsampling library
# Copyright (C) 2018  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
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
from py_depthsampling.eccentricity.ecc_main import eccentricity
# *****************************************************************************


# *****************************************************************************
# *** Define parameters

# Region of interest ('v1' or 'v2'):
lstRoi = ['v1', 'v2']

# Hemispheres ('lh' or 'rh'):
lstHmsph = ['lh', 'rh']

# List of subject IDs:
lstSubId = ['20171023',  # '20171109',
            '20171204_01',
            '20171204_02',
            '20171211',
            '20171213',
            '20180111',
            '20180118']

# Condition levels (used to complete file names):
# lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst', 'Pd_min_Ps_sst', 'Pd_min_Cd_sst',
#           'Pd_trn', 'Cd_trn', 'Ps_trn', 'Pd_min_Ps_trn', 'Pd_min_Cd_trn']
lstCon = ['Pd_sst', 'Pd_trn']

# Path of vtk files with eccentricity information (subject ID and hemisphere
# left open):
strVtkEcc = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_eccentricity.vtk'  #noqa

# Eccentricity bins:
vecEccBin = np.array([0.1,
                      1.0,
                      1.5,
                      2.0,
                      2.5,
                      3.0,
                      3.5,
                      4.0,
                      4.5,
                      5.0,
                      5.5,
                      6.0])

# Path of vtk file with statistical parameters (at several depth levels;
# subject ID, hemisphere, and condition left open):
strVtkParam = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/feat_level_2_{}_cope.vtk'  #noqa
# Number of depth levels in the parameter vtk files:
varNumDpth = 11

# Beginning of string which precedes vertex data in data vtk files:
strPrcdData = 'SCALARS'
# Number of header lines in vtk files:
# varNumHdrRoi = 1
# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Paths of csv files with ROI information (created with paraview on a vtk mesh
# in the same space as the above vtk files; subject ID, hemisphere, and ROI
# left open):
strCsvRoi = '/home/john/PhD/GitHub/PacMan/analysis/{}/08_depthsampling/{}/{}_mod.csv'  #noqa

# Number of header lines in ROI CSV file:
varNumHdrRoi = 1

# Paths of vtk files with intensity information for thresholding (at all depth
# levels, e.g. R2 from pRF analysis; subject ID, and hemisphere left open):
strVtkThr = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_R2.vtk'  #noqa
# Threshold (e.g. minimum R2 value - if vertex is below this value at any
# depth level, vertex is excluded):
varThr = 0.1

# Use separate lookup tables for negative values?
lgcNegLkp = True

# Output basename (ROI, hemisphere, and condition left open):
strPathOut = '/home/john/PhD/PacMan_Plots/eccentricity/plots/ecc_{}_{}_{}'
# *****************************************************************************


# *****************************************************************************
# *** Loop through ROIs / conditions

# Loop through ROIs, hemispheres, and conditions to create plots:
for idxRoi in range(len(lstRoi)):
    for idxHmsph in range(len(lstHmsph)):
        for idxCon in range(len(lstCon)):

            # Call main function:
            eccentricity(lstSubId, strVtkEcc.format('{}', lstHmsph[idxHmsph]),
                         vecEccBin, strVtkParam.format('{}',
                         lstHmsph[idxHmsph], lstCon[idxCon]), varNumDpth,
                         strPrcdData, varNumLne, strCsvRoi.format('{}',
                         lstHmsph[idxHmsph], lstRoi[idxRoi]), varNumHdrRoi,
                         strVtkThr.format('{}', lstHmsph[idxHmsph]), varThr,
                         lgcNegLkp, strPathOut.format(lstHmsph[idxHmsph],
                         lstRoi[idxRoi], lstCon[idxCon]))
# *****************************************************************************
