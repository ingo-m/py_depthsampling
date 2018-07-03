# -*- coding: utf-8 -*-
"""
Estimate cortical depth point spread function.

The cortical depth point spread function is estimated from visual field
projections of percent signal change (which can to be created using
`py_depthsampling.project.project_main.py`). The point spread function is
modelled as a Gaussian. In addition to the Gaussian filter, the visual field
projections are scaled (in order to account for increasing signal towards the
cortical surface). The lowest cortical depth level (i.e. closest to white
matter) is taken as a reference, and the point spread function is estimated by
reducing the residuals between the visual field projection of each depth level
and the reference visual field projection.
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


import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from py_depthsampling.psf_2D.utilities import psf
from py_depthsampling.psf_2D.utilities import psf_diff


# -----------------------------------------------------------------------------
# *** Define parameters

# Load projection from (ROI, condition, depth level label left open):
strPthNpy = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/project/{}_{}_{}.npy'  #noqa

# Depth level labels (to complete input file names). First depth level in list
# is used as reference for estimation of point spread function.
lstDpthLbl = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# ROI ('v1','v2', or 'v3'):
lstRoi = ['v1', 'v2', 'v3']

# Output path & prefix for plots (ROI, condition, depth level label left open):
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/psf_2D_pe/{}_{}_{}'  #noqa

# File type suffix for plot:
# strFlTp = '.svg'
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Condition levels (used to complete file names):
lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst']

# Extent of visual space from centre of the screen in negative x-direction
# (i.e. from the fixation point to the left end of the screen) in degrees of
# visual angle.
varExtXmin = -5.19
# Extent of visual space from centre of the screen in positive x-direction
# (i.e. from the fixation point to the right end of the screen) in degrees of
# visual angle.
varExtXmax = 5.19
# Extent of visual space from centre of the screen in negative y-direction
# (i.e. from the fixation point to the lower end of the screen) in degrees of
# visual angle.
varExtYmin = -5.19
# Extent of visual space from centre of the screen in positive y-direction
# (i.e. from the fixation point to the upper end of the screen) in degrees of
# visual angle.
varExtYmax = 5.19
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Parent loop

print('-Estimate cortical depth point spread function')

# Number of ROIs/conditions/depths:
varNumDpth = len(lstDpthLbl)
varNumRoi = len(lstRoi)
varNumCon = len(lstCon)

# Loop through ROIs, conditions, depth levels:
for idxRoi in range(varNumRoi):
    for idxCon in range(varNumCon):
        for idxDpth in range(varNumDpth):  #noqa

            # The first entry in the list of depth levels is assumed to be the
            # deepest depth level, and used as the reference for the estimation
            # of the point spread function.
            if idxDpth == 0:

                # File name of npy file for referecnecondition:
                strPthNpyTmp = strPthNpy.format(lstRoi[idxRoi],
                                                lstCon[idxCon],
                                                lstDpthLbl[idxDpth])

                # Load visual field projection:
                aryDeep = np.load(strPthNpyTmp)

            else:

                # File name of npy file for current condition:
                strPthNpyTmp = strPthNpy.format(lstRoi[idxRoi],
                                                lstCon[idxCon],
                                                lstDpthLbl[idxDpth])

                # Load visual field projection:
                aryVfp = np.load(strPthNpyTmp)








strTmp01 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/project/v1_Pd_sst_deepGM.npy'
strTmp02 = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/project/v1_Pd_sst_superficialGM.npy'


ary01 = np.load(strTmp01)
ary02 = np.load(strTmp02)







varNum = 100

vecSd = np.linspace(0.0, 50.0, num=varNum)
vecFct = np.linspace(0.0, 5.0, num=varNum)

aryRes = np.zeros((varNum, varNum))

for idxSd in range(varNum):
    print(idxSd)
    for idxFct in range(varNum):
        aryRes[idxSd, idxFct] = psf_diff_02(vecSd[idxSd],
                                            vecFct[idxFct],
                                            ary01,
                                            ary02)

# aryRatio = np.divide(aryResSum, aryRes)
# aryResSum = np.copy(aryRes)



tplIdxMin = np.unravel_index(aryRes.argmin(), aryRes.shape)
varFitSd = vecSd[tplIdxMin[0]]
varFitFct = vecFct[tplIdxMin[1]]



varInitSd = 25.0
varInitFct = 2.5
vecInit = np.array([varInitSd, varInitFct])
# Sequence of (min, max) pairs:
lstBnds = [(0.0, 50.0), (0.0, 5.0)]

dicOptm = minimize(psf_diff_01, vecInit, args=(ary01, ary02), bounds=lstBnds)

print(dicOptm.x[0])
print(dicOptm.x[1])

aryFit01 = psf(ary01, dicOptm.x[0], dicOptm.x[1])
