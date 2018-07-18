# -*- coding: utf-8 -*-
"""
Estimate cortical depth point spread function.

The cortical depth point spread function is estimated from visual field
projections of percent signal change (which can be created using
`py_depthsampling.project.project_singlesubject`). The point spread function is
modelled as a Gaussian. In addition to the Gaussian filter, the visual field
projections are scaled (in order to account for increasing signal towards the
cortical surface). The lowest cortical depth level (i.e. closest to white
matter) is taken as a reference, and the point spread function is estimated by
reducing the residuals between the visual field projection of each depth level
and the reference visual field projection.

The point spread function is estimated on group averages. In order to get an
estimate of the variance, bootstrap confidence interval are created (sampling
single subject visual field projections with replacement).
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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from py_depthsampling.psf_2D.psf_2D_estimate import estm_psf


# -----------------------------------------------------------------------------
# *** Define parameters

# PSF parameters are saved to or loaded from pickled dataframe. Path of
# dataframe (number of samples left open):
strPthDf = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/psf_2D/dataframe_{}.pickle'  #noqa

# Load projection from (ROI, condition, depth level label left open):
strPthNpz = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/project_single_subject/{}_{}_{}.npz'  #noqa

# Depth level labels (to complete input file names). First depth level in list
# is used as reference for estimation of point spread function.
lstDpthLbl = [str(x) for x in range(11)]

# ROI ('v1','v2', or 'v3'):
lstRoi = ['v1']  # , 'v2', 'v3']

# Output path & prefix for summary plots (file name left open). Set to `None`
# if plot should not be created.
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/psf_2D_pe/{}'  #noqa

# Output path & prefix for plots of visual field projections after application
# of fitted PSF (file name left open). Set to `None` if plot should not be
# created.
strPthPltVfp = None  # '/home/john/Dropbox/PacMan_Plots/psf_2D_pe/{}'

# File type suffix for plot:
# strFlTp = '.svg'
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Condition levels (used to complete file names):
lstCon = ['Pd_sst']  # , 'Ps_sst', 'Cd_sst']

# Initial guess for PSF parameters (width and scaling factor; SD in degree of
# visual angle):
varInitSd = 1.0
varInitFct = 5.0

# Limits for PSF parameters [SD is in degrees of visual angle]:
tplBndSd = (0.0, 2.0)
tplBndFct = (0.0, 10.0)

# Extent of visual space from centre of the screen (assumed to be the same in
# positive/negative x/y direction:
varExtmax = 2.0 * 5.19

# Save result from model fitting (i.e. parameters of PSF) to disk (pandas data
# frame saved as csv for import in R). If `None`, data frame is not created.
strPthCsv = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/psf_2D/dataframe.csv'  #noqa

# Number of bootstrapping iterations:
varNumIt = 100

# Lower and upper bound of bootstrap confidence intervals:
varConLw = 5.0
varConUp = 95.0
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Preparations

# Get dimension of visual space model (assumed to be the same for x and y
# directions, and for all ROIs/conditions/depth levels):
objNpzTmp = np.load(strPthNpz.format(lstRoi[0], lstCon[0], lstDpthLbl[0]))
aryTmp = objNpzTmp['aryVslSpc']
varSzeVsm = aryTmp.shape[1]

# Get number of subjects:
varNumSub = aryTmp.shape[0]

# Scaling factor from degrees of visual angle to array dimensions:
varScl = (float(varSzeVsm) / float(varExtmax))

# Scale Gaussian SD from degree of visual angle to array dimensions:
varInitSd = (varInitSd * varScl)
tplBndSd = ((tplBndSd[0] * varScl), (tplBndSd[1] * varScl))

# Bring initial values and bounds into shape expected by scipy optimize:
vecInit = np.array([varInitSd, varInitFct])
lstBnds = [tplBndSd, tplBndFct]

# Number of ROIs/conditions/depths:
varNumDpth = len(lstDpthLbl)
varNumRoi = len(lstRoi)
varNumCon = len(lstCon)

# Total number of samples for dataframe (`(varNumDpth - 1)` because there are
# no PSF parameters for the reference depth level):
varNumSmpl = (varNumRoi * varNumCon * (varNumDpth - 1))

# Feature list (column names for dataframe). For width & scaling, there are
# columns for lower & upper bounds of bootstrapping confidence interval.
lstFtr = ['ROI', 'Condition', 'Depth', 'Width', 'Width CI low', 'Width CI up',
          'Scaling', 'Scaling CI low', 'Scaling CI up', 'Residuals']

# Coutner for dataframe samples:
idxSmpl = 0

# Bootstrap parameters:

# We will sample subjects with replacement. How many subjects to sample on each
# bootstrap iteration:
varNumBooSmp = varNumSub

# Random array with subject indicies for bootstrapping of the form
# aryRnd[varNumIt, varNumBooSmp]. Each row includes the indicies of the
# subjects to the sampled on that iteration.
aryRnd = np.random.randint(0,
                           high=varNumSub,
                           size=(varNumIt, varNumBooSmp))
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Parent loop

# Check whether dataframe with correct number of samples already exists; if
# yes, load from disk.
strPthDf = strPthDf.format(str(varNumSmpl))

if os.path.isfile(strPthDf):

    print('--Load PSF parameters from dataframe.')

    # Load existing dataframe:
    objDf = pd.read_pickle(strPthDf)

else:

    print('-Estimate cortical depth point spread function')

    # Dataframe for PSF model parameters:
    objDf = pd.DataFrame(0.0, index=np.arange(varNumSmpl), columns=lstFtr)

    # Loop through ROIs, conditions, depth levels:
    for idxRoi in range(varNumRoi):
        for idxCon in range(varNumCon):
            for idxDpth in range(varNumDpth):  #noqa

                # Estimate PSF parameters.

                # When processing the lowest depth level, the corresponding
                # visual field projection has to bee returned (because it will
                # act as the reference for subsequent depth levels). Otherwise,
                # only the dataframe with the PSF parameters is returned (and
                # the reference visual field projection is passed into the
                # function).

                if idxDpth == 0:

                    objDf, aryDeep, aryGrpDeep, aryDeepNorm = estm_psf(
                        idxRoi, idxCon, idxDpth, objDf, lstRoi, lstCon,
                        lstDpthLbl, strPthNpz, vecInit, lstBnds, strPthPltVfp,
                        varNumIt, varSzeVsm, strFlTp, varNumSub, aryRnd,
                        varScl, varConLw, varConUp, None, None, None, idxSmpl)

                else:

                    objDf = estm_psf(idxRoi, idxCon, idxDpth, objDf, lstRoi,
                                     lstCon, lstDpthLbl, strPthNpz, vecInit,
                                     lstBnds, strPthPltVfp, varNumIt,
                                     varSzeVsm, strFlTp, varNumSub, aryRnd,
                                     varScl, varConLw, varConUp, aryDeep,
                                     aryGrpDeep, aryDeepNorm, idxSmpl)

                # Increment counter for dataframe sample index:
                idxSmpl += 1

    # Save dataframe to pickle:
    objDf.to_pickle(strPthDf)


# -----------------------------------------------------------------------------
# *** Plot results

if not (strPthPltOt is None):

    # -------------------------------------------------------------------------
    # ** PSF width by depth & condition

    for idxRoi in range(varNumRoi):
        for idxCon in range(varNumCon):

            # Output file name:
            strFleNme = ('PSF_width_by_depth_'
                         + lstRoi[idxRoi]
                         + '_'
                         + lstCon[idxCon]
                         + strFlTp)

            # Output path:
            strPthTmp = (strPthPltOt.format(strFleNme))

            # Select data corrsponding to current ROI and condition:
            objLgcRoi = ((objDf["ROI"] == idxRoi)
                         & (objDf["Condition"] == idxCon))

            # Get PSF width from relevant section of dataframe (current ROI and
            # condition):
            vecSd = objDf[objLgcRoi]["Width"].values
            vecSdLow = objDf[objLgcRoi]["Width CI low"].values
            vecSdUp = objDf[objLgcRoi]["Width CI up"].values

            # Matplotlib `yerr` are +/- sizes relative to the data, therefore
            # subtract data points from error values:
            vecSdLow = np.subtract(vecSd, vecSdLow)
            vecSdUp = np.subtract(vecSdUp, vecSd)

            # Adjust shape for matplotlib:
            arySdErr = np.array([vecSdLow, vecSdUp])

            # Number of bars (one less than number of depth levels, because
            # there are no parameters for the reference depth level).
            vecX = np.arange(0, (varNumDpth - 1))

            varSizeX = 700
            varSizeY = 700
            varDpi = 80

            # Create figure:
            fgr01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                                        (varSizeY * 0.5) / varDpi),
                               dpi=varDpi)

            # Create axis:
            axs01 = fgr01.subplots(1, 1)

            # Create plot:
            plot01 = axs01.bar(vecX, vecSd, yerr=arySdErr)

            # Save figure:
            fgr01.savefig(strPthTmp)


# -----------------------------------------------------------------------------
# *** Save PSF parameters to CSV file

# Create dataframe & save csv?
if (not (strPthCsv is None)):

    print('--Saving dataframe to csv.')

    # Activate the pandas conversion), for conversions of pandas to R objects:
    pandas2ri.activate()

    # We use an R function from python to write the dataframe to a csv file.
    # Get reference to R function:
    fncR = robjects.r('write.csv')

    # Save csv to disk (using R function):
    fncR(objDf, strPthCsv)

# # Alternative grid search implementation:
#
# varNum = 100
#
# vecSd = np.linspace(0.0, 50.0, num=varNum)
# vecFct = np.linspace(0.0, 5.0, num=varNum)
#
# aryRes = np.zeros((varNum, varNum))
#
# for idxSd in range(varNum):
#     print(idxSd)
#     for idxFct in range(varNum):
#         aryRes[idxSd, idxFct] = psf_diff_02(vecSd[idxSd],
#                                             vecFct[idxFct],
#                                             ary01,
#                                             ary02)
#
# tplIdxMin = np.unravel_index(aryRes.argmin(), aryRes.shape)
# varFitSd = vecSd[tplIdxMin[0]]
# varFitFct = vecFct[tplIdxMin[1]]
