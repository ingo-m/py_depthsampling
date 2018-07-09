# -*- coding: utf-8 -*-
"""
Calculate similarity between visual field projections and stimulus model.

Fit an explicit model of a the spatial extent of stimulus-evoked activation (in
visual space) to empirically observed activation patterns.
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
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
from scipy.optimize import minimize
from py_depthsampling.psf_2D.utilities_stim_model import psf_stim_mdl
from py_depthsampling.psf_2D.utilities_stim_model import psf_diff_stim_mdl
from py_depthsampling.psf_2D.utilities_stim_model import plot_psf_params
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from py_depthsampling.project.plot import plot as plot_vfp


# -----------------------------------------------------------------------------
# *** Define parameters

# Load projection from (ROI, condition, depth level label left open):
strPthNpy = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/project/{}_{}_{}.npy'  #noqa

# Depth level labels (to complete input file names).
lstDpthLbl = [str(x) for x in range(11)]

# ROI ('v1','v2', or 'v3'):
lstRoi = ['v1', 'v2', 'v3']

# Output path & prefix for summary plots (dependent and independent variable
# left open). Set to `None` if plot should not be created.
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/psf_2D_pe_stim_model/PSF_{}_by_{}'  #noqa

# Output path & prefix for plots of modelled visual field projections. Set to
# `None` if plot should not be created.
strPthPltVfp = '/home/john/Dropbox/PacMan_Plots/psf_2D_pe_stim_model/{}'

# File type suffix for plot:
# strFlTp = '.svg'
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Condition levels (used to complete file names):
# lstCon = ['Ps_sst']
lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst']

# Only fit in left side of visual field?
lgcLftOnly = True

# Initial guess for PSF parameters (width and scaling factor for stimulus
# centre, stimulus edge, and periphery; SD in degree of visual angle):
varInitSd = 1.0
varInitFctCntr = 1.0
varInitFctEdge = 1.0
varInitFctPeri = 1.0

# Limits for PSF parameters [SD is in degrees of visual angle]:
tplBndSd = (0.0, 5.0)
tplBndCntr = (-50.0, 50.0)
tplBndEdge = (-50.0, 50.0)
tplBndPeri = (-50.0, 50.0)

# Save result from model fitting (i.e. parameters of PSF) to disk (pandas data
# frame saved as csv for import in R). If `None`, data frame is not created.
strPthCsv = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/psf_2D_stim_model/dataframe.csv'  #noqa

# Extent of visual space from centre of the screen (assumed to be the same in
# positive/negative x/y direction:
varExtMin = -5.19
varExtMax = 5.19
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Preparations

# Get dimension of visual space model (assumed to be the same for x and y
# directions, and for all ROIs/conditions/depth levels):
aryTmp = np.load(strPthNpy.format(lstRoi[0], lstCon[0], lstDpthLbl[0]))
varSzeVsm = aryTmp.shape[0]

# Scaling factor from degrees of visual angle to array dimensions:
varScl = (float(varSzeVsm) / float(varExtMax))

# Scale Gaussian SD from degree of visual angle to array dimensions:
varInitSd = (varInitSd * varScl)
tplBndSd = ((tplBndSd[0] * varScl), (tplBndSd[1] * varScl))

# Bring initial values and bounds into shape expected by scipy optimize:
vecInit = np.array([varInitSd, varInitFctCntr, varInitFctEdge, varInitFctPeri])
lstBnds = [tplBndSd, tplBndCntr, tplBndEdge, tplBndPeri]

# Number of ROIs/conditions/depths:
varNumDpth = len(lstDpthLbl)
varNumRoi = len(lstRoi)
varNumCon = len(lstCon)

# Total number of samples for dataframe:
varNumSmpl = (varNumRoi * varNumCon * varNumDpth)

# Feature list (column names for dataframe):
lstFtr = ['ROI', 'Condition', 'Depth', 'Width', 'PSC centre', 'PSC edge',
          'PSC periphery', 'Residuals']

# Dataframe for PSF model parameters:
objDf = pd.DataFrame(0.0, index=np.arange(varNumSmpl), columns=lstFtr)

# Coutner for dataframe samples:
idxSmpl = 0
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Prepare stimulus masks

# Vector with visual space x-coordinates:
vecVslX = np.linspace(varExtMin, varExtMax, num=varSzeVsm,
                      endpoint=True)

# Vector with visual space y-coordinates:
vecVslY = np.linspace(varExtMin, varExtMax, num=varSzeVsm,
                      endpoint=True)

# Eccentricity map of visual space:
aryEcc = np.sqrt(
                 np.add(
                        np.power(vecVslX[:, None], 2.0),
                        np.power(vecVslY[None, :], 2.0)
                        )
                 )

# Polar angle map of visual space:
aryPol = np.arctan2(vecVslY[:, None], vecVslX[None, :])

# Array representing PacMan shape (binary mask):
aryPacMan = np.multiply(
                        np.less_equal(aryEcc, 3.75),
                        np.logical_or(
                                      np.less(aryPol, np.deg2rad(-35.0)),
                                      np.greater(aryPol, np.deg2rad(35.0))
                                      )
                        ).astype(np.float64)

# Get binary mask of PacMan edge using gradient of PacMan:
lstGrd = np.gradient(aryPacMan.astype(np.float64))
aryEdge = np.greater(
                     np.add(
                            np.absolute(lstGrd[0]),
                            np.absolute(lstGrd[1])
                            ),
                     0.0).astype(np.float64)

# Scale width of edge mask (so that it becomes less dependent on resolution of
# visual field model):
varEdgeWidth = 0.01
varSd = varEdgeWidth * float(varSzeVsm)
aryEdge = gaussian_filter(aryEdge, varSd, order=0, mode='nearest',
                          truncate=4.0)
aryEdge = np.greater(aryEdge, 0.075).astype(np.float64)

# Visual space outside of stimulus:
aryPeri = np.less(np.add(aryPacMan, aryEdge), 0.1).astype(np.float64)

# Remove overlapping part between PacMan centre & edge (so that the masks are
# mutually exclusive):
aryPacMan = np.greater(np.subtract(aryPacMan, aryEdge),
                       0.0).astype(np.float64)

# At this point, the reference frame of the masks does not fit that of the
# empirical visual field projections. Rotate the masks to fit empirical data:
aryPacMan = np.rot90(aryPacMan, k=3)
aryEdge = np.rot90(aryEdge, k=3)
aryPeri = np.rot90(aryPeri, k=3)

# Set right side of visual field to zero:
if lgcLftOnly:
    aryPacMan[np.greater_equal(vecVslX, 0.0), :] = 0.0
    aryEdge[np.greater_equal(vecVslX, 0.0), :] = 0.0
    aryPeri[np.greater_equal(vecVslX, 0.0), :] = 0.0
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Parent loop

print('-Estimate cortical depth point spread function')

# Loop through ROIs, conditions, depth levels:
for idxRoi in range(varNumRoi):
    for idxCon in range(varNumCon):
        for idxDpth in range(varNumDpth):  #noqa

            # File name of npy file for current condition:
            strPthNpyTmp = strPthNpy.format(lstRoi[idxRoi],
                                            lstCon[idxCon],
                                            lstDpthLbl[idxDpth])

            # Load visual field projection:
            aryTrgt = np.load(strPthNpyTmp)

            # Set right side of visual field to zero:
            if lgcLftOnly:
                aryTrgt[np.greater_equal(vecVslX, 0.0), :] = 0.0

            # Fit point spread function:
            dicOptm = minimize(psf_diff_stim_mdl,
                               vecInit,
                               args=(aryPacMan, aryEdge, aryPeri, aryTrgt,
                                     lgcLftOnly, vecVslX),
                               bounds=lstBnds)

            # Fitted model parameters:
            tplFit = (dicOptm.x[0], dicOptm.x[1], dicOptm.x[2], dicOptm.x[3])

            # Calculate sum of model residuals:
            varTmpRes = psf_diff_stim_mdl(tplFit,
                                          aryPacMan,
                                          aryEdge,
                                          aryPeri,
                                          aryTrgt,
                                          lgcLftOnly,
                                          vecVslX)

            # Convert width from array indices to degrees of visual angle:
            varTmpSd = (dicOptm.x[0] / varScl)

            # Features to dataframe:
            objDf.at[idxSmpl, 'ROI'] = idxRoi
            objDf.at[idxSmpl, 'Condition'] = idxCon
            objDf.at[idxSmpl, 'Depth'] = idxDpth
            objDf.at[idxSmpl, 'Width'] = varTmpSd
            objDf.at[idxSmpl, 'PSC centre'] = dicOptm.x[1]
            objDf.at[idxSmpl, 'PSC edge'] = dicOptm.x[2]
            objDf.at[idxSmpl, 'PSC periphery'] = dicOptm.x[3]
            objDf.at[idxSmpl, 'Residuals'] = varTmpRes

            idxSmpl += 1

            # Plot visual field projection after applying PSF:
            if not(strPthPltVfp is None):

                # Apply fitted parameters to reference visual field projection:
                aryFit = psf_stim_mdl(aryPacMan, aryEdge, aryPeri,
                                      dicOptm.x[0], dicOptm.x[1], dicOptm.x[2],
                                      dicOptm.x[3])

                # Set right side of visual field to zero:
                if lgcLftOnly:
                    aryFit[np.greater_equal(vecVslX, 0.0), :] = 0.0

                # Output path for plot:
                strPthPltOtTmp = (strPthPltVfp.format((lstRoi[idxRoi]
                                                       + '_'
                                                       + lstCon[idxCon]
                                                       + '_'
                                                       + lstDpthLbl[idxDpth]))
                                  + strFlTp)

                # Plot title:
                strTmpTtl = (lstRoi[idxRoi]
                             + ' '
                             + lstCon[idxCon]
                             + ' '
                             + lstDpthLbl[idxDpth])

                # Create plot:
                plot_vfp(aryFit,
                         strTmpTtl,
                         'x-position',
                         'y-position',
                         strPthPltOtTmp,
                         tpleLimX=(-5.19, 5.19, 3.0),
                         tpleLimY=(-5.19, 5.19, 3.0))
# -----------------------------------------------------------------------------


print(objDf)

# -----------------------------------------------------------------------------
# *** Plot results

# Size of confidence intervals:
varCi = 90

# List of x and y variables for plot:
lstX = ['ROI'] * 5
lstY = ['Width', 'PSC centre', 'PSC edge', 'PSC periphery', 'Residuals']
lstHue = ['Depth'] * 5

for idxPlt in range(len(lstX)):

    # Plot results:
    plot_psf_params((strPthPltOt.format(lstY[idxPlt],
                                        lstHue[idxPlt]) + strFlTp),
                    lstX[idxPlt],
                    lstY[idxPlt],
                    lstHue[idxPlt],
                    objDf,
                    lstRoi,
                    varNumDpth,
                    varCi=varCi)

# List of x and y variables for plot:
lstX = ['ROI'] * 5
lstY = ['Width', 'PSC centre', 'PSC edge', 'PSC periphery', 'Residuals']
lstHue = ['Condition'] * 5

for idxPlt in range(len(lstX)):

    # Plot results:
    plot_psf_params((strPthPltOt.format(lstY[idxPlt],
                                        lstHue[idxPlt]) + strFlTp),
                    lstX[idxPlt],
                    lstY[idxPlt],
                    lstHue[idxPlt],
                    objDf,
                    lstRoi,
                    varNumDpth,
                    varCi=varCi)
# -----------------------------------------------------------------------------


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
# -----------------------------------------------------------------------------
