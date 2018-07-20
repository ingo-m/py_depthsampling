# -*- coding: utf-8 -*-
"""
Calculate similarity between visual field projections and stimulus model.

Fit an explicit model of a the spatial extent of stimulus-evoked activation (in
visual space) to empirically observed activation patterns. The empirical visual
field can be created using `py_depthsampling.project.project_main`.
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
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
from py_depthsampling.psf_2D.psf_stim_model_estimate import estm_psf_stim_mdl
from py_depthsampling.psf_2D.utilities_stim_model import plot_psf_params
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import seaborn as sns


# -----------------------------------------------------------------------------
# *** Define parameters

# PSF parameters are saved to or loaded to/from pickled dataframe. Also, the
# bootstrap distributions of PSF parameters (Gaussian width and scaling factor)
# are saved to or loaded from and npz file. Path of respective files (number of
# samples and iterations left open):
strPthData = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/psf_2D_stim_model/dataframe_{}_samples_{}_iterations'  #noqa

# Load visual field projection from (ROI, condition, depth level label left
# open):
strPthVfp = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/project_single_subject/{}_{}_{}.npz'  #noqa

# Depth level labels (to complete input file names).
lstDpthLbl = [str(x) for x in range(11)]
# lstDpthLbl = ['0']

# ROI ('v1','v2', or 'v3'):
lstRoi = ['v1', 'v2', 'v3']

# Output path & prefix for summary plots (dependent and independent variable
# left open). Set to `None` if plot should not be created.
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/psf_2D_pe_stim_model/PSF_{}_by_{}'  #noqa

# Output path & prefix for plots of modelled visual field projections. Set to
# `None` if plot should not be created.
strPthPltVfp = None  # '/home/john/Dropbox/PacMan_Plots/psf_2D_pe_stim_model/{}'  #noqa

# File type suffix for plot:
# strFlTp = '.svg'
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Condition levels (used to complete file names):
# lstCon = ['Ps_sst']
lstCon = ['Pd_sst', 'Ps_sst', 'Cd_sst']

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
strPthCsv = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/psf_2D_stim_model/dataframe_{}_samples_{}_iterations.csv'  #noqa

# Extent of visual space from centre of the screen (assumed to be the same in
# positive/negative x/y direction:
varExtMin = -5.19
varExtMax = 5.19

# Number of bootstrapping iterations:
varNumIt = 10

# Lower and upper bound of bootstrap confidence intervals:
varConLw = 5.0
varConUp = 95.0
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Preparations

# Get dimension of visual space model (assumed to be the same for x and y
# directions, and for all ROIs/conditions/depth levels):
objNpzTmp = np.load(strPthVfp.format(lstRoi[0], lstCon[0], lstDpthLbl[0]))
aryTmp = objNpzTmp['aryVslSpc']
varSzeVsm = aryTmp.shape[1]

# Get number of subjects:
varNumSub = aryTmp.shape[0]

# Scaling factor from degrees of visual angle to array dimensions:
varScl = (float(varSzeVsm) / float(varExtMax))

# Scale Gaussian SD from degree of visual angle to array dimensions:
varInitSd = (varInitSd * varScl)
tplBndSd = ((tplBndSd[0] * varScl), (tplBndSd[1] * varScl))

# Bring initial values and bounds into shape expected by scipy optimize:
vecInit = np.array([varInitSd, varInitFctCntr, varInitFctEdge, varInitFctPeri])
lstBnds = [tplBndSd, tplBndCntr, tplBndEdge, tplBndPeri]

# Number of depths, ROIs, and conditions:
varNumDpth = len(lstDpthLbl)
varNumRoi = len(lstRoi)
varNumCon = len(lstCon)

# Total number of samples for dataframe:
varNumSmpl = (varNumRoi * varNumCon * varNumDpth)

# Feature list (column names for dataframe):
lstFtr = ['ROI', 'Condition', 'Depth', 'Width', 'PSC centre', 'PSC edge',
          'PSC periphery', 'Residuals']

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

# Account for shape of PacMan centre (small filled square region around
# fixation):
aryBox = np.multiply(
                     np.less_equal(np.absolute(vecVslY), 0.5)[:, None],
                     np.less_equal(np.absolute(vecVslY), 0.5)[None, :]
                     )
aryPacMan[aryBox] = 1.0

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

# Crop visual field (only keep left hemifield):
if lgcLftOnly:
    # Remember original dimensions:
    tplDim = aryPacMan.shape
    # Index of vertical meridian:
    varMrdnV = int(np.around(float(tplDim[0] * 0.5)))
    # Crop visual field model:
    aryPacMan = aryPacMan[0:varMrdnV, :]
    aryEdge = aryEdge[0:varMrdnV, :]
    aryPeri = aryPeri[0:varMrdnV, :]
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Parent loop

# Complete paths for dataframe and npz files:
strPthDf = (strPthData.format(str(varNumSmpl), str(varNumIt)) + '.pickle')
strPthNpz = (strPthData.format(str(varNumSmpl), str(varNumIt)) + '.npz')

# Check whether dataframe with correct number of samples already exists; if
# yes, load from disk.
if os.path.isfile(strPthDf):

    print('--Load PSF parameters from dataframe.')

    # Load existing dataframe:
    objDf = pd.read_pickle(strPthDf)

    # Load bootstrap distribution of PSF parameters:
    objNpz = np.load(strPthNpz)
    aryBooResSd = objNpz['aryBooResSd']
    aryBooResFct = objNpz['aryBooResFct']

else:

    print('-Estimate cortical depth point spread function')

    # Dataframe for PSF model parameters:
    objDf = pd.DataFrame(0.0, index=np.arange(varNumSmpl), columns=lstFtr)

    # Loop through ROIs, conditions, depth levels:
    for idxRoi in range(varNumRoi):
        for idxCon in range(varNumCon):
            for idxDpth in range(varNumDpth):  #noqa

                # Estimate PSF parameters:
                objDf = estm_psf_stim_mdl(idxRoi, idxCon, idxDpth, idxSmpl,
                                          lstRoi, lstCon, lstDpthLbl,
                                          lgcLftOnly, varMrdnV, vecInit,
                                          aryPacMan, aryEdge, aryPeri, vecVslX,
                                          lstBnds, varScl, tplDim, strPthVfp,
                                          strPthPltVfp, strFlTp, objDf)

                # Increment counter for dataframe sample index:
                idxSmpl += 1

    # Save dataframe to pickle:
    objDf.to_pickle(strPthDf)

    # Save boostrap distribution of PSF parameters to npz file:
    np.savez(strPthNpz,
             aryBooResSd=aryBooResSd,
             aryBooResFct=aryBooResFct)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Plot results

# Size of confidence intervals:
varCi = 90

# ** Parameters by depth

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
                    varCi=varCi,
                    strClrmp="continuous")

# ** Parameters by condition

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
                    varNumCon,
                    varCi=varCi,
                    lstConLbls=lstCon,
                    strClrmp="categorical")

# ** PSF width by depth & condition

# Output path:
strPthTmp = (strPthPltOt.format('Width', 'Depth_and_Condition') + strFlTp)

# Create seaborn colour palette:
objClr = sns.light_palette((210, 90, 60), input="husl",
                           n_colors=varNumDpth)

# Draw nested barplot:
fgr02 = sns.factorplot(x="ROI", y="Width", hue="Depth", data=objDf, size=6,
                       kind="bar", legend=True, palette=objClr, ci=varCi,
                       col="Condition")

# Set column titles:
for objAx, strTtl in zip(fgr02.axes.flat, lstCon):
    objAx.set_title(strTtl)

# Set x-axis labels to upper case ROI labels:
lstRoiUp = [x.upper() for x in lstRoi]
fgr02.set_xticklabels(lstRoiUp)

# Save figure:
fgr02.savefig(strPthTmp)
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
