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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from py_depthsampling.psf_2D.utilities import psf
from py_depthsampling.psf_2D.utilities import psf_diff
from py_depthsampling.project.plot import plot
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri


# -----------------------------------------------------------------------------
# *** Define parameters

# Load projection from (ROI, condition, depth level label left open):
strPthNpz = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/project_single_subject/{}_{}_{}.npz'  #noqa

# Depth level labels (to complete input file names). First depth level in list
# is used as reference for estimation of point spread function.
lstDpthLbl = [str(x) for x in range(11)]

# ROI ('v1','v2', or 'v3'):
#lstRoi = ['v1', 'v2', 'v3']
lstRoi = ['v2']

# Output path & prefix for summary plots (file name left open). Set to `None`
# if plot should not be created.
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/psf_2D_pe/{}'  #noqa

# Output path & prefix for plots of visual field projections after application
# of fitted PSF (file name left open). Set to `None` if plot should not be
# created.
strPthPltVfp = '/home/john/Dropbox/PacMan_Plots/psf_2D_pe/{}'

# File type suffix for plot:
# strFlTp = '.svg'
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Condition levels (used to complete file names):
lstCon = ['Pd_sst', 'Ps_sst', 'Cd_sst']
lstCon = ['Ps_sst']

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
varNumIt = 1000

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

# Dataframe for PSF model parameters:
objDf = pd.DataFrame(0.0, index=np.arange(varNumSmpl), columns=lstFtr)

# Coutner for dataframe samples:
idxSmpl = 0

# Bootstrap parameters:

# We will sample subjects with replacement. How many subjects to sample on each
# iteration:
varNumSmp = varNumSub

# Random array with subject indicies for bootstrapping of the form
# aryRnd[varNumIt, varNumSmp]. Each row includes the indicies of the subjects
# to the sampled on that iteration.
aryRnd = np.random.randint(0,
                           high=varNumSub,
                           size=(varNumIt, varNumSmp))
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Parent loop

print('-Estimate cortical depth point spread function')

# Loop through ROIs, conditions, depth levels:
for idxRoi in range(varNumRoi):
    for idxCon in range(varNumCon):
        for idxDpth in range(varNumDpth):  #noqa

            print(('--ROI: '
                   + lstRoi[idxRoi]
                   + ' | Condition: '
                   + lstCon[idxCon]
                   + ' | Depth level: '
                   + lstDpthLbl[idxDpth]))

            # -----------------------------------------------------------------
            # *** Load data

            print('---Load data')

            # The first entry in the list of depth levels is assumed to be the
            # deepest depth level, and used as the reference for the estimation
            # of the point spread function.
            if idxDpth == 0:

                # File name of npz file for reference condition:
                strPthNpzTmp = strPthNpz.format(lstRoi[idxRoi],
                                                lstCon[idxCon],
                                                lstDpthLbl[idxDpth])

                # Load visual field projections. `aryVslSpc` contains single
                # subject visual field projections (shape: `aryVslSpc[idxSub,
                # x, y]`). `aryNorm` contains normalisation factors for visual
                # space projection (same shape as `aryVslSpc`).
                objNpz = np.load(strPthNpzTmp)
                aryDeep = objNpz['aryVslSpc']
                aryDeepNorm = objNpz['aryNorm']

                # Add up single subject visual field projections:
                aryGrpDeep = np.sum(aryDeep, axis=0)
                aryGrpDeepNorm = np.sum(aryDeepNorm, axis=0)

                # Normalise:
                aryGrpDeep = np.divide(aryGrpDeep, aryGrpDeepNorm)

            else:

                # File name of npz file for current condition:
                strPthNpzTmp = strPthNpz.format(lstRoi[idxRoi],
                                                lstCon[idxCon],
                                                lstDpthLbl[idxDpth])

                # Load visual field projections. `aryVslSpc` contains single
                # subject visual field projections (shape: `aryVslSpc[idxSub,
                # x, y]`). `aryNorm` contains normalisation factors for visual
                # space projection (same shape as `aryVslSpc`).
                objNpz = np.load(strPthNpzTmp)
                aryTrgt = objNpz['aryVslSpc']
                aryTrgtNorm = objNpz['aryNorm']

                # Add up single subject visual field projections:
                aryGrpTrgt = np.sum(aryTrgt, axis=0)
                aryGrpTrgtNorm = np.sum(aryTrgtNorm, axis=0)

                # Normalise:
                aryGrpTrgt = np.divide(aryGrpTrgt, aryGrpTrgtNorm)

                # -------------------------------------------------------------
                # *** Calculate group level PSF

                print('---Calculate group level PSF')

                # We first calculate the PSF on the full, empirical group
                # average visual field projection.

                # Fit point spread function:
                dicOptm = minimize(psf_diff,
                                   vecInit,
                                   args=(aryGrpDeep, aryGrpTrgt),
                                   bounds=lstBnds)

                # Calculate sum of model residuals:
                varTmpRes = psf_diff((dicOptm.x[0], dicOptm.x[1]),
                                     aryGrpDeep,
                                     aryGrpTrgt)

                # Convert width from array indices to degrees of visual angle:
                varTmpSd = (dicOptm.x[0] / varScl)

                # Features to dataframe:
                objDf.at[idxSmpl, 'ROI'] = idxRoi
                objDf.at[idxSmpl, 'Condition'] = idxCon
                objDf.at[idxSmpl, 'Depth'] = idxDpth
                objDf.at[idxSmpl, 'Width'] = varTmpSd
                objDf.at[idxSmpl, 'Scaling'] = dicOptm.x[1]
                objDf.at[idxSmpl, 'Residuals'] = varTmpRes

                # idxSmpl += 1

            # -----------------------------------------------------------------
            # *** Create plots

            if not(strPthPltVfp is None) and not(idxDpth == 0):

                # ** Plot least squares fit visual field projection

                # Apply fitted parameters to reference visual field projection:
                aryFit = psf(aryGrpDeep, dicOptm.x[0], dicOptm.x[1])

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
                plot(aryFit,
                     strTmpTtl,
                     'x-position',
                     'y-position',
                     strPthPltOtTmp,
                     tpleLimX=(-5.19, 5.19, 3.0),
                     tpleLimY=(-5.19, 5.19, 3.0),
                     varMin=-2.5,
                     varMax=2.5)

                # ** Plot residuals visual field projection

                # Calculate residuals:
                aryRes = np.subtract(aryGrpTrgt, aryFit)

                # Output path for plot:
                strPthPltOtTmp = (strPthPltVfp.format((lstRoi[idxRoi]
                                                       + '_'
                                                       + lstCon[idxCon]
                                                       + '_'
                                                       + lstDpthLbl[idxDpth]
                                                       + '_residuals'))
                                  + strFlTp)

                # Plot title:
                strTmpTtl = (lstRoi[idxRoi]
                             + ' '
                             + lstCon[idxCon]
                             + ' '
                             + lstDpthLbl[idxDpth])

                # Create plot:
                plot(aryRes,
                     strTmpTtl,
                     'x-position',
                     'y-position',
                     strPthPltOtTmp,
                     tpleLimX=(-5.19, 5.19, 3.0),
                     tpleLimY=(-5.19, 5.19, 3.0),
                     varMin=None,
                     varMax=None)

                # ** Plot residuals by PSC

                # PSC and residuals into datafram:
                objDfRes = pd.DataFrame(data=np.array([aryGrpDeep.flatten(),
                                                       aryRes.flatten()]).T,
                                        columns=['PSC', 'Residuals'])

                # Plot residuals vs. PSC:
                objAx = sns.regplot('PSC', 'Residuals', data=objDfRes,
                                    marker='.')
                objFgr = objAx.get_figure()

                # Output path for plot:
                strPthPltOtTmp = (strPthPltVfp.format((lstRoi[idxRoi]
                                                       + '_'
                                                       + lstCon[idxCon]
                                                       + '_'
                                                       + lstDpthLbl[idxDpth]
                                                       + '_residuals_by_PSC'))
                                  + strFlTp)

                # Save figure:
                objFgr.savefig(strPthPltOtTmp)
                # plt.clf(objFgr)
                plt.close(objFgr)

            # -----------------------------------------------------------------
            # *** Bootstrap confidence intervals

            if not(idxDpth == 0):

                print('---Bootstrap confidence intervals')

                # Arrays for bootstrap samples (visual field projection and
                # normalisation array, for reference and target depth levels):
                aryBooDeep = np.zeros((varNumIt, varNumSub, varSzeVsm,
                                       varSzeVsm))
                aryBooDeepNorm = np.zeros((varNumIt, varNumSub, varSzeVsm,
                                           varSzeVsm))
                aryBooTrgt = np.zeros((varNumIt, varNumSub, varSzeVsm,
                                       varSzeVsm))
                aryBooTrgtNorm = np.zeros((varNumIt, varNumSub, varSzeVsm,
                                           varSzeVsm))

                # Loop through bootstrap iterations:
                for idxIt in range(varNumIt):

                    # Indices of current bootstrap sample:
                    vecRnd = aryRnd[idxIt, :]

                    # Put current bootstrap sample into array:
                    aryBooDeep[idxIt, :, :, :] = aryDeep[vecRnd, :, :]
                    aryBooDeepNorm[idxIt, :, :, :] = aryDeepNorm[vecRnd, :, :]
                    aryBooTrgt[idxIt, :, :, :] = aryTrgt[vecRnd, :, :]
                    aryBooTrgtNorm[idxIt, :, :, :] = aryTrgtNorm[vecRnd, :, :]

                # Mean for each bootstrap sample (across subjects within the
                # bootstrap sample). Afterwards, arrays have the following
                # shape: `aryBoo*[varNumIt, varSzeVsm, varSzeVsm]`.
                aryBooDeep = np.mean(aryBooDeep, axis=1)
                aryBooDeepNorm = np.mean(aryBooDeepNorm, axis=1)
                aryBooTrgt = np.mean(aryBooTrgt, axis=1)
                aryBooTrgtNorm = np.mean(aryBooTrgtNorm, axis=1)

                # Normalise:
                aryBooDeep = np.divide(aryBooDeep, aryBooDeepNorm)
                aryBooTrgt = np.divide(aryBooTrgt, aryBooTrgtNorm)

                # Vectors for bootstrapping results:
                vecBooResSd = np.zeros(varNumIt)
                vecBooResFct = np.zeros(varNumIt)

                # The actual bootstrap PSF model fitting:
                for idxIt in range(varNumIt):

                    # Fit point spread function:
                    dicOptm = minimize(psf_diff,
                                       vecInit,
                                       args=(aryBooDeep[idxIt, :, :],
                                             aryBooTrgt[idxIt, :, :]),
                                       bounds=lstBnds)

                    # Convert width from array indices to degrees of visual
                    # angle:
                    varTmpSd = (dicOptm.x[0] / varScl)

                    # Bootstrapping results to vector:
                    vecBooResSd[idxIt] = varTmpSd
                    vecBooResFct[idxIt] = dicOptm.x[1]

                # Percentile bootstrap:
                vecPrctSd = np.percentile(vecBooResSd, (varConLw, varConUp))
                vecPrctFct = np.percentile(vecBooResFct, (varConLw, varConUp))

                # Features to dataframe:
                objDf.at[idxSmpl, 'Width CI low'] = vecPrctSd[0]
                objDf.at[idxSmpl, 'Width CI up'] = vecPrctSd[1]
                objDf.at[idxSmpl, 'Scaling CI low'] = vecPrctFct[0]
                objDf.at[idxSmpl, 'Scaling CI up'] = vecPrctFct[1]

                idxSmpl += 1




strPthDf = '/home/john/Desktop/tmp/objDf.pickle'
objDf.to_pickle(strPthDf)
# objDf = pd.read_pickle(strPthDf)


# -----------------------------------------------------------------------------
# *** Plot results

if not (strPthPltOt is None):

#    # -------------------------------------------------------------------------
#    # ** PSF width by condition
#
#    # Output path:
#    strPthTmp = (strPthPltOt.format('PSF_width_by_condition') + strFlTp)
#
#    # Create seaborn colour palette:
#    lstClr = ["amber", "greyish", "faded green"]
#    objClr = sns.xkcd_palette(lstClr)
#
#    # Draw nested barplot:
#    fgr01 = sns.factorplot(x="ROI", y="Width", hue="Condition", data=objDf,
#                           size=6, kind="bar", palette=objClr, ci=varCi)
#
#    # Set x-axis labels to upper case ROI labels:
#    lstRoiUp = [x.upper() for x in lstRoi]
#    fgr01.set_xticklabels(lstRoiUp)
#
#    # Set hue labels (i.e. condition labels in legend):
#    for objTxt, strLbl in zip(fgr01._legend.texts, lstCon):
#        objTxt.set_text(strLbl)
#
#    # Save figure:
#    fgr01.savefig(strPthTmp)

#    # -------------------------------------------------------------------------
#    # ** PSF width by depth
#
#    # Output path:
#    strPthTmp = (strPthPltOt.format('PSF_width_by_depth') + strFlTp)
#
#    # Create seaborn colour palette:
#    objClr = sns.light_palette((210, 90, 60), input="husl",
#                               n_colors=varNumDpth)
#
#    # Draw nested barplot:
#    fgr02 = sns.factorplot(x="ROI", y="Width", hue="Depth", data=objDf, size=6,
#                           kind="bar", legend=True, palette=objClr, ci=varCi)
#
#    fgr02.set_xticklabels(lstRoiUp)
#
#    # Save figure:
#    fgr02.savefig(strPthTmp)

    # -------------------------------------------------------------------------
    # ** PSF width by depth & condition


    for idxRoi in range(varNumRoi):
        for idxCon in range(varNumCon):

            # Output file name:
            strFleNme = (lstRoi[idxRoi]
                         + '_'
                         + lstCon[idxCon]
                         + '_PSF_width_by_depth'
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

#    # -------------------------------------------------------------------------
#    # ** PSF factor by condition
#
#    # Output path:
#    strPthTmp = (strPthPltOt.format('PSF_factor_by_condition') + strFlTp)
#
#    # Create seaborn colour palette:
#    colors = ["amber", "greyish", "faded green"]
#    objClr = sns.xkcd_palette(colors)
#
#    # Draw nested barplot:
#    fgr01 = sns.factorplot(x="ROI", y="Scaling", hue="Condition", data=objDf,
#                           size=6, kind="bar", palette=objClr, ci=varCi)
#
#    # Set x-axis labels to upper case ROI labels:
#    lstRoiUp = [x.upper() for x in lstRoi]
#    fgr01.set_xticklabels(lstRoiUp)
#
#    # Set hue labels (i.e. condition labels in legend):
#    for objTxt, strLbl in zip(fgr01._legend.texts, lstCon):
#        objTxt.set_text(strLbl)
#
#    # Save figure:
#    fgr01.savefig(strPthTmp)

#    # -------------------------------------------------------------------------
#    # ** PSF factor by depth
#
#    # Output path:
#    strPthTmp = (strPthPltOt.format('PSF_factor_by_depth') + strFlTp)
#
#    # Create seaborn colour palette:
#    objClr = sns.light_palette((210, 90, 60), input="husl",
#                               n_colors=varNumDpth)
#
#    # Draw nested barplot:
#    fgr02 = sns.factorplot(x="ROI", y="Scaling", hue="Depth", data=objDf,
#                           size=6, kind="bar", legend=True, palette=objClr,
#                           ci=varCi)
#
#    fgr02.set_xticklabels(lstRoiUp)
#
#    # Save figure:
#    fgr02.savefig(strPthTmp)

#    # -------------------------------------------------------------------------
#    # ** Residuals by condition
#
#    # Output path:
#    strPthTmp = (strPthPltOt.format('Residuals_by_condition') + strFlTp)
#
#    # Create seaborn colour palette:
#    colors = ["amber", "greyish", "faded green"]
#    objClr = sns.xkcd_palette(colors)
#
#    # Draw nested barplot:
#    fgr01 = sns.factorplot(x="ROI", y="Residuals", hue="Condition", data=objDf,
#                           size=6, kind="bar", palette=objClr, ci=varCi)
#
#    # Set x-axis labels to upper case ROI labels:
#    lstRoiUp = [x.upper() for x in lstRoi]
#    fgr01.set_xticklabels(lstRoiUp)
#
#    # Set hue labels (i.e. condition labels in legend):
#    for objTxt, strLbl in zip(fgr01._legend.texts, lstCon):
#        objTxt.set_text(strLbl)
#
#    # Save figure:
#    fgr01.savefig(strPthTmp)
#
#    # -------------------------------------------------------------------------
#    # ** Residuals by depth
#
#    # Output path:
#    strPthTmp = (strPthPltOt.format('Residuals_by_depth') + strFlTp)
#
#    # Create seaborn colour palette:
#    objClr = sns.light_palette((210, 90, 60), input="husl",
#                               n_colors=varNumDpth)
#
#    # Draw nested barplot:
#    fgr02 = sns.factorplot(x="ROI", y="Residuals", hue="Depth", data=objDf,
#                           size=6, kind="bar", legend=True, palette=objClr,
#                           ci=varCi)
#
#    fgr02.set_xticklabels(lstRoiUp)
#
#    # Save figure:
#    fgr02.savefig(strPthTmp)
#
#    # -------------------------------------------------------------------------
#    # ** Residuals by depth & condition
#
#    # Output path:
#    strPthTmp = (strPthPltOt.format('Residuals_by_depth_and_cond') + strFlTp)
#
#    # Create seaborn colour palette:
#    objClr = sns.light_palette((210, 90, 60), input="husl",
#                               n_colors=varNumDpth)
#
#    # Draw nested barplot:
#    fgr02 = sns.factorplot(x="ROI", y="Residuals", hue="Depth", data=objDf,
#                           size=6, kind="bar", legend=True, palette=objClr,
#                           ci=varCi, col="Condition")
#
#    # Set column titles:
#    for objAx, strTtl in zip(fgr02.axes.flat, lstCon):
#        objAx.set_title(strTtl)
#
#    fgr02.set_xticklabels(lstRoiUp)
#
#    # Save figure:
#    fgr02.savefig(strPthTmp)
#
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
