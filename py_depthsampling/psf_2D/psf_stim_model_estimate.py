# -*- coding: utf-8 -*-
"""Calculate similarity between visual field projections and stimulus model."""

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
import pandas as pd  #noqa
from scipy.optimize import minimize
from py_depthsampling.psf_2D.utilities_stim_model import psf_stim_mdl
from py_depthsampling.psf_2D.utilities_stim_model import psf_diff_stim_mdl
from py_depthsampling.project.plot import plot as plot_vfp


def estm_psf_stim_mdl(idxRoi, idxCon, idxDpth, idxSmpl, lstRoi, lstCon,
                      lstDpthLbl, lgcLftOnly, varMrdnV, vecInit, aryPacMan,
                      aryEdge, aryPeri, vecVslX, lstBnds, varScl, tplDim,
                      strPthVfp, strPthPltVfp, strFlTp, aryRnd, varNumIt,
                      varConLw, varConUp, varNumSub, varSzeVsm, objDf):
    """
    Calculate similarity between visual field projections and stimulus model.

    See `psf_from_stim_model_main.py` for details.
    """
    print(('--ROI: '
           + lstRoi[idxRoi]
           + ' | Condition: '
           + lstCon[idxCon]
           + ' | Depth level: '
           + lstDpthLbl[idxDpth]))

    # -------------------------------------------------------------------------
    # *** Load data

    print('---Load visual field projection.')

    # File name of npy file for current condition:
    strPthVfpTmp = strPthVfp.format(lstRoi[idxRoi],
                                    lstCon[idxCon],
                                    lstDpthLbl[idxDpth])

    # Load visual field projections. `aryVslSpc` contains single subject visual
    # field projections (shape: `aryVslSpc[idxSub, x, y]`). `aryNorm` contains
    # normalisation factors for visual field projection (same shape as
    # `aryVslSpc`).
    objNpz = np.load(strPthVfpTmp)
    aryTrgt = objNpz['aryVslSpc']
    aryTrgtNorm = objNpz['aryNorm']

    # Add up single subject visual field projections:
    aryGrpTrgt = np.sum(aryTrgt, axis=0)
    aryGrpTrgtNorm = np.sum(aryTrgtNorm, axis=0)

    # Normalise:
    aryGrpTrgt = np.divide(aryGrpTrgt, aryGrpTrgtNorm)

    # -------------------------------------------------------------------------
    # *** Calculate group level PSF

    print('---Calculate group level model fit.')

    # Crop visual field (only keep left hemifield):
    if lgcLftOnly:
        aryGrpTrgt = aryGrpTrgt[0:varMrdnV, :]

    # Fit point spread function:
    dicOptm = minimize(psf_diff_stim_mdl,
                       vecInit,
                       args=(aryPacMan, aryEdge, aryPeri, aryGrpTrgt,
                             vecVslX),
                       bounds=lstBnds)

    # Fitted model parameters:
    tplFit = (dicOptm.x[0], dicOptm.x[1], dicOptm.x[2], dicOptm.x[3])

    # Calculate sum of model residuals:
    varTmpRes = psf_diff_stim_mdl(tplFit,
                                  aryPacMan,
                                  aryEdge,
                                  aryPeri,
                                  aryGrpTrgt,
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

    # -------------------------------------------------------------------------
    # *** Create plots (modelled VFP and residuals)

    # Plot results?
    if not(strPthPltVfp is None):

        # ** Plot modelled visual field projection

        # Apply fitted parameters to reference visual field projection:
        aryFit = psf_stim_mdl(aryPacMan, aryEdge, aryPeri,
                              dicOptm.x[0], dicOptm.x[1], dicOptm.x[2],
                              dicOptm.x[3])

        # Set right side of visual field to zero:
        if lgcLftOnly:
            aryTmp = np.zeros(tplDim)
            aryTmp[0:varMrdnV, :] = aryFit
            aryFit = aryTmp

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
                 tpleLimY=(-5.19, 5.19, 3.0),
                 varMin=-2.5,
                 varMax=2.5)

        # ** Plot model residuals

        # Set right side of visual field to zero:
        if lgcLftOnly:
            aryTmp = np.zeros(tplDim)
            aryTmp[0:varMrdnV, :] = aryGrpTrgt
            aryGrpTrgt = aryTmp

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
        plot_vfp(aryRes,
                 strTmpTtl,
                 'x-position',
                 'y-position',
                 strPthPltOtTmp,
                 tpleLimX=(-5.19, 5.19, 3.0),
                 tpleLimY=(-5.19, 5.19, 3.0),
                 varMin=None,
                 varMax=None)

    # -------------------------------------------------------------------------
    # *** Bootstrap confidence intervals

    print('---Bootstrap confidence intervals')

    # Arrays for bootstrap samples (visual field projection and normalisation
    # array):
    aryBooTrgt = np.zeros((varNumIt, varNumSub, varSzeVsm, varSzeVsm))
    aryBooTrgtNorm = np.zeros((varNumIt, varNumSub, varSzeVsm, varSzeVsm))

    # Loop through bootstrap iterations:
    for idxIt in range(varNumIt):

        # Indices of current bootstrap sample:
        vecRnd = aryRnd[idxIt, :]

        # Put current bootstrap sample into array:
        aryBooTrgt[idxIt, :, :, :] = aryTrgt[vecRnd, :, :]
        aryBooTrgtNorm[idxIt, :, :, :] = aryTrgtNorm[vecRnd, :, :]

    # Sum over each bootstrap sample (across subjects within the bootstrap
    # sample). Afterwards, arrays have the following shape: `aryBoo*[varNumIt,
    # varSzeVsm, varSzeVsm]`.
    aryBooTrgt = np.sum(aryTrgt, axis=1)
    aryBooTrgtNorm = np.sum(aryTrgtNorm, axis=1)

    # Normalise:
    aryBooTrgt = np.divide(aryBooTrgt, aryBooTrgtNorm)

    # Vectors for bootstrapping distributions (for PSF width, and scaling
    # factors for stimulu centre, edge, and periphery):
    vecBooSd = np.zeros(varNumIt)
    vecBooFctCntr = np.zeros(varNumIt)
    vecBooFctEdge = np.zeros(varNumIt)
    vecBooFctPeri = np.zeros(varNumIt)

    # Crop visual field (only keep left hemifield):
    if lgcLftOnly:
        aryBooTrgt = aryBooTrgt[:, 0:varMrdnV, :]

    # The actual bootstrap PSF model fitting:
    for idxIt in range(varNumIt):

        # Fit point spread function:
        dicOptm = minimize(psf_diff_stim_mdl,
                           vecInit,
                           args=(aryPacMan, aryEdge, aryPeri,
                                 aryBooTrgt[idxIt, :, :], vecVslX),
                           bounds=lstBnds)

        # Convert width from array indices to degrees of visual angle:
        varTmpSd = (dicOptm.x[0] / varScl)

        # Bootstrapping results to vector:
        vecBooSd[idxIt] = varTmpSd
        vecBooFctCntr[idxIt] = dicOptm.x[1]
        vecBooFctEdge[idxIt] = dicOptm.x[2]
        vecBooFctPeri[idxIt] = dicOptm.x[3]

    # Percentile bootstrap confidence intervals:
    # vecPrctSd = np.percentile(vecBooSd, (varConLw, varConUp))
    # vecPrctFctCntr = np.percentile(vecBooFctCntr, (varConLw, varConUp))
    # vecPrctFctEdge = np.percentile(vecBooFctEdge, (varConLw, varConUp))
    # vecPrctFctPeri = np.percentile(vecBooFctPeri, (varConLw, varConUp))

    # -------------------------------------------------------------------------
    # *** Return

    return objDf, vecBooSd, vecBooFctCntr, vecBooFctEdge, vecBooFctPeri
# -----------------------------------------------------------------------------
