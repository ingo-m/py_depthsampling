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
import multiprocessing as mp
from scipy.optimize import minimize
from py_depthsampling.psf_2D.utilities_stim_model import psf_stim_mdl
from py_depthsampling.psf_2D.utilities_stim_model import psf_diff_stim_mdl
from py_depthsampling.project.plot import plot as plot_vfp
from py_depthsampling.psf_2D.psf_stim_model_estimate_par import est_par


def estm_psf_stim_mdl(idxRoi, idxCon, idxDpth, idxSmpl, lstRoi, lstCon,  #noqa
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
                 varMin=-3.0,
                 varMax=3.0)

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
    aryBooTrgt = np.sum(aryBooTrgt, axis=1)
    aryBooTrgtNorm = np.sum(aryBooTrgtNorm, axis=1)

    # Normalise:
    aryBooTrgt = np.divide(aryBooTrgt, aryBooTrgtNorm)

    # Crop visual field (only keep left hemifield):
    if lgcLftOnly:
        aryBooTrgt = aryBooTrgt[:, 0:varMrdnV, :]

    # ** Parallelised bootstrap

    # Number of processes to run in parallel:
    varPar = 11

    # Split data for parallel processing (list elements have shape
    # aryBooTrgt[varNumItChnk, varSzeVsm, varSzeVsm], where varNumItChnk are
    # the number of bootstrap samples per chunk).
    lstBooTrgt = np.array_split(aryBooTrgt, varPar, axis=0)
    del(aryBooTrgt)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for processes:
    lstPrcs = [None] * varPar

    # Empty list for results of parallel processes:
    lstRes = [None] * varPar

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=est_par,
                                     args=(idxPrc,
                                           lstBooTrgt[idxPrc],
                                           vecInit,
                                           aryPacMan,
                                           aryEdge,
                                           aryPeri,
                                           vecVslX,
                                           lstBnds,
                                           varScl,
                                           queOut)
                                     )

        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstRes[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    del(lstBooTrgt)

    # List for bootstrapping distributions (for PSF width, and scaling
    # factors for stimulu centre, edge, and periphery):
    lstBooSd = []
    lstBooFctCntr = []
    lstBooFctEdge = []
    lstBooFctPeri = []

    # Append bootstrap results to lists:
    for idxPrc in range(0, varPar):

        # The list with results from parallel processes has the shape:
        # lstOut[idxPrc, vecBooSd, vecBooFctCntr, vecBooFctEdge, vecBooFctPeri]
        # One such list per process is contained in `lstRes`.
        lstOutTmp = lstRes[idxPrc]
        lstBooSd.append(lstOutTmp[1])
        lstBooFctCntr.append(lstOutTmp[2])
        lstBooFctEdge.append(lstOutTmp[3])
        lstBooFctPeri.append(lstOutTmp[4])

    # Bootstrap distribution from list to array:
    vecBooSd = np.concatenate(lstBooSd, axis=0)
    vecBooFctCntr = np.concatenate(lstBooFctCntr, axis=0)
    vecBooFctEdge = np.concatenate(lstBooFctEdge, axis=0)
    vecBooFctPeri = np.concatenate(lstBooFctPeri, axis=0)

    # -------------------------------------------------------------------------
    # *** Return

    return objDf, vecBooSd, vecBooFctCntr, vecBooFctEdge, vecBooFctPeri
# -----------------------------------------------------------------------------
