# -*- coding: utf-8 -*-
"""Estimate cortical depth point spread function."""

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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from py_depthsampling.psf_2D.utilities import psf
from py_depthsampling.psf_2D.utilities import psf_diff
from py_depthsampling.project.plot import plot


def estm_psf(idxRoi, idxCon, idxDpth, objDf, lstRoi, lstCon, lstDpthLbl,
             strPthNpz, vecInit, lstBnds, strPthPltVfp, varNumIt, varSzeVsm,
             strFlTp, varNumSub, aryRnd, varScl, varConLw, varConUp,
             aryDeep, aryGrpDeep, aryDeepNorm, idxSmpl):
    """
    Estimate cortical depth point spread function.

    Parameters of the cortical depth point spread function are estimated for
    a given set of ROI, condition, and depth level. Bootstrap confidence
    intervals for parameters are estimated, and (optionally) plots of the
    fitted visual field projections and residuals are created.

    See `psf_2D_main.py` for details.
    """
    print(('--ROI: '
           + lstRoi[idxRoi]
           + ' | Condition: '
           + lstCon[idxCon]
           + ' | Depth level: '
           + lstDpthLbl[idxDpth]))

    # -------------------------------------------------------------------------
    # *** Load data

    print('---Load data')

    # The first entry in the list of depth levels is assumed to be the deepest
    # depth level, and used as the reference for the estimation of the point
    # spread function.
    if idxDpth == 0:

        # File name of npz file for reference condition:
        strPthNpzTmp = strPthNpz.format(lstRoi[idxRoi],
                                        lstCon[idxCon],
                                        lstDpthLbl[idxDpth])

        # Load visual field projections. `aryVslSpc` contains single subject
        # visual field projections (shape: `aryVslSpc[idxSub, x, y]`).
        # `aryNorm` contains normalisation factors for visual space projection
        # (same shape as `aryVslSpc`).
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

        # Load visual field projections. `aryVslSpc` contains single subject
        # visual field projections (shape: `aryVslSpc[idxSub, x, y]`).
        # `aryNorm` contains normalisation factors for visual space projection
        # (same shape as `aryVslSpc`).
        objNpz = np.load(strPthNpzTmp)
        aryTrgt = objNpz['aryVslSpc']
        aryTrgtNorm = objNpz['aryNorm']

        # Add up single subject visual field projections:
        aryGrpTrgt = np.sum(aryTrgt, axis=0)
        aryGrpTrgtNorm = np.sum(aryTrgtNorm, axis=0)

        # Normalise:
        aryGrpTrgt = np.divide(aryGrpTrgt, aryGrpTrgtNorm)

        # ---------------------------------------------------------------------
        # *** Calculate group level PSF

        print('---Calculate group level PSF')

        # We first calculate the PSF on the full, empirical group average
        # visual field projection.

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
        # varTmpSd = (dicOptm.x[0] / varScl)

        # Features to dataframe:
        objDf.at[idxSmpl, 'ROI'] = idxRoi
        objDf.at[idxSmpl, 'Condition'] = idxCon
        objDf.at[idxSmpl, 'Depth'] = idxDpth
        # objDf.at[idxSmpl, 'Width'] = varTmpSd
        # objDf.at[idxSmpl, 'Scaling'] = dicOptm.x[1]
        objDf.at[idxSmpl, 'Residuals'] = varTmpRes

    # -------------------------------------------------------------------------
    # *** Create plots

    if not(strPthPltVfp is None) and not(idxDpth == 0):

        # ** Plot least squares fit visual field projection

        # Apply fitted parameters to reference visual field projection:
        aryFit = psf(aryGrpDeep, dicOptm.x[0], dicOptm.x[1])

        # Output path for plot:
        strPthPltVfpTmp = (strPthPltVfp.format((lstRoi[idxRoi]
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
             strPthPltVfpTmp,
             tpleLimX=(-5.19, 5.19, 3.0),
             tpleLimY=(-5.19, 5.19, 3.0),
             varMin=-2.5,
             varMax=2.5)

        # ** Plot residuals visual field projection

        # Calculate residuals:
        aryRes = np.subtract(aryGrpTrgt, aryFit)

        # Output path for plot:
        strPthPltVfpTmp = (strPthPltVfp.format((lstRoi[idxRoi]
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
             strPthPltVfpTmp,
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

    # -------------------------------------------------------------------------
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

        # Median for each bootstrap sample (across subjects within the
        # bootstrap sample). Afterwards, arrays have the following shape:
        # `aryBoo*[varNumIt, varSzeVsm, varSzeVsm]`.
        aryBooDeep = np.median(aryBooDeep, axis=1)
        aryBooDeepNorm = np.median(aryBooDeepNorm, axis=1)
        aryBooTrgt = np.median(aryBooTrgt, axis=1)
        aryBooTrgtNorm = np.median(aryBooTrgtNorm, axis=1)

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

            # Convert width from array indices to degrees of visual angle:
            varTmpSd = (dicOptm.x[0] / varScl)

            # Bootstrapping results to vector:
            vecBooResSd[idxIt] = varTmpSd
            vecBooResFct[idxIt] = dicOptm.x[1]

        # Percentile bootstrap confidence intervals:
        vecPrctSd = np.percentile(vecBooResSd, (varConLw, varConUp))
        vecPrctFct = np.percentile(vecBooResFct, (varConLw, varConUp))

        # Features to dataframe:
        objDf.at[idxSmpl, 'Width'] = np.median(vecBooResSd)
        objDf.at[idxSmpl, 'Width CI low'] = vecPrctSd[0]
        objDf.at[idxSmpl, 'Width CI up'] = vecPrctSd[1]
        objDf.at[idxSmpl, 'Scaling'] = np.median(vecBooResFct)
        objDf.at[idxSmpl, 'Scaling CI low'] = vecPrctFct[0]
        objDf.at[idxSmpl, 'Scaling CI up'] = vecPrctFct[1]

    # -------------------------------------------------------------------------
    # *** Return

    # When processing the lowest depth level, the corresponding visual field
    # projection has to bee returned (because it will act as the reference
    # for subsequent depth levels). Otherwise, the dataframe with the PSF
    # parameters and vectors with the bootstrapping distribution of paramters
    # are returned (the latter is needed in order to calculate the across-
    # condition average).
    if idxDpth == 0:
        return objDf, aryDeep, aryGrpDeep, aryDeepNorm
    else:
        return objDf, vecPrctSd, vecPrctFct
# -----------------------------------------------------------------------------
