# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

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

import numpy as np
from scipy.optimize import curve_fit
from ds_crfFunc import crf_power
from ds_crfFunc import crf_hyper


def crf_fit(vecEmpX, aryEmpY, strFunc='power', varNumX=1000, varXmin=0.0,
            varXmax=1.0, strAvr='mean'):
    """
    Fit contrast response function.

    Parameters
    ----------
    vecEmpX : np.array
        Empirical x-values at which model will be fitted (e.g. stimulus
        contrast levels at which stimuli were presented), of the form
        vecEmpX[idxCon].
    aryEmpY : np.array
        Empirical y-values to fit (e.g. measured response), of the form
        aryEmpY[idxSub, idxCon]
    strFunc : str
        Which function to fit ('power' for power function, 'hyper' for
        hyperbolic ratio). See respective module for more details.
    varNumX : int
        Number of x-values for which to solve the function when calculating
        model fit.
    varXmin : float
        Minimum x-value for which function will be fitted.
    varXmax : float
        Maximum x-value for which function will be fitted.
    strAvr : str
        How to calculate the average of the y-values; 'mean' or 'median'.

    Returns
    -------
    vecMdlY : np.array
        Fitted y-values (predicted response based on CRF model), of the form
        vecMdlY[varNumX]
    varHlfMax : float
        Predicted response at 50 percent contrast based on CRF model.
    varSemi : float
        Semisaturation contrast (predicted contrast needed to elicit 50 percent
        of the response amplitude that would be expected with a 100 percent
        contrast stimulus).
    vecRes : np.array
        Residual variance at empirical contrast levels (of the form
        vecRes[varNumCon]).

    Notes
    -----
    Function of the depth sampling pipeline.
    """
    # *** Average across subjects

    # Number of subjects:
    # varNumSubs = aryEmpY.shape[0]

    # Number of conditions:
    varNumCon = vecEmpX.shape[0]

    if strAvr == 'mean':
        # Across-subjects mean for measured response:
        vecEmpYMne = np.mean(aryEmpY, axis=0)
    elif strAvr == 'median':
        # Across-subjects median for measured response:
        vecEmpYMne = np.median(aryEmpY, axis=0)

    # Standard error of the mean (across subjects):
    # aryEmpYSem = np.divide(np.std(aryEmpY, axis=0),
    #                               np.sqrt(varNumSubs))

    # *** Fit contrast reponse function

    # Vector for which the function will be fitted:
    vecMdlX = np.linspace(varXmin, varXmax, num=varNumX, endpoint=True)

    if strFunc == 'power':

        # Lower limits for parameters (factor, exponent) - for power function:
        vecLimPowLw = np.array([0.0, 0.0])

        # Upper limits for parameters (factor, exponent) - for power function:
        vecLimPowUp = np.array([10.0, 1.0])

        vecMdlPar, vecMdlCov = curve_fit(crf_power,
                                         vecEmpX,
                                         vecEmpYMne,
                                         maxfev=100000,
                                         bounds=(vecLimPowLw, vecLimPowUp),
                                         p0=(0.5, 0.5))

    elif strFunc == 'hyper':

        # Lower limits for parameters (maximum response, semisaturation
        # contrast, and exponent) - for hyperbolic function:
        vecLimHypLw = np.array([0.0, 0.0, 0.0])

        # Upper limits for parameters (maximum response, semisaturation
        # contrast, and exponent) - for hyperbolic function:
        vecLimHypUp = np.array([np.inf, np.inf, np.inf])

        vecMdlPar, vecMdlCov = curve_fit(crf_hyper,
                                         vecEmpX,
                                         vecEmpYMne,
                                         maxfev=100000,
                                         bounds=(vecLimHypLw, vecLimHypUp),
                                         p0=(0.01, 0.01, 0.5))

    # *** Apply reponse function

    # Calculate fitted y-values:
    if strFunc == 'power':
        vecMdlY = crf_power(vecMdlX,
                            vecMdlPar[0],
                            vecMdlPar[1])
    elif strFunc == 'hyper':
        vecMdlY = crf_hyper(vecMdlX,
                            vecMdlPar[0],
                            vecMdlPar[1],
                            vecMdlPar[2])

    # *** Calculate response at 50% contrast

    # The response at half maximum contrast (i.e. at a luminance contrast of
    # 50%):
    if strFunc == 'power':
        varHlfMax = crf_power(0.5,
                              vecMdlPar[0],
                              vecMdlPar[1])
    elif strFunc == 'hyper':
        varHlfMax = crf_hyper(0.5,
                              vecMdlPar[0],
                              vecMdlPar[1],
                              vecMdlPar[2])

    # *** Calculate semisaturation contrast

    # The maximum response (defined as the response at 100% luminance
    # contrast):
    if strFunc == 'power':
        varResp50 = crf_power(1.0,
                              vecMdlPar[0],
                              vecMdlPar[1])
    elif strFunc == 'hyper':
        varResp50 = crf_hyper(1.0,
                              vecMdlPar[0],
                              vecMdlPar[1],
                              vecMdlPar[2])

    # Half maximum response:
    varResp50 = np.multiply(varResp50, 0.5)

    # Search for the luminance contrast level at half maximum response. A
    # while loop is more practical than an analytic solution - it is easy
    # to implement and reliable because of the contraint nature of the
    # problem. The problem is contraint because the luminance contrast has
    # to be between zero and one.

    # Initial value for the semisaturation contrast (will be incremented until
    # the half maximum response is reached).
    varSemi = 0.0

    # Initial value for the resposne.
    varRespTmp = 0.0

    # Increment the contrast level until the half maximum response is
    # reached:
    while np.less(varRespTmp, varResp50):
        varSemi += 0.000001
        if strFunc == 'power':
            varRespTmp = crf_power(varSemi,
                                   vecMdlPar[0],
                                   vecMdlPar[1])
        elif strFunc == 'hyper':
            varRespTmp = crf_hyper(varSemi,
                                   vecMdlPar[0],
                                   vecMdlPar[1],
                                   vecMdlPar[2])

    # *** Calculate residual variance

    # Array for residual variance:
    vecRes = np.zeros(varNumCon)

    # In order to assess the fit of the model, we calculate the deviation of
    # the measured response from the fitted model (average across conditions).
    # First we have to calculate the deviation for each condition.
    for idxCon in range(0, varNumCon):

        # Model prediction for current contrast level:
        if strFunc == 'power':
            varTmp = crf_power(vecEmpX[idxCon],
                               vecMdlPar[0],
                               vecMdlPar[1])
        elif strFunc == 'hyper':
            varTmp = crf_hyper(vecEmpX[idxCon],
                               vecMdlPar[0],
                               vecMdlPar[1],
                               vecMdlPar[2])

        # Residual = absolute of difference between prediction and
        #            measurement
        vecRes[idxCon] = np.absolute(np.subtract(vecEmpYMne[idxCon], varTmp))

    return vecMdlY, varHlfMax, varSemi, vecRes
