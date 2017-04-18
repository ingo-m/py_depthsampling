# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.
"""

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


def crf_fit(aryX,
            aryY):
    """
    Fit contrast response function.

    Parameters
    ----------
    aryX : np.array
        x-values at which model will be fitted (e.g. stimulus contrast),
        of the form aryX[idxSub, idxCon].
    aryY : np.array
        y-values to fit (e.g. response), same shape as aryX.
    

    Returns
    -------


    Notes
    -----


    Function of the depth sampling pipeline.
    """
    # *** Average across subjects
    

    
    # Number of subjects:
    # varNumSubs = lstDpth[idxIn].shape[0]

    # Number of conditions:
    varNumCon = vecX.shape[0]

    # Across-subjects mean for measured response:
    lstDpthMne[idxIn] = np.mean(lstDpth[idxIn], axis=0)

    # Standard error of the mean (across subjects):
    lstDpthSem[idxIn] = np.divide(np.std(lstDpth[idxIn], axis=0),
                                  np.sqrt(varNumSubs))
    
    
    # ----------------------------------------------------------------------------
    # *** Fit CRF across depth levels
    
    # Append function type to output file:
    strPthOt = (strPthOt + '_' + strFunc)
    
    # Number of x-values for which to solve the function:
    varNumX = 1000
    
    # Vector for which the function will be fitted:
    vecX = np.linspace(varXmin, varXmax, num=varNumX, endpoint=True)
    
    # List of vectors for y-values of fitted function (for each depth level):
    lstFit = [np.zeros((varNumDpth, varNumX)) for i in range(varNumIn)]
    
    # List of vectors for response at half maximum contrast:
    lstHlfMaxResp = [np.zeros((1, varNumDpth)) for i in range(varNumIn)]
    
    # List of vectors for contrast at half maximum response:
    lstHlfMaxCont = [np.zeros((1, varNumDpth)) for i in range(varNumIn)]
    
    # List of arrays for residual variance:
    lstRes = [np.zeros((varNumCon, varNumDpth)) for i in range(varNumIn)]
    
    # Loop through ROIs (i.e. V1 and V2):
    for idxIn in range(0, varNumIn):  #noqa
    
        # We fit the contrast response function separately for all depth levels.
    
        # Loop through depth levels:
        for idxDpth in range(0, varNumDpth):
    
            # --------------------------------------------------------------------
            # *** Fit contrast reponse function
    
            if strFunc == 'power':
                vecMdlPar, vecMdlCov = curve_fit(crf_power,
                                                 vecCon,
                                                 lstDpthMne[idxIn][:, idxDpth],
                                                 maxfev=100000,
                                                 bounds=(vecLimPowLw, vecLimPowUp),
                                                 p0=(0.5, 0.5))
    
            elif strFunc == 'hyper':
                vecMdlPar, vecMdlCov = curve_fit(crf_hyper,
                                                 vecCon,
                                                 lstDpthMne[idxIn][:, idxDpth],
                                                 maxfev=100000,
                                                 bounds=(vecLimHypLw, vecLimHypUp),
                                                 p0=(0.01, 0.01, 0.5))
    
            # --------------------------------------------------------------------
            # *** Apply reponse function
    
            # Calculate fitted y-values:
            if strFunc == 'power':
                lstFit[idxIn][idxDpth, :] = crf_power(vecX,
                                                      vecMdlPar[0],
                                                      vecMdlPar[1])
            elif strFunc == 'hyper':
                lstFit[idxIn][idxDpth, :] = crf_hyper(vecX,
                                                      vecMdlPar[0],
                                                      vecMdlPar[1],
                                                      vecMdlPar[2])
    
            # --------------------------------------------------------------------
            # *** Calculate response at half maximum contrast
    
            # The response at half maximum contrast (i.e. at a luminance contrast
            # of 50%):
            if strFunc == 'power':
                lstHlfMaxResp[idxIn][0, idxDpth] = crf_power(0.5,
                                                             vecMdlPar[0],
                                                             vecMdlPar[1])
            elif strFunc == 'hyper':
                lstHlfMaxResp[idxIn][0, idxDpth] = crf_hyper(0.5,
                                                             vecMdlPar[0],
                                                             vecMdlPar[1],
                                                             vecMdlPar[2])
    
            # --------------------------------------------------------------------
            # *** Calculate contrast at half maximum response
    
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
    
            # Initial value for the contrast level (will be incremented until the
            # half maximum response is reached).
            varHlfMaxCont = 0.0
    
            # Initial value for the resposne.
            varRespTmp = 0.0
    
            # Increment the contrast level until the half maximum response is
            # reached:
            while np.less(varRespTmp, varResp50):
                varHlfMaxCont += 0.000001
                if strFunc == 'power':
                    varRespTmp = crf_power(varHlfMaxCont,
                                           vecMdlPar[0],
                                           vecMdlPar[1])
                elif strFunc == 'hyper':
                    varRespTmp = crf_hyper(varHlfMaxCont,
                                           vecMdlPar[0],
                                           vecMdlPar[1],
                                           vecMdlPar[2])
            lstHlfMaxCont[idxIn][0, idxDpth] = varHlfMaxCont
    
            # --------------------------------------------------------------------
            # *** Calculate residual variance
    
            # In order to assess the fit of the model, we calculate the deviation
            # of the measured response from the fitted model (average across
            # conditions). First we have to calculate the deviation for each
            # condition.
            for idxCon in range(0, varNumCon):
    
                # Model prediction for current contrast level:
                if strFunc == 'power':
                    varTmp = crf_power(vecCon[idxCon],
                                       vecMdlPar[0],
                                       vecMdlPar[1])
                elif strFunc == 'hyper':
                    varTmp = crf_hyper(vecCon[idxCon],
                                       vecMdlPar[0],
                                       vecMdlPar[1],
                                       vecMdlPar[2])
    
                # Residual = absolute of difference between prediction and
                #            measurement
                lstRes[idxIn][idxCon, idxDpth] = \
                    np.absolute(np.subtract(lstDpthMne[idxIn][idxCon, idxDpth],
                                            varTmp))