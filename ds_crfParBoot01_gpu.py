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
import time
import theano as th
import theano.tensor as T


#aryDpth = np.load('/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy')
#aryDpth = np.array([aryDpth])
#vecEmpX = np.array([0.025, 0.061, 0.163, 0.72])
#strFunc='power'
#varNumIt=1000
#varNumX=1000


def crf_par_01_t(aryDpth, vecEmpX, strFunc='power', varNumIt=1000,
                 varNumX=1000, varXmin=0.0, varXmax=1.0):
    """
    Parallelised bootstrapping of contrast response function, level 1.

    Parameters
    ----------
    aryDpth : np.array
        Array with empirical response data, of the form
        aryDpth[idxRoi, idxSub, idxCon, idxDpt].
    vecEmpX : np.array
        Empirical x-values at which model will be fitted (e.g. stimulus
        contrast levels at which stimuli were presented), of the form
        vecEmpX[idxCon].
    strFunc : str
        Which contrast response function to fit. 'power' for power function, or
        'hyper' for hyperbolic ratio function.
    varNumIt : int
        Number of bootstrapping iterations (i.e. how many times to sample).
    varPar : int
        Number of process to run in parallel.
    varNumX : int
        Number of x-values for which to solve the function when calculating
        model fit.
    varXmin : float
        Minimum x-value for which function will be fitted.
    varXmax : float
        Maximum x-value for which function will be fitted.

    Returns
    -------
    aryMdlY : np.array
        Fitted y-values (predicted response based on CRF model), of the form
        aryMdlY[idxRoi, idxIteration, idxDpt, idxContrast]
    aryHlfMax : np.array
        Predicted response at 50 percent contrast based on CRF model. Array of
        the form aryHlfMax[idxRoi, idxIteration, idxDpt].
    arySemi : np.array
        Semisaturation contrast (predicted contrast needed to elicit 50 percent
        of the response amplitude that would be expected with a 100 percent
        contrast stimulus). Array of the form
        arySemi[idxRoi, idxIteration, idxDpt].
    aryRes : np.array
        Residual variance at empirical contrast levels. Array of the form
        aryRes[idxRoi, idxIteration, idxCondition, idxDpt].

    Notes
    -----
    NOTE: HYPERBOLIC RATIO NOT YET IMPLEMENTED FOR THEANO.
    
    This function parallelises the contrast response function fitting by
    calling a second-level function using the multiprocessing module.

    Function of the depth sampling pipeline.
    """
    # ------------------------------------------------------------------------
    # *** Prepare bootstrapping
    print('---Preparing bootstrapping')

    # Number of ROIs:
    varNumIn = aryDpth.shape[0]

    # Number of subjects:
    varNumSubs = aryDpth.shape[1]

    # Number of conditions:
    varNumCon = aryDpth.shape[2]

    # Number of depth levels:
    varNumDpth = aryDpth.shape[3]

    # Random array with subject indicies for bootstrapping of the form
    # aryRnd[varNumIt, varNumSmp]. Each row includes the indicies of the
    # subjects to the sampled on that iteration.
    aryRnd = np.random.randint(0,
                               high=varNumSubs,
                               size=(varNumIt, varNumSubs))

    ## Initialise array for random samples:
    #aryDpthRnd = np.zeros((varNumIt, varNumIn, varNumSubs, varNumCon, varNumDpth))
    #
    ## Fill array with resampled samples:
    #for idxIt in range(varNumIt):
    #    aryDpthRnd[idxIt, :, :, :, :] = aryDpth[:, aryRnd[idxIt, :], :, :]
    #
    ## Take mean within random samples:
    #aryDpthRnd = np.mean(aryDpthRnd, axis=2)
    #
    ## Total number of CRF models to fit:
    #varNumTtl = (varNumIn * varNumDpth * varNumIt)
    #
    ## Reshape:
    #aryDpthRnd = np.reshape(aryDpthRnd, (varNumTtl, varNumCon))

    # Total number of CRF models to fit:
    varNumTtl = (varNumIn * varNumDpth * varNumIt)

    # Array for resampled samples:
    aryDpthRnd = np.zeros((varNumTtl, varNumSubs, varNumCon))

    # Fill array with resampled samples:
    varCnt = 0
    for idxIn in range(varNumIn):
        for idxIt in range(varNumIt):
            for idxDpth in range(varNumDpth):
                aryDpthRnd[varCnt, :, :] = aryDpth[idxIn,
                                                   aryRnd[idxIt, :],
                                                   :,
                                                   idxDpth]
    
                varCnt += 1
    
    # Take mean within random samples:
    aryDpthRnd = np.mean(aryDpthRnd, axis=1)


    # ------------------------------------------------------------------------
    # *** Theano CRF fitting

    print('---Theano CRF fitting')

    # Boradcast array with X data, and change data type to float 32:
    aryEmpX = np.broadcast_to(vecEmpX, (varNumTtl, vecEmpX.shape[0]))
    aryEmpX = aryEmpX.astype(th.config.floatX)
    aryDpthRnd = aryDpthRnd.astype(th.config.floatX)

    # Check time:
    varTme01 = time.time()

    # The CRF:
    # varR = varA * np.power(varC, varB)
    def model(aryC, vecA, vecB):
       return T.mul(T.pow(aryC, vecB[:, None]), vecA[:, None])

    # Initialise theano arrays for emprical X and Y data:
    TaryEmpX = T.matrix()
    TaryDpthRnd = T.matrix()

    # Initialise model parameters:
    vecA = np.ones((varNumTtl), dtype=th.config.floatX)
    vecB = np.ones((varNumTtl), dtype=th.config.floatX)

    # Create shared theano object for model parameters:
    TvecA = th.shared(vecA)
    TvecB = th.shared(vecB)

    # Model prediction for theano:
    TobjMdlPre = model(TaryEmpX, TvecA, TvecB)

    # Learning rate:
    varLrnRt = np.float32(0.01)

    # Cost function:
    # cost = T.mean(T.sqr(y - Y))
    TobjCst = T.sum(T.sqr(T.sub(TobjMdlPre, TaryDpthRnd)))

    # Gradients for cost function
    TobGrd01 = T.grad(cost=TobjCst, wrt=TvecA)
    TobGrd02 = T.grad(cost=TobjCst, wrt=TvecB)

    # How to update the cost function:
    lstUp = [(TvecA, (TvecA - TobGrd01 * varLrnRt)),
             (TvecB, (TvecB - TobGrd02 * varLrnRt))]

    # Define the theano function that will be optimised:
    TcrfPwOp = th.function(inputs=[TaryEmpX, TaryDpthRnd],
                        outputs=TobjCst,
                        updates=lstUp)  # allow_input_downcast=True)

    # Do not check input data type:
    # train.trust_input = True

    ## Array for theano model parameters, of the form
    ## aryMdlParT[idxTotalIterations, freeModelParameters]:
    aryMdlParT = np.zeros((varNumTtl, 2)).astype(th.config.floatX)

    # Optimise function:
    for idxThn in range(1000):
        TcrfPwOp(aryEmpX, aryDpthRnd)

    # Save model parameter A:
    aryMdlParT[:, 0] = TvecA.get_value()

    # Save model parameter B:
    aryMdlParT[:, 1] = TvecB.get_value()

    # Check time:
    varTme02 = time.time()

    # Report time:
    varTme03 = varTme02 - varTme01
    print(('---Elapsed time: ' + str(varTme03) + 's for ' + str(varNumTtl)
           + ' iterations.'))

    # ------------------------------------------------------------------------
    # *** Apply CRF

    ### NOTE: This part can be optimised by using theano. ###

    # Arrays for y-values of fitted function (for each iteration & depth
    # level):
    aryMdlY = np.zeros((varNumTtl, varNumX))
    # Array for responses at half maximum contrast:
    aryHlfMax = np.zeros((varNumTtl))
    # Array for semisaturation contrast:
    arySemi = np.zeros((varNumTtl))
    # Arrays for residual variance:
    aryRes = np.zeros((varNumTtl, varNumCon))






    # Arrays for y-values of fitted function (for each iteration & depth
    # level):
    aryMdlY = np.zeros((varNumIn, varNumIt, varNumDpt, varNumX))
    # Array for responses at half maximum contrast:
    aryHlfMax = np.zeros((varNumIn, varNumIt, varNumDpt))
    # Array for semisaturation contrast:
    arySemi = np.zeros((varNumIn, varNumIt, varNumDpt))
    # Arrays for residual variance:
    aryRes = np.zeros((varNumIn, varNumIt, varNumCon, varNumDpt))

    # Reshaped array for model parameters:
    aryMdlParShp = np.zeros((varNumIn, varNumIt, varNumDpt, 2))

    # Reshape model parameters:
    varCnt = 0
    for idxIn in range(varNumIn):
        for idxIt in range(varNumIt):
            for idxDpth in range(varNumDpth):
                # Put model parameters into reshaped array:
                aryMdlParShp[idxIn, idxIt, idxDpth, :] = \
                    aryDpthRnd[varCnt, :]
                varCnt += 1




    # Loop through iterations:
    for idxIt in range(varNumTtl):
        
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















    return aryMdlY, aryHlfMax, arySemi, aryRes
