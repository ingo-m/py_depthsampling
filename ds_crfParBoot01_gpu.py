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


def crf_par_01_gpu(aryDpth, vecEmpX, strFunc='power', varNumIt=1000,
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
        aryMdlY[idxRoi, idxIteration, idxDpt, varNumX]
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

    print('---Theano CRF evaluation')

    # Check time:
    varTme01 = time.time()

    # Vector for which the function will be fitted:
    vecMdlX = np.linspace(varXmin, varXmax, num=varNumX, endpoint=True)

    # Boradcast array with X data, and change data type to float 32:
    aryMdlX = np.broadcast_to(vecMdlX, (varNumTtl, varNumX))
    aryMdlX = aryMdlX.astype(th.config.floatX)

    # Change data type to float 32:
    aryMdlParT = aryMdlParT.astype(th.config.floatX)

    # Initialise theano arrays for mmodel X data:
    TaryMdlX = T.matrix()

    # Create shared theano object for fitted model parameters:
    TvecMdlParA = th.shared(aryMdlParT[:, 0])
    TvecMdlParB = th.shared(aryMdlParT[:, 1])

    # Model to evaluate, like before (i.e. similar to the model that was
    # optimised, but this time with the fitted parameter values as input):
    TobjMdlEval = model(TaryMdlX, TvecMdlParA, TvecMdlParB)

    # Function definition for evaluation:
    TcrfPwEval = th.function([TaryMdlX], TobjMdlEval)

    # Evaluate function (get predicted y values of CRF for all resampling
    # iterations). Returns arrays for y-values of fitted function (for each
    # iteration & depth  level), of the form aryMdlY[varNumTtl, varNumX]
    aryMdlY = TcrfPwEval(aryMdlX)

    # Check time:
    varTme02 = time.time()

    # Report time:
    varTme03 = varTme02 - varTme01
    print(('---Elapsed time: ' + str(varTme03) + 's for ' + str(varNumTtl)
           + ' iterations.'))

    # ------------------------------------------------------------------------
    # *** Calculate predicted response at 50% contrast

    # Vector for which the function will be fitted (contrast = 0.5):
    vecMdl50 = np.ones((varNumTtl, 1))
    vecMdl50 = np.multiply(vecMdl50, 0.5)
    vecMdl50 = vecMdl50.astype(dtype=th.config.floatX)

    # Evaluate function. Returns array for responses at half maximum
    # contrast, of the form aryHlfMax[varNumTtl, 1]
    aryHlfMax = TcrfPwEval(vecMdl50)

    # ------------------------------------------------------------------------
    # *** Calculate predicted response at empirical contrast levels

    # We calculate the predicted response at the actually tested empirical
    # contrast levels in order to subsequently calculate the residual
    # variance of the model fit at those contrast levels.

    # Evaluate function (get predicted y values of CRF for empirically measured
    # contrast values):
    aryMdlEmpX = TcrfPwEval(aryEmpX)

    # ------------------------------------------------------------------------
    # *** Calculate semisaturation contrast

    # We first need to calculate the response at 100% contrast.

    # Vector for which the function will be fitted (contrast = 1.0):
    vecMdl100 = np.ones((varNumTtl, 1))
    vecMdl100 = vecMdl100.astype(dtype=th.config.floatX)

    # Evaluate function. Returns array for responses at half maximum
    # contrast, of the form aryResp100[varNumTtl, 1]
    aryResp100 = TcrfPwEval(vecMdl100)

    # Half maximum response:
    aryResp50 = np.multiply(aryResp100, 0.5)
    aryResp50 = aryResp50.astype(dtype=th.config.floatX)

    # Initialise theano arrays for half maximum response:
    TaryResp50 = T.matrix()

    # Initialise vector for Semisaturation constant:
    arySemi = np.ones((varNumTtl,1 ))
    arySemi = np.multiply(arySemi, 0.5)
    arySemi = arySemi.astype(dtype=th.config.floatX)

    # Create shared theano object for model parameters:
    TarySemi = th.shared(arySemi)

    # Model for finding semisaturation contrast:
    TobjMdlSemi = model(TarySemi, TvecMdlParA, TvecMdlParB)

    # Cost function for finding semisaturation contrast:
    TobjCst = T.sum(T.sqr(T.sub(TobjMdlSemi[:], TaryResp50[:])))

    # Gradient for cost function
    TobGrdSemi = T.grad(cost=TobjCst, wrt=TarySemi)

    # How to update the cost function:
    lstUp = [(TarySemi, (TarySemi - TobGrdSemi * varLrnRt))]

    # Define the theano function that will be optimised:
    TcrfPwSemi = th.function(inputs=[TaryResp50],
                             outputs=TobjCst,
                             updates=lstUp)  # allow_input_downcast=True)

    # Optimise function:
    for idxThn in range(1000):
        TcrfPwSemi(aryResp50)

    # Save semisaturation contrast:
    arySemi = TarySemi.get_value()

    # ------------------------------------------------------------------------
    # *** Reshape

    # Reshape array with fitted y-values, from
    #     aryMdlY[varNumTtl, varNumX]
    # to
    #     aryMdlY[idxRoi, idxIteration, idxDpt, varNumX]
    aryMdlYRs = np.zeros((varNumIn, varNumIt, varNumDpth, varNumCon, varNumX),
                         dtype=th.config.floatX)
    varCnt = 0
    for idxIn in range(varNumIn):
        for idxIt in range(varNumIt):
            for idxDpth in range(varNumDpth):
                    aryMdlYRs[idxIn, idxIt, idxDpth, :] = aryMdlY[varCnt, :]
                    varCnt += 1
    del(aryMdlY)
    aryMdlY = np.copy(aryMdlYRs)
    del(aryMdlYRs)

    # Reshape array with predicted response at 50 percent contrast, from
    #     aryHlfMax[varNumTtl, 1]
    # to
    #     aryHlfMax[idxRoi, idxIteration, idxDpt]
    aryHlfMaxRs = np.zeros((varNumIn, varNumIt, varNumDpth),
                           dtype=th.config.floatX)
    varCnt = 0
    for idxIn in range(varNumIn):
        for idxIt in range(varNumIt):
            for idxDpth in range(varNumDpth):
                aryHlfMaxRs[idxIn, idxIt, idxDpth] = aryHlfMax[varCnt, :]
                varCnt += 1
    del(aryHlfMax)
    aryHlfMax = np.copy(aryHlfMaxRs)
    del(aryHlfMaxRs)

    # Reshape array with predicted response at 50 percent contrast, from
    #     arySemi[varNumTtl, 1]
    # to
    #     arySemi[idxRoi, idxIteration, idxDpt]
    arySemiRs = np.zeros((varNumIn, varNumIt, varNumDpth),
                           dtype=th.config.floatX)
    varCnt = 0
    for idxIn in range(varNumIn):
        for idxIt in range(varNumIt):
            for idxDpth in range(varNumDpth):
                arySemiRs[idxIn, idxIt, idxDpth] = arySemi[varCnt, :]
                varCnt += 1
    del(arySemi)
    arySemi = np.copy(arySemiRs)
    del(arySemiRs)

    # Reshape array with predicted response at empirical constrast values, from
    #     aryMdlEmpX[varNumTtl, varNumCon]
    # to
    #     aryMdlEmpX[idxRoi, idxIteration, idxCondition, idxDpt]
    aryMdlEmpXRs = np.zeros((varNumIn, varNumIt, varNumCon, varNumDpth),
                           dtype=th.config.floatX)
    varCnt = 0
    for idxIn in range(varNumIn):
        for idxIt in range(varNumIt):
            for idxDpth in range(varNumDpth):
                aryMdlEmpXRs[idxIn, idxIt, :, idxDpth] = aryMdlEmpX[varCnt, :]
                varCnt += 1
    del(aryMdlEmpX)
    aryMdlEmpX = np.copy(aryMdlEmpXRs)
    del(aryMdlEmpXRs)

    # ------------------------------------------------------------------------
    # *** Calculate residual variance

    # Mean across subjects of (full) empirical dataset:
    aryEmpYMne = np.mean(aryDpth, axis=1)
    aryEmpYMne = aryEmpYMne.astype(dtype=th.config.floatX)

    # Residual variance at empirical contrast levels. Array of the form
    # aryRes[idxRoi, idxIteration, idxCondition, idxDpt].
    aryRes = np.subtract(aryMdlEmpX,
                         aryEmpYMne[:, None, :, :])

    return aryMdlY, aryHlfMax, arySemi, aryRes
