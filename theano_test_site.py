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
from scipy.optimize import curve_fit
from ds_crfFunc import crf_power



aryDpth = np.load('/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy')
aryDpth = np.array([aryDpth])
vecEmpX = np.array([0.025, 0.061, 0.163, 0.72])
strFunc='power'
varNumIt=100000
varNumX=1000


#def crf_par_01_t(aryDpth, vecEmpX, strFunc='power', varNumIt=1000,
#                 varNumX=1000):
#    """
#    Parallelised bootstrapping of contrast response function, level 1.
#
#    Parameters
#    ----------
#    aryDpth : np.array
#        Array with empirical response data, of the form
#        aryDpth[idxRoi, idxSub, idxCon, idxDpt].
#    vecEmpX : np.array
#        Empirical x-values at which model will be fitted (e.g. stimulus
#        contrast levels at which stimuli were presented), of the form
#        vecEmpX[idxCon].
#    strFunc : str
#        Which contrast response function to fit. 'power' for power function, or
#        'hyper' for hyperbolic ratio function.
#    varNumIt : int
#        Number of bootstrapping iterations (i.e. how many times to sample).
#    varPar : int
#        Number of process to run in parallel.
#    varNumX : int
#        Number of x-values for which to solve the function when calculating
#        model fit.
#
#    Returns
#    -------
#    aryMdlY : np.array
#        Fitted y-values (predicted response based on CRF model), of the form
#        aryMdlY[idxRoi, idxIteration, idxDpt, idxContrast]
#    aryHlfMax : np.array
#        Predicted response at 50 percent contrast based on CRF model. Array of
#        the form aryHlfMax[idxRoi, idxIteration, idxDpt].
#    arySemi : np.array
#        Semisaturation contrast (predicted contrast needed to elicit 50 percent
#        of the response amplitude that would be expected with a 100 percent
#        contrast stimulus). Array of the form
#        arySemi[idxRoi, idxIteration, idxDpt].
#    aryRes : np.array
#        Residual variance at empirical contrast levels. Array of the form
#        aryRes[idxRoi, idxIteration, idxCondition, idxDpt].
#
#    Notes
#    -----
#    This function parallelises the contrast response function fitting by
#    calling a second-level function using the multiprocessing module.
#
#    Function of the depth sampling pipeline.
#    """
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

## Total number of CRF models to fit:
#varNumTtl = (varNumIn * varNumDpth * varNumIt)
#
## Array for resampled samples:
#aryDpthRnd = np.zeros((varNumTtl, varNumCon))
#
## Fill array with resampled samples:
#varCnt = 0
#for idxIn in range(varNumIn):
#    for idxIt in range(varNumIt):
#        for idxDpth in range(varNumDpth):
#            aryDpthRnd[varCnt, :] = np.mean(aryDpth[idxIn,
#                                                    aryRnd[idxIt, :],
#                                                    :,
#                                                    idxDpth],
#                                            axis=0)
#            varCnt += 1


# ------------------------------------------------------------------------
# *** CRF fitting

print('---CRF fitting')

# Check time:
varTme01 = time.time()

# *** Fit contrast reponse function
if strFunc == 'power':

    # Array for model parameters, of the form
    # aryMdlPar[idxTotalIterations, freeModelParameters]:
    aryMdlPar = np.zeros((varNumTtl, 2))

    # Array for model covariance matrices, of the form
    # aryMdlCov[idxTotalIterations, covarianceMatrixElements]:
    aryMdlCov = np.zeros((varNumTtl, 2, 2))
    
    # Lower limits for parameters (factor, exponent) - for power function:
    vecLimPowLw = np.array([0.0, 0.0])

    # Upper limits for parameters (factor, exponent) - for power function:
    vecLimPowUp = np.array([10.0, 1.0])

    # Loop through resampling iterations and fit function:
    for idxIt in range(0, varNumTtl):
        
        # Curve fitting:
        aryMdlPar[idxIt, :], aryMdlCov[idxIt, :, :] = \
            curve_fit(crf_power,
                      vecEmpX,
                      aryDpthRnd[idxIt, :],
                      maxfev=100000,
                      bounds=(vecLimPowLw, vecLimPowUp),
                      p0=(0.5, 0.5))

# Check time:
varTme02 = time.time()

# Report time:
varTme03 = varTme02 - varTme01
print(('---Elapsed time: ' + str(varTme03) + 's for ' + str(varNumTtl)
       + ' iterations.'))


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


y = model(TaryEmpX, TvecA, TvecB)

# Learning rate:
varLrnRt = np.float32(0.01)

# Cost function:
# cost = T.mean(T.sqr(y - Y))
TobjCst = T.sum(T.sqr(T.sub(y, TaryDpthRnd)))

# Gradients for cost function
TobGrd01 = T.grad(cost=TobjCst, wrt=TvecA)
TobGrd02 = T.grad(cost=TobjCst, wrt=TvecB)

# How to update the cost function:
lstUp = [(TvecA, (TvecA - TobGrd01 * varLrnRt)),
         (TvecB, (TvecB - TobGrd02 * varLrnRt))]

train = th.function(inputs=[TaryEmpX, TaryDpthRnd],
                    outputs=TobjCst,
                    updates=lstUp)  # allow_input_downcast=True)

# Do not check input data type:
# train.trust_input = True

## Array for theano model parameters, of the form
## aryMdlParT[idxTotalIterations, freeModelParameters]:
aryMdlParT = np.zeros((varNumTtl, 2)).astype(th.config.floatX)


## Loop through resampling iterations and fit function:
# for idxIt01 in range(0, varNumTtl):

for idxThn in range(1000):
    train(aryEmpX, aryDpthRnd)

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

# Correlation between results:
varCor01 = np.corrcoef(aryMdlPar[:, 0], aryMdlParT[:, 0])[0][1]
varCor02 = np.corrcoef(aryMdlPar[:, 1], aryMdlParT[:, 1])[0][1]

print('---Correlation model parameter 01: ' + str(varCor01))
print('---Correlation model parameter 02: ' + str(varCor02))

