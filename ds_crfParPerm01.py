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

import itertools
import numpy as np
import multiprocessing as mp
from ds_crfParPerm02 import crf_par_perm_02


def crf_par_perm_01(aryDpth01, aryDpth02, vecEmpX, strFunc='power',
                    varNumIt=1000, varPar=10, varNumX=1000):
    """
    Parallelised permutation testing on contrast response function, level 1.

    Parameters
    ----------
    aryDpth01 : np.array
        Array with single-subject cortical depth profiles for first condition
        (e.g. V1), of the form: aryDpth[idxSub, idxCondition, idxDpth].
        aryDpth01 and aryDpth02 must have same dimensions.
    aryDpth02 : np.array
        Array with single-subject cortical depth profiles for second condition
        (e.g. V2), of the form: aryDpth[idxSub, idxCondition, idxDpth].
        aryDpth01 and aryDpth02 must have same dimensions.
    vecEmpX : np.array
        Empirical x-values at which model will be fitted (e.g. stimulus
        contrast levels at which stimuli were presented), of the form
        vecEmpX[idxCon].
    strFunc : str
        Which contrast response function to fit. 'power' for power function, or
        'hyper' for hyperbolic ratio function.
    varNumIt : int or None
        Number of resampling iterations. Set to `None` in case of small enough
        sample size for exact test (i.e. all possible resamples), otherwise
        Monte Carlo resampling is performed.
    varPar : int
        Number of process to run in parallel.
    varNumX : int
        Number of x-values for which to solve the function when calculating
        model fit.

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
    This function parallelises the contrast response function fitting by
    calling a second-level function using the multiprocessing module.

    Function of the depth sampling pipeline.
    """
    # ------------------------------------------------------------------------
    # *** Prepare resampling

    print('---Preparing resampling')

    # Get number of subjects from input array:
    varNumSub = aryDpth01.shape[0]

    # Random array that is used to permute V1 and V2 labels within subjects,
    # of the form aryRnd[idxIteration, idxSub]. For each iteration and subject,
    # there is either a zero or a one. 'Zero' means that the actual V1 value
    # gets assigned to the permuted 'V1' group and the actual V2 value gets
    # assigned to the permuted 'V2' group. 'One' means that the labels are
    # switched, i.e. the actual V1 label get assignet to the 'V2' group and
    # vice versa.
    if not(varNumIt is None):
        # Monte Carlo resampling:
        aryRnd = np.random.randint(0, high=2, size=(varNumIt, varNumSub))
    else:
        # In case of tractable number of permutations, create a list of all
        # possible permutations (Bernoulli sequence).
        lstBnl = list(itertools.product([0, 1], repeat=varNumSub))
        aryRnd = np.array(lstBnl)
        # Number of resampling cases:
        varNumIt = len(lstBnl)

    # ------------------------------------------------------------------------
    # *** Parallelised CRF fitting

    print('---Creating parallel processes')

    # Empty list for results:
    lstOut = [None for i in range(varPar)]

    # Empty list for processes:
    lstPrc = [None for i in range(varPar)]

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # List into which the chunks of the randomisation-array for the parallel
    # processes will be put:
    lstRnd = [None for i in range(varPar)]

    # Vector with the indicies at which the randomisation-array will be
    # separated in order to be chunked up for the parallel processes:
    vecIdxChnks = np.linspace(0,
                              varNumIt,
                              num=varPar,
                              endpoint=False)
    vecIdxChnks = np.hstack((vecIdxChnks, varNumIt))

    # Put randomisation indicies into chunks:
    for idxChnk in range(0, varPar):
        # Index of first iteration to be included in current chunk:
        varTmpChnkSrt = int(vecIdxChnks[idxChnk])
        # Index of last iteration to be included in current chunk:
        varTmpChnkEnd = int(vecIdxChnks[(idxChnk+1)])
        # Put array into list:
        lstRnd[idxChnk] = aryRnd[varTmpChnkSrt:varTmpChnkEnd, :]

    # We don't need the original randomisation array anymore:
    del(aryRnd)

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrc[idxPrc] = mp.Process(target=crf_par_perm_02,
                                    args=(idxPrc,
                                          aryDpth01,
                                          aryDpth02,
                                          vecEmpX,
                                          strFunc,
                                          lstRnd[idxPrc],
                                          varNumX,
                                          queOut))
        # Daemon (kills processes when exiting):
        lstPrc[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrc[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstOut[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrc[idxPrc].join()

    # ------------------------------------------------------------------------
    # *** Collect results

    print('---Collecting results from parallel processes')

    # List for results from parallel processes, in order to join the results:
    lstMdlY = [None for i in range(varPar)]
    lstHlfMax = [None for i in range(varPar)]
    lstSemi = [None for i in range(varPar)]
    lstRes = [None for i in range(varPar)]

    # Put output into correct order:
    for idxPrc in range(0, varPar):

        # Index of results (first item in output list):
        varTmpIdx = lstOut[idxPrc][0]

        # Put fitting results into list, in correct order:
        lstMdlY[varTmpIdx] = lstOut[idxPrc][1]
        lstHlfMax[varTmpIdx] = lstOut[idxPrc][2]
        lstSemi[varTmpIdx] = lstOut[idxPrc][3]
        lstRes[varTmpIdx] = lstOut[idxPrc][4]

    # Concatenate output vectors (into the same order as the voxels that were
    # included in the fitting):

    # Arrays for y-values of fitted function (for each iteration & depth
    # level):
    # aryMdlY = np.zeros((varNumIn, varNumIt, varNumDpt, varNumX))
    # Array for responses at half maximum contrast:
    # aryHlfMax = np.zeros((varNumIn, varNumIt, varNumDpt))
    # List of vectors for semisaturation contrast:
    # arySemi = np.zeros((varNumIn, varNumIt, varNumDpt))
    # List of arrays for residual variance:
    # aryRes = np.zeros((varNumIn, varNumIt, varNumCon, varNumDpt))
    for idxPrc in range(0, varPar):
        aryMdlY = np.concatenate(lstMdlY, axis=1)
        aryHlfMax = np.concatenate(lstHlfMax, axis=1)
        arySemi = np.concatenate(lstSemi, axis=1)
        aryRes = np.concatenate(lstRes, axis=1)

    return aryMdlY, aryHlfMax, arySemi, aryRes
