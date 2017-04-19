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
import multiprocessing as mp
from ds_crfParBoot02 import crf_par_02


def crf_par_01(lstDpth, vecEmpX, varNumIt=1000, varPar=10, varNumX=1000):
    """
    Parallelised bootstrapping of contrast response function, level 1.

    Parameters
    ----------
    lstDpth : list
        List of arrays with empirical response data, of the form
        lstDpth[idxRoi][idxSub, idxCon, idxDpt].
    vecEmpX : np.array
        Empirical x-values at which model will be fitted (e.g. stimulus
        contrast levels at which stimuli were presented), of the form
        vecEmpX[idxCon].
    varNumIt : int
        Number of bootstrapping iterations (i.e. how many times to sample).
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
        Predicted response at 50% contrast based on CRF model. Array of the
        form aryHlfMax[idxRoi, idxIteration, idxDpt].
    arySemi : np.array
        Semisaturation contrast (predicted contrast needed to elicit 50% of
        the response amplitude that would be expected with a 100% contrast
        stimulus). Array of the form arySemi[idxRoi, idxIteration, idxDpt].
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
    # *** Prepare bootstrapping

    print('---Preparing bootstrapping')

    # Number of subjects:
    varNumSubs = lstDpth[0].shape[0]

    # We will sample subjects with replacement. How many subjects to sample on
    # each iteration:
    varNumSmp = varNumSubs

    # Random array with subject indicies for bootstrapping of the form
    # aryRnd[varNumIt, varNumSmp]. Each row includes the indicies of the
    # subjects to the sampled on that iteration.
    aryRnd = np.random.randint(0,
                               high=varNumSubs,
                               size=(varNumIt, varNumSmp))

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

    # We don't need the original array with the functional data anymore:
    del(aryRnd)

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrc[idxPrc] = mp.Process(target=crf_par_02,
                                    args=(idxPrc,
                                          lstDpth,
                                          vecEmpX,
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
