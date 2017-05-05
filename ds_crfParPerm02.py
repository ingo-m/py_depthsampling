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
from ds_crfFit import crf_fit


def crf_par_perm_02(idxPrc, aryDpth01, aryDpth02, vecEmpX, strFunc, aryRnd,
                    varNumX, queOut):
    """
    Parallelised permutation testing on contrast response function, level 2.

    Parameters
    ----------
    idxPrc : int
        Process ID of parallel process. Needed to put output in order in
        parent function.
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
    aryRnd : np.array
        Array with randomised subject indicies for bootstrapping of the form
        aryRnd[idxIteration, varNumSamples]. Each row includes the indicies of
        the subjects to be sampled on that iteration.
    varNumX : int
        Number of x-values for which to solve the function when calculating
        model fit.
    queOut : multiprocessing.queues.Queue
        Queue to put results on.

    Returns
    -------
    lstOut : list
        List with results, containing the following objects:
    idxPrc : int
        Process ID of parallel process. Needed to put output in order in
        parent function.
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
    This function is supposed to be called in parallel, using the
    multiprocessing module. This function calls a function which performs
    the actual least squares fitting.

    Function of the depth sampling pipeline.
    """
    # ----------------------------------------------------------------------------
    # *** Fit contrast response function

    # Number of ROIs:
    varNumIn = 2

    # Get number of conditions from input array:
    varNumCon = aryDpth01.shape[1]

    # Get number of depth levels from input array:
    varNumDpt = aryDpth01.shape[2]

    # Number of iterations (for resampling):
    varNumIt = aryRnd.shape[0]

    # We need two versions of the randomisation array, one for sampling from
    # the first input array (e.g. V1), and a second version to sample from the
    # second input array (e.g. V2). (I.e. the second version is the opposite
    # of the first version.)
    aryRnd01 = np.equal(aryRnd, 1)
    aryRnd02 = np.equal(aryRnd, 0)
    del(aryRnd)

    # Arrays for y-values of fitted function (for each iteration & depth
    # level):
    aryMdlY = np.zeros((varNumIn, varNumIt, varNumDpt, varNumX))

    # Array for responses at half maximum contrast:
    aryHlfMax = np.zeros((varNumIn, varNumIt, varNumDpt))

    # List of vectors for semisaturation contrast:
    arySemi = np.zeros((varNumIn, varNumIt, varNumDpt))

    # List of arrays for residual variance:
    aryRes = np.zeros((varNumIn, varNumIt, varNumCon, varNumDpt))

    # Only print status messages if this is the first of several parallel
    # processes:
    if idxPrc == 0:
        print('------Fitting contrast response functions')

        # Prepare status indicator. Number of steps of the status
        # indicator:
        varStsStpSze = 20

        # Vector with iterations at which to give status feedback:
        vecStatItr = np.linspace(0,
                                 varNumIt,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatItr = np.ceil(vecStatItr)
        vecStatItr = vecStatItr.astype(int)

        # Vector with corresponding percentage values at which to give
        # status  feedback:
        vecStatPrc = np.linspace(0,
                                 100,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrc = np.ceil(vecStatPrc)
        vecStatPrc = vecStatPrc.astype(int)

        # Counter for status indicator:
        varCntSts01 = 0
        varCntSts02 = 0

    # Loop through resampling iterations:
    for idxIt in range(0, varNumIt):

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 0:

            # Status indicator:
            if varCntSts02 == vecStatItr[varCntSts01]:

                # Prepare status message:
                strStsMsg = ('------------Progress: '
                             + str(vecStatPrc[varCntSts01])
                             + ' %')

                print(strStsMsg)

                # Only increment counter if the last value has not been
                # reached yet:
                if varCntSts01 < varStsStpSze:
                    varCntSts01 = varCntSts01 + int(1)

            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

        # ** Permutation group 1

        # CRF fitting for the first permutation group. We first need to assign
        # the values to be fitted based on the randomly permuted labels:
        aryDpthRnd = np.zeros(aryDpth01.shape)

        # Assign values from original group 1 to permutation group 1:
        aryDpthRnd[aryRnd01[idxIt, :], :, :] = aryDpth01[aryRnd01[idxIt, :],
                                                         :, :]

        # Assign values from original group 2 to permutation group 1:
        aryDpthRnd[aryRnd02[idxIt, :], :, :] = aryDpth02[aryRnd02[idxIt, :],
                                                         :, :]

        # Loop through depth levels:
        for idxDpt in range(0, varNumDpt):

            # Fit CRF:
            (aryMdlY[0, idxIt, idxDpt, :],
             aryHlfMax[0, idxIt, idxDpt],
             arySemi[0, idxIt, idxDpt],
             aryRes[0, idxIt, :, idxDpt]) = crf_fit(vecEmpX,
                                                    aryDpthRnd[:, :, idxDpt],
                                                    strFunc=strFunc,
                                                    varNumX=varNumX,
                                                    varXmin=0.0,
                                                    varXmax=1.0)

        # ** Permutation group 2

        # CRF fitting for the second permutation group. We first need to assign
        # the values to be fitted based on the randomly permuted labels:
        aryDpthRnd = np.zeros(aryDpth01.shape)

        # Assign values from original group 1 to permutation group 2:
        aryDpthRnd[aryRnd02[idxIt, :], :, :] = aryDpth01[aryRnd02[idxIt, :],
                                                         :, :]

        # Assign values from original group 2 to permutation group 2:
        aryDpthRnd[aryRnd01[idxIt, :], :, :] = aryDpth02[aryRnd01[idxIt, :],
                                                         :, :]

        # Loop through depth levels:
        for idxDpt in range(0, varNumDpt):

            # Fit CRF:
            (aryMdlY[1, idxIt, idxDpt, :],
             aryHlfMax[1, idxIt, idxDpt],
             arySemi[1, idxIt, idxDpt],
             aryRes[1, idxIt, :, idxDpt]) = crf_fit(vecEmpX,
                                                    aryDpthRnd[:, :, idxDpt],
                                                    strFunc=strFunc,
                                                    varNumX=varNumX,
                                                    varXmin=0.0,
                                                    varXmax=1.0)

    # Output list:
    lstOut = [idxPrc, aryMdlY, aryHlfMax, arySemi, aryRes]

    queOut.put(lstOut)
