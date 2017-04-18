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


def crf_par(idxPrc,
            lstDpth,
            vecEmpX,
            aryRnd,
            varNumX,
            queOut):
    """
    Parallelise fitting of contrast response function.

    Parameters
    ----------
    idxPrc : int
        Process ID of parallel process. Needed to put output in order in
        parent function.
    lstDpth : list
        List of arrays with empirical response data, of the form
        lstDpth[idxRoi][idxSub, idxCon, idxDpt].
    vecEmpX : np.array
        Empirical x-values at which model will be fitted (e.g. stimulus
        contrast levels at which stimuli were presented), of the form
        vecEmpX[idxCon].
    aryRnd : np.array
        Array with randomised subject indicies for bootstrapping of the form
        aryRnd[idxIteration, varNumSamples]. Each row includes the indicies of
        the subjects to the sampled on that iteration.
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
    Function of the depth sampling pipeline.
    """
    # ----------------------------------------------------------------------------
    # *** Fit contrast response function

    # Number of inputs (ROIs, e.g. V1 & V2):
    varNumIn = len(lstDpth)

    # Number of conditions:
    varNumCon = lstDpth[0].shape[1]  # same as vecEmpX.shape[0]

    # Number of depth levels:
    varNumDpt = lstDpth[0].shape[2]

    # Number of iterations (for bootstrapping):
    varNumIt = aryRnd.shape[0]

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

    # Loop through ROIs (i.e. V1 and V2):
    for idxIn in range(0, varNumIn):

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 0:
            print('---------ROI: '
                  + str(idxIn + 1)
                  + ' of '
                  + str(varNumIn))

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

        # Loop through bootstrapping iterations:
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

            # Indicies of subjects to sample on current iteration:
            vecSmpl = aryRnd[idxIt, :]

            # Loop through depth levels:
            for idxDpt in range(0, varNumDpt):

                # Access contrast response profiles of current subset of
                # subjects and current depth level:
                aryEmpY = lstDpth[idxIn][vecSmpl, :, idxDpt]

                # Fit CRF:
                (aryMdlY[idxIn, idxIt, idxDpt, :],
                 aryHlfMax[idxIn, idxIt, idxDpt],
                 arySemi[idxIn, idxIt, idxDpt],
                 aryRes[idxIn, idxIt, :, idxDpt]) = crf_fit(vecEmpX,
                                                            aryEmpY,
                                                            strFunc='power',
                                                            varNumX=varNumX,
                                                            varXmin=0.0,
                                                            varXmax=1.0)

    # Output list:
    lstOut = [idxPrc, aryMdlY, aryHlfMax, arySemi, aryRes]

    queOut.put(lstOut)
