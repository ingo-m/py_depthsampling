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
import rpy2.robjects as robjects
from py_depthsampling.plot.plt_dpth_prfl import plt_dpth_prfl


def boot_plot(objDpth, strPath, lstCon, lstConLbl, varNumIt=10000,  #noqa
              varConLw=2.5, varConUp=97.5, strTtl='', varYmin=0.0, varYmax=2.0,
              tplPadY=(0.0, 0.0),
              strXlabel='Cortical depth level (equivolume)',
              strYlabel='fMRI signal change [arbitrary units]',
              lgcLgnd=False, lstDiff=None, vecNumInc=None, strParam='mean'):
    """
    Plot across-subject cortical depth profiles with confidence intervals.

    Parameters
    ----------
    objDpth : np.array or str
        Array with single-subject cortical depth profiles, of the form:
        aryDpth[idxSub, idxCondition, idxDpth]. Either a numpy array or a
        string with the path to an npy file containing the array.
    strPath : str
        Output path for plot.
    lstCon : list
        Abbreviated condition levels used to complete file names (e.g. 'Pd').
    lstConLbl : list
        List containing condition labels (strings). Number of condition labels
        has to be the same as number of conditions in `lstCon`.
    varNumIt : int
        Number of bootstrap iterations.
    varConLw : float
        Lower bound of the percentile bootstrap confidence interval in
        percent (i.e. in range of [0, 100]).
    varConUp : float
        Upper bound of the percentile bootstrap confidence interval in
        percent (i.e. in range of [0, 100]).
    strTtl : str
        Plot title.
    varYmin : float
        Minimum of Y axis.
    varYmax : float
        Maximum of Y axis.
    tplPadY : tuple
        Padding around labelled values on y.
    strXlabel : str
        Label for x axis.
    strYlabel : str
        Label for y axis.
    lgcLgnd : bool
        Whether to show a legend.
    lstDiff : list or None
        If None, the depth profiles are plotted separately for each condition.
        If a list of tuples of condition indices is provided, on each
        bootstrapping iteration the difference between the two conditions is
        calculated, and is plotted. The the second condition from the tuple is
        subtracted from the first (e.g. if lstDiff = [(0, 1)], then condition 1
        is subtracted from condition 0).
    vecNumInc : np.array
        1D array with weights for weighted averaging over subjects (e.g. number
        of vertices per subject). If the array is loaded from disk (i.e. if
        `objDpth` is the path to an `*.npz` file stored on disk), `vecNumInc`
        has to be in the `*.npz` file. If `objDpth` is a numpy array containing
        the data, `vecNumInc` should also be provided as an input arguments.
        Otherwise, weights are set to be equal across subjects.
    strParam : string
        Which parameter to plot; 'mean' or 'median'. If `strParam = 'median'`,
        an R function is imported for calculating the weighted median.
        Dependency (in python): `rpy2`, dependency (in R): `spatstat`.

    Returns
    -------
    None : None
        This function has no return value.

    Notes
    -----
    Plot across-subject mean or median cortical depth profiles with percentile
    bootstrap confidence intervals. This function bootstraps (i.e. resamples
    with replacement) from an array of single-subject depth profiles,
    calculates a confidence interval of the mean/median across bootstrap
    iterations and plots the empirical mean/median & bootstrap confidence
    intervals along the cortical depth.

    Function of the depth sampling pipeline.
    """
    # -------------------------------------------------------------------------
    # *** Prepare bootstrapping

    # Test whether the input is a numpy array or a string (with the path to a
    # numpy array):
    lgcAry = (type(objDpth) == np.ndarray)
    lgcStr = (type(objDpth) == str)

    # If input is a string, load array from npy file:
    if lgcAry:
        aryDpth = objDpth
        # If weights are not provided, set equal weight of one for each
        # subject:
        if vecNumInc is None:
            vecNumInc = np.ones((aryDpth.shape[0]))

    elif lgcStr:
        # Load array for first condition to get dimensions:
        objNpz = np.load(objDpth.format(lstCon[0]))
        aryTmpDpth = objNpz['arySubDpthMns']
        # Number of subjects:
        varNumSub = aryTmpDpth.shape[0]
        # Get number of depth levels from input array:
        varNumDpth = aryTmpDpth.shape[1]
        # Number of conditions:
        varNumCon = len(lstCon)
        # Array for depth profiles of form aryDpth[subject, condition, depth]:
        aryDpth = np.zeros((varNumSub, varNumCon, varNumDpth))
        # Load single-condition arrays from disk:
        for idxCon in range(varNumCon):
            objNpz = np.load(objDpth.format(lstCon[idxCon]))
            aryDpth[:, idxCon, :] = objNpz['arySubDpthMns']
        # Array with number of vertices (for weighted averaging across
        # subjects), shape: vecNumInc[subjects].
        vecNumInc = objNpz['vecNumInc']

    else:
        print(('---Error in bootPlot: input needs to be numpy array or path '
               + 'to numpy array.'))

    # Get number of subjects from input array:
    varNumSub = aryDpth.shape[0]
    # Get number of conditions from input array:
    varNumCon = aryDpth.shape[1]
    # Get number of depth levels from input array:
    varNumDpth = aryDpth.shape[2]

    # We will sample subjects with replacement. How many subjects to sample on
    # each iteration:
    varNumSmp = varNumSub

    # Random array with subject indicies for bootstrapping of the form
    # aryRnd[varNumIt, varNumSmp]. Each row includes the indicies of the
    # subjects to the sampled on that iteration.
    aryRnd = np.random.randint(0,
                               high=varNumSub,
                               size=(varNumIt, varNumSmp))

    if lstDiff is None:
        # Array for bootstrap samples, of the form
        # aryBoo[idxIteration, idxSubject, idxCondition, idxDpth]):
        aryBoo = np.zeros((varNumIt, varNumSub, varNumCon, varNumDpth))
    else:
        # Set number of comparisons:
        varNumCon = len(lstDiff)
        # Array for bootstrap samples:
        aryBoo = np.zeros((varNumIt, varNumSub, varNumCon, varNumDpth))

    # Array with number of vertices per subject for each bootstrapping sample
    # (needed for weighted averaging), shape: aryWght[iterations, subjects]
    aryWght = np.zeros((varNumIt, varNumSub))

    # ------------------------------------------------------------------------
    # *** Bootstrap

    # Loop through bootstrap iterations:
    for idxIt in range(varNumIt):
        # Indices of current bootstrap sample:
        vecRnd = aryRnd[idxIt, :]
        if lstDiff is None:
            # Put current bootstrap sample into array:
            aryBoo[idxIt, :, :, :] = aryDpth[vecRnd, :, :]
        else:
            # Calculate difference between conditions:
            for idxDiff in range(varNumCon):
                aryBoo[idxIt, :, idxDiff, :] = \
                    np.subtract(aryDpth[vecRnd, lstDiff[idxDiff][0], :],
                                aryDpth[vecRnd, lstDiff[idxDiff][1], :])

        # Put number of vertices per subject into respective array (for
        # weighted averaging):
        aryWght[idxIt, :] = vecNumInc[vecRnd]

    if strParam == 'mean':

        # Mean for each bootstrap sample (across subjects within the bootstrap
        # sample):

        # Sum of weights over subjects (i.e. total number of vertices across
        # subjects, one value per iteration; for scaling).
        vecSum = np.sum(aryWght, axis=1)

        # Multiply depth profiles by weights (weights are broadcasted over
        # conditions and depth levels):
        aryTmp = np.multiply(aryBoo, aryWght[:, :, None, None])

        # Sum over subjects, and scale by number of vertices (sum of vertices
        # is broadcasted over conditions and depth levels):
        aryBooMne = np.divide(
                              np.sum(aryTmp, axis=1),
                              vecSum[:, None, None]
                              )

    elif strParam == 'median':

        # Define R function for calculation of weighted median:
        strFuncR = """
         funcR <- function(lstData, lstWght){
         library(spatstat)
         varWm <- weighted.median(lstData, lstWght)
         return(varWm)
         }
        """
        objFuncR = robjects.r(strFuncR)

        # Array for weighted median difference between conditions:
        aryBooMne = np.zeros((varNumIt, varNumCon, varNumDpth))

        # Calculate weighted median difference between conditions in R (yes
        # this is slow):
        for idxIt in range(varNumIt):
            for idxCon in range(varNumCon):
                for idxDpth in range(varNumDpth):
                    aryBooMne[idxIt, idxCon, idxDpth] = \
                        objFuncR(list(aryBoo[idxIt, :, idxCon, idxDpth]),
                                 list(aryWght[idxIt, :]))[0]

    # Delete large bootstrap array:
    del(aryBoo)

    # Percentile bootstrap for mean:
    aryPrct = np.percentile(aryBooMne, (varConLw, varConUp), axis=0)

    # ------------------------------------------------------------------------
    # *** Plot result

    if lstDiff is None:

        if strParam == 'mean':

            # Sum of weights over subjects (i.e. total number of vertices
            # across subjects; for scaling).
            varSum = np.sum(vecNumInc)

            # Multiply depth profiles by weights (weights are broadcasted over
            # conditions and depth levels):
            aryTmp = np.multiply(aryDpth, vecNumInc[:, None, None])

            # Sum over subjects, and scale by total number of vertices:
            aryEmpMne = np.divide(
                                  np.sum(aryTmp, axis=0),
                                  varSum
                                  )

        elif strParam == 'median':

            # Array for weighted median difference between conditions:
            aryEmpMne = np.zeros((varNumCon, varNumDpth))

            # Calculate weighted median in R (yes this is slow):
            for idxCon in range(varNumCon):
                for idxDpth in range(varNumDpth):
                    aryEmpMne[idxCon, idxDpth] = \
                        objFuncR(list(aryDpth[:, idxCon, idxDpth]),
                                 list(vecNumInc))[0]

    else:

        # Empirical mean difference between conditions:
        aryEmpMne = np.zeros((varNumCon, varNumDpth))

        for idxDiff in range(varNumCon):

            if strParam == 'mean':

                # Sum of weights over subjects (i.e. total number of vertices
                # across subjects; for scaling).
                varSum = np.sum(vecNumInc)

                # Difference in cortical depth profiles between conditions:
                aryDiff = np.subtract(aryDpth[:, lstDiff[idxDiff][0], :],
                                      aryDpth[:, lstDiff[idxDiff][1], :])

                # Un-comment this for SEM (overwrites bootstrapping results),
                # for comparison:
                # aryPrct[0, idxDiff, :] = np.divide(np.std(aryDiff, axis=0),
                #                                    np.sqrt(varNumSub)) * -1
                # aryPrct[1, idxDiff, :] = np.divide(np.std(aryDiff, axis=0),
                #                                    np.sqrt(varNumSub)) * 1

                # Multiply depth profiles by weights (weights are broadcasted
                # over depth levels):
                aryDiff = np.multiply(aryDiff, vecNumInc[:, None])

                # Sum over subjects, and scale by total number of vertices:
                aryEmpMne[idxDiff, :] = np.divide(
                                                  np.sum(aryDiff, axis=0),
                                                  varSum
                                                  )

                # Un-comment this for SEM (overwrites bootstrapping results),
                # for comparison:
                # aryPrct[0, idxDiff, :] = np.add(aryPrct[0, idxDiff, :],
                #                                 aryEmpMne[idxDiff, :])
                # aryPrct[1, idxDiff, :] = np.add(aryPrct[1, idxDiff, :],
                #                                 aryEmpMne[idxDiff, :])

            elif strParam == 'median':

                # Calculate weighted median difference between conditions in R
                # (yes this is slow):
                for idxDiff in range(varNumCon):

                    # Difference in cortical depth profiles between conditions:
                    aryDiff = np.subtract(aryDpth[:, lstDiff[idxDiff][0], :],
                                          aryDpth[:, lstDiff[idxDiff][1], :])

                    for idxDpth in range(varNumDpth):
                        aryEmpMne[idxDiff, idxDpth] = \
                            objFuncR(list(aryDiff[:, idxDpth]),
                                     list(vecNumInc))[0]

        # Create condition labels for differences:
        lstDiffLbl = [None] * varNumCon
        for idxDiff in range(varNumCon):
            lstDiffLbl[idxDiff] = ((lstConLbl[lstDiff[idxDiff][0]])
                                   + ' minus '
                                   + (lstConLbl[lstDiff[idxDiff][1]]))
        lstConLbl = lstDiffLbl

    plt_dpth_prfl(aryEmpMne, None, varNumDpth, varNumCon, 80.0, varYmin,
                  varYmax, False, lstConLbl, strXlabel, strYlabel, strTtl,
                  lgcLgnd, strPath, varSizeX=1800.0, varSizeY=1600.0,
                  varNumLblY=6, tplPadY=tplPadY, aryCnfLw=aryPrct[0, :, :],
                  aryCnfUp=aryPrct[1, :, :])
    # ------------------------------------------------------------------------
