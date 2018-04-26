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
from py_depthsampling.plot.plt_dpth_prfl import plt_dpth_prfl


def boot_plot(objDpth, strPath, lstConLbl, varNumIt=10000, varConLw=2.5,
              varConUp=97.5, strTtl='', varYmin=0.0, varYmax=2.0,
              strXlabel='Cortical depth level (equivolume)',
              strYlabel='fMRI signal change [arbitrary units]',
              lgcLgnd=False, lstDiff=None):
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
    lstConLbl : list
        List containing condition labels (strings). Number of condition labels
        has to be the same as number of conditions in `objDpth`.
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

    Returns
    -------
    None : None
        This function has no return value.

    Notes
    -----
    Plot across-subject median cortical depth profiles with percentile
    bootstrap confidence intervals. This function bootstraps (i.e. resamples
    with replacement) from an array of single-subject depth profiles,
    calculates a confidence interval of the median across bootstrap iterations
    and plots the empirical median & bootstrap confidence intervals along the
    cortical depth.

    Function of the depth sampling pipeline.
    """
    # ------------------------------------------------------------------------
    # *** Prepare bootstrapping

    # Test whether the input is a numpy array or a string (with the path to a
    # numpy array):
    lgcAry = (type(objDpth) == np.ndarray)
    lgcStr = (type(objDpth) == str)

    # If input is a string, load array from npy file:
    if lgcAry:
        aryDpth = objDpth
    elif lgcStr:
        aryDpth = np.load(objDpth)
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
        # Array for bootstrap samples, of the form
        # aryBoo[idxIteration, idxSubject, 1, idxDpth]) (3rd dimension is one
        # because the array will hold the difference between two conditions):
        aryBoo = np.zeros((varNumIt, varNumSub, varNumCon, varNumDpth))

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
            # NOTE: Relative difference score leads to inconsistent results.
            # Calculate normalised difference between conditions (difference
            # score ranging from -1 to 1; ((A - B) / abs(A + B)):
            # for idxDiff in range(varNumCon):
            #     aryBoo[idxIt, :, idxDiff, :] = \
            #         np.divide(
            #             np.subtract(
            #                 aryDpth[vecRnd, lstDiff[idxDiff][0], :],
            #                 aryDpth[vecRnd, lstDiff[idxDiff][1], :]
            #                 ),
            #             np.absolute(
            #                 np.add(
            #                     aryDpth[vecRnd, lstDiff[idxDiff][0], :],
            #                     aryDpth[vecRnd, lstDiff[idxDiff][1], :]
            #                     )
            #                 )
            #             )
            # Calculate difference between conditions:
            for idxDiff in range(varNumCon):
                aryBoo[idxIt, :, idxDiff, :] = \
                    np.subtract(aryDpth[vecRnd, lstDiff[idxDiff][0], :],
                                aryDpth[vecRnd, lstDiff[idxDiff][1], :])

    # Median for each bootstrap sample (across subjects within the bootstrap
    # sample):
    aryBooMed = np.median(aryBoo, axis=1)

    # Delete large bootstrap array:
    del(aryBoo)

    # Percentile bootstrap for median:
    aryPrct = np.percentile(aryBooMed, (varConLw, varConUp), axis=0)

    # ------------------------------------------------------------------------
    # *** Plot result

    if lstDiff is None:

        # Empirical median:
        aryEmpMed = np.median(aryDpth, axis=0)

    else:

        # Empirical median difference between conditions:
        aryEmpMed = np.zeros((varNumCon, varNumDpth))
        # NOTE: Relative difference score leads to inconsistent results.
        # for idxDiff in range(varNumCon):
        #     aryEmpMed[idxDiff, :] = np.median(
        #         np.divide(
        #             np.subtract(
        #                 aryDpth[:, lstDiff[idxDiff][0], :],
        #                 aryDpth[:, lstDiff[idxDiff][1], :]
        #                 ),
        #             np.absolute(
        #                 np.add(
        #                     aryDpth[:, lstDiff[idxDiff][0], :],
        #                     aryDpth[:, lstDiff[idxDiff][1], :]
        #                     )
        #                 )
        #             ),
        #         axis=0)
        for idxDiff in range(varNumCon):
            aryEmpMed[idxDiff, :] = np.median(
                    np.subtract(aryDpth[:, lstDiff[idxDiff][0], :],
                                aryDpth[:, lstDiff[idxDiff][1], :]),
                    axis=0)

        # Create condition labels for differences:
        lstDiffLbl = [None] * varNumCon
        for idxDiff in range(varNumCon):
            lstDiffLbl[idxDiff] = ((lstConLbl[lstDiff[idxDiff][0]])
                                   + ' minus '
                                   + (lstConLbl[lstDiff[idxDiff][1]]))
        lstConLbl = lstDiffLbl

    print('np.min(aryEmpMed)')
    print(np.min(aryEmpMed))
    print('np.max(aryEmpMed)')
    print(np.max(aryEmpMed))

    plt_dpth_prfl(aryEmpMed, None, varNumDpth, varNumCon, 80.0, varYmin,
                  varYmax, False, lstConLbl, strXlabel, strYlabel, strTtl,
                  lgcLgnd, strPath, varSizeX=1800.0, varSizeY=1600.0,
                  varNumLblY=5, varPadY=(0.1, 0.1), aryCnfLw=aryPrct[0, :, :],
                  aryCnfUp=aryPrct[1, :, :])
    # ------------------------------------------------------------------------
