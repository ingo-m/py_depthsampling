# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

# Part of py_depthsampling library
# Copyright (C) 2018  Ingo Marquardt
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
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from py_depthsampling.plot.plt_dpth_prfl import plt_dpth_prfl


def diff_sem(objDpth, strPath, lstCon, lstConLbl, strTtl='', varYmin=0.0,  #noqa
             varYmax=2.0, tplPadY=(0.0, 0.0), varNumLblY=5, varDpi=80.0,
             strXlabel='Cortical depth level (equivolume)',
             strYlabel='fMRI signal change [arbitrary units]',
             lgcLgnd=False, lstDiff=None, vecNumInc=None, strParam='mean'):
    """
    Plot across-subject cortical depth profiles with SEM.

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
    strTtl : str
        Plot title.
    varYmin : float
        Minimum of Y axis.
    varYmax : float
        Maximum of Y axis.
    tplPadY : tuple
        Padding around labelled values on y.
    varNumLblY : int
        Number of labels on y axis.
    varDpi : float
        Resolution of the output figure.
    strXlabel : str
        Label for x axis.
    strYlabel : str
        Label for y axis.
    lgcLgnd : bool
        Whether to show a legend.
    lstDiff : list or None
        If None, the depth profiles are plotted separately for each condition.
        If a list of tuples of condition indices is provided, the difference
        between the two conditions is calculated, and is plotted. The the
        second condition from the tuple is subtracted from the first (e.g. if
        lstDiff = [(0, 1)], then condition 1 is subtracted from condition 0).
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
    Plot across-subject mean or median cortical depth profiles with SEM.

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

    # Define R function for calculation of weighted median:
    strFuncR = """
     funcR <- function(lstData, lstWght){
     library(spatstat)
     varWm <- weighted.median(lstData, lstWght)
     return(varWm)
     }
    """
    objFuncR = robjects.r(strFuncR)

    if not(lstDiff is None):
        # Set number of comparisons:
        varNumCon = len(lstDiff)

    # Array for SEM:
    arySem = np.zeros((varNumCon, varNumDpth))

    # ------------------------------------------------------------------------
    # *** Calculate parameters

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

                # Multiply depth profiles by weights (weights are broadcasted
                # over depth levels):
                aryDiff = np.multiply(aryDiff, vecNumInc[:, None])

                # Sum over subjects, and scale by total number of vertices:
                aryEmpMne[idxDiff, :] = np.divide(
                                                  np.sum(aryDiff, axis=0),
                                                  varSum
                                                  )

                # Formula for SEM according to Franz & Loftus (2012). Standard
                # errors and confidence intervals in within-subjects designs:
                # generalizing Loftus and Masson (1994) and avoiding the biases
                # of alternative accounts. Psychonomic Bulletin & Review,
                # 19(3), p. 398.
                arySem[idxDiff, :] = \
                    np.sqrt(
                            np.multiply(
                                        np.divide(
                                                  1.0,
                                                  np.multiply(
                                                              float(varNumSub),
                                                              (float(varNumSub) - 1.0)
                                                              )
                                                  ),
                                        np.sum(
                                               np.power(
                                                        np.subtract(
                                                                    np.subtract(
                                                                                aryDpth[:, lstDiff[idxDiff][0], :],
                                                                                aryDpth[:, lstDiff[idxDiff][1], :]
                                                                                ),
                                                                    np.mean(
                                                                            np.subtract(
                                                                                        aryDpth[:, lstDiff[idxDiff][0], :],
                                                                                        aryDpth[:, lstDiff[idxDiff][1], :]),
                                                                            axis=0
                                                                            ),
                                                                    ),
                                                        2.0
                                                        ),
                                               axis=0
                                               )
                                        )
                             )

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

                # TODO: SEM for median

        # Create condition labels for differences:
        lstDiffLbl = [None] * varNumCon
        for idxDiff in range(varNumCon):
            lstDiffLbl[idxDiff] = ((lstConLbl[lstDiff[idxDiff][0]])
                                   + ' minus '
                                   + (lstConLbl[lstDiff[idxDiff][1]]))
        lstConLbl = lstDiffLbl

    # ------------------------------------------------------------------------
    # *** Plot results

    # For the plots of condition differences we use a different colour schemea
    # as for the plots of individual condition depth profiles.

    # Prepare colour map:
    objClrNorm = colors.Normalize(vmin=0, vmax=9)
    objCmap = plt.cm.tab10
    aryClr = np.zeros((varNumCon, 3))

    # Use custom colour scheme for PacMan data (three differences):
    if varNumCon == 3:
        aryClr[0, :] = objCmap(objClrNorm(9))[0:3]
        aryClr[1, :] = objCmap(objClrNorm(6))[0:3]
        aryClr[2, :] = objCmap(objClrNorm(8))[0:3]

    # Use tab10 colour map (but leave out first items, as those are used for
    # single condition plots).
    else:
        for idxCon in range(varNumCon):
            aryClr[idxCon, :] = objCmap(objClrNorm(varNumCon + 2 - idxCon))

    plt_dpth_prfl(aryEmpMne, arySem, varNumDpth, varNumCon, varDpi, varYmin,
                  varYmax, False, lstConLbl, strXlabel, strYlabel, strTtl,
                  lgcLgnd, strPath, varSizeX=1800.0, varSizeY=1600.0,
                  varNumLblY=varNumLblY, tplPadY=tplPadY, aryClr=aryClr)
    # ------------------------------------------------------------------------
