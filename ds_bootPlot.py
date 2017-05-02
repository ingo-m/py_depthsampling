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
from ds_pltAcrDpth import funcPltAcrDpth

# objDpth = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy'
# strPath = '/home/john/Desktop/tmp/tmp.png'
# varNumIt=10000
# varConLw=2.5
# varConUp=97.5

def bootPlot(objDpth, strPath, varNumIt=10000, varConLw=2.5, varConUp=97.5,
             strTtl='', strXlabel='Cortical depth level (equivolume)',
             strYlabel = 'fMRI signal change [arbitrary units]',
             lgcLgnd=False):
    """
    Plot across-subject median cortical depth profiles with percentile
    bootstrap confidence intervals.

    Parameters
    ----------
    objDpth : np.array or str
        Array with single-subject cortical depth profiles, of the form:
        aryDpth[idxSub, idxCondition, idxDpth]. Either a numpy array or a
        string with the path to an npy file containing the array.
    strPath : str
        Output path for plot.
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
    strXlabel : str
        Label for x axis.
    strYlabel : str
        Label for y axis.
    lgcLgnd : bool
        Whether to show a legend.

    Returns
    -------
    None : None
        This function has no return value.

    Notes
    -----
    Plot across-subject median cortical depth profiles with percentile
    bootstrap confidence intervals. This function bootstraps (i.e. resamples
    with replacement) from an array of single-subject depth profiles,
    calculates the median & percentile range across bootstrap iterations and
    plots the resulting median & confidence intervals along the cortical depth.

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

    # Array for bootstrap samples, of the form
    # aryBoo[idxIteration, idxSubject, idxCondition, idxDpth]):
    aryBoo = np.zeros((varNumIt, varNumSub, varNumCon, varNumDpth))

    # ------------------------------------------------------------------------
    # *** Bootstrap
    
    # Loop through bootstrap iterations:
    for idxIt in range(varNumIt):
        # Indices of current bootstrap sample:
        vecRnd = aryRnd[idxIt, :]
        # Put current bootstrap sample into array:
        aryBoo[idxIt, :, :, :] = aryDpth[vecRnd, :, :]

    # Median for each bootstrap sample (across subjects within the bootstrap
    # sample):
    aryMedi01 = np.median(aryBoo, axis=1)

    # Median across bootstrap samples:
    aryMedi02 = np.median(aryMedi01, axis=0)

    # Delete large bootstrap array:
    del(aryBoo)

    # Percentile bootstrap for median:
    aryPrct = np.percentile(aryMedi01, (varConLw, varConUp), axis=0)

    # ------------------------------------------------------------------------
    # *** Plot result

    # Condition labels:
    lstConLbl = ['2.5%', '6.1%', '16.3%', '72.0%']

    # Labels for axes:
    strXlabel = 'Cortical depth level (equivolume)'
    strYlabel = 'fMRI signal change [arbitrary units]'

    # Mean?
    # aryMedi02 = np.mean(aryDpth, axis=0)

    funcPltAcrDpth(aryMedi02, None, varNumDpth, varNumCon, 80.0, 0.0, 2.0,
                   False, lstConLbl, strXlabel, strYlabel, strTtl, lgcLgnd,
                   strPath, varSizeX=1800.0, varSizeY=1600.0, varNumLblY=5,
                   varPadY=(0.1, 0.1), aryCnfLw=aryPrct[0, :, :],
                   aryCnfUp=aryPrct[1, :, :])
    # ------------------------------------------------------------------------

# bootPlot(objDpth, strPath, strTtl='V1 before deconvolution', varNumIt=10000)
