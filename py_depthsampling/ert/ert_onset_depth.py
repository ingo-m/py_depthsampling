# -*- coding: utf-8 -*-
"""Function of the depth sampling library."""

# Part of py_depthsampling library
# Copyright (C) 2018  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

import pickle
import numpy as np
from scipy.interpolate import griddata
from py_depthsampling.ert.utilities import onset
from py_depthsampling.plot.plt_dpth_prfl import plt_dpth_prfl


def ert_onset_depth(lstPthPic, strPthPlt, lstConLbl, varTr, varBse,
                    strTtl='Response onset time difference', strFleTpe='.svg'):
    """
    Plot response onset times over cortical depth.

    Parameters
    ----------
    lstPthPic : list
        List of strings with path(s) of pickle file containing event related
        time courses. The pickles can be created using `py_depthsampling.ert`.
        They contain one list per subject. Each list contains a numpy array
        with the event related time courses (separately for conditions and
        depth levels), and an integer - the number of vertices that contribute
        to the subject's time course. By passing in several strings, more than
        one region of interest can be plotted/analysed.
    strPthPlt : str
        Path & file name for plot.
    lstConLbl : list
        List of strings containing condition labels (one per input pickle).
    varTr : float
        Volume TR of the measurement, in seconds.
    varBse : int
        Time point of first volume after stimulus onset (index in event related
        time course). In other words, the index of the first volume on which
        the stimulus was on.
    strTtl : str
        Title for plot.
    strFleTpe : str
        File extension.

    Returns
    -------
    This function has no return value.

    Notes
    -----
    Plot response onset time by cortical depth. Event-related time courses
    are upsampled in order to estimate the onset time, separately for each
    subject.

    """
    # *************************************************************************
    # *** Preparations

    # Number of ROIs:
    varNumRoi = len(lstPthPic)

    # Number of bootstrap iterations:
    varNumIt = 1000

    # z-threshold for peak finding. Peak is identified if signal is above/below
    # varThr times mean baseline signal.
    varThr = 2.5

    # Temporal upsampling factor:
    varUp = 100

    # Upper and lower bound for percentile bootstrap confidence interval:
    varConLw = 5.0
    varConUp = 95.0

    # *************************************************************************
    # *** Loop through ROIs

    # Note that here, 'ROI' may refer to stimulus & edge ROIs within e.g. V1.

    for idxRoi in range(varNumRoi):

        # Path of current input pickle file:
        strPthPic = lstPthPic[idxRoi]

        # Load previously prepared event-related timecourses from pickle:
        dicAllSubsRoiErt = pickle.load(open(strPthPic, 'rb'))

        # Get number of subjects, conditions, cortical depth levels, time
        # points (volumes):
        varNumSub = len(dicAllSubsRoiErt)

        tplShpe = list(dicAllSubsRoiErt.values())[0][0].shape
        varNumCon = tplShpe[0]
        varNumDpth = tplShpe[1]
        varNumVol = tplShpe[2]

        # On first iteration, initialise arrays:
        if idxRoi == 0:

            # Array for indices of response onset:
            aryFirst = np.zeros((varNumRoi, varNumDpth), dtype=np.int32)

            # Array with information on whether there is a detectable response
            # at all (`True` if a response onset was detected).
            aryLgc = np.zeros((varNumRoi, varNumDpth), dtype=np.bool)

            # Array for bootstrap onset times (needed for calculation of
            # bootstrap confidence intervals of response onset difference).
            aryFirstBoo = np.zeros((varNumRoi, varNumIt, varNumDpth),
                                   dtype=np.float32)

            # Array for percentile bootstrap of response onset (for confidence
            # interval of response onset). Shape: aryFirstPrc[ROI, depth,
            # upper/lower bound].
            aryFirstPrc = np.zeros((varNumRoi, varNumDpth, 2), dtype=np.int32)

            # Array with information on whether there is a detectable response
            # at all for each bootstrap iteration (`True` if a response onset
            # was detected).
            aryLgcBoo = np.zeros((varNumRoi, varNumIt, varNumDpth),
                                 dtype=np.bool)

            # Bootstrap preparations. We will sample subjects with replacement.
            # How many subjects to sample on each iteration:
            varNumSmp = varNumSub

            # Random array with subject indicies for bootstrapping of the form
            # aryRnd[varNumIt, varNumSmp]. Each row includes the indicies of
            # the subjects to be sampled on that iteration.
            aryRnd = np.random.randint(0,
                                       high=varNumSub,
                                       size=(varNumIt, varNumSmp))

        # *********************************************************************
        # *** Subtract baseline mean

        # The input to this function are timecourses that have been normalised
        # to the pre-stimulus baseline. The datapoints are signal intensity
        # relative to the pre-stimulus baseline, and the pre-stimulus baseline
        # has a mean of one. We subtract one, so that the datapoints are
        # percent signal change relative to baseline.
        for strSubID, lstItem in dicAllSubsRoiErt.items():
            # Get event related time courses from list (second entry in list is
            # the number of vertices contained in this ROI).
            aryRoiErt = lstItem[0]
            # Subtract baseline mean:
            aryRoiErt = np.subtract(aryRoiErt, 1.0)
            # Is this line necessary (hard copy)?
            dicAllSubsRoiErt[strSubID] = [aryRoiErt, lstItem[1]]

        # *********************************************************************
        # *** Create group level ERT

        # Create across-subjects data array of the form:
        # aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]
        aryAllSubsRoiErt = np.zeros((varNumSub, varNumCon, varNumDpth,
                                     varNumVol))

        # Vector for number of vertices per subject (used for weighted
        # averaging):
        vecNumVrtcs = np.zeros((varNumSub), dtype=np.float32)

        idxSub = 0

        for lstItem in dicAllSubsRoiErt.values():

            # Get event related time courses from list.
            aryRoiErt = lstItem[0]

            # Get number of vertices for this subject:
            vecNumVrtcs[idxSub] = int(lstItem[1])

            aryAllSubsRoiErt[idxSub, :, :, :] = aryRoiErt

            idxSub += 1

        # The number of vertices are going to be used as weights, so we cast
        # to float:
        vecNumVrtcs = vecNumVrtcs.astype(np.float32)

        # *********************************************************************
        # *** Upsample timecourses

        # New number of volumes:
        varNumVolUp = varNumVol * varUp

        # Current shape:
        # aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]

        # Within subject mean (over conditions):
        aryMneWthn = np.mean(aryAllSubsRoiErt, axis=1)

        # New shape:
        # aryAllSubsRoiErt[varNumSub, varNumDpth, varNumVol]

        # Position of original datapoints in time:
        vecPosEmp = np.arange(0.0, float(varNumVol))

        # Position of upsampled datapoints in time:
        vecPosUp = np.linspace(0.0, float(varNumVol - 1), num=varNumVolUp,
                               endpoint=True)

        # Array for upsampled timecourses:
        aryErtUp = np.zeros((varNumSub, varNumDpth, varNumVolUp),
                            dtype=np.float32)

        # Loop through subjects and depth levels (upsampling in 1D):
        for idxSub in range(varNumSub):
            for idxDpth in range(varNumDpth):

                # Interpolation:
                aryErtUp[idxSub, idxDpth, :] = griddata(
                        vecPosEmp, aryMneWthn[idxSub, idxDpth, :], vecPosUp,
                        method='cubic').astype(np.float32)

        # *********************************************************************
        # *** Compute onset time

        # Weighted mean across subjects:
        aryErtUpMne = np.average(aryErtUp, weights=vecNumVrtcs, axis=0)

        # Add array dimension, new shape: aryErtUpMne[1, depth, volumes]
        aryErtUpMne = aryErtUpMne[None, ].astype(np.float32)

        # Scale baseline interval:
        varBseUp = (varUp * varBse) - 1

        # Calculate onset times:
        aryTmp01, aryTmp02 = onset(aryErtUpMne, varBseUp, varThr)
        aryFirst[idxRoi, :] = aryTmp01[0, :]
        aryLgc[idxRoi, :] = aryTmp02

        # *********************************************************************
        # *** Bootstrap onset time

        # Array for bootstrap samples:
        aryBoo = np.zeros((varNumIt, varNumSub, varNumDpth, varNumVolUp),
                          dtype=np.float32)

        # Array with number of vertices per subject for each bootstrapping
        # sample (needed for weighted averaging), shape: aryWght[iterations,
        # subjects]
        aryWght = np.zeros((varNumIt, varNumSub), dtype=np.float32)

        # Loop through bootstrap iterations:
        for idxIt in range(varNumIt):
            # Indices of current bootstrap sample:
            vecRnd = aryRnd[idxIt, :]
            # Put current bootstrap sample into array:
            aryBoo[idxIt, :, :, :] = aryErtUp[vecRnd, :, :]
            # Put number of vertices per subject into respective array (for
            # weighted averaging):
            aryWght[idxIt, :] = vecNumVrtcs[vecRnd]

        # Weightes mean for each bootstrap sample (across subjects within the
        # bootstrap sample):

        # Sum of weights over subjects (i.e. total number of vertices across
        # subjects, one value per iteration; for scaling).
        vecSum = np.sum(aryWght, axis=1)

        # Multiply depth profiles by weights (weights are broadcasted over
        # depth levels and volumes):
        aryTmp = np.multiply(aryBoo, aryWght[:, :, None, None])

        # Sum over subjects, and scale by number of vertices (sum of vertices
        # is broadcasted over conditions and depth levels):
        aryBooMne = np.divide(
                              np.sum(aryTmp, axis=1),
                              vecSum[:, None, None]
                              )
        # Resulting shape: aryBooMne[iterations, depth, volumes].

        # Delete large bootstrap array:
        del(aryBoo)

        # Calculate onset times, return value has shape [iterations, depth].
        aryFirstBoo[idxRoi, :, :], aryLgcBoo[idxRoi, :, :] = onset(
            aryBooMne, varBseUp, varThr)

    # *************************************************************************
    # *** Percentile boostrap - onset time

    # Was an onset detected in both ROIs? Shape: aryLgcBoo[iterations, depth].
    aryLgcBoo = np.min(aryLgcBoo, axis=0)

    # In order to exclude cases in which no onset was detected, we need to loop
    # through ROIs and depth levels (np.percentile does not accept weights).
    for idxRoi in range(varNumRoi):

        for idxDpth in range(varNumDpth):

            # Temporary vector for onset times on iterations without iterations
            # where no response was detected.
            vecTmp = aryFirstBoo[idxRoi, :, idxDpth]
            vecTmp = vecTmp[aryLgcBoo[:, idxDpth]]

            vecPrct = np.percentile(vecTmp,
                                    (varConLw, varConUp),
                                    axis=0)

            aryFirstPrc[idxRoi, idxDpth, 0] = vecPrct[0]
            aryFirstPrc[idxRoi, idxDpth, 1] = vecPrct[1]

    # *************************************************************************
    # *** Percentile boostrap - onset time difference

    # Was a response detected in both ROIs? Shape: vecLgc[depth].
    # vecLgc = np.min(aryLgc, axis=0)

    # Group level onset time difference, shape: vec[depth].
    vecDiff = np.subtract(aryFirst[0, :], aryFirst[1, :])

    # Scale result to seconds & new shape (for plot function):
    aryDiff = np.divide(
                        np.multiply(vecDiff,
                                    varTr),
                        float(varUp)).reshape(1, varNumDpth)

    # Onset difference for bootstrap samples:
    aryDiffBoo = np.subtract(aryFirstBoo[0, :, :],
                             aryFirstBoo[1, :, :])
    # Shape: aryDiffBoo[varNumIt, varNumDpth]

    # Scale result to seconds:
    aryDiffBoo = np.divide(
                           np.multiply(aryDiffBoo,
                                       varTr),
                           float(varUp))

    # Array for percentile bootstrap CI of onset difference.
    aryOnsetPrct = np.zeros((2, varNumDpth))
    # Shape: aryOnsetPrct[lower/upper bound, depth]

    # Was an onset detected in both ROIs? Shape: aryLgcBoo[iterations, depth].
    # aryLgcBoo = np.min(aryLgcBoo, axis=0)

    # In order to exclude cases in which no onset was detected, we need to
    # loop through depth levels (np.percentile does not accept weights).
    for idxDpth in range(varNumDpth):

        # Temporary vector for onset times on iterations without iterations
        # where no response was detected.
        vecTmp = aryDiffBoo[:, idxDpth]
        vecTmp = vecTmp[aryLgcBoo[:, idxDpth]]

        # Percentile bootstrap:
        aryOnsetPrct[:, idxDpth] = np.percentile(vecTmp,
                                                 (varConLw, varConUp),
                                                 axis=0)

    # *************************************************************************
    # *** Plot onset time

    # Scale result to seconds:
    aryFirst = np.divide(
                         np.multiply(aryFirst.astype(np.float64),
                                     varTr),
                         float(varUp))
    aryFirstPrc = np.divide(
                            np.multiply(aryFirstPrc.astype(np.float64),
                                        varTr),
                            float(varUp))

    # Subtract per-stimulus baseline:
    aryFirst = np.subtract(aryFirst, (float(varBse) * varTr))
    aryFirstPrc = np.subtract(aryFirstPrc, (float(varBse) * varTr))

    # Output file path:
    strPthOut = strPthPlt + 'onset_by_depth' + strFleTpe

    # Plot parameters:
    varYmin = 0.0
    varYmax = 6.0
    varNumLblY = 4
    tplPadY = (0.0, 0.1)
    lgcLgnd = True
    strXlabel = 'Cortical depth'
    strYlabel = 'Onset time [s]'
    aryClr = np.array([[49.0, 128.0, 182.0],
                       [253.0, 134.0, 47.0]])
    aryClr = np.divide(aryClr, 255.0)

    plt_dpth_prfl(aryFirst, None, varNumDpth, 2, 80.0, varYmin,
                  varYmax, False, lstConLbl, strXlabel, strYlabel, strTtl,
                  lgcLgnd, strPthOut, varSizeX=1200.0, varSizeY=1000.0,
                  varNumLblY=varNumLblY, tplPadY=tplPadY, aryClr=aryClr,
                  aryCnfLw=aryFirstPrc[:, :, 0], aryCnfUp=aryFirstPrc[:, :, 1])

    # *************************************************************************
    # *** Plot onset time difference

    # Plot parameters:
    varYmin = 0.0
    varYmax = 3.0
    varNumLblY = 4
    tplPadY = (0.0, 0.1)
    lgcLgnd = False
    strXlabel = 'Cortical depth'
    strYlabel = 'Time difference [s]'

    # Output file path:
    strPthOut = strPthPlt + 'onsetdiff_by_depth' + strFleTpe

    plt_dpth_prfl(aryDiff, None, varNumDpth, 1, 80.0, varYmin,
                  varYmax, False, lstConLbl, strXlabel, strYlabel, strTtl,
                  lgcLgnd, strPthOut, varSizeX=1200.0, varSizeY=1000.0,
                  varNumLblY=varNumLblY, tplPadY=tplPadY,
                  aryCnfLw=aryOnsetPrct[0, :][None, :],
                  aryCnfUp=aryOnsetPrct[1, :][None, :])

    # *************************************************************************


if __name__ == "__main__":

    # *************************************************************************
    # *** Define parameters

    # Meta-condition (within or outside of retinotopic stimulus area):
    lstMtaCn = ['stimulus', 'periphery']

    # Condition label:
    lstConLbl = ['stimulus', 'edge']

    # Region of interest ('v1' or 'v2'):
    lstRoi = ['v1', 'v2', 'v3']

    # Hemispheres ('lh' or 'rh'):
    lstHmsph = ['rh']

    # Output path for plots. ROI,hemisphere, and depth level left open):
    strPlt = '/home/john/Dropbox/PacMan_Plots/era_onset/{}_{}_'

    # Output file extension:
    strFleTpe = '.png'

    # Name of pickle file from which to load time course data (metacondition,
    # ROI, and hemisphere left open):
    strPthPic = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/era_{}_{}.pickle'  #noqa

    # Volume TR [s]:
    varTr = 2.079

    # Time point of first volume after stimulus onset (index in event related
    # time course):
    varBse = 5

    # *************************************************************************
    # *** Create plots

    # Loop through ROIs, hemispheres, and depth levels to create plots:
    for idxRoi in range(len(lstRoi)):
        for idxHmsph in range(len(lstHmsph)):

            # Complete path of input pickle (stimulus centre):
            strPthPic01 = strPthPic.format(lstMtaCn[0], lstRoi[idxRoi],
                                           lstHmsph[idxHmsph])

            # Complete path of input pickle (stimulus edge):
            strPthPic02 = strPthPic.format(lstMtaCn[1], lstRoi[idxRoi],
                                           lstHmsph[idxHmsph])

            lstPthPic = [strPthPic01, strPthPic02]

            strTitleTmp = lstRoi[idxRoi].upper()

            strPltTmp = strPlt.format(lstRoi[idxRoi], lstHmsph[idxHmsph])

            ert_onset_depth(lstPthPic, strPltTmp, lstConLbl, varTr, varBse,
                            strTtl=strTitleTmp, strFleTpe=strFleTpe)
    # *************************************************************************
