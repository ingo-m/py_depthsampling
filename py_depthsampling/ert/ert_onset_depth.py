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
from scipy.stats import ttest_1samp
from scipy.interpolate import griddata
from py_depthsampling.ert.ert_plt import ert_plt
from py_depthsampling.plot.plt_dpth_prfl import plt_dpth_prfl


#lstPthPic = ['/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/era_v1_rh.pickle',
#             '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/periphery/era_v1_rh.pickle']
#varSkip=2
#idxRoi = 1
#lstBse = [2, 3, 4]
## Timepoint of first stimulus volume:
#varBse = 5
#varTr = 2.079

def ert_onset_depth(lstPthPic, strPthPlt, lstConLbl, varTr, varBse,
                    strTtl='Response onset time difference'):
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

            # Grand mean:
            # aryGrndMne = np.zeros((varNumRoi, varNumVol))

            # Standard error:
            # aryGrndSem = np.zeros((varNumRoi, varNumVol))

            # Vector for index of onset:
            # lstOnset = [None] * varNumRoi

            # Array for indices of response onset:
            aryFirst = np.zeros((varNumRoi, varNumSub, varNumDpth))

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
        vecNumVrtcs = np.zeros((varNumSub))

        idxSub = 0

        for lstItem in dicAllSubsRoiErt.values():

            # Get event related time courses from list.
            aryRoiErt = lstItem[0]

            # Get number of vertices for this subject:
            vecNumVrtcs[idxSub] = lstItem[1]

            aryAllSubsRoiErt[idxSub, :, :, :] = aryRoiErt

            idxSub += 1

        # *********************************************************************
        # *** Upsample timecourses

        # Temporal upsampling factor:
        varUp = 1000

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
        aryErtUp = np.zeros((varNumSub, varNumDpth, varNumVolUp))

        # Loop through subjects and depth levels (upsampling in 1D):
        for idxSub in range(varNumSub):
            for idxDpth in range(varNumDpth):

                # Interpolation:
                aryErtUp[idxSub, idxDpth, :] = griddata(
                        vecPosEmp, aryMneWthn[idxSub, idxDpth, :], vecPosUp,
                        method='cubic')

        # *********************************************************************
        # *** Compute onset time

        # Mean (over time) in pre-stimulus period, separately for each subject and
        # depth level.
        aryBseMne = np.mean(aryMneWthn[:, :, :varBse], axis=2)

        # Mean (over time) in pre-stimulus period, separately for each subject and
        # depth level.
        aryBseSd = np.std(aryMneWthn[:, :, :varBse], axis=2)

        # z-threshold
        varThr = 2.0

        # z-score interval around mean:
        aryLimUp = np.add(aryBseMne, np.multiply(aryBseSd, varThr))
        aryLimLow = np.subtract(aryBseMne, np.multiply(aryBseSd, varThr))

        # Which timepoints are above threshold?
        aryLgc = np.logical_or(
                               np.greater(aryErtUp, aryLimUp[:, :, None]),
                               np.less(aryErtUp, aryLimLow[:, :, None])
                               )

        # Set volumes before baseline to false (avoiding false positives on
        # first and second volume due to uncomplete recovery of signal in the
        # volumes before pre-stimulus baseline):
        aryLgc[:, :, :(varUp * varBse)] = False

        # Find first volume over threshold:
        aryFirst[idxRoi, :, :] = np.argmax(aryLgc, axis=2)

    # *************************************************************************
    # *** Response onset difference
    
    # Single subject onset time difference, shape: aryDiff[subject, depth].
    aryDiff = np.subtract(aryFirst[0, :, :], aryFirst[1, :, :])

    # Weighted mean over subjects:
    vecDiff = np.average(aryDiff, weights=vecNumVrtcs, axis=0)

    # Weighted variance:
    vecVar = np.average(
                        np.power(
                                 np.subtract(
                                             aryDiff,
                                             vecDiff[None, :]
                                             ),
                                 2.0
                                 ),
                        axis=0,
                        weights=vecNumVrtcs
                        )

    # Weighted standard deviation:
    vecSd = np.sqrt(vecVar)

    # Calculate standard error of the mean (for error bar):
    vecSem = np.divide(vecSd, np.sqrt(varNumSub))

    # Scale result to seconds & new shape (for plot function):
    aryDiff = np.divide(
                        np.multiply(vecDiff,
                                    varTr),
                        varUp).reshape(1, varNumDpth)
    arySem = np.divide(
                       np.multiply(vecSem,
                                   varTr),
                       varUp).reshape(1, varNumDpth)

    # *************************************************************************
    # *** Create plot

    varYmin = -1.0
    varYmax = 3.0
    varNumLblY = 5
    tplPadY = (0.1, 0.1)
    lgcLgnd = False
    strXlabel = 'Cortical depth'
    strYlabel = 'Time difference [s]'
    

    plt_dpth_prfl(aryDiff, arySem, varNumDpth, 1, 80.0, varYmin,
                  varYmax, False, lstConLbl, strXlabel, strYlabel, strTtl,
                  lgcLgnd, strPthPlt, varSizeX=1800.0, varSizeY=1600.0,
                  varNumLblY=varNumLblY, tplPadY=tplPadY)
    # *************************************************************************


if __name__ == "__main__":

    # *************************************************************************
    # *** Define parameters

    # Meta-condition (within or outside of retinotopic stimulus area):
    lstMtaCn = ['stimulus', 'periphery']

    # Condition label:
    lstConLbl = lstMtaCn

    # Region of interest ('v1' or 'v2'):
    lstRoi = ['v1', 'v2', 'v3']

    # Hemispheres ('lh' or 'rh'):
    lstHmsph = ['rh']

    # Output path for plots. ROI,hemisphere, and depth level left open):
    strPlt = '/home/john/Dropbox/PacMan_Plots/era_onset/onset_by_depth_{}_{}.svg'

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
                            strTtl=strTitleTmp)
    # *************************************************************************
