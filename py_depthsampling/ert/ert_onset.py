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
from py_depthsampling.ert.ert_plt import ert_plt


def ert_onset(lstPthPic, strPthPlt, lstConLbl, strTitle=' ', varSkip=2):
    """
    Plot event-related time courses and compute response onset time.

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
    varSkip : int
        Number of pre-stimulus volumes to skip in response onset time
        calculation. This is used because the input time courses may contain
        volumes before the onset of the pre-stimulus baseline (in case of
        PacMan study, three volumes before stimulus onset are used as baseline,
        but the timecourses contain an additional two volumes before that which
        are included in the plots).

    Returns
    -------
    This function has no return value.

    Notes
    -----
    Create plot of event-related timecourse, and calcualte onset time of
    response using one-sample t-test. Onset time is highlighted in the plot.
    By providing the path of more than one pickle file, several ROI can be
    compared (e.g. time course of central & edge ROI in PacMan experiments).

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

        # On first iteration, initialise array for mean (across subjects,
        # conditions, depths) ERT:
        if idxRoi == 0:

            # Grand mean:
            aryGrndMne = np.zeros((varNumRoi, varNumVol))

            # Standard error:
            aryGrndSem = np.zeros((varNumRoi, varNumVol))

            # Vector for index of onset:
            lstOnset = [None] * varNumRoi

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
        # *** Compute onset time

        # First, average over conditions and depth levels within subject. New
        # shape: aryMneWthn[varNumSub, varNumVol].
        aryMneWthn = np.mean(aryAllSubsRoiErt, axis=(1, 2))

        # One-sample t-test, testing the hypothesis that the signal is
        # different from zeros, separately for each time point (volume).
        vecT, vecP = ttest_1samp(aryMneWthn, 0.0, axis=0)

        # Bonferroni-corrected p-threshold (alpha level of 0.05 is corrected
        # for number of time points, excluding the time points before the start
        # of the baseline period).
        varCorctP = np.divide(0.05, (float(varNumVol) - varSkip))

        # For which timepoints is the p-value below threshold?
        vecLgc = np.less_equal(vecP, varCorctP)

        # Set volumes before baseline to false:
        vecLgc[:varSkip] = False

        # Find first volume over threshold:
        varFirst = np.argmax(vecLgc)
        lstOnset[idxRoi] = varFirst

        # *********************************************************************
        # *** Weighted average (for plot)

        # Calculate mean ERT across subjects (weighted by number of vertices
        # per subject) - for the plot.
        aryMneTmp = np.average(aryMneWthn, weights=vecNumVrtcs, axis=0)

        # Weighted variance:
        aryVar = np.average(
                            np.power(
                                     np.subtract(
                                                 aryMneWthn,
                                                 aryMneTmp[None, :]
                                                 ),
                                     2.0
                                     ),
                            axis=0,
                            weights=vecNumVrtcs
                            )

        # Weighted standard deviation:
        arySd = np.sqrt(aryVar)

        # Calculate standard error of the mean (for error bar):
        arySem = np.divide(arySd, np.sqrt(varNumSub))

        aryGrndMne[idxRoi, :] = np.copy(aryMneTmp)
        aryGrndSem[idxRoi, :] = np.copy(arySem)

    # *************************************************************************
    # *** Create plot

    # Limits of axes:
    varYmin = -0.04
    varYmax = 0.02

    # Number of labels on y-axis:
    varYnum = 4

    # Convert y-axis values to percent (i.e. divide label values by 100)?
    lgcCnvPrct = True

    # Label for axes:
    strXlabel = 'Time [s]'
    strYlabel = 'fMRI signal change [%]'

    # Volume index of start of stimulus period (i.e. index of first volume
    # during which stimulus was on - for the plot):
    varStimStrt = 5

    # Volume index of end of stimulus period (i.e. index of last volume during
    # which stimulus was on - for the plot):
    varStimEnd = 10

    # Volume TR (in seconds, for the plot):
    varTr = 2.079

    # Convert stimulus onset & offset times from volume indicies to seconds:
    varStimStrt = float(varStimStrt) * varTr
    varStimEnd = float(varStimEnd) * varTr

    # Plot legend?
    lgcLgnd = True

    # Figure scaling factor:
    varDpi = 90.0

    ert_plt(aryGrndMne,
            aryGrndSem,
            1,
            varNumRoi,
            varNumVol,
            varDpi,
            varYmin,
            varYmax,
            varStimStrt,
            varStimEnd,
            varTr,
            lstConLbl,
            lgcLgnd,
            strXlabel,
            strYlabel,
            lgcCnvPrct,
            strTitle,
            strPthPlt,
            varTmeScl=1.0,
            varXlbl=5,
            varYnum=varYnum,
            tplPadY=(0.001, 0.001),
            lstVrt=lstOnset)
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

    # Output path for plots - prfix (ROI and hemisphere left open):
    strPlt = '/home/john/Dropbox/PacMan_Plots/era_onset/{}_{}.svg'

    # Name of pickle file from which to load time course data (metacondition,
    # ROI, and hemisphere left open):
    strPthPic = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/era_{}_{}.pickle'  #noqa

    # *************************************************************************
    # *** Create plots

    # Loop through ROIs, hemispheres, and conditions to create plots:
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

            ert_onset(lstPthPic, strPltTmp, lstConLbl, strTitle=strTitleTmp)

    # *************************************************************************
