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
from py_depthsampling.ert.ert_plt import ert_plt


def ert_onset_sngle(lstPthPic, strPthPlt, lstConLbl, strTitle=' ',
                    lstDpth=None):
    """
    Plot single-subject event-related time courses.

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
    lstDpth : list
        Nested list with depth levels to average over. For instance, if
        `lstDpth = [0, 1, 2]`, the average over the first three depth levels is
        calculated. If `lstDpth= [None]`,  average over all depth levels.

    Returns
    -------
    This function has no return value.

    Notes
    -----
    Create plot of single-subject event-related timecourse. By providing the 
    path of more than one pickle file, several ROI can be compared (e.g. time
    course of central & edge ROI in PacMan experiments).

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

            # Grand mean ERT (across subjects, conditions, depths):
            aryGrndMne = np.zeros((varNumRoi, varNumVol))

            # Standard error:
            aryGrndSem = np.zeros((varNumRoi, varNumVol))

            # Vector for index of onset:
            lstOnset = [None] * varNumRoi
            
            # List for single-subject ERTs (will be concatenated later):
            lstSngle = []

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
        # *** Average across depth levels and conditions

        # Current shape:
        # aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]

        if lstDpth is None:

            # Average over conditions and over all depth levels within subject.
            # New shape: aryMneWthn[varNumSub, varNumVol].
            aryMneWthn = np.mean(aryAllSubsRoiErt, axis=(1, 2))

        elif len(lstDpth) == 1:

            # If there is only one depth level, averaging over depth levels
            # does not make sense. Take that depth level, and average over
            # conditions.

            # Select depth level:
            aryAllSubsRoiErt = aryAllSubsRoiErt[:, :, lstDpth[0], :]
            # New shape: aryAllSubsRoiErt[varNumSub, varNumCon, varNumVol]

            # Mean over conditions:
            aryMneWthn = np.mean(aryAllSubsRoiErt, axis=1)

        else:

            # Select depth levels:
            aryAllSubsRoiErt = aryAllSubsRoiErt[:, :, lstDpth, :]

            # Mean over conditions & depth levels:
            aryMneWthn = np.mean(aryAllSubsRoiErt, axis=(1, 2))
            
        # Append to list for single-subject ERTs:
        lstSngle.append(aryMneWthn)


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
    # *** Combine single subject & group average data for plot
    
    # Concatenate single subject data (resulting array shape:
    # arySngle[subject*ROIs, time]).
    arySngle = np.concatenate(lstSngle, axis=0)

    # Corresponding dummy array for error shading:
    arySngleErr = np.zeros(arySngle.shape)

    # Concatenate group average and single subject data:
    aryConc = np.concatenate((aryGrndMne, arySngle), axis=0)
    aryConcErr = np.concatenate((aryGrndSem, arySngleErr), axis=0)


    # RGB colours for two conditions - makeshift solution, will not work for
    # more than two conditions.
    lstClr01 = [float(x)/255.0 for x  in [255, 127, 14]]
    lstClr02 = [float(x)/255.0 for x  in [31, 119, 180]]
    lstClr = [[lstClr01] * varNumSub
              + [lstClr02] * varNumSub][0]

    # Prepend colour list for the group average lines:
    lstClr = [lstClr01] + [lstClr02] + lstClr

    # Line thickness group averages:
    varLneGrp = 8.0
    # Line thickness single subject:
    varLneSngl = 2.0
    # List with line thickness for all lines:
    lstLne = [varLneGrp] * varNumRoi + [varLneSngl] * (varNumRoi * varNumSub)

    # Total number of lines:
    varNumLne = len(lstLne)

    # *************************************************************************
    # *** Create plot

    # Limits of axes:
    varYmin = -0.04
    varYmax = 0.02

    # Number of labels on y-axis:
    varYnum = 4

    # Padding at x & y axis limits
    tplPadY = (0.008, 0.005)

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

    ert_plt(aryConc,
            aryConcErr,
            1,
            varNumLne,
            varNumVol,
            varDpi,
            varYmin,
            varYmax,
            varStimStrt,
            varStimEnd,
            varTr,
            None,
            lgcLgnd,
            strXlabel,
            strYlabel,
            lgcCnvPrct,
            strTitle,
            strPthPlt,
            varTmeScl=1.0,
            varXlbl=5,
            varYnum=varYnum,
            tplPadY=tplPadY,
            lstLne=lstLne,
            lstClr=lstClr)
    # *************************************************************************


if __name__ == "__main__":

    # *************************************************************************
    # *** Define parameters

    # Meta-condition (within or outside of retinotopic stimulus area):
    lstMtaCn = ['stimulus', 'periphery']

    # Condition label:
    lstConLbl = lstMtaCn

    # Region of interest ('v1' or 'v2'):
    lstRoi = ['v1']  # , 'v2', 'v3']

    # Hemispheres ('lh' or 'rh'):
    lstHmsph = ['rh']

    # Nested list with depth levels to average over. For instance, if `lstDpth
    # = [[0, 1, 2], [3, 4, 5]]`, on a first iteration, the average over the
    # first three depth levels is calculated, and on a second iteration the
    # average over the subsequent three depth levels is calculated. If
    # 1lstDpth= [[None]]1, average over all depth levels.
    lstDpth = [None]
    # lstDpth = [[x] for x in range(11)]
    # Depth level condition labels (output file will contain this label):
    lstDpthLbl = ['allGM']
    # lstDpthLbl = [str(x) for x in range(11)]

    # Output path for plots. ROI, hemisphere, and depth level left open:
    strPlt = '/home/john/Dropbox/PacMan_Plots/era_onset_sngle/{}_{}_{}.svg'

    # Name of pickle file from which to load time course data (metacondition,
    # ROI, and hemisphere left open):
    strPthPic = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/era_{}_{}.pickle'  #noqa

    # *************************************************************************
    # *** Create plots

    # Loop through ROIs, hemispheres, and depth levels to create plots:
    for idxRoi in range(len(lstRoi)):
        for idxHmsph in range(len(lstHmsph)):
            for idxDpth in range(len(lstDpth)):

                # Complete path of input pickle (stimulus centre):
                strPthPic01 = strPthPic.format(lstMtaCn[0], lstRoi[idxRoi],
                                               lstHmsph[idxHmsph])

                # Complete path of input pickle (stimulus edge):
                strPthPic02 = strPthPic.format(lstMtaCn[1], lstRoi[idxRoi],
                                               lstHmsph[idxHmsph])

                lstPthPic = [strPthPic01, strPthPic02]

                strTitleTmp = lstRoi[idxRoi].upper()

                strPltTmp = strPlt.format(lstRoi[idxRoi], lstHmsph[idxHmsph],
                                          lstDpthLbl[idxDpth])

                ert_onset_sngle(lstPthPic, strPltTmp, lstConLbl,
                                strTitle=strTitleTmp, lstDpth=lstDpth[idxDpth])
    # *************************************************************************
