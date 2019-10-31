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
from py_depthsampling.ert.ert_get_sub_data import ert_get_sub_data
from py_depthsampling.ert.ert_plt import ert_plt


def ert_main(lstSubId, lstCon, lstConLbl, strMtaCn, lstHmsph, strRoi,
             strVtkMsk, strVtkPth, varTr, varNumDpth, varNumVol, varStimStrt,
             varStimEnd, strPthPic, lgcPic, strPltOtPre, strPltOtSuf,
             varNumLne=2, strPrcdData='SCALARS', strXlabel='Time [s]',
             strYlabel='Percent signal change', varAcrSubsYmin=-0.06,
             varAcrSubsYmax=0.04, tplPadY=(0.001, 0.001), lgcCnvPrct=True,
             lgcLgnd01=True, lgcLgnd02=True, varTmeScl=1.0, varXlbl=5,
             varYnum=6, varDpi=100.0):
    """
    Plot event-related timecourses sampled across cortical depth levels.

    Parameters
    ----------
    lstSubId : list
        List of subject IDs (list of strings, e.g. ['20171023', ...]).
    lstCon : list
        Condition levels (used to complete file names).
    lstConLbl : list
        Condition labels (for plot legend).
    strMtaCn : string
        Metacondition ('stimulus' or 'periphery').
    lstHmsph : list
        Hemispheres ('rh' and/or 'lh').
    strRoi : string
        Region of interest ('v1', 'v2', or 'v3').
    strVtkMsk : string
        Path of vertex inclusion mask (subject ID, hemisphere, subject ID, ROI,
        and metacondition left open).
    strVtkPth : string
        Base name of single-volume vtk meshes that together make up the
        timecourse (subject ID, hemisphere, stimulus level, and volume index
        left open).
    varTr : int
        Volume TR (in seconds, for the plot).
    varNumDpth : int
        Number of cortical depths.
    varNumVol : int
        Number of timepoints in functional time series.
    varStimStrt : int
        Volume index of start of stimulus period (i.e. index of first volume
        during which stimulus was on - for the plot).
    varStimEnd : int
        Volume index of end of stimulus period (i.e. index of last volume
        during which stimulus was on - for the plot).
    strPthPic : string
        Name of pickle file from which to load time course data or save time
        course data to (metacondition and ROI left open).
    lgcPic : bool
        Load data from previously prepared pickle? If 'False', data is loaded
        from vtk meshes and saved as pickle.
    strPltOtPre : string
        Output path for plots - prefix, i.e. path and file name (metacondition
        and ROI left open).
    strPltOtSuf : string
        Output path for plots - suffix, i.e. file extension.
    varNumLne : int
        Number of lines between vertex-identification-string and first data
        point.
    strPrcdData : string
        Beginning of string which precedes vertex data in data vtk files (i.e.
        in the statistical maps).
    strXlabel : string
        Label for x axis.
    strYlabel : string
        Label for y axis.
    varAcrSubsYmin : float
        Lower limit of y-axis.
    varAcrSubsYmax : float
        Upper limit of y-axis.
    tplPadY : tuple
        Padding around labelled values on y.
    lgcCnvPrct : bool
        Convert y-axis values to percent (i.e. divide label values by 100)?
    lgcLgnd01 : bool
        Whether to plot legend - single subject plots.
    lgcLgnd02 : bool
        Whether to plot legend - group plots.
    varTmeScl : float
        Time scaling factor (factor by which timecourse was temporally
        upsampled; if it was not upsampled, varTmeScl = 1.0).
    varXlbl : int
        Which x-values to label on the axis (e.g., if `varXlbl = 2`, every
        second x-value is labelled).
    varYnum : int
        Number of labels on the y axis.
    varDpi : float
        Resolution of resulting figure.

    Returns
    -------
    This function has no return value. Plots are saved to disk.

    Notes
    -----
    The input to this module are custom-made 'mesh time courses'. Timecourses
    have to be cut into event-related segments and averaged across trials
    (using the 'cut_sgmnts.py' script of the depth-sampling library, or
    automatically as part of the PacMan analysis pipeline,
    n_03x_py_evnt_rltd_avrgs.py). Depth-sampling has to be performed with CBS
    tools, resulting in a 3D mesh for each time point. Here, 3D meshes (with
    values for all depth-levels at one point in time, for one condition) are
    combined across time and conditions to be plotted and analysed.
    """
    # *************************************************************************
    # *** Preparations

    # Convert stimulus onset & offset times from volume indicies to seconds:
    varStimStrt = float(varStimStrt) * varTr
    varStimEnd = float(varStimEnd) * varTr

    # *************************************************************************
    # *** Load data

    print('-Event-related timecourses depth sampling')

    # Number of subjects:
    varNumSub = len(lstSubId)

    # Number of conditions:
    varNumCon = len(lstCon)

    # Complete strings:
    strPthPic = strPthPic.format(strMtaCn, strRoi)
    strPltOtPre = strPltOtPre.format(strMtaCn, strRoi)

    if lgcPic:

        print('---Loading data pickle')

        # Load previously prepared event-related timecourses from pickle:
        dicAllSubsRoiErt = pickle.load(open(strPthPic, 'rb'))

    else:

        print('---Loading data from vtk meshes')

        # Dictionary for ROI event-related averages. NOTE: Once the
        # Depth-sampling can be scripted, this array should be extended to
        # contain one timecourse per trial (per subject & depth level).

        # The keys for the dictionary will be the subject IDs, and for each
        # subject there is an array of the form:
        # aryRoiErt[varNumCon, varNumDpth, varNumVol]
        dicAllSubsRoiErt = {}

        # Loop through subjects and load data:
        for strSubID in lstSubId:

            print(('------Subject: ' + strSubID))

            # List for event related time courses from each hemisphere:
            lstErt = [None] * len(lstHmsph)

            # Loop through hemispheres:
            for idxHmsph in range(len(lstHmsph)):

                # Complete file path of vertex inclusion mask for current
                # subject:
                # strVtkMskTmp = strVtkMsk.format(strSubID, lstHmsph[idxHmsph],
                #                                 strSubID, strRoi, strMtaCn)

                # Load data for current subject (returns a list with two
                # elements:
                # First, an array of the form
                #    aryRoiErt[varNumCon, varNumDpth, varNumVol]
                # and the number of vertices contained in the ROI (a single
                # integer):
                # lstErt[idxHmsph] = ert_get_sub_data(strSubID,
                #                                     lstHmsph[idxHmsph],
                #                                     strVtkMskTmp,
                #                                     strVtkPth,
                #                                     lstCon,
                #                                     varNumVol,
                #                                     varNumDpth,
                #                                     strPrcdData,
                #                                     varNumLne)

                # -------------------------------------------------------------
                # --- Makeshift solution for asymmetrical ROIs ---

                # In the surface experiment, the central ROI needs to be
                # slightly different for the 'Kanizsa' condition than for the
                # 'Kanizsa rotated' condition. In the 'Kanizsa' condition, the
                # central ROI is a square (avoiding the illusory contours). In
                # the 'Kanizsa roated' condition, the central ROI is a diamond
                # of the size than the square (rotated by 45 deg). The diamond
                # avoids the rotated Kanizsa inducer (which extends further
                # towards the centre of the field of view, because the 'mouth'
                # of the inducer is oriented away from the centre).

                # Similarly, in the texture/uniform background control
                # experiment, the ROI needs to be ajusted according for the
                # square vs. Pac-Man stimulus.

                # Array for event-related timecourses:
                aryRoiErt = np.zeros((varNumCon, varNumDpth, varNumVol),
                                     dtype=np.float16)

                # lstCon = ['bright_square', 'kanizsa', 'kanizsa_rotated']
                # lstMtaCn = ['centre', 'edge', 'background']

                # Number of vertices:
                varNumVrtc = 0

                # Loop through conditions:
                for idxCon in range(varNumCon):

                    # Current condition:
                    strTmpCon = lstCon[idxCon]

                    # ROI for background or square-centre conditions, without
                    # adjustments. Will be replaced for other condition/ROI
                    # combinations.
                    strMtaCnTmp = strMtaCn

                    # *** Main surface experiment ***
                    # If processing the central ROI for the 'Kanizsa rotated'
                    # condition, don't use the square ROI mask, but use the
                    # diamond ROI instead.
                    lgcTmp01 = ((strTmpCon == 'kanizsa_rotated')
                                and (strMtaCn == 'centre'))

                    if lgcTmp01:
                        print(('------Using diamond ROI (instead of square '
                               + 'ROI) for Kanizsa rotated condition.'))
                        # Use diamond ROI:
                        strMtaCnTmp = 'diamond'

                    # *** Uniform/texture background control experiment ***
                    # The Pac-Man stimulus and the square stimulus have
                    # different ROIs.

                    # 'Bright square' stimulus:
                    elif 'bright_square_' in strTmpCon:
                        print(('------Using square ROI.'))
                        if strMtaCn == 'centre':
                            # Centre ROI:
                            strMtaCnTmp = 'square_centre'
                        elif strMtaCn == 'edge':
                            # Edge ROI:
                            strMtaCnTmp = 'square_edge'

                    # 'Pac-Man' stimulus:
                    elif 'pacman_static_' in strTmpCon:
                        print(('------Using Pac-Man ROI.'))
                        if strMtaCn == 'centre':
                            # Centre ROI:
                            strMtaCnTmp = 'pacman_centre'
                        elif strMtaCn == 'edge':
                            # Edge ROI:
                            strMtaCnTmp = 'pacman_edge'

                    # Complete file path of vertex inclusion mask for current
                    # subject:
                    strVtkMskTmp = strVtkMsk.format(strSubID,
                                                    lstHmsph[idxHmsph],
                                                    strSubID,
                                                    strRoi,
                                                    strMtaCnTmp)

                    # Load data for current subject (returns a list with two
                    # elements:
                    # First, an array of the form
                    #    aryRoiErt[varNumCon, varNumDpth, varNumVol]
                    # and the number of vertices contained in the ROI (a single
                    # integer):
                    aryRoiErt[idxCon, :, :], varTmp01 = ert_get_sub_data(
                        strSubID,
                        lstHmsph[idxHmsph],
                        strVtkMskTmp,
                        strVtkPth,
                        [lstCon[idxCon]],  # Makeshift solution asymmetric ROIs
                        varNumVol,
                        varNumDpth,
                        strPrcdData,
                        varNumLne)

                    # Number of vertices should not depend on order of
                    # conditions; use maximum across conditions:
                    if varNumVrtc < varTmp01:
                        varNumVrtc = varTmp01

                lstErt[idxHmsph] = [np.copy(aryRoiErt), varNumVrtc]
                # --- End of makeshift solution ---
                # -------------------------------------------------------------

            # In case only one hemisphere is analysed, there is no need to
            # average.
            if (len(lstHmsph) == 1):

                # Single hemisphere to dictionary:
                dicAllSubsRoiErt[strSubID] = lstErt[0]

            else:

                # Weighted average of both hemispheres:
                aryErt = np.add(
                                np.multiply(lstErt[0][0],
                                            float(lstErt[0][1])),
                                np.multiply(lstErt[1][0],
                                            float(lstErt[1][1]))
                                )

                # Number of vertices of both hemispheres together:
                varNumVrtc = np.add(lstErt[0][1], lstErt[1][1])

                # Weighted average of both hemispheres:
                aryErt = np.divide(aryErt, varNumVrtc)

                # Mean of both hemispheres to dictionary:
                dicAllSubsRoiErt[strSubID] = [aryErt, varNumVrtc]

        # Save event-related timecourses to disk as pickle:
        pickle.dump(dicAllSubsRoiErt, open(strPthPic, 'wb'))

    # *************************************************************************
    # *** Subtract baseline mean

    # The input to this function are timecourses that have been normalised to
    # the pre-stimulus baseline. The datapoints are signal intensity relative
    # to the pre-stimulus baseline, and the pre-stimulus baseline has a mean of
    # one. We subtract one, so that the datapoints are percent signal change
    # relative to baseline.
    for strSubID, lstItem in dicAllSubsRoiErt.items():
        # Get event related time courses from list (second entry in list is the
        # number of vertices contained in this ROI).
        aryRoiErt = lstItem[0]
        # Subtract baseline mean:
        aryRoiErt = np.subtract(aryRoiErt, 1.0)
        # Is this line necessary (hard copy)?
        dicAllSubsRoiErt[strSubID] = [aryRoiErt, lstItem[1]]

    # *************************************************************************
    # *** Plot single subjet results

    if False:

        print('---Ploting single-subjects event-related averages')

        # Loop through subjects:
        for strSubID, lstItem in dicAllSubsRoiErt.items():

            # Get event related time courses from list (second entry in list is
            # the number of vertices contained in this ROI).
            aryRoiErt = lstItem[0]

            # Calculate mean across depth (within subjects):
            aryRoiErt = np.mean(aryRoiErt, axis=1)

            # Title for plot:
            strTmpTtl = strSubID

            # Output filename:
            strTmpPth = (strPltOtPre + strSubID + strPltOtSuf)

            # We don't have the variances across trials (within subjects),
            # therefore we create an empty array as a placeholder. NOTE:
            # This should be replaced by between-trial variance once the
            # depth sampling is fully scriptable.
            aryDummy = np.zeros(aryRoiErt.shape)

            # Plot single subject ERT (mean over depth levels):
            ert_plt(aryRoiErt,
                    aryDummy,
                    1,  # varNumDpth
                    varNumCon,
                    varNumVol,
                    varDpi,
                    varAcrSubsYmin,
                    varAcrSubsYmax,
                    varStimStrt,
                    varStimEnd,
                    varTr,
                    lstConLbl,
                    lgcLgnd01,
                    strXlabel,
                    strYlabel,
                    lgcCnvPrct,
                    strTmpTtl,
                    strTmpPth,
                    varTmeScl=varTmeScl,
                    varXlbl=varXlbl,
                    varYnum=varYnum,
                    tplPadY=tplPadY)

    # *************************************************************************
    # *** Plot across-subjects average

    print('---Ploting across-subjects average')

    # Create across-subjects data array of the form:
    # aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]
    aryAllSubsRoiErt = np.zeros((varNumSub, varNumCon, varNumDpth, varNumVol))

    # Vector for number of vertices per subject (used for weighted averaging):
    vecNumVrtcs = np.zeros((varNumSub))

    idxSub = 0

    for lstItem in dicAllSubsRoiErt.values():

        # Get event related time courses from list.
        aryRoiErt = lstItem[0]

        # Get number of vertices for this subject:
        vecNumVrtcs[idxSub] = lstItem[1]

        aryAllSubsRoiErt[idxSub, :, :, :] = aryRoiErt

        idxSub += 1

    # Calculate mean event-related time courses (mean across subjects):
    aryRoiErtMean = np.average(aryAllSubsRoiErt, weights=vecNumVrtcs, axis=0)

    # Weighted variance:
    aryAcrSubDpthVar = np.average(
                                  np.power(
                                           np.subtract(
                                                       aryAllSubsRoiErt,
                                                       aryRoiErtMean[None,
                                                                     :, :, :]
                                                       ),
                                           2.0
                                           ),
                                  axis=0,
                                  weights=vecNumVrtcs
                                  )

    # Weighted standard deviation:
    aryAcrSubDpthSd = np.sqrt(aryAcrSubDpthVar)

    # Calculate standard error of the mean (for error bar):
    aryRoiErtSem = np.divide(aryAcrSubDpthSd,
                             np.sqrt(varNumSub))

    # Loop through depth levels:
    # for idxDpth in range(0, varNumDpth):
    if False:  # for idxDpth in [0, 5, 10]:

        # Title for plot:
        # strTmpTtl = ('Event-related average, depth level ' + str(idxDpth))
        strTmpTtl = ''

        # Output filename:
        strTmpPth = (strPltOtPre + 'acr_subs_dpth_' + str(idxDpth)
                     + strPltOtSuf)

        # The mean array now has the form:
        # aryRoiErtMean[varNumCon, varNumDpth, varNumVol]

        # We create one plot per depth-level.
        ert_plt(aryRoiErtMean[:, idxDpth, :],
                aryRoiErtSem[:, idxDpth, :],
                varNumDpth,
                varNumCon,
                varNumVol,
                varDpi,
                varAcrSubsYmin,
                varAcrSubsYmax,
                varStimStrt,
                varStimEnd,
                varTr,
                lstConLbl,
                lgcLgnd02,
                strXlabel,
                strYlabel,
                lgcCnvPrct,
                strTmpTtl,
                strTmpPth,
                varTmeScl=varTmeScl,
                varXlbl=varXlbl,
                varYnum=varYnum,
                tplPadY=tplPadY)

    # *************************************************************************
    # *** Plot across-subjects average (mean across depth levels)

    print('---Ploting across-subjects average - mean across depth levels')

    # Event-related time courses have the form:
    # aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]

    # Calculate mean across depth (within subjects):
    aryMneDpth = np.mean(aryAllSubsRoiErt, axis=2)

    # Now of the form:
    # aryMneDpth[varNumSub, varNumCon, varNumVol]

    # Calculate mean across subjects:
    aryMneDpthSub = np.average(aryMneDpth, weights=vecNumVrtcs, axis=0)

    # Weighted variance (across subjects):
    aryVarDpthSub = np.average(
                               np.power(
                                        np.subtract(
                                                    aryMneDpth,
                                                    aryMneDpthSub[None, :, :]
                                                    ),
                                        2.0
                                        ),
                               axis=0,
                               weights=vecNumVrtcs
                               )

    # Weighted standard deviation (across subjects):
    arySdDpthSub = np.sqrt(aryVarDpthSub)

    # Now of the form:
    # aryMneDpthSub[varNumCon, varNumVol]
    # arySdDpthSub[varNumCon, varNumVol]

    # Title for plot:
    strTmpTtl = ''

    # Output filename:
    strTmpPth = (strPltOtPre + 'acr_dpth_acr_subs' + strPltOtSuf)

    # We create one plot per depth-level.
    ert_plt(aryMneDpthSub,
            arySdDpthSub,
            varNumDpth,
            varNumCon,
            varNumVol,
            varDpi,
            varAcrSubsYmin,
            varAcrSubsYmax,
            varStimStrt,
            varStimEnd,
            varTr,
            lstConLbl,
            lgcLgnd02,
            strXlabel,
            strYlabel,
            lgcCnvPrct,
            strTmpTtl,
            strTmpPth,
            varTmeScl=varTmeScl,
            varXlbl=varXlbl,
            varYnum=varYnum,
            tplPadY=tplPadY)
    # *************************************************************************
