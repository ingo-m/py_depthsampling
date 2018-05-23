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


def ert_main(lstSubId, lstCon, lstConLbl, strMtaCn, strHmsph, strRoi,
             strVtkMsk, strVtkPth, varTr, varNumDpth, varNumVol, varStimStrt,
             varStimEnd, strPthPic, lgcPic, strPltOtPre, strPltOtSuf,
             varNumLne=2, strPrcdData='SCALARS', strXlabel='Time [s]',
             strYlabel='Percent signal change', varAcrSubsYmin=-0.06,
             varAcrSubsYmax=0.04, lgcCnvPrct=True, lgcLgnd01=True,
             lgcLgnd02=True, varTmeScl=1.0, varXlbl=5, varDpi=70.0):
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
    strHmsph : string
        Hemisphere ('rh' or 'lh').
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
        course data to (metacondition, ROI, and hemisphere left open).
    lgcPic : bool
        Load data from previously prepared pickle? If 'False', data is loaded
        from vtk meshes and saved as pickle.
    strPltOtPre : string
        Output path for plots - prefix, i.e. path and file name (metacondition,
        ROI, and hemisphere left open).
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

    # Complete strings:
    strPthPic = strPthPic.format(strMtaCn, strRoi, strHmsph)
    strPltOtPre = strPltOtPre.format(strMtaCn, strRoi, strHmsph)

    # Number of subjects:
    varNumSub = len(lstSubId)

    # Number of conditions:
    varNumCon = len(lstCon)

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

            # Complete file path of vertex inclusion mask for current subject:
            strVtkMskTmp = strVtkMsk.format(strSubID, strHmsph, strSubID,
                                            strRoi, strMtaCn)

            # Load data for current subject (returns array of the form:
            # aryRoiErt[varNumCon, varNumDpth, varNumVol]):
            dicAllSubsRoiErt[strSubID] = ert_get_sub_data(strSubID,
                                                          strHmsph,
                                                          strVtkMskTmp,
                                                          strVtkPth,
                                                          lstCon,
                                                          varNumVol,
                                                          varNumDpth,
                                                          strPrcdData,
                                                          varNumLne)

        # Save event-related timecourses to disk as pickle:
        pickle.dump(dicAllSubsRoiErt, open(strPthPic, 'wb'))

    # *************************************************************************
    # *** Subtract baseline mean

    # The input to this function are timecourses that have been normalised to
    # the pre-stimulus baseline. The datapoints are signal intensity relative
    # to the pre-stimulus baseline, and the pre-stimulus baseline has a mean of
    # one. We subtract one, so that the datapoints are percent signal change
    # relative to baseline.
    for strSubID, aryRoiErt in dicAllSubsRoiErt.items():
        aryRoiErt = np.subtract(aryRoiErt, 1.0)
        # Is this line necessary (hard copy)?
        dicAllSubsRoiErt[strSubID] = aryRoiErt

    # *************************************************************************
    # *** Plot single subjet results

    if True:

        print('---Ploting single-subjects event-related averages')

        # Loop through subjects:
        for strSubID, aryRoiErt in dicAllSubsRoiErt.items():

            # Loop through depth levels (we only create plots for three depth
            # levels):
            for idxDpth in [0, 5, 10]:

                # Title for plot:
                strTmpTtl = (strSubID + ' ERA, depth level ' + str(idxDpth))

                # Output filename:
                strTmpPth = (strPltOtPre + strSubID + '_dpth_' + str(idxDpth)
                             + strPltOtSuf)

                # We don't have the variances across trials (within subjects),
                # therefore we create an empty array as a placeholder. NOTE:
                # This should be replaced by between-trial variance once the
                # depth sampling is fully scriptable.
                aryDummy = np.zeros(aryRoiErt[:, idxDpth, :].shape)

                # We create one plot per depth-level.
                ert_plt(aryRoiErt[:, idxDpth, :],
                        aryDummy,
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
                        lgcLgnd01,
                        strXlabel,
                        strYlabel,
                        lgcCnvPrct,
                        strTmpTtl,
                        strTmpPth,
                        varTmeScl=varTmeScl,
                        varXlbl=varXlbl)

    # *************************************************************************
    # *** Plot across-subjects average

    print('---Ploting across-subjects average')

    # Create across-subjects data array of the form:
    # aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]
    aryAllSubsRoiErt = np.zeros((varNumSub, varNumCon, varNumDpth, varNumVol))
    idxSub = 0
    for aryRoiErt in dicAllSubsRoiErt.values():
        aryAllSubsRoiErt[idxSub, :, :, :] = aryRoiErt
        idxSub += 1

    # Calculate mean event-related time courses (mean across subjects):
    aryRoiErtMean = np.mean(aryAllSubsRoiErt, axis=0)

    # Calculate standard error of the mean (for error bar):
    aryRoiErtSem = np.divide(np.std(aryAllSubsRoiErt, axis=0),
                             np.sqrt(varNumSub))

    # Loop through depth levels:
    # for idxDpth in range(0, varNumDpth):
    for idxDpth in [0, 5, 10]:

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
                varXlbl=varXlbl)

    # *************************************************************************
    # *** Plot across-subjects average (mean across depth levels)

    print('---Ploting across-subjects average - mean across depth levels')

    # Event-related time courses have the form:
    # aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]

    # Calculate mean across depth:
    aryMneDpth = np.mean(aryAllSubsRoiErt, axis=2)

    # Now of the form:
    # aryMneTmp[varNumSub, varNumCon, varNumVol]

    # Calculate mean across subjects:
    aryMneDpthSub = np.mean(aryMneDpth, axis=0)

    # Calculate SD across subjects:
    arySdDpthSub = np.std(aryMneDpth, axis=0)

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
            varXlbl=varXlbl)
    # *************************************************************************
