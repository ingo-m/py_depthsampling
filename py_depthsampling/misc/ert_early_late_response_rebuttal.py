#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot cortical depth profiles separately for early & late response.

Analysis to address reviewer comment (2), second revision round, PacMan paper
eLife submission.

Reviewer comment:

> 2. [...] If the negative BOLD response is indeed a composite of a positive
> and negative BOLD response, it would be interesting to see how the laminar
> effects can perhaps decompose this composite. A laminar analysis conducted on
> the first and later parts of the response separately may be highly insightful
> here.

Note: Right hemisphere only.
"""


import pickle
import numpy as np
from py_depthsampling.diff.diff_sem import diff_sem
from py_depthsampling.drain_model.drain_model_main import drain_model


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of pickle files with event-related timecourses (ROI and stimulus
# conditionleft open):
pathPic = '/media/ssd_dropbox/Dropbox/University/PhD/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/era_{}_rh.pickle'

# Path for npz files with depth profiles (both for intermediate files to
# perform draining correction, and corrected depth profiles; file name left
# open):
pathNpz = '/media/ssd_dropbox/Dropbox/University/PhD/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/{}.npz'

# List of ROIs:
lstRois = ['v1', 'v2', 'v3']

# For the PacMan project, the event related pickled timecourses should contain
# 19 time points. The index of the first volume during which the stimulus was
# on is 5, and the last volume during which the stimulus was on is 10. Define
# for which time windows to construct new depth profiles (list of tuples with
# volume indices):
lstTmeWins = [(5, 6, 7),
              (8, 9, 10)]

# List of conditions (has to match whichever order of conditions was used to
# create pickles with event related timecourses, see `ert.py`):
lstCon = ['Pd', 'Ps', 'Cd']


# -----------------------------------------------------------------------------
# *** Create npz files with depth profiles

# Create npz files with depth profiles based on event-related timecourses, to
# be used for draining effect removal.

# Loop through ROIs
for strRoi in lstRois:

    # Load pickle with event-related timecourses of current ROI:
    dicAllSubsRoiErt = pickle.load(open(pathPic.format(strRoi), 'rb'))

    # Number of subjects:
    varNumSubs = len(dicAllSubsRoiErt)

    # Number of conditions:
    varNumCon = list(dicAllSubsRoiErt.values())[0][0].shape[0]

    # Number of depth levels:
    varNumDpth = list(dicAllSubsRoiErt.values())[0][0].shape[1]

    # Loop through time windows and construct arrays for drain model.
    for idxTmeWin, tplTmeWin in enumerate(lstTmeWins):

        # To apply the drain model, we need to save depth profiles to numpy
        # arrays with shape aryDpth[subject, condition, depth].
        aryDpth = np.zeros((varNumSubs, varNumCon, varNumDpth))

        # Array for number of vertices per subject:
        vecNumInc = np.zeros(varNumSubs)

        # Loop through subjects:
        for idxSub, strSub in enumerate(list(dicAllSubsRoiErt.keys())):

            # Take mean over time for volumes in current time window, for
            # current subject:
            aryDpth[idxSub, :, :] = np.mean(dicAllSubsRoiErt[strSub][0]
                                            [:, :, tplTmeWin],
                                            axis=2)

            # Number of vertices for current subject:
            vecNumInc[idxSub] = dicAllSubsRoiErt[strSub][1]

        # Loop through conditions, to save separate npz file for each condition
        # (for compatibility with subsequent functions):
        for idxCon, strCon in enumerate(lstCon):

            # Path for npz file:
            pathNpzTmp = pathNpz.format((strRoi
                                         + '_rh_'
                                         + strCon
                                         + '_dpth_prfls_time_win_'
                                         + str(idxTmeWin)))

            # Save npz file with single-condition depth profiles of current
            # time window to disk:
            np.savez(pathNpzTmp,
                     aryDpth=aryDpth[:, idxCon, :],
                     vecNumInc=vecNumInc)

# -----------------------------------------------------------------------------
# *** Apply draining effect model

# Loop through ROIs
for strRoi in lstRois:

    # Loop through time windows and construct arrays for drain model.
    for idxTmeWin, tplTmeWin in enumerate(lstTmeWins):

         # Path of npz file with un-corrected depth profiles:
        pathNpzTmp = pathNpz.format((strRoi
                                     + '_dpth_prfls_time_win_'
                                     + str(idxTmeWin)))

        # Call drain model function:
        drain_model(1,  # Which draining model to use
                    strRoi,
                    '',  # Deprecated
                    strPthPrf.format(lstMetaCon[idxMtaCn],
                                     lstRoi[idxRoi],
                                     lstHmsph[idxHmsph],
                                     '{}'),
                    strPthPrfOt.format(lstMetaCon[idxMtaCn],
                                       lstRoi[idxRoi],
                                       lstHmsph[idxHmsph],
                                       {},
                                       lstMdl[idxMdl]),
                    strPthPltOt.format(lstMetaCon[idxMtaCn],
                                       lstRoi[idxRoi],
                                       lstRoi[idxRoi],
                                       lstHmsph[idxHmsph],
                                       lstNstCon[idxCon][0],
                                       str(lstMdl[idxMdl])),
                    strFlTp,
                    varDpi,
                    strXlabel,
                    strYlabel,
                    lstNstCon[idxCon],
                    lstNstConLbl[idxCon],
                    varNumIt,
                    varCnfLw,
                    varCnfUp,
                    varNseRndSd,
                    varNseSys,
                    lstFctr,
                    varAcrSubsYmin01,
                    varAcrSubsYmax01,
                    varAcrSubsYmin02,
                    varAcrSubsYmax02)

