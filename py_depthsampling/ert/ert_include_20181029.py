# -*- coding: utf-8 -*-

"""
Adjust event-related time courses from pilot session.

The event-related time courses from pilot session 20181029 differ from
following sessions in two important aspect: (1) the pilot session did not
include the 'Kanizsa rotated' control condition, (2) the volume TR in the
pilot session was slightly shorter (1947 ms instead of 2079 ms in subsequent
sessions). Here, the data from the pilot session are adjusted and integrated
into the pickle file containing data from subsequent sessions. The ERT pickle
file for the pilot session first needs to be created separately from that of
the following sessions.
"""

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


import numpy as np
import pickle
from scipy.interpolate import griddata


# -----------------------------------------------------------------------------
# ### Parameters

# Paths of pickle file with even-related timecourses for all subjects except
# pilot session. The pilot session will be appended to this file, and the file
# will be overwritten (ROI left open).
strPthPic01 = '/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/{}.pickle'  #noqa

# Path of pickle file with event-related time courses of pilot session (ROI
# left open).
strPthPic02 = '/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/{}_20181029.pickle'  #noqa

# List of ROIs:
lstRoi = ['centre/era_v1',
          'centre/era_v2',
          'centre/era_v3',
          'background/era_v1',
          'background/era_v2',
          'background/era_v3',
          'edge/era_v1',
          'edge/era_v2',
          'edge/era_v3']

# The order of conditions in the ERT files (within the respective numpy array)
# is supposed to be as follow:
#     - All subjects (expect pilot):
#       lstCon = ['bright_square', 'kanizsa', 'kanizsa_rotated']
#     - Pilot session:
#       lstCon = ['bright_square', 'kanizsa']

# Volume index of start of stimulus period (i.e. index of first volume during
# which stimulus was on):
varStimStrt = 5

# Volume TR:
varTr01 = 2.079

# Volume TR of pilot session:
varTr02 = 1.947

# Session ID of pilot session:
strSesId02 = '20181029'
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# ### Loop through ROIs

print('-Adjusting event-related time courses from pilot session.')

for idxRoi in lstRoi:

    # -------------------------------------------------------------------------
    # ### Preparations

    # Complete filepaths:
    strFle01 = strPthPic01.format(idxRoi)
    strFle02 = strPthPic02.format(idxRoi)

    print(('---ERT file, all subjects (but pilot session): ' + strFle01))
    print(('---ERT file, pilot session:                    ' + strFle02))

    # Load previously prepared event-related timecourses from pickle:
    dicRoiErt01 = pickle.load(open(strFle01, 'rb'))

    # Load previously prepared event-related timecourses from pickle, pilot
    # session:
    dicRoiErt02 = pickle.load(open(strFle02, 'rb'))

    # Event-related timecourses from pilot session:
    aryErt02 = dicRoiErt02[strSesId02][0]

    # Number of conditions in pilot session:
    varNumCon02 = aryErt02.shape[0]

    # Number of depth levels:
    varNumDpth = aryErt02.shape[1]

    # Number of timepoints:
    varNumVol = aryErt02.shape[2]

    # Number of vertices in pilot session:
    varNumVrtc = dicRoiErt02[strSesId02][1]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # ### Temporal interpolation

    # Interpolate data from pilot session to the sampling rate (TR) of
    # subsequent sessions.

    # Position of time points for all but pilot session:
    vecPos01 = np.linspace((-1.0 * varStimStrt * varTr01),
                           (float(varNumVol - varStimStrt) * varTr01),
                           num=varNumVol,
                           endpoint=False)

    # Position of time points for pilot session:
    vecPos02 = np.linspace((-1.0 * varStimStrt * varTr02),
                           (float(varNumVol - varStimStrt) * varTr02),
                           num=varNumVol,
                           endpoint=False)

    # Array for interpolated timepoints:
    aryErtInt02 = np.zeros(aryErt02.shape)

    # Loop through conditions:
    for idxCon in range(varNumCon02):

        # Loop through depth levels:
        for idxDpth in range(varNumDpth):

            # Interpolation:
            aryErtInt02[idxCon, idxDpth, :] = griddata(
                vecPos02, aryErt02[idxCon, idxDpth, :], vecPos01,
                method='cubic', fill_value=1.0)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # ### Match condition

    # Control condition 'Kanizsa rotated' was not included in the pilot
    # session. Fill the ERT array with the corresponding timecourse from the
    # same subject from a separate session (20181108). The missing condition is
    # assumed to be the third condition in the ERT array (see above for
    # details).

    # Session ID of matching session:
    strSesId01 = '20181108'

    # Get 'kanizsa_rotated' condition from matching session:
    aryMtch = dicRoiErt01[strSesId01][0][2, :, :]

    # Array for matched ERT (pilot session):
    aryErtMtch02 = np.zeros(((varNumCon02 + 1), varNumDpth, varNumVol))

    # Place interpolated ERT from pilot session:
    aryErtMtch02[0:2, :, :] = aryErtInt02

    # Place ERT from matching session:
    aryErtMtch02[2, :, :] = aryMtch
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # ### Save results

    # Place adjusted ERT in dictionary:
    dicRoiErt01[strSesId02] = [aryErtMtch02, varNumVrtc]

    # Save adjusted ERT array to pickle.
    pickle.dump(dicRoiErt01, open(strFle01, 'wb'))
    # -------------------------------------------------------------------------
