# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

Function of the event-related timecourses depth sampling sub-pipeline.
"""

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

import os
import numpy as np
from py_depthsampling.get_data.load_vtk_single import load_vtk_single
from py_depthsampling.get_data.load_vtk_multi import load_vtk_multi


def ert_get_sub_data(strSubId,
                     strHmsph,
                     strVtkMsk,
                     strVtkPth,
                     lstCon,
                     varNumVol,
                     varNumDpth,
                     strPrcdData,
                     varNumLne):
    """
    Load data for event-related timecourse analysis.

    Load vtk meshes for the event-related average timecourses analysis, for one
    subject. Each vtk mesh is one 3D volume. This script loads all meshes for
    all conditions for one subject.
    """
    # *************************************************************************
    # *** Load vtk mask (ROI)

    # Load vtk mask (with ROI definition - the event-related time course for
    # each depth-level is averaged across this ROI):
    vecVtkMsk = load_vtk_single(strVtkMsk,
                                strPrcdData,
                                varNumLne)
    # *************************************************************************

    # *************************************************************************
    # *** Load 3D vtk meshes

    # Number of conditions:
    varNumCon = len(lstCon)

    # Number of vertices:
    varNumVrtc = vecVtkMsk.shape[0]

    # Array to be filled with data:
    aryErt = np.zeros((varNumCon, varNumDpth, varNumVol, varNumVrtc),
                      dtype=np.float16)

    # Loading time courses from single vtk files is very slow. The first time
    # the time courses are accessed, we therefore save the data to disk in form
    # of a *.npy file, and delete the vtk files. Here, we first check whether
    # the vtk files are available. If yes, we load them and create a new *.npy
    # file. In this way, we do not risk loading an outdated *.npy file (because
    # we only load an *.npy file if no vtk (new) files are available).

    # Path of vtk file of first volume:
    strVtkPthTmp = strVtkPth.format(strSubId,
                                    strHmsph,
                                    lstCon[0],
                                    str(0).zfill(3))

    # Check whether first vtk file exists:
    if os.path.isfile(strVtkPthTmp):

        # Loop through conditions:
        for idxCon in range(0, varNumCon):

            # Loop through volumes:
            for idxVol in range(0, varNumVol):

                # Complete file path of current volume:
                strVtkPthTmp = strVtkPth.format(strSubId,
                                                strHmsph,
                                                lstCon[idxCon],
                                                str(idxVol).zfill(3))

                # Load vtk mesh for current timepoint:
                aryTmp = load_vtk_multi(strVtkPthTmp,
                                        strPrcdData,
                                        varNumLne,
                                        varNumDpth).astype(np.float16)

                aryErt[idxCon, :, idxVol, :] = aryTmp.T

                # Delete vtk file:
                os.remove(strVtkPthTmp)

            # Save array of current condition to disk (in case data need to be
            # accessed again (conditions are saved separately because
            # composition of conditions can change for plot):
            strPthNpy = os.path.join(os.path.split(strVtkPthTmp)[0],
                                     ('aryErt_' + lstCon[idxCon] + '.npy'))
            np.save(strPthNpy, aryErt[idxCon, :, :, :])

    # Vtk file does not exist, attempt loading (previously created) *.npy file:
    else:

        # Loop through conditions:
        for idxCon in range(0, varNumCon):

            # Vtk file path of current condition:
            strVtkPthTmp = strVtkPth.format(strSubId,
                                            strHmsph,
                                            lstCon[idxCon],
                                            str(0).zfill(3))

            # Path of *.npy file to load (file name is hard-coded, not optimal
            # but still serves the purpose):
            strPthNpy = os.path.join(os.path.split(strVtkPthTmp)[0],
                                     ('aryErt_' + lstCon[idxCon] + '.npy'))

            aryErt[idxCon, :, :, :] = np.load(strPthNpy).astype(np.float16)
    # *************************************************************************

    # *************************************************************************
    # Extract ROI timecourses

    # Note that the ROI time courses extraction is done after saving the data
    # to disk in *.npy format. This is on purpose; the file size would be much
    # smaller otherwise, but changing the ROI would require to load vtk data
    # again, which is slower than loading *.npy and would defeat the purpose of
    # creating the *.npy file in the first place.

    # Get indicies of vertices with value greater than threshold. (The vtk mask
    # is supposed to contain ones for vertices that are included, and zeros
    # elsewhere).
    vecInc = np.greater_equal(vecVtkMsk, 0.5)

    print('---------Subject: ' + strSubId + ' --- Number of vertices in ROI: '
          + str(np.sum(vecInc)))
    print('------------Based on vtk mask: ' + strVtkMsk)

    # Apply selection to timecourses:
    aryErt = aryErt[:, :, :, vecInc]

    # Get number of vertices (for weighted across-subjects averaging):
    varNumVrtc = aryErt.shape[3]

    # Average across vertices:
    aryErt = np.mean(aryErt, axis=3)
    # *************************************************************************

    # *************************************************************************
    # *** Return

    return [aryErt, varNumVrtc]
    # *************************************************************************
