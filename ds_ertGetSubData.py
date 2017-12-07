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

import numpy as np
from ds_loadVtkSingle import funcLoadVtkSingle
from ds_loadVtkMulti import funcLoadVtkMulti


def funcGetSubData(strSubId,
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

    The purpose of this function is to load vtk meshes for the event-related
    average timecourses analysis, for one subject. Each vtk mesh is one 3D
    volume. This script loads all meshes for all conditions for one subject.
    """
    # *************************************************************************
    # *** Load vtk mask (ROI)

    # Load vtk mask (with ROI definition - the event-related time course for
    # each depth-level is averaged across this ROI):
    vecVtkMsk = funcLoadVtkSingle(strVtkMsk,
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
                      dtype=np.float32)

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
            aryTmp = funcLoadVtkMulti(strVtkPthTmp,
                                      strPrcdData,
                                      varNumLne,
                                      varNumDpth).astype(np.float32)

            aryErt[idxCon, :, idxVol, :] = aryTmp.T
    # *************************************************************************

    # *************************************************************************
    # Extract ROI timecourses

    # Get indicies of vertices with value greater than threshold. (The vtk mask
    # is supposed to contain ones for vertices that are included, and zeros
    # elsewhere).
    vecInc = np.greater_equal(vecVtkMsk, 0.5)

    print('---------Subject: ' + strSubId + ' --- Number of vertices in ROI: '
          + str(np.sum(vecInc)))
    print('------------Based on vtk mask: ' + strVtkMsk)

    # Apply selection to timecourses:
    aryErt = aryErt[:, :, :, vecInc]

    # Average across vertices:
    aryErt = np.mean(aryErt, axis=3)
    # *************************************************************************

    # *************************************************************************
    # *** Return

    return aryErt
    # *************************************************************************
