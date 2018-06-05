# -*- coding: utf-8 -*-
"""Project parameter estimates into a visual space representation."""

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
from py_depthsampling.project.utilities import get_data


def load_par(strSub, strCon, strRoi, strPthData, strPthR2, strPthX, strPthY,
             strPthSd, strCsvRoi, varNumDpth, idxPrc, queOut):
    """
    Load single subject vtk meshes in parallel.

    Notes
    -----
    The list with results is not returned directly, but placed on a
    multiprocessing queue.
    """
    # Temporary input paths for left hemisphere:
    strPthLhData = strPthData.format(strSub, 'lh', strCon)
    strPthLhR2 = strPthR2.format(strSub, 'lh')
    strPthLhSd = strPthSd.format(strSub, 'lh')
    strPthLhX = strPthX.format(strSub, 'lh')
    strPthLhY = strPthY.format(strSub, 'lh')
    strCsvLhRoi = strCsvRoi.format(strSub, 'lh', strRoi)

    # Load single subject data for left hemisphere:
    vecLhData, vecLhR2, vecLhSd, vecLhX, vecLhY = get_data(
        strPthLhData, strPthLhR2, strPthLhSd, strPthLhX, strPthLhY,
        strCsvLhRoi, varNumDpth=varNumDpth)

    # Temporary input paths for right hemisphere:
    strPthRhData = strPthData.format(strSub, 'rh', strCon)
    strPthRhR2 = strPthR2.format(strSub, 'rh')
    strPthRhSd = strPthSd.format(strSub, 'rh')
    strPthRhX = strPthX.format(strSub, 'rh')
    strPthRhY = strPthY.format(strSub, 'rh')
    strCsvRhRoi = strCsvRoi.format(strSub, 'rh', strRoi)

    # Load single subject data for right hemisphere:
    vecRhData, vecRhR2, vecRhSd, vecRhX, vecRhY = get_data(
        strPthRhData, strPthRhR2, strPthRhSd, strPthRhX, strPthRhY,
        strCsvRhRoi, varNumDpth=varNumDpth)

    # Concatenate LH and RH data:
    vecData = np.concatenate([vecLhData, vecRhData])
    vecR2 = np.concatenate([vecLhR2, vecRhR2])
    vecSd = np.concatenate([vecLhSd, vecRhSd])
    vecX = np.concatenate([vecLhX, vecRhX])
    vecY = np.concatenate([vecLhY, vecRhY])

    # Create list containing subject data, and the process ID:
    lstOut = [idxPrc, vecData, vecR2, vecSd, vecX, vecY]

    # Put output to queue:
    queOut.put(lstOut)
