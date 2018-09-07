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


def load_par(strSub, strCon, strRoi, strPthData, strPthMneEpi, strPthR2,
             strPthX, strPthY, strPthSd, strCsvRoi, varNumDpth, lstDpth,
             varTr, idxPrc, queOut):
    """
    Load single subject vtk meshes in parallel.

    Notes
    -----
    The list with results is not returned directly, but placed on a
    multiprocessing queue.
    """
    # Temporary input paths for left hemisphere:
    if '.npy' in strPthData:
        # Time series data:
        strPthLhData = strPthData.format(strSub, 'lh', strCon, strCon)
    else:
        # Statistical map (PEs):
        strPthLhData = strPthData.format(strSub, 'lh', strCon)

    strPthLhMneEpi = strPthMneEpi.format(strSub, 'lh')
    strPthLhR2 = strPthR2.format(strSub, 'lh')
    strPthLhSd = strPthSd.format(strSub, 'lh')
    strPthLhX = strPthX.format(strSub, 'lh')
    strPthLhY = strPthY.format(strSub, 'lh')
    strCsvLhRoi = strCsvRoi.format(strSub, 'lh', strRoi)

    # Load single subject data for left hemisphere:
    vecLhData, vecLhMneEpi, vecLhR2, vecLhSd, vecLhX, vecLhY = get_data(
        strPthLhData, strPthLhMneEpi, strPthLhR2, strPthLhSd, strPthLhX,
        strPthLhY, strCsvLhRoi, varNumDpth=varNumDpth, lstDpth=lstDpth,
        varTr=varTr)

    # Temporary input paths for right hemisphere:
    if '.npy' in strPthData:
        # Time series data:
        strPthRhData = strPthData.format(strSub, 'rh', strCon, strCon)
    else:
        # Statistical map (PEs):
        strPthRhData = strPthData.format(strSub, 'rh', strCon)

    strPthRhMneEpi = strPthMneEpi.format(strSub, 'rh')
    strPthRhR2 = strPthR2.format(strSub, 'rh')
    strPthRhSd = strPthSd.format(strSub, 'rh')
    strPthRhX = strPthX.format(strSub, 'rh')
    strPthRhY = strPthY.format(strSub, 'rh')
    strCsvRhRoi = strCsvRoi.format(strSub, 'rh', strRoi)

    # Load single subject data for right hemisphere:
    vecRhData, vecRhMneEpi, vecRhR2, vecRhSd, vecRhX, vecRhY = get_data(
        strPthRhData, strPthRhMneEpi, strPthRhR2, strPthRhSd, strPthRhX,
        strPthRhY, strCsvRhRoi, varNumDpth=varNumDpth, lstDpth=lstDpth,
        varTr=varTr)

    # Concatenate LH and RH data:
    vecData = np.concatenate([vecLhData, vecRhData])
    vecMneEpi = np.concatenate([vecLhMneEpi, vecRhMneEpi])
    vecR2 = np.concatenate([vecLhR2, vecRhR2])
    vecSd = np.concatenate([vecLhSd, vecRhSd])
    vecX = np.concatenate([vecLhX, vecRhX])
    vecY = np.concatenate([vecLhY, vecRhY])

    # Create list containing subject data, and the process ID:
    lstOut = [idxPrc, vecData, vecMneEpi, vecR2, vecSd, vecX, vecY]

    # Put output to queue:
    queOut.put(lstOut)
