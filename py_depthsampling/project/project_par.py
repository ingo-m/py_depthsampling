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
from py_depthsampling.project.utilities import crt_gauss


def project_par(idxPrc, vecData, vecX, vecY, vecSd, vecR2, varThrR2, varNumX,
                varNumY, varExtXmin, varExtXmax, varExtYmin, varExtYmax,
                queOut):
    """
    Project vertex data into visual space, in parallel.

    Notes
    -----
    The list with results is not returned directly, but placed on a
    multiprocessing queue.
    """
    # Visual space array (2D array with bins of locations in visual
    # space):
    aryVslSpc = np.zeros((varNumX, varNumY))

    # Vector with visual space coordinates of elements in `aryVslSpc`:
    vecCorX = np.linspace(varExtXmin, varExtXmax, num=varNumX,
                          endpoint=True)
    vecCorY = np.linspace(varExtYmin, varExtYmax, num=varNumY,
                          endpoint=True)

    # Convert pRF parameters (position and size) from degree
    # of visual angle into array indices:
    vecIdxX = (np.abs(vecCorX[None, :] - vecX[:, None])).argmin(axis=1)
    vecIdxY = (np.abs(vecCorY[None, :] - vecY[:, None])).argmin(axis=1)

    # The pRF size is converted from degree visual angle to
    # relative size with respect to the size of the array
    # representing the visual space (`aryVslSpc`).
    vecIdxSd = (vecSd
                / ((np.abs(varExtXmin) + varExtXmax)
                   + (np.abs(varExtYmin) + varExtYmax)) * 0.5
                * ((varNumX + varNumY) * 0.5))

    # Exclude vertices with a pRF size of zero, and such with an R2
    # value below threshold:
    lgcInc = np.multiply(np.greater(vecSd, 0.0),
                         np.greater(vecR2, varThrR2))

    # Number of vertices:
    varNumVrtc = vecData.shape[0]

    # Loop through vertices:
    for idxVrtx in range(varNumVrtc):

        # Only include vertex is SD is not zero, and R2 is above threshold:
        if lgcInc[idxVrtx]:

            # Create Gaussian at current pRF position:
            aryTmpGauss = crt_gauss(varNumX,
                                    varNumY,
                                    vecIdxX[idxVrtx],
                                    vecIdxY[idxVrtx],
                                    vecIdxSd[idxVrtx])

            # Scale Gaussian to have its maximum at one:
            aryTmpGauss = np.divide(aryTmpGauss, np.max(aryTmpGauss))

            # Multiply current data value (e.g. parameter estimate)
            # with Gaussian:
            aryTmpGauss = np.multiply(aryTmpGauss, vecData[idxVrtx])

            # Add current pRF sample to visual space map:
            aryVslSpc = np.add(aryVslSpc, aryTmpGauss)

    # Create list containing subject data, and the process ID:
    lstOut = [idxPrc, aryVslSpc]

    # Put output to queue:
    queOut.put(lstOut)
