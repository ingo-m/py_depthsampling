# -*- coding: utf-8 -*-
"""Project parameter estimates into 1D visual space representation."""

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
from py_depthsampling.psf_1D.utilities import crt_gauss_1D


def project_ecc_par(idxPrc, vecData, vecX, vecY, vecSd, vecR2, varThrR2,
                    varNumEcc, varExtXmin, varExtXmax, varExtYmin, varExtYmax,
                    queOut):
    """
    Project vertex data into 1D eccentricity space, in parallel.

    Notes
    -----
    The list with results is not returned directly, but placed on a
    multiprocessing queue.

    """
    # Visual space array (1D array with eccentricity bins):
    vecVslSpc = np.zeros((varNumEcc))

    # Array for normalisation (parameter estimates are summed up over the
    # eccentricity vector; the array is needed to normalise the sum):
    vecNorm = np.zeros((varNumEcc))

    # Maximum eccentricity in visual field:
    varEccMax = np.sqrt(
                        np.add(
                               np.power(varExtXmax, 2.0),
                               np.power(varExtYmax, 2.0)
                               )
                        )

    # Vector with visual space coordinates of elements in `vecVslSpc`:
    vecCorEcc = np.linspace(0.0, varEccMax, num=varNumEcc, endpoint=True)

    # Calculate vertex eccentricities (only x- and y-positions are loaded from
    # disk):
    vecEcc = np.sqrt(
                     np.add(
                            np.power(vecX, 2.0),
                            np.power(vecY, 2.0)
                            )
                     )

    # Convert eccentricities from degree of visual angle into array indices:
    vecIdxEcc = (np.abs(vecCorEcc[None, :] - vecEcc[:, None])).argmin(axis=1)

    # The pRF size is converted from degree visual angle to relative size with
    # respect to the size of the array representing the visual space
    # (`vecVslSpc`).
    vecIdxSd = (vecSd
                / ((np.abs(varExtXmin) + varExtXmax)
                   + (np.abs(varExtYmin) + varExtYmax)) * 0.5
                * varNumEcc)

    # Exclude vertices with a pRF size of zero, and such with an R2 value below
    # threshold:
    lgcInc = np.multiply(np.greater(vecSd, 0.0),
                         np.greater(vecR2, varThrR2))

    # Only include left visual field:
    lgcInc = np.multiply(lgcInc,
                         np.less_equal(vecX, 0.0))

    # Number of vertices:
    varNumVrtc = vecData.shape[0]

    # Loop through vertices:
    for idxVrtx in range(varNumVrtc):

        # Only include vertex is SD is not zero, and R2 is above threshold:
        if lgcInc[idxVrtx]:

            # Create Gaussian at current pRF position:
            vecTmpGauss = crt_gauss_1D(varNumEcc,
                                       vecIdxEcc[idxVrtx],
                                       vecIdxSd[idxVrtx])

            # Scale Gaussian to have its maximum at one:
            # vecTmpGauss = np.divide(vecTmpGauss, np.max(vecTmpGauss))

            # Add non-scaled Gaussian to normalisation array:
            vecNorm = np.add(vecNorm, vecTmpGauss)

            # Multiply current data value (e.g. parameter estimate)
            # with Gaussian:
            vecTmpGauss = np.multiply(vecTmpGauss, vecData[idxVrtx])

            # Add current pRF sample to visual space map:
            vecVslSpc = np.add(vecVslSpc, vecTmpGauss)

    # Create list containing subject data, and the process ID:
    lstOut = [idxPrc, vecVslSpc, vecNorm]

    # Put output to queue:
    queOut.put(lstOut)
