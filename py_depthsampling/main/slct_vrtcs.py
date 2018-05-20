# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

(C) Ingo Marquardt, 2018
"""

# Part of py_depthsampling library
# Copyright (C) 2017  Ingo Marquardt
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


def slct_vrtcs(varNumCon,           # Number of conditions  #noqa
               lstDpthData01,       # List with depth-sampled data I
               lgcSlct01,           # Criterion 1 - Yes or no?
               aryRoiVrtx,          # Criterion 1 - Data (ROI)
               lgcSlct02,           # Criterion 2 - Yes or no?
               arySlct02,           # Criterion 2 - Data
               varThrSlct02,        # Criterion 2 - Threshold
               lgcSlct03,           # Criterion 3 - Yes or no?
               arySlct03,           # Criterion 3 - Data
               varThrSlct03,        # Criterion 3 - Threshold
               lgcSlct04,           # Criterion 4 - Yes or no?
               arySlct04,           # Criterion 4 - Data
               tplThrSlct04,        # Criterion 4 - Threshold
               idxPrc):             # Process ID
    """Select vertices. See ds_main.py for more information."""
    # *************************************************************************
    # Preparations

    # Original number of vertices in mesh:
    varOrigNumVtkVrtc = lstDpthData01[0].shape[0]

    # Only print status message if this is the first of several parallel
    # processes:
    if idxPrc == 0:
        print('---------Total number of vertices in vtk mesh: '
              + str(varOrigNumVtkVrtc))

    # Vertex inclusion vector. Contains '1' for each vertex that is included,
    # otherwise '0'.
    vecInc = np.ones(varOrigNumVtkVrtc, dtype=bool)

    # Initialise number of included vertices:
    varNumInc = np.sum(vecInc)
    # *************************************************************************

    # *************************************************************************
    # *** (1) Select vertices contained within the ROI

    # Apply first selection criterion (ROI)?
    if lgcSlct01:

        # Only print status messages if this is the first of several parallel
        # processes:
        if idxPrc == 0:
            print('---------Select vertices contained within the ROI')

        # The second column of the array "aryRoiVrtx" contains the indicies of
        # the vertices contained in the ROI. We extract that information:
        vecRoiIdx = aryRoiVrtx[:, 1].astype(np.int64)

        # If using the first criterion, re-initialise the inclusion vector and
        # set it to 'True' only for vertices contained within the ROI:
        vecInc = np.zeros(varOrigNumVtkVrtc, dtype=bool)
        vecInc[vecRoiIdx] = True

        # Update number of included vertices:
        varNumInc = np.sum(vecInc)

        # Only print status messages if this is the first of several parallel
        # processes:
        if idxPrc == 0:
            print('------------Remaining vertices: ' + str(varNumInc))
    # *************************************************************************

    # *************************************************************************
    # *** (2) Selection criterion 2
    #         Vertices that are BELOW a certain threshold are excluded - median
    #         across depth levels.

    if lgcSlct02:

        if idxPrc == 0:
            print('---------Select vertices based on criterion 2')

        # Get median value across cortical depths:
        vecMneSlct02 = np.median(arySlct02, axis=1)

        # Check whether vertex values are above the exclusion threshold:
        vecSlct02 = np.greater(vecMneSlct02, varThrSlct02)

        # Apply second vertex selection criterion to inclusion-vector:
        vecInc = np.logical_and(vecInc,
                                vecSlct02)

        # Update number of included vertices:
        varNumInc = np.sum(vecInc)

        # Only print status messages if this is the first of several parallel
        # processes:
        if idxPrc == 0:
            print('------------Remaining vertices: ' + str(varNumInc))
    # *************************************************************************

    # *************************************************************************
    # *** (3) Selection criterion 3
    #         Vertices that are BELOW a certain threshold are excluded -
    #         minimum across depth levels.

    if lgcSlct03:

        if idxPrc == 0:
            print('---------Select vertices based on criterion 3')

        # Get minimum value across cortical depths:
        vecMneSlct03 = np.min(arySlct03, axis=1)

        # Check whether vertex values are above the exclusion threshold:
        vecSlct03 = np.greater(vecMneSlct03, varThrSlct03)

        # Apply second vertex selection criterion to inclusion-vector:
        vecInc = np.logical_and(vecInc,
                                vecSlct03)

        # Update number of included vertices:
        varNumInc = np.sum(vecInc)

        # Only print status messages if this is the first of several parallel
        # processes:
        if idxPrc == 0:
            print('------------Remaining vertices: ' + str(varNumInc))
    # *************************************************************************

    # *************************************************************************
    # *** (2) Selection criterion 4
    #         Vertices that are WITHIN INTERVAL are included - median across
    #         depth levels).

    if lgcSlct04:

        if idxPrc == 0:
            print('---------Select vertices based on criterion 4')

        # Get median value across cortical depths:
        vecMneSlct04 = np.median(arySlct04, axis=1)

        # Check whether vertex values are within the interval (lower and upper
        # bound):
        vecSlct04lowbound = np.greater(vecMneSlct04, tplThrSlct04[0])
        vecSlct04upbound = np.less(vecMneSlct04, tplThrSlct04[1])
        vecSlct04 = np.logical_and(vecSlct04lowbound, vecSlct04upbound)

        # Apply second vertex selection criterion to inclusion-vector:
        vecInc = np.logical_and(vecInc,
                                vecSlct04)

        # Update number of included vertices:
        varNumInc = np.sum(vecInc)

        # Only print status messages if this is the first of several parallel
        # processes:
        if idxPrc == 0:
            print('------------Remaining vertices: ' + str(varNumInc))
    # *************************************************************************

    # *************************************************************************
    # *** Apply inclusion-vector to data

    if idxPrc == 0:
        print('---------Applying inclusion criteria to data.')

    # Loop through conditions (corresponding to input files):
    for idxIn in range(0, varNumCon):

        # Get array for current condition:
        aryTmp = lstDpthData01[idxIn]

        # Selcet vertices that survived all previous inclusion criteria:
        aryTmp = aryTmp[vecInc, :]

        # Put array back into list:
        lstDpthData01[idxIn] = aryTmp

    if idxPrc == 0:
        print('---------Final number of vertices: ' + str(varNumInc))
    # *************************************************************************

    # *************************************************************************
    # *** Return
    return lstDpthData01, varNumInc, vecInc
    # *************************************************************************
