# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

(C) Ingo Marquardt, 2018
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


def funcSlctVrtcs(varNumCon,           # Number of conditions
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
                  varThrSlct04,        # Criterion 4 - Threshold
                  idxPrc):             # Process ID
    """Function for selecting vertices. See ds_main.py for more information."""
    # **************************************************************************
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

        # Loop through conditions (corresponding to input files):
        for idxIn in range(0, varNumCon):

            # Get array for current condition:
            aryTmp = lstDpthData01[idxIn]

            # Get original number of vertices in vtk mesh:
            varOrigNumVtkVrtc = aryTmp.shape[0]

            # Selcet vertices that are included in the patch of interest:
            aryTmp = aryTmp[vecRoiIdx, :]

            # Put array back into list:
            lstDpthData01[idxIn] = aryTmp

        # Initialise inclusion vector:
        vecInc = np.ones((vecRoiIdx.shape[0]), dtype=bool)

        # Initialise number of included vertices:
        varNumInc = np.sum(vecInc)

        # Only print status messages if this is the first of several parallel
        # processes:
        if idxPrc == 0:
            print('------------Remaining vertices: ' + str(varNumInc))
    # **************************************************************************

    # **************************************************************************
    # *** (2) Selection criterion 2
    #         Vertices that are BELOW a certain threshold at any depth level
    #         are excluded.

    if lgcSlct02:

        if idxPrc == 0:
            print('---------Select vertices based criterion 2 (vertices below '
                  + 'threshold at any depth level will be excluded)')

        # Get minimum value across cortical depths:
        vecMinSlct02 = np.min(arySlct02, axis=1)

        # Check whether vertex values are above the exclusion threshold:
        vecSlct02 = np.greater(vecMinSlct02, varThrSlct02)

        # Extract values for ROI:
        vecSlct02 = vecSlct02[vecRoiIdx]

        # Apply second vertex selection criterion to inclusion-vector:
        vecInc = np.logical_and(vecInc,
                                vecSlct02)

        # Update number of included vertices:
        varNumInc = np.sum(vecInc)

        # Only print status messages if this is the first of several parallel
        # processes:
        if idxPrc == 0:
            print('------------Remaining vertices: ' + str(varNumInc))
    # **************************************************************************

    # **************************************************************************
    # *** (3) Selection criterion 3
    #         Vertices that are BELOW a certain threshold at any depth level
    #         are excluded.

    if lgcSlct03:

        if idxPrc == 0:
            print('---------Select vertices based criterion 3 (vertices below '
                  + 'threshold at any depth level will be excluded)')

        # Get minimum value across cortical depths:
        vecMinSlct03 = np.min(arySlct03, axis=1)

        # Check whether vertex values are above the exclusion threshold:
        vecSlct03 = np.greater(vecMinSlct03, varThrSlct03)

        # Extract values for ROI:
        vecSlct03 = vecSlct03[vecRoiIdx]

        # Apply second vertex selection criterion to inclusion-vector:
        vecInc = np.logical_and(vecInc,
                                vecSlct03)

        # Update number of included vertices:
        varNumInc = np.sum(vecInc)

        # Only print status messages if this is the first of several parallel
        # processes:
        if idxPrc == 0:
            print('------------Remaining vertices: ' + str(varNumInc))
    # **************************************************************************

    # **************************************************************************
    # *** (2) Selection criterion 4
    #         Vertices that are BELOW a certain threshold at any depth level
    #         are excluded.

    if lgcSlct04:

        if idxPrc == 0:
            print('---------Select vertices based criterion 4 (vertices below '
                  + 'threshold at any depth level will be excluded)')

        # Get minimum value across cortical depths:
        vecMinSlct04 = np.min(arySlct04, axis=1)

        # Check whether vertex values are above the exclusion threshold:
        vecSlct04 = np.greater(vecMinSlct04, varThrSlct04)

        # Extract values for ROI:
        vecSlct04 = vecSlct04[vecRoiIdx]

        # Apply second vertex selection criterion to inclusion-vector:
        vecInc = np.logical_and(vecInc,
                                vecSlct04)

        # Update number of included vertices:
        varNumInc = np.sum(vecInc)

        # Only print status messages if this is the first of several parallel
        # processes:
        if idxPrc == 0:
            print('------------Remaining vertices: ' + str(varNumInc))
    # **************************************************************************

    # **************************************************************************
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
    # **************************************************************************

# TODO

    # **************************************************************************
    # *** Update inclusion vector

    if idxPrc == 0:
        print('---------Total number of vertices in vtk mesh: '
              + str(varOrigNumVtkVrtc))
        print('---------Size of ROI patch: ' + str(vecInc.size) + ' vertices')
        print('---------Number of vertices after exlusion criteria 2, 3, & 4: '
              + str(np.sum(vecInc)))

    # We update the inclusion vector, i.e. we set those vertices that were
    # excluded when using criterion 5 and/or 6 to false. First we create two
    # arrays of the same size as the original inclusion vector:
    if lgcVtk02:
        vecIncCrit05 = np.zeros((vecInc.shape), dtype=bool)
    else:
        vecIncCrit05 = np.ones((vecInc.shape), dtype=bool)

    if lgcPeRng:
        vecIncCrit06 = np.zeros((vecInc.shape), dtype=bool)
    else:
        vecIncCrit06 = np.ones((vecInc.shape), dtype=bool)

    # We bring the result from criterion 5 into the shape of the full data
    # array (i.e. the array with all vertices, at the beginning of this
    # function):
    if lgcVtk02:
        vecIncCrit05[vecInc] = vecLgcZconAny

        if idxPrc == 0:
            print('---------Number of vertices surviving criterion 5: '
                  + str(np.sum(vecIncCrit05)))

    # The same for criterion 6:
    if lgcPeRng:
        if lgcVtk02:
            # If criterion 5 is selected as well, we first have to account for
            # the fact that the input to criterion 6 does not have the shape
            # of the original array (because vertices have been removed in
            # criterion 5):
            vecTmp = np.zeros((vecIncCrit06[vecInc].shape),
                              dtype=bool)
            vecTmp[vecLgcZconAny] = vecLgcPrctl
            vecIncCrit06[vecInc] = vecTmp
        else:
            vecIncCrit06[vecInc] = vecLgcPrctl

        if idxPrc == 0:
            print('---------Number of vertices surviving criterion 6: '
                  + str(np.sum(vecIncCrit06)))

    # Multiply the main inclusion vector with the vectors for criterion 5 & 6:
    vecInc = np.multiply(vecInc,
                         np.multiply(vecIncCrit05,
                                     vecIncCrit06)
                         )

    # Update number of vertices included:
    varNumInc = np.sum(vecInc)

    if idxPrc == 0:
        print('---------Final number of vertices: ' + str(varNumInc))
    # **************************************************************************

    # **************************************************************************
    # *** Return
    return lstDpthData01, varNumInc, varThrZcon, vecInc
    # **************************************************************************
