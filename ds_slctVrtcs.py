# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

@author: Ingo Marquardt, 04.11.2016
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

def funcSlctVrtcs(varNumCon,      # Number of conditions  #noqa
                  lstDpthData01,  # List with depth-sampled data I
                  aryRoiVrtx,     # Array with ROI definition (1st crit.)
                  lgcSlct02,      # Criterion 2 - Yes or no?
                  vecSlct02,      # Criterion 2 - Data
                  varThrSlct02,   # Criterion 2 - Threshold
                  lgcSlct03,      # Criterion 3 - Yes or no?
                  arySlct03,      # Criterion 3 - Data
                  varThrSlct03,   # Criterion 3 - Threshold
                  lgcSlct04,      # Criterion 4 - Yes or no?
                  arySlct04,      # Criterion 4 - Data
                  varThrSlct04,     # Criterion 4 - Threshold
                  lgcVtk02,       # Criterion 5 - Yes or no?
                  lstDpthData02,  # Criterion 5 - VTK path
                  varNumVrtx,     # Criterion 5 - Num vrtx to include
                  lgcPeRng,       # Criterion 6 - Yes or no?
                  varPeRngLw,     # Criterion 6 - Lower bound
                  varPeRngUp,     # Criterion 6 - Upper bound
                  idxPrc,         # Process ID (to manage status messages)
                  ):
    """Function for selecting vertices. See ds_main.py for more information."""
    # **************************************************************************
    # *** (1) Select vertices contained within the ROI

    # Only print status messages if this is the first of several parallel
    # processes:
    if idxPrc == 0:
        print('---------Select vertices contained within the ROI')

    # The second column of the array "aryRoiVrtx" contains the indicies of the
    # vertices contained in the ROI. We extract that information:
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

    # Apply ROI selection to second set of vtk data:
    if lgcVtk02:

            # Loop through conditions (corresponding to input files):
            for idxIn in range(0, varNumCon):

                # Get array for current condition:
                aryTmp = lstDpthData02[idxIn]

                # Selcet vertices that are included in the patch of interest:
                aryTmp = aryTmp[vecRoiIdx, :]

                # Put array back into list:
                lstDpthData02[idxIn] = aryTmp

    # Initialise dummy inclusion vector:
    vecInc = np.ones((vecRoiIdx.shape[0]), dtype=bool)

    # Initialise number of included vertices:
    varNumInc = np.sum(vecInc)

    # Only print status messages if this is the first of several parallel
    # processes:
    if idxPrc == 0:
        print('------------Remaining vertices: ' + str(varNumInc))
    # **************************************************************************

    # **************************************************************************
    # *** (2) Select vertices based on intensity criterion

    if lgcSlct02:

        if idxPrc == 0:
            print('---------Select vertices based on intensity criteria')

        # Previously, we selected those vertices in the data array that are
        # contained within the ROI. In order to apply the intensity-criterion,
        # we have to extract the data corresponding to the ROI from the
        # intensity array (e.g. extract pRF-overlap-ratio data for the ROI).

        # Selcet vertices from patch-selection arrays:
        vecSlct02 = vecSlct02[vecRoiIdx]

        # Get indicies of vertices with value greater than the intensity
        # criterion:
        vecInc = np.greater_equal(vecSlct02, varThrSlct02)

        # Update number of included vertices:
        varNumInc = np.sum(vecInc)

        # Only print status messages if this is the first of several parallel
        # processes:
        if idxPrc == 0:
            print('------------Remaining vertices: ' + str(varNumInc))
    # **************************************************************************

    # **************************************************************************
    # *** (3) Multi-depth level criterion I
    #         Vertices that are BELOW a certain threshold at any depth level
    #         are excluded.

    if lgcSlct03:

        if idxPrc == 0:
            print('---------Select vertices based on multi-depth level ' +
                  'criterion I')

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
    # *** (4) Multi-depth level criterion II
    #         vertices BELOW threshold at any depth level are excluded.

    if lgcSlct04:

        if idxPrc == 0:
            print('---------Select vertices based on multi-depth level ' +
                  'criterion II')

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

    # **************************************************************************
    # *** (5) Multi-level data distribution criterion I
    #         Selection based on combination of z-conjunction-mask mask and
    #         distribution of z-values.

    if lgcVtk02:

        if idxPrc == 0:
            print('---------Vertex selection based on combintation of '
                  + 'z-conjunction mask and distribution of z-values.')

        # Calculate z-conjunction mask. First, we check for each vertex whether
        # its data value (e.g. z-score) exceeds a certain value in ALL input
        # files (i.e. in all conditions). For instance, one could exclude
        # vertices that do not have a z-score of at least 1.5 in all stimulus
        # conditions. If this condition is fulfilled at least at one depth
        # level, the entire 'column' is included.

        # The second set of depth-data (i.e. the z-scores) need to be put into
        # the same shape as the first set of depth-data (i.e. the parameter
        # estimates). In other words, the same exclusion criteria have to be
        # applied to both data sets:

        # Loop through conditions (corresponding to input files):
        for idxIn in range(0, varNumCon):

            # Get array for current condition:
            aryTmp = lstDpthData02[idxIn]

            # Selcet vertices that survived all previous inclusion criteria:
            aryTmp = aryTmp[vecInc, :]

            # Put array back into list:
            lstDpthData02[idxIn] = aryTmp

        # Get minimum data value (e.g. z score) across conditions

        # Array for entire depth data, of the form
        # aryDpthData02[condition, vertex, depthlevel]
        aryDpthData02 = np.zeros((varNumCon,
                                  lstDpthData02[0].shape[0],
                                  lstDpthData02[0].shape[1]))

        # Loop through conditions (corresponding to input files):
        for idxIn in range(0, varNumCon):

            # Get array for current condition:
            aryDpthData02[idxIn, :, :] = lstDpthData02[idxIn]

        if idxPrc == 0:
            print('------------Attempting to select ' + str(varNumVrtx)
                  + ' vertices')

        # Initial value for number of included vertices:
        varNumIncTmp = varNumInc

        # Initial value for z-conjunction threshold:
        varThrZcon = 0.0

        # If we attempt to select more vertices than have survived all
        # inclusion criteria until now, the following while-loop will be
        # skipped. We create a dummy-version of the vector that would have
        # been created in the while-loop (all 'columns' that have survived
        # until now are also included after this criterion has been applied).
        if not(np.greater(varNumIncTmp, varNumVrtx)):
            vecLgcZconAny = np.ones(np.sum(vecInc), dtype=bool)

        # We increase the z conjunction threshold until we have reached the
        # number of vertices we would like to inlcude:
        while np.greater(varNumIncTmp, varNumVrtx):

            # Increase z-conjunction-threshold for next interation of the
            # while-loop:
            varThrZcon = varThrZcon + 0.001

            # Create boolean array for z-conjunction criterion:
            aryLgcZcon = np.ones((lstDpthData02[0].shape[0],
                                  lstDpthData02[0].shape[1]),
                                 dtype=bool)

            # List for logical arrays (for testing whether the vertex exceeds
            # the threshold in all conditions):
            lstLgcZcon = [None] * varNumCon

            # Test whether vertex exceeds threshold (separately for each
            # condition):
            for idxIn in range(0, varNumCon):
                lstLgcZcon[idxIn] = np.greater_equal(aryDpthData02[idxIn],
                                                     varThrZcon)

            # Create conjunction by multipying all logical test results:
            for idxIn in range(0, varNumCon):
                aryLgcZcon = np.multiply(lstLgcZcon[idxIn],
                                         aryLgcZcon)

            # At this point, aryLgcZcon holds the information whether each
            # vertex exceeds the z-threshold in all conditions, separately for
            # each depth level. Next, we test for each 'column' whether this
            # z-conjunction condition is fulfilled at least at one depth level
            # (we take the maximum across depth level):
            vecLgcZconAny = np.max(aryLgcZcon, axis=1)

            # Update number of included vertices:
            varNumIncTmp = np.sum(vecLgcZconAny)

        if idxPrc == 0:
            print('------------Selected ' + str(varNumIncTmp) + ' vertices, '
                  + 'based on a z-conjunction-threshold of ' + str(varThrZcon))

        # Apply z-conjunction criterion on first set of depth-data (i.e.
        # parameter estimates):

        # Loop through conditions (corresponding to input files):
        for idxIn in range(0, varNumCon):

            # Get array for current condition:
            aryTmp = lstDpthData01[idxIn]

            # Put array back into list:
            lstDpthData01[idxIn] = aryTmp[vecLgcZconAny, :]
    # **************************************************************************

    # **************************************************************************
    # *** (6) Multi-level data distribution criterion II
    #         Excludes vertices whose across-depth-maximum-value is at the
    #         lower and/or upper end of the distribution across vertices.
    #         (E.g. percentile of parameter estimates.)

    if lgcPeRng:

        if idxPrc == 0:
            print('---------Select vertices based on multi-depth data ' +
                  'distribution criterion (e.g. percentile of parameter ' +
                  'estimates)')

        # Get maximum PE value across depth levels

        # Loop through conditions (corresponding to input files):
        for idxIn in range(0, varNumCon):

            # Get array for current condition:
            aryTmp = lstDpthData01[idxIn]

            # On first iteration of the loop, create array for maxium PE values
            # across depth:
            if idxIn == 0:
                vecPeMax = np.zeros(aryTmp.shape[0])

            # Maximum PE across depths:
            vecPeMaxTmp = np.max(aryTmp, axis=1)

            # Check whether values in current PE file are greater than maximum
            # so far:
            vecLgcTmp = np.greater(vecPeMaxTmp, vecPeMax)

            # Update maximum PE vector:
            vecPeMax[vecLgcTmp] = vecPeMaxTmp[vecLgcTmp]

        # Lower percentile & indicies of vertices above:
        varPrctlLw = np.percentile(vecPeMax, varPeRngLw)
        vecLgcLw = np.greater_equal(vecPeMax, varPrctlLw)

        # Upper percentile:
        varPrctlUp = np.percentile(vecPeMax, varPeRngUp)
        vecLgcUp = np.less_equal(vecPeMax, varPrctlUp)

        # Conjunction of the two criteria:
        vecLgcPrctl = np.logical_and(vecLgcLw, vecLgcUp)

        # Select vertices that fulfill the PE range criterion

        # Number of vertices before exclusion:
        varTmp01 = aryTmp.shape[0]

        # Loop through conditions (corresponding to input files):
        for idxIn in range(0, varNumCon):

            # Get array for current condition:
            aryTmp = lstDpthData01[idxIn]

            # Put array back into list:
            lstDpthData01[idxIn] = aryTmp[vecLgcPrctl, :]

        # Number of vertices after exclusion:
        varTmp02 = lstDpthData01[0].shape[0]

        # Number of excluded varticies:
        varTmp03 = np.subtract(varTmp01, varTmp02)

        # Ratio of vertices excluded:
        varTmp04 = np.around(np.multiply(np.divide(float(varTmp03),
                                                   float(varTmp01)),
                                         100.0),
                             decimals=1)

        if idxPrc == 0:
            print(('------------Excluded ' + str(varTmp03) + ' vertices out '
                   + 'of ' + str(varTmp01) + ', i.e. ' + str(varTmp04) + '%'))
    # **************************************************************************

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
