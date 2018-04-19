# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

@author: Ingo Marquardt, 26.10.2016
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

import csv
import numpy as np


def funcLoadCsvRoi(strCsvRoi,
                   varNumHdrRoi):
    """
    Function for loading ROI definition from csv file.

    The csv file to be loaded can be created with paraview, by selecting
    vertices on a vtk mesh (e.g. a retinotopic region on the inflated cortex).
    Here, we only extract the indicies of the vertices included in the ROI.
    Therefore, it is important that the ROI is defined on the same vtk mesh
    that is used for further analysis (because only then do the vertex indicies
    in the csv file correspond to those in the vtk meshes).
    """
    # print('---------Importing ROI csv file.')

    # Open file with ROI information:
    fleCsvRoi = open(strCsvRoi, 'r')

    # Read file  with ROI information:
    csvIn = csv.reader(fleCsvRoi,
                       delimiter='\n',
                       skipinitialspace=True)

    # Create empty list for CSV data:
    lstCsvRoi = []

    # Loop through csv object to fill list with csv data:
    for lstTmp in csvIn:
        for strTmp in lstTmp:
            lstCsvRoi.append(strTmp[:])

    # Close file:
    fleCsvRoi.close()

    # Next we would like to access the numeric data in the list into which we
    # placed the vtk data. We skip the header and convert the data to numeric
    # values.

    # Number of verticies in the list (i.e. number of elements in the list
    # minus header):
    varNumRoiVrtx = len(lstCsvRoi) - varNumHdrRoi

    # Create empty array for ROI vertex data. The second column will contain
    # the IDs of the vertices contained in the ROI.
    aryRoiVrtx = np.zeros((varNumRoiVrtx, 5))

    # Loop through list and access numeric vertex data:
    for idxVrtx in range(0, varNumRoiVrtx):
        # When indexing the list, we add one because we skip the header:
        aryRoiVrtx[idxVrtx, :] = lstCsvRoi[(idxVrtx + 1)].split(',')

    # Delete original list:
    del(lstCsvRoi)

    # Return the vertex array:
    return aryRoiVrtx
