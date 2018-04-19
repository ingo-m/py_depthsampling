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


def funcLoadVtkMulti(strVtkIn,
                     strPrcdData,
                     varNumLne,
                     varNumDpth):
    """
    Function for loading vtk file with multiple data points per vertex.

    The vtk file to be loaded is supposed to be a cortex mesh with multiple
    values per vertex, e.g. statistical parameters at several cortical depth
    levels.
    """
    # print('---------Importing vtk file with multiple values per vertex: '
    #       + strVtkIn)

    # Open file:
    fleVtkIn = open(strVtkIn, 'r')

    # Read file:
    csvIn = csv.reader(fleVtkIn,
                       delimiter='\n',
                       skipinitialspace=True)

    # Create empty list for vertex data:
    lstVtkData = []

    # Loop through csv object to fill list with csv data:
    for lstTmp in csvIn:
        for strTmp in lstTmp:
            lstVtkData.append(strTmp[:])

    # Close file:
    fleVtkIn.close()

    # Get index of string (as specified above) which precedes the vertex data:
    for idxSrch in range(0, len(lstVtkData)):
        if lstVtkData[idxSrch].startswith((strPrcdData)):
            varIdxTmp = idxSrch

    # Get number of vertices from the line preceding the specified string:
    strNumDataVrtx = lstVtkData[(varIdxTmp - 1)]

    # The number of vertecies is preceded by the string 'POINT_DATA' in vtk
    # files. We extract the number behind the string:
    varNumDataVrtx = int(strNumDataVrtx[11:])

    # Index of first vertex data point:
    varIdxFrst = varIdxTmp + varNumLne

    # Array for numeric vertex data:
    aryVtkData = np.zeros((varNumDataVrtx, varNumDpth))

    # Get numeric vertex data:
    for idxData in range(0, varNumDataVrtx):

        # Line of current vertex in csv list:
        varTmpStrt = varIdxFrst + idxData

        # Put vertex data into array:
        aryVtkData[idxData, :] = \
            lstVtkData[varTmpStrt].split(' ')[0:varNumDpth]

    # Return vertex data:
    return aryVtkData
