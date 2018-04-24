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

import csv
import os
import numpy as np  #noqa


def vtk_msk(strSubId,        # Data struc - Subject ID
            strVtkDpth01,    # Data struc - Path first data vtk file
            strPrcdData,     # Data struc - Str. prcd. VTK data
            varNumLne,       # Data struc - Lns. prcd. data VTK
            strCsvRoi,       # Data struc - ROI CSV fle (for output naming)
            vecInc,          # Vertex inclusion vector
            strMetaCon=''):  # Metacondition (stimulus or periphery)
    """
    Create surface mask for selected vertices.

    This function creates a vtk file containing a mask of those vertices that
    have been selected for depth sampling (vtk file that can be opened in
    paraview).
    """
    # *************************************************************************
    # *** Access vtk data

    # Open file:
    fleVtkIn = open(strVtkDpth01, 'r')

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
            varIdxPreData = idxSrch

    # Get number of vertices from the line preceding the specified string:
    strNumDataVrtx = lstVtkData[(varIdxPreData - 1)]

    # The number of vertecies is preceded by the string 'POINT_DATA' in vtk
    # files. We extract the number behind the string:
    varNumDataVrtx = int(strNumDataVrtx[11:])

    # Index of first vertex data point:
    varIdxFrst = varIdxPreData + varNumLne
    # *************************************************************************

    # *************************************************************************
    # *** Replace vtk data with mask

    # Change header (file name):
    lstVtkData[1] = (strSubId + '_vertex_inclusion_mask')

    # We change the string that precedes the numerical vertex data. The first
    # word (probably 'SCALARS') we leave as it is (needs to be supplied as
    # an input to this function anyways in order to find the numeric data).
    # Second is the name associated with the data points (that is displayed
    # in paraview), we give some sensible name. Finally, we specify the
    # datatype as float (integer type would be sufficient, but we keep it as
    # float for consistency), followed by the number of data points per vertex
    # (which is one data point per vertex).
    lstVtkData[varIdxPreData] = (strPrcdData + ' ROI_MASK ' + 'float 1')

    # Change default lookup table:
    lstVtkData[(varIdxPreData + 1)] = 'LOOKUP_TABLE viridis'

    # Loop through vertices and replace values:
    # for idxVrtx in range(varIdxFrst, (varIdxFrst + varNumDataVrtx)):
    for idxVrtx in range(0, varNumDataVrtx):

        # If the vertex with the current index is supposed to be included, we
        # set its value to one:
        if vecInc[idxVrtx]:
            # Since the header and the section with the vertex coordinates
            # precedes the numeric vertex data, we have to add the row number
            # of the first numeric data point to the index:
            lstVtkData[(idxVrtx + varIdxFrst)] = '1.0'
        # If the current vertex is not to be included, we set its value to
        # zero:
        else:
            lstVtkData[(idxVrtx + varIdxFrst)] = '0.0'
    # *************************************************************************

    # *************************************************************************
    # *** Save mask

    # Get directory of input vtk file:
    strVtkOt = os.path.abspath(os.path.join(os.path.dirname(strVtkDpth01)))

    # Get file name of CSV file used for ROI selection, without file extension
    # (this is needed to know which ROI definition was used to created this
    # vertex inclusion mask, e.g. V1 or V2):
    strRoi = os.path.splitext(os.path.split(strCsvRoi)[-1])[0]

    # Add output file name:
    if strMetaCon == '':
        strVtkOt = (strVtkOt + '/' + strSubId + '_vertex_inclusion_mask_'
                    + strRoi + '.vtk')
    else:
        strVtkOt = (strVtkOt + '/' + strSubId + '_vertex_inclusion_mask_'
                    + strRoi + '_' + strMetaCon + '.vtk')

    # Create output csv object:
    objCsvOt = open(strVtkOt, 'w')

    # Save manipulated list to disk:
    csvOt = csv.writer(objCsvOt, lineterminator='\n')

    # Write output list data to file (row by row):
    for strTmp in lstVtkData:
        csvOt.writerow([strTmp])

    # Close:
    objCsvOt.close()
    # *************************************************************************
