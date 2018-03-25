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

#strSubId = '20150930'
#strVtkDpth01 = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_pe_stim_lvl_01.vtk'
#strPrcdData = 'SCALARS'
#varNumLne = 2
#varNumHdrRoi = 1
#strCsvRoi = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/v1.csv'
#vecInc = np.ones(9049)

#from ds_loadCsvRoi import funcLoadCsvRoi
#aryRoiVrtx = funcLoadCsvRoi(strCsvRoi, varNumHdrRoi)


def funcCrtVtkMsk(strSubId,      # Data struc - Subject ID
                  strVtkDpth01,  # Data struc - Path first data vtk file
                  strPrcdData,   # Data struc - Str. prcd. VTK data
                  varNumLne,     # Data struc - Lns. prcd. data VTK
                  strCsvRoi,     # Data struc - ROI CSV fle (for output naming)
                  aryRoiVrtx,    # Array with ROI definition (1st crit.)
                  vecInc,        # Vertex inclusion vector
                  ):
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
    lstVtkData[varIdxPreData] = (strPrcdData + ' SurfaceMask ' + 'float 1')

    # Change default lookup table:
    lstVtkData[(varIdxPreData + 1)] = 'LOOKUP_TABLE viridis'

    # TODO: Make this section more elegant.

    # If no ROI mask is provided, 'aryRoiVrtx' is set to zero, and we cannot
    # use its values.
    try:
        # If a ROI mask is provided (in form of a CSV file), the vertex
        # selection procedure (the result of which is vecInc) is only carried
        # out within the ROI. Therefore, we need the indicies of the vertices
        # that are within the ROI. All verticies that are not in the ROI are not
        # considered for depth sampling, and are therefore set to zero here. The
        # indicies of the vertices that are included in the ROI are in the
        # second (i.e. index=1) column of aryRoiVrtx (i.e. the indicies relative
        # to the full vtk mesh).
        vecRoiVrtx = aryRoiVrtx[:, 1].astype(np.uint32)
    except:
        vecRoiVrtx = np.arange(varNumDataVrtx, dtype=np.uint32)

    # Since the output is supposed to be float, we have to change vecInc from
    # boolean to float:
    vecInc = vecInc.astype(np.float32)

    # Index for accessing the verticies included in the CSV ROI:
    varCount = 0

    # Loop through vertices and replace values:
    # for idxVrtx in range(varIdxFrst, (varIdxFrst + varNumDataVrtx)):
    for idxVrtx in range(0, varNumDataVrtx):

        # If the vertex with the current index is within the ROI (as defined
        # by the CSV file), we replace its data value with the value in the
        # inclusion vector vecInc (i.e. one or zero):
        if idxVrtx in vecRoiVrtx:
            # Since the header and the section with the vertex coordinates
            # precedes the numeric vertex data, we have to add the row number
            # of the first numeric data point to the index:
            lstVtkData[(idxVrtx + varIdxFrst)] = \
                str(np.around(vecInc[varCount], decimals=1))
            varCount = varCount + 1
        # If the current vertex is not in the ROI (as defined by the CSV
        # file), it can by definition also not considered for depth sampling,
        # so we set its value to zero:
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
    strVtkOt = (strVtkOt + '/' + strSubId + '_vertex_inclusion_mask_' +
                strRoi + '.vtk')

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
