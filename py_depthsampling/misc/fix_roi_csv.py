# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

(C) Ingo Marquardt, 2018
"""

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


import csv
import numpy as np


def fix_roi_csv(strCsvRoi, strCsvRoiOut, strVtkIn, varNumHdrRoi=1,  #noqa
                strPrcdCoor='POINTS', varNumLne=1, varRnd=1):
    """
    Fix indices in ROI definitions.

    Parameters
    ----------
    strCsvRoi : string
        Base path of csv files with ROI definition (i.e. patch of cortex
        selected on the surface, e.g. V1 or V2).
    strCsvRoiOut : string
        Output path of modified csv file.
    strVtkIn : string
        Path of reference vtk file. The coordinates of vertices in the ROI csv
        file are compared with the coordinate of vertices in this vtk mesh, and
        the indices in the ROI file are replaced with the indices in this vtk
        file.
    varNumHdrRoi : int
        Number of header lines in ROI CSV file.
    strPrcdCoor : string
        Beginning of string which precedes vertex coordinates in data vtk
        files.
    varNumLne : int
        Number of lines between string ('strPrcdCoor') and coordinate in vtk
        file.
    varRnd : int
        Number of decimal places after rounding for comparison of vertex
        coordinates between csv ROI and vtk mesh.

    Returns
    -------
    This function has no return value. The indices in the csv file are updated,
    and changes are written to disk.

    Notes
    -----
    Regions of interest (ROIs) created with paraview on a cortical surface may
    have inconsistent vertex indices with respect to vtk meshes. This may be
    the case if existing ROIs are loaded from disk (loading a paraview 'state'
    file) and then edited. Here, such edited ROIs are loaded, and their
    indicies are replaced with those from a reference VTK mesh file, based on
    the correspondence of vertex coordinates between the VTK mesh and the CSV
    ROI.
    """
    print('-Fixing vertex indices in ROI CSV file.')
    print(('---CSV ROI: ' + strCsvRoi))

    # -------------------------------------------------------------------------
    # *** Load vtk file

    # Load single-depth-level vtk file (e.g. 'polar_angle_thr.vtk' that was
    # used to delineate ROI). Different to the standard ROI loading module
    # (i.e. 'load_vtk_single'), we do not only load the data, but the indices
    # and coordinates of vertices.

    # Open vtk file:
    fleVtkIn = open(strVtkIn, 'r')

    # Read file  vtk file:
    csvIn = csv.reader(fleVtkIn,
                       delimiter='\n',
                       skipinitialspace=True)

    # Create empty list for vertex data:
    lstVtkData = []

    # Loop through csv object to fill list with vertex data:
    for lstTmp in csvIn:
        for strTmp in lstTmp:
            lstVtkData.append(strTmp[:])

    # Close file:
    fleVtkIn.close()

    # Get index of string that precedes the vertex coordinates:
    for idxSrch in range(0, len(lstVtkData)):
        if lstVtkData[idxSrch].startswith((strPrcdCoor)):
            varIdxTmp = idxSrch

    # Get number of vertices from the line preceding the specified string. The
    # string is supposed to be of the form 'POINTS 252382 float'. We split the
    # string and assume that the second element is the number of vertices.
    varNumDataVrtx = int(lstVtkData[varIdxTmp].split()[1])

    # Index of first vertex data point:
    varIdxFrst = varIdxTmp + varNumLne

    # Array for vertex coordinates (columns correspond to vertex ID, x-, y-,
    # and z-position, respectively):
    aryVtk = np.zeros((varNumDataVrtx, 4))

    # Get vertex coordinates:
    for idxData in range(0, varNumDataVrtx):

        # Line of current vertex in csv list:
        varTmpStrt = varIdxFrst + idxData

        # Put vertex data into array:
        aryVtk[idxData, 0] = float(idxData)
        aryVtk[idxData, 1] = float(lstVtkData[varTmpStrt].split()[0])
        aryVtk[idxData, 2] = float(lstVtkData[varTmpStrt].split()[1])
        aryVtk[idxData, 3] = float(lstVtkData[varTmpStrt].split()[2])

    # 'aryVtk' now contains all vertex indices and coordinates:
    # aryVtk[idxVertex, x, y, z]

    # -------------------------------------------------------------------------
    # *** Load CSV ROI

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

    # -------------------------------------------------------------------------
    # *** Fix CSV ROI indices

    # Round coordinates:
    vecXrnd01 = np.around(aryVtk[:, 1], decimals=varRnd)
    vecYrnd01 = np.around(aryVtk[:, 2], decimals=varRnd)
    vecZrnd01 = np.around(aryVtk[:, 3], decimals=varRnd)

    # Round with one extra decimal point:
    vecXrnd02 = np.around(aryVtk[:, 1], decimals=(varRnd + 1))
    vecYrnd02 = np.around(aryVtk[:, 2], decimals=(varRnd + 1))
    vecZrnd02 = np.around(aryVtk[:, 3], decimals=(varRnd + 1))

    # List for vertices for which no correspondence is found:
    lstErr = []

    # Loop through CSV file:
    for idxCsv in range(varNumHdrRoi, len(lstCsvRoi)):

        # Get current line of csv file (split into list of strings,
        # corresponding to parameter value (e.g. polar angle), vertex ID,
        # x-coordinate, y-coordinate, z-coordinate.
        lstTmp = lstCsvRoi[idxCsv].split(',')

        # Coordinates:
        varTmpX = np.around(float(lstTmp[2]), decimals=varRnd)
        varTmpY = np.around(float(lstTmp[3]), decimals=varRnd)
        varTmpZ = np.around(float(lstTmp[4]), decimals=varRnd)

        # Create sets of vertex coordinates in vtk mesh that coincide with the
        # coordinate of the current vertex from the csv file:
        setTmpX = set(np.where(np.equal(vecXrnd01, varTmpX))[0])
        setTmpY = set(np.where(np.equal(vecYrnd01, varTmpY))[0])
        setTmpZ = set(np.where(np.equal(vecZrnd01, varTmpZ))[0])

        # List of intersections (i.e. vertices that have the same x, y, and z
        # coordinate as the current vertex from the csv file):
        lstInt = list(set.intersection(setTmpX, setTmpY, setTmpZ))

        # If everything went ok, there should only be one vertex at the current
        # position (otherwise consider reducing the rounding of coordinates):
        if (len(lstInt) > 1):

            # Round one decimal point less, and see if there is a unique
            # vertex.

            # Coordinates:
            varTmpX = np.around(float(lstTmp[2]), decimals=(varRnd + 1))
            varTmpY = np.around(float(lstTmp[3]), decimals=(varRnd + 1))
            varTmpZ = np.around(float(lstTmp[4]), decimals=(varRnd + 1))

            # Create sets of vertex coordinates in vtk mesh that coincide with
            # the coordinate of the current vertex from the csv file:
            setTmpX = set(np.where(np.equal(vecXrnd02, varTmpX))[0])
            setTmpY = set(np.where(np.equal(vecYrnd02, varTmpY))[0])
            setTmpZ = set(np.where(np.equal(vecZrnd02, varTmpZ))[0])

            # List of intersections (i.e. vertices that have the same x, y, and
            # z coordinate as the current vertex from the csv file):
            lstInt = list(set.intersection(setTmpX, setTmpY, setTmpZ))

            if (len(lstInt) > 1):

                # Error message:
                strErrMsg = ('---Error: More than one vertex from reference '
                             + 'VTK file has the same coordinates as vertex '
                             + 'number '
                             + str(idxCsv)
                             + ' from CSV ROI file. There should only be one '
                             + 'vertex per coordinate. Consider reduced '
                             + 'rounding of vertex coordinates.')
                raise ValueError(strErrMsg)

        if (len(lstInt) == 1):

            # The list of vertices at intersection of coordinates has length
            # one, containing the index of the vertex (in the vtk mesh)
            # corresponding to the current vertex in the CSV ROI. We replace
            # the (possibly wrong) vertex index in the CSV ROI with the vtk
            # mesh index.

            # Convert index to string:
            strTmpVtkIdx = str(lstInt[0])

            # Replace old (wrong) index with new index:
            lstTmp[1] = strTmpVtkIdx

            # Converte list of strings to string (same as original line from
            # csv file, just with new index):
            strTmp = ','.join(lstTmp)

            # Put modified line into csv list:
            lstCsvRoi[idxCsv] = strTmp

        elif (len(lstInt) == 0):

            # Remember vertices without correspondence:
            lstErr.append(idxCsv)

    # Counter for vertices for which no correspondence can be established:
    varErrCnt = 0

    # If no corresponding points could be found, first check whether new
    # (modified) vertex IDs of preceeding and suceeding vertices are in
    # sequence (e.g. if preceeding ID is 60075 and suceeding ID is 60077,
    # assign ID = 60076 to respective vertex). If this is not the case, remove
    # this vertex from the ROI. Loop through missing elements in reverse order,
    # because list indices change when removing elements.
    for idxErr in list(reversed(lstErr)):

        # Skip first and last vertices (there would be no preceeding or
        # suceeding value for interpolation):
        if (idxErr > 1) and (len(lstCsvRoi) > idxErr):

            # Preceeding index:
            strTmpPre = lstCsvRoi[(idxErr - 1)].split(',')[1]
            # Suceesing index:
            strTmpSce = lstCsvRoi[(idxErr + 1)].split(',')[1]

            # Is succeeding index = preceeding index plus two?
            if ((int(strTmpPre) + 2) == int(strTmpSce)):

                # Get current line of csv file (split into list of strings,
                # corresponding to parameter value (e.g. polar angle), vertex
                # ID, x-coordinate, y-coordinate, z-coordinate.
                lstTmp = lstCsvRoi[idxErr].split(',')

                # Convert index to string:
                strTmpVtkIdx = str((int(strTmpPre) + 1))

                # Replace old (wrong) index with new index:
                lstTmp[1] = strTmpVtkIdx

                # Converte list of strings to string (same as original line
                # from csv file, just with new index):
                strTmp = ','.join(lstTmp)

                # Put modified line into csv list:
                lstCsvRoi[idxErr] = strTmp

            else:

                # Delete vertex.
                del(lstCsvRoi[idxErr])
                # Increment error counter:
                varErrCnt += 1

        else:

            # Delete vertex.
            del(lstCsvRoi[idxErr])
            # Increment error counter:
            varErrCnt += 1

    strTmp = ('---Number of vertices in ROI CSV file for which no '
              + 'corresponding vertex could be found in VTK mesh: '
              + str(varErrCnt))
    print(strTmp)

    # Replace header (to avoid problems with multiple delimiters, i.e. ' ' and
    # ',').
    lstCsvRoi[0] = 'header'

    # -------------------------------------------------------------------------
    # *** Save modified CSV file

    # Create output csv object:
    objCsvOt = open(strCsvRoiOut, 'w')

    # Save manipulated list to disk:
    csvOt = csv.writer(objCsvOt, delimiter=' ', lineterminator='\n')

    # Write output list data to file (row by row):
    for strTmp in lstCsvRoi:
        csvOt.writerow([strTmp])

    # Close:
    objCsvOt.close()
    # -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Run function

# Region of interest ('v1' or 'v2'):
lstRoi = ['v1', 'v2', 'v3']

# Hemispheres ('lh' or 'rh'):
lstHmsph = ['lh', 'rh']

# List of subject identifiers:
lstSubIds = ['20171023',  # '20171109',
             '20171204_01',
             '20171204_02',
             '20171211',
             '20171213',
             '20180111',
             '20180118']

# Path of input csv files (subject ID, hemisphere, and ROI left open):
strCsvRoi = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/{}.csv'

# Path of output csv files (subject ID, hemisphere, and ROI left open):
strCsvRoiOut = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/{}_mod.csv'

# Path of reference vtk file (subject ID and hemisphere left open):
strVtkIn = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/polar_angle_thr.vtk'  #noqa

for idxSub in lstSubIds:
    for idxHmpsh in lstHmsph:
        for idxRoi in lstRoi:

            fix_roi_csv(strCsvRoi.format(idxSub, idxHmpsh, idxRoi),
                        strCsvRoiOut.format(idxSub, idxHmpsh, idxRoi),
                        strVtkIn.format(idxSub, idxHmpsh))
