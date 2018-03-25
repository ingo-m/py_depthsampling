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

import numpy as np  # noqa
from ds_loadCsvRoi import funcLoadCsvRoi
# from ds_loadVtkSingle import funcLoadVtkSingle
from ds_loadVtkMulti import funcLoadVtkMulti
from ds_slctVrtcs import funcSlctVrtcs
from ds_crtVtkMsk import funcCrtVtkMsk
from ds_pltAcrDpth import funcPltAcrDpth


def funcAcrSubGetSubsData(idxPrc,              # Process ID
                          strSubId,            # Data struc - Subject ID
                          lstVtkDpth01,        # Data struc - Pth vtk I
                          varNumDpth,          # Data struc - Num. depth levels
                          strPrcdData,         # Data struc - Str prcd VTK data
                          varNumLne,           # Data struc - Lns prcd data VTK
                          lgcSlct01,           # Criterion 1 - Yes or no?
                          strCsvRoi,           # Criterion 1 - CSV path
                          varNumHdrRoi,        # Criterion 1 - Header lines
                          lgcSlct02,           # Criterion 2 - Yes or no?
                          strVtkSlct02,        # Criterion 2 - VTK path
                          varThrSlct02,        # Criterion 2 - Threshold
                          lgcSlct03,           # Criterion 3 - Yes or no?
                          strVtkSlct03,        # Criterion 3 - VTK path
                          varThrSlct03,        # Criterion 3 - Threshold
                          lgcSlct04,           # Criterion 4 - Yes or no?
                          strVtkSlct04,        # Criterion 4 - VTK path
                          varThrSlct04,        # Criterion 4 - Threshold
                          lgcNormDiv,          # Normalisation - Yes or no?
                          varNormIdx,          # Normalisation - Reference
                          varDpi,              # Plot - Dots per inch
                          varYmin,             # Plot - Minimum of Y axis
                          varYmax,             # Plot - Maximum of Y axis
                          lstConLbl,           # Plot - Condition labels
                          strXlabel,           # Plot - X axis label
                          strYlabel,           # Plot - Y axis label
                          strTitle,            # Plot - Title
                          strPltOtPre,         # Plot - Output file path prefix
                          strPltOtSuf,         # Plot - Output file path suffix
                          queOut):             # Queue for output list
    """
    Obtaining & plotting single subject data for across subject analysis.

    This function loads the data for each subject for a multi-subject analysis
    and passes the data to the parent function for visualisation.
    """
    # Only print status messages if this is the first of several parallel
    # processes:
    if idxPrc == 0:
        print('------Loading single subject data: ' + strSubId)

    # **************************************************************************
    # *** Import data

    # Import CSV file with ROI definition
    if lgcSlct01:
        if idxPrc == 0:
            print('---------Importing CSV file with ROI definition (first '
                  + 'criterion)')
        aryRoiVrtx = funcLoadCsvRoi(strCsvRoi, varNumHdrRoi)
    # Otherwise, create dummy vector (for function I/O)
    else:
        aryRoiVrtx = 0

    # Import second criterion vtk file (all depth levels)
    if lgcSlct02:
        if idxPrc == 0:
            print('---------Importing second criterion vtk file (all depth '
                  + 'levels).')
        arySlct02 = funcLoadVtkMulti(strVtkSlct02,
                                     strPrcdData,
                                     varNumLne,
                                     varNumDpth)
    # Otherwise, create dummy vector (for function I/O)
    else:
        arySlct02 = 0

    # Import third criterion vtk file (all depth levels)
    if lgcSlct03:
        if idxPrc == 0:
            print('---------Importing third criterion vtk file (all depth '
                  + 'levels).')
        arySlct03 = funcLoadVtkMulti(strVtkSlct03,
                                     strPrcdData,
                                     varNumLne,
                                     varNumDpth)
    # Otherwise, create dummy array (for function I/O)
    else:
        arySlct03 = 0

    # Import fourth criterion vtk file (all depth levels)
    if lgcSlct04:
        if idxPrc == 0:
            print('---------Importing fourth criterion vtk file (all depth '
                  + 'levels).')
        arySlct04 = funcLoadVtkMulti(strVtkSlct04,
                                     strPrcdData,
                                     varNumLne,
                                     varNumDpth)
    # Otherwise, create dummy array (for function I/O):
    else:
        arySlct04 = 0

    # Import depth data vtk files
    if idxPrc == 0:
        print('---------Importing depth data vtk files.')
    # Number of input files (i.e. number of conditions):
    varNumCon = len(lstVtkDpth01)
    # List for input data:
    lstDpthData01 = [None] * varNumCon
    # Loop through input data files:
    for idxIn in range(0, varNumCon):
        # Import data from file:
        lstDpthData01[idxIn] = funcLoadVtkMulti(lstVtkDpth01[idxIn],
                                                strPrcdData,
                                                varNumLne,
                                                varNumDpth)
        if idxPrc == 0:
            print('------------File ' + str(idxIn + 1) + ' out of '
                  + str(varNumCon))
    # **************************************************************************

    # **************************************************************************
    # *** Select vertices

    lstDpthData01, varNumInc, vecInc = \
        funcSlctVrtcs(varNumCon,           # Number of conditions
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
                      idxPrc)              # Process ID
    # **************************************************************************

    # **************************************************************************
    # *** Create VTK mesh mask

    if idxPrc == 0:
        print('---------Creating VTK mesh mask.')

    # We would like to be able to visualise the selected vertices on the
    # cortical surface, i.e. on a vtk mesh.
    funcCrtVtkMsk(strSubId,         # Data struc - Subject ID
                  lstVtkDpth01[0],  # Data struc - Path first data vtk file
                  strPrcdData,      # Data struc - Str. prcd. VTK data
                  varNumLne,        # Data struc - Lns. prcd. data VTK
                  strCsvRoi,        # Data struc - ROI CSV fle (outpt. naming)
                  aryRoiVrtx,       # Array with ROI definition (1st crit.)
                  vecInc)           # Vector with included vertices
    # **************************************************************************

    # **************************************************************************
    # *** Calculate mean & conficende interval

    if idxPrc == 0:
        print('---------Plot results - mean over vertices.')

    # Prepare arrays for results (mean & confidence interval):
    aryDpthMean = np.zeros((varNumCon, varNumDpth))
    aryDpthConf = np.zeros((varNumCon, varNumDpth))

    # Fill array with data - loop through input files:
    for idxIn in range(0, varNumCon):

        # Loop through depth levels:
        for idxDpth in range(0, varNumDpth):

            # Retrieve all vertex data for current input file & current depth
            # level:
            aryTmp = lstDpthData01[idxIn][:, idxDpth]

            # Calculate mean over vertices:
            varTmp = np.mean(aryTmp)
            # Place mean in array:
            aryDpthMean[idxIn, idxDpth] = varTmp

            # Calculate 95% confidence interval for the mean, obtained by
            # multiplying the standard error of the mean (SEM) by 1.96. We
            # obtain  the SEM by dividing the standard deviation by the
            # squareroot of the sample size n. We get n by taking 1/8 of the
            # number of vertices,  which corresponds to the number of voxels in
            # native resolution.
            varTmp = np.multiply(np.divide(np.std(aryTmp),
                                           np.sqrt(aryTmp.size * 0.125)),
                                 1.96)
            # Place confidence interval in array:
            aryDpthConf[idxIn, idxDpth] = varTmp

            # Calculate standard error of the mean.
            # varTmp = np.divide(np.std(aryTmp),
            #                    np.sqrt(aryTmp.size * 0.125))
            # Place SEM in array:
            # aryDpthConf[idxIn, idxDpth] = varTmp

            # Calculate standard deviation over vertices:
            # varTmp = np.std(aryTmp)

            # Place standard deviation in array:
            # aryDpthConf[idxIn, idxDpth] = varTmp

    # Normalise by division:
    if lgcNormDiv:

        if idxPrc == 0:
            print('---------Normalisation by division.')

        # Vector for subtraction:
        # vecSub = np.array(aryDpthMean[varNormIdx, :], ndmin=2)
        # Divide all rows by reference row:
        # aryDpthMean = np.divide(aryDpthMean, vecSub)

        # Calculate 'grand mean', i.e. the mean PE across depth levels and
        # conditions:
        varGrndMean = np.mean(aryDpthMean)
        # varGrndMean = np.median(aryDpthMean)

        # Divide all values by the grand mean:
        aryDpthMean = np.divide(aryDpthMean, varGrndMean)
        aryDpthConf = np.divide(aryDpthConf, varGrndMean)
    # **************************************************************************

    # **************************************************************************
    # *** Create plot

    # File name for figure:
    strPltOt = strPltOtPre + strSubId + strPltOtSuf

    # Title, including information about number of vertices:
    strTitleTmp = (strTitle
                   + ', '
                   + str(varNumInc)
                   + ' vertices')

    funcPltAcrDpth(aryDpthMean,  # Data: aryData[Condition, Depth]
                   aryDpthConf,  # Error shading: aryError[Condition, Depth]
                   varNumDpth,   # Number of depth levels (on the x-axis)
                   varNumCon,    # Number of conditions (separate lines)
                   varDpi,       # Resolution of the output figure
                   varYmin,      # Minimum of Y axis
                   varYmax,      # Maximum of Y axis
                   False,        # Boolean: whether to convert y axis to %
                   lstConLbl,    # Labels for conditions (separate lines)
                   strXlabel,    # Label on x axis
                   strYlabel,    # Label on y axis
                   strTitleTmp,  # Figure title
                   True,         # Boolean: whether to plot a legend
                   strPltOt)
    # **************************************************************************

    # **************************************************************************
    # *** Return

    # Output list:
    lstOut = [idxPrc,
              aryDpthMean]

    queOut.put(lstOut)
    # **************************************************************************
