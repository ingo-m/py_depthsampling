# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

@author: Ingo Marquardt, 05.12.2016
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
from ds_loadVtkSingle import funcLoadVtkSingle
from ds_loadVtkMulti import funcLoadVtkMulti
from ds_loadCsvRoi import funcLoadCsvRoi


def funcParamEccDpthGet(strVtkEcc,
                        strPrcdData,
                        varNumLne,
                        strVtkParam,
                        varNumDpth,
                        strCsvRoi,
                        varNumHdrRoi,
                        vecEccBin,
                        strVtkThr,
                        varThr):
    """
    Load data for eccentricity & cortical depth analysis.

    This function loads pRF eccentricity information & statistical parameters
    (e.g. parameter estimates) from vtk files, samples them within an ROI
    defined by a csv file, and returns an array with mean statistical
    parameters separately for eccentricity bins & cortical depths (i.e. PE by
    eccentricity and cortical depth).

    This function is part of a tool for analysis of cortical-depth-dependent
    fMRI responses at different retinotopic eccentricities. (Which is a part
    of the depth sampling pipeline.)
    """
    # *************************************************************************
    # *** Import data

    # Import the eccentricity information (one value per vertex):
    vecEcc = funcLoadVtkSingle(strVtkEcc,
                               strPrcdData,
                               varNumLne)

    # Import the parameter estimates (several values per vertex - one per
    # cortical depth level):
    aryParam = funcLoadVtkMulti(strVtkParam,
                                strPrcdData,
                                varNumLne,
                                varNumDpth)

    # Import ROI definition (csv file, list of vertices):
    aryRoiVrtx = funcLoadCsvRoi(strCsvRoi,
                                varNumHdrRoi)

    # Import intensity data for vertex selection (e.g. R2 values):
    aryVtkThr = funcLoadVtkMulti(strVtkThr,
                                 strPrcdData,
                                 varNumLne,
                                 varNumDpth)
    # *************************************************************************

    # *************************************************************************
    # *** Select vertices contained within the ROI

    # The second column of the array "aryRoiVrtx" contains the indicies of the
    # vertices contained in the ROI. We extract that information:
    vecRoiIdx = aryRoiVrtx[:, 1].astype(np.int64)

    # Selcet vertices that are included in the patch of interest:
    vecEcc = vecEcc[vecRoiIdx]
    aryParam = aryParam[vecRoiIdx, :]
    # *************************************************************************

    # *************************************************************************
    # *** Select vertices based on intensity criterion

    # We would like to exclude vertices if the intensity (e.g. R2 value) is
    # below the threshold at any depth level. Get minimum value across
    # cortical depths:
    vecVtkThr = np.min(aryVtkThr, axis=1)

    # Previously, we selected those vertices in the data array that are
    # contained within the ROI. In order to apply the intensity-criterion, we
    # have to extract the data corresponding to the ROI from the intensity
    # array (extract e.g. R2 data for the ROI).

    # Selcet vertices from patch-selection arrays:
    vecVtkThr = vecVtkThr[vecRoiIdx]

    # Get indicies of vertices with value greater than the intensity
    # criterion:
    vecInc = np.greater_equal(vecVtkThr, varThr)

    # Apply selection to eccentricity data and statistical parameters:
    vecEcc = vecEcc[vecInc]
    aryParam = aryParam[vecInc, :]
    # *************************************************************************

    # *************************************************************************
    # *** Create eccenticity bins

    # Add extra dimension to vector so that it can be stacked:
    vecEcc = np.array(vecEcc, ndmin=2).T

    # Merge eccentricity information and statistical parameters into one array:
    aryData = np.hstack((vecEcc, aryParam))

    # Delete unneeded objects:
    del aryParam

    # Sort by eccentricity:
    aryData = aryData[aryData[:, 0].argsort()]

    # Number of eccentricity bins:
    varEccNum = vecEccBin.shape[0]

    # Get indicies of data values for eccentricity bins:
    vecEccIdx = np.zeros((varEccNum), dtype=np.int64)

    # Initialise counter for accessing the eccentricity bins:
    varCnt = 0

    # Loop through vertices and get indicies of eccentricity bins:
    for idxVrtx in range(0, aryData.shape[0]):

        # Only continue if the last bin has not been reached yet:
        if varCnt < varEccNum:

            # Is the current vertex eccentricity value greater than the current
            # eccentricity-bin value?
            if vecEccBin[varCnt] < aryData[idxVrtx, 0]:

                # Put the index of the first vertex that belongs to the current
                # bin into the vector:
                vecEccIdx[varCnt] = idxVrtx

                varCnt += 1

    # If the last eccentricity bin has not been reached (i.e. if no vertex had
    # an eccentricity value greater than the upper limit of the last bin) we
    # set the respective index value to that of the vertex with the largest
    # eccentricity plus one (the result is that the vertex with the largest
    # eccentricity is the last vertex in the last bin):
    if vecEccIdx[-1] < 1.0:
        vecEccIdx[-1] = aryData.shape[0]
    # *************************************************************************

    # *************************************************************************
    # *** Average within eccentricity bins

    # Array for average statistical parameter within bin:
    aryMean = np.zeros(((varEccNum - 1), aryData.shape[1]))

    print(('---------' + str(vecEccIdx)))

    # Array for number of vertices in each bin:
    vecBinNumVrtc = np.zeros((varEccNum - 1))

    # Loop through eccentricity bins:
    for idxEcc in range(0, (varEccNum - 1)):

        # Indicies for accessing the vertex data according to their
        # eccentricity:
        varTmpFrst = vecEccIdx[idxEcc]
        varTmpLast = vecEccIdx[(idxEcc + 1)]

        # Number of vertices in current eccentricity bin:
        vecBinNumVrtc[idxEcc] =  varTmpLast - varTmpFrst

        # Report vertices in current eccentricity bin:
        strTmp = ('---------Eccentricity bin ' + str(idxEcc) + ' - Number ' +
                  'of vertices: ' + str(vecBinNumVrtc[idxEcc]))
        print(strTmp)

        # Calculate the mean across eccentricities:
        aryMean[idxEcc, :] = np.mean(aryData[varTmpFrst:varTmpLast, :],
                                     axis=0)

    # Remove first column from array (the eccentricity column):
    aryMean = aryMean[:, 1:]
    # *************************************************************************

    # *************************************************************************
    # *** Return

    # Return the average statistical parameter for each eccentricity bin &
    # depth level, and the eccentricity information within the ROI (for
    # histrogram plot):
    return aryMean, vecEcc, vecBinNumVrtc
    # *************************************************************************
