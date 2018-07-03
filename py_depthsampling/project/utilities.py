# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

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


import numpy as np
import scipy as sp
from py_depthsampling.get_data.load_csv_roi import load_csv_roi
from py_depthsampling.get_data.load_vtk_multi import load_vtk_multi


def get_data(strData, strPthMneEpi, strPthR2, strPthSd, strPthX, strPthY,
             strCsvRoi, varNumDpth=11, strPrcdData='SCALARS', varNumLne=2,
             varNumHdrRoi=1, lstDpth=None):
    """
    Load data for projection into visual space.

    Parameters
    ----------
    strData : string
        Path of vtk mesh with data to project into visual space (e.g. parameter
        estimates).
    strPthR2 : string
        Path of vtk mesh with R2 values from pRF mapping (at multiple depth
        levels).
    strPthSd : string
        Path of vtk mesh with pRF sizes (at multiple depth levels).
    strPthX : string
        Path of vtk mesh with pRF x positions (at multiple depth levels).
    strPthY : string
        Path of vtk mesh with pRF y positions (at multiple depth levels).
    strCsvRoi : string
        Path of csv file with ROI definition.
    varNumDpth : int
        Number of cortical depths.
    strPrcdData : string
        Beginning of string which precedes vertex data in data vtk files.
    varNumLne : int
        Number of lines between vertex-identification-string and first data
        point in vtk meshes.
    varNumHdrRoi : int
        Number of header lines in ROI CSV file.
    lstDpth : list
        List with depth levels to average over. For instance, if `lstDpth = [0,
        1, 2]`, the average over the first three depth levels is calculated. If
        `None`, average over all depth levels.

    Returns
    -------
    vecData : np.array
        Array with data values contained within the ROI.
    vecR2 : np.array
        Array with R2 values of all vertices contained in the ROI.
    vecSd : np.array
        Array with pRF sizes of all vertices contained in the ROI.
    vecX : np.array
        Array with x positions of all vertices contained in the ROI.
    vecY : np.array
        Array with y positions of all vertices contained in the ROI.

    Notes
    -----
    Load data from vtk meshes for projection into visual space.
    """
    # -------------------------------------------------------------------------
    # *** Load data

    # Load data to be projected:
    aryData = load_vtk_multi(strData,
                             strPrcdData,
                             varNumLne,
                             varNumDpth)

    # Load mean EPI:
    aryMneEpi = load_vtk_multi(strPthMneEpi,
                               strPrcdData,
                               varNumLne,
                               varNumDpth)

    # Load R2 map:
    aryR2 = load_vtk_multi(strPthR2,
                           strPrcdData,
                           varNumLne,
                           varNumDpth)

    # Load SD map:
    arySd = load_vtk_multi(strPthSd,
                           strPrcdData,
                           varNumLne,
                           varNumDpth)

    # Load x position map:
    aryX = load_vtk_multi(strPthX,
                          strPrcdData,
                          varNumLne,
                          varNumDpth)

    # Load y position map:
    aryY = load_vtk_multi(strPthY,
                          strPrcdData,
                          varNumLne,
                          varNumDpth)

    # Import CSV file with ROI definition
    aryRoiVrtx = load_csv_roi(strCsvRoi, varNumHdrRoi)

    # -------------------------------------------------------------------------
    # *** Apply ROI mask

    # The second column of the array "aryRoiVrtx" contains the indicies of
    # the vertices contained in the ROI. We extract that information:
    vecRoiIdx = aryRoiVrtx[:, 1].astype(np.int64)

    # Only keep vertices that are contained in the ROI:
    aryData = aryData[vecRoiIdx, :]
    aryMneEpi = aryMneEpi[vecRoiIdx, :]
    aryR2 = aryR2[vecRoiIdx, :]
    arySd = arySd[vecRoiIdx, :]
    aryX = aryX[vecRoiIdx, :]
    aryY = aryY[vecRoiIdx, :]

    # -------------------------------------------------------------------------
    # *** Average across depth levels

    if lstDpth is None:

        # Average over all depth levels:
        vecData = np.mean(aryData, axis=1)
        vecMneEpi = np.mean(aryMneEpi, axis=1)
        vecR2 = np.mean(aryR2, axis=1)
        vecSd = np.mean(arySd, axis=1)
        vecX = np.mean(aryX, axis=1)
        vecY = np.mean(aryY, axis=1)

    elif len(lstDpth) == 1:

        # If there is only one depth level, averaging over depth levels
        # does not make sense.
        vecData = aryData.flatten()
        vecMneEpi = aryMneEpi.flatten()
        vecR2 = aryR2.flatten()
        vecSd = arySd.flatten()
        vecX = aryX.flatten()
        vecY = aryY.flatten()

    else:

        # Average over selected depth levels:
        vecData = np.mean(aryData[:, lstDpth], axis=1)
        vecMneEpi = np.mean(aryMneEpi[:, lstDpth], axis=1)
        # Average pRF parameters across cortical depth:
        vecR2 = np.mean(aryR2, axis=1)
        vecSd = np.mean(arySd, axis=1)
        vecX = np.mean(aryX, axis=1)
        vecY = np.mean(aryY, axis=1)

    return vecData, vecMneEpi, vecR2, vecSd, vecX, vecY


def crt_gauss(varSizeX, varSizeY, varPosX, varPosY, varSd):
    """
    Create 2D Gaussian kernel.

    Parameters
    ----------
    varSizeX : int, positive
        Width of the visual field.
    varSizeY : int, positive
        Height of the visual field..
    varPosX : int, positive
        X position of centre of 2D Gauss.
    varPosY : int, positive
        Y position of centre of 2D Gauss.
    varSd : float, positive
        Standard deviation of 2D Gauss.

    Returns
    -------
    aryGauss : 2d numpy array, shape [varSizeX, varSizeY]
        2d Gaussian.
    """
    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    aryX, aryY = sp.mgrid[0:varSizeX,
                          0:varSizeY]

    # The actual creation of the Gaussian array:
    aryGauss = (
        (np.square((aryX - varPosX))
         + np.square((aryY - varPosY))
         ) /
        (2.0 * np.square(varSd))
        )
    aryGauss = np.exp(-aryGauss) / (2.0 * np.pi * np.square(varSd))

    return aryGauss
