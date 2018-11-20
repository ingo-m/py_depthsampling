# -*- coding: utf-8 -*-
"""Main routine for analysis & visualisation of depth sampling results."""

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
import multiprocessing as mp
from py_depthsampling.get_data.acr_subs_get_data import acr_subs_get_data
from py_depthsampling.plot.plt_dpth_prfl_acr_subs import plt_dpth_prfl_acr_subs


def ds_main(strRoi, lstHmsph, lstSubIds, lstCon, lstConLbl, strVtkDpth01,
            lgcSlct01, strCsvRoi, varNumHdrRoi, lgcSlct02, strVtkSlct02,
            varThrSlct02, lgcSlct03, strVtkSlct03, varThrSlct03, lgcSlct04,
            strVtkSlct04, tplThrSlct04, varNumDpth, strPrcdData, varNumLne,
            strTitle, lstLimY, varAcrSubsYmin, varAcrSubsYmax, strXlabel,
            strYlabel, strPltOtPre, strPltOtSuf, varDpi, varNormIdx,
            lgcNormDiv, strDpthMeans, strMetaCon='', varNumLblY=5,
            tplPadY=(0.0, 0.0)):
    """
    Delineate ROIs and create cortical depth profiles from VTK meshes.

    Main routine for analysis & visualisation of depth sampling results.
    """
    # *************************************************************************
    # *** Plot and retrieve single subject data

    print('-Visualisation of depth sampling results')

    print('---Plotting & retrieving single subject data')

    print(('   ROI: ' + strRoi + ' Condition: ' + lstCon[0]
           + ' Metacondition: ' + strMetaCon))

    # Number of hemispheres:
    varNumHmsph = len(lstHmsph)

    # Number of subjects:
    varNumSubs = len(lstSubIds)

    # Number of conditions (i.e. number of data vtk files per subject):
    varNumCon = len(lstCon)

    # Array for single-subject depth sampling results:
    arySubDpthMns = np.zeros((varNumSubs, varNumHmsph, varNumCon, varNumDpth))

    # Vector for number of vertices contained in the ROI:
    vecNumInc = np.zeros((varNumSubs, varNumHmsph))

    # Loop through hemispheres:
    for idxHmsph in range(varNumHmsph):

        # List for single subject data (mean PE over depth levels):
        lstSubData01 = [None] * varNumSubs

        # Empty list to collect results from parallelised function:
        lstParResult = [None] * varNumSubs

        # Empty list for processes:
        lstPrcs = [None] * varNumSubs

        # Create a queue to put the results in:
        queOut = mp.Queue()

        # Loop through subjects:
        for idxSub in range(varNumSubs):

            # Current hemisphere:
            strHmsph = lstHmsph[idxHmsph]

            # Create list with complete file names for the data to be
            # depth-sampled:
            lstVtkDpth01 = [strVtkDpth01.format(lstSubIds[idxSub],
                                                strHmsph,
                                                strTmp) for strTmp in lstCon]

            # Complete file paths:
            strCsvRoiTmp = strCsvRoi.format(lstSubIds[idxSub], strHmsph,
                                            strRoi)
            strVtkSlct02Tmp = strVtkSlct02.format(lstSubIds[idxSub], strHmsph)
            strVtkSlct03Tmp = strVtkSlct03.format(lstSubIds[idxSub], strHmsph)
            strVtkSlct04Tmp = strVtkSlct04.format(lstSubIds[idxSub], strHmsph,
                                                  strMetaCon)
            strPltOtSufTmp = strPltOtSuf.format(('_' + strHmsph))

            # Prepare processes that plot & return single subject data:
            lstPrcs[idxSub] = \
                mp.Process(
                    target=acr_subs_get_data,
                    args=(idxSub,              # Process ID
                          lstSubIds[idxSub],   # Data struc - Subject ID
                          lstVtkDpth01,        # Data struc - Pth vtk I
                          varNumDpth,          # Data struc - Num depth lvls
                          strPrcdData,         # Data struc - Str prcd VTK
                          varNumLne,           # Data struc - Lns prcd VTK
                          lgcSlct01,           # Criterion 1 - Yes or no?
                          strCsvRoiTmp,        # Criterion 1 - CSV path
                          varNumHdrRoi,        # Criterion 1 - Header lines
                          lgcSlct02,           # Criterion 2 - Yes or no?
                          strVtkSlct02Tmp,     # Criterion 2 - VTK path
                          varThrSlct02,        # Criterion 2 - Threshold
                          lgcSlct03,           # Criterion 3 - Yes or no?
                          strVtkSlct03Tmp,     # Criterion 3 - VTK path
                          varThrSlct03,        # Criterion 3 - Threshold
                          lgcSlct04,           # Criterion 4 - Yes or no?
                          strVtkSlct04Tmp,     # Criterion 4 - VTK path
                          tplThrSlct04,        # Criterion 4 - Threshold
                          lgcNormDiv,          # Normalisation - Yes or no?
                          varNormIdx,          # Normalisation - Reference
                          varDpi,              # Plot - dots per inch
                          lstLimY[idxSub][0],  # Plot - Minimum of Y axis
                          lstLimY[idxSub][1],  # Plot - Maximum of Y axis
                          lstConLbl,           # Plot - Condition labels
                          strXlabel,           # Plot - X axis label
                          strYlabel,           # Plot - Y axis label
                          strTitle,            # Plot - Title
                          strPltOtPre,      # Plot - Output file path prefix
                          strPltOtSufTmp,   # Plot - Output file path suffix
                          strMetaCon,       # Metacondition (stim/periphery)
                          queOut))          # Queue for output list

        # Daemon (kills processes when exiting):
        lstPrcs[idxSub].Daemon = True

        # Start processes:
        for idxSub in range(0, varNumSubs):
            lstPrcs[idxSub].start()

        # Collect results from queue:
        for idxSub in range(0, varNumSubs):
            lstParResult[idxSub] = queOut.get(True)

        # Join processes:
        for idxSub in range(0, varNumSubs):
            lstPrcs[idxSub].join()

        # List for arrays with depth profiles.
        lstSubData01 = [None] * varNumSubs

        # Put output into correct order:
        for idxRes in range(varNumSubs):

            # Index of results (first item in output list):
            varTmpIdx = lstParResult[idxRes][0]

            # Put fitting results into list, in correct order:
            lstSubData01[varTmpIdx] = lstParResult[idxRes][1]
            vecNumInc[varTmpIdx, idxHmsph] = lstParResult[idxRes][2]

        # Retrieve single-subject data from list:
        for idxSub in range(varNumSubs):
            arySubDpthMns[idxSub, idxHmsph, :, :] = np.copy(
                lstSubData01[idxSub])

    # Array for single-subject depth sampling results, averaged over
    # hemispheres:
    arySubDpthMns02 = np.zeros((varNumSubs, varNumCon, varNumDpth))

    # Average across hemispheres. Because the function used for weighted
    # averaging does not work with broadcasting, we have to loop through
    # subjects.
    for idxSub in range(varNumSubs):

        # Get array for current subject:
        aryTmp = np.copy(arySubDpthMns[idxSub, :, :, :])

        # Replace nan by zero (in case of empty depth profile - these will be
        # weighted with zero in the across hemispheres averaging anyway.
        aryTmp = np.nan_to_num(aryTmp)

        # Average across hemispheres:
        arySubDpthMns02[idxSub, :, :] = np.average(
            aryTmp, axis=0, weights=vecNumInc[idxSub, :])

    del(arySubDpthMns)
    arySubDpthMns = arySubDpthMns02

    # Add number of vertices over hemispheres:
    vecNumInc = np.sum(vecNumInc, axis=1)
    # *************************************************************************

    # *************************************************************************
    # *** Save results

    # We save the mean parameter estimates of all subjects to disk. This file
    # can be used to plot results from different ROIs in one plot. The depth
    # profile for each condition is saved to a separate file (for consistency):

    for idxCon in range(varNumCon):

        # Form of the array that is saved to disk:
        # arySubDpthMns[subject, depth]

        # In addition, a vector with the number of vertices (for that ROI in
        # tha subject) is saved, in order to be able to normalise when
        # averaging over subjects. Shape: vecNumInc[subject]

        # Save subject-level depth profiles, and number of vertices per
        # subject:
        np.savez(strDpthMeans.format(lstCon[idxCon]),
                 arySubDpthMns=arySubDpthMns[:, idxCon, :],
                 vecNumInc=vecNumInc)
    # *************************************************************************

    # *************************************************************************
    # *** Plot mean over subjects

    print('---Plot results - mean over subjects.')

    plt_dpth_prfl_acr_subs(arySubDpthMns,
                           varNumSubs,
                           varNumDpth,
                           varNumCon,
                           varDpi,
                           varAcrSubsYmin,
                           varAcrSubsYmax,
                           lstConLbl,
                           strXlabel,
                           strYlabel,
                           strTitle,
                           strPltOtPre,
                           strPltOtSuf.format(''),
                           strErr='sem',
                           vecWghts=vecNumInc,
                           varNumLblY=varNumLblY,
                           tplPadY=tplPadY)
    # *************************************************************************
