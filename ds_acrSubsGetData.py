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
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ds_loadCsvRoi import funcLoadCsvRoi
from ds_loadVtkSingle import funcLoadVtkSingle
from ds_loadVtkMulti import funcLoadVtkMulti
from ds_slctVrtcs import funcSlctVrtcs
from ds_crtVtkMsk import funcCrtVtkMsk


def funcAcrSubGetSubsData(idxPrc,        # Process ID  #noqa
                          strSubId,      # Data struc - Subject ID
                          lstVtkDpth01,  # Data struc - Pth vtk I
                          strCsvRoi,     # Data struc - ROI CSV fle
                          varNumDpth,    # Data struc - Num. depth levels
                          varNumHdrRoi,  # Data struc - Header lines CSV
                          strPrcdData,   # Data struc - Str. prcd. VTK data
                          varNumLne,     # Data struc - Lns. prcd. data VTK
                          lgcSlct02,     # Criterion 2 - Yes or no?
                          strVtkSlct02,  # Criterion 2 - VTK path
                          varThrSlct02,  # Criterion 2 - Threshold
                          lgcSlct03,     # Criterion 3 - Yes or no?
                          strVtkSlct03,  # Criterion 3 - VTK path
                          varThrSlct03,  # Criterion 3 - Threshold
                          lgcMskExcl,    # Criterion 4 - Yes or no?
                          strVtkExcl,    # Criterion 4 - VTK path
                          varThrExcl,    # Criterion 4 - Threshold
                          lgcVtk02,      # Criterion 5 - Yes or no?
                          lstVtkDpth02,  # Criterion 5 - VTK path
                          varNumVrtx,    # Criterion 5 - Num vrtx to include
                          lgcPeRng,      # Criterion 6 - Yes or no?
                          varPeRngLw,    # Criterion 6 - Lower limit
                          varPeRngUp,    # Criterion 6 - Upper limit
                          lgcNormDiv,    # Normalisation - Yes or no?
                          varNormIdx,    # Normalisation - Which reference
                          varDpi,        # Plot - dots per inch
                          varYmin,       # Plot - Minimum of Y axis
                          varYmax,       # Plot - Maximum of Y axis
                          lstConLbl,     # Plot - Condition labels
                          strXlabel,     # Plot - X axis label
                          strYlabel,     # Plot - Y axis label
                          strTitle,      # Plot - Title
                          strPltOtPre,   # Plot - Output file path prefix
                          strPltOtSuf,   # Plot - Output file path suffix
                          queOut,        # Queue for output list
                          ):
    """
    Obtaining & plotting single subject data for across subject analysis.

    This function loads the data for each subject in an across subject analysis
    and passes the data to the parent function for visualisation.
    """
    # Only print status messages if this is the first of several parallel
    # processes:
    if idxPrc == 0:
        print('------Loading single subject data: ' + strSubId)

    # **************************************************************************
    # *** Import data

    # Import CSV file with ROI definition
    if idxPrc == 0:
        print('---------Importing CSV file with ROI definition (first '
              + 'criterion)')
    aryRoiVrtx = funcLoadCsvRoi(strCsvRoi, varNumHdrRoi)

    # Import second criterion vtk file (one depth level)
    if lgcSlct02:
        if idxPrc == 0:
            print('---------Importing second criterion vtk file (one depth '
                  + 'level).')
        vecSlct02 = funcLoadVtkSingle(strVtkSlct02, strPrcdData, varNumLne)
    # Otherwise, create dummy vector (for function I/O)
    else:
        vecSlct02 = 0

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

    # Import exclusion mask vtk file (fourth criterion, all depth levels)
    if lgcMskExcl:
        if idxPrc == 0:
            print('---------Importing exclusion mask vtk file (all depth '
                  + 'levels).')
        aryExcl = funcLoadVtkMulti(strVtkExcl,
                                   strPrcdData,
                                   varNumLne,
                                   varNumDpth)
    # Otherwise, create dummy array (for function I/O):
    else:
        aryExcl = 0

    # Import first set of data vtk files
    if idxPrc == 0:
        print('---------Importing first set of data vtk files.')
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

    # Import second set of data vtk files
    if lgcVtk02:
        if idxPrc == 0:
            print('---------Importing second set of data vtk files.')
        # Number of input files (i.e. number of conditions):
        varNumCon = len(lstVtkDpth02)
        # List for input data:
        lstDpthData02 = [None] * varNumCon
        # Loop through input data files:
        for idxIn in range(0, varNumCon):
            # Import data from file:
            lstDpthData02[idxIn] = funcLoadVtkMulti(lstVtkDpth02[idxIn],
                                                    strPrcdData,
                                                    varNumLne,
                                                    varNumDpth)
            if idxPrc == 0:
                print('------------File ' + str(idxIn + 1) + ' out of '
                      + str(varNumCon))
    # Otherwise, create dummy array (for function I/O):
    else:
        lstDpthData02 = 0
    # **************************************************************************

    # **************************************************************************
    # *** Select vertices

    lstDpthData01, varNumInc, varThrZcon, vecInc = \
        funcSlctVrtcs(varNumCon,      # Number of conditions
                      lstDpthData01,  # List with depth-sampled data I
                      aryRoiVrtx,     # Array with ROI definition (1st crit.)
                      lgcSlct02,      # Criterion 2 - Yes or no?
                      vecSlct02,      # Criterion 2 - Data
                      varThrSlct02,   # Criterion 2 - Threshold
                      lgcSlct03,      # Criterion 3 - Yes or no?
                      arySlct03,      # Criterion 3 - Data
                      varThrSlct03,   # Criterion 3 - Threshold
                      lgcMskExcl,     # Criterion 4 - Yes or no? (excl. mask)
                      aryExcl,        # Criterion 4 - Data (excl. mask)
                      varThrExcl,     # Criterion 4 - Threshold (excl. mask)
                      lgcVtk02,       # Criterion 5 - Yes or no?
                      lstDpthData02,  # Criterion 5 - Depth-sampled data II
                      varNumVrtx,     # Criterion 5 - Num vrtx to include
                      lgcPeRng,       # Criterion 6 - Yes or no?
                      varPeRngLw,     # Criterion 6 - Lower bound
                      varPeRngUp,     # Criterion 6 - Upper bound
                      idxPrc,         # Process ID (to manage status messages)
                      )
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
    # *** Plot results - mean over vertices

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
            # Place standard deviation in array:
            aryDpthConf[idxIn, idxDpth] = varTmp

            # Calculate standard error of the mean.
            varTmp = np.divide(np.std(aryTmp),
                               np.sqrt(aryTmp.size * 0.125))
            # Place standard deviation in array:
            aryDpthConf[idxIn, idxDpth] = varTmp

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

        # Divide all values by the grand mean:
        aryDpthMean = np.divide(aryDpthMean, varGrndMean)
        aryDpthConf = np.divide(aryDpthConf, varGrndMean)

    # Create figure:
    fgr01 = plt.figure(figsize=(800.0/varDpi, 600.0/varDpi),
                       dpi=varDpi)
    # Create axis:
    axs01 = fgr01.add_subplot(111)

    # Vector for x-data:
    vecX = range(0, varNumDpth)

    # Prepare colour map:
    objClrNorm = colors.Normalize(vmin=0, vmax=(varNumCon - 1))
    objCmap = plt.cm.winter

    # Loop through input files:
    for idxIn in range(0, varNumCon):

        # Adjust the colour of current line:
        # vecClrTmp = np.array([(float(idxIn) / float(varNumCon)),
        #                       (1.0 - (float(idxIn) / float(varNumCon))),
        #                       (float(idxIn) / float(varNumCon))
        #                       ])
        vecClrTmp = objCmap(objClrNorm(varNumCon - 1 - idxIn))

        # Plot depth profile for current input file:
        plt01 = axs01.plot(vecX,  #noqa
                           aryDpthMean[idxIn, :],
                           color=vecClrTmp,
                           alpha=0.9,
                           label=('Luminance contrast '
                                  + lstConLbl[idxIn]),
                           linewidth=8.0,
                           antialiased=True)

        # Plot error shading:
        plot02 = axs01.fill_between(vecX,  #noqa
                                    np.subtract(aryDpthMean[idxIn, :],
                                                aryDpthConf[idxIn, :]),
                                    np.add(aryDpthMean[idxIn, :],
                                           aryDpthConf[idxIn, :]),
                                    alpha=0.4,
                                    edgecolor=vecClrTmp,
                                    facecolor=vecClrTmp,
                                    linewidth=0,
                                    # linestyle='dashdot',
                                    antialiased=True)

    # Set x-axis range:
    axs01.set_xlim([-1, varNumDpth])
    # Set y-axis range:
    axs01.set_ylim([varYmin, varYmax])

    # Which x values to label with ticks (WM & CSF boundary):
    axs01.set_xticks([-0.5, (varNumDpth - 0.5)])
    # Labels for x ticks:
    axs01.set_xticklabels(['WM', 'CSF'])

    # Set x & y tick font size:
    axs01.tick_params(labelsize=13)

    # Adjust labels:
    axs01.set_xlabel(strXlabel,
                     fontsize=13)
    axs01.set_ylabel(strYlabel,
                     fontsize=13)

    # Adjust title:
    strTitle = (strSubId + ' ' + strTitle + ' - Number of vertices: ' +
                str(varNumInc) + ', z-conjunction-threshold = ' +
                str(np.around((varThrZcon), decimals=2)))
    axs01.set_title(strTitle, fontsize=10)

    # Legend for axis 1:
    axs01.legend(loc=0,
                 prop={'size': 13})

    # # Add vertical grid lines:
    #    axs01.xaxis.grid(which=u'major',
    #                     color=([0.5,0.5,0.5]),
    #                     linestyle='-',
    #                     linewidth=0.2)

    # File name for figure:
    strPltOt = strPltOtPre + strSubId + strPltOtSuf

    # Save figure:
    fgr01.savefig(strPltOt,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)

    # Close figure:
    plt.close(fgr01)
    # **************************************************************************

    # **************************************************************************
    # *** Return

    # Output list:
    lstOut = [idxPrc,
              aryDpthMean]

    queOut.put(lstOut)
    # **************************************************************************
