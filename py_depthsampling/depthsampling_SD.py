# -*- coding: utf-8 -*-
"""
VTK depth samling across subjects.

Visualise cortical depth sampling results from vtk files. Vertices are selected
according to several criteria:

    (1) Selection criterion 1 - the vertex has to be contained within the ROI
        (as defined by by a csv file containing the indices of the ROI
        vertices; this csv file can be created with paraview based on a
        retinotopic map).
    (2) Selection criterion 2 -  vertices that are BELOW threshold at any depth
        levels are excluded. (For example, a venogram, or a T2* weighted EPI
        image with low intensities around veins that is defined at all depth
        level can be used.)
    (3) Selection criterion 3 - same as (2). Vertices that are BELOW threshold
        at any depth level are excluded.
    (4) Selection criterion 4 - Vertices that are WITHIN INTERVAL are included
        (one depth level, e.g. retinotopic eccentricity).
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


from py_depthsampling.main.main import ds_main


# *****************************************************************************
# *** Define parameters

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['periphery']

# Region of interest ('v1' or 'v2'):
lstRoi = ['v1', 'v2', 'v3']

# Hemispheres ('lh' or 'rh'):
lstHmsph = ['rh'] #, 'lh']

# List of subject identifiers:
lstSubIds = ['20171023',  # '20171109',
             '20171204_01',
             '20171204_02',
             '20171211',
             '20171213',
             '20180111',
             '20180118']

# Condition levels (used to complete file names) - nested list:
lstNstCon = [['SD']]

# Condition labels:
lstNstConLbl = lstNstCon

# Base path of vtk files with depth-sampled data, e.g. parameter estimates
# (with subject ID, hemisphere, and stimulus level left open):
# strVtkDpth01 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/feat_level_2_{}_zstat.vtk'  #noqa
strVtkDpth01 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_{}.vtk'  #noqa

# (1)
# Restrict vertex selection to region of interest (ROI)?
lgcSlct01 = True
# Base path of csv files with ROI definition (i.e. patch of cortex selected on
# the surface, e.g. V1 or V2) - i.e. the first vertex selection criterion (with
# subject ID, hemisphere, and ROI left open):
# NOTE: The '_mod' subscript indicates that the csv files have been processed
# by the funtion `py_depthsampling.misc.fix_roi_csv.fix_roi_csv` in order to
# ensure that the indices of the ROI definition and the vtk meshes are
# congruent.
strCsvRoi = '/home/john/PhD/GitLab/PacMan/analysis/{}/08_depthsampling/{}/{}_mod.csv'  #noqa
# Number of header lines in ROI CSV file:
varNumHdrRoi = 1

# (2)
# Use vertex selection criterion 2 (vertices that are BELOW threshold are
# excluded - median across depth levels):
lgcSlct02 = True
# Path of vtk files with for vertex selection criterion. This vtk file is
# supposed to contain one set of data values for each depth level. (With
# subject ID and hemisphere left open.)
strVtkSlct02 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_R2.vtk'  #noqa
# Threshold for vertex selection:
varThrSlct02 = 0.15

# (3)
# Use vertex selection criterion 3 (vertices that are BELOW threshold are
# excluded - minimum across depth levels):
lgcSlct03 = True
# Path of vtk files with for vertex selection criterion. This vtk file is
# supposed to contain one set of data values for each depth level. (With
# subject ID and hemisphere left open.)
strVtkSlct03 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/combined_mean.vtk'  #noqa
# Threshold for vertex selection:
varThrSlct03 = 7000.0

# (4)
# Use vertex selection criterion 4 (vertices that are WITHIN INTERVAL are
# included - median across depth levels):
lgcSlct04 = True
# Path of vtk files with for vertex selection criterion. This vtk file is
# supposed to contain one set of data values for each depth level. (With
# subject ID and hemisphere left open.)
strVtkSlct04 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_eccentricity.vtk'  #noqa
# Threshold for vertex selection - list of tuples (interval per meta-condition,
# e.g. within & outside stimulus area):
lstThrSlct04 = [(3.5, 4.0)]

# Number of cortical depths:
varNumDpth = 11

# Beginning of string which precedes vertex data in data vtk files (i.e. in the
# statistical maps):
strPrcdData = 'SCALARS'

# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Label for axes:
strXlabel = 'Cortical depth level'
# strYlabel = 'z-value'
strYlabel = 'SD [deg]'

# Output path for plots - prefix:
# strPltOtPre = '/home/john/Dropbox/PacMan_Plots/z/{}/plots_{}/'
strPltOtPre = '/home/john/Dropbox/PacMan_Plots/SD/SD_{}_{}_'

# Output path for plots - suffix:
strPltOtSuf = '_{}_{}_{}.png'

# Figure scaling factor:
varDpi = 100.0

# If normalisation - data from which input file to divide by?
# (Indexing starts at zero.) Note: This functionality is not used at the
# moment. Instead of dividing by a reference condition, all profiles are
# divided by the grand mean within subjects before averaging across subjects
# (if lgcNormDiv is true).
varNormIdx = 0

# Normalise by division?
lgcNormDiv = False

# Output path for depth samling results (within subject means):
# strDpthMeans = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}_zstat.npz'  #noqa
strDpthMeans = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}.npz'  #noqa

# Maximum number of processes to run in parallel: *** NOT IMPLEMENTED
# varPar = 10
# *****************************************************************************


# *****************************************************************************
# *** Loop through ROIs / conditions

# Loop through ROIs, hemispheres, and conditions to create plots:
for idxMtaCn in range(len(lstMetaCon)):  #noqa
    for idxRoi in range(len(lstRoi)):
        for idxHmsph in range(len(lstHmsph)):
            for idxCon in range(len(lstNstCon)):

                # Limits of axes need to be adjusted based on ROI, condition,
                # hemisphere.

                # Limits of y-axis for SINGLE SUBJECT PLOTS (list of tuples,
                # [(Ymin, Ymax)]):

                if idxRoi == 0:  # v1
                    if (idxCon == 0) or (idxCon == 4):  # Simple contrasts
                        lstLimY = [(-4.0, 2.0)] * len(lstSubIds)
                    else:  # Differential contrasts
                        lstLimY = [(-1.0, 1.0)] * len(lstSubIds)

                elif (idxRoi == 1) or (idxRoi == 2):  # v2 & v3
                    if (idxCon == 0) or (idxCon == 4):  # Simple contrasts
                        lstLimY = [(-4.0, 2.0)] * len(lstSubIds)
                    else:  # Differential contrasts
                        lstLimY = [(-1.0, 1.0)] * len(lstSubIds)

                # Limits of y-axis for ACROSS SUBJECT PLOTS:

                # Stimulus:
                if lstMetaCon[idxMtaCn] == 'stimulus':

                    if (idxCon == 0) or (idxCon == 4):  # Simple contrasts
                        # Limits of y-axis for across subject plot:
                        varAcrSubsYmin = -5.0
                        varAcrSubsYmax = 1.0
                    else:  # Differential contrasts
                        # Limits of y-axis for across subject plot:
                        varAcrSubsYmin = -1.0
                        varAcrSubsYmax = 1.0

                # Periphery:
                if lstMetaCon[idxMtaCn] == 'periphery':

                    if (idxCon == 0) or (idxCon == 4):  # Simple contrasts
                        # Limits of y-axis for across subject plot:
                        varAcrSubsYmin = 0.0
                        varAcrSubsYmax = 2.0
                    else:  # Differential contrasts
                        # Limits of y-axis for across subject plot:
                        varAcrSubsYmin = -1.0
                        varAcrSubsYmax = 1.0

                # Title for mean plot:
                strTitle = lstRoi[idxRoi].upper()

                # Call main function:
                ds_main(lstRoi[idxRoi], lstHmsph[idxHmsph], lstSubIds,
                        lstNstCon[idxCon], lstNstConLbl[idxCon], strVtkDpth01,
                        lgcSlct01, strCsvRoi, varNumHdrRoi, lgcSlct02,
                        strVtkSlct02, varThrSlct02, lgcSlct03, strVtkSlct03,
                        varThrSlct03, lgcSlct04, strVtkSlct04,
                        lstThrSlct04[idxMtaCn], varNumDpth, strPrcdData,
                        varNumLne, strTitle, lstLimY, varAcrSubsYmin,
                        varAcrSubsYmax, strXlabel, strYlabel,
                        strPltOtPre.format(lstMetaCon[idxMtaCn],
                        lstRoi[idxRoi]), strPltOtSuf.format(
                        lstHmsph[idxHmsph], lstRoi[idxRoi],
                        lstNstCon[idxCon][0]), varDpi, varNormIdx, lgcNormDiv,
                        strDpthMeans.format(lstMetaCon[idxMtaCn],
                        lstRoi[idxRoi], lstHmsph[idxHmsph], '{}'),
                        strMetaCon=lstMetaCon[idxMtaCn])
# *****************************************************************************
