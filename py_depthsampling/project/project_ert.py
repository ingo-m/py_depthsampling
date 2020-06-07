# -*- coding: utf-8 -*-
"""
Project event-related time courses into visual field.

Function of the depth sampling library.

Timecourses have to be cut into event-related segments and averaged across
trials (using the 'cut_sgmnts.py' script of the depth-sampling library, or
automatically as part of the PacMan analysis pipeline,
n_03x_py_evnt_rltd_avrgs.py). Depth-sampling has to be performed with CBS
tools, resulting in a 3D mesh for each time point.
"""


from py_depthsampling.project.project_main import project


# *****************************************************************************
# *** Define parameters

# Load/save existing projection from/to (ROI, condition, depth level label, and
# volume index left open):
strPthNpy = '/media/ssd_dropbox/Dropbox/University/PhD/PacMan_Depth_Data/Higher_Level_Analysis/project_ert/{}_{}_{}_volume_{}.npy'  #noqa

# Output path & prefix for plots (ROI, condition, depth level label left open):
strPthPltOt = '/media/ssd_dropbox/Dropbox/University/PhD/PacMan_Plots/project/pe_for_elife_production/{}_{}_{}'  #noqa

# Region of interest ('v1' or 'v2'):
lstRoi = ['v1']

# List of subject identifiers:
lstSubIds = ['20171023',  # '20171109',
             '20171204_01',
             '20171204_02',
             '20171211',
             '20171213',
             '20180111',
             '20180118']

# Condition levels (used to complete file names):
lstCon = ['pacman_static']

# Condition labels (for plot legend):
lstConLbl = ['Pacman static']

# Number of cortical depths:
varNumDpth = 11

# Number of timepoints:
varNumVol = 19

# Beginning of string which precedes vertex data in data vtk files (i.e. in the
# statistical maps):
strPrcdData = 'SCALARS'

# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Limits of y-axis:
varAcrSubsYmin = -0.04
varAcrSubsYmax = 0.02

# Convert y-axis values to percent (i.e. divide label values by 100)?
# lgcCnvPrct = False

# Nested list with depth levels to average over. For instance, if `lstDpth =
# [[0, 1, 2], [3, 4, 5]]`, on a first iteration, the average over the first
# three depth levels is calculated, and on a second iteration the average over
# the subsequent three depth levels is calculated. If `lstDpth= [[None]]`,
# average over all depth levels.
lstDpth = [None]
# Depth level condition labels (output file will contain this label):
lstDpthLbl = ['allGM']

# File type suffix for plot:
strFlTp = '.svg'
# strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Path of csv file with ROI definition (subject ID, hemisphere, and ROI left
# open).
strCsvRoi = '/home/john/PhD/GitLab/PacMan/analysis/{}/08_depthsampling/{}/{}_mod.csv'  #noqa

# Path of vtk mesh with data to project into visual space (e.g. parameter
# estimates; subject ID, hemisphere, condition, and condition level left open).
strPthData =  '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}_era/{}/aryErt_{}.npy'  #noqa

# Path of mean EPI (for scaling to percent signal change; subject ID and
# hemisphere left open). Not used here, but necessary because of modularity
# (multiprocessing function does not take kwargs).
strPthMneEpi = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/combined_mean.vtk'  #noqa

# Path of vtk mesh with R2 values from pRF mapping (at multiple depth levels;
# subject ID and hemisphere left open).
strPthR2 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_R2.vtk'  #noqa

# Path of vtk mesh with pRF sizes (at multiple depth levels; subject ID and
# hemisphere left open).
strPthSd = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_SD.vtk'  #noqa

# Path of vtk mesh with pRF x positions at multiple depth levels; subject ID
# and hemisphere left open).
strPthX = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_x_pos.vtk'  #noqa

# Path of vtk mesh with pRF y positions at multiple depth levels; subject ID
# and hemisphere left open).
strPthY = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_y_pos.vtk'  #noqa

# Extent of visual space from centre of the screen in negative x-direction
# (i.e. from the fixation point to the left end of the screen) in degrees of
# visual angle.
varExtXmin = -5.19
# Extent of visual space from centre of the screen in positive x-direction
# (i.e. from the fixation point to the right end of the screen) in degrees of
# visual angle.
varExtXmax = 5.19
# Extent of visual space from centre of the screen in negative y-direction
# (i.e. from the fixation point to the lower end of the screen) in degrees of
# visual angle.
varExtYmin = -5.19
# Extent of visual space from centre of the screen in positive y-direction
# (i.e. from the fixation point to the upper end of the screen) in degrees of
# visual angle.
varExtYmax = 5.19

# R2 threshold for vertex inclusion (vertices with R2 value below threshold are
# not considered for plot):
varThrR2 = 0.15

# Number of bins for visual space representation in x- and y-direction (ratio
# of number of x and y bins should correspond to ratio of size of visual space
# in x- and y-directions).
varNumX = 200
varNumY = 200

# Minimum and maximum of colour bar:
varMin = -0.02
varMax = 0.02
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Parent loop

print('-Project parametric map into visual space')

# Number of subjects:
varNumSub = len(lstSubIds)

# Loop through depth levels, ROIs, and conditions:
for idxDpth in range(len(lstDpth)):  #noqa
    for idxRoi in range(len(lstRoi)):
        for idxCon in range(len(lstCon)):
            for idxVol in range(varNumVol):

                project(lstRoi[idxRoi], lstCon[idxCon], lstDpth[idxDpth],
                        lstDpthLbl[idxDpth], strPthNpy.format('{}', '{}', '{}',
                        idxVol), varNumSub, lstSubIds, strPthData,
                        strPthMneEpi, strPthR2, strPthX, strPthY, strPthSd,
                        strCsvRoi, varNumDpth, varThrR2, varNumX, varNumY,
                        varExtXmin, varExtXmax, varExtYmin, varExtYmax,
                        strPthPltOt, strFlTp, varMin=varMin, varMax=varMax,
                        varTr=idxVol)
# -----------------------------------------------------------------------------
