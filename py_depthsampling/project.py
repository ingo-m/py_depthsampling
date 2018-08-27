# -*- coding: utf-8 -*-
"""Project parameter estimates into a visual space representation."""

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


from py_depthsampling.project.project_main import project


# -----------------------------------------------------------------------------
# *** Define parameters

# Load/save existing projection from/to (ROI, condition, depth level label left
# open):
strPthNpy = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/project/{}_{}_{}.npy'  #noqa
# strPthNpy = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/project/{}_{}_{}_z.npy'  #noqa

# List of subject identifiers:
lstSubIds = ['20171023',  # '20171109',
             '20171204_01',
             '20171204_02',
             '20171211',
             '20171213',
             '20180111',
             '20180118']

# Nested list with depth levels to average over. For instance, if `lstDpth =
# [[0, 1, 2], [3, 4, 5]]`, on a first iteration, the average over the first
# three depth levels is calculated, and on a second iteration the average over
# the subsequent three depth levels is calculated. If 1lstDpth= [[None]]1,
# average over all depth levels.
lstDpth = [None, [0, 1, 2], [4, 5, 6], [8, 9, 10]]
# lstDpth = [[x] for x in range(11)]
# Depth level condition labels (output file will contain this label):
lstDpthLbl = ['allGM', 'deepGM', 'midGM', 'superficialGM']
# lstDpthLbl = [str(x) for x in range(11)]

# ROI ('v1' or 'v2'):
lstRoi = ['v1', 'v2', 'v3']

# Output path & prefix for plots (ROI, condition, depth level label left open):
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/project/pe/{}_{}_{}'  #noqa
# strPthPltOt = '/home/john/Dropbox/PacMan_Plots/project/z/{}_{}_{}'  #noqa
strPthPltOt = '/home/john/Desktop/tmp/project/{}_{}_{}'

# File type suffix for plot:
strFlTp = '.svg'
# strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Condition levels (used to complete file names):
# lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst',
#           'Pd_trn', 'Cd_trn', 'Ps_trn',
#           'Pd_min_Ps_sst', 'Pd_min_Cd_sst', 'Cd_min_Ps_sst',
#           'Pd_min_Cd_Ps_sst',
#           'Pd_min_Cd_Ps_trn',
#           'Pd_min_Ps_trn', 'Pd_min_Cd_trn', 'Cd_min_Ps_trn']
# lstCon = ['polar_angle', 'x_pos', 'y_pos', 'SD', 'R2']
lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst']
# lstCon = ['Pd_min_Ps_sst', 'Pd_min_Cd_sst', 'Cd_min_Ps_sst',
#           'Pd_min_Cd_Ps_sst']

# Path of vtk mesh with data to project into visual space (e.g. parameter
# estimates; subject ID, hemisphere, and contion level left open).
strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/feat_level_2_{}_cope.vtk'  #noqa
# strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/feat_level_2_{}_zstat.vtk'  #noqa
# strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_{}.vtk'  #noqa

# Path of mean EPI (for scaling to percent signal change; subject ID and
# hemisphere left open):
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

# Path of csv file with ROI definition (subject ID, hemisphere, and ROI left
# open).
strCsvRoi = '/home/john/PhD/GitLab/PacMan/analysis/{}/08_depthsampling/{}/{}_mod.csv'  #noqa
# strCsvRoi = '/Users/john/1_PhD/GitLab/PacMan/analysis/{}/08_depthsampling/{}/{}_mod.csv'  #noqa

# Number of cortical depths.
varNumDpth = 11

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

            project(lstRoi[idxRoi], lstCon[idxCon], lstDpth[idxDpth],
                    lstDpthLbl[idxDpth], strPthNpy, varNumSub,
                    lstSubIds, strPthData, strPthMneEpi, strPthR2, strPthX,
                    strPthY, strPthSd, strCsvRoi, varNumDpth, varThrR2,
                    varNumX, varNumY, varExtXmin, varExtXmax, varExtYmin,
                    varExtYmax, strPthPltOt, strFlTp)
# -----------------------------------------------------------------------------
