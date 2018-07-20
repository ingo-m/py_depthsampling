# -*- coding: utf-8 -*-
"""
Function of the depth sampling library.

Function of the event-related timecourses depth sampling library.

Plot event-related timecourses sampled across cortical depth levels.

NOTE: This version is used to plot event related timecourses for an additional
experimental condition with long stimulus blocks (in a subset of subjects).

The input to this module are custom-made 'mesh time courses'. Timecourses have
to be cut into event-related segments and averaged across trials (using the
'cut_sgmnts.py' script of the depth-sampling library, or automatically as part
of the PacMan analysis pipeline, n_03x_py_evnt_rltd_avrgs.py). Depth-sampling
has to be performed with CBS tools, resulting in a 3D mesh for each time point.
Here, 3D meshes (with values for all depth-levels at one point in time, for one
condition) are combined across time and conditions to be plotted and analysed.
"""


from py_depthsampling.ert.ert_main_PacMan import ert_main


# *****************************************************************************
# *** Define parameters

# Load data from previously prepared pickle? If 'False', data is loaded from
# vtk meshes and saved as pickle.
lgcPic = False

# Meta-condition (within or outside of retinotopic stimulus area):
lstMtaCn = ['stimulus', 'periphery']

# Region of interest ('v1' or 'v2'):
lstRoi = ['v1', 'v2']

# Hemispheres ('lh' or 'rh'):
lstHmsph = ['lh', 'rh']

# List of subject identifiers:
lstSubIds = ['20171211',
             '20171213',
             '20180111',
             '20180118']

# Name of pickle file from which to load time course data or save time course
# data to (metacondition, ROI, and hemisphere left open):
strPthPic = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/era_long_{}_{}.pickle'  #noqa

# Condition levels (used to complete file names):
lstCon = ['pacman_dynamic_long']

# Condition labels (for plot legend):
lstConLbl = ['Pacman dynamic long']

# Base name of vertex inclusion masks (subject ID, hemisphere, subject ID,
# ROI, and metacondition left open):
strVtkMsk = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/{}_vertex_inclusion_mask_{}_mod_{}.vtk'  #noqa

# Base name of single-volume vtk meshes that together make up the timecourse
# (subject ID, hemisphere, stimulus level, and volume index left open):
strVtkPth = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}_era/{}/vol_{}.vtk'  #noqa

# Number of cortical depths:
varNumDpth = 11

# Number of timepoints:
varNumVol = 400

# Beginning of string which precedes vertex data in data vtk files (i.e. in the
# statistical maps):
strPrcdData = 'SCALARS'

# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Limits of y-axis:
varAcrSubsYmin = -0.03
varAcrSubsYmax = 0.02

# Convert y-axis values to percent (i.e. divide label values by 100)?
lgcCnvPrct = True

# Number of labels on the y axis:
varYnum = 6

# Which x-values to label on the axis (e.g., if `varXlbl = 2`, every second
# x-value is labelled).
varXlbl = 10

# Volume TR (in seconds, for the plot):
varTr = 2.079

# Time scaling factor (factor by which timecourse was temporally upsampled; if
# it was not upsampled, varTmeScl = 1.0):
varTmeScl = 10.0

# Stimulus onset in seconds (for the plot), in volumes (will be converted to
# seconds later):
varStimStrt = 5.0  # 5 volumes prestimulus interval in ERT

# Stimulus offset in seconds (for the plot), in volumes (will be converted to
# seconds later). (25 s stimulus plus prestimulus interval.)
varStimEnd = varStimStrt + (25.0 / varTr)

# Plot legend - single subject plots:
lgcLgnd01 = True
# Plot legend - across subject plots:
lgcLgnd02 = True

# Label for axes:
strXlabel = 'Time [s]'
strYlabel = 'fMRI signal change [%]'

# Output path for plots - prfix (metacondition, ROI, and hemisphere left open):
strPltOtPre = '/home/john/Dropbox/PacMan_Plots/era_long/{}/{}_{}/'
# Output path for plots - suffix:
strPltOtSuf = '_ert_long.svg'

# Figure scaling factor:
varDpi = 70.0
# *****************************************************************************


# *****************************************************************************
# *** Loop through ROIs / conditions

# Loop through ROIs, hemispheres, and conditions to create plots:
for idxMtaCn in range(len(lstMtaCn)):
    for idxRoi in range(len(lstRoi)):
        for idxHmsph in range(len(lstHmsph)):

                # Call main function:
                ert_main(lstSubIds, lstCon, lstConLbl, lstMtaCn[idxMtaCn],
                         lstHmsph[idxHmsph], lstRoi[idxRoi], strVtkMsk,
                         strVtkPth, varTr, varNumDpth, varNumVol, varStimStrt,
                         varStimEnd, strPthPic, lgcPic, strPltOtPre,
                         strPltOtSuf, strXlabel=strXlabel, strYlabel=strYlabel,
                         varAcrSubsYmin=varAcrSubsYmin,
                         varAcrSubsYmax=varAcrSubsYmax, tplPadY=(0.005, 0.005),
                         lgcCnvPrct=lgcCnvPrct, lgcLgnd01=lgcLgnd01,
                         lgcLgnd02=lgcLgnd02, varTmeScl=varTmeScl,
                         varXlbl=varXlbl, varYnum=varYnum, varDpi=varDpi)
# *****************************************************************************
