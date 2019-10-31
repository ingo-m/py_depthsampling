# -*- coding: utf-8 -*-
"""
Function of the depth sampling library.

THIS VERSION: texture / uniform background control experiment
              (sessions 20181128 & 20190207)

Function of the event-related timecourses depth sampling library.

Plot event-related timecourses sampled across cortical depth levels.

The input to this module are custom-made 'mesh time courses'. Timecourses have
to be cut into event-related segments and averaged across trials (using the
'cut_sgmnts.py' script of the depth-sampling library, or automatically as part
of the PacMan analysis pipeline, n_03x_py_evnt_rltd_avrgs.py). Depth-sampling
has to be performed with CBS tools, resulting in a 3D mesh for each time point.
Here, 3D meshes (with values for all depth-levels at one point in time, for one
condition) are combined across time and conditions to be plotted and analysed.
"""


from py_depthsampling.ert.ert_main_surface import ert_main


# *****************************************************************************
# *** Define parameters

# Load data from previously prepared pickle? If 'False', data is loaded from
# vtk meshes and saved as pickle.
lgcPic = False

# Meta-condition (within or outside of retinotopic stimulus area):
lstMtaCn = ['centre',
            'edge',
            'background']

# Region of interest ('v1' or 'v2'):
lstRoi = ['v1', 'v2']

# Hemispheres ('lh' or 'rh'):
lstHmsph = ['lh', 'rh']

# List of subject identifiers:
lstSubIds = ['20181128', '20190207']

# Name of pickle file from which to load time course data or save time course
# data to (metacondition and ROI left open):
strPthPic = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/txtr_uni_bckgr_cntrl_exp/era_{}_{}.pickle'  #noqa

# Condition levels (used to complete file names):
lstCon = ['bright_square_txtr',
          'bright_square_uni',
          'pacman_static_txtr',
          'pacman_static_uni']

# Condition labels (for plot legend):
lstConLbl = lstCon

# Base name of vertex inclusion masks (subject ID, hemisphere, subject ID,
# ROI, and metacondition left open):
strVtkMsk = '/media/sf_D_DRIVE/MRI_Data_PhD/09_surface/{}/cbs/{}/{}_vertex_inclusion_mask_{}_{}.vtk'  #noqa

# Base name of single-volume vtk meshes that together make up the timecourse
# (subject ID, hemisphere, stimulus level, and volume index left open):
strVtkPth = '/media/sf_D_DRIVE/MRI_Data_PhD/09_surface/{}/cbs/{}_era/{}/vol_{}.vtk'  #noqa

# Number of cortical depths:
varNumDpth = 11

# Number of timepoints:
varNumVol = 20

# Beginning of string which precedes vertex data in data vtk files (i.e. in the
# statistical maps):
strPrcdData = 'SCALARS'

# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Convert y-axis values to percent (i.e. divide label values by 100)?
lgcCnvPrct = True

# Label for axes:
strXlabel = 'Time [s]'
strYlabel = 'fMRI signal change [%]'

# Volume index of start of stimulus period (i.e. index of first volume during
# which stimulus was on - for the plot):
varStimStrt = 5
# Volume index of end of stimulus period (i.e. index of last volume during
# which stimulus was on - for the plot):
varStimEnd = 11
# Volume TR (in seconds, for the plot):
varTr = 2.079

# Plot legend - single subject plots:
lgcLgnd01 = True
# Plot legend - across subject plots:
lgcLgnd02 = True

# Output path for plots - prfix (metacondition and ROI left open):
strPltOtPre = '/home/john/Dropbox/PacMan_Plots/era_onset_texture_control_experiment/{}_{}_'
# Output path for plots - suffix:
strPltOtSuf = '_ert.svg'

# Figure scaling factor:
varDpi = 100.0
# *****************************************************************************


# *****************************************************************************
# *** Loop through ROIs / conditions

# Loop through ROIs, hemispheres, and conditions to create plots:
for idxMtaCn in range(len(lstMtaCn)):

    if 'centre' in lstMtaCn[idxMtaCn]:
        # Limits of y-axis:
        varAcrSubsYmin = -0.03
        varAcrSubsYmax = 0.02
        # Number of labels on y-axis:
        varYnum = 6
        # Padding around labelled values on y:
        tplPadY = (0.003, 0.002)

    if 'background'in lstMtaCn[idxMtaCn]:
        # Limits of y-axis:
        varAcrSubsYmin = -0.01
        varAcrSubsYmax = 0.01
        # Number of labels on y-axis:
        varYnum = 3
        # Padding around labelled values on y:
        tplPadY = (0.006, 0.006)

    if 'edge' in lstMtaCn[idxMtaCn]:
        # Limits of y-axis:
        varAcrSubsYmin = -0.0
        varAcrSubsYmax = 0.04
        # Number of labels on y-axis:
        varYnum = 5
        # Padding around labelled values on y:
        tplPadY = (0.006, 0.002)

    for idxRoi in range(len(lstRoi)):

        # Call main function:
        ert_main(lstSubIds, lstCon, lstConLbl, lstMtaCn[idxMtaCn],
                 lstHmsph, lstRoi[idxRoi], strVtkMsk, strVtkPth, varTr,
                 varNumDpth, varNumVol, varStimStrt, varStimEnd, strPthPic,
                 lgcPic, strPltOtPre, strPltOtSuf,
                 varAcrSubsYmin=varAcrSubsYmin,
                 varAcrSubsYmax=varAcrSubsYmax, tplPadY=tplPadY,
                 varYnum=varYnum, strXlabel=strXlabel, strYlabel=strYlabel)
# *****************************************************************************