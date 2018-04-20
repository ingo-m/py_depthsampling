"""
Create event related segments from 4D nii files.

The purpose of this script is to cut out trial segments and to create average
time courses from 4D nii files. The inputs to this script are a list of 4D
nii files, and a corresponding list of design matrices including 3 columns,
describing the occurrence of the condition of interest in the nii files.

All nii files need to have the same image dimensions, and in order for the
averaging to be sensible, all condition blocks need to be of the same length.

There are three options for normalisation (de-meaning):
    (1) No normalisation.
    (2) Segments are normalised trial-by-trial; i.e. each time-course
        segment is divided by its own pre-stimulus baseline before averaging
        across trials.
    (3) The event-related average is normalised; i.e. segments are first
        averaged, and subsequently the mean timecourse is divided by the mean
        pre-stimulus baseline.

Note: This script assumes that there is an equal number of stimulus-condition
blocks across runs.

@author: Tristar / Ingo Marquardt & Marian Schneider
"""

# %% Import modules
import os
import time
import numpy as np
import nibabel as nb

# %% Check time
varTme01 = time.clock()

# %% Define functions


def fncLoadNii(strPathIn):
    """Function for loading nii files."""
    # print(('------Loading: ' + strPathIn))
    # Load nii file (this doesn't load the data into memory yet):
    niiTmp = nb.load(strPathIn)
    # Load data into array:
    aryTmp = niiTmp.get_data()
    # Get headers:
    hdrTmp = niiTmp.header
    # Get 'affine':
    aryAff = niiTmp.affine
    # Output nii data as numpy array and header:
    return aryTmp, hdrTmp, aryAff


# %% Define parameters

# Subject IDs & run IDs for each subject (for some subjects, not all runs can
# be included):
dicSubId = {'20150930': ['01', '02', '03', '04', '05', '06'],
            '20151118': ['01', '03', '04', '05', '06'],
            '20151127_01': ['01', '02', '03', '04', '05'],
            '20151130_02': ['02', '03', '04', '05', '06'],
            '20161205': ['02', '03', '04', '05', '06'],
            '20161207': ['01', '02', '03', '04', '05', '06'],
            '20161212_02': ['02', '03', '04', '05', '06'],
            '20161214': ['02', '03', '04'],
            '20161219_01': ['01', '02', '03', '04', '05', '06'],
            '20161219_02': ['02', '03', '04', '05', '06']
            }

# Parent directory:
strPthPrnt = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/{}/nii_distcor/feat_level_1/'  #noqa

# Path of 4D nii files (location within parent directory):
strInNii = 'func_{}.feat/filtered_func_data.nii.gz'

# Directory containing design matrices (EV files):
strPthEV = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/FSL_MRI_Metadata/version_01/run_{}_txt_eventmatrix.txt'  #noqa

# Output directory:
strPthOt = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/{}/nii_distcor/func_regAcrssRuns_cube_averages/'  #noqa

# Volume TR of input nii files:
varTR = 2.940

# Number of stimulus levels
varNumStimLvls = 4

# Number of volumes that will be included in the average segment before the
# onset of the condition block. NOTE: We get the start time and the duration
# of each block from the design matrices (EV files). In order for the averaging
# to work (and in order for it to be sensible in a conceptual sense), all
# blocks that are included in the average need to have the same duration.
varVolsPre = 3

# Number of volumes that will be included in the average segment after the
# end of the condition block:
varVolsPst = 7

# Options for de-meaning (see abvoe for more detailed information). Options
# are:
#   1 = No normalisation.
#   2 = Segments are normalised trial-by-trial (before averaging).
#   3 = The event-related average is normalised (after averaging).
varDemean = 2

# Which time points to use for the de-meaning, relative to the stimulus
# condition onset. (I.e., if you specify -3 and 0, the three volumes
# preceeding the onset of the stimulus are used - the interval is non-inclusive
# at the end.)
tplBase = (-3, 0)

# Whether or not to also produces individual event-related segments for each
# trial (in addition to average across all trials):
lgcSegs = False


# %% Loop through subjects:

for strSubId in dicSubId.keys():  #noqa

    # %% Preparations

    print('-Processing subject: ' + strSubId)

    # Input & output directories, formatted for this subject:
    strPthPrntTmp = strPthPrnt.format(strSubId)
    strPthOtTmp = strPthOt.format(strSubId)

    # Number of input 4D nii files:
    varNumRuns = len(dicSubId[strSubId])

    # Load first nii file in order to get image dimensions (headers, and
    # therefore image dimensions, are assummed to be identical across runs):
    _, hdr01, aryAff01 = fncLoadNii((strPthPrnt.format(strSubId)
                                     + strInNii.format(dicSubId[strSubId][0])))

    # Image dimensions:
    aryDim = np.copy(hdr01['dim'])

    # Check whether directory for segments of each trial already exists, if not
    # create it:
    if lgcSegs:
        # Target directory for segments:
        strPathSegs = (strPthOtTmp + 'segs')
        # Check whether target directory for segments exists:
        lgcDir = os.path.isdir(strPathSegs)
        # If directory does exist, delete it:
        if not(lgcDir):
            # Create direcotry for segments:
            os.mkdir(strPathSegs)
        # print('---Trial segments will be saved at: ' + strPathSegs)

    # Convert number of pre-stimulus volumes to integer, just in case the user
    # entered a float:
    varVolsPre = int(np.around(varVolsPre, 0))

    # %% Load design matrices (EV files)

    print('---Load design matrices (EV files)')

    # Empty list that will be filled with EV data:
    lstEV = []

    for idxRun in range(0, varNumRuns):
        # print('---Loading: ' + lstIn02[idxRun])
        # Read text file:
        aryTmp = np.loadtxt(strPthEV.format(dicSubId[strSubId][idxRun]),
                            comments='#',
                            delimiter=' ',
                            skiprows=0,
                            usecols=(0, 1, 2)
                            )
        # Append current csv object to list:
        lstEV.append(aryTmp)

    # %% Create event-related segments

    print('---Create event-related segments')

    # Duration of the first stimulus block (all stimulus blocks are assumed to
    # be of the same duration) [seconds]:
    varDurStim = lstEV[0][lstEV[0][:, 0] == 3, 1:][0, 1]

    # Calculate length of segments to be created during the averaging (pre-
    # stimulus  interval + stimulus interval + post-stimulus interval), in
    # number of volumes:
    varSegDur = int(np.around((float(varVolsPre) +
                               (varDurStim / float(varTR)) +
                               float(varVolsPst)
                               )))

    # Number of stimulus-condition blocks per run (assumed to be equal across
    # runs):
    varNumBlck = len(lstEV[0][lstEV[0][:, 0] == 3, 1:][:, 0])

    # Array that will be filled with event-related segments:
    aryTrials = np.empty((aryDim[1],
                          aryDim[2],
                          aryDim[3],
                          varNumStimLvls,
                          varNumRuns,
                          varNumBlck,
                          varSegDur,
                          ), dtype='float32')

    # Loop through runs:
    for idxRun in np.arange(varNumRuns):

        strInNiiTmp = strInNii.format(dicSubId[strSubId][idxRun])

        print('------Processing run: ' + strInNiiTmp)

        # Load nii data

        # Load input 4D nii files:
        aryTmpRun, _, _ = fncLoadNii((strPthPrntTmp + strInNiiTmp))

        # Loop through stimulus levels (conditions):
        for idxCon in np.arange(varNumStimLvls):

            print('---------Processing stim level: ' + str((idxCon + 1)))

            # Retrieve start times & duration of the current condition (add 3
            # here because relevant stim levels are coded as 3,4,5,6):
            aryFsl = lstEV[idxRun][lstEV[idxRun][:, 0] == idxCon + 3, 1:]

            # Loop through blocks:
            for idxBlck in np.arange(varNumBlck):

                # Start time of current block (in seconds, as in EV file):
                varTmpStr = (aryFsl[idxBlck, 0] / varTR)

                # Stop time of current condition:
                varTmpEnd = varTmpStr + (aryFsl[idxBlck, 1] / varTR)
                # Add post-condition interval to stop time:
                varTmpEnd = varTmpEnd + varVolsPst

                # We need to remove rounding error from the start and stop
                # indices, and convert them to integers:
                varTmpStr = int(np.around(varTmpStr, 0))
                varTmpEnd = int(np.around(varTmpEnd, 0))
                # print('---------------varTmpStr:' + str(varTmpStr))
                # print('---------------varTmpEnd:' + str(varTmpEnd))

                # Cut segment pertaining to current run and block:
                aryTmpBlck = np.copy(aryTmpRun[:,
                                               :,
                                               :,
                                               (varTmpStr - varVolsPre):varTmpEnd])

    #            # Trial-by-trial normalisation:
    #            if varDemean == 2:
    #                # Get prestimulus baseline:
    #                aryTmpBse = np.copy(aryTmpRun[:,
    #                                              :,
    #                                              :,
    #                                              (varTmpStr + tplBase[0]):
    #                                              (varTmpStr + tplBase[1])])
    #                # Mean for each voxel over time (i.e. over the pre-stimulus
    #                # baseline):
    #                aryTmpBseMne = np.mean(aryTmpBse, axis=3)
    #                # Get indicies of voxels that have a non-zero prestimulus
    #                # baseline:
    #                aryTmpNonZero = np.not_equal(aryTmpBseMne, 0.0)
    #                # Divide all voxels that are non-zero in the pre-stimulus
    #                # baseline by the prestimulus baseline:
    #                aryTmpBlck[aryTmpNonZero] = \
    #                    np.divide(aryTmpBlck[aryTmpNonZero],
    #                              aryTmpBseMne[aryTmpNonZero, None])

                # Assign current stimulus segment to array:
                aryTrials[:, :, :, idxCon, idxRun, idxBlck, :] = aryTmpBlck

    # %% Trial-by-trial normalisation:

    if varDemean == 2:

        # Get prestimulus baseline:
        aryBse = np.copy(aryTrials[:,
                                   :,
                                   :,
                                   :,
                                   :,
                                   :,
                                   (varVolsPre + tplBase[0]):
                                   (varVolsPre + tplBase[1])])

        # Mean for each voxel over time (i.e. over the pre-stimulus baseline):
        aryBseMne = np.mean(aryBse, axis=6)
        # Get indicies of voxels that have a non-zero prestimulus baseline:
        aryNonZero = np.not_equal(aryBseMne, 0.0)
        # Divide all voxels that are non-zero in the pre-stimulus baseline by
        # the prestimulus baseline:
        aryTrials[aryNonZero] = np.divide(aryTrials[aryNonZero],
                                          aryBseMne[aryNonZero, None])

    # %% Average over trials

    # Get dimensions:
    tplShpe = aryTrials.shape

    # Combine the dimensions corresponding to runs and blocks:
    aryTrials = np.reshape(aryTrials,
                           (tplShpe[0],
                            tplShpe[1],
                            tplShpe[2],
                            tplShpe[3],
                            (tplShpe[4] * tplShpe[5]),
                            tplShpe[6]))

    # Average over run- & block-dimension (create event-related average):
    aryEvntRltAvrg = np.mean(aryTrials, axis=4)

    # %% Normalisation on event-related averages:

    if varDemean == 3:

        # Get prestimulus baseline:
        aryBse = np.copy(aryEvntRltAvrg[:,
                                        :,
                                        :,
                                        :,
                                        (varVolsPre + tplBase[0]):
                                        (varVolsPre + tplBase[1])])

        # Mean for each voxel over time (i.e. over the pre-stimulus baseline):
        aryBseMne = np.mean(aryBse, axis=4)
        # Get indicies of voxels that have a non-zero prestimulus baseline:
        aryNonZero = np.not_equal(aryBseMne, 0.0)
        # Divide all voxels that are non-zero in the pre-stimulus baseline by
        # the prestimulus baseline:
        aryEvntRltAvrg[aryNonZero] = np.divide(aryEvntRltAvrg[aryNonZero],
                                               aryBseMne[aryNonZero, None])

    # %% Save segments npy

    if False:

        print('---Saving segments as npy file')

        # Adjust output filename:
        if varDemean == 1:
            strTmp = (strPthOtTmp + 'era')
        elif varDemean == 2:
            strTmp = (strPthOtTmp + 'era_norm')
        elif varDemean == 3:
            strTmp = (strPthOtTmp + 'era_norm')

        # Save array as npy file:
        np.save(strTmp, aryTrials)

    # %% Save event-related averages as nii

    print('---Save event-related averages as nii')

    # Loop through stimulus levels (conditions):
    for idxCon in np.arange(varNumStimLvls):

        print('---------Processing stim level: ' + str((idxCon + 1)))

        # Adjust output filename:
        if varDemean == 1:
            strTmp = (strPthOtTmp + 'er_avrg_stim_lvl_0' +
                      str((idxCon + 1)) + '.nii')
        elif varDemean == 2:
            strTmp = (strPthOtTmp + 'er_avrg_demeaned_stim_lvl_0' +
                      str((idxCon + 1)) + '.nii')
        elif varDemean == 3:
            strTmp = (strPthOtTmp + 'er_avrg_demeaned_stim_lvl_0' +
                      str((idxCon + 1)) + '.nii')

        # Create nii object for results:
        niiOut = nb.Nifti1Image(aryEvntRltAvrg[:, :, :, idxCon, :],
                                aryAff01,
                                header=hdr01)
        # Save nii:
        nb.save(niiOut, strTmp)

# %% Check & report time

varTme02 = time.clock()
varTme03 = varTme02 - varTme01
print('-Elapsed time: ' + str(varTme03) + ' s')
print('-Done.')
