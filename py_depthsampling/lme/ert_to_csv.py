# -*- coding: utf-8 -*-
"""
Save event-related timecourses to csv file.
"""


import pickle
import numpy as np
import pandas as pd
# import seaborn as sns


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of npz files with cortical depth profiles (metacondition and ROI left
# open):
strPckl = '/Users/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/{}/era_{}.pickle'  #noqa

# List of ROIs:
lstRoi = ['v1', 'v2']

# List of metaconditions:
lstMta = ['centre', 'edge', 'background']

# List of conditions:
lstCon = ['kanizsa',
          'kanizsa_rotated',
          'bright_square']

# Output path for data csv file (metacondition left open):
strCsv = '/Users/john/1_PhD/GitLab/py_depthsampling/py_depthsampling/lme/ert_{}.csv'  #noqa

# Crop ERT (start & end time point), do not crop if 'None'.
tplCrp = (5, 16)


# -----------------------------------------------------------------------------
# *** Preparations

# Number of metacondition:
varNumMta = len(lstMta)

# Number of conditions:
#varNumCon = len(lstCon)

# Number of ROIs:
varNumRoi = len(lstRoi)

# List of features for dataframe:
lstFtr = ['ROI', 'Condition', 'Subject', 'Volume', 'Vertices', 'PSC']


# -----------------------------------------------------------------------------
# *** Load data from disk

# Fill dataframe with cortical depth profiles:
for idxMta in lstMta:
    for idxRoi in lstRoi:

        # *********************************************************************
        # *** Load data from disk

        # Path of current input pickle file:
        strPcklTmp = strPckl.format(idxMta, idxRoi)

        # Load previously prepared event-related timecourses from pickle:
        dicAllSubsRoiErt = pickle.load(open(strPcklTmp, 'rb'))

        # Get number of subjects, conditions, cortical depth levels, time
        # points (volumes):
        varNumSub = len(dicAllSubsRoiErt)

        tplShpe = list(dicAllSubsRoiErt.values())[0][0].shape
        varNumCon = tplShpe[0]
        varNumDpth = tplShpe[1]
        varNumVol = tplShpe[2]

        # On first iteration, initialise vector for number of vertices per
        # subject:
        # if idxRoi == 0:


        # *********************************************************************
        # *** Subtract baseline mean

        # The input to this function are timecourses that have been normalised
        # to the pre-stimulus baseline. The datapoints are signal intensity
        # relative to the pre-stimulus baseline, and the pre-stimulus baseline
        # has a mean of one. We subtract one, so that the datapoints are
        # percent signal change relative to baseline.
        for strSubID, lstItem in dicAllSubsRoiErt.items():
            # Get event related time courses from list (second entry in list is
            # the number of vertices contained in this ROI).
            aryRoiErt = lstItem[0]
            # Subtract baseline mean:
            aryRoiErt = np.subtract(aryRoiErt, 1.0)
            # Is this line necessary (hard copy)?
            dicAllSubsRoiErt[strSubID] = [aryRoiErt, lstItem[1]]

        # *********************************************************************
        # *** Create group level ERT

        # Create across-subjects data array:
        aryAllSubsRoiErt = np.zeros((varNumSub, varNumCon, varNumDpth,
                                     varNumVol))

        # Vector for number of vertices per subject (used for weighted
        # averaging):
        vecNumVrtcs = np.zeros((varNumSub))

        idxSub = 0

        for lstItem in dicAllSubsRoiErt.values():

            # Get event related time courses from list.
            aryRoiErt = lstItem[0]

            # Get number of vertices for this subject:
            vecNumVrtcs[idxSub] = lstItem[1]

            aryAllSubsRoiErt[idxSub, :, :, :] = aryRoiErt

            idxSub += 1

        # *********************************************************************
        # *** Average across cortical depth

        # Current shape:
        # aryAllSubsRoiErt[varNumSub, varNumCon, varNumDpth, varNumVol]

        # Take mean across cortical depth levels:
        aryAllSubsRoiErt = np.mean(aryAllSubsRoiErt, axis=2)

        # New shape:
        # aryAllSubsRoiErt[varNumSub, varNumCon, varNumVol]

        # *********************************************************************
        # *** Crop time course

        if not tplCrp is None:
            # Crop time course:
            aryAllSubsRoiErt = aryAllSubsRoiErt[:, :, tplCrp[0]:tplCrp[1]]
            # Update number of volumes:
            varNumVol = aryAllSubsRoiErt.shape[2]

    # *************************************************************************
    # *** Put samples into dataframe

    # Number of samples:
    varNumSmpl = (varNumRoi * varNumCon * varNumSub * varNumVol)

    # Create dataframe:
    objDf = pd.DataFrame('0', index=np.arange(varNumSmpl), columns=lstFtr)

    # Dictionary for dataframe column datatypes.
    dicType = {'ROI': str,
               'Condition': str,
               'Subject': np.int16,
               'Volume': np.int16,
               'Vertices': np.int32,
               'PSC': np.float64}

    # Set datatype:
    objDf.astype(dicType)

    # Counter for samples:
    idxSmpl = 0

    # Loop through samples:
    for idxRoi in lstRoi:
        for idxCon in range(varNumCon):
            for idxSub in range(varNumSub):
                for idxVol in range(varNumVol):

                    # Data to dataframe:
                    objDf.at[idxSmpl, 'ROI'] = idxRoi.upper()
                    objDf.at[idxSmpl, 'Condition'] = lstCon[idxCon].upper()
                    objDf.at[idxSmpl, 'Subject'] = idxSub
                    objDf.at[idxSmpl, 'Volume'] = idxVol
                    objDf.at[idxSmpl, 'Vertices'] = vecNumVrtcs[idxSub]
                    objDf.at[idxSmpl, 'PSC'] = aryAllSubsRoiErt[idxSub,
                                                                idxCon,
                                                                idxVol]

                    # Increment counter:
                    idxSmpl += 1


    # *************************************************************************
    # *** Save data to csv

    objDf.to_csv(strCsv.format(idxMta), sep=';', index=False)
    # *************************************************************************


# -----------------------------------------------------------------------------
# *** Create plot

# Set datatype to float:
# objDf = objDf.astype({"PSC": np.float64})

# Create plot separately for conditions:
# sns.lineplot(x="Depth",
#              y="PSC",
#              hue="Condition",
#              data=objDf)
# -----------------------------------------------------------------------------
