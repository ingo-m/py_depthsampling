# -*- coding: utf-8 -*-
"""Save cortical depth profiles to csv file."""


import numpy as np
import pandas as pd
# import seaborn as sns


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of npz files with cortical depth profiles (ROI and condition left open):
# strNpz = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/{}_rh_{}_sst_deconv_model_1.npz'  #noqa
strNpz = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/periphery/{}_rh_{}_trn_deconv_model_1.npz'  #noqa

# List of ROIs:
lstRoi = ['v1', 'v2', 'v3']

# List of conditions:
lstCon = ['Pd', 'Ps', 'Cd']

# Output path for data csv file:
# strCsv = '/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/depth_data_stimulus.csv'  #noqa
strCsv = '/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/depth_data_periphery.csv'  #noqa


# -----------------------------------------------------------------------------
# *** Preparations

# Number of conditions:
varNumCon = len(lstCon)

# Number of ROIs:
varNumRoi = len(lstRoi)

# Get number of subjects from data:
objNpz01 = np.load(strNpz.format(lstRoi[0], lstCon[0]))
ary01 = objNpz01['arySubDpthMns']
varNumSub = ary01.shape[0]

# Number of depth levels:
varNumDpth = ary01.shape[1]

# List of features for dataframe:
lstFtr = ['ROI', 'Condition', 'Subject', 'Depth', 'PSC']

# Number of samples:
varNumSmpl = (varNumRoi * varNumCon * varNumSub * varNumDpth)

# Create dataframe:
objDf = pd.DataFrame('0', index=np.arange(varNumSmpl), columns=lstFtr)

# Dictionary for dataframe column datatypes.
dicType = {'ROI': str,
           'Condition': str,
           'Subject': np.int16,
           'Depth': np.int16,
           'PSC': np.float64}

# Set datatype:
objDf.astype(dicType)


# -----------------------------------------------------------------------------
# *** Load data from disk

# Counter for samples:
idxSmpl = 0

# Fill dataframe with cortical depth profiles:
for idxRoi in lstRoi:
    for idxCon in lstCon:

        # Load npz file:
        objNpz01 = np.load(strNpz.format(idxRoi, idxCon))
        ary01 = objNpz01['arySubDpthMns']

        for idxSub in range(varNumSub):
            for idxDpth in range(varNumDpth):

                # Data to dataframe:
                objDf.at[idxSmpl, 'ROI'] = idxRoi.upper()
                objDf.at[idxSmpl, 'Condition'] = idxCon.upper()
                objDf.at[idxSmpl, 'Subject'] = idxSub
                objDf.at[idxSmpl, 'Depth'] = idxDpth
                objDf.at[idxSmpl, 'PSC'] = ary01[idxSub, idxDpth]

                # Increment counter:
                idxSmpl += 1


# -----------------------------------------------------------------------------
# *** Save data to csv

objDf.to_csv(strCsv, sep=';', index=False)


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
