# -*- coding: utf-8 -*-
"""
Repeated measures anova on cortical depth profiles.
"""


import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of npz files with cortical depth profiles (ROI and condition left open):
strNpz = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/{}_rh_{}_sst.npz'
# strNpz = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/periphery/{}_rh_{}_sst.npz'

# List of ROIs:
lstRoi = ['v1', 'v2', 'v3']

# List of conditions:
lstCon = ['Pd', 'Ps', 'Cd']

# Output path for data csv file:
strCsv = '/home/john/Desktop/anovarm.csv'


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
        # print(strNpz.format(idxRoi, idxCon))
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
# *** Fit repeated measures ANOVA

# Create ANOVA object:
objAnov = AnovaRM(objDf,
                  'PSC',
                  'Subject',
                  within=['ROI', 'Condition', 'Depth'],
                  between=None,
                  aggregate_func=None)

# Fit model:
objResult = objAnov.fit()

# Print results:
# print(objResult.anova_table)
print(objResult.summary())

# -----------------------------------------------------------------------------
# *** Save data to csv

<<<<<<< HEAD
objDf.to_csv(strCsv, sep=';', index=False)
=======
# objDf.to_csv(strCsv, sep=';', index=False)
>>>>>>> surface
# -----------------------------------------------------------------------------
