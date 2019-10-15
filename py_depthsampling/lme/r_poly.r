#' Hypothesis test on cortical depth profile peak positions
#' 
#' Perform a test for difference in peak positions between ROIs in cortical
#' depth profiles of condition differences. Single-subject peak position values
#' for ROIs are loaded from csv file, and can be created using
#' ~/py_depthsampling/py_depthsampling/poly/fit_poly_test_roi.py

library(nlme)

# Input file path:
strCsv <- '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/poly/v1_and_v3_condition_Pd_sst_minus_Ps_sst_plus_Cd_sst.csv'
#strCsv <- '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/poly/v1_and_v3_condition_Pd_sst_minus_Cd_sst.csv'

# Read CSV into R
objDf <- read.csv(file=strCsv,
                  header=TRUE,
                  sep=';')

head(objDf)

# Weights = ~1/n where n is number of vertices.

mdlNull = lme(PeakPosition ~ 1,
              objDf,
              random=(~1|Subject),
              weights=(~1/Vertices),
              correlation=corSymm(form=(~1|Subject/ROI)),
              method='ML')

mdlFull = lme(PeakPosition ~ ROI,
              objDf,
              random=(~1|Subject),
              weights=(~1/Vertices),
              correlation=corSymm(form=(~1|Subject/ROI)),
              method='ML')

summary(mdlFull)

# Model comparison:
anova(mdlNull, mdlFull)
