################################################################################
# Repeated measures ANOVA on permutation null distribution.
#
# Test for a difference between conditions (e.g. PacMan Dynamic vs. PacMan
# Static) in cortical depth profiles. Cortical depth levels are treated as
# repeated measures.
#
#
# There are four inputs:
#
# aryEmp01 & aryEmp02
# Arrays with empirical cortical depth profiles for two conditions (e.g. PacMan
# Dynamic and PacMan static), shape aryDpthEmp01[subject, depth].
#
# aryRnd01 & aryRnd02:
# Arrays with null distribution cortical depth profiles for two 'conditions'
# with permuted labels, shape aryDpthRnd01[iteration, subject, depth].
# Can be created with `py_depthsampling.permutation.py`.
#
# The test is performed as follows:
# - Load numpy arrays from disk
# - Loop through resampling iterations (in aryDpthRnd01 & aryDpthRnd02), and
#   test for a difference between the two randomised 'conditions', using a
#   repeated measures ANOVA with factors condition, subject, depth level
#   (the repeated measure), and cortical depth signal as a dependent variabel.
# - The F-statistics resulting from the ANOVA (one for each resampling
#   iteration) constitute the null distribution.
# - Conduct same ANOVA on empirical depth profiles.
# - Compare F-statistic from ANOVA on empirical profils with the null
#   distribution to assess significance.
#
# Requires: RcppCNPy library.
################################################################################

library(RcppCNPy)


aryEmp01[1, ]

# Input files:
aryPthInEmp <- c('/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v1_rh_Pd_sst_forR.npy',
                 '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v1_rh_Ps_sst_forR.npy',
                 '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/v1_rh_vecNumInc_forR.npy')

# TODO convert npz to npy in python

aryEmp01  <- npyLoad(aryPthInEmp[1])
aryEmp02  <- npyLoad(aryPthInEmp[2])


dfEmp01 = data.frame(aryEmp01)

?data.frame



  # Put data into data frame object:
  datEmpSemi  <- data.frame(aryEmpSemi)

  # Adjust column names:
  colnames(datEmpSemi) <- c('Signal', 'Depth', 'ROI', 'Subject')

  # Linear model:
  mdlLin      <- lm(Signal ~ Depth + ROI + Subject, datEmpSemi)

  # Parameter estimates (coefficients):
  vecPe       <- coefficients(mdlLin)

  # Number of coefficients to test against:
  varNumPe    <- length(vecPe)

  # Array for bootstrapping results. Columns correspond to iterations, rows
  # correspond to the three parameter estimates for each iteration.
  aryBoot     <- matrix(rep(0, varNumPe * varNumIt), ncol=varNumPe)

  # Loop through iterations:
  for (idxIt in 1:varNumIt) {
    # The parametric bootstrap. The 'simulate()' function simulates new
    # parameter estimates based on the full model. The model fitting is repeated
    # on the simulated data, and the original data is put into this new model.
    # The resulting parameter estimates are saved into aryBoot.
    aryBoot[idxIt,] <-coefficients(lm(unlist(simulate(mdlLin)) ~ Depth + ROI + Subject, datEmpSemi))
  }

  # There are four coefficients: (1) intercept, (2) depth, (3) ROI, (4) Subject.

  # (1) Intercept

  # Mean parameter estimate across bootstrap iterations:
  varPe01Mne <- mean(aryBoot[,1])
  # Absolute difference between the PE of each iteration and the mean PE across
  # iterations (null distribution, i.e. distribution of the PE assuming that the
  # mean value of the PE is zero):
  vecPe01Abs <-  abs(aryBoot[,1] - mean(varPe01Mne))
  # Logical test: For each iteration, is the PE from the null distribution greater
  # than the PE found on the full model?
  vecPe01Lgc <- vecPe01Abs > abs(vecPe[1])
  # Ratio of iterations with bootstrapped PE under H0 greater than empirical PE
  # (p-value):
  varPe01P <- mean(vecPe01Lgc)
  # Fix decimal places:
  varPe01P <- format(round(varPe01P, 5), nsmall = 5)
  # Confidence interval for PE:
  vecPe01Conf <- quantile(aryBoot[,1], c(.025, 0.975))

  strResults <- c(strResults,
                  'INTERCEPT:',
                  '',
                  'p-value:',
                  varPe01P,
                  '',
                  'Confidence interval:',
                  vecPe01Conf,
                  '')

  # (2) Cortical depth level

  # Mean parameter estimate across bootstrap iterations:
  varPe02Mne <- mean(aryBoot[,2])
  # Absolute difference between the PE of each iteration and the mean PE across
  # iterations (null distribution, i.e. distribution of the PE assuming that the
  # mean value of the PE is zero):
  vecPe02Abs <-  abs(aryBoot[,2] - mean(varPe02Mne))
  # Logical test: For each iteration, is the PE from the null distribution greater
  # than the PE found on the full model?
  vecPe02Lgc <- vecPe02Abs > abs(vecPe[2])
  # Ratio of iterations with bootstrapped PE under H0 greater than empirical PE
  # (p-value):
  varPe02P <- mean(vecPe02Lgc)
  # Fix decimal places:
  varPe02P <- format(round(varPe02P, 5), nsmall = 5)
  # Confidence interval for PE:
  vecPe02Conf <- quantile(aryBoot[,2], c(.025, 0.975))

  strResults <- c(strResults,
                  'CORTICAL DEPTH LEVEL:',
                  '',
                  'p-value:',
                  varPe02P,
                  '',
                  'Confidence interval:',
                  vecPe02Conf,
                  '')

  # (3) ROI

  # Mean parameter estimate across bootstrap iterations:
  varPe03Mne <- mean(aryBoot[,3])
  # Absolute difference between the PE of each iteration and the mean PE across
  # iterations (null distribution, i.e. distribution of the PE assuming that the
  # mean value of the PE is zero):
  vecPe03Abs <-  abs(aryBoot[,3] - mean(varPe03Mne))
  # Logical test: For each iteration, is the PE from the null distribution greater
  # than the PE found on the full model?
  vecPe03Lgc <- vecPe03Abs > abs(vecPe[3])
  # Ratio of iterations with bootstrapped PE under H0 greater than empirical PE
  # (p-value):
  varPe03P <- mean(vecPe03Lgc)
  # Fix decimal places:
  varPe03P <- format(round(varPe03P, 5), nsmall = 5)
  # Confidence interval for PE:
  vecPe03Conf <- quantile(aryBoot[,3], c(.025, 0.975))

  strResults <- c(strResults,
                  'ROI:',
                  '',
                  'p-value:',
                  varPe03P,
                  '',
                  'Confidence interval:',
                  vecPe03Conf,
                  '')

  # (4) Subject

  # Mean parameter estimate across bootstrap iterations:
  varPe04Mne <- mean(aryBoot[,4])
  # Absolute difference between the PE of each iteration and the mean PE across
  # iterations (null distribution, i.e. distribution of the PE assuming that the
  # mean value of the PE is zero):
  vecPe04Abs <-  abs(aryBoot[,4] - mean(varPe04Mne))
  # Logical test: For each iteration, is the PE from the null distribution greater
  # than the PE found on the full model?
  vecPe04Lgc <- vecPe04Abs > abs(vecPe[4])
  # Ratio of iterations with bootstrapped PE under H0 greater than empirical PE
  # (p-value):
  varPe04P <- mean(vecPe04Lgc)
  # Fix decimal places:
  varPe04P <- format(round(varPe04P, 5), nsmall = 5)
  # Confidence interval for PE:
  vecPe04Conf <- quantile(aryBoot[,4], c(.025, 0.975))

  strResults <- c(strResults,
                  'SUBJECT:',
                  '',
                  'p-value:',
                  varPe04P,
                  '',
                  'Confidence interval:',
                  vecPe04Conf,
                  '')

}

strResults <- c(strResults,
                '#######################################################################')

# File to save results to:
fleResults <- file('/home/john/Dropbox/ParCon_Manuscript/Tables/lin_reg_boot.txt')

# Write results to file:
writeLines(c(strResults), fleResults)

# Close results file.
close(fleResults)
