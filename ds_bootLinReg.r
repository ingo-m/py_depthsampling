"
Parametric bootstraping of linear regerssion.

Input: array created with ds_testSemi.py, of the form aryEmpSemiR[idxDpt, 3],
where the first dimension corresponds to the number of (cortical depth levels
* number of ROIs), and the second dimension corresponds to three columns for
the linear model, representing: the signal (i.e. the semisaturation constant,
which is the idenpendent variable), the depth level (1st dependent variable)
and the ROI membership (2nd dependent variable).

Requires: RcppCNPy library.

For more information see
https://stats.stackexchange.com/questions/83012/how-to-obtain-p-values-of-coefficients-from-bootstrap-regression
"

library(RcppCNPy)

# Load empirical semisaturation constant from disk (i.e. semisaturation constant
# fitted on the full dataset; needs to be created with ds_crfMain.py):
aryEmpSemi  <- npyLoad('/Users/john/Dropbox/Sonstiges/Higher_Level_Analysis/aryEmpSemi_corrected_power.npy')

# Put data into data frame object:
datEmpSemi  <- data.frame(aryEmpSemi)

# Adjust column names:
colnames(datEmpSemi) <- c('Signal', 'Depth', 'ROI')

# Number of iterations:
varNumIt    <- 10000

# Linear model:
mdlLin      <- lm(Signal ~ Depth + ROI, datEmpSemi)

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
  # on the simulated, data, and the original data is put into this new model.
  # The resulting parameter estimates are saved into aryBoot.
  aryBoot[idxIt,] <-coefficients(lm(unlist(simulate(mdlLin)) ~ Depth + ROI, datEmpSemi))
}

# There are three coefficient: (1) intercept, (2) depth, (3) ROI.

# Get the p-values for coefficient
P_val1 <-mean( abs(aryBoot[,1] - mean(aryBoot[,1]) )> abs( vecPe[1]))
P_val2 <-mean( abs(aryBoot[,2] - mean(aryBoot[,2]) )> abs( vecPe[2]))
P_val3 <-mean( abs(aryBoot[,3] - mean(aryBoot[,3]) )> abs( vecPe[3]))

#and some parametric bootstrap confidence intervals (2.5%, 97.5%)
ConfInt1 <- quantile(aryBoot[,1], c(.025, 0.975))
ConfInt2 <- quantile(aryBoot[,2], c(.025, 0.975))
ConfInt3 <- quantile(aryBoot[,3], c(.025, 0.975))
